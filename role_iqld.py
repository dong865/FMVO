"""Endogenous-role IQLD models for the expert-behavior role-discovery design.

Stage 4 runtime: roles are discovered from unlabelled expert behavior (frozen
discovery artifact), the online role predictor is row-normalized per member
(roles may repeat or be unused), the actor is shared and role-conditioned, and
the critics are permutation-invariant masked set networks. The fixed 3x3
Sinkhorn assignment and geometric slot losses are removed; checkpoints are
versioned (v6) so pre-EPN and fixed-slot checkpoints fail with an explicit message
instead of silently mixing role mechanisms.
"""

import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn

from .additional_models import PredictionFeatures, normalized_entities
from .config import CrazyfliePhysics, TaskConfig
from .features import OBS_DIM, build_observations
from .iqld_hyperparameters import IQLDHyperparameters
from .safety import project_obstacle_velocity, project_pairwise_velocity


ROLE_IQLD_CHECKPOINT_VERSION = 6

# Runtime safety guard: project pursuer velocity commands away from pairwise
# close approaches earlier and harder than the in-environment projection. The
# environment projection alone lets true separations dip a few millimetres
# below the emergency distance because it acts on commanded velocities while
# the delayed velocity PID tracks them with lag. The guard acts through
# rate-limited actions (max_accel*dt per step), so it must engage earlier.
#
# Distances rescaled 2026-10-15 to match the airframe. The previous floors
# (0.34 m pursuer-pursuer, 0.38 m pursuer-evader) were 3.7-4.1x the 0.092 m
# motor-to-motor span and fought the policy every time it tried to close the
# 0.72 m capture ring, which is why real encirclement could not complete.
# New values keep a lead over TaskConfig (emergency 0.12 / warning 0.20) to
# still compensate for PID lag, but no longer exceed the capture radius
# budget.
RUNTIME_GUARD_WARNING_DISTANCE = 0.25
RUNTIME_GUARD_FLOOR_DISTANCE = 0.14
RUNTIME_GUARD_SEPARATION_SPEED = 0.15
# Pursuer-evader constraints need a larger floor and anticipation: the evader
# is uncontrolled and faster (0.50 vs 0.45 m/s), so a retreating pursuer
# cannot outrun it once close. Projecting against the linearly extrapolated
# evader position engages the retreat ~0.4 s earlier.
RUNTIME_GUARD_EVADER_WARNING_DISTANCE = 0.30
RUNTIME_GUARD_EVADER_FLOOR_DISTANCE = 0.17
RUNTIME_GUARD_EVADER_SEPARATION_SPEED = 0.25
RUNTIME_GUARD_EVADER_ANTICIPATION_SECONDS = 0.4


class EndogenousRolePredictor(nn.Module):
    """Row-normalized per-member role logits from local observations.

    Each member independently maps its local observation to K role logits;
    softmax over roles gives a row-normalized distribution. Multiple members
    may share a role and a role may be unused. No Sinkhorn column forcing.
    """

    def __init__(self, obs_dim: int, k_roles: int, hidden_dim: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.k_roles = k_roles
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, k_roles),
        )

    def forward(self, actor_obs: torch.Tensor) -> torch.Tensor:
        # actor_obs: [B, N, obs_dim] -> [B, N, K] logits
        if actor_obs.ndim != 3:
            raise ValueError("role predictor expects [B, N, obs_dim]")
        return self.network(actor_obs)


class SharedRoleActor(nn.Module):
    """Shared role-conditioned actor: action_i = actor(obs_i, role_emb_i).

    The same network serves every pursuer and team size; the role enters
    through a soft role embedding (row-normalized assignment x role matrix).

    With ``stochastic=True`` the same trunk additionally exposes a per-action
    log-std parameter and :meth:`sample` implements a reparameterised
    tanh-squashed Gaussian (SAC/IAC-style self-bootstrapping actor).  The
    deterministic trunk parameters are unchanged, so IQLD checkpoints load
    into the stochastic variant (missing ``log_std`` is skipped) and
    deterministic inference is identical for both modes.
    """

    def __init__(self, obs_dim: int, k_roles: int, action_dim: int = 2,
                 hidden_dim: int = 64, stochastic: bool = False,
                 log_std_init: float = -0.5):
        super().__init__()
        self.obs_dim = obs_dim
        self.k_roles = k_roles
        self.action_dim = action_dim
        self.stochastic = stochastic
        self.role_matrix = nn.Parameter(torch.randn(k_roles, hidden_dim) * 0.1)
        self.network = nn.Sequential(
            nn.Linear(obs_dim + hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        if stochastic:
            # pre-tanh mean comes from network; log_std is per action dim.
            self.log_std = nn.Parameter(
                torch.full((action_dim,), float(log_std_init))
            )
        else:
            self.log_std = None

    def forward(self, actor_obs, role_logits):
        soft_z = torch.softmax(role_logits, dim=-1)          # [B, N, K]
        role_embedding = soft_z @ self.role_matrix            # [B, N, D]
        pre_tanh = self.network(
            torch.cat((actor_obs, role_embedding), dim=-1)
        )
        return torch.tanh(pre_tanh)

    def _pre_tanh(self, actor_obs, role_logits) -> torch.Tensor:
        soft_z = torch.softmax(role_logits, dim=-1)
        role_embedding = soft_z @ self.role_matrix
        return self.network(torch.cat((actor_obs, role_embedding), dim=-1))

    def deterministic(self, actor_obs, role_logits) -> torch.Tensor:
        """Mean action (tanh-squashed pre-tanh output); mode-agnostic."""
        return torch.tanh(self._pre_tanh(actor_obs, role_logits))

    def sample(self, actor_obs, role_logits) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reparameterised tanh-Gaussian sample and per-agent log-prob.

        Returns (actions [B, N, A], log_prob [B, N]) with log_prob summed over
        the two action dimensions; gradients flow through the sample.
        """
        if not self.stochastic:
            actions = self.deterministic(actor_obs, role_logits)
            zeros = torch.zeros_like(actions[..., 0])
            return actions, zeros
        pre_tanh = self._pre_tanh(actor_obs, role_logits)
        std = torch.exp(torch.clamp(self.log_std, -5.0, 1.0))
        noise = torch.randn_like(pre_tanh)
        squashed = torch.tanh(pre_tanh + std * noise)
        # log N(u; mean=pre_tanh, std) summed over the action dims, then the
        # tanh change-of-variable correction (log(1 - a^2)) per agent.
        log_normal = (
            -0.5 * noise.square()
            - std.log()
            - 0.5 * math.log(2.0 * math.pi)
        ).sum(dim=-1)
        log_correction = torch.log(
            1.0 - squashed * squashed + 1e-6
        ).sum(dim=-1)
        log_prob = log_normal - log_correction
        return squashed, log_prob


class MaskedDeepSetsValue(nn.Module):
    """Permutation-invariant masked set value over member tokens."""

    def __init__(self, d_model: int, k_roles: int, context_dim: int,
                 hidden_dim: int = 64):
        super().__init__()
        self.token = nn.Sequential(
            nn.Linear(d_model + k_roles, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.context = nn.Sequential(
            nn.Linear(context_dim, hidden_dim), nn.ReLU()
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, embeddings, role_logits, mask, prediction_context):
        # embeddings [B, N, D], role_logits [B, N, K], mask [B, N]
        soft_z = torch.softmax(role_logits, dim=-1)
        pursuer_embeddings = embeddings[:, : soft_z.shape[1]]
        tokens = self.token(
            torch.cat((pursuer_embeddings, soft_z), dim=-1)
        )                                                      # [B, N, H]
        weights = mask.float().unsqueeze(-1)
        pooled = (tokens * weights).sum(dim=1)
        pooled = pooled / weights.sum(dim=1).clamp_min(1.0)
        context = self.context(prediction_context)
        return self.head(torch.cat((pooled, context), dim=1)).squeeze(1)


class MaskedDeepSetsQ(nn.Module):
    """Permutation-invariant masked set Q over member tokens and actions."""

    def __init__(self, d_model: int, k_roles: int, context_dim: int,
                 action_dim: int = 2,
                 hidden_dim: int = 64):
        super().__init__()
        self.token = nn.Sequential(
            nn.Linear(d_model + k_roles + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.context = nn.Sequential(
            nn.Linear(context_dim, hidden_dim), nn.ReLU()
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, embeddings, role_logits, actions, mask, prediction_context
    ):
        soft_z = torch.softmax(role_logits, dim=-1)
        pursuer_embeddings = embeddings[:, : soft_z.shape[1]]
        tokens = self.token(
            torch.cat((pursuer_embeddings, soft_z, actions), dim=-1)
        )                                                      # [B, N, H]
        weights = mask.float().unsqueeze(-1)
        pooled = (tokens * weights).sum(dim=1)
        pooled = pooled / weights.sum(dim=1).clamp_min(1.0)
        context = self.context(prediction_context)
        return self.head(torch.cat((pooled, context), dim=1)).squeeze(1)


class RoleIQLDAgent(PredictionFeatures):
    baseline_name = "no_graph_iqld_maddpg"

    def __init__(
        self,
        config: TaskConfig,
        device: str = "cpu",
        d_model: int = 64,
        hyperparameters: Optional[IQLDHyperparameters] = None,
        k_roles: Optional[int] = None,
        actor_mode: str = "iqld",
        log_std_init: float = -1.0,
        hidden_dim: int = 64,
    ):
        if actor_mode not in ("iqld", "iac"):
            raise ValueError("actor_mode must be 'iqld' or 'iac'")
        self.log_std_init = float(log_std_init)
        self.actor_mode = actor_mode
        self.iqld_prior = None
        self.device = torch.device(device)
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.hyperparameters = hyperparameters or IQLDHyperparameters()
        self.k_roles = k_roles or config.num_pursuers
        self.prediction_context_dim = config.prediction_steps * 2 + 1
        self._init_prediction(config, self.device)
        from .additional_models import SituationEncoder

        self.situation_encoder = SituationEncoder(d_model).to(self.device)
        self.role_predictor = EndogenousRolePredictor(
            self.actor_obs_dim, self.k_roles, hidden_dim=hidden_dim
        ).to(self.device)
        self.actor = SharedRoleActor(
            self.actor_obs_dim,
            self.k_roles,
            stochastic=(actor_mode == "iac"),
            log_std_init=self.log_std_init,
            hidden_dim=hidden_dim,
        ).to(self.device)
        self.value = MaskedDeepSetsValue(
            d_model, self.k_roles, self.prediction_context_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)
        self.target_value = MaskedDeepSetsValue(
            d_model, self.k_roles, self.prediction_context_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)
        self.q1 = MaskedDeepSetsQ(
            d_model, self.k_roles, self.prediction_context_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)
        self.q2 = MaskedDeepSetsQ(
            d_model, self.k_roles, self.prediction_context_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)
        self.target_value.load_state_dict(self.value.state_dict())
        if actor_mode == "iac":
            self.q1_target = MaskedDeepSetsQ(
                d_model, self.k_roles, self.prediction_context_dim
            ).to(self.device)
            self.q2_target = MaskedDeepSetsQ(
                d_model, self.k_roles, self.prediction_context_dim
            ).to(self.device)
            self.q1_target.load_state_dict(self.q1.state_dict())
            self.q2_target.load_state_dict(self.q2.state_dict())
            for module in (self.q1_target, self.q2_target):
                for parameter in module.parameters():
                    parameter.requires_grad = False
        else:
            self.q1_target = None
            self.q2_target = None
        policy_parameters = (
            list(self.situation_encoder.parameters())
            + list(self.role_predictor.parameters())
            + list(self.actor.parameters())
        )
        self.policy_optimizer = torch.optim.Adam(
            policy_parameters, lr=self.hyperparameters.policy_learning_rate
        )
        self.value_optimizer = torch.optim.Adam(
            self.value.parameters(), lr=self.hyperparameters.value_learning_rate
        )
        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=self.hyperparameters.critic_learning_rate,
        )

    def _mask(self, positions: torch.Tensor) -> torch.Tensor:
        """Return a [B, N] mask of active pursuers (all active by default)."""
        count = positions.shape[1] - 1
        return torch.ones(
            positions.shape[0], count, dtype=torch.bool, device=positions.device
        )

    def encode(self, positions, velocities):
        embeddings = self.situation_encoder(positions, velocities)
        mask = self._mask(positions)
        return embeddings, mask

    def policy(self, actor_obs, positions, velocities):
        embeddings, mask = self.encode(positions, velocities)
        role_logits = self.role_predictor(actor_obs)
        actions = self.actor.deterministic(actor_obs, role_logits)
        return actions, embeddings, role_logits, mask

    def rollout_policy(self, actor_obs, positions, velocities, *, stochastic=False):
        """Return rollout actions, optionally from the SAC target distribution."""
        actions, embeddings, role_logits, mask = self.policy(
            actor_obs, positions, velocities
        )
        if stochastic:
            if self.actor_mode != "iac":
                raise ValueError("stochastic rollout requires actor_mode='iac'")
            actions, _ = self.sample_actions(actor_obs, role_logits)
        return actions, embeddings, role_logits, mask

    def sample_actions(self, actor_obs, role_logits):
        """IAC/SAC-mode stochastic actions + per-agent log-probabilities."""
        return self.actor.sample(actor_obs, role_logits)

    def update_q_targets(self, tau: float = 0.005) -> None:
        """Polyak-average the online critics into their target networks."""
        if self.q1_target is None:
            return
        with torch.no_grad():
            for online, target in (
                (self.q1, self.q1_target),
                (self.q2, self.q2_target),
            ):
                for target_parameter, online_parameter in zip(
                    target.parameters(), online.parameters()
                ):
                    target_parameter.mul_(1.0 - tau)
                    target_parameter.add_(tau * online_parameter)

    def prediction_context(self, actor_obs):
        """Return the shared EPN prediction/confidence block for Q and V."""
        context = actor_obs[..., -self.prediction_context_dim:]
        return context.mean(dim=1)

    def save(self, path: str, extra: Optional[Dict[str, object]] = None) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": ROLE_IQLD_CHECKPOINT_VERSION,
                "baseline": self.baseline_name,
                "task_config": self.config.to_dict(),
                "physics": CrazyfliePhysics().__dict__,
                "obs_dim": OBS_DIM,
                "actor_obs_dim": self.actor_obs_dim,
                "d_model": self.d_model,
                "hidden_dim": self.hidden_dim,
                "k_roles": self.k_roles,
                "iqld_hyperparameters": self.hyperparameters.to_dict(),
                **{
                    name: getattr(self, name).state_dict()
                    for name in (
                        "situation_encoder",
                        "role_predictor",
                        "actor",
                        "value",
                        "q1",
                        "q2",
                        "epn",
                    )
                },
                **(
                    {
                        "q1_target": self.q1_target.state_dict(),
                        "q2_target": self.q2_target.state_dict(),
                    }
                    if self.actor_mode == "iac"
                    else {}
                ),
                "extra": {
                    **(extra or {}),
                    "actor_mode": self.actor_mode,
                },
            },
            destination,
        )


class RuntimeSafetyGuard:
    """Pairwise safety action projection shared by runtime and rollouts.

    Tracks the environment's commanded velocities internally so the guard can
    be applied to any stream of measured states and rate-limited actions,
    inside or outside deployment.
    """

    def __init__(self, config: TaskConfig):
        self.config = config
        self._command_estimate = np.zeros(
            (config.num_pursuers, 2), dtype=np.float64
        )

    def reset(self) -> None:
        self._command_estimate = np.zeros(
            (self.config.num_pursuers, 2), dtype=np.float64
        )

    def apply(
        self,
        raw_actions: np.ndarray,
        positions: np.ndarray,
        velocities: np.ndarray,
        origin,
        virtual_obstacles,
    ) -> np.ndarray:
        """Project pursuer actions to keep pairwise separation above a floor.

        Mirrors the environment's command integration and safety projections
        so the internal command estimate tracks the environment's commanded
        velocities, then applies two guard tiers — a pairwise projection for
        pursuer-pursuer pairs and an anticipated repulsion for pursuer-evader
        pairs — and converts the correction back into rate-limited actions.
        """
        cfg = self.config
        step = cfg.max_accel * cfg.dt
        positions = np.asarray(positions, dtype=np.float64)
        velocities = np.asarray(velocities, dtype=np.float64)
        actions = np.clip(np.asarray(raw_actions, dtype=np.float64), -1.0, 1.0)
        norms = np.linalg.norm(actions, axis=1, keepdims=True)
        actions = actions * np.minimum(1.0, 1.0 / np.maximum(norms, 1e-12))
        estimate = self._command_estimate + actions * step
        speed_limits = np.array(
            [cfg.max_speed] * cfg.num_pursuers + [cfg.evader_max_speed]
        )
        estimate = self._limit_rows(estimate, cfg.max_speed)
        # Pursuer-pursuer constraints via the shared pairwise projection.
        guarded = project_pairwise_velocity(
            estimate,
            positions[: cfg.num_pursuers],
            warning_distance=RUNTIME_GUARD_WARNING_DISTANCE,
            emergency_distance=RUNTIME_GUARD_FLOOR_DISTANCE,
            separation_speed=RUNTIME_GUARD_SEPARATION_SPEED,
            speed_limits=cfg.max_speed,
        ).velocities
        # Pursuer-evader constraints with anticipation: full correction on the
        # pursuer (the evader is uncontrolled), against the current and the
        # linearly extrapolated evader position.
        guarded = guarded + self._evader_correction(guarded, positions, velocities)
        guarded = self._limit_rows(guarded, cfg.max_speed)
        output = (guarded - self._command_estimate) / step
        output = self._limit_rows(np.clip(output, -1.0, 1.0), 1.0)
        # Keep the estimate faithful to what the environment will do with the
        # output actions: integrate, then apply the environment's own
        # projections (identical parameters, same measured state).
        integrated = self._command_estimate + output * step
        integrated = self._limit_rows(integrated, cfg.max_speed)
        tracked = np.vstack((integrated, velocities[cfg.evader_index]))
        tracked = project_pairwise_velocity(
            tracked,
            positions,
            warning_distance=cfg.safety_warning_distance,
            emergency_distance=cfg.safety_emergency_distance,
            separation_speed=cfg.safety_separation_speed,
            speed_limits=speed_limits,
        ).velocities
        tracked = project_obstacle_velocity(
            tracked,
            positions,
            origin,
            virtual_obstacles or (),
            safety_margin=cfg.obstacle_safety_margin,
            warning_margin=cfg.obstacle_warning_margin,
            separation_speed=cfg.safety_separation_speed,
            speed_limits=speed_limits,
        ).velocities
        self._command_estimate = tracked[: cfg.num_pursuers]
        return output.astype(np.float32)

    def _evader_correction(
        self,
        velocities: np.ndarray,
        positions: np.ndarray,
        measured_velocities: np.ndarray,
    ) -> np.ndarray:
        """Repulsive correction keeping pursuers off the evader and its ghost.

        Unlike the pairwise projection, the full correction lands on the
        pursuer: the evader is uncontrolled and faster, so sharing the
        correction would leave too little retreat authority.
        """
        cfg = self.config
        evader_position = positions[cfg.evader_index]
        evader_velocity = measured_velocities[cfg.evader_index]
        ghost_position = (
            evader_position
            + evader_velocity * RUNTIME_GUARD_EVADER_ANTICIPATION_SECONDS
        )
        span = (
            RUNTIME_GUARD_EVADER_WARNING_DISTANCE
            - RUNTIME_GUARD_EVADER_FLOOR_DISTANCE
        )
        correction = np.zeros_like(velocities)
        for point in (evader_position, ghost_position):
            for agent in range(cfg.num_pursuers):
                displacement = positions[agent] - point
                distance = float(np.linalg.norm(displacement))
                if distance >= RUNTIME_GUARD_EVADER_WARNING_DISTANCE:
                    continue
                normal = displacement / max(distance, 1e-12)
                urgency = np.clip(
                    (RUNTIME_GUARD_EVADER_WARNING_DISTANCE - distance) / span,
                    0.0,
                    1.0,
                )
                target = RUNTIME_GUARD_EVADER_SEPARATION_SPEED * urgency
                radial = float(np.dot(velocities[agent] - evader_velocity, normal))
                if radial < target:
                    correction[agent] += (target - radial) * normal
        return correction

    @staticmethod
    def _limit_rows(vectors: np.ndarray, limit: float) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors * np.minimum(1.0, limit / np.maximum(norms, 1e-12))


class RoleIQLDRuntime:
    baseline_name = RoleIQLDAgent.baseline_name

    def __init__(self, checkpoint: str, device: str = "cpu",
                 safety_guard: bool = True):
        self.device = torch.device(device)
        try:
            data = torch.load(checkpoint, map_location=self.device, weights_only=False)
        except TypeError:
            data = torch.load(checkpoint, map_location=self.device)
        version = data.get("schema_version")
        if version != ROLE_IQLD_CHECKPOINT_VERSION:
            raise ValueError(
                "role IQLD checkpoints must be schema version %d "
                "(v5 and fixed-slot v4 checkpoints are legacy and require "
                "explicit "
                "diagnostic entry points); found %s"
                % (ROLE_IQLD_CHECKPOINT_VERSION, version)
            )
        if data.get("physics") != CrazyfliePhysics().__dict__:
            raise ValueError("checkpoint Crazyflie physics do not match ROS2")
        self.config = TaskConfig.from_dict(data["task_config"])
        actor_mode = str(
            (data.get("extra") or {}).get("actor_mode", "iqld")
        )
        self.agent = RoleIQLDAgent(
            self.config,
            device,
            int(data.get("d_model", 64)),
            IQLDHyperparameters(**data.get("iqld_hyperparameters", {})),
            k_roles=int(data.get("k_roles", self.config.num_pursuers)),
            actor_mode=actor_mode,
            hidden_dim=int(data.get("hidden_dim", 64)),
        )
        for name in (
            "situation_encoder",
            "role_predictor",
            "actor",
            "value",
            "q1",
            "q2",
            "epn",
        ):
            getattr(self.agent, name).load_state_dict(data[name])
            getattr(self.agent, name).eval()
        self.safety_guard = bool(safety_guard)
        self._guard = (
            RuntimeSafetyGuard(self.config) if self.safety_guard else None
        )

    def reset(self):
        self.agent.reset_history()
        if self._guard is not None:
            self._guard.reset()

    def _apply_safety_guard(
        self,
        raw_actions: np.ndarray,
        positions: np.ndarray,
        velocities: np.ndarray,
        origin,
        virtual_obstacles,
    ) -> np.ndarray:
        return self._guard.apply(
            raw_actions, positions, velocities, origin, virtual_obstacles
        )

    def actions(self, positions, velocities, origin, virtual_obstacles=None):
        state = np.concatenate(
            (
                (positions[-1] - origin) / self.config.arena_half_extent,
                velocities[-1] / self.config.velocity_scale,
            )
        )
        self.agent.update_history(state)
        prediction, confidence = self.agent.predict()
        observations = build_observations(
            positions, velocities, origin, self.config, virtual_obstacles
        )
        actor_obs = self.agent.augment(observations, prediction, confidence)
        normalized_pos, normalized_vel = normalized_entities(
            positions, velocities, origin, self.config
        )
        with torch.no_grad():
            actions, _, _, _ = self.agent.policy(
                torch.as_tensor(
                    actor_obs[None], dtype=torch.float32, device=self.device
                ),
                torch.as_tensor(
                    normalized_pos[None], dtype=torch.float32, device=self.device
                ),
                torch.as_tensor(
                    normalized_vel[None], dtype=torch.float32, device=self.device
                ),
            )
        raw_actions = actions[0].cpu().numpy()
        if not self.safety_guard:
            return raw_actions
        return self._apply_safety_guard(
            raw_actions, positions, velocities, origin, virtual_obstacles
        )
