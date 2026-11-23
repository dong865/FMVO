"""Subtask(FiLM)-conditioned Role-IQLD for the discovery+conditional-distillation
mainline capture test (scan-only, development build).

Design goal: on the SAME capable base that can already capture (the v11
contract-v2 Role-IQLD), add a causal *subtask* (sequence mode) condition that
warms up identically to the base (FiLM starts as identity) and a matched
control that receives an *uninformative* (uniform) mode. Only the information
carried by the mode differs, so the capture delta isolates whether the
sequence-aware discovered subtask adds value over the current state alone.

Nothing here is wired into the deployment runtime or the gate: this is the
experimental surface for the mainline's capture-level A/B.

    python scan_subtask_iqld_capture.py --dataset <contract-v2 expert> \
        --init-checkpoint <v11_best_safe.pt> --k <K> --out <dir>
"""

from typing import Optional

import numpy as np
import torch
from torch import nn

from .additional_models import normalized_entities
from .role_iqld import RoleIQLDAgent, RuntimeSafetyGuard, SharedRoleActor


class SubtaskFiLMActor(nn.Module):
    """SharedRoleActor + per-mode FiLM on the first hidden layer.

    The base network and role_matrix are structurally identical to
    SharedRoleActor so a v11 state dict loads with ``strict=False``. The FiLM
    branch is initialised to identity (scale -> 1, shift -> 0) so at start the
    conditioned action equals the base action regardless of mode.
    """

    def __init__(self, obs_dim: int, k_roles: int, k_subtasks: int,
                 action_dim: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.k_roles = k_roles
        self.k_subtasks = k_subtasks
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.role_matrix = nn.Parameter(torch.randn(k_roles, hidden_dim) * 0.1)
        # identical network structure to SharedRoleActor (indexed 0,2,4)
        self.network = nn.Sequential(
            nn.Linear(obs_dim + hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
        # FiLM conditioning on the first hidden activation
        self.mode_matrix = nn.Parameter(torch.randn(k_subtasks, hidden_dim) * 0.1)
        self.film_scale = nn.Linear(hidden_dim, hidden_dim)
        self.film_shift = nn.Linear(hidden_dim, hidden_dim)
        # identity init: scale -> 0 (so 1+scale=1), shift -> 0
        nn.init.zeros_(self.film_scale.weight)
        nn.init.zeros_(self.film_scale.bias)
        nn.init.zeros_(self.film_shift.weight)
        nn.init.zeros_(self.film_shift.bias)

    def forward(self, actor_obs, role_logits, mode_soft=None):
        soft_z = torch.softmax(role_logits, dim=-1)          # [B, N, K_r]
        role_embedding = soft_z @ self.role_matrix            # [B, N, D]
        x = torch.cat((actor_obs, role_embedding), dim=-1)    # [B, N, obs+D]
        h = torch.tanh(self.network[0](x))                    # [B, N, D]
        if mode_soft is not None:
            mode_embedding = mode_soft @ self.mode_matrix            # [B, N, D]
            scale = 1.0 + self.film_scale(mode_embedding)
            shift = self.film_shift(mode_embedding)
            h = h * scale + shift
        h = torch.tanh(self.network[2](h))
        return torch.tanh(self.network[4](h))


class CausalSubtaskPredictor(nn.Module):
    """p(z | current per-pursuer state): 8-dim -> K subtask logits.

    Shared across pursuers (each pursuer contributes its own focal window); the
    offline discovery assigned a mode to each focal window so the same learned
    map is used for every teammate at runtime.
    """

    def __init__(self, state_dim: int = 8, k_subtasks: int = 2,
                 hidden_dim: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, k_subtasks),
        )

    def forward(self, states):
        return self.network(states)


def mode_soft_from_predictor(predictor, actor_obs, positions, velocities,
                             config, device="cpu"):
    """Causal per-pursuer subtask soft assignment.

    ``positions``/``velocities`` are the *normalized* tensors passed to the
    agent policy ([B, N_entities, 2], evader last). The 8-dim per-pursuer
    state matches the representation used to train the mode label of each
    discovery window (focal relation to evader + masked teammate mean).
    """
    num = config.num_pursuers
    pursuer_pos = positions[:, :num]                       # [B, num, 2]
    pursuer_vel = velocities[:, :num]
    evader_pos = positions[:, num:num + 1].expand(-1, num, -1)
    evader_vel = velocities[:, num:num + 1].expand(-1, num, -1)
    focal_pos = pursuer_pos - evader_pos                   # [B, num, 2]
    focal_vel = pursuer_vel - evader_vel
    # masked teammate mean: diff[i,j] = p_j - p_i ; exclude j == i
    diff_pos = pursuer_pos.unsqueeze(2) - pursuer_pos.unsqueeze(1)  # [B,num,num,2]
    diff_vel = pursuer_vel.unsqueeze(2) - pursuer_vel.unsqueeze(1)
    eye = torch.eye(num, device=device)
    mask = (1.0 - eye).unsqueeze(-1)                       # [num,num,1]
    tm_pos = (diff_pos * mask).sum(dim=1) / (num - 1)      # [B,num,2]
    tm_vel = (diff_vel * mask).sum(dim=1) / (num - 1)
    state = torch.cat(
        (focal_pos, focal_vel, tm_pos, tm_vel), dim=-1
    ).reshape(-1, 8)                                        # [B*num, 8]
    logits = predictor(state)
    return torch.softmax(logits, dim=-1).reshape(
        positions.shape[0], num, predictor.network[-1].out_features
    )


class RoleIQLDSubtaskAgent(RoleIQLDAgent):
    """Role-IQLD with a causal subtask FiLM condition.

    ``condition_mode`` selects a real (causal) vs uniform (uninformative) mode;
    both use the SAME SubtaskFiLMActor, so the A/B differs only in the mode
    signal. Warm start: pass the v11 agent via ``warm_agent`` (or call
    ``warm_start_from``) so the shared actor network is copied and the FiLM
    begins at identity.
    """

    def __init__(self, config, device="cpu", d_model=64,
                 hyperparameters=None, k_roles=None, k_subtasks=2,
                 condition_mode: bool = True):
        super().__init__(config, device, d_model, hyperparameters, k_roles)
        self.k_subtasks = k_subtasks
        self.condition_mode = condition_mode
        # build the conditioned actor warm from the base actor
        self.actor = SubtaskFiLMActor(
            self.actor_obs_dim, self.k_roles, k_subtasks
        ).to(self.device)
        self.subtask_predictor = CausalSubtaskPredictor(
            8, k_subtasks
        ).to(self.device)
        # include the new parameters in the policy optimizer
        self.policy_optimizer = torch.optim.Adam(
            list(self.situation_encoder.parameters())
            + list(self.role_predictor.parameters())
            + list(self.actor.parameters())
            + list(self.subtask_predictor.parameters()),
            lr=self.hyperparameters.policy_learning_rate,
        )

    def warm_start_from(self, base_agent):
        """Copy base (v11) weights into the shared modules + actor backbone.

        The conditioned actor's network/role_matrix are overwritten by the base
        actor's; mode/film parameters keep their identity initialization.
        """
        self.situation_encoder.load_state_dict(
            base_agent.situation_encoder.state_dict()
        )
        self.role_predictor.load_state_dict(base_agent.role_predictor.state_dict())
        self.actor.load_state_dict(base_agent.actor.state_dict(), strict=False)
        self.value.load_state_dict(base_agent.value.state_dict())
        self.target_value.load_state_dict(base_agent.target_value.state_dict())
        self.q1.load_state_dict(base_agent.q1.state_dict())
        self.q2.load_state_dict(base_agent.q2.state_dict())
        self.epn.load_state_dict(base_agent.epn.state_dict())

    def policy(self, actor_obs, positions, velocities):
        embeddings, mask = self.encode(positions, velocities)
        role_logits = self.role_predictor(actor_obs)
        num = actor_obs.shape[1]  # per-pursuer dimension
        if self.condition_mode:
            mode_soft = mode_soft_from_predictor(
                self.subtask_predictor, actor_obs, positions, velocities,
                self.config, self.device,
            )                                       # [B, num, K]
        else:
            mode_soft = torch.full(
                (positions.shape[0], num, self.k_subtasks),
                1.0 / self.k_subtasks, device=self.device,
            )
        actions = self.actor(actor_obs, role_logits, mode_soft)
        return actions, embeddings, role_logits, mask


class SubtaskRoleIQLDRuntime:
    """Runtime wrapper exposing the ``.actions/.reset`` interface for evaluation.

    Mirrors ``RoleIQLDRuntime`` but drives a ``RoleIQLDSubtaskAgent`` so the
    deployment-evaluation harness can roll out a conditioned/unconditioned
    subtask policy identically to the base runtime.
    """

    def __init__(self, agent: RoleIQLDSubtaskAgent, safety_guard: bool = True):
        self.agent = agent
        self.device = agent.device
        self.config = agent.config
        self.safety_guard = bool(safety_guard)
        self._guard = (
            RuntimeSafetyGuard(self.config) if safety_guard else None
        )

    def reset(self):
        self.agent.reset_history()
        if self._guard is not None:
            self._guard.reset()

    def actions(self, positions, velocities, origin, virtual_obstacles=None):
        state = np.concatenate(
            (
                (positions[-1] - origin) / self.config.arena_half_extent,
                velocities[-1] / self.config.velocity_scale,
            )
        )
        self.agent.update_history(state)
        prediction, confidence = self.agent.predict()
        from .features import build_observations

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
        return self._guard.apply(
            raw_actions, positions, velocities, origin, virtual_obstacles
        )
