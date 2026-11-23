"""Models for the migrated no-graph IQLD and no-graph IRL baselines."""

from collections import deque
from pathlib import Path
from typing import Deque, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn

from .config import CrazyfliePhysics, TaskConfig
from .features import OBS_DIM, build_observations
from .model import EvaderPredictionNetwork


ADDITIONAL_CHECKPOINT_VERSION = 1


def load_checkpoint(path: str, device: torch.device) -> Dict[str, object]:
    try:
        data = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        data = torch.load(path, map_location=device)
    if data.get("schema_version") != ADDITIONAL_CHECKPOINT_VERSION:
        raise ValueError("unsupported checkpoint schema")
    if data.get("obs_dim") != OBS_DIM:
        raise ValueError("checkpoint observation definition does not match")
    if data.get("physics") != CrazyfliePhysics().__dict__:
        raise ValueError("checkpoint Crazyflie physics do not match ROS2")
    return data


class PredictionFeatures:
    """Shared EPN history and augmented-observation behavior."""

    def _init_prediction(self, config: TaskConfig, device: torch.device) -> None:
        self.config = config
        self.device = device
        self.actor_obs_dim = OBS_DIM + config.prediction_steps * 2 + 1
        self.epn = EvaderPredictionNetwork(config.prediction_steps).to(device)
        self.epn_optimizer = torch.optim.Adam(self.epn.parameters(), lr=1e-3)
        self.history: Deque[np.ndarray] = deque(maxlen=config.history_steps)

    def reset_history(self) -> None:
        self.history.clear()

    def update_history(self, state: np.ndarray) -> None:
        self.history.append(np.asarray(state, dtype=np.float32))

    def predict(self) -> Tuple[np.ndarray, float]:
        if len(self.history) < 3:
            return np.zeros((self.config.prediction_steps, 2), np.float32), 0.0
        sequence = torch.as_tensor(
            np.asarray(self.history), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            prediction, confidence = self.epn(sequence)
        return prediction[0].cpu().numpy(), float(confidence[0, 0].cpu())

    def augment(
        self,
        observations: np.ndarray,
        prediction: np.ndarray,
        confidence: float,
    ) -> np.ndarray:
        extra = np.concatenate((prediction.reshape(-1), [confidence])).astype(
            np.float32
        )
        return np.concatenate((observations, np.repeat(extra[None], 3, axis=0)), axis=1)


def normalized_entities(
    positions: np.ndarray,
    velocities: np.ndarray,
    origin: np.ndarray,
    config: TaskConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(positions) - np.asarray(origin)[None]
    pos = pos / config.arena_half_extent
    vel = np.asarray(velocities) / config.velocity_scale
    return pos.astype(np.float32), vel.astype(np.float32)


class ReplayBuffer:
    """FIFO transition buffer with O(1) random access.

    A plain list with a logical start pointer: deque random indexing is O(n),
    which made each ``sample`` traverse the whole buffer (~800k pointer hops
    per 256-sample batch at 30k transitions) and slowed BC/discriminator
    updates to ~40 ms/step. Compaction is amortized so ``add`` stays O(1).
    """

    def __init__(self, capacity: int = 200000):
        self.capacity = capacity
        self.data: list = []
        self._start = 0

    def __len__(self) -> int:
        return len(self.data) - self._start

    def add(self, *transition) -> None:
        self.data.append(tuple(np.asarray(item).copy() for item in transition))
        if len(self.data) - self._start > self.capacity:
            self._start += 1
        if self._start > 65536:  # amortized compaction keeps add() O(1)
            self.data = self.data[self._start:]
            self._start = 0

    def pop_oldest(self) -> None:
        """Drop the oldest transition (used by bounded DAgger aggregation)."""
        if len(self) > 0:
            self._start += 1
            if self._start > 65536:
                self.data = self.data[self._start:]
                self._start = 0

    def sample(self, batch_size: int, device: torch.device):
        size = len(self)
        indices = np.random.choice(size, batch_size, replace=False)
        base = self._start
        columns = list(zip(*(self.data[base + index] for index in indices)))
        return tuple(
            torch.as_tensor(np.stack(column), dtype=torch.float32, device=device)
            for column in columns
        )


class DeterministicActor(nn.Module):
    def __init__(self, obs_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.Tanh(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)


class JointCritic(nn.Module):
    def __init__(self, global_obs_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(global_obs_dim + 6, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, observations: torch.Tensor, actions: torch.Tensor):
        inputs = torch.cat((observations.flatten(1), actions.flatten(1)), dim=1)
        return self.network(inputs).squeeze(1)


class IRDiscriminator(nn.Module):
    """Global state-action critic from the migrated IRL baseline.

    WGAN-GP form: the output is an unbounded Wasserstein score (no sigmoid) so
    the critic never saturates against a finite target. As long as the policy
    keeps changing, the critic retains a nonzero gradient signal for the
    policy, unlike a BCE discriminator that converges onto its soft targets
    and then freezes (expert=0.8/policy=0.2 exactly, imitation gradient ~0).
    """

    def __init__(self, obs_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear((obs_dim + 2) * 3, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, observations: torch.Tensor, actions: torch.Tensor):
        values = torch.cat((observations.flatten(1), actions.flatten(1)), dim=1)
        return self.network(values).squeeze(1)


class IRLAgent(PredictionFeatures):
    baseline_name = "no_graph_irl_maddpg"

    def __init__(self, config: TaskConfig, device: str = "cpu"):
        self.device = torch.device(device)
        self._init_prediction(config, self.device)
        self.actor = DeterministicActor(self.actor_obs_dim).to(self.device)
        self.target_actor = DeterministicActor(self.actor_obs_dim).to(self.device)
        global_dim = self.actor_obs_dim * 3
        self.critics = nn.ModuleList(
            JointCritic(global_dim).to(self.device) for _ in range(3)
        )
        self.target_critics = nn.ModuleList(
            JointCritic(global_dim).to(self.device) for _ in range(3)
        )
        self.target_actor.load_state_dict(self.actor.state_dict())
        for target, source in zip(self.target_critics, self.critics):
            target.load_state_dict(source.state_dict())
        self.discriminator = IRDiscriminator(self.actor_obs_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=5e-4)
        self.critic_optimizers = [
            torch.optim.Adam(critic.parameters(), lr=1e-3) for critic in self.critics
        ]
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=1e-4
        )

    def soft_update(self, tau: float = 0.005) -> None:
        pairs = [(self.target_actor, self.actor)] + list(
            zip(self.target_critics, self.critics)
        )
        with torch.no_grad():
            for target, source in pairs:
                for target_parameter, parameter in zip(
                    target.parameters(), source.parameters()
                ):
                    target_parameter.mul_(1.0 - tau).add_(parameter, alpha=tau)

    def save(self, path: str, extra: Optional[Dict[str, object]] = None) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": ADDITIONAL_CHECKPOINT_VERSION,
                "baseline": self.baseline_name,
                "task_config": self.config.to_dict(),
                "physics": CrazyfliePhysics().__dict__,
                "obs_dim": OBS_DIM,
                "actor_obs_dim": self.actor_obs_dim,
                "actor": self.actor.state_dict(),
                "critics": [critic.state_dict() for critic in self.critics],
                "epn": self.epn.state_dict(),
                "discriminator": self.discriminator.state_dict(),
                "extra": extra or {},
            },
            destination,
        )


class SituationEncoder(nn.Module):
    """Transformer encoder over three pursuers and one evader."""

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.type_embedding = nn.Embedding(2, 4)
        self.input_projection = nn.Linear(8, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.normalization = nn.LayerNorm(d_model)

    def forward(self, positions: torch.Tensor, velocities: torch.Tensor):
        type_ids = torch.tensor([0, 0, 0, 1], device=positions.device)
        types = self.type_embedding(type_ids).unsqueeze(0)
        types = types.expand(positions.shape[0], -1, -1)
        inputs = torch.cat((positions, velocities, types), dim=-1)
        return self.normalization(self.transformer(self.input_projection(inputs)))


class TargetAssignmentNetwork(nn.Module):
    """Differentiable assignment retained even though 3V1 has one target."""

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.prototype = nn.Parameter(torch.randn(1, d_model))
        self.projection = nn.Linear(d_model, d_model)

    def forward(self, pursuer_embeddings: torch.Tensor):
        scores = torch.einsum(
            "bnd,td->bnt", self.projection(pursuer_embeddings), self.prototype
        )
        return torch.softmax(scores, dim=-1)


class PathControlNetwork(nn.Module):
    def __init__(self, obs_dim: int, d_model: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim + d_model + 1, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.Tanh(),
        )

    def forward(self, obs, assignment, embedding):
        return self.network(torch.cat((obs, assignment, embedding), dim=-1))


class IQLDValueNet(nn.Module):
    def __init__(self, d_model: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(d_model * 4, 256), nn.ReLU(), nn.Linear(256, 1)
        )

    def forward(self, embeddings):
        return self.network(embeddings.flatten(1)).squeeze(1)


class IQLDQNet(nn.Module):
    def __init__(self, d_model: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(d_model * 4 + 9, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, embeddings, assignment, actions):
        inputs = torch.cat(
            (embeddings.flatten(1), assignment.flatten(1), actions.flatten(1)),
            dim=1,
        )
        return self.network(inputs).squeeze(1)


class IQLDAgent(PredictionFeatures):
    baseline_name = "no_graph_iqld_maddpg"

    def __init__(self, config: TaskConfig, device: str = "cpu", d_model: int = 64):
        self.device = torch.device(device)
        self.d_model = d_model
        self._init_prediction(config, self.device)
        self.situation_encoder = SituationEncoder(d_model).to(self.device)
        self.assignment = TargetAssignmentNetwork(d_model).to(self.device)
        self.path_controller = PathControlNetwork(self.actor_obs_dim, d_model).to(
            self.device
        )
        self.value = IQLDValueNet(d_model).to(self.device)
        self.target_value = IQLDValueNet(d_model).to(self.device)
        self.q1 = IQLDQNet(d_model).to(self.device)
        self.q2 = IQLDQNet(d_model).to(self.device)
        self.target_value.load_state_dict(self.value.state_dict())
        policy_parameters = list(self.situation_encoder.parameters())
        policy_parameters += list(self.assignment.parameters())
        policy_parameters += list(self.path_controller.parameters())
        self.policy_optimizer = torch.optim.Adam(policy_parameters, lr=3e-4)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=3e-4)
        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=3e-4
        )

    def policy(self, actor_obs, positions, velocities):
        embeddings = self.situation_encoder(positions, velocities)
        assignments = self.assignment(embeddings[:, :3])
        actions = []
        for agent in range(3):
            actions.append(
                self.path_controller(
                    actor_obs[:, agent],
                    assignments[:, agent],
                    embeddings[:, agent],
                )
            )
        return torch.stack(actions, dim=1), embeddings, assignments

    def save(self, path: str, extra: Optional[Dict[str, object]] = None) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": ADDITIONAL_CHECKPOINT_VERSION,
                "baseline": self.baseline_name,
                "task_config": self.config.to_dict(),
                "physics": CrazyfliePhysics().__dict__,
                "obs_dim": OBS_DIM,
                "actor_obs_dim": self.actor_obs_dim,
                "d_model": self.d_model,
                "situation_encoder": self.situation_encoder.state_dict(),
                "assignment": self.assignment.state_dict(),
                "path_controller": self.path_controller.state_dict(),
                "value": self.value.state_dict(),
                "q1": self.q1.state_dict(),
                "q2": self.q2.state_dict(),
                "epn": self.epn.state_dict(),
                "extra": extra or {},
            },
            destination,
        )


class IRLRuntime:
    baseline_name = IRLAgent.baseline_name

    def __init__(self, checkpoint: str, device: str = "cpu"):
        self.device = torch.device(device)
        data = load_checkpoint(checkpoint, self.device)
        if data.get("baseline") != self.baseline_name:
            raise ValueError("checkpoint is not a no_graph_irl_maddpg model")
        self.config = TaskConfig.from_dict(data["task_config"])
        self.agent = IRLAgent(self.config, device)
        self.agent.actor.load_state_dict(data["actor"])
        self.agent.epn.load_state_dict(data["epn"])
        self.agent.actor.eval()
        self.agent.epn.eval()

    def reset(self):
        self.agent.reset_history()

    def actions(self, positions, velocities, origin, virtual_obstacles=None):
        state = np.concatenate(
            (
                (positions[3] - origin) / self.config.arena_half_extent,
                velocities[3] / self.config.velocity_scale,
            )
        )
        self.agent.update_history(state)
        prediction, confidence = self.agent.predict()
        observations = build_observations(
            positions, velocities, origin, self.config, virtual_obstacles
        )
        actor_obs = self.agent.augment(observations, prediction, confidence)
        with torch.no_grad():
            actions = self.agent.actor(
                torch.as_tensor(actor_obs, dtype=torch.float32, device=self.device)
            )
        return actions.cpu().numpy()


class IQLDRuntime:
    baseline_name = IQLDAgent.baseline_name

    def __init__(self, checkpoint: str, device: str = "cpu"):
        self.device = torch.device(device)
        data = load_checkpoint(checkpoint, self.device)
        if data.get("baseline") != self.baseline_name:
            raise ValueError("checkpoint is not a no_graph_iqld_maddpg model")
        self.config = TaskConfig.from_dict(data["task_config"])
        self.agent = IQLDAgent(self.config, device, int(data.get("d_model", 64)))
        for name in (
            "situation_encoder",
            "assignment",
            "path_controller",
            "value",
            "q1",
            "q2",
            "epn",
        ):
            getattr(self.agent, name).load_state_dict(data[name])
            getattr(self.agent, name).eval()

    def reset(self):
        self.agent.reset_history()

    def actions(self, positions, velocities, origin, virtual_obstacles=None):
        state = np.concatenate(
            (
                (positions[3] - origin) / self.config.arena_half_extent,
                velocities[3] / self.config.velocity_scale,
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
            actions, _, _ = self.agent.policy(
                torch.as_tensor(
                    actor_obs[None], dtype=torch.float32, device=self.device
                ),
                torch.as_tensor(
                    normalized_pos[None],
                    dtype=torch.float32,
                    device=self.device,
                ),
                torch.as_tensor(
                    normalized_vel[None],
                    dtype=torch.float32,
                    device=self.device,
                ),
            )
        return actions[0].cpu().numpy()
