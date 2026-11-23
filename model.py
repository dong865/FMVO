"""MAPPO + evader prediction model and deployment checkpoint loader."""

from collections import deque
from pathlib import Path
from typing import Deque, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from .config import CrazyfliePhysics, TaskConfig
from .features import OBS_DIM, build_observations


CHECKPOINT_VERSION = 1


class SquashedNormal:
    """Tanh-squashed Gaussian with corrected action log-probability."""

    def __init__(self, location: torch.Tensor, scale: torch.Tensor):
        self.base = Normal(location, scale)

    def rsample(self) -> torch.Tensor:
        return torch.tanh(self.base.rsample())

    def sample(self) -> torch.Tensor:
        return torch.tanh(self.base.sample())

    @property
    def mean(self) -> torch.Tensor:
        return torch.tanh(self.base.mean)

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        epsilon = torch.finfo(actions.dtype).eps
        bounded = actions.clamp(-1.0 + epsilon, 1.0 - epsilon)
        latent = torch.atanh(bounded)
        correction = torch.log1p(-bounded.square() + epsilon)
        return self.base.log_prob(latent) - correction

    def entropy(self) -> torch.Tensor:
        return -self.log_prob(self.rsample())


class EvaderPredictionNetwork(nn.Module):
    def __init__(self, future_steps: int, hidden_dim: int = 64):
        super().__init__()
        self.future_steps = future_steps
        self.lstm = nn.LSTM(4, hidden_dim, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, future_steps * 2),
        )
        self.confidence = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, history: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, (hidden, _) = self.lstm(history)
        latent = hidden[-1]
        prediction = self.decoder(latent).view(-1, self.future_steps, 2)
        return prediction, self.confidence(latent)


class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int = 2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.7))

    def distribution(self, observations: torch.Tensor) -> SquashedNormal:
        mean = self.network(observations)
        std = self.log_std.exp().clamp(0.05, 0.8).expand_as(mean)
        return SquashedNormal(mean, std)

    def act(self, observations: torch.Tensor, deterministic: bool = False):
        distribution = self.distribution(observations)
        actions = distribution.mean if deterministic else distribution.sample()
        log_prob = distribution.log_prob(actions).sum(-1)
        return actions, log_prob


class CentralCritic(nn.Module):
    def __init__(self, global_obs_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(global_obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )

    def forward(self, global_observations: torch.Tensor) -> torch.Tensor:
        return self.network(global_observations).squeeze(-1)


class MAPPOAgent:
    """Trainable policy retaining the MAPPO + EPN baseline structure."""

    def __init__(self, config: TaskConfig, device: str = "cpu"):
        self.config = config
        self.device = torch.device(device)
        self.actor_obs_dim = OBS_DIM + config.prediction_steps * 2 + 1
        self.actor = Actor(self.actor_obs_dim).to(self.device)
        self.critic = CentralCritic(self.actor_obs_dim * config.num_pursuers).to(
            self.device
        )
        self.epn = EvaderPredictionNetwork(config.prediction_steps).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.epn_optimizer = torch.optim.Adam(self.epn.parameters(), lr=1e-3)
        self.history: Deque[np.ndarray] = deque(maxlen=config.history_steps)

    def reset_history(self) -> None:
        self.history.clear()

    def update_history(self, evader_state: np.ndarray) -> None:
        self.history.append(np.asarray(evader_state, dtype=np.float32))

    def predict_evader(self) -> Tuple[np.ndarray, float]:
        if len(self.history) < 3:
            return np.zeros((self.config.prediction_steps, 2), np.float32), 0.0
        history = torch.as_tensor(
            np.asarray(self.history), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            prediction, confidence = self.epn(history)
        return prediction[0].cpu().numpy(), float(confidence[0, 0].cpu())

    def actor_observations(
        self,
        observations: np.ndarray,
        prediction: np.ndarray,
        confidence: float,
    ) -> np.ndarray:
        extra = np.concatenate(
            (np.asarray(prediction, np.float32).reshape(-1), [confidence])
        ).astype(np.float32)
        return np.concatenate(
            (
                observations,
                np.repeat(extra[None, :], self.config.num_pursuers, axis=0),
            ),
            axis=1,
        )

    def save(self, path: str, extra: Optional[Dict[str, object]] = None) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": CHECKPOINT_VERSION,
                "task_config": self.config.to_dict(),
                "physics": CrazyfliePhysics().__dict__,
                "obs_dim": OBS_DIM,
                "actor_obs_dim": self.actor_obs_dim,
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "epn": self.epn.state_dict(),
                "extra": extra or {},
            },
            destination,
        )


class PolicyRuntime:
    """Deterministic inference for offline checks and real flight."""

    def __init__(self, checkpoint: str, device: str = "cpu"):
        self.device = torch.device(device)
        try:
            data = torch.load(checkpoint, map_location=self.device, weights_only=False)
        except TypeError:  # PyTorch < 2.0
            data = torch.load(checkpoint, map_location=self.device)
        if data.get("schema_version") != CHECKPOINT_VERSION:
            raise ValueError("unsupported or legacy checkpoint; retrain with train.py")
        if data.get("obs_dim") != OBS_DIM:
            raise ValueError(
                "checkpoint observation definition does not match this code"
            )
        self.config = TaskConfig.from_dict(data["task_config"])
        self.agent = MAPPOAgent(self.config, device=device)
        self.agent.actor.load_state_dict(data["actor"])
        self.agent.epn.load_state_dict(data["epn"])
        self.agent.actor.eval()
        self.agent.epn.eval()

    def reset(self) -> None:
        self.agent.reset_history()

    def actions(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        origin: np.ndarray,
        virtual_obstacles=None,
    ) -> np.ndarray:
        local_evader = positions[3] - origin
        evader_state = np.concatenate(
            (
                local_evader / self.config.arena_half_extent,
                velocities[3] / self.config.velocity_scale,
            )
        ).astype(np.float32)
        self.agent.update_history(evader_state)
        prediction, confidence = self.agent.predict_evader()
        observations = build_observations(
            positions, velocities, origin, self.config, virtual_obstacles
        )
        actor_obs = self.agent.actor_observations(observations, prediction, confidence)
        tensor = torch.as_tensor(actor_obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action, _ = self.agent.actor.act(tensor, deterministic=True)
        return action.cpu().numpy()
