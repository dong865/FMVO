"""Pure, decomposed reward shared by the learning baselines."""

from dataclasses import dataclass, fields
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .config import TaskConfig


@dataclass(frozen=True)
class RewardBreakdown:
    """Per-pursuer reward components for one environment step."""

    distance_progress: np.ndarray
    angular_progress: np.ndarray
    time: np.ndarray
    action: np.ndarray
    pairwise_warning: np.ndarray
    obstacle_warning: np.ndarray
    capture_candidate: np.ndarray
    capture: np.ndarray
    safety_failure: np.ndarray
    escape_failure: np.ndarray

    @property
    def total(self) -> np.ndarray:
        total = np.zeros_like(self.distance_progress, dtype=np.float32)
        for component in fields(self):
            total += getattr(self, component.name)
        return total

    def info(self) -> Dict[str, np.ndarray]:
        return {
            f"reward_{component.name}": getattr(self, component.name).copy()
            for component in fields(self)
        }


def _common(value: float, count: int) -> np.ndarray:
    return np.full(count, value, dtype=np.float32)


def _pairwise_urgency(config: TaskConfig, positions: np.ndarray) -> float:
    delta = positions[:, None, :] - positions[None, :, :]
    distances = np.linalg.norm(delta, axis=-1)
    row, column = np.triu_indices(positions.shape[0], k=1)
    urgency = np.clip(
        (config.safety_warning_distance - distances[row, column])
        / (config.safety_warning_distance - config.safety_emergency_distance),
        0.0,
        1.0,
    )
    return float(urgency.sum())


def _obstacle_urgency(
    config: TaskConfig,
    positions: np.ndarray,
    origin: np.ndarray,
    virtual_obstacles: Sequence[Tuple[float, float, float]],
) -> float:
    total = 0.0
    for x, y, radius in virtual_obstacles:
        center = origin + np.array([x, y], dtype=np.float32)
        clearance = np.linalg.norm(positions - center, axis=1)
        clearance -= radius + config.obstacle_safety_margin
        urgency = np.clip(
            (config.obstacle_warning_margin - clearance)
            / config.obstacle_warning_margin,
            0.0,
            1.0,
        )
        total += float(urgency.sum())
    return total


def compute_reward(
    config: TaskConfig,
    positions: np.ndarray,
    origin: np.ndarray,
    actions: np.ndarray,
    progress: float,
    angular_progress: float,
    capture_candidate: bool,
    captured: bool,
    safety_failure: bool,
    virtual_obstacles: Optional[
        Sequence[Tuple[float, float, float]]
    ] = None,
    evader_succeeded: bool = False,
) -> RewardBreakdown:
    """Compute the approved 50 Hz reward without mutating environment state."""

    positions = np.asarray(positions, dtype=np.float32)
    origin = np.asarray(origin, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    if positions.shape != (config.num_entities, 2):
        raise ValueError(
            f"positions must have shape ({config.num_entities}, 2)"
        )
    if origin.shape != (2,):
        raise ValueError("origin must have shape (2,)")
    if actions.shape != (config.num_pursuers, 2):
        raise ValueError(
            f"actions must have shape ({config.num_pursuers}, 2)"
        )

    dt = config.dt
    pairwise_urgency = _pairwise_urgency(config, positions)
    obstacles = (
        config.virtual_obstacles
        if virtual_obstacles is None
        else tuple(virtual_obstacles)
    )
    obstacle_urgency = _obstacle_urgency(
        config, positions, origin, obstacles
    )
    common = lambda value: _common(value, config.num_pursuers)
    return RewardBreakdown(
        distance_progress=common(
            config.distance_progress_reward_scale * progress
        ),
        angular_progress=common(
            config.angular_progress_reward_scale * angular_progress
        ),
        time=common(-config.time_penalty_rate * dt),
        action=(
            -config.action_penalty_rate
            * np.sum(actions * actions, axis=1)
            * dt
        ).astype(np.float32),
        pairwise_warning=common(
            -config.safety_warning_penalty_rate * pairwise_urgency * dt
        ),
        obstacle_warning=common(
            -config.obstacle_warning_penalty_rate * obstacle_urgency * dt
        ),
        capture_candidate=common(
            config.capture_candidate_reward_rate * dt * capture_candidate
        ),
        capture=common(config.capture_reward * captured),
        safety_failure=common(
            -config.safety_failure_penalty * safety_failure
        ),
        escape_failure=common(
            -config.evader_escape_failure_penalty * evader_succeeded
        ),
    )
