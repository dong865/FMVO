"""Explicit collection of qualified Game Theory demonstrations."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple

import numpy as np

from .config import CrazyfliePhysics, TaskConfig
from .environment import ThreeVOneEnv
from .expert_dataset import (
    ExpertEpisode,
    append_expert_episodes,
    episode_is_qualified,
    load_expert_dataset,
)
from .paper_game_theory import GameTheoryRuntime
from .training_common import zero_prediction_observations


@dataclass(frozen=True)
class EpisodeRollout:
    """Outcome and transitions from one expert attempt."""

    identifier: str
    transitions: Tuple[tuple, ...]
    captured: bool
    collision_free: bool
    obstacle_collision: bool
    minimum_separation: float


@dataclass(frozen=True)
class CollectionSummary:
    """Summary of one explicit collection request."""

    attempted: int
    accepted: int
    rejected: int
    total_persisted: int
    minimum_separation: float


def rollout_game_theory_episode(
    config: TaskConfig,
    seed: int,
    attempt: int,
    max_steps: int,
    scene_options=None,
) -> EpisodeRollout:
    """Roll out one Game Theory episode in the production training dynamics."""
    environment = ThreeVOneEnv(config, seed=seed, scene_options=scene_options)
    controller = GameTheoryRuntime(config)
    observations, _ = environment.reset()
    controller.reset()
    transitions = []
    captured = False
    collision_free = True
    obstacle_collision = False
    minimum_separation = float("inf")
    for _ in range(max_steps):
        positions, velocities = environment.control_state()
        actor_observations = zero_prediction_observations(observations, config)
        actions = controller.actions(
            positions,
            velocities,
            environment.origin,
            environment.virtual_obstacles,
        )
        next_observations, rewards, terminated, truncated, info = environment.step(
            actions
        )
        next_positions, next_velocities = environment.control_state()
        transitions.append(
            (
                positions,
                velocities,
                actor_observations,
                actions,
                rewards,
                next_positions,
                next_velocities,
                zero_prediction_observations(next_observations, config),
                np.float32(terminated or truncated),
            )
        )
        observations = next_observations
        separation = float(info["minimum_inter_drone_distance"])
        collided = bool(np.any(info["obstacle_collisions"]))
        minimum_separation = min(minimum_separation, separation)
        obstacle_collision = obstacle_collision or collided
        collision_free = bool(
            collision_free
            and separation >= config.safety_emergency_distance
            and not collided
        )
        if terminated or truncated:
            captured = bool(info["captured"])
            break
    return EpisodeRollout(
        identifier=f"seed{seed}-attempt{attempt}",
        transitions=tuple(transitions),
        captured=captured,
        collision_free=collision_free,
        obstacle_collision=obstacle_collision,
        minimum_separation=minimum_separation,
    )


def collect_qualified_expert_dataset(
    config: TaskConfig,
    path,
    target_new_episodes: int,
    max_attempts: int,
    seed: int,
    max_steps: int,
    episode_runner: Callable = None,
    scene_options=None,
) -> CollectionSummary:
    if episode_runner is None:
        if scene_options is not None:
            from functools import partial

            episode_runner = partial(
                rollout_game_theory_episode, scene_options=scene_options
            )
        else:
            episode_runner = rollout_game_theory_episode
    """Collect and atomically persist only qualified expert episodes."""
    if target_new_episodes <= 0:
        raise ValueError("target_new_episodes must be positive")
    if max_attempts < target_new_episodes:
        raise ValueError("max_attempts must not be below target_new_episodes")
    path = Path(path).expanduser()
    accepted = 0
    rejected = 0
    minimum_separation = float("inf")
    for attempt in range(max_attempts):
        rollout = episode_runner(config, seed + attempt, attempt, max_steps)
        minimum_separation = min(minimum_separation, rollout.minimum_separation)
        qualified = episode_is_qualified(
            rollout.captured,
            rollout.collision_free,
            rollout.obstacle_collision,
            rollout.minimum_separation,
            rollout.transitions,
        )
        episodes = ()
        if qualified:
            episodes = (
                ExpertEpisode(
                    identifier=rollout.identifier,
                    seed=seed + attempt,
                    attempt=attempt,
                    minimum_separation=rollout.minimum_separation,
                    transitions=rollout.transitions,
                ),
            )
        before = (
            len(load_expert_dataset(path, expected_config=config).episodes)
            if path.exists()
            else 0
        )
        dataset = append_expert_episodes(
            path,
            config,
            episodes,
            physics_config=CrazyfliePhysics(),
            attempted_episodes=1,
        )
        added = len(dataset.episodes) > before
        accepted += int(qualified and added)
        rejected += int(not (qualified and added))
        if accepted >= target_new_episodes:
            return CollectionSummary(
                attempted=attempt + 1,
                accepted=accepted,
                rejected=rejected,
                total_persisted=len(dataset.episodes),
                minimum_separation=minimum_separation,
            )
    total = len(load_expert_dataset(path, expected_config=config).episodes)
    raise RuntimeError(
        f"collected {accepted} of {target_new_episodes} qualified episodes "
        f"after {max_attempts} attempts; {total} episodes are persisted"
    )
