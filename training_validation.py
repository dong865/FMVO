"""Deterministic final-task validation for training checkpoints."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .config import TaskConfig
from .environment import ThreeVOneEnv


TRAINING_VALIDATION_SEEDS = (607, 613, 619)
CURRENT_SCENARIO_CONTRACT_VERSION = 2


@dataclass(frozen=True)
class ValidationResult:
    capture_rate: float
    collision_free_rate: float
    minimum_separation: float
    evader_success_rate: float = 0.0
    timeout_rate: float = 0.0
    safety_failure_rate: float = 0.0
    mean_final_goal_progress: float = 0.0

    def tensorboard(self):
        return {
            "validation/capture_rate": self.capture_rate,
            "validation/collision_free_rate": self.collision_free_rate,
            "validation/minimum_separation": self.minimum_separation,
            "validation/evader_success_rate": self.evader_success_rate,
            "validation/timeout_rate": self.timeout_rate,
            "validation/safety_failure_rate": self.safety_failure_rate,
            "validation/mean_final_goal_progress": self.mean_final_goal_progress,
        }


def checkpoint_is_better(
    candidate: ValidationResult,
    incumbent: Optional[ValidationResult],
    minimum_collision_free_rate: float = 0.95,
    minimum_separation: float = 0.30,
) -> bool:
    """Order safe checkpoints by capture first, then safety margins."""

    if candidate.collision_free_rate < minimum_collision_free_rate:
        return False
    if candidate.minimum_separation < minimum_separation:
        return False
    if incumbent is None:
        return True
    return (
        candidate.capture_rate,
        candidate.collision_free_rate,
        candidate.minimum_separation,
    ) > (
        incumbent.capture_rate,
        incumbent.collision_free_rate,
        incumbent.minimum_separation,
    )


def evaluate_final_task(
    runtime,
    final_config: TaskConfig,
    episodes: int = 20,
    seeds: Sequence[int] = TRAINING_VALIDATION_SEEDS,
    scene_options=None,
) -> ValidationResult:
    """Evaluate deterministic policy actions on fresh final-task environments."""

    if episodes <= 0:
        raise ValueError("validation episodes must be positive")
    if not seeds:
        raise ValueError("validation seeds must not be empty")
    config = TaskConfig.from_dict(final_config.to_dict())
    if config.scenario_contract_version != CURRENT_SCENARIO_CONTRACT_VERSION:
        raise ValueError(
            "formal validation requires scenario contract v2; regenerate old "
            "expert data and checkpoints"
        )
    environments = [
        ThreeVOneEnv(
            TaskConfig.from_dict(config.to_dict()),
            seed=int(seed),
            scene_options=scene_options,
        )
        for seed in seeds
    ]
    captures = 0
    evader_successes = 0
    timeouts = 0
    safety_failures = 0
    final_goal_progress = 0.0
    collision_free_episodes = 0
    overall_minimum = float("inf")
    for episode in range(episodes):
        environment = environments[episode % len(environments)]
        environment.reset()
        runtime.reset()
        captured = False
        outcome = "timeout"
        collision_free = True
        episode_minimum = float("inf")
        for _ in range(config.max_episode_steps):
            positions, velocities = environment.control_state()
            actions = np.asarray(
                runtime.actions(
                    positions,
                    velocities,
                    environment.origin,
                    environment.virtual_obstacles,
                ),
                dtype=np.float32,
            )
            _, _, terminated, truncated, info = environment.step(actions)
            separation = float(info["minimum_inter_drone_distance"])
            episode_minimum = min(episode_minimum, separation)
            collision_free = bool(
                collision_free
                and separation >= config.safety_emergency_distance
                and not np.any(info["obstacle_collisions"])
            )
            if terminated or truncated:
                captured = bool(info["captured"])
                outcome = str(info["outcome"])
                break
        captures += int(captured)
        evader_successes += int(outcome == "evader_succeeded")
        timeouts += int(outcome == "timeout")
        safety_failures += int(outcome == "safety_failure")
        final_goal_progress += (
            1.0
            - float(info["distance_to_goal"])
            / max(environment.initial_goal_distance, 1e-8)
        )
        collision_free_episodes += int(collision_free)
        overall_minimum = min(overall_minimum, episode_minimum)
    return ValidationResult(
        capture_rate=captures / episodes,
        collision_free_rate=collision_free_episodes / episodes,
        minimum_separation=overall_minimum,
        evader_success_rate=evader_successes / episodes,
        timeout_rate=timeouts / episodes,
        safety_failure_rate=safety_failures / episodes,
        mean_final_goal_progress=final_goal_progress / episodes,
    )


def validate_checkpoint(
    baseline: str,
    checkpoint: Path,
    final_config: TaskConfig,
    episodes: int,
    device: str,
    scene_options=None,
) -> ValidationResult:
    """Load the deployment runtime and evaluate the exact saved checkpoint."""

    from .runtime import load_runtime

    runtime = load_runtime(baseline, str(checkpoint), device=device)
    return evaluate_final_task(
        runtime, final_config, episodes=episodes, scene_options=scene_options
    )


def add_validation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--validation-episodes", type=int, default=20)
