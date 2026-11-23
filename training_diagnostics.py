"""Shared episode diagnostics for every learning baseline."""

from typing import Dict

import numpy as np

from .curriculum import CurriculumStage


_REWARD_COMPONENTS = (
    "distance_progress",
    "angular_progress",
    "time",
    "action",
    "pairwise_warning",
    "obstacle_warning",
    "capture_candidate",
    "capture",
    "safety_failure",
    "escape_failure",
)


class EpisodeDiagnostics:
    """Accumulate reward and policy diagnostics without retaining trajectories."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._steps = 0
        self._reward_sums = {name: 0.0 for name in _REWARD_COMPONENTS}
        self._radial_alignment_sum = 0.0
        self._minimum_separation = float("inf")
        self._inter_drone_failure_steps = 0
        self._obstacle_collision_steps = 0

    def update(
        self,
        info: Dict[str, object],
        actions: np.ndarray,
        positions: np.ndarray,
    ) -> None:
        actions = np.asarray(actions, dtype=np.float32)
        positions = np.asarray(positions, dtype=np.float32)
        if actions.shape != (3, 2):
            raise ValueError("actions must have shape (3, 2)")
        if positions.shape != (4, 2):
            raise ValueError("positions must have shape (4, 2)")

        for name in _REWARD_COMPONENTS:
            values = np.asarray(info[f"reward_{name}"], dtype=np.float32)
            if values.shape != (3,):
                raise ValueError(f"reward_{name} must have shape (3,)")
            self._reward_sums[name] += float(values.mean())

        inward = positions[3] - positions[:3]
        inward_norm = np.linalg.norm(inward, axis=1, keepdims=True)
        action_norm = np.linalg.norm(actions, axis=1, keepdims=True)
        valid = (inward_norm[:, 0] > 1e-8) & (action_norm[:, 0] > 1e-8)
        alignment = np.zeros(3, dtype=np.float32)
        alignment[valid] = np.sum(
            inward[valid] / inward_norm[valid]
            * actions[valid] / action_norm[valid],
            axis=1,
        )
        self._radial_alignment_sum += float(alignment.mean())
        self._minimum_separation = min(
            self._minimum_separation,
            float(info["minimum_inter_drone_distance"]),
        )
        obstacle_collision = bool(np.any(info.get("obstacle_collisions", False)))
        self._obstacle_collision_steps += int(obstacle_collision)
        self._inter_drone_failure_steps += int(
            bool(info.get("safety_failure", False)) and not obstacle_collision
        )
        self._steps += 1

    def tensorboard(self, stage: CurriculumStage) -> Dict[str, float]:
        denominator = max(self._steps, 1)
        result = {
            f"reward/{name}": value / denominator
            for name, value in self._reward_sums.items()
        }
        result.update(
            {
                "policy/radial_action_alignment": (
                    self._radial_alignment_sum / denominator
                ),
                "safety/minimum_separation": (
                    self._minimum_separation
                    if np.isfinite(self._minimum_separation)
                    else 0.0
                ),
                "safety/inter_drone_failure_steps": float(
                    self._inter_drone_failure_steps
                ),
                "safety/obstacle_collision_steps": float(
                    self._obstacle_collision_steps
                ),
                "curriculum/stage": float(stage.index),
                "curriculum/spawn_distance_min": float(
                    stage.config.pursuer_spawn_distance_range[0]
                ),
                "curriculum/spawn_distance_max": float(
                    stage.config.pursuer_spawn_distance_range[1]
                ),
                "curriculum/goal_min_distance": float(
                    stage.config.evader_goal_min_distance
                ),
                "curriculum/evader_policy_scale": float(
                    stage.config.evader_policy_scale
                ),
            }
        )
        return result
