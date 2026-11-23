"""Private-goal artificial-potential-field controller for the evader."""

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from .config import TaskConfig


@dataclass(frozen=True)
class PotentialFieldBreakdown:
    """Individual acceleration requests before the final vector limit."""

    goal: np.ndarray
    pursuers: np.ndarray
    obstacles: np.ndarray
    boundary: np.ndarray
    tangent: np.ndarray
    total: np.ndarray


def _unit(vector: np.ndarray, fallback: np.ndarray) -> Tuple[np.ndarray, float]:
    distance = float(np.linalg.norm(vector))
    if distance <= 1e-9:
        return np.asarray(fallback, dtype=np.float64), 0.0
    return vector / distance, distance


def _smooth_repulsion(
    clearance: float,
    influence_distance: float,
    gain: float,
) -> float:
    if clearance >= influence_distance:
        return 0.0
    normalized = 1.0 - max(clearance, 0.0) / influence_distance
    return gain * normalized * normalized


class PotentialFieldEvader:
    """Move toward a private goal while avoiding threats and local minima."""

    def __init__(self, config: TaskConfig):
        self.config = config

    def acceleration(
        self,
        positions: np.ndarray,
        goal: np.ndarray,
        origin: np.ndarray,
        obstacles: Sequence[Tuple[float, float, float]],
    ) -> np.ndarray:
        return self.with_breakdown(positions, goal, origin, obstacles).total

    def with_breakdown(
        self,
        positions: np.ndarray,
        goal: np.ndarray,
        origin: np.ndarray,
        obstacles: Sequence[Tuple[float, float, float]],
    ) -> PotentialFieldBreakdown:
        positions = np.asarray(positions, dtype=np.float64)
        goal = np.asarray(goal, dtype=np.float64)
        origin = np.asarray(origin, dtype=np.float64)
        evader = positions[-1]

        goal_direction, goal_distance = _unit(
            goal - evader, np.asarray((1.0, 0.0))
        )
        slowdown_distance = max(2.0 * self.config.evader_goal_radius, 1e-9)
        goal_strength = self.config.evader_goal_attraction_gain * min(
            goal_distance / slowdown_distance, 1.0
        )
        goal_term = goal_direction * goal_strength

        pursuer_term = np.zeros(2, dtype=np.float64)
        fallback_directions = (
            np.asarray((1.0, 0.0)),
            np.asarray((-0.5, 3.0 ** 0.5 / 2.0)),
            np.asarray((-0.5, -(3.0 ** 0.5) / 2.0)),
        )
        for index, pursuer in enumerate(positions[:-1]):
            direction, distance = _unit(
                evader - pursuer,
                fallback_directions[index % len(fallback_directions)],
            )
            pursuer_term += direction * _smooth_repulsion(
                distance,
                self.config.evader_pursuer_influence_distance,
                self.config.evader_pursuer_repulsion_gain,
            )

        obstacle_term = np.zeros(2, dtype=np.float64)
        blocking_strength = 0.0
        tangent_sign = 1.0
        for obstacle_x, obstacle_y, radius in obstacles:
            centre = origin + np.asarray((obstacle_x, obstacle_y))
            away, centre_distance = _unit(
                evader - centre, -goal_direction
            )
            clearance = centre_distance - float(radius)
            obstacle_strength = _smooth_repulsion(
                clearance,
                self.config.evader_obstacle_influence_distance,
                self.config.evader_obstacle_repulsion_gain,
            )
            obstacle_term += away * obstacle_strength

            relative = centre - evader
            along_route = float(np.dot(relative, goal_direction))
            lateral_cross = (
                goal_direction[0] * relative[1]
                - goal_direction[1] * relative[0]
            )
            corridor = (
                float(radius)
                + self.config.evader_obstacle_influence_distance
            )
            if 0.0 < along_route < goal_distance and abs(lateral_cross) < corridor:
                candidate = max(
                    obstacle_strength,
                    self.config.evader_obstacle_repulsion_gain
                    * (1.0 - abs(lateral_cross) / corridor),
                )
                if candidate > blocking_strength:
                    blocking_strength = candidate
                    tangent_sign = 1.0 if lateral_cross <= 0.0 else -1.0

        boundary_term = np.zeros(2, dtype=np.float64)
        extent = self.config.arena_half_extent
        influence = self.config.evader_boundary_influence_distance
        gain = self.config.evader_boundary_repulsion_gain
        relative_evader = evader - origin
        for axis in range(2):
            low_clearance = relative_evader[axis] + extent
            high_clearance = extent - relative_evader[axis]
            boundary_term[axis] += _smooth_repulsion(
                low_clearance, influence, gain
            )
            boundary_term[axis] -= _smooth_repulsion(
                high_clearance, influence, gain
            )

        subtotal = goal_term + pursuer_term + obstacle_term + boundary_term
        if (
            blocking_strength > 0.0
            or np.linalg.norm(subtotal) < self.config.evader_stagnation_threshold
        ):
            perpendicular = np.asarray((-goal_direction[1], goal_direction[0]))
            activation = max(blocking_strength, 1.0)
            tangent_term = (
                perpendicular
                * tangent_sign
                * self.config.evader_tangential_gain
                * min(activation, 1.0)
            )
        else:
            tangent_term = np.zeros(2, dtype=np.float64)

        requested = subtotal + tangent_term
        limit = self.config.evader_max_accel * self.config.evader_policy_scale
        norm = float(np.linalg.norm(requested))
        if limit <= 0.0:
            total = np.zeros(2, dtype=np.float64)
        elif norm > limit:
            total = requested * (limit / norm)
        else:
            total = requested
        return PotentialFieldBreakdown(
            goal=goal_term,
            pursuers=pursuer_term,
            obstacles=obstacle_term,
            boundary=boundary_term,
            tangent=tangent_term,
            total=total,
        )
