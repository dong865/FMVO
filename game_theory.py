"""50 Hz IBR/potential-game baseline adapted to the ROS2 state contract."""

from itertools import permutations
from typing import Optional

import numpy as np

from .config import TaskConfig
from .features import point_in_triangle
from .paper_game_theory import GameTheoryRuntime

__all__ = ["GameTheoryRuntime"]


class AlphaBetaFilter:
    """Constant-velocity tracker serving the same role as the original KF."""

    def __init__(self, dt: float, alpha: float = 0.75, beta: float = 0.20):
        self.dt = dt
        self.alpha = alpha
        self.beta = beta
        self.position = np.zeros(2, dtype=np.float32)
        self.velocity = np.zeros(2, dtype=np.float32)
        self.initialized = False

    def reset(self) -> None:
        self.initialized = False
        self.position.fill(0.0)
        self.velocity.fill(0.0)

    def update(self, measured_position: np.ndarray) -> None:
        measured_position = np.asarray(measured_position, dtype=np.float32)
        if not self.initialized:
            self.position[:] = measured_position
            self.initialized = True
            return
        predicted = self.position + self.velocity * self.dt
        residual = measured_position - predicted
        self.position[:] = predicted + self.alpha * residual
        self.velocity[:] += self.beta * residual / self.dt

    def predict(self, horizon: float) -> np.ndarray:
        return self.position + self.velocity * horizon


class LegacyGameTheoryRuntime:
    """Three-pursuer IBR controller with formation potential utilities."""

    baseline_name = "game_theory"

    def __init__(self, config: Optional[TaskConfig] = None):
        self.config = config or TaskConfig()
        self.filter = AlphaBetaFilter(self.config.dt)

    def reset(self) -> None:
        self.filter.reset()

    @staticmethod
    def _angle_difference(first: float, second: float) -> float:
        return abs((first - second + np.pi) % (2.0 * np.pi) - np.pi)

    def _formation_targets(
        self, positions: np.ndarray, predicted_evader: np.ndarray
    ) -> np.ndarray:
        relative = positions[:3] - predicted_evader
        current_angles = np.arctan2(relative[:, 1], relative[:, 0])
        escape = self.filter.velocity
        if np.linalg.norm(escape) > 1e-4:
            base_angle = np.arctan2(escape[1], escape[0]) + np.pi
        else:
            base_angle = float(np.mean(current_angles))
        ideal_angles = base_angle + np.arange(3) * 2.0 * np.pi / 3.0
        role_order = min(
            permutations(range(3)),
            key=lambda order: sum(
                self._angle_difference(
                    float(current_angles[i]), float(ideal_angles[order[i]])
                )
                for i in range(3)
            ),
        )
        distances = np.linalg.norm(relative, axis=1)
        enclosed = point_in_triangle(predicted_evader, positions[:3])
        if enclosed:
            radius = np.clip(
                distances.max() * 0.72,
                self.config.capture_radius * 0.65,
                self.config.capture_radius,
            )
        else:
            radius = np.clip(
                distances.max() * 0.45,
                self.config.capture_radius * 0.65,
                self.config.capture_radius * 1.8,
            )
        targets = []
        for i, role in enumerate(role_order):
            angle = ideal_angles[role]
            targets.append(
                predicted_evader
                + radius * np.array([np.cos(angle), np.sin(angle)])
            )
        return np.asarray(targets, dtype=np.float32)

    def _utility(
        self,
        agent: int,
        candidate: np.ndarray,
        actions: np.ndarray,
        positions: np.ndarray,
        velocities: np.ndarray,
        targets: np.ndarray,
        origin: np.ndarray,
    ) -> float:
        horizon = 0.40
        trial_actions = actions.copy()
        trial_actions[agent] = candidate
        predicted_positions = (
            positions[:3]
            + velocities[:3] * horizon
            + 0.5 * trial_actions * horizon * horizon
        )
        own_target_cost = np.linalg.norm(
            predicted_positions[agent] - targets[agent]
        )
        separation_cost = 0.0
        for other in range(3):
            if other == agent:
                continue
            distance = np.linalg.norm(
                predicted_positions[agent] - predicted_positions[other]
            )
            if distance < self.config.collision_radius * 1.5:
                separation_cost += 8.0 * (
                    self.config.collision_radius * 1.5 - distance
                )
        local = predicted_positions[agent] - origin
        boundary_excess = np.maximum(
            np.abs(local) - self.config.arena_half_extent * 0.9, 0.0
        ).sum()
        obstacle_cost = 0.0
        predicted_local = predicted_positions[agent] - origin
        for ox, oy, radius in self.config.virtual_obstacles:
            distance = np.linalg.norm(predicted_local - np.array([ox, oy]))
            clearance = distance - radius - self.config.obstacle_safety_margin
            if clearance < 0.0:
                obstacle_cost += 50.0 + 100.0 * abs(clearance)
            elif clearance < 0.30:
                obstacle_cost += 5.0 * ((0.30 - clearance) / 0.30) ** 2
        effort = np.linalg.norm(candidate) / self.config.max_accel
        return (
            -own_target_cost
            - separation_cost
            - 10.0 * boundary_excess
            - obstacle_cost
            - 0.03 * effort
        )

    def actions(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        origin: np.ndarray,
    ) -> np.ndarray:
        positions = np.asarray(positions, dtype=np.float32)
        velocities = np.asarray(velocities, dtype=np.float32)
        self.filter.update(positions[3])
        predicted_evader = self.filter.predict(0.40)
        targets = self._formation_targets(positions, predicted_evader)
        desired_velocity = targets - positions[:3]
        norms = np.linalg.norm(desired_velocity, axis=1, keepdims=True)
        desired_velocity *= np.minimum(
            1.0, self.config.max_speed / np.maximum(norms, 1e-8)
        )
        actions = np.clip(
            (desired_velocity - velocities[:3]) / 0.40,
            -self.config.max_accel,
            self.config.max_accel,
        )

        directions = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
        candidates = [np.zeros(2, dtype=np.float32)]
        candidates.extend(
            self.config.max_accel
            * np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
            for angle in directions
        )
        # Sequential best responses approximate the original finite-action IBR.
        for _ in range(3):
            for agent in range(3):
                actions[agent] = max(
                    candidates,
                    key=lambda candidate: self._utility(
                        agent,
                        candidate,
                        actions,
                        positions,
                        velocities,
                        targets,
                        origin,
                    ),
                )
        return np.clip(actions / self.config.max_accel, -1.0, 1.0)
