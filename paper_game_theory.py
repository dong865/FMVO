"""Finite-horizon exact-potential IBR expert for the 3-versus-1 task."""

from itertools import permutations
from typing import Dict, Optional

import numpy as np

from .config import TaskConfig
from .features import point_in_triangle


class AlphaBetaFilter:
    """Constant-velocity target estimator used by simulation and deployment."""

    def __init__(
        self,
        dt: float,
        alpha: float = 0.75,
        beta: float = 0.20,
        max_speed: Optional[float] = None,
    ):
        self.dt = dt
        self.alpha = alpha
        self.beta = beta
        self.max_speed = max_speed
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
        speed = float(np.linalg.norm(self.velocity))
        if self.max_speed is not None and speed > self.max_speed:
            self.velocity *= self.max_speed / max(speed, 1e-8)

    def predict(self, horizon: float) -> np.ndarray:
        return self.position + self.velocity * horizon


class GameTheoryRuntime:
    """Sequential best response over one shared finite-horizon potential."""

    baseline_name = "game_theory"

    def __init__(self, config: Optional[TaskConfig] = None):
        self.config = config or TaskConfig()
        self.filter = AlphaBetaFilter(
            self.config.dt, max_speed=self.config.evader_max_speed
        )
        self.formation_angle: Optional[float] = None
        self.previous_order = None

    def reset(self) -> None:
        self.filter.reset()
        self.formation_angle = None
        self.previous_order = None

    @staticmethod
    def _angle_difference(first: float, second: float) -> float:
        return abs((first - second + np.pi) % (2.0 * np.pi) - np.pi)

    def _rate_limited_formation_angle(self, desired: float) -> float:
        if self.formation_angle is None:
            self.formation_angle = float(desired)
        else:
            difference = (
                desired - self.formation_angle + np.pi
            ) % (2.0 * np.pi) - np.pi
            # Follow the evader's heading fast enough to keep the triangle
            # closed on a turning evader (evader max accel 0.8 m/s^2); 1x dt
            # (0.02 rad/step) lagged behind hard cuts and left an escape gap.
            maximum_change = 3.0 * self.config.dt
            self.formation_angle += float(
                np.clip(difference, -maximum_change, maximum_change)
            )
        return self.formation_angle

    def _formation_roles(self, positions, predicted_evader):
        relative = positions[:3] - predicted_evader
        current_angles = np.arctan2(relative[:, 1], relative[:, 0])
        if np.linalg.norm(self.filter.velocity) > 1e-4:
            base_angle = np.arctan2(
                self.filter.velocity[1], self.filter.velocity[0]
            )
        else:
            base_angle = np.arctan2(
                np.sin(current_angles).mean(), np.cos(current_angles).mean()
            )
        base_angle = self._rate_limited_formation_angle(base_angle)
        ideal_angles = base_angle + np.arange(3) * 2.0 * np.pi / 3.0
        order = min(
            permutations(range(3)),
            key=lambda candidate: (
                sum(
                    self._angle_difference(
                        current_angles[i], ideal_angles[candidate[i]]
                    )
                    for i in range(3)
                )
                + self.config.role_switch_penalty
                * (
                    sum(candidate[i] != self.previous_order[i] for i in range(3))
                    if self.previous_order is not None
                    else 0
                )
            ),
        )
        self.previous_order = order
        distances = np.linalg.norm(relative, axis=1)
        enclosed = point_in_triangle(positions[3], positions[:3])
        if enclosed:
            radius = self.config.role_min_radius
        else:
            radius = np.clip(
                distances.max() * 0.45,
                self.config.role_min_radius,
                self.config.role_max_radius,
            )
        role_targets = predicted_evader + radius * np.stack(
            (np.cos(ideal_angles), np.sin(ideal_angles)), axis=1
        )
        assignment = np.zeros((3, 3), dtype=np.float32)
        assignment[np.arange(3), np.asarray(order)] = 1.0
        return (
            role_targets.astype(np.float32),
            assignment,
            (assignment @ role_targets).astype(np.float32),
        )

    def _formation_targets(self, positions, predicted_evader):
        return self._formation_roles(positions, predicted_evader)[2]

    def _team_potential(
        self,
        actions,
        positions,
        velocities,
        targets,
        origin,
        virtual_obstacles=None,
    ) -> float:
        horizon = self.config.ibr_horizon_seconds
        predicted = positions[:3] + velocities[:3] * horizon
        predicted += 0.5 * actions * horizon * horizon
        value = -float(np.linalg.norm(predicted - targets, axis=1).sum())
        predicted_evader = positions[3] + velocities[3] * horizon
        relative = predicted - predicted_evader
        evader_distances = np.linalg.norm(relative, axis=1)
        for first in range(3):
            for second in range(first + 1, 3):
                distance = np.linalg.norm(predicted[first] - predicted[second])
                shortfall = max(
                    self.config.safety_warning_distance - distance,
                    0.0,
                )
                value -= 12.0 * shortfall
        for distance in evader_distances:
            shortfall = max(
                self.config.safety_warning_distance - distance,
                0.0,
            )
            # Safety margin: at 0.30 (emergency) this penalty (3.2) exceeds the
            # capture bonus (+2.0) by a clear margin, so the expert brakes
            # instead of trading safety for capture. Penalty 20 allowed close
            # passes at 0.297-0.300 (all safety failures sat just under the
            # 0.30 floor) because 20*0.10 == the capture bonus; 28 reduced but
            # did not eliminate them (4/60 robust at 0.297-0.300).
            value -= 40.0 * shortfall
        angles = np.sort(np.arctan2(relative[:, 1], relative[:, 0]))
        wrapped = np.concatenate((angles, angles[:1] + 2.0 * np.pi))
        maximum_gap = float(np.max(np.diff(wrapped)))
        value -= 0.8 * max(maximum_gap - 2.0 * np.pi / 3.0, 0.0)
        maximum_distance = float(np.max(evader_distances))
        value -= 0.5 * maximum_distance
        if (
            point_in_triangle(predicted_evader, predicted)
            and maximum_distance <= self.config.capture_radius
            and float(np.min(evader_distances))
            >= self.config.safety_emergency_distance
        ):
            value += 2.0
        local = predicted - origin
        excess = np.maximum(np.abs(local) - self.config.arena_half_extent * 0.90, 0.0)
        value -= 10.0 * float(excess.sum())
        obstacles = (
            self.config.virtual_obstacles
            if virtual_obstacles is None
            else tuple(virtual_obstacles)
        )
        for point in local:
            for ox, oy, radius in obstacles:
                clearance = np.linalg.norm(point - np.array([ox, oy]))
                clearance -= radius + self.config.obstacle_safety_margin
                if clearance < 0.0:
                    value -= 50.0 + 100.0 * abs(clearance)
                elif clearance < 0.30:
                    value -= 5.0 * ((0.30 - clearance) / 0.30) ** 2
        value -= 0.03 * float(
            np.linalg.norm(actions, axis=1).sum() / self.config.max_accel
        )
        return value

    def _candidates(self):
        angles = np.linspace(
            0.0,
            2.0 * np.pi,
            self.config.ibr_candidate_directions,
            endpoint=False,
        )
        candidates = [np.zeros(2, dtype=np.float32)]
        candidates.extend(
            self.config.max_accel
            * np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
            for angle in angles
        )
        return candidates

    def actions_with_trace(
        self, positions, velocities, origin, virtual_obstacles=None
    ) -> Dict[str, object]:
        positions = np.asarray(positions, dtype=np.float32)
        velocities = np.asarray(velocities, dtype=np.float32)
        origin = np.asarray(origin, dtype=np.float32)
        self.filter.update(positions[3])
        predicted_evader = self.filter.predict(self.config.ibr_horizon_seconds)
        role_targets, assignment, targets = self._formation_roles(
            positions, predicted_evader
        )
        horizon = self.config.ibr_horizon_seconds
        desired_velocity = (targets - positions[:3]) / max(horizon, 1e-6)
        norms = np.linalg.norm(desired_velocity, axis=1, keepdims=True)
        desired_velocity *= np.minimum(
            1.0, self.config.max_speed / np.maximum(norms, 1e-8)
        )
        actions = np.clip(
            (desired_velocity - velocities[:3]) / max(horizon, 1e-6),
            -self.config.max_accel,
            self.config.max_accel,
        )
        candidates = self._candidates()
        process_actions = []
        potential_gains = []
        for _ in range(self.config.ibr_rounds):
            before = self._team_potential(
                actions, positions, velocities, targets, origin,
                virtual_obstacles,
            )
            for agent in range(3):
                best = actions[agent].copy()
                best_value = self._team_potential(
                    actions, positions, velocities, targets, origin,
                    virtual_obstacles,
                )
                for candidate in candidates:
                    trial = actions.copy()
                    trial[agent] = candidate
                    value = self._team_potential(
                        trial, positions, velocities, targets, origin,
                        virtual_obstacles,
                    )
                    if value > best_value:
                        best, best_value = candidate, value
                actions[agent] = best
            after = self._team_potential(
                actions, positions, velocities, targets, origin,
                virtual_obstacles,
            )
            process_actions.append(
                np.clip(actions / self.config.max_accel, -1.0, 1.0).copy()
            )
            potential_gains.append(max(after - before, 0.0))

        residual = 0.0
        final_value = self._team_potential(
            actions, positions, velocities, targets, origin,
            virtual_obstacles,
        )
        for agent in range(3):
            for candidate in candidates:
                trial = actions.copy()
                trial[agent] = candidate
                residual = max(
                    residual,
                    self._team_potential(
                        trial, positions, velocities, targets, origin,
                        virtual_obstacles,
                    )
                    - final_value,
                )
        return {
            "actions": np.clip(actions / self.config.max_accel, -1.0, 1.0),
            "process_actions": np.asarray(process_actions, dtype=np.float32),
            "potential_gains": np.asarray(potential_gains, dtype=np.float32),
            "br_residual": float(max(residual, 0.0)),
            "role_assignment": assignment,
            "role_targets": role_targets,
        }

    def actions(self, positions, velocities, origin, virtual_obstacles=None):
        return self.actions_with_trace(
            positions, velocities, origin, virtual_obstacles
        )["actions"]
