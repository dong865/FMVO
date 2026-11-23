"""50 Hz 3V1 environment with a noisy mocap-to-velocity-PID control chain."""

from collections import deque
from typing import Deque, Dict, Optional, Tuple

import numpy as np

from .config import CrazyfliePhysics, TaskConfig
from .features import build_observations, point_in_triangle
from .evader_potential_field import PotentialFieldEvader
from .reward import compute_reward
from .scenario import (
    EpisodeScenario,
    ScenarioGenerator,
    SceneSamplingOptions,
)
from .safety import (
    pairwise_distances,
    project_obstacle_velocity,
    project_pairwise_velocity,
)


class ThreeVOneEnv:
    """Planar training dynamics aligned with real ``cmdVelocityWorld`` use.

    ``positions`` and ``velocities`` are the true physical state. Policies must
    consume :meth:`control_state`, which returns delayed/noisy mocap state.
    Normalized policy accelerations are integrated into velocity commands. A
    delayed velocity PID and randomized plant then produce the true motion.
    """

    def __init__(
        self,
        config: Optional[TaskConfig] = None,
        seed: int = 0,
        scene_options: Optional[SceneSamplingOptions] = None,
    ):
        self.config = config or TaskConfig()
        self.physics = CrazyfliePhysics()
        self.rng = np.random.default_rng(seed)
        self.scenario_generator = ScenarioGenerator(
            self.config, self.rng, scene_options
        )
        self.evader_controller = PotentialFieldEvader(self.config)
        self.current_scenario: Optional[EpisodeScenario] = None
        self.virtual_obstacles = tuple(self.config.virtual_obstacles)
        self.evader_goal = np.zeros(2, dtype=np.float32)
        self.initial_goal_distance = 0.0
        self.previous_goal_distance = 0.0
        self.goal_progress = 0.0
        self.origin = np.zeros(2, dtype=np.float32)
        num_entities = self.config.num_entities
        self.active_pursuer_mask = np.ones(
            self.config.num_pursuers, dtype=bool
        )
        self.positions = np.zeros((num_entities, 2), dtype=np.float32)
        self.velocities = np.zeros((num_entities, 2), dtype=np.float32)
        self.measured_positions = np.zeros((num_entities, 2), dtype=np.float32)
        self.measured_velocities = np.zeros((num_entities, 2), dtype=np.float32)
        self.commanded_velocities = np.zeros((num_entities, 2), dtype=np.float32)
        self.applied_velocities = np.zeros((num_entities, 2), dtype=np.float32)
        self.reference_positions = np.zeros((num_entities, 2), dtype=np.float32)
        self.pid_integral = np.zeros((num_entities, 2), dtype=np.float32)
        self.pid_previous_error = np.zeros((num_entities, 2), dtype=np.float32)
        self.mass_scales = np.ones(num_entities, dtype=np.float32)
        self.actuator_scales = np.ones(num_entities, dtype=np.float32)
        self.wind_bias = np.zeros(2, dtype=np.float32)
        self.inter_drone_airflow_acceleration = np.zeros(
            (num_entities, 2), dtype=np.float32
        )
        self.num_pursuers = self.config.num_pursuers
        self.num_entities = self.config.num_entities
        self.evader_index = self.config.evader_index
        self._pursuer_slice = slice(0, self.num_pursuers)
        self.mocap_delay_steps = 0
        self.communication_delay_steps = 0
        self.command_delay_steps = 0
        self.frame_dropped = False
        self.step_count = 0
        self.capture_count = 0
        self.previous_mean_distance = 0.0
        self.safety_intervened = False
        self.safety_emergency = False
        self.capture_candidate = False
        self.obstacle_safety_intervened = False
        self.obstacle_safety_emergency = False
        self.minimum_obstacle_clearance = float("inf")
        self.previous_angular_gap = 2.0 * np.pi
        self._measurement_queue: Deque[Tuple[np.ndarray, np.ndarray]] = deque()
        self._command_queue: Deque[np.ndarray] = deque()

    def _sample_integer(self, maximum: int) -> int:
        if not self.config.disturbances_enabled:
            return 0
        return int(self.rng.integers(0, max(0, maximum) + 1))

    def _reset_domain_parameters(self) -> None:
        cfg = self.config
        if cfg.disturbances_enabled:
            low, high = cfg.mass_scale_range
            self.mass_scales[:] = self.rng.uniform(
                low, high, size=self.num_entities
            )
            low, high = cfg.actuator_scale_range
            self.actuator_scales[:] = self.rng.uniform(
                low, high, size=self.num_entities
            )
            self.wind_bias[:] = self.rng.normal(0.0, cfg.wind_bias_std, size=2)
        else:
            self.mass_scales.fill(1.0)
            self.actuator_scales.fill(1.0)
            self.wind_bias.fill(0.0)
        self.mocap_delay_steps = self._sample_integer(cfg.mocap_delay_steps_max)
        self.communication_delay_steps = self._sample_integer(
            cfg.communication_delay_steps_max
        )
        self.command_delay_steps = self._sample_integer(
            cfg.velocity_command_delay_steps_max
        )

    def reset(self) -> Tuple[np.ndarray, Dict[str, object]]:
        self.current_scenario = self.scenario_generator.sample()
        self.positions[self._pursuer_slice] = np.asarray(
            self.current_scenario.pursuer_starts, dtype=np.float32
        )
        self.positions[self.evader_index] = np.asarray(
            self.current_scenario.evader_start, dtype=np.float32
        )
        self.evader_goal = np.asarray(
            self.current_scenario.evader_goal, dtype=np.float32
        )
        self.virtual_obstacles = tuple(
            self.current_scenario.virtual_obstacles
        )
        self.velocities.fill(0.0)
        self.commanded_velocities.fill(0.0)
        self.applied_velocities.fill(0.0)
        self.pid_integral.fill(0.0)
        self.pid_previous_error.fill(0.0)
        self.inter_drone_airflow_acceleration.fill(0.0)
        self._enforce_virtual_obstacles()
        self.reference_positions[:] = self.positions
        self._reset_domain_parameters()

        observation_delay = self.mocap_delay_steps + self.communication_delay_steps
        self._measurement_queue.clear()
        for _ in range(observation_delay):
            self._measurement_queue.append(
                (self.positions.copy(), self.velocities.copy())
            )
        self._command_queue.clear()
        for _ in range(self.command_delay_steps):
            self._command_queue.append(
                np.zeros((self.num_entities, 2), dtype=np.float32)
            )
        self._set_measurement(self.positions, self.velocities)
        self.frame_dropped = False
        self.step_count = 0
        self.capture_count = 0
        self.safety_intervened = False
        self.safety_emergency = False
        self.capture_candidate = False
        self.obstacle_safety_intervened = False
        self.obstacle_safety_emergency = False
        self.minimum_obstacle_clearance = float("inf")
        self.previous_mean_distance = self._mean_distance()
        self.previous_angular_gap = self._maximum_angular_gap()
        self.initial_goal_distance = self._distance_to_goal()
        self.previous_goal_distance = self.initial_goal_distance
        self.goal_progress = 0.0
        return self.observations(), self._info(False, False)

    def _set_measurement(self, positions: np.ndarray, velocities: np.ndarray) -> None:
        cfg = self.config
        if cfg.disturbances_enabled:
            position_noise = self.rng.normal(
                0.0, cfg.mocap_position_noise_std, positions.shape
            )
            velocity_noise = self.rng.normal(
                0.0, cfg.mocap_velocity_noise_std, velocities.shape
            )
        else:
            position_noise = 0.0
            velocity_noise = 0.0
        self.measured_positions[:] = positions + position_noise
        self.measured_velocities[:] = velocities + velocity_noise

    def _update_measurement(self) -> None:
        self._measurement_queue.append((self.positions.copy(), self.velocities.copy()))
        delayed_positions, delayed_velocities = self._measurement_queue.popleft()
        dropout = (
            self.config.disturbances_enabled
            and self.rng.random() < self.config.mocap_frame_drop_probability
        )
        self.frame_dropped = bool(dropout)
        if not dropout:
            self._set_measurement(delayed_positions, delayed_velocities)

    def control_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the state visible to an external ROS/mocap controller."""
        return self.measured_positions.copy(), self.measured_velocities.copy()

    def observations(self) -> np.ndarray:
        return build_observations(
            self.measured_positions,
            self.measured_velocities,
            self.origin,
            self.config,
            self.virtual_obstacles,
        )

    def evader_state(self) -> np.ndarray:
        """Delayed/noisy normalized evader state expected by the EPN."""
        return np.concatenate(
            (
                self.measured_positions[self.evader_index]
                / self.config.arena_half_extent,
                self.measured_velocities[self.evader_index]
                / self.config.velocity_scale,
            )
        ).astype(np.float32)

    def _pursuer_positions(self) -> np.ndarray:
        return self.positions[self._pursuer_slice]

    def _mean_distance(self) -> float:
        return float(
            np.linalg.norm(
                self.positions[self._pursuer_slice]
                - self.positions[self.evader_index],
                axis=1,
            ).mean()
        )

    def _maximum_angular_gap(self) -> float:
        relative = (
            self.positions[self._pursuer_slice]
            - self.positions[self.evader_index]
        )
        angles = np.sort(np.arctan2(relative[:, 1], relative[:, 0]))
        wrapped = np.concatenate((angles, angles[:1] + 2.0 * np.pi))
        return float(np.max(np.diff(wrapped)))

    def _is_enclosed(self) -> bool:
        """Evader inside the active pursuers' convex hull.

        3V1 keeps the historical triangle rule; larger teams use the angular
        coverage criterion (max gap < pi), which agrees on three points.
        """
        pursuers = self._pursuer_positions()
        if self.num_pursuers == 3:
            return point_in_triangle(self.positions[self.evader_index], pursuers)
        return self._maximum_angular_gap() < np.pi

    @staticmethod
    def _limit_vectors(vectors: np.ndarray, limit: float) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors * np.minimum(1.0, limit / np.maximum(norms, 1e-8))

    def _evader_acceleration(self) -> np.ndarray:
        return self.evader_controller.acceleration(
            self.measured_positions,
            self.evader_goal,
            self.origin,
            self.virtual_obstacles,
        ).astype(np.float32)

    def _distance_to_goal(self) -> float:
        return float(
            np.linalg.norm(self.positions[self.evader_index] - self.evader_goal)
        )

    def _enforce_virtual_obstacles(self) -> np.ndarray:
        collisions = np.zeros(self.num_entities, dtype=bool)
        for ox, oy, radius in self.virtual_obstacles:
            center = self.origin + np.array([ox, oy], dtype=np.float32)
            safe_radius = radius + self.config.obstacle_safety_margin
            for agent in range(self.num_entities):
                offset = self.positions[agent] - center
                distance = float(np.linalg.norm(offset))
                if distance < safe_radius:
                    collisions[agent] = True
                    direction = (
                        np.array([1.0, 0.0], dtype=np.float32)
                        if distance < 1e-6
                        else offset / distance
                    )
                    self.positions[agent] = center + direction * safe_radius
                    inward = float(np.dot(self.velocities[agent], direction))
                    if inward < 0.0:
                        self.velocities[agent] -= inward * direction
                    self.velocities[agent] *= 0.25
        return collisions

    def _capture_candidate(
        self,
        distances: np.ndarray,
        minimum_distance: float,
        enclosed: bool,
    ) -> bool:
        """Hook for subclasses to override the capture geometry rule."""
        cfg = self.config
        return bool(
            enclosed
            and np.max(distances) <= cfg.capture_radius
            and minimum_distance >= cfg.safety_emergency_distance
        )

    def _delayed_velocity_command(self) -> np.ndarray:
        self._command_queue.append(self.commanded_velocities.copy())
        delayed = self._command_queue.popleft()
        self.applied_velocities[:] = delayed
        return delayed

    def _inter_drone_airflow(self) -> np.ndarray:
        """Approximate same-height rotor outwash and near-field turbulence."""
        cfg = self.config
        airflow = np.zeros((self.num_entities, 2), dtype=np.float32)
        if not cfg.disturbances_enabled:
            self.inter_drone_airflow_acceleration[:] = airflow
            return airflow
        for first in range(self.num_entities):
            for second in range(first + 1, self.num_entities):
                offset = self.positions[first] - self.positions[second]
                distance = float(np.linalg.norm(offset))
                if distance >= cfg.inter_drone_airflow_radius:
                    continue
                direction = (
                    np.array([1.0, 0.0], dtype=np.float32)
                    if distance < 1e-6
                    else offset / distance
                )
                proximity = (1.0 - distance / cfg.inter_drone_airflow_radius) ** 2
                radial = cfg.inter_drone_airflow_strength * proximity * direction
                turbulence_scale = cfg.inter_drone_airflow_turbulence_std * proximity
                airflow[first] += radial
                airflow[second] -= radial
                airflow[first] += self.rng.normal(0.0, turbulence_scale, size=2)
                airflow[second] += self.rng.normal(0.0, turbulence_scale, size=2)
        self.inter_drone_airflow_acceleration[:] = airflow
        return airflow

    def _apply_velocity_pid(self, target_velocity: np.ndarray) -> np.ndarray:
        cfg = self.config
        error = target_velocity - self.velocities
        self.pid_integral += error * cfg.dt
        np.clip(
            self.pid_integral,
            -cfg.velocity_pid_integral_limit,
            cfg.velocity_pid_integral_limit,
            out=self.pid_integral,
        )
        derivative = (error - self.pid_previous_error) / cfg.dt
        self.pid_previous_error[:] = error
        pid_acceleration = (
            cfg.velocity_pid_kp * error
            + cfg.velocity_pid_ki * self.pid_integral
            + cfg.velocity_pid_kd * derivative
        )
        pid_acceleration *= (self.actuator_scales / self.mass_scales)[:, None]
        pid_acceleration = self._limit_vectors(
            pid_acceleration, cfg.velocity_pid_max_accel
        )
        if cfg.disturbances_enabled:
            room_gust = self.rng.normal(
                0.0, cfg.wind_gust_std, (self.num_entities, 2)
            )
        else:
            room_gust = np.zeros((self.num_entities, 2), dtype=np.float32)
        airflow = self._inter_drone_airflow()
        return (
            pid_acceleration
            + self.wind_bias[None, :]
            + room_gust
            + airflow
            - cfg.linear_drag * self.velocities
        ).astype(np.float32)

    def _info(
        self,
        captured: bool,
        enclosed: bool,
        capture_candidate: bool = False,
        safety_failure: bool = False,
        evader_succeeded: bool = False,
        outcome: str = "ongoing",
    ) -> Dict[str, object]:
        position_error = np.linalg.norm(
            self.reference_positions - self.positions, axis=1
        )
        velocity_error = np.linalg.norm(
            self.applied_velocities - self.velocities, axis=1
        )
        observation_error = np.linalg.norm(
            self.measured_positions - self.positions, axis=1
        )
        pair_distance = pairwise_distances(self.positions)
        return {
            "captured": captured,
            "enclosed": enclosed,
            "capture_candidate": capture_candidate,
            "safety_failure": safety_failure,
            "evader_succeeded": evader_succeeded,
            "outcome": outcome,
            "evader_goal": self.evader_goal.copy(),
            "distance_to_goal": self._distance_to_goal(),
            "goal_progress": self.goal_progress,
            "safety_intervened": self.safety_intervened,
            "safety_emergency": self.safety_emergency,
            "obstacle_safety_intervened": self.obstacle_safety_intervened,
            "obstacle_safety_emergency": self.obstacle_safety_emergency,
            "minimum_obstacle_clearance": self.minimum_obstacle_clearance,
            "mean_distance": self._mean_distance(),
            "obstacle_collisions": np.zeros(self.num_entities, dtype=bool),
            "active_pursuer_mask": self.active_pursuer_mask.copy(),
            "reference_positions": self.reference_positions.copy(),
            "true_positions": self.positions.copy(),
            "measured_positions": self.measured_positions.copy(),
            "position_tracking_error": position_error.astype(np.float32),
            "velocity_tracking_error": velocity_error.astype(np.float32),
            "mocap_observation_error": observation_error.astype(np.float32),
            "mocap_frame_dropped": self.frame_dropped,
            "mocap_delay_steps": self.mocap_delay_steps,
            "communication_delay_steps": self.communication_delay_steps,
            "velocity_command_delay_steps": self.command_delay_steps,
            "mass_scales": self.mass_scales.copy(),
            "actuator_scales": self.actuator_scales.copy(),
            "wind_bias": self.wind_bias.copy(),
            "inter_drone_airflow_acceleration": (
                self.inter_drone_airflow_acceleration.copy()
            ),
            "minimum_inter_drone_distance": float(pair_distance.min()),
        }

    def step(self, actions: np.ndarray):
        cfg = self.config
        actions = np.clip(np.asarray(actions, dtype=np.float32), -1.0, 1.0)
        if actions.shape != (cfg.num_pursuers, 2):
            raise ValueError(
                f"actions must have shape ({cfg.num_pursuers}, 2)"
            )
        actions = self._limit_vectors(actions, 1.0)

        self.commanded_velocities[self._pursuer_slice] += (
            actions * cfg.max_accel * cfg.dt
        )
        self.commanded_velocities[self._pursuer_slice] = self._limit_vectors(
            self.commanded_velocities[self._pursuer_slice], cfg.max_speed
        )
        self.commanded_velocities[self.evader_index] += (
            self._evader_acceleration() * cfg.dt
        )
        self.commanded_velocities[self.evader_index:] = self._limit_vectors(
            self.commanded_velocities[self.evader_index:],
            cfg.evader_max_speed,
        )
        speed_limits = np.array(
            [cfg.max_speed] * cfg.num_pursuers + [cfg.evader_max_speed]
        )
        safety = project_pairwise_velocity(
            self.commanded_velocities,
            self.measured_positions,
            warning_distance=cfg.safety_warning_distance,
            emergency_distance=cfg.safety_emergency_distance,
            separation_speed=cfg.safety_separation_speed,
            speed_limits=speed_limits,
        )
        obstacle_safety = project_obstacle_velocity(
            safety.velocities,
            self.measured_positions,
            self.origin,
            self.virtual_obstacles,
            safety_margin=cfg.obstacle_safety_margin,
            warning_margin=cfg.obstacle_warning_margin,
            separation_speed=cfg.safety_separation_speed,
            speed_limits=speed_limits,
        )
        self.commanded_velocities[:] = obstacle_safety.velocities
        self.obstacle_safety_intervened = obstacle_safety.intervened
        self.obstacle_safety_emergency = obstacle_safety.emergency
        self.minimum_obstacle_clearance = obstacle_safety.minimum_clearance
        self.safety_intervened = safety.intervened or obstacle_safety.intervened
        self.safety_emergency = safety.emergency or obstacle_safety.emergency
        self.reference_positions += self.commanded_velocities * cfg.dt
        np.clip(
            self.reference_positions,
            -cfg.arena_half_extent,
            cfg.arena_half_extent,
            out=self.reference_positions,
        )

        delayed_velocity = self._delayed_velocity_command()
        acceleration = self._apply_velocity_pid(delayed_velocity)
        self.velocities += acceleration * cfg.dt
        self.velocities[self._pursuer_slice] = self._limit_vectors(
            self.velocities[self._pursuer_slice],
            cfg.max_speed * cfg.physical_speed_limit_scale,
        )
        self.velocities[self.evader_index:] = self._limit_vectors(
            self.velocities[self.evader_index:],
            cfg.evader_max_speed * cfg.physical_speed_limit_scale,
        )
        self.velocities = self.velocities.astype(np.float32)
        self.positions += self.velocities * cfg.dt

        extent = cfg.arena_half_extent
        for agent in range(self.num_entities):
            for axis in range(2):
                if abs(self.positions[agent, axis]) > extent:
                    self.positions[agent, axis] = (
                        np.sign(self.positions[agent, axis]) * extent
                    )
                    self.velocities[agent, axis] *= -0.25
        obstacle_collisions = self._enforce_virtual_obstacles()

        distances = np.linalg.norm(
            self.positions[self._pursuer_slice]
            - self.positions[self.evader_index],
            axis=1,
        )
        enclosed = self._is_enclosed()
        minimum_distance = float(np.min(pairwise_distances(self.positions)))
        safety_failure = bool(
            minimum_distance < cfg.safety_emergency_distance
            or np.any(obstacle_collisions)
        )
        self.capture_candidate = self._capture_candidate(
            distances, minimum_distance, enclosed
        )
        self.capture_count = self.capture_count + 1 if self.capture_candidate else 0
        captured = self.capture_count >= cfg.capture_hold_steps
        distance_to_goal = self._distance_to_goal()
        self.goal_progress = self.previous_goal_distance - distance_to_goal
        self.previous_goal_distance = distance_to_goal
        reached_goal = distance_to_goal <= cfg.evader_goal_radius

        self.step_count += 1
        timed_out = self.step_count >= cfg.max_episode_steps
        if safety_failure:
            captured = False
            evader_succeeded = False
            outcome = "safety_failure"
        elif captured:
            evader_succeeded = False
            outcome = "captured"
        elif reached_goal:
            evader_succeeded = True
            outcome = "evader_succeeded"
        elif timed_out:
            evader_succeeded = False
            outcome = "timeout"
        else:
            evader_succeeded = False
            outcome = "ongoing"

        mean_distance = float(distances.mean())
        progress = self.previous_mean_distance - mean_distance
        self.previous_mean_distance = mean_distance
        angular_gap = self._maximum_angular_gap()
        angular_progress = self.previous_angular_gap - angular_gap
        self.previous_angular_gap = angular_gap
        reward_breakdown = compute_reward(
            config=cfg,
            positions=self.positions,
            origin=self.origin,
            actions=actions,
            progress=progress,
            angular_progress=angular_progress,
            capture_candidate=self.capture_candidate,
            captured=captured,
            safety_failure=safety_failure,
            virtual_obstacles=self.virtual_obstacles,
            evader_succeeded=evader_succeeded,
        )

        terminated = safety_failure or captured or evader_succeeded
        truncated = timed_out and not terminated
        self._update_measurement()
        info = self._info(
            captured,
            enclosed,
            self.capture_candidate,
            safety_failure,
            evader_succeeded,
            outcome,
        )
        info["mean_distance"] = mean_distance
        info["obstacle_collisions"] = obstacle_collisions.copy()
        info.update(reward_breakdown.info())
        return (
            self.observations(),
            reward_breakdown.total,
            terminated,
            truncated,
            info,
        )
