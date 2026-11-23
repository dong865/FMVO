"""Configuration shared by training and real flight."""

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class CrazyfliePhysics:
    """Parameters copied from crazyflie_sim/backend/np.py (CF2 model)."""

    mass_kg: float = 0.034
    inertia_kg_m2: Tuple[float, float, float] = (
        16.571710e-6,
        16.655602e-6,
        29.261652e-6,
    )
    arm_length_m: float = 0.046
    thrust_to_torque: float = 0.006
    gravity_m_s2: float = 9.81
    rpm_to_force_poly: Tuple[float, float, float] = (
        2.55077341e-08,
        -4.92422570e-05,
        -1.51910248e-01,
    )


@dataclass
class TaskConfig:
    """Training/deployment contract in metres and seconds."""

    control_hz: float = 50.0
    num_pursuers: int = 3
    num_evaders: int = 1
    arena_half_extent: float = 2.5
    flight_height: float = 1.0
    # The scattered pursuit task requires a modest pursuer capability
    # advantage; otherwise an evader already moving toward a distant goal is
    # kinematically unreachable from many valid initial states.
    max_speed: float = 0.56
    max_accel: float = 0.94
    evader_max_speed: float = 0.50
    evader_max_accel: float = 0.80
    capture_radius: float = 0.72
    # Safety distances are derived from the physical airframe, not tuned for
    # margin: CrazyfliePhysics.arm_length_m = 0.046 m, so the full motor-to-
    # motor span is 2 x 0.046 = 0.092 m.  The old values (collision 0.26 /
    # emergency 0.30 / warning 0.43) were 3-9x the airframe and made real
    # encirclement impossible: `real_flight.apply_safety` zeroes the velocity
    # of EVERY drone as soon as any pair is closer than the emergency
    # separation, and `project_pairwise_velocity` actively prevents closing
    # below the warning distance -- both of which fire long before the
    # 0.72 m capture ring can be formed.
    #
    # collision_radius      = full span (2 arm lengths)
    # safety_emergency      = slightly longer than the full span
    # safety_warning        = approach warning, must exceed emergency
    collision_radius: float = 0.09
    safety_warning_distance: float = 0.20
    safety_emergency_distance: float = 0.12
    safety_separation_speed: float = 0.10
    max_virtual_obstacles: int = 4
    virtual_obstacles: Tuple[Tuple[float, float, float], ...] = (
        (0.00, 0.95, 0.22),
        (0.95, -0.60, 0.22),
    )
    scenario_contract_version: int = 2
    procedural_obstacles: bool = True
    evader_goal_radius: float = 0.25
    evader_goal_min_distance: float = 4.0
    diagonal_boundary_clearance_range: Tuple[float, float] = (0.45, 0.60)
    pursuer_spawn_distance_range: Tuple[float, float] = (0.90, 1.60)
    spawn_min_separation: float = 0.55
    obstacle_free_probability: float = 0.20
    procedural_obstacle_count_range: Tuple[int, int] = (1, 4)
    procedural_obstacle_radius_range: Tuple[float, float] = (0.14, 0.28)
    scenario_sampling_attempts: int = 256
    # Obstacle margins: the drone only needs its own span of clearance from a
    # cylinder, not the previous 0.15/0.25 m (which pushed the team so far off
    # the route that encirclement geometry could not close).
    obstacle_safety_margin: float = 0.08
    obstacle_warning_margin: float = 0.15
    capture_hold_seconds: float = 0.50
    episode_seconds: float = 35.0
    history_seconds: float = 0.50
    prediction_stride: int = 10
    prediction_steps: int = 5
    z_kp: float = 1.5
    z_kd: float = 0.25
    max_vertical_speed: float = 0.25
    pursuer_spawn_radius_range: Tuple[float, float] = (1.25, 1.70)
    evader_spawn_extent: float = 0.45
    evader_policy_scale: float = 1.0
    evader_goal_attraction_gain: float = 1.00
    evader_pursuer_repulsion_gain: float = 0.65
    evader_pursuer_influence_distance: float = 1.40
    evader_obstacle_repulsion_gain: float = 0.80
    evader_obstacle_influence_distance: float = 0.75
    evader_boundary_repulsion_gain: float = 0.60
    evader_boundary_influence_distance: float = 0.55
    evader_tangential_gain: float = 0.20
    evader_stagnation_threshold: float = 0.05
    distance_progress_reward_scale: float = 20.0
    angular_progress_reward_scale: float = 2.0
    time_penalty_rate: float = 0.015
    action_penalty_rate: float = 0.02
    safety_warning_penalty_rate: float = 2.0
    obstacle_warning_penalty_rate: float = 2.0
    capture_candidate_reward_rate: float = 2.0
    capture_reward: float = 10.0
    safety_failure_penalty: float = 5.0
    evader_escape_failure_penalty: float = 10.0

    disturbances_enabled: bool = True
    mocap_position_noise_std: float = 0.003
    mocap_velocity_noise_std: float = 0.015
    mocap_delay_steps_max: int = 1
    communication_delay_steps_max: int = 1
    velocity_command_delay_steps_max: int = 2
    mocap_frame_drop_probability: float = 0.03
    velocity_pid_kp: float = 6.0
    velocity_pid_ki: float = 1.0
    velocity_pid_kd: float = 0.02
    velocity_pid_integral_limit: float = 0.30
    velocity_pid_max_accel: float = 1.20
    mass_scale_range: Tuple[float, float] = (0.90, 1.10)
    actuator_scale_range: Tuple[float, float] = (0.90, 1.10)
    wind_bias_std: float = 0.005
    wind_gust_std: float = 0.003
    inter_drone_airflow_radius: float = 0.65
    inter_drone_airflow_strength: float = 0.22
    inter_drone_airflow_turbulence_std: float = 0.04
    linear_drag: float = 0.08
    physical_speed_limit_scale: float = 1.35

    num_roles: int = 3
    role_sinkhorn_iterations: int = 8
    role_temperature: float = 0.35
    role_switch_penalty: float = 0.15
    role_min_radius: float = 0.52
    role_max_radius: float = 1.25
    ibr_rounds: int = 3
    ibr_candidate_directions: int = 16
    ibr_horizon_seconds: float = 0.75

    def __post_init__(self) -> None:
        if self.arena_half_extent <= 0.0:
            raise ValueError("arena_half_extent must be positive")
        if self.scenario_contract_version <= 0:
            raise ValueError("scenario_contract_version must be positive")
        clearance_low, clearance_high = self.diagonal_boundary_clearance_range
        if (
            clearance_low <= 0.0
            or clearance_high < clearance_low
            or clearance_high >= self.arena_half_extent
        ):
            raise ValueError(
                "diagonal boundary clearances must be positive, ordered, "
                "and smaller than the arena half extent"
            )
        shortest_diagonal = (
            2.0 ** 0.5
            * 2.0
            * (self.arena_half_extent - clearance_high)
        )
        if (
            self.evader_goal_min_distance <= 0.0
            or self.evader_goal_min_distance > shortest_diagonal
        ):
            raise ValueError(
                "evader_goal_min_distance is impossible for the corner sectors"
            )
        if self.evader_goal_radius <= 0.0:
            raise ValueError("evader_goal_radius must be positive")
        distance_low, distance_high = self.pursuer_spawn_distance_range
        if distance_low <= 0.0 or distance_high < distance_low:
            raise ValueError(
                "pursuer spawn distances must be positive and ordered"
            )
        if self.spawn_min_separation < self.safety_warning_distance:
            raise ValueError(
                "spawn_min_separation must not be below safety_warning_distance"
            )
        if not 0.0 <= self.obstacle_free_probability <= 1.0:
            raise ValueError("obstacle_free_probability must lie in [0, 1]")
        obstacle_count_low, obstacle_count_high = (
            self.procedural_obstacle_count_range
        )
        if (
            obstacle_count_low < 0
            or obstacle_count_high < obstacle_count_low
            or obstacle_count_high > self.max_virtual_obstacles
        ):
            raise ValueError(
                "procedural obstacle counts must be ordered and fit feature slots"
            )
        radius_low, radius_high = self.procedural_obstacle_radius_range
        if radius_low <= 0.0 or radius_high < radius_low:
            raise ValueError(
                "procedural obstacle radii must be positive and ordered"
            )
        if self.scenario_sampling_attempts <= 0:
            raise ValueError("scenario_sampling_attempts must be positive")
        spawn_low, spawn_high = self.pursuer_spawn_radius_range
        if spawn_low <= 0.0 or spawn_high < spawn_low or spawn_high > self.arena_half_extent:
            raise ValueError("pursuer spawn radii must be positive, ordered, and inside the arena")
        if self.evader_spawn_extent <= 0.0 or self.evader_spawn_extent > self.arena_half_extent:
            raise ValueError("evader_spawn_extent must be positive and inside the arena")
        if not 0.0 <= self.evader_policy_scale <= 1.0:
            raise ValueError("evader_policy_scale must lie in [0, 1]")
        for name in (
            "evader_goal_attraction_gain",
            "evader_pursuer_repulsion_gain",
            "evader_obstacle_repulsion_gain",
            "evader_boundary_repulsion_gain",
            "evader_tangential_gain",
            "evader_stagnation_threshold",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        for name in (
            "evader_pursuer_influence_distance",
            "evader_obstacle_influence_distance",
            "evader_boundary_influence_distance",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.num_pursuers not in (3, 4, 5) or self.num_evaders != 1:
            raise ValueError("supported tasks require 3-5 pursuers and one evader")
        if self.num_roles <= 0:
            raise ValueError("num_roles must be positive")
        if min(
            self.max_speed,
            self.max_accel,
            self.evader_max_speed,
            self.evader_max_accel,
        ) <= 0.0:
            raise ValueError("pursuer and evader motion limits must be positive")
        if self.capture_radius <= self.collision_radius:
            raise ValueError("capture_radius must exceed collision_radius")
        if self.safety_emergency_distance < self.collision_radius:
            raise ValueError(
                "safety_emergency_distance must not be below collision_radius"
            )
        if self.safety_warning_distance <= self.safety_emergency_distance:
            raise ValueError(
                "safety_warning_distance must exceed safety_emergency_distance"
            )
        if self.safety_separation_speed < 0.0:
            raise ValueError("safety_separation_speed must be non-negative")
        if self.obstacle_warning_margin <= 0.0:
            raise ValueError("obstacle_warning_margin must be positive")

    @property
    def arena_size_m(self) -> float:
        return 2.0 * self.arena_half_extent

    @property
    def num_entities(self) -> int:
        """Number of physical agents represented by environment state."""
        return self.num_pursuers + self.num_evaders

    @property
    def evader_index(self) -> int:
        """Index of the single evader after the pursuer entity block."""
        return self.num_pursuers

    @property
    def velocity_scale(self) -> float:
        """Common observation scale, valid for both pursuers and evader."""
        return max(self.max_speed, self.evader_max_speed)

    @property
    def dt(self) -> float:
        return 1.0 / self.control_hz

    @property
    def max_episode_steps(self) -> int:
        return round(self.episode_seconds * self.control_hz)

    @property
    def history_steps(self) -> int:
        return max(3, round(self.history_seconds * self.control_hz))

    @property
    def capture_hold_steps(self) -> int:
        return max(1, round(self.capture_hold_seconds * self.control_hz))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "TaskConfig":
        valid = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in valid})
