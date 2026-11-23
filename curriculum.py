"""Safety-gated curriculum whose last stage is the unmodified final task."""

import argparse
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Tuple

import numpy as np

from .config import TaskConfig


@dataclass(frozen=True)
class CurriculumStage:
    """One training stage and its independently cloned environment config."""

    index: int
    name: str
    config: TaskConfig


def _config_with(final_config: TaskConfig, **updates) -> TaskConfig:
    values: Dict[str, object] = final_config.to_dict()
    values.update(updates)
    return TaskConfig.from_dict(values)


def curriculum_stages(final_config: TaskConfig) -> Tuple[CurriculumStage, ...]:
    """Build six increasingly difficult stages that all retain the private goal."""
    stages = (
        (
            "short_diagonal_open",
            _config_with(
                final_config,
                evader_goal_min_distance=3.0,
                diagonal_boundary_clearance_range=(0.75, 0.90),
                evader_policy_scale=0.40,
                disturbances_enabled=False,
                procedural_obstacles=False,
                virtual_obstacles=(),
            ),
        ),
        (
            "full_diagonal_open",
            _config_with(
                final_config,
                evader_policy_scale=0.60,
                disturbances_enabled=False,
                procedural_obstacles=False,
                virtual_obstacles=(),
            ),
        ),
        (
            "limited_random_obstacles",
            _config_with(
                final_config,
                evader_policy_scale=0.80,
                disturbances_enabled=False,
                procedural_obstacles=True,
                procedural_obstacle_count_range=(1, 2),
                obstacle_free_probability=0.35,
            ),
        ),
        (
            "full_procedural_obstacles",
            _config_with(
                final_config,
                disturbances_enabled=False,
                procedural_obstacles=True,
            ),
        ),
        (
            "reduced_disturbances",
            _config_with(
                final_config,
                disturbances_enabled=True,
                procedural_obstacles=True,
                mocap_position_noise_std=final_config.mocap_position_noise_std * 0.5,
                mocap_velocity_noise_std=final_config.mocap_velocity_noise_std * 0.5,
                mocap_delay_steps_max=final_config.mocap_delay_steps_max // 2,
                communication_delay_steps_max=(
                    final_config.communication_delay_steps_max // 2
                ),
                velocity_command_delay_steps_max=min(
                    1, final_config.velocity_command_delay_steps_max
                ),
                mocap_frame_drop_probability=(
                    final_config.mocap_frame_drop_probability * 0.5
                ),
                mass_scale_range=(0.95, 1.05),
                actuator_scale_range=(0.95, 1.05),
                wind_bias_std=final_config.wind_bias_std * 0.5,
                wind_gust_std=final_config.wind_gust_std * 0.5,
                inter_drone_airflow_strength=(
                    final_config.inter_drone_airflow_strength * 0.5
                ),
                inter_drone_airflow_turbulence_std=(
                    final_config.inter_drone_airflow_turbulence_std * 0.5
                ),
            ),
        ),
        ("final_robust", _config_with(final_config)),
    )
    return tuple(
        CurriculumStage(index=index, name=name, config=config)
        for index, (name, config) in enumerate(stages)
    )


def stage_zero_stall_message(diagnostics: Dict[str, float]) -> str:
    """Format actionable evidence for a stage-zero learning failure."""

    reward_means = {
        key.removeprefix("reward/"): value
        for key, value in diagnostics.items()
        if key.startswith("reward/")
    }
    return (
        "stage 0 stalled after 50 policy updates: captures=0; "
        "collision_source=("
        f"inter_drone_steps={diagnostics.get('safety/inter_drone_failure_steps', 0.0):.0f}, "
        f"obstacle_steps={diagnostics.get('safety/obstacle_collision_steps', 0.0):.0f}); "
        "radial_action_alignment="
        f"{diagnostics.get('policy/radial_action_alignment', 0.0):.4f}; "
        f"reward_component_means={reward_means}; "
        "minimum_separation="
        f"{diagnostics.get('safety/minimum_separation', 0.0):.4f}"
    )


class CurriculumController:
    """Promote only when rolling capture and safety gates both pass."""

    def __init__(
        self,
        final_config: TaskConfig,
        window: int = 50,
        promotion_capture_rate: float = 0.80,
        promotion_collision_free_rate: float = 0.95,
        demotion_capture_rate: float = 0.0,
        demotion_collision_free_rate: float = 0.0,
    ) -> None:
        if window <= 0:
            raise ValueError("curriculum window must be positive")
        for value in (
            promotion_capture_rate,
            promotion_collision_free_rate,
            demotion_capture_rate,
            demotion_collision_free_rate,
        ):
            if value < 0.0 or value > 1.0:
                raise ValueError("curriculum rates must lie in [0, 1]")
        self._final_config = _config_with(final_config)
        self.stages = curriculum_stages(self._final_config)
        self.stage_index = 0
        self.window = window
        self.promotion_capture_rate = promotion_capture_rate
        self.promotion_collision_free_rate = promotion_collision_free_rate
        self.demotion_capture_rate = demotion_capture_rate
        self.demotion_collision_free_rate = demotion_collision_free_rate
        self._history: Deque[Tuple[bool, bool]] = deque(maxlen=window)
        self._stage_policy_updates = 0
        self._stage_captures = 0
        # Lifetime captures never reset on promotion/demotion. The cold-start
        # stall guard keys on this, so a run that already captured (and only
        # later changed stage) is never misreported as never-learning.
        self._lifetime_captures = 0

    @property
    def current(self) -> CurriculumStage:
        return self.stages[self.stage_index]

    def validation_config(self) -> TaskConfig:
        """Return a fresh final-task config independent of training stage."""
        return _config_with(self._final_config)

    def rates(self) -> Tuple[float, float]:
        if not self._history:
            return 0.0, 0.0
        values = np.asarray(self._history, dtype=float)
        return float(values[:, 0].mean()), float(values[:, 1].mean())

    def record_policy_update(self) -> None:
        self._stage_policy_updates += 1

    def stage_zero_stalled(self, max_updates: int = 50) -> bool:
        if max_updates <= 0:
            raise ValueError("max_updates must be positive")
        return bool(
            self.stage_index == 0
            and self._stage_policy_updates >= max_updates
            and self._lifetime_captures == 0
        )

    def update(self, captured: bool, collision_free: bool) -> CurriculumStage:
        self._history.append((bool(captured), bool(collision_free)))
        self._stage_captures += int(bool(captured))
        self._lifetime_captures += int(bool(captured))
        if len(self._history) < self.window:
            return self.current
        capture_rate, collision_free_rate = self.rates()
        if (
            capture_rate >= self.promotion_capture_rate
            and collision_free_rate >= self.promotion_collision_free_rate
            and self.stage_index < len(self.stages) - 1
        ):
            self.stage_index += 1
            self._history.clear()
            self._stage_policy_updates = 0
            self._stage_captures = 0
        elif (
            self.stage_index > 0
            and capture_rate < self.demotion_capture_rate
            and collision_free_rate < self.demotion_collision_free_rate
            and (
                self.demotion_capture_rate > 0.0
                or self.demotion_collision_free_rate > 0.0
            )
        ):
            self.stage_index -= 1
            self._history.clear()
            self._stage_policy_updates = 0
            self._stage_captures = 0
        return self.current


def add_curriculum_args(parser: argparse.ArgumentParser) -> None:
    """Add consistent optional curriculum settings to a training CLI."""
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="enable safety-gated training stages; validation stays final",
    )
    parser.add_argument("--curriculum-window", type=int, default=50)
    parser.add_argument(
        "--curriculum-stall-updates", type=int, default=50
    )
    parser.add_argument(
        "--curriculum-capture-rate", type=float, default=0.80
    )
    parser.add_argument(
        "--curriculum-collision-free-rate", type=float, default=0.90
    )


def controller_from_args(
    args: argparse.Namespace, final_config: TaskConfig
):
    """Create a controller only when the CLI explicitly enables curriculum."""
    if not getattr(args, "curriculum", False):
        return None
    return CurriculumController(
        final_config,
        window=args.curriculum_window,
        promotion_capture_rate=args.curriculum_capture_rate,
        promotion_collision_free_rate=(
            args.curriculum_collision_free_rate
        ),
    )
