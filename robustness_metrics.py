"""Episode aggregation for sim-to-real disturbance diagnostics."""

from typing import Dict

import numpy as np


class RobustnessMetrics:
    """Accumulate tracking, observation and link metrics from environment info."""

    def __init__(self) -> None:
        self.samples = 0
        self.position_error_sum = 0.0
        self.position_error_max = 0.0
        self.velocity_error_sum = 0.0
        self.observation_error_sum = 0.0
        self.airflow_acceleration_sum = 0.0
        self.airflow_acceleration_max = 0.0
        self.minimum_inter_drone_distance = float("inf")
        self.dropped_frames = 0
        self.last_info = None

    def update(self, info: Dict[str, object]) -> None:
        position_error = np.asarray(info["position_tracking_error"])
        velocity_error = np.asarray(info["velocity_tracking_error"])
        observation_error = np.asarray(info["mocap_observation_error"])
        airflow = np.linalg.norm(
            np.asarray(info["inter_drone_airflow_acceleration"]), axis=1
        )
        self.samples += 1
        self.position_error_sum += float(position_error.mean())
        self.position_error_max = max(
            self.position_error_max, float(position_error.max())
        )
        self.velocity_error_sum += float(velocity_error.mean())
        self.observation_error_sum += float(observation_error.mean())
        self.airflow_acceleration_sum += float(airflow.mean())
        self.airflow_acceleration_max = max(
            self.airflow_acceleration_max, float(airflow.max())
        )
        self.minimum_inter_drone_distance = min(
            self.minimum_inter_drone_distance,
            float(info["minimum_inter_drone_distance"]),
        )
        self.dropped_frames += int(info["mocap_frame_dropped"])
        self.last_info = info

    def tensorboard(self) -> Dict[str, float]:
        count = max(self.samples, 1)
        values = {
            "tracking/mean_position_error_m": self.position_error_sum / count,
            "tracking/max_position_error_m": self.position_error_max,
            "tracking/mean_velocity_error_mps": self.velocity_error_sum / count,
            "mocap/mean_observation_error_m": self.observation_error_sum / count,
            "mocap/frame_drop_rate": self.dropped_frames / count,
            "airflow/mean_acceleration_mps2": (
                self.airflow_acceleration_sum / count
            ),
            "airflow/max_acceleration_mps2": self.airflow_acceleration_max,
            "airflow/minimum_inter_drone_distance_m": (
                self.minimum_inter_drone_distance
                if self.samples
                else 0.0
            ),
        }
        if self.last_info is not None:
            info = self.last_info
            values.update(
                {
                    "latency/mocap_steps": info["mocap_delay_steps"],
                    "latency/communication_steps": info[
                        "communication_delay_steps"
                    ],
                    "latency/velocity_command_steps": info[
                        "velocity_command_delay_steps"
                    ],
                    "domain/mass_scale_mean": np.mean(info["mass_scales"]),
                    "domain/actuator_scale_mean": np.mean(
                        info["actuator_scales"]
                    ),
                    "domain/wind_bias_mps2": np.linalg.norm(info["wind_bias"]),
                }
            )
        return values
