"""Observation construction used identically in simulation and real flight."""

import numpy as np

from .config import TaskConfig


BASE_OBS_DIM = 20
OBSTACLE_FEATURES = 3
OBS_DIM = BASE_OBS_DIM + 4 * OBSTACLE_FEATURES


def observation_dim(config: TaskConfig) -> int:
    """Return the per-pursuer feature width for the active team size."""
    return OBS_DIM + 4 * (config.num_pursuers - 3)


def build_observations(
    positions: np.ndarray,
    velocities: np.ndarray,
    origin: np.ndarray,
    config: TaskConfig,
    virtual_obstacles=None,
) -> np.ndarray:
    """Build one obstacle-aware observation for each active pursuer.

    Agent order is stable: pursuers first, evader last. The feature vector contains
    self state, teammates' relative states, evader relative state, and four
    signed distances to the local square geofence.
    """
    positions = np.asarray(positions, dtype=np.float32)
    velocities = np.asarray(velocities, dtype=np.float32)
    origin = np.asarray(origin, dtype=np.float32)
    expected_shape = (config.num_entities, 2)
    if positions.shape != expected_shape or velocities.shape != expected_shape:
        raise ValueError(
            "observations require positions/velocities shaped "
            f"{expected_shape}"
        )

    scale_p = float(config.arena_half_extent)
    scale_v = float(config.velocity_scale)
    obstacles = (
        config.virtual_obstacles
        if virtual_obstacles is None
        else tuple(virtual_obstacles)
    )
    local = positions - origin[None, :]
    result = []
    for i in range(config.num_pursuers):
        own = np.concatenate((local[i] / scale_p, velocities[i] / scale_v))
        teammates = []
        for j in range(config.num_pursuers):
            if j != i:
                teammates.extend((positions[j] - positions[i]) / scale_p)
                teammates.extend((velocities[j] - velocities[i]) / scale_v)
        target = np.concatenate(
            (
                (positions[config.evader_index] - positions[i]) / scale_p,
                (velocities[config.evader_index] - velocities[i]) / scale_v,
            )
        )
        x, y = local[i]
        boundary = (
            np.array(
                [scale_p - x, scale_p + x, scale_p - y, scale_p + y],
                dtype=np.float32,
            )
            / scale_p
        )
        obstacle_features = []
        for index in range(config.max_virtual_obstacles):
            if index < len(obstacles):
                ox, oy, radius = obstacles[index]
                relative = np.array([ox, oy], dtype=np.float32) - local[i]
                clearance = (
                    np.linalg.norm(relative) - radius - config.obstacle_safety_margin
                )
                obstacle_features.extend(relative / scale_p)
                obstacle_features.append(clearance / scale_p)
            else:
                obstacle_features.extend((0.0, 0.0, 2.0))
        result.append(
            np.concatenate((own, teammates, target, boundary, obstacle_features))
        )
    obs = np.asarray(result, dtype=np.float32)
    if obs.shape != (config.num_pursuers, observation_dim(config)):
        raise RuntimeError(f"unexpected observation shape {obs.shape}")
    return np.clip(obs, -5.0, 5.0)


def point_in_triangle(point: np.ndarray, triangle: np.ndarray) -> bool:
    """Return whether a 2-D point is inside a non-degenerate triangle."""
    a, b, c = triangle
    v0, v1, v2 = c - a, b - a, point - a
    d00, d01 = np.dot(v0, v0), np.dot(v0, v1)
    d02, d11, d12 = np.dot(v0, v2), np.dot(v1, v1), np.dot(v1, v2)
    denominator = d00 * d11 - d01 * d01
    if abs(float(denominator)) < 1e-8:
        return False
    u = (d11 * d02 - d01 * d12) / denominator
    v = (d00 * d12 - d01 * d02) / denominator
    return bool(u >= 0.0 and v >= 0.0 and u + v <= 1.0)
