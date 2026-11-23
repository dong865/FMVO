"""Shared pairwise safety projection for simulation and real flight."""

from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np


@dataclass(frozen=True)
class SafetyResult:
    """Result of projecting velocity commands onto pairwise safety constraints."""

    velocities: np.ndarray
    intervened: bool
    emergency: bool
    minimum_distance: float


@dataclass(frozen=True)
class ObstacleSafetyResult:
    """Result of projecting commands away from virtual obstacles."""

    velocities: np.ndarray
    intervened: bool
    emergency: bool
    minimum_clearance: float


def _as_agent_vectors(values: np.ndarray, name: str) -> np.ndarray:
    """Return finite floating-point agent vectors after validating their shape."""
    vectors = np.asarray(values, dtype=float)
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError(f"{name} must have shape (num_agents, dimensions)")
    if not np.all(np.isfinite(vectors)):
        raise ValueError(f"{name} must contain only finite values")
    return vectors


def pairwise_distances(positions: np.ndarray) -> np.ndarray:
    """Compute symmetric Euclidean distances with an infinite diagonal."""
    position_vectors = _as_agent_vectors(positions, "positions")
    offsets = position_vectors[:, None, :] - position_vectors[None, :, :]
    distances = np.linalg.norm(offsets, axis=-1)
    np.fill_diagonal(distances, np.inf)
    return distances


def _speed_limit_array(
    speed_limits: Union[float, np.ndarray], num_agents: int
) -> np.ndarray:
    limits = np.asarray(speed_limits, dtype=float)
    if limits.ndim == 0:
        limits = np.full(num_agents, float(limits))
    if limits.shape != (num_agents,):
        raise ValueError("speed_limits must be a scalar or one value per agent")
    if not np.all(np.isfinite(limits)) or np.any(limits <= 0.0):
        raise ValueError("speed_limits must be finite and positive")
    return limits


def _clip_speeds(velocities: np.ndarray, limits: np.ndarray) -> np.ndarray:
    speeds = np.linalg.norm(velocities, axis=1)
    scales = np.minimum(1.0, limits / np.maximum(speeds, 1e-12))
    return velocities * scales[:, None]


def project_pairwise_velocity(
    velocities: np.ndarray,
    positions: np.ndarray,
    warning_distance: float,
    emergency_distance: float,
    separation_speed: float,
    speed_limits: Union[float, np.ndarray],
) -> SafetyResult:
    """Remove closing motion near agents and separate emergency pairs."""
    commands = _as_agent_vectors(velocities, "velocities")
    position_vectors = _as_agent_vectors(positions, "positions")
    if commands.shape != position_vectors.shape:
        raise ValueError("velocities and positions must have identical shapes")
    if not np.isfinite(warning_distance) or not np.isfinite(emergency_distance):
        raise ValueError("safety distances must be finite")
    if warning_distance <= emergency_distance or emergency_distance <= 0.0:
        raise ValueError("warning_distance must exceed positive emergency_distance")
    if not np.isfinite(separation_speed) or separation_speed < 0.0:
        raise ValueError("separation_speed must be finite and non-negative")

    limits = _speed_limit_array(speed_limits, commands.shape[0])
    projected = commands.copy()
    distances = pairwise_distances(position_vectors)
    minimum_distance = float(np.min(distances))
    emergency = minimum_distance < emergency_distance

    for _ in range(3):
        for first in range(commands.shape[0] - 1):
            for second in range(first + 1, commands.shape[0]):
                distance = distances[first, second]
                if distance >= warning_distance:
                    continue
                displacement = position_vectors[first] - position_vectors[second]
                if distance <= 1e-12:
                    normal = np.zeros(commands.shape[1], dtype=float)
                    normal[first % commands.shape[1]] = 1.0
                else:
                    normal = displacement / distance
                relative_radial_speed = float(
                    np.dot(projected[first] - projected[second], normal)
                )
                urgency = np.clip(
                    (warning_distance - distance)
                    / (warning_distance - emergency_distance),
                    0.0,
                    1.0,
                )
                target_speed = separation_speed * urgency
                if relative_radial_speed < target_speed:
                    correction = 0.5 * (target_speed - relative_radial_speed) * normal
                    projected[first] += correction
                    projected[second] -= correction

    projected = _clip_speeds(projected, limits)
    intervened = not np.allclose(projected, commands, rtol=0.0, atol=1e-7)
    return SafetyResult(
        velocities=projected,
        intervened=intervened,
        emergency=emergency,
        minimum_distance=minimum_distance,
    )


def project_obstacle_velocity(
    velocities: np.ndarray,
    positions: np.ndarray,
    origin: np.ndarray,
    obstacles: Sequence[Sequence[float]],
    safety_margin: float,
    warning_margin: float,
    separation_speed: float,
    speed_limits: Union[float, np.ndarray],
) -> ObstacleSafetyResult:
    """Project horizontal commands away from cylindrical virtual obstacles."""
    commands = _as_agent_vectors(velocities, "velocities")
    position_vectors = _as_agent_vectors(positions, "positions")
    if commands.shape != position_vectors.shape or commands.shape[1] != 2:
        raise ValueError("obstacle projection requires matching (agents, 2) arrays")
    origin_vector = np.asarray(origin, dtype=float)
    if origin_vector.shape != (2,) or not np.all(np.isfinite(origin_vector)):
        raise ValueError("origin must be a finite two-dimensional vector")
    if safety_margin <= 0.0 or warning_margin <= 0.0:
        raise ValueError("obstacle safety and warning margins must be positive")
    if separation_speed < 0.0 or not np.isfinite(separation_speed):
        raise ValueError("separation_speed must be finite and non-negative")

    limits = _speed_limit_array(speed_limits, commands.shape[0])
    projected = commands.copy()
    emergency = False
    minimum_clearance = float("inf")
    for obstacle in obstacles:
        if len(obstacle) != 3:
            raise ValueError("each obstacle must contain x, y, and radius")
        offset_x, offset_y, radius = (float(value) for value in obstacle)
        if radius <= 0.0:
            raise ValueError("obstacle radius must be positive")
        center = origin_vector + np.array([offset_x, offset_y])
        safe_radius = radius + safety_margin
        activation_radius = safe_radius + warning_margin
        for agent in range(commands.shape[0]):
            displacement = position_vectors[agent] - center
            distance = float(np.linalg.norm(displacement))
            clearance = distance - safe_radius
            minimum_clearance = min(minimum_clearance, clearance)
            if distance <= 1e-12:
                normal = np.array([1.0, 0.0])
            else:
                normal = displacement / distance
            if clearance <= 0.0:
                emergency = True
            if distance >= activation_radius:
                continue
            urgency = np.clip(
                (activation_radius - distance) / warning_margin,
                0.0,
                1.0,
            )
            target_speed = separation_speed * urgency
            radial_speed = float(np.dot(projected[agent], normal))
            if radial_speed < target_speed:
                projected[agent] += (target_speed - radial_speed) * normal

    projected = _clip_speeds(projected, limits)
    intervened = not np.allclose(projected, commands, rtol=0.0, atol=1e-7)
    return ObstacleSafetyResult(
        velocities=projected,
        intervened=intervened,
        emergency=emergency,
        minimum_clearance=minimum_clearance,
    )
