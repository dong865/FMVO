"""Virtual cylindrical-obstacle configuration shared by training and flight."""

import json
from pathlib import Path
from typing import Optional, Sequence, Tuple

from .config import TaskConfig


# Coordinates are relative to the initial four-drone centroid. Each tuple is
# (x [m], y [m], radius [m]); cylinders are treated as spanning flight height.
DEFAULT_VIRTUAL_OBSTACLES = (
    (0.00, 0.55, 0.16),
    (0.55, -0.35, 0.16),
)


def validate_obstacles(
    obstacles: Sequence[Sequence[float]], maximum: int
) -> Tuple[Tuple[float, float, float], ...]:
    if len(obstacles) > maximum:
        raise ValueError(
            f"at most {maximum} virtual obstacles are supported; "
            f"received {len(obstacles)}"
        )
    validated = []
    for index, obstacle in enumerate(obstacles):
        if len(obstacle) != 3:
            raise ValueError(f"obstacle {index} must contain [x, y, radius]")
        x, y, radius = (float(value) for value in obstacle)
        if radius <= 0.0:
            raise ValueError(f"obstacle {index} radius must be positive")
        validated.append((x, y, radius))
    return tuple(validated)


def configure_obstacles(
    config: TaskConfig,
    path: Optional[str] = None,
    use_defaults: bool = True,
) -> TaskConfig:
    """Apply built-in obstacles or an explicit JSON layout to ``config``."""
    if path is None:
        obstacles = DEFAULT_VIRTUAL_OBSTACLES if use_defaults else ()
        margin = config.obstacle_safety_margin
    else:
        source = Path(path).expanduser().resolve()
        with source.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if payload.get("coordinate_frame") != "initial_swarm_centroid":
            raise ValueError(
                "obstacle coordinate_frame must be initial_swarm_centroid"
            )
        entries = payload.get("obstacles", [])
        obstacles = [
            (entry["x"], entry["y"], entry["radius"])
            for entry in entries
        ]
        margin = float(
            payload.get("safety_margin", config.obstacle_safety_margin)
        )
        if margin <= 0.0:
            raise ValueError("obstacle safety_margin must be positive")
    config.virtual_obstacles = validate_obstacles(
        obstacles, config.max_virtual_obstacles
    )
    config.obstacle_safety_margin = margin
    return config


def obstacle_centers_world(config: TaskConfig, origin):
    """Return ``(center_x, center_y, radius)`` in the current world frame."""
    return tuple(
        (origin[0] + x, origin[1] + y, radius)
        for x, y, radius in config.virtual_obstacles
    )
