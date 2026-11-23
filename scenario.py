"""Deterministic episode geometry for goal-directed pursuit-evasion."""

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from .config import TaskConfig


Point = Tuple[float, float]
Obstacle = Tuple[float, float, float]


@dataclass(frozen=True)
class SceneSamplingOptions:
    """Spawn/route/solvability knobs for :class:`ScenarioGenerator`.

    Deliberately NOT part of TaskConfig: TaskConfig feeds the task fingerprint,
    so adding fields there would invalidate every existing checkpoint/dataset.
    Scene tuning therefore lives here, defaulting to byte-identical legacy
    sampling (options=None in the generator reproduces the pre-refactor code).
    """

    # --- pursuer team template (route-relative to the evader spawn) ---------
    interceptor_distance_range: Tuple[float, float] = (1.30, 1.50)
    """Distance of the interceptor ahead of the evader along the route."""
    interceptor_lateral_half_width: float = 0.08
    """Jitter half-width of the interceptor across the route."""
    flank_forward_range: Tuple[float, float] = (0.85, 0.95)
    """Forward (along-route) distance band of the two flank pursuers."""
    flank_lateral_range: Tuple[float, float] = (0.65, 0.75)
    """Lateral (across-route) distance band of the two flank pursuers."""
    spawn_min_separation: Optional[float] = None
    """Optional override of config.spawn_min_separation."""
    spawn_distance_range: Optional[Tuple[float, float]] = None
    """Optional override of config.pursuer_spawn_distance_range (evader-to-
    pursuer distance band accepted by the generator).  Widening/raising this
    band is the main difficulty axis: a farther team needs more time to close,
    during which the faster evader can slip away."""
    spawn_layout: str = "route_arc"
    """'route_arc' keeps the benchmark template; 'scattered' independently
    distributes pursuers across the map for GUI demonstrations; 'arc' places
    N>=3 pursuers on a forward-facing arc (used for hard 4V1/5V1)."""
    arc_half_spread: float = 0.85
    """Half-width (radians) of the forward-facing arc used when
    spawn_layout='arc' and num_pursuers > 3."""

    route_layout: str = "diagonal"
    """'diagonal' uses the benchmark corner-to-corner route. 'interior'
    starts the evader near the arena centre with a long, randomly oriented
    escape route, leaving room to evade before reaching a boundary."""

    # --- route / goal -------------------------------------------------------
    goal_distance_range: Optional[Tuple[float, float]] = None
    """When set, the goal is placed at a sampled route distance from the evader
    spawn (instead of the legacy opposite-corner sector). Harder scenes use
    shorter routes: less time for the pursuers to establish capture."""
    corner_clearance_range: Optional[Tuple[float, float]] = None
    """Override of config.diagonal_boundary_clearance_range (corner sectors)."""
    route_min_distance: Optional[float] = None
    """Override of config.evader_goal_min_distance (used by legacy corner mode)."""

    # --- solvability prefilter ----------------------------------------------
    solvability_mode: str = "none"
    """'none' (legacy, no capturability check) or 'analytic' (fast conservative
    open-field bound, documented caveat: obstacles ignored, so it guarantees
    solvability for the obstacle-free/light cases and is validated empirically
    against an omniscient-boosted oracle in scan_scene_solvability.py)."""
    solvability_speed_multiplier: float = 1.0
    """Pursuer worst-case speed used by the analytic bound as a fraction of
    config.max_speed (1.0 = nominal; <1 = conservative)."""
    solvability_hold_margin_seconds: float = 0.3
    """Extra seconds the analytic bound demands beyond capture hold + arrival."""
    solvability_ring_radius: float = 0.55
    """Radius of the equilateral capture ring used by the analytic bound."""

    # --- diagnostics ---------------------------------------------------------
    mode_name: str = "legacy"

    # --- obstacle placement --------------------------------------------------
    obstacle_mode: str = "random"
    """'random' (legacy procedural placement) or 'corridor': place 2-3 larger
    obstacles in the mid-route band at a lateral offset from the evader route,
    channelling the pursuers while keeping the evader's own route free (the
    free_route_exists check still applies to the evader only)."""
    corridor_obstacle_count_range: Tuple[int, int] = (2, 3)
    corridor_obstacle_radius_range: Tuple[float, float] = (0.22, 0.32)
    corridor_fraction_range: Tuple[float, float] = (0.30, 0.75)
    """Route fraction band along which corridor obstacles are centred."""
    corridor_lateral_band: Tuple[float, float] = (0.55, 1.05)
    """Lateral offset band of corridor obstacles from the route."""


# Difficulty presets for the scene audit / benchmark sweep.
def scene_preset(name: str, **overrides) -> SceneSamplingOptions:
    """Named difficulty presets built on top of the legacy template."""
    base = SceneSamplingOptions()
    presets: dict = {
        # Spread the flank arc out and pull the intercept line back: gaps the
        # (faster) evader can exploit, but geometry still lets a coordinated
        # trio capture from close range.
        "hard_flanks": dict(
            interceptor_distance_range=(1.05, 1.25),
            interceptor_lateral_half_width=0.12,
            flank_forward_range=(0.55, 0.70),
            flank_lateral_range=(0.90, 1.15),
            mode_name="hard_flanks",
        ),
        # Shorten the evader's route to its goal: less time to establish
        # capture before the evader can reach it.
        "hard_route": dict(
            goal_distance_range=(2.4, 3.2),
            mode_name="hard_route",
        ),
        # Both: spread arc AND short route.
        "hard_both": dict(
            interceptor_distance_range=(1.05, 1.25),
            interceptor_lateral_half_width=0.12,
            flank_forward_range=(0.55, 0.70),
            flank_lateral_range=(0.90, 1.15),
            goal_distance_range=(2.4, 3.2),
            mode_name="hard_both",
        ),
        # Strategic obstacles in the mid-route band channelling the pursuers.
        "hard_obstacles": dict(
            obstacle_mode="corridor",
            corridor_obstacle_count_range=(2, 3),
            corridor_obstacle_radius_range=(0.22, 0.32),
            mode_name="hard_obstacles",
        ),
        # Farther spawns: template scaled up and the accepted distance band
        # widened/raised. far1 ~1.8m, far2 ~2.3m, far3 ~2.8m mean separation.
        "far1": dict(
            interceptor_distance_range=(1.60, 1.85),
            flank_forward_range=(1.10, 1.35),
            flank_lateral_range=(1.10, 1.40),
            spawn_distance_range=(1.30, 2.10),
            mode_name="far1",
        ),
        "far2": dict(
            interceptor_distance_range=(2.05, 2.35),
            flank_forward_range=(1.45, 1.75),
            flank_lateral_range=(1.55, 1.95),
            spawn_distance_range=(1.70, 2.80),
            mode_name="far2",
        ),
        "far3": dict(
            interceptor_distance_range=(2.55, 2.95),
            flank_forward_range=(1.85, 2.25),
            flank_lateral_range=(1.95, 2.40),
            spawn_distance_range=(2.10, 3.40),
            mode_name="far3",
        ),
        # Hard multi-pursuer presets: forward-facing arc with far-like distances.
        # These mirror far1 spacing but use the arc layout for teams > 3.
        "hard_4v1": dict(
            spawn_layout="arc",
            spawn_distance_range=(1.30, 2.10),
            arc_half_spread=1.05,
            mode_name="hard_4v1",
        ),
        "hard_5v1": dict(
            spawn_layout="arc",
            spawn_distance_range=(1.30, 2.10),
            arc_half_spread=1.15,
            mode_name="hard_5v1",
        ),
        # Wide open gaps in the initial arc (teacher must out-think, not out-run).
        "hard_gap": dict(
            interceptor_distance_range=(0.70, 0.95),
            interceptor_lateral_half_width=0.30,
            flank_forward_range=(0.30, 0.55),
            flank_lateral_range=(1.15, 1.60),
            mode_name="hard_gap",
        ),
        # Strategic obstacles on a short route: channelled intercept under time.
        "hard_choke": dict(
            goal_distance_range=(2.4, 3.2),
            obstacle_mode="corridor",
            corridor_obstacle_count_range=(2, 3),
            corridor_obstacle_radius_range=(0.22, 0.32),
            mode_name="hard_choke",
        ),
        "gui_scattered": dict(
            spawn_layout="scattered",
            spawn_distance_range=(1.45, 6.0),
            mode_name="gui_scattered",
        ),
        "natural_pursuit": dict(
            spawn_layout="scattered",
            spawn_distance_range=(1.45, 6.0),
            mode_name="natural_pursuit",
        ),
        # Pursuit visualization/evaluation geometry: the team begins behind
        # and to the sides of an interior evader.  There is deliberately no
        # interceptor waiting on the escape line, so capture requires a chase
        # followed by coordinated lateral closure.
        "open_pursuit": dict(
            route_layout="interior",
            spawn_layout="pursuit_fan",
            spawn_distance_range=(1.75, 2.15),
            arc_half_spread=1.45,
            goal_distance_range=(2.90, 3.05),
            mode_name="open_pursuit",
        ),
    }
    if name in ("legacy", "none"):
        return SceneSamplingOptions(mode_name="legacy")
    if name == "hard":
        # Recommended difficulty preset: far1 geometry (teacher ~0.80, oracle
        # solvable ~100%), exposed under a stable benchmark name.
        name = "far1"
    if name not in presets:
        raise ValueError(f"unknown scene preset {name!r}; known: {sorted(presets)}")
    merged = dict(presets[name])
    merged.update(overrides)
    merged["mode_name"] = name
    return SceneSamplingOptions(**merged)


@dataclass(frozen=True)
class EpisodeScenario:
    """Immutable geometry sampled once at the start of an episode."""

    pursuer_starts: Tuple[Point, Point, Point]
    evader_start: Point
    evader_goal: Point
    virtual_obstacles: Tuple[Obstacle, ...]
    diagonal_index: int
    route_length: float


def _point_in_triangle(point: np.ndarray, triangle: np.ndarray) -> bool:
    signs = []
    for index in range(3):
        edge = triangle[(index + 1) % 3] - triangle[index]
        relative = point - triangle[index]
        signs.append(float(edge[0] * relative[1] - edge[1] * relative[0]))
    return (
        all(value >= -1e-10 for value in signs)
        or all(value <= 1e-10 for value in signs)
    )


def _minimum_pairwise_distance(points: np.ndarray) -> float:
    return min(
        float(np.linalg.norm(points[left] - points[right]))
        for left in range(len(points))
        for right in range(left + 1, len(points))
    )


def sample_scattered_pursuers(
    rng,
    evader,
    arena_half_extent,
    count=3,
    minimum_evader_distance=1.45,
    minimum_pairwise_distance=1.25,
    required_team_diameter=2.75,
    attempts=2048,
):
    """Independently distribute pursuers across the arena without enclosure."""
    evader = np.asarray(evader, dtype=np.float64)
    limit = float(arena_half_extent) - 0.10
    for _ in range(int(attempts)):
        pursuers = rng.uniform(-limit, limit, size=(int(count), 2))
        distances = np.linalg.norm(pursuers - evader, axis=1)
        if float(np.min(distances)) < minimum_evader_distance:
            continue
        if _minimum_pairwise_distance(pursuers) < minimum_pairwise_distance:
            continue
        team_diameter = max(
            float(np.linalg.norm(pursuers[i] - pursuers[j]))
            for i in range(int(count))
            for j in range(i + 1, int(count))
        )
        if team_diameter < required_team_diameter:
            continue
        if float(np.ptp(distances)) < 0.35:
            continue
        if int(count) == 3 and _point_in_triangle(evader, pursuers):
            continue
        return pursuers
    raise RuntimeError("failed to sample map-wide scattered pursuers")


def _blocked(
    point: np.ndarray,
    obstacles: Sequence[Obstacle],
    margin: float,
) -> bool:
    return any(
        np.linalg.norm(point - np.asarray((x, y))) <= radius + margin
        for x, y, radius in obstacles
    )


def free_route_exists(
    start: Sequence[float],
    goal: Sequence[float],
    obstacles: Sequence[Obstacle],
    config: TaskConfig,
) -> bool:
    """Check start-to-goal reachability on an obstacle-inflated 8-neighbour grid."""

    if not obstacles:
        return True
    extent = config.arena_half_extent
    grid_size = 45
    coordinates = np.linspace(-extent, extent, grid_size)

    def nearest_index(point: Sequence[float]) -> Tuple[int, int]:
        value = np.asarray(point, dtype=np.float64)
        column = int(np.argmin(np.abs(coordinates - value[0])))
        row = int(np.argmin(np.abs(coordinates - value[1])))
        return row, column

    start_cell = nearest_index(start)
    goal_cell = nearest_index(goal)
    occupied = np.zeros((grid_size, grid_size), dtype=bool)
    for row, y_value in enumerate(coordinates):
        for column, x_value in enumerate(coordinates):
            occupied[row, column] = _blocked(
                np.asarray((x_value, y_value)),
                obstacles,
                config.obstacle_safety_margin,
            )
    occupied[start_cell] = False
    occupied[goal_cell] = False
    frontier = deque((start_cell,))
    visited = {start_cell}
    neighbours = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    )
    while frontier:
        row, column = frontier.popleft()
        if (row, column) == goal_cell:
            return True
        for row_step, column_step in neighbours:
            candidate = row + row_step, column + column_step
            if (
                0 <= candidate[0] < grid_size
                and 0 <= candidate[1] < grid_size
                and not occupied[candidate]
                and candidate not in visited
            ):
                visited.add(candidate)
                frontier.append(candidate)
    return False


def _straight_time(
    distance: float,
    max_speed: float,
    max_accel: float,
) -> float:
    """Time for a unit starting from rest to cover ``distance`` with a speed
    cap and a trapezoidal velocity ramp (v = a t up to v_max)."""
    if distance <= 0.0:
        return 0.0
    ramp_distance = max_speed * max_speed / (2.0 * max_accel)
    if distance <= ramp_distance:
        return float(np.sqrt(2.0 * distance / max_accel))
    return float(
        max_speed / max_accel + (distance - ramp_distance) / max_speed
    )


def _straight_travel(time: float, max_speed: float, max_accel: float) -> float:
    """Inverse of :func:`_straight_time` for a unit starting from rest."""
    if time <= 0.0:
        return 0.0
    ramp_time = max_speed / max_accel
    if time <= ramp_time:
        return 0.5 * max_accel * time * time
    ramp_distance = 0.5 * max_accel * ramp_time * ramp_time
    return float(ramp_distance + max_speed * (time - ramp_time))


def _analytic_solvable_open_field(
    start: np.ndarray,
    goal: np.ndarray,
    pursuers: np.ndarray,
    config: TaskConfig,
    options: SceneSamplingOptions,
) -> Tuple[bool, str]:
    """Fast conservative open-field solvability bound (obstacles ignored).

    A scene is accepted only if there is a route point ``x`` ahead of the
    evader where all three pursuers can reach an enclosing ring (equilateral
    capture triangle of radius ``capture_radius``, best rotation + assignment)
    no later than the evader's own straight-line arrival, plus a hold+slack
    margin, and still before the evader's goal.  ``options.solvability_mode``
    must be 'analytic' for this to run.  This is an optimistic-for-pursuers
    model (straight-line evader), so leaks are quantified empirically against
    an omniscient-boosted oracle in ``scan_scene_solvability.py`` and the
    margin parameters below are the tuning knobs.
    """
    speed_multiplier = max(options.solvability_speed_multiplier, 1e-3)
    pursuer_speed = config.max_speed * speed_multiplier
    pursuer_accel = config.max_accel * speed_multiplier
    evader_speed = config.evader_max_speed
    evader_accel = config.evader_max_accel

    route = goal - start
    route_length = float(np.linalg.norm(route))
    if route_length <= 1e-6:
        return False, "degenerate route"
    forward = route / route_length
    lateral = np.asarray((-forward[1], forward[0]))

    time_to_goal = _straight_time(
        max(0.0, route_length - config.evader_goal_radius),
        evader_speed,
        evader_accel,
    )
    # Never solvable if even the goal-line reach is impossible within episode.
    episode_budget = config.episode_seconds

    ring_radius = options.solvability_ring_radius
    vertices = np.asarray(
        [
            np.cos(np.arange(3) * 2.0 * np.pi / 3.0),
            np.sin(np.arange(3) * 2.0 * np.pi / 3.0),
        ]
    ).T * ring_radius
    # candidate intercept points along the route
    probe = np.linspace(0.8, min(route_length - config.evader_goal_radius, 3.5), 12)
    if probe.size == 0 or route_length < 1.2:
        return False, "route too short for an intercept probe"
    best_margin = -np.inf
    best_detail = ""
    for x in probe:
        centre = start + forward * x
        # evader arrives at x after travelling x (straight)
        evader_arrival = _straight_time(
            x, evader_speed, evader_accel
        )
        if evader_arrival > time_to_goal + 1e-9:
            continue
        best_rotation_margin = -np.inf
        for rotation in np.linspace(0.0, np.pi / 3.0, 5, endpoint=False):
            rotated = vertices @ np.asarray(
                ((np.cos(rotation), -np.sin(rotation)),
                 (np.sin(rotation), np.cos(rotation)))
            )
            target = centre + rotated
            best_bottleneck = float("inf")
            for permutation in ((0, 1, 2), (0, 2, 1), (1, 0, 2),
                                (1, 2, 0), (2, 0, 1), (2, 1, 0)):
                arrival = 0.0
                for index, vertex in enumerate(permutation):
                    distance = float(
                        np.linalg.norm(pursuers[index] - target[vertex])
                    )
                    arrival = max(
                        arrival,
                        _straight_time(distance, pursuer_speed, pursuer_accel),
                    )
                best_bottleneck = min(best_bottleneck, arrival)
            # capture must be established before the evader can leave the ring
            margin = evader_arrival - (
                best_bottleneck + config.capture_hold_seconds
            )
            best_rotation_margin = max(best_rotation_margin, margin)
        if best_rotation_margin > best_margin:
            best_margin = best_rotation_margin
            best_detail = f"x={x:.2f}"
    slack = options.solvability_hold_margin_seconds
    feasible_goal = time_to_goal <= episode_budget - 1.0
    if best_margin >= slack and feasible_goal:
        return True, ""
    return False, (
        f"analytic bound fails (best margin {best_margin:.2f}s < slack {slack}s; "
        f"goal time {time_to_goal:.2f}s; {best_detail})"
    )


class ScenarioGenerator:
    """Sample reproducible diagonal routes and safe randomized geometry."""

    _START_SIGNS = (
        (-1.0, -1.0),
        (-1.0, 1.0),
        (1.0, 1.0),
        (1.0, -1.0),
    )

    def __init__(
        self,
        config: TaskConfig,
        rng: np.random.Generator,
        options: Optional[SceneSamplingOptions] = None,
    ):
        self.config = config
        self.rng = rng
        self.options = options or SceneSamplingOptions()
        self.reject_counts: dict = {}
        self.accept_count = 0

    def _reject(self, reason: str) -> None:
        self.reject_counts[reason] = self.reject_counts.get(reason, 0) + 1

    def sample(self) -> EpisodeScenario:
        last_failure = "no candidate sampled"
        for _ in range(self.config.scenario_sampling_attempts):
            candidate, last_failure = self._sample_candidate()
            if candidate is not None:
                self.accept_count += 1
                return candidate
        raise RuntimeError(
            "failed to sample a valid scenario after "
            f"{self.config.scenario_sampling_attempts} attempts: {last_failure}"
        )

    def _sample_candidate(self):
        options = self.options
        config = self.config
        diagonal_index = int(self.rng.integers(0, 4))
        clearance_low, clearance_high = (
            options.corner_clearance_range
            or config.diagonal_boundary_clearance_range
        )
        if options.route_layout == "interior":
            # Near-diagonal routes provide about 3 m of usable escape distance
            # from the centre while keeping both endpoints inside the square.
            diagonal_heading = (
                -0.75 * np.pi + 0.5 * np.pi * diagonal_index
            )
            heading = float(diagonal_heading + self.rng.uniform(-0.05, 0.05))
            direction = np.asarray((np.cos(heading), np.sin(heading)))
            start = self.rng.uniform(-0.15, 0.15, size=2)
            low, high = options.goal_distance_range or (2.90, 3.05)
            target_distance = float(self.rng.uniform(low, high))
            goal = start + direction * target_distance
            diagonal_index = int(((heading + np.pi) % (2.0 * np.pi)) // (0.5 * np.pi))
        else:
            start_sign = np.asarray(self._START_SIGNS[diagonal_index])
            goal_sign = -start_sign
            start = start_sign * (
                config.arena_half_extent
                - self.rng.uniform(clearance_low, clearance_high, size=2)
            )
            goal = None
        if goal is None and options.goal_distance_range is None:
            goal = goal_sign * (
                config.arena_half_extent
                - self.rng.uniform(clearance_low, clearance_high, size=2)
            )
        elif goal is None:
            # Place the goal on the evader's route at a sampled distance.
            anchor = goal_sign * (config.arena_half_extent - clearance_low)
            route = anchor - start
            direction = route / float(np.linalg.norm(route))
            low, high = options.goal_distance_range
            low = max(low, 1.4)
            target_distance = float(
                self.rng.uniform(
                    low, min(high, float(np.linalg.norm(route)) - 1e-6)
                )
            )
            goal = start + direction * target_distance
            route_min_floor = (
                options.route_min_distance
                if options.route_min_distance is not None
                else 1.4
            )
            if target_distance < route_min_floor:
                return None, "goal-distance route below floor"
            if np.max(np.abs(goal)) > config.arena_half_extent - 0.15:
                return None, "goal-distance goal outside arena margin"
        route = goal - start
        route_length = float(np.linalg.norm(route))
        route_min = (
            options.route_min_distance
            if options.route_min_distance is not None
            else config.evader_goal_min_distance
        )
        if options.goal_distance_range is None and route_length < route_min:
            return None, "diagonal route shorter than minimum"
        forward = route / route_length
        lateral = np.asarray((-forward[1], forward[0]))

        num_pursuers = int(config.num_pursuers)
        if options.spawn_layout == "pursuit_fan":
            # Rear-facing fan: nobody is pre-positioned between the evader and
            # its goal.  The slight angular/radial jitter prevents a visibly
            # scripted formation while preserving the open escape corridor.
            distance_low, distance_high = options.spawn_distance_range
            half_spread = float(options.arc_half_spread)
            offsets = []
            for index in range(num_pursuers):
                fraction = index / max(num_pursuers - 1, 1)
                spread = -half_spread + 2.0 * half_spread * fraction
                spread += float(self.rng.uniform(-0.035, 0.035))
                radius = float(self.rng.uniform(distance_low, distance_high))
                unit = -np.cos(spread) * forward + np.sin(spread) * lateral
                offsets.append(radius * unit)
            pursuers = start[None, :] + np.asarray(offsets, dtype=float)
        elif options.spawn_layout == "scattered":
            pursuers = sample_scattered_pursuers(
                self.rng,
                start,
                config.arena_half_extent,
                count=num_pursuers,
            )
        elif num_pursuers == 3:
            interceptor_distance = float(
                self.rng.uniform(*options.interceptor_distance_range)
            )
            interceptor_offset = (
                forward * interceptor_distance
                + lateral
                * self.rng.uniform(
                    -options.interceptor_lateral_half_width,
                    options.interceptor_lateral_half_width,
                )
            )
            flank_forward = self.rng.uniform(*options.flank_forward_range, size=2)
            flank_lateral = self.rng.uniform(*options.flank_lateral_range, size=2)
            pursuers = np.asarray(
                (
                    start + interceptor_offset,
                    start + forward * flank_forward[0]
                    + lateral * flank_lateral[0],
                    start + forward * flank_forward[1]
                    - lateral * flank_lateral[1],
                )
            )
        elif num_pursuers > 3 and options.spawn_layout == "arc":
            # Hard 4V1/5V1: pursuers on a forward-facing arc with configurable
            # spread and far-like distances. The arc is centered on the route
            # direction so the evader is not enclosed.
            distance_low, distance_high = (
                options.spawn_distance_range
                if options.spawn_distance_range is not None
                else config.pursuer_spawn_distance_range
            )
            low_r = max(float(distance_low), 0.5)
            high_r = float(distance_high)
            half_spread = float(options.arc_half_spread)
            offsets = []
            for index in range(num_pursuers):
                spread = -half_spread + 2.0 * half_spread * index / max(
                    num_pursuers - 1, 1
                )
                radius = float(self.rng.uniform(low_r, high_r))
                unit = np.cos(spread) * forward + np.sin(spread) * lateral
                jitter = self.rng.normal(0.0, 0.04, size=2)
                offsets.append(radius * unit + jitter)
            pursuers = start[None, :] + np.asarray(offsets, dtype=float)
        else:
            # 4V1/5V1: pursuers form a fan AHEAD of the evader along the route,
            # staying inside the arena and never enclosing the evader. Larger
            # spreads would push spawns toward the walls (rejected), so the fan
            # half-width is capped and the arena checks reject rare outliers.
            distance_low, distance_high = (
                options.spawn_distance_range
                if options.spawn_distance_range is not None
                else config.pursuer_spawn_distance_range
            )
            low_r = max(float(distance_low), 0.5)
            high_r = min(float(distance_high), 1.5)
            half_spread = 0.85
            offsets = []
            for index in range(num_pursuers):
                spread = -half_spread + 2.0 * half_spread * index / max(
                    num_pursuers - 1, 1
                )
                radius = float(self.rng.uniform(low_r, high_r))
                unit = np.cos(spread) * forward + np.sin(spread) * lateral
                jitter = self.rng.normal(0.0, 0.06, size=2)
                offsets.append(radius * unit + jitter)
            pursuers = start[None, :] + np.asarray(offsets, dtype=float)
        pursuers = pursuers[self.rng.permutation(num_pursuers)]
        all_starts = np.vstack((pursuers, start))
        if np.max(np.abs(all_starts)) > config.arena_half_extent:
            return None, "spawn outside arena"
        distances = np.linalg.norm(pursuers - start, axis=1)
        distance_low, distance_high = (
            options.spawn_distance_range
            if options.spawn_distance_range is not None
            else config.pursuer_spawn_distance_range
        )
        if np.any(distances < distance_low) or np.any(distances > distance_high):
            return None, "pursuer outside route-relative spawn range"
        spawn_separation = (
            options.spawn_min_separation
            if options.spawn_min_separation is not None
            else config.spawn_min_separation
        )
        if _minimum_pairwise_distance(all_starts) < spawn_separation:
            return None, "spawn separation below contract"
        if num_pursuers == 3 and _point_in_triangle(start, pursuers):
            return None, "evader initially enclosed"
        if num_pursuers > 3:
            # angular-coverage enclosure check: evader must not be encircled
            relative = pursuers - start
            angles = np.sort(np.arctan2(relative[:, 1], relative[:, 0]))
            wrapped = np.concatenate((angles, angles[:1] + 2.0 * np.pi))
            if float(np.max(np.diff(wrapped))) < np.pi:
                return None, "evader initially enclosed"

        if options.solvability_mode == "analytic":
            solvable, reason = _analytic_solvable_open_field(
                start, goal, pursuers, config, options
            )
            if not solvable:
                self._reject("analytic_unsolvable: " + reason)
                return None, "analytic solvability bound fails"

        obstacles = self._sample_obstacles(
            (*pursuers, start, goal), start, goal
        )
        if obstacles is None:
            return None, "could not place safe procedural obstacles"
        if not free_route_exists(start, goal, obstacles, config):
            return None, "procedural obstacles block start-to-goal route"
        scene = EpisodeScenario(
            pursuer_starts=tuple(
                (float(point[0]), float(point[1])) for point in pursuers
            ),
            evader_start=(float(start[0]), float(start[1])),
            evader_goal=(float(goal[0]), float(goal[1])),
            virtual_obstacles=obstacles,
            diagonal_index=diagonal_index,
            route_length=route_length,
        )
        return scene, ""

    def _sample_obstacles(
        self,
        protected_points: Iterable[Sequence[float]],
        start: Sequence[float] = (0.0, 0.0),
        goal: Sequence[float] = (0.0, 0.0),
    ):
        if not self.config.procedural_obstacles:
            return tuple(tuple(item) for item in self.config.virtual_obstacles)
        if self.rng.random() < self.config.obstacle_free_probability:
            return ()
        options = self.options
        if options.obstacle_mode == "corridor":
            return self._sample_corridor_obstacles(
                protected_points, start, goal
            )
        count_low, count_high = self.config.procedural_obstacle_count_range
        count = int(self.rng.integers(count_low, count_high + 1))
        radius_low, radius_high = self.config.procedural_obstacle_radius_range
        protected = tuple(np.asarray(point) for point in protected_points)
        obstacles = []
        placement_attempts = max(32, 16 * count)
        for _ in range(placement_attempts):
            if len(obstacles) == count:
                break
            radius = float(self.rng.uniform(radius_low, radius_high))
            limit = (
                self.config.arena_half_extent
                - radius
                - self.config.obstacle_safety_margin
            )
            centre = self.rng.uniform(-limit, limit, size=2)
            if any(
                np.linalg.norm(centre - point)
                < radius + self.config.obstacle_safety_margin
                for point in protected
            ):
                continue
            if any(
                np.linalg.norm(centre - np.asarray((x, y)))
                < radius + other_radius + self.config.obstacle_safety_margin
                for x, y, other_radius in obstacles
            ):
                continue
            obstacles.append((float(centre[0]), float(centre[1]), radius))
        if len(obstacles) != count:
            return None
        return tuple(obstacles)
    def _sample_corridor_obstacles(
        self,
        protected_points: Iterable[Sequence[float]],
        start: Sequence[float],
        goal: Sequence[float],
    ):
        """Place obstacles in the mid-route band at a lateral offset, channelling
        the pursuers around them while the evader's own route stays free."""
        options = self.options
        cfg = self.config
        protected = tuple(np.asarray(point) for point in protected_points)
        start_v = np.asarray(start, dtype=float)
        goal_v = np.asarray(goal, dtype=float)
        route = goal_v - start_v
        route_length = float(np.linalg.norm(route))
        if route_length <= 1e-6:
            return None
        forward = route / route_length
        lateral = np.asarray((-forward[1], forward[0]))
        fraction_low, fraction_high = options.corridor_fraction_range
        count_low, count_high = options.corridor_obstacle_count_range
        count = int(self.rng.integers(count_low, count_high + 1))
        radius_low, radius_high = options.corridor_obstacle_radius_range
        lateral_low, lateral_high = options.corridor_lateral_band
        obstacles = []
        attempts = 0
        while len(obstacles) < count and attempts < 96:
            attempts += 1
            radius = float(self.rng.uniform(radius_low, radius_high))
            fraction = float(
                self.rng.uniform(fraction_low, fraction_high)
            )
            side = float(self.rng.choice((-1.0, 1.0)))
            lateral_offset = side * self.rng.uniform(
                lateral_low, lateral_high
            )
            centre = start_v + forward * fraction * route_length
            centre = centre + lateral * lateral_offset
            limit = (
                cfg.arena_half_extent - radius - cfg.obstacle_safety_margin
            )
            if np.any(np.abs(centre) > limit):
                continue
            if any(
                np.linalg.norm(centre - point)
                < radius + cfg.obstacle_safety_margin
                for point in protected
            ):
                continue
            if any(
                np.linalg.norm(centre - np.asarray((x, y)))
                < radius + other_radius + cfg.obstacle_safety_margin
                for x, y, other_radius in obstacles
            ):
                continue
            obstacles.append((float(centre[0]), float(centre[1]), radius))
        if len(obstacles) != count:
            return None
        return tuple(obstacles)
