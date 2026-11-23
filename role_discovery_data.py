"""Role-free, variable-team expert behavior windows for IQLD."""

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

from .expert_dataset import ExpertEpisode


@dataclass(frozen=True)
class BehaviorWindow:
    """One focal agent's role-free behavior over a fixed time window."""

    episode_id: str
    focal_slot: int
    start: int
    focal: np.ndarray
    teammates: np.ndarray
    teammate_mask: np.ndarray
    actions: np.ndarray
    effects: np.ndarray


@dataclass(frozen=True)
class RoleEpisodeSplit:
    """Episode-disjoint partitions used by role discovery."""

    train: Tuple[ExpertEpisode, ...]
    selection: Tuple[ExpertEpisode, ...]
    test: Tuple[ExpertEpisode, ...]


def split_role_episodes(
    episodes: Iterable[ExpertEpisode],
    selection_fraction: float,
    test_fraction: float,
    seed: int,
) -> RoleEpisodeSplit:
    """Split complete episodes deterministically before window construction."""
    episodes = tuple(episodes)
    if not 0.0 <= selection_fraction < 1.0 or not 0.0 <= test_fraction < 1.0:
        raise ValueError("split fractions must lie in [0, 1)")
    if selection_fraction + test_fraction >= 1.0:
        raise ValueError("split fractions must leave training episodes")

    order = np.random.default_rng(seed).permutation(len(episodes))
    selection_count = int(round(len(episodes) * selection_fraction))
    test_count = int(round(len(episodes) * test_fraction))
    selection_end = selection_count
    test_end = selection_end + test_count
    selection = tuple(episodes[index] for index in order[:selection_end])
    test = tuple(episodes[index] for index in order[selection_end:test_end])
    train = tuple(episodes[index] for index in order[test_end:])
    return RoleEpisodeSplit(train=train, selection=selection, test=test)


def _transition_arrays(transition, max_pursuers: int):
    if not isinstance(transition, (tuple, list)) or len(transition) != 9:
        raise ValueError("expert transition must contain nine columns")
    positions = np.asarray(transition[0], dtype=np.float32)
    velocities = np.asarray(transition[1], dtype=np.float32)
    actions = np.asarray(transition[3], dtype=np.float32)
    next_positions = np.asarray(transition[5], dtype=np.float32)
    next_velocities = np.asarray(transition[6], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 2:
        raise ValueError("expert actions must have shape (pursuers, 2)")
    count = actions.shape[0]
    if count < 3 or count > max_pursuers:
        raise ValueError("expert pursuer count must lie in [3, max_pursuers]")
    for name, value in (
        ("positions", positions),
        ("velocities", velocities),
        ("next_positions", next_positions),
        ("next_velocities", next_velocities),
    ):
        invalid = (
            value.ndim != 2
            or value.shape[0] != count + 1
            or value.shape[1] < 2
        )
        if invalid:
            raise ValueError(
                f"{name} entity count must equal pursuers plus one"
            )
    return positions, velocities, actions, next_positions, next_velocities


def _read_only(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value).copy()
    result.setflags(write=False)
    return result


def build_behavior_windows(
    episodes: Iterable[ExpertEpisode],
    window_steps: int,
    max_pursuers: int = 5,
) -> Tuple[BehaviorWindow, ...]:
    """Build identity-free focal-agent windows without crossing episodes."""
    if window_steps <= 0:
        raise ValueError("window_steps must be positive")
    if max_pursuers < 3:
        raise ValueError("max_pursuers must be at least three")

    windows = []
    for episode in episodes:
        validated = tuple(
            _transition_arrays(transition, max_pursuers)
            for transition in episode.transitions
        )
        if not validated:
            continue
        counts = {item[2].shape[0] for item in validated}
        if len(counts) != 1:
            raise ValueError(
                "pursuer count must remain constant within an episode"
            )
        count = counts.pop()
        for start in range(len(validated) - window_steps + 1):
            rows = validated[start:start + window_steps]
            for focal_slot in range(count):
                focal_values = np.zeros((window_steps, 4), dtype=np.float32)
                teammate_values = np.zeros(
                    (window_steps, max_pursuers - 1, 4), dtype=np.float32
                )
                teammate_mask = np.zeros(
                    (window_steps, max_pursuers - 1), dtype=bool
                )
                action_values = np.zeros((window_steps, 2), dtype=np.float32)
                effect_values = np.zeros((window_steps, 6), dtype=np.float32)
                teammate_indices = [
                    index for index in range(count) if index != focal_slot
                ]
                for offset, row in enumerate(rows):
                    (
                        positions,
                        velocities,
                        actions,
                        next_positions,
                        next_velocities,
                    ) = row
                    evader = count
                    focal_relation = np.concatenate(
                        (
                            positions[focal_slot, :2] - positions[evader, :2],
                            velocities[focal_slot, :2]
                            - velocities[evader, :2],
                        )
                    )
                    teammate_relation = np.concatenate(
                        (
                            positions[teammate_indices, :2]
                            - positions[focal_slot, :2],
                            velocities[teammate_indices, :2]
                            - velocities[focal_slot, :2],
                        ),
                        axis=1,
                    )
                    next_relation = np.concatenate(
                        (
                            next_positions[focal_slot, :2]
                            - next_positions[evader, :2],
                            next_velocities[focal_slot, :2]
                            - next_velocities[evader, :2],
                        )
                    )
                    effect = np.concatenate(
                        (
                            next_relation[:2] - focal_relation[:2],
                            next_relation[2:] - focal_relation[2:],
                            next_velocities[evader, :2]
                            - velocities[evader, :2],
                        )
                    )
                    focal_values[offset] = focal_relation
                    teammate_values[offset, : count - 1] = teammate_relation
                    teammate_mask[offset, : count - 1] = True
                    action_values[offset] = actions[focal_slot]
                    effect_values[offset] = effect
                windows.append(
                    BehaviorWindow(
                        episode_id=episode.identifier,
                        focal_slot=focal_slot,
                        start=start,
                        focal=_read_only(focal_values),
                        teammates=_read_only(teammate_values),
                        teammate_mask=_read_only(teammate_mask),
                        actions=_read_only(action_values),
                        effects=_read_only(effect_values),
                    )
                )
    return tuple(windows)


# ---------------------------------------------------------------------------
# Leakage guards for v2 role-free discovery artifacts (Task 3).
# ---------------------------------------------------------------------------
ROLE_FREE_V2_VERSION = "2"

# Keys that MUST NOT appear in a confirmatory discovery artifact: they leak
# teacher roles, hidden intent, stage, or privileged scenario state.
FORBIDDEN_DISCOVERY_KEYS = frozenset(
    {
        "roles",
        "role_assignment",
        "role_targets",
        "c",
        "escape",
        "feint_steps",
        "phase",
        "phases",
    }
)


def reject_forbidden_discovery_keys(payload) -> None:
    """Raise if a serializable transition/episode payload leaks privileged keys."""
    if isinstance(payload, dict):
        keys = set(payload.keys())
    elif isinstance(payload, np.ndarray) and payload.dtype.names is not None:
        keys = set(payload.dtype.names)
    else:
        keys = set()
    leaked = sorted(keys & FORBIDDEN_DISCOVERY_KEYS)
    if leaked:
        raise ValueError(
            "role-free discovery payload leaks forbidden keys: "
            + ", ".join(leaked)
        )


def reject_legacy_discovery_artifact(path) -> None:
    """Reject legacy split_s3-style npz files as confirmatory discovery data."""
    loaded = np.load(path, allow_pickle=True)
    records = loaded["records"]
    if len(records) == 0:
        raise ValueError(f"discovery artifact {path} is empty")
    sample = records[0]
    keys = (
        set(sample.keys())
        if isinstance(sample, dict)
        else (set(sample.dtype.names) if sample.dtype.names is not None else set())
    )
    leaked = sorted(keys & FORBIDDEN_DISCOVERY_KEYS)
    if leaked:
        raise ValueError(
            f"rejecting legacy/leaking discovery artifact {path}; "
            f"found forbidden keys {leaked}"
        )
