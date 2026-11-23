"""Persistent, versioned expert demonstrations for imitation baselines."""

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

import numpy as np
import torch


EXPERT_DATASET_VERSION = 1
CURRENT_SCENARIO_CONTRACT_VERSION = 2
MIN_EXPERT_SEPARATION_M = 0.30
TRANSITION_COLUMNS = 9


@dataclass(frozen=True)
class ExpertEpisode:
    """One accepted Game Theory rollout and its episode boundary."""

    identifier: str
    seed: int
    attempt: int
    minimum_separation: float
    transitions: Tuple[tuple, ...]


@dataclass(frozen=True)
class ExpertDataset:
    """Versioned expert data bound to one task and physics definition."""

    version: int
    task_fingerprint: str
    task_config: dict
    physics_config: dict
    source: str
    episodes: Tuple[ExpertEpisode, ...]
    attempted_episodes: int
    created_at: str
    updated_at: str


def _dictionary(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if is_dataclass(value):
        return asdict(value)
    raise TypeError("configuration must be a dict or dataclass")


def task_fingerprint(task_config: Any, physics_config: Any) -> str:
    """Return a stable fingerprint of the complete task/physics contract."""
    payload = {
        "task_config": _dictionary(task_config),
        "physics_config": _dictionary(physics_config),
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _transitions_are_finite(transitions: Iterable[tuple]) -> bool:
    transitions = tuple(transitions)
    if not transitions:
        return False
    for item in transitions:
        if not isinstance(item, (tuple, list)) or len(item) != TRANSITION_COLUMNS:
            return False
        for value in item:
            try:
                array = np.asarray(value)
            except (TypeError, ValueError):
                return False
            if array.dtype == object or not np.all(np.isfinite(array)):
                return False
    return True


def episode_is_qualified(
    captured: bool,
    collision_free: bool,
    obstacle_collision: bool,
    minimum_separation: float,
    transitions: Iterable[tuple],
) -> bool:
    """Return whether an episode is safe and useful for expert imitation."""
    return bool(
        captured
        and collision_free
        and not obstacle_collision
        and math.isfinite(float(minimum_separation))
        and float(minimum_separation) >= MIN_EXPERT_SEPARATION_M
        and _transitions_are_finite(transitions)
    )


def _episode_payload(episode: ExpertEpisode) -> dict:
    return {
        "identifier": episode.identifier,
        "seed": episode.seed,
        "attempt": episode.attempt,
        "minimum_separation": episode.minimum_separation,
        "transitions": episode.transitions,
    }


def _dataset_payload(dataset: ExpertDataset) -> dict:
    payload = asdict(dataset)
    payload["episodes"] = [_episode_payload(item) for item in dataset.episodes]
    return payload


def _load_payload(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _from_payload(payload: dict) -> ExpertDataset:
    if payload.get("version") != EXPERT_DATASET_VERSION:
        raise ValueError("unsupported expert dataset version")
    episodes = tuple(
        ExpertEpisode(
            identifier=item["identifier"],
            seed=int(item["seed"]),
            attempt=int(item["attempt"]),
            minimum_separation=float(item["minimum_separation"]),
            transitions=tuple(tuple(column for column in row) for row in item["transitions"]),
        )
        for item in payload["episodes"]
    )
    return ExpertDataset(
        version=int(payload["version"]),
        task_fingerprint=payload["task_fingerprint"],
        task_config=payload["task_config"],
        physics_config=payload["physics_config"],
        source=payload["source"],
        episodes=episodes,
        attempted_episodes=int(payload["attempted_episodes"]),
        created_at=payload["created_at"],
        updated_at=payload["updated_at"],
    )


def load_expert_dataset(
    path: Any, *, expected_config: Any, expected_physics: Any = None
) -> ExpertDataset:
    """Load and validate an expert dataset against the requested task."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    dataset = _from_payload(_load_payload(path))
    dataset_contract = int(
        dataset.task_config.get("scenario_contract_version", 1)
    )
    expected_contract = int(
        _dictionary(expected_config).get("scenario_contract_version", 1)
    )
    if (
        dataset_contract != CURRENT_SCENARIO_CONTRACT_VERSION
        or expected_contract != CURRENT_SCENARIO_CONTRACT_VERSION
    ):
        raise ValueError(
            "expert dataset uses an obsolete scenario contract; regenerate "
            "expert data and checkpoints for contract v2"
        )
    physics = expected_physics if expected_physics is not None else dataset.physics_config
    expected = task_fingerprint(expected_config, physics)
    if dataset.task_fingerprint != expected:
        raise ValueError(
            "expert dataset task fingerprint does not match; regenerate data "
            "for the current task contract"
        )
    if any(
        not episode_is_qualified(
            True, True, False, item.minimum_separation, item.transitions
        )
        for item in dataset.episodes
    ):
        raise ValueError("expert dataset contains an unqualified episode")
    return dataset


def append_expert_episodes(
    path: Any,
    task_config: Any,
    episodes: Iterable[ExpertEpisode],
    *,
    physics_config: Any,
    attempted_episodes: int = 0,
) -> ExpertDataset:
    """Atomically append qualified, unique episodes to a persistent dataset."""
    path = Path(path).expanduser()
    task_values = _dictionary(task_config)
    physics_values = _dictionary(physics_config)
    fingerprint = task_fingerprint(task_values, physics_values)
    new_episodes = tuple(episodes)
    for item in new_episodes:
        if not episode_is_qualified(
            True, True, False, item.minimum_separation, item.transitions
        ):
            raise ValueError(f"episode {item.identifier!r} is not qualified")

    now = datetime.now(timezone.utc).isoformat()
    if path.exists():
        current = load_expert_dataset(
            path,
            expected_config=task_values,
            expected_physics=physics_values,
        )
        existing = list(current.episodes)
        created_at = current.created_at
        attempts = current.attempted_episodes + int(attempted_episodes)
    else:
        existing = []
        created_at = now
        attempts = int(attempted_episodes)

    identifiers = {item.identifier for item in existing}
    existing.extend(item for item in new_episodes if item.identifier not in identifiers)
    dataset = ExpertDataset(
        version=EXPERT_DATASET_VERSION,
        task_fingerprint=fingerprint,
        task_config=task_values,
        physics_config=physics_values,
        source="game_theory",
        episodes=tuple(existing),
        attempted_episodes=attempts,
        created_at=created_at,
        updated_at=now,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(_dataset_payload(dataset), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return dataset


def filter_episodes(
    dataset: "ExpertDataset",
    max_transitions: Optional[int] = None,
    min_separation: Optional[float] = None,
) -> "ExpertDataset":
    """Keep only useful expert experience: fast captures with safety margin.

    Long episodes (slow captures) contain mostly wandering/low-value behavior;
    episodes whose minimum separation hugs the 0.30 m boundary teach actions
    that risk real collisions. Both are dropped (or kept, if no limit given),
    so imitation learning focuses on effective encirclement transitions.
    """
    episodes = dataset.episodes
    if max_transitions is not None:
        episodes = tuple(
            e for e in episodes if len(e.transitions) <= max_transitions
        )
    if min_separation is not None:
        episodes = tuple(
            e for e in episodes if e.minimum_separation >= min_separation
        )
    return ExpertDataset(
        version=dataset.version,
        task_fingerprint=dataset.task_fingerprint,
        task_config=dataset.task_config,
        physics_config=dataset.physics_config,
        source=dataset.source,
        episodes=episodes,
        attempted_episodes=dataset.attempted_episodes,
        created_at=dataset.created_at,
        updated_at=dataset.updated_at,
    )
