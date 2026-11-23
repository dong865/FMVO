"""Shared recording and source-fallback control for successful episodes."""

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List

import numpy as np

from .episode_artifact import EpisodeArtifact, validate_episode


@dataclass(frozen=True)
class SearchSource:
    name: str
    attempts: int


def search_sources(
    sources: Iterable[SearchSource],
    seed_start: int,
    rollout: Callable[[SearchSource, int], Dict[str, object]],
):
    """Try every primary seed before advancing to the next source."""
    for source in sources:
        for offset in range(source.attempts):
            seed = seed_start + offset
            result = rollout(source, seed)
            print(f"scene attempt source={source.name} seed={seed} "
                  f"accepted={bool(result.get('accepted'))}", flush=True)
            if result.get("accepted"):
                result = dict(result)
                result.update(source=source.name, seed=seed)
                return result
    raise RuntimeError("no successful safe episode found within attempt budgets")


def build_artifact(frames: List[Dict[str, object]], metadata):
    if len(frames) < 2:
        raise ValueError("an episode needs at least two recorded frames")
    names = ("timestamps", "positions", "velocities", "actions",
             "commanded_velocities")
    source_names = ("timestamp", "positions", "velocities", "actions",
                    "commanded_velocities")
    arrays = {
        name: np.asarray([frame[source] for frame in frames])
        for name, source in zip(names, source_names)
    }
    dt = np.diff(arrays["timestamps"])
    acceleration = np.zeros_like(arrays["commanded_velocities"], dtype=np.float32)
    acceleration[1:] = np.diff(arrays["commanded_velocities"], axis=0) / dt[:, None, None]
    arrays["applied_accelerations"] = acceleration
    optional = set(frames[0]).difference(source_names)
    for name in optional:
        if all(name in frame for frame in frames):
            value = np.asarray([frame[name] for frame in frames])
            if value.dtype != object:
                arrays[name] = value
    artifact = EpisodeArtifact(arrays=arrays, metadata=dict(metadata))
    validate_episode(artifact)
    return artifact

