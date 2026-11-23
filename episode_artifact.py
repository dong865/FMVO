"""Versioned, pickle-free successful-episode artifacts."""

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Dict, Mapping

import numpy as np


REQUIRED_ARRAYS = {
    "timestamps", "positions", "velocities", "actions",
    "applied_accelerations", "commanded_velocities",
}


@dataclass
class EpisodeArtifact:
    arrays: Dict[str, np.ndarray]
    metadata: Dict[str, object]


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def validate_episode(artifact: EpisodeArtifact) -> None:
    missing = REQUIRED_ARRAYS.difference(artifact.arrays)
    if missing:
        raise ValueError(f"missing required arrays: {sorted(missing)}")
    arrays = {k: np.asarray(v) for k, v in artifact.arrays.items()}
    timestamps = arrays["timestamps"]
    if timestamps.ndim != 1 or len(timestamps) < 2:
        raise ValueError("timestamps must contain at least two frames")
    if not np.all(np.diff(timestamps) > 0):
        raise ValueError("timestamps must be strictly increasing")
    frames = len(timestamps)
    for name, value in arrays.items():
        if value.dtype == object:
            raise ValueError(f"{name} must not use object dtype")
        if value.ndim == 0 or value.shape[0] != frames:
            raise ValueError(f"{name} must have leading dimension {frames}")
        if np.issubdtype(value.dtype, np.number) and not np.all(np.isfinite(value)):
            raise ValueError(f"{name} contains non-finite values")
    for name in ("positions", "velocities", "commanded_velocities"):
        if arrays[name].ndim != 3 or arrays[name].shape[1] != 4:
            raise ValueError(f"{name} must have shape [T,4,D]")
    if arrays["actions"].shape != (frames, 3, 2):
        raise ValueError("actions must have shape [T,3,2]")
    if arrays["applied_accelerations"].shape[:2] != (frames, 4):
        raise ValueError("applied_accelerations must have shape [T,4,D]")
    if int(artifact.metadata.get("format_version", -1)) != 1:
        raise ValueError("unsupported format_version")


def save_episode(path, artifact: EpisodeArtifact) -> Path:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    validate_episode(artifact)
    arrays = {name: np.asarray(value) for name, value in artifact.arrays.items()}
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
    digest = sha256(path.read_bytes()).hexdigest()
    manifest = _json_safe(dict(artifact.metadata))
    manifest["npz_sha256"] = digest
    manifest_path = path.with_suffix(".json")
    manifest_tmp = manifest_path.with_name(manifest_path.name + ".tmp")
    manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest_tmp.replace(manifest_path)
    return path


def load_episode(path) -> EpisodeArtifact:
    path = Path(path).expanduser().resolve()
    metadata = json.loads(path.with_suffix(".json").read_text())
    expected = metadata.pop("npz_sha256", None)
    actual = sha256(path.read_bytes()).hexdigest()
    if expected != actual:
        raise ValueError("episode NPZ checksum mismatch")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    artifact = EpisodeArtifact(arrays=arrays, metadata=metadata)
    validate_episode(artifact)
    return artifact


def episode_metrics(artifact: EpisodeArtifact) -> Dict[str, np.ndarray]:
    validate_episode(artifact)
    pos = np.asarray(artifact.arrays["positions"], dtype=np.float64)[..., :2]
    vel = np.asarray(artifact.arrays["velocities"], dtype=np.float64)
    cmd = np.asarray(artifact.arrays["commanded_velocities"], dtype=np.float64)
    relative = pos[:, :3] - pos[:, 3:4]
    angles = np.sort(np.arctan2(relative[..., 1], relative[..., 0]), axis=1)
    wrapped = np.concatenate((angles, angles[:, :1] + 2.0 * np.pi), axis=1)
    delta = pos[:, :, None, :] - pos[:, None, :, :]
    pair = np.linalg.norm(delta, axis=-1)
    diagonal = np.arange(4)
    pair[:, diagonal, diagonal] = np.inf
    result = {
        "pursuer_evader_distance": np.linalg.norm(relative, axis=-1),
        "minimum_pair_distance": np.min(pair, axis=(1, 2)),
        "maximum_angular_gap": np.max(np.diff(wrapped, axis=1), axis=1),
        "speed": np.linalg.norm(vel, axis=-1),
        "command_speed": np.linalg.norm(cmd, axis=-1),
    }
    obstacles = artifact.metadata.get("virtual_obstacles", [])
    if obstacles:
        clearances = []
        margin = float(artifact.metadata.get("obstacle_safety_margin", 0.0))
        for ox, oy, radius in obstacles:
            clearances.append(
                np.linalg.norm(pos - np.asarray([ox, oy]), axis=-1)
                - float(radius) - margin
            )
        result["minimum_obstacle_clearance"] = np.min(clearances, axis=(0, 2))
    else:
        result["minimum_obstacle_clearance"] = np.full(len(pos), np.inf)
    return result
