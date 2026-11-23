"""Deterministic atomic leaderboards for Role-IQLD tuning trials."""

import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Tuple

from .tuning_manifest import TrialManifest
from .tuning_policy import candidate_rank


_MARKDOWN_COLUMNS = (
    "run ID",
    "state",
    "hypothesis",
    "changed parameter",
    "seed",
    "curriculum stage",
    "robust capture",
    "collision-free rate",
    "minimum separation",
    "mean capture time",
    "checkpoint",
    "rejection reason",
)


def _ranking_metrics(manifest: TrialManifest) -> SimpleNamespace:
    """Expose persisted metrics through the attribute API used by ranking."""
    return SimpleNamespace(**manifest.metrics)


def _ordered(manifests: Iterable[TrialManifest]) -> Tuple[TrialManifest, ...]:
    """Sort safety-first while keeping run-ID ordering deterministic on ties."""
    ordered = sorted(manifests, key=lambda item: item.run_id)
    ordered.sort(
        key=lambda item: candidate_rank(_ranking_metrics(item)), reverse=True
    )
    return tuple(ordered)


def _atomic_write(path: Path, data: bytes) -> None:
    """Durably replace one leaderboard without exposing partial content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".%s." % path.name,
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        directory_descriptor = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _markdown_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        value = format(value, ".6g")
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\n", " ")
    )


def _markdown(manifests: Iterable[TrialManifest]) -> bytes:
    header = "| " + " | ".join(_MARKDOWN_COLUMNS) + " |"
    separator = "| " + " | ".join("---" for _ in _MARKDOWN_COLUMNS) + " |"
    rows = [header, separator]
    for manifest in manifests:
        metrics = manifest.metrics
        values = (
            manifest.run_id,
            manifest.state.value,
            manifest.hypothesis,
            metrics.get("changed_parameter"),
            manifest.seed,
            metrics.get("curriculum_stage"),
            metrics.get("robust_capture_rate"),
            metrics.get("collision_free_rate"),
            metrics.get("minimum_separation"),
            metrics.get("mean_capture_time"),
            manifest.paths.get("checkpoint"),
            manifest.reason,
        )
        rows.append(
            "| " + " | ".join(_markdown_value(item) for item in values) + " |"
        )
    return ("\n".join(rows) + "\n").encode("utf-8")


def write_leaderboards(root: Path, manifests: Iterable[TrialManifest]) -> None:
    """Atomically regenerate JSON and Markdown from all known trials."""
    root = Path(root).expanduser().resolve()
    ordered = _ordered(manifests)
    payload = {"trials": [manifest.to_dict() for manifest in ordered]}
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    _atomic_write(root / "leaderboard.json", encoded)
    _atomic_write(root / "leaderboard.md", _markdown(ordered))
