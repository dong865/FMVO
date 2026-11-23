"""Durable manifests for restartable IQLD tuning trials."""

import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union

import fcntl


class TrialState(str, Enum):
    """Lifecycle states for a tuning trial."""

    PLANNED = "planned"
    RUNNING = "running"
    SHORT_VALIDATION = "short_validation"
    MULTISEED_CONFIRMATION = "multiseed_confirmation"
    FORMAL_EVALUATION = "formal_evaluation"
    APPROVED = "approved"
    REJECTED = "rejected"
    FAILED = "failed"
    CANCELLED = "cancelled"


ALLOWED_TRANSITIONS = {
    TrialState.PLANNED: {TrialState.RUNNING, TrialState.CANCELLED},
    TrialState.RUNNING: {TrialState.SHORT_VALIDATION, TrialState.FAILED},
    TrialState.SHORT_VALIDATION: {
        TrialState.MULTISEED_CONFIRMATION,
        TrialState.REJECTED,
        TrialState.FAILED,
    },
    TrialState.MULTISEED_CONFIRMATION: {
        TrialState.FORMAL_EVALUATION,
        TrialState.REJECTED,
        TrialState.FAILED,
    },
    TrialState.FORMAL_EVALUATION: {
        TrialState.APPROVED,
        TrialState.REJECTED,
        TrialState.FAILED,
    },
}


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class TrialManifest:
    """All durable metadata required to reproduce and audit one trial."""

    run_id: str
    revision: int
    created_at: str
    updated_at: str
    state: TrialState
    hypothesis: str
    diagnostic_class: Optional[str]
    parent_run_id: Optional[str]
    seed: int
    budgets: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    git_metadata: Dict[str, Any] = field(default_factory=dict)
    environment_metadata: Dict[str, Any] = field(default_factory=dict)
    dataset_metadata: Dict[str, Any] = field(default_factory=dict)
    paths: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    reason: Optional[str] = None

    @classmethod
    def create(
        cls,
        run_id: str,
        hypothesis: str,
        seed: int,
        parameters: Dict[str, Any],
        *,
        diagnostic_class: Optional[str] = None,
        parent_run_id: Optional[str] = None,
        budgets: Optional[Dict[str, Any]] = None,
        git_metadata: Optional[Dict[str, Any]] = None,
        environment_metadata: Optional[Dict[str, Any]] = None,
        dataset_metadata: Optional[Dict[str, Any]] = None,
        paths: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        reason: Optional[str] = None,
    ) -> "TrialManifest":
        """Build a new revision-zero trial manifest."""
        timestamp = _timestamp()
        return cls(
            run_id=run_id,
            revision=0,
            created_at=timestamp,
            updated_at=timestamp,
            state=TrialState.PLANNED,
            hypothesis=hypothesis,
            diagnostic_class=diagnostic_class,
            parent_run_id=parent_run_id,
            seed=seed,
            budgets=dict(budgets or {}),
            parameters=dict(parameters),
            git_metadata=dict(git_metadata or {}),
            environment_metadata=dict(environment_metadata or {}),
            dataset_metadata=dict(dataset_metadata or {}),
            paths=dict(paths or {}),
            metrics=dict(metrics or {}),
            reason=reason,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe representation with string enum values."""
        data = asdict(self)
        data["state"] = self.state.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrialManifest":
        """Restore a manifest written by :func:`save_manifest`."""
        values = dict(data)
        values["state"] = TrialState(values["state"])
        return cls(**values)


@contextmanager
def _manifest_lock(path: Path):
    """Serialize writers through a stable sibling lock file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(".%s.lock" % path.name)
    descriptor = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _save_manifest_unlocked(path: Path, manifest: TrialManifest) -> None:
    """Atomically replace *path* with canonical, durable JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        manifest.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8") + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".%s." % path.name, suffix=".tmp", dir=str(path.parent)
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        directory_descriptor = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def save_manifest(path: Path, manifest: TrialManifest) -> None:
    """Atomically persist a manifest while excluding concurrent writers."""
    with _manifest_lock(path):
        _save_manifest_unlocked(path, manifest)


def load_manifest(path: Path) -> TrialManifest:
    """Load a manifest from its atomic JSON representation."""
    with Path(path).open(encoding="utf-8") as handle:
        return TrialManifest.from_dict(json.load(handle))


def transition_manifest(
    path: Path,
    expected: Union[TrialState, TrialManifest],
    target: TrialState,
    reason: Optional[str] = None,
    *,
    expected_revision: Optional[int] = None,
) -> TrialManifest:
    """Compare-and-swap a manifest state and write the next revision.

    Passing a prior :class:`TrialManifest` as *expected* compares both its
    state and revision.  Callers holding only a state can pass
    ``expected_revision`` to obtain the same optimistic-concurrency check.
    """
    with _manifest_lock(path):
        current = load_manifest(path)
        if isinstance(expected, TrialManifest):
            expected_state = expected.state
            expected_revision = expected.revision
        else:
            expected_state = TrialState(expected)
        target = TrialState(target)
        if current.state != expected_state:
            raise ValueError(
                "expected state %s, found %s"
                % (expected_state.name, current.state.name)
            )
        if expected_revision is not None and current.revision != expected_revision:
            raise ValueError(
                "expected revision %s, found %s"
                % (expected_revision, current.revision)
            )
        if target not in ALLOWED_TRANSITIONS.get(current.state, set()):
            raise ValueError(
                "illegal transition %s -> %s" % (current.state.name, target.name)
            )
        next_manifest = replace(
            current,
            revision=current.revision + 1,
            updated_at=_timestamp(),
            state=target,
            reason=reason,
        )
        _save_manifest_unlocked(path, next_manifest)
        return next_manifest
