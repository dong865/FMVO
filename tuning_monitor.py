"""Fail-closed health monitoring for IQLD TensorBoard training runs."""

import math
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from .config import TaskConfig
from .tuning_manifest import TrialManifest


# Deployment/validation standard: accepted checkpoints and formal evaluation
# must maintain minimum inter-drone separation of at least 0.30 m. This is the
# threshold enforced by the validation gate (tuning_policy) and must NOT change.
MINIMUM_SEPARATION_M = 0.30

# Trial-level hard-safety floor during TRAINING. The environment itself
# terminates episodes at safety_emergency_distance (0.30 m) as a learning
# signal (penalty + next episode), so the trial monitor observing the latest
# episode's minimum separation must not fail the whole trial at that same
# threshold — exploration would kill every trial at its first transient dip.
# A trial is failed only when the state drops below the physical collision
# radius, i.e. when the environment's own safety projection failed.
TRIAL_HARD_SEPARATION_M = TaskConfig().collision_radius
PRETRAIN_REQUIRED_TAGS = frozenset({"loss/behavior_clone", "assignment/pretrain_geometric_cost", "assignment/pretrain_entropy", "iqld/pretrain_mse_before", "iqld/pretrain_mse_after", "iqld/pretrain_cosine_before", "iqld/pretrain_cosine_after", "iqld/pretrain_gate_passed"})
ONLINE_REQUIRED_TAGS = frozenset({"episode/captured", "episode/obstacle_collisions", "metrics/cumulative_capture_rate", "curriculum/stage", "curriculum/capture_rate", "curriculum/collision_free_rate", "loss/q", "loss/value", "loss/policy", "loss/epn", "assignment/geometric_cost", "assignment/entropy", "buffer/online_replay_size", "buffer/expert_replay_size", "iqld/expert_fraction_actual", "policy/radial_action_alignment", "progress/environment_steps", "safety/minimum_separation", "safety/obstacle_collision_steps"})


@dataclass(frozen=True)
class MonitorSnapshot:
    """Latest scalar, process, and GPU evidence for one training run."""

    latest: Dict[str, float]
    last_event_time: Optional[float]
    stale: bool
    missing_tags: Tuple[str, ...]
    tag_steps: Dict[str, int] = field(default_factory=dict)
    run_id: Optional[str] = None
    pid: Optional[int] = None
    process_alive: Optional[bool] = None
    gpu_evidence: Tuple[str, ...] = ()
    read_error: Optional[str] = None


def _event_files(event_dir: Path) -> Tuple[Path, ...]:
    return tuple(path for path in event_dir.rglob("events.out.tfevents.*") if path.is_file())


def _required_tags(latest: Dict[str, float], tag_steps: Dict[str, int]) -> frozenset:
    """Select online requirements only after actual online progress has begun."""
    if any(tag_steps.get(tag, 0) > 0 for tag in ONLINE_REQUIRED_TAGS):
        return ONLINE_REQUIRED_TAGS
    return PRETRAIN_REQUIRED_TAGS


def read_trial_snapshot(event_dir: Path, now: float, heartbeat_seconds: float) -> MonitorSnapshot:
    """Read scalars and event-file heartbeat, failing closed on unavailable data."""
    event_dir = Path(event_dir)
    files = _event_files(event_dir) if event_dir.exists() else ()
    last_event_time = max((path.stat().st_mtime for path in files), default=None)
    latest: Dict[str, float] = {}
    tag_steps: Dict[str, int] = {}
    read_error = None
    if files:
        try:
            accumulator = EventAccumulator(str(event_dir))
            accumulator.Reload()
            for tag in accumulator.Tags().get("scalars", ()):
                events = accumulator.Scalars(tag)
                if events:
                    latest[tag] = float(events[-1].value)
                    tag_steps[tag] = int(events[-1].step)
            if not latest:
                read_error = "event stream has no readable scalars"
        except Exception as error:
            read_error = str(error) or error.__class__.__name__
    required = _required_tags(latest, tag_steps)
    return MonitorSnapshot(latest=latest, last_event_time=last_event_time, stale=last_event_time is None or now - last_event_time > heartbeat_seconds, missing_tags=tuple(sorted(required.difference(latest))), tag_steps=tag_steps, read_error=read_error)


def _read_proc_stat(pid: int) -> str:
    return Path("/proc/%d/stat" % pid).read_text(encoding="utf-8")


def process_is_alive(pid: int, stat_reader: Callable[[int], str] = _read_proc_stat) -> bool:
    """Return Linux process liveness, treating inaccessible and zombie states as dead."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        tail = stat_reader(pid).rsplit(")", 1)[1].lstrip().split()
        state = tail[0]
    except (IndexError, OSError, PermissionError):
        return False
    return state not in {"Z", "X"}


def monitor_trial(manifest: TrialManifest, event_dir: Path, pid: int, now: float, heartbeat_seconds: float, gpu_reader: Optional[Callable[[int], Iterable[str]]] = None) -> MonitorSnapshot:
    """Bind event telemetry to a manifest, process identity, and real GPU evidence."""
    snapshot = read_trial_snapshot(event_dir, now, heartbeat_seconds)
    alive = process_is_alive(pid)
    gpu_evidence: Tuple[str, ...] = ()
    read_error = snapshot.read_error
    if gpu_reader is not None:
        try:
            gpu_evidence = tuple(str(item) for item in gpu_reader(pid))
        except Exception as error:
            message = "gpu reader error: " + (str(error) or error.__class__.__name__)
            read_error = message if read_error is None else read_error + "; " + message
    return replace(snapshot, run_id=manifest.run_id, pid=pid, process_alive=alive, gpu_evidence=gpu_evidence, read_error=read_error)


def failure_reason(snapshot: MonitorSnapshot, process_alive: Optional[bool] = None) -> Optional[str]:
    """Return every safety-critical failure, including contradictory liveness evidence."""
    failures = []
    evidence = snapshot.process_alive
    if evidence is False or process_alive is False:
        failures.append("training process is not alive")
    if evidence is not None and process_alive is not None and evidence != process_alive:
        failures.append("contradictory process liveness evidence")
    if snapshot.stale:
        failures.append("event stream is stale")
    if snapshot.read_error is not None:
        failures.append("event read error: " + snapshot.read_error)
    if snapshot.missing_tags:
        failures.append("missing required tags: " + ", ".join(snapshot.missing_tags))
    non_finite = sorted(tag for tag, value in snapshot.latest.items() if not math.isfinite(value))
    if non_finite:
        failures.append("non-finite metrics: " + ", ".join(non_finite))
    if snapshot.latest.get("safety/minimum_separation", TRIAL_HARD_SEPARATION_M) < TRIAL_HARD_SEPARATION_M:
        failures.append("minimum separation below %.2f m" % TRIAL_HARD_SEPARATION_M)
    # Obstacle collisions during TRAINING are episode-level learning signals
    # (the environment terminates + penalizes; the policy learns to avoid).
    # They must not fail the whole trial; the obstacle-free requirement is
    # enforced by the deployment gate during validation.
    return "; ".join(failures) if failures else None
