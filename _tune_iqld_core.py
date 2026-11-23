"""Run one restartable phase of sequential Role-IQLD continuous tuning."""

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple

from .iqld_hyperparameters import (
    IQLDHyperparameters,
    add_iqld_hyperparameter_args,
)
from .tuning_manifest import (
    TrialManifest,
    TrialState,
    load_manifest,
    save_manifest,
    transition_manifest,
)
from .tuning_monitor import failure_reason, read_trial_snapshot
from .tuning_policy import (
    DiagnosticClass,
    classify_failure,
    deployment_gate_passed,
    next_single_change,
)
from .tuning_report import write_leaderboards


_TRAINING_MODULE = "crazyflie_examples.three_v_one_baseline.train_iqld"
_TERMINAL_STATES = frozenset(
    {
        TrialState.APPROVED,
        TrialState.REJECTED,
        TrialState.FAILED,
        TrialState.CANCELLED,
    }
)


@dataclass(frozen=True)
class ControllerResult:
    """Durable state returned after one controller phase."""

    manifest: TrialManifest
    manifest_path: Path
    launched_pid: Optional[int] = None


def _expert_min_reward(value: str):
    """Parse --expert-min-reward; 'none' disables the transition filter."""
    if value in ("none", "None", "disable"):
        return None
    return float(value)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse controller arguments without importing or launching training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="runs/iqld_continuous_tuning")
    parser.add_argument(
        "--expert-dataset",
        default="checkpoints/three_v_one_expert/final_robust_safe_expert.pt",
    )
    parser.add_argument(
        "--expert-max-transitions", type=int, default=1000,
        help="drop expert episodes longer than this; 0 disables",
    )
    parser.add_argument(
        "--expert-min-separation", type=float, default=0.32,
        help="drop expert episodes below this minimum separation; 0 disables",
    )
    parser.add_argument(
        "--expert-min-reward", type=_expert_min_reward, default=0.0,
        help="drop expert transitions below this mean reward; 'none' keeps all",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--train-episodes", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=1750)
    parser.add_argument("--monitor-interval-seconds", type=float, default=30.0)
    parser.add_argument("--heartbeat-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--short-validation-episodes", type=int, default=20)
    parser.add_argument(
        "--confirmation-seeds", type=int, nargs="+", default=[11, 21, 31]
    )
    parser.add_argument(
        "--formal-seeds",
        type=int,
        nargs="+",
        default=[101, 211, 307, 401, 503],
    )
    parser.add_argument("--formal-episodes-per-seed", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--no-curriculum",
        action="store_true",
        help="train IQLD directly on the final task guided by the expert "
        "dataset (Game-Theory-guided), skipping curriculum stages",
    )
    parser.add_argument(
        "--epn-augment-expert",
        action="store_true",
        help="train the EPN on expert trajectories and rebuild expert "
        "observations with real predictions (train/inference consistency)",
    )
    parser.add_argument("--epn-pretrain-epochs", type=int, default=5)
    parser.add_argument(
        "--role-artifact", default=None,
        help="frozen role-discovery artifact: sets k_roles to K_star",
    )
    add_iqld_hyperparameter_args(parser)
    args = parser.parse_args(argv)
    if args.train_episodes <= 0 or args.max_steps <= 0:
        parser.error("training episodes and max steps must be positive")
    if args.monitor_interval_seconds < 0 or args.heartbeat_timeout_seconds <= 0:
        parser.error("monitor interval must be non-negative and heartbeat positive")
    if args.short_validation_episodes <= 0 or args.formal_episodes_per_seed <= 0:
        parser.error("validation episode budgets must be positive")
    if not args.confirmation_seeds or not args.formal_seeds:
        parser.error("confirmation and formal seeds must not be empty")
    return args


def deterministic_run_id(
    generation: int,
    seed: int,
    parameters: dict,
    parent_run_id: Optional[str] = None,
) -> str:
    """Return a stable human-readable ID derived from resolved trial inputs."""
    if generation < 0:
        raise ValueError("generation must be non-negative")
    payload = {
        "generation": int(generation),
        "parent_run_id": parent_run_id,
        "parameters": parameters,
        "seed": int(seed),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]
    label = "baseline" if generation == 0 else "child"
    return "g%03d-%s-seed%d-%s" % (generation, label, seed, digest)


def _generation(run_id: str) -> int:
    match = re.match(r"^g(\d+)-", run_id)
    return int(match.group(1)) if match else 0


def plan_child_trial(
    parent: TrialManifest,
    diagnostic: DiagnosticClass,
    history: Iterable[Any] = (),
) -> TrialManifest:
    """Plan one child with exactly one policy-approved parameter change."""
    diagnostic = DiagnosticClass(diagnostic)
    resolved = IQLDHyperparameters(**parent.parameters)
    change = next_single_change(diagnostic, resolved, history)
    generation = _generation(parent.run_id) + 1
    parameters = change.parameters.to_dict()
    run_id = deterministic_run_id(
        generation, parent.seed, parameters, parent_run_id=parent.run_id
    )
    paths = {}
    if parent.paths.get("run_dir"):
        root = Path(parent.paths["run_dir"]).parent.parent
        paths = _trial_paths(root, run_id)
    return TrialManifest.create(
        run_id=run_id,
        hypothesis=(
            "change %s from %s to %s after %s"
            % (change.field, change.before, change.after, diagnostic.value)
        ),
        seed=parent.seed,
        parameters=parameters,
        diagnostic_class=diagnostic.value,
        parent_run_id=parent.run_id,
        budgets=dict(parent.budgets),
        git_metadata=dict(parent.git_metadata),
        environment_metadata=dict(parent.environment_metadata),
        dataset_metadata=dict(parent.dataset_metadata),
        paths=paths,
        metrics={
            "changed_parameter": change.field,
            "parameter_before": change.before,
            "parameter_after": change.after,
        },
    )


def _dataset_metadata(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def _trial_paths(root: Path, run_id: str) -> dict:
    run_dir = root / "trials" / run_id
    output_dir = run_dir / "checkpoints"
    return {
        "run_dir": str(run_dir),
        "manifest": str(run_dir / "manifest.json"),
        "stdout": str(run_dir / "stdout.log"),
        "event_dir": str(root / "tensorboard" / run_id),
        "tensorboard_root": str(root / "tensorboard"),
        "output_dir": str(output_dir),
        "checkpoint": str(output_dir / "no_graph_iqld_maddpg_latest.pt"),
    }


def _new_manifest(
    args: argparse.Namespace, root: Path
) -> Tuple[Path, TrialManifest]:
    parameters = IQLDHyperparameters.from_args(args).to_dict()
    run_id = deterministic_run_id(0, args.seed, parameters)
    paths = _trial_paths(root, run_id)
    manifest_path = Path(paths["manifest"])
    if manifest_path.exists():
        raise FileExistsError("trial already exists; pass --resume: %s" % run_id)
    dataset = Path(args.expert_dataset).expanduser().resolve()
    if not dataset.is_file():
        raise FileNotFoundError("expert dataset is missing: %s" % dataset)
    manifest = TrialManifest.create(
        run_id=run_id,
        hypothesis="measure resolved Role-IQLD baseline",
        seed=args.seed,
        parameters=parameters,
        budgets={
            "train_episodes": args.train_episodes,
            "max_steps": args.max_steps,
            "short_validation_episodes": args.short_validation_episodes,
            "confirmation_seeds": list(args.confirmation_seeds),
            "formal_seeds": list(args.formal_seeds),
            "formal_episodes_per_seed": args.formal_episodes_per_seed,
        },
        environment_metadata={
            "device": args.device,
            "python_executable": sys.executable,
        },
        dataset_metadata=_dataset_metadata(dataset),
        paths=paths,
    )
    save_manifest(manifest_path, manifest)
    return manifest_path, manifest


def _all_manifests(root: Path) -> Tuple[Tuple[Path, TrialManifest], ...]:
    found = [
        (path, load_manifest(path))
        for path in root.glob("trials/*/manifest.json")
    ]
    found.sort(key=lambda item: (item[1].created_at, item[1].run_id))
    return tuple(found)


def _select_manifest(
    args: argparse.Namespace, root: Path
) -> Tuple[Path, TrialManifest]:
    existing = _all_manifests(root)
    if args.resume and existing:
        path, manifest = existing[-1]
        _warn_resume_mismatch(args, manifest)
        return path, manifest
    return _new_manifest(args, root)


def _warn_resume_mismatch(
    args: argparse.Namespace, manifest: TrialManifest
) -> None:
    """Warn when --resume silently reuses a trial built from different args."""
    mismatches = []
    if _generation(manifest.run_id) == 0:
        expected = deterministic_run_id(
            0, args.seed, IQLDHyperparameters.from_args(args).to_dict()
        )
        if manifest.run_id != expected:
            mismatches.append(
                "run identity %s does not match resolved arguments %s"
                % (manifest.run_id, expected)
            )
    if manifest.seed != args.seed:
        mismatches.append(
            "seed %d does not match --seed %d" % (manifest.seed, args.seed)
        )
    if manifest.budgets.get("train_episodes") != args.train_episodes:
        mismatches.append(
            "train episodes %s do not match --train-episodes %d"
            % (manifest.budgets.get("train_episodes"), args.train_episodes)
        )
    if mismatches:
        print(
            "warning: --resume reuses %s despite changed arguments: %s"
            % (manifest.run_id, "; ".join(mismatches)),
            file=sys.stderr,
        )


def _persist_details(
    path: Path, manifest: TrialManifest, **changes
) -> TrialManifest:
    updated = replace(manifest, **changes)
    save_manifest(path, updated)
    return updated


def _snapshot_metrics(snapshot) -> dict:
    """Latest telemetry normalized to manifest metric keys."""
    metrics = dict(snapshot.latest)
    if "curriculum/stage" in metrics:
        metrics["curriculum_stage"] = metrics.pop("curriculum/stage")
    return metrics


def _child_argv(args: argparse.Namespace, manifest: TrialManifest) -> list:
    parameters = IQLDHyperparameters(**manifest.parameters)
    dataset = Path(manifest.dataset_metadata["path"]).resolve()
    argv = [
        sys.executable,
        "-m",
        _TRAINING_MODULE,
        "--episodes",
        str(manifest.budgets["train_episodes"]),
        "--max-steps",
        str(manifest.budgets["max_steps"]),
        "--seed",
        str(manifest.seed),
        "--device",
        str(args.device),
        "--expert-dataset",
        str(dataset),
        "--output",
        str(Path(manifest.paths["output_dir"]).resolve()),
        "--tensorboard-dir",
        str(Path(manifest.paths["tensorboard_root"]).resolve()),
        "--run-name",
        manifest.run_id,
        "--validation-episodes",
        str(manifest.budgets["short_validation_episodes"]),
    ]
    if not getattr(args, "no_curriculum", False):
        argv.append("--curriculum")
    if getattr(args, "epn_augment_expert", False):
        argv.append("--epn-augment-expert")
        argv.extend(("--epn-pretrain-epochs", str(args.epn_pretrain_epochs)))
    if getattr(args, "role_artifact", None):
        argv.extend(("--role-artifact", str(args.role_artifact)))
    if getattr(args, "expert_max_transitions", None) is not None:
        argv.extend(("--expert-max-transitions", str(args.expert_max_transitions)))
    if getattr(args, "expert_min_separation", None) is not None:
        argv.extend(("--expert-min-separation", str(args.expert_min_separation)))
    argv.extend(
        (
            "--expert-min-reward",
            "none" if getattr(args, "expert_min_reward", None) is None
            else str(args.expert_min_reward),
        )
    )
    for field, value in parameters.to_dict().items():
        argv.extend(("--" + field.replace("_", "-"), str(value)))
    return argv


def _terminal(
    root: Path,
    path: Path,
    current: TrialManifest,
    target: TrialState,
    reason: str,
    *,
    diagnostic: Optional[DiagnosticClass] = None,
    metrics: Optional[dict] = None,
) -> TrialManifest:
    transitioned = transition_manifest(path, current, target, reason=reason)
    changes = {}
    if diagnostic is not None:
        changes["diagnostic_class"] = diagnostic.value
    if metrics is not None:
        combined = dict(transitioned.metrics)
        combined.update(metrics)
        changes["metrics"] = combined
    if changes:
        transitioned = _persist_details(path, transitioned, **changes)
    write_leaderboards(root, [item[1] for item in _all_manifests(root)])
    return transitioned


def _run_training(
    args: argparse.Namespace,
    root: Path,
    manifest_path: Path,
    manifest: TrialManifest,
    process_factory: Callable,
    snapshot_reader: Callable,
    process_group_signaler: Callable[[int, int], None],
    sleep: Callable[[float], None],
    clock: Callable[[], float],
) -> ControllerResult:
    manifest = transition_manifest(manifest_path, manifest, TrialState.RUNNING)
    run_dir = Path(manifest.paths["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = Path(manifest.paths["stdout"])
    with stdout_path.open("ab", buffering=0) as stdout:
        process = process_factory(
            _child_argv(args, manifest),
            stdout=stdout,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            shell=False,
        )
        launched_pid = int(process.pid)
        paths = dict(manifest.paths)
        paths["child_pid"] = launched_pid
        manifest = _persist_details(manifest_path, manifest, paths=paths)
        started_at = clock()
        while True:
            observed_at = clock()
            snapshot = snapshot_reader(
                Path(manifest.paths["event_dir"]),
                observed_at,
                args.heartbeat_timeout_seconds,
            )
            returncode = process.poll()
            reason = failure_reason(
                snapshot, process_alive=True if returncode is None else None
            )
            awaiting_first_event = (
                returncode is None
                and snapshot.last_event_time is None
                and observed_at - started_at < args.heartbeat_timeout_seconds
            )
            if awaiting_first_event:
                reason = None
            if reason is not None:
                diagnostic = classify_failure(
                    snapshot, validation=None, process=returncode is None
                )
                if returncode is None:
                    process_group_signaler(launched_pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process_group_signaler(launched_pid, signal.SIGKILL)
                        process.wait()
                failed = _terminal(
                    root,
                    manifest_path,
                    manifest,
                    TrialState.FAILED,
                    reason,
                    diagnostic=diagnostic,
                    metrics=_snapshot_metrics(snapshot),
                )
                return ControllerResult(failed, manifest_path, launched_pid)
            if returncode is not None:
                if returncode != 0:
                    failed = _terminal(
                        root,
                        manifest_path,
                        manifest,
                        TrialState.FAILED,
                        "unexpected child exit code %d" % returncode,
                        diagnostic=DiagnosticClass.PROCESS_FAILURE,
                        metrics=_snapshot_metrics(snapshot),
                    )
                    return ControllerResult(failed, manifest_path, launched_pid)
                completed = transition_manifest(
                    manifest_path, manifest, TrialState.SHORT_VALIDATION
                )
                completed = _persist_details(
                    manifest_path,
                    completed,
                    metrics={**completed.metrics, **_snapshot_metrics(snapshot)},
                )
                return ControllerResult(completed, manifest_path, launched_pid)
            sleep(args.monitor_interval_seconds)
