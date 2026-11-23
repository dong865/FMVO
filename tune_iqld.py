"""Run one restartable phase of sequential Role-IQLD continuous tuning."""

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

from ._tune_iqld_core import (
    ControllerResult,
    _TERMINAL_STATES,
    _all_manifests,
    _run_training,
    _select_manifest,
    _terminal,
    deterministic_run_id,
    parse_args,
    plan_child_trial,
)
from .deployment_evaluation import GateReport
from .tuning_manifest import TrialManifest, TrialState, transition_manifest
from .tuning_monitor import read_trial_snapshot
from .tuning_policy import (
    DiagnosticClass,
    deployment_gate_passed,
)
from .tuning_report import write_leaderboards


def _persist_metrics(path: Path, manifest: TrialManifest, metrics: dict):
    """Persist measured metrics without changing an already-CASed state."""
    from ._tune_iqld_core import _persist_details

    return _persist_details(
        path, manifest, metrics={**manifest.metrics, **metrics}
    )


def _default_checkpoint_loader(path: Path, device: str):
    from .runtime import load_runtime

    return load_runtime("no_graph_iqld_maddpg", str(path), device=device)


def _default_short_validation(runtime, manifest: TrialManifest, args):
    from .config import TaskConfig
    from .training_validation import evaluate_final_task

    return evaluate_final_task(
        runtime,
        TaskConfig(),
        episodes=args.short_validation_episodes,
        seeds=(args.seed,),
    )


def _default_confirmation(runtime, manifest: TrialManifest, args):
    from .config import TaskConfig
    from .training_validation import evaluate_final_task

    return evaluate_final_task(
        runtime,
        TaskConfig(),
        episodes=len(args.confirmation_seeds) * args.short_validation_episodes,
        seeds=tuple(args.confirmation_seeds),
    )


def _default_formal(runtime, manifest: TrialManifest, args):
    from .config import TaskConfig
    from .deployment_evaluation import evaluate_runtime

    return evaluate_runtime(
        runtime,
        TaskConfig(),
        seeds=tuple(args.formal_seeds),
        episodes_per_seed=args.formal_episodes_per_seed,
        max_steps=args.max_steps,
    )


def _validation_metrics(result) -> dict:
    return {
        "robust_capture_rate": float(result.capture_rate),
        "collision_free_rate": float(result.collision_free_rate),
        "minimum_separation": float(result.minimum_separation),
    }


def _validation_miss(result) -> Optional[str]:
    missing = []
    if result.capture_rate < 0.80:
        missing.append("capture rate below 0.80")
    if result.collision_free_rate < 0.95:
        missing.append("collision-free rate below 0.95")
    if result.minimum_separation < 0.30:
        missing.append("minimum separation below 0.30 m")
    return "; ".join(missing) if missing else None


def _run_validation(
    args,
    root: Path,
    path: Path,
    manifest: TrialManifest,
    checkpoint_loader: Callable,
    runner: Callable,
    target: TrialState,
) -> ControllerResult:
    checkpoint = Path(manifest.paths["checkpoint"]).expanduser().resolve()
    # prefer the safety-qualified best checkpoint, then the capture peak, then
    # the final latest checkpoint
    best_safe = checkpoint.with_name("no_graph_iqld_maddpg_best_safe.pt")
    if best_safe.is_file():
        checkpoint = best_safe
    elif getattr(args, "prefer_best_capture", False):
        candidate = checkpoint.with_name("no_graph_iqld_maddpg_best_capture.pt")
        if candidate.is_file():
            checkpoint = candidate
    try:
        runtime = checkpoint_loader(checkpoint, args.device)
    except Exception as error:
        reason = "checkpoint load failed: %s" % (
            str(error) or error.__class__.__name__
        )
        failed = _terminal(
            root,
            path,
            manifest,
            TrialState.FAILED,
            reason,
            diagnostic=DiagnosticClass.PROCESS_FAILURE,
        )
        return ControllerResult(failed, path)
    try:
        result = runner(runtime, manifest, args)
        metrics = _validation_metrics(result)
        miss = _validation_miss(result)
    except Exception as error:
        reason = "validation failed: %s" % (str(error) or error.__class__.__name__)
        failed = _terminal(
            root,
            path,
            manifest,
            TrialState.FAILED,
            reason,
            diagnostic=DiagnosticClass.PROCESS_FAILURE,
        )
        return ControllerResult(failed, path)
    if miss is not None:
        rejected = _terminal(
            root,
            path,
            manifest,
            TrialState.REJECTED,
            miss,
            metrics=metrics,
        )
        return ControllerResult(rejected, path)
    advanced = transition_manifest(path, manifest, target)
    return ControllerResult(_persist_metrics(path, advanced, metrics), path)


def _formal_metrics(report: GateReport) -> dict:
    capture_times = [
        value
        for value in (
            report.nominal.mean_capture_time_seconds,
            report.robust.mean_capture_time_seconds,
        )
        if value is not None
    ]
    return {
        "nominal_capture_rate": report.nominal.capture_rate,
        "robust_capture_rate": report.robust.capture_rate,
        "collision_free_rate": min(
            report.nominal.collision_free_rate,
            report.robust.collision_free_rate,
        ),
        "minimum_separation": min(
            report.nominal.minimum_separation,
            report.robust.minimum_separation,
        ),
        "mean_capture_time": max(capture_times) if capture_times else None,
        "formal_report": report.to_dict(),
    }


def _run_formal(
    args,
    root: Path,
    path: Path,
    manifest: TrialManifest,
    checkpoint_loader: Callable,
    runner: Callable,
) -> ControllerResult:
    checkpoint = Path(manifest.paths["checkpoint"]).expanduser().resolve()
    try:
        runtime = checkpoint_loader(checkpoint, args.device)
    except Exception as error:
        failed = _terminal(
            root,
            path,
            manifest,
            TrialState.FAILED,
            "checkpoint load failed: %s"
            % (str(error) or error.__class__.__name__),
            diagnostic=DiagnosticClass.PROCESS_FAILURE,
        )
        return ControllerResult(failed, path)
    try:
        report = runner(runtime, manifest, args)
    except Exception as error:
        failed = _terminal(
            root,
            path,
            manifest,
            TrialState.FAILED,
            "formal evaluation failed: %s"
            % (str(error) or error.__class__.__name__),
            diagnostic=DiagnosticClass.PROCESS_FAILURE,
        )
        return ControllerResult(failed, path)
    if not isinstance(report, GateReport) or not deployment_gate_passed(report):
        reason = (
            "formal approval requires a typed complete GateReport"
            if not isinstance(report, GateReport)
            else "formal deployment gates were not met"
        )
        metrics = _formal_metrics(report) if isinstance(report, GateReport) else {}
        rejected = _terminal(
            root,
            path,
            manifest,
            TrialState.REJECTED,
            reason,
            metrics=metrics,
        )
        return ControllerResult(rejected, path)
    approved = _terminal(
        root,
        path,
        manifest,
        TrialState.APPROVED,
        "typed complete GateReport passed every formal gate",
        metrics=_formal_metrics(report),
    )
    return ControllerResult(approved, path)


def run_controller(
    args,
    process_factory=subprocess.Popen,
    *,
    snapshot_reader=read_trial_snapshot,
    process_group_signaler=os.killpg,
    sleep=time.sleep,
    clock=time.time,
    checkpoint_loader=None,
    short_validation_runner=None,
    confirmation_runner=None,
    formal_runner=None,
) -> ControllerResult:
    """Run exactly one durable phase, never overlapping child processes."""
    root = Path(args.root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    path, manifest = _select_manifest(args, root)
    if manifest.state in _TERMINAL_STATES:
        write_leaderboards(root, [item[1] for item in _all_manifests(root)])
        return ControllerResult(manifest, path)
    if manifest.state == TrialState.RUNNING:
        failed = _terminal(
            root,
            path,
            manifest,
            TrialState.FAILED,
            "controller restarted without a trusted child process handle; "
            "recorded PID was not signalled",
            diagnostic=DiagnosticClass.PROCESS_FAILURE,
        )
        return ControllerResult(failed, path)
    if manifest.state == TrialState.PLANNED:
        return _run_training(
            args,
            root,
            path,
            manifest,
            process_factory,
            snapshot_reader,
            process_group_signaler,
            sleep,
            clock,
        )
    loader = checkpoint_loader or _default_checkpoint_loader
    if manifest.state == TrialState.SHORT_VALIDATION:
        return _run_validation(
            args,
            root,
            path,
            manifest,
            loader,
            short_validation_runner or _default_short_validation,
            TrialState.MULTISEED_CONFIRMATION,
        )
    if manifest.state == TrialState.MULTISEED_CONFIRMATION:
        return _run_validation(
            args,
            root,
            path,
            manifest,
            loader,
            confirmation_runner or _default_confirmation,
            TrialState.FORMAL_EVALUATION,
        )
    if manifest.state == TrialState.FORMAL_EVALUATION:
        return _run_formal(
            args,
            root,
            path,
            manifest,
            loader,
            formal_runner or _default_formal,
        )
    raise AssertionError("unhandled trial state: %s" % manifest.state)


def main(argv: Optional[Sequence[str]] = None) -> None:
    result = run_controller(parse_args(argv))
    print(json.dumps(result.manifest.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
