"""Fail-closed decisions for sequential Role-IQLD tuning trials."""

import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

from .deployment_evaluation import (
    DEFAULT_EVALUATION_SEEDS,
    GateReport,
    wilson_interval,
)
from .iqld_hyperparameters import IQLDHyperparameters
from .tuning_manifest import TrialState
from .tuning_monitor import MINIMUM_SEPARATION_M, TRIAL_HARD_SEPARATION_M, MonitorSnapshot
from .training_validation import ValidationResult


FORMAL_SEEDS = (101, 211, 307, 401, 503)
FORMAL_EPISODES_PER_SEED = 100
MINIMUM_COLLISION_FREE_RATE = 0.95
MINIMUM_ROBUST_CAPTURE_RATE = 0.80
MINIMUM_NOMINAL_CAPTURE_RATE = 0.95


class DiagnosticClass(str, Enum):
    """Primary actionable reason that a trial may not advance."""

    PROCESS_FAILURE = "process_failure"
    STALE_TELEMETRY = "stale_telemetry"
    MISSING_TELEMETRY = "missing_telemetry"
    NON_FINITE_METRICS = "non_finite_metrics"
    INTER_DRONE_SAFETY = "inter_drone_safety"
    OBSTACLE_COLLISION = "obstacle_collision"
    CURRICULUM_STALL = "curriculum_stall"
    INSUFFICIENT_EXPLORATION = "insufficient_exploration"


class TuningPaused(RuntimeError):
    """Raised when repeated evidence needs human review before another trial."""


@dataclass(frozen=True)
class ParameterChange:
    """One auditable hyperparameter delta and its resolved child settings."""

    field: str
    before: float
    after: float
    parameters: IQLDHyperparameters


def _is_finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _latest(snapshot: MonitorSnapshot, name: str, default: float = 0.0) -> float:
    value = snapshot.latest.get(name, default)
    return float(value) if _is_finite(value) else default


def _validation_is_unsafe(validation: Optional[Any]) -> bool:
    if validation is None:
        return False
    if isinstance(validation, GateReport):
        return not _hard_safe(validation)
    collision_free = getattr(validation, "collision_free_rate", None)
    minimum_separation = getattr(validation, "minimum_separation", None)
    return (
        collision_free is not None
        and (
            not _is_finite(collision_free)
            or collision_free < MINIMUM_COLLISION_FREE_RATE
        )
    ) or (
        minimum_separation is not None
        and (
            not _is_finite(minimum_separation)
            or minimum_separation < MINIMUM_SEPARATION_M
        )
    )


def classify_failure(
    snapshot: MonitorSnapshot,
    validation: Optional[ValidationResult],
    process: Optional[bool],
) -> DiagnosticClass:
    """Classify evidence in safety and infrastructure precedence order."""
    if process is False or snapshot.process_alive is False:
        return DiagnosticClass.PROCESS_FAILURE
    if snapshot.stale:
        return DiagnosticClass.STALE_TELEMETRY
    if snapshot.read_error is not None or snapshot.missing_tags:
        return DiagnosticClass.MISSING_TELEMETRY
    if any(not _is_finite(value) for value in snapshot.latest.values()):
        return DiagnosticClass.NON_FINITE_METRICS
    if (
        _latest(snapshot, "safety/minimum_separation", TRIAL_HARD_SEPARATION_M)
        < TRIAL_HARD_SEPARATION_M
        or _validation_is_unsafe(validation)
    ):
        return DiagnosticClass.INTER_DRONE_SAFETY
    if (
        _latest(snapshot, "curriculum/stage") == 0.0
        and _latest(snapshot, "curriculum/capture_rate") <= 0.0
        and _latest(snapshot, "progress/environment_steps") > 0.0
    ):
        return DiagnosticClass.CURRICULUM_STALL
    return DiagnosticClass.INSUFFICIENT_EXPLORATION


def _rate(result: Any, names: Sequence[str], default: float = 0.0) -> float:
    for name in names:
        value = getattr(result, name, None)
        if value is not None:
            return float(value)
    return default


def _mean_capture_time(result: Any) -> float:
    if isinstance(result, GateReport):
        values = (
            result.nominal.mean_capture_time_seconds,
            result.robust.mean_capture_time_seconds,
        )
        finite = [
            float(value)
            for value in values
            if value is not None and _is_finite(value)
        ]
        return max(finite) if finite else math.inf
    value = _rate(
        result,
        ("mean_capture_time_seconds", "mean_capture_time", "capture_time"),
        math.inf,
    )
    return value if _is_finite(value) else math.inf


def _obstacle_free(result: Any) -> bool:
    if isinstance(result, GateReport):
        return all(
            item.obstacle_collision_free
            for scenario in (result.nominal, result.robust)
            for item in scenario.episode_results
        )
    collisions = _rate(result, ("obstacle_collisions", "obstacle_collision_count"))
    return _is_finite(collisions) and collisions == 0.0


def _hard_safe(result: Any) -> bool:
    if isinstance(result, GateReport):
        scenarios = (result.nominal, result.robust)
        return (
            all(
                _is_finite(scenario.collision_free_rate)
                and scenario.collision_free_rate >= MINIMUM_COLLISION_FREE_RATE
                and _is_finite(scenario.minimum_separation)
                and scenario.minimum_separation >= MINIMUM_SEPARATION_M
                and scenario.command_limit_violations == 0
                for scenario in scenarios
            )
            and _obstacle_free(result)
        )
    collision_free = _rate(result, ("collision_free_rate", "collision_free"))
    minimum_separation = _rate(
        result, ("minimum_separation", "min_sep", "minimum_separation_m")
    )
    return (
        _is_finite(collision_free)
        and collision_free >= MINIMUM_COLLISION_FREE_RATE
        and _is_finite(minimum_separation)
        and minimum_separation >= MINIMUM_SEPARATION_M
        and _obstacle_free(result)
    )


def candidate_rank(result: Any) -> Tuple[bool, float, float, float, float]:
    """Return a lexicographic safety-first candidate ordering key."""
    if isinstance(result, GateReport):
        collision_free = min(
            result.nominal.collision_free_rate, result.robust.collision_free_rate
        )
        robust_capture = result.robust.capture_rate
        nominal_capture = result.nominal.capture_rate
    else:
        collision_free = _rate(result, ("collision_free_rate", "collision_free"))
        robust_capture = _rate(
            result,
            ("robust_capture_rate", "robust_capture", "capture_rate", "capture"),
        )
        nominal_capture = _rate(
            result,
            (
                "nominal_capture_rate",
                "nominal_capture",
                "capture_rate",
                "capture",
            ),
        )
    return (
        _hard_safe(result),
        collision_free,
        robust_capture,
        nominal_capture,
        -_mean_capture_time(result),
    )


def _episode_evidence_complete(episode: Any, scenario: str) -> bool:
    if episode.scenario != scenario or episode.seed not in FORMAL_SEEDS:
        return False
    if (
        not isinstance(episode.captured, bool)
        or not isinstance(episode.collision_free, bool)
        or not isinstance(episode.obstacle_collision_free, bool)
        or not isinstance(episode.command_limit_violations, int)
    ):
        return False
    if not all(
        _is_finite(value)
        for value in (episode.minimum_separation, episode.safety_intervention_rate)
    ):
        return False
    if (
        episode.capture_time_seconds is not None
        and not _is_finite(episode.capture_time_seconds)
    ):
        return False
    return episode.obstacle_collision_free and episode.command_limit_violations == 0


def _seed_record_matches_episodes(record: Any, episodes: Sequence[Any]) -> bool:
    if record.episodes != len(episodes) or record.episodes != FORMAL_EPISODES_PER_SEED:
        return False
    if (
        not isinstance(record.captures, int)
        or not isinstance(record.collision_free_episodes, int)
        or not isinstance(record.command_limit_violations, int)
        or not _is_finite(record.minimum_separation)
        or not _is_finite(record.mean_safety_intervention_rate)
    ):
        return False
    if (
        record.captures < 0
        or record.captures > record.episodes
        or record.collision_free_episodes < 0
        or record.collision_free_episodes > record.episodes
        or record.command_limit_violations != 0
    ):
        return False
    if (
        record.mean_capture_time_seconds is not None
        and not _is_finite(record.mean_capture_time_seconds)
    ):
        return False
    captures = sum(item.captured for item in episodes)
    collision_free = sum(item.collision_free for item in episodes)
    capture_times = [
        item.capture_time_seconds
        for item in episodes
        if item.capture_time_seconds is not None
    ]
    mean_capture_time = (
        sum(capture_times) / len(capture_times) if capture_times else None
    )
    mean_intervention = sum(
        item.safety_intervention_rate for item in episodes
    ) / len(episodes)
    if (
        record.captures != captures
        or record.collision_free_episodes != collision_free
        or not math.isclose(
            record.minimum_separation,
            min(item.minimum_separation for item in episodes),
        )
        or not math.isclose(record.mean_safety_intervention_rate, mean_intervention)
    ):
        return False
    if (record.mean_capture_time_seconds is None) != (mean_capture_time is None):
        return False
    if (
        mean_capture_time is not None
        and not math.isclose(record.mean_capture_time_seconds, mean_capture_time)
    ):
        return False
    confidence_intervals = (
        wilson_interval(record.captures, record.episodes)
        + wilson_interval(record.collision_free_episodes, record.episodes)
    )
    return all(_is_finite(value) for value in confidence_intervals)


def _gate_report_complete(report: GateReport) -> bool:
    """Verify exact raw evidence and per-seed metric/CI records."""
    if tuple(DEFAULT_EVALUATION_SEEDS) != FORMAL_SEEDS:
        return False
    for expected_name, scenario in (
        ("nominal", report.nominal),
        ("robust", report.robust),
    ):
        if scenario.scenario != expected_name:
            return False
        by_seed = {}
        for episode in scenario.episode_results:
            if not _episode_evidence_complete(episode, expected_name):
                return False
            by_seed.setdefault(episode.seed, []).append(episode)
        if set(by_seed) != set(FORMAL_SEEDS):
            return False
        if any(
            len(episodes) != FORMAL_EPISODES_PER_SEED
            for episodes in by_seed.values()
        ):
            return False
        if len(scenario.episode_results) != len(FORMAL_SEEDS) * FORMAL_EPISODES_PER_SEED:
            return False
        records = {record.seed: record for record in scenario.seed_results}
        if len(records) != len(scenario.seed_results) or set(records) != set(FORMAL_SEEDS):
            return False
        for seed, episodes in by_seed.items():
            record = records[seed]
            if record.scenario != expected_name:
                return False
            if not _seed_record_matches_episodes(record, episodes):
                return False
        aggregates = (
            scenario.capture_rate,
            scenario.collision_free_rate,
            scenario.minimum_separation,
            scenario.mean_safety_intervention_rate,
            scenario.mean_capture_time_seconds,
        )
        if not all(value is not None and _is_finite(value) for value in aggregates):
            return False
    return True


def deployment_gate_passed(results: Any) -> bool:
    """Permit deployment only with complete, typed formal evidence."""
    if not isinstance(results, GateReport):
        return False
    if not _gate_report_complete(results) or not _hard_safe(results):
        return False
    return (
        results.nominal.capture_rate >= MINIMUM_NOMINAL_CAPTURE_RATE
        and results.robust.capture_rate >= MINIMUM_ROBUST_CAPTURE_RATE
    )


def _history_value(item: Any, name: str) -> Any:
    if isinstance(item, Mapping):
        return item.get(name)
    return getattr(item, name, None)


def _same_class_failures(
    diagnostic: DiagnosticClass, history: Iterable[Any]
) -> int:
    count = 0
    for item in reversed(tuple(history)):
        state = _history_value(item, "state")
        if isinstance(state, TrialState):
            state = state.value
        if state not in {
            TrialState.REJECTED.value,
            TrialState.FAILED.value,
            "rejected",
            "failed",
        }:
            break
        observed = _history_value(item, "diagnostic_class")
        observed = observed.value if isinstance(observed, DiagnosticClass) else observed
        if observed != diagnostic.value:
            break
        count += 1
    return count


_DELTAS = {
    DiagnosticClass.PROCESS_FAILURE: (
        "policy_learning_rate",
        (lambda value, _: value * 0.5, lambda value, _: value * 2.0),
    ),
    DiagnosticClass.STALE_TELEMETRY: (
        "value_learning_rate",
        (lambda value, _: value * 0.5, lambda value, _: value * 2.0),
    ),
    DiagnosticClass.MISSING_TELEMETRY: (
        "value_learning_rate",
        (lambda value, _: value * 0.5, lambda value, _: value * 2.0),
    ),
    DiagnosticClass.NON_FINITE_METRICS: (
        "critic_learning_rate",
        (lambda value, _: value * 0.5, lambda value, _: value * 2.0),
    ),
    DiagnosticClass.INTER_DRONE_SAFETY: (
        "assignment_cost_weight",
        (lambda value, _: value + 0.05, lambda value, _: value * 0.5),
    ),
    DiagnosticClass.OBSTACLE_COLLISION: (
        "assignment_entropy_weight",
        (lambda value, _: value + 0.01, lambda value, _: value * 0.5),
    ),
    DiagnosticClass.CURRICULUM_STALL: (
        "expert_fraction_start",
        (
            lambda value, _: min(1.0, value + 0.05),
            lambda value, parameters: max(
                parameters.expert_fraction_end, value - 0.05
            ),
        ),
    ),
    DiagnosticClass.INSUFFICIENT_EXPLORATION: (
        "exploration_noise",
        (lambda value, _: value + 0.05, lambda value, _: value * 0.5),
    ),
}


def next_single_change(
    diagnostic: DiagnosticClass,
    parent_parameters: IQLDHyperparameters,
    history: Iterable[Any],
) -> ParameterChange:
    """Create one validated, non-environmental hyperparameter change."""
    if _same_class_failures(diagnostic, history) >= 3:
        raise TuningPaused(
            "three consecutive %s failures require human review" % diagnostic.value
        )
    field, transforms = _DELTAS[diagnostic]
    before = float(getattr(parent_parameters, field))
    for transform in transforms:
        after = float(transform(before, parent_parameters))
        if after == before or not _is_finite(after):
            continue
        try:
            parameters = replace(parent_parameters, **{field: after})
        except ValueError:
            continue
        return ParameterChange(field, before, after, parameters)
    raise TuningPaused(
        "parameter space exhausted for %s: %s cannot change"
        % (diagnostic.value, field)
    )
