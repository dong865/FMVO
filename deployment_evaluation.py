"""Evaluate every 3V1 baseline against shared simulation deployment gates."""

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .config import TaskConfig
from .environment import ThreeVOneEnv
from .runtime import BASELINES, load_runtime


DEFAULT_EVALUATION_SEEDS = (101, 211, 307, 401, 503)


def evaluation_seeds() -> Tuple[int, ...]:
    """Return the held-out seed schedule shared by all baselines."""
    return DEFAULT_EVALUATION_SEEDS


def wilson_interval(
    successes: int, trials: int, z: float = 1.959963984540054
) -> Tuple[float, float]:
    """Return the two-sided Wilson score interval for a binomial rate."""
    if trials <= 0:
        raise ValueError("trials must be positive")
    if successes < 0 or successes > trials:
        raise ValueError("successes must be between zero and trials")
    observed = successes / trials
    denominator = 1.0 + z * z / trials
    center = (observed + z * z / (2.0 * trials)) / denominator
    margin = (
        z
        * math.sqrt(
            observed * (1.0 - observed) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


@dataclass(frozen=True)
class EpisodeResult:
    """Metrics from one deterministic scenario/seed/episode evaluation."""

    scenario: str
    seed: int
    episode: int
    captured: bool
    collision_free: bool
    capture_time_seconds: Optional[float]
    minimum_separation: float
    safety_intervention_rate: float
    command_limit_violations: int
    inter_drone_collision_free: bool = True
    obstacle_collision_free: bool = True
    outcome: str = "unknown"
    evader_succeeded: bool = False
    timed_out: bool = False
    safety_failure: bool = False
    final_goal_progress: float = 0.0


@dataclass(frozen=True)
class SeedResult:
    """Aggregate metrics for one held-out seed."""

    scenario: str
    seed: int
    episodes: int
    captures: int
    collision_free_episodes: int
    mean_capture_time_seconds: Optional[float]
    minimum_separation: float
    mean_safety_intervention_rate: float
    command_limit_violations: int

    @classmethod
    def from_episodes(cls, episodes: Sequence[EpisodeResult]) -> "SeedResult":
        if not episodes:
            raise ValueError("seed aggregation requires at least one episode")
        scenario = episodes[0].scenario
        seed = episodes[0].seed
        if any(item.scenario != scenario or item.seed != seed for item in episodes):
            raise ValueError("seed aggregation received mixed scenario or seed data")
        capture_times = [
            item.capture_time_seconds
            for item in episodes
            if item.capture_time_seconds is not None
        ]
        return cls(
            scenario=scenario,
            seed=seed,
            episodes=len(episodes),
            captures=sum(item.captured for item in episodes),
            collision_free_episodes=sum(item.collision_free for item in episodes),
            mean_capture_time_seconds=(
                float(np.mean(capture_times)) if capture_times else None
            ),
            minimum_separation=min(item.minimum_separation for item in episodes),
            mean_safety_intervention_rate=float(
                np.mean([item.safety_intervention_rate for item in episodes])
            ),
            command_limit_violations=sum(
                item.command_limit_violations for item in episodes
            ),
        )


@dataclass(frozen=True)
class ScenarioResult:
    """Aggregate result for nominal or robust dynamics."""

    scenario: str
    episode_results: Tuple[EpisodeResult, ...]
    seed_results: Tuple[SeedResult, ...]

    @classmethod
    def from_episodes(
        cls, scenario: str, episodes: Sequence[EpisodeResult]
    ) -> "ScenarioResult":
        values = tuple(episodes)
        if any(item.scenario != scenario for item in values):
            raise ValueError("scenario aggregation received mixed data")
        grouped: Dict[int, list] = {}
        for item in values:
            grouped.setdefault(item.seed, []).append(item)
        seeds = tuple(
            SeedResult.from_episodes(grouped[seed]) for seed in sorted(grouped)
        )
        return cls(scenario, values, seeds)

    @property
    def episodes(self) -> int:
        return len(self.episode_results)

    @property
    def captures(self) -> int:
        return sum(item.captured for item in self.episode_results)

    @property
    def capture_rate(self) -> float:
        return self.captures / self.episodes if self.episodes else 0.0

    @property
    def collision_free_episodes(self) -> int:
        return sum(item.collision_free for item in self.episode_results)

    @property
    def collision_free_rate(self) -> float:
        return self.collision_free_episodes / self.episodes if self.episodes else 0.0

    @property
    def inter_drone_collision_free_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(
            item.inter_drone_collision_free for item in self.episode_results
        ) / self.episodes

    @property
    def obstacle_collision_free_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(
            item.obstacle_collision_free for item in self.episode_results
        ) / self.episodes

    @property
    def minimum_separation(self) -> float:
        if not self.episode_results:
            return 0.0
        return min(item.minimum_separation for item in self.episode_results)

    @property
    def mean_capture_time_seconds(self) -> Optional[float]:
        values = [
            item.capture_time_seconds
            for item in self.episode_results
            if item.capture_time_seconds is not None
        ]
        return float(np.mean(values)) if values else None

    @property
    def mean_safety_intervention_rate(self) -> float:
        if not self.episode_results:
            return 0.0
        return float(
            np.mean(
                [item.safety_intervention_rate for item in self.episode_results]
            )
        )

    @property
    def command_limit_violations(self) -> int:
        return sum(item.command_limit_violations for item in self.episode_results)

    def rate_interval(self, attribute: str) -> Tuple[float, float]:
        if not self.episodes:
            return 0.0, 1.0
        return wilson_interval(int(getattr(self, attribute)), self.episodes)

    def to_dict(self) -> Dict[str, object]:
        return {
            "scenario": self.scenario,
            "episodes": self.episodes,
            "captures": self.captures,
            "capture_rate": self.capture_rate,
            "capture_rate_wilson95": self.rate_interval("captures"),
            "collision_free_episodes": self.collision_free_episodes,
            "collision_free_rate": self.collision_free_rate,
            "inter_drone_collision_free_rate": (
                self.inter_drone_collision_free_rate
            ),
            "obstacle_collision_free_rate": self.obstacle_collision_free_rate,
            "collision_free_rate_wilson95": self.rate_interval(
                "collision_free_episodes"
            ),
            "mean_capture_time_seconds": self.mean_capture_time_seconds,
            "minimum_separation": self.minimum_separation,
            "mean_safety_intervention_rate": self.mean_safety_intervention_rate,
            "command_limit_violations": self.command_limit_violations,
            "seeds": [asdict(item) for item in self.seed_results],
        }


@dataclass(frozen=True)
class GateThresholds:
    """Approved quantitative threshold and sample-size contract."""

    nominal_capture_rate: float = 0.95
    robust_capture_rate: float = 0.80
    collision_free_rate: float = 0.95
    minimum_separation: float = 0.30
    required_seeds: int = 5
    episodes_per_seed: int = 100


@dataclass(frozen=True)
class GateReport:
    """Combined nominal and robust deployment-readiness report."""

    nominal: ScenarioResult
    robust: ScenarioResult
    thresholds: GateThresholds

    @classmethod
    def from_episode_results(
        cls,
        nominal: Sequence[EpisodeResult],
        robust: Sequence[EpisodeResult],
        thresholds: Optional[GateThresholds] = None,
    ) -> "GateReport":
        return cls(
            nominal=ScenarioResult.from_episodes("nominal", nominal),
            robust=ScenarioResult.from_episodes("robust", robust),
            thresholds=thresholds or GateThresholds(),
        )

    def _sample_complete(self, scenario: ScenarioResult) -> bool:
        required = self.thresholds.episodes_per_seed
        expected_seeds = set(
            evaluation_seeds()[: self.thresholds.required_seeds]
        )
        actual_seeds = {item.seed for item in scenario.seed_results}
        if not expected_seeds.issubset(actual_seeds):
            return False
        if len(scenario.seed_results) < self.thresholds.required_seeds:
            return False
        return all(item.episodes >= required for item in scenario.seed_results)

    @property
    def failed_gates(self) -> Tuple[str, ...]:
        failures = []
        if not self._sample_complete(self.nominal) or not self._sample_complete(
            self.robust
        ):
            failures.append("sample_size")
        if self.nominal.capture_rate < self.thresholds.nominal_capture_rate:
            failures.append("nominal_capture_rate")
        if self.robust.capture_rate < self.thresholds.robust_capture_rate:
            failures.append("robust_capture_rate")
        if min(
            self.nominal.collision_free_rate,
            self.robust.collision_free_rate,
        ) < self.thresholds.collision_free_rate:
            failures.append("collision_free_rate")
        if min(
            self.nominal.minimum_separation,
            self.robust.minimum_separation,
        ) < self.thresholds.minimum_separation:
            failures.append("minimum_separation")
        if (
            self.nominal.command_limit_violations
            + self.robust.command_limit_violations
            > 0
        ):
            failures.append("command_limit_violations")
        return tuple(failures)

    @property
    def deployment_ready(self) -> bool:
        return not self.failed_gates

    def to_dict(self) -> Dict[str, object]:
        return {
            "status": (
                "deployment_ready" if self.deployment_ready else "not_deployment_ready"
            ),
            "deployment_ready": self.deployment_ready,
            "failed_gates": self.failed_gates,
            "thresholds": asdict(self.thresholds),
            "nominal": self.nominal.to_dict(),
            "robust": self.robust.to_dict(),
        }


def _command_limit_violations(environment: ThreeVOneEnv) -> int:
    config = environment.config
    pursuer_speeds = np.linalg.norm(environment.commanded_velocities[:3], axis=1)
    evader_speed = float(np.linalg.norm(environment.commanded_velocities[3]))
    return int(np.count_nonzero(pursuer_speeds > config.max_speed + 1e-6)) + int(
        evader_speed > config.evader_max_speed + 1e-6
    )


def _run_scenario(
    runtime,
    config: TaskConfig,
    seeds: Sequence[int],
    episodes_per_seed: int,
    scenario: str,
    max_steps: Optional[int] = None,
    scene_options=None,
) -> Tuple[EpisodeResult, ...]:
    if config.scenario_contract_version != 2:
        raise ValueError(
            "formal evaluation requires scenario contract v2; regenerate old "
            "expert data and checkpoints"
        )
    results = []
    step_limit = max_steps or config.max_episode_steps
    for seed in seeds:
        environment = ThreeVOneEnv(
            config, seed=int(seed), scene_options=scene_options
        )
        for episode in range(episodes_per_seed):
            environment.reset()
            runtime.reset()
            captured = False
            collision_free = True
            inter_drone_collision_free = True
            obstacle_collision_free = True
            minimum_separation = float("inf")
            intervention_steps = 0
            limit_violations = 0
            for _ in range(step_limit):
                positions, velocities = environment.control_state()
                actions = np.asarray(
                    runtime.actions(
                        positions,
                        velocities,
                        environment.origin,
                        environment.virtual_obstacles,
                    ),
                    dtype=np.float32,
                )
                if actions.shape != (config.num_pursuers, 2):
                    raise ValueError(
                        "runtime actions must have shape "
                        f"({config.num_pursuers}, 2), received {actions.shape}"
                    )
                if not np.all(np.isfinite(actions)):
                    limit_violations += int(np.count_nonzero(~np.isfinite(actions)))
                    actions = np.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
                _, _, terminated, truncated, info = environment.step(actions)
                separation = float(info["minimum_inter_drone_distance"])
                minimum_separation = min(minimum_separation, separation)
                obstacle_collision = bool(np.any(info["obstacle_collisions"]))
                inter_drone_collision_free = bool(
                    inter_drone_collision_free
                    and separation >= config.safety_emergency_distance
                )
                obstacle_collision_free = bool(
                    obstacle_collision_free and not obstacle_collision
                )
                collision_free = bool(
                    inter_drone_collision_free and obstacle_collision_free
                )
                intervention_steps += int(info["safety_intervened"])
                limit_violations += _command_limit_violations(environment)
                if terminated or truncated:
                    captured = bool(info["captured"])
                    break
            capture_time = (
                environment.step_count / config.control_hz if captured else None
            )
            results.append(
                EpisodeResult(
                    scenario=scenario,
                    seed=int(seed),
                    episode=episode,
                    captured=captured,
                    collision_free=collision_free,
                    capture_time_seconds=capture_time,
                    minimum_separation=minimum_separation,
                    safety_intervention_rate=(
                        intervention_steps / max(environment.step_count, 1)
                    ),
                    command_limit_violations=limit_violations,
                    inter_drone_collision_free=inter_drone_collision_free,
                    obstacle_collision_free=obstacle_collision_free,
                    outcome=str(info["outcome"]),
                    evader_succeeded=bool(info["evader_succeeded"]),
                    timed_out=bool(info["outcome"] == "timeout"),
                    safety_failure=bool(info["safety_failure"]),
                    final_goal_progress=(
                        1.0
                        - float(info["distance_to_goal"])
                        / max(environment.initial_goal_distance, 1e-8)
                    ),
                )
            )
    return tuple(results)


def evaluate_runtime(
    runtime,
    config: TaskConfig,
    seeds: Sequence[int] = DEFAULT_EVALUATION_SEEDS,
    episodes_per_seed: int = 100,
    max_steps: Optional[int] = None,
    thresholds: Optional[GateThresholds] = None,
    scene_options=None,
) -> GateReport:
    """Evaluate one runtime under identical nominal and robust scenarios."""
    if not seeds or episodes_per_seed <= 0:
        raise ValueError("evaluation requires seeds and positive episodes_per_seed")
    robust_config = replace(config, disturbances_enabled=True)
    nominal_config = replace(config, disturbances_enabled=False)
    nominal = _run_scenario(
        runtime, nominal_config, seeds, episodes_per_seed, "nominal",
        max_steps, scene_options,
    )
    robust = _run_scenario(
        runtime, robust_config, seeds, episodes_per_seed, "robust",
        max_steps, scene_options,
    )
    return GateReport.from_episode_results(nominal, robust, thresholds)


def _write_report(report: GateReport, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(report.to_dict(), stream, indent=2, sort_keys=True)
    fieldnames = tuple(EpisodeResult.__dataclass_fields__)
    with (output / "episodes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for item in report.nominal.episode_results + report.robust.episode_results:
            writer.writerow(asdict(item))
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as error:
        raise RuntimeError(
            "TensorBoard is required for deployment evaluation"
        ) from error
    writer = SummaryWriter(log_dir=str(output / "tensorboard"))
    for step, scenario in enumerate((report.nominal, report.robust), start=1):
        writer.add_scalar(
            f"{scenario.scenario}/capture_rate", scenario.capture_rate, step
        )
        writer.add_scalar(
            f"{scenario.scenario}/collision_free_rate",
            scenario.collision_free_rate,
            step,
        )
        writer.add_scalar(
            f"{scenario.scenario}/minimum_separation",
            scenario.minimum_separation,
            step,
        )
        writer.add_scalar(
            f"{scenario.scenario}/safety_intervention_rate",
            scenario.mean_safety_intervention_rate,
            step,
        )
    writer.add_scalar("gates/deployment_ready", report.deployment_ready, 1)
    writer.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", choices=BASELINES, required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=evaluation_seeds())
    parser.add_argument("--episodes-per-seed", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--scene-preset", default=None,
        help="scene difficulty preset (None=legacy, or scene_preset() name "
             "such as 'hard'); the SAME scene generator is used for nominal "
             "and robust scenarios",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = TaskConfig()
    scene_options = None
    if args.scene_preset:
        from .scenario import scene_preset

        scene_options = scene_preset(args.scene_preset)
    runtime = load_runtime(args.baseline, args.checkpoint, device=args.device)
    report = evaluate_runtime(
        runtime,
        config,
        seeds=tuple(args.seeds),
        episodes_per_seed=args.episodes_per_seed,
        max_steps=args.max_steps,
        scene_options=scene_options,
    )
    output = Path(args.output).expanduser().resolve()
    _write_report(report, output)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    print(f"report_dir={output}")


if __name__ == "__main__":
    main()
