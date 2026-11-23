"""Test policy stability on a fixed EpisodeScenario with varying environment noise."""

import argparse
from pathlib import Path

import numpy as np
import torch

from .config import TaskConfig
from .environment import ThreeVOneEnv
from .role_iqld import RoleIQLDRuntime
from .scenario import SceneSamplingOptions


def run_fixed_scenario(checkpoint: str, scenario_path: str, num_runs: int = 100,
                       base_seed: int = 1000, device: str = "cuda"):
    runtime = RoleIQLDRuntime(checkpoint, device=device, safety_guard=True)
    config = runtime.config
    scenario = torch.load(scenario_path, map_location="cpu", weights_only=False)

    results = []
    for run in range(num_runs):
        env = ThreeVOneEnv(config, seed=base_seed + run, scene_options=SceneSamplingOptions())
        # Monkey-patch scenario generator to always return the fixed scenario
        env.scenario_generator.sample = lambda: scenario
        env.reset()
        runtime.reset()

        captured = False
        length = 0
        min_separation = float("inf")
        for _ in range(config.max_episode_steps):
            positions, velocities = env.control_state()
            actions = runtime.actions(
                positions, velocities, env.origin, env.virtual_obstacles
            )
            _, _, terminated, truncated, info = env.step(actions)
            length += 1
            min_separation = min(min_separation, float(info["minimum_inter_drone_distance"]))
            if terminated or truncated:
                captured = bool(info.get("captured", False))
                break
        results.append({
            "captured": captured,
            "length": length,
            "min_separation": min_separation,
            "outcome": info.get("outcome", "unknown"),
        })

    captures = sum(1 for r in results if r["captured"])
    collision_free = sum(
        1 for r in results
        if r["min_separation"] >= config.safety_emergency_distance and r["outcome"] != "safety_failure"
    )
    print(f"Fixed scenario repeatability test: {scenario_path}")
    print(f"Runs: {num_runs}")
    print(f"Captures: {captures} ({captures / num_runs * 100:.1f}%)")
    print(f"Collision-free: {collision_free} ({collision_free / num_runs * 100:.1f}%)")
    print(f"Mean episode length: {np.mean([r['length'] for r in results]):.1f} steps")
    print(f"Mean min separation: {np.mean([r['min_separation'] for r in results]):.3f} m")

    outcomes = {}
    for r in results:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
    print("Outcomes:")
    for outcome, count in sorted(outcomes.items()):
        print(f"  {outcome}: {count} ({count / num_runs * 100:.1f}%)")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run_fixed_scenario(args.checkpoint, args.scenario, args.runs, args.seed, args.device)


if __name__ == "__main__":
    main()
