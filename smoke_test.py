"""Offline end-to-end contract test for the migrated baseline."""

import argparse
import tempfile
from pathlib import Path

import numpy as np

from .config import TaskConfig
from .environment import ThreeVOneEnv
from .features import OBS_DIM
from .model import MAPPOAgent, PolicyRuntime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()
    config = TaskConfig()
    env = ThreeVOneEnv(config, seed=123)
    observations, _ = env.reset()
    assert observations.shape == (3, OBS_DIM)
    with tempfile.TemporaryDirectory(prefix="three_v_one_") as directory:
        checkpoint = Path(directory) / "smoke.pt"
        MAPPOAgent(config).save(str(checkpoint), {"purpose": "smoke_test"})
        runtime = PolicyRuntime(str(checkpoint))
        for _ in range(args.steps):
            measured_positions, measured_velocities = env.control_state()
            actions = runtime.actions(
                measured_positions,
                measured_velocities,
                env.origin,
                env.virtual_obstacles,
            )
            assert actions.shape == (3, 2) and np.isfinite(actions).all()
            _, _, terminated, truncated, _ = env.step(actions)
            if terminated or truncated:
                env.reset()
                runtime.reset()
    print(f"PASS: {args.steps} steps at {config.control_hz:.0f} Hz")


if __name__ == "__main__":
    main()
