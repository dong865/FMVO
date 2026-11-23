"""Offline checkpoint/runtime contract test for every migrated baseline."""

import tempfile
from pathlib import Path

import numpy as np

from .additional_models import IRLAgent
from .role_iqld import RoleIQLDAgent
from .config import TaskConfig
from .environment import ThreeVOneEnv
from .model import MAPPOAgent
from .runtime import load_runtime


def main():
    config = TaskConfig()
    environment = ThreeVOneEnv(config, seed=123)
    environment.reset()
    with tempfile.TemporaryDirectory(prefix="three_v_one_all_") as directory:
        root = Path(directory)
        checkpoints = {
            "mappo_open": root / "mappo.pt",
            "no_graph_iqld_maddpg": root / "iqld.pt",
            "no_graph_irl_maddpg": root / "irl.pt",
        }
        MAPPOAgent(config).save(str(checkpoints["mappo_open"]))
        RoleIQLDAgent(config).save(str(checkpoints["no_graph_iqld_maddpg"]))
        IRLAgent(config).save(str(checkpoints["no_graph_irl_maddpg"]))
        cases = list(checkpoints.items()) + [("game_theory", None)]
        for baseline, checkpoint in cases:
            runtime = load_runtime(
                baseline, None if checkpoint is None else str(checkpoint)
            )
            runtime.reset()
            measured_positions, measured_velocities = environment.control_state()
            action = runtime.actions(
                measured_positions,
                measured_velocities,
                environment.origin,
                environment.virtual_obstacles,
            )
            assert action.shape == (3, 2)
            assert np.isfinite(action).all()
            assert np.max(np.abs(action)) <= 1.00001
            print(f"PASS: {baseline} action={action.shape}")


if __name__ == "__main__":
    main()
