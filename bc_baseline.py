"""Generate a pure-BC checkpoint (no RL/IRL updates) to quantify the BC baseline.

Compares against trained IRL policies: if pure BC captures far exceed the
trained policy, the RL/IRL updates are destroying the expert behavior that BC
learned, and the deployment path should anchor on BC.
"""

import argparse
from pathlib import Path

import torch

from crazyflie_examples.three_v_one_baseline.additional_models import IRLAgent, ReplayBuffer
from crazyflie_examples.three_v_one_baseline.config import TaskConfig
from crazyflie_examples.three_v_one_baseline.training_common import load_expert_replays
from crazyflie_examples.three_v_one_baseline.train_irl import behavior_clone


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert-dataset", required=True)
    parser.add_argument("--bc-steps", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--min-qualified-expert-episodes", type=int, default=20)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = TaskConfig()
    agent = IRLAgent(config, args.device)
    expert, holdout, summary = load_expert_replays(
        args.expert_dataset,
        config,
        minimum_episodes=args.min_qualified_expert_episodes,
        holdout_fraction=0.20,
        seed=args.seed,
        max_transitions=1000,
        min_separation=0.32,
        min_reward=0.0,
    )
    loss = behavior_clone(agent, expert, args.bc_steps, args.batch_size)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    agent.save(
        str(out),
        {
            "episode": 0,
            "bc_steps": args.bc_steps,
            "bc_loss": loss,
            "expert_transitions": len(expert),
            "note": "pure BC, no RL/IRL updates",
        },
    )
    print(f"bc_loss={loss:.5f} transitions={len(expert)} saved={out}")


if __name__ == "__main__":
    main()
