"""Evaluate the migrated 50 Hz game-theory baseline without ROS."""

import argparse

import numpy as np

from .config import TaskConfig
from .environment import ThreeVOneEnv
from .environment_options import add_environment_args, apply_environment_args
from .expert_collection import collect_qualified_expert_dataset
from .obstacles import configure_obstacles
from .paper_game_theory import GameTheoryRuntime
from .robustness_metrics import RobustnessMetrics
from .tensorboard_logging import TrainingLogger, add_tensorboard_args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=1750)
    parser.add_argument("--obstacles-file", default=None)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--collect-qualified-expert", action="store_true")
    parser.add_argument(
        "--expert-dataset",
        default="checkpoints/three_v_one_expert/final_robust_safe_expert.pt",
    )
    parser.add_argument("--target-qualified-episodes", type=int, default=20)
    parser.add_argument("--max-collection-attempts", type=int, default=100)
    parser.add_argument(
        "--scene-preset", default=None,
        help="scene difficulty preset for collection (None=legacy, or a "
             "scene_preset() name such as 'hard')",
    )
    add_tensorboard_args(parser)
    add_environment_args(parser)
    args = parser.parse_args()
    config = TaskConfig()
    apply_environment_args(config, args)
    scene_options = None
    if args.scene_preset:
        from .scenario import scene_preset

        scene_options = scene_preset(args.scene_preset)
    if args.obstacles_file:
        configure_obstacles(config, args.obstacles_file)
    if args.collect_qualified_expert:
        logger = TrainingLogger(args, "game_theory_expert_collection", config)
        summary = collect_qualified_expert_dataset(
            config,
            args.expert_dataset,
            args.target_qualified_episodes,
            args.max_collection_attempts,
            args.seed,
            args.max_steps,
            scene_options=scene_options,
        )
        logger.scalars(
            {
                "expert/attempted": summary.attempted,
                "expert/accepted": summary.accepted,
                "expert/rejected": summary.rejected,
                "expert/acceptance_rate": summary.accepted / summary.attempted,
                "expert/minimum_separation_m": summary.minimum_separation,
                "expert/total_persisted": summary.total_persisted,
            },
            summary.attempted,
        )
        logger.close()
        print(summary)
        return
    environment = ThreeVOneEnv(config, seed=args.seed)
    controller = GameTheoryRuntime(config)
    logger = TrainingLogger(args, "game_theory", config)
    captures = 0
    returns = []
    total_steps = 0
    for episode in range(args.episodes):
        environment.reset()
        controller.reset()
        episode_return = 0.0
        captured = False
        obstacle_collisions = 0
        action_magnitudes = []
        robustness = RobustnessMetrics()
        for _ in range(args.max_steps):
            measured_positions, measured_velocities = environment.control_state()
            action = controller.actions(
                measured_positions,
                measured_velocities,
                environment.origin,
                environment.virtual_obstacles,
            )
            action_magnitudes.append(float(np.linalg.norm(action, axis=1).mean()))
            _, reward, terminated, truncated, info = environment.step(action)
            total_steps += 1
            obstacle_collisions += int(
                np.count_nonzero(info["obstacle_collisions"][: config.num_pursuers])
            )
            robustness.update(info)
            episode_return += float(np.mean(reward))
            if terminated or truncated:
                captured = bool(info["captured"])
                break
        captures += int(captured)
        returns.append(episode_return)
        logger.scalars(
            {
                "episode/return": episode_return,
                "episode/length": environment.step_count,
                "episode/captured": int(captured),
                "episode/obstacle_collisions": obstacle_collisions,
                "metrics/cumulative_capture_rate": captures / (episode + 1),
                "metrics/cumulative_mean_return": float(np.mean(returns)),
                "policy/mean_action_magnitude": (
                    float(np.mean(action_magnitudes)) if action_magnitudes else 0.0
                ),
                "progress/environment_steps": total_steps,
                **robustness.tensorboard(),
            },
            episode + 1,
        )
        logger.flush()
        print(
            f"episode={episode + 1}/{args.episodes} "
            f"return={episode_return:.3f} captured={captured}"
        )
    print(
        f"capture_rate={captures / max(args.episodes, 1):.2%} "
        f"mean_return={np.mean(returns):.3f}"
    )
    logger.close()


if __name__ == "__main__":
    main()
