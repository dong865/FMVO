"""Train the migrated no-graph IRL-MADDPG baseline at 50 Hz."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .additional_models import IRLAgent, ReplayBuffer, load_checkpoint
from .config import TaskConfig
from .curriculum import (
    add_curriculum_args,
    controller_from_args,
    curriculum_stages,
    stage_zero_stall_message,
)
from .environment import ThreeVOneEnv
from .environment_options import add_environment_args, apply_environment_args
from .obstacles import configure_obstacles
from .robustness_metrics import RobustnessMetrics
from .tensorboard_logging import TrainingLogger, add_tensorboard_args
from .training_diagnostics import EpisodeDiagnostics
from .training_validation import (
    add_validation_args,
    checkpoint_is_better,
    validate_checkpoint,
)
from .training_common import (
    add_expert_dataset_args,
    current_actor_observations,
    load_expert_replays,
    make_epn_samples,
    train_epn,
    zero_prediction_observations,
)


def behavior_clone(agent: IRLAgent, expert: ReplayBuffer, steps, batch_size):
    losses = []
    for _ in range(steps):
        if len(expert) < batch_size:
            break
        _, _, observations, actions, *_ = expert.sample(batch_size, agent.device)
        prediction = agent.actor(observations.flatten(0, 1)).view(-1, 3, 2)
        loss = nn.functional.mse_loss(prediction, actions)
        agent.actor_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.actor.parameters(), 5.0)
        agent.actor_optimizer.step()
        losses.append(float(loss.detach().cpu()))
    agent.target_actor.load_state_dict(agent.actor.state_dict())
    return float(np.mean(losses)) if losses else 0.0


def discriminator_metrics(expert_score, policy_score):
    """Return detached Wasserstein-score diagnostics for expert and policy."""
    expert_mean = expert_score.detach().mean().cpu()
    policy_mean = policy_score.detach().mean().cpu()
    return {
        "expert_score": float(expert_mean),
        "policy_score": float(policy_mean),
        "separation": float(expert_mean - policy_mean),
    }


def update_discriminator(
    agent: IRLAgent,
    expert: ReplayBuffer,
    policy: ReplayBuffer,
    batch_size: int,
) -> dict:
    if len(expert) < batch_size or len(policy) < batch_size:
        return {
            "loss": 0.0,
            "expert_score": 0.0,
            "policy_score": 0.0,
            "separation": 0.0,
        }
    _, _, expert_obs, expert_actions, *_ = expert.sample(batch_size, agent.device)
    _, _, policy_obs, policy_actions, *_ = policy.sample(batch_size, agent.device)
    expert_score = agent.discriminator(expert_obs, expert_actions)
    policy_score = agent.discriminator(policy_obs, policy_actions)
    metrics = discriminator_metrics(expert_score, policy_score)
    # WGAN-GP Wasserstein loss: the critic maximizes expert-vs-policy score
    # separation. Unlike BCE-with-soft-targets, this has no fixed saturation
    # point: while the policy drifts, the critic keeps a gradient for it, so
    # the imitation signal survives the whole run (v7's BCE critic froze at
    # exactly expert=0.8/policy=0.2 after ~100 episodes and the policy then
    # regressed on env reward alone).
    loss = policy_score.mean() - expert_score.mean()

    alpha = torch.rand(batch_size, 1, 1, device=agent.device)
    interpolated_obs = alpha * expert_obs + (1.0 - alpha) * policy_obs
    interpolated_actions = alpha * expert_actions + (1.0 - alpha) * policy_actions
    interpolated_obs.requires_grad_(True)
    interpolated_actions.requires_grad_(True)
    interpolated_score = agent.discriminator(interpolated_obs, interpolated_actions)
    gradients = torch.autograd.grad(
        interpolated_score.sum(),
        (interpolated_obs, interpolated_actions),
        create_graph=True,
    )
    flat_gradient = torch.cat((gradients[0].flatten(1), gradients[1].flatten(1)), dim=1)
    loss += 10.0 * ((flat_gradient.norm(2, dim=1) - 1.0) ** 2).mean()
    agent.discriminator_optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(agent.discriminator.parameters(), 1.0)
    agent.discriminator_optimizer.step()
    metrics["loss"] = float(loss.detach().cpu())
    return metrics


def update_maddpg(agent: IRLAgent, replay: ReplayBuffer, batch_size: int):
    if len(replay) < batch_size:
        return {"actor": 0.0, "critic": 0.0}
    (
        _,
        _,
        observations,
        actions,
        rewards,
        _,
        _,
        next_observations,
        dones,
    ) = replay.sample(batch_size, agent.device)
    gamma = 0.99 ** (agent.config.dt / 0.02)  # per-step 0.99 discount
    with torch.no_grad():
        next_actions = agent.target_actor(next_observations.flatten(0, 1)).view(
            -1, 3, 2
        )
    critic_losses = []
    for index, (critic, target_critic, optimizer) in enumerate(
        zip(agent.critics, agent.target_critics, agent.critic_optimizers)
    ):
        with torch.no_grad():
            target = rewards[:, index] + gamma * (1.0 - dones) * target_critic(
                next_observations, next_actions
            )
        value = critic(observations, actions)
        critic_loss = nn.functional.smooth_l1_loss(value, target)
        optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(critic.parameters(), 5.0)
        optimizer.step()
        critic_losses.append(float(critic_loss.detach().cpu()))

    predicted_actions = agent.actor(observations.flatten(0, 1)).view(-1, 3, 2)
    required = [
        [parameter.requires_grad for parameter in critic.parameters()]
        for critic in agent.critics
    ]
    for critic in agent.critics:
        for parameter in critic.parameters():
            parameter.requires_grad_(False)
    actor_loss = -torch.stack(
        [critic(observations, predicted_actions) for critic in agent.critics]
    ).mean()
    actor_loss += 0.05 * predicted_actions.square().mean()
    agent.actor_optimizer.zero_grad()
    actor_loss.backward()
    nn.utils.clip_grad_norm_(agent.actor.parameters(), 5.0)
    agent.actor_optimizer.step()
    for critic, flags in zip(agent.critics, required):
        for parameter, flag in zip(critic.parameters(), flags):
            parameter.requires_grad_(flag)
    agent.soft_update()
    return {
        "actor": float(actor_loss.detach().cpu()),
        "critic": float(np.mean(critic_losses)),
    }


def discriminator_reward(agent: IRLAgent, observations, actions) -> float:
    with torch.no_grad():
        obs_tensor = torch.as_tensor(
            observations[None], dtype=torch.float32, device=agent.device
        )
        action_tensor = torch.as_tensor(
            actions[None], dtype=torch.float32, device=agent.device
        )
        # WGAN-GP critic score. Unbounded and Lipschitz-smooth, so it keeps a
        # gradient for the policy as long as the policy deviates from expert
        # behavior. (The BCE -log(1-D)-log(2) form froze at a constant -0.47/
        # step once the discriminator saturated, killing imitation entirely.)
        return float(agent.discriminator(obs_tensor, action_tensor)[0])


def train(args: argparse.Namespace) -> Path:
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = TaskConfig()
    apply_environment_args(config, args)
    if args.obstacles_file:
        configure_obstacles(config, args.obstacles_file)
    scene_options = None
    if getattr(args, "scene_preset", None):
        from .scenario import scene_preset

        scene_options = scene_preset(args.scene_preset)
    curriculum = controller_from_args(args, config)
    final_stage = curriculum_stages(config)[-1]
    validation_config = (
        curriculum.validation_config() if curriculum is not None else final_stage.config
    )
    training_config = (
        curriculum.current.config if curriculum is not None else config
    )
    environment = ThreeVOneEnv(
        training_config, seed=args.seed, scene_options=scene_options
    )
    agent = IRLAgent(config, args.device)
    logger = TrainingLogger(args, "no_graph_irl_maddpg", config)
    replay = ReplayBuffer(args.replay_size)
    if args.init_checkpoint:
        data = load_checkpoint(args.init_checkpoint, agent.device)
        agent.actor.load_state_dict(data["actor"])
        for critic, weights in zip(agent.critics, data["critics"]):
            critic.load_state_dict(weights)
        agent.epn.load_state_dict(data["epn"])
        agent.target_actor.load_state_dict(agent.actor.state_dict())
        for target, source in zip(agent.target_critics, agent.critics):
            target.load_state_dict(source.state_dict())
        print(f"initialized from {args.init_checkpoint}")
    expert_replay, expert_holdout, expert_summary = load_expert_replays(
        args.expert_dataset,
        config,
        minimum_episodes=args.min_qualified_expert_episodes,
        holdout_fraction=args.expert_holdout_fraction,
        seed=args.seed,
        max_transitions=args.expert_max_transitions,
        min_separation=args.expert_min_separation,
        min_reward=args.expert_min_reward,
    )
    # NOTE: expert transitions are NOT injected into the online replay. The
    # discriminator distinguishes expert vs policy behavior; the critic must
    # learn the TRUE task value from the policy's own (mixed-reward) data.
    # Mixing expert trajectories (pure env rewards, +capture) with policy
    # trajectories (safety-failure penalties) makes Q oscillate and inflate.
    bc_steps = 0 if args.init_checkpoint else args.bc_steps
    bc_loss = behavior_clone(agent, expert_replay, bc_steps, args.batch_size)
    print(
        f"expert_transitions={len(expert_replay)} "
        f"holdout_transitions={len(expert_holdout)} bc_loss={bc_loss:.5f}"
    )
    logger.scalars(
        {
            "expert/transitions": len(expert_replay),
            "expert/holdout_transitions": len(expert_holdout),
            "expert/episodes": expert_summary.total_episodes,
            "expert/minimum_separation": expert_summary.minimum_separation,
            "loss/behavior_clone": bc_loss,
        },
        0,
    )

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    latest = output / "no_graph_irl_maddpg_latest.pt"
    epn_samples = []
    total_steps = 0
    total_captures = 0
    best_validation = None
    # WGAN-GP critic scores are unbounded and drift (v8: expert=-4/policy=-10.6
    # after a few episodes), which drowned the env reward. Zero-center the IRL
    # reward by subtracting an EMA of the policy's own score so only the
    # relative expert-vs-policy signal remains.
    disc_reward_bias = None
    stage_diagnostics = EpisodeDiagnostics()
    for episode in range(1, args.episodes + 1):
        environment.reset()
        agent.reset_history()
        evader_history = []
        episode_return = 0.0
        episode_learned_reward = 0.0
        episode_mixed_return = 0.0
        last_losses = {"actor": 0.0, "critic": 0.0}
        discriminator = {
            "loss": 0.0,
            "expert_score": 0.0,
            "policy_score": 0.0,
            "separation": 0.0,
        }
        captured = False
        collision_free = True
        obstacle_collisions = 0
        action_magnitudes = []
        noise = 0.0
        robustness = RobustnessMetrics()
        diagnostics = EpisodeDiagnostics()
        episode_stage = (
            curriculum.current if curriculum is not None else final_stage
        )
        for _ in range(args.max_steps):
            positions, velocities = environment.control_state()
            actor_obs, state = current_actor_observations(agent, environment)
            evader_history.append(state)
            with torch.no_grad():
                action = (
                    agent.actor(
                        torch.as_tensor(
                            actor_obs, dtype=torch.float32, device=agent.device
                        )
                    )
                    .cpu()
                    .numpy()
                )
            noise = args.exploration_noise * max(
                0.1, 1.0 - episode / max(args.episodes, 1)
            )
            action = np.clip(
                action + np.random.normal(0.0, noise, action.shape), -1.0, 1.0
            )
            action_magnitudes.append(float(np.linalg.norm(action, axis=1).mean()))
            next_obs, rewards, terminated, truncated, info = environment.step(action)
            obstacle_collisions += int(
                np.count_nonzero(info["obstacle_collisions"][: config.num_pursuers])
            )
            robustness.update(info)
            diagnostics.update(info, action, info["true_positions"])
            stage_diagnostics.update(info, action, info["true_positions"])
            collision_free = bool(
                collision_free
                and info["minimum_inter_drone_distance"]
                >= config.safety_emergency_distance
                and not np.any(info["obstacle_collisions"])
            )
            learned_reward = discriminator_reward(agent, actor_obs, action)
            if disc_reward_bias is None:
                disc_reward_bias = learned_reward
            else:
                # Fast EMA (0.005, time constant ~200 steps): the 0.001 EMA
                # lagged the policy's drift, so early positive learned rewards
                # inflated Q (v8.2: actor loss -27, Q~+27) and the inflated Q
                # then dragged the policy away from expert behavior (radial
                # alignment 0.55 -> 0.046, 0 captures by ep 131). A fast bias
                # keeps the IRL reward centered on the policy's CURRENT level.
                disc_reward_bias = (
                    0.995 * disc_reward_bias + 0.005 * learned_reward
                )
            learned_reward -= disc_reward_bias
            mixed_rewards = rewards + args.irl_lambda * learned_reward
            next_actor_obs = zero_prediction_observations(next_obs, config)
            next_positions, next_velocities = environment.control_state()
            replay.add(
                positions,
                velocities,
                actor_obs,
                action,
                mixed_rewards,
                next_positions,
                next_velocities,
                next_actor_obs,
                np.float32(terminated or truncated),
            )
            total_steps += 1
            episode_return += float(np.mean(rewards))
            episode_learned_reward += learned_reward
            episode_mixed_return += float(np.mean(mixed_rewards))
            if total_steps >= args.warmup_steps:
                last_losses = update_maddpg(agent, replay, args.batch_size)
                if curriculum is not None:
                    curriculum.record_policy_update()
                # Discriminator updates slower than the policy: a per-step
                # critic learns faster than the actor (v8.x separation ~7-8
                # from ep ~50) and dominates the game, beating the policy off
                # its BC-initialized expert behavior (radial alignment
                # 0.6 -> 0.05, 0 captures). Every K steps gives the policy
                # room to explore while the imitation signal stays alive.
                if (
                    args.disc_update_interval > 0
                    and total_steps % args.disc_update_interval == 0
                ):
                    discriminator = update_discriminator(
                        agent, expert_replay, replay, args.batch_size
                    )
            if terminated or truncated:
                captured = bool(info["captured"])
                break
        total_captures += int(captured)
        # BC anchor: pull the policy back onto the expert distribution after
        # each episode. v8.1-v8.3 all regressed from BC (radial 0.6 -> 0.05)
        # under pure RL/IRL updates; re-anchoring on filtered expert
        # transitions implements the expert-experience-first principle and
        # stops the drift that destroys capture behavior.
        if args.bc_anchor_steps > 0:
            behavior_clone(agent, expert_replay, args.bc_anchor_steps, args.batch_size)
        if curriculum is not None:
            previous_stage = curriculum.current.index
            stage = curriculum.update(captured, collision_free)
            curriculum_capture, curriculum_safety = curriculum.rates()
            if stage.index != previous_stage:
                environment = ThreeVOneEnv(
                    stage.config,
                    seed=args.seed + episode,
                    scene_options=scene_options,
                )
                stage_diagnostics.reset()
        else:
            stage = None
            curriculum_capture = float(captured)
            curriculum_safety = float(collision_free)
        if curriculum is not None and curriculum.stage_zero_stalled():
            raise RuntimeError(
                stage_zero_stall_message(
                    stage_diagnostics.tensorboard(curriculum.current)
                )
            )
        epn_samples.extend(make_epn_samples(evader_history, config))
        epn_samples = epn_samples[-args.epn_replay_size:]
        epn_loss = train_epn(agent, epn_samples)
        agent.save(str(latest), {"episode": episode, "total_steps": total_steps})
        if args.validation_interval > 0 and episode % args.validation_interval == 0:
            validation = validate_checkpoint(
                "no_graph_irl_maddpg",
                latest,
                validation_config,
                args.validation_episodes,
                args.device,
                scene_options=scene_options,
            )
            logger.scalars(validation.tensorboard(), episode)
            if checkpoint_is_better(validation, best_validation):
                best = output / "no_graph_irl_maddpg_best_safe.pt"
                agent.save(
                    str(best),
                    {"episode": episode, "validation": validation.__dict__},
                )
                best_validation = validation
        logger.scalars(
            {
                "episode/environment_return": episode_return,
                "episode/learned_reward": episode_learned_reward,
                "episode/mixed_return": episode_mixed_return,
                "episode/length": environment.step_count,
                "episode/captured": int(captured),
                "episode/obstacle_collisions": obstacle_collisions,
                "metrics/cumulative_capture_rate": total_captures / episode,
                "curriculum/stage": (
                    stage.index if stage is not None else 5
                ),
                "curriculum/capture_rate": curriculum_capture,
                "curriculum/collision_free_rate": curriculum_safety,
                "loss/actor": last_losses["actor"],
                "loss/critic": last_losses["critic"],
                "loss/discriminator": discriminator["loss"],
                "discriminator/expert_score": (
                    discriminator["expert_score"]
                ),
                "discriminator/policy_score": (
                    discriminator["policy_score"]
                ),
                "discriminator/separation": discriminator["separation"],
                "buffer/expert_replay_fraction": min(
                    len(expert_replay) / max(len(replay), 1), 1.0
                ),
                "loss/epn": epn_loss,
                "exploration/noise_std": noise,
                "policy/mean_action_magnitude": (
                    float(np.mean(action_magnitudes)) if action_magnitudes else 0.0
                ),
                "buffer/replay_size": len(replay),
                "buffer/epn_samples": len(epn_samples),
                "progress/environment_steps": total_steps,
                **robustness.tensorboard(),
                **diagnostics.tensorboard(episode_stage),
            },
            episode,
        )
        logger.flush()
        print(
            f"episode={episode}/{args.episodes} return={episode_return:.3f} "
            f"captured={captured} actor={last_losses['actor']:.4f} "
            f"critic={last_losses['critic']:.4f} "
            f"disc={discriminator['loss']:.4f} epn={epn_loss:.4f}"
        )
    logger.close()
    return latest


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=1750)
    parser.add_argument("--bc-steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--replay-size", type=int, default=200000)
    parser.add_argument("--epn-replay-size", type=int, default=20000)
    parser.add_argument("--exploration-noise", type=float, default=0.15)
    parser.add_argument("--irl-lambda", type=float, default=0.3)
    parser.add_argument("--disc-update-interval", type=int, default=1)
    parser.add_argument("--bc-anchor-steps", type=int, default=0)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--obstacles-file", default=None)
    parser.add_argument("--scene-preset", default=None)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", default="checkpoints/no_graph_irl")
    add_expert_dataset_args(parser)
    add_curriculum_args(parser)
    add_validation_args(parser)
    add_tensorboard_args(parser)
    add_environment_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
