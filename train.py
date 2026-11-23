"""Retrain the migrated 3V1 MAPPO + EPN baseline at 50 Hz."""

import argparse
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn

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
from .model import MAPPOAgent
from .robustness_metrics import RobustnessMetrics
from .tensorboard_logging import TrainingLogger, add_tensorboard_args
from .training_diagnostics import EpisodeDiagnostics
from .training_validation import (
    add_validation_args,
    checkpoint_is_better,
    validate_checkpoint,
)


def make_epn_samples(
    states: Sequence[np.ndarray], config: TaskConfig
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create history/future pairs without crossing an episode boundary."""
    samples = []
    last_offset = config.prediction_stride * config.prediction_steps
    for now in range(config.history_steps - 1, len(states) - last_offset):
        history = np.asarray(
            states[now - config.history_steps + 1: now + 1], dtype=np.float32
        )
        target_indices = [
            now + config.prediction_stride * (i + 1)
            for i in range(config.prediction_steps)
        ]
        target = np.asarray(
            [states[i][:2] for i in target_indices], dtype=np.float32
        )
        samples.append((history, target))
    return samples


def train_epn(
    agent: MAPPOAgent,
    samples: List[Tuple[np.ndarray, np.ndarray]],
    batch_size: int = 128,
) -> float:
    if not samples:
        return 0.0
    random.shuffle(samples)
    losses = []
    for start in range(0, len(samples), batch_size):
        batch = samples[start: start + batch_size]
        history = torch.as_tensor(
            np.stack([item[0] for item in batch]),
            dtype=torch.float32,
            device=agent.device,
        )
        target = torch.as_tensor(
            np.stack([item[1] for item in batch]),
            dtype=torch.float32,
            device=agent.device,
        )
        prediction, confidence = agent.epn(history)
        prediction_loss = nn.functional.smooth_l1_loss(prediction, target)
        with torch.no_grad():
            error = torch.sqrt(
                ((prediction - target) ** 2).mean(dim=(1, 2)) + 1e-8
            )
            confidence_target = torch.exp(-4.0 * error).unsqueeze(1)
        confidence_loss = nn.functional.mse_loss(confidence, confidence_target)
        loss = prediction_loss + 0.1 * confidence_loss
        agent.epn_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.epn.parameters(), 5.0)
        agent.epn_optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def ppo_update(
    agent: MAPPOAgent,
    rollout: Dict[str, np.ndarray],
    gamma: float,
    gae_lambda: float,
    epochs: int = 6,
    batch_size: int = 256,
) -> Dict[str, float]:
    for name in ("rewards", "dones", "values", "actor_obs", "actions"):
        if not np.all(np.isfinite(rollout[name])):
            raise ValueError(f"MAPPO rollout {name} contains non-finite values")
    if not np.isfinite(rollout["bootstrap"]):
        raise ValueError("MAPPO rollout bootstrap is non-finite")
    rewards = rollout["rewards"]
    dones = rollout["dones"]
    values = rollout["values"]
    bootstrap = float(rollout["bootstrap"])
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_value = bootstrap if t == len(rewards) - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        gae = delta + gamma * gae_lambda * nonterminal * gae
        advantages[t] = gae
    returns = advantages + values
    return_mean = float(returns.mean())
    return_std = float(returns.std())
    returns = (returns - return_mean) / (return_std + 1e-8)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    advantage_mean = float(advantages.mean())
    advantage_std = float(advantages.std())
    value_target_mean = float(returns.mean())

    actor_obs = torch.as_tensor(
        rollout["actor_obs"].reshape(-1, agent.actor_obs_dim),
        dtype=torch.float32,
        device=agent.device,
    )
    actions = torch.as_tensor(
        rollout["actions"].reshape(-1, 2),
        dtype=torch.float32,
        device=agent.device,
    )
    old_log_probs = torch.as_tensor(
        rollout["log_probs"].reshape(-1),
        dtype=torch.float32,
        device=agent.device,
    )
    actor_advantages = torch.as_tensor(
        np.repeat(advantages, agent.config.num_pursuers),
        dtype=torch.float32,
        device=agent.device,
    )
    global_obs = torch.as_tensor(
        rollout["actor_obs"].reshape(len(rewards), -1),
        dtype=torch.float32,
        device=agent.device,
    )
    target_returns = torch.as_tensor(
        returns, dtype=torch.float32, device=agent.device
    )

    actor_losses, critic_losses, actor_grad_norms, critic_grad_norms = [], [], [], []
    actor_indices = np.arange(len(actor_obs))
    critic_indices = np.arange(len(global_obs))
    for _ in range(epochs):
        np.random.shuffle(actor_indices)
        for start in range(0, len(actor_indices), batch_size):
            idx = actor_indices[start: start + batch_size]
            dist = agent.actor.distribution(actor_obs[idx])
            log_prob = dist.log_prob(actions[idx]).sum(-1)
            ratio = torch.exp(log_prob - old_log_probs[idx])
            unclipped = ratio * actor_advantages[idx]
            clipped = torch.clamp(ratio, 0.8, 1.2) * actor_advantages[idx]
            actor_loss = (
                -torch.min(unclipped, clipped).mean()
                - 0.01 * dist.entropy().sum(-1).mean()
            )
            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_norm = nn.utils.clip_grad_norm_(agent.actor.parameters(), 5.0)
            actor_grad_norms.append(
                min(float(actor_norm.detach().cpu()), 5.0)
            )
            agent.actor_optimizer.step()
            actor_losses.append(float(actor_loss.detach().cpu()))

        np.random.shuffle(critic_indices)
        for start in range(0, len(critic_indices), batch_size):
            idx = critic_indices[start: start + batch_size]
            value = agent.critic(global_obs[idx])
            critic_loss = nn.functional.smooth_l1_loss(
                value, target_returns[idx]
            )
            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_norm = nn.utils.clip_grad_norm_(agent.critic.parameters(), 5.0)
            critic_grad_norms.append(
                min(float(critic_norm.detach().cpu()), 5.0)
            )
            agent.critic_optimizer.step()
            critic_losses.append(float(critic_loss.detach().cpu()))
    return {
        "actor_loss": float(np.mean(actor_losses)),
        "critic_loss": float(np.mean(critic_losses)),
        "advantage_mean": advantage_mean,
        "advantage_std": advantage_std,
        "value_target_mean": value_target_mean,
        "actor_grad_norm": float(np.mean(actor_grad_norms)),
        "critic_grad_norm": float(np.mean(critic_grad_norms)),
    }


def train(args: argparse.Namespace) -> Path:
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = TaskConfig()
    apply_environment_args(config, args)
    if args.obstacles_file:
        configure_obstacles(config, args.obstacles_file)
    curriculum = controller_from_args(args, config)
    final_stage = curriculum_stages(config)[-1]
    validation_config = (
        curriculum.validation_config() if curriculum is not None else final_stage.config
    )
    training_config = (
        curriculum.current.config if curriculum is not None else config
    )
    env = ThreeVOneEnv(training_config, seed=args.seed)
    agent = MAPPOAgent(config, device=args.device)
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    logger = TrainingLogger(args, "mappo_open", config)

    # Preserve the original baseline's physical discount horizon after changing
    # its 0.5 s decision step to the deployment period of 0.02 s.
    gamma = 0.99 ** (config.dt / 0.5)
    gae_lambda = 0.95 ** (config.dt / 0.5)
    observations, _ = env.reset()
    agent.reset_history()
    episode_states: List[np.ndarray] = []
    epn_replay: List[Tuple[np.ndarray, np.ndarray]] = []
    episode_return = 0.0
    episode_obstacle_collisions = 0
    episode_collision_free = True
    completed_episodes = 0
    total_steps = 0
    episode_robustness = RobustnessMetrics()
    episode_diagnostics = EpisodeDiagnostics()
    stage_diagnostics = EpisodeDiagnostics()
    episode_stage = curriculum.current if curriculum is not None else final_stage
    best_validation = None

    for iteration in range(1, args.iterations + 1):
        storage = {
            key: []
            for key in (
                "actor_obs",
                "actions",
                "log_probs",
                "rewards",
                "values",
                "dones",
            )
        }
        captures = 0
        episodes_this_rollout = 0
        action_magnitudes = []
        confidences = []
        rollout_robustness = RobustnessMetrics()
        for _ in range(args.rollout_steps):
            evader_state = env.evader_state()
            episode_states.append(evader_state.copy())
            agent.update_history(evader_state)
            prediction, confidence = agent.predict_evader()
            actor_obs = agent.actor_observations(
                observations, prediction, confidence
            )
            actor_tensor = torch.as_tensor(
                actor_obs, dtype=torch.float32, device=agent.device
            )
            global_tensor = actor_tensor.reshape(1, -1)
            with torch.no_grad():
                action, log_prob = agent.actor.act(actor_tensor)
                value = agent.critic(global_tensor)[0]
            next_obs, rewards, terminated, truncated, info = env.step(
                action.cpu().numpy()
            )
            total_steps += 1
            action_magnitudes.append(
                float(np.linalg.norm(action.cpu().numpy(), axis=1).mean())
            )
            confidences.append(float(np.mean(confidence)))
            episode_obstacle_collisions += int(
                np.count_nonzero(
                    info["obstacle_collisions"][: config.num_pursuers]
                )
            )
            episode_robustness.update(info)
            rollout_robustness.update(info)
            episode_diagnostics.update(
                info,
                action.cpu().numpy(),
                info["true_positions"],
            )
            stage_diagnostics.update(
                info,
                action.cpu().numpy(),
                info["true_positions"],
            )
            episode_collision_free = bool(
                episode_collision_free
                and info["minimum_inter_drone_distance"]
                >= config.safety_emergency_distance
                and not np.any(info["obstacle_collisions"])
            )
            done = terminated or truncated
            storage["actor_obs"].append(actor_obs)
            storage["actions"].append(action.cpu().numpy())
            storage["log_probs"].append(log_prob.cpu().numpy())
            storage["rewards"].append(float(np.mean(rewards)))
            storage["values"].append(float(value.cpu()))
            storage["dones"].append(float(done))
            episode_return += float(np.mean(rewards))
            observations = next_obs
            if done:
                captures += int(info["captured"])
                completed_episodes += 1
                episodes_this_rollout += 1
                epn_replay.extend(make_epn_samples(episode_states, config))
                epn_replay = epn_replay[-args.epn_replay_size:]
                if curriculum is not None:
                    previous_stage = curriculum.current.index
                    stage = curriculum.update(
                        bool(info["captured"]), episode_collision_free
                    )
                    curriculum_capture, curriculum_safety = curriculum.rates()
                    if stage.index != previous_stage:
                        env = ThreeVOneEnv(
                            stage.config,
                            seed=args.seed + completed_episodes,
                        )
                        stage_diagnostics.reset()
                else:
                    stage = None
                    curriculum_capture = float(info["captured"])
                    curriculum_safety = float(episode_collision_free)
                logger.scalars(
                    {
                        "episode/return": episode_return,
                        "episode/length": env.step_count,
                        "episode/captured": int(info["captured"]),
                        "episode/obstacle_collisions": episode_obstacle_collisions,
                        "curriculum/stage": (
                            stage.index if stage is not None else 5
                        ),
                        "curriculum/capture_rate": curriculum_capture,
                        "curriculum/collision_free_rate": curriculum_safety,
                        **episode_robustness.tensorboard(),
                        **episode_diagnostics.tensorboard(episode_stage),
                    },
                    completed_episodes,
                )
                print(
                    f"episode={completed_episodes} "
                    f"return={episode_return:.3f} "
                    f"captured={bool(info['captured'])} "
                    f"steps={env.step_count}"
                )
                observations, _ = env.reset()
                agent.reset_history()
                episode_states = []
                episode_return = 0.0
                episode_obstacle_collisions = 0
                episode_collision_free = True
                episode_robustness = RobustnessMetrics()
                episode_diagnostics.reset()
                episode_stage = (
                    curriculum.current if curriculum is not None else final_stage
                )

        current_prediction, current_confidence = agent.predict_evader()
        current_actor_obs = agent.actor_observations(
            observations, current_prediction, current_confidence
        )
        with torch.no_grad():
            bootstrap = float(
                agent.critic(
                    torch.as_tensor(
                        current_actor_obs.reshape(1, -1),
                        dtype=torch.float32,
                        device=agent.device,
                    )
                )[0].cpu()
            )
        rollout = {key: np.asarray(value) for key, value in storage.items()}
        rollout["bootstrap"] = bootstrap
        losses = ppo_update(agent, rollout, gamma, gae_lambda)
        if curriculum is not None:
            curriculum.record_policy_update()
            if curriculum.stage_zero_stalled(
                getattr(args, "curriculum_stall_updates", 50)
            ):
                raise RuntimeError(
                    stage_zero_stall_message(
                        stage_diagnostics.tensorboard(curriculum.current)
                    )
                )
        epn_loss = train_epn(agent, epn_replay)
        latest = output / "mappo_epn_3v1_latest.pt"
        agent.save(latest, {"iteration": iteration, "gamma": gamma})
        if args.validation_interval > 0 and iteration % args.validation_interval == 0:
            validation = validate_checkpoint(
                "mappo_open",
                latest,
                validation_config,
                args.validation_episodes,
                args.device,
            )
            logger.scalars(validation.tensorboard(), iteration)
            if checkpoint_is_better(validation, best_validation):
                best = output / "mappo_epn_3v1_best_safe.pt"
                agent.save(
                    best,
                    {"iteration": iteration, "validation": validation.__dict__},
                )
                best_validation = validation
        logger.scalars(
            {
                "loss/actor": losses["actor_loss"],
                "loss/critic": losses["critic_loss"],
                "loss/epn": epn_loss,
                "diagnostics/advantage_mean": losses["advantage_mean"],
                "diagnostics/advantage_std": losses["advantage_std"],
                "diagnostics/value_target_mean": (
                    losses["value_target_mean"]
                ),
                "diagnostics/actor_grad_norm": losses["actor_grad_norm"],
                "diagnostics/critic_grad_norm": (
                    losses["critic_grad_norm"]
                ),
                "rollout/mean_reward": float(np.mean(rollout["rewards"])),
                "rollout/captures": captures,
                "rollout/episodes": episodes_this_rollout,
                "rollout/capture_rate": captures / max(episodes_this_rollout, 1),
                "policy/mean_action_magnitude": np.mean(action_magnitudes),
                "prediction/mean_confidence": np.mean(confidences),
                "buffer/epn_samples": len(epn_replay),
                "progress/environment_steps": total_steps,
                **rollout_robustness.tensorboard(),
            },
            iteration,
        )
        logger.flush()
        print(
            f"iteration={iteration}/{args.iterations} "
            f"actor={losses['actor_loss']:.4f} "
            f"critic={losses['critic_loss']:.4f} epn={epn_loss:.4f} "
            f"captures={captures} checkpoint={latest}"
        )
    logger.close()
    return latest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--epn-replay-size", type=int, default=20000)
    parser.add_argument("--obstacles-file", default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", default="checkpoints/three_v_one")
    add_curriculum_args(parser)
    add_validation_args(parser)
    add_tensorboard_args(parser)
    add_environment_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
