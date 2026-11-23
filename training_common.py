"""Expert-data and EPN helpers shared by the additional baselines."""

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from .config import TaskConfig
from .config import CrazyfliePhysics
from .additional_models import ReplayBuffer
from .environment import ThreeVOneEnv
from .expert_dataset import filter_episodes, load_expert_dataset
from .features import OBS_DIM, build_observations
from .iqld_replay import EpisodeBalancedReplay
from .paper_game_theory import GameTheoryRuntime


DEFAULT_EXPERT_DATASET = (
    "checkpoints/three_v_one_expert/final_robust_safe_expert.pt"
)


def safety_penalty(
    positions: np.ndarray, config: TaskConfig, scale: float, dt: float
) -> np.ndarray:
    """Training-only quadratic inter-drone safety penalty (3,).

    The environment's linear safety-warning penalty (rate 2.0) is dominated by
    the +10 capture reward, so learned formations drift to ~0.29-0.30 m spacing.
    This steeper quadratic penalty (default 0 = disabled) teaches the Q-function
    to avoid tight formations without touching TaskConfig (which would
    invalidate the expert dataset fingerprint).
    """
    if scale <= 0.0:
        return np.zeros(3, dtype=np.float32)
    delta = positions[:, None, :] - positions[None, :, :]
    distances = np.linalg.norm(delta, axis=-1)
    row, column = np.triu_indices(positions.shape[0], k=1)
    urgency = np.clip(
        (config.safety_warning_distance - distances[row, column])
        / (
            config.safety_warning_distance
            - config.safety_emergency_distance
        ),
        0.0,
        1.0,
    )
    penalty = float(-scale * (urgency * urgency).sum() * dt)
    return np.full(3, penalty, dtype=np.float32)


def capture_margin_reward(
    positions: np.ndarray, config: TaskConfig, bonus: float, dt: float
) -> np.ndarray:
    """Training-only reward for enclosing the evader at a safe spacing (3,).

    The env's capture-candidate reward is active exactly when the pursuers form
    a tight triangle around the evader, which pushes inter-drone spacing toward
    the 0.30 m safety boundary. This bonus (default 0 = disabled) rewards the
    close-encirclement phase (max pursuer-evader distance within capture_radius)
    proportionally to inter-drone margin above the emergency distance, so the
    Q-function prefers the expert's looser 0.35-0.40 m formations over the
    boundary-risky 0.30-0.33 m ones.
    """
    if bonus <= 0.0:
        return np.zeros(3, dtype=np.float32)
    pursuers = positions[:3]
    evader = positions[3]
    distances = np.linalg.norm(pursuers - evader, axis=1)
    if float(np.max(distances)) > config.capture_radius:
        return np.zeros(3, dtype=np.float32)
    delta = positions[:3, None, :] - positions[None, :3, :]
    pairwise = np.linalg.norm(delta, axis=-1)
    row, column = np.triu_indices(3, k=1)
    minimum = float(pairwise[row, column].min())
    margin = np.clip(
        (minimum - config.safety_emergency_distance) / 0.10, 0.0, 1.0
    )
    return np.full(
        3, float(bonus * margin * dt), dtype=np.float32
    )


def capture_balance_penalty(
    positions: np.ndarray, config: TaskConfig, scale: float, dt: float
) -> np.ndarray:
    """Training-only penalty for unbalanced encirclement (3,).

    Replay diagnosis shows failures happen when ONE pursuer over-commits to the
    evader (0.30-0.31 m) while teammates lag (0.5-1.6 m), bunching the near
    pursuers below the 0.30 m inter-drone floor. This penalty (default 0 =
    disabled) fires during the close-encirclement phase (all pursuers within
    1.5 * capture_radius of the evader) and grows with the pursuer-evader
    distance spread (max - min), so balanced triangles are worth more than
    over-committed ones.
    """
    if scale <= 0.0:
        return np.zeros(3, dtype=np.float32)
    distances = np.linalg.norm(positions[:3] - positions[3], axis=1)
    if float(np.max(distances)) > 1.5 * config.capture_radius:
        return np.zeros(3, dtype=np.float32)
    spread = float(np.max(distances) - np.min(distances))
    penalty = float(-scale * np.clip(spread / 0.50, 0.0, 1.0) * dt)
    return np.full(3, penalty, dtype=np.float32)


@dataclass(frozen=True)
class ExpertReplaySummary:
    """Episode and transition counts for one deterministic cache split."""

    total_episodes: int
    training_episodes: int
    holdout_episodes: int
    training_transitions: int
    holdout_transitions: int
    filtered_transitions: int
    minimum_separation: float
    task_fingerprint: str


def _expert_min_reward_arg(value: str):
    """Parse --expert-min-reward; 'none' disables the transition filter."""
    if value in ("none", "None", "disable"):
        return None
    return float(value)


def add_expert_dataset_args(parser: argparse.ArgumentParser) -> None:
    """Add persistent expert-cache arguments shared by IQLD and IRL."""
    parser.add_argument("--expert-dataset", default=DEFAULT_EXPERT_DATASET)
    parser.add_argument("--min-qualified-expert-episodes", type=int, default=20)
    parser.add_argument("--expert-holdout-fraction", type=float, default=0.20)
    parser.add_argument(
        "--expert-max-transitions", type=int, default=1000,
        help="drop expert episodes longer than this (slow captures contain "
             "low-value wandering transitions); 0 disables",
    )
    parser.add_argument(
        "--expert-min-separation", type=float, default=0.32,
        help="drop expert episodes whose minimum separation is below this "
             "(marginal episodes teach boundary-risky actions); 0 disables",
    )
    parser.add_argument(
        "--expert-min-reward", type=_expert_min_reward_arg, default=0.0,
        help="drop expert transitions whose mean reward is below this "
             "(no-progress/penalty samples dilute encirclement behavior); "
             "'none' keeps all transitions including safety-warning examples",
    )


def _collection_command(path: Path, minimum_episodes: int) -> str:
    return (
        "PYTHONPATH=. python -m "
        "crazyflie_examples.three_v_one_baseline.evaluate_game_theory "
        "--collect-qualified-expert "
        f"--expert-dataset {path} "
        f"--target-qualified-episodes {minimum_episodes} "
        f"--max-collection-attempts {max(100, minimum_episodes * 5)}"
    )


def load_expert_replays(
    path,
    config: TaskConfig,
    *,
    minimum_episodes: int = 20,
    holdout_fraction: float = 0.20,
    seed: int = 0,
    max_transitions: Optional[int] = 1000,
    min_separation: Optional[float] = 0.32,
    min_reward: Optional[float] = 0.0,
    safety_penalty_scale: float = 0.0,
    capture_margin_bonus: float = 0.0,
    capture_balance_penalty_scale: float = 0.0,
    episode_balanced: bool = False,
):
    """Load a qualified cache and split it by complete episode."""
    path = Path(path).expanduser()
    command = _collection_command(path, minimum_episodes)
    if not path.is_file():
        raise FileNotFoundError(
            f"qualified expert dataset is missing: {path}\nCreate it with:\n{command}"
        )
    try:
        dataset = load_expert_dataset(
            path,
            expected_config=config,
            expected_physics=CrazyfliePhysics(),
        )
    except ValueError as error:
        raise ValueError(
            f"{error}\nRebuild a compatible dataset with:\n{command}"
        ) from error
    if max_transitions or min_separation:
        dataset = filter_episodes(
            dataset,
            max_transitions=max_transitions or None,
            min_separation=min_separation or None,
        )
    if len(dataset.episodes) < minimum_episodes:
        raise ValueError(
            f"expert dataset has {len(dataset.episodes)} qualified episodes "
            f"after quality filtering; {minimum_episodes} required\n"
            f"Extend it with:\n{command}"
        )
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("expert holdout fraction must lie in (0, 1)")
    holdout_count = max(1, round(len(dataset.episodes) * holdout_fraction))
    if holdout_count >= len(dataset.episodes):
        raise ValueError("expert cache must leave at least one training episode")
    indices = np.random.default_rng(seed).permutation(len(dataset.episodes))
    holdout_indices = set(indices[:holdout_count].tolist())
    replay_type = EpisodeBalancedReplay if episode_balanced else ReplayBuffer
    train_replay = replay_type()
    holdout_replay = replay_type()
    filtered_transitions = 0
    for index, episode in enumerate(dataset.episodes):
        replay = holdout_replay if index in holdout_indices else train_replay
        prepared = []
        for transition in episode.transitions:
            reward = float(np.mean(np.asarray(transition[4])))
            if min_reward is not None and reward < min_reward:
                filtered_transitions += 1
                continue
            rewards = np.asarray(transition[4], dtype=np.float32)
            if safety_penalty_scale or capture_margin_bonus or capture_balance_penalty_scale:
                positions = np.asarray(transition[0], dtype=np.float32)
                rewards = rewards + safety_penalty(
                    positions, config, safety_penalty_scale, config.dt
                )
                rewards = rewards + capture_margin_reward(
                    positions, config, capture_margin_bonus, config.dt
                )
                rewards = rewards + capture_balance_penalty(
                    positions, config, capture_balance_penalty_scale, config.dt
                )
            prepared.append((
                transition[0],
                transition[1],
                transition[2],
                transition[3],
                rewards,
                transition[5],
                transition[6],
                transition[7],
                transition[8],
            ))
        if episode_balanced:
            replay.add_episode(prepared)
        else:
            for transition in prepared:
                replay.add(*transition)
    summary = ExpertReplaySummary(
        total_episodes=len(dataset.episodes),
        training_episodes=len(dataset.episodes) - holdout_count,
        holdout_episodes=holdout_count,
        training_transitions=len(train_replay),
        holdout_transitions=len(holdout_replay),
        filtered_transitions=filtered_transitions,
        minimum_separation=min(
            episode.minimum_separation for episode in dataset.episodes
        ),
        task_fingerprint=dataset.task_fingerprint,
    )
    return train_replay, holdout_replay, summary


def load_epn_augmented_expert_replays(
    agent,
    path,
    config: TaskConfig,
    *,
    minimum_episodes: int = 20,
    holdout_fraction: float = 0.20,
    seed: int = 0,
    max_transitions: Optional[int] = 1000,
    min_separation: Optional[float] = 0.32,
    min_reward: Optional[float] = 0.0,
    epn_epochs: int = 5,
    safety_penalty_scale: float = 0.0,
    capture_margin_bonus: float = 0.0,
    capture_balance_penalty_scale: float = 0.0,
    episode_balanced: bool = False,
) -> Tuple[ReplayBuffer, ReplayBuffer, ExpertReplaySummary]:
    """Load expert data whose observations carry real EPN predictions.

    The stored expert observations zero out the EPN prediction block (the
    collector cannot run a trained EPN). Training on zeroed predictions while
    evaluating with real predictions is a train/inference mismatch; this loader
    trains the agent's EPN on the expert evader trajectories first, then
    rebuilds every transition's observation with the trained EPN's prediction
    so behavior-cloning and RL see the same features the runtime will use.
    """
    path = Path(path).expanduser()
    dataset = load_expert_dataset(
        path,
        expected_config=config,
        expected_physics=CrazyfliePhysics(),
    )
    if max_transitions or min_separation:
        dataset = filter_episodes(
            dataset,
            max_transitions=max_transitions or None,
            min_separation=min_separation or None,
        )
    if len(dataset.episodes) < minimum_episodes:
        raise ValueError(
            f"expert dataset has {len(dataset.episodes)} qualified episodes "
            f"after quality filtering; {minimum_episodes} required"
        )
    # 1) train the EPN on the expert evader trajectories
    epn_samples: List[Tuple[np.ndarray, np.ndarray]] = []
    for episode in dataset.episodes:
        states = []
        for transition in episode.transitions:
            positions, velocities = transition[0], transition[1]
            states.append(
                np.concatenate(
                    (
                        positions[3] / config.arena_half_extent,
                        velocities[3] / config.velocity_scale,
                    )
                ).astype(np.float32)
            )
        epn_samples.extend(make_epn_samples(states, config))
    for _ in range(epn_epochs):
        train_epn(agent, epn_samples)
    # 2) rebuild observations with real predictions and split by episode
    indices = np.random.default_rng(seed).permutation(len(dataset.episodes))
    holdout_count = max(1, round(len(dataset.episodes) * holdout_fraction))
    if holdout_count >= len(dataset.episodes):
        raise ValueError("expert cache must leave at least one training episode")
    holdout_indices = set(indices[:holdout_count].tolist())
    replay_type = EpisodeBalancedReplay if episode_balanced else ReplayBuffer
    train_replay = replay_type()
    holdout_replay = replay_type()
    filtered_transitions = 0
    for index, episode in enumerate(dataset.episodes):
        replay = holdout_replay if index in holdout_indices else train_replay
        states = []
        for transition in episode.transitions:
            positions, velocities = transition[0], transition[1]
            states.append(
                np.concatenate(
                    (
                        positions[3] / config.arena_half_extent,
                        velocities[3] / config.velocity_scale,
                    )
                ).astype(np.float32)
            )
        prepared = []
        for step, transition in enumerate(episode.transitions):
            reward = float(np.mean(np.asarray(transition[4])))
            if min_reward is not None and reward < min_reward:
                filtered_transitions += 1
                continue
            prediction, confidence = _epn_prediction_at(
                agent, states, step, config
            )
            positions, velocities = transition[0], transition[1]
            observations = expert_base_observations(transition, 2, config)
            next_positions, next_velocities = transition[5], transition[6]
            next_observations = expert_base_observations(
                transition, 7, config
            )
            actor_obs = agent.augment(observations, prediction, confidence)
            next_obs = agent.augment(next_observations, prediction, confidence)
            rewards = np.asarray(transition[4], dtype=np.float32)
            if safety_penalty_scale or capture_margin_bonus or capture_balance_penalty_scale:
                rewards = rewards + safety_penalty(
                    positions, config, safety_penalty_scale, config.dt
                )
                rewards = rewards + capture_margin_reward(
                    positions, config, capture_margin_bonus, config.dt
                )
                rewards = rewards + capture_balance_penalty(
                    positions, config, capture_balance_penalty_scale, config.dt
                )
            prepared.append((
                positions,
                velocities,
                actor_obs,
                transition[3],
                rewards,
                next_positions,
                next_velocities,
                next_obs,
                transition[8],
            ))
        if episode_balanced:
            replay.add_episode(prepared)
        else:
            for transition in prepared:
                replay.add(*transition)
    summary = ExpertReplaySummary(
        total_episodes=len(dataset.episodes),
        training_episodes=len(dataset.episodes) - holdout_count,
        holdout_episodes=holdout_count,
        training_transitions=len(train_replay),
        holdout_transitions=len(holdout_replay),
        filtered_transitions=filtered_transitions,
        minimum_separation=min(
            episode.minimum_separation for episode in dataset.episodes
        ),
        task_fingerprint=dataset.task_fingerprint,
    )
    return train_replay, holdout_replay, summary


def expert_base_observations(transition, column: int, config: TaskConfig):
    """Recover stored base observations, preserving per-episode obstacles."""
    stored = np.asarray(transition[column], dtype=np.float32)
    expected_rows = config.num_pursuers
    if stored.ndim != 2 or stored.shape[0] != expected_rows:
        raise ValueError(
            "stored expert observations must have one row per pursuer"
        )
    if stored.shape[1] < OBS_DIM:
        raise ValueError("stored expert observations are missing base features")
    return stored[:, :OBS_DIM].copy()


def _epn_prediction_at(agent, states, step: int, config: TaskConfig):
    """EPN prediction for the history window ending at ``step``."""
    lo = max(0, step - config.history_steps + 1)
    history = states[lo:step + 1]
    if len(history) < 3:
        return np.zeros((config.prediction_steps, 2), np.float32), 0.0
    sequence = torch.as_tensor(
        np.asarray(history, dtype=np.float32), dtype=torch.float32,
        device=agent.device,
    ).unsqueeze(0)
    with torch.no_grad():
        prediction, confidence = agent.epn(sequence)
    return prediction[0].cpu().numpy(), float(confidence[0, 0].cpu())


@dataclass(frozen=True)
class ExpertQuality:
    """Capture and safety summary for one generated expert dataset."""

    episodes: int
    captures: int
    collision_free_episodes: int
    minimum_separation: float

    @property
    def capture_rate(self) -> float:
        return self.captures / self.episodes if self.episodes else 0.0

    @property
    def collision_free_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return self.collision_free_episodes / self.episodes


def require_expert_quality(
    quality: ExpertQuality,
    min_capture_rate: float,
    min_collision_free_rate: float,
    minimum_separation: float = 0.30,
) -> ExpertQuality:
    """Reject expert experience that cannot support safe imitation."""
    if quality.episodes <= 0:
        raise ValueError("expert quality requires at least one episode")
    if quality.capture_rate < min_capture_rate:
        raise ValueError(
            "expert capture rate "
            f"{quality.capture_rate:.1%} is below {min_capture_rate:.1%}"
        )
    if quality.collision_free_rate < min_collision_free_rate:
        raise ValueError(
            "expert collision-free rate "
            f"{quality.collision_free_rate:.1%} is below "
            f"{min_collision_free_rate:.1%}"
        )
    if quality.minimum_separation < minimum_separation:
        raise ValueError(
            "expert minimum separation "
            f"{quality.minimum_separation:.3f} m is below "
            f"{minimum_separation:.3f} m"
        )
    return quality


def zero_prediction_observations(
    observations: np.ndarray, config: TaskConfig
) -> np.ndarray:
    extra = np.zeros(config.prediction_steps * 2 + 1, dtype=np.float32)
    return np.concatenate((observations, np.repeat(extra[None], 3, axis=0)), axis=1)


def collect_expert_transitions(
    config: TaskConfig,
    episodes: int,
    max_steps: int,
    seed: int,
    return_process: bool = False,
    return_quality: bool = False,
):
    """Collect IBR expert transitions in the exact 50 Hz training dynamics."""
    environment = ThreeVOneEnv(config, seed=seed)
    expert = GameTheoryRuntime(config)
    transitions = []
    process_samples = []
    captures = 0
    collision_free_episodes = 0
    minimum_separation = float("inf")
    for _ in range(episodes):
        observations, _ = environment.reset()
        expert.reset()
        collision_free = True
        for _ in range(max_steps):
            positions, velocities = environment.control_state()
            actor_obs = zero_prediction_observations(observations, config)
            trace = expert.actions_with_trace(
                positions,
                velocities,
                environment.origin,
                environment.virtual_obstacles,
            )
            actions = trace["actions"]
            if return_process:
                process_samples.append(
                    {
                        "positions": positions.copy(),
                        "origin": environment.origin.copy(),
                        "velocities": velocities.copy(),
                        "actor_obs": actor_obs.copy(),
                        "process_actions": trace["process_actions"].copy(),
                        "potential_gains": trace["potential_gains"].copy(),
                        "br_residual": np.float32(trace["br_residual"]),
                    }
                )
            next_obs, rewards, terminated, truncated, info = environment.step(actions)
            next_positions, next_velocities = environment.control_state()
            next_actor_obs = zero_prediction_observations(next_obs, config)
            transitions.append(
                (
                    positions,
                    velocities,
                    actor_obs,
                    actions,
                    rewards,
                    next_positions,
                    next_velocities,
                    next_actor_obs,
                    np.float32(terminated or truncated),
                )
            )
            observations = next_obs
            separation = float(info["minimum_inter_drone_distance"])
            minimum_separation = min(minimum_separation, separation)
            collision_free = bool(
                collision_free
                and separation >= config.safety_emergency_distance
                and not np.any(info["obstacle_collisions"])
            )
            if terminated or truncated:
                captures += int(info["captured"])
                break
        collision_free_episodes += int(collision_free)
    quality = ExpertQuality(
        episodes=episodes,
        captures=captures,
        collision_free_episodes=collision_free_episodes,
        minimum_separation=minimum_separation,
    )
    if return_process and return_quality:
        return transitions, captures, process_samples, quality
    if return_process:
        return transitions, captures, process_samples
    if return_quality:
        return transitions, captures, quality
    return transitions, captures


def make_epn_samples(
    states: Sequence[np.ndarray], config: TaskConfig
) -> List[Tuple[np.ndarray, np.ndarray]]:
    samples = []
    last_offset = config.prediction_stride * config.prediction_steps
    for now in range(config.history_steps - 1, len(states) - last_offset):
        history = np.asarray(
            states[now - config.history_steps + 1: now + 1],
            dtype=np.float32,
        )
        indices = [
            now + config.prediction_stride * (step + 1)
            for step in range(config.prediction_steps)
        ]
        target = np.asarray([states[index][:2] for index in indices])
        samples.append((history, target.astype(np.float32)))
    return samples


def train_epn(agent, samples, batch_size: int = 128) -> float:
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
            error = torch.sqrt(((prediction - target) ** 2).mean(dim=(1, 2)) + 1e-8)
            confidence_target = torch.exp(-4.0 * error).unsqueeze(1)
        loss = prediction_loss + 0.1 * nn.functional.mse_loss(
            confidence, confidence_target
        )
        agent.epn_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.epn.parameters(), 5.0)
        agent.epn_optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def evader_state(
    positions: np.ndarray,
    velocities: np.ndarray,
    origin: np.ndarray,
    config: TaskConfig,
    virtual_obstacles=None,
) -> np.ndarray:
    evader = config.evader_index
    return np.concatenate(
        (
            (positions[evader] - origin) / config.arena_half_extent,
            velocities[evader] / config.velocity_scale,
        )
    ).astype(np.float32)


def current_actor_observations(agent, environment: ThreeVOneEnv):
    positions, velocities = environment.control_state()
    state = evader_state(
        positions,
        velocities,
        environment.origin,
        environment.config,
        environment.virtual_obstacles,
    )
    agent.update_history(state)
    prediction, confidence = agent.predict()
    observations = build_observations(
        positions,
        velocities,
        environment.origin,
        environment.config,
        environment.virtual_obstacles,
    )
    return agent.augment(observations, prediction, confidence), state
