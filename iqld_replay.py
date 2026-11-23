"""Prioritized mixing of immutable expert and online IQLD replay."""

from typing import Tuple

import numpy as np
import torch


class EpisodeBalancedReplay:
    """Immutable expert replay sampled uniformly by episode, then transition."""

    def __init__(self):
        self.episodes = []
        self._size = 0

    def __len__(self):
        return self._size

    def add_episode(self, transitions):
        episode = [
            tuple(np.asarray(item).copy() for item in transition)
            for transition in transitions
        ]
        if not episode:
            return
        self.episodes.append(episode)
        self._size += len(episode)

    def sample(self, batch_size: int, device: torch.device):
        if not self.episodes:
            raise ValueError("cannot sample an empty episode-balanced replay")
        episode_indices = np.random.choice(len(self.episodes), batch_size)
        rows = []
        for episode_index in episode_indices:
            episode = self.episodes[int(episode_index)]
            rows.append(episode[np.random.randint(len(episode))])
        columns = list(zip(*rows))
        return tuple(
            torch.as_tensor(np.stack(column), dtype=torch.float32, device=device)
            for column in columns
        )


def expert_fraction_for_episode(
    episode: int,
    total_episodes: int,
    start: float = 0.75,
    end: float = 0.50,
    anneal_fraction: float = 0.50,
) -> float:
    """Anneal expert sampling over the first half of training."""
    if total_episodes <= 0:
        raise ValueError("total_episodes must be positive")
    if not 0.0 < anneal_fraction <= 1.0:
        raise ValueError("anneal_fraction must lie in (0, 1]")
    anneal_episodes = max(total_episodes * anneal_fraction, 1.0)
    progress = min(max(float(episode), 0.0) / anneal_episodes, 1.0)
    return start + progress * (end - start)


def _allocation(expert_size, online_size, batch_size, fraction):
    expert_count = min(int(round(batch_size * fraction)), expert_size)
    online_count = min(batch_size - expert_count, online_size)
    missing = batch_size - expert_count - online_count
    if missing:
        extra_expert = min(missing, expert_size - expert_count)
        expert_count += extra_expert
        missing -= extra_expert
    if missing:
        extra_online = min(missing, online_size - online_count)
        online_count += extra_online
        missing -= extra_online
    if missing:
        raise ValueError("combined replay does not contain a complete batch")
    return expert_count, online_count


def sample_mixed_batch(
    expert_replay,
    online_replay,
    batch_size: int,
    expert_fraction: float,
    device: torch.device,
) -> Tuple[tuple, float]:
    """Sample one shuffled batch without replacement within either source."""
    if not 0.0 <= expert_fraction <= 1.0:
        raise ValueError("expert_fraction must lie in [0, 1]")
    expert_count, online_count = _allocation(
        len(expert_replay), len(online_replay), batch_size, expert_fraction
    )
    parts = []
    if expert_count:
        parts.append(expert_replay.sample(expert_count, device))
    if online_count:
        parts.append(online_replay.sample(online_count, device))
    columns = tuple(
        torch.cat(items, dim=0) for items in zip(*parts)
    )
    order = torch.randperm(batch_size, device=device)
    return tuple(column[order] for column in columns), expert_count / batch_size
