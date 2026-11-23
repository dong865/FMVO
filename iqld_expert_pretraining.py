"""Game-theory experience pretraining for Role-IQLD.

Only expert state-action experience is used. The expert's hard role permutation
is deliberately absent: Sinkhorn matching remains an IQLD-owned latent module.
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class ActionAgreement:
    """Held-out agreement between IQLD and expert velocity actions."""

    mse: float
    cosine_similarity: float
    finite: bool


def evaluate_action_agreement(agent, replay, batch_size: int) -> ActionAgreement:
    """Evaluate expert action prediction without updating or using train data."""
    if not len(replay):
        raise ValueError("action agreement requires holdout transitions")
    count = min(len(replay), batch_size)
    modules = (agent.situation_encoder, agent.role_predictor, agent.actor)
    training_states = tuple(module.training for module in modules)
    for module in modules:
        module.eval()
    try:
        positions, velocities, actor_obs, actions, *_ = replay.sample(
            count, agent.device
        )
        positions = positions / agent.config.arena_half_extent
        velocities = velocities / agent.config.velocity_scale
        with torch.no_grad():
            prediction, _, _, _ = agent.policy(
                actor_obs, positions, velocities
            )
            mse = nn.functional.mse_loss(prediction, actions)
            cosine = nn.functional.cosine_similarity(
                prediction.flatten(1), actions.flatten(1), dim=1
            ).mean()
        finite = bool(torch.isfinite(mse) and torch.isfinite(cosine))
        return ActionAgreement(
            mse=float(mse.cpu()),
            cosine_similarity=float(cosine.cpu()),
            finite=finite,
        )
    finally:
        for module, training in zip(modules, training_states):
            module.train(training)


def require_pretraining_improvement(
    before: ActionAgreement, after: ActionAgreement
) -> ActionAgreement:
    """Stop IQLD when cloning does not improve held-out expert agreement."""
    if (
        not after.finite
        or after.mse >= before.mse
        or after.cosine_similarity <= 0.0
    ):
        raise RuntimeError(
            "IQLD expert pretraining agreement gate failed: "
            f"MSE {before.mse:.6f}->{after.mse:.6f}, "
            f"cosine={after.cosine_similarity:.6f}"
        )
    return after


def behavior_clone(
    agent, expert_replay, steps: int, batch_size: int
) -> Dict[str, float]:
    metrics = []
    for _ in range(steps):
        if len(expert_replay) < batch_size:
            break
        positions, velocities, actor_obs, actions, *_ = expert_replay.sample(
            batch_size, agent.device
        )
        positions /= agent.config.arena_half_extent
        velocities /= agent.config.velocity_scale
        prediction, _, _, _ = agent.policy(
            actor_obs, positions, velocities
        )
        clone_loss = nn.functional.mse_loss(prediction, actions)
        loss = clone_loss
        agent.policy_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(agent.situation_encoder.parameters())
            + list(agent.role_predictor.parameters())
            + list(agent.actor.parameters()),
            5.0,
        )
        agent.policy_optimizer.step()
        metrics.append(
            (
                float(clone_loss.detach().cpu()),
                0.0,
                0.0,
            )
        )
    if not metrics:
        return {"clone": 0.0, "assignment_cost": 0.0, "assignment_entropy": 0.0}
    means = np.asarray(metrics).mean(axis=0)
    return dict(zip(("clone", "assignment_cost", "assignment_entropy"), means))
