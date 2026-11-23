"""Online causal role predictor and role-conditioned policy distillation.

Stage 4 of the expert-behavior role-discovery design: a causal prior
p_theta(z | current observable state) is distilled from the frozen discovery
posterior with soft targets, and a shared role-conditioned actor is trained by
value-consistent behavior cloning. This module is environment-agnostic: it
consumes role-free windows and the frozen discovery artifact, and its networks
accept masked variable-team tensors so the same code serves 3V1/4V1/5V1 once
the variable-team environment is merged.
"""

import numpy as np
import torch

from .role_discovery import (
    RoleDiscoveryArtifact,
    RoleMixture,
    windows_to_batch,
)


def window_current_state(window) -> np.ndarray:
    """Return the first-step observable relational state (identity-free).

    The causal predictor may only see the current state at the window start,
    never the action/effect window or future context.
    """
    focal = np.asarray(window.focal)[0]              # [4]
    teammates = np.asarray(window.teammates)[0]      # [M, 4]
    mask = np.asarray(window.teammate_mask)[0]       # [M]
    masked = teammates * mask[..., None]
    count = max(float(mask.sum()), 1.0)
    teammate_mean = masked.sum(axis=0) / count       # [4]
    return np.concatenate((focal, teammate_mean)).astype(np.float32)


def states_to_tensor(windows, device="cpu"):
    """Stack current-state vectors into a [B, 8] tensor."""
    states = np.stack([window_current_state(w) for w in windows])
    return torch.as_tensor(states, dtype=torch.float32, device=device)


def _validated_sample_weights(sample_weights, count: int, device: str):
    if sample_weights is None:
        weights = torch.ones(count, device=device)
    else:
        weights = torch.as_tensor(
            sample_weights, dtype=torch.float32, device=device
        ).detach()
    if weights.shape != (count,):
        raise ValueError("sample_weights must contain one value per window")
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("sample_weights must be finite and non-negative")
    return weights


class CausalRolePredictor(torch.nn.Module):
    """p_theta(z | current relational state): row-normalized per window."""

    def __init__(self, state_dim: int, k_roles: int, hidden_dim: int = 32):
        super().__init__()
        self.state_dim = state_dim
        self.k_roles = k_roles
        self.device = torch.device("cpu")
        self.network = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, k_roles),
        )

    def forward(self, states):
        return self.network(states)


class RoleConditionedActor(torch.nn.Module):
    """Shared actor: action = actor(local_state, soft_role_embedding)."""

    def __init__(self, state_dim: int, k_roles: int, action_dim: int = 2,
                 hidden_dim: int = 32):
        super().__init__()
        self.state_dim = state_dim
        self.k_roles = k_roles
        self.action_dim = action_dim
        self.device = torch.device("cpu")
        self.role_matrix = torch.nn.Parameter(
            torch.randn(k_roles, hidden_dim) * 0.1
        )
        self.network = torch.nn.Sequential(
            torch.nn.Linear(state_dim + hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, action_dim),
            torch.nn.Tanh(),
        )

    def forward(self, states, soft_z):
        role_embedding = soft_z @ self.role_matrix          # [B, D]
        return self.network(
            torch.cat((states, role_embedding), dim=1)
        )


def posterior_soft_targets(artifact, windows, device="cpu"):
    """Frozen posterior soft assignments q_phi(z | window) as targets."""
    mixture = RoleMixture(
        artifact.hyperparameters.get("context_dim", 16),
        int(np.asarray(windows[0].focal).shape[0]),
        artifact.k_star,
    )
    # the artifact persists nested per-module state dicts
    mixture.encoder.load_state_dict(artifact.mixture_state["encoder"])
    mixture.posterior_net.load_state_dict(artifact.mixture_state["posterior"])
    mixture.decoder.load_state_dict(artifact.mixture_state["decoder"])
    mixture.to(device)
    mixture.eval()
    context, actions, effects = windows_to_batch(windows, device)
    with torch.no_grad():
        encoded = mixture.encoder(context)
        logits = mixture.posterior_net(encoded, actions, effects)
    return torch.softmax(logits, dim=1)


def distill_online_predictor(
    artifact: RoleDiscoveryArtifact,
    windows,
    steps: int,
    batch_size: int,
    seed: int,
    device: str = "cpu",
    sample_weights=None,
) -> CausalRolePredictor:
    """Train p_theta to match the frozen posterior with soft targets."""
    if not windows:
        raise ValueError("distill_online_predictor requires windows")
    torch.manual_seed(seed)
    np.random.seed(seed)
    predictor = CausalRolePredictor(8, artifact.k_star).to(device)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=1e-3)
    states = states_to_tensor(windows, device)
    targets = posterior_soft_targets(artifact, windows, device)
    count = len(windows)
    sample_weights = _validated_sample_weights(
        sample_weights, count, device
    )
    for step in range(steps):
        indices = np.random.default_rng(seed + step).choice(
            count, size=min(batch_size, count), replace=False
        )
        optimizer.zero_grad()
        logits = predictor(states[indices])
        role_error = torch.nn.functional.kl_div(
            torch.log_softmax(logits, dim=1),
            targets[indices],
            reduction="none",
        ).sum(dim=1)
        loss = (sample_weights[indices] * role_error).mean()
        loss.backward()
        optimizer.step()
    return predictor


def iqld_value_weights(
    q1,
    q2,
    value,
    temperature: float,
    uncertainty_scale: float = 0.5,
    max_weight: float = 20.0,
):
    """Return detached advantage weights with twin-Q uncertainty penalty."""
    if q1.shape != q2.shape or q1.shape != value.shape:
        raise ValueError("q1, q2, and value must have matching shapes")
    if temperature <= 0.0 or uncertainty_scale < 0.0 or max_weight <= 0.0:
        raise ValueError("invalid IQL value-weight hyperparameters")
    with torch.no_grad():
        advantage = torch.minimum(q1, q2) - value
        disagreement = (q1 - q2).abs()
        uncertainty = disagreement / (disagreement.median() + 1e-6)
        exponent = temperature * advantage - uncertainty_scale * uncertainty
        return torch.exp(exponent).clamp(max=max_weight)


def train_role_conditioned_bc(
    windows,
    soft_z,
    steps: int,
    batch_size: int,
    seed: int,
    k_roles: int,
    device: str = "cpu",
    sample_weights=None,
) -> RoleConditionedActor:
    """Value-weighted cloning conditioned on soft role assignments.

    ``soft_z`` may come from the frozen posterior (offline functional gate)
    or from the online causal predictor (deployment path).
    """
    if not windows:
        raise ValueError("train_role_conditioned_bc requires windows")
    torch.manual_seed(seed)
    np.random.seed(seed)
    actor = RoleConditionedActor(8, k_roles).to(device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    states = states_to_tensor(windows, device)
    actions = torch.as_tensor(
        np.stack([np.asarray(w.actions)[0] for w in windows]),
        dtype=torch.float32, device=device,
    )
    if soft_z.shape[0] != len(windows):
        raise ValueError("soft_z count must match windows")
    sample_weights = _validated_sample_weights(
        sample_weights, len(windows), device
    )
    count = len(windows)
    for step in range(steps):
        indices = np.random.default_rng(seed + step).choice(
            count, size=min(batch_size, count), replace=False
        )
        optimizer.zero_grad()
        prediction = actor(states[indices], soft_z[indices])
        behavior_error = (prediction - actions[indices]).square().mean(dim=1)
        loss = (sample_weights[indices] * behavior_error).mean()
        loss.backward()
        optimizer.step()
    return actor


def train_no_role_bc(
    windows,
    steps: int,
    batch_size: int,
    seed: int,
    k_roles: int,
    device: str = "cpu",
    sample_weights=None,
) -> RoleConditionedActor:
    """Train the matched actor with an uninformative uniform role input."""
    uniform_z = torch.full(
        (len(windows), k_roles), 1.0 / k_roles,
        dtype=torch.float32, device=device,
    )
    return train_role_conditioned_bc(
        windows,
        uniform_z,
        steps,
        batch_size,
        seed,
        k_roles,
        device,
        sample_weights,
    )


def role_conditioned_action_error(
    actor: RoleConditionedActor,
    soft_z,
    windows,
    device: str = "cpu",
) -> float:
    """Held-out MSE of role-conditioned action predictions."""
    actor.eval()
    states = states_to_tensor(windows, device)
    actions = torch.as_tensor(
        np.stack([np.asarray(w.actions)[0] for w in windows]),
        dtype=torch.float32, device=device,
    )
    with torch.no_grad():
        prediction = actor(states, soft_z)
    return float(torch.nn.functional.mse_loss(
        prediction, actions
    ).detach().cpu())


def no_role_action_error(actor, windows, k_roles: int, device="cpu") -> float:
    """Held-out error with the matched actor and uniform role input."""
    uniform_z = torch.full(
        (len(windows), k_roles), 1.0 / k_roles,
        dtype=torch.float32, device=device,
    )
    return role_conditioned_action_error(actor, uniform_z, windows, device)
