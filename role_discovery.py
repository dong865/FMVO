"""Offline latent-role discovery from role-free expert behavior windows.

This is stage 3 of the expert-behavior role-discovery design
(docs/superpowers/specs/2026-08-28-expert-behavior-role-discovery-design.md).

The discovery stage consumes role-free per-agent windows by duck-typed
contract (the parallel role-discovery data layer produces the same fields)
and produces a versioned, fingerprinted frozen artifact for downstream role
distillation. No environment, task-config, or shared data-layer file is
modified here.
"""

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class RoleFreeWindow:
    """One focal agent's role-free behavior over a fixed time window.

    Fields match the parallel data layer's window contract:
      focal          [T, 4]   focal-relative position/velocity to the evader
      teammates      [T, M, 4]  teammate-relative states (padded, masked)
      teammate_mask  [T, M]   bool, true for active teammates
      actions        [T, 2]   focal action sequence
      effects        [T, 6]   relational transition effect sequence
    """

    episode_id: str
    focal_slot: int
    start: int
    focal: np.ndarray
    teammates: np.ndarray
    teammate_mask: np.ndarray
    actions: np.ndarray
    effects: np.ndarray


def validate_window(window) -> None:
    """Validate one window's shapes and finiteness without mutating it."""
    focal = np.asarray(window.focal)
    teammates = np.asarray(window.teammates)
    mask = np.asarray(window.teammate_mask)
    actions = np.asarray(window.actions)
    effects = np.asarray(window.effects)
    if focal.ndim != 2 or focal.shape[1] != 4:
        raise ValueError("window focal must have shape [T, 4]")
    if teammates.ndim != 3 or teammates.shape[2] != 4:
        raise ValueError("window teammates must have shape [T, M, 4]")
    if mask.shape != (teammates.shape[0], teammates.shape[1]):
        raise ValueError(
            "window teammate_mask shape must match teammates [T, M]"
        )
    if actions.ndim != 2 or actions.shape[1] != 2:
        raise ValueError("window actions must have shape [T, 2]")
    if effects.ndim != 2 or effects.shape[1] != 6:
        raise ValueError("window effects must have shape [T, 6]")
    steps = {focal.shape[0], teammates.shape[0], actions.shape[0],
             effects.shape[0]}
    if len(steps) != 1:
        raise ValueError("all window arrays must share the same step count")
    for name, value in (
        ("focal", focal),
        ("teammates", teammates),
        ("actions", actions),
        ("effects", effects),
    ):
        if not np.isfinite(value).all():
            raise ValueError(f"window {name} must be finite")


def make_synthetic_windows(
    mode_count: int,
    windows_per_mode: int,
    window_steps: int,
    seed: int,
    teammate_count: int = 3,
    state_cue: bool = False,
) -> Tuple[RoleFreeWindow, ...]:
    """Generate windows whose action/effect patterns differ per latent mode.

    Each mode has a distinct constant action direction and a distinct effect
    offset; the relational context is randomized. This makes the latent mode
    discoverable only from behavior (action/effect), not from geometry or
    identity. The hidden mode is recoverable from ordering (mode-major) for
    tests only; the model never receives a mode label.
    """
    if mode_count < 1:
        raise ValueError("mode_count must be positive")
    if windows_per_mode < 1:
        raise ValueError("windows_per_mode must be positive")
    if window_steps < 1:
        raise ValueError("window_steps must be positive")
    rng = np.random.default_rng(seed)
    windows = []
    for mode in range(mode_count):
        angle = mode * 2.0 * np.pi / mode_count
        action_direction = np.array(
            [np.cos(angle), np.sin(angle)], dtype=np.float32
        )
        effect_offset = np.array(
            [np.cos(angle), np.sin(angle), 0.0, 0.0, 0.1 * mode, -0.1 * mode],
            dtype=np.float32,
        )
        for index in range(windows_per_mode):
            focal_noise = 0.1 if state_cue else 0.2
            focal = rng.normal(
                0.0, focal_noise, size=(window_steps, 4)
            ).astype(np.float32)
            if state_cue:
                # the mode is revealed in the initial state (formation
                # offset) so a causal (state-only) predictor can identify it
                state_offset = np.concatenate(
                    (0.3 * np.array([np.cos(angle), np.sin(angle)]),
                     np.zeros(2))
                ).astype(np.float32)
                focal[0] = focal[0] + state_offset
            teammates = rng.normal(
                0.0, 0.2, size=(window_steps, teammate_count, 4)
            ).astype(np.float32)
            actions = np.repeat(
                action_direction[None, :], window_steps, axis=0
            ).astype(np.float32)
            effects = np.repeat(
                effect_offset[None, :], window_steps, axis=0
            ).astype(np.float32)
            mask = np.ones((window_steps, teammate_count), dtype=bool)
            windows.append(
                RoleFreeWindow(
                    episode_id=f"ep-mode{mode}-{index}",
                    focal_slot=0,
                    start=0,
                    focal=focal,
                    teammates=teammates,
                    teammate_mask=mask,
                    actions=actions,
                    effects=effects,
                )
            )
    return tuple(windows)


def split_windows_by_episode(
    windows: Sequence[RoleFreeWindow],
    selection_fraction: float,
    test_fraction: float,
    seed: int,
) -> Tuple[Tuple[RoleFreeWindow, ...], ...]:
    """Split windows by complete episode (no window crosses splits)."""
    if not 0.0 <= selection_fraction < 1.0 or not 0.0 <= test_fraction < 1.0:
        raise ValueError("split fractions must lie in [0, 1)")
    if selection_fraction + test_fraction >= 1.0:
        raise ValueError("split fractions must leave training windows")
    episodes = sorted({window.episode_id for window in windows})
    order = np.random.default_rng(seed).permutation(len(episodes))
    selection_count = int(round(len(episodes) * selection_fraction))
    test_count = int(round(len(episodes) * test_fraction))
    selection_end = selection_count
    test_end = selection_end + test_count
    by_id = {episode_id: [] for episode_id in episodes}
    for window in windows:
        by_id[window.episode_id].append(window)
    selection_ids = {episodes[index] for index in order[:selection_end]}
    test_ids = {episodes[index] for index in order[selection_end:test_end]}
    train_ids = {episodes[index] for index in order[test_end:]}
    groups = []
    for ids in (train_ids, selection_ids, test_ids):
        group = tuple(
            window
            for episode_id in sorted(ids)
            for window in by_id[episode_id]
        )
        groups.append(group)
    return tuple(groups)


def windows_to_batch(windows, device="cpu"):
    """Stack windows into (context, action_target, effect_target) tensors.

    ``context`` has shape [B, T, 17]: the focal-relative state plus the
    masked teammate summary (mean, max, and count) per step. The max keeps the
    formation configuration (who is ahead/behind) that the mean alone cancels,
    so latent roles remain discoverable. Targets are [B, T, 2] actions and
    [B, T, 6] effects.
    """
    import torch

    focal = np.stack([np.asarray(w.focal) for w in windows])
    teammates = np.stack([np.asarray(w.teammates) for w in windows])
    mask = np.stack([np.asarray(w.teammate_mask) for w in windows])
    actions = np.stack([np.asarray(w.actions) for w in windows])
    effects = np.stack([np.asarray(w.effects) for w in windows])
    masked = teammates * mask[..., None]
    counts = np.clip(mask.sum(axis=2, keepdims=True), 1.0, None)
    teammate_mean = masked.sum(axis=2) / counts
    teammate_max = masked.max(axis=2)
    teammate_min = masked.min(axis=2)
    teammate_count = mask.sum(axis=2, keepdims=True)
    context = np.concatenate(
        (
            focal,
            teammate_mean,
            teammate_max,
            teammate_min,
            teammate_count,
        ),
        axis=2,
    ).astype(np.float32)
    return (
        torch.as_tensor(context, dtype=torch.float32, device=device),
        torch.as_tensor(actions, dtype=torch.float32, device=device),
        torch.as_tensor(effects, dtype=torch.float32, device=device),
    )


class TemporalContextEncoder(torch.nn.Module):
    """Permutation-invariant temporal encoder from a window to a context."""

    def __init__(self, window_steps: int, hidden_dim: int = 32):
        super().__init__()
        self.window_steps = window_steps
        self.hidden_dim = hidden_dim
        self.device = torch.device("cpu")
        self.per_step = torch.nn.Sequential(
            torch.nn.Linear(17, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )
        self.gru = torch.nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.head = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, context):
        # context: [B, T, 8]
        steps = self.per_step(context)          # [B, T, H]
        _, hidden = self.gru(steps)             # [1, B, H]
        return self.head(hidden[-1])            # [B, H]


class ContextOnlyDecoder(torch.nn.Module):
    """Baseline decoder predicting action/effect windows from context only."""

    def __init__(
        self, context_dim: int, window_steps: int, hidden_dim: int = 32
    ):
        super().__init__()
        self.window_steps = window_steps
        self.device = torch.device("cpu")
        self.network = torch.nn.Sequential(
            torch.nn.Linear(context_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )
        self.action_head = torch.nn.Linear(hidden_dim, 2)
        self.effect_head = torch.nn.Linear(hidden_dim, 6)

    def forward(self, context):
        features = self.network(context)        # [B, H]
        action_pred = self.action_head(features).unsqueeze(1).repeat(
            1, self.window_steps, 1
        )
        effect_pred = self.effect_head(features).unsqueeze(1).repeat(
            1, self.window_steps, 1
        )
        return action_pred, effect_pred


class RolePosterior(torch.nn.Module):
    """Training posterior q_phi(z | context, action_window, effect_window)."""

    def __init__(self, context_dim: int, window_steps: int, k_roles: int):
        super().__init__()
        self.window_steps = window_steps
        self.k_roles = k_roles
        self.device = torch.device("cpu")
        behavior_dim = window_steps * (2 + 6)
        self.behavior_encoder = torch.nn.Sequential(
            torch.nn.Linear(behavior_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
        )
        self.head = torch.nn.Linear(context_dim + 32, k_roles)

    def forward(self, context, actions, effects):
        behavior = torch.cat(
            (
                actions.reshape(actions.shape[0], -1),
                effects.reshape(effects.shape[0], -1),
            ),
            dim=1,
        )
        encoded = self.behavior_encoder(behavior)
        return self.head(torch.cat((context, encoded), dim=1))


class RoleDecoder(torch.nn.Module):
    """Decoder p_omega(action, effect | context, z) with a discrete bottleneck.

    The role enters the output LINEARLY as a per-role prototype offset
    (action_proto / effect_proto). A soft mixture of K prototypes is a convex
    combination; three non-collinear mode actions cannot be represented by
    two prototypes, so K roles can represent at most K modes. The shared
    context-dependent term cannot encode the mode because the discovery
    context is identity/role-free.
    """

    def __init__(self, context_dim: int, window_steps: int, k_roles: int,
                 hidden_dim: int = 32):
        super().__init__()
        self.window_steps = window_steps
        self.k_roles = k_roles
        self.device = torch.device("cpu")
        self.action_proto = torch.nn.Parameter(
            torch.randn(k_roles, 2) * 0.1
        )
        self.effect_proto = torch.nn.Parameter(
            torch.randn(k_roles, 6) * 0.1
        )
        self.context_net = torch.nn.Sequential(
            torch.nn.Linear(context_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )
        self.action_context = torch.nn.Linear(hidden_dim, 2)
        self.effect_context = torch.nn.Linear(hidden_dim, 6)

    def forward(self, context, soft_z):
        action_offset = soft_z @ self.action_proto           # [B, 2]
        effect_offset = soft_z @ self.effect_proto           # [B, 6]
        shared = self.context_net(context)                   # [B, D]
        action_pred = (
            self.action_context(shared) + action_offset
        ).unsqueeze(1).repeat(1, self.window_steps, 1)
        effect_pred = (
            self.effect_context(shared) + effect_offset
        ).unsqueeze(1).repeat(1, self.window_steps, 1)
        return action_pred, effect_pred


class RoleMixture(torch.nn.Module):
    """Bundles the temporal encoder, posterior, and bottleneck decoder."""

    def __init__(self, context_dim: int, window_steps: int, k_roles: int,
                 beta: float = 0.01, sticky_weight: float = 0.01,
                 entropy_penalty: float = 0.02, gumbel_tau: float = 0.5):
        super().__init__()
        self.context_dim = context_dim
        self.window_steps = window_steps
        self.k_roles = k_roles
        self.beta = beta
        self.sticky_weight = sticky_weight
        self.entropy_penalty = entropy_penalty
        self.gumbel_tau = gumbel_tau
        self.device = torch.device("cpu")
        self.encoder = TemporalContextEncoder(window_steps, context_dim)
        self.posterior_net = RolePosterior(
            context_dim, window_steps, k_roles
        )
        self.decoder = RoleDecoder(context_dim, window_steps, k_roles)

    def posterior(self, context, actions, effects):
        """Return role logits from raw context, encoded internally."""
        encoded = self.encoder(context)
        return self.posterior_net(encoded, actions, effects)

    def decode(self, context, soft_z):
        return self.decoder(context, soft_z)

    def elbo_loss(self, context, actions, effects):
        encoded = self.encoder(context)                      # [B, D]
        logits = self.posterior_net(encoded, actions, effects)
        soft_z = torch.softmax(logits, dim=1)
        # soft assignments during training (smooth gradients coordinate the
        # role assignment); selection evaluation uses the hard argmax so the
        # frozen bottleneck is discrete and K roles represent at most K modes
        action_pred, effect_pred = self.decode(encoded, soft_z)
        nll_action = torch.nn.functional.mse_loss(
            action_pred, actions
        )
        nll_effect = torch.nn.functional.mse_loss(
            effect_pred, effects
        )
        k = self.k_roles
        log_q = torch.log_softmax(logits, dim=1)
        kl = torch.mean(torch.sum(soft_z * (log_q + np.log(k)), dim=1))
        # entropy penalty pushes assignments toward one-hot so the z
        # bottleneck is discrete: K roles can represent at most K modes
        entropy = -torch.mean(torch.sum(soft_z * log_q, dim=1))
        return (
            nll_action + nll_effect + self.beta * kl
            + self.entropy_penalty * entropy
        )


def _kmeans_centroids(windows, k_roles: int, seed: int) -> np.ndarray:
    """K-means over per-window (action_mean, effect_mean) summaries.

    Used only to initialize the decoder prototypes so the ELBO starts from a
    good role assignment (standard mixture-model warm start). The centroids
    are in the behavior space the decoder predicts.
    """
    summaries = np.stack(
        [
            np.concatenate(
                (
                    np.asarray(w.actions).mean(axis=0),
                    np.asarray(w.effects).mean(axis=0),
                )
            )
            for w in windows
        ]
    ).astype(np.float32)
    rng = np.random.default_rng(seed)
    centroids = summaries[rng.choice(len(summaries), k_roles, replace=False)]
    for _ in range(50):
        distances = np.linalg.norm(
            summaries[:, None, :] - centroids[None, :, :], axis=2
        )
        labels = np.argmin(distances, axis=1)
        new_centroids = np.stack(
            [
                summaries[labels == k].mean(axis=0) if (labels == k).any()
                else centroids[k]
                for k in range(k_roles)
            ]
        )
        if np.allclose(new_centroids, centroids, atol=1e-6):
            break
        centroids = new_centroids
    return centroids


def train_mixture(
    windows,
    k_roles: int,
    steps: int,
    batch_size: int,
    seed: int,
    context_dim: int = 16,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    kmeans_init: bool = True,
) -> RoleMixture:
    """Train a RoleMixture deterministically on role-free windows."""
    if len(windows) == 0:
        raise ValueError("train_mixture requires at least one window")
    torch.manual_seed(seed)
    np.random.seed(seed)
    window_steps = np.asarray(windows[0].focal).shape[0]
    mixture = RoleMixture(context_dim, window_steps, k_roles).to(device)
    if kmeans_init:
        centroids = _kmeans_centroids(windows, k_roles, seed)
        with torch.no_grad():
            mixture.decoder.action_proto.copy_(
                torch.as_tensor(centroids[:, :2], dtype=torch.float32)
            )
            mixture.decoder.effect_proto.copy_(
                torch.as_tensor(centroids[:, 2:], dtype=torch.float32)
            )
    optimizer = torch.optim.Adam(mixture.parameters(), lr=learning_rate)
    context, actions, effects = windows_to_batch(windows, device)
    count = len(windows)
    for step in range(steps):
        indices = np.random.default_rng(seed + step).choice(
            count, size=min(batch_size, count), replace=False
        )
        optimizer.zero_grad()
        loss = mixture.elbo_loss(context[indices], actions[indices],
                                 effects[indices])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mixture.parameters(), 5.0)
        optimizer.step()
    return mixture


def reconstruction_loss(mixture, context, actions, effects, device="cpu"):
    """Return (effect_nll, action_nll) using the trained decoder (no KL).

    Evaluation uses the deterministic hard (argmax) role so the selection
    loss reflects the discrete z bottleneck, matching training semantics.
    """
    mixture.eval()
    with torch.no_grad():
        encoded = mixture.encoder(context)
        logits = mixture.posterior_net(encoded, actions, effects)
        soft_z = torch.softmax(logits, dim=1)
        hard_z = torch.zeros_like(soft_z)
        hard_z.scatter_(1, torch.argmax(soft_z, dim=1, keepdim=True), 1.0)
        action_pred, effect_pred = mixture.decode(encoded, hard_z)
    nll_action = float(torch.nn.functional.mse_loss(
        action_pred, actions
    ).detach().cpu())
    nll_effect = float(torch.nn.functional.mse_loss(
        effect_pred, effects
    ).detach().cpu())
    return nll_effect, nll_action


def model_complexity(k_roles: int, per_role_penalty: float) -> float:
    """Frozen complexity penalty: linear in the candidate role count."""
    return float(k_roles * per_role_penalty)


def selection_loss(
    mixture,
    windows,
    alpha_action: float,
    lambda_complexity: float,
    per_role_penalty: float,
    device: str = "cpu",
) -> float:
    """Frozen selection loss: effect NLL + alpha*action NLL + complexity."""
    context, actions, effects = windows_to_batch(windows, device)
    nll_effect, nll_action = reconstruction_loss(
        mixture, context, actions, effects, device
    )
    return (
        nll_effect
        + alpha_action * nll_action
        + lambda_complexity
        * model_complexity(mixture.k_roles, per_role_penalty)
    )


def component_occupancy(mixture, windows, device="cpu") -> np.ndarray:
    """Mean soft role assignment per component over the given windows."""
    context, actions, effects = windows_to_batch(windows, device)
    mixture.eval()
    with torch.no_grad():
        encoded = mixture.encoder(context)
        logits = mixture.posterior_net(encoded, actions, effects)
        soft_z = torch.softmax(logits, dim=1)
    return soft_z.mean(dim=0).detach().cpu().numpy()


def select_k_star(
    loss_table,
    min_support: float = 0.05,
) -> Tuple[int, dict]:
    """Frozen one-standard-error rule with occupancy merge/reject.

    ``loss_table`` maps K -> (mean_selection_loss, bootstrap_se). The smallest
    K whose mean is within one standard error of the minimum is chosen; roles
    with occupancy below ``min_support`` are then rejected (the count shrinks,
    never grows to satisfy a desired number).
    """
    if len(loss_table) < 2:
        raise ValueError("select_k_star requires at least two candidates")
    ks = sorted(loss_table)
    means = {k: loss_table[k][0] for k in ks}
    ses = {k: loss_table[k][1] for k in ks}
    min_k = min(ks, key=lambda k: means[k])
    threshold = means[min_k] + ses[min_k]
    eligible = [k for k in ks if means[k] <= threshold]
    k_star = min(eligible)
    summary = {
        "minimum_k": min_k,
        "threshold": threshold,
        "eligible": eligible,
        "bootstrap": {k: loss_table[k][1] for k in ks},
    }
    return k_star, summary


def selection_loss_table(
    train_windows,
    selection_windows,
    candidate_ks,
    steps: int,
    batch_size: int,
    seed: int,
    alpha_action: float = 0.1,
    lambda_complexity: float = 0.01,
    per_role_penalty: float = 0.02,
    bootstrap_episodes: int = 20,
    device: str = "cpu",
) -> dict:
    """Fit each candidate K identically and report (mean, bootstrap SE)."""
    if not candidate_ks:
        raise ValueError("candidate_ks must not be empty")
    table = {}
    for k in candidate_ks:
        mixture = train_mixture(
            train_windows, k_roles=k, steps=steps,
            batch_size=batch_size, seed=seed, device=device,
        )
        # episode-level bootstrap over the selection split
        episodes = sorted({w.episode_id for w in selection_windows})
        by_id = {episode_id: [] for episode_id in episodes}
        for window in selection_windows:
            by_id[window.episode_id].append(window)
        boot_means = []
        rng = np.random.default_rng(seed + k)
        for _ in range(bootstrap_episodes):
            sample = tuple(
                window
                for episode_id in rng.choice(episodes, size=len(episodes),
                                             replace=True)
                for window in by_id[str(episode_id)]
            )
            if not sample:
                continue
            boot_means.append(
                selection_loss(
                    mixture, sample, alpha_action,
                    lambda_complexity, per_role_penalty, device,
                )
            )
        mean = float(np.mean(boot_means)) if boot_means else float("inf")
        se = float(np.std(boot_means)) if len(boot_means) > 1 else 0.0
        table[k] = (mean, se)
    return table


def artifact_from_mixture(
    mixture: "RoleMixture",
    windows,
    seed: int,
    selection_summary: dict,
) -> "RoleDiscoveryArtifact":
    """Wrap a trained mixture into a frozen artifact (bypassing selection).

    Used by tests and pipeline stages that fit a known role count directly.
    """
    import subprocess

    model_device = next(mixture.parameters()).device
    occupancy = component_occupancy(mixture, windows, str(model_device))
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    )
    source_commit = (
        commit.stdout.strip() if commit.returncode == 0 else "unknown"
    )
    return RoleDiscoveryArtifact(
        schema_version=1,
        method_version=1,
        dataset_fingerprint=_fingerprint(windows),
        split_episode_ids=tuple(
            sorted({window.episode_id for window in windows})
        ),
        candidate_ks=(mixture.k_roles,),
        frozen_rule={"rule": "direct-fit"},
        hyperparameters={
            "context_dim": mixture.context_dim,
            "device": str(model_device),
        },
        k_star=mixture.k_roles,
        selection_summary=selection_summary,
        occupancy=tuple(float(value) for value in occupancy),
        mixture_state=_freeze_mixture(mixture),
        role_prototypes=np.concatenate(
            (
                mixture.decoder.action_proto.detach().cpu().numpy(),
                mixture.decoder.effect_proto.detach().cpu().numpy(),
            ),
            axis=1,
        ),
        source_commit=source_commit,
        seeds={"training": seed},
    )


def _fingerprint(windows) -> str:
    """Stable fingerprint over episode ids and window shapes/counts."""
    import hashlib

    digest = hashlib.sha256()
    digest.update(str(len(windows)).encode())
    for window in windows:
        digest.update(window.episode_id.encode())
        digest.update(str(np.asarray(window.actions).shape).encode())
    return digest.hexdigest()[:16]


@dataclass(frozen=True)
class RoleDiscoveryArtifact:
    """Versioned, fingerprinted frozen output of the discovery stage."""

    schema_version: int
    method_version: int
    dataset_fingerprint: str
    split_episode_ids: Tuple[str, ...]
    candidate_ks: Tuple[int, ...]
    frozen_rule: dict
    hyperparameters: dict
    k_star: int
    selection_summary: dict
    occupancy: Tuple[float, ...]
    mixture_state: dict
    role_prototypes: np.ndarray
    source_commit: str
    seeds: dict


def save_artifact(artifact: RoleDiscoveryArtifact, path: str) -> None:
    """Atomically persist the frozen discovery artifact."""
    import tempfile
    from pathlib import Path

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": artifact.schema_version,
        "method_version": artifact.method_version,
        "dataset_fingerprint": artifact.dataset_fingerprint,
        "split_episode_ids": tuple(artifact.split_episode_ids),
        "candidate_ks": tuple(artifact.candidate_ks),
        "frozen_rule": dict(artifact.frozen_rule),
        "hyperparameters": dict(artifact.hyperparameters),
        "k_star": artifact.k_star,
        "selection_summary": dict(artifact.selection_summary),
        "occupancy": tuple(float(value) for value in artifact.occupancy),
        "mixture_state": artifact.mixture_state,
        "role_prototypes": np.asarray(artifact.role_prototypes),
        "source_commit": artifact.source_commit,
        "seeds": dict(artifact.seeds),
    }
    with tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.", suffix=".tmp",
        dir=destination.parent, delete=False,
    ) as temporary:
        torch.save(payload, temporary.name)
        temporary.flush()
        import os
        os.replace(temporary.name, destination)


def load_artifact(
    path: str, expected_fingerprint: str
) -> RoleDiscoveryArtifact:
    """Load and validate a frozen discovery artifact."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported or missing discovery artifact schema")
    actual = payload.get("dataset_fingerprint")
    if expected_fingerprint is not None and actual != expected_fingerprint:
        raise ValueError(
            "discovery artifact fingerprint mismatch: "
            f"{actual} != {expected_fingerprint}"
        )
    return RoleDiscoveryArtifact(
        schema_version=payload["schema_version"],
        method_version=payload["method_version"],
        dataset_fingerprint=payload["dataset_fingerprint"],
        split_episode_ids=tuple(payload["split_episode_ids"]),
        candidate_ks=tuple(payload["candidate_ks"]),
        frozen_rule=dict(payload["frozen_rule"]),
        hyperparameters=dict(payload["hyperparameters"]),
        k_star=int(payload["k_star"]),
        selection_summary=dict(payload["selection_summary"]),
        occupancy=tuple(float(value) for value in payload["occupancy"]),
        mixture_state=payload["mixture_state"],
        role_prototypes=np.asarray(payload["role_prototypes"]),
        source_commit=payload["source_commit"],
        seeds=dict(payload["seeds"]),
    )


def _freeze_mixture(mixture: RoleMixture) -> dict:
    """Collect deterministic state dicts for the frozen artifact."""
    mixture.eval()
    return {
        "encoder": mixture.encoder.state_dict(),
        "posterior": mixture.posterior_net.state_dict(),
        "decoder": mixture.decoder.state_dict(),
    }


def _shrink_effective_roles(
    k_star: int,
    occupancy: Sequence[float],
    min_support: float = 0.05,
) -> Tuple[int, dict]:
    """Shrink the selected role count to components with real support.

    One-standard-error selection picks the smallest competitive K; when the
    fitted posterior leaves some of those components nearly empty (occupancy
    below min_support), the confirmatory role count is reduced to the
    supported components. Never grows the count. Mirrors select_k_star_v2 so
    discover_roles and the v2 helper suite agree.
    """
    occupancy = np.asarray(list(occupancy), dtype=float)
    supported = int(np.sum(occupancy >= min_support))
    effective = max(supported, 1)
    effective = min(effective, int(k_star))
    return int(effective), {
        "shrink": True if effective < int(k_star) else False,
        "requested_k": int(k_star),
        "effective_k": effective,
        "min_support": float(min_support),
        "occupancy": occupancy.tolist(),
    }


def discover_roles(
    windows,
    candidate_ks,
    selection_fraction: float,
    test_fraction: float,
    seed: int,
    steps: int,
    batch_size: int,
    final_steps: int = 0,
    alpha_action: float = 0.1,
    lambda_complexity: float = 0.01,
    per_role_penalty: float = 0.02,
    bootstrap_episodes: int = 20,
    device: str = "cpu",
) -> RoleDiscoveryArtifact:
    """Run the full discovery stage and return the frozen artifact."""
    import subprocess

    for window in windows:
        validate_window(window)
    train, selection, test = split_windows_by_episode(
        windows, selection_fraction, test_fraction, seed
    )
    if not train or not selection or not test:
        raise ValueError(
            "discovery requires non-empty train/selection/test splits"
        )
    table = selection_loss_table(
        train,
        selection,
        candidate_ks=candidate_ks,
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        alpha_action=alpha_action,
        lambda_complexity=lambda_complexity,
        per_role_penalty=per_role_penalty,
        bootstrap_episodes=bootstrap_episodes,
        device=device,
    )
    k_star, summary = select_k_star(table)
    # the frozen posterior must be well converged for downstream distillation;
    # selection candidates stay on the frozen selection budget
    mixture = train_mixture(
        train,
        k_roles=k_star,
        steps=final_steps or max(steps, 1500),
        batch_size=batch_size, seed=seed, device=device,
    )
    occupancy = component_occupancy(mixture, selection, device)
    # Occupancy-aware shrink: 1-SE may over-select on near-degenerate signal;
    # components below the frozen minimum support are removed (never grown).
    effective_k, shrink_summary = _shrink_effective_roles(
        k_star, occupancy, min_support=0.05
    )
    if effective_k != int(k_star):
        # Re-fit with the shrunk count so mixture_state, occupancy and role
        # prototypes are mutually consistent for downstream distillation.
        mixture = train_mixture(
            train,
            k_roles=effective_k,
            steps=final_steps or max(steps, 1500),
            batch_size=batch_size, seed=seed, device=device,
        )
        occupancy = component_occupancy(mixture, selection, device)
        k_star = effective_k
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    )
    source_commit = (
        commit.stdout.strip() if commit.returncode == 0 else "unknown"
    )
    prototypes = np.concatenate(
        (
            mixture.decoder.action_proto.detach().cpu().numpy(),
            mixture.decoder.effect_proto.detach().cpu().numpy(),
        ),
        axis=1,
    )
    return RoleDiscoveryArtifact(
        schema_version=1,
        method_version=1,
        dataset_fingerprint=_fingerprint(windows),
        split_episode_ids=tuple(
            sorted({window.episode_id for window in windows})
        ),
        candidate_ks=tuple(candidate_ks),
        frozen_rule={
            "rule": "smallest-K-within-one-SE-of-minimum",
            "alpha_action": alpha_action,
            "lambda_complexity": lambda_complexity,
            "per_role_penalty": per_role_penalty,
        },
        hyperparameters={
            "steps": steps,
            "final_steps": final_steps or max(steps, 1500),
            "batch_size": batch_size,
            "selection_fraction": selection_fraction,
            "test_fraction": test_fraction,
            "bootstrap_episodes": bootstrap_episodes,
            "device": device,
        },
        k_star=effective_k,
        selection_summary={
            "table": {str(k): list(v) for k, v in table.items()},
            **summary,
            **shrink_summary,
        },
        occupancy=tuple(float(value) for value in occupancy),
        mixture_state=_freeze_mixture(mixture),
        role_prototypes=prototypes,
        source_commit=source_commit,
        seeds={"selection": seed, "training": seed},
    )


def build_windows_from_episodes(
    episodes,
    window_steps: int,
    max_pursuers: int = 5,
) -> Tuple[RoleFreeWindow, ...]:
    """Build identity-free windows from expert episodes (provisional adapter).

    The parallel role-discovery data layer owns the canonical window builder;
    this adapter mirrors the same contract (episode_id, focal, teammates,
    teammate_mask, actions, effects) so the discovery stage can run end-to-end
    before that layer is merged. It never serializes role fields and never
    crosses episode boundaries.
    """
    if window_steps <= 0:
        raise ValueError("window_steps must be positive")
    windows = []
    for episode in episodes:
        transitions = tuple(episode.transitions)
        if not transitions:
            continue
        count = np.asarray(transitions[0][3]).shape[0]
        if count < 3 or count > max_pursuers:
            raise ValueError(
                "expert pursuer count must lie in [3, max_pursuers]"
            )
        for start in range(len(transitions) - window_steps + 1):
            rows = transitions[start:start + window_steps]
            for focal_slot in range(count):
                focal_values = np.zeros((window_steps, 4), dtype=np.float32)
                teammate_values = np.zeros(
                    (window_steps, max_pursuers - 1, 4), dtype=np.float32
                )
                teammate_mask = np.zeros(
                    (window_steps, max_pursuers - 1), dtype=bool
                )
                action_values = np.zeros((window_steps, 2), dtype=np.float32)
                effect_values = np.zeros((window_steps, 6), dtype=np.float32)
                teammate_indices = [
                    index for index in range(count) if index != focal_slot
                ]
                for offset, transition in enumerate(rows):
                    positions = np.asarray(transition[0], dtype=np.float32)
                    velocities = np.asarray(transition[1], dtype=np.float32)
                    actions = np.asarray(transition[3], dtype=np.float32)
                    next_positions = np.asarray(
                        transition[5], dtype=np.float32
                    )
                    next_velocities = np.asarray(
                        transition[6], dtype=np.float32
                    )
                    evader = count
                    focal_relation = np.concatenate(
                        (
                            positions[focal_slot, :2] - positions[evader, :2],
                            velocities[focal_slot, :2]
                            - velocities[evader, :2],
                        )
                    )
                    teammate_relation = np.concatenate(
                        (
                            positions[teammate_indices, :2]
                            - positions[focal_slot, :2],
                            velocities[teammate_indices, :2]
                            - velocities[focal_slot, :2],
                        ),
                        axis=1,
                    )
                    next_relation = np.concatenate(
                        (
                            next_positions[focal_slot, :2]
                            - next_positions[evader, :2],
                            next_velocities[focal_slot, :2]
                            - next_velocities[evader, :2],
                        )
                    )
                    effect = np.concatenate(
                        (
                            next_relation[:2] - focal_relation[:2],
                            next_relation[2:] - focal_relation[2:],
                            next_velocities[evader, :2]
                            - velocities[evader, :2],
                        )
                    )
                    focal_values[offset] = focal_relation
                    teammate_values[offset, : count - 1] = teammate_relation
                    teammate_mask[offset, : count - 1] = True
                    action_values[offset] = actions[focal_slot]
                    effect_values[offset] = effect
                window = RoleFreeWindow(
                    episode_id=episode.identifier,
                    focal_slot=focal_slot,
                    start=start,
                    focal=_read_only(focal_values),
                    teammates=_read_only(teammate_values),
                    teammate_mask=_read_only(teammate_mask),
                    actions=_read_only(action_values),
                    effects=_read_only(effect_values),
                )
                validate_window(window)
                windows.append(window)
    return tuple(windows)


def _read_only(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value).copy()
    result.setflags(write=False)
    return result


# ---------------------------------------------------------------------------
# Task 4: v2 synthetic generator options + occupancy-aware K selection.
# (Added after the original suite; all existing defaults unchanged.)
# ---------------------------------------------------------------------------
def make_synthetic_windows_v2(
    mode_count: int,
    windows_per_mode: int,
    window_steps: int,
    seed: int,
    teammate_count: int = 3,
    state_cue: bool = False,
    noise: float = 0.2,
    behavior_group: Tuple[int, ...] = (),
    inactive_modes: int = 0,
    imbalance: float = 1.0,
) -> Tuple[RoleFreeWindow, ...]:
    """v2 generator with Task-4 controls.

    behavior_group: mode indices that share one behavior (repeated roles).
    inactive_modes: extra modes whose windows carry ZERO behavior (an
                    unused/background cluster when the fitter sees more K).
    imbalance: factor multiplying windows_per_mode for every other mode.
    """
    if mode_count < 1 or windows_per_mode < 1 or window_steps < 1:
        raise ValueError("mode_count/windows_per_mode/window_steps positive")
    rng = np.random.default_rng(seed)
    windows: list = []
    dirs: dict = {}

    def behavior_for(mode: int):
        if mode in dirs:
            return dirs[mode]
        if behavior_group:
            base = mode % max(len(behavior_group), 1)
            group_mode = behavior_group[base % len(behavior_group)]
            angle = group_mode * 2.0 * np.pi / mode_count
        else:
            group_mode = mode
            angle = mode * 2.0 * np.pi / mode_count
        direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        effect_offset = np.concatenate(
            (direction,
             np.array([0.0, 0.0, 0.1 * group_mode, -0.1 * group_mode],
                      dtype=np.float32))
        ).astype(np.float32)
        dirs[mode] = (direction, effect_offset)
        return dirs[mode]

    for mode in range(mode_count + inactive_modes):
        count = windows_per_mode
        if imbalance != 1.0 and mode % 2 == 1:
            count = max(1, int(round(count * imbalance)))
        inactive = mode >= mode_count
        for index in range(count):
            if inactive:
                direction = np.zeros(2, dtype=np.float32)
                effect_offset = np.zeros(6, dtype=np.float32)
            else:
                direction, effect_offset = behavior_for(mode)
            focal_noise = 0.1 if state_cue else noise
            focal = rng.normal(0.0, focal_noise,
                               size=(window_steps, 4)).astype(np.float32)
            if state_cue and not inactive:
                angle = np.arctan2(direction[1], direction[0])
                offset = np.concatenate(
                    (0.3 * np.array([np.cos(angle), np.sin(angle)]),
                     np.zeros(2))
                ).astype(np.float32)
                focal[0] = focal[0] + offset
            teammates = rng.normal(0.0, 0.2,
                                   size=(window_steps, teammate_count, 4)
                                   ).astype(np.float32)
            actions = np.repeat(direction[None, :], window_steps,
                                axis=0).astype(np.float32)
            effects = np.repeat(effect_offset[None, :], window_steps,
                                axis=0).astype(np.float32)
            mask = np.ones((window_steps, teammate_count), dtype=bool)
            windows.append(RoleFreeWindow(
                episode_id=f"ep-mode{mode}-{index}",
                focal_slot=0, start=0, focal=focal, teammates=teammates,
                teammate_mask=mask, actions=actions, effects=effects,
            ))
    return tuple(windows)


def selection_loss_table_with_occupancy(
    train_windows,
    selection_windows,
    candidate_ks,
    steps: int,
    batch_size: int,
    seed: int,
    alpha_action: float = 0.1,
    lambda_complexity: float = 0.01,
    per_role_penalty: float = 0.02,
    bootstrap_episodes: int = 20,
    inits: int = 2,
    device: str = "cpu",
) -> Tuple[dict, dict]:
    """Fit each candidate K (best of inits), report (mean, SE) and occupancy.

    Occupancy is measured on the frozen selection split; used to reject
    components whose support is too low.
    """
    table: dict = {}
    occupancy: dict = {}
    episodes = sorted({w.episode_id for w in selection_windows})
    by_id = {episode_id: [] for episode_id in episodes}
    for window in selection_windows:
        by_id[window.episode_id].append(window)
    rng_boot = np.random.default_rng(seed)
    bootstrap_samples = []
    for _ in range(bootstrap_episodes):
        sample = tuple(
            window
            for episode_id in rng_boot.choice(
                episodes, size=len(episodes), replace=True)
            for window in by_id[str(episode_id)]
        )
        if sample:
            bootstrap_samples.append(sample)
    fitted = {}
    for k in candidate_ks:
        best = None
        for init in range(inits):
            mixture = train_mixture(
                train_windows, k_roles=k, steps=steps,
                batch_size=batch_size, seed=seed + init * 101 + k,
                device=device,
            )
            boot_means = []
            for sample in bootstrap_samples:
                boot_means.append(selection_loss(
                    mixture, sample, alpha_action, lambda_complexity,
                    per_role_penalty, device))
            mean = float(np.mean(boot_means)) if boot_means else float("inf")
            if best is None or mean < best[0]:
                best = (mean, mixture, np.asarray(boot_means, dtype=float))
        mean, mixture, boot_means = best
        fitted[k] = (mean, mixture, boot_means)
        occ = component_occupancy(mixture, selection_windows, device)
        occupancy[k] = occ
    min_k = min(fitted, key=lambda candidate: fitted[candidate][0])
    reference = fitted[min_k][2]
    for k, (mean, _mixture, boot_means) in fitted.items():
        paired = boot_means - reference
        paired_se = (float(np.std(paired, ddof=1))
                     if len(paired) > 1 else 0.0)
        table[k] = (mean, paired_se)
    return table, occupancy


def select_k_star_v2(
    loss_table: dict,
    occupancy_table: dict,
    min_support: float = 0.05,
) -> Tuple[int, dict]:
    """Frozen one-SE K choice, then shrink to components with support.

    The 1-SE eligible set is taken from loss_table (means[0], se[1]). Among
    eligible candidates we pick the smallest K whose fitted components all
    carry >= min_support occupancy; if none does, we keep the smallest
    eligible K and shrink the reported count to the supported components
    (>=2, else 1).
    """
    ks = sorted(loss_table)
    if len(ks) < 2:
        raise ValueError("select_k_star_v2 requires at least two candidates")
    means = {k: loss_table[k][0] for k in ks}
    ses = {k: loss_table[k][1] for k in ks}
    min_k = min(ks, key=lambda k: means[k])
    threshold = means[min_k]
    eligible = [k for k in ks if means[k] - means[min_k] <= ses[k]]
    occ = {
        k: np.asarray(occupancy_table.get(k, np.ones(k)))
        for k in ks
    }
    chosen = None
    for k in sorted(eligible):
        supported = int((occ[k] >= min_support).sum())
        if supported >= 1:
            chosen = (k, supported)
            break
    if chosen is None:
        k0 = min(eligible)
        supported = int((occ[k0] >= min_support).sum())
        chosen = (k0, max(supported, 1))
    k_star, effective = chosen
    summary = {
        "minimum_k": min_k,
        "threshold": threshold,
        "paired_one_se": True,
        "paired_margins": {int(k): ses[k] for k in ks},
        "eligible": eligible,
        "selected_k": k_star,
        "effective_roles": effective,
        "occupancy": {int(k): occ[k].tolist() for k in ks},
    }
    return effective, summary
