"""Resolved, immutable tuning parameters for Role-IQLD training."""

import argparse
import math
from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class IQLDHyperparameters:
    """Tuning values that must be recorded with every Role-IQLD run."""

    expert_fraction_start: float = 0.75
    expert_fraction_end: float = 0.50
    expert_anneal_fraction: float = 0.50
    assignment_cost_weight: float = 0.10
    assignment_entropy_weight: float = 0.01
    policy_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    value_learning_rate: float = 3e-4
    expectile: float = 0.70
    advantage_temperature: float = 3.0
    exploration_noise: float = 0.15
    minimum_exploration_noise: float = 0.10
    # Per-environment-step discount for the IQL target. The legacy value
    # 0.99 ** (dt / 0.5) ~ 0.9996 accumulated bootstrap error over the whole
    # episode (critic instability seen on the IRL track); per-step 0.99 keeps
    # Q bounded while the dense distance-progress reward still guides approach.
    discount: float = 0.99
    # Number of behavior-clone steps before RL starts. The IRL track measured
    # clone loss 0.099 at 4k steps vs 0.059 at 10k steps (filtered expert);
    # longer cloning gives the RL phase a better initialization.
    bc_steps: int = 2000
    # Training-only quadratic inter-drone safety penalty scale (0 = disabled).
    # The env's linear safety-warning penalty is dominated by the +10 capture
    # reward, so learned formations drift to ~0.29-0.30 m spacing; a steeper
    # training-side penalty (no TaskConfig change, expert dataset fingerprint
    # intact) teaches the Q-function to keep >= safety_emergency_distance.
    safety_penalty_scale: float = 0.0
    # Training-only reward (rate per second) for enclosing the evader at a
    # safe inter-drone spacing (0 = disabled). The capture-candidate reward
    # incentivizes tight triangles; this bonus rewards the expert's looser
    # 0.35-0.40 m formations instead.
    capture_margin_bonus: float = 0.0
    # Training-only penalty for unbalanced encirclement (0 = disabled). Failures
    # show one pursuer over-committing (0.30-0.31 m to evader) while teammates
    # lag; this penalizes the pursuer-evader distance spread during the close
    # phase so balanced triangles are preferred.
    capture_balance_penalty: float = 0.0

    def __post_init__(self) -> None:
        if not all(math.isfinite(getattr(self, field.name)) for field in fields(self)):
            raise ValueError("all IQLD hyperparameters must be finite")
        if not 0.0 <= self.expert_fraction_end <= self.expert_fraction_start <= 1.0:
            raise ValueError("expert fraction must satisfy 0 <= end <= start <= 1")
        if not 0.0 < self.expert_anneal_fraction <= 1.0:
            raise ValueError("expert anneal fraction must lie in (0, 1]")
        if not 0.5 <= self.expectile < 1.0:
            raise ValueError("expectile must lie in [0.5, 1)")
        if not 0.0 < self.discount <= 1.0:
            raise ValueError("discount must lie in (0, 1]")
        if self.bc_steps <= 0:
            raise ValueError("bc_steps must be positive")
        if self.safety_penalty_scale < 0.0:
            raise ValueError("safety_penalty_scale must be non-negative")
        if self.capture_margin_bonus < 0.0:
            raise ValueError("capture_margin_bonus must be non-negative")
        if self.capture_balance_penalty < 0.0:
            raise ValueError("capture_balance_penalty must be non-negative")
        if min(
            self.policy_learning_rate,
            self.critic_learning_rate,
            self.value_learning_rate,
        ) <= 0.0:
            raise ValueError("learning rates must be positive")
        if min(
            self.assignment_cost_weight, self.assignment_entropy_weight
        ) < 0.0:
            raise ValueError("assignment weights must be non-negative")
        if min(self.exploration_noise, self.minimum_exploration_noise) < 0.0:
            raise ValueError("exploration noise must be non-negative")
        if self.advantage_temperature <= 0.0:
            raise ValueError("advantage temperature must be positive")

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "IQLDHyperparameters":
        """Build a resolved parameter set from the training CLI namespace."""
        return cls(
            **{
                field.name: getattr(args, field.name, field.default)
                for field in fields(cls)
            }
        )

    def to_dict(self) -> dict:
        """Return JSON-serializable resolved training values."""
        return asdict(self)


def add_iqld_hyperparameter_args(parser: argparse.ArgumentParser) -> None:
    """Expose every approved Role-IQLD tuning value on a training CLI."""
    defaults = IQLDHyperparameters()
    parser.add_argument("--expert-fraction-start", type=float, default=defaults.expert_fraction_start)
    parser.add_argument("--expert-fraction-end", type=float, default=defaults.expert_fraction_end)
    parser.add_argument("--expert-anneal-fraction", type=float, default=defaults.expert_anneal_fraction)
    parser.add_argument("--assignment-cost-weight", type=float, default=defaults.assignment_cost_weight)
    parser.add_argument("--assignment-entropy-weight", type=float, default=defaults.assignment_entropy_weight)
    parser.add_argument("--policy-learning-rate", type=float, default=defaults.policy_learning_rate)
    parser.add_argument("--critic-learning-rate", type=float, default=defaults.critic_learning_rate)
    parser.add_argument("--value-learning-rate", type=float, default=defaults.value_learning_rate)
    parser.add_argument("--expectile", type=float, default=defaults.expectile)
    parser.add_argument("--advantage-temperature", type=float, default=defaults.advantage_temperature)
    parser.add_argument("--exploration-noise", type=float, default=defaults.exploration_noise)
    parser.add_argument("--minimum-exploration-noise", type=float, default=defaults.minimum_exploration_noise)
    parser.add_argument("--discount", type=float, default=defaults.discount)
    parser.add_argument("--bc-steps", type=int, default=defaults.bc_steps)
    parser.add_argument("--safety-penalty-scale", type=float, default=defaults.safety_penalty_scale)
    parser.add_argument("--capture-margin-bonus", type=float, default=defaults.capture_margin_bonus)
    parser.add_argument("--capture-balance-penalty", type=float, default=defaults.capture_balance_penalty)
