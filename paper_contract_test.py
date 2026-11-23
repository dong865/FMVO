"""Fast contract checks for the paper-aligned 5 m pursuit environment."""

import numpy as np
import torch

from .additional_models import normalized_entities
from .config import TaskConfig
from .environment import ThreeVOneEnv
from .paper_game_theory import GameTheoryRuntime
from .role_iqld import RoleIQLDAgent


def main() -> None:
    config = TaskConfig(disturbances_enabled=False)
    assert np.isclose(config.arena_size_m, 5.0)
    assert config.max_speed >= config.evader_max_speed
    assert config.max_accel >= config.evader_max_accel
    assert config.max_episode_steps == 1750

    environment = ThreeVOneEnv(config, seed=2026)
    observations, _ = environment.reset()
    assert observations.shape[0] == 3
    assert np.max(np.abs(environment.positions)) <= config.arena_half_extent

    positions, velocities = environment.control_state()
    expert = GameTheoryRuntime(config)
    trace = expert.actions_with_trace(
        positions,
        velocities,
        environment.origin,
        environment.virtual_obstacles,
    )
    assert trace["actions"].shape == (3, 2)
    assert trace["process_actions"].shape == (config.ibr_rounds, 3, 2)
    assert np.all(trace["potential_gains"] >= -1e-7)
    assert trace["br_residual"] >= 0.0
    assignment = trace["role_assignment"]
    assert np.allclose(assignment.sum(axis=0), 1.0)
    assert np.allclose(assignment.sum(axis=1), 1.0)

    normalized_position, normalized_velocity = normalized_entities(
        positions, velocities, environment.origin, config
    )
    agent = RoleIQLDAgent(config, d_model=16)
    actor_obs = np.zeros((1, 3, agent.actor_obs_dim), dtype=np.float32)
    with torch.no_grad():
        actions, _, soft_assignment, role_targets = agent.policy(
            torch.as_tensor(actor_obs),
            torch.as_tensor(normalized_position[None]),
            torch.as_tensor(normalized_velocity[None]),
        )
    assert actions.shape == (1, 3, 2)
    assert role_targets.shape == (1, 3, 2)
    assert torch.allclose(soft_assignment.sum(dim=1), torch.ones((1, 3)), atol=2e-3)
    assert torch.allclose(soft_assignment.sum(dim=2), torch.ones((1, 3)), atol=2e-3)

    for _ in range(300):
        environment.step(np.ones((3, 2), dtype=np.float32))
    pursuer_speed = np.linalg.norm(environment.velocities[:3], axis=1).max()
    allowed = config.max_speed * config.physical_speed_limit_scale
    assert pursuer_speed <= allowed + 1e-6
    assert allowed >= config.evader_max_speed * config.physical_speed_limit_scale
    print("PASS: 5m arena, speed hierarchy, IBR trace, and role assignment")


if __name__ == "__main__":
    main()
