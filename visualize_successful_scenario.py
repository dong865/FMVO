"""Run the best offline model on gui_scattered, pick a complex successful episode,
and plot trajectories + role assignments.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import TaskConfig
from .environment import ThreeVOneEnv
from .role_iqld import RoleIQLDRuntime
from .scenario import scene_preset


def run_episodes(checkpoint: str, num_episodes: int = 50, seed: int = 777,
                 device: str = "cuda"):
    runtime = RoleIQLDRuntime(checkpoint, device=device, safety_guard=True)
    config = runtime.config
    scene_options = scene_preset("gui_scattered")
    env = ThreeVOneEnv(config, seed=seed, scene_options=scene_options)

    episodes = []
    for ep in range(num_episodes):
        env.reset()
        runtime.reset()
        trajectory = {
            "pursuer_positions": [],
            "evader_positions": [],
            "actions": [],
            "role_logits": [],
            "role_probs": [],
            "rewards": [],
            "info": [],
            "obstacles": env.virtual_obstacles,
        }
        done = False
        captured = False
        while not done:
            positions, velocities = env.control_state()
            origin = env.origin
            obs = env.last_observation if hasattr(env, "last_observation") else None

            # Get actions and role info directly from agent
            from .features import build_observations
            from .additional_models import normalized_entities
            state = np.concatenate(
                ((positions[-1] - origin) / config.arena_half_extent,
                 velocities[-1] / config.velocity_scale)
            )
            runtime.agent.update_history(state)
            prediction, confidence = runtime.agent.predict()
            observations = build_observations(
                positions, velocities, origin, config, env.virtual_obstacles
            )
            actor_obs = runtime.agent.augment(observations, prediction, confidence)
            normalized_pos, normalized_vel = normalized_entities(
                positions, velocities, origin, config
            )
            with torch.no_grad():
                actions, embeddings, role_logits, mask = runtime.agent.policy(
                    torch.as_tensor(actor_obs[None], dtype=torch.float32, device=runtime.device),
                    torch.as_tensor(normalized_pos[None], dtype=torch.float32, device=runtime.device),
                    torch.as_tensor(normalized_vel[None], dtype=torch.float32, device=runtime.device),
                )
            raw_actions = actions[0].cpu().numpy()
            role_probs = torch.softmax(role_logits[0], dim=-1).cpu().numpy()

            # Apply safety guard like runtime does
            safe_actions = runtime._apply_safety_guard(
                raw_actions, positions, velocities, origin, env.virtual_obstacles
            )

            trajectory["pursuer_positions"].append(positions[:config.num_pursuers].copy())
            trajectory["evader_positions"].append(positions[config.evader_index].copy())
            trajectory["actions"].append(safe_actions.copy())
            trajectory["role_logits"].append(role_logits[0].cpu().numpy().copy())
            trajectory["role_probs"].append(role_probs.copy())

            _, rewards, terminated, truncated, info = env.step(safe_actions)
            trajectory["rewards"].append(rewards.copy())
            trajectory["info"].append(info)
            done = terminated or truncated
            captured = bool(info.get("captured", False))
            if done:
                break

        # Compute complexity metrics
        starts = np.asarray(trajectory["pursuer_positions"][0])
        evader_start = np.asarray(trajectory["evader_positions"][0])
        team_diameter = max(
            np.linalg.norm(starts[i] - starts[j])
            for i in range(config.num_pursuers)
            for j in range(i + 1, config.num_pursuers)
        )
        max_pursuer_evader_dist = float(
            np.max(np.linalg.norm(starts - evader_start[None], axis=1))
        )
        trajectory["team_diameter"] = team_diameter
        trajectory["max_pursuer_evader_dist"] = max_pursuer_evader_dist
        trajectory["length"] = len(trajectory["pursuer_positions"])
        trajectory["captured"] = captured
        trajectory["outcome"] = info.get("outcome", "unknown")
        trajectory["scenario"] = env.current_scenario
        episodes.append(trajectory)

    return episodes


def select_complex_success(episodes, min_team_diameter=2.5,
                           min_pursuer_evader_dist=2.5,
                           require_obstacles=False):
    successes = [
        ep for ep in episodes
        if ep["captured"]
        and ep["team_diameter"] >= min_team_diameter
        and ep["max_pursuer_evader_dist"] >= min_pursuer_evader_dist
        and (not require_obstacles or len(ep.get("obstacles", ())) > 0)
    ]
    if not successes:
        successes = [
            ep for ep in episodes
            if ep["captured"]
            and (not require_obstacles or len(ep.get("obstacles", ())) > 0)
        ]
    if not successes:
        return None
    # pick the one with most obstacles first, then largest team diameter
    return max(successes, key=lambda ep: (len(ep.get("obstacles", ())), ep["team_diameter"]))


def plot_trajectory(ep, config, output_path: str):
    pursuers = np.asarray(ep["pursuer_positions"])  # [T, 3, 2]
    evader = np.asarray(ep["evader_positions"])      # [T, 2]
    role_probs = np.asarray(ep["role_probs"])        # [T, 3, K]
    T = len(pursuers)
    colors = ["tab:blue", "tab:green", "tab:orange"]
    role_names = ["Role 0", "Role 1", "Role 2"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Trajectory plot
    ax = axes[0]
    extent = config.arena_half_extent
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal")
    ax.set_title("gui_scattered: complex successful capture")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    # Plot obstacles
    obstacles = ep.get("obstacles", ())
    for ox, oy, orad in obstacles:
        circle = plt.Circle((ox, oy), orad, color="gray", alpha=0.4,
                            label="Obstacle" if ox == obstacles[0][0] else "")
        ax.add_patch(circle)

    # Plot start positions
    for i in range(config.num_pursuers):
        ax.plot(pursuers[0, i, 0], pursuers[0, i, 1], "o", color=colors[i],
                markersize=10, label=f"Pursuer {i} start")
    ax.plot(evader[0, 0], evader[0, 1], "rx", markersize=12, label="Evader start")

    # Plot trajectories with fading alpha
    for i in range(config.num_pursuers):
        for t in range(T - 1):
            alpha = 0.3 + 0.7 * (t / max(T - 1, 1))
            ax.plot(pursuers[t:t+2, i, 0], pursuers[t:t+2, i, 1],
                    color=colors[i], alpha=alpha, linewidth=1.5)
    ax.plot(evader[:, 0], evader[:, 1], "r-", alpha=0.7, linewidth=1.5,
            label="Evader path")

    # Plot capture positions
    for i in range(config.num_pursuers):
        ax.plot(pursuers[-1, i, 0], pursuers[-1, i, 1], "s", color=colors[i],
                markersize=8)
    ax.plot(evader[-1, 0], evader[-1, 1], "r*", markersize=14)

    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)

    # Role probability plot
    ax = axes[1]
    time = np.arange(T) * config.dt
    for i in range(config.num_pursuers):
        for k in range(role_probs.shape[2]):
            ax.plot(time, role_probs[:, i, k],
                    color=colors[i], linestyle=["-", "--", ":"][k],
                    label=f"P{i} {role_names[k]}")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("role probability")
    ax.set_title("Endogenous role assignment over time")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right", fontsize=6, ncol=3)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"saved trajectory plot to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="successful_scenario.png")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print(f"Running {args.episodes} episodes on gui_scattered...")
    episodes = run_episodes(args.checkpoint, args.episodes, args.seed, args.device)
    success_count = sum(1 for ep in episodes if ep["captured"])
    print(f"Successful captures: {success_count}/{len(episodes)}")

    ep = select_complex_success(episodes, require_obstacles=True)
    if ep is None:
        print("No successful episode with obstacles found")
        return

    # Save the fixed scenario for repeatability tests
    scenario_path = str(Path(args.output).with_suffix(".pt"))
    torch.save(ep["scenario"], scenario_path)
    print(f"saved fixed scenario to {scenario_path}")

    config = TaskConfig()
    print(f"\nSelected episode:")
    print(f"  captured={ep['captured']}")
    print(f"  outcome={ep['outcome']}")
    print(f"  length={ep['length']} steps ({ep['length'] * config.dt:.1f} s)")
    print(f"  team_diameter={ep['team_diameter']:.3f} m")
    print(f"  max_pursuer_evader_dist={ep['max_pursuer_evader_dist']:.3f} m")
    obs = ep.get("obstacles", ())
    print(f"  obstacles={len(obs)}")
    for i, (ox, oy, orad) in enumerate(obs):
        print(f"    obstacle {i}: center=({ox:.2f}, {oy:.2f}), radius={orad:.3f} m")

    # Role usage summary
    role_probs = np.asarray(ep["role_probs"])
    role_hard = role_probs.argmax(axis=2)  # [T, 3]
    print(f"\nRole assignments (hard argmax):")
    for i in range(config.num_pursuers):
        roles, counts = np.unique(role_hard[:, i], return_counts=True)
        role_str = ", ".join(f"R{r}:{c}({c/len(role_hard)*100:.1f}%)" for r, c in zip(roles, counts))
        print(f"  Pursuer {i}: {role_str}")

    # Number of role switches per pursuer
    switches = np.sum(np.diff(role_hard, axis=0) != 0, axis=0)
    print(f"\nRole switches per pursuer: {switches.tolist()}")

    plot_trajectory(ep, config, args.output)


if __name__ == "__main__":
    main()
