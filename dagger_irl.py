"""DAgger-style online expert correction for the 3V1 IRL baseline.

Diagnosis: pure BC reaches only ~10% capture on the expert collection seeds
(experts themselves capture 100% there). BC is open-loop (fit state->action),
but closed-loop rollouts drift off the expert state distribution (covariate
shift) and errors compound. DAgger re-labels the policy's OWN rollouts with
online expert (Game Theory) actions and re-trains, learning only the states
the policy actually encounters — the "filter expert experience, skip useless
samples" principle applied online.

Run: dagger_irl.py --rounds 3 --rollout-seeds 20 --bc-steps 30000 ...
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")
from crazyflie_examples.three_v_one_baseline.additional_models import (  # noqa: E402
    IRLAgent,
    ReplayBuffer,
)
from crazyflie_examples.three_v_one_baseline.config import TaskConfig  # noqa: E402
from crazyflie_examples.three_v_one_baseline.environment import ThreeVOneEnv  # noqa: E402
from crazyflie_examples.three_v_one_baseline.game_theory import GameTheoryRuntime  # noqa: E402
from crazyflie_examples.three_v_one_baseline.training_common import (  # noqa: E402
    load_expert_replays,
    zero_prediction_observations,
)
from crazyflie_examples.three_v_one_baseline.train_irl import behavior_clone  # noqa: E402


def collect_policy_rollout(
    agent: IRLAgent,
    config: TaskConfig,
    expert: GameTheoryRuntime,
    seeds,
    max_steps: int,
):
    """Roll out the policy (closed loop); label every state with the expert.

    The expert's internal evader filter is updated frame by frame, so labels
    follow the same temporal order the expert saw during collection.
    """
    transitions = []
    captured = 0
    collision_free = 0
    for seed in seeds:
        env = ThreeVOneEnv(config, seed=seed)
        obs, _ = env.reset()
        expert.reset()
        cf = True
        for _ in range(max_steps):
            positions, velocities = env.control_state()
            actor_obs = zero_prediction_observations(obs, config)
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
            expert_action = expert.actions(
                positions, velocities, env.origin, env.virtual_obstacles
            )
            next_obs, rewards, terminated, truncated, info = env.step(action)
            next_positions, next_velocities = env.control_state()
            transitions.append(
                (
                    positions,
                    velocities,
                    actor_obs,
                    expert_action,  # DAgger: expert label on policy state
                    rewards,
                    next_positions,
                    next_velocities,
                    zero_prediction_observations(next_obs, config),
                    np.float32(terminated or truncated),
                )
            )
            obs = next_obs
            if info["minimum_inter_drone_distance"] < config.safety_emergency_distance:
                cf = False
            if terminated or truncated:
                break
        captured += int(info["captured"])
        collision_free += int(cf)
    return transitions, captured, collision_free


def evaluate_capture(agent: IRLAgent, config: TaskConfig, seeds, max_steps):
    captured = 0
    total = 0
    for seed in seeds:
        env = ThreeVOneEnv(config, seed=seed)
        obs, _ = env.reset()
        for _ in range(max_steps):
            actor_obs = zero_prediction_observations(obs, config)
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
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        captured += int(info["captured"])
        total += 1
    return captured, total


def mixed_behavior_clone(
    agent: IRLAgent,
    expert: ReplayBuffer,
    dagger: ReplayBuffer,
    steps: int,
    batch_size: int,
):
    """BC on expert data + DAgger data, sampled proportional to buffer size.

    Bounded aggregation keeps the original expert data forever and only trims
    the DAgger buffer to the most recent K rounds, so the expert behavior
    basis is never popped (a single shared queue would evict the expert data
    first — the earliest entries — and destroy the BC foundation).
    """
    losses = []
    for _ in range(steps):
        sources = [(expert, len(expert)), (dagger, len(dagger))]
        total = sum(size for _, size in sources)
        if total < batch_size:
            break
        pick = np.random.rand() * total
        cumulative = 0.0
        chosen = expert
        for source, size in sources:
            cumulative += size
            if pick < cumulative:
                chosen = source
                break
        if len(chosen) < batch_size:
            continue
        _, _, observations, actions, *_ = chosen.sample(batch_size, agent.device)
        prediction = agent.actor(observations.flatten(0, 1)).view(-1, 3, 2)
        loss = torch.nn.functional.mse_loss(prediction, actions)
        agent.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 5.0)
        agent.actor_optimizer.step()
        losses.append(float(loss.detach().cpu()))
    agent.target_actor.load_state_dict(agent.actor.state_dict())
    return float(np.mean(losses)) if losses else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--rollout-seeds", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1750)
    parser.add_argument("--bc-steps", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--max-aggregate-rounds",
        type=int,
        default=0,
        help="keep only the most recent K rounds of DAgger data in the training "
        "buffer (0 = unbounded aggregation; bounded aggregation keeps fresh "
        "policy-state data dominant so the policy keeps adapting)",
    )
    parser.add_argument(
        "--disturbances",
        action="store_true",
        help="roll out with disturbances enabled (robust scenario coverage)",
    )
    args = parser.parse_args()

    config = TaskConfig()
    if args.disturbances:
        config = replace(config, disturbances_enabled=True)
        print("[info] rollout config: disturbances ENABLED (robust coverage)")
    agent = IRLAgent(config, args.device)
    expert = GameTheoryRuntime(config)
    train_replay, _, _ = load_expert_replays(
        args.expert_dataset,
        config,
        minimum_episodes=20,
        holdout_fraction=0.20,
        seed=args.seed,
        max_transitions=1000,
        min_separation=0.32,
        min_reward=0.0,
    )

    # Evaluation seeds: expert collection seeds for in-distribution comparison.
    eval_seeds = [1013 + i for i in range(20)]
    # Rollout seeds: mix deployment-gate seeds (101/211/307/401/503) with
    # offsets so online correction covers the deployment state distribution,
    # not just the BC-friendly collection seeds.
    rollout_seeds = []
    for base in (101, 211, 307, 401, 503):
        rollout_seeds.extend(base + 1000 * i for i in range(max(1, args.rollout_seeds // 5)))
    rollout_seeds = rollout_seeds[: args.rollout_seeds]

    # Separate buffers: expert data is permanent; DAgger data is bounded to the
    # most recent K rounds so fresh policy states stay dominant.
    dagger_replay = ReplayBuffer()

    loss = behavior_clone(agent, train_replay, args.bc_steps, args.batch_size)
    cap, tot = evaluate_capture(agent, config, eval_seeds, args.max_steps)
    print(f"round 0 (pure BC): bc_loss={loss:.5f} capture {cap}/{tot} = {cap/tot:.1%}")

    round_frames = []
    for round_index in range(1, args.rounds + 1):
        transitions, captured, cf = collect_policy_rollout(
            agent, config, expert, rollout_seeds, args.max_steps
        )
        print(
            f"  round {round_index}: policy rollout capture {captured}/{len(rollout_seeds)}, "
            f"colfree {cf}/{len(rollout_seeds)}, frames {len(transitions)}"
        )
        for transition in transitions:
            dagger_replay.add(*transition)
        round_frames.append(len(transitions))
        if args.max_aggregate_rounds > 0:
            # Trim only the DAgger buffer to the most recent K rounds.
            overflow = len(round_frames) - args.max_aggregate_rounds
            while overflow > 0:
                for _ in range(round_frames.pop(0)):
                    dagger_replay.pop_oldest()
                overflow -= 1
        if args.max_aggregate_rounds > 0:
            loss = mixed_behavior_clone(
                agent, train_replay, dagger_replay, args.bc_steps, args.batch_size
            )
        else:
            for transition in transitions:
                train_replay.add(*transition)
            loss = behavior_clone(agent, train_replay, args.bc_steps, args.batch_size)
        cap, tot = evaluate_capture(agent, config, eval_seeds, args.max_steps)
        print(f"  round {round_index}: bc_loss={loss:.5f} capture {cap}/{tot} = {cap/tot:.1%}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    agent.save(
        str(out),
        {
            "episode": 0,
            "rounds": args.rounds,
            "note": "DAgger: BC + online expert relabeling of policy rollouts",
            "capture_on_eval_seeds": f"{cap}/{tot}",
        },
    )
    print(f"saved {out}")


if __name__ == "__main__":
    main()
