"""Train Role-IQLD from game-theory expert experience and environment replay."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .additional_models import ReplayBuffer, normalized_entities
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
from .iqld_expert_pretraining import (
    behavior_clone,
    evaluate_action_agreement,
    require_pretraining_improvement,
)
from .iqld_hyperparameters import (
    IQLDHyperparameters,
    add_iqld_hyperparameter_args,
)
from .iqld_replay import expert_fraction_for_episode, sample_mixed_batch
from .robustness_metrics import RobustnessMetrics
from .role_distillation import iqld_value_weights
from .role_iqld import (
    ROLE_IQLD_CHECKPOINT_VERSION,
    MaskedDeepSetsValue,
    RoleIQLDAgent,
    RuntimeSafetyGuard,
)
from .paper_game_theory import GameTheoryRuntime
from .tensorboard_logging import TrainingLogger, add_tensorboard_args
from .training_diagnostics import EpisodeDiagnostics
from .training_validation import (
    add_validation_args,
    checkpoint_is_better,
    validate_checkpoint,
)
from .training_common import (
    add_expert_dataset_args,
    capture_balance_penalty,
    capture_margin_reward,
    current_actor_observations,
    load_epn_augmented_expert_replays,
    load_expert_replays,
    make_epn_samples,
    safety_penalty,
    train_epn,
)


def update_iqld(
    agent: RoleIQLDAgent,
    expert_replay: ReplayBuffer,
    online_replay: ReplayBuffer,
    batch_size: int,
    expert_fraction: float,
):
    """IQL update plus end-to-end Sinkhorn role learning."""
    empty = {
        "q": 0.0,
        "v": 0.0,
        "policy": 0.0,
        "assignment_cost": 0.0,
        "assignment_entropy": 0.0,
        "expert_fraction_actual": 0.0,
    }
    if len(expert_replay) + len(online_replay) < batch_size:
        return empty
    (
        positions,
        velocities,
        actor_obs,
        data_actions,
        rewards,
        next_positions,
        next_velocities,
        next_obs,
        dones,
    ), actual_expert_fraction = sample_mixed_batch(
        expert_replay,
        online_replay,
        batch_size,
        expert_fraction,
        agent.device,
    )
    positions /= agent.config.arena_half_extent
    velocities /= agent.config.velocity_scale
    next_positions /= agent.config.arena_half_extent
    next_velocities /= agent.config.velocity_scale
    team_reward = rewards.mean(dim=1)
    gamma = agent.hyperparameters.discount

    embeddings, mask = agent.encode(positions, velocities)
    prediction_context = agent.prediction_context(actor_obs)
    policy_actions, _, policy_role_logits, mask = agent.policy(
        actor_obs, positions, velocities
    )
    with torch.no_grad():
        next_embeddings, next_mask = agent.encode(
            next_positions, next_velocities
        )
        next_role_logits = agent.role_predictor(next_obs)
        next_prediction_context = agent.prediction_context(next_obs)
        online_v = agent.target_value(
            next_embeddings,
            next_role_logits,
            next_mask,
            next_prediction_context,
        )
        # Optional offline value-function memory (OVM): anchor the TD target so
        # the online critic never drops below the frozen offline estimate.
        # This directly counteracts 'unlearning' of the offline initialization
        # during online fine-tuning. Off by default (no behaviour change).
        if getattr(agent, "actor_mode", "iqld") == "iac":
            # SAC-style self-bootstrap: critic learns at the CURRENT policy's
            # own next-state actions (with entropy bonus), so Q is corrected
            # exactly where the actor operates - the loop that in-sample IQL
            # cannot close.  target-Q nets absorb the bootstrap.
            next_actions, next_log_probs = agent.actor.sample(
                next_obs, next_role_logits
            )
            next_log_prob_team = next_log_probs.mean(dim=1)
            target_q1 = agent.q1_target(
                next_embeddings,
                next_role_logits,
                next_actions,
                next_mask,
                next_prediction_context,
            )
            target_q2 = agent.q2_target(
                next_embeddings,
                next_role_logits,
                next_actions,
                next_mask,
                next_prediction_context,
            )
            target_v = torch.minimum(target_q1, target_q2) - (
                agent.entropy_coef * next_log_prob_team
            )
        else:
            offline_value = getattr(agent, "offline_value", None)
            if offline_value is not None:
                mem_v = offline_value(
                    next_embeddings,
                    next_role_logits,
                    next_mask,
                    next_prediction_context,
                )
                target_v = torch.maximum(online_v, mem_v)
            else:
                target_v = online_v
        target = team_reward + gamma * (1.0 - dones) * target_v
    q1 = agent.q1(
        embeddings.detach(), policy_role_logits.detach(), data_actions, mask,
        prediction_context,
    )
    q2 = agent.q2(
        embeddings.detach(), policy_role_logits.detach(), data_actions, mask,
        prediction_context,
    )
    q_loss = nn.functional.smooth_l1_loss(q1, target)
    q_loss += nn.functional.smooth_l1_loss(q2, target)
    # Cal-QL-style calibration: penalise the critic when the Q of the current
    # policy's actions exceeds a frozen reference value V_ref (behaviour/expert
    # value). This stops the Q-function from being over-optimistic about OOD
    # actions the actor generates, so a later -E[Qmin(s,pi)] actor term does not
    # chase hallucinated boundary actions. alpha via agent.calql_alpha (default
    # 0 = unchanged). V_ref is the frozen offline value when provided.
    calql_alpha = float(getattr(agent, "calql_alpha", 0.0))
    if calql_alpha > 0.0 and getattr(agent, "offline_value", None) is not None:
        with torch.no_grad():
            pi_actions, _, _, _ = agent.policy(
                actor_obs, positions, velocities
            )
        v_ref = agent.offline_value(
            embeddings.detach(), policy_role_logits.detach(), mask,
            prediction_context.detach(),
        )
        q1_pi_c = agent.q1(
            embeddings.detach(), policy_role_logits.detach(),
            pi_actions, mask, prediction_context.detach(),
        )
        q2_pi_c = agent.q2(
            embeddings.detach(), policy_role_logits.detach(),
            pi_actions, mask, prediction_context.detach(),
        )
        cal = torch.relu(torch.minimum(q1_pi_c, q2_pi_c) - v_ref).mean()
        q_loss = q_loss + calql_alpha * cal
    agent.q_optimizer.zero_grad()
    q_loss.backward()
    nn.utils.clip_grad_norm_(
        list(agent.q1.parameters()) + list(agent.q2.parameters()), 5.0
    )
    agent.q_optimizer.step()
    agent.update_q_targets(getattr(agent, "q_target_tau", 0.005))

    value = agent.value(
        embeddings.detach(), policy_role_logits.detach(), mask,
        prediction_context,
    )
    difference = torch.minimum(q1.detach(), q2.detach()) - value
    expectile_weight = torch.where(
        difference > 0.0,
        agent.hyperparameters.expectile,
        1.0 - agent.hyperparameters.expectile,
    )
    value_loss = (expectile_weight * difference.square()).mean()
    agent.value_optimizer.zero_grad()
    value_loss.backward()
    nn.utils.clip_grad_norm_(agent.value.parameters(), 5.0)
    agent.value_optimizer.step()

    weights = iqld_value_weights(
        q1,
        q2,
        value,
        agent.hyperparameters.advantage_temperature,
    )
    behavior_error = (policy_actions - data_actions).square().mean(dim=(1, 2))
    anchor_loss = (weights * behavior_error).mean()
    if getattr(agent, "actor_mode", "iqld") == "iac":
        # IAC/SAC-style policy: entropy-regularised maximisation of
        # min(Q1,Q2) at reparameterised stochastic actions, anchored to the
        # data by the advantage-weighted BC term.  Critics remain in-sample
        # (IQL expectile targets above), so Q(s, a~pi) is never used as a
        # TD target and cannot chase OOD hallucinations as hard as bare Q-max.
        sampled_actions, log_probs = agent.actor.sample(
            actor_obs, policy_role_logits
        )
        q1_sampled = agent.q1(
            embeddings.detach(),
            policy_role_logits.detach(),
            sampled_actions,
            mask,
            prediction_context.detach(),
        )
        q2_sampled = agent.q2(
            embeddings.detach(),
            policy_role_logits.detach(),
            sampled_actions,
            mask,
            prediction_context.detach(),
        )
        log_prob_mean = log_probs.mean(dim=1)  # team-level entropy per batch
        sac_loss = (
            agent.entropy_coef * log_prob_mean
            - torch.minimum(q1_sampled, q2_sampled)
        ).mean()
        prior_term = 0.0
        if getattr(agent, "iqld_prior", None) is not None:
            with torch.no_grad():
                prior_actions = agent.iqld_prior.deterministic(
                    actor_obs, policy_role_logits.detach()
                )
            prior_term = (
                (policy_actions - prior_actions).square().mean()
            )
        policy_loss = (
            agent.anchor_weight * anchor_loss
            + getattr(agent, "prior_weight", 0.0) * prior_term
            + getattr(agent, "sac_weight", 1.0) * sac_loss
        )
    else:
        policy_loss = anchor_loss
    # Optional Q-maximization term: push the ACTOR toward actions with higher
    # Qmin (not just cloning the expert/behaviour data), so online fine-tuning
    # can exceed the expert baseline. lambda via agent.qmax_lambda (default 0 =
    # pure weighted-BC, unchanged). Q params are not in policy_optimizer, so
    # their gradients here do not update them.
    qmax_lambda = float(getattr(agent, "qmax_lambda", 0.0))
    if qmax_lambda > 0.0:
        q1_pi = agent.q1(
            embeddings.detach(), policy_role_logits.detach(),
            policy_actions, mask, prediction_context.detach(),
        )
        q2_pi = agent.q2(
            embeddings.detach(), policy_role_logits.detach(),
            policy_actions, mask, prediction_context.detach(),
        )
        policy_loss = policy_loss - qmax_lambda * torch.minimum(
            q1_pi, q2_pi
        ).mean()
    agent.policy_optimizer.zero_grad()
    policy_loss.backward()
    nn.utils.clip_grad_norm_(
        list(agent.situation_encoder.parameters())
        + list(agent.role_predictor.parameters())
        + list(agent.actor.parameters()),
        5.0,
    )
    agent.policy_optimizer.step()

    with torch.no_grad():
        for target_parameter, parameter in zip(
            agent.target_value.parameters(), agent.value.parameters()
        ):
            target_parameter.mul_(0.995).add_(parameter, alpha=0.005)
    return {
        "q": float(q_loss.detach().cpu()),
        "v": float(value_loss.detach().cpu()),
        "policy": float(policy_loss.detach().cpu()),
        "assignment_cost": 0.0,
        "assignment_entropy": 0.0,
        "expert_fraction_actual": actual_expert_fraction,
    }


def log_hparams_for_final_validation(
    logger: TrainingLogger,
    hyperparameters: IQLDHyperparameters,
    validation,
) -> None:
    """Log resolved HParams from the final measured validation result."""
    if validation is not None:
        logger.hparams(
            hyperparameters.to_dict(),
            {"hparam/robust_capture_rate": validation.capture_rate},
        )


def _load_init_checkpoint(agent: RoleIQLDAgent, checkpoint: str) -> None:
    """Load a v6 checkpoint into the agent for guard-in-loop fine-tuning."""
    data = torch.load(checkpoint, map_location=agent.device, weights_only=False)
    version = data.get("schema_version")
    if version != ROLE_IQLD_CHECKPOINT_VERSION:
        raise ValueError(
            "init checkpoints must be schema version "
            f"{ROLE_IQLD_CHECKPOINT_VERSION}; found {version}"
        )
    k_roles = int(data.get("k_roles", agent.k_roles))
    if k_roles != agent.k_roles:
        raise ValueError(
            f"init checkpoint k_roles={k_roles} does not match agent "
            f"k_roles={agent.k_roles}; pass the matching role artifact"
        )
    for name in (
        "situation_encoder",
        "role_predictor",
        "actor",
        "value",
        "q1",
        "q2",
        "epn",
    ):
        if name == "actor" and agent.actor_mode == "iac":
            # iqld warm start: deterministic trunk matches, stochastic
            # log_std parameter is freshly initialised.
            getattr(agent, name).load_state_dict(data[name], strict=False)
        else:
            getattr(agent, name).load_state_dict(data[name])
    agent.target_value.load_state_dict(agent.value.state_dict())
    if agent.actor_mode == "iac":
        agent.q1_target.load_state_dict(agent.q1.state_dict())
        agent.q2_target.load_state_dict(agent.q2.state_dict())
        # freeze the loaded IQLD deterministic policy as the bootstrap prior
        import copy as _copy

        agent.iqld_prior = _copy.deepcopy(agent.actor)
        for _parameter in agent.iqld_prior.parameters():
            _parameter.requires_grad = False
        agent.iqld_prior.eval()
        print("frozen IQLD policy prior captured for IAC bootstrap", flush=True)
    print(f"init_checkpoint={checkpoint} k_roles={k_roles}")


def train(args: argparse.Namespace) -> Path:
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    hyperparameters = IQLDHyperparameters.from_args(args)
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
    scene_options = None
    if args.scene_preset:
        from .scenario import scene_preset

        scene_options = scene_preset(args.scene_preset)
    environment = ThreeVOneEnv(
        training_config, seed=args.seed, scene_options=scene_options
    )
    k_roles = None
    if args.role_artifact:
        from .role_discovery import load_artifact

        role_artifact = load_artifact(args.role_artifact, expected_fingerprint=None)
        k_roles = role_artifact.k_star
    agent = RoleIQLDAgent(
        config, args.device, args.d_model, hyperparameters, k_roles=k_roles,
        actor_mode=args.actor_mode,
        log_std_init=float(getattr(args, "log_std_init", -1.0)),
        hidden_dim=args.hidden_dim,
    )
    agent.entropy_coef = float(args.entropy_coef)
    agent.anchor_weight = float(args.anchor_weight)
    agent.sac_weight = float(args.sac_weight)
    agent.prior_weight = float(args.prior_weight)
    agent.q_target_tau = float(args.q_target_tau)
    if args.actor_mode == "iac":
        print(
            f"IAC actor mode: entropy={agent.entropy_coef} "
            f"sac_weight={agent.sac_weight} anchor={agent.anchor_weight} "
            f"tau={agent.q_target_tau}",
            flush=True,
        )
    agent.qmax_lambda = args.policy_qmax_lambda
    agent.calql_alpha = args.calql_alpha
    if args.policy_qmax_lambda > 0.0:
        print(f"policy Q-maximization enabled (lambda={args.policy_qmax_lambda})",
              flush=True)
    if args.calql_alpha > 0.0:
        print(f"Cal-QL critic calibration enabled (alpha={args.calql_alpha})",
              flush=True)
    online_replay = ReplayBuffer(args.replay_size)
    if args.epn_augment_expert:
        expert_replay, expert_holdout, expert_summary = (
            load_epn_augmented_expert_replays(
                agent,
                args.expert_dataset,
                config,
                minimum_episodes=args.min_qualified_expert_episodes,
                holdout_fraction=args.expert_holdout_fraction,
                seed=args.seed,
                max_transitions=args.expert_max_transitions,
                min_separation=args.expert_min_separation,
                min_reward=args.expert_min_reward,
                epn_epochs=args.epn_pretrain_epochs,
                safety_penalty_scale=hyperparameters.safety_penalty_scale,
                capture_margin_bonus=hyperparameters.capture_margin_bonus,
                capture_balance_penalty_scale=hyperparameters.capture_balance_penalty,
                episode_balanced=(args.expert_sampling == "episode"),
            )
        )
    else:
        expert_replay, expert_holdout, expert_summary = load_expert_replays(
            args.expert_dataset,
            config,
            minimum_episodes=args.min_qualified_expert_episodes,
            holdout_fraction=args.expert_holdout_fraction,
            seed=args.seed,
            max_transitions=args.expert_max_transitions,
            min_separation=args.expert_min_separation,
            min_reward=args.expert_min_reward,
            safety_penalty_scale=hyperparameters.safety_penalty_scale,
            capture_margin_bonus=hyperparameters.capture_margin_bonus,
            capture_balance_penalty_scale=hyperparameters.capture_balance_penalty,
            episode_balanced=(args.expert_sampling == "episode"),
        )
    agreement_before = evaluate_action_agreement(
        agent, expert_holdout, len(expert_holdout)
    )
    if args.init_checkpoint:
        _load_init_checkpoint(agent, args.init_checkpoint)
        if args.offline_value_memory:
            memory = torch.load(
                args.init_checkpoint, map_location=agent.device,
                weights_only=False,
            )
            frozen = MaskedDeepSetsValue(
                agent.d_model, agent.k_roles, agent.prediction_context_dim
            ).to(agent.device)
            frozen.load_state_dict(memory["value"])
            for param in frozen.parameters():
                param.requires_grad = False
            frozen.eval()
            agent.offline_value = frozen
            print("offline value memory anchored (frozen offline value)", flush=True)
        agreement_after = evaluate_action_agreement(
            agent, expert_holdout, len(expert_holdout)
        )
        pretrain = {
            "clone": 0.0,
            "assignment_cost": 0.0,
            "assignment_entropy": 0.0,
        }
    else:
        pretrain = behavior_clone(agent, expert_replay, hyperparameters.bc_steps, args.batch_size)
        agreement_after = evaluate_action_agreement(
            agent, expert_holdout, len(expert_holdout)
        )
        require_pretraining_improvement(agreement_before, agreement_after)
    logger = TrainingLogger(args, "no_graph_iqld_maddpg", config)
    print(
        f"expert_transitions={len(expert_replay)} "
        f"holdout_transitions={len(expert_holdout)} "
        f"bc_loss={pretrain['clone']:.5f}"
    )
    logger.scalars(
        {
            "expert/transitions": len(expert_replay),
            "expert/holdout_transitions": len(expert_holdout),
            "expert/episodes": expert_summary.total_episodes,
            "expert/minimum_separation": expert_summary.minimum_separation,
            "loss/behavior_clone": pretrain["clone"],
            "assignment/pretrain_geometric_cost": pretrain["assignment_cost"],
            "assignment/pretrain_entropy": pretrain["assignment_entropy"],
            "iqld/pretrain_mse_before": agreement_before.mse,
            "iqld/pretrain_mse_after": agreement_after.mse,
            "iqld/pretrain_cosine_before": agreement_before.cosine_similarity,
            "iqld/pretrain_cosine_after": agreement_after.cosine_similarity,
            "iqld/pretrain_gate_passed": 1.0,
        },
        0,
    )
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    latest = output / "no_graph_iqld_maddpg_latest.pt"
    rollout_guard = (
        RuntimeSafetyGuard(config) if args.rollout_safety_guard else None
    )
    dagger_expert = (
        GameTheoryRuntime(config) if args.dagger_online else None
    )
    if dagger_expert is not None:
        print("DAgger online mode: student rollouts will be relabelled by GameTheoryRuntime",
              flush=True)
        if args.calql_alpha > 0.0 or args.policy_qmax_lambda > 0.0:
            print("WARNING: DAgger mode disables Cal-QL/Q-max; ignoring non-zero alpha/lambda",
                  flush=True)
            agent.calql_alpha = 0.0
            agent.qmax_lambda = 0.0
    epn_samples = []
    total_steps = 0
    total_captures = 0
    best_validation = None
    best_capture_validation = None
    latest_validation = None
    stage_diagnostics = EpisodeDiagnostics()
    # persisted metadata so checkpoints record whether online fine-tuning used
    # the offline value-function memory anchor (OVM); deployment ignores it.
    training_meta = {
        "offline_value_memory": bool(args.offline_value_memory),
        "calql_alpha": float(args.calql_alpha),
        "policy_qmax_lambda": float(args.policy_qmax_lambda),
        "expert_sampling": str(args.expert_sampling),
        "iac_rollout_sampling": bool(args.iac_rollout_sampling),
        "dagger_online": bool(args.dagger_online),
    }

    for episode in range(1, args.episodes + 1):
        # conservative->adaptive schedule for the frozen-IQLD prior anchor:
        # strong at start (keeps SAC local), decays to a floor so late
        # bootstrap can move the policy beyond the offline level.
        if agent.actor_mode == "iac" and getattr(agent, "iqld_prior", None) is not None:
            progress = max(0.0, min(1.0, (episode - 1) / max(args.episodes, 1)))
            agent.prior_weight = float(args.prior_weight) * max(
                0.2, 1.0 - progress
            )
        requested_expert_fraction = expert_fraction_for_episode(
            episode - 1,
            args.episodes,
            hyperparameters.expert_fraction_start,
            hyperparameters.expert_fraction_end,
            hyperparameters.expert_anneal_fraction,
        )
        environment.reset()
        agent.reset_history()
        if rollout_guard is not None:
            rollout_guard.reset()
        if dagger_expert is not None:
            dagger_expert.reset()
        evader_history = []
        episode_return = 0.0
        captured = False
        collision_free = True
        obstacle_collisions = 0
        action_magnitudes = []
        robustness = RobustnessMetrics()
        diagnostics = EpisodeDiagnostics()
        episode_stage = (
            curriculum.current if curriculum is not None else final_stage
        )
        losses = {
            "q": 0.0,
            "v": 0.0,
            "policy": 0.0,
            "assignment_cost": 0.0,
            "assignment_entropy": 0.0,
            "expert_fraction_actual": 0.0,
        }
        noise = 0.0
        for _ in range(args.max_steps):
            positions, velocities = environment.control_state()
            actor_obs, state = current_actor_observations(agent, environment)
            evader_history.append(state)
            if total_steps < args.random_steps:
                action = np.random.uniform(-1.0, 1.0, (3, 2)).astype(np.float32)
                noise = 1.0
            else:
                norm_pos, norm_vel = normalized_entities(
                    positions, velocities, environment.origin, config
                )
                with torch.no_grad():
                    action, _, _, _ = agent.rollout_policy(
                        torch.as_tensor(
                            actor_obs[None], dtype=torch.float32, device=agent.device
                        ),
                        torch.as_tensor(
                            norm_pos[None], dtype=torch.float32, device=agent.device
                        ),
                        torch.as_tensor(
                            norm_vel[None], dtype=torch.float32, device=agent.device
                        ),
                        stochastic=(
                            agent.actor_mode == "iac"
                            and args.iac_rollout_sampling
                        ),
                    )
                action = action[0].cpu().numpy()
                if agent.actor_mode == "iac" and args.iac_rollout_sampling:
                    noise = float(
                        torch.exp(torch.clamp(agent.actor.log_std, -5.0, 1.0))
                        .mean().cpu()
                    )
                else:
                    noise = hyperparameters.exploration_noise * max(
                        hyperparameters.minimum_exploration_noise,
                        1.0 - episode / max(args.episodes, 1)
                    )
                    action = np.clip(
                        action + np.random.normal(0.0, noise, action.shape),
                        -1.0,
                        1.0,
                    )
            if rollout_guard is not None:
                action = rollout_guard.apply(
                    action,
                    positions,
                    velocities,
                    environment.origin,
                    environment.virtual_obstacles,
                )
            action_magnitudes.append(float(np.linalg.norm(action, axis=1).mean()))
            next_obs, rewards, terminated, truncated, info = environment.step(action)
            if dagger_expert is not None:
                # DAgger: student drives the rollout, but the replay buffer stores
                # the expert action for the visited state.
                expert_action = dagger_expert.actions(
                    positions,
                    velocities,
                    environment.origin,
                    environment.virtual_obstacles,
                )
                stored_action = expert_action.astype(np.float32)
            else:
                stored_action = action
            next_positions, next_velocities = environment.control_state()
            if (
                hyperparameters.safety_penalty_scale
                or hyperparameters.capture_margin_bonus
                or hyperparameters.capture_balance_penalty
            ):
                rewards = rewards + safety_penalty(
                    positions,
                    config,
                    hyperparameters.safety_penalty_scale,
                    config.dt,
                )
                rewards = rewards + capture_margin_reward(
                    positions,
                    config,
                    hyperparameters.capture_margin_bonus,
                    config.dt,
                )
                rewards = rewards + capture_balance_penalty(
                    positions,
                    config,
                    hyperparameters.capture_balance_penalty,
                    config.dt,
                )
            online_replay.add(
                positions,
                velocities,
                actor_obs,
                stored_action,
                rewards,
                next_positions,
                next_velocities,
                np.concatenate(
                    (next_obs, actor_obs[:, -agent.prediction_context_dim:]),
                    axis=1,
                ),
                np.float32(terminated or truncated),
            )
            total_steps += 1
            episode_return += float(np.mean(rewards))
            obstacle_collisions += int(
                np.count_nonzero(info["obstacle_collisions"][: config.num_pursuers])
            )
            robustness.update(info)
            diagnostics.update(info, action, info["true_positions"])
            stage_diagnostics.update(info, action, info["true_positions"])
            collision_free = bool(
                collision_free
                and info["minimum_inter_drone_distance"]
                >= config.safety_emergency_distance
                and not np.any(info["obstacle_collisions"])
            )
            if total_steps >= args.learning_starts:
                for _ in range(args.updates_per_step):
                    losses = update_iqld(
                        agent,
                        expert_replay,
                        online_replay,
                        args.batch_size,
                        requested_expert_fraction,
                    )
                    if curriculum is not None:
                        curriculum.record_policy_update()
            if terminated or truncated:
                captured = bool(info["captured"])
                break

        total_captures += int(captured)
        if curriculum is not None:
            previous_stage = curriculum.current.index
            stage = curriculum.update(captured, collision_free)
            curriculum_capture, curriculum_safety = curriculum.rates()
            if stage.index != previous_stage:
                environment = ThreeVOneEnv(
                    stage.config,
                    seed=args.seed + episode,
                    scene_options=scene_options,
                )
                stage_diagnostics.reset()
        else:
            stage = None
            curriculum_capture = float(captured)
            curriculum_safety = float(collision_free)
        if curriculum is not None and curriculum.stage_zero_stalled():
            raise RuntimeError(
                stage_zero_stall_message(
                    stage_diagnostics.tensorboard(curriculum.current)
                )
            )
        epn_samples.extend(make_epn_samples(evader_history, config))
        epn_samples = epn_samples[-args.epn_replay_size:]
        epn_loss = train_epn(agent, epn_samples)
        agent.save(str(latest), {"episode": episode, "total_steps": total_steps, **training_meta})
        if args.validation_interval > 0 and episode % args.validation_interval == 0:
            validation = validate_checkpoint(
                "no_graph_iqld_maddpg",
                latest,
                validation_config,
                args.validation_episodes,
                args.device,
                scene_options=scene_options,
            )
            logger.scalars(validation.tensorboard(), episode)
            latest_validation = validation
            if checkpoint_is_better(validation, best_validation):
                best = output / "no_graph_iqld_maddpg_best_safe.pt"
                agent.save(
                    str(best),
                    {"episode": episode, "validation": validation.__dict__,
                     **training_meta},
                )
                best_validation = validation
            # capture-best checkpoint: the deployment gate values capture
            # primarily, and checkpoints oscillate; keep the best-capture
            # phase so a mid-training peak is not lost
            if (
                best_capture_validation is None
                or validation.capture_rate > best_capture_validation.capture_rate
            ):
                best_capture = output / "no_graph_iqld_maddpg_best_capture.pt"
                agent.save(
                    str(best_capture),
                    {"episode": episode, "validation": validation.__dict__,
                     **training_meta},
                )
                best_capture_validation = validation
        logger.scalars(
            {
                "episode/return": episode_return,
                "episode/length": environment.step_count,
                "episode/captured": int(captured),
                "episode/obstacle_collisions": obstacle_collisions,
                "metrics/cumulative_capture_rate": total_captures / episode,
                "curriculum/stage": (
                    stage.index if stage is not None else 5
                ),
                "curriculum/capture_rate": curriculum_capture,
                "curriculum/collision_free_rate": curriculum_safety,
                "loss/q": losses["q"],
                "loss/value": losses["v"],
                "loss/policy": losses["policy"],
                "loss/epn": epn_loss,
                "assignment/geometric_cost": losses["assignment_cost"],
                "assignment/entropy": losses["assignment_entropy"],
                "exploration/noise_std": noise,
                "policy/mean_action_magnitude": (
                    float(np.mean(action_magnitudes)) if action_magnitudes else 0.0
                ),
                "buffer/online_replay_size": len(online_replay),
                "buffer/expert_replay_size": len(expert_replay),
                "iqld/expert_fraction_requested": requested_expert_fraction,
                "iqld/expert_fraction_actual": losses[
                    "expert_fraction_actual"
                ],
                "buffer/epn_samples": len(epn_samples),
                "progress/environment_steps": total_steps,
                **robustness.tensorboard(),
                **diagnostics.tensorboard(episode_stage),
            },
            episode,
        )
        logger.flush()
        print(
            f"episode={episode}/{args.episodes} return={episode_return:.3f} "
            f"captured={captured} q={losses['q']:.4f} "
            f"v={losses['v']:.4f} policy={losses['policy']:.4f} "
            f"epn={epn_loss:.4f}"
        )
    log_hparams_for_final_validation(logger, hyperparameters, latest_validation)
    logger.close()
    return latest


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=1750)
    parser.add_argument("--random-steps", type=int, default=0)
    parser.add_argument(
        "--epn-augment-expert", action="store_true",
        help="train the EPN on expert trajectories and rebuild expert "
             "observations with real predictions before BC/RL "
             "(fixes the zeroed-prediction train/inference mismatch)",
    )
    parser.add_argument("--epn-pretrain-epochs", type=int, default=5)
    parser.add_argument(
        "--expert-sampling", choices=("episode", "transition"),
        default="episode",
        help="sample expert replay uniformly by episode (default), then by "
             "transition; 'transition' restores the historical behavior",
    )
    parser.add_argument("--learning-starts", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replay-size", type=int, default=200000)
    parser.add_argument("--epn-replay-size", type=int, default=20000)
    parser.add_argument("--updates-per-step", type=int, default=1)
    add_iqld_hyperparameter_args(parser)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument(
        "--hidden-dim", type=int, default=64,
        help="hidden width of the role predictor, actor, and DeepSets V/Q MLPs",
    )
    parser.add_argument(
        "--role-artifact", default=None,
        help="frozen role-discovery artifact: sets k_roles to K_star and "
             "provides the role library for the endogenous-role IQLD",
    )
    parser.add_argument("--obstacles-file", default=None)
    parser.add_argument(
        "--calql-alpha", type=float, default=0.0,
        help="Cal-QL-style critic calibration weight: penalise Q(s,pi(s)) above "
             "a frozen reference value (needs --offline-value-memory), stopping "
             "the critic from over-estimating OOD actor actions; enables a later "
             "actor Q-max term without chasing boundary hallucinations",
    )
    parser.add_argument(
        "--actor-mode", default="iqld", choices=("iqld", "iac"),
        help="'iqld': deterministic advantage-weighted BC actor (unchanged). "
             "'iac': stochastic tanh-Gaussian actor with SAC-style "
             "entropy-regularised Q-maximisation over in-sample critics "
             "(IQLD x SAC/RLPD self-bootstrap hybrid)",
    )
    parser.add_argument(
        "--log-std-init", type=float, default=-1.0,
        help="initial log-std of the IAC stochastic actor (std=exp); "
             "-1.0 => std 0.37, keep it small during warm start",
    )
    parser.add_argument(
        "--iac-rollout-sampling", action="store_true",
        help="sample IAC environment rollouts from the same tanh-Gaussian "
             "actor distribution used by the SAC target; deployment remains "
             "deterministic and external Gaussian exploration is disabled",
    )
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument(
        "--prior-weight", type=float, default=2.0,
        help="weight of the frozen-IQLD-policy anchor beta*E||pi_theta(s) - "
             "pi_iqld(s)||^2 in the IAC policy loss: keeps the SAC bootstrap "
             "inside the distribution the offline phase learned",
    )
    parser.add_argument(
        "--sac-weight", type=float, default=1.0,
        help="weight of the SAC Q-max+entropy term in the IAC policy loss; "
             "<=0.3 together with the calql/OVM anchor keeps the actor from "
             "chasing OOD Q hallucination while still bootstrapping",
    )
    parser.add_argument(
        "--anchor-weight", type=float, default=1.0,
        help="weight of the advantage-weighted BC anchor in the IAC policy "
             "loss (keeps the actor near the data; anneal to let it bootstrap)",
    )
    parser.add_argument("--q-target-tau", type=float, default=0.005)
    parser.add_argument(
        "--scene-preset", default=None,
        help="scene difficulty preset for training environments and "
             "validation (None=legacy, or a scene_preset() name such as "
             "'hard'); expert-replay batches are unaffected (they carry "
             "their own scene geometry)",
    )
    parser.add_argument(
        "--policy-qmax-lambda", type=float, default=0.0,
        help="weight of the actor Q-maximization term (-E[Qmin(s,pi(s))]) in "
             "the policy loss; >0 lets the actor exceed the expert instead of "
             "only cloning it (used during online fine-tuning)",
    )
    parser.add_argument(
        "--offline-value-memory", action="store_true",
        help="anchor online critic TD targets with a frozen offline value "
             "function (max(online, offline-memory)) to prevent unlearning of "
             "the offline initialization during online fine-tuning",
    )
    parser.add_argument(
        "--init-checkpoint", default=None,
        help="v6 checkpoint to fine-tune from; skips BC pretraining and its "
             "agreement gate",
    )
    parser.add_argument(
        "--rollout-safety-guard", action="store_true",
        help="apply the runtime safety guard to rollout actions so training "
             "matches the guarded deployment loop",
    )
    parser.add_argument(
        "--dagger-online", action="store_true",
        help="DAgger-style online refinement: the student policy drives "
             "environment rollouts but the online replay stores actions from "
             "a GameTheoryRuntime expert. This keeps online learning inside "
             "the expert distribution and avoids Cal-QL/SAC OOD bootstrap. "
             "When enabled, calql-alpha and policy-qmax-lambda are ignored.",
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", default="checkpoints/no_graph_iqld")
    add_expert_dataset_args(parser)
    add_curriculum_args(parser)
    add_validation_args(parser)
    add_tensorboard_args(parser)
    add_environment_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
