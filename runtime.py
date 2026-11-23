"""Runtime registry for all migrated 3V1 baselines."""

from typing import Optional


BASELINES = (
    "mappo_open",
    "no_graph_iqld_maddpg",
    "no_graph_irl_maddpg",
    "game_theory",
)


def load_runtime(baseline: str, checkpoint: Optional[str], device: str = "cpu"):
    if baseline == "game_theory":
        from .paper_game_theory import GameTheoryRuntime

        if checkpoint:
            print("[info] game_theory does not use a checkpoint; ignoring it")
        return GameTheoryRuntime()
    if not checkpoint:
        raise ValueError(f"baseline {baseline} requires a trained checkpoint")
    if baseline == "mappo_open":
        from .model import PolicyRuntime

        return PolicyRuntime(checkpoint, device=device)
    if baseline == "no_graph_iqld_maddpg":
        from .role_iqld import RoleIQLDRuntime as IQLDRuntime

        return IQLDRuntime(checkpoint, device=device)
    if baseline == "no_graph_irl_maddpg":
        from .additional_models import IRLRuntime

        return IRLRuntime(checkpoint, device=device)
    raise ValueError(f"unknown baseline {baseline}; expected one of {BASELINES}")
