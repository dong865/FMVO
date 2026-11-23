"""Driver helper: plan and persist the next child trial from the latest manifest.

Usage (from src/crazyswarm2/crazyflie_examples):
  python -m crazyflie_examples.three_v_one_baseline.plan_next_trial \
      --root runs/iqld_continuous_tuning --diagnostic inter_drone_safety

Prints the child run id; the next `tune_iqld --resume` invocation runs it.
"""

import argparse
import glob
from pathlib import Path

from crazyflie_examples.three_v_one_baseline._tune_iqld_core import (
    plan_child_trial,
)
from crazyflie_examples.three_v_one_baseline.tuning_manifest import (
    load_manifest,
    save_manifest,
)
from crazyflie_examples.three_v_one_baseline.tuning_policy import DiagnosticClass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--diagnostic",
        required=True,
        choices=[item.value for item in DiagnosticClass],
    )
    parser.add_argument("--episodes", type=int, default=1000)
    args = parser.parse_args()

    root = Path(args.root)
    manifests = sorted(glob.glob(str(root / "trials" / "*" / "manifest.json")))
    if not manifests:
        raise SystemExit("no manifests found under %s" % root)
    parent = load_manifest(manifests[-1])
    diagnostic = DiagnosticClass(args.diagnostic)
    child = plan_child_trial(parent, diagnostic, history=())
    child_path = Path(child.paths["manifest"])
    if child_path.exists():
        raise SystemExit("child already exists: %s" % child.run_id)
    child_path.parent.mkdir(parents=True, exist_ok=True)
    save_manifest(child_path, child)
    changed = child.metrics.get("changed_parameter")
    print(
        "planned %s (%s %s -> %s) after %s; launch with tune_iqld --resume"
        % (
            child.run_id,
            changed,
            child.metrics.get("parameter_before"),
            child.metrics.get("parameter_after"),
            diagnostic.value,
        )
    )


if __name__ == "__main__":
    main()
