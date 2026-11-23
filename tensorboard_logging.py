"""Shared TensorBoard logging for the 3V1 training and evaluation tools."""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def add_tensorboard_args(parser: argparse.ArgumentParser) -> None:
    """Add consistent TensorBoard command-line options to an entry point."""
    parser.add_argument(
        "--tensorboard-dir",
        default="runs/three_v_one",
        help="root directory for TensorBoard event files",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="run subdirectory (default: baseline, time and seed)",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="disable TensorBoard event logging",
    )


class TrainingLogger:
    """Small, optional wrapper around ``SummaryWriter``."""

    def __init__(
        self,
        args: argparse.Namespace,
        baseline: str,
        config: Any,
    ) -> None:
        self.writer = None
        self.log_dir: Optional[Path] = None
        self._hparams_logged = False
        if getattr(args, "no_tensorboard", False):
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as error:
            raise RuntimeError(
                "TensorBoard is enabled but unavailable. Install it with "
                "`python -m pip install tensorboard`, or pass --no-tensorboard."
            ) from error

        run_name = getattr(args, "run_name", None)
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            run_name = f"{baseline}_{timestamp}_seed{args.seed}"
        run_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(run_name)).strip("._")
        if not run_name:
            raise ValueError("--run-name must contain at least one valid character")
        root = Path(args.tensorboard_dir).expanduser().resolve()
        self.log_dir = root / run_name
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        arguments = json.dumps(vars(args), indent=2, default=str, sort_keys=True)
        config_values = (
            config.to_dict() if hasattr(config, "to_dict") else vars(config)
        )
        configuration = json.dumps(
            config_values, indent=2, default=str, sort_keys=True
        )
        self.writer.add_text("configuration/arguments", f"```json\n{arguments}\n```", 0)
        self.writer.add_text(
            "configuration/task", f"```json\n{configuration}\n```", 0
        )
        print(f"tensorboard_logdir={self.log_dir}")

    def scalar(self, tag: str, value: Any, step: int) -> None:
        if self.writer is not None:
            self.writer.add_scalar(tag, float(value), step)

    def scalars(self, values: Dict[str, Any], step: int) -> None:
        for tag, value in values.items():
            self.scalar(tag, value, step)

    def hparams(self, parameters: Dict[str, Any], metrics: Dict[str, Any]) -> None:
        """Record one resolved HParams experiment summary for this run."""
        if self.writer is not None and not self._hparams_logged:
            self.writer.add_hparams(parameters, metrics, run_name=".")
            self._hparams_logged = True

    def flush(self) -> None:
        if self.writer is not None:
            self.writer.flush()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None

