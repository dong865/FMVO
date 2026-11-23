"""Shared command-line options for robust and idealized simulation."""

import argparse

from .config import TaskConfig


def add_environment_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ideal-dynamics",
        action="store_true",
        help=(
            "disable mocap/link/actuator noise and domain randomization; "
            "the velocity PID plant remains active"
        ),
    )


def apply_environment_args(config: TaskConfig, args: argparse.Namespace) -> None:
    if getattr(args, "ideal_dynamics", False):
        config.disturbances_enabled = False
