"""Parent-side JSON bridge for the isolated PyBullet GUI process."""

import json
import subprocess
import sys

import numpy as np


PROTOCOL_PREFIX = "@3V1_GUI@"


def _json_default(value):
    """Convert NumPy values carried by simulation state into JSON values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(
        f"Object of type {value.__class__.__name__} is not JSON serializable"
    )


class ProcessViewer:
    """Expose the viewer API while PyBullet runs in a clean child process."""

    def __init__(self, config, trail_length=300, modern_opengl=True,
                 assets=True, assets_path=None):
        command = [
            sys.executable,
            "-m",
            (
                "crazyflie_examples.three_v_one_baseline."
                "pybullet_render_server"
            ),
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.pending_events = set()
        response = self._request(
            {
                "config": config.to_dict(),
                "trail_length": trail_length,
                "modern_opengl": modern_opengl,
                "assets": assets,
                "assets_path": assets_path,
            }
        )
        if not response.get("ready"):
            raise RuntimeError("PyBullet render process did not become ready")

    def _request(self, message):
        if self.process.poll() is not None:
            raise RuntimeError(
                "PyBullet render process exited with "
                f"{self.process.returncode}"
            )
        encoded = json.dumps(message, default=_json_default) + "\n"
        self.process.stdin.write(encoded)
        self.process.stdin.flush()
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError("PyBullet render process closed its output")
            if line.startswith(PROTOCOL_PREFIX):
                payload = line[len(PROTOCOL_PREFIX):]
                response = json.loads(payload)
                self.pending_events.update(response.get("events", []))
                return response
            print(line, end="")

    def reset_episode(self):
        self._request({"command": "reset"})

    def set_scenario(self, goal, obstacles, evader_start=None, extra=None):
        message = {
            "command": "scenario",
            "goal": list(goal),
            "obstacles": [list(item) for item in obstacles],
            "evader_start": (
                None if evader_start is None else list(evader_start)
            ),
        }
        if extra:
            message["extra"] = extra
        self._request(message)

    def toggle_trails(self):
        self._request({"command": "toggle_trails"})

    def _configure_camera(self):
        self._request({"command": "camera"})

    def update(self, positions, step, episode, captured, paused):
        self._request(
            {
                "command": "update",
                "positions": positions.tolist(),
                "step": step,
                "episode": episode,
                "captured": captured,
                "paused": paused,
            }
        )

    def keyboard(self):
        events = self.pending_events
        self.pending_events = set()
        return events

    def close(self):
        if self.process.poll() is None:
            try:
                message = json.dumps({"command": "close"}) + "\n"
                self.process.stdin.write(message)
                self.process.stdin.flush()
            except BrokenPipeError:
                pass
            try:
                self.process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=2.0)
