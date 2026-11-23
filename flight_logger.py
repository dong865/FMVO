"""Stream 3V1 flight telemetry to disk (JSONL per frame + NPZ summary).

The runner used to keep every measured state and every commanded velocity in
memory only, so a flight left nothing behind.  This module records, per 50 Hz
control step:

* measured mocap positions / heights / velocities of all four agents,
* the velocity commands actually sent after the safety filter,
* evader distance to each pursuer, minimum pairwise separation,
* the paper capture criterion (enclosed in the pursuer triangle, all pursuers
  inside ``capture_radius``, no pair below ``safety_emergency_distance``,
  held for ``capture_hold_steps``).

Frames are appended to ``<tag>.jsonl`` immediately (crash safe).  ``close()``
additionally writes ``<tag>.summary.json`` (metadata + outcome) and
``<tag>.npz`` (arrays ready for plotting).
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .config import TaskConfig
from .features import point_in_triangle
from .safety import pairwise_distances


def file_sha256(path: str, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


class CaptureTracker:
    """Reproduce the simulation capture contract on measured mocap poses."""

    def __init__(self, config: TaskConfig) -> None:
        self.config = config
        self.count = 0
        self.captured = False
        self.events: List[float] = []

    def update(self, positions: np.ndarray, timestamp: float) -> bool:
        """Return True on the first frame the ring has been held long enough."""
        pursuers = positions[:3]
        evader = positions[3]
        distances = np.linalg.norm(pursuers - evader, axis=1)
        min_pair = float(np.min(pairwise_distances(positions)))
        enclosed = point_in_triangle(evader, pursuers)
        candidate = bool(
            enclosed
            and float(np.max(distances)) <= self.config.capture_radius
            and min_pair >= self.config.safety_emergency_distance
        )
        self.count = self.count + 1 if candidate else 0
        newly_captured = (
            not self.captured and self.count >= self.config.capture_hold_steps
        )
        if newly_captured:
            self.captured = True
            self.events.append(float(timestamp))
        return newly_captured

    def metrics(self, positions: np.ndarray) -> Dict[str, object]:
        pursuers = positions[:3]
        evader = positions[3]
        distances = np.linalg.norm(pursuers - evader, axis=1)
        min_pair = float(np.min(pairwise_distances(positions)))
        return {
            "pursuer_distances": [round(float(value), 4) for value in distances],
            "min_separation": round(min_pair, 4),
            "enclosed": bool(point_in_triangle(evader, pursuers)),
        }


class FlightRecorder:
    """Crash-safe per-frame recorder for one 3V1 flight."""

    def __init__(
        self,
        log_dir: str,
        tag: str,
        metadata: Dict[str, object],
        config: TaskConfig,
    ) -> None:
        self.config = config
        self.directory = Path(log_dir).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.tag = tag
        self.metadata = dict(metadata)
        self.stem = f"flight_3v1_{tag}"
        self.jsonl_path = self.directory / f"{self.stem}.jsonl"
        self._handle = self.jsonl_path.open("w", encoding="utf-8")
        self.rows: List[Dict[str, object]] = []
        self.t_start: Optional[float] = None
        self.tracker = CaptureTracker(config)

    def record(
        self,
        timestamp: float,
        positions: np.ndarray,
        heights: np.ndarray,
        velocities: np.ndarray,
        commands: np.ndarray,
        phase: str = "flight",
    ) -> bool:
        """Append one frame; returns True when the ring is first captured."""
        if self.t_start is None:
            self.t_start = timestamp
        metrics = self.tracker.metrics(positions)
        newly_captured = self.tracker.update(positions, timestamp - self.t_start)
        row = {
            "t": round(float(timestamp - self.t_start), 4),
            "phase": phase,
            "positions": [[round(float(v), 4) for v in xy] for xy in positions],
            "heights": [round(float(v), 4) for v in heights],
            "velocities": [[round(float(v), 4) for v in vxy] for vxy in velocities],
            "commands": [[round(float(v), 4) for v in vxy] for vxy in commands],
            "pursuer_distances": metrics["pursuer_distances"],
            "min_separation": metrics["min_separation"],
            "enclosed": metrics["enclosed"],
            "capture_candidate": bool(
                self.tracker.count > 0
            ),
            "captured": bool(self.tracker.captured),
            "separation_guard": bool(
                metrics["min_separation"] < self.config.safety_emergency_distance
            ),
        }
        self.rows.append(row)
        self._handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return newly_captured

    def _arrays(self) -> Dict[str, np.ndarray]:
        rows = self.rows
        if not rows:
            return {}
        return {
            "t": np.asarray([row["t"] for row in rows], dtype=np.float32),
            "phase": np.asarray([row.get("phase", "flight") for row in rows], dtype="U12"),
            "positions": np.asarray([row["positions"] for row in rows], dtype=np.float32),
            "heights": np.asarray([row["heights"] for row in rows], dtype=np.float32),
            "velocities": np.asarray([row["velocities"] for row in rows], dtype=np.float32),
            "commands": np.asarray([row["commands"] for row in rows], dtype=np.float32),
            "pursuer_distances": np.asarray(
                [row["pursuer_distances"] for row in rows], dtype=np.float32
            ),
            "min_separation": np.asarray(
                [row["min_separation"] for row in rows], dtype=np.float32
            ),
            "captured": np.asarray([row["captured"] for row in rows], dtype=bool),
            "enclosed": np.asarray([row["enclosed"] for row in rows], dtype=bool),
        }

    def close(self, outcome: str, extra: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        """Flush the stream, write summary JSON and NPZ; return the summary."""
        summary: Dict[str, object] = {
            "outcome": outcome,
            "frames": len(self.rows),
            "duration_s": None,
            "captured": bool(self.tracker.captured),
            "capture_time_s": self.tracker.events[0] if self.tracker.events else None,
            "capture_events_s": [round(value, 3) for value in self.tracker.events],
        }
        if self.rows:
            summary["duration_s"] = round(float(self.rows[-1]["t"]), 3)
            summary["min_separation"] = round(
                float(min(row["min_separation"] for row in self.rows)), 4
            )
            summary["min_pursuer_distance"] = round(
                float(min(min(row["pursuer_distances"]) for row in self.rows)), 4
            )
            summary["height_min"] = round(
                float(min(min(row["heights"]) for row in self.rows)), 3
            )
            summary["height_mean"] = round(
                float(np.mean([np.mean(row["heights"]) for row in self.rows])), 3
            )
            summary["height_max"] = round(
                float(max(max(row["heights"]) for row in self.rows)), 3
            )
            summary["separation_guard_frames"] = int(
                sum(1 for row in self.rows if row["separation_guard"])
            )
        if extra:
            summary.update(extra)
        try:
            self._handle.flush()
            self._handle.close()
        except OSError:
            pass
        np.savez(self.directory / f"{self.stem}.npz", **self._arrays())
        document = {
            "tag": self.tag,
            "metadata": self.metadata,
            "summary": summary,
            "frames": len(self.rows),
        }
        (self.directory / f"{self.stem}.summary.json").write_text(
            json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return summary
