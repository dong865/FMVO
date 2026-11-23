"""Interactive 3D viewer that mirrors the PyBullet GUI scene."""

from collections import deque

import matplotlib.pyplot as plt
import numpy as np


PURSUER_COLORS = ("#1f77e8", "#19b66a", "#9b4de0")
EVADER_COLOR = "#ed3b2f"


class MatplotlibViewer:
    """Display snapshots without participating in environment dynamics."""

    def __init__(self, config, trail_length=300):
        self.config = config
        self.altitude = config.flight_height
        self.extent = config.arena_half_extent
        self.trails_enabled = True
        self.trails = [deque(maxlen=trail_length) for _ in range(4)]
        self.pending_events = set()
        self.closed = False
        plt.ion()
        self.figure = plt.figure(figsize=(11, 8))
        self.figure.canvas.manager.set_window_title("3V1 Pursuit GUI")
        self.axes = self.figure.add_subplot(111, projection="3d")
        self.figure.canvas.mpl_connect("key_press_event", self._on_key)
        self.figure.canvas.mpl_connect("close_event", self._on_close)
        self._configure_axes()
        self._draw_static_scene()
        self.goal_marker = self.axes.scatter(
            [], [], [], s=180, marker="*", color="#ffd21f",
            edgecolor="black", label="Evader goal",
        )
        self.route_line, = self.axes.plot(
            [], [], [], color="#d5a700", linestyle="--", linewidth=1.8
        )
        self.scenario_obstacles = self.axes.scatter(
            [], [], [], s=[], color="#e64b13", alpha=0.55,
            edgecolor="#9c2906", label="Episode obstacles",
        )
        colors = PURSUER_COLORS + (EVADER_COLOR,)
        self.markers = []
        self.arm_lines = []
        self.trail_lines = []
        for index, color in enumerate(colors):
            marker = self.axes.scatter(
                [0.0],
                [0.0],
                [self.altitude],
                s=90,
                color=color,
                edgecolor="black",
                linewidth=0.8,
                depthshade=True,
                label=f"P{index + 1}" if index < 3 else "Evader",
            )
            self.markers.append(marker)
            drone_arms = []
            for _ in range(2):
                line, = self.axes.plot(
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [self.altitude, self.altitude],
                    color=color,
                    linewidth=3.0,
                )
                drone_arms.append(line)
            self.arm_lines.append(drone_arms)
            trail, = self.axes.plot(
                [], [], [], color=color, linewidth=1.4, alpha=0.75
            )
            self.trail_lines.append(trail)
        self.formation_lines = []
        for _ in range(3):
            line, = self.axes.plot(
                [], [], [], color="#10c9d4", linewidth=1.8, alpha=0.9
            )
            self.formation_lines.append(line)
        self.status = self.axes.text2D(
            0.02,
            0.97,
            "",
            transform=self.axes.transAxes,
            verticalalignment="top",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "gray"},
        )
        self.axes.legend(loc="upper right")
        self.figure.tight_layout()
        self.figure.show()
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def _configure_axes(self):
        self.axes.set_xlim(-self.extent, self.extent)
        self.axes.set_ylim(-self.extent, self.extent)
        self.axes.set_zlim(0.0, max(2.0, self.altitude * 1.9))
        self.axes.set_xlabel("X / m")
        self.axes.set_ylabel("Y / m")
        self.axes.set_zlabel("Z / m")
        self.axes.set_title("3V1 pursuit-evasion — 50 Hz planar dynamics")
        self.axes.set_box_aspect((1.0, 1.0, 0.55))
        self.axes.view_init(elev=38.0, azim=-52.0)
        self.axes.grid(True, alpha=0.35)

    def _draw_static_scene(self):
        floor = np.linspace(-self.extent, self.extent, 2)
        floor_x, floor_y = np.meshgrid(floor, floor)
        self.axes.plot_surface(
            floor_x,
            floor_y,
            np.zeros_like(floor_x),
            color="#dfe4e8",
            alpha=0.42,
            shade=False,
        )
        boundary_x = [
            -self.extent,
            self.extent,
            self.extent,
            -self.extent,
            -self.extent,
        ]
        boundary_y = [
            -self.extent,
            -self.extent,
            self.extent,
            self.extent,
            -self.extent,
        ]
        self.axes.plot(
            boundary_x,
            boundary_y,
            [0.01] * 5,
            color="#303840",
            linewidth=2.5,
        )
        theta = np.linspace(0.0, 2.0 * np.pi, 42)
        height = max(1.8, self.altitude * 1.8)
        z_values = np.linspace(0.0, height, 2)
        theta_grid, z_grid = np.meshgrid(theta, z_values)
        for index, (x, y, radius) in enumerate(
            self.config.virtual_obstacles, start=1
        ):
            cylinder_x = x + radius * np.cos(theta_grid)
            cylinder_y = y + radius * np.sin(theta_grid)
            self.axes.plot_surface(
                cylinder_x,
                cylinder_y,
                z_grid,
                color="#e64b13",
                alpha=0.32,
                shade=True,
            )
            safe_radius = radius + self.config.obstacle_safety_margin
            self.axes.plot(
                x + safe_radius * np.cos(theta),
                y + safe_radius * np.sin(theta),
                [self.altitude] * len(theta),
                color="#ff930f",
                linewidth=2.2,
                linestyle="--",
            )
            self.axes.text(
                x,
                y,
                height + 0.05,
                f"O{index}",
                color="#9c2906",
                horizontalalignment="center",
            )

    def _on_key(self, event):
        mapping = {
            " ": ord(" "),
            "space": ord(" "),
            "n": ord("n"),
            "r": ord("r"),
            "t": ord("t"),
            "c": ord("c"),
            "q": ord("q"),
        }
        if event.key in mapping:
            self.pending_events.add(mapping[event.key])

    def _on_close(self, _event):
        self.closed = True
        self.pending_events.add(ord("q"))

    def reset_episode(self):
        for trail in self.trails:
            trail.clear()
        for line in self.trail_lines:
            line.set_data_3d([], [], [])

    def set_scenario(self, goal, obstacles, evader_start=None):
        goal = np.asarray(goal, dtype=float)
        self.goal_marker._offsets3d = (
            [goal[0]], [goal[1]], [self.altitude]
        )
        if evader_start is not None:
            start = np.asarray(evader_start, dtype=float)
            self.route_line.set_data_3d(
                [start[0], goal[0]],
                [start[1], goal[1]],
                [self.altitude, self.altitude],
            )
        obstacles = tuple(obstacles)
        self.scenario_obstacles._offsets3d = (
            [item[0] for item in obstacles],
            [item[1] for item in obstacles],
            [self.altitude for _ in obstacles],
        )
        self.scenario_obstacles.set_sizes(
            [max(70.0, 900.0 * item[2] ** 2) for item in obstacles]
        )

    def toggle_trails(self):
        self.trails_enabled = not self.trails_enabled
        for line in self.trail_lines:
            line.set_visible(self.trails_enabled)

    def _configure_camera(self):
        self.axes.view_init(elev=38.0, azim=-52.0)
        self.figure.canvas.draw_idle()

    def update(self, positions, step, episode, captured, paused):
        arm = 0.075
        for index, position in enumerate(positions):
            x, y = (float(value) for value in position)
            self.markers[index]._offsets3d = ([x], [y], [self.altitude])
            first, second = self.arm_lines[index]
            first.set_data_3d(
                [x - arm, x + arm],
                [y, y],
                [self.altitude, self.altitude],
            )
            second.set_data_3d(
                [x, x],
                [y - arm, y + arm],
                [self.altitude, self.altitude],
            )
            if self.trails_enabled and step % 3 == 0:
                self.trails[index].append((x, y, self.altitude))
            if self.trails[index]:
                values = tuple(zip(*self.trails[index]))
                self.trail_lines[index].set_data_3d(*values)
        for index, line in enumerate(self.formation_lines):
            start = positions[index]
            end = positions[(index + 1) % 3]
            line.set_data_3d(
                [start[0], end[0]],
                [start[1], end[1]],
                [self.altitude, self.altitude],
            )
        state = "PAUSED" if paused else "RUNNING"
        if captured:
            state = "CAPTURED"
        self.status.set_text(
            f"episode={episode}  step={step}  {state}\n"
            f"control={self.config.control_hz:.0f} Hz\n"
            "Space pause | N step | R reset | T trails | C camera | Q quit"
        )
        if step % 2 == 0 or paused or captured:
            self.figure.canvas.draw_idle()
            self.figure.canvas.flush_events()
            plt.pause(0.001)

    def keyboard(self):
        if self.closed:
            return {ord("q")}
        events = self.pending_events
        self.pending_events = set()
        return events

    def close(self):
        if plt.fignum_exists(self.figure.number):
            plt.close(self.figure)
