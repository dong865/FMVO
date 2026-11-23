"""PyBullet-only render process for the 3V1 GUI.

Do not import NumPy, Torch, or the simulation environment here. On some Linux
NVIDIA/PyBullet combinations the ExampleBrowser GUI thread crashes when those
numeric extension modules share its process. The parent sends JSON snapshots.
"""

from collections import deque
import json
import math
import os
import sys
import time

import pybullet as bullet

# Asset-based scene (gym-pybullet-drones meshes). Kept optional so a plain
# pybullet install without the assets dir still falls back to debug geometry.
try:
    from . import pybullet_assets as _assets
except Exception:  # pragma: no cover - defensive
    _assets = None


PROTOCOL_PREFIX = "@3V1_GUI@"
PURSUER_COLORS = (
    (0.12, 0.48, 0.95, 1.0),
    (0.10, 0.72, 0.42, 1.0),
    (0.64, 0.30, 0.92, 1.0),
)
EVADER_COLOR = (0.95, 0.24, 0.18, 1.0)
TRAIL_COLORS = (
    (0.25, 0.58, 1.0),
    (0.20, 0.82, 0.52),
    (0.74, 0.42, 1.0),
    (1.0, 0.34, 0.24),
)


def replacement_kwargs(item_id):
    """Only replace a debug item after PyBullet returned a valid ID."""
    return {"replaceItemUniqueId": item_id} if item_id >= 0 else {}


def _reply(payload):
    sys.stdout.write(PROTOCOL_PREFIX + json.dumps(payload) + "\n")
    sys.stdout.flush()


class RenderServer:
    def __init__(self, config, modern_opengl=True, trail_length=300,
                 assets=True, assets_path=None):
        options = "" if modern_opengl else "--opengl2"
        self.client = bullet.connect(bullet.GUI, options=options)
        if self.client < 0:
            raise RuntimeError("could not connect to the PyBullet GUI")
        self.config = config
        self.extent = float(config["arena_half_extent"])
        self.altitude = float(config["flight_height"])
        self.control_hz = float(config["control_hz"])
        self.obstacles = config["virtual_obstacles"]
        self.obstacle_margin = float(config["obstacle_safety_margin"])
        self.use_assets = bool(assets) and _assets is not None
        if assets_path and _assets is not None:
            os.environ["MTV2_PYBULLET_ASSETS"] = str(assets_path)
        self.trails_enabled = True
        self.trails = [deque(maxlen=trail_length) for _ in range(4)]
        self.trail_lines = []
        self.formation_lines = [-1, -1, -1]
        self.status_text = -1
        self.drone_ids = []
        self.goal_items = []
        self.goal_bodies = []
        self.episode_markers = []
        time.sleep(0.5)
        bullet.setGravity(0.0, 0.0, -9.81, physicsClientId=self.client)
        bullet.setTimeStep(1.0 / self.control_hz, physicsClientId=self.client)
        for _ in range(10):
            bullet.stepSimulation(physicsClientId=self.client)
        self.configure_camera()
        self._create_arena()
        self._create_obstacles()
        colors = PURSUER_COLORS + (EVADER_COLOR,)
        self.drone_ids = [self._create_drone(color) for color in colors]

    def configure_camera(self):
        bullet.resetDebugVisualizerCamera(
            cameraDistance=max(4.8, self.extent * 2.2),
            cameraYaw=50.0,
            cameraPitch=-38.0,
            cameraTargetPosition=[0.0, 0.0, 0.65],
            physicsClientId=self.client,
        )

    def _create_arena(self):
        if self.use_assets:
            self._create_asset_arena()
            return
        self._create_geometry_arena()

    def _create_geometry_arena(self):
        """Legacy look: light floor slab + grid lines + dark boundary."""
        floor_visual = bullet.createVisualShape(
            bullet.GEOM_BOX,
            halfExtents=[self.extent, self.extent, 0.015],
            rgbaColor=[0.90, 0.92, 0.94, 1.0],
            specularColor=[0.1, 0.1, 0.1],
            physicsClientId=self.client,
        )
        bullet.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=floor_visual,
            basePosition=[0.0, 0.0, -0.02],
            physicsClientId=self.client,
        )
        grid_color = [0.72, 0.75, 0.78]
        for index in range(13):
            value = -self.extent + 2.0 * self.extent * index / 12.0
            bullet.addUserDebugLine(
                [value, -self.extent, 0.002],
                [value, self.extent, 0.002],
                grid_color,
                lineWidth=1.0,
                physicsClientId=self.client,
            )
            bullet.addUserDebugLine(
                [-self.extent, value, 0.002],
                [self.extent, value, 0.002],
                grid_color,
                lineWidth=1.0,
                physicsClientId=self.client,
            )
        self._draw_boundary()

    def _create_asset_arena(self):
        """gym-pybullet-drones look: textured ground plane + arena fence."""
        try:
            _assets.load_ground(bullet, self.client, self.extent)
        except Exception as exc:  # pragma: no cover - defensive fallback
            print(f"[render] plane.urdf unavailable, using plain floor ({exc})")
            floor_visual = bullet.createVisualShape(
                bullet.GEOM_BOX,
                halfExtents=[self.extent, self.extent, 0.015],
                rgbaColor=[0.78, 0.80, 0.82, 1.0],
                specularColor=[0.1, 0.1, 0.1],
                physicsClientId=self.client,
            )
            bullet.createMultiBody(
                baseMass=0.0,
                baseVisualShapeIndex=floor_visual,
                basePosition=[0.0, 0.0, -0.02],
                physicsClientId=self.client,
            )
        _assets.load_fence(bullet, self.client, self.extent)
        self._draw_boundary(raised=0.06)

    def _draw_boundary(self, raised=0.015, color=None, width=3.0):
        color = color or [0.20, 0.24, 0.28]
        corners = (
            (-self.extent, -self.extent),
            (self.extent, -self.extent),
            (self.extent, self.extent),
            (-self.extent, self.extent),
        )
        for index in range(4):
            start = corners[index]
            end = corners[(index + 1) % 4]
            bullet.addUserDebugLine(
                [start[0], start[1], raised],
                [end[0], end[1], raised],
                color,
                lineWidth=width,
                physicsClientId=self.client,
            )

    def _add_circle(self, x, y, radius, z, color, width):
        points = []
        for index in range(33):
            angle = 2.0 * math.pi * index / 32.0
            points.append(
                [
                    x + radius * math.cos(angle),
                    y + radius * math.sin(angle),
                    z,
                ]
            )
        for start, end in zip(points[:-1], points[1:]):
            bullet.addUserDebugLine(
                start,
                end,
                color,
                lineWidth=width,
                physicsClientId=self.client,
            )

    def _create_obstacles(self):
        height = max(1.8, self.altitude * 1.8)
        core_color = [0.92, 0.22, 0.04]
        safety_color = [1.0, 0.58, 0.05]
        for index, item in enumerate(self.obstacles, start=1):
            x, y, radius = (float(value) for value in item)
            safe_radius = radius + self.obstacle_margin
            if self.use_assets:
                # Reference 6V2 GUI: clean translucent cylinder, without the
                # legacy diagnostic rings and oversized obstacle label.
                _assets.load_obstacle(
                    bullet, self.client, x, y, radius, height,
                    model="architrave.urdf",
                )
            else:
                self._add_circle(x, y, radius, 0.02, core_color, 3.0)
                self._add_circle(x, y, radius, height, core_color, 3.0)
                self._add_circle(
                    x, y, safe_radius, self.altitude, safety_color, 2.0
                )
                for line_index in range(8):
                    angle = 2.0 * math.pi * line_index / 8.0
                    edge_x = x + radius * math.cos(angle)
                    edge_y = y + radius * math.sin(angle)
                    bullet.addUserDebugLine(
                        [edge_x, edge_y, 0.02],
                        [edge_x, edge_y, height],
                        core_color,
                        lineWidth=2.0,
                        physicsClientId=self.client,
                    )
                label_z = height + 0.12
                bullet.addUserDebugText(
                    f"virtual obstacle {index}",
                    [x, y, label_z],
                    textColorRGB=[0.50, 0.12, 0.02],
                    textSize=1.1,
                    physicsClientId=self.client,
                )

    def _create_drone(self, color):
        if self.use_assets:
            # Crazyflie 2X mesh from gym-pybullet-drones, tinted per-role
            return _assets.load_drone(
                bullet, self.client, color=color,
                base_position=(0.0, 0.0, self.altitude),
            )
        return self._create_geometry_drone(color)

    def _create_geometry_drone(self, color):
        """Legacy primitive quadcopter look (cylinder body + box arms)."""
        body_visual = bullet.createVisualShape(
            bullet.GEOM_CYLINDER,
            radius=0.040,
            length=0.025,
            rgbaColor=color,
            physicsClientId=self.client,
        )
        arm_visual = bullet.createVisualShape(
            bullet.GEOM_BOX,
            halfExtents=[0.058, 0.006, 0.006],
            rgbaColor=[0.12, 0.12, 0.14, 1.0],
            physicsClientId=self.client,
        )
        rotor_visual = bullet.createVisualShape(
            bullet.GEOM_CYLINDER,
            radius=0.026,
            length=0.004,
            rgbaColor=[0.15, 0.16, 0.18, 0.88],
            physicsClientId=self.client,
        )
        visuals = [arm_visual, arm_visual] + [rotor_visual] * 4
        positions = [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.052, 0.0, 0.010],
            [-0.052, 0.0, 0.010],
            [0.0, 0.052, 0.010],
            [0.0, -0.052, 0.010],
        ]
        identity = [0.0, 0.0, 0.0, 1.0]
        quarter_turn = bullet.getQuaternionFromEuler([0.0, 0.0, math.pi / 2.0])
        orientations = [identity, quarter_turn] + [identity] * 4
        count = len(visuals)
        return bullet.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=body_visual,
            basePosition=[0.0, 0.0, self.altitude],
            linkMasses=[0.0] * count,
            linkCollisionShapeIndices=[-1] * count,
            linkVisualShapeIndices=visuals,
            linkPositions=positions,
            linkOrientations=orientations,
            linkInertialFramePositions=[[0.0, 0.0, 0.0]] * count,
            linkInertialFrameOrientations=[identity] * count,
            linkParentIndices=[0] * count,
            linkJointTypes=[bullet.JOINT_FIXED] * count,
            linkJointAxis=[[0.0, 0.0, 1.0]] * count,
            physicsClientId=self.client,
        )

    def clear_trails(self):
        for line_id in self.trail_lines:
            bullet.removeUserDebugItem(line_id, physicsClientId=self.client)
        self.trail_lines.clear()
        for trail in self.trails:
            trail.clear()

    def toggle_trails(self):
        self.trails_enabled = not self.trails_enabled
        if not self.trails_enabled:
            self.clear_trails()

    def set_scenario(self, message):
        for item in self.goal_items:
            bullet.removeUserDebugItem(item, physicsClientId=self.client)
        for item in self.episode_markers:
            bullet.removeUserDebugItem(item, physicsClientId=self.client)
        for body in self.goal_bodies:
            bullet.removeBody(body, physicsClientId=self.client)
        self.goal_items = []
        self.goal_bodies = []
        self.episode_markers = []
        goal = message["goal"]
        extra = message.get("extra") or {}
        if extra.get("mouths") or not self.use_assets:
            self.goal_items.append(
                bullet.addUserDebugText(
                    "EVADER GOAL",
                    [goal[0], goal[1], self.altitude + 0.12],
                    textColorRGB=[0.95, 0.70, 0.02],
                    textSize=0.75,
                    physicsClientId=self.client,
                )
            )
        start = message.get("evader_start")
        if start is not None:
            self.goal_items.append(
                bullet.addUserDebugLine(
                    [start[0], start[1], self.altitude],
                    [goal[0], goal[1], self.altitude],
                    [0.95, 0.70, 0.02],
                    lineWidth=2.0,
                    physicsClientId=self.client,
                )
            )
        if not extra.get("mouths"):
            radius = float(self.config.get("evader_goal_radius", 0.30))
            if self.use_assets:
                self.goal_bodies.append(
                    _assets.load_goal(bullet, self.client, goal, radius)
                )
            else:
                self._add_ring_marker(
                    goal[0], goal[1], radius, self.altitude,
                    [0.95, 0.70, 0.02], 3.0,
                )
        self._draw_scenario_extra(extra)

    def _draw_scenario_extra(self, extra):
        """MtV2 split-coverage markers: escape mouths, decoy, hidden side."""
        if not extra:
            return
        ring_radius = float(extra.get("mouth_radius", 0.30))
        for mouth in extra.get("mouths", ()):
            x, y = (float(value) for value in mouth)
            self.episode_markers.append(
                bullet.addUserDebugText(
                    "MOUTH", [x, y - ring_radius - 0.18, self.altitude],
                    textColorRGB=[0.04, 0.55, 0.28],
                    textSize=1.0,
                    physicsClientId=self.client,
                )
            )
            self._add_ring_marker(
                x, y, ring_radius, self.altitude,
                [0.04, 0.60, 0.30], 2.4,
            )
        decoy = extra.get("decoy")
        if decoy is not None:
            x, y = (float(value) for value in decoy)
            self.episode_markers.append(
                bullet.addUserDebugText(
                    "DECOY", [x, y - ring_radius - 0.18, self.altitude],
                    textColorRGB=[0.91, 0.44, 0.04],
                    textSize=1.0,
                    physicsClientId=self.client,
                )
            )
            self._add_ring_marker(
                x, y, ring_radius, self.altitude,
                [0.91, 0.44, 0.04], 2.4,
            )
        escape = extra.get("escape")
        if escape is not None:
            # gold inner ring highlights the true hidden mouth this episode
            x, y = (float(value) for value in escape)
            self._add_ring_marker(
                x, y, max(0.14, ring_radius * 0.55), self.altitude,
                [0.95, 0.70, 0.02], 3.0,
            )
        info = extra.get("info")
        if info:
            self.episode_markers.append(
                bullet.addUserDebugText(
                    str(info),
                    [-self.extent, self.extent - 0.25, 0.05],
                    textColorRGB=[0.10, 0.12, 0.16],
                    textSize=1.3,
                    physicsClientId=self.client,
                )
            )

    def _add_ring_marker(self, x, y, radius, z, color, width):
        """Add a debug circle kept in self.episode_markers (so it is cleared
        together with the other per-episode markers)."""
        segments = 48
        for index in range(segments):
            angle_0 = 2.0 * math.pi * index / segments
            angle_1 = 2.0 * math.pi * (index + 1) / segments
            start = [x + radius * math.cos(angle_0),
                     y + radius * math.sin(angle_0), z]
            end = [x + radius * math.cos(angle_1),
                   y + radius * math.sin(angle_1), z]
            line_id = bullet.addUserDebugLine(
                start, end, color, lineWidth=width,
                physicsClientId=self.client,
            )
            self.episode_markers.append(line_id)

    def update(self, message):
        positions = message["positions"]
        world = [[row[0], row[1], self.altitude] for row in positions]
        for drone_id, position in zip(self.drone_ids, world):
            bullet.resetBasePositionAndOrientation(
                drone_id,
                position,
                [0.0, 0.0, 0.0, 1.0],
                physicsClientId=self.client,
            )
        for index in range(3):
            self.formation_lines[index] = bullet.addUserDebugLine(
                world[index],
                world[(index + 1) % 3],
                [0.05, 0.80, 0.85],
                lineWidth=2.0,
                **replacement_kwargs(self.formation_lines[index]),
                physicsClientId=self.client,
            )
        step = int(message["step"])
        if self.trails_enabled and step % 3 == 0:
            self._update_trails(world)
        state = "PAUSED" if message["paused"] else "RUNNING"
        if message["captured"]:
            state = "CAPTURED"
        text = (
            f"3V1 | episode {message['episode']} | step {step} | {state}\n"
            f"control={self.control_hz:.0f} Hz\n"
            "SPACE pause  N single-step  R reset  T trails  C camera  Q quit"
        )
        self.status_text = bullet.addUserDebugText(
            text,
            [-self.extent, -self.extent, 1.95],
            textColorRGB=[0.05, 0.08, 0.12],
            textSize=0.65,
            **replacement_kwargs(self.status_text),
            physicsClientId=self.client,
        )
        bullet.stepSimulation(physicsClientId=self.client)

    def _update_trails(self, positions):
        for index, position in enumerate(positions):
            trail = self.trails[index]
            if trail:
                line_id = bullet.addUserDebugLine(
                    trail[-1],
                    position,
                    TRAIL_COLORS[index],
                    lineWidth=1.5,
                    physicsClientId=self.client,
                )
                self.trail_lines.append(line_id)
            trail.append(position)
        maximum = sum(trail.maxlen for trail in self.trails)
        while len(self.trail_lines) > maximum:
            line_id = self.trail_lines.pop(0)
            bullet.removeUserDebugItem(line_id, physicsClientId=self.client)

    def keyboard(self):
        events = bullet.getKeyboardEvents(physicsClientId=self.client)
        triggered = bullet.KEY_WAS_TRIGGERED
        return [key for key, state in events.items() if state & triggered]

    def close(self):
        if bullet.isConnected(self.client):
            bullet.disconnect(self.client)


def main():
    first_line = sys.stdin.readline()
    if not first_line:
        return
    initialization = json.loads(first_line)
    server = RenderServer(
        initialization["config"],
        modern_opengl=initialization.get("modern_opengl", True),
        trail_length=initialization.get("trail_length", 300),
        assets=initialization.get("assets", True),
        assets_path=initialization.get("assets_path"),
    )
    _reply({"ready": True})
    try:
        for line in sys.stdin:
            message = json.loads(line)
            command = message.get("command")
            if command == "close":
                break
            if command == "update":
                server.update(message)
            elif command == "reset":
                server.clear_trails()
            elif command == "toggle_trails":
                server.toggle_trails()
            elif command == "camera":
                server.configure_camera()
            elif command == "scenario":
                server.set_scenario(message)
            _reply({"events": server.keyboard()})
    finally:
        server.close()


if __name__ == "__main__":
    main()
