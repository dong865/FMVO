"""PyBullet 3D viewer for the unchanged lightweight 3V1 environment."""

import argparse
from collections import deque
import time

import numpy as np

from .environment import ThreeVOneEnv
from .features import point_in_triangle
from .obstacles import configure_obstacles
from .matplotlib_viewer import MatplotlibViewer
from .pybullet_process_viewer import ProcessViewer
from .runtime import BASELINES, load_runtime
from .scenario import scene_preset

try:
    from . import pybullet_assets as _assets
except Exception:  # pragma: no cover - defensive
    _assets = None


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


def _pybullet():
    try:
        import pybullet as bullet
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "PyBullet is required for the 3D viewer: pip install pybullet"
        ) from error
    except ImportError as error:
        raise RuntimeError(
            "PyBullet could not load. Check that its binary is "
            "compatible with the active NumPy version (commonly numpy<2)."
        ) from error
    return bullet


class PyBulletViewer:
    """Render environment state without participating in its dynamics."""

    def __init__(
        self, config, headless=False, trail_length=300, modern_opengl=True,
        assets=True, assets_path=None,
    ):
        self.bullet = _pybullet()
        connection_mode = self.bullet.DIRECT if headless else self.bullet.GUI
        options = "" if headless or modern_opengl else "--opengl2"
        self.client = self.bullet.connect(connection_mode, options=options)
        if self.client < 0:
            raise RuntimeError("could not connect to the PyBullet GUI")
        if not headless:
            time.sleep(1.0)
        self.config = config
        self.headless = headless
        self.trails_enabled = True
        self.trails = [deque(maxlen=trail_length) for _ in range(4)]
        self.trail_lines = []
        self.formation_lines = [-1, -1, -1]
        self.status_text = -1
        self.drone_ids = []
        self.episode_markers = []
        self.goal_bodies = []
        self.use_assets = bool(assets) and _assets is not None
        if assets_path and _assets is not None:
            import os

            os.environ["MTV2_PYBULLET_ASSETS"] = str(assets_path)
        self._configure_camera()
        self._build_scene()

    def _configure_camera(self):
        extent = self.config.arena_half_extent
        self.bullet.resetDebugVisualizerCamera(
            cameraDistance=max(4.8, extent * 2.2),
            cameraYaw=50.0,
            cameraPitch=-38.0,
            cameraTargetPosition=[0.0, 0.0, 0.65],
            physicsClientId=self.client,
        )

    def _build_scene(self):
        bullet = self.bullet
        bullet.resetSimulation(physicsClientId=self.client)
        bullet.setGravity(0.0, 0.0, -9.81, physicsClientId=self.client)
        bullet.setTimeStep(self.config.dt, physicsClientId=self.client)
        bullet.configureDebugVisualizer(
            bullet.COV_ENABLE_GUI, 1, physicsClientId=self.client
        )
        bullet.configureDebugVisualizer(
            bullet.COV_ENABLE_SHADOWS, 1, physicsClientId=self.client
        )
        self._create_arena()
        self._create_obstacles()
        colors = PURSUER_COLORS + (EVADER_COLOR,)
        self.drone_ids = [self._create_drone(color) for color in colors]

    def _create_arena(self):
        if self.use_assets:
            self._create_asset_arena()
            return
        self._create_geometry_arena()

    def _create_geometry_arena(self):
        bullet = self.bullet
        extent = self.config.arena_half_extent
        floor_visual = bullet.createVisualShape(
            bullet.GEOM_BOX,
            halfExtents=[extent, extent, 0.015],
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
        grid_values = np.linspace(-extent, extent, 13)
        for value in grid_values:
            bullet.addUserDebugLine(
                [float(value), -extent, 0.002],
                [float(value), extent, 0.002],
                grid_color,
                lineWidth=1.0,
                physicsClientId=self.client,
            )
            bullet.addUserDebugLine(
                [-extent, float(value), 0.002],
                [extent, float(value), 0.002],
                grid_color,
                lineWidth=1.0,
                physicsClientId=self.client,
            )
        self._draw_boundary(raised=0.015)

    def _create_asset_arena(self):
        """gym-pybullet-drones look: textured ground plane + arena fence."""
        bullet = self.bullet
        extent = self.config.arena_half_extent
        try:
            _assets.load_ground(bullet, self.client, extent)
        except Exception as exc:  # pragma: no cover - defensive fallback
            print(f"[viewer] plane.urdf unavailable, plain floor ({exc})")
            floor_visual = bullet.createVisualShape(
                bullet.GEOM_BOX,
                halfExtents=[extent, extent, 0.015],
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
        _assets.load_fence(bullet, self.client, extent)
        self._draw_boundary(raised=0.06, color=[0.55, 0.58, 0.62], width=2.0)

    def _draw_boundary(self, raised=0.015, color=None, width=3.0):
        bullet = self.bullet
        extent = self.config.arena_half_extent
        color = color or [0.20, 0.24, 0.28]
        corners = (
            (-extent, -extent),
            (extent, -extent),
            (extent, extent),
            (-extent, extent),
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
        angles = np.linspace(0.0, 2.0 * np.pi, 33)
        points = [
            [x + radius * np.cos(angle), y + radius * np.sin(angle), z]
            for angle in angles
        ]
        for start, end in zip(points[:-1], points[1:]):
            self.bullet.addUserDebugLine(
                start,
                end,
                color,
                lineWidth=width,
                physicsClientId=self.client,
            )

    def _create_obstacles(self):
        bullet = self.bullet
        height = max(1.8, self.config.flight_height * 1.8)
        core_color = [0.92, 0.22, 0.04]
        safety_color = [1.0, 0.58, 0.05]
        for index, (x, y, radius) in enumerate(
            self.config.virtual_obstacles, start=1
        ):
            safe_radius = radius + self.config.obstacle_safety_margin
            if self.use_assets:
                _assets.load_obstacle(
                    bullet, self.client, x, y, radius, height,
                    model="architrave.urdf",
                )
            else:
                self._add_circle(x, y, radius, 0.02, core_color, 3.0)
                self._add_circle(x, y, radius, height, core_color, 3.0)
                self._add_circle(
                    x,
                    y,
                    safe_radius,
                    self.config.flight_height,
                    safety_color,
                    2.0,
                )
                for angle in np.linspace(
                    0.0, 2.0 * np.pi, 8, endpoint=False
                ):
                    edge_x = x + radius * np.cos(angle)
                    edge_y = y + radius * np.sin(angle)
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
            return _assets.load_drone(
                self.bullet, self.client, color=color,
                base_position=(0.0, 0.0, self.config.flight_height),
            )
        return self._create_geometry_drone(color)

    def _create_geometry_drone(self, color):
        """Legacy primitive quadcopter look (cylinder body + box arms)."""
        bullet = self.bullet
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
        link_visuals = [arm_visual, arm_visual] + [rotor_visual] * 4
        link_positions = [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.052, 0.0, 0.010],
            [-0.052, 0.0, 0.010],
            [0.0, 0.052, 0.010],
            [0.0, -0.052, 0.010],
        ]
        identity = [0.0, 0.0, 0.0, 1.0]
        quarter_turn = bullet.getQuaternionFromEuler([0.0, 0.0, np.pi / 2.0])
        link_orientations = [identity, quarter_turn] + [identity] * 4
        count = len(link_visuals)
        return bullet.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=body_visual,
            basePosition=[0.0, 0.0, self.config.flight_height],
            linkMasses=[0.0] * count,
            linkCollisionShapeIndices=[-1] * count,
            linkVisualShapeIndices=link_visuals,
            linkPositions=link_positions,
            linkOrientations=link_orientations,
            linkInertialFramePositions=[[0.0, 0.0, 0.0]] * count,
            linkInertialFrameOrientations=[identity] * count,
            linkParentIndices=[0] * count,
            linkJointTypes=[bullet.JOINT_FIXED] * count,
            linkJointAxis=[[0.0, 0.0, 1.0]] * count,
            physicsClientId=self.client,
        )

    def reset_episode(self):
        self.clear_trails()

    def set_scenario(self, goal, obstacles, evader_start=None, extra=None):
        bullet = self.bullet
        del obstacles
        extra = extra or {}
        for item in self.episode_markers:
            bullet.removeUserDebugItem(item, physicsClientId=self.client)
        for body in self.goal_bodies:
            bullet.removeBody(body, physicsClientId=self.client)
        self.episode_markers = []
        self.goal_bodies = []
        goal = np.asarray(goal, dtype=float)
        if not extra.get("mouths"):
            if self.use_assets:
                self.goal_bodies.append(_assets.load_goal(
                    bullet, self.client, goal,
                    self.config.evader_goal_radius,
                ))
            else:
                self._add_ring_marker(
                    goal[0], goal[1], self.config.evader_goal_radius,
                    self.config.flight_height, [0.95, 0.70, 0.02], 3.0,
                )
        if evader_start is not None:
            start = np.asarray(evader_start, dtype=float)
            line_id = bullet.addUserDebugLine(
                [start[0], start[1], self.config.flight_height],
                [goal[0], goal[1], self.config.flight_height],
                [0.95, 0.70, 0.02],
                lineWidth=2.0,
                physicsClientId=self.client,
            )
            self.episode_markers.append(line_id)
        self._draw_scenario_extra(extra)

    def _add_ring_marker(self, x, y, radius, z, color, width):
        bullet = self.bullet
        for index in range(48):
            angle_0 = 2.0 * np.pi * index / 48.0
            angle_1 = 2.0 * np.pi * (index + 1) / 48.0
            start = [x + radius * np.cos(angle_0),
                     y + radius * np.sin(angle_0), z]
            end = [x + radius * np.cos(angle_1),
                   y + radius * np.sin(angle_1), z]
            line_id = bullet.addUserDebugLine(
                start, end, color, lineWidth=width,
                physicsClientId=self.client,
            )
            self.episode_markers.append(line_id)

    def _draw_scenario_extra(self, extra):
        """MtV2 split-coverage markers: escape mouths, decoy, hidden side."""
        bullet = self.bullet
        if not extra:
            return
        ring_radius = float(extra.get("mouth_radius", 0.30))
        for mouth in extra.get("mouths", ()):
            x, y = (float(value) for value in mouth)
            label_id = bullet.addUserDebugText(
                "MOUTH", [x, y - ring_radius - 0.18, self.config.flight_height],
                textColorRGB=[0.04, 0.55, 0.28],
                textSize=1.0,
                physicsClientId=self.client,
            )
            self.episode_markers.append(label_id)
            self._add_ring_marker(
                x, y, ring_radius, self.config.flight_height,
                [0.04, 0.60, 0.30], 2.4,
            )
        decoy = extra.get("decoy")
        if decoy is not None:
            x, y = (float(value) for value in decoy)
            label_id = bullet.addUserDebugText(
                "DECOY", [x, y - ring_radius - 0.18, self.config.flight_height],
                textColorRGB=[0.91, 0.44, 0.04],
                textSize=1.0,
                physicsClientId=self.client,
            )
            self.episode_markers.append(label_id)
            self._add_ring_marker(
                x, y, ring_radius, self.config.flight_height,
                [0.91, 0.44, 0.04], 2.4,
            )
        escape = extra.get("escape")
        if escape is not None:
            # gold inner ring highlights the true hidden mouth this episode
            x, y = (float(value) for value in escape)
            self._add_ring_marker(
                x, y, max(0.14, ring_radius * 0.55), self.config.flight_height,
                [0.95, 0.70, 0.02], 3.0,
            )
        info = extra.get("info")
        if info:
            text_id = bullet.addUserDebugText(
                str(info),
                [-self.config.arena_half_extent,
                 self.config.arena_half_extent - 0.25, 0.05],
                textColorRGB=[0.10, 0.12, 0.16],
                textSize=1.3,
                physicsClientId=self.client,
            )
            self.episode_markers.append(text_id)

    def clear_trails(self):
        for line_id in self.trail_lines:
            self.bullet.removeUserDebugItem(
                line_id, physicsClientId=self.client
            )
        self.trail_lines.clear()
        for trail in self.trails:
            trail.clear()

    def toggle_trails(self):
        self.trails_enabled = not self.trails_enabled
        if not self.trails_enabled:
            self.clear_trails()

    def update(self, positions, step, episode, captured, paused):
        altitude = self.config.flight_height
        world_positions = np.column_stack(
            (positions[:, 0], positions[:, 1], np.full(4, altitude))
        )
        for drone_id, position in zip(self.drone_ids, world_positions):
            self.bullet.resetBasePositionAndOrientation(
                drone_id,
                position.tolist(),
                [0.0, 0.0, 0.0, 1.0],
                physicsClientId=self.client,
            )
        self._update_formation(world_positions)
        if self.trails_enabled and step % 3 == 0:
            self._update_trails(world_positions)
        self._update_status(positions, step, episode, captured, paused)
        self.bullet.stepSimulation(physicsClientId=self.client)

    def _update_formation(self, positions):
        color = [0.05, 0.80, 0.85]
        for index in range(3):
            start = positions[index].tolist()
            end = positions[(index + 1) % 3].tolist()
            self.formation_lines[index] = self.bullet.addUserDebugLine(
                start,
                end,
                color,
                lineWidth=2.0,
                **replacement_kwargs(self.formation_lines[index]),
                physicsClientId=self.client,
            )

    def _update_trails(self, positions):
        for index, position in enumerate(positions):
            trail = self.trails[index]
            if trail:
                line_id = self.bullet.addUserDebugLine(
                    trail[-1],
                    position.tolist(),
                    TRAIL_COLORS[index],
                    lineWidth=1.5,
                    physicsClientId=self.client,
                )
                self.trail_lines.append(line_id)
            trail.append(position.tolist())
        maximum_lines = sum(trail.maxlen for trail in self.trails)
        while len(self.trail_lines) > maximum_lines:
            line_id = self.trail_lines.pop(0)
            self.bullet.removeUserDebugItem(
                line_id, physicsClientId=self.client
            )

    def _update_status(self, positions, step, episode, captured, paused):
        enclosed = point_in_triangle(positions[3], positions[:3])
        state = "PAUSED" if paused else "RUNNING"
        if captured:
            state = "CAPTURED"
        text = (
            f"3V1 | episode {episode} | step {step} | {state}\n"
            f"enclosed={enclosed} | control={self.config.control_hz:.0f} Hz\n"
            "SPACE pause  N single-step  R reset  T trails  C camera  Q quit"
        )
        self.status_text = self.bullet.addUserDebugText(
            text,
            [
                -self.config.arena_half_extent,
                -self.config.arena_half_extent,
                1.95,
            ],
            textColorRGB=[0.05, 0.08, 0.12],
            textSize=0.65,
            **replacement_kwargs(self.status_text),
            physicsClientId=self.client,
        )

    def keyboard(self):
        if self.headless:
            return set()
        events = self.bullet.getKeyboardEvents(physicsClientId=self.client)
        triggered = self.bullet.KEY_WAS_TRIGGERED
        return {key for key, state in events.items() if state & triggered}

    def close(self):
        if self.bullet.isConnected(self.client):
            self.bullet.disconnect(self.client)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", nargs="?", default=None)
    parser.add_argument("--baseline", choices=BASELINES, default="game_theory")
    parser.add_argument("--obstacles-file", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--scene-preset",
        choices=(
            "legacy", "hard", "hard_flanks", "hard_route", "hard_both",
            "hard_obstacles", "far1", "far2", "far3", "hard_gap",
            "hard_choke", "gui_scattered",
        ),
        default="gui_scattered",
        help="GUI episode geometry (gui_scattered independently distributes "
             "pursuers across the map; legacy restores the original arc)",
    )
    parser.add_argument("--episodes", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--trail-length", type=int, default=300)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--renderer",
        choices=("matplotlib", "pybullet"),
        default="matplotlib",
    )
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument(
        "--modern-opengl",
        dest="modern_opengl", action="store_true", default=True,
        help="use native OpenGL3+ rendering (default)",
    )
    parser.add_argument(
        "--opengl2",
        dest="modern_opengl", action="store_false",
        help="use limited OpenGL2 fallback when OpenGL3+ cannot start",
    )
    parser.add_argument(
        "--no-assets", dest="assets", action="store_false", default=True,
        help="render with plain debug geometry instead of gym-pybullet-drones "
             "meshes (ground plane, fence, pillars, Crazyflie drones)",
    )
    parser.add_argument(
        "--assets-path", default=None,
        help="override the gym-pybullet-drones asset directory "
             "(default: /media/yudong/data/gym-pybullet-drones-main or "
             "$MTV2_PYBULLET_ASSETS)",
    )
    return parser.parse_args()


def run(args):
    if args.speed <= 0.0:
        raise ValueError("--speed must be positive")
    if args.episodes < 0:
        raise ValueError("--episodes cannot be negative")
    runtime = load_runtime(args.baseline, args.checkpoint, device=args.device)
    config = runtime.config
    if args.obstacles_file:
        configure_obstacles(config, args.obstacles_file)
    environment = ThreeVOneEnv(
        config,
        seed=args.seed,
        scene_options=scene_preset(args.scene_preset),
    )
    if args.headless:
        viewer = PyBulletViewer(config, headless=True,
                                assets=args.assets,
                                assets_path=args.assets_path)
    elif args.renderer == "pybullet":
        viewer = ProcessViewer(
            config,
            args.trail_length,
            modern_opengl=args.modern_opengl,
            assets=args.assets,
            assets_path=args.assets_path,
        )
    else:
        viewer = MatplotlibViewer(config, args.trail_length)
    max_steps = args.max_steps or config.max_episode_steps
    episode = 1
    step = 0
    paused = False
    single_step = False
    captured = False

    def reset_episode():
        nonlocal step, captured
        environment.reset()
        runtime.reset()
        viewer.reset_episode()
        viewer.set_scenario(
            environment.evader_goal,
            environment.virtual_obstacles,
            environment.current_scenario.evader_start,
        )
        step = 0
        captured = False

    reset_episode()
    viewer.update(environment.positions, step, episode, captured, paused)
    try:
        while args.episodes == 0 or episode <= args.episodes:
            frame_start = time.monotonic()
            keys = viewer.keyboard()
            if ord("q") in keys:
                break
            if ord(" ") in keys:
                paused = not paused
            if ord("r") in keys:
                reset_episode()
            if ord("t") in keys:
                viewer.toggle_trails()
            if ord("c") in keys:
                viewer._configure_camera()
            if ord("n") in keys:
                single_step = True

            should_step = not paused or single_step
            if should_step:
                measured_positions, measured_velocities = (
                    environment.control_state()
                )
                actions = runtime.actions(
                    measured_positions,
                    measured_velocities,
                    environment.origin,
                    environment.virtual_obstacles,
                )
                _, _, terminated, truncated, info = environment.step(actions)
                step += 1
                captured = bool(info["captured"])
                single_step = False
                episode_done = terminated or truncated or step >= max_steps
                viewer.update(
                    environment.positions, step, episode, captured, paused
                )
                if episode_done:
                    print(
                        f"episode={episode} steps={step} "
                        f"captured={captured}"
                    )
                    episode += 1
                    if args.episodes and episode > args.episodes:
                        break
                    reset_episode()
            else:
                viewer.update(
                    environment.positions, step, episode, captured, paused
                )

            if not args.no_realtime:
                elapsed = time.monotonic() - frame_start
                delay = config.dt / args.speed - elapsed
                if delay > 0.0:
                    time.sleep(delay)
    finally:
        viewer.close()


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
