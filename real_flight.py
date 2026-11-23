"""Run a retrained 3V1 checkpoint through Crazyswarm2 cmdVelocityWorld."""

import argparse
import json
import time
from typing import List

import numpy as np
import torch

from .evader_potential_field import PotentialFieldEvader
from .flight_logger import FlightRecorder, file_sha256
from .obstacles import configure_obstacles
from .runtime import BASELINES, load_runtime


def schedule_velocity_replay(
    cfs, commands, timestamps, safety_monitor=None, sleep_fn=time.sleep
):
    """Send a recorded four-agent velocity sequence and always finish stopped."""
    commands = np.asarray(commands, dtype=np.float32)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if commands.ndim != 3 or commands.shape[1] != 4 or commands.shape[2] not in (2, 3):
        raise ValueError("commands must have shape [T,4,2|3]")
    if len(timestamps) != len(commands) or not np.all(np.diff(timestamps) > 0):
        raise ValueError("timestamps must be strictly increasing and match commands")
    try:
        for index, frame in enumerate(commands):
            if safety_monitor is not None and not safety_monitor(index):
                raise RuntimeError(f"safety monitor rejected replay frame {index}")
            for cf, command in zip(cfs, frame):
                velocity = np.zeros(3, dtype=np.float32)
                velocity[: len(command)] = command
                cf.cmdVelocityWorld(velocity.tolist(), yawRate=0.0)
            if index + 1 < len(commands):
                sleep_fn(float(timestamps[index + 1] - timestamps[index]))
    finally:
        for cf in cfs:
            cf.cmdVelocityWorld([0.0, 0.0, 0.0], yawRate=0.0)


def limit_rows(vectors: np.ndarray, maximum: float) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors * np.minimum(1.0, maximum / np.maximum(norms, 1e-8))


def select_roles(cfs, role_names: List[str]):
    by_name = {cf.prefix.lstrip("/"): cf for cf in cfs}
    missing = [name for name in role_names if name not in by_name]
    if missing:
        raise RuntimeError(
            f"roles {missing} are absent; " f"available Crazyflies: {sorted(by_name)}"
        )
    if len(set(role_names)) != 4:
        raise ValueError("--roles needs four distinct names: 3 pursuers, then evader")
    return [by_name[name] for name in role_names]


def evader_velocity(
    positions, current_velocity, origin, config, goal
) -> np.ndarray:
    acceleration = PotentialFieldEvader(config).acceleration(
        positions,
        np.asarray(goal, dtype=np.float32),
        origin,
        config.virtual_obstacles,
    )
    return limit_rows(
        (current_velocity + acceleration * config.dt)[None, :],
        config.evader_max_speed,
    )[0]


def apply_safety(velocity, positions, origin, config):
    """与仿真一致的成对/障碍物安全投影，外加场地边界保护。"""
    from .safety import project_obstacle_velocity, project_pairwise_velocity

    speed_limits = np.array(
        [config.max_speed] * config.num_pursuers + [config.evader_max_speed],
        dtype=np.float32,
    )
    pairwise = project_pairwise_velocity(
        velocity[:, :2],
        positions,
        warning_distance=config.safety_warning_distance,
        emergency_distance=config.safety_emergency_distance,
        separation_speed=config.safety_separation_speed,
        speed_limits=speed_limits,
    )
    obstacle = project_obstacle_velocity(
        pairwise.velocities,
        positions,
        origin,
        config.virtual_obstacles,
        safety_margin=config.obstacle_safety_margin,
        warning_margin=config.obstacle_warning_margin,
        separation_speed=config.safety_separation_speed,
        speed_limits=speed_limits,
    )
    out = velocity.copy()
    out[:, :2] = obstacle.velocities
    # 仿真中参考位置会被裁剪在场地内；实飞用边界推力做等效保护
    local = positions - origin
    margin = 0.20
    for i in range(4):
        for axis in range(2):
            if local[i, axis] > config.arena_half_extent - margin:
                out[i, axis] = min(out[i, axis], -0.05)
            elif local[i, axis] < -config.arena_half_extent + margin:
                out[i, axis] = max(out[i, axis], 0.05)
    return out


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint",
        nargs="?",
        help="trained checkpoint; omitted only for game_theory",
    )
    parser.add_argument("--baseline", choices=BASELINES, default="mappo_open")
    parser.add_argument(
        "--roles",
        default="cf1,cf2,cf3,cf4",
        help="three pursuer names followed by the evader name",
    )
    parser.add_argument("--duration", type=float, default=35.0)
    parser.add_argument("--takeoff-height", type=float, default=None)
    parser.add_argument("--max-speed", type=float, default=0.56)
    parser.add_argument("--max-accel", type=float, default=0.94)
    parser.add_argument(
        "--obstacles-file",
        default=None,
        help="JSON obstacle layout; uses checkpoint/default if omitted",
    )
    parser.add_argument(
        "--emergency-separation",
        type=float,
        default=0.12,
        help="pairwise hard-stop distance; below this ALL drones are frozen. "
             "Must match TaskConfig.safety_emergency_distance (0.12 m, i.e. "
             "slightly more than the 0.092 m airframe span). Larger values "
             "prevent the team from ever closing the capture ring.",
    )
    parser.add_argument(
        "--evader-mode", choices=("escape", "stationary"), default="escape"
    )
    parser.add_argument(
        "--evader-goal-x",
        type=float,
        default=1.9802,
        help="evader-private goal X relative to the mocap arena origin "
             "(default = fixed scenario evader_goal 1.9802)",
    )
    parser.add_argument(
        "--evader-goal-y",
        type=float,
        default=-1.9759,
        help="evader-private goal Y relative to the mocap arena origin "
             "(default = fixed scenario evader_goal -1.9759)",
    )
    parser.add_argument(
        "--scene-origin",
        type=float,
        nargs=2,
        default=None,
        metavar=("X", "Y"),
        help="固定场景原点（动捕绝对坐标）。给出后飞行坐标系原点取该点，"
             "不再用四机初始质心，用于按仿真原坐标复现场景；缺省 = 四机质心",
    )
    parser.add_argument(
        "--align-spawns",
        default=None,
        metavar="JSON",
        help="起飞后先微调到精确出生点的 JSON 文件：{\"P1\": [x, y], \"P2\": ..., "
             "\"P3\": ..., \"E\": ...}（坐标相对飞行原点）。给出后每架起飞后会"
             "先 goTo 到该点（小幅度修正），到位后再进入 50 Hz 闭环",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--log-dir",
        default="flight_data",
        help="directory for per-frame trajectory logs (default: ./flight_data)",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="log file tag (default: timestamp)",
    )
    parser.add_argument("--replay-data", default=None,
                        help="successful episode NPZ; bypasses online policy")
    parser.add_argument("--acknowledge-open-loop", action="store_true",
                        help="required confirmation that velocity replay can drift")
    parser.add_argument("--max-replay-speed", type=float, default=0.55)
    return parser.parse_args()


def _run_recorded_velocity(args):
    if not args.acknowledge_open_loop:
        raise SystemExit("--replay-data requires --acknowledge-open-loop")
    from crazyflie_py import Crazyswarm
    from .episode_artifact import load_episode
    from .mocap_bridge import Nokov

    artifact = load_episode(args.replay_data)
    commands = artifact.arrays["commanded_velocities"]
    speeds = np.linalg.norm(commands, axis=-1)
    if float(speeds.max()) > args.max_replay_speed:
        raise RuntimeError(f"recorded speed {speeds.max():.3f} exceeds "
                           f"--max-replay-speed {args.max_replay_speed:.3f}")
    role_names = [item.strip().lstrip("/") for item in args.roles.split(",")]
    swarm = Crazyswarm()
    cfs = select_roles(swarm.allcfs.crazyflies, role_names)
    mocap = Nokov(cfs)
    airborne = False
    low_level_started = False
    try:
        deadline = time.monotonic() + 8.0
        while not mocap.ready() and time.monotonic() < deadline:
            mocap.get_drone_state()
        if not mocap.ready():
            raise RuntimeError("not all replay roles have mocap transforms")
        initial = mocap.get_drone_state().numpy()
        origin = initial[:, :2].mean(axis=0)
        extent = float(artifact.metadata["arena_half_extent"])
        separation = float(artifact.metadata.get("safety_emergency_distance",
                                                  args.emergency_separation))
        for cf in cfs:
            cf.arm(True)
        swarm.timeHelper.sleep(1.0)
        height = args.takeoff_height or float(artifact.metadata.get("flight_height", 0.7))
        for cf in cfs:
            cf.takeoff(targetHeight=height, duration=2.5)
        airborne = True
        swarm.timeHelper.sleep(3.0)

        def safe(_index):
            state = mocap.get_drone_state().numpy()
            if mocap.stale():
                return False
            xy = state[:, :2]
            if np.any(np.abs(xy - origin) > extent):
                return False
            delta = xy[:, None] - xy[None, :]
            distance = np.linalg.norm(delta, axis=-1)
            distance[np.diag_indices(4)] = np.inf
            return bool(distance.min() >= separation)

        low_level_started = True
        print(f"[replay] open-loop source={args.replay_data} frames={len(commands)}")
        schedule_velocity_replay(cfs, commands, artifact.arrays["timestamps"],
                                 safety_monitor=safe)
    finally:
        if low_level_started:
            for cf in cfs:
                cf.notifySetpointsStop()
        if airborne:
            for cf in cfs:
                cf.land(targetHeight=0.05, duration=2.5)
            swarm.timeHelper.sleep(3.0)
        for cf in cfs:
            cf.arm(False)
        mocap.end_program()
        print("[done] replay stopped, landed and disarmed")


def _require_healthy(cfs, expected: int = 513, timeout: float = 6.0) -> None:
    """起飞前检查：四架 /health 的 supervisor.info 必须都等于 513。

    513 = can_be_armed(1) + 0x200；出现 640/672(crashed)、576(locked)、544(tumbled)
    或 0 时说明飞控状态异常，需要断电重开对应飞机。
    """
    import rclpy
    from crazyflie_interfaces.msg import LogDataGeneric

    names = [cf.prefix.lstrip("/") for cf in cfs]
    latest: Dict[str, float] = {}
    node = rclpy.create_node("preflight_health_check")
    try:
        def make_callback(name: str):
            def _callback(msg) -> None:
                if len(msg.values) >= 3:
                    latest[name] = float(msg.values[2])
            return _callback

        for name in names:
            node.create_subscription(
                LogDataGeneric, f"/{name}/health", make_callback(name), 10
            )
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            if all(name in latest for name in names):
                break
    finally:
        node.destroy_node()

    missing = [name for name in names if name not in latest]
    if missing:
        raise RuntimeError(f"起飞前检查失败：未收到 health 数据：{missing}")
    bad = {
        name: int(value)
        for name, value in latest.items()
        if abs(value - expected) > 0.5
    }
    if bad:
        raise RuntimeError(
            f"起飞前检查失败：supervisor.info 不是 {expected}：{bad}"
            "（640/672=crashed、576=locked、544=tumbled、0=未就绪 —— 请断电重开对应飞机）"
        )
    print(
        f"[preflight] supervisor.info 全部 {expected}："
        f"{ {n: int(v) for n, v in latest.items()} }",
        flush=True,
    )


def _record_snapshot(recorder, mocap, phase: str) -> None:
    """Record one pose-only frame (zero commands) for takeoff/align/landing."""
    if recorder is None:
        return
    try:
        state = mocap.get_drone_state().numpy()
    except Exception:
        return
    try:
        recorder.record(
            time.monotonic(),
            state[:, :2],
            state[:, 2],
            state[:, 7:9],
            np.zeros((4, 2), dtype=np.float32),
            phase=phase,
        )
    except OSError:
        pass


def main():
    args = parse_args()
    if args.replay_data:
        _run_recorded_velocity(args)
        return
    from crazyflie_py import Crazyswarm
    from .mocap_bridge import Nokov
    runtime = load_runtime(args.baseline, args.checkpoint, device=args.device)
    config = runtime.config
    if args.obstacles_file:
        configure_obstacles(config, args.obstacles_file)
    if abs(config.control_hz - 50.0) > 1e-6:
        raise RuntimeError(
            f"checkpoint was trained at {config.control_hz} Hz; "
            "this runner requires 50 Hz"
        )
    target_height = args.takeoff_height or config.flight_height
    role_names = [item.strip().lstrip("/") for item in args.roles.split(",")]
    if len(role_names) != 4:
        raise ValueError(
            "--roles requires exactly four comma-separated Crazyflie names"
        )

    swarm = Crazyswarm()
    time_helper = swarm.timeHelper
    allcfs = swarm.allcfs
    cfs = select_roles(allcfs.crazyflies, role_names)
    mocap = Nokov(cfs)
    airborne = False
    low_level_started = False
    recorder = None
    outcome = "completed"
    try:
        print("[preflight] waiting for all four /tf rigid bodies")
        deadline = time.monotonic() + 8.0
        while not mocap.ready() and time.monotonic() < deadline:
            mocap.get_drone_state()
        if not mocap.ready():
            unseen = [role_names[i] for i, seen in enumerate(mocap.seen) if not seen]
            raise RuntimeError(f"no mocap transform received for {unseen}")

        initial = mocap.get_drone_state().numpy()
        if args.scene_origin is not None:
            origin = np.asarray(args.scene_origin, dtype=np.float32)
            print(
                f"[preflight] scene-origin 模式：原点 = 动捕绝对坐标 "
                f"({origin[0]:+.3f}, {origin[1]:+.3f})，不再使用四机质心",
                flush=True,
            )
        else:
            origin = initial[:, :2].mean(axis=0)
        evader_goal = origin + np.asarray(
            (args.evader_goal_x, args.evader_goal_y), dtype=np.float32
        )
        if np.any(np.abs(evader_goal - origin) > config.arena_half_extent):
            raise ValueError("evader goal must lie inside the configured arena")
        for ox, oy, radius in config.virtual_obstacles:
            center = origin + np.array([ox, oy], dtype=np.float32)
            clearances = (
                np.linalg.norm(initial[:, :2] - center, axis=1)
                - radius
                - config.obstacle_safety_margin
            )
            if np.any(clearances <= 0.0):
                unsafe = np.where(clearances <= 0.0)[0].tolist()
                raise RuntimeError(
                    f"initial drones {unsafe} are inside virtual obstacle "
                    f"at local ({ox}, {oy})"
                )
        tag = args.tag or time.strftime("%m%d_%H%M%S")
        metadata = {
            "script": "three_v_one_baseline.real_flight",
            "baseline": args.baseline,
            "checkpoint": args.checkpoint,
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "roles": role_names,
            "origin": [round(float(value), 4) for value in origin],
            "origin_mode": "scene" if args.scene_origin is not None else "centroid",
            "align_spawns": args.align_spawns,
            "evader_goal": [round(float(value), 4) for value in evader_goal],
            "takeoff_height": target_height,
            "duration": args.duration,
            "max_speed": args.max_speed,
            "max_accel": args.max_accel,
            "emergency_separation": args.emergency_separation,
            "evader_mode": args.evader_mode,
            "virtual_obstacles": [
                list(map(float, item)) for item in config.virtual_obstacles
            ],
            "capture_radius": config.capture_radius,
            "capture_hold_steps": config.capture_hold_steps,
            "control_hz": config.control_hz,
        }
        recorder = FlightRecorder(args.log_dir, tag, metadata, config)
        print(f"[log] frame stream: {recorder.jsonl_path}")
        captured_once = False
        command_xy = np.zeros((4, 2), dtype=np.float32)
        runtime.reset()
        print(
            f"[preflight] origin={origin}, roles={role_names}, "
            f"virtual_obstacles={config.virtual_obstacles}, "
            f"evader_goal={evader_goal}, "
            "control=50 Hz"
        )
        _require_healthy(cfs)
        # 逐架顺序解锁（每次 arm 之间留 1 s，降低四机同时收包时的丢失概率）
        for index, cf in enumerate(cfs):
            print(f"[arm] {role_names[index]}: arm", flush=True)
            cf.arm(True)
            time_helper.sleep(1.0)
        time_helper.sleep(1.0)
        # 逐架顺序起飞：一架确认离地后再命令下一架，避免四机同时收指令时的丢包/冲突。
        # 每架同时发高层 takeoff 与速度爬升，兼容 enHighLevel=1/0 两种模式。
        airborne_flags = [False] * 4
        for index, cf in enumerate(cfs):
            print(f"[takeoff] {role_names[index]}: 发 takeoff", flush=True)
            cf.takeoff(targetHeight=target_height, duration=2.5)
            for _attempt in range(2):
                deadline = time.monotonic() + 4.0
                while time.monotonic() < deadline:
                    state = mocap.get_drone_state().numpy()
                    heights = state[:, 2]
                    for j, other in enumerate(cfs):
                        if j == index:
                            z_error = target_height - float(heights[j])
                            vertical = float(np.clip(1.2 * z_error, -config.max_vertical_speed, config.max_vertical_speed))
                            other.cmdVelocityWorld([0.0, 0.0, vertical], yawRate=0.0)
                        elif airborne_flags[j]:
                            vertical = float(np.clip(
                                1.2 * (target_height - float(heights[j])), -0.2, 0.2))
                            other.cmdVelocityWorld([0.0, 0.0, vertical], yawRate=0.0)
                    if float(heights[index]) >= 0.9 * target_height:
                        break
                    time.sleep(0.02)
                heights = mocap.get_drone_state().numpy()[:, 2]
                if float(heights[index]) >= 0.5 * target_height:
                    airborne_flags[index] = True
                    break
                print(
                    f"[takeoff] {role_names[index]} 未离地，重新 arm + 重发 takeoff "
                    f"(第 {_attempt + 1} 次)",
                    flush=True,
                )
                # 若刚才发生过复位，飞控参数会回到默认、且处于未解锁状态，
                # 因此先重新 arm 再发 takeoff。
                try:
                    cf.arm(True)
                except Exception:
                    pass
                time_helper.sleep(1.5)
                cf.takeoff(targetHeight=target_height, duration=2.5)
            if not airborne_flags[index]:
                raise RuntimeError(
                    f"takeoff failed for {role_names[index]} "
                    f"(height={float(mocap.get_drone_state().numpy()[index, 2]):.2f})"
                )
        airborne = True
        time_helper.sleep(1.0)
        takeoff_heights = mocap.get_drone_state().numpy()[:, 2]
        print(
            f"[flight] takeoff heights={np.round(takeoff_heights, 2)} "
            f"(target {target_height})",
            flush=True,
        )
        _record_snapshot(recorder, mocap, "takeoff")

        if args.align_spawns:
            with open(args.align_spawns, "r", encoding="utf-8") as stream:
                spawn_offsets = json.load(stream)
            print(f"[align] 起飞后微调到精确出生点：{args.align_spawns}", flush=True)
            targets = []
            for index, name in enumerate(role_names):
                role_label = ("P1", "P2", "P3", "E")[index]
                key = name if name in spawn_offsets else role_label
                if key not in spawn_offsets:
                    raise ValueError(
                        f"align-spawns 缺少 {name}（也没有角色键 {role_label}）"
                    )
                targets.append(
                    np.asarray(origin + np.asarray(spawn_offsets[key], dtype=np.float32),
                               dtype=np.float32)
                )

            # 先给每架发高层 goTo（兼容 enHighLevel=1 的飞机），再进入速度闭环保持。
            state0 = mocap.get_drone_state().numpy()
            for index, cf in enumerate(cfs):
                delta = float(np.linalg.norm(targets[index] - state0[index, :2]))
                if delta < 0.02:
                    continue
                duration = max(2.0, min(8.0, delta / 0.25 + 1.0))
                cf.goTo(
                    [float(targets[index][0]), float(targets[index][1]), target_height],
                    0.0,
                    duration,
                    relative=False,
                )

            # 所有飞机同时做位置保持：每架都朝自己的出生点做速度闭环，
            # 避免“等待对齐期间”的漂移。
            def _hold_all(active_index: int, speed_limit: float) -> np.ndarray:
                state = mocap.get_drone_state().numpy()
                heights = state[:, 2]
                for j, other in enumerate(cfs):
                    error_vector = targets[j] - state[j, :2]
                    distance = float(np.linalg.norm(error_vector))
                    if j == active_index:
                        speed = float(min(speed_limit, 0.8 * distance))
                    else:
                        speed = float(min(0.10, 0.6 * distance))
                    direction = error_vector / max(distance, 1e-6)
                    vertical = float(np.clip(
                        1.2 * (target_height - float(heights[j])), -0.2, 0.2))
                    other.cmdVelocityWorld(
                        [float(direction[0] * speed), float(direction[1] * speed), vertical],
                        yawRate=0.0,
                    )
                return state

            align_deadline = time.monotonic() + 12.0
            while time.monotonic() < align_deadline:
                state = _hold_all(active_index=-1, speed_limit=0.15)
                if all(
                    float(np.linalg.norm(state[j, :2] - targets[j])) <= 0.06
                    for j in range(4)
                ):
                    break
                time.sleep(0.02)
            _record_snapshot(recorder, mocap, "align")
            final_state = mocap.get_drone_state().numpy()
            for index, name in enumerate(role_names):
                error = float(np.linalg.norm(final_state[index, :2] - targets[index]))
                print(f"[align] {name}: 到位误差 {error:.3f} m", flush=True)
            errors = [
                (role_names[index],
                 float(np.linalg.norm(final_state[index, :2] - targets[index])))
                for index in range(4)
            ]
            bad = [(name, err) for name, err in errors if err > 0.10]
            if bad:
                raise RuntimeError(
                    "对齐出生点失败：" + ", ".join(f"{n} {e:.2f} m" for n, e in bad)
                )
            time.sleep(0.5)

        total_steps = round(args.duration * config.control_hz)
        for step in range(total_steps):
            loop_start = time.monotonic()
            state = mocap.get_drone_state().numpy()
            if mocap.stale():
                raise RuntimeError("mocap /tf stream is stale; stopping flight")
            positions = state[:, :2]
            measured_velocity = np.nan_to_num(
                state[:, 7:9], nan=0.0, posinf=0.0, neginf=0.0
            )
            normalized_action = runtime.actions(
                positions,
                measured_velocity,
                origin,
                config.virtual_obstacles,
            )
            command_xy[:3] += (
                normalized_action * min(args.max_accel, config.max_accel) * config.dt
            )
            command_xy[:3] = limit_rows(
                command_xy[:3], min(args.max_speed, config.max_speed)
            )
            if args.evader_mode == "escape":
                command_xy[3] = evader_velocity(
                    positions,
                    command_xy[3],
                    origin,
                    config,
                    evader_goal,
                )
            else:
                command_xy[3] = 0.0

            velocity = np.zeros((4, 3), dtype=np.float32)
            velocity[:, :2] = command_xy
            z_error = target_height - state[:, 2]
            velocity[:, 2] = np.clip(
                config.z_kp * z_error - config.z_kd * state[:, 9],
                -config.max_vertical_speed,
                config.max_vertical_speed,
            )
            velocity = apply_safety(velocity, positions, origin, config)
            command_xy[:] = velocity[:, :2]
            if recorder is not None:
                try:
                    if recorder.record(
                        time.monotonic(),
                        positions,
                        state[:, 2],
                        measured_velocity,
                        velocity[:, :2],
                    ):
                        captured_once = True
                        print(
                            f"[capture] ring held {config.capture_hold_steps} "
                            f"steps at t={step / config.control_hz:.2f}s；"
                            "停止追逃，准备悬停后原地下降",
                            flush=True,
                        )
                        break
                except OSError as exc:
                    print(f"[log] frame stream stopped: {exc}")
                    recorder = None
            if step % 100 == 0:
                pair = np.linalg.norm(
                    positions[:, None, :] - positions[None, :, :], axis=2
                ) + np.eye(4) * 100.0
                print(
                    f"[flight] t={step / config.control_hz:5.1f}s "
                    f"heights={np.round(state[:, 2], 2)} "
                    f"min_sep={np.min(pair):.2f} captured={captured_once}"
                )
            if (
                np.linalg.norm(positions[3] - evader_goal)
                <= config.evader_goal_radius
            ):
                print("[success] evader reached its private goal")
                outcome = "evader_reached_goal"
                break
            for cf, command in zip(cfs, velocity):
                cf.cmdVelocityWorld(command.tolist(), yawRate=0.0)
            low_level_started = True

            elapsed = time.monotonic() - loop_start
            remaining = config.dt - elapsed
            if remaining > 0.0:
                time.sleep(remaining)
            elif step % 50 == 0:
                print(
                    f"[warning] 50 Hz deadline missed: " f"loop={elapsed * 1000:.1f} ms"
                )
        if captured_once:
            outcome = "captured"
            print(
                "[capture] 已判定捕获：全体悬停 1 s，然后原地下降",
                flush=True,
            )
            hover_until = time.monotonic() + 1.0
            while time.monotonic() < hover_until:
                state = mocap.get_drone_state().numpy()
                heights = state[:, 2]
                for index, cf in enumerate(cfs):
                    vertical = float(np.clip(
                        1.2 * (target_height - float(heights[index])), -0.2, 0.2))
                    cf.cmdVelocityWorld([0.0, 0.0, vertical], yawRate=0.0)
                time.sleep(0.02)
            print("[land] 原地速度下降", flush=True)
            descend_deadline = time.monotonic() + 8.0
            while time.monotonic() < descend_deadline:
                state = mocap.get_drone_state().numpy()
                heights = state[:, 2]
                if all(float(h) <= 0.10 for h in heights):
                    break
                for index, cf in enumerate(cfs):
                    vertical = -0.2 if float(heights[index]) > 0.10 else 0.0
                    cf.cmdVelocityWorld([0.0, 0.0, vertical], yawRate=0.0)
                time.sleep(0.02)
            for _ in range(10):
                for cf in cfs:
                    cf.cmdVelocityWorld([0.0, 0.0, 0.0], yawRate=0.0)
                time.sleep(config.dt)
            for cf in cfs:
                try:
                    cf.notifySetpointsStop()
                except Exception:
                    pass
            airborne = False
            low_level_started = False

    except KeyboardInterrupt:
        outcome = "interrupted"
        print("[stop] interrupted by operator")
    except Exception:
        outcome = "error"
        raise
    finally:
        if low_level_started:
            for _ in range(10):
                for cf in cfs:
                    cf.cmdVelocityWorld([0.0, 0.0, 0.0], yawRate=0.0)
                time.sleep(config.dt)
            for cf in cfs:
                cf.notifySetpointsStop()
        if airborne:
            for cf in cfs:
                cf.land(targetHeight=0.05, duration=2.5)
            # 速度下降兜底（enHighLevel=0 时高层 land 无效）
            land_deadline = time.monotonic() + 6.0
            while time.monotonic() < land_deadline:
                try:
                    state = mocap.get_drone_state().numpy()
                except Exception:
                    break
                heights = state[:, 2]
                if all(float(h) <= 0.10 for h in heights):
                    break
                for index, cf in enumerate(cfs):
                    vertical = -0.2 if float(heights[index]) > 0.10 else 0.0
                    cf.cmdVelocityWorld([0.0, 0.0, vertical], yawRate=0.0)
                time.sleep(0.02)
            for _ in range(10):
                for cf in cfs:
                    cf.cmdVelocityWorld([0.0, 0.0, 0.0], yawRate=0.0)
                time.sleep(config.dt)
            for cf in cfs:
                try:
                    cf.notifySetpointsStop()
                except Exception:
                    pass
            time_helper.sleep(1.0)
        for cf in cfs:
            cf.arm(False)
        _record_snapshot(recorder, mocap, "landed")
        if recorder is not None:
            try:
                summary = recorder.close(outcome)
                print(f"[log] {recorder.jsonl_path}")
                print(
                    f"[log] summary outcome={summary['outcome']} "
                    f"frames={summary['frames']} "
                    f"duration={summary['duration_s']}s "
                    f"captured={summary['captured']} "
                    f"min_sep={summary.get('min_separation')} "
                    f"min_pursuer_dist={summary.get('min_pursuer_distance')}"
                )
            except OSError as exc:
                print(f"[log] failed to finalize log: {exc}")
        mocap.end_program()
        print("[done] landed and disarmed")


if __name__ == "__main__":
    with torch.no_grad():
        main()
