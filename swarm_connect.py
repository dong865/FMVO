"""带超时与诊断的 Crazyswarm 连接助手（3V1 实机入口共用）。

为什么需要它
------------
``crazyflie_py.Crazyswarm()`` 内部对每个服务都是**无超时**的 ``wait_for_service()``
（见 ``crazyflie_py/crazyswarm_py.py::Crazyswarm.__init__`` →
``CrazyflieServer``）。因此只要满足下面任一条，构造就会**静默卡住、终端一个字都不输出**：

* 终端 1 的 ``ros2 launch crazyflie launch.py`` 没起来；
* ``crazyflies.yaml`` 里 enabled 的飞机没有全部开机（crazyswarm2 要等**所有**
  enabled 飞机连上才开始创建服务）；
* 电台 URI 不对／无线棒没插好。

本模块把构造放进守护线程并 ``join(timeout)``：超时后打印一份 ROS 图诊断
（节点、``/takeoff``、``/cfX/start_trajectory`` 服务），然后抛 ``RuntimeError``，
让操作员立刻知道卡在哪，而不是对着一个"像死了"的终端。

用法::

    from crazyflie_py import Crazyswarm
    from .swarm_connect import connect_swarm

    print("[preflight] 正在连接 crazyflie server（若长时间无输出，请确认终端 1 已 launch）",
          flush=True)
    swarm = connect_swarm(Crazyswarm, timeout=20.0, expected_robots=4)
"""

from __future__ import annotations

import threading
from typing import Callable, Optional, Tuple

#: 本机 3V1 编队的机数（3 追捕 + 1 逃逸）
DEFAULT_EXPECTED_ROBOTS = 4

#: 默认连接超时（秒）。老版 994 行 real_flight.py 用的也是这个量级。
DEFAULT_CONNECT_TIMEOUT = 20.0


def _ros_graph_snapshot(expected: int = 0):
    """连接超时时抓一份 ROS 图快照（节点 + 关键服务），便于一步定位。"""
    try:
        import rclpy

        if not rclpy.ok():
            rclpy.init()
        node = rclpy.create_node("vone_connect_diag")
        try:
            services = sorted(name for name, _ in node.get_service_names_and_types())
            nodes = sorted(node.get_node_names())
        finally:
            node.destroy_node()
        takeoff = [s for s in services if s.endswith("/takeoff")]
        start_traj = [s for s in services if s.endswith("/start_trajectory")]
        return nodes, takeoff, start_traj
    except Exception as exc:  # noqa: BLE001 - 诊断失败不能掩盖原始超时
        return None, f"（无法读取 ROS 图：{exc}）", None


def read_connectivity_from_launch_log(lookback_s: float = 900.0) -> dict:
    """解析 crazyflie_server 的进程日志，得出每架飞机的连接结果。

    日志位置：``~/.ros/log/python3_<pid>_<ts>.log``（server 进程自己的日志；
    ``launch.log`` 里只有 launch 级消息，不含连接结果）。

    为什么需要：crazyswarm2 **要等 crazyflies.yaml 里所有 enabled 的飞机都连上**才创建
    ``/cfX/start_trajectory`` 服务。少一架就一直没有服务，实飞脚本只能超时退出，
    而报错里看不出**是哪一架**没连上（得自己去翻终端 1）。

    只取"最近 lookback 秒内被写过、且内容含 crazyflie_server 的最新一个文件"，
    并在其中按出现顺序取每架飞机的**最后状态**，避免把上一轮的旧结果混进来。
    """
    import glob
    import os
    import re
    import time

    now = time.time()
    files = [f for f in glob.glob(os.path.expanduser("~/.ros/log/**/*.log"), recursive=True)
             if now - os.path.getmtime(f) <= float(lookback_s)]
    files.sort(key=os.path.getmtime, reverse=True)
    pat = re.compile(r"\[(cf\d+)\]\s*(is (?:fully )?connected|connection Failed|"
                     r"could not be found|can not be found)")
    for path in files:
        try:
            with open(path, "r", errors="ignore") as handle:
                lines = handle.readlines()
        except OSError:
            continue
        if not any("crazyflie_server" in ln for ln in lines):
            continue
        status = {}
        for ln in lines:
            if "crazyflie_server" not in ln:
                continue
            m = pat.search(ln)
            if m:
                status[m.group(1)] = ("ok" if "connected" in m.group(2) else "fail")
        if status:
            connected = sorted(n for n, v in status.items() if v == "ok")
            failed = sorted(n for n, v in status.items() if v == "fail")
            return {"connected": connected, "failed": failed, "log": path}
    return {"connected": [], "failed": [], "log": None}


def connect_swarm(
    swarm_factory: Callable[[], object],
    timeout: float = DEFAULT_CONNECT_TIMEOUT,
    expected_robots: int = DEFAULT_EXPECTED_ROBOTS,
    expected_names=None,
) -> object:
    """在带超时的线程里构造 ``Crazyswarm``，超时给出可操作的诊断。

    ``swarm_factory`` 通常直接传 ``Crazyswarm`` 类本身。
    """
    result: dict = {}

    def worker() -> None:
        try:
            result["swarm"] = swarm_factory()
        except Exception as exc:  # noqa: BLE001 - 交给主线程重新抛出
            result["error"] = exc

    thread = threading.Thread(target=worker, daemon=True, name="connect_swarm")
    thread.start()
    thread.join(timeout=float(timeout))

    if "error" in result:
        raise RuntimeError(
            f"连接 crazyflie 服务器失败：{result['error']}"
        ) from result["error"]

    if thread.is_alive():
        lines = [
            f"连接 crazyflie 服务器超时（>{timeout:.0f}s），无人机不会有任何反应。请依次检查：",
            "  1) 终端 1 是否在跑：ros2 launch crazyflie launch.py gui:=False rviz:=True",
            "  2) crazyflies.yaml 里 enabled 的飞机是否都已开机"
            "（crazyswarm2 要等全部连上才建服务）",
            "  3) 电台地址是否正确：python3 radio_check.py --battery",
            f"  4) 服务是否出现：ros2 service list | grep start_trajectory"
            f"   # 应有 {expected_robots} 个" if expected_robots else
            "  4) 服务是否出现：ros2 service list | grep start_trajectory",
        ]
        nodes, takeoff, start_traj = _ros_graph_snapshot(expected_robots)
        if nodes and start_traj is not None:
            lines.append(f"  当前 ROS 图：节点 {nodes}")
            lines.append(f"              含 /takeoff 的服务：{takeoff}")
            lines.append(f"              含 /start_trajectory 的服务：{start_traj}")
            if not start_traj:
                lines.append(
                    "  → 一个 /cfX/start_trajectory 都没有：crazyflie_server 还没创建服务，"
                    "说明有 enabled 的飞机没连上"
                )
                # 自动读 server 的日志，直接点名是哪一架（省掉翻终端 1）。
                # 没读到数据时绝不下"某几架没连上"的结论（那是假结论）——
                # 旧 launch 用 output='screen'，server 的连接行只到终端、不落盘。
                try:
                    info = read_connectivity_from_launch_log()
                    if not (info["connected"] or info["failed"]):
                        lines.append("  （server 的连接结果没落盘，无法自动判断是哪一架；"
                                     "请到终端 1 看哪一架没有 is connected!。"
                                     "已把 launch 改成 output=both，下次会自动点名）")
                        lines.append("     注意：mocap 里能看到刚体不等于电台连上了；"
                                     "4 架 enabled 少一架就一个服务都建不出来。")
                    else:
                        lines.append(f"  ✅ 已连上：{info['connected']}"
                                     + (f"   ❌ 连接失败：{info['failed']}"
                                        if info["failed"] else ""))
                        if expected_names:
                            missing = [n for n in expected_names
                                       if n not in info["connected"]]
                            if missing:
                                lines.append(f"  ⛔ **没连上的机：{missing}**"
                                             "（server 没认到这些机，所以服务建不出来）")
                                lines.append("     处置顺序：① 给这些机**换满电电池**；"
                                             "② 断电 10 s 重新上电（清掉旧链路/飞控状态）；"
                                             "③ 仍连不上就停掉 server，单独验证电台地址：")
                                for n in missing:
                                    addr = "E7E7E7E7" + n.replace("cf", "").zfill(2)
                                    lines.append("        python3 radio_check.py "
                                                 f"radio://0/80/2M/{addr} --battery")
                        lines.append(f"     （来源：{info['log']}）")
                except Exception as exc:                  # noqa: BLE001
                    lines.append(f"  （自动读取连接日志失败：{exc}）")
            elif expected_robots and len(start_traj) < expected_robots:
                lines.append(
                    f"  → 只发现 {len(start_traj)} 个（需要 {expected_robots} 个）："
                    "少量飞机没连上，crazyswarm2 会一直等"
                )
        else:
            lines.append(f"  ROS 图读取结果：{takeoff}")
        lines.append(
            "  查看终端 1 日志里每架飞机的连接结果（例如 [cf8] is connected! / "
            "connection failed），并把没连上的那几架先排除"
        )
        lines.append(
            "  快速核对：ros2 node list | grep crazyflie_server ; "
            "ros2 service list | grep -c start_trajectory"
        )
        lines.append("  查看服务器日志（最直接）：")
        lines.append("     ls -t ~/.ros/log/*/launch.log | head -1        # 本次 launch 日志")
        lines.append(
            "     grep -h 'connection Failed\\|is connected\\|can not be found' "
            "~/.ros/log/python3_*.log | tail -20"
        )
        lines.append(
            "     → 若出现 '[cfX] connection Failed' + "
            "'One or more Crazyflies can not be found'，"
        )
        lines.append("       说明该架没连上，crazyswarm2 会直接退出服务器（其余飞机也会一起断开）")
        raise RuntimeError("\n".join(lines))

    return result["swarm"]


def add_connect_timeout_arg(parser, default: float = DEFAULT_CONNECT_TIMEOUT) -> None:
    """给 argparse parser 加上统一的 ``--connect-timeout``。"""
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=default,
        help="连接 crazyflie server 的超时秒数（默认 %(default)s）；"
             "超时会打印 ROS 图诊断而不是静默卡住",
    )


def announce_connecting(expected_robots: int = DEFAULT_EXPECTED_ROBOTS,
                        timeout: float = DEFAULT_CONNECT_TIMEOUT) -> None:
    """连接前先打印一行，避免"运行后没反应"的错觉。"""
    print(
        f"[preflight] 正在连接 crazyflie server 并等待 {expected_robots} 架飞机"
        f"（超时 {timeout:.0f}s；若长时间无输出，请确认终端 1 已 "
        f"ros2 launch crazyflie launch.py）",
        flush=True,
    )
