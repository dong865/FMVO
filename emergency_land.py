"""Rescue script: slowly land every connected Crazyflie at its current position.

Use when the control script has crashed, hung, or was killed (Ctrl+C inside
real_flight.py already performs a slow descent itself; this script is for the
case where that process is gone and the drones are still airborne).

IMPORTANT: if the control script is still running and streaming velocity
setpoints, its low-level stream will override this landing command. Stop the
control script first (or use the /emergency_land topic trigger of real_flight).

Usage (in another terminal, after `ros2 launch crazyflie ...` is up):
    python3 -m crazyflie_examples.three_v_one_baseline.emergency_land [height] [duration]
"""

import sys

from crazyflie_py import Crazyswarm


def main():
    land_height = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
    duration = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0

    swarm = Crazyswarm()
    allcfs = swarm.allcfs
    time_helper = swarm.timeHelper

    names = sorted(cf.prefix.lstrip("/") for cf in allcfs.crazyflies)
    print(
        f"[rescue] {len(names)} drones connected: {names}",
        flush=True,
    )
    print(
        f"[rescue] stopping setpoint streams, then landing all drones "
        f"to {land_height} m over {duration} s",
        flush=True,
    )
    for cf in allcfs.crazyflies:
        cf.notifySetpointsStop()
    time_helper.sleep(0.5)
    for cf in allcfs.crazyflies:
        cf.land(targetHeight=land_height, duration=duration)
    time_helper.sleep(duration + 2.0)
    for cf in allcfs.crazyflies:
        cf.arm(False)
    print("[rescue] done - drones landed and disarmed", flush=True)


if __name__ == "__main__":
    main()
