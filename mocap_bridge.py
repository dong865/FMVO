"""Nokov reader based on pid_swarm_test.py; mocap topics are unchanged."""

import time

import rclpy
import torch

from crazyflie_examples.fake.swarm.subscriber import TFSubscriber


def quaternion_to_euler(quaternion: torch.Tensor) -> torch.Tensor:
    w, x, y, z = torch.unbind(quaternion, dim=quaternion.dim() - 1)
    return torch.stack(
        (
            torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)),
            torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0)),
            torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)),
        ),
        dim=-1,
    )


class Nokov:
    """Same TF input contract and state layout as ``pid_swarm_test.Nokov``."""

    def __init__(self, cfs):
        self.log = None
        self.cfs = cfs
        self.num_cf = len(cfs)
        self.drone_state = torch.zeros((self.num_cf, 28))
        self.drone_state[..., 3] = 1.0
        self.node = TFSubscriber(self.update_drone_state)
        self.last_time = 0.0
        self.cf_map = {cf.prefix[1:]: index for index, cf in enumerate(cfs)}
        self.seen = torch.zeros(self.num_cf, dtype=torch.bool)
        self.last_wall_update = 0.0

    def update_drone_state(self, log):
        self.log = log
        self.last_wall_update = time.monotonic()

    def get_drone_state(self):
        rclpy.spin_once(self.node, timeout_sec=0.1)
        if self.log is not None:
            last_pos = self.drone_state[..., :3].clone()
            last_quat = self.drone_state[..., 3:7].clone()
            last_rpy = quaternion_to_euler(last_quat)
            newest_time = self.last_time
            for transform in self.log.transforms:
                stamp = transform.header.stamp
                transform_time = stamp.sec + stamp.nanosec / 1e9
                if transform.child_frame_id not in self.cf_map:
                    continue
                drone_id = self.cf_map[transform.child_frame_id]
                newest_time = max(newest_time, transform_time)
                self.seen[drone_id] = True
                pose = self.drone_state[drone_id]
                pose[0] = transform.transform.translation.x
                pose[1] = transform.transform.translation.y
                pose[2] = transform.transform.translation.z
                pose[3] = transform.transform.rotation.w
                pose[4] = transform.transform.rotation.x
                pose[5] = transform.transform.rotation.y
                pose[6] = transform.transform.rotation.z
                if pose[3] < 0:
                    pose[3:7] *= -1
            dt = newest_time - self.last_time
            if dt < 0.005:
                dt = 0.01
            else:
                self.drone_state[..., 7:10] = (
                    self.drone_state[..., :3] - last_pos
                ) / dt
            current_rpy = quaternion_to_euler(self.drone_state[..., 3:7])
            self.drone_state[..., 10:13] = (current_rpy - last_rpy) / dt
            self.last_time = newest_time
        return self.drone_state.clone()

    def ready(self) -> bool:
        return bool(torch.all(self.seen))

    def stale(self, timeout: float = 0.25) -> bool:
        return (
            self.last_wall_update == 0.0
            or time.monotonic() - self.last_wall_update > timeout
        )

    def end_program(self):
        self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
