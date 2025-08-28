import os
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from open_quadruped_interfaces.msg import JointAngles

# Optional imports; the node will warn if mujoco/mjx aren't present
try:
    import mujoco
except Exception as e:  # noqa: BLE001
    mujoco = None

try:
    from mujoco import mjx
except Exception as e:  # noqa: BLE001
    mjx = None


class MJXSimNode(Node):
    def __init__(self):
        super().__init__('mjx_sim')
        self.declare_parameter('mjcf_path', '')
        self.declare_parameter('timestep', 0.002)
        self.declare_parameter('realtime', True)
        self.declare_parameter('actuator_mapping', {})

        self.mjcf_path = str(self.get_parameter('mjcf_path').value)
        self.dt = float(self.get_parameter('timestep').value)
        self.realtime = bool(self.get_parameter('realtime').value)
        self.actuator_mapping = dict(self.get_parameter('actuator_mapping').value)

        if mujoco is None:
            self.get_logger().warn('mujoco not available. Install with `uv add mujoco`.')
        if mjx is None:
            self.get_logger().warn('mujoco-mjx not available. Install with `uv add mujoco-mjx jax jaxlib`.')

        self.model = None
        self.data = None
        if mujoco is not None and self.mjcf_path and os.path.exists(self.mjcf_path):
            try:
                self.model = mujoco.MjModel.from_xml_path(self.mjcf_path)
                self.data = mujoco.MjData(self.model)
                self.get_logger().info(f'Loaded MJCF: {self.mjcf_path}')
            except Exception as e:  # noqa: BLE001
                self.get_logger().error(f'Failed to load MJCF: {e}')

        self.sub = self.create_subscription(JointAngles, 'joint_angles', self.on_joint_angles, 10)
        self.timer = self.create_timer(self.dt, self.step)
        self.last_time = time.time()

    def on_joint_angles(self, msg: JointAngles):
        # Map incoming joint angles (rad) into model actuators if available.
        if self.model is None or self.data is None:
            return
        # Map joint arrays into actuator ctrl by name mapping if provided
        if not self.actuator_mapping:
            return
        def set_by_list(names, values):
            for n, v in zip(names, values):
                if self.model is None:
                    continue
                aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                if aid >= 0:
                    self.data.ctrl[aid] = float(v)
        set_by_list(self.actuator_mapping.get('fl', []), msg.fl)
        set_by_list(self.actuator_mapping.get('fr', []), msg.fr)
        set_by_list(self.actuator_mapping.get('bl', []), msg.bl)
        set_by_list(self.actuator_mapping.get('br', []), msg.br)

    def step(self):
        if self.model is None or self.data is None or mujoco is None:
            return
        mujoco.mj_step(self.model, self.data)
        if self.realtime:
            now = time.time()
            elapsed = now - self.last_time
            sleep_t = max(0.0, self.dt - elapsed)
            time.sleep(sleep_t)
            self.last_time = now + sleep_t


def main():
    rclpy.init()
    node = MJXSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
