import math
from typing import Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from open_quadruped_interfaces.msg import JointAngles

from open_quadruped_control.control_library import leg_ik, body_ik


class InterfaceProcess(Node):
    def __init__(self):
        super().__init__('interface_process')
        # params
        self.declare_parameter('yaw_limit', 15.0)
        self.declare_parameter('pitch_limit', 15.0)
        self.declare_parameter('roll_limit', 15.0)

        self.yaw_limit = float(self.get_parameter('yaw_limit').value)
        self.pitch_limit = float(self.get_parameter('pitch_limit').value)
        self.roll_limit = float(self.get_parameter('roll_limit').value)

        self.body_mode = 0
        self.gait_mode = 1
        self.mode = self.body_mode

        # IK models
        self.leg_model = leg_ik.LegIKModel(109.868, 144.580, 11.369, 63.763)
        self.body_model = body_ik.BodyIKModel(76.655, 229.3, 130)

        self.buttons: Optional[list[int]] = None
        self.axes: Optional[list[float]] = None

        self.sub = self.create_subscription(Joy, 'joy', self.controller_callback, 10)
        self.pub = self.create_publisher(JointAngles, 'joint_angles', 10)
        self.timer = self.create_timer(0.1, self.loop)  # 10 Hz

    def controller_callback(self, msg: Joy):
        if len(msg.buttons) > 7 and msg.buttons[7] == 1:
            self.mode = self.gait_mode
        else:
            self.mode = self.body_mode
        self.buttons = list(msg.buttons)
        self.axes = list(msg.axes)

    def loop(self):
        if self.axes is None or self.buttons is None:
            return
        if self.mode == self.body_mode:
            debug = 'in body mode'
            yaw = (self.axes[2] if len(self.axes) > 2 else 0.0) * self.yaw_limit
            pitch = (self.axes[5] if len(self.axes) > 5 else 0.0) * self.pitch_limit
            roll = (self.axes[0] if len(self.axes) > 0 else 0.0) * self.roll_limit
            self.body_model.reset_pose()
            self.body_model.transform(math.radians(yaw), math.radians(pitch), math.radians(roll))
            htf_vecs = self.body_model.get_htf_vectors()
        else:
            debug = 'in gait mode (not implemented)'
            # TODO: integrate gait planner; for now reuse neutral body IK
            self.body_model.reset_pose()
            htf_vecs = self.body_model.get_htf_vectors()
        self.get_logger().debug(debug)

        ja_m = self.leg_model.ja_from_htf_vecs(htf_vecs)
        msg = JointAngles()
        msg.fl = [float(x) for x in ja_m[0]]
        msg.fr = [float(x) for x in ja_m[1]]
        msg.bl = [float(x) for x in ja_m[2]]
        msg.br = [float(x) for x in ja_m[3]]
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = InterfaceProcess()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
