import os
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from builtin_interfaces.msg import Time as TimeMsg

try:
    import mujoco
    from mujoco import viewer  # noqa: F401  # not used but ensures render backend availability
except Exception:
    mujoco = None


class MujocoCameraPublisher(Node):
    def __init__(self):
        super().__init__('mujoco_camera_publisher')
        self.declare_parameter('mjcf_path', '')
        self.declare_parameter('camera', 'track')
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('frame_id', 'camera_link')
        self.declare_parameter('rate', 15.0)

        self.mjcf_path = str(self.get_parameter('mjcf_path').value)
        self.cam_name = str(self.get_parameter('camera').value)
        self.width = int(self.get_parameter('width').value)
        self.height = int(self.get_parameter('height').value)
        self.frame_id = str(self.get_parameter('frame_id').value)
        self.rate = float(self.get_parameter('rate').value)

        if mujoco is None:
            self.get_logger().error('mujoco not available. Install via `uv add mujoco`.')
            raise SystemExit(1)

        if not self.mjcf_path or not os.path.exists(self.mjcf_path):
            self.get_logger().error(f'Invalid mjcf_path: {self.mjcf_path}')
            raise SystemExit(1)

        self.model = mujoco.MjModel.from_xml_path(self.mjcf_path)
        self.data = mujoco.MjData(self.model)

        # Camera id
        self.cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.cam_name)
        if self.cam_id < 0:
            self.get_logger().error(f'Camera {self.cam_name} not found in MJCF.')
            raise SystemExit(1)

        # Renderer
        try:
            self.renderer = mujoco.Renderer(self.model, self.height, self.width)
            self.get_logger().info(f'Renderer created {self.width}x{self.height} for camera {self.cam_name}.')
        except Exception as e:
            self.get_logger().error(f'Failed to create Renderer: {e}')
            raise SystemExit(1)

        self.pub_rgb = self.create_publisher(Image, 'camera/image_raw', 10)
        self.pub_depth = self.create_publisher(Image, 'camera/depth/image_raw', 10)
        self.pub_info = self.create_publisher(CameraInfo, 'camera/camera_info', 10)

        self.timer = self.create_timer(1.0 / max(self.rate, 1e-3), self.tick)

    def tick(self):
        # Step a bit to ensure valid state for rendering
        mujoco.mj_step(self.model, self.data)

        # Set camera and render
        self.renderer.update_scene(self.data, camera=self.cam_name)
        rgb = self.renderer.render()
        depth = self.renderer.render(depth=True)

        now = self.get_clock().now().to_msg()
        self.pub_rgb.publish(self._image_msg(rgb, now, 'rgb8'))
        # Depth in meters, float32
        depth = depth.astype(np.float32)
        self.pub_depth.publish(self._image_msg(depth, now, '32FC1'))
        self.pub_info.publish(self._camera_info(now))

    def _image_msg(self, img: np.ndarray, stamp: TimeMsg, encoding: str) -> Image:
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.height, msg.width = img.shape[:2]
        msg.encoding = encoding
        if img.dtype != np.uint8 and encoding != '32FC1':
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        msg.step = int(img.strides[0])
        msg.data = img.tobytes()
        return msg

    def _camera_info(self, stamp: TimeMsg) -> CameraInfo:
        msg = CameraInfo()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.width = self.width
        msg.height = self.height
        # Simple pinhole defaults; for precise values compute from MJCF camera fovy
        cam = self.model.cam_fovy[self.cam_id]
        f = 0.5 * self.height / np.tan(0.5 * np.deg2rad(cam))
        msg.k = [f, 0, self.width/2, 0, f, self.height/2, 0, 0, 1]
        msg.p = [f, 0, self.width/2, 0, 0, f, self.height/2, 0, 0, 0, 1, 0]
        return msg


def main():
    rclpy.init()
    node = MujocoCameraPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
