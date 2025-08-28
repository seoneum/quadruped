import os
import numpy as np
import tarfile


def _ensure_meshes_unpacked(mjcf_path: str, logger=None):
    import os
    try:
        base = os.path.dirname(mjcf_path)
        meshdir = os.path.join(base, 'meshes')
        if not os.path.isdir(meshdir):
            return
        # quick check for any STL present
        try:
            has_stl = any(name.lower().endswith('.stl') for name in os.listdir(meshdir))
        except Exception:
            has_stl = False
        if has_stl:
            return
        # find archive to unpack
        cands = [os.path.join(meshdir, f) for f in os.listdir(meshdir) if f.endswith('.tar.gz')]
        if not cands:
            return
        archive = sorted(cands)[-1]
        if logger:
            logger.info(f'Unpacking mesh archive: {archive}')
        else:
            print(f'[info] Unpacking mesh archive: {archive}')
        with tarfile.open(archive, 'r:gz') as tf:
            tf.extractall(meshdir)
    except Exception as e:
        msg = f'Failed to auto-unpack meshes: {e}'
        if logger:
            try:
                logger.warn(msg)
            except Exception:
                logger.info(msg)
        else:
            print('[warn]', msg)
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

try:
    import mujoco
except Exception:
    mujoco = None


class MujocoLidarPublisher(Node):
    def __init__(self):
        super().__init__('mujoco_lidar_publisher')
        self.declare_parameter('mjcf_path', '')
        self.declare_parameter('camera', 'lidar_cam')
        self.declare_parameter('width', 1024)
        self.declare_parameter('height', 64)
        self.declare_parameter('frame_id', 'lidar_link')
        self.declare_parameter('rate', 10.0)
        self.declare_parameter('max_range', 50.0)

        self.mjcf_path = str(self.get_parameter('mjcf_path').value)
        self.cam_name = str(self.get_parameter('camera').value)
        self.width = int(self.get_parameter('width').value)
        self.height = int(self.get_parameter('height').value)
        self.frame_id = str(self.get_parameter('frame_id').value)
        self.rate = float(self.get_parameter('rate').value)
        self.max_range = float(self.get_parameter('max_range').value)

        if mujoco is None:
            self.get_logger().error('mujoco not available. Install via `uv add mujoco`.')
            raise SystemExit(1)
        if not self.mjcf_path or not os.path.exists(self.mjcf_path):
            self.get_logger().error(f'Invalid mjcf_path: {self.mjcf_path}')
            raise SystemExit(1)

        # Ensure meshes are unpacked from the bundled archive if needed
        _ensure_meshes_unpacked(self.mjcf_path, self.get_logger())
        self.model = mujoco.MjModel.from_xml_path(self.mjcf_path)
        self.data = mujoco.MjData(self.model)

        self.cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.cam_name)
        if self.cam_id < 0:
            self.get_logger().error(f'Camera {self.cam_name} not found in MJCF.')
            raise SystemExit(1)

        # Renderer for depth
        try:
            self.renderer = mujoco.Renderer(self.model, self.height, self.width)
            self.get_logger().info(f'Lidar renderer {self.width}x{self.height} for camera {self.cam_name}.')
        except Exception as e:
            self.get_logger().error(f'Failed to create Renderer: {e}')
            raise SystemExit(1)

        self.pub = self.create_publisher(PointCloud2, 'lidar/points', 10)
        self.timer = self.create_timer(1.0 / max(self.rate, 1e-3), self.tick)

    def tick(self):
        mujoco.mj_step(self.model, self.data)
        self.renderer.update_scene(self.data, camera=self.cam_name)
        depth = self.renderer.render(depth=True)  # (H, W) float in meters
        # Convert depth to point cloud
        cloud = self.depth_to_points(depth)
        msg = self.numpy_to_pointcloud2(cloud, self.frame_id)
        self.pub.publish(msg)

    def depth_to_points(self, depth: np.ndarray) -> np.ndarray:
        H, W = depth.shape
        depth = np.clip(depth, 0.0, self.max_range)
        # Intrinsics from fovy
        fovy = self.model.cam_fovy[self.cam_id]
        fy = 0.5 * H / np.tan(0.5 * np.deg2rad(fovy))
        fx = fy  # square pixels assumption
        cx = W / 2.0
        cy = H / 2.0
        # pixel grid
        xs = np.arange(W)
        ys = np.arange(H)
        xv, yv = np.meshgrid(xs, ys)
        Z = depth
        X = (xv - cx) * Z / fx
        Y = (yv - cy) * Z / fy
        pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3).astype(np.float32)
        # filter zeros/inf
        mask = np.isfinite(Z).reshape(-1) & (Z.reshape(-1) > 0.0)
        return pts[mask]

    def numpy_to_pointcloud2(self, points: np.ndarray, frame_id: str) -> PointCloud2:
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = frame_id
        # Define fields x,y,z as float32
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        # to bytes
        data = points.tobytes()
        msg = PointCloud2(
            header=header,
            height=1,
            width=points.shape[0],
            fields=fields,
            is_bigendian=False,
            point_step=12,
            row_step=12 * points.shape[0],
            data=data,
            is_dense=True,
        )
        return msg


def main():
    rclpy.init()
    node = MujocoLidarPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
