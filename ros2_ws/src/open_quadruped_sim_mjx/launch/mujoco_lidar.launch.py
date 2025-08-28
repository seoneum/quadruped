
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    mjcf_path = LaunchConfiguration('mjcf_path')
    cam = LaunchConfiguration('camera')
    return LaunchDescription([
        DeclareLaunchArgument('mjcf_path', default_value=''),
        DeclareLaunchArgument('camera', default_value='lidar_cam'),
        Node(
            package='open_quadruped_sim_mjx',
            executable='mujoco_lidar_publisher',
            name='mujoco_lidar',
            parameters=[{'mjcf_path': mjcf_path, 'camera': cam, 'width': 1024, 'height': 64, 'rate': 10.0}],
            output='screen',
        ),
    ])
