
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    mjcf_path = LaunchConfiguration('mjcf_path')
    cam = LaunchConfiguration('camera')
    return LaunchDescription([
        DeclareLaunchArgument('mjcf_path', default_value=''),
        DeclareLaunchArgument('camera', default_value='track'),
        Node(
            package='open_quadruped_control',
            executable='interface_process',
            name='interface_process',
            output='screen',
        ),
        Node(
            package='open_quadruped_sim_mjx',
            executable='mjx_sim',
            name='mjx_sim',
            parameters=[{'mjcf_path': mjcf_path, 'timestep': 0.002, 'realtime': True}],
            output='screen',
        ),
        Node(
            package='open_quadruped_sim_mjx',
            executable='mujoco_camera_publisher',
            name='mujoco_rgbd',
            parameters=[{'mjcf_path': mjcf_path, 'camera': cam, 'width': 640, 'height': 480, 'rate': 15.0}],
            output='screen',
        ),
        Node(
            package='open_quadruped_sim_mjx',
            executable='mujoco_lidar_publisher',
            name='mujoco_lidar',
            parameters=[{'mjcf_path': mjcf_path, 'camera': cam, 'width': 1024, 'height': 64, 'rate': 10.0}],
            output='screen',
        ),
    ])
