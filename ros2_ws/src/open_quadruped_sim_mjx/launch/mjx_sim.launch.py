
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument


def generate_launch_description():
    mjcf_path = LaunchConfiguration('mjcf_path')
    return LaunchDescription([
        DeclareLaunchArgument('mjcf_path', default_value=''),
        Node(
            package='open_quadruped_sim_mjx',
            executable='mjx_sim',
            name='mjx_sim',
            parameters=[{'mjcf_path': mjcf_path, 'timestep': 0.002, 'realtime': True}],
            output='screen',
        )
    ])
