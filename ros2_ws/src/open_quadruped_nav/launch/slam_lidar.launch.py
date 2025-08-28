from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    points = LaunchConfiguration('points', default='lidar/points')
    scan   = LaunchConfiguration('scan',   default='scan')
    frame  = LaunchConfiguration('frame_id', default='lidar_link')
    return LaunchDescription([
        DeclareLaunchArgument('points', default_value=points),
        DeclareLaunchArgument('scan', default_value=scan),
        DeclareLaunchArgument('frame_id', default_value=frame),
        Node(
            package='pointcloud_to_laserscan', executable='cloud_to_scan', name='cloud_to_scan', output='screen',
            parameters=[{'output_frame_id': frame}],
            remappings=[('cloud', points), ('scan', scan)]
        ),
        Node(
            package='slam_toolbox', executable='async_slam_toolbox_node', name='slam_toolbox', output='screen',
            remappings=[('scan', scan)]
        )
    ])
