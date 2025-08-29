from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # OrbbecSDK_ROS2 wrapper (modern SDK)
        Node(
            package='orbbec_camera',
            executable='orbbec_camera_node',
            name='orbbec_astra',
            output='screen',
            parameters=[{
                'enable_color': True,
                'enable_depth': True,
                'depth_align': True,
                'fps': 30,
                'resolution': '640x480',
                'frame_id': 'camera_link',
            }]
        )
    ])
