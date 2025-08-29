from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='livox_ros_driver2',
            executable='livox_ros_driver_node',
            name='livox_mid360',
            output='screen',
            parameters=[{
                'multi_topic': True,
                'xfer_format': 1,   # 0: Livox custom, 1: PointCloud2
                'publish_freq': 10.0,
                'frame_id': 'livox_frame',
                # 추가 포트/브로드캐스트 설정은 사용자 네트워크에 맞게
            }]
        )
    ])
