from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    rgb  = LaunchConfiguration('rgb',  default='camera/image_raw')
    depth= LaunchConfiguration('depth',default='camera/depth/image_raw')
    info = LaunchConfiguration('info', default='camera/camera_info')
    frame= LaunchConfiguration('frame_id', default='base_link')
    return LaunchDescription([
        DeclareLaunchArgument('rgb',   default_value=rgb),
        DeclareLaunchArgument('depth', default_value=depth),
        DeclareLaunchArgument('info',  default_value=info),
        DeclareLaunchArgument('frame_id', default_value=frame),
        # rgbd_sync packs RGB+Depth+Info to /rgbd_image
        Node(
            package='rtabmap_ros', executable='rgbd_sync', name='rgbd_sync', output='screen',
            remappings=[
                ('rgb/image', rgb),
                ('depth/image', depth),
                ('rgb/camera_info', info),
            ]
        ),
        Node(
            package='rtabmap_ros', executable='rtabmap', name='rtabmap', output='screen',
            parameters=[{'frame_id': frame}],
            remappings=[('rgbd_image', 'rgbd_sync/output')]
        )
    ])
