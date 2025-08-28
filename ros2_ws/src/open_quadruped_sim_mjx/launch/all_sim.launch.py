
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    mjcf_path = LaunchConfiguration('mjcf_path')
    cam = LaunchConfiguration('camera')
    # Default actuator mapping for simple quadruped.xml (can be overridden)
    default_mapping = {
        'fl': ['hip_front_left', 'knee_front_left'],
        'fr': ['hip_front_right', 'knee_front_right'],
        'bl': ['hip_back_left', 'knee_back_left'],
        'br': ['hip_back_right', 'knee_back_right'],
    }
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
            parameters=[{'mjcf_path': mjcf_path, 'timestep': 0.002, 'realtime': True, 'actuator_mapping': default_mapping}],
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
        # Static TFs so mapping has consistent frames (tweak as needed)
        Node(
            package='tf2_ros', executable='static_transform_publisher', name='tf_cam', output='screen',
            arguments=['0','0','0.2','0','0','0','base_link','camera_link']
        ),
        Node(
            package='tf2_ros', executable='static_transform_publisher', name='tf_lidar', output='screen',
            arguments=['0','0','0.25','0','0','0','base_link','lidar_link']
        ),
    ])
