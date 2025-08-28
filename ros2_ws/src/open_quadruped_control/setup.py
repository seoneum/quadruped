from setuptools import setup
package_name = 'open_quadruped_control'
setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name, f'{package_name}.control_library'],
    package_dir={package_name: 'open_quadruped_control', f'{package_name}.control_library': 'open_quadruped_control/control_library'},
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='OpenQuadruped Maintainer',
    maintainer_email='you@example.com',
    description='ROS 2 control nodes for OpenQuadruped (rclpy port)',
    license='MIT',
    entry_points={
        'console_scripts': [
            'interface_process = open_quadruped_control.nodes.interface_process:main',
        ],
    },
)
