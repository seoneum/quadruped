from setuptools import setup
package_name = 'open_quadruped_nav'
setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    package_dir={package_name: 'open_quadruped_nav'},
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/slam_rgbd.launch.py', 'launch/slam_lidar.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='OpenQuadruped Maintainer',
    maintainer_email='you@example.com',
    description='Navigation/SLAM launch files for OpenQuadruped (rtabmap_ros / slam_toolbox).',
    license='MIT',
)
