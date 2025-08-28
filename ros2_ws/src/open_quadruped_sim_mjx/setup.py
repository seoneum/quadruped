from setuptools import setup
package_name = 'open_quadruped_sim_mjx'
setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    package_dir={package_name: 'open_quadruped_sim_mjx'},
    include_package_data=True,
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='OpenQuadruped Maintainer',
    maintainer_email='you@example.com',
    description='ROS 2 node integrating MuJoCo/MJX simulation with OpenQuadruped',
    license='MIT',
    entry_points={
        'console_scripts': [
            'mjx_sim = open_quadruped_sim_mjx.nodes.mjx_sim:main',
        ],
    },
)
