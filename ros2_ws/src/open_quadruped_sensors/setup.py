from setuptools import setup

package_name = 'open_quadruped_sensors'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/livox_mid360.launch.py', 'launch/orbbec_astra.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='seoneum',
    maintainer_email='seoneum@users.noreply.github.com',
    description='Hardware sensor integration for Livox MID-360 and Orbbec Astra',
    license='MIT',
)
