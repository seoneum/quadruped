# OpenQuadruped ROS2 + MuJoCo-MJX port

This repository contains a ROS 2 (rclpy) port of the OpenQuadruped project with scaffolding for MuJoCo/MJX-based simulation and uv-managed Python dependencies.

## Structure
- ros2_ws/src/open_quadruped_interfaces: ROS 2 msg (JointAngles)
- ros2_ws/src/open_quadruped_control: rclpy port of interface_process + vendored control_library
- ros2_ws/src/open_quadruped_sim_mjx: MuJoCo/MJX-driven simulator stub

## Build (colcon)
1) Source your ROS 2 distro (e.g., Humble):
   source /opt/ros/humble/setup.bash
2) Build:
   cd ros2_ws && rm -f src/COLCON_IGNORE && colcon build --symlink-install
3) Source overlay:
   source install/setup.bash

## Run
- Start joystick driver publishing sensor_msgs/msg/Joy
- Start control node:
  ros2 run open_quadruped_control interface_process
- Start MJX sim (requires mujoco + mujoco-mjx installed via uv):
  ros2 run open_quadruped_sim_mjx mjx_sim --ros-args -p mjcf_path:=/path/to/model.xml

## Python deps with uv
- Install uv: https://github.com/astral-sh/uv
- From repo root:
  uv venv && uv sync
  uv add mujoco mujoco-mjx jax jaxlib numpy
Note: ROS 2 Python packages (rclpy, sensor_msgs) are provided by your ROS 2 installation, not pip.

## Next steps
- Provide/convert an MJCF for the quadruped and map JointAngles to actuators
- Implement gait mode using the existing gait planner
- Add launch files


## RL training with MJX (ARS)
- Setup: `uv venv && uv sync`
- Run: `python rl/train_ars.py`
- Edit `rl/train_ars.py` to point to your MJCF and actuator names.
- For large-scale parallelism, increase `ARSCfg.n_envs` and ensure JAX is using GPU.
