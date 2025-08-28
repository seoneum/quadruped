# OpenQuadruped ROS2 + MuJoCo-MJX (Quardred-based)

ROS 2 Humble + MuJoCo/MJX 기반의 사족보행 로봇 프로젝트 스캐폴딩입니다. 이 레포는 Quardred_08272115_minimum.xml을 기준으로 ROS 2 노드, 시뮬, 센서 퍼블리셔, MJX 병렬 RL 훈련(ARS)을 제공합니다. Python 의존성은 uv로 관리합니다.

## 프로젝트 구조
- ros2_ws/src/open_quadruped_interfaces: ROS 2 msg (JointAngles)
- ros2_ws/src/open_quadruped_control: rclpy 포트(interface_process) + vendored control_library
- ros2_ws/src/open_quadruped_sim_mjx: MuJoCo/MJX 시뮬레이터 + RGBD/LiDAR 퍼블리셔 + 런치 파일 + MJCF 자산
  - open_quadruped_sim_mjx/assets/Quardred_08272115_minimum.xml: 메인 MJCF (관절/모터/카메라 포함)
  - open_quadruped_sim_mjx/assets/meshes/all_meshes_Quardred_08272115_minimum_YYYY-MM-DD.tar.gz: 모든 STL이 들어있는 단일 아카이브(개별 STL은 깃에서 제거)
- rl: MJX 병렬 환경 + ARS 학습/평가 스크립트
- scripts/mjcf_validate.py: STL 존재, 액추에이터/조인트 목록 확인 도구

## 의존성 설치(uv)
1) uv 설치: https://github.com/astral-sh/uv
2) 레포 루트에서:
   uv venv && uv sync
   # 필요시 수동 추가
   uv add mujoco mujoco-mjx jax jaxlib numpy

ROS 2 Python 패키지(rclpy, sensor_msgs 등)는 ROS 2 설치(apt)로 제공됩니다.

## STL 자산(단일 아카이브)
- 경로: ros2_ws/src/open_quadruped_sim_mjx/open_quadruped_sim_mjx/assets/meshes/all_meshes_Quardred_08272115_minimum_YYYY-MM-DD.tar.gz
- 개별 STL은 깃에서 제거되어 있습니다. MuJoCo는 개별 파일을 읽으므로, 풀어서 사용해야 합니다.
- 풀기:
  cd ros2_ws/src/open_quadruped_sim_mjx/open_quadruped_sim_mjx/assets/meshes
  tar -xzf all_meshes_Quardred_08272115_minimum_*.tar.gz

## 빌드(colcon)
1) ROS 2 Humble 설정:
   source /opt/ros/humble/setup.bash
2) 빌드:
   cd ros2_ws && rm -f src/COLCON_IGNORE && colcon build --symlink-install
3) 오버레이 소스:
   source install/setup.bash

## 시뮬 레디(Quardred 기준)
- MJCF: assets/Quardred_08272115_minimum.xml
- 관절(8개): Left_Hip_Joint, Right_Hip_Joint, Left_Hip_Joint_1, Right_Hip_Joint_1, Lower_Leg_33, Lower_Leg_134, Lower_Leg_1, Lower_Leg_4_1
- 모터(8개): act_<joint_name> (±2.3 N·m, ctrl ∈ [-1,1])
- 카메라: track(기본 RGBD), lidar_cam(깊이→PointCloud 변환용)
- ROS 2 런치(모든 것):
  ros2 launch open_quadruped_sim_mjx all_sim.launch.py mjcf_path:=<절대경로>/ros2_ws/src/open_quadruped_sim_mjx/open_quadruped_sim_mjx/assets/Quardred_08272115_minimum.xml camera:=track
- 기본 actuator_mapping(다리 순서 [hip, knee]):
  fl: [act_Left_Hip_Joint, act_Lower_Leg_33]
  fr: [act_Right_Hip_Joint, act_Lower_Leg_134]
  bl: [act_Left_Hip_Joint_1, act_Lower_Leg_1]
  br: [act_Right_Hip_Joint_1, act_Lower_Leg_4_1]

## 센서 토픽
- RGB: /camera/image_raw (encoding rgb8)
- Depth: /camera/depth/image_raw (encoding 32FC1, meters)
- CameraInfo: /camera/camera_info
- LiDAR 포인트클라우드(Depth→PointCloud2): /lidar/points

## SLAM 런치
- RGBD(rtabmap_ros):
  ros2 launch open_quadruped_nav slam_rgbd.launch.py
- LiDAR(slam_toolbox):
  ros2 launch open_quadruped_nav slam_lidar.launch.py

## RL: MJX 병렬 ARS
- 학습:
  uv run python rl/train_ars.py
  - 내부 기본 MJCF: Quardred_08272115_minimum.xml
  - actuator_names: 위 매핑과 동일 순서
  - 설정: ARSCfg(n_envs=64, horizon=500, …)
- 평가:
  uv run python rl/eval_ars.py <checkpoint.npz>

## 개발 팁
- STL이 풀려 있어야 시뮬이 메시에 접근합니다. 실행 전 meshes 폴더에 아카이브를 반드시 풀어주세요.
- 액추에이터 토크 한계(±2.3 N·m)는 PDI-HV5523MG 스펙 반영 값입니다. 변경 필요시 MJCF와 코드의 torque_limit을 함께 수정하세요.
- 카메라 fovy/마운트는 기본값으로 설정되어 있으며, 실제 모델에 맞춰 조정 가능.
