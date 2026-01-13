# ROS2 Humble & LiDAR (Slamtec C1) Setup Guide

Jetson Orin Nano 환경에서 ROS2 Humble을 설치하고 Slamtec C1 LiDAR를 설정 및 실행하는 가이드입니다.

## 환경 사양

| 항목 | 내용 |
|------|------|
| Device | Jetson Orin Nano |
| OS | Ubuntu 22.04.5 LTS (jammy) |
| JetPack | 6.2 |
| Architecture | arm64 (aarch64) |
| ROS2 Distribution | Humble |
| LiDAR Model | Slamtec C1 |

## 1. 환경 확인

### Ubuntu 버전 확인
```bash
lsb_release -a
# 출력: Ubuntu 22.04.5 LTS (jammy)
```

### 아키텍처 확인
```bash
dpkg --print-architecture
# 출력: arm64 (Jetson용 필수)
```

## 2. Universe Repository 활성화

ROS2 의존성 패키지를 위해 Universe 저장소를 활성화합니다.

```bash
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository universe
sudo apt update
```

### Repository 확인
```bash
grep -R "universe" /etc/apt/sources.list /etc/apt/sources.list.d/
```

## 3. 필수 도구 설치

```bash
sudo apt install -y curl gnupg lsb-release
```

## 4. ROS2 GPG Key 등록

```bash
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
```

### Key 파일 확인
```bash
ls -l /usr/share/keyrings/ros-archive-keyring.gpg
```

## 5. ROS2 Humble Repository 등록

```bash
echo "deb [arch=arm64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu jammy main" | \
sudo tee /etc/apt/sources.list.d/ros2.list
```

### Repository 파일 확인
```bash
cat /etc/apt/sources.list.d/ros2.list
```

## 6. 패키지 인덱스 업데이트

```bash
sudo apt update
# 출력에 "Get: http://packages.ros.org/ros2/ubuntu jammy/main arm64 Packages" 포함 확인
```

### ROS2 패키지 검색
```bash
apt search ros-humble-desktop
```

## 7. ROS2 Humble Desktop 설치

```bash
# Jetson 환경에 권장되는 Desktop 패키지 설치 (desktop-full 아님)
sudo apt install -y ros-humble-desktop
```

## 8. ROS2 환경 설정

### 환경 변수 로드
```bash
source /opt/ros/humble/setup.bash
```

### 자동 로드 설정 (영구 적용)
```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### ROS 배포판 확인
```bash
echo $ROS_DISTRO
# 출력: humble
```

## 9. 개발 도구 설치

### rosdep 설치 (의존성 관리)
```bash
sudo apt install -y python3-rosdep
```

### rosdep 초기화 (시스템당 1회)
```bash
sudo rosdep init
rosdep update
```

### colcon 설치 (ROS2 빌드 시스템)
```bash
sudo apt install -y python3-colcon-common-extensions
```

## 10. ROS2 Workspace 생성

```bash
# 표준 ROS2 workspace 생성
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws

# 초기 빌드 (빈 workspace도 정상)
colcon build

# Workspace overlay 적용
source install/setup.bash
```

### Workspace 자동 로드 설정
```bash
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
```

## 11. ROS2 통신 검증 (DDS Test)

두 개의 터미널에서 다음 명령을 실행하여 ROS2 통신을 확인합니다.

**터미널 1:**
```bash
ros2 run demo_nodes_cpp talker
```

**터미널 2:**
```bash
ros2 run demo_nodes_py listener
```

✅ 메시지가 수신되면 ROS2 설치 성공

---

## 12. Slamtec C1 LiDAR 드라이버 설치

### 소스코드 다운로드
```bash
cd ~/ros2_ws/src
git clone https://github.com/Slamtec/sllidar_ros2.git
```

**참고:** [Slamtec sllidar_ros2 GitHub Repository](https://github.com/Slamtec/sllidar_ros2)

## 13. 의존성 설치 및 빌드

```bash
cd ~/ros2_ws

# 의존성 자동 설치
rosdep update
rosdep install -i --from-path src --rosdistro humble -y

# 빌드 실행
colcon build --symlink-install
```

빌드 완료 후 `build`, `install`, `log` 디렉토리가 생성됩니다.

## 14. 환경 설정 적용

```bash
source install/setup.bash
```

⚠️ **매번 새 터미널을 열 때마다 실행 필요** (또는 이미 `.bashrc`에 등록되어 있으면 자동 적용)

## 15. USB 포트 권한 설정

LiDAR를 USB에 연결하면 일반적으로 `/dev/ttyUSB0`로 인식됩니다.

```bash
# 포트 권한 부여
sudo chmod 666 /dev/ttyUSB0
```

### 영구 권한 설정 (선택사항)
```bash
# udev rule 생성
sudo nano /etc/udev/rules.d/99-sllidar.rules

# 다음 내용 추가:
# KERNEL=="ttyUSB[0-9]*", MODE="0666"

# udev 규칙 재로드
sudo udevadm control --reload-rules
sudo udevadm trigger
```

## 16. LiDAR C1 실행

```bash
ros2 launch sllidar_ros2 sllidar_c1_launch.py
```

### 실행 확인
터미널에서 에러 메시지 없이 **"SLLidar 확인 완료"** 또는 유사한 메시지가 표시되어야 합니다.

## 17. RViz2를 사용한 시각화

### RViz2 실행
새 터미널을 열어 다음 명령 실행:

```bash
rviz2
```

### Fixed Frame 설정
1. 좌측 패널에서 **Global Options** → **Fixed Frame** 찾기
2. 기본값이 `map`으로 되어 있을 것입니다
3. `laser`로 직접 입력하여 변경

⚠️ **중요:** Fixed Frame이 올바르지 않으면 데이터가 화면에 표시되지 않습니다.

### LaserScan 토픽 추가
1. 좌측 하단의 **[Add]** 버튼 클릭
2. **[By topic]** 탭 선택
3. `/scan` 토픽 아래의 **LaserScan** 선택
4. **OK** 클릭

### 시각화 확인
- 화면 중심에 점(Point)들이 찍히는지 확인
- 점이 너무 작다면 LaserScan 설정의 **Size (m)** 값을 `0.05` 정도로 증가

## 18. LiDAR 데이터 검증

### Scan 데이터 확인
```bash
ros2 topic echo /scan --once
```

### 샘플 Python 스크립트 실행
```bash
python3 scan_exist_test.py \
  --rear \
  --fov_deg 0.25 \
  --max_range 1.5 \
  --min_hits 1
```

---

## LiDAR C1 기술 사양

| 파라미터 | 값 |
|---------|-----|
| **각도 범위** | -180° ~ +180° (360°) |
| **각도 분해능** | ~0.5° (0.00873878 rad) |
| **포인트 수** | ~720 points/scan |
| **거리 범위** | 0.05m ~ 16.0m |
| **Latency** | 100ms |
| **Scan Time** | ~0.099초 (10Hz) |

### 거리별 포인트 간격 (예시)

| 거리 | 포인트 간격 |
|------|------------|
| 1m | ~0.9cm |
| 6m | ~5.2cm |

### 각도 계산
```
각도 분해능 = 0.00873878 rad × (180 / π) ≈ 0.50°
전체 포인트 수 = 360° / 0.5° ≈ 720 points
```

---

## 최종 상태

✅ ROS2 Humble 설치 완료  
✅ ROS2 Workspace 생성 및 설정  
✅ rosdep 및 colcon 설치  
✅ Slamtec C1 LiDAR 드라이버 빌드  
✅ USB 포트 권한 설정  
✅ LiDAR 실행 및 RViz2 시각화  
✅ 실시간 환경 인식 데이터 수집 가능  

🚀 **Jetson Orin Nano 환경에서 LiDAR 기반 로봇 개발 준비 완료**

---

## 문제 해결 (Troubleshooting)

### LiDAR가 인식되지 않는 경우
```bash
# USB 장치 확인
lsusb

# 시리얼 포트 확인
ls -l /dev/ttyUSB*

# 권한 재설정
sudo chmod 666 /dev/ttyUSB0
```

### RViz2에서 데이터가 보이지 않는 경우
1. Fixed Frame이 `laser`로 설정되었는지 확인
2. `/scan` 토픽이 정상 발행되는지 확인:
```bash
ros2 topic list
ros2 topic hz /scan
```

### 빌드 오류 발생 시
```bash
# workspace 정리 후 재빌드
cd ~/ros2_ws
rm -rf build install log
colcon build --symlink-install
```

---

## 참고 자료

- [Slamtec sllidar_ros2 GitHub](https://github.com/Slamtec/sllidar_ros2)
- [ROS2 Humble Documentation](https://docs.ros.org/en/humble/)
- [Jetson Orin Nano Developer Guide](https://developer.nvidia.com/embedded/jetson-orin-nano)
