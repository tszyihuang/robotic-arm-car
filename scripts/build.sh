#!/usr/bin/env bash
set -euo pipefail
CAR_PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
CAR_ROS_SETUP="/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
if [[ ! -f "$CAR_ROS_SETUP" ]]; then
  echo "找不到 $CAR_ROS_SETUP，请先安装 ROS 2。" >&2
  exit 1
fi
set +u
source "$CAR_ROS_SETUP"
set -u
cd "$CAR_PROJECT_DIR"
colcon build --symlink-install --base-paths src --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3 "$@"
