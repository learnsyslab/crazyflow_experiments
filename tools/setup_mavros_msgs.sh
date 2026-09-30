#!/usr/bin/env bash

# Builds mavros_msgs into ros_ws, since robostack ships no package for it. The PX4 deployment
# (deploy/px4.py) needs it to talk to the MAVROS instance on the companion computer. Only the
# message package is checked out, the rest of MAVROS would need MAVLink and GeographicLib.
#
# Pixi sources activation scripts, so exiting aborts activation and makes pixi report the failure.
# Each step is checked explicitly rather than through `set -euo pipefail`, because the ROS setup
# script sourced at the end relies on unset variables and on commands that return non-zero.

if [ -z "${PIXI_PROJECT_ROOT:-}" ]; then
  echo "[Setup MAVROS msgs] ERROR: not running inside a Pixi environment." >&2
  exit 1
fi

ROS_WS="${PIXI_PROJECT_ROOT}/ros_ws"
MAVROS_SRC="${ROS_WS}/src/mavros"
# Pin to the MAVROS version running on the companion computer so the message definitions match,
# e.g. MAVROS_MSGS_REF=2.10.1. `ros2 pkg xml mavros_msgs | grep version` prints the installed one.
MAVROS_MSGS_REF="${MAVROS_MSGS_REF:-ros2}"

if ! mkdir -p "${ROS_WS}/src"; then
  echo "[Setup MAVROS msgs] ERROR: could not create ${ROS_WS}/src." >&2
  exit 1
fi

if [ ! -d "${MAVROS_SRC}/.git" ]; then
  echo "[Setup MAVROS msgs] Cloning mavros_msgs (${MAVROS_MSGS_REF})..."
  if ! git clone --depth 1 --filter=blob:none --sparse --branch "${MAVROS_MSGS_REF}" \
    https://github.com/mavlink/mavros "${MAVROS_SRC}"; then
    echo "[Setup MAVROS msgs] ERROR: failed to clone mavros." >&2
    exit 1
  fi
  if ! git -C "${MAVROS_SRC}" sparse-checkout set mavros_msgs; then
    echo "[Setup MAVROS msgs] ERROR: failed to check out mavros_msgs." >&2
    exit 1
  fi
fi

# The marker is the installed package rather than install/setup.sh, which setup_mocap.sh creates as
# well: a --packages-select that matches nothing leaves colcon at exit code 0, so only the package
# itself proves the build ran.
MAVROS_MSGS_MARKER="${ROS_WS}/install/mavros_msgs/share/mavros_msgs/package.xml"

if [ ! -f "${MAVROS_MSGS_MARKER}" ]; then
  echo "[Setup MAVROS msgs] Running colcon build..."
  if ! (cd "${ROS_WS}" && colcon build --packages-select mavros_msgs); then
    echo "[Setup MAVROS msgs] ERROR: colcon build of mavros_msgs failed in ros_ws." >&2
    exit 1
  fi
fi

if [ ! -f "${MAVROS_MSGS_MARKER}" ]; then
  echo "[Setup MAVROS msgs] ERROR: ${MAVROS_MSGS_MARKER} is missing after the build." >&2
  exit 1
fi

. "${ROS_WS}/install/setup.sh"
