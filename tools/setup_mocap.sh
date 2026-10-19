#!/usr/bin/env bash

# Pixi sources activation scripts, so exiting aborts activation and makes pixi report the failure.
# Each step is checked explicitly rather than through `set -euo pipefail`, because the ROS setup
# script sourced at the end relies on unset variables and on commands that return non-zero.

if [ -z "${PIXI_PROJECT_ROOT:-}" ]; then
  echo "[Setup Mocap] ERROR: not running inside a Pixi environment." >&2
  exit 1
fi

ROS_WS="${PIXI_PROJECT_ROOT}/ros_ws"

MOCAP_MARKER="${ROS_WS}/install/motion_capture_tracking/share/motion_capture_tracking/package.xml"

# Pixi can activate environments in parallel (e.g. when installing all envs). This executes the
# setup scripts concurrently. We lock builds to avoid races.
(
  flock 9

  if ! mkdir -p "${ROS_WS}/src"; then
    echo "[Setup Mocap] ERROR: could not create ${ROS_WS}/src." >&2
    exit 1
  fi

  if [ ! -d "${ROS_WS}/src/motion_capture_tracking/.git" ]; then
    echo "[Setup Mocap] Cloning motion_capture_tracking..."
    if ! git clone --recurse-submodules https://github.com/utiasDSL/motion_capture_tracking \
      "${ROS_WS}/src/motion_capture_tracking"; then
      echo "[Setup Mocap] ERROR: failed to clone motion_capture_tracking." >&2
      exit 1
    fi
  fi

  if [ ! -f "${MOCAP_MARKER}" ]; then
    echo "[Setup Mocap] Running colcon build..."
    if ! (cd "${ROS_WS}" && colcon build --packages-select motion_capture_tracking \
      --cmake-args -DCMAKE_POLICY_VERSION_MINIMUM=3.5); then
      echo "[Setup Mocap] ERROR: colcon build of motion_capture_tracking failed in ros_ws." >&2
      exit 1
    fi
  fi
) 9>"${PIXI_PROJECT_ROOT}/.pixi/setup.lock" || exit 1

if [ ! -f "${MOCAP_MARKER}" ]; then
  echo "[Setup Mocap] ERROR: ${MOCAP_MARKER} is missing after the build." >&2
  exit 1
fi

. "${ROS_WS}/install/setup.sh"
