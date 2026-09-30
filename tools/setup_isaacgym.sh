#!/usr/bin/env bash
set -euo pipefail

# Fetch IsaacGym, which is not on PyPI. Pixi resolves the path of the isaacgym pypi dependency
# while solving the aerial-gym environment, so this has to run before `pixi install` rather than
# from an activation script, which would only run after that environment already exists.
#
# Unlike the activation scripts this one is executed rather than sourced, so exiting is safe.

DOWNLOAD_URL="https://developer.nvidia.com/isaac-gym-preview-4"

if [ -z "${PIXI_PROJECT_ROOT:-}" ]; then
  echo "[Setup IsaacGym] ERROR: not running inside a Pixi environment." >&2
  exit 1
fi

ISAACGYM_DIR="${PIXI_PROJECT_ROOT}/submodules/isaacgym"

if [ -d "${ISAACGYM_DIR}" ]; then
  echo "[Setup IsaacGym] Already present at ${ISAACGYM_DIR}."
  exit 0
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "[Setup IsaacGym] ERROR: curl is not installed." >&2
  exit 1
fi

archive=$(mktemp --suffix=.tar.gz)
trap 'rm -f "${archive}"' EXIT

echo "[Setup IsaacGym] Downloading IsaacGym Preview 4 (about 200 MB)..."
echo "[Setup IsaacGym] It is covered by the NVIDIA license shipped inside the archive."
if ! curl -fL --retry 3 -o "${archive}" "${DOWNLOAD_URL}"; then
  echo "[Setup IsaacGym] ERROR: download failed from ${DOWNLOAD_URL}." >&2
  exit 1
fi

echo "[Setup IsaacGym] Extracting to submodules..."
if ! tar xzf "${archive}" -C "${PIXI_PROJECT_ROOT}/submodules"; then
  echo "[Setup IsaacGym] ERROR: could not extract the archive." >&2
  exit 1
fi

if [ ! -d "${ISAACGYM_DIR}/python" ]; then
  echo "[Setup IsaacGym] ERROR: ${ISAACGYM_DIR}/python is missing after extraction." >&2
  echo "[Setup IsaacGym] The archive layout changed, so the pypi dependency path no longer fits." >&2
  exit 1
fi

echo "[Setup IsaacGym] IsaacGym is ready. Run 'pixi install' next."
