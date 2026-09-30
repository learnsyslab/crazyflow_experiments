#!/usr/bin/env bash
set -euo pipefail

# Pixi sources activation scripts, so any exit aborts activation. The success path must therefore
# fall through to the end instead of exiting. For the same reason this script uses `git -C` rather
# than `cd`, which would move the caller's working directory.

if [ -z "${PIXI_PROJECT_ROOT:-}" ]; then
  echo "[Setup Flightmare] ERROR: not running inside a Pixi environment." >&2
  exit 1
fi

if [ ! -f "${PIXI_PROJECT_ROOT}/pixi.lock" ]; then
  echo "[Setup Flightmare] ERROR: pixi environment is not properly set up." >&2
  exit 1
fi

FLIGHTMARE_DIR="${PIXI_PROJECT_ROOT}/submodules/flightmare"
PATCH_FILE="${PIXI_PROJECT_ROOT}/tools/flightmare_build.patch"
BUILD_DIR="${FLIGHTMARE_DIR}/flightlib/build"
SITE_PACKAGES="$(python -c 'import site; print(site.getsitepackages()[0])')"

# flightlib resolves its environment quadrotor configuration relative to this variable
export FLIGHTMARE_PATH="${FLIGHTMARE_DIR}"

if [ ! -e "${FLIGHTMARE_DIR}/.git" ]; then
  echo "[Setup Flightmare] ERROR: submodules/flightmare is not checked out." >&2
  echo "[Setup Flightmare] Run 'git submodule update --init submodules/flightmare' first." >&2
  exit 1
fi

# Flightmare uses master for googletest, pybind11 and yaml-cpp, but the branches no longer exist. We
# patch the deps to fixed versions instead
if git -C "${FLIGHTMARE_DIR}" apply --reverse --check "${PATCH_FILE}" 2>/dev/null; then
  echo "[Setup Flightmare] Build patch already applied."
elif git -C "${FLIGHTMARE_DIR}" apply --check "${PATCH_FILE}" 2>/dev/null; then
  echo "[Setup Flightmare] Applying build patch..."
  git -C "${FLIGHTMARE_DIR}" apply "${PATCH_FILE}"
else
  echo "[Setup Flightmare] ERROR: tools/flightmare_build.patch does not apply." >&2
  echo "[Setup Flightmare] The submodule likely moved off the pinned commit. Refresh the patch." >&2
  exit 1
fi

# flightlib's setup.py deletes build/ on every run. Pixi invokes it, so we would rebuild every time.
# Instead, we build it once and copy the resulting flightgym*.so into site-packages.
# EIGEN_FROM_SYSTTEM is turned off so Eigen comes from the flightlib pin. The misspelling is
# intentional to match the CMake variable in flightlib.
if [ ! -e "${SITE_PACKAGES}"/flightgym*.so ]; then
  echo "[Setup Flightmare] Building flightgym, this takes a few minutes..."
  git -C "${FLIGHTMARE_DIR}" clean -qfdx flightlib/build
  mkdir -p "${BUILD_DIR}"
  # flightlib expects zmq and zmqpp where its Ubuntu image kept them, on the default include path.
  # Conda scopes OpenCV to its own subdirectory, so the environment root is added back explicitly.
  cmake -S "${FLIGHTMARE_DIR}/flightlib" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-I${CONDA_PREFIX}/include" \
    -DEIGEN_FROM_SYSTTEM=OFF -DPYTHON_EXECUTABLE="$(which python)" > "${BUILD_DIR}/cmake.log" 2>&1
  cmake --build "${BUILD_DIR}" -j "$(nproc)" > "${BUILD_DIR}/build.log" 2>&1
  # -Ofast silently breaks flightlib on new compilers. The test suite covers this, so we run it once
  # to make sure everything is working.
  echo "[Setup Flightmare] Verifying the build against the flightlib test suite..."
  (cd "${BUILD_DIR}" && ./test_lib > "${BUILD_DIR}/test.log" 2>&1) || {
    echo "[Setup Flightmare] ERROR: flightlib test suite failed, see ${BUILD_DIR}/test.log" >&2
    exit 1
  }
  cp "${BUILD_DIR}"/flightgym*.so "${SITE_PACKAGES}/"
  echo "[Setup Flightmare] Built and verified flightgym."
fi

# The Unity player renders the camera images. It is a release asset, so it is fetched into
# flightrender once.
UNITY_URL="https://github.com/uzh-rpg/flightmare/releases/download/0.0.5/RPG_Flightmare.tar.xz"
RENDER_DIR="${FLIGHTMARE_DIR}/flightrender"
if [ -z "$(find "${RENDER_DIR}" -name RPG_Flightmare.x86_64 -type f)" ]; then
  echo "[Setup Flightmare] Downloading the Unity player (about 900 MB)..."
  archive=$(mktemp --suffix=.tar.xz)
  if ! curl -fL --retry 3 -o "${archive}" "${UNITY_URL}"; then
    rm -f "${archive}"
    echo "[Setup Flightmare] ERROR: download failed from ${UNITY_URL}." >&2
    exit 1
  fi
  if ! tar xJf "${archive}" -C "${RENDER_DIR}"; then
    rm -f "${archive}"
    echo "[Setup Flightmare] ERROR: could not extract the Unity player archive." >&2
    exit 1
  fi
  rm -f "${archive}"
  binary=$(find "${RENDER_DIR}" -name RPG_Flightmare.x86_64 -type f)
  if [ -z "${binary}" ]; then
    echo "[Setup Flightmare] ERROR: RPG_Flightmare.x86_64 is missing after extraction." >&2
    echo "[Setup Flightmare] The archive layout changed, so the render benchmark cannot find it." >&2
    exit 1
  fi
  chmod +x "${binary}"
  echo "[Setup Flightmare] Downloaded the Unity player."
fi

echo "[Setup Flightmare] Flightmare is ready!"
