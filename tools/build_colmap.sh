#!/usr/bin/env bash
# Build COLMAP with the CASPAR GPU bundle adjustment backend.
#
# COLMAP does not provide binaries with CASPAR, so we build it from source.
#
# PoseLib and faiss are fetched rather than taken from conda, because neither ships a cmake
# config in this environment.
set -euo pipefail

VERSION=4.1.1
: "${CONDA_PREFIX:?run inside an activated environment, use \`pixi run -e vision colmap-build\`}"
PREFIX="$CONDA_PREFIX"
SRC="${PIXI_PROJECT_ROOT:-$PWD}/.pixi-src/colmap/src"
BUILD="${PIXI_PROJECT_ROOT:-$PWD}/.pixi-src/colmap/build"

# A build with CASPAR_ENABLED off carries the guard message as well as the backend's name, so the
# absence of the guard is what separates the two.
caspar_enabled() {
    if grep -qa "built without CASPAR_ENABLED" "$1"; then
        return 1
    fi
    grep -qa CASPAR "$1"
}

if [ -x "$PREFIX/bin/colmap" ] && caspar_enabled "$PREFIX/bin/colmap"; then
    echo "colmap with CASPAR is already installed at $PREFIX/bin/colmap"
    echo "delete it to force a rebuild"
    exit 0
fi

if [ ! -d "$SRC" ]; then
    git clone --depth 1 --branch "$VERSION" https://github.com/colmap/colmap.git "$SRC"
fi

cmake -S "$SRC" -B "$BUILD" -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
    -DCUDA_ENABLED=ON \
    -DCASPAR_ENABLED=ON \
    -DCMAKE_CUDA_ARCHITECTURES=native \
    -DGUI_ENABLED=OFF \
    -DOPENGL_ENABLED=OFF \
    -DCGAL_ENABLED=OFF \
    -DLSD_ENABLED=OFF \
    -DONNX_ENABLED=ON \
    -DFETCH_ONNX=ON \
    -DFETCH_FAISS=ON \
    -DFETCH_POSELIB=ON
cmake --build "$BUILD" --target install

# This is now the only place CASPAR is verified, so fail here rather than an hour into a
# reconstruction that turned out to refuse the backend.
if ! caspar_enabled "$PREFIX/bin/colmap"; then
    echo "built $PREFIX/bin/colmap, but it carries no CASPAR backend" >&2
    exit 1
fi
echo "built $PREFIX/bin/colmap with CASPAR"
