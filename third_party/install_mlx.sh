#!/usr/bin/env bash

set -euo pipefail

MLX_VERSION="v0.32.2"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MLX_DIR="${SCRIPT_DIR}/mlx"
MLX_PATCH="${SCRIPT_DIR}/mlx_metallib_from_memory.patch"
BUILD_DIR="${1:-${MLX_DIR}/build}"

if [[ "$(uname -s)" != "Darwin" ]] || [[ "$(uname -m)" != "arm64" ]]; then
  echo "error: MLX with the Metal backend only builds on Apple Silicon macOS" >&2
  exit 1
fi

if ! command -v cmake >/dev/null 2>&1; then
  echo "error: cmake (>= 3.25) is required to build MLX" >&2
  exit 1
fi

if ! metal_check="$(xcrun -sdk macosx metal --version 2>&1)"; then
  echo "error: the Metal shader compiler is unavailable:" >&2
  echo "  ${metal_check}" >&2
  if [[ "${metal_check}" == *"Metal Toolchain"* ]]; then
    # Xcode 26 splits the shader compiler out into a downloadable component. Note the
    # missing sudo: the download runs through the calling user's MobileAsset session,
    # and as root it fails with "Failed fetching catalog".
    echo "hint: run 'xcodebuild -downloadComponent MetalToolchain' (without sudo)." >&2
  else
    echo "hint: MLX needs a full Xcode install (the Command Line Tools alone do not ship" \
         "the shader compiler). Install Xcode, then run" \
         "'sudo xcode-select -s /Applications/Xcode.app'," \
         "'sudo xcodebuild -license accept' and 'sudo xcodebuild -runFirstLaunch'." >&2
  fi
  exit 1
fi

if [[ ! -d "${MLX_DIR}" ]]; then
  echo "==> Cloning MLX ${MLX_VERSION}"
  git clone \
    --branch "${MLX_VERSION}" \
    --depth 1 \
    https://github.com/ml-explore/mlx.git \
    "${MLX_DIR}"
elif [[ ! -d "${MLX_DIR}/.git" ]] || \
     [[ "$(git -C "${MLX_DIR}" describe --tags --exact-match 2>/dev/null)" != "${MLX_VERSION}" ]]; then
  echo "error: ${MLX_DIR} is not MLX ${MLX_VERSION}" >&2
  exit 1
fi

echo "==> MLX ${MLX_VERSION} is available at ${MLX_DIR}"

# Upstream can only load mlx.metallib from a file path. libwaifu embeds it in the binary
# instead, so it needs the in-memory entry point this patch adds. `git apply --reverse
# --check` is how we tell "already patched" from "patch does not fit", which keeps the
# script idempotent across re-runs.
if git -C "${MLX_DIR}" apply --reverse --check "${MLX_PATCH}" >/dev/null 2>&1; then
  echo "==> ${MLX_PATCH##*/} is already applied"
elif git -C "${MLX_DIR}" apply "${MLX_PATCH}"; then
  echo "==> Applied ${MLX_PATCH##*/}"
else
  echo "error: failed to apply ${MLX_PATCH}" >&2
  exit 1
fi

echo "==> Configuring MLX in ${BUILD_DIR}"
cmake -S "${MLX_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DMLX_BUILD_METAL=ON \
  -DMLX_BUILD_CPU=ON \
  -DMLX_BUILD_TESTS=OFF \
  -DMLX_BUILD_EXAMPLES=OFF \
  -DMLX_BUILD_BENCHMARKS=OFF \
  -DMLX_BUILD_PYTHON_BINDINGS=OFF \
  -DMLX_BUILD_PYTHON_STUBS=OFF

echo "==> Building mlx (static)"
cmake --build "${BUILD_DIR}" --target mlx --parallel "$(sysctl -n hw.ncpu)"

MLX_LIB="${BUILD_DIR}/libmlx.a"
if [[ ! -f "${MLX_LIB}" ]]; then
  echo "error: build finished but ${MLX_LIB} was not produced" >&2
  exit 1
fi

echo "==> mlx is available at ${MLX_LIB}"
exit 0
