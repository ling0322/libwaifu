#!/usr/bin/env bash

set -euo pipefail

FLASH_ATTN_CUTLASS_REVISION="dc4817921edda44a549197ff3a9dcf5df0636e7b"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${1:-${SCRIPT_DIR}/flash-attention/build}"
FLASH_ATTN_DIR="${SCRIPT_DIR}/flash-attention/csrc/flash_attn"
FLASH_ATTN_CUTLASS_DIR="${SCRIPT_DIR}/flash-attention/csrc/cutlass"

if [[ ! -d "${FLASH_ATTN_CUTLASS_DIR}" ]]; then
  echo "==> Cloning FlashAttention CUTLASS ${FLASH_ATTN_CUTLASS_REVISION}"
  git init --quiet "${FLASH_ATTN_CUTLASS_DIR}"
  git -C "${FLASH_ATTN_CUTLASS_DIR}" remote add origin https://github.com/NVIDIA/cutlass.git
  git -C "${FLASH_ATTN_CUTLASS_DIR}" fetch --quiet --depth 1 origin "${FLASH_ATTN_CUTLASS_REVISION}"
  git -C "${FLASH_ATTN_CUTLASS_DIR}" checkout --quiet --detach FETCH_HEAD
elif [[ ! -d "${FLASH_ATTN_CUTLASS_DIR}/.git" ]] || \
     [[ "$(git -C "${FLASH_ATTN_CUTLASS_DIR}" rev-parse HEAD)" != "${FLASH_ATTN_CUTLASS_REVISION}" ]]; then
  echo "error: ${FLASH_ATTN_CUTLASS_DIR} is not CUTLASS ${FLASH_ATTN_CUTLASS_REVISION}" >&2
  exit 1
fi

echo "==> FlashAttention CUTLASS ${FLASH_ATTN_CUTLASS_REVISION} is available at ${FLASH_ATTN_CUTLASS_DIR}"

# How many kernels to compile at once.
#
# One of these takes about 3 GB in cicc, so the usual "one job per core" turns a machine with
# more cores than memory into a swap storm or an OOM kill -- which is why this used to build
# serially whenever it could not hand the job to systemd. Serial is a poor trade on a machine
# that has the memory: the count is worked out from memory instead, enough jobs to fill
# FLASH_ATTN_MEMORY_SHARE percent of what is actually available, and never more than there are
# cores to run them on.
FLASH_ATTN_KERNEL_MEMORY_GB="${FLASH_ATTN_KERNEL_MEMORY_GB:-3}"
FLASH_ATTN_MEMORY_SHARE="${FLASH_ATTN_MEMORY_SHARE:-70}"

# The memory this build may use, in whole GB.
memory_gb() {
  local total=0 limit

  if [[ -r /proc/meminfo ]]; then
    total=$(awk '/^MemTotal:/ { print int($2 / 1048576) }' /proc/meminfo)
  elif command -v sysctl >/dev/null 2>&1; then
    total=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 ))
  fi

  # In a container /proc/meminfo is the host's, while the cgroup is what this process may
  # actually have; take whichever is smaller so the build is not sized for memory it cannot get.
  local limit_file
  for limit_file in /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory/memory.limit_in_bytes; do
    [[ -r "${limit_file}" ]] || continue
    limit=$(cat "${limit_file}")
    [[ "${limit}" =~ ^[0-9]+$ ]] || continue
    limit=$(( limit / 1073741824 ))
    if (( limit > 0 && (total == 0 || limit < total) )); then
      total=${limit}
    fi
  done

  echo "${total}"
}

parallel_jobs() {
  local total cores jobs
  total=$(memory_gb)
  cores=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)

  # Unknown memory is treated as no memory to spare, which is the old serial behaviour.
  jobs=$(( total * FLASH_ATTN_MEMORY_SHARE / 100 / FLASH_ATTN_KERNEL_MEMORY_GB ))
  (( jobs < 1 )) && jobs=1
  (( jobs > cores )) && jobs=${cores}
  echo "${jobs}"
}

FLASH_ATTN_PARALLEL="${FLASH_ATTN_PARALLEL:-$(parallel_jobs)}"
export CMAKE_BUILD_PARALLEL_LEVEL="${FLASH_ATTN_PARALLEL}"
export MAKEFLAGS="-j${FLASH_ATTN_PARALLEL}"
export NINJAFLAGS="-j${FLASH_ATTN_PARALLEL}"

echo "==> Configuring FlashAttention in ${BUILD_DIR}"
cmake_args=(-S "${FLASH_ATTN_DIR}" -B "${BUILD_DIR}")
if [[ -n "${FLASH_ATTN_CUDA_ARCH:-}" ]]; then
  cmake_args+=(-DCMAKE_CUDA_ARCHITECTURES="${FLASH_ATTN_CUDA_ARCH}")
fi
cmake "${cmake_args[@]}"

echo "==> Building flash_attn (${FLASH_ATTN_PARALLEL} at a time, of $(memory_gb) GB and $(nproc 2>/dev/null || echo '?') cores)"
# Where systemd is there, the job count is still a guess and the scope is what makes it safe: a
# kernel that takes more than its share is stopped rather than left to take the machine down.
if command -v systemd-run >/dev/null 2>&1 && [[ -n "${XDG_RUNTIME_DIR:-}" ]]; then
  systemd-run --user --scope --quiet \
    -p MemoryMax="${FLASH_ATTN_MEMORY_MAX:-80%}" \
    -p MemorySwapMax=2G \
    cmake --build "${BUILD_DIR}" --target flash_attn --parallel "${FLASH_ATTN_PARALLEL}"
else
  cmake --build "${BUILD_DIR}" --target flash_attn --parallel "${FLASH_ATTN_PARALLEL}"
fi

echo "==> flash_attn is available at ${BUILD_DIR}"
exit 0
