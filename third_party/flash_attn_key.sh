#!/usr/bin/env bash
#
# The identity of a prebuilt FlashAttention archive.
#
# Everything that decides what the archive contains, reduced to something short enough to put in
# a file name: the script that pins the FlashAttention and CUTLASS revisions, the sources and
# CMakeLists built from them, and the CUDA the whole thing is compiled against. Change any of
# those and the key changes with them, so an archive built before the change cannot be mistaken
# for one built after it.
#
# This exists so that staleness is impossible rather than unlikely. A workflow `paths:` filter
# answers "did a commit touch these files", which is not the same question as "is the archive we
# are about to link the one this source produces" -- a revert, a rebase, a deleted asset or a
# bumped toolkit all part the two. The build publishes under this key and the release asks for
# it by name, so the only archive either can reach is the right one.
#
# Blob hashes rather than file contents, because git already tracks exactly this and does it the
# same way on every host -- no line-ending or permission surprises between a Linux checkout and
# a Windows one.
set -euo pipefail

cuda_version="${1:?usage: flash_attn_key.sh <cuda-version>}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

{
  echo "cuda=${cuda_version}"
  git -C "${REPO_ROOT}" ls-files -s \
      third_party/install_flash_attn.sh \
      third_party/flash-attention/csrc/flash_attn
} | sha256sum | cut -c1-12
