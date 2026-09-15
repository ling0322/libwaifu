#!/usr/bin/env bash
#
# Wrap a freshly built FlashAttention archive, with a note saying what it is.
#
# Shared by the Windows and Linux halves of flash-attn.yml so the two produce the same shape:
# the same manifest fields, in the same order, beside a library whose only difference is the
# name its platform gives a static archive.
#
#   package_flash_attn.sh <platform> <library-name> <asset-name> <key>
set -euo pipefail

platform="${1:?platform}"
libname="${2:?library name}"
asset="${3:?asset name}"
key="${4:?key}"

lib="third_party/flash-attention/build/${libname}"

# The top-level CMakeLists looks for exactly this path, so a build that put something somewhere
# else has not produced anything -- and saying so here beats an EXISTS check failing a release
# three steps later with nothing to point at.
if [ ! -f "${lib}" ]; then
  echo "expected ${lib}, found:" >&2
  ls -la third_party/flash-attention/build/ >&2 || true
  exit 1
fi

rm -rf dist
mkdir -p dist
cp "${lib}" dist/

# Provenance, so a surprising archive can be read rather than guessed at. The key says which
# source built it; the rest says what it can be linked beside.
{
  echo "key:        ${key}"
  echo "platform:   ${platform}"
  echo "library:    ${libname}"
  echo "built:      $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "commit:     ${GITHUB_SHA:-unknown}"
  echo "run:        ${GITHUB_RUN_ID:-unknown}"
  echo "cuda:       ${CUDA_VERSION:-unknown}"
  echo "cutlass:    $(git -C third_party/flash-attention/csrc/cutlass rev-parse HEAD)"
  echo "arch:       $(grep -oE '[0-9]+-real' third_party/flash-attention/csrc/flash_attn/CMakeLists.txt | tr '\n' ' ')"
} > dist/MANIFEST.txt
cat dist/MANIFEST.txt

# Both produce a zip; which tool is present is what differs. A Windows runner has 7z and no zip
# worth relying on, and a Linux one has zip. Guessing wrong here would fail at the last step of
# an hour-long build, so the platform this was told about picks.
# Resolved before anything changes directory, so the archive lands where the caller named it
# whether that was a bare file name or a path.
case "${asset}" in
  /*) out="${asset}" ;;
  *)  out="$(pwd)/${asset}" ;;
esac

rm -f "${out}"
case "${platform}" in
  windows) (cd dist && 7z a "${out}" .) ;;
  *)       (cd dist && zip -qr "${out}" .) ;;
esac
echo "packaged ${asset}"
