#!/usr/bin/env bash
#
# Fetch the prebuilt FlashAttention archive belonging to the source in this checkout.
#
#   fetch_flash_attn.sh <platform> <library-name>
#
# The name is computed rather than chosen: third_party/flash_attn_key.sh hashes everything that
# decides what the archive contains, so an archive built from other source has a different name
# and cannot be picked up by accident. Finding nothing is a failure that says which key it wanted
# -- silently linking an older archive would be the bad outcome this is arranged to prevent.
set -euo pipefail

platform="${1:?platform}"
libname="${2:?library name}"

key=$(bash third_party/flash_attn_key.sh "${CUDA_VERSION}")
asset="flash_attn-${platform}-cuda${CUDA_VERSION}-${key}.zip"
echo "looking for ${asset}"

if ! gh release download "${PREBUILT_TAG}" --pattern "${asset}" --dir prebuilt; then
  echo "::error::No prebuilt FlashAttention for ${platform}, key ${key}."
  echo "::error::Run the 'FlashAttention prebuild' workflow on this commit, then tag again."
  exit 1
fi

mkdir -p third_party/flash-attention/build
case "${platform}" in
  windows) 7z x "prebuilt/${asset}" -othird_party/flash-attention/build -y ;;
  *)       unzip -o "prebuilt/${asset}" -d third_party/flash-attention/build ;;
esac
cat third_party/flash-attention/build/MANIFEST.txt

# The top-level CMakeLists looks for exactly this, and would otherwise say so much later.
test -f "third_party/flash-attention/build/${libname}"
echo "have third_party/flash-attention/build/${libname}"
