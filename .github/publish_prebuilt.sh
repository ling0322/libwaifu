#!/usr/bin/env bash
#
# Put a prebuilt archive into the release that holds them, creating that release the first time.
#
#   publish_prebuilt.sh <tag> <asset>
#
# A prerelease, so a dependency never presents itself as the project's latest release. Assets are
# added rather than replaced: each platform and key is a different archive, and a release tag cut
# months ago should still find the one it was built against.
set -euo pipefail

tag="${1:?tag}"
asset="${2:?asset}"

notes="Prebuilt FlashAttention static libraries, linked by the release workflow.

One asset per platform per build key -- see third_party/flash_attn_key.sh -- each carrying a
MANIFEST.txt naming the CUDA version, CUTLASS revision and architectures it was built for.
Built by .github/workflows/flash-attn.yml."

# Two runners can reach this at the same moment, and the loser of that race would otherwise fail
# on a release the winner had just made. Create, and if it already exists, carry on.
gh release view "${tag}" >/dev/null 2>&1 || \
  gh release create "${tag}" \
    --prerelease \
    --title "FlashAttention prebuilt archives" \
    --notes "${notes}" \
  || gh release view "${tag}" >/dev/null

gh release upload "${tag}" "${asset}" --clobber
echo "published ${asset} to ${tag}"
