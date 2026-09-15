#!/usr/bin/env bash
#
# Put a built artifact on the release its tag names, making that release the first time.
#
#   attach_release.sh <artifact>
#
# Shared by the platform jobs, which finish in whatever order they finish: `create` covers the
# first one to arrive, and the `||` covers everyone after it, including a re-run.
set -euo pipefail

artifact="${1:?artifact}"

# An alpha is not what someone arriving at the repository should be handed as the current
# release, so anything that says so is marked for what it is.
prerelease=""
case "${GITHUB_REF_NAME}" in
  *alpha*|*beta*|*rc*) prerelease="--prerelease" ;;
esac

gh release create "${GITHUB_REF_NAME}" --generate-notes ${prerelease} "${artifact}" \
  || gh release upload "${GITHUB_REF_NAME}" "${artifact}" --clobber

echo "attached ${artifact} to ${GITHUB_REF_NAME}"
