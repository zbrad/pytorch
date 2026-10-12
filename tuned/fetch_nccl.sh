#!/bin/bash
# tuned/fetch_nccl.sh <release-tag> <dest-dir> — download a zbrad/nccl release, check the tarball
# against its published sha256, and extract it into <dest-dir> (the tarball's top directory is
# stripped, so <dest-dir>/lib/libnccl_static.a and <dest-dir>/include/nccl.h exist). Idempotent:
# a <dest-dir> already holding that tag is left alone. Needs `gh` logged in. Never fetches a tag
# whose files do not match their checksum, and never leaves a half-extracted <dest-dir>.
set -euo pipefail
TAG="${1:?usage: tuned/fetch_nccl.sh <release-tag> <dest-dir>}"
DEST="${2:?usage: tuned/fetch_nccl.sh <release-tag> <dest-dir>}"
REPO="zbrad/nccl"

if [[ -f "${DEST}/.tag" && "$(cat "${DEST}/.tag")" == "${TAG}" && -f "${DEST}/lib/libnccl_static.a" ]]; then
    exit 0
fi
command -v gh >/dev/null || { echo "ERROR: fetch_nccl.sh needs the gh CLI to download ${REPO} ${TAG}." >&2; exit 1; }

WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
echo "[tuned/fetch_nccl] downloading ${REPO} ${TAG}"
gh release download "${TAG}" --repo "${REPO}" --pattern '*.tar.gz' --pattern '*.tar.gz.sha256' --dir "${WORK}"
TARBALL="$(find "${WORK}" -maxdepth 1 -name '*.tar.gz' | head -1)"
[[ -f "${TARBALL}" && -f "${TARBALL}.sha256" ]] || { echo "ERROR: ${REPO} ${TAG} has no tarball and .sha256 asset pair." >&2; exit 1; }
(cd "${WORK}" && sha256sum -c "$(basename "${TARBALL}").sha256" >/dev/null) \
    || { echo "ERROR: ${TARBALL} does not match its published sha256." >&2; exit 1; }
tar -xzf "${TARBALL}" -C "${WORK}"
TOP="$(find "${WORK}" -mindepth 1 -maxdepth 1 -type d | head -1)"
grep -q "^tag=${TAG}$" "${TOP}/BUILD_INFO" || { echo "ERROR: BUILD_INFO in the tarball does not say tag=${TAG}." >&2; exit 1; }
[[ -f "${TOP}/lib/libnccl_static.a" && -f "${TOP}/include/nccl.h" ]] || { echo "ERROR: tarball lacks lib/libnccl_static.a or include/nccl.h." >&2; exit 1; }
echo "${TAG}" > "${TOP}/.tag"
mkdir -p "$(dirname "${DEST}")"
[[ "${DEST}" == */build/*/nccl/* ]] || { echo "ERROR: refusing to replace unexpected destination ${DEST}" >&2; exit 1; }
rm -rf -- "${DEST}"
mv "${TOP}" "${DEST}"
echo "[tuned/fetch_nccl] ${TAG} verified and extracted to ${DEST}"
