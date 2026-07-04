#!/usr/bin/env bash
# Download, verify, and (for zips) extract the datasets that exceed the project's git
# budget (GitLab free tier: 10 GiB repo+LFS per project) and therefore live only on the
# workstation at /data/shared/raw. Every URL, size, and sha256 comes from data/manifest.csv;
# rows whose source is a MAST product label resolve through the public MAST download API.
set -euo pipefail

MANIFEST="${FETCH_MANIFEST:-$(cd "$(dirname "$0")" && pwd)/manifest.csv}"
DEST_ROOT="${FETCH_DEST_ROOT:-/data/shared/raw}"
DATASETS="${FETCH_DATASETS:-speed_plus,ifu_cubes}"
EXTRACT="${FETCH_EXTRACT:-1}"
CURL_RETRIES="${FETCH_CURL_RETRIES:-5}"
MAST_API="${FETCH_MAST_API:-https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:JWST/product}"

fail() { echo "[fail] $*" >&2; exit 1; }

if [ ! -f "${MANIFEST}" ]; then
  fail "manifest not found at ${MANIFEST}; set FETCH_MANIFEST"
fi

fetch_row() {
  local relpath="$1" bytes="$2" sha_expected="$3" source="$4" dest url got
  dest="${DEST_ROOT}/${relpath}"
  case "${source}" in
    http://*|https://*) url="${source}" ;;
    MAST:*) url="${MAST_API}/$(basename "${relpath}")" ;;
    *) fail "row ${relpath} has source '${source}' which is neither a URL nor a MAST label" ;;
  esac
  mkdir -p "$(dirname "${dest}")"
  if [ -f "${dest}" ] && [ "$(stat -c %s "${dest}")" = "${bytes}" ]; then
    echo "[fetch] cache hit: ${dest} complete (${bytes} bytes, mtime $(stat -c %y "${dest}" | cut -d. -f1))"
  else
    echo "[fetch] downloading ${url} -> ${dest} (${bytes} bytes expected) at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    curl -L --fail --retry "${CURL_RETRIES}" -C - -o "${dest}" "${url}"
  fi
  got="$(sha256sum "${dest}" | cut -d' ' -f1)"
  if [ "${got}" != "${sha_expected}" ]; then
    fail "sha256 mismatch for ${dest}: got ${got}, manifest says ${sha_expected}"
  fi
  echo "[fetch] verified ${relpath} sha256=${got:0:12}..."
  if [ "${EXTRACT}" = "1" ] && [[ "${dest}" == *.zip ]]; then
    echo "[fetch] extracting ${dest} into $(dirname "${dest}") (existing files skipped)"
    unzip -n -q "${dest}" -d "$(dirname "${dest}")"
  fi
}

total=0
for ds in $(echo "${DATASETS}" | tr ', ' '\n' | sed '/^$/d'); do
  rows="$(awk -F, -v d="${ds}" '$1 == d {print}' "${MANIFEST}")"
  if [ -z "${rows}" ]; then
    fail "no rows for dataset '${ds}' in ${MANIFEST}"
  fi
  echo "[fetch] dataset ${ds}: $(wc -l <<< "${rows}") file(s), $(awk -F, '{s+=$3} END {printf "%.2f GiB", s/1024/1024/1024}' <<< "${rows}")"
  while IFS=, read -r _ relpath bytes sha source _; do
    fetch_row "${relpath}" "${bytes}" "${sha}" "${source}"
    total=$((total + 1))
  done <<< "${rows}"
done
echo "[fetch] done: ${total} file(s) present and verified under ${DEST_ROOT}"
