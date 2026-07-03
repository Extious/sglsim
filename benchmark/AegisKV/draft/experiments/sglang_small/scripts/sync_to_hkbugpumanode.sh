#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-hkbugpumanode}"
REMOTE_DIR="${REMOTE_DIR:-~/aegiskv_sglang_small}"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if command -v rsync >/dev/null 2>&1; then
  rsync -av --exclude '/results/' "${LOCAL_DIR}/" "${REMOTE}:${REMOTE_DIR}/"
else
  tar -C "${LOCAL_DIR}" --exclude './results' -czf - . \
    | ssh "${REMOTE}" "mkdir -p ${REMOTE_DIR}/results && tar -C ${REMOTE_DIR} -xzf -"
fi
echo "synced ${LOCAL_DIR} to ${REMOTE}:${REMOTE_DIR}"
