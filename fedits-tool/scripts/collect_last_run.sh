#!/usr/bin/env bash
set -euo pipefail

LAST_DIR=$(ls -1dt outputs/runs/* 2>/dev/null | head -n 1 || true)
if [ -z "${LAST_DIR}" ]; then
  echo "No runs found under outputs/runs/"
  exit 1
fi

echo "Last run: ${LAST_DIR}"
echo "Files:"
ls -lh "${LAST_DIR}"
