#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-9999}"
ROUNDS="${ROUNDS:-5}"
M="${M:-10}"
DEADLINE="${DEADLINE:-25}"

source .venv/bin/activate
python orchestrator/orchestrator.py --mode rpc --host "${HOST}" --port "${PORT}" --rounds "${ROUNDS}" --m "${M}" --deadline "${DEADLINE}"
