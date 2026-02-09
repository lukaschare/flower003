#!/usr/bin/env bash
set -euo pipefail

ROUNDS="${ROUNDS:-5}"
M="${M:-10}"
DEADLINE="${DEADLINE:-25}"

source .venv/bin/activate
python orchestrator/orchestrator.py --mode mock --rounds "${ROUNDS}" --m "${M}" --deadline "${DEADLINE}"
