#!/usr/bin/env bash
set -euo pipefail
exec python -m brief_eval_rig.runner.orchestrator \
    corpus/sample/sample-001.json \
    --lineup local \
    --ollama-url "${OLLAMA_BASE_URL:-http://localhost:11435}"
