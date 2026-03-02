#!/usr/bin/env bash
# -------------------------------------------------------------------
# Start the full model stack: LLM + Embedding + Reranker servers.
#
# Launches all three in the background, waits for each to become
# healthy, then keeps running until Ctrl+C.
#
# Prerequisites:
#   - llama-server (or llama-cpp-python server) for LLM
#   - pip install "infinity-emb[all]" for embedding + reranking
#
# Usage:
#   bash scripts/start_model_stack.sh           # start all
#   bash scripts/start_model_stack.sh --no-llm  # skip LLM (already running)
# -------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ports (match defaults.yaml)
LLM_PORT="${LLM_PORT:-8080}"
EMBEDDING_PORT="${EMBEDDING_PORT:-8081}"
RERANKER_PORT="${RERANKER_PORT:-8082}"

SKIP_LLM=false
PIDS=()

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-llm)  SKIP_LLM=true; shift ;;
        *)         echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

cleanup() {
    echo ""
    echo "Shutting down model stack..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            echo "  Stopped PID $pid"
        fi
    done
    echo "Done."
    exit 0
}

trap cleanup SIGINT SIGTERM

wait_for_health() {
    local name="$1"
    local url="$2"
    local max_wait="${3:-120}"
    local elapsed=0

    printf "  Waiting for %s at %s " "$name" "$url"
    while [ $elapsed -lt $max_wait ]; do
        if curl -sf "$url" > /dev/null 2>&1; then
            echo "✅ (${elapsed}s)"
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
        printf "."
    done
    echo " TIMEOUT after ${max_wait}s ❌"
    return 1
}

# -------------------------------------------------------------------
# Launch servers
# -------------------------------------------------------------------

echo "============================================="
echo "  Agentic Workbench — Model Stack"
echo "============================================="
echo ""

# 1. LLM server
if [ "$SKIP_LLM" = false ]; then
    echo "[1/3] Starting LLM server on :${LLM_PORT}..."
    bash "${SCRIPT_DIR}/start_model_server.sh" &
    PIDS+=($!)
else
    echo "[1/3] LLM server — SKIPPED (--no-llm)"
fi

# 2. Embedding server
echo "[2/3] Starting embedding server on :${EMBEDDING_PORT}..."
EMBEDDING_PORT="${EMBEDDING_PORT}" bash "${SCRIPT_DIR}/start_embedding_server.sh" &
PIDS+=($!)

# 3. Reranker server
echo "[3/3] Starting reranker server on :${RERANKER_PORT}..."
RERANKER_PORT="${RERANKER_PORT}" bash "${SCRIPT_DIR}/start_reranker_server.sh" &
PIDS+=($!)

echo ""
echo "All servers launched. Waiting for health..."
echo ""

# -------------------------------------------------------------------
# Health checks
# -------------------------------------------------------------------

ALL_HEALTHY=true

if [ "$SKIP_LLM" = false ]; then
    if ! wait_for_health "LLM" "http://127.0.0.1:${LLM_PORT}/health" 120; then
        ALL_HEALTHY=false
    fi
fi

if ! wait_for_health "Embedding" "http://127.0.0.1:${EMBEDDING_PORT}/health" 120; then
    ALL_HEALTHY=false
fi

if ! wait_for_health "Reranker" "http://127.0.0.1:${RERANKER_PORT}/health" 120; then
    ALL_HEALTHY=false
fi

echo ""
echo "============================================="
if [ "$ALL_HEALTHY" = true ]; then
    echo "  All services healthy ✅"
    echo ""
    echo "  LLM:       http://127.0.0.1:${LLM_PORT}"
    echo "  Embedding:  http://127.0.0.1:${EMBEDDING_PORT}"
    echo "  Reranker:   http://127.0.0.1:${RERANKER_PORT}"
else
    echo "  Some services failed to start ⚠️"
    echo "  Check output above for details."
fi
echo "============================================="
echo ""
echo "Press Ctrl+C to stop all servers."
echo ""

# Keep alive
wait
