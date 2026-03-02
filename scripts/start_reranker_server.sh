#!/usr/bin/env bash
# -------------------------------------------------------------------
# Start the Infinity reranker server.
#
# Serves BAAI/bge-reranker-v2-m3 on port 8082.
# The model is downloaded on first run and cached thereafter.
#
# Prerequisites:
#   pip install "infinity-emb[all]"
#
# Usage:
#   bash scripts/start_reranker_server.sh
#   bash scripts/start_reranker_server.sh --port 8082
#   RERANKER_MODEL=BAAI/bge-reranker-v2-m3 bash scripts/start_reranker_server.sh
# -------------------------------------------------------------------

set -euo pipefail

# Configurable via environment
MODEL="${RERANKER_MODEL:-BAAI/bge-reranker-v2-m3}"
PORT="${RERANKER_PORT:-8082}"
HOST="${RERANKER_HOST:-0.0.0.0}"

# Parse CLI args (override env)
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)  MODEL="$2";  shift 2 ;;
        --port)   PORT="$2";   shift 2 ;;
        --host)   HOST="$2";   shift 2 ;;
        *)        echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "============================================="
echo "  Infinity Reranker Server"
echo "============================================="
echo "  Model:  ${MODEL}"
echo "  Host:   ${HOST}"
echo "  Port:   ${PORT}"
echo "  URL:    http://${HOST}:${PORT}"
echo "============================================="
echo ""

# Check infinity is installed
if ! command -v infinity_emb &> /dev/null; then
    echo "ERROR: infinity_emb not found."
    echo ""
    echo "Install with:"
    echo '  pip install "infinity-emb[all]"'
    echo ""
    exit 1
fi

# Workaround: optimum >= 1.24 removed bettertransformer, which
# infinity_emb imports unconditionally at startup.  Uninstalling
# optimum avoids the crash; torch SDPA replaces it anyway.
if python3 -c "from optimum.bettertransformer import BetterTransformer" 2>/dev/null; then
    : # optimum is fine
else
    if python3 -c "import optimum" 2>/dev/null; then
        echo "⚠️  Detected incompatible 'optimum' version (missing bettertransformer)."
        echo "   Removing it — torch SDPA provides the same optimization natively."
        pip uninstall optimum -y --quiet 2>/dev/null || true
    fi
fi

# Launch
exec infinity_emb v2 \
    --model-id "${MODEL}" \
    --port "${PORT}" \
    --host "${HOST}" \
    --engine torch