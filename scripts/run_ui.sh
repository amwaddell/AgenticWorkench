#!/usr/bin/env bash
# Launch the Streamlit chat UI for the agentic workbench.
#
# Usage:
#   bash scripts/run_ui.sh              # default port 8501
#   bash scripts/run_ui.sh --port 8502  # custom port
#
# Prerequisites:
#   1. pip install -r requirements.txt
#   2. LanceDB indexes built (vector + keyword FTS)
#   3. Model server running: bash scripts/start_model_server.sh
#   4. (Optional) phoenix serve — to view traces

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Ensure we're in the project root so relative paths in config work
cd "$PROJECT_ROOT"

echo "=========================================="
echo "  Agentic Workbench — Streamlit Chat UI"
echo "=========================================="
echo ""
echo "Project root: $PROJECT_ROOT"
echo ""

# Check that streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "ERROR: streamlit not found. Install with:"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Check that the LanceDB index exists
if [ ! -d "$PROJECT_ROOT/data/indexes/active/lancedb" ]; then
    echo "WARNING: LanceDB index not found at data/indexes/active/lancedb"
    echo "  Build it with: python scripts/build_vector_index.py"
    echo ""
fi

# Launch Streamlit
echo "Starting Streamlit..."
echo "  UI:     http://localhost:8501"
echo "  Phoenix: http://localhost:6006 (if running)"
echo ""

exec streamlit run src/workbench/ui/streamlit_app.py \
    --server.headless true \
    --browser.gatherUsageStats false \
    "$@"
