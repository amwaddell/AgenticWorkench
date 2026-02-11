#!/bin/bash
# Start local llama-cpp-python server with Ministral model
#
# Usage: bash scripts/start_model_server.sh

set -e

# Configuration
MODEL_PATH="data/models/ministral-8b-instruct-2410-q4_k_m.gguf"
HOST="127.0.0.1"
PORT="8080"
CONTEXT_SIZE=8192
N_GPU_LAYERS=-1  # Use -1 for full GPU offload on Metal

echo "Starting llama-cpp-python server..."
echo "Model: $MODEL_PATH"
echo "Host: $HOST:$PORT"
echo "Context size: $CONTEXT_SIZE"
echo ""

# Check if model file exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ ERROR: Model file not found: $MODEL_PATH"
    echo ""
    echo "Please download a GGUF model file and place it in data/models/"
    echo "For example:"
    echo "  1. Download Ministral-8B GGUF from Hugging Face"
    echo "  2. Place it in: data/models/"
    echo "  3. Update MODEL_PATH in this script if needed"
    exit 1
fi

# Create data/models directory if it doesn't exist
mkdir -p "$(dirname "$MODEL_PATH")"

# Start the server
echo "Server starting on http://$HOST:$PORT"
echo "Press Ctrl+C to stop"
echo ""

python3 -m llama_cpp.server \
    --model "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --n_ctx "$CONTEXT_SIZE" \
    --n_gpu_layers "$N_GPU_LAYERS" \
    --verbose false

# Alternative minimal command:
# python3 -m llama_cpp.server --model "$MODEL_PATH"