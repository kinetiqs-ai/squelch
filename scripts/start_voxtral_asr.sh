#!/bin/bash
# Start a vLLM Realtime server for Voxtral Mini 4B ASR.

set -euo pipefail

VOXTRAL_MODEL="${VOXTRAL_MODEL:-mistralai/Voxtral-Mini-4B-Realtime-2602}"
VOXTRAL_HOST="${VOXTRAL_HOST:-0.0.0.0}"
VOXTRAL_PORT="${VOXTRAL_PORT:-8082}"
VOXTRAL_GPU_MEMORY_UTILIZATION="${VOXTRAL_GPU_MEMORY_UTILIZATION:-0.35}"
VOXTRAL_MAX_MODEL_LEN="${VOXTRAL_MAX_MODEL_LEN:-32768}"
VOXTRAL_COMPILATION_CONFIG="${VOXTRAL_COMPILATION_CONFIG:-{\"cudagraph_mode\":\"PIECEWISE\"}}"

echo "============================================"
echo "Starting Voxtral Realtime ASR"
echo "============================================"
echo "  Model: $VOXTRAL_MODEL"
echo "  Listen: $VOXTRAL_HOST:$VOXTRAL_PORT"
echo "  GPU memory utilization: $VOXTRAL_GPU_MEMORY_UTILIZATION"
echo "============================================"

exec vllm serve "$VOXTRAL_MODEL" \
    --host "$VOXTRAL_HOST" \
    --port "$VOXTRAL_PORT" \
    --tokenizer-mode mistral \
    --gpu-memory-utilization "$VOXTRAL_GPU_MEMORY_UTILIZATION" \
    --max-model-len "$VOXTRAL_MAX_MODEL_LEN" \
    --compilation-config "$VOXTRAL_COMPILATION_CONFIG"
