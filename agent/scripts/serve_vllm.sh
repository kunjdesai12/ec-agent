#!/usr/bin/env bash
# Launch vLLM serving Qwen2.5-7B-Instruct GPTQ-Int4 on G6.xlarge (L4 24GB).
#
# Hermes tool-call parser is required so vLLM extracts <tool_call> JSON
# from Qwen's raw output and returns structured tool_calls in the
# OpenAI-compatible response (which is what LangGraph expects).
#
# Run this on the GPU host. The FastAPI service points at it via
# VLLM_BASE_URL (default: http://<gpu-host>:8001/v1).

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4}"
PORT="${PORT:-8001}"
MAX_LEN="${MAX_LEN:-16384}"
GPU_UTIL="${GPU_UTIL:-0.90}"
SERVED_NAME="${SERVED_NAME:-$MODEL}"

# Install once on a fresh box:
#   pip install -U vllm

echo "Launching vLLM"
echo "  model:           $MODEL"
echo "  port:            $PORT"
echo "  max-model-len:   $MAX_LEN"
echo "  gpu-mem-util:    $GPU_UTIL"

exec vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --served-model-name "$SERVED_NAME" \
  --quantization gptq \
  --dtype half \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --enable-prefix-caching \
  --disable-log-requests
