#!/bin/bash
set -e

MODEL_DIR="${MODEL_DIR:-/models}"
LFM_PORT="${LFM_PORT:-8090}"
PROXY_PORT="${PROXY_PORT:-8091}"
QUANT="${QUANT:-Q8_0}"

if [ ! -f "$MODEL_DIR/LFM2.5-Audio-1.5B-${QUANT}.gguf" ]; then
    echo "Downloading LFM2.5-Audio-1.5B-GGUF (${QUANT})..."
    huggingface-cli download LiquidAI/LFM2.5-Audio-1.5B-GGUF \
        "LFM2.5-Audio-1.5B-${QUANT}.gguf" \
        "mmproj-LFM2.5-Audio-1.5B-${QUANT}.gguf" \
        "vocoder-LFM2.5-Audio-1.5B-${QUANT}.gguf" \
        "tokenizer-LFM2.5-Audio-1.5B-${QUANT}.gguf" \
        --local-dir "$MODEL_DIR"
    echo "Download complete."
fi

RUNNER_DIR="${RUNNER_DIR:-/runner}"

echo "Starting LFM server on port ${LFM_PORT}..."
LD_LIBRARY_PATH="${RUNNER_DIR}:${LD_LIBRARY_PATH}" \
    "${RUNNER_DIR}/llama-liquid-audio-server" \
    -m "${MODEL_DIR}/LFM2.5-Audio-1.5B-${QUANT}.gguf" \
    -mm "${MODEL_DIR}/mmproj-LFM2.5-Audio-1.5B-${QUANT}.gguf" \
    -mv "${MODEL_DIR}/vocoder-LFM2.5-Audio-1.5B-${QUANT}.gguf" \
    --tts-speaker-file "${MODEL_DIR}/tokenizer-LFM2.5-Audio-1.5B-${QUANT}.gguf" \
    --host 127.0.0.1 --port "${LFM_PORT}" &

LFM_PID=$!

echo "Waiting for LFM server..."
for i in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:${LFM_PORT}/" >/dev/null 2>&1 || \
       curl -sf "http://127.0.0.1:${LFM_PORT}/v1/chat/completions" >/dev/null 2>&1; then
        break
    fi
    # Check if process died
    if ! kill -0 $LFM_PID 2>/dev/null; then
        echo "LFM server process died"
        exit 1
    fi
    sleep 1
done
echo "LFM server ready."

echo "Starting transcription proxy on port ${PROXY_PORT}..."
exec uv run --frozen python transcription_proxy.py \
    --host 0.0.0.0 --port "${PROXY_PORT}" --lfm-url "http://127.0.0.1:${LFM_PORT}"
