ARG CUDA_VERSION=12.8.1
ARG UBUNTU_VERSION=24.04

# Stage 1: Build llama.cpp with CUDA + liquid audio support
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake build-essential curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build
RUN git clone --depth 1 -b tarek/feat/os-lfm2.5-audio-1.5b-upstream \
    https://github.com/tdakhran/llama.cpp.git .

RUN cmake -B build \
    -DGGML_CUDA=ON \
    -DLLAMA_CURL=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="60;70;75;80;86;89;90" && \
    cmake --build build --config Release -j$(nproc) --target llama-liquid-audio-cli llama-liquid-audio-server

# Stage 2: Runtime
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg curl python3 python3-dev libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy built binaries + libs
COPY --from=builder /build/build/bin/llama-liquid-audio-cli /runner/
COPY --from=builder /build/build/bin/llama-liquid-audio-server /runner/
COPY --from=builder /build/build/src/libllama.so* /runner/
COPY --from=builder /build/build/ggml/src/libggml*.so* /runner/
COPY --from=builder /build/build/examples/liquid-audio/libliquid-audio.so /runner/
COPY --from=builder /build/build/common/libmtmd.so* /runner/
RUN chmod +x /runner/llama-liquid-audio-cli /runner/llama-liquid-audio-server

# Download models
ARG QUANT=Q8_0
RUN uv run python -c "\
from huggingface_hub import hf_hub_download; \
import os; \
repo = 'LiquidAI/LFM2.5-Audio-1.5B-GGUF'; \
q = '${QUANT}'; \
d = '/models'; \
os.makedirs(d, exist_ok=True); \
[hf_hub_download(repo, f, local_dir=d) for f in [ \
    f'LFM2.5-Audio-1.5B-{q}.gguf', \
    f'mmproj-LFM2.5-Audio-1.5B-{q}.gguf', \
    f'vocoder-LFM2.5-Audio-1.5B-{q}.gguf', \
    f'tokenizer-LFM2.5-Audio-1.5B-{q}.gguf', \
]]"

COPY transcription_proxy.py entrypoint.sh ./

ENV MODEL_DIR=/models
ENV RUNNER_DIR=/runner
ENV LFM_PORT=8090
ENV PROXY_PORT=8091
ENV QUANT=${QUANT}
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV LD_LIBRARY_PATH=/runner:${LD_LIBRARY_PATH}

EXPOSE 8091

CMD ["bash", "entrypoint.sh"]
