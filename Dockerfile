FROM python:3.13-slim AS base

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg curl unzip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
RUN uv run python -c "import huggingface_hub" 2>/dev/null || uv add huggingface-hub

COPY transcription_proxy.py entrypoint.sh ./

# Download runner
ARG RUNNER_URL=https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-GGUF/resolve/main/runners/llama-liquid-audio-ubuntu-x64.zip
RUN mkdir -p /runner && \
    curl -L -o /tmp/runner.zip "${RUNNER_URL}" && \
    unzip /tmp/runner.zip -d /tmp/runner-extract && \
    mv /tmp/runner-extract/llama-liquid-audio-ubuntu-x64/* /runner/ && \
    chmod +x /runner/llama-liquid-audio-cli /runner/llama-liquid-audio-server && \
    rm -rf /tmp/runner.zip /tmp/runner-extract

# Download models (Q8_0 by default, override with --build-arg QUANT=Q4_0)
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

ENV MODEL_DIR=/models
ENV RUNNER_DIR=/runner
ENV LFM_PORT=8090
ENV PROXY_PORT=8091
ENV QUANT=${QUANT}

EXPOSE 8091

CMD ["bash", "entrypoint.sh"]
