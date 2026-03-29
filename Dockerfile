ARG CUDA_VERSION=12.8.1
ARG UBUNTU_VERSION=24.04

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg python3 python3-dev build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Layer 1: deps (cached unless pyproject.toml or uv.lock changes)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Layer 2: app code (changes often, but tiny)
COPY transcription_proxy.py .

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV HF_HOME=/cache/huggingface

EXPOSE 8091

CMD ["uv", "run", "--frozen", "python3", "transcription_proxy.py", "--host", "0.0.0.0", "--port", "8091"]
