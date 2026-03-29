# Voice Transcription Proxy

OpenAI-compatible transcription service powered by [LFM2.5-Audio-1.5B](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B) (Liquid AI).

Uses the official `liquid-audio` Python package with PyTorch for native GPU inference — no llama.cpp needed.

## Endpoints

| Endpoint | Protocol | Description |
|---|---|---|
| `POST /v1/audio/transcriptions` | HTTP | Whisper-compatible file upload |
| `POST /v1/audio/transcriptions` + `stream=true` | HTTP SSE | Streaming transcription |
| `ws:///v1/realtime?intent=transcription` | WebSocket | OpenAI Realtime API with server VAD |
| `GET /health` | HTTP | Health + GPU status |

## Run locally

```bash
uv sync
uv run python transcription_proxy.py --port 8091
```

## Docker

```bash
docker build -t voice .
docker run --gpus all -p 8091:8091 voice

# With persistent model cache
docker run --gpus all -p 8091:8091 -v voice-cache:/cache voice
```

## Kubernetes

```yaml
containers:
  - name: voice
    image: ghcr.io/anthaathi/voice-transcription-proxy:latest
    ports:
      - containerPort: 8091
    resources:
      limits:
        nvidia.com/gpu: 1
    volumeMounts:
      - name: cache
        mountPath: /cache
```

## Architecture

```
┌─────────────────────────────────────────┐
│ Single process (FastAPI + liquid-audio)  │
│                                          │
│  Model loaded on GPU at startup          │
│  ├─ POST /v1/audio/transcriptions        │
│  ├─ WebSocket /v1/realtime (webrtcvad)   │
│  └─ ASR via LFM2AudioModel              │
└─────────────────────────────────────────┘
```

Model is auto-downloaded from HuggingFace on first start.
Mount `/cache` volume to persist across restarts.
