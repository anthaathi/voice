# Voice Transcription Proxy

Single-container OpenAI-compatible transcription service powered by [LFM2.5-Audio-1.5B](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-GGUF) (Liquid AI).

Bundles the LFM server + model files + transcription proxy in one image.

## Endpoints

| Endpoint | Protocol | Description |
|---|---|---|
| `POST /v1/audio/transcriptions` | HTTP | Whisper-compatible file upload transcription |
| `POST /v1/audio/transcriptions` + `stream=true` | HTTP SSE | Streaming transcription |
| `ws:///v1/realtime?intent=transcription` | WebSocket | OpenAI Realtime API compatible with server VAD (webrtcvad) |
| `GET /health` | HTTP | Health check |

## Run locally (no Docker)

```bash
# Start LFM server separately, then:
uv run python transcription_proxy.py --host 0.0.0.0 --port 8091 --lfm-url http://127.0.0.1:8090
```

## Docker

```bash
# Build (downloads ~1.5GB model during build)
docker build -t voice .

# Build with Q4_0 quantization (~850MB, faster)
docker build --build-arg QUANT=Q4_0 -t voice .

# Run
docker run -p 8091:8091 voice
```

## Kubernetes

```bash
kubectl apply -f k8s/
```

## Architecture

```
┌──────────────────────────────────────┐
│  Container                           │
│                                      │
│  entrypoint.sh                       │
│    ├─ llama-liquid-audio-server :8090│
│    │   (LFM2.5 model, CPU inference)│
│    │                                 │
│    └─ transcription_proxy.py  :8091  │
│        (FastAPI, WebSocket, VAD)     │
│        └─ proxies to LFM :8090      │
└──────────────────────────────────────┘
```

## WebSocket protocol (OpenAI Realtime compatible)

```
Client → input_audio_buffer.append  (base64 PCM16 24kHz mono)
Server → input_audio_buffer.speech_started  (webrtcvad detected speech)
Server → input_audio_buffer.speech_stopped  (1.2s silence)
Server → input_audio_buffer.committed
Server → conversation.item.input_audio_transcription.delta  (streaming tokens)
Server → conversation.item.input_audio_transcription.completed  (full transcript)
```
