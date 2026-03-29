# Voice Transcription Proxy

OpenAI-compatible transcription proxy for [LFM2.5-Audio-1.5B](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-GGUF) (Liquid AI).

## Endpoints

| Endpoint | Protocol | Description |
|---|---|---|
| `POST /v1/audio/transcriptions` | HTTP | Whisper-compatible file upload transcription |
| `POST /v1/audio/transcriptions` + `stream=true` | HTTP SSE | Streaming transcription |
| `ws:///v1/realtime?intent=transcription` | WebSocket | OpenAI Realtime API compatible with server VAD (webrtcvad) |
| `GET /health` | HTTP | Health check |

## Run locally

```bash
uv run transcription-proxy/transcription_proxy.py \
  --host 0.0.0.0 --port 8091 --lfm-url http://127.0.0.1:8090
```

## Docker

```bash
docker build -t voice-transcription-proxy transcription-proxy/
docker run -p 8091:8091 voice-transcription-proxy
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
