# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "fastapi",
#     "uvicorn",
#     "httpx",
#     "python-multipart",
#     "soundfile",
#     "websockets",
#     "numpy",
#     "webrtcvad-wheels",
# ]
# ///
"""Whisper-compatible /v1/audio/transcriptions proxy for LFM2.5-Audio server."""

import argparse
import asyncio
import base64
import io
import json
import logging
import subprocess
import tempfile
import time

import struct
import uuid

import httpx
import numpy as np
import soundfile as sf
import uvicorn
import webrtcvad
from starlette.websockets import WebSocketState

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("lfm-proxy")
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI(title="LFM2.5 Transcription Proxy")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LFM_BASE_URL = "http://127.0.0.1:8090"


def ensure_wav(audio_bytes: bytes, filename: str) -> bytes:
    try:
        data, sr = sf.read(io.BytesIO(audio_bytes))
        log.info(f"Audio decoded natively: {len(data)} samples, {sr}Hz, duration={len(data)/sr:.2f}s, from={filename}")
        buf = io.BytesIO()
        sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return buf.read()
    except Exception:
        log.info(f"soundfile can't decode {filename}, falling back to ffmpeg")
        return convert_with_ffmpeg(audio_bytes, filename)


def convert_with_ffmpeg(audio_bytes: bytes, filename: str) -> bytes:
    ext = filename.rsplit(".", 1)[-1] if "." in filename else "webm"
    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as infile:
        infile.write(audio_bytes)
        infile.flush()
        outpath = infile.name + ".wav"
        try:
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", infile.name, "-ar", "16000", "-ac", "1", "-f", "wav", outpath],
                capture_output=True, timeout=30,
            )
            if result.returncode != 0:
                log.error(f"ffmpeg failed: {result.stderr.decode()[-500:]}")
                return audio_bytes
            with open(outpath, "rb") as f:
                wav_bytes = f.read()
            data, sr = sf.read(io.BytesIO(wav_bytes))
            log.info(f"Audio converted via ffmpeg: {len(data)} samples, {sr}Hz, duration={len(data)/sr:.2f}s")
            return wav_bytes
        except Exception as e:
            log.error(f"ffmpeg conversion error: {e}")
            return audio_bytes
        finally:
            import os
            os.unlink(infile.name)
            if os.path.exists(outpath):
                os.unlink(outpath)


def build_lfm_payload(wav_bytes: bytes) -> dict:
    encoded = base64.b64encode(wav_bytes).decode("utf-8")
    return {
        "model": "",
        "stream": True,
        "max_tokens": 2048,
        "reset_context": True,
        "messages": [
            {"role": "system", "content": "Perform ASR."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": encoded, "format": "wav"},
                    }
                ],
            },
        ],
    }


async def transcribe_via_lfm(wav_bytes: bytes, language: str | None = None) -> str:
    payload = build_lfm_payload(wav_bytes)
    log.info(f"Sending to LFM (buffered): audio_base64_len={len(payload['messages'][1]['content'][0]['input_audio']['data'])}")

    text_parts = []
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            f"{LFM_BASE_URL}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
        ) as resp:
            log.info(f"LFM response status: {resp.status_code}")
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                if not line.startswith("data: "):
                    log.warning(f"Unexpected SSE line: {line[:200]}")
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk["choices"][0].get("delta", {})
                    if content := delta.get("content"):
                        text_parts.append(content)
                except (json.JSONDecodeError, KeyError, IndexError) as e:
                    log.warning(f"Failed to parse chunk: {e}, raw: {data_str[:200]}")
                    continue

    result = "".join(text_parts).strip()
    log.info(f"Transcription result: '{result}'")
    return result


async def transcribe_via_lfm_stream(wav_bytes: bytes, language: str | None = None):
    payload = build_lfm_payload(wav_bytes)
    log.info(f"Sending to LFM (streaming): audio_base64_len={len(payload['messages'][1]['content'][0]['input_audio']['data'])}")

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            f"{LFM_BASE_URL}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
        ) as resp:
            log.info(f"LFM response status: {resp.status_code}")
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                if not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    yield "data: [DONE]\n\n"
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk["choices"][0].get("delta", {})
                    if content := delta.get("content"):
                        sse_chunk = {"text": content}
                        yield f"data: {json.dumps(sse_chunk)}\n\n"
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form(default=""),
    language: str | None = Form(default=None),
    response_format: str = Form(default="json"),
    prompt: str | None = Form(default=None),
    temperature: float = Form(default=0.0),
    stream: bool = Form(default=False),
):
    audio_bytes = await file.read()
    log.info(f"Received file: name={file.filename}, content_type={file.content_type}, size={len(audio_bytes)} bytes, stream={stream}")
    if len(audio_bytes) == 0:
        return {"text": ""}
    wav_bytes = ensure_wav(audio_bytes, file.filename or "audio.wav")
    log.info(f"WAV bytes: {len(wav_bytes)} bytes")

    if stream:
        return StreamingResponse(
            transcribe_via_lfm_stream(wav_bytes, language),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    text = await transcribe_via_lfm(wav_bytes, language)

    if response_format == "text":
        return JSONResponse(content=text, media_type="text/plain")
    elif response_format == "verbose_json":
        return {
            "task": "transcribe",
            "language": language or "en",
            "duration": 0.0,
            "text": text,
            "segments": [],
        }
    else:
        return {"text": text}


def pcm16_24k_to_wav_16k(pcm_bytes: bytes) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".pcm", delete=False) as infile:
        infile.write(pcm_bytes)
        infile.flush()
        outpath = infile.name + ".wav"
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-f", "s16le", "-ar", "24000", "-ac", "1",
                    "-i", infile.name,
                    "-ar", "16000", "-ac", "1",
                    "-f", "wav", outpath,
                ],
                capture_output=True, timeout=10,
            )
            if result.returncode != 0:
                log.error(f"pcm16_24k_to_wav_16k ffmpeg failed: {result.stderr.decode()[-300:]}")
                samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                buf = io.BytesIO()
                sf.write(buf, samples, 24000, format="WAV", subtype="PCM_16")
                buf.seek(0)
                return buf.read()
            with open(outpath, "rb") as f:
                return f.read()
        finally:
            import os
            os.unlink(infile.name)
            if os.path.exists(outpath):
                os.unlink(outpath)


def make_event_id() -> str:
    return f"evt_{uuid.uuid4().hex[:24]}"


def make_item_id() -> str:
    return f"item_{uuid.uuid4().hex[:24]}"


async def _transcribe_and_send(ws: WebSocket, pcm_bytes: bytes, item_id: str):
    wav_bytes = pcm16_24k_to_wav_16k(pcm_bytes)
    duration = len(pcm_bytes) / (24000 * 2)
    log.info(f"VAD transcribe: {len(pcm_bytes)} PCM bytes, {duration:.2f}s, WAV={len(wav_bytes)} bytes")

    payload = build_lfm_payload(wav_bytes)
    text_parts = []
    content_index = 0

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            f"{LFM_BASE_URL}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
        ) as resp:
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk["choices"][0].get("delta", {})
                    if content := delta.get("content"):
                        text_parts.append(content)
                        if ws.client_state == WebSocketState.CONNECTED:
                            await ws.send_json({
                                "event_id": make_event_id(),
                                "type": "conversation.item.input_audio_transcription.delta",
                                "item_id": item_id,
                                "content_index": content_index,
                                "delta": content,
                            })
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    transcript = "".join(text_parts).strip()
    log.info(f"VAD transcription: '{transcript}'")

    if ws.client_state == WebSocketState.CONNECTED:
        await ws.send_json({
            "event_id": make_event_id(),
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": item_id,
            "content_index": content_index,
            "transcript": transcript,
        })


SAMPLE_RATE = 24000
BYTES_PER_SAMPLE = 2
SAMPLES_PER_MS = SAMPLE_RATE // 1000



def _resample_24k_to_16k(samples_24k: np.ndarray) -> np.ndarray:
    n_out = int(len(samples_24k) * 16000 / 24000)
    indices = (np.arange(n_out) * 24000 / 16000).astype(np.int64)
    return samples_24k[np.clip(indices, 0, len(samples_24k) - 1)]


def _check_speech_webrtcvad(vad: webrtcvad.Vad, chunk_24k_int16: np.ndarray) -> bool:
    resampled = _resample_24k_to_16k(chunk_24k_int16)
    frame_size = 480
    speech_frames = 0
    total_frames = 0
    for i in range(0, len(resampled) - frame_size + 1, frame_size):
        frame = resampled[i:i + frame_size].tobytes()
        if vad.is_speech(frame, 16000):
            speech_frames += 1
        total_frames += 1
    if total_frames == 0:
        return False
    return speech_frames > total_frames * 0.3


@app.websocket("/v1/realtime")
async def realtime_ws(ws: WebSocket, intent: str = "transcription"):
    subprotocol = None
    requested_protocols = []
    for header_name, header_value in ws.headers.raw:
        if header_name == b"sec-websocket-protocol":
            requested_protocols = [p.strip() for p in header_value.decode().split(",")]
            break

    if "realtime" in requested_protocols:
        subprotocol = "realtime"
    elif requested_protocols:
        subprotocol = requested_protocols[0]

    log.info(f"WS connect: origin={ws.headers.get('origin')}, selected={subprotocol}")
    await ws.accept(subprotocol=subprotocol)

    session_id = f"sess_{uuid.uuid4().hex[:20]}"

    SILENCE_MS = 1200
    PREFIX_MS = 300
    MIN_SPEECH_MS = 500
    MAX_SPEECH_MS = 8000

    silence_threshold = int(SILENCE_MS * SAMPLES_PER_MS)
    prefix_bytes = int(PREFIX_MS * SAMPLES_PER_MS * BYTES_PER_SAMPLE)
    min_speech_bytes = int(MIN_SPEECH_MS * SAMPLES_PER_MS * BYTES_PER_SAMPLE)
    max_speech_bytes = int(MAX_SPEECH_MS * SAMPLES_PER_MS * BYTES_PER_SAMPLE)

    vad = webrtcvad.Vad(2)
    audio_buffer = bytearray()
    pre_speech_buffer = bytearray()
    speech_active = False
    silence_samples = 0
    speech_confirm = 0
    total_samples = 0
    item_id = ""

    await ws.send_json({
        "event_id": make_event_id(),
        "type": "session.created",
        "session": {
            "id": session_id,
            "object": "realtime.transcription_session",
            "type": "transcription",
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "transcription": {"model": "lfm2.5-audio-1.5b"},
                    "turn_detection": {"type": "server_vad", "threshold": 0.5, "silence_duration_ms": SILENCE_MS},
                }
            },
        },
    })

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type", "")

            if msg_type == "input_audio_buffer.append":
                audio_b64 = msg.get("audio", "")
                if not audio_b64:
                    continue
                chunk_bytes = base64.b64decode(audio_b64)
                chunk_samples = np.frombuffer(chunk_bytes, dtype=np.int16)
                n_samples = len(chunk_samples)
                total_samples += n_samples

                is_speech = _check_speech_webrtcvad(vad, chunk_samples)

                if not speech_active:
                    pre_speech_buffer.extend(chunk_bytes)
                    if len(pre_speech_buffer) > prefix_bytes:
                        pre_speech_buffer = pre_speech_buffer[-prefix_bytes:]

                    if is_speech:
                        speech_confirm += 1
                    else:
                        speech_confirm = 0

                    if speech_confirm >= 2:
                        speech_active = True
                        silence_samples = 0
                        speech_confirm = 0
                        item_id = make_item_id()
                        audio_buffer.clear()
                        audio_buffer.extend(pre_speech_buffer)
                        audio_buffer.extend(chunk_bytes)
                        ms = int(total_samples / SAMPLES_PER_MS)
                        log.info(f"speech_started at {ms}ms")
                        if ws.client_state == WebSocketState.CONNECTED:
                            await ws.send_json({
                                "event_id": make_event_id(),
                                "type": "input_audio_buffer.speech_started",
                                "audio_start_ms": ms,
                                "item_id": item_id,
                            })
                else:
                    audio_buffer.extend(chunk_bytes)

                    if not is_speech:
                        silence_samples += n_samples
                    else:
                        silence_samples = 0

                    do_commit = silence_samples >= silence_threshold or len(audio_buffer) >= max_speech_bytes
                    if do_commit:
                        ms = int(total_samples / SAMPLES_PER_MS)
                        reason = "max_len" if len(audio_buffer) >= max_speech_bytes else "silence"
                        speech_active = False
                        silence_samples = 0

                        log.info(f"speech_stopped at {ms}ms ({reason})")
                        if ws.client_state == WebSocketState.CONNECTED:
                            await ws.send_json({"event_id": make_event_id(), "type": "input_audio_buffer.speech_stopped", "audio_end_ms": ms, "item_id": item_id})
                            await ws.send_json({"event_id": make_event_id(), "type": "input_audio_buffer.committed", "item_id": item_id, "previous_item_id": None})

                        pcm = bytes(audio_buffer)
                        audio_buffer.clear()
                        pre_speech_buffer.clear()

                        if len(pcm) >= min_speech_bytes:
                            asyncio.create_task(_transcribe_and_send(ws, pcm, item_id))
                        else:
                            log.info(f"Skipped: too short")

            elif msg_type == "input_audio_buffer.clear":
                audio_buffer.clear()
                pre_speech_buffer.clear()
                speech_active = False
                silence_samples = 0
                speech_confirm = 0
                await ws.send_json({"event_id": make_event_id(), "type": "input_audio_buffer.cleared"})

            elif msg_type == "input_audio_buffer.commit":
                pcm = bytes(audio_buffer) if audio_buffer else bytes(pre_speech_buffer)
                cid = make_item_id()
                audio_buffer.clear()
                pre_speech_buffer.clear()
                speech_active = False
                silence_samples = 0
                speech_confirm = 0
                dur = len(pcm) / (SAMPLES_PER_MS * BYTES_PER_SAMPLE)

                await ws.send_json({"event_id": make_event_id(), "type": "input_audio_buffer.committed", "item_id": cid, "previous_item_id": None})
                if dur < 300:
                    await ws.send_json({"event_id": make_event_id(), "type": "conversation.item.input_audio_transcription.completed", "item_id": cid, "content_index": 0, "transcript": ""})
                else:
                    log.info(f"Manual commit: {dur:.0f}ms")
                    asyncio.create_task(_transcribe_and_send(ws, pcm, cid))

            elif msg_type in ("session.update", "transcription_session.update"):
                resp_type = "transcription_session.updated" if "transcription" in msg_type else "session.updated"
                await ws.send_json({"event_id": make_event_id(), "type": resp_type, "session": {"id": session_id, "type": "transcription"}})

            elif msg_type == "response.cancel":
                pass

    except WebSocketDisconnect:
        log.info(f"WS session {session_id} disconnected")
    except Exception as e:
        log.error(f"WS error: {e}")
        if ws.client_state == WebSocketState.CONNECTED:
            await ws.close(code=1011, reason=str(e))


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "lfm2.5-audio-1.5b",
                "object": "model",
                "owned_by": "LiquidAI",
            }
        ],
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--lfm-url", default="http://127.0.0.1:8090")
    args = parser.parse_args()
    LFM_BASE_URL = args.lfm_url
    uvicorn.run(app, host=args.host, port=args.port)
