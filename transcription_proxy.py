"""OpenAI-compatible transcription proxy powered by LFM2.5-Audio (liquid-audio)."""

import argparse
import asyncio
import base64
import io
import json
import logging
import os
import subprocess
import tempfile
import uuid

import numpy as np
import soundfile as sf
import torch
import torchaudio
import uvicorn
import webrtcvad
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState
from starlette.websockets import WebSocketState

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("lfm-proxy")

HF_REPO = os.environ.get("HF_REPO", "LiquidAI/LFM2.5-Audio-1.5B")

app = FastAPI(title="LFM2.5 Transcription Proxy")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

processor: LFM2AudioProcessor | None = None
model: LFM2AudioModel | None = None
model_lock = asyncio.Lock()


@app.on_event("startup")
async def load_model():
    global processor, model
    log.info(f"Loading model from {HF_REPO}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = LFM2AudioProcessor.from_pretrained(HF_REPO).eval()
    model = LFM2AudioModel.from_pretrained(HF_REPO).eval()
    if device == "cuda":
        model = model.cuda()
    log.info(f"Model loaded on {device}")


STOP_TOKENS = {"<|im_end|>", "<|endoftext|>", "<|end|>", "</s>"}


def _run_asr(wav_tensor: torch.Tensor, sample_rate: int) -> list[str]:
    chat = ChatState(processor)
    chat.new_turn("system")
    chat.add_text("Perform ASR.")
    chat.end_turn()
    chat.new_turn("user")
    chat.add_audio(wav_tensor, sample_rate)
    chat.end_turn()
    chat.new_turn("assistant")

    tokens = []
    for t in model.generate_sequential(**chat, max_new_tokens=2048):
        if t.numel() == 1:
            text = processor.text.decode(t)
            if text in STOP_TOKENS:
                break
            tokens.append(text)
    return tokens


def _ensure_wav_tensor(audio_bytes: bytes, filename: str) -> tuple[torch.Tensor, int]:
    try:
        data, sr = sf.read(io.BytesIO(audio_bytes))
        return torch.tensor(data, dtype=torch.float32).unsqueeze(0), sr
    except Exception:
        pass

    ext = filename.rsplit(".", 1)[-1] if "." in filename else "webm"
    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as infile:
        infile.write(audio_bytes)
        infile.flush()
        outpath = infile.name + ".wav"
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", infile.name, "-ar", "16000", "-ac", "1", "-f", "wav", outpath],
                capture_output=True, timeout=30,
            )
            wav, sr = torchaudio.load(outpath)
            return wav, sr
        finally:
            os.unlink(infile.name)
            if os.path.exists(outpath):
                os.unlink(outpath)


def _pcm16_24k_to_tensor(pcm_bytes: bytes) -> tuple[torch.Tensor, int]:
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return torch.tensor(samples, dtype=torch.float32).unsqueeze(0), 24000


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model_name: str = Form(default="", alias="model"),
    language: str | None = Form(default=None),
    response_format: str = Form(default="json"),
    prompt: str | None = Form(default=None),
    temperature: float = Form(default=0.0),
    stream: bool = Form(default=False),
):
    audio_bytes = await file.read()
    log.info(f"Received: name={file.filename}, size={len(audio_bytes)}, stream={stream}")
    if len(audio_bytes) == 0:
        return {"text": ""}

    wav_tensor, sr = _ensure_wav_tensor(audio_bytes, file.filename or "audio.wav")
    log.info(f"Audio: {wav_tensor.shape}, {sr}Hz")

    async with model_lock:
        tokens = await asyncio.to_thread(_run_asr, wav_tensor, sr)

    text = "".join(tokens).strip()
    for st in STOP_TOKENS:
        text = text.replace(st, "")
    text = text.strip()
    log.info(f"Transcription: '{text}'")

    if stream:
        async def sse_gen():
            for tok in tokens:
                yield f"data: {json.dumps({'text': tok})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(sse_gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    if response_format == "text":
        return JSONResponse(content=text, media_type="text/plain")
    elif response_format == "verbose_json":
        return {"task": "transcribe", "language": language or "en", "duration": 0.0, "text": text, "segments": []}
    else:
        return {"text": text}


# --- Streaming ASR for WebSocket ---

def _run_asr_streaming(wav_tensor: torch.Tensor, sample_rate: int):
    chat = ChatState(processor)
    chat.new_turn("system")
    chat.add_text("Perform ASR.")
    chat.end_turn()
    chat.new_turn("user")
    chat.add_audio(wav_tensor, sample_rate)
    chat.end_turn()
    chat.new_turn("assistant")

    for t in model.generate_sequential(**chat, max_new_tokens=2048):
        if t.numel() == 1:
            text = processor.text.decode(t)
            if text in STOP_TOKENS:
                break
            yield text


async def _transcribe_and_send(ws: WebSocket, pcm_bytes: bytes, item_id: str):
    wav_tensor, sr = _pcm16_24k_to_tensor(pcm_bytes)
    duration = len(pcm_bytes) / (24000 * 2)
    log.info(f"Transcribing: {duration:.2f}s")

    text_parts = []
    content_index = 0

    def _generate():
        return list(_run_asr_streaming(wav_tensor, sr))

    async with model_lock:
        tokens = await asyncio.to_thread(_generate)

    for tok in tokens:
        text_parts.append(tok)
        if ws.client_state == WebSocketState.CONNECTED:
            await ws.send_json({
                "event_id": f"evt_{uuid.uuid4().hex[:24]}",
                "type": "conversation.item.input_audio_transcription.delta",
                "item_id": item_id,
                "content_index": content_index,
                "delta": tok,
            })

    transcript = "".join(text_parts).strip()
    for st in STOP_TOKENS:
        transcript = transcript.replace(st, "")
    transcript = transcript.strip()
    log.info(f"Transcription: '{transcript}'")

    if ws.client_state == WebSocketState.CONNECTED:
        await ws.send_json({
            "event_id": f"evt_{uuid.uuid4().hex[:24]}",
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": item_id,
            "content_index": content_index,
            "transcript": transcript,
        })


# --- WebSocket VAD helpers ---

SAMPLE_RATE = 24000
BYTES_PER_SAMPLE = 2
SAMPLES_PER_MS = SAMPLE_RATE // 1000


def _resample_24k_to_16k(samples_24k: np.ndarray) -> np.ndarray:
    n_out = int(len(samples_24k) * 16000 / 24000)
    indices = (np.arange(n_out) * 24000 / 16000).astype(np.int64)
    return samples_24k[np.clip(indices, 0, len(samples_24k) - 1)]


def _check_speech(vad: webrtcvad.Vad, chunk_24k_int16: np.ndarray) -> bool:
    resampled = _resample_24k_to_16k(chunk_24k_int16)
    frame_size = 480
    speech_frames = 0
    total_frames = 0
    for i in range(0, len(resampled) - frame_size + 1, frame_size):
        if vad.is_speech(resampled[i:i + frame_size].tobytes(), 16000):
            speech_frames += 1
        total_frames += 1
    return total_frames > 0 and speech_frames > total_frames * 0.3


def _make_eid():
    return f"evt_{uuid.uuid4().hex[:24]}"


def _make_iid():
    return f"item_{uuid.uuid4().hex[:24]}"


@app.websocket("/v1/realtime")
async def realtime_ws(ws: WebSocket, intent: str = "transcription"):
    requested = []
    for name, val in ws.headers.raw:
        if name == b"sec-websocket-protocol":
            requested = [p.strip() for p in val.decode().split(",")]
            break
    subproto = "realtime" if "realtime" in requested else (requested[0] if requested else None)
    log.info(f"WS connect: origin={ws.headers.get('origin')}, subprotocol={subproto}")
    await ws.accept(subprotocol=subproto)

    sid = f"sess_{uuid.uuid4().hex[:20]}"
    SILENCE_MS, PREFIX_MS, MIN_SPEECH_MS, MAX_SPEECH_MS = 1200, 300, 500, 8000

    silence_thresh = int(SILENCE_MS * SAMPLES_PER_MS)
    prefix_bytes = int(PREFIX_MS * SAMPLES_PER_MS * BYTES_PER_SAMPLE)
    min_bytes = int(MIN_SPEECH_MS * SAMPLES_PER_MS * BYTES_PER_SAMPLE)
    max_bytes = int(MAX_SPEECH_MS * SAMPLES_PER_MS * BYTES_PER_SAMPLE)

    vad = webrtcvad.Vad(2)
    buf = bytearray()
    pre_buf = bytearray()
    active = False
    silence = 0
    confirm = 0
    total = 0
    iid = ""

    await ws.send_json({
        "event_id": _make_eid(), "type": "session.created",
        "session": {
            "id": sid, "object": "realtime.transcription_session", "type": "transcription",
            "audio": {"input": {"format": {"type": "audio/pcm", "rate": 24000}, "transcription": {"model": "lfm2.5-audio-1.5b"}, "turn_detection": {"type": "server_vad"}}},
        },
    })

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            mt = msg.get("type", "")

            if mt == "input_audio_buffer.append":
                b64 = msg.get("audio", "")
                if not b64:
                    continue
                chunk = base64.b64decode(b64)
                samples = np.frombuffer(chunk, dtype=np.int16)
                n = len(samples)
                total += n
                is_speech = _check_speech(vad, samples)

                if not active:
                    pre_buf.extend(chunk)
                    if len(pre_buf) > prefix_bytes:
                        pre_buf = pre_buf[-prefix_bytes:]
                    if is_speech:
                        confirm += 1
                    else:
                        confirm = 0
                    if confirm >= 2:
                        active = True
                        silence = 0
                        confirm = 0
                        iid = _make_iid()
                        buf.clear()
                        buf.extend(pre_buf)
                        buf.extend(chunk)
                        ms = int(total / SAMPLES_PER_MS)
                        log.info(f"speech_started at {ms}ms")
                        if ws.client_state == WebSocketState.CONNECTED:
                            await ws.send_json({"event_id": _make_eid(), "type": "input_audio_buffer.speech_started", "audio_start_ms": ms, "item_id": iid})
                else:
                    buf.extend(chunk)
                    silence = silence + n if not is_speech else 0
                    if silence >= silence_thresh or len(buf) >= max_bytes:
                        ms = int(total / SAMPLES_PER_MS)
                        active = False
                        silence = 0
                        log.info(f"speech_stopped at {ms}ms")
                        if ws.client_state == WebSocketState.CONNECTED:
                            await ws.send_json({"event_id": _make_eid(), "type": "input_audio_buffer.speech_stopped", "audio_end_ms": ms, "item_id": iid})
                            await ws.send_json({"event_id": _make_eid(), "type": "input_audio_buffer.committed", "item_id": iid, "previous_item_id": None})
                        pcm = bytes(buf)
                        buf.clear()
                        pre_buf.clear()
                        if len(pcm) >= min_bytes:
                            asyncio.create_task(_transcribe_and_send(ws, pcm, iid))

            elif mt == "input_audio_buffer.clear":
                buf.clear()
                pre_buf.clear()
                active = False
                silence = 0
                confirm = 0
                await ws.send_json({"event_id": _make_eid(), "type": "input_audio_buffer.cleared"})

            elif mt == "input_audio_buffer.commit":
                pcm = bytes(buf) if buf else bytes(pre_buf)
                cid = _make_iid()
                buf.clear()
                pre_buf.clear()
                active = False
                silence = 0
                confirm = 0
                dur = len(pcm) / (SAMPLES_PER_MS * BYTES_PER_SAMPLE)
                await ws.send_json({"event_id": _make_eid(), "type": "input_audio_buffer.committed", "item_id": cid, "previous_item_id": None})
                if dur < 300:
                    await ws.send_json({"event_id": _make_eid(), "type": "conversation.item.input_audio_transcription.completed", "item_id": cid, "content_index": 0, "transcript": ""})
                else:
                    log.info(f"Manual commit: {dur:.0f}ms")
                    asyncio.create_task(_transcribe_and_send(ws, pcm, cid))

            elif mt in ("session.update", "transcription_session.update"):
                rt = "transcription_session.updated" if "transcription" in mt else "session.updated"
                await ws.send_json({"event_id": _make_eid(), "type": rt, "session": {"id": sid, "type": "transcription"}})

            elif mt == "response.cancel":
                pass

    except WebSocketDisconnect:
        log.info(f"WS session {sid} disconnected")
    except Exception as e:
        log.error(f"WS error: {e}")
        if ws.client_state == WebSocketState.CONNECTED:
            await ws.close(code=1011, reason=str(e))


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": "lfm2.5-audio-1.5b", "object": "model", "owned_by": "LiquidAI"}]}


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None, "device": "cuda" if torch.cuda.is_available() else "cpu"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8091)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
