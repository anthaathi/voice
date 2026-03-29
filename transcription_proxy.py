"""OpenAI-compatible transcription + conversation proxy powered by LFM2.5-Audio.

Endpoints:
  POST /v1/audio/transcriptions  — Whisper-compatible ASR
  WS   /v1/realtime?intent=transcription — streaming ASR with server VAD
  WS   /v1/realtime?intent=conversation  — speech-to-speech (interleaved gen)
  GET  /                                 — test UI
"""

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
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from liquid_audio import LFM2AudioModel, LFM2AudioProcessor, ChatState, LFMModality
from starlette.websockets import WebSocketState

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("lfm-proxy")

HF_REPO = os.environ.get("HF_REPO", "LiquidAI/LFM2.5-Audio-1.5B")

app = FastAPI(title="LFM2.5 Audio Proxy")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

processor: LFM2AudioProcessor | None = None
model: LFM2AudioModel | None = None
model_lock = asyncio.Lock()

STOP_TOKENS = {"<|im_end|>", "<|endoftext|>", "<|end|>", "</s>"}
SAMPLE_RATE = 24000
BYTES_PER_SAMPLE = 2
SAMPLES_PER_MS = SAMPLE_RATE // 1000
VAD_FRAME_MS = 30
VAD_FRAME_SAMPLES_16K = int(16000 * VAD_FRAME_MS / 1000)


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_eid():
    return f"evt_{uuid.uuid4().hex[:24]}"

def _make_iid():
    return f"item_{uuid.uuid4().hex[:24]}"

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

def _clean_transcript(text: str) -> str:
    for st in STOP_TOKENS:
        text = text.replace(st, "")
    return text.strip()


# ---------------------------------------------------------------------------
# ASR generation (sequential)
# ---------------------------------------------------------------------------

def _run_asr(wav_tensor: torch.Tensor, sample_rate: int) -> list[str]:
    chat = ChatState(processor)
    chat.new_turn("system"); chat.add_text("Perform ASR."); chat.end_turn()
    chat.new_turn("user"); chat.add_audio(wav_tensor, sample_rate); chat.end_turn()
    chat.new_turn("assistant")
    tokens = []
    for t in model.generate_sequential(**chat, max_new_tokens=2048):
        if t.numel() == 1:
            text = processor.text.decode(t)
            if text in STOP_TOKENS:
                break
            tokens.append(text)
    return tokens

def _run_asr_streaming(wav_tensor: torch.Tensor, sample_rate: int):
    chat = ChatState(processor)
    chat.new_turn("system"); chat.add_text("Perform ASR."); chat.end_turn()
    chat.new_turn("user"); chat.add_audio(wav_tensor, sample_rate); chat.end_turn()
    chat.new_turn("assistant")
    for t in model.generate_sequential(**chat, max_new_tokens=2048):
        if t.numel() == 1:
            text = processor.text.decode(t)
            if text in STOP_TOKENS:
                break
            yield text


# ---------------------------------------------------------------------------
# Interleaved generation (speech-to-speech)
# ---------------------------------------------------------------------------

def _run_interleaved(
    wav_tensor: torch.Tensor,
    sample_rate: int,
    system_prompt: str = "Respond with interleaved text and audio.",
    max_new_tokens: int = 512,
    audio_temperature: float = 1.0,
    audio_top_k: int = 4,
) -> tuple[list[str], list[torch.Tensor]]:
    chat = ChatState(processor)
    chat.new_turn("system"); chat.add_text(system_prompt); chat.end_turn()
    chat.new_turn("user"); chat.add_audio(wav_tensor, sample_rate); chat.end_turn()
    chat.new_turn("assistant")

    text_tokens: list[str] = []
    audio_tokens: list[torch.Tensor] = []

    for t in model.generate_interleaved(
        **chat,
        max_new_tokens=max_new_tokens,
        audio_temperature=audio_temperature,
        audio_top_k=audio_top_k,
    ):
        if t.numel() == 1:
            text = processor.text.decode(t)
            if text in STOP_TOKENS:
                break
            text_tokens.append(text)
        else:
            audio_tokens.append(t)

    return text_tokens, audio_tokens


def _decode_audio_tokens(tokens: list[torch.Tensor]) -> bytes:
    if len(tokens) < 2:
        return b""
    audio_codes = torch.stack(tokens[:-1], 1).unsqueeze(0)
    waveform = processor.decode(audio_codes)
    pcm_f32 = waveform.cpu().squeeze().numpy()
    pcm_i16 = (np.clip(pcm_f32, -1.0, 1.0) * 32767).astype(np.int16)
    return pcm_i16.tobytes()


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------

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

    text = _clean_transcript("".join(tokens))
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


# ---------------------------------------------------------------------------
# WebSocket: transcription worker
# ---------------------------------------------------------------------------

async def _transcribe_and_send(ws: WebSocket, pcm_bytes: bytes, item_id: str):
    wav_tensor, sr = _pcm16_24k_to_tensor(pcm_bytes)
    duration = len(pcm_bytes) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
    log.info(f"Transcribing: {duration:.2f}s")

    def _generate():
        return list(_run_asr_streaming(wav_tensor, sr))

    async with model_lock:
        tokens = await asyncio.to_thread(_generate)

    for tok in tokens:
        if ws.client_state == WebSocketState.CONNECTED:
            await ws.send_json({
                "event_id": _make_eid(),
                "type": "conversation.item.input_audio_transcription.delta",
                "item_id": item_id, "content_index": 0, "delta": tok,
            })

    transcript = _clean_transcript("".join(tokens))
    log.info(f"Transcription: '{transcript}'")

    if ws.client_state == WebSocketState.CONNECTED:
        await ws.send_json({
            "event_id": _make_eid(),
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": item_id, "content_index": 0, "transcript": transcript,
        })


async def _transcription_worker(ws: WebSocket, queue: asyncio.Queue):
    while True:
        job = await queue.get()
        if job is None:
            break
        try:
            await _transcribe_and_send(ws, job["pcm"], job["item_id"])
        except Exception as e:
            log.error(f"Transcription error: {e}")
        finally:
            queue.task_done()


# ---------------------------------------------------------------------------
# WebSocket: conversation worker (interleaved speech-to-speech)
# ---------------------------------------------------------------------------

async def _generate_response_and_send(ws: WebSocket, pcm_bytes: bytes, item_id: str, system_prompt: str):
    wav_tensor, sr = _pcm16_24k_to_tensor(pcm_bytes)
    duration = len(pcm_bytes) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
    log.info(f"Generating response for {duration:.2f}s audio")

    response_id = f"resp_{uuid.uuid4().hex[:20]}"
    output_item_id = _make_iid()

    if ws.client_state == WebSocketState.CONNECTED:
        await ws.send_json({
            "event_id": _make_eid(), "type": "response.created",
            "response": {"id": response_id, "status": "in_progress"},
        })

    async with model_lock:
        text_tokens, audio_tokens = await asyncio.to_thread(
            _run_interleaved, wav_tensor, sr, system_prompt
        )

    for tok in text_tokens:
        if ws.client_state == WebSocketState.CONNECTED:
            await ws.send_json({
                "event_id": _make_eid(), "type": "response.audio_transcript.delta",
                "response_id": response_id, "item_id": output_item_id, "delta": tok,
            })

    audio_pcm = b""
    if audio_tokens:
        audio_pcm = await asyncio.to_thread(_decode_audio_tokens, audio_tokens)

    CHUNK = 48000
    for i in range(0, len(audio_pcm), CHUNK):
        chunk = audio_pcm[i:i + CHUNK]
        if ws.client_state == WebSocketState.CONNECTED:
            await ws.send_json({
                "event_id": _make_eid(), "type": "response.audio.delta",
                "response_id": response_id, "item_id": output_item_id,
                "delta": base64.b64encode(chunk).decode(),
            })

    transcript = _clean_transcript("".join(text_tokens))
    log.info(f"Response: '{transcript}', audio: {len(audio_tokens)} tokens, {len(audio_pcm)} bytes")

    if ws.client_state == WebSocketState.CONNECTED:
        await ws.send_json({"event_id": _make_eid(), "type": "response.audio_transcript.done", "response_id": response_id, "item_id": output_item_id, "transcript": transcript})
        await ws.send_json({"event_id": _make_eid(), "type": "response.audio.done", "response_id": response_id, "item_id": output_item_id})
        await ws.send_json({"event_id": _make_eid(), "type": "response.done", "response_id": response_id, "status": "completed"})


async def _conversation_worker(ws: WebSocket, queue: asyncio.Queue, system_prompt: str):
    while True:
        job = await queue.get()
        if job is None:
            break
        try:
            await _generate_response_and_send(ws, job["pcm"], job["item_id"], system_prompt)
        except Exception as e:
            log.error(f"Conversation error: {e}")
        finally:
            queue.task_done()


# ---------------------------------------------------------------------------
# Audio pipeline: VAD, state machine, endpoint detection
# ---------------------------------------------------------------------------

def _resample_24k_to_16k(samples: np.ndarray) -> np.ndarray:
    n_out = int(len(samples) * 16000 / 24000)
    indices = (np.arange(n_out) * 24000 / 16000).astype(np.int64)
    return samples[np.clip(indices, 0, len(samples) - 1)]

def _vad_frames(vad: webrtcvad.Vad, chunk_24k: np.ndarray) -> list[bool]:
    resampled = _resample_24k_to_16k(chunk_24k)
    results = []
    fs = VAD_FRAME_SAMPLES_16K
    for i in range(0, len(resampled) - fs + 1, fs):
        frame = resampled[i:i + fs].tobytes()
        results.append(vad.is_speech(frame, 16000))
    return results


class AudioPipelineConfig:
    frame_ms: int = 30
    speech_start_frames: int = 3
    speech_stop_silence_ms: int = 800
    min_utterance_ms: int = 300
    max_utterance_ms: int = 12000
    preroll_ms: int = 200
    ring_buffer_size: int = 8
    ring_buffer_threshold: int = 5
    vad_aggressiveness: int = 2

    def __init__(self, **kw):
        for k, v in kw.items():
            if hasattr(self, k):
                setattr(self, k, v)


class _State:
    IDLE = "idle"
    SPEAKING = "speaking"
    SILENCE_PENDING = "silence_pending"


class AudioPipeline:
    def __init__(self, cfg: AudioPipelineConfig | None = None):
        self.cfg = cfg or AudioPipelineConfig()
        self.vad = webrtcvad.Vad(self.cfg.vad_aggressiveness)
        self.state = _State.IDLE
        self.utterance_buf = bytearray()
        self.preroll_buf = bytearray()
        self.preroll_max = int(self.cfg.preroll_ms * SAMPLES_PER_MS * BYTES_PER_SAMPLE)
        self.silence_samples = 0
        self.silence_threshold = int(self.cfg.speech_stop_silence_ms * SAMPLES_PER_MS)
        self.min_bytes = int(self.cfg.min_utterance_ms * SAMPLES_PER_MS * BYTES_PER_SAMPLE)
        self.max_bytes = int(self.cfg.max_utterance_ms * SAMPLES_PER_MS * BYTES_PER_SAMPLE)
        self.ring_buffer = [False] * self.cfg.ring_buffer_size
        self.ring_idx = 0
        self.speech_start_count = 0
        self.total_samples = 0
        self.item_id = ""

    def _ring_vote(self) -> bool:
        return sum(self.ring_buffer) >= self.cfg.ring_buffer_threshold

    def _push_ring(self, is_speech: bool):
        self.ring_buffer[self.ring_idx % self.cfg.ring_buffer_size] = is_speech
        self.ring_idx += 1

    def feed(self, chunk_bytes: bytes) -> list[dict]:
        events = []
        chunk_samples = np.frombuffer(chunk_bytes, dtype=np.int16)
        n = len(chunk_samples)
        self.total_samples += n

        frame_decisions = _vad_frames(self.vad, chunk_samples)
        chunk_speech = sum(frame_decisions) > len(frame_decisions) * 0.3 if frame_decisions else False
        self._push_ring(chunk_speech)
        smoothed_speech = self._ring_vote()

        if self.state == _State.IDLE:
            self.preroll_buf.extend(chunk_bytes)
            if len(self.preroll_buf) > self.preroll_max:
                self.preroll_buf = self.preroll_buf[-self.preroll_max:]
            if smoothed_speech:
                self.speech_start_count += 1
            else:
                self.speech_start_count = 0
            if self.speech_start_count >= self.cfg.speech_start_frames:
                self.state = _State.SPEAKING
                self.silence_samples = 0
                self.speech_start_count = 0
                self.item_id = _make_iid()
                self.utterance_buf.clear()
                self.utterance_buf.extend(self.preroll_buf)
                self.utterance_buf.extend(chunk_bytes)
                ms = int(self.total_samples / SAMPLES_PER_MS)
                events.append({"type": "speech_started", "audio_start_ms": ms, "item_id": self.item_id})

        elif self.state == _State.SPEAKING:
            self.utterance_buf.extend(chunk_bytes)
            if not smoothed_speech:
                self.silence_samples += n
                if self.silence_samples >= self.silence_threshold:
                    self.state = _State.SILENCE_PENDING
            else:
                self.silence_samples = 0
            if len(self.utterance_buf) >= self.max_bytes:
                events.extend(self._finalize("max_length"))
            if self.state == _State.SILENCE_PENDING:
                events.extend(self._finalize("silence"))

        return events

    def _finalize(self, reason: str) -> list[dict]:
        events = []
        ms = int(self.total_samples / SAMPLES_PER_MS)
        pcm = bytes(self.utterance_buf)
        self.utterance_buf.clear()
        self.preroll_buf.clear()
        self.state = _State.IDLE
        self.silence_samples = 0
        self.speech_start_count = 0
        self.ring_buffer = [False] * self.cfg.ring_buffer_size

        events.append({"type": "speech_stopped", "audio_end_ms": ms, "item_id": self.item_id})
        if len(pcm) >= self.min_bytes:
            events.append({"type": "utterance_ready", "pcm": pcm, "item_id": self.item_id, "reason": reason})
        else:
            log.info(f"Utterance too short ({len(pcm) / (SAMPLES_PER_MS * BYTES_PER_SAMPLE):.0f}ms), discarded")
        return events

    def flush(self) -> list[dict]:
        if self.state in (_State.SPEAKING, _State.SILENCE_PENDING) and self.utterance_buf:
            return self._finalize("flush")
        pcm = bytes(self.preroll_buf) if self.preroll_buf else b""
        self.preroll_buf.clear()
        self.state = _State.IDLE
        if len(pcm) >= self.min_bytes:
            iid = self.item_id or _make_iid()
            return [{"type": "utterance_ready", "pcm": pcm, "item_id": iid, "reason": "manual_commit"}]
        return []


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/v1/realtime")
async def realtime_ws(ws: WebSocket, intent: str = "transcription"):
    requested = []
    for name, val in ws.headers.raw:
        if name == b"sec-websocket-protocol":
            requested = [p.strip() for p in val.decode().split(",")]
            break
    subproto = "realtime" if "realtime" in requested else (requested[0] if requested else None)
    log.info(f"WS connect: intent={intent}, origin={ws.headers.get('origin')}, subprotocol={subproto}")
    await ws.accept(subprotocol=subproto)

    sid = f"sess_{uuid.uuid4().hex[:20]}"
    cfg = AudioPipelineConfig()
    pipeline = AudioPipeline(cfg)
    tx_queue: asyncio.Queue = asyncio.Queue(maxsize=10)
    system_prompt = "Respond with interleaved text and audio."

    is_conversation = intent == "conversation"

    if is_conversation:
        worker_task = asyncio.create_task(_conversation_worker(ws, tx_queue, system_prompt))
    else:
        worker_task = asyncio.create_task(_transcription_worker(ws, tx_queue))

    session_type = "conversation" if is_conversation else "transcription"
    await ws.send_json({
        "event_id": _make_eid(), "type": "session.created",
        "session": {
            "id": sid, "object": "realtime.session", "type": session_type,
            "input_audio_format": "pcm16",
            "turn_detection": {
                "type": "server_vad",
                "silence_duration_ms": cfg.speech_stop_silence_ms,
                "prefix_padding_ms": cfg.preroll_ms,
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

            mt = msg.get("type", "")

            if mt == "input_audio_buffer.append":
                b64 = msg.get("audio", "")
                if not b64:
                    continue
                chunk = base64.b64decode(b64)
                events = pipeline.feed(chunk)
                for evt in events:
                    if evt["type"] == "speech_started":
                        log.info(f"speech_started at {evt['audio_start_ms']}ms")
                        if ws.client_state == WebSocketState.CONNECTED:
                            await ws.send_json({"event_id": _make_eid(), "type": "input_audio_buffer.speech_started", "audio_start_ms": evt["audio_start_ms"], "item_id": evt["item_id"]})
                    elif evt["type"] == "speech_stopped":
                        log.info(f"speech_stopped at {evt['audio_end_ms']}ms")
                        if ws.client_state == WebSocketState.CONNECTED:
                            await ws.send_json({"event_id": _make_eid(), "type": "input_audio_buffer.speech_stopped", "audio_end_ms": evt["audio_end_ms"], "item_id": evt["item_id"]})
                    elif evt["type"] == "utterance_ready":
                        dur = len(evt["pcm"]) / (SAMPLES_PER_MS * BYTES_PER_SAMPLE)
                        log.info(f"utterance_ready: {dur:.0f}ms ({evt['reason']})")
                        if ws.client_state == WebSocketState.CONNECTED:
                            await ws.send_json({"event_id": _make_eid(), "type": "input_audio_buffer.committed", "item_id": evt["item_id"], "previous_item_id": None})
                        await tx_queue.put({"pcm": evt["pcm"], "item_id": evt["item_id"]})

            elif mt == "input_audio_buffer.clear":
                pipeline = AudioPipeline(cfg)
                await ws.send_json({"event_id": _make_eid(), "type": "input_audio_buffer.cleared"})

            elif mt == "input_audio_buffer.commit":
                events = pipeline.flush()
                cid = _make_iid()
                await ws.send_json({"event_id": _make_eid(), "type": "input_audio_buffer.committed", "item_id": cid, "previous_item_id": None})
                if events:
                    for evt in events:
                        if evt["type"] == "utterance_ready":
                            dur = len(evt["pcm"]) / (SAMPLES_PER_MS * BYTES_PER_SAMPLE)
                            log.info(f"Manual commit: {dur:.0f}ms")
                            await tx_queue.put({"pcm": evt["pcm"], "item_id": evt.get("item_id", cid)})
                else:
                    if not is_conversation:
                        await ws.send_json({"event_id": _make_eid(), "type": "conversation.item.input_audio_transcription.completed", "item_id": cid, "content_index": 0, "transcript": ""})

            elif mt in ("session.update", "transcription_session.update"):
                sd = msg.get("session", {})
                td = sd.get("turn_detection", {})
                if "silence_duration_ms" in td:
                    cfg.speech_stop_silence_ms = td["silence_duration_ms"]
                if "prefix_padding_ms" in td:
                    cfg.preroll_ms = td["prefix_padding_ms"]
                if "threshold" in td:
                    agg = int(max(0, min(3, td["threshold"] * 3)))
                    cfg.vad_aggressiveness = agg
                if sd.get("system_prompt"):
                    system_prompt = sd["system_prompt"]
                pipeline = AudioPipeline(cfg)
                rt = "transcription_session.updated" if "transcription" in mt else "session.updated"
                await ws.send_json({"event_id": _make_eid(), "type": rt, "session": {"id": sid, "type": session_type}})

            elif mt == "response.cancel":
                pass

    except WebSocketDisconnect:
        log.info(f"WS session {sid} disconnected")
    except Exception as e:
        log.error(f"WS error: {e}")
        if ws.client_state == WebSocketState.CONNECTED:
            await ws.close(code=1011, reason=str(e))
    finally:
        await tx_queue.put(None)
        await worker_task


# ---------------------------------------------------------------------------
# Other endpoints
# ---------------------------------------------------------------------------

@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": "lfm2.5-audio-1.5b", "object": "model", "owned_by": "LiquidAI"}]}

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None, "device": "cuda" if torch.cuda.is_available() else "cpu"}


# ---------------------------------------------------------------------------
# Test UI
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def ui():
    return TEST_UI_HTML

TEST_UI_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LFM2.5 Audio Test</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,sans-serif;background:#0a0a0a;color:#e0e0e0;min-height:100vh;display:flex;flex-direction:column;align-items:center;padding:2rem}
h1{font-size:1.5rem;margin-bottom:0.5rem;color:#fff}
.subtitle{color:#888;font-size:0.85rem;margin-bottom:2rem}
.tabs{display:flex;gap:0.5rem;margin-bottom:1.5rem;flex-wrap:wrap;justify-content:center}
.tab{padding:0.5rem 1.25rem;border:1px solid #333;border-radius:6px;background:transparent;color:#aaa;cursor:pointer;font-size:0.9rem;transition:all .15s}
.tab.active{background:#1a1a2e;color:#7c8cf8;border-color:#7c8cf8}
.panel{display:none;width:100%;max-width:640px}
.panel.active{display:block}
.card{background:#111;border:1px solid #222;border-radius:10px;padding:1.5rem;margin-bottom:1rem}
.btn{padding:0.6rem 1.5rem;border:none;border-radius:6px;font-size:0.9rem;cursor:pointer;transition:all .15s;font-weight:500}
.btn-primary{background:#7c8cf8;color:#fff}
.btn-primary:hover{background:#6b7bf7}
.btn-danger{background:#e04040;color:#fff}
.btn-danger:hover{background:#c03030}
.btn-green{background:#22c55e;color:#fff}
.btn-green:hover{background:#16a34a}
.btn:disabled{opacity:0.4;cursor:not-allowed}
.btn-row{display:flex;gap:0.75rem;margin-top:1rem;align-items:center;flex-wrap:wrap}
.status{font-size:0.8rem;color:#888;margin-left:0.5rem}
.status.active{color:#4ade80}
.status.error{color:#ef4444}
.status.warn{color:#f0c040}
.result{background:#0d0d0d;border:1px solid #222;border-radius:8px;padding:1rem;margin-top:1rem;min-height:80px;font-family:'JetBrains Mono',monospace;font-size:0.85rem;white-space:pre-wrap;word-break:break-word;line-height:1.6}
.result .interim{color:#7c8cf8}
.result .final{color:#4ade80}
.result .ai{color:#c084fc}
.log{background:#0d0d0d;border:1px solid #222;border-radius:8px;padding:0.75rem;margin-top:0.75rem;max-height:200px;overflow-y:auto;font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:#666;line-height:1.5}
.log .evt{color:#888}.log .speech{color:#f0c040}.log .delta{color:#7c8cf8}.log .done{color:#4ade80}.log .err{color:#ef4444}.log .audio{color:#c084fc}
label{font-size:0.85rem;color:#aaa;display:block;margin-bottom:0.3rem}
input[type=file]{margin-top:0.25rem;font-size:0.85rem;color:#ccc}
.meter{height:4px;background:#222;border-radius:2px;margin-top:0.75rem;overflow:hidden}
.meter-bar{height:100%;background:#7c8cf8;border-radius:2px;transition:width 0.1s;width:0%}
.meter-bar.speaking{background:#4ade80}
.meter-bar.ai{background:#c084fc}
</style>
</head>
<body>
<h1>LFM2.5-Audio</h1>
<p class="subtitle">Speech-to-Text &amp; Speech-to-Speech powered by Liquid AI</p>

<div class="tabs">
  <button class="tab active" onclick="switchTab('rest')">REST Upload</button>
  <button class="tab" onclick="switchTab('ws')">WS Transcription</button>
  <button class="tab" onclick="switchTab('conv')">WS Conversation</button>
</div>

<!-- REST panel -->
<div id="rest-panel" class="panel active">
  <div class="card">
    <label>Upload audio file (wav, mp3, webm, etc.)</label>
    <input type="file" id="audioFile" accept="audio/*">
    <div class="btn-row">
      <button class="btn btn-primary" id="restBtn" onclick="transcribeRest()">Transcribe</button>
      <label style="display:flex;align-items:center;gap:4px"><input type="checkbox" id="restStream"> Stream</label>
      <span class="status" id="restStatus"></span>
    </div>
    <div class="result" id="restResult">Upload a file and click Transcribe...</div>
  </div>
</div>

<!-- WS Transcription panel -->
<div id="ws-panel" class="panel">
  <div class="card">
    <div class="btn-row">
      <button class="btn btn-primary" id="wsStartBtn" onclick="wsStart('transcription')">Start</button>
      <button class="btn btn-danger" id="wsStopBtn" onclick="wsStop('ws')" disabled>Stop</button>
      <span class="status" id="wsStatus"></span>
    </div>
    <div class="meter"><div class="meter-bar" id="wsMeter"></div></div>
    <div class="result" id="wsResult">Click Start to begin transcription...</div>
    <div class="log" id="wsLog"></div>
  </div>
</div>

<!-- WS Conversation panel -->
<div id="conv-panel" class="panel">
  <div class="card">
    <div class="btn-row">
      <button class="btn btn-green" id="convStartBtn" onclick="wsStart('conversation')">Start Conversation</button>
      <button class="btn btn-danger" id="convStopBtn" onclick="wsStop('conv')" disabled>Stop</button>
      <span class="status" id="convStatus"></span>
    </div>
    <div class="meter"><div class="meter-bar" id="convMeter"></div></div>
    <div class="result" id="convResult">Click Start to begin a voice conversation...</div>
    <div class="log" id="convLog"></div>
  </div>
</div>

<script>
function switchTab(t) {
  document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(el => el.classList.remove('active'));
  const tabs = document.querySelectorAll('.tab');
  const map = {rest:0, ws:1, conv:2};
  tabs[map[t]].classList.add('active');
  document.getElementById(t + '-panel').classList.add('active');
}

// --- REST ---
async function transcribeRest() {
  const file = document.getElementById('audioFile').files[0];
  if (!file) { alert('Select a file'); return; }
  const stream = document.getElementById('restStream').checked;
  const btn = document.getElementById('restBtn');
  const status = document.getElementById('restStatus');
  const result = document.getElementById('restResult');
  btn.disabled = true; status.textContent = 'Uploading...'; status.className = 'status active';
  result.textContent = '';
  const fd = new FormData();
  fd.append('file', file);
  fd.append('stream', stream ? 'true' : 'false');
  try {
    const t0 = performance.now();
    const resp = await fetch('/v1/audio/transcriptions', { method: 'POST', body: fd });
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    if (stream) {
      const reader = resp.body.getReader();
      const dec = new TextDecoder();
      let buf = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const lines = buf.split('\n');
        buf = lines.pop();
        for (const line of lines) {
          if (!line.startsWith('data: ') || line === 'data: [DONE]') continue;
          try { const d = JSON.parse(line.slice(6)); result.textContent += d.text; } catch {}
        }
      }
    } else {
      const data = await resp.json();
      result.innerHTML = '<span class="final">' + (data.text || '(empty)') + '</span>';
    }
    status.textContent = 'Done in ' + (performance.now() - t0).toFixed(0) + 'ms'; status.className = 'status active';
  } catch (e) { status.textContent = e.message; status.className = 'status error'; }
  btn.disabled = false;
}

// --- WebSocket shared state ---
const sessions = {};

function _log(prefix, cls, msg) {
  const el = document.getElementById(prefix + 'Log');
  const d = document.createElement('div');
  d.className = cls;
  d.textContent = '[' + new Date().toLocaleTimeString() + '] ' + msg;
  el.appendChild(d);
  el.scrollTop = el.scrollHeight;
}

function _render(prefix, finalText, interim, aiText) {
  const el = document.getElementById(prefix + 'Result');
  let html = '';
  if (finalText) html += '<span class="final">' + finalText + '</span>';
  if (interim) html += (finalText ? ' ' : '') + '<span class="interim">' + interim + '</span>';
  if (aiText) html += '\n<span class="ai">AI: ' + aiText + '</span>';
  el.innerHTML = html || '<span style="color:#666">Listening...</span>';
}

async function wsStart(intent) {
  const prefix = intent === 'conversation' ? 'conv' : 'ws';
  const status = document.getElementById(prefix + 'Status');
  document.getElementById(prefix + 'StartBtn').disabled = true;
  document.getElementById(prefix + 'StopBtn').disabled = false;
  document.getElementById(prefix + 'Log').innerHTML = '';

  const s = { ws: null, audioCtx: null, mediaStream: null, processor: null, accum: '', finalText: '', aiText: '', playCtx: null };
  sessions[prefix] = s;

  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  s.ws = new WebSocket(proto + '//' + location.host + '/v1/realtime?intent=' + intent, ['realtime']);

  s.ws.onopen = () => {
    status.textContent = 'Connected'; status.className = 'status active';
    _log(prefix, 'evt', 'WebSocket connected (intent=' + intent + ')');
    s.ws.send(JSON.stringify({ type: 'transcription_session.update', session: {
      input_audio_format: 'pcm16',
      turn_detection: { type: 'server_vad', threshold: 0.5, silence_duration_ms: 1000 }
    }}));
  };

  // Audio playback for conversation mode
  if (intent === 'conversation') {
    s.playCtx = new AudioContext({ sampleRate: 24000 });
    s.playQueue = [];
    s.playing = false;
  }

  s.ws.onmessage = (e) => {
    try {
      const msg = JSON.parse(e.data);
      const t = msg.type;

      if (t === 'session.created') _log(prefix, 'evt', 'session: ' + msg.session?.id);
      else if (t.includes('.updated')) _log(prefix, 'evt', 'session updated');

      else if (t === 'input_audio_buffer.speech_started') {
        _log(prefix, 'speech', 'speech_started');
        status.textContent = 'Speech detected'; status.className = 'status active';
        document.getElementById(prefix + 'Meter').classList.add('speaking');
      }
      else if (t === 'input_audio_buffer.speech_stopped') {
        _log(prefix, 'speech', 'speech_stopped');
        status.textContent = 'Processing...'; status.className = 'status warn';
        document.getElementById(prefix + 'Meter').classList.remove('speaking');
      }
      else if (t === 'input_audio_buffer.committed') _log(prefix, 'evt', 'committed: ' + msg.item_id);

      // Transcription events
      else if (t === 'conversation.item.input_audio_transcription.delta') {
        s.accum += msg.delta;
        _log(prefix, 'delta', 'delta: ' + msg.delta);
        _render(prefix, s.finalText, s.accum, s.aiText);
      }
      else if (t === 'conversation.item.input_audio_transcription.completed') {
        if (msg.transcript) {
          s.finalText += (s.finalText ? ' ' : '') + msg.transcript;
          _log(prefix, 'done', 'completed: ' + msg.transcript);
        }
        s.accum = '';
        _render(prefix, s.finalText, '', s.aiText);
        status.textContent = 'Listening'; status.className = 'status active';
      }

      // Conversation response events
      else if (t === 'response.created') {
        _log(prefix, 'evt', 'response started');
        s.aiText = '';
        status.textContent = 'AI responding...'; status.className = 'status warn';
        document.getElementById(prefix + 'Meter').classList.add('ai');
      }
      else if (t === 'response.audio_transcript.delta') {
        s.aiText += msg.delta;
        _log(prefix, 'delta', 'ai: ' + msg.delta);
        _render(prefix, s.finalText, '', s.aiText);
      }
      else if (t === 'response.audio.delta') {
        const pcm = atob(msg.delta);
        const buf = new ArrayBuffer(pcm.length);
        const view = new Uint8Array(buf);
        for (let i = 0; i < pcm.length; i++) view[i] = pcm.charCodeAt(i);
        const i16 = new Int16Array(buf);
        const f32 = new Float32Array(i16.length);
        for (let i = 0; i < i16.length; i++) f32[i] = i16[i] / 32768;
        if (s.playCtx) {
          const ab = s.playCtx.createBuffer(1, f32.length, 24000);
          ab.getChannelData(0).set(f32);
          s.playQueue.push(ab);
          _playNext(s);
        }
        _log(prefix, 'audio', 'audio chunk: ' + f32.length + ' samples');
      }
      else if (t === 'response.audio_transcript.done') {
        _log(prefix, 'done', 'ai transcript: ' + msg.transcript);
      }
      else if (t === 'response.audio.done') {
        _log(prefix, 'done', 'audio done');
        document.getElementById(prefix + 'Meter').classList.remove('ai');
      }
      else if (t === 'response.done') {
        _log(prefix, 'done', 'response complete');
        status.textContent = 'Listening'; status.className = 'status active';
      }
      else if (t === 'error') {
        _log(prefix, 'err', 'error: ' + msg.error?.message);
      }
    } catch (ex) { console.error(ex); }
  };

  s.ws.onerror = () => { status.textContent = 'Error'; status.className = 'status error'; _log(prefix, 'err', 'WebSocket error'); };
  s.ws.onclose = () => { status.textContent = 'Disconnected'; status.className = 'status'; _log(prefix, 'evt', 'Disconnected'); };

  try {
    s.mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    s.audioCtx = new AudioContext({ sampleRate: 24000 });
    const source = s.audioCtx.createMediaStreamSource(s.mediaStream);

    const analyser = s.audioCtx.createAnalyser();
    analyser.fftSize = 256;
    source.connect(analyser);
    const dataArr = new Uint8Array(analyser.frequencyBinCount);
    const meterEl = document.getElementById(prefix + 'Meter');
    function tick() {
      if (!s.audioCtx) return;
      analyser.getByteFrequencyData(dataArr);
      let sum = 0; for (let i = 0; i < dataArr.length; i++) sum += dataArr[i];
      meterEl.style.width = Math.min(100, (sum / dataArr.length / 255) * 300) + '%';
      requestAnimationFrame(tick);
    }
    tick();

    s.processor = s.audioCtx.createScriptProcessor(4096, 1, 1);
    s.processor.onaudioprocess = (ev) => {
      if (!s.ws || s.ws.readyState !== WebSocket.OPEN) return;
      const f32 = ev.inputBuffer.getChannelData(0);
      const i16 = new Int16Array(f32.length);
      for (let i = 0; i < f32.length; i++) {
        const v = Math.max(-1, Math.min(1, f32[i]));
        i16[i] = v < 0 ? v * 0x8000 : v * 0x7FFF;
      }
      const bytes = new Uint8Array(i16.buffer);
      let bin = ''; for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
      s.ws.send(JSON.stringify({ type: 'input_audio_buffer.append', audio: btoa(bin) }));
    };
    source.connect(s.processor);
    s.processor.connect(s.audioCtx.destination);
    status.textContent = 'Listening'; status.className = 'status active';
  } catch (e) { status.textContent = e.message; status.className = 'status error'; }
}

function _playNext(s) {
  if (s.playing || !s.playQueue.length || !s.playCtx) return;
  s.playing = true;
  const buf = s.playQueue.shift();
  const src = s.playCtx.createBufferSource();
  src.buffer = buf;
  src.connect(s.playCtx.destination);
  src.onended = () => { s.playing = false; _playNext(s); };
  src.start();
}

async function wsStop(prefix) {
  const s = sessions[prefix];
  if (!s) return;
  document.getElementById(prefix + 'StartBtn').disabled = false;
  document.getElementById(prefix + 'StopBtn').disabled = true;
  document.getElementById(prefix + 'Meter').style.width = '0%';

  if (s.processor) { s.processor.disconnect(); s.processor = null; }
  if (s.audioCtx) { s.audioCtx.close(); s.audioCtx = null; }
  if (s.mediaStream) { s.mediaStream.getTracks().forEach(t => t.stop()); s.mediaStream = null; }

  if (s.ws && s.ws.readyState === WebSocket.OPEN) {
    s.ws.send(JSON.stringify({ type: 'input_audio_buffer.commit' }));
    await new Promise(r => {
      const orig = s.ws.onmessage;
      const timeout = setTimeout(r, 10000);
      s.ws.onmessage = (e) => {
        if (orig) orig(e);
        try {
          const msg = JSON.parse(e.data);
          if (msg.type === 'conversation.item.input_audio_transcription.completed' ||
              msg.type === 'response.done') { clearTimeout(timeout); r(); }
        } catch {}
      };
    });
    s.ws.close();
  }
  s.ws = null;
  if (s.playCtx) { s.playCtx.close(); s.playCtx = null; }
  document.getElementById(prefix + 'Status').textContent = 'Stopped';
  document.getElementById(prefix + 'Status').className = 'status';
}
</script>
</body>
</html>"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8091)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
