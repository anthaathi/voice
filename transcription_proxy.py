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
from pathlib import Path
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

MIN_PEAK_AMPLITUDE = 0.05

async def _transcribe_and_send(ws: WebSocket, pcm_bytes: bytes, item_id: str):
    wav_tensor, sr = _pcm16_24k_to_tensor(pcm_bytes)
    duration = len(pcm_bytes) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
    peak = float(wav_tensor.abs().max())
    log.info(f"Transcribing: {duration:.2f}s, peak={peak:.4f}")

    if peak < MIN_PEAK_AMPLITUDE:
        log.info(f"Rejected: peak {peak:.4f} < {MIN_PEAK_AMPLITUDE} (silence/noise)")
        if ws.client_state == WebSocketState.CONNECTED:
            await ws.send_json({
                "event_id": _make_eid(),
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": item_id, "content_index": 0, "transcript": "",
            })
        return

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
    peak = float(wav_tensor.abs().max())
    if peak < MIN_PEAK_AMPLITUDE:
        log.info(f"Conversation: rejected silence (peak={peak:.4f})")
        return
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
    speech_start_frames: int = 3
    speech_stop_silence_ms: int = 800
    min_utterance_ms: int = 800
    max_utterance_ms: int = 15000
    preroll_ms: int = 500
    ring_buffer_size: int = 8
    ring_buffer_threshold: int = 5
    vad_aggressiveness: int = 3

    def __init__(self, **kw):
        for k, v in kw.items():
            if hasattr(self, k):
                setattr(self, k, v)


class _State:
    IDLE = "idle"
    SPEAKING = "speaking"


class AudioPipeline:
    def __init__(self, cfg: AudioPipelineConfig | None = None):
        self.cfg = cfg or AudioPipelineConfig()
        self.vad = webrtcvad.Vad(self.cfg.vad_aggressiveness)
        self.state = _State.IDLE
        self.utterance_buf = bytearray()
        self.preroll_buf = bytearray()
        self.preroll_max = int(self.cfg.preroll_ms * SAMPLES_PER_MS * BYTES_PER_SAMPLE)
        self.min_bytes = int(self.cfg.min_utterance_ms * SAMPLES_PER_MS * BYTES_PER_SAMPLE)
        self.max_bytes = int(self.cfg.max_utterance_ms * SAMPLES_PER_MS * BYTES_PER_SAMPLE)
        self.ring_buffer = [False] * self.cfg.ring_buffer_size
        self.ring_idx = 0
        self.speech_start_count = 0
        self.silence_frames = 0
        self.silence_frames_threshold = int(self.cfg.speech_stop_silence_ms / VAD_FRAME_MS)
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
        self.total_samples += len(chunk_samples)

        frame_decisions = _vad_frames(self.vad, chunk_samples)

        for is_speech in frame_decisions:
            self._push_ring(is_speech)
            smoothed = self._ring_vote()

            if self.state == _State.IDLE:
                if smoothed:
                    self.speech_start_count += 1
                else:
                    self.speech_start_count = 0

                if self.speech_start_count >= self.cfg.speech_start_frames:
                    self.state = _State.SPEAKING
                    self.silence_frames = 0
                    self.speech_start_count = 0
                    self.item_id = _make_iid()
                    self.utterance_buf.clear()
                    self.utterance_buf.extend(self.preroll_buf)
                    ms = int(self.total_samples / SAMPLES_PER_MS)
                    events.append({"type": "speech_started", "audio_start_ms": ms, "item_id": self.item_id})

            elif self.state == _State.SPEAKING:
                if not smoothed:
                    self.silence_frames += 1
                    if self.silence_frames >= self.silence_frames_threshold:
                        events.extend(self._finalize("silence"))
                else:
                    self.silence_frames = 0

        if self.state == _State.IDLE:
            self.preroll_buf.extend(chunk_bytes)
            if len(self.preroll_buf) > self.preroll_max:
                self.preroll_buf = self.preroll_buf[-self.preroll_max:]
        elif self.state == _State.SPEAKING:
            self.utterance_buf.extend(chunk_bytes)
            if len(self.utterance_buf) >= self.max_bytes:
                events.extend(self._finalize("max_length"))

        return events

    def _finalize(self, reason: str) -> list[dict]:
        events = []
        ms = int(self.total_samples / SAMPLES_PER_MS)
        pcm = bytes(self.utterance_buf)
        self.utterance_buf.clear()
        self.preroll_buf.clear()
        self.state = _State.IDLE
        self.silence_frames = 0
        self.speech_start_count = 0
        self.ring_buffer = [False] * self.cfg.ring_buffer_size

        events.append({"type": "speech_stopped", "audio_end_ms": ms, "item_id": self.item_id})
        if len(pcm) >= self.min_bytes:
            events.append({"type": "utterance_ready", "pcm": pcm, "item_id": self.item_id, "reason": reason})
        else:
            dur_ms = len(pcm) / (SAMPLES_PER_MS * BYTES_PER_SAMPLE)
            log.info(f"Utterance too short ({dur_ms:.0f}ms < {self.cfg.min_utterance_ms}ms), discarded")
        return events

    def flush(self) -> list[dict]:
        if self.state == _State.SPEAKING and self.utterance_buf:
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

_INDEX_HTML = Path(__file__).parent / "index.html"

@app.get("/", response_class=HTMLResponse)
async def ui():
    return _INDEX_HTML.read_text()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8091)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
