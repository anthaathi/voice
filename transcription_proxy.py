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
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
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




# ---------------------------------------------------------------------------
# Audio pipeline: noise suppression, VAD, endpoint detection, transcription
# ---------------------------------------------------------------------------

SAMPLE_RATE = 24000
BYTES_PER_SAMPLE = 2
SAMPLES_PER_MS = SAMPLE_RATE // 1000
VAD_FRAME_MS = 30
VAD_FRAME_SAMPLES_16K = int(16000 * VAD_FRAME_MS / 1000)


def _make_eid():
    return f"evt_{uuid.uuid4().hex[:24]}"


def _make_iid():
    return f"item_{uuid.uuid4().hex[:24]}"


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


def _denoise_pcm(pcm_bytes: bytes, sample_rate: int = 24000) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".pcm", delete=False) as inf:
        inf.write(pcm_bytes)
        inf.flush()
        outp = inf.name + ".clean.pcm"
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "-i", inf.name,
                    "-af", "highpass=f=200,lowpass=f=3000,afftdn=nt=w:om=o:nr=20",
                    "-f", "s16le", "-ar", str(sample_rate), "-ac", "1", outp,
                ],
                capture_output=True, timeout=15,
            )
            if result.returncode == 0 and os.path.exists(outp):
                with open(outp, "rb") as f:
                    return f.read()
        except Exception as e:
            log.warning(f"Denoise failed: {e}")
        finally:
            os.unlink(inf.name)
            if os.path.exists(outp):
                os.unlink(outp)
    return pcm_bytes


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
    denoise: bool = True

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

        elif self.state == _State.SILENCE_PENDING:
            pass

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
            log.info(f"Utterance too short: {len(pcm) / (SAMPLES_PER_MS * BYTES_PER_SAMPLE):.0f}ms, discarded")

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


async def _transcription_worker(ws: WebSocket, queue: asyncio.Queue, denoise: bool):
    while True:
        job = await queue.get()
        if job is None:
            break
        pcm_bytes, item_id = job["pcm"], job["item_id"]
        try:
            if denoise:
                pcm_bytes = await asyncio.to_thread(_denoise_pcm, pcm_bytes, SAMPLE_RATE)
            await _transcribe_and_send(ws, pcm_bytes, item_id)
        except Exception as e:
            log.error(f"Transcription error: {e}")
        finally:
            queue.task_done()


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
    cfg = AudioPipelineConfig()
    pipeline = AudioPipeline(cfg)
    tx_queue: asyncio.Queue = asyncio.Queue(maxsize=10)
    worker_task = asyncio.create_task(_transcription_worker(ws, tx_queue, cfg.denoise))

    await ws.send_json({
        "event_id": _make_eid(), "type": "session.created",
        "session": {
            "id": sid, "object": "realtime.transcription_session", "type": "transcription",
            "audio": {"input": {
                "format": {"type": "audio/pcm", "rate": 24000},
                "transcription": {"model": "lfm2.5-audio-1.5b"},
                "turn_detection": {
                    "type": "server_vad",
                    "silence_duration_ms": cfg.speech_stop_silence_ms,
                    "prefix_padding_ms": cfg.preroll_ms,
                },
            }},
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
                pipeline = AudioPipeline(cfg)
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
    finally:
        await tx_queue.put(None)
        await worker_task


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": "lfm2.5-audio-1.5b", "object": "model", "owned_by": "LiquidAI"}]}


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None, "device": "cuda" if torch.cuda.is_available() else "cpu"}


@app.get("/", response_class=HTMLResponse)
async def ui():
    return TEST_UI_HTML


TEST_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LFM2.5 Transcription Test</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,sans-serif;background:#0a0a0a;color:#e0e0e0;min-height:100vh;display:flex;flex-direction:column;align-items:center;padding:2rem}
h1{font-size:1.5rem;margin-bottom:0.5rem;color:#fff}
.subtitle{color:#888;font-size:0.85rem;margin-bottom:2rem}
.tabs{display:flex;gap:0.5rem;margin-bottom:1.5rem}
.tab{padding:0.5rem 1.25rem;border:1px solid #333;border-radius:6px;background:transparent;color:#aaa;cursor:pointer;font-size:0.9rem;transition:all .15s}
.tab.active{background:#1a1a2e;color:#7c8cf8;border-color:#7c8cf8}
.panel{display:none;width:100%;max-width:600px}
.panel.active{display:block}
.card{background:#111;border:1px solid #222;border-radius:10px;padding:1.5rem;margin-bottom:1rem}
.btn{padding:0.6rem 1.5rem;border:none;border-radius:6px;font-size:0.9rem;cursor:pointer;transition:all .15s;font-weight:500}
.btn-primary{background:#7c8cf8;color:#fff}
.btn-primary:hover{background:#6b7bf7}
.btn-danger{background:#e04040;color:#fff}
.btn-danger:hover{background:#c03030}
.btn:disabled{opacity:0.4;cursor:not-allowed}
.btn-row{display:flex;gap:0.75rem;margin-top:1rem;align-items:center}
.status{font-size:0.8rem;color:#888;margin-left:0.5rem}
.status.active{color:#4ade80}
.status.error{color:#ef4444}
.result{background:#0d0d0d;border:1px solid #222;border-radius:8px;padding:1rem;margin-top:1rem;min-height:80px;font-family:'JetBrains Mono',monospace;font-size:0.85rem;white-space:pre-wrap;word-break:break-word;line-height:1.6}
.result .interim{color:#7c8cf8}
.result .final{color:#4ade80}
.log{background:#0d0d0d;border:1px solid #222;border-radius:8px;padding:0.75rem;margin-top:0.75rem;max-height:200px;overflow-y:auto;font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:#666;line-height:1.5}
.log .evt{color:#888}.log .speech{color:#f0c040}.log .delta{color:#7c8cf8}.log .done{color:#4ade80}.log .err{color:#ef4444}
label{font-size:0.85rem;color:#aaa;display:block;margin-bottom:0.3rem}
input[type=file]{margin-top:0.25rem;font-size:0.85rem;color:#ccc}
.meter{height:4px;background:#222;border-radius:2px;margin-top:0.75rem;overflow:hidden}
.meter-bar{height:100%;background:#7c8cf8;border-radius:2px;transition:width 0.1s;width:0%}
</style>
</head>
<body>
<h1>LFM2.5-Audio Transcription</h1>
<p class="subtitle">OpenAI-compatible STT powered by Liquid AI</p>

<div class="tabs">
  <button class="tab active" onclick="switchTab('rest')">REST API</button>
  <button class="tab" onclick="switchTab('ws')">WebSocket Realtime</button>
</div>

<div id="rest-panel" class="panel active">
  <div class="card">
    <label>Upload audio file (wav, mp3, webm, etc.)</label>
    <input type="file" id="audioFile" accept="audio/*">
    <div class="btn-row">
      <button class="btn btn-primary" id="restBtn" onclick="transcribeRest()">Transcribe</button>
      <label><input type="checkbox" id="restStream"> Stream</label>
      <span class="status" id="restStatus"></span>
    </div>
    <div class="result" id="restResult">Upload a file and click Transcribe...</div>
  </div>
</div>

<div id="ws-panel" class="panel">
  <div class="card">
    <div class="btn-row">
      <button class="btn btn-primary" id="wsStartBtn" onclick="wsStart()">Start Recording</button>
      <button class="btn btn-danger" id="wsStopBtn" onclick="wsStop()" disabled>Stop</button>
      <span class="status" id="wsStatus"></span>
    </div>
    <div class="meter"><div class="meter-bar" id="wsMeter"></div></div>
    <div class="result" id="wsResult">Click Start Recording to begin...</div>
    <div class="log" id="wsLog"></div>
  </div>
</div>

<script>
function switchTab(t) {
  document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(el => el.classList.remove('active'));
  document.querySelector(`.tab[onclick*="${t}"]`).classList.add('active');
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
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

    if (stream) {
      const reader = resp.body.getReader();
      const dec = new TextDecoder();
      let buf = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const lines = buf.split('\\n');
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
    const ms = (performance.now() - t0).toFixed(0);
    status.textContent = `Done in ${ms}ms`; status.className = 'status active';
  } catch (e) {
    status.textContent = e.message; status.className = 'status error';
  }
  btn.disabled = false;
}

// --- WebSocket ---
let ws = null, audioCtx = null, mediaStream = null, processor = null;
let wsAccum = '', wsFinalText = '';

function wsLog(cls, msg) {
  const el = document.getElementById('wsLog');
  const d = document.createElement('div');
  d.className = cls;
  d.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
  el.appendChild(d);
  el.scrollTop = el.scrollHeight;
}

function wsRender() {
  const el = document.getElementById('wsResult');
  let html = '';
  if (wsFinalText) html += '<span class="final">' + wsFinalText + '</span>';
  if (wsAccum) html += (wsFinalText ? ' ' : '') + '<span class="interim">' + wsAccum + '</span>';
  el.innerHTML = html || '<span style="color:#666">Listening...</span>';
}

async function wsStart() {
  const status = document.getElementById('wsStatus');
  document.getElementById('wsStartBtn').disabled = true;
  document.getElementById('wsStopBtn').disabled = false;
  document.getElementById('wsLog').innerHTML = '';
  wsAccum = ''; wsFinalText = '';

  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${proto}//${location.host}/v1/realtime?intent=transcription`, ['realtime']);

  ws.onopen = () => {
    status.textContent = 'Connected'; status.className = 'status active';
    wsLog('evt', 'WebSocket connected');
    ws.send(JSON.stringify({ type: 'transcription_session.update', session: {
      input_audio_format: 'pcm16',
      turn_detection: { type: 'server_vad', threshold: 0.5, silence_duration_ms: 1200 }
    }}));
  };

  ws.onmessage = (e) => {
    try {
      const msg = JSON.parse(e.data);
      if (msg.type === 'session.created') wsLog('evt', 'session.created: ' + msg.session?.id);
      else if (msg.type === 'transcription_session.updated') wsLog('evt', 'session updated');
      else if (msg.type === 'input_audio_buffer.speech_started') { wsLog('speech', 'speech_started'); status.textContent = 'Speech detected'; }
      else if (msg.type === 'input_audio_buffer.speech_stopped') { wsLog('speech', 'speech_stopped'); status.textContent = 'Processing...'; }
      else if (msg.type === 'input_audio_buffer.committed') wsLog('evt', 'committed: ' + msg.item_id);
      else if (msg.type === 'conversation.item.input_audio_transcription.delta') {
        wsAccum += msg.delta;
        wsLog('delta', 'delta: ' + msg.delta);
        wsRender();
      } else if (msg.type === 'conversation.item.input_audio_transcription.completed') {
        if (msg.transcript) {
          wsFinalText += (wsFinalText ? ' ' : '') + msg.transcript;
          wsLog('done', 'completed: ' + msg.transcript);
        }
        wsAccum = '';
        wsRender();
        status.textContent = 'Listening'; status.className = 'status active';
      } else if (msg.type === 'error') {
        wsLog('err', 'error: ' + msg.error?.message);
      }
    } catch {}
  };

  ws.onerror = () => { status.textContent = 'Error'; status.className = 'status error'; wsLog('err', 'WebSocket error'); };
  ws.onclose = () => { status.textContent = 'Disconnected'; status.className = 'status'; wsLog('evt', 'Disconnected'); };

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioCtx = new AudioContext({ sampleRate: 24000 });
    const source = audioCtx.createMediaStreamSource(mediaStream);

    const analyser = audioCtx.createAnalyser();
    analyser.fftSize = 256;
    source.connect(analyser);
    const dataArr = new Uint8Array(analyser.frequencyBinCount);
    const meterEl = document.getElementById('wsMeter');
    function tick() {
      if (!audioCtx) return;
      analyser.getByteFrequencyData(dataArr);
      let sum = 0; for (let i = 0; i < dataArr.length; i++) sum += dataArr[i];
      meterEl.style.width = Math.min(100, (sum / dataArr.length / 255) * 300) + '%';
      requestAnimationFrame(tick);
    }
    tick();

    processor = audioCtx.createScriptProcessor(4096, 1, 1);
    processor.onaudioprocess = (e) => {
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      const f32 = e.inputBuffer.getChannelData(0);
      const i16 = new Int16Array(f32.length);
      for (let i = 0; i < f32.length; i++) {
        const s = Math.max(-1, Math.min(1, f32[i]));
        i16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
      }
      const bytes = new Uint8Array(i16.buffer);
      let bin = ''; for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
      ws.send(JSON.stringify({ type: 'input_audio_buffer.append', audio: btoa(bin) }));
    };
    source.connect(processor);
    processor.connect(audioCtx.destination);

    status.textContent = 'Listening'; status.className = 'status active';
  } catch (e) {
    status.textContent = e.message; status.className = 'status error';
  }
}

async function wsStop() {
  document.getElementById('wsStartBtn').disabled = false;
  document.getElementById('wsStopBtn').disabled = true;
  document.getElementById('wsMeter').style.width = '0%';

  if (processor) { processor.disconnect(); processor = null; }
  if (audioCtx) { audioCtx.close(); audioCtx = null; }
  if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }

  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'input_audio_buffer.commit' }));
    await new Promise(r => {
      const orig = ws.onmessage;
      const timeout = setTimeout(r, 10000);
      ws.onmessage = (e) => {
        if (orig) orig(e);
        try {
          const msg = JSON.parse(e.data);
          if (msg.type === 'conversation.item.input_audio_transcription.completed') { clearTimeout(timeout); r(); }
        } catch {}
      };
    });
    ws.close();
  }
  ws = null;
  document.getElementById('wsStatus').textContent = 'Stopped';
  document.getElementById('wsStatus').className = 'status';
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
