"""VoxCPM2 TTS server - OpenAI-style /v1/audio/speech endpoint.

Auth: Bearer token must match env TTS_API_KEY (falls back to VLLM_API_KEY).
Run:  uvicorn tts_server:app --host 0.0.0.0 --port 8002

Fixed voice (optional, env):
  TTS_REFERENCE_WAV   path to a clean 5-15 s clip of the voice to clone; every request uses this speaker
  TTS_REFERENCE_TEXT  exact transcript of that clip; enables VoxCPM2's closer "continuation" cloning
  TTS_SEED            integer; same seed + same text -> same audio. A request's "seed" field overrides it
Without a reference, VoxCPM2 invents a new speaker per request (a "(description)" prefix in the text steers it).

Named voices: a request's "voice" field picks <TTS_VOICES_DIR>/<voice>.wav (default dir: voices/ next to this file),
cloned with <voice>.txt as its transcript when that file exists. E.g. voice "id" -> voices/id.wav + voices/id.txt.
No voice, or one with no file, uses the fixed voice above (the response's X-Voice header says which was used).
"""
import asyncio
import hmac
import io
import os
import random
import re
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

MODEL_ID = os.environ.get("TTS_MODEL", "openbmb/VoxCPM2")
API_KEY = os.environ.get("TTS_API_KEY") or os.environ.get("VLLM_API_KEY")
REFERENCE_WAV = os.environ.get("TTS_REFERENCE_WAV") or None
REFERENCE_TEXT = os.environ.get("TTS_REFERENCE_TEXT") or None
DEFAULT_SEED = int(os.environ["TTS_SEED"]) if os.environ.get("TTS_SEED") else None
VOICES_DIR = Path(os.environ.get("TTS_VOICES_DIR") or Path(__file__).parent / "voices")

app = FastAPI(title="VoxCPM2 TTS")
state = {"model": None, "sr": 48000}
gpu_lock = asyncio.Lock()  # one synthesis at a time on the GPU


class SpeechRequest(BaseModel):
    input: str = Field(..., min_length=1, max_length=4000)
    model: Optional[str] = "voxcpm2"
    voice: Optional[str] = None            # a named voice in VOICES_DIR (e.g. "id"); unknown or None: the fixed voice
    response_format: Optional[str] = "wav"  # "wav" or "pcm" (raw 16-bit little-endian mono)
    sample_rate: Optional[int] = None       # e.g. 8000 telephony, 16000, 24000; default = model rate
    cfg_value: Optional[float] = 2.0
    seed: Optional[int] = None              # overrides TTS_SEED for this request


def check_auth(authorization: Optional[str]) -> None:
    if not API_KEY:
        return
    expected = f"Bearer {API_KEY}"
    if not authorization or not hmac.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.on_event("startup")
def load_model() -> None:
    if REFERENCE_WAV and not Path(REFERENCE_WAV).is_file():  # fail at boot, not on the first call
        raise RuntimeError(f"TTS_REFERENCE_WAV not found: {REFERENCE_WAV}")
    from voxcpm import VoxCPM
    model = VoxCPM.from_pretrained(MODEL_ID, load_denoiser=False)
    state["model"] = model
    state["sr"] = getattr(getattr(model, "tts_model", None), "sample_rate", 48000)


@app.get("/health")
def health():
    if state["model"] is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    return {"status": "ok", "reference": bool(REFERENCE_WAV), "reference_text": bool(REFERENCE_TEXT), "seed": DEFAULT_SEED,
            "voices": sorted(p.stem for p in VOICES_DIR.glob("*.wav"))}


def voice_clip(name: Optional[str]) -> tuple[Optional[str], Optional[str], str]:
    """(wav path, transcript, name used) for a named voice; the fixed voice when it has no clip.
    Only plain names: "id", "ar_female", never a path."""
    if name and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name) and (VOICES_DIR / f"{name}.wav").is_file():
        txt = VOICES_DIR / f"{name}.txt"
        return str(VOICES_DIR / f"{name}.wav"), (txt.read_text(encoding="utf-8").strip() if txt.is_file() else None), name
    return REFERENCE_WAV, REFERENCE_TEXT, "default"


@app.get("/v1/models")
def models(authorization: Optional[str] = Header(None)):
    check_auth(authorization)
    return {"object": "list", "data": [{"id": "voxcpm2", "object": "model", "owned_by": "openbmb"}]}


def synthesize(text: str, cfg_value: float, seed: Optional[int], ref_wav: Optional[str], ref_text: Optional[str]) -> np.ndarray:
    if seed is not None:
        # Seed the global RNGs rather than passing seed= to generate(), so it works on any voxcpm version.
        # Safe because gpu_lock serializes synthesis.
        import torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)  # also seeds CUDA
    kwargs = {}
    if ref_wav:
        kwargs["reference_wav_path"] = ref_wav
        if ref_text:
            kwargs.update(prompt_wav_path=ref_wav, prompt_text=ref_text)
    wav = state["model"].generate(text=text, cfg_value=cfg_value, **kwargs)
    return np.asarray(wav, dtype=np.float32).squeeze()


@app.post("/v1/audio/speech")
async def speech(req: SpeechRequest, authorization: Optional[str] = Header(None)):
    check_auth(authorization)
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model loading")
    fmt = (req.response_format or "wav").lower()
    if fmt not in ("wav", "pcm"):
        raise HTTPException(status_code=400, detail="response_format must be 'wav' or 'pcm'")

    seed = req.seed if req.seed is not None else DEFAULT_SEED
    ref_wav, ref_text, voice = voice_clip(req.voice)
    loop = asyncio.get_running_loop()
    async with gpu_lock:
        audio = await loop.run_in_executor(None, synthesize, req.input, req.cfg_value or 2.0, seed, ref_wav, ref_text)

    sr = state["sr"]
    if req.sample_rate and req.sample_rate != sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=req.sample_rate)
        sr = req.sample_rate

    if fmt == "pcm":
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()
        return Response(content=pcm, media_type="audio/pcm", headers={"X-Sample-Rate": str(sr), "X-Voice": voice})

    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return Response(content=buf.getvalue(), media_type="audio/wav", headers={"X-Sample-Rate": str(sr), "X-Voice": voice})
