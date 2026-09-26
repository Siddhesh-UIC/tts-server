"""VoxCPM2 TTS server - OpenAI-style /v1/audio/speech endpoint.

Auth: Bearer token must match env TTS_API_KEY (falls back to VLLM_API_KEY).
Run:  uvicorn tts_server:app --host 0.0.0.0 --port 8002
"""
import asyncio
import hmac
import io
import os
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

MODEL_ID = os.environ.get("TTS_MODEL", "openbmb/VoxCPM2")
API_KEY = os.environ.get("TTS_API_KEY") or os.environ.get("VLLM_API_KEY")

app = FastAPI(title="VoxCPM2 TTS")
state = {"model": None, "sr": 48000}
gpu_lock = asyncio.Lock()  # one synthesis at a time on the GPU


class SpeechRequest(BaseModel):
    input: str = Field(..., min_length=1, max_length=4000)
    model: Optional[str] = "voxcpm2"
    voice: Optional[str] = None            # accepted for OpenAI compatibility, ignored
    response_format: Optional[str] = "wav"  # "wav" or "pcm" (raw 16-bit little-endian mono)
    sample_rate: Optional[int] = None       # e.g. 8000 telephony, 16000, 24000; default = model rate
    cfg_value: Optional[float] = 2.0


def check_auth(authorization: Optional[str]) -> None:
    if not API_KEY:
        return
    expected = f"Bearer {API_KEY}"
    if not authorization or not hmac.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.on_event("startup")
def load_model() -> None:
    from voxcpm import VoxCPM
    model = VoxCPM.from_pretrained(MODEL_ID, load_denoiser=False)
    state["model"] = model
    state["sr"] = getattr(getattr(model, "tts_model", None), "sample_rate", 48000)


@app.get("/health")
def health():
    if state["model"] is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    return {"status": "ok"}


@app.get("/v1/models")
def models(authorization: Optional[str] = Header(None)):
    check_auth(authorization)
    return {"object": "list", "data": [{"id": "voxcpm2", "object": "model", "owned_by": "openbmb"}]}


def synthesize(text: str, cfg_value: float) -> np.ndarray:
    wav = state["model"].generate(text=text, cfg_value=cfg_value)
    return np.asarray(wav, dtype=np.float32).squeeze()


@app.post("/v1/audio/speech")
async def speech(req: SpeechRequest, authorization: Optional[str] = Header(None)):
    check_auth(authorization)
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model loading")
    fmt = (req.response_format or "wav").lower()
    if fmt not in ("wav", "pcm"):
        raise HTTPException(status_code=400, detail="response_format must be 'wav' or 'pcm'")

    loop = asyncio.get_running_loop()
    async with gpu_lock:
        audio = await loop.run_in_executor(None, synthesize, req.input, req.cfg_value or 2.0)

    sr = state["sr"]
    if req.sample_rate and req.sample_rate != sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=req.sample_rate)
        sr = req.sample_rate

    if fmt == "pcm":
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()
        return Response(content=pcm, media_type="audio/pcm", headers={"X-Sample-Rate": str(sr)})

    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return Response(content=buf.getvalue(), media_type="audio/wav", headers={"X-Sample-Rate": str(sr)})
