# tts-server

VoxCPM2 text-to-speech behind an OpenAI-style `POST /v1/audio/speech` endpoint, running on the GPU VM on port 8002. API details are in the model API reference in `SAM-voice-pipe`.

## Fixed voice (reference clip + seed)

Without a reference, VoxCPM2 invents a new speaker for every request, so the voice changes from sentence to sentence. Set these environment variables on the VM (Northflank service → environment) to lock it to one voice:

| Variable | Value | Effect |
| --- | --- | --- |
| `TTS_REFERENCE_WAV` | Path on the VM to a clean 5–15 s WAV, e.g. `/workspace/voices/agent.wav` | Every request is spoken in this voice |
| `TTS_REFERENCE_TEXT` | The exact words spoken in that clip | Closer clone (VoxCPM2 "continuation" mode); leave unset if unsure |
| `TTS_SEED` | Any integer, e.g. `42` | Same text → same audio every time; requests can override with a `"seed"` field |

If `TTS_REFERENCE_WAV` points to a missing file, the server refuses to start (check the `[tts]` log). `GET /health` reports what is active: `{"status":"ok","reference":true,"reference_text":true,"seed":42}`.

### Picking the voice

`voices/candidate-1.wav` … `candidate-5.wav` are five voices generated from the description `(A warm, calm young woman, speaking softly, slowly and politely)`. All five say:

> Hello, and thank you for calling. I'm here to help you with your account today. Please take your time, and tell me how I can assist you.

1. Listen and pick one. If you don't like any, generate more from the voice pipe with a different `TTS_VOICE`.
2. Copy it to the VM, e.g. `/workspace/voices/agent.wav`. Use a persistent volume path, or it is lost on restart.
3. Set `TTS_REFERENCE_WAV=/workspace/voices/agent.wav`, `TTS_REFERENCE_TEXT=` the sentence above, and `TTS_SEED=42`.
4. Deploy the updated `tts_server.py` to the VM (wherever `/workspace/start.sh` starts it from) and restart the service.
5. Check `curl https://<tts-host>/health` shows `"reference": true`.

Any recording works as a reference, including a real person's voice, but only with their consent.

## Named voices (one per language)

A request's `voice` field picks a clip from `voices/` next to `tts_server.py` (on the VM `/workspace/voices/`; set `TTS_VOICES_DIR` to use another folder):

| `voice` | Clip | Transcript (closer cloning) |
| --- | --- | --- |
| `id` | `voices/id.wav`: Indonesian female, 12.4 s | `voices/id.txt` |
| `ar` | `voices/ar.wav`: Arabic female, 13.4 s | `voices/ar.txt` |
| none, or a name with no `.wav` | the fixed voice (`TTS_REFERENCE_WAV`) | `TTS_REFERENCE_TEXT` |

The voice pipe sends the call's language (`en`, `id`, `ar`) as `voice`, so English keeps the fixed voice until there is an `en.wav`. Each response's `X-Voice` header names the voice used, and `GET /health` lists them under `"voices"`.

Adding a voice: a clean 5–15 s mono WAV that starts and ends on a whole sentence, plus a `.txt` with exactly the words spoken (numbers and titles as said, e.g. "Mister", not "Mr."). Only plain names (letters, digits, `-`, `_`). It takes effect on the next request; no restart needed.

## Run

```bash
uvicorn tts_server:app --host 0.0.0.0 --port 8002
```

Auth: `Authorization: Bearer <key>`, matching `TTS_API_KEY` (falls back to `VLLM_API_KEY`).
