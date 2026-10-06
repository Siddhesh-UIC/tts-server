"""GPU VM dashboard: live service status, GPU/host load, token throughput and a request feed.

Runs on its own port (default 3000), reads only public signals (/health, vLLM /metrics,
nvidia-smi, /proc), so it cannot disturb the model servers. Request events arrive on
POST /ingest from the telemetry gateway (localhost only).

Env: DASH_PORT [3000]  DASH_USER [admin]  DASH_PASSWORD [falls back to VLLM_API_KEY]
     DASH_SERVICES [llm:8000,stt:8001,tts:8002,sahabat:8003]  DASH_DISK [/workspace]
"""
import asyncio, base64, collections, contextlib, hmac, json, os, shutil, time

import httpx, uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Route

E = os.environ.get
PORT = int(E("DASH_PORT", "3000"))
USER = E("DASH_USER", "admin")
PASSWORD = E("DASH_PASSWORD") or E("VLLM_API_KEY") or ""
SERVICES = [(s, int(p)) for s, p in (x.split(":") for x in E("DASH_SERVICES", "llm:8000,stt:8001,tts:8002,sahabat:8003").split(","))]
DISK = E("DASH_DISK", "/workspace")
HIST = 300  # 5 minutes at 1 s

client = httpx.AsyncClient(timeout=1.5)
state = {"ts": 0, "gpu": {}, "host": {}, "services": {}, "hist": {}}
events = collections.deque(maxlen=500); event_seq = 0
_prev = {}  # previous counter values per service, for rates
_cpu_prev = None; _disk = {"ts": 0}; _up_since = {}

def parse_metrics(text):
    v, model = {}, None
    for line in text.splitlines():
        if not line.startswith("vllm:"): continue
        try:
            name, val = line.rsplit(" ", 1); key = name.split("{")[0]
            v[key] = v.get(key, 0.0) + float(val)
            if model is None and 'model_name="' in name: model = name.split('model_name="', 1)[1].split('"', 1)[0]
        except ValueError: pass
    return v, model

def g(v, *ks):
    for k in ks:
        if k in v: return v[k]
    return None

async def poll_service(name, port, now):
    s = {"name": name, "port": port}
    try: code = (await client.get(f"http://127.0.0.1:{port}/health")).status_code
    except Exception: code = 0
    s["state"] = {200: "up", 503: "loading"}.get(code, "down")
    if s["state"] == "up": _up_since.setdefault(name, now)
    else: _up_since.pop(name, None)
    s["up_seconds"] = round(now - _up_since[name]) if name in _up_since else None
    if s["state"] != "up": return s
    try: r = await client.get(f"http://127.0.0.1:{port}/metrics")
    except Exception: return s
    if r.status_code != 200 or "vllm:" not in r.text:
        s["kind"] = "custom"; return s
    v, model = parse_metrics(r.text); s["kind"] = "vllm"; s["model"] = model
    s["running"] = g(v, "vllm:num_requests_running"); s["waiting"] = g(v, "vllm:num_requests_waiting")
    s["kv"] = g(v, "vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")
    cur = {"gen": g(v, "vllm:generation_tokens_total"), "prompt": g(v, "vllm:prompt_tokens_total"),
           "req": g(v, "vllm:request_success_total"),
           "ttft_s": g(v, "vllm:time_to_first_token_seconds_sum"), "ttft_c": g(v, "vllm:time_to_first_token_seconds_count"),
           "e2e_s": g(v, "vllm:e2e_request_latency_seconds_sum"), "e2e_c": g(v, "vllm:e2e_request_latency_seconds_count"), "t": now}
    s["totals"] = {"requests": cur["req"], "prompt_tokens": cur["prompt"], "generated_tokens": cur["gen"]}
    hist = _prev.setdefault(name, collections.deque(maxlen=61))
    hist.append(cur)
    def rate(k, back):
        if len(hist) < 2: return 0.0
        a, b = hist[max(0, len(hist) - 1 - back)], hist[-1]
        if a.get(k) is None or b.get(k) is None or b["t"] <= a["t"] or b[k] < a[k]: return 0.0
        return (b[k] - a[k]) / (b["t"] - a["t"])
    def avg(ks, kc, back):
        if len(hist) < 2: return None
        a, b = hist[max(0, len(hist) - 1 - back)], hist[-1]
        if None in (a.get(ks), b.get(ks), a.get(kc), b.get(kc)) or b[kc] <= a[kc]: return None
        return (b[ks] - a[ks]) / (b[kc] - a[kc])
    s["gen_tps"] = round(rate("gen", 1)); s["prompt_tps"] = round(rate("prompt", 1))
    s["req_per_min"] = round(rate("req", 60) * 60, 1)
    s["ttft_ms_1m"] = None if (x := avg("ttft_s", "ttft_c", 60)) is None else round(x * 1000)
    s["latency_ms_1m"] = None if (x := avg("e2e_s", "e2e_c", 60)) is None else round(x * 1000)
    return s

async def poll_gpu():
    try:
        p = await asyncio.create_subprocess_exec("nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,power.limit",
                                                 "--format=csv,noheader,nounits", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        out, _ = await asyncio.wait_for(p.communicate(), 3)
        n, mu, mt, ut, te, pw, pl = [x.strip() for x in out.decode().splitlines()[0].split(",")]
        return {"name": n, "vram_used_gib": round(float(mu) / 1024, 1), "vram_total_gib": round(float(mt) / 1024, 1),
                "util": float(ut), "temp_c": float(te), "power_w": round(float(pw)), "power_limit_w": round(float(pl))}
    except Exception: return {}

def poll_host(now):
    global _cpu_prev
    h = {}
    try:
        f = [int(x) for x in open("/proc/stat").readline().split()[1:9]]; tot, idle = sum(f), f[3] + f[4]
        if _cpu_prev and tot > _cpu_prev[0]: h["cpu"] = round(100 * (1 - (idle - _cpu_prev[1]) / (tot - _cpu_prev[0])), 1)
        _cpu_prev = (tot, idle)
    except Exception: pass
    h["vcpus"] = os.cpu_count()
    try:
        mi = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo")}
        h["ram_total_gib"] = round(mi["MemTotal"] / 2**20, 1); h["ram_used_gib"] = round((mi["MemTotal"] - mi["MemAvailable"]) / 2**20, 1)
    except Exception: pass
    if now - _disk["ts"] > 30:
        try: du = shutil.disk_usage(DISK); _disk.update(ts=now, total=round(du.total / 2**30, 1), used=round(du.used / 2**30, 1))
        except Exception: _disk["ts"] = now
    h["disk_path"] = DISK; h["disk_total_gib"] = _disk.get("total"); h["disk_used_gib"] = _disk.get("used")
    return h

def push_hist(key, val):
    state["hist"].setdefault(key, collections.deque(maxlen=HIST)).append(val)

async def collector():
    tick = 0
    while True:
        now = time.time()
        results = await asyncio.gather(*(poll_service(n, p, now) for n, p in SERVICES))
        state["services"] = {s["name"]: s for s in results}
        if tick % 2 == 0: state["gpu"] = await poll_gpu() or state["gpu"]
        state["host"] = poll_host(now); state["ts"] = now
        push_hist("gpu_util", state["gpu"].get("util", 0))
        for s in results: push_hist(f"{s['name']}_tps", s.get("gen_tps", 0) or 0)
        tick += 1
        await asyncio.sleep(max(0.2, 1 - (time.time() - now)))

def snapshot():
    return {"ts": state["ts"], "gpu": state["gpu"], "host": state["host"], "services": list(state["services"].values()),
            "hist": {k: list(v) for k, v in state["hist"].items()}, "event_seq": event_seq}

# ------------------------------------------------------------------ auth
def authorized(req):
    if not PASSWORD: return True
    h = req.headers.get("authorization", "")
    if not h.startswith("Basic "): return False
    try: u, _, p = base64.b64decode(h[6:]).decode().partition(":")
    except Exception: return False
    return hmac.compare_digest(u, USER) and hmac.compare_digest(p, PASSWORD)

def need_auth():
    return Response("Sign in to view the GPU dashboard.", status_code=401, headers={"WWW-Authenticate": 'Basic realm="GPU dashboard"'})

# ------------------------------------------------------------------ routes
async def health(req): return JSONResponse({"status": "ok"})

async def ingest(req: Request):
    global event_seq
    if req.client and req.client.host not in ("127.0.0.1", "::1"): return Response(status_code=403)
    try: ev = await req.json()
    except Exception: return Response(status_code=400)
    event_seq += 1; ev["_seq"] = event_seq; events.append(ev)
    return Response(status_code=204)

async def api_snapshot(req):
    if not authorized(req): return need_auth()
    snap = snapshot(); snap["events"] = list(events)[-100:]; return JSONResponse(snap)

async def api_stream(req: Request):
    if not authorized(req): return need_auth()
    async def gen():
        last = 0
        first = True
        while True:
            if await req.is_disconnected(): break
            snap = snapshot()
            new = [e for e in events if e["_seq"] > last][-(200 if first else 100):]
            if new: last = new[-1]["_seq"]
            snap["events"] = new; first = False
            yield f"data: {json.dumps(snap, ensure_ascii=False)}\n\n"
            await asyncio.sleep(1)
    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

async def page(req):
    if not authorized(req): return need_auth()
    return HTMLResponse(HTML)

@contextlib.asynccontextmanager
async def lifespan(app):
    task = asyncio.create_task(collector()); yield; task.cancel()

app = Starlette(routes=[Route("/", page), Route("/health", health), Route("/ingest", ingest, methods=["POST"]),
                        Route("/api/snapshot", api_snapshot), Route("/api/stream", api_stream)], lifespan=lifespan)

HTML = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html"), encoding="utf-8").read()

if __name__ == "__main__":
    if not PASSWORD: print("[dashboard] WARNING: no DASH_PASSWORD or VLLM_API_KEY set - dashboard is open to anyone", flush=True)
    print(f"[dashboard] serving on :{PORT}, services {SERVICES}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
