"""Telemetry gateway for the GPU VM model servers.

Sits in front of the LLM, STT and TTS servers, forwards every request unchanged
(streaming included) and publishes one telemetry event per model request to Solace,
plus periodic status events. If Solace is unreachable the proxy keeps working.

Env (defaults in brackets):
  GW_HOST [0.0.0.0]   GW_PORTS [llm:8000,stt:8001,tts:8002]
  GW_UPSTREAMS [llm:http://127.0.0.1:9000,stt:http://127.0.0.1:9001,tts:http://127.0.0.1:9002]
  SOLACE_HOST, SOLACE_VPN, SOLACE_USERNAME, SOLACE_PASSWORD   (unset = print events to stdout)
  TELEMETRY_PREFIX [ivr/demo/gpu1]   TELEMETRY_TEXT [preview|none|full]   TELEMETRY_PREVIEW_CHARS [200]
  TELEMETRY_STATUS_SEC [10]   SOLACE_TRUST_STORE [/etc/ssl/certs]
  DASHBOARD_INGEST [http://127.0.0.1:3000/ingest]  (request events are also sent to the dashboard; empty = off)
"""
import asyncio, contextlib, json, os, queue, re, socket, threading, time, uuid
from datetime import datetime, timezone

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route

def env(k, d=""): return os.environ.get(k, d)

PREFIX = env("TELEMETRY_PREFIX", "ivr/demo/gpu1").strip("/")
TEXT_MODE = env("TELEMETRY_TEXT", "preview").lower()
PREVIEW = int(env("TELEMETRY_PREVIEW_CHARS", "200"))
STATUS_SEC = float(env("TELEMETRY_STATUS_SEC", "10"))
PORTS = {int(p): s for s, p in (x.split(":") for x in env("GW_PORTS", "llm:8000,stt:8001,tts:8002").split(","))}
UPSTREAMS = dict(x.split(":", 1) for x in env(
    "GW_UPSTREAMS", "llm:http://127.0.0.1:9000,stt:http://127.0.0.1:9001,tts:http://127.0.0.1:9002").split(","))
TRACKED = {"llm": ("/v1/chat/completions", "/v1/completions"),
           "stt": ("/v1/audio/transcriptions", "/v1/audio/translations"),
           "tts": ("/v1/audio/speech",)}
HOP = {b"host", b"content-length", b"connection", b"keep-alive", b"transfer-encoding", b"accept-encoding", b"te", b"upgrade"}
RESP_DROP = {"content-length", "transfer-encoding", "connection", "content-encoding", "keep-alive"}

def now_iso(): return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
def topic_safe(s, d="none"): s = re.sub(r"[^A-Za-z0-9._-]", "_", str(s or ""))[:64]; return s or d
def clip(s):
    if s is None or TEXT_MODE == "none": return None
    s = str(s); return s if TEXT_MODE == "full" else s[:PREVIEW]

# ---------------------------------------------------------------- publisher
class Publisher:
    """Background-thread Solace publisher. Never blocks or breaks request handling."""
    def __init__(self):
        self.q = queue.Queue(maxsize=10000); self.dropped = 0; self.sent = 0
        self.enabled = bool(env("SOLACE_HOST"))
        threading.Thread(target=self._run, daemon=True).start()

    def publish(self, topic, payload):
        try: self.q.put_nowait((topic, json.dumps(payload, ensure_ascii=False)))
        except queue.Full: self.dropped += 1

    def _connect(self):
        from solace.messaging.messaging_service import MessagingService, RetryStrategy
        props = {"solace.messaging.transport.host": env("SOLACE_HOST"),
                 "solace.messaging.service.vpn-name": env("SOLACE_VPN"),
                 "solace.messaging.authentication.basic.username": env("SOLACE_USERNAME"),
                 "solace.messaging.authentication.basic.password": env("SOLACE_PASSWORD"),
                 "solace.messaging.tls.trust-store-path": env("SOLACE_TRUST_STORE", "/etc/ssl/certs")}
        svc = (MessagingService.builder().from_properties(props)
               .with_reconnection_retry_strategy(RetryStrategy.forever_retry(3000)).build())
        svc.connect()
        pub = svc.create_direct_message_publisher_builder().on_back_pressure_reject(buffer_capacity=1000).build()
        pub.start()
        return svc, pub

    def _forward_to_dashboard(self, topic, data):
        url = env("DASHBOARD_INGEST", "http://127.0.0.1:3000/ingest")
        if not url or "/request/" not in topic: return
        try:
            import urllib.request
            req = urllib.request.Request(url, data=data.encode(), headers={"Content-Type": "application/json"}, method="POST")
            urllib.request.urlopen(req, timeout=0.3).close()
        except Exception: pass   # dashboard down or not installed: ignore

    def _run(self):
        svc = pub = None
        while True:
            topic, data = self.q.get()
            self._forward_to_dashboard(topic, data)
            if not self.enabled:
                print(f"[telemetry] {topic} {data}", flush=True); continue
            try:
                if pub is None:
                    svc, pub = self._connect()
                    print(f"[gateway] connected to Solace {env('SOLACE_HOST')} vpn={env('SOLACE_VPN')}", flush=True)
                from solace.messaging.resources.topic import Topic
                msg = svc.message_builder().with_http_content_header("application/json", "utf-8").build(data)
                pub.publish(msg, Topic.of(topic)); self.sent += 1
            except Exception as e:
                self.dropped += 1
                print(f"[gateway] publish failed ({type(e).__name__}: {e}); retrying connection in 5 s", flush=True)
                try:
                    if svc: svc.disconnect()
                except Exception: pass
                svc = pub = None; time.sleep(5)

PUB = Publisher()
client = httpx.AsyncClient(timeout=httpx.Timeout(None, connect=5.0), limits=httpx.Limits(max_connections=200))

# ---------------------------------------------------------------- request analysis
def user_text(messages):
    for m in reversed(messages or []):
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, list): c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
            return c
    return None

class Tracker:
    """Collects facts about one request while its response streams through."""
    def __init__(self, svc, req, body):
        self.svc, self.t0 = svc, time.perf_counter()
        h = req.headers
        self.ev = {"v": 1, "service": svc, "endpoint": req.url.path, "ts": now_iso(),
                   "request_id": h.get("x-request-id") or uuid.uuid4().hex[:12],
                   "call_id": h.get("x-call-id"), "turn_id": h.get("x-turn-id"),
                   "client": (h.get("x-forwarded-for") or (req.client.host if req.client else "")).split(",")[0].strip()}
        self.stream = False; self.sse_buf = ""; self.body_buf = bytearray(); self.bytes_out = 0
        self.content = []; self.tools = []; self.finish = None; self.usage = None; self.ttft = None
        self.ctype = ""
        if svc == "llm":
            try:
                j = json.loads(body or b"{}")
                self.ev["model"] = j.get("model"); self.stream = bool(j.get("stream"))
                self.ev["messages"] = len(j.get("messages") or []); self.ev["tools_offered"] = len(j.get("tools") or [])
                self.ev["input"] = clip(user_text(j.get("messages")) if "messages" in j else j.get("prompt"))
            except Exception: pass
        elif svc == "tts":
            try:
                j = json.loads(body or b"{}"); txt = j.get("input") or ""
                self.ev.update(model=j.get("model"), input_chars=len(txt), input=clip(txt),
                               format=(j.get("response_format") or "wav"), sample_rate_req=j.get("sample_rate"))
            except Exception: pass
        elif svc == "stt":
            self.ev["request_bytes"] = len(body or b"")

    def feed(self, chunk):
        self.bytes_out += len(chunk)
        if self.svc == "llm" and self.stream:
            self.sse_buf += chunk.decode("utf-8", "ignore")
            while "\n" in self.sse_buf:
                line, self.sse_buf = self.sse_buf.split("\n", 1); line = line.strip()
                if not line.startswith("data:"): continue
                data = line[5:].strip()
                if data == "[DONE]": continue
                try: j = json.loads(data)
                except Exception: continue
                if j.get("usage"): self.usage = j["usage"]
                self.ev.setdefault("model", j.get("model"))
                for ch in j.get("choices") or []:
                    d = ch.get("delta") or {}
                    if (d.get("content") or d.get("tool_calls")) and self.ttft is None:
                        self.ttft = time.perf_counter() - self.t0
                    if d.get("content"): self.content.append(d["content"])
                    for tc in d.get("tool_calls") or []:
                        n = (tc.get("function") or {}).get("name")
                        if n: self.tools.append(n)
                    if ch.get("finish_reason"): self.finish = ch["finish_reason"]
        elif self.svc == "tts" and self.ctype.startswith("audio/"):
            if len(self.body_buf) < 64: self.body_buf += chunk[:64]
        elif len(self.body_buf) < 4_000_000:
            self.body_buf += chunk

    def finish_event(self, status, headers, error=None):
        e = self.ev; e["status"] = status; e["latency_ms"] = round((time.perf_counter() - self.t0) * 1000)
        e["bytes_out"] = self.bytes_out
        if error: e["error"] = error
        if self.svc == "llm":
            e["stream"] = self.stream
            if not self.stream:
                try:
                    j = json.loads(self.body_buf or b"{}"); self.usage = j.get("usage"); e.setdefault("model", j.get("model"))
                    ch = (j.get("choices") or [{}])[0]; msg = ch.get("message") or {}
                    self.finish = ch.get("finish_reason"); self.content = [msg.get("content") or ch.get("text") or ""]
                    self.tools = [(t.get("function") or {}).get("name") for t in msg.get("tool_calls") or []]
                    if status >= 400: e["error"] = clip(j.get("error") or j.get("detail") or j)
                except Exception: pass
            else: e["ttft_ms"] = round(self.ttft * 1000) if self.ttft is not None else None
            u = self.usage or {}
            e["tokens"] = {"prompt": u.get("prompt_tokens"), "completion": u.get("completion_tokens"),
                           "reasoning": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                           "total": u.get("total_tokens")}
            if e["latency_ms"] and u.get("completion_tokens"):
                e["tokens_per_s"] = round(u["completion_tokens"] / (e["latency_ms"] / 1000), 1)
            e["finish_reason"] = self.finish; e["tool_calls"] = [t for t in self.tools if t]
            out = "".join(c for c in self.content if c)
            e["output"] = clip(out) if out else None
        elif self.svc == "stt":
            try:
                j = json.loads(self.body_buf or b"{}")
                e["output"] = clip(j.get("text")); e["output_chars"] = len(j.get("text") or "")
                e["audio_seconds"] = (j.get("usage") or {}).get("seconds")
                if status >= 400: e["error"] = clip(j)
            except Exception:
                if status >= 400: e["error"] = clip(bytes(self.body_buf).decode("utf-8", "ignore"))
        elif self.svc == "tts":
            sr = int(headers.get("x-sample-rate") or 0)
            if sr and status < 400:
                pcm = self.bytes_out - (44 if e.get("format") == "wav" else 0)
                e["sample_rate"] = sr; e["audio_seconds"] = round(max(pcm, 0) / 2 / sr, 2)
            if status >= 400: e["error"] = clip(bytes(self.body_buf).decode("utf-8", "ignore"))
        t = f"{PREFIX}/{self.svc}/request/{topic_safe(e.get('call_id'))}/{topic_safe(e.get('turn_id'), '0')}"
        PUB.publish(t, e)

# ---------------------------------------------------------------- proxy
async def proxy(req: Request):
    svc = PORTS.get(req.scope["server"][1]) if req.scope.get("server") else None
    if svc not in UPSTREAMS: return Response("unknown port", status_code=500)
    body = await req.body()
    tracked = req.method == "POST" and req.url.path in TRACKED.get(svc, ())
    tr = Tracker(svc, req, body) if tracked else None
    if tr and svc == "llm" and tr.stream:   # ask vLLM to include token usage in the final stream chunk
        try:
            j = json.loads(body); so = j.get("stream_options") or {}; so["include_usage"] = True
            j["stream_options"] = so; body = json.dumps(j).encode()
        except Exception: pass
    headers = [(k, v) for k, v in req.headers.raw if k.lower() not in HOP]
    if tr: headers.append((b"x-request-id", tr.ev["request_id"].encode()))
    url = UPSTREAMS[svc] + req.url.path + (("?" + req.url.query) if req.url.query else "")
    try:
        up = await client.send(client.build_request(req.method, url, headers=headers, content=body), stream=True)
    except Exception as e:
        if tr: tr.finish_event(502, {}, error=f"upstream unreachable: {type(e).__name__}")
        return Response(f"upstream {svc} unreachable", status_code=502)
    if tr: tr.ctype = up.headers.get("content-type", "")
    out_headers = {k: v for k, v in up.headers.items() if k.lower() not in RESP_DROP}

    async def body_iter():
        err = None
        try:
            async for chunk in up.aiter_raw():
                if tr: tr.feed(chunk)
                yield chunk
        except Exception as e:
            err = f"{type(e).__name__}"; raise
        finally:
            await up.aclose()
            if tr: tr.finish_event(up.status_code, up.headers, error=err)
    return StreamingResponse(body_iter(), status_code=up.status_code, headers=out_headers)

# ---------------------------------------------------------------- status loop
def parse_metrics(text):
    v = {}
    for line in text.splitlines():
        if not line.startswith("vllm:"): continue
        try: name, val = line.rsplit(" ", 1); v[name.split("{")[0]] = v.get(name.split("{")[0], 0) + float(val)
        except ValueError: pass
    g = lambda *ks: next((v[k] for k in ks if k in v), None)
    return {"running": g("vllm:num_requests_running"), "waiting": g("vllm:num_requests_waiting"),
            "kv_cache_usage": g("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
            "prompt_tokens_total": g("vllm:prompt_tokens_total"), "generation_tokens_total": g("vllm:generation_tokens_total"),
            "requests_total": g("vllm:request_success_total")}

async def status_loop():
    while True:
        for svc, base in UPSTREAMS.items():
            ev = {"v": 1, "service": svc, "ts": now_iso()}
            try:
                r = await client.get(base + "/health", timeout=3); ev["health"] = r.status_code
            except Exception: ev["health"] = 0
            ev["state"] = {200: "up", 503: "loading"}.get(ev["health"], "down")
            if svc in ("llm", "stt") and ev["health"] == 200:
                try: ev.update(parse_metrics((await client.get(base + "/metrics", timeout=3)).text))
                except Exception: pass
            PUB.publish(f"{PREFIX}/{svc}/status", ev)
        try:
            p = await asyncio.create_subprocess_exec("nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
                                                     "--format=csv,noheader,nounits", stdout=asyncio.subprocess.PIPE)
            out, _ = await asyncio.wait_for(p.communicate(), 5)
            mu, mt, ut, te, pw = [x.strip() for x in out.decode().splitlines()[0].split(",")]
            PUB.publish(f"{PREFIX}/gpu/status", {"v": 1, "service": "gpu", "ts": now_iso(), "vram_used_mib": float(mu),
                        "vram_total_mib": float(mt), "util_pct": float(ut), "temp_c": float(te), "power_w": float(pw),
                        "telemetry_sent": PUB.sent, "telemetry_dropped": PUB.dropped})
        except Exception: pass
        await asyncio.sleep(STATUS_SEC)

@contextlib.asynccontextmanager
async def lifespan(app):
    PUB.publish(f"{PREFIX}/gateway/log/info", {"v": 1, "service": "gateway", "ts": now_iso(), "event": "started",
                "ports": PORTS, "upstreams": UPSTREAMS, "text_mode": TEXT_MODE})
    task = asyncio.create_task(status_loop())
    yield
    task.cancel(); await client.aclose()

app = Starlette(routes=[Route("/{path:path}", proxy, methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])],
                lifespan=lifespan)

def main():
    host = env("GW_HOST", "0.0.0.0"); socks = []
    for port in PORTS:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port)); socks.append(s)
    print(f"[gateway] listening on {host} {PORTS} -> {UPSTREAMS}; telemetry {'Solace ' + env('SOLACE_HOST') if PUB.enabled else 'stdout'} prefix={PREFIX} text={TEXT_MODE}", flush=True)
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", timeout_keep_alive=75))
    asyncio.run(server.serve(sockets=socks))

if __name__ == "__main__":
    main()
