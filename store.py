"""SQLite history for the GPU dashboard: one row of averaged metrics every few seconds,
plus every request event from the telemetry gateway. Standard library only.

Disk use is small: about 5 MB per day of metrics, plus the request events.
Old rows are deleted automatically (see the retention settings in dashboard.py).
"""
import json, math, os, re, sqlite3, threading, time

MAX_COLS = ("running", "waiting")        # per-service columns aggregated with MAX instead of AVG
SAFE = re.compile(r"^[a-z][a-z0-9_]{0,40}$")


class Store:
    def __init__(self, path, sample_days=30, event_days=30):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path, self.sample_days, self.event_days = path, sample_days, event_days
        self.lock = threading.Lock()
        self.db = sqlite3.connect(path, check_same_thread=False, timeout=10)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS samples (ts INTEGER PRIMARY KEY)")
        self.db.execute("""CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, service TEXT, model TEXT, call_id TEXT, turn_id TEXT,
            status INTEGER, latency_ms REAL, ttft_ms REAL, prompt_tokens INTEGER, completion_tokens INTEGER, data TEXT)""")
        self.db.execute("CREATE INDEX IF NOT EXISTS events_ts ON events(ts)")
        self.db.execute("CREATE INDEX IF NOT EXISTS events_call ON events(call_id)")
        self.db.commit()
        self.cols = {r[1] for r in self.db.execute("PRAGMA table_info(samples)")}
        self._pruned = 0.0

    # ------------------------------------------------------------ metrics
    def add_sample(self, ts, values):
        """values: {column_name: float or None}. New column names are added on the fly."""
        values = {k: v for k, v in values.items() if SAFE.match(k)}
        with self.lock:
            for k in values:
                if k not in self.cols:
                    self.db.execute(f"ALTER TABLE samples ADD COLUMN {k} REAL"); self.cols.add(k)
            keys = list(values)
            self.db.execute(f"INSERT OR REPLACE INTO samples (ts{''.join(', ' + k for k in keys)}) VALUES (?{', ?' * len(keys)})",
                            [int(ts)] + [values[k] for k in keys])
            self.db.commit()
            self._maybe_prune()

    def history(self, since, until, points=480):
        """Averages per time bucket so any range returns about `points` rows."""
        bucket = max(5, math.ceil((until - since) / points))
        with self.lock:
            cols = sorted(c for c in self.cols if c != "ts")
            if not cols: return {"bucket": bucket, "t": [], "series": {}}
            agg = ", ".join(f"{'MAX' if c.endswith(MAX_COLS) else 'AVG'}({c})" for c in cols)
            rows = self.db.execute(f"SELECT (ts / ?) * ? AS b, {agg} FROM samples WHERE ts >= ? AND ts <= ? GROUP BY b ORDER BY b",
                                   (bucket, bucket, int(since), int(until))).fetchall()
        out = {"bucket": bucket, "t": [r[0] for r in rows], "series": {}}
        for i, c in enumerate(cols, start=1):
            vals = [None if r[i] is None else round(r[i], 3) for r in rows]
            if any(v is not None for v in vals): out["series"][c] = vals
        return out

    # ------------------------------------------------------------ request events
    def add_event(self, ev):
        tok = ev.get("tokens") or {}
        with self.lock:
            cur = self.db.execute(
                "INSERT INTO events (ts, service, model, call_id, turn_id, status, latency_ms, ttft_ms, prompt_tokens, completion_tokens, data)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (time.time(), ev.get("service"), ev.get("model"), ev.get("call_id"), ev.get("turn_id"), ev.get("status"),
                 ev.get("latency_ms"), ev.get("ttft_ms"), tok.get("prompt"), tok.get("completion"), json.dumps(ev, ensure_ascii=False)))
            self.db.commit()
            return cur.lastrowid

    def events(self, before=None, limit=200, q=None, since=None):
        sql, args = "SELECT id, data FROM events WHERE 1=1", []
        if before: sql += " AND id < ?"; args.append(int(before))
        if since: sql += " AND ts >= ?"; args.append(float(since))
        if q: sql += " AND data LIKE ?"; args.append(f"%{q}%")
        sql += " ORDER BY id DESC LIMIT ?"; args.append(max(1, min(int(limit), 1000)))
        with self.lock:
            rows = self.db.execute(sql, args).fetchall()
        out = []
        for i, d in reversed(rows):
            try: ev = json.loads(d)
            except Exception: continue
            ev["_seq"] = i; out.append(ev)
        return out

    def last_event_id(self):
        with self.lock:
            return self.db.execute("SELECT COALESCE(MAX(id), 0) FROM events").fetchone()[0]

    # ------------------------------------------------------------ housekeeping
    def _maybe_prune(self):          # called with the lock held
        now = time.time()
        if now - self._pruned < 3600: return
        self._pruned = now
        self.db.execute("DELETE FROM samples WHERE ts < ?", (int(now - self.sample_days * 86400),))
        self.db.execute("DELETE FROM events WHERE ts < ?", (now - self.event_days * 86400,))
        self.db.commit()

    def size_mb(self):
        try: return round(sum(os.path.getsize(self.path + s) for s in ("", "-wal") if os.path.exists(self.path + s)) / 2**20, 1)
        except OSError: return None
