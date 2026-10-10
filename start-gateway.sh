#!/bin/bash
# Telemetry gateway, run separately from start.sh (Northflank command:
#   -c '/workspace/start-gateway.sh & exec /workspace/start.sh').
# Listens on 7000-7003 and forwards to the models on 8000-8003, recording one event per request.
# Point Northflank's public ports at 7000-7003 to trace outside traffic; the models are untouched.
# If the gateway crashes it restarts here within 2 s; it never restarts the models.
export PATH=/workspace/bin:$PATH
unset PIP_CONSTRAINT
source /workspace/venvs/gateway/bin/activate
export GW_PORTS=${GW_PORTS:-llm:7000,stt:7001,tts:7002,sahabat:7003}
export GW_UPSTREAMS=${GW_UPSTREAMS:-llm:http://127.0.0.1:8000,stt:http://127.0.0.1:8001,tts:http://127.0.0.1:8002,sahabat:http://127.0.0.1:8003}
LOG=/workspace/logs/gateway.log
mkdir -p /workspace/logs
cd /workspace/gateway
while true; do
  echo "[gateway] starting $(date -u +%FT%TZ)" >> "$LOG"
  python gateway.py >> "$LOG" 2>&1
  echo "[gateway] exited with code $?, restarting in 2 s" >> "$LOG"
  sleep 2
done
