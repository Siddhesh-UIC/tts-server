#!/bin/bash
# Dashboard on port 3000. Runs outside start.sh's supervision: if it crashes it restarts
# itself here, and it can never take the model servers down.
export PATH=/workspace/bin:$PATH
unset PIP_CONSTRAINT
source /workspace/venvs/gateway/bin/activate
cd /workspace/dashboard
while true; do
  python dashboard.py
  echo "[dashboard] exited with code $?, restarting in 5 s"
  sleep 5
done
