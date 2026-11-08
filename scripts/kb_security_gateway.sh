#!/usr/bin/env bash
# A throw-away gateway for scripts/kb_security_sweep.py: port 8093, its own SurrealDB database, ledger and files, and small
# limits so every refusal can be provoked cheaply. It shares the engines and SurrealDB server of the main stack, which must
# be running. Usage: scripts/kb_security_gateway.sh start|stop|status
set -euo pipefail
cd "$(dirname "$0")/.."
D=run/kb-security; BIN=rust/target/release/nafs-gateway; PID=$D/gateway.pid
case "${1:-status}" in
  start)
    mkdir -p $D/files
    cp data/runtime/verifier_rates.json $D/ 2>/dev/null || true
    python3 - <<'PY'
import re
s = open("config/nafs-aql.toml").read()
def sub(key, val, quote=True):
    global s
    new = f'"{val}"' if quote else str(val)
    s, n = re.subn(rf'(?m)^({key}\s*=\s*)("[^"]*"|\d+)', lambda m: m.group(1) + new, s, count=1)
    assert n == 1, key
sub("bind", "127.0.0.1:8093")
for k, v in [("calibration_file", "gate_calibration.json"), ("params_file", "gate_params.json"), ("rates_file", "verifier_rates.json"),
             ("sqlite_path", "nafs.sqlite"), ("trace_path", "trace.jsonl")]:
    sub(k, "run/kb-security/" + v)
sub("database", "kb_security")
sub("files_dir", "run/kb-security/files")
for k, v in [("max_upload_mb", 2), ("max_request_mb", 3), ("max_documents_per_base", 5), ("max_chunks_per_document", 60),
             ("concurrent_uploads", 1), ("parse_timeout_s", 120)]:
    sub(k, v, quote=False)
open("run/kb-security/nafs-aql.toml", "w").write(s)
PY
    if [ -f $PID ] && kill -0 "$(cat $PID)" 2>/dev/null; then echo "already running (pid $(cat $PID))"; exit 0; fi
    nohup "$BIN" --config $D/nafs-aql.toml > $D/gateway.log 2>&1 & echo $! > $PID
    for i in $(seq 1 40); do curl -sf localhost:8093/healthz >/dev/null && break; sleep 0.5; done
    echo "security-sweep gateway on http://127.0.0.1:8093 (pid $(cat $PID))" ;;
  stop)
    if [ -f $PID ]; then p=$(cat $PID); if ps -p "$p" -o command= 2>/dev/null | grep -q "run/kb-security/nafs-aql.toml"; then kill "$p" && echo stopped; fi; rm -f $PID; fi ;;
  status) curl -s localhost:8093/healthz || echo "down"; echo ;;
  *) echo "usage: $0 start|stop|status"; exit 2 ;;
esac
