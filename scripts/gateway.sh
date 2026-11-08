#!/usr/bin/env bash
# Start/stop the Rust gateway (UI on http://127.0.0.1:8080). Usage: scripts/gateway.sh start|stop|restart|status|logs
# Set NAFS_WEB_DIR=rust/crates/nafs-web/assets to serve the UI from disk while editing it.
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=rust/target/release/nafs-gateway; PID=run/gateway.pid; mkdir -p run logs
case "${1:-status}" in
  start) [ -x "$BIN" ] || (cd rust && cargo build --release -p nafs-gateway)
         # The pid must be a gateway: after a reboot the pid file names whatever was given that number (seen 2026-11-08).
         if [ -f $PID ] && ps -p "$(cat $PID)" -o command= 2>/dev/null | grep -q "nafs-gateway"; then echo "gateway already running (pid $(cat $PID))"; exit 0; fi
         nohup "$BIN" --config "${NAFS_CONFIG:-config/nafs-aql.toml}" > logs/gateway.log 2>&1 & echo $! > $PID
         for i in $(seq 1 30); do curl -sf localhost:8080/healthz >/dev/null && break; sleep 0.5; done
         echo "gateway on http://127.0.0.1:8080 (pid $(cat $PID))"; curl -s localhost:8080/healthz; echo ;;
  stop) # Only a process that is our gateway is signalled: the pid file's, then an orphan still serving :8080.
        is_gateway() { ps -p "$1" -o command= 2>/dev/null | grep -q "nafs-gateway"; }
        if [ -f $PID ]; then p=$(cat $PID); if is_gateway "$p"; then kill "$p" 2>/dev/null && echo "gateway stopped"; fi; rm -f $PID; fi
        p=$(lsof -ti tcp:8080 -sTCP:LISTEN 2>/dev/null | head -1 || true)
        if [ -n "$p" ] && is_gateway "$p"; then kill "$p" 2>/dev/null && echo "stopped the gateway still serving :8080 (pid $p)"; fi ;;
  restart) "$0" stop; sleep 1; "$0" start ;;
  status) curl -s localhost:8080/healthz || echo "gateway down"; echo ;;
  logs) tail -n 60 logs/gateway.log ;;
  *) echo "usage: $0 start|stop|restart|status|logs"; exit 2 ;;
esac
