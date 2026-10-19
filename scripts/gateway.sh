#!/usr/bin/env bash
# Start/stop the Rust gateway (UI on http://127.0.0.1:8080). Usage: scripts/gateway.sh start|stop|restart|status|logs
# Set NAFS_WEB_DIR=rust/crates/nafs-web/assets to serve the UI from disk while editing it.
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=rust/target/release/nafs-gateway; PID=run/gateway.pid; mkdir -p run logs
case "${1:-status}" in
  start) [ -x "$BIN" ] || (cd rust && cargo build --release -p nafs-gateway)
         if [ -f $PID ] && kill -0 "$(cat $PID)" 2>/dev/null; then echo "gateway already running (pid $(cat $PID))"; exit 0; fi
         nohup "$BIN" --config "${NAFS_CONFIG:-config/nafs-aql.toml}" > logs/gateway.log 2>&1 & echo $! > $PID
         for i in $(seq 1 30); do curl -sf localhost:8080/healthz >/dev/null && break; sleep 0.5; done
         echo "gateway on http://127.0.0.1:8080 (pid $(cat $PID))"; curl -s localhost:8080/healthz; echo ;;
  stop) [ -f $PID ] && kill "$(cat $PID)" 2>/dev/null && echo "gateway stopped"; rm -f $PID; pkill -f "target/release/nafs-gateway" 2>/dev/null || true ;;
  restart) "$0" stop; sleep 1; "$0" start ;;
  status) curl -s localhost:8080/healthz || echo "gateway down"; echo ;;
  logs) tail -n 60 logs/gateway.log ;;
  *) echo "usage: $0 start|stop|restart|status|logs"; exit 2 ;;
esac
