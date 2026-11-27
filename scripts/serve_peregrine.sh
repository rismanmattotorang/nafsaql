#!/usr/bin/env bash
# The Peregrine backend (Phase 39): RantAI's CPU inference engine (github.com/RantAI-dev/peregrine, Apache-2.0, checked out in
# third_party/peregrine) behind the gateway's OpenAI-compatible API, one process per model like scripts/serve_smiley.sh, so
# either role can be moved to it on its own. Usage: scripts/serve_peregrine.sh start|start-one <role>|stop|stop-one <role>|status|logs|embed|stop-embed|build
#   System 1 :8211  Qwen3-1.7B Q4_K_M        System 2 :8212  Qwen3-4B-Thinking-2507 Q4_K_M
#   embed    :8213  Qwen3-Embedding-0.6B Q8_0 (the Knowledge Base's /v1/embeddings; point [knowledge] embed_url at it)
# Switch the gateway with POST /v1/system/backend {"name": "peregrine"} or per role {"system1": "peregrine", "system2": "smiley"}
# (the System page). THREADS (default 3, as for Smiley Core, so the A/B is like for like), SLOTS (users at once, default 4),
# KV (f16|f32|i8), CTX, S1_MODEL, S2_MODEL and PEREGRINE_BIN override the defaults.
set -euo pipefail
cd "$(dirname "$0")/.."
SRC=third_party/peregrine
BIN=${PEREGRINE_BIN:-$SRC/target/release/peregrine}
THREADS=${THREADS:-3}
RUN=run; mkdir -p "$RUN" logs
build() {
  [ -d "$SRC" ] || { echo "missing $SRC: git clone https://github.com/RantAI-dev/peregrine $SRC && git -C $SRC checkout 194bfb7"; exit 1; }
  (cd "$SRC" && cargo build --release -p peregrine)
}
start_one() { # name port model ctx extra...
  local name=$1 port=$2 model=$3 ctx=$4; shift 4
  if [ -f "$RUN/peregrine-$name.pid" ] && kill -0 "$(cat "$RUN/peregrine-$name.pid")" 2>/dev/null; then echo "peregrine $name already running (pid $(cat $RUN/peregrine-$name.pid))"; return; fi
  [ -x "$BIN" ] || build
  [ -f "$model" ] || { echo "missing $model: run scripts/fetch_models.sh"; exit 1; }
  nohup "$BIN" serve -m "$model" --alias "$name" -t "$THREADS" -c "$ctx" --port "$port" --kv "${KV:-f16}" "$@" > "logs/peregrine-$name.log" 2>&1 &
  echo $! > "$RUN/peregrine-$name.pid"; echo "started peregrine $name on :$port (pid $!)"
}
wait_up() { for p in "$@"; do for i in $(seq 1 180); do curl -sf "localhost:$p/health" >/dev/null && { echo ":$p healthy"; break; }; sleep 1; done; done; }
S1=${S1_MODEL:-models/Qwen3-1.7B-Q4_K_M.gguf}; S2=${S2_MODEL:-models/Qwen3-4B-Thinking-2507-Q4_K_M.gguf}
case "${1:-status}" in
  build) build ;;
  start)
    start_one system1 8211 "$S1" "${CTX:-8192}" --slots "${SLOTS:-4}"
    start_one system2 8212 "$S2" "${CTX:-8192}" --slots "${SLOTS:-4}"
    wait_up 8211 8212 ;;
  start-one)
    case "${2:?usage: start-one system1|system2}" in
      system1) start_one system1 8211 "$S1" "${CTX:-8192}" --slots "${SLOTS:-4}"; wait_up 8211 ;;
      system2) start_one system2 8212 "$S2" "${CTX:-8192}" --slots "${SLOTS:-4}"; wait_up 8212 ;;
      *) echo "unknown role $2"; exit 2 ;;
    esac ;;
  embed)
    E=${EMBED_MODEL:-models/Qwen3-Embedding-0.6B-Q8_0.gguf}
    start_one embed "${EMBED_PORT:-8213}" "$E" 8192 --slots "${SLOTS:-4}"
    wait_up "${EMBED_PORT:-8213}" ;;
  stop-embed) [ -f "$RUN/peregrine-embed.pid" ] && kill "$(cat $RUN/peregrine-embed.pid)" 2>/dev/null && echo "stopped peregrine embed"; rm -f "$RUN/peregrine-embed.pid" ;;
  stop)  for n in system1 system2 embed; do [ -f "$RUN/peregrine-$n.pid" ] && kill "$(cat $RUN/peregrine-$n.pid)" 2>/dev/null && echo "stopped peregrine $n"; rm -f "$RUN/peregrine-$n.pid"; done ;;
  stop-one) n=${2:?usage: stop-one system1|system2|embed}; [ -f "$RUN/peregrine-$n.pid" ] && kill "$(cat $RUN/peregrine-$n.pid)" 2>/dev/null && echo "stopped peregrine $n"; rm -f "$RUN/peregrine-$n.pid" ;;
  status) for p in 8211 8212 8213; do printf ":%s " $p; curl -sf "localhost:$p/health" || echo "down"; echo; done ;;
  logs) tail -n 40 logs/peregrine-system1.log logs/peregrine-system2.log ;;
  *) echo "usage: $0 build|start|start-one <role>|stop|stop-one <role>|status|logs|embed|stop-embed"; exit 2 ;;
esac
