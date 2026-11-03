#!/usr/bin/env bash
# The Smiley Core backend (Phase 15): the vendored Rust engine behind the gateway's OpenAI-compatible API, one process per
# model, like scripts/serve_llama.sh. Usage: scripts/serve_smiley.sh start|stop|stop-one|status|logs
#   System 1 :8201  Qwen3-1.7B Q4_K_M        System 2 :8202  Qwen3-4B-Thinking-2507 Q4_K_M
# Both servers run on the same four cores as the llama-servers; switch the gateway with POST /v1/system/backend
# {"name": "smiley"} (the System page), which warms the pair first. SLOTS (users served at once, Phase 18),
# THREADS, S1_MODEL, S2_MODEL, KV (f16|i8),
# DSPARK (a drafter directory for System 2, greedy requests only) override the defaults.
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=${SMILEY_BIN:-rust/target/release/nafs-smiley-server}
# 3, not 4: on this shared host decode (memory-bound) is 4-8 % faster with 3 threads (6 of 6 interleaved pairs, benchmarks/engine_profile.json),
# prefill 16-17 % slower, and a core stays free for the verifier sandbox and the gateway. A quiet host may prefer 4: THREADS=4.
THREADS=${THREADS:-3}
RUN=run; mkdir -p "$RUN" logs
start_one() { # name port model ctx extra...
  local name=$1 port=$2 model=$3 ctx=$4; shift 4
  if [ -f "$RUN/smiley-$name.pid" ] && kill -0 "$(cat "$RUN/smiley-$name.pid")" 2>/dev/null; then echo "smiley $name already running (pid $(cat $RUN/smiley-$name.pid))"; return; fi
  [ -x "$BIN" ] || (cd rust && cargo build --release -p nafs-smiley-server)
  [ -f "$model" ] || { echo "missing $model: run scripts/fetch_models.sh"; exit 1; }
  nohup "$BIN" -m "$model" --alias "$name" -t "$THREADS" -c "$ctx" --port "$port" --kv "${KV:-f16}" --slots "${SLOTS:-4}" "$@" > "logs/smiley-$name.log" 2>&1 &
  echo $! > "$RUN/smiley-$name.pid"; echo "started smiley $name on :$port (pid $!)"
}
case "${1:-status}" in
  start)
    S1=${S1_MODEL:-models/Qwen3-1.7B-Q4_K_M.gguf}; S2=${S2_MODEL:-models/Qwen3-4B-Thinking-2507-Q4_K_M.gguf}
    start_one system1 8201 "$S1" 8192
    if [ -n "${DSPARK:-}" ]; then start_one system2 8202 "$S2" 8192 --dspark "$DSPARK"; else start_one system2 8202 "$S2" 8192; fi
    for p in 8201 8202; do for i in $(seq 1 120); do curl -sf "localhost:$p/health" >/dev/null && { echo ":$p healthy"; break; }; sleep 1; done; done ;;
  stop)  for n in system1 system2; do [ -f "$RUN/smiley-$n.pid" ] && kill "$(cat $RUN/smiley-$n.pid)" 2>/dev/null && echo "stopped smiley $n"; rm -f "$RUN/smiley-$n.pid"; done ;;
  stop-one) n=${2:?usage: stop-one system1|system2}; [ -f "$RUN/smiley-$n.pid" ] && kill "$(cat $RUN/smiley-$n.pid)" 2>/dev/null && echo "stopped smiley $n"; rm -f "$RUN/smiley-$n.pid" ;;
  status) for p in 8201 8202; do printf ":%s " $p; curl -sf "localhost:$p/health" || echo "down"; echo; done ;;
  logs) tail -n 40 logs/smiley-system1.log logs/smiley-system2.log ;;
  *) echo "usage: $0 start|stop|stop-one <name>|status|logs"; exit 2 ;;
esac
