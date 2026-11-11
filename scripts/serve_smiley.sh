#!/usr/bin/env bash
# The Smiley Core backend (Phase 15): the vendored Rust engine behind the gateway's OpenAI-compatible API, one process per
# model, like scripts/serve_llama.sh. Usage: scripts/serve_smiley.sh start|stop|stop-one|status|logs|embed|stop-embed
#   System 1 :8201  Qwen3-1.7B Q4_K_M        System 2 :8202  Qwen3-4B-Thinking-2507 Q4_K_M
#   embed    :8203  Qwen3-Embedding-0.6B Q8_0 (`embed`, Phase 20 Knowledge Base; scripts/fetch_models.sh embed)
# Both servers run on the same four cores as the llama-servers; switch the gateway with POST /v1/system/backend
# {"name": "smiley"} (the System page), which warms the pair first. SLOTS (users served at once, Phase 18),
# THREADS, PREFILL_THREADS, S1_MODEL, S2_MODEL, KV (f16|i8),
# DSPARK (a drafter directory for System 2, greedy requests only) override the defaults. LITTLEBIT=1 runs the
# projections a <model>.gguf.slb sidecar covers from it (experimental: post-training sidecars are not accurate).
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=${SMILEY_BIN:-rust/target/release/nafs-smiley-server}
# 3, not 4: on this shared host decode (memory-bound) is 4-8 % faster with 3 threads (6 of 6 interleaved pairs, benchmarks/engine_profile.json),
# prefill 16-17 % slower, and a core stays free for the verifier sandbox and the gateway. A quiet host may prefer 4: THREADS=4.
THREADS=${THREADS:-3}
# Reading a prompt is bound by arithmetic, not by memory, and scales with every core: with PREFILL_THREADS=4 the quick model
# reads a prompt 23 % faster than with 3 (7 of 8 interleaved pairs, benchmarks/engine_phase27/prefill_threads.json) while
# tokens are still written by THREADS. Default: every core. The embedding server only reads, so it uses this width too.
PREFILL_THREADS=${PREFILL_THREADS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}
RUN=run; mkdir -p "$RUN" logs
start_one() { # name port model ctx extra...
  local name=$1 port=$2 model=$3 ctx=$4; shift 4
  if [ -f "$RUN/smiley-$name.pid" ] && kill -0 "$(cat "$RUN/smiley-$name.pid")" 2>/dev/null; then echo "smiley $name already running (pid $(cat $RUN/smiley-$name.pid))"; return; fi
  [ -x "$BIN" ] || (cd rust && cargo build --release -p nafs-smiley-server)
  [ -f "$model" ] || { echo "missing $model: run scripts/fetch_models.sh"; exit 1; }
  nohup "$BIN" -m "$model" --alias "$name" -t "$THREADS" --prefill-threads "$PREFILL_THREADS" -c "$ctx" --port "$port" --kv "${KV:-f16}" --slots "${SLOTS:-4}" ${LITTLEBIT:+--littlebit} "$@" > "logs/smiley-$name.log" 2>&1 &
  echo $! > "$RUN/smiley-$name.pid"; echo "started smiley $name on :$port (pid $!)"
}
case "${1:-status}" in
  start)
    S1=${S1_MODEL:-models/Qwen3-1.7B-Q4_K_M.gguf}; S2=${S2_MODEL:-models/Qwen3-4B-Thinking-2507-Q4_K_M.gguf}
    start_one system1 8201 "$S1" 8192
    if [ -n "${DSPARK:-}" ]; then start_one system2 8202 "$S2" 8192 --dspark "$DSPARK"; else start_one system2 8202 "$S2" 8192; fi
    for p in 8201 8202; do for i in $(seq 1 120); do curl -sf "localhost:$p/health" >/dev/null && { echo ":$p healthy"; break; }; sleep 1; done; done ;;
  embed) # The Knowledge Base's embedding server: /v1/embeddings, no chat. EMBED_MODEL, EMBED_PORT and EMBED_THREADS override.
    E=${EMBED_MODEL:-models/Qwen3-Embedding-0.6B-Q8_0.gguf}
    if [ -f "$RUN/smiley-embed.pid" ] && kill -0 "$(cat "$RUN/smiley-embed.pid")" 2>/dev/null; then echo "smiley embed already running (pid $(cat $RUN/smiley-embed.pid))"; exit 0; fi
    [ -x "$BIN" ] || (cd rust && cargo build --release -p nafs-smiley-server)
    [ -f "$E" ] || { echo "missing $E: run scripts/fetch_models.sh embed"; exit 1; }
    nohup "$BIN" -m "$E" --alias embed -t "${EMBED_THREADS:-$PREFILL_THREADS}" --port "${EMBED_PORT:-8203}" --embedding > logs/smiley-embed.log 2>&1 &
    echo $! > "$RUN/smiley-embed.pid"; echo "started smiley embed on :8203 (pid $!)"
    for i in $(seq 1 60); do curl -sf localhost:8203/health >/dev/null && { echo ":8203 healthy"; break; }; sleep 1; done ;;
  stop-embed) [ -f "$RUN/smiley-embed.pid" ] && kill "$(cat $RUN/smiley-embed.pid)" 2>/dev/null && echo "stopped smiley embed"; rm -f "$RUN/smiley-embed.pid" ;;
  stop)  for n in system1 system2 embed; do [ -f "$RUN/smiley-$n.pid" ] && kill "$(cat $RUN/smiley-$n.pid)" 2>/dev/null && echo "stopped smiley $n"; rm -f "$RUN/smiley-$n.pid"; done ;;
  stop-one) n=${2:?usage: stop-one system1|system2}; [ -f "$RUN/smiley-$n.pid" ] && kill "$(cat $RUN/smiley-$n.pid)" 2>/dev/null && echo "stopped smiley $n"; rm -f "$RUN/smiley-$n.pid" ;;
  status) for p in 8201 8202 8203; do printf ":%s " $p; curl -sf "localhost:$p/health" || echo "down"; echo; done ;;
  logs) tail -n 40 logs/smiley-system1.log logs/smiley-system2.log ;;
  *) echo "usage: $0 start|stop|stop-one <name>|status|logs|embed|stop-embed"; exit 2 ;;
esac
