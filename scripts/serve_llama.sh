#!/usr/bin/env bash
# Native CPU serving with llama-server (Profile A). Usage: scripts/serve_llama.sh start|stop|status|logs
# System 1 :8001  Qwen3-1.7B Q4_K_M, thinking off per request; 4 slots (batched decode: 4 sessions, or a draft + k=2 samples)
# System 2 :8002  Qwen3-4B-Thinking-2507 Q4_K_M; per-request thinking_budget_tokens (budget forcing)
# EAGLE=1 adds EAGLE-3 speculative heads to both systems (P7-T04); EAGLE_N sets the draft length (default 4).
# SPEC=1 adds speculative decoding to System 2 with a Qwen3-0.6B Q8_0 draft (scripts/fetch_models.sh draft).
#   Opt-in: its speed-up on this CPU is not yet measured on an idle machine (scripts/perf_tuning.sh spec).
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=${LLAMA_BIN:-third_party/llama.cpp/build/bin/llama-server}
THREADS=${THREADS:-4}
RUN=run; mkdir -p "$RUN" logs
start_one() { # name port model slots ctx extra...
  local name=$1 port=$2 model=$3 slots=$4 ctx=$5; shift 5
  if [ -f "$RUN/$name.pid" ] && kill -0 "$(cat "$RUN/$name.pid")" 2>/dev/null; then echo "$name already running (pid $(cat $RUN/$name.pid))"; return; fi
  [ -f "$model" ] || { echo "missing $model: run scripts/fetch_models.sh"; exit 1; }
  nohup "$BIN" -m "$model" --alias "$name" -t "$THREADS" -c "$ctx" -np "$slots" --host 127.0.0.1 --port "$port" \
        --jinja --reasoning-format deepseek --cache-reuse 256 --metrics "$@" > "logs/$name.log" 2>&1 &
  echo $! > "$RUN/$name.pid"; echo "started $name on :$port (pid $!)"
}
case "${1:-status}" in
  start)
    # The Tazkiya registry (OR-005) names the active artefacts; S1_MODEL / S2_MODEL override it.
    reg_model() { [ -f config/models.registry.json ] && python3 -I -c "import json,sys; print(json.load(open('config/models.registry.json'))['active'][sys.argv[1]]['gguf'])" "$1" 2>/dev/null || true; }
    S1=${S1_MODEL:-$(reg_model system1)}; S1=${S1:-models/Qwen3-1.7B-Q4_K_M.gguf}
    S2=${S2_MODEL:-$(reg_model system2)}; S2=${S2:-models/Qwen3-4B-Thinking-2507-Q4_K_M.gguf}
    # EAGLE=1 serves each model with its EAGLE-3 speculative head (P7-T04, FR-081): a one-layer drafter that
    # reads the target's hidden states; 4 draft tokens per step measured best on this CPU (scripts/eagle3_ab.py).
    S1_EXTRA=(); S2_EXTRA=()
    if [ "${EAGLE:-0}" = 1 ]; then
      for f in models/Qwen3-1.7B-eagle3-Q8_0.gguf models/Qwen3-4B-eagle3-Q8_0.gguf; do [ -f "$f" ] || { echo "EAGLE=1 needs $f: see docs/roadmap/STATUS.md (P7-T04)"; exit 1; }; done
      S1_EXTRA=(-md models/Qwen3-1.7B-eagle3-Q8_0.gguf --spec-type draft-eagle3 --spec-draft-n-max "${EAGLE_N:-4}" --spec-draft-p-min 0.5)
      S2_EXTRA=(-md models/Qwen3-4B-eagle3-Q8_0.gguf --spec-type draft-eagle3 --spec-draft-n-max "${EAGLE_N:-4}" --spec-draft-p-min 0.5)
    fi
    start_one system1 8001 "$S1" 4 16384 ${S1_EXTRA[@]+"${S1_EXTRA[@]}"}
    if [ "${SPEC:-0}" = 1 ]; then
      [ -f models/Qwen3-0.6B-Q8_0.gguf ] || { echo "SPEC=1 needs models/Qwen3-0.6B-Q8_0.gguf: scripts/fetch_models.sh draft"; exit 1; }
      S2_EXTRA=(-md models/Qwen3-0.6B-Q8_0.gguf --spec-draft-n-max "${SPEC_N:-6}" --spec-draft-n-min 1 --spec-draft-p-min 0.6)
    fi
    start_one system2 8002 "$S2" 2 16384 ${S2_EXTRA[@]+"${S2_EXTRA[@]}"}
    for p in 8001 8002; do for i in $(seq 1 90); do curl -sf "localhost:$p/health" >/dev/null && { echo ":$p healthy"; break; }; sleep 1; done; done ;;
  stop)  for n in system1 system2; do [ -f "$RUN/$n.pid" ] && kill "$(cat $RUN/$n.pid)" 2>/dev/null && echo "stopped $n"; rm -f "$RUN/$n.pid"; done ;;
  stop-one) n=${2:?usage: stop-one system1|system2}; [ -f "$RUN/$n.pid" ] && kill "$(cat $RUN/$n.pid)" 2>/dev/null && echo "stopped $n"; rm -f "$RUN/$n.pid" ;;
  status) for p in 8001 8002; do printf ":%s " $p; curl -sf "localhost:$p/health" || echo "down"; echo; done ;;
  logs) tail -n 40 logs/system1.log logs/system2.log ;;
  *) echo "usage: $0 start|stop|stop-one <name>|status|logs"; exit 2 ;;
esac
