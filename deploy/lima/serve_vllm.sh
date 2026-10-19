#!/usr/bin/env bash
# Run INSIDE the nafs-vllm VM: start/stop the vLLM CPU servers (OpenAI-compatible API).
#   start      (default, fits a 32 GB host)  one server :8101 "qwen3-1.7b", Qwen3-1.7B BF16 serving BOTH roles:
#              System 1 = enable_thinking false, System 2 = enable_thinking true + thinking_token_budget (Profile B pattern)
#   start-s1 / start-s2  two-model layout (S2 = Qwen3-4B-Thinking-2507 BF16, ~12 GB more RAM; needs a >= 48 GB host)
# Ports are forwarded to the macOS host by Lima (deploy/lima/nafs-vllm.yaml).
set -euo pipefail
ENV="$HOME/vllm-env"; source "$ENV/bin/activate"
TC=$(ls /usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 2>/dev/null || true)
IOMP=$(cat "$HOME/.iomp_path" 2>/dev/null || true)
export LD_PRELOAD="${TC}${IOMP:+:$IOMP}"
export VLLM_CPU_OMP_THREADS_BIND=${VLLM_CPU_OMP_THREADS_BIND:-0-2}   # leave one core for the API front-ends
export VLLM_CPU_NUM_OF_RESERVED_CPU=1
# First-use torch.compile of new shapes on a 4-core AVX2 CPU can exceed the 300 s default and kill the engine.
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}
# Eager mode (no torch.compile): on a 4-core AVX2 host with 12 GiB the first-request graph compile took >10 min and
# grew the worker to 10.4 GB RSS until the OOM killer stopped it. Set VLLM_COMPILE=1 to re-enable compilation on larger hosts.
EAGER=$([ "${VLLM_COMPILE:-0}" = 1 ] && echo "" || echo "--enforce-eager")
# BF16 needs AVX512-BF16/AMX on CPU; without them oneDNN cannot build the matmul and vLLM falls back to a very slow
# torch path (measured: an 8-token reply did not finish in 10 minutes). Use FP32 (AVX2/FMA oneDNN kernels) instead.
DTYPE=${VLLM_DTYPE:-$(grep -q avx512_bf16 /proc/cpuinfo && echo bfloat16 || echo float32)}
RUN="$HOME/run"; LOG=/nafs-aql/logs; mkdir -p "$RUN"
# Stage weights from the shared /models mount (virtiofs, ~30 MB/s) onto the VM's own disk once; later starts load fast.
stage() { local src=$1 dst="$HOME/models/$(basename "$1")"
  if [ ! -f "$dst/.staged" ]; then echo "staging $(basename "$src") to VM disk (one time)..." >&2; mkdir -p "$dst"; cp -r "$src"/. "$dst"/ && touch "$dst/.staged"; fi
  echo "$dst"; }
start_one() { # name port model_dir kv_gib maxlen extra...
  local name=$1 port=$2 model; model=$(stage "$3"); local kv=$4 maxlen=$5; shift 5
  if [ -f "$RUN/$name.pid" ] && kill -0 "$(cat $RUN/$name.pid)" 2>/dev/null; then echo "$name already running"; return; fi
  VLLM_CPU_KVCACHE_SPACE=$kv nohup vllm serve "$model" --served-model-name "$name" --host 0.0.0.0 --port "$port" \
     --dtype "$DTYPE" --max-model-len "$maxlen" --max-num-seqs 4 --max-num-batched-tokens 2048 --block-size 128 \
     --enable-prefix-caching --max-logprobs 5 --reasoning-parser qwen3 $EAGER "$@" > "$LOG/vllm_$name.log" 2>&1 &
  echo $! > "$RUN/$name.pid"; echo "started $name (pid $!) -> logs/vllm_$name.log"
}
case "${1:-status}" in
  start|start-single) start_one qwen3-1.7b 8101 /models/hf/Qwen3-1.7B 2 6144 \
              --reasoning-config '{"reasoning_start_str": "<think>", "reasoning_end_str": "\n\nI have to give the answer based on my reasoning so far.\n</think>\n\n"}' ;;
  start-s1) start_one system1 8101 /models/hf/Qwen3-1.7B 1 8192 ;;
  start-s2) start_one system2 8102 /models/hf/Qwen3-4B-Thinking-2507 2 8192 \
              --reasoning-config '{"reasoning_start_str": "<think>", "reasoning_end_str": "\n\nI have to give the answer based on my reasoning so far.\n</think>\n\n"}' ;;
  stop) for n in qwen3-1.7b system1 system2; do [ -f "$RUN/$n.pid" ] && kill "$(cat $RUN/$n.pid)" 2>/dev/null && echo "stopped $n"; rm -f "$RUN/$n.pid"; done; pkill -f "vllm serve" || true ;;
  status) for p in 8101 8102; do printf ":%s " $p; curl -sf "localhost:$p/v1/models" | head -c 120 || printf "down"; echo; done ;;
  *) echo "usage: $0 start|start-s1|start-s2|stop|status"; exit 2 ;;
esac
