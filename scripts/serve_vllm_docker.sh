#!/usr/bin/env bash
# vLLM CPU backend in Docker (Profile A on Linux containers; P7-T10). Serves Qwen3-1.7B in FP32 from the host's
# Hugging Face weights in the carstenz/vllm-smoke image (vLLM 0.30, x86 CPU build, offline), on :8101, as the
# gateway's "vllm" backend set expects (config/nafs-aql.toml: one model in two modes). FP32: on this AVX2-only VM
# oneDNN has no BF16 kernels and the BF16 torch fallback never returns a token (A-050); FP32 is 6.8 GB of weights.
#   scripts/serve_vllm_docker.sh start|stop|status|logs|wait
# Stand-in on a host without 10 GiB for the VM: VLLM_MODEL_DIR=models/hf/Qwen3-0.6B VLLM_SERVED_NAME=qwen3-0.6b (set the
# same name in the vllm backend set of config/nafs-aql.toml).
# Memory: vLLM's three processes cost about 3 GB beyond the weights. Qwen3-1.7B FP32 (6.8 GB) therefore needs a Docker VM
# of 11-12 GiB (it was OOM-killed at 10 GiB); the 0.6B stand-in needs 7.5 GiB with VLLM_CTX=1024 and a 256 MiB KV cache.
# Set Docker Desktop > Resources > Memory; on a host short of memory the VM itself may fail to start above that.
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE=${VLLM_IMAGE:-carstenz/vllm-smoke:0.1.0-cpu}
NAME=nafs-vllm
MODEL_DIR=${VLLM_MODEL_DIR:-$PWD/models/hf/Qwen3-1.7B}
SERVED=${VLLM_SERVED_NAME:-qwen3-1.7b}   # the model name the gateway's vllm backend set expects
PORT=${VLLM_PORT:-8101}
case "${1:-status}" in
  start)
    [ -f "$MODEL_DIR/config.json" ] || { echo "missing $MODEL_DIR (scripts/fetch_models.sh hf)"; exit 1; }
    docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "image $IMAGE not found"; exit 1; }
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    docker run -d --name "$NAME" -p "127.0.0.1:$PORT:$PORT" -v "$MODEL_DIR:/models/served:ro" \
      -e VLLM_TARGET_DEVICE=cpu -e VLLM_CPU_OMP_THREADS_BIND=auto \
      -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 --shm-size 1g --entrypoint /opt/venv/bin/python "$IMAGE" \
      -m vllm.entrypoints.openai.api_server --model /models/served --served-model-name "$SERVED" \
      --host 0.0.0.0 --port "$PORT" --dtype "${VLLM_DTYPE:-float32}" --kv-cache-memory-bytes "${VLLM_KV_BYTES:-268435456}" \
      --max-model-len "${VLLM_CTX:-1024}" --max-num-seqs "${VLLM_SEQS:-2}" --max-num-batched-tokens "${VLLM_BATCH:-256}" \
      --enable-prefix-caching --reasoning-parser qwen3 --max-logprobs 5 --enforce-eager >/dev/null
    echo "started $NAME on :$PORT (docker logs $NAME); run '$0 wait' for readiness" ;;
  wait)
    for i in $(seq 1 360); do
      curl -sf "localhost:$PORT/v1/models" >/dev/null 2>&1 && { echo "ready"; exit 0; }
      docker ps --format '{{.Names}}' | grep -q "^$NAME$" || { echo "container exited:"; docker logs "$NAME" 2>&1 | tail -20; exit 1; }
      sleep 5
    done; echo "not ready after 30 min"; exit 1 ;;
  stop) docker rm -f "$NAME" >/dev/null 2>&1 && echo "stopped $NAME" || echo "$NAME not running" ;;
  status) docker ps --filter "name=^$NAME$" --format '{{.Names}} {{.Status}}'; curl -sf "localhost:$PORT/v1/models" >/dev/null 2>&1 && echo ":$PORT serving" || echo ":$PORT not serving" ;;
  logs) docker logs --tail 60 "$NAME" ;;
  *) echo "usage: $0 start|wait|stop|status|logs"; exit 2 ;;
esac
