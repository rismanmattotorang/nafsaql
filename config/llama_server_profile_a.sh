#!/usr/bin/env bash
# Sample llama-server launch for Profile A on the reference machine (MR-003). Tested 2026-10-18 with llama.cpp 797de63.
# Both servers expose the OpenAI-compatible API consumed by the Rust gateway (IF-006).
set -euo pipefail
MODELS=${MODELS:-./models}
BIN=${LLAMA_BIN:-llama-server}
THREADS=${THREADS:-4}            # physical cores (i3-8100: 4, no SMT)
K=${K:-4}                        # cheap samples for the gate: llama-server caps `n` by the slot count, so -np >= K
# System 1: Qwen3-1.7B, thinking off per request via chat_template_kwargs {"enable_thinking": false}
$BIN -m "$MODELS/Qwen3-1.7B-Q4_K_M.gguf" --alias system1 -t "$THREADS" -c 8192 -np "$K" \
     --port 8001 --host 127.0.0.1 --jinja --reasoning-format deepseek --cache-reuse 256 \
     --spec-type ngram-simple --metrics &
# System 2: Qwen3-4B-Thinking-2507, reasoning budget enforced server-side (FR-024); gateway may also force the answer
$BIN -m "$MODELS/Qwen3-4B-Thinking-2507-Q4_K_M.gguf" --alias system2 -t "$THREADS" -c 8192 -np 1 \
     --port 8002 --host 127.0.0.1 --jinja --reasoning-format deepseek --reasoning-budget 1024 \
     --cache-reuse 256 --spec-type ngram-simple --metrics &
wait
