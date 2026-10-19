# benchmarks

Measured results. Reference machine: Intel Core i3-8100 (4 cores, AVX2, no AVX-512), 32 GB DDR4, macOS 15.4.1, llama.cpp commit 797de63, 4 threads.

`reference-machine/` (2026-10-18)

| File | Content |
|------|---------|
| `llama_bench_*.txt` | `llama-bench` pp512 / tg128, 3 repetitions, for Qwen3-0.6B Q8_0, Qwen3-1.7B Q4_K_M, Qwen3-4B Q4_K_M, Qwen3-4B-Thinking-2507 Q4_K_M |
| `server_probe_qwen3_1.7b.jsonl` | llama-server OpenAI-API probe: `n` cap, parallel sampled log-probabilities, thinking with a reasoning budget |
| `ollama_probe_qwen3_1.7b.jsonl` | Ollama thinking versus non-thinking timing for Qwen3-1.7B |
| `crate_versions.txt` | `cargo search` output used to pin the Rust crates (2026-10-18) |
| `llama_bench_h7_qwen35_vs_qwen3.md` | Idle-machine llama-bench of Qwen3-1.7B vs Qwen3.5-2B and Qwen3-4B vs Qwen3.5-4B (hypothesis H7, gate G0) |
| `vllm_probe_qwen3_1.7b_fp32.txt` | vLLM 0.31 CPU in the Lima VM: decode speed, logprobs, reasoning separation, `thinking_token_budget` |
| `../mvp_acceptance.json` | MVP acceptance tests MVP-A1..A9 (`scripts/mvp_acceptance.py`) |

The four `llama_bench_qwen3_*.txt` files were taken while downloads were running and are lower bounds; `llama_bench_h7_qwen35_vs_qwen3.md` was measured on an idle machine (Qwen3-1.7B decodes at 22.9 tok/s idle versus 15.5 tok/s under load).

Re-run:

```bash
scripts/fetch_models.sh all
third_party/llama.cpp/build/bin/llama-bench -m models/Qwen3-4B-Q4_K_M.gguf -t 4 -p 512 -n 128 -r 3
```

The probes live in `scripts/probes/` (`server_probe.py` expects a llama-server on port 18081 with alias `qwen3-1.7b`).
