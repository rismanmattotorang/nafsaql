# Decision 008: engine knobs, the model switch, and a cascade in the harness (Phase 14)

Date 2026-11-01. Status: **adopted in part**. Research and numbers:
[`docs/research/engine_switching_harness.md`](../research/engine_switching_harness.md). Report: section "Phase 14".
Evidence files are in `benchmarks/` (listed below).

## Context

The owner asked to assess the inference engine, the switch between the two models and the Codex-style harness, to research what
concurrency, parallelism and current methods can still give on the target machine, and to implement what pays. Phases 7, 11
and 13 had measured what changes *what* the engine computes (speculation, formats, batching, prefix caching); the engine's own
knobs, the cost of the switch, and a cascade inside the agent had not been measured. The phase ran on a host shared with
another workload of the owner's (load average 8-67 against about 2 idle), which bounds what the engine numbers can say.

## Decision

| Requirement | What | Default | Evidence |
|---|---|---|---|
| FR-097 | Harness cascade: the first step on System 1 as one `n`-choice request, each candidate judged by the verifier, System 2 only when none passes | **on** (`harness.cascade = true`, `cascade_samples = 4`) | `harness_ab_cascade.json`: 10 tasks interleaved, 10/10 verified in both arms, wall time 1,467 s against 1,987 s (-26 %), faster in 7 of 10 pairs (sign test p = 0.17), 6 of 10 tasks verified by System 1 alone (median 34 s against 85 s on System 2); indicative at n = 10 |
| FR-098 | Retries with Codex's backoff of a model call that fails for a reason other than the run's cap | **on** (`harness.retries = 2`) | unit tests; a dropped stream no longer ends the run as `model_error` |
| FR-099 | Warm-first backend switch: the target pair is primed with every constant prefix before the atomic swap; in-flight requests finish on the old pair | **on** | `POST /v1/system/backend` reports `warmed_prefixes`, `warm_ms`, `in_flight`; cold against warm first token 9.8 s against 1 s (decision 005) |
| (serving) | Engine flags: flash attention, micro-batch, threads, polling, memory locking | **unchanged** | `engine_tuning.json`: inconclusive on the shared host (per-round ratios 0.07x-3.8x; load 8-67); no flag beat the baseline on both tests for either model |
| (serving) | Co-scheduling of the two servers | **unchanged** | `cosched_ab.json`: when both servers decode each keeps 47-65 % of its solo rate and the pair writes 7.1-7.8 tokens/s in every variant (polling, S1 threads); nothing to arbitrate, nothing to freeze |
| (measured) | The switch inside a request (draft to decision, decision to System 2's first token, System 1 streams after the decision) | — | `handoff_latency.json`: decision to System 2's first token 1.1-4.7 s (median 1.5 s), gate 0.8-0.9 s without the rescue, no System 1 slot busy after the decision; nothing to freeze, no prefetch built |

## Not adopted

- **Parallel tool calls and dispatch while streaming** (Codex `tools/parallel.rs`, `handle_output_item_done`): one action per
  reply, and under the grammar the server stops at the end of the action; nothing is left to overlap.
- **Ternary models and table-lookup kernels** (BitNet b1.58, T-MAC): they change the model; no Qwen3 variant exists.
- **Checkpoint-loading switches** (ServerlessLLM): both models stay resident; the switch is routing plus a warm prefix.
- **`--prio`**: refused on macOS (SCHED_FIFO needs privileges); `-ub 128`: 0.6x on both models in the blocked run.

## Re-open conditions

- **Engine flags:** re-run `scripts/engine_tuning.py` on an idle host (load about 2); adopt a flag only when its median over
  rounds beats the baseline on both `pp512` and `tg64@d1024` for the model it serves.
- **Cascade off / on:** a 30-task interleaved run on a quiet host where the cascade loses verified tasks, or where its first step costs more than it saves on the tasks System 1 cannot do; then `cascade = false`, or `cascade_samples = 2`.
- **Co-scheduling:** a host with more cores than the two servers' threads, or a llama.cpp with a working thread priority on macOS; then re-run `scripts/cosched_ab.py`.
