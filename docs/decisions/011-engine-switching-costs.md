# Decision 011: Smiley Core's threads, a prefix store for switching, and an exact continuation (Phase 17)

Date 2026-11-02. Status: **adopted.** Research note: [`docs/research/smiley_engine_perf.md`](../research/smiley_engine_perf.md). Report: section "Phase 17".

## Context

The owner asked for an assessment of Smiley Core's speed, of the switch between the two models, and of the harness's cost, with research on
concurrency and parallelism on this machine and on current methods. The measurements (shared host, interleaved) found the engine within
60-95 % of the machine's memory bandwidth, so the gains were not in more parallelism but in three places where work was repeated or waited on.

## Decision

| What | Where | Default |
|---|---|---|
| **Three decode threads** instead of four | `scripts/serve_smiley.sh` (`THREADS`), the systemd units, the launchd plists | 3 |
| **A prefix store**: finished conversations' caches kept as snapshots and restored for a prompt that shares a long beginning (`smiley_core::prefix`, `KvCache::snapshot/restore`, `--prefix-cache-mb`) | `smiley-core`, `nafs-smiley-server` | 768 MB per model |
| **An exact continuation** after the reasoning budget: the assistant prefix repeats the generated text token for token (`continuation_prefix`) | `nafs-smiley-server` | always |
| The server reports the engine's phase split and the store's statistics in `timings.smiley` | `nafs-smiley-server` | always |

## Evidence

- **Threads**: decode at three threads beat four in six of six interleaved pairs (+4 % System 1, +8 % System 2 at the median); cold prefill is 16-17 %
  slower at three, and most prompts are now cache hits while decode dominates a request. The memory ceiling is 19.5 GB/s at four threads and 18.4 at three.
- **Prefix store**: a chat, agent or arithmetic prompt after a different kind cost 5.8 s (System 1) and 14 s (System 2) of prefill, against 0.1 s on
  llama.cpp's slots; it now costs 0.1 s and 0.2 s. The answers are byte-identical with the store on and off (`switch_cost.py --verify`).
- **Continuation**: the old prefix had a second newline the model never wrote, so the cache stopped matching at the first reasoning token and every reasoning
  token was prefilled again (108 tokens and 2.2 s at a budget of 200; 7-30 s on System 2). Now 5 tokens and 0.2 s. On six escalated requests the
  cold prefill fell from 700-1,160 tokens (45-78 s) to 350-390 (22-24 s) and the mean request time from 182 s to 133 s.
- **Switching**: the System 1 to System 2 handoff is 0.08 s of gate time, 0.4-1.4 s to the first reasoning token (median 0.85 s), and System 1 is idle at the
  decision: nothing to freeze, because each model is its own process with its own weights. This is at or below llama.cpp's 1.5 s.

## Not adopted

- **Speculative sampling** with prompt-lookup drafts: lossless at any temperature but engine work, and System 2's time is reasoning text that does not copy the
  prompt; named as the next engine item.
- **Per-phase thread width**: changes the engine's most delicate primitive for a gain the store already mostly provides.
- **More prefetch, spin or sampler tuning**: inside the host's noise (prefetch 2048/4096 +2-3 % at the median; sampling is 0.1-0.3 % of decode).

## Re-open conditions

- A quiet host (no other workloads): re-run `scripts/engine_profile.py`; four threads may win (the memory ceiling is 6 % higher), set `THREADS=4`.
- Speculative sampling lands, or the workload becomes dominated by code edits that copy the prompt: re-measure prompt-lookup drafting.
