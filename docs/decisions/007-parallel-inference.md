# Decision 007: parallel and concurrent inference on four CPU cores (Phase 13)

Date 2026-10-31. Status: **adopted in part**. Research and numbers: [`docs/research/parallel_inference.md`](../research/parallel_inference.md).
Report: section "Phase 13". Evidence files are in `benchmarks/` (listed below).

## Context

The owner asked to assess inference and gate performance and to use the four cores in parallel. The gate and the verifiers are not
the cost (under 2.5 % of model time in the 100-prompt run); decode is (System 2 thinking 77.6 %, drafts 17.9 %), and six prompts
(failed code and timeouts) used 52 % of it.

## Decision

| Requirement | What | Default | Evidence |
|---|---|---|---|
| FR-095 | draw the always-drawn agreement samples (unverified maths) in the draft's batch, and drop them when the draft is stopped early | **on** | `fuse_ab.json`: -24 % time to decision, 23 of 28 pairs, p = 0.0005; escalation rate not distinguishable (7 and 5 of 28) |
| FR-096 | on a failed exact check, draw 4 System 1 samples in one request and verify each as it finishes before escalating | **on** (`gate.rescue_samples = 4`) | `s1_rescue.json` (6 of 32 failed drafts rescued), `rescue_ab.json` (22 to 20 escalations; break-even 106 s against 250-580 s) |
| (serving) | System 2 with 4 slots, 24,576 tokens of context (6,144 per slot) | **on** | `batch_scaling_*.json`: 2.7x decode at short context, 1.5x at 1,600 tokens; the System 2 queue cost 476 s in the 100-prompt run |
| FR-094 | concurrent hedged rounds on System 2: one future per candidate (generate, patch, verify), completion order, first verified wins, `n`-choice requests, width policy, `repair_rounds`, `hedge_after_failure`, a round deadline | **off** (`deliberation.parallel.enabled = false`) | `parallel_ab.json`: no gain; `batch_scaling_system2_depth*.json`: no width pays at repair depth |
| (backend) | `Backend::stream_n`: `n` choices in one request, each delivered when it ends | used by the rescue and by rounds | `prefill_share.json`: 31.7 s, 124.5 s (four separate), 39.1 s (n = 4) |

## Not adopted

- **Concurrent hedging on System 2.** The batch gain falls from 2.7x to 1.5x between short context and 1,600 tokens, and a repair prompt
  is that deep, so a round of four costs 2.2 to 3.4 single attempts and is no faster than the two serial repairs it replaces; run on the 8 code
  items that reach System 2, the outcomes match the sequential loop and failing items are not faster. Kept for hopeless tasks
  (p of 0.07 or less) and for faster hosts: switch it on, and set `cost_factors`, after measuring them there.
- **n-gram self-speculation** (`ngram_ab_system1.json`): 1.12x at best (code repair), 0.84x at worst (JSON).
- **q8_0 KV cache with flash attention** (`batch_scaling_system2_kvq8_*.json`): no gain at depth.
- **Hogwild!-style shared-cache workers, learned parallel transformations (ParScale), table-lookup low-bit kernels (T-MAC)**:
  need trained models or a different runtime; reported small models struggle to cooperate.

## Defects the measurements found, all fixed

1. Candidates sent as separate requests each prefilled the whole prompt (3.9x for four). Fixed by `n`-choice requests.
2. The budget for the next attempt was sized from the single-stream speed, so a round overran the request cap and ended as HTTP 504. Fixed:
   the budget scales by the cost of the new width, and a round has its own deadline 5 s inside the cap and returns the best candidate.
3. An error in the last candidate of a round discarded the best candidate found so far. Fixed: it is returned as the best unverified one.

## Re-open conditions

- **Hedging on:** a host where `batch_scaling.py --depth 1200` shows a 4-stream gain above about 2.5x (a faster memory system), or items
  where one attempt rarely succeeds (p of 0.07 or less); then run `scripts/parallel_ab.py` on at least 20 items.
- **Rescue samples above 4 or below 3:** `s1_rescue.py` showed nothing rescued by 1 or 2 samples and 19 % by 4.
