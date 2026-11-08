# What this machine cannot do, and the plan for each (Phase 23)

Written 2026-11-08 from what the Phase 20 to 23 measurements showed. The reference machine is a 4-core Intel i3-8100
with no GPU, shared with other work (load average 6 to 60 while these were measured). Four things are blocked by it, not
by code. For each: what is blocked, the evidence, what to run where, what it costs, and what result would settle it.

## 1. The gate's promotion floor: 2,900 labelled drafts (no new hardware)

**Blocked.** The calibrated gate cannot be promoted until the ledger holds 2,900 held-out labelled samples (NFR-020).
It holds 203 (`benchmarks/gate_probe_run_*.json`): the fit is well calibrated (held-out ECE 0.024) but at this size the
fixed rule is the better rule (it keeps 78.2 % of drafts at 10.1 % wrong; the certified bins keep 54.5 % at 9.9 %, or
5.4 % at 12.4 % at the default `max_risk`; `benchmarks/gate_report.txt`). Bounds of about 100 drafts per bin are too wide.

**Plan.** Switch on the label stream (`[labels] enabled = true`, FR-R6). While no user request is in flight it answers
GSM8K and MBPP train tasks with the quick model alone and lets the exact verifier judge each draft. The measured cost of
one labelled draft was a median 64 s under load (p90 125 s); alone it should be about 20 s. 2,900 drafts are therefore
16 to 32 hours of idle time (`per_hour = 90` caps it at about 32). `recalibrate_every_h = 24` refits each day and
promotes only if the held-out tests pass; `GET /v1/gate/labels` shows the count and the bins as they grow.

**Then measure** (the end-to-end test that could not be run honestly at n = 203): `gate_report` on the full ledger for the
held-out comparison, then a paired `nafs_blind` evaluation, fixed against calibrated on 100 fresh GSM8K test items
(`POST /v1/gate` to switch mode between arms; arms interleaved per item). **Settled when** the calibrated rule keeps at
least as many drafts as the fixed rule at no more wrong, or the data say it cannot, in which case the fixed rule stays
and A-136 becomes a result.

## 2. Release-scale evaluation (a quiet host, or ten nights)

**Blocked.** Gate G2 needs 700 paired items, the baseline 4,600 tasks, and each paired item costs a quick draft and a
reasoning run (minutes each on this CPU, hours under load); the measured deliberate path decodes at about 2 tokens a second
while the host is shared.

**Plan.** Either a quiet host with four or more fast cores, or nightly slices here: `POST /v1/eval/runs` takes `offset`
and `max_items`, so 70 pairs a night finishes G2 in ten nights, resuming after any restart. Label every result
`SHARED MACHINE`. **Settled when** the gates dashboard (`GET /v1/gates`) computes G2 from stored runs.

## 3. Memorilla's general reading ability: a GPU pre-train

**Blocked.** A module learns to read memory in two stages on about 3.3 million examples before it is fine-tuned on a base
(`training/memorilla/GPU_RECIPE.md`). On this CPU an example costs 30 to 90 s, so the stages are years.

**Evidence from this round.** Fine-tuned from scratch on one base (234 training examples, 66 passages, one epoch of the
CPU run) the module's validation loss fell from 8.77 to 1.46, but:

- in the f32 trainer, claim accuracy with the module is 0.55 and without it 0.55 (20 claims): no knowledge was read;
- served on the quantised model, `memory` mode answers 0 of 20 and writes degenerate text (`0000…`, `None None…`), where
  closed-book also scores 0 of 20 but fluently;
- `hybrid` (memory tokens and two passages) answers 17 of 20 against `rag`'s 18 of 20 with four, so the memory tokens add
  nothing that was measured (the control that separates them, `rag@2`, is in `scripts/kb_modes_ab.py`).

That is what the paper predicts for a module with no pre-training. It does not show that the pre-trained module would
fail, and it does not show that it would help.

**Plan.** Rent about 10 H100-hours (an estimate: 4 GPUs for 2 to 3 hours with Qwen3-1.7B as decoder), run Stages 1 and 2
from the recipe with our decoder and encoder, convert with `convert_checkpoint.py`, and fine-tune per base here from
`--init`. **Settled when**, on the paper bases, claim accuracy with memory exceeds claim accuracy without it by at least 15
points in the f32 trainer, `memory`-mode answers are fluent on the quantised model, and `hybrid` beats `rag@2`. If the
f32 gain appears and the quantised answers are still degenerate, the defect is in serving (the placeholder positions or the
quantisation), and the recipe's warning that the tokens "transfer, but quality should be checked on the quantised model"
becomes the work item.

## 4. Speed studies that need an idle host

**Blocked and waiting** (STATUS, "Known issues"): NFR-007 concurrency (2.15 over 1.25 against a loaded host), EAGLE-3 and
repacked weights, speculative decoding and the engine's flag matrix, the 100-prompt run, and every speed number labelled
`SHARED MACHINE`. **Plan.** One session on an idle machine with `scripts/perf_tuning.sh`, `scripts/eagle3_ab.py`,
`scripts/concurrency_ab.py` and `scripts/live_sweep.sh`, with `uptime` recorded before each. Nothing here needs code.

## Order

1 first, because it needs no purchase and runs while the machine is otherwise idle; 4 whenever an idle hour exists; 2 as
nightly slices once 1 is running; 3 when a GPU budget is approved.
