# The gate and the agent harness: the gaps against vLLM Semantic Router and DeepSeek Harness, and the specification that closes them (Phase 22)

Version 1.0, 2026-11-08. Research: [`router_harness_comparison.md`](../research/router_harness_comparison.md).
Requirement ids: FR-R (the gate, "router"), FR-H (the harness); they continue the FR-0xx gate requirements of
`nafs_aql_spec.pdf` and the FR-09x harness requirements of `harness_integration.md`.

**The ground rule**, as in every phase: affordable on four CPU cores with no GPU. Nothing here adds a model call to
a request. Both projects measured their own gates at 79 ms to 9.8 s per request on a CPU; ours stays at the 0.08 s
it costs today.

## 1. The gap table

| # | Gap | Evidence | Closes in |
|---|---|---|---|
| **R-1** | The calibrated gate is promoted on an *average* bound (held-out ECE ≤ 0.05); nothing certifies the error rate of the bin a draft actually falls in | Semantic Router's System One cascade accepts only in a bin whose one-sided Clopper–Pearson upper bound at α/#bins is ≤ `max_risk`, with the artefact pinned by hash | FR-R1 |
| **R-2** | A missing gate feature (no log-probabilities from a backend, no samples) is silently a default; the decision does not say so | Their three-valued rules: an errored leaf is *unknown*, and the decision declares `on_unknown: match / no_match / fail_request` | FR-R2 |
| **R-3** | A question asked again after an answer is treated like a new question | Their `reask` signal: cosine ≥ 0.8 to the previous user turn, a prior that the last answer failed | FR-R3 |
| **R-4** | A retrieval-memory hit can be a near-identical question with the opposite polarity ("is X enabled" / "is X disabled") | Their cache polarity guard: ≤ 2-token difference with a negation cue or an antonym pair on one side only | FR-R4 |
| **R-5** | The error estimate has no prior from the question's own difficulty | Their contrastive prototype banks: a hard bank and an easy bank of example questions, scored `0.75·best + 0.25·mean(top-2)`, margin → hard / medium / easy | FR-R5 (specified) |
| **H-1** | The transcript's size is a byte estimate; compaction fires a step early or late | `dsh` anchors on the server's reported `prompt_tokens` of the last call plus a heuristic delta | FR-H1 |
| **H-2** | Compaction is one step: the transcript is replaced by a task message and a summary | `dsh` prunes long tool results first (8,192 → 4,096 head + 1,024 tail), summarises only if still over 0.8 × window, never splits a call from its result, rejects a summary that is not smaller | FR-H2 |
| **H-3** | A reply cut at the token budget is parsed as if it were whole | `dsh` drops every tool call from a message that ended on `max-tokens` | FR-H3 |
| **H-4** | The stall guard reacts to the same *failure*; nothing names the same *attempt* before it fails | `dsh`'s repeat-tool chain: thresholds 3 / 5 / 8, gentle then detailed, naming the call | FR-H4 |
| **H-5** | An observation's middle is gone once truncated | `dsh` spills a long result to a file and shows a head-and-tail preview with the path | FR-H5 (specified) |
| **K-1** | One ingestion queue: a 187-byte upload waited five minutes behind a 517-passage re-index, and the graph build behind both | Found by the live sweep of this phase | FR-K61 |

## 2. The specification — the gate

- **FR-R1 — certified per-bin risk.** The fitted calibration keeps, beside the isotonic map, the held-out
  reliability bins of ê₁ with their counts `n_b` and errors `k_b`, and for each bin the one-sided Clopper–Pearson
  upper bound on the error rate at `α / B` (Bonferroni over the `B` pre-declared bins; `α = 0.05`): the smallest
  `u` with `I_u(k+1, n−k) ≥ 1 − α/B`, found by bisection on the regularised incomplete beta; for `k = 0`,
  `u = 1 − (α/B)^{1/n}`. A bin is **certified** when `u ≤ [gate] max_risk`. In calibrated mode (not budget mode,
  which fixes the escalation rate by design) a draft is kept only if its ê₁ falls in a certified bin; a draft whose bin is uncertified is escalated, and the decision record
  says `rule: uncertified_bin` with the bound. The certified bins are stored with the calibration and reported by
  `GET /v1/gate/calibration`. Promotion keeps the ECE test and adds: at least one certified bin, or the mode
  stays fixed.
- **FR-R2 — a declared unknown.** When the draft has no log-probabilities (a backend without them) or the samples
  it needs are absent, the gate does not substitute a default silently: `[gate] on_unknown` is `escalate`
  (default), `keep` or `fixed` (fall back to the fixed rule on what is present), the decision's `rule` is
  `unknown_<policy>`, and the explanation names the missing feature.
- **FR-R3 — reask.** The orchestrator compares the user's message with the previous user turn of the same
  conversation by the Jaccard overlap of their content words; at `[gate] reask_overlap` (default 0.6) or above the
  turn is a reask. A reask of an unverified draft escalates under the fixed rule (`[gate] reask_escalates`,
  default on), the feature is recorded on the decision, and it is logged so the next calibration can learn its
  weight. Lexical rather than cosine because the gate runs in `nafs-core` without the embedder and the cost must
  stay at nothing.
- **FR-R4 — the polarity guard on retrieval memory.** A retrieval-memory hit is dropped when its question's token
  set differs from the query's by at most two tokens and a negation cue (`not, no, never, without, cannot, n't`)
  or a swapped antonym pair (`enable/disable, on/off, add/remove, allow/deny, start/stop, open/close, true/false,
  before/after, more/less, increase/decrease`) is on one side only. Pure function, tested.
- **FR-R5 — difficulty prior (specified).** Two prototype banks of example questions per domain, embedded once by
  the knowledge embedder, clustered at cosine 0.9 to at most 8 medoids; the margin `score(hard) − score(easy)` with
  `score = 0.75·best + 0.25·mean(top-2)` becomes a gate feature. Deferred: it needs a labelled example set per
  domain and a measured gain on the calibration before it enters the feature vector, which would invalidate
  stored calibrations.

## 3. The specification — the harness

- **FR-H1 — an anchored token meter.** After a reply, the transcript records the server's `prompt_tokens` and the
  number of messages it held; the estimate is that count plus bytes/3 of what was appended since, and a
  compaction or a prune clears the anchor. Implemented: `Transcript::anchor`.
- **FR-H2 — prune, then compact.** When the transcript is over `compact_at_tokens`, older observations (every user
  message but the task and the newest exchange) longer than 1,200 bytes are cut to a 600-byte head and a 200-byte
  tail with a marker saying how much was left out; only if the transcript is still over the budget is it compacted
  into the task message. The newest exchange is never cut, and a prune never separates an action from its
  observation. Implemented: `Transcript::prune_older`; the step records `pruned`.
- **FR-H3 — no edit from a cut reply.** A patch or a whole file in a reply whose `finish_reason` is `length` is
  refused with an instruction to send a shorter edit; a `run` is still read. Implemented in `parse_allowed`.
- **FR-H4 — the repeat notice.** The same action text sent a third time draws a gentle notice, a fifth time a
  detailed one naming the action's first line; the notice is added after the stall guard has taken the step's
  signature, so it never hides an identical failure from it. Implemented: `repeat_notice`.
- **FR-H5 — spill (specified).** An observation over `max_observation_bytes` is kept whole in the run's scratch
  directory and the model sees a head, a tail and the path. Deferred: the action vocabulary has no *read*; the
  model reads only by running Python, so a spilled file needs either a `read` action or a `run cat` convention,
  and that is a prompt change to measure, not a mechanism.

## 4. The specification — ingestion

- **FR-K61 — two lanes.** Parsing, chunking and storing run on a fast lane and set the document `ready` (words
  searchable at once); vectors and the graph queue on a slow lane, one job at a time (NFR-K06 unchanged). A new
  upload never waits behind another base's embedding or extraction. Implemented.

## 5. What Phase 22 implements, and what it defers

Implemented: FR-R1, FR-R2, FR-R3, FR-R4, FR-H1 to FR-H4, FR-K61. Specified and deferred, with the reason: FR-R5
(a labelled example set and a calibration measurement first), FR-H5 (a read action first).

## 6. Acceptance

1. The calibration reports certified bins with their bounds, and a synthetic calibration with one bad bin keeps
   drafts only in the good ones (unit test).
2. A draft with no log-probabilities produces a decision whose rule is `unknown_<policy>` and whose explanation
   names the feature (unit test).
3. A re-asked question is recorded as such and, unverified, escalates under the fixed rule (unit test).
4. "Is caching enabled?" does not retrieve a stored answer to "Is caching disabled?" (unit test).
5. The harness tests of FR-H1 to FR-H4 pass; a long older observation is pruned before any compaction, and the
   newest exchange is never cut.
6. A small upload reaches `ready` while a large re-index runs (the knowledge sweep's ingest check passes under
   load).
7. `fmt`, `clippy -D warnings`, `cargo deny`, the Rust suite and the live sweeps are clean.
