# Decision 015: Certified gate risk, a declared unknown, reask, and a two-phase harness (Phase 22)

Date 2026-11-08. Status: **adopted; FR-R1 to FR-R4, FR-H1 to FR-H4 and FR-K61 implemented; FR-R5 and FR-H5
specified.** Research: [`docs/research/router_harness_comparison.md`](../research/router_harness_comparison.md).
Specification: [`docs/spec/router_harness_upgrade_spec.md`](../spec/router_harness_upgrade_spec.md).

## Context

The owner asked for vLLM Semantic Router (`third_party/semantic-router`, Go) and DeepSeek Harness
(`third_party/deepseek-harness`, TypeScript) to be read against our gate and our agent harness, the gaps closed, and
Nafs-Aql left superior in Rust. Both were read in full. The finding has the same shape as the OntoSmith read of
Phase 21: we are ahead on what decides correctness (an expected-loss gate with a priced λ, exact verifiers whose
verdict is final, a stall guard, rollback, a sandbox with the network denied), and behind on a handful of pure
mechanisms that cost no model call.

## Decision

| Question | Decision | Why |
|---|---|---|
| How is a calibrated draft kept? | Only in a held-out ê₁ bin whose one-sided Clopper–Pearson upper bound at α/B (α = 0.05, B = 5) is ≤ `[gate] max_risk` (0.15); otherwise `rule: uncertified_bin`. Promotion requires at least one certified bin. | ECE bounds the average; the per-bin bound is stronger and our verifiers supply the error counts for free. The bins are stored with the calibration and served by `GET /v1/gate/calibration`. |
| A draft without log-probabilities? | `[gate] on_unknown` = `escalate` / `keep` / `fixed`; the decision's rule is `unknown_<policy>` and the explanation names the missing feature. | A missing feature was a silent default; Semantic Router's three-valued rules declare it. |
| How is a reask detected? | Jaccard overlap of content words with the previous user turn, ≥ `[gate] reask_overlap` (0.6); a reask of an unverified draft escalates under the fixed rule and the feature is logged. | Lexical, not cosine: the gate runs without the embedder and must stay at its 0.08 s. |
| Retrieval memory and opposite questions? | A hit whose question differs from the query by ≤ 2 content words with a negation cue or an antonym on one side only is dropped (`polarity_conflict`). | "Is caching enabled" must not retrieve the stored answer to "is caching disabled". |
| A difficulty prior from prototype banks (FR-R5)? | Specified, deferred. | Needs a labelled example set per domain and a measured gain; adding a feature invalidates stored calibrations. |
| The harness's transcript meter? | Anchored on the server's reported `prompt_tokens` plus bytes/3 of what was appended since; a prune or a compaction clears the anchor. | Bytes/4 over the whole transcript compacted a step early or late; prefill is the cost on a CPU. |
| Compaction? | Two phases: older observations over `prune_over` bytes (default 1,200: a 600-byte head and a 200-byte tail; never the task, never the newest exchange, never an action apart from its observation) are cut; only if still over `compact_at_tokens` is the transcript compacted. The step records `pruned`. **Measured inert** (16 paired runs at the default: no step pruned, 8 of 16 solved in each arm, identical actions; at 600 bytes 2 of 16 runs pruned, still identical): the model is never shown more than about 930 bytes of evidence. Kept on as insurance against a long `run` output; no gain claimed. | One long test log was thought to take the whole transcript down with it; the measurement says the evidence is capped before it gets there. |
| A reply cut at the token budget? | Its patch or whole file is refused with an instruction to send a shorter edit; a `run` is still read. | A half patch was applied as if whole. |
| The same action again? | A third identical action draws a gentle notice, a fifth a detailed one naming it; added after the stall guard takes the step's signature. | The stall guard reacts to the same failure; this names the same attempt before it fails, and must not hide the failure from the guard (a test caught exactly that). |
| Spill long observations to a file (FR-H5)? | Specified, deferred. | The action vocabulary has no `read`; a spilled file needs one, which is a prompt change to measure. |
| Ingestion under load? | Two lanes: parse/chunk/store fast and `ready` at once; vectors and the graph on a slow lane, one job at a time. | The live sweep found a 187-byte upload waiting five minutes behind a 517-passage re-index. |

## What we did not copy, and why

A ≥ 0.3B model as the per-request gate (79 ms to 9.8 s on a CPU by their own measurement); model-judged cascades that
serve the last response rather than the best; hand-weighted min-max utilities without a price of latency;
model-claimed completion and unbounded loops; a plugin architecture whose composition bugs fill two of four
postmortems; a session log uploaded to the provider. Each is either a model call we cannot afford on four cores or a
weaker rule than the one we have.

## Consequences

- Configuration gains `[gate] max_risk`, `risk_alpha`, `on_unknown`, `reask_overlap`, `reask_escalates`, each with a
  default, so an older file still loads. A calibration fitted before Phase 22 has no bins and is not promoted until
  refit (`POST /v1/gate/calibrate`).
- `GateRule` gains `uncertified_bin`, `unknown_escalate`, `unknown_keep`; `GateFeatures` gains `reask`; a harness step
  gains `pruned`. All are additive in the API.
- Assumptions A-127 to A-130. Unit tests for each mechanism; the harness suite stays at 90 tests passing.
