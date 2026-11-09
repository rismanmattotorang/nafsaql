# Closing the measured gaps against vLLM Semantic Router and DeepSeek Harness: specification (Phase 24)

Version 1.0, 2026-11-09. Research and measurements: [`semantic_router_deepseek_h2h.md`](../research/semantic_router_deepseek_h2h.md).
Continues [`router_harness_upgrade_spec.md`](router_harness_upgrade_spec.md) (FR-R, FR-H) and
[`knowledge_base_spec.md`](knowledge_base_spec.md) (FR-K). New ids: FR-P (per-request cost), FR-H6, NFR-P.

**Ground rule, unchanged:** four CPU cores, no GPU, no model call added to any request, nothing leaves the machine.

## 1. Strategy

The Phase 22 read found *mechanisms* worth porting; the Phase 24 measurements asked a different question: **on the work
both sides do on every request, who pays more?** The answer sets the order:

1. Fix what is measurably slower on the per-request path (a gap with a number beats a gap with an opinion): the sentence
   packer, the vector dot product, exact search over a base's vectors.
2. Make what already exists visible (the label stream the calibrated gate depends on).
3. Specify, but do not build, what needs a model per request or a surface this project does not have, and say why.
4. Make the comparison itself reproducible, so the next upstream change can be re-measured with one command.

## 2. Requirements implemented

- **FR-P1 — an AVX2/FMA dot product.** `nafs_kb::vcache::dot` selects, once per process, an AVX2 + FMA kernel (64 floats a
  step on eight independent accumulators, a horizontal add at the end) when the CPU has both, and `dot_portable`
  otherwise. Both read only `min(a.len(), b.len())` floats. Test: the two agree to 1e-3 relative on pseudo-random
  inputs of 14 lengths from 0 to 1,024 (empty, tail only, one block, several blocks).
  *Measured:* n=1024, 1.9 µs → 199 ns; against their assembly 1.3–1.5× slower at n ≥ 768 (G-A, open).
- **FR-P2 — a packer that does its words once.** `pack_units` computes each sentence's words once and uses them for
  both inverse document frequency and relevance. Output identical (existing tests unchanged).
  *Measured:* 16K tokens to 512, 27.1 ms → 8.96 ms; 2.4–5.6× faster than their `Compress`.
- **FR-P3 — an optional exact vector cache.** `[knowledge] vector_cache_mb` (default 0, off) keeps a normalised copy of a
  base's vectors, loaded in the background document by document; while loading, and for a base over the cap, the store's
  index answers. Results equal the store's exact search (test `the_cached_search_is_exact`). *Measured:* 164–296 s to
  load 46,000 passages, then 18–22 ms a query; the warm HNSW answers in 0.37 s, so the default stays off.
- **FR-P4 — the label stream in the UI.** The Quality page's "How Auto decides" card shows whether the stream is
  practising, how many labelled drafts the ledger holds against the promotion floor, and the share wrong.
- **NFR-P1 — reproducible head-to-head.** `benchmarks/headtohead/run.sh` runs, interleaved for `ROUNDS` rounds, the
  upstream Go benchmarks that compile offline and the Rust counterparts, records the load average per round, and
  `summarise.py` writes the median per pair to `summary.json`. Every row of the research note comes from that file.

## 3. Requirements specified and not built, with the reason

- **FR-H6 — replay.** Every model request of a run can be rebuilt from the run's trace: the trace stores the message list
  as appended (task, actions, observations, notices, prune and compaction markers with the replaced ranges), and
  `nafs-harness replay <run>` prints the request the model saw at step *n* byte for byte. Why not yet: a `Step` records the
  action, the observation and two booleans (`pruned`, `compacted`), not the text that replaced the older messages, so a
  replay after a prune or compaction cannot be exact; the fix is a record per replacement, which is cheap but changes the trace format and must be versioned. Acceptance: for 20
  recorded runs, `replay` at every step equals the logged request hash.
- **FR-R5 — difficulty prior** and **FR-H5 — spill:** see the router/harness spec §5; unchanged.
- **Signal breadth, trained jailbreak/PII classifiers.** Not adopted: a ≥ 0.3B classifier per request costs more than the
  quick model's own answer on four cores (their own measurement: 9.8 s median on 4 cores). Our patterns and structure
  checks stay; a classifier is reconsidered when a GPU is present.
- **Envoy `ext_proc`, Kubernetes operator.** Not adopted: a mesh filter is a different product. The gateway already
  speaks the OpenAI API, which Envoy can route to.
- **Subagents at scale, browser and computer use.** Not adopted: a decode per subagent on this CPU is minutes.

## 4. Embedder decision (G-D)

Qwen3-Embedding-4B (Q4_K_M, 2,560 dimensions) against 0.6B (Q8_0, 1,024), same 122 passages at 1,200 characters, 49
quoted and 20 paraphrased questions, one embedding server each (shared machine):

| | quoted R@1 / MRR (vector only) | paraphrased R@1 / MRR (vector only) | hybrid best weight, pooled MRR |
|---|---|---|---|
| 0.6B | 0.63 / 0.716 | 0.25 / 0.361 | w = 0.2: 0.804 |
| 4B | 0.73 / 0.774 | 0.45 / 0.534 | w = 0.5: 0.787 |

Words alone (w = 0) score 0.909 on quoted and 0.450 on paraphrased questions. The 4B model is the better *vector*
retriever, but in the hybrid, which is what ships, it is not better than the 0.6B one and costs about 4× the time to
embed (≈ 1 passage per 45–60 s under load against seconds), 2.5× the storage per passage, a separate index (the vector
index has one width per database) and a full re-embed. **Not adopted** (decision 017); `config/models.manifest.json`
carries the 4B entry and `scripts/fetch_models.sh embed4b` fetches it, so the choice can be revisited on a GPU host.

## 5. Acceptance

| Id | Test | Result (2026-11-09) |
|---|---|---|
| FR-P1 | `the_fast_dot_agrees_with_the_portable_one`; `h2h` dot rows | pass; 199 ns at n=1024 |
| FR-P2 | `nafs-core` knowledge tests; `h2h` packer rows | pass; 8.96 ms at 16K |
| FR-P3 | `the_cached_search_is_exact`, `odd_inputs_are_handled` | pass |
| FR-P4 | Quality page screenshot, `/v1/gate/labels` | see docs/screenshots |
| NFR-P1 | `ROUNDS=5 benchmarks/headtohead/run.sh` | five rounds, load 24–29, `summary.json` |
| G-A | Rust dot ≤ 1.1× upstream assembly at n ≥ 768 | **not met** (1.3–1.5×); recorded |
