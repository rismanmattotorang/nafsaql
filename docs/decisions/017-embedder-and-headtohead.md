# Decision 017: Keep the 0.6B embedder; close the measured gaps against Semantic Router and DeepSeek Harness (Phase 24)

Date 2026-11-09. Status: **adopted.** Research [`semantic_router_deepseek_h2h.md`](../research/semantic_router_deepseek_h2h.md),
specification [`headtohead_spec.md`](../spec/headtohead_spec.md). Fusion sweeps: `benchmarks/kb_fusion_sweep_embedder_*.json`;
per-embedder ranks: `benchmarks/kb_embedder_0p6b.json`, `kb_embedder_4b.json`. All timings SHARED MACHINE.

## Decisions

1. **The knowledge embedder stays Qwen3-Embedding-0.6B (`vector_weight` 0.1 default).** The 4B model improves vector-only
   retrieval (paraphrased R@1 0.25 → 0.45, quoted 0.63 → 0.73) but the hybrid, which ships, does not improve: the best
   pooled MRR over 69 questions is 0.804 for the 0.6B model at w = 0.2 and 0.787 for the 4B model at w = 0.5, and the 4B
   model costs about four times the embedding time, 2.5 times the storage and a separate index. The 4B entry stays in the
   manifest as an option for a GPU host.
2. **`vector_weight` stays 0.1.** The sweep puts 0.2 slightly ahead on paraphrased questions (MRR 0.576 against 0.510) and
   0.1 ahead or equal on quoted ones (0.909 against 0.897). 20 paraphrased questions cannot separate them (one question is
   0.05 of R@1), so the earlier, larger, pooled choice stands.
3. **The head-to-head is part of the repository,** reproducible with one command, with the caveat that three upstream
   packages compile offline and the rest need downloads that were not approved.
4. **No new gap is closed by claiming it.** The AVX2 dot product is 1.3–1.5× slower than upstream's assembly and stays
   recorded so.
