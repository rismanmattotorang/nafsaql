# Decision 018: Search from an in-process index; a graph without a model; settings by a 256-question sweep (Phase 25)

Date 2026-11-10. Status: **adopted.** Research [`knowledge_sota_2026.md`](../research/knowledge_sota_2026.md),
specification [`knowledge_sota_spec.md`](../spec/knowledge_sota_spec.md). Measurements: `benchmarks/kb_phase25/`.
All timings SHARED MACHINE (load 6 to 8 for the Phase 25 runs, 24 to 129 for the baseline).

## Decisions

1. **A search reads memory, not the store.** SurrealDB stays the source of truth; `nafs_kb::index` holds each base's
   passages, words, vectors and term graph per document within `index_memory_mb` (512). Words search 177 ms → 0.086 ms,
   hybrid 740 ms → 152 ms at low load (the store path took 5.3 s under load). The store path stays for bases that do not
   fit and behind `"engine": "store"`.
2. **The index is as accurate as the store path, not more.** Pooled hybrid MRR over 256 questions 0.388 against 0.379;
   no set separates them. The claim made is speed.
3. **Every base gets a graph from its text, with no model call** (terms by rule, passage-term links, under a millisecond
   for 847 passages against 17 to 28 s a passage for the model-built graph). The model-built graph stays as the layer of
   typed, quoted relations a person reads.
4. **Personalized PageRank over that graph is built and off by default.** It changes nothing in the first 8 passages
   on any of the three sets; it raises recall at 20 on reworded questions (0.329 → up to 0.377). It serves `graph` mode
   and waits for a reranker.
5. **`b = 0.5`; reciprocal-rank fusion at 0.1 stays; flat field weights; no feedback terms; `rag_lexical_rescue` stays
   10.0.** Convex fusion, BM25F field weights and RM3-style feedback were each measured and not adopted (spec §6.3).
6. **The embedder keeps the previous text's cache rows** (question embedding 297 ms → 146 ms, identical vectors),
   built because the index left the embedding as 99.9 % of a hybrid search.
7. **A re-read reuses the vectors of unchanged passages.**
8. **No "state of the art" claim.** No public benchmark was downloaded; the reworded-question gap (answer first for
   12 %) is the open problem, and what would close it (a reranker or stronger embedder, questions written per passage,
   public sets to compare on) needs the owner's approval of downloads or hours of model time.

## Not adopted, with the number

| Candidate | Measured | Why not |
|---|---|---|
| Convex-combination fusion (α 0.3) | pooled MRR 0.387 against 0.388 | better on quoted and two-passage, worse on reworded |
| BM25F field weights (0 to 3 on title, heading, context) | within one question of flat on every set | no effect on this base |
| Feedback terms (5 passages, 8 terms) | quoted R@1 0.883 → 0.750 | drifts on questions the words already answer |
| Graph arm in the default search | first 8 unchanged | no gain where the prompt reads |
| An index file per base (FR-K77) | load from the store 0.7 to 1.1 s | a second copy that can disagree, for a second saved |
| ANN library, tantivy, semantic chunking, community summaries, GraphRAG query modes | research note §5 | cost or no gain at this scale |
