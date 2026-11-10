# Documentation index

## All documents

| | |
|---|---|
| [Technical report](report/nafs_aql_report.pdf) | the model, the proofs, the implementation and its measured results |
| [Slides](slides/nafs_aql_slides.pdf) | the same story in 40 slides |
| [Specification](spec/nafs_aql_spec.pdf) | 100 requirements with identifiers, interfaces and a traceability matrix |
| [Roadmap](roadmap/nafs_aql_roadmap.pdf) · [Status](roadmap/STATUS.md) | phases, gates, what is done and what is not |
| [Assumptions](../ASSUMPTIONS.md) | every judgement call and its impact if wrong (A-001 to A-130) |
| [Results write-up](publication/nafs_aql_lab_results.pdf) | numbers generated from stored runs |
| [Quantum consciousness and neuroscience review](research/quantum_neuro_review.md) | Phase 8: what transfers (sequential evidence) and what does not (quantum mechanisms), with proofs and measurements |
| [Parallel and concurrent inference](research/parallel_inference.md) · [Decision 007](decisions/007-parallel-inference.md) | Phase 13: batch scaling at depth, what was adopted (fused samples, System 1 rescue, `n`-choice rounds, 4 slots) and what was measured and left off |
| [Engine knobs, the model switch, the harness cascade](research/engine_switching_harness.md) · [Decision 008](decisions/008-engine-switching-harness.md) | Phase 14: the flag matrix and co-scheduling measured (unchanged), the switch timed, the cascade and retries built and measured, the warm-first backend switch |
| [Smiley Core as the third backend](research/smiley_backend.md) · [Decision 009](decisions/009-smiley-backend.md) | Phase 15: the owner's Rust CPU engine vendored and served behind the gateway; the engines compared alone and the whole service compared on each backend; which one is the default and why |
| [Knowledge base specification](spec/knowledge_base_spec.md) · [Decision 014](decisions/014-knowledge-base-memorilla.md) · [Memorilla and knowledge research](research/memorilla_knowledge.md) | Phase 20: documents into passages, vectors and a graph in SurrealDB; the six knowledge modes; Memorilla memory tokens in Candle and Smiley Core; what was measured and what waits for a GPU |
| [OntoSmith comparison](research/ontosmith_comparison.md) · [Graph RAG upgrade spec](spec/graph_rag_upgrade_spec.md) | Phase 21: OntoSmith read in full against the knowledge base, the twelve gaps, the nine closed (graph as a fused arm, contextual BM25, sentence packer, ontology as data, verified quotes, two-hop walk) and the three deferred |
| [Knowledge hardening and retrieval at scale](decisions/016-knowledge-hardening-phase8.md) · [What this machine cannot do](roadmap/machine_bound_plan.md) | Phase 23: the eight hardening gaps, the retrieval measurements (fusion weight, query expansion, passage size, scale), the three open measurements, and the plan for what needs another machine |
| [Knowledge module: state of the art and assessment](research/knowledge_sota_2026.md) · [Specification](spec/knowledge_sota_spec.md) · [Decision 018](decisions/018-knowledge-index-phase25.md) | Phase 25: the in-process index (words 177 ms to 0.086 ms), a graph of every base with no model call, a 256-question sweep with what it adopted and what it rejected, the interface audit, and the open reworded-question gap |
| [Head to head with Semantic Router and DeepSeek Harness](research/semantic_router_deepseek_h2h.md) · [Specification](spec/headtohead_spec.md) · [Decision 017](decisions/017-embedder-and-headtohead.md) | Phase 24: what each project is, a measured comparison of the same work on both sides, the gaps closed and the ones left open, and why a larger embedder was not adopted |
| [Semantic Router and DeepSeek Harness comparison](research/router_harness_comparison.md) · [Router and harness upgrade spec](spec/router_harness_upgrade_spec.md) · [Decision 015](decisions/015-router-harness-phase22.md) | Phase 22: where the gate and the harness were already ahead, the pure mechanisms copied (certified per-bin risk, a declared unknown, reask, polarity guard, anchored meter, prune-then-compact, repeat notice, two-lane ingestion), and what was not |
| [Smiley Core 06c2b88 and LittleBit](research/smiley_core_update.md) · [Decision 013](decisions/013-smiley-core-littlebit.md) | Phase 19: the merge, the two defects it exposed (a crash and a compiler-dependent 4× slowdown), LittleBit measured |
| [Serving many users from Smiley Core](research/smiley_multiuser.md) · [Decision 012](decisions/012-multiuser-engine.md) | Phase 18: what vLLM does, what applies on four cores, the batched multi-user engine and what it measures |
| [Smiley Core's speed, switching and the harness's cost](research/smiley_engine_perf.md) · [Decision 011](decisions/011-engine-switching-costs.md) | Phase 17: the engine against the machine's bandwidth, threads, the prefix store, the exact continuation, the handoff |
| [The code agent against modern harnesses](research/harness_gaps.md) · [Decision 010](decisions/010-harness-gaps.md) | Phase 16: what current harnesses and papers do, the gaps in Nafs-Aql's, what was built and measured, and what was not adopted and why |
| [End-to-end code review](research/code_review_2026-11.md) | Nov 2026: what four independent reviews found across the crates, scripts, deployment files and CI, what was fixed and tested, and what was left and why |
| [Math and code harness](research/harness_integration.md) · [Decision 005](decisions/005-harness-integration.md) | Phase 10: what was taken from Open Math Harness and Codex, what was not, and why |
| [Codex-style harness](research/harness_integration.md#5-phase-11-the-codex-agent-harness-adapted-to-small-models-on-a-cpu) · [Decision 006](decisions/006-codex-harness.md) | Phase 11: the Codex agent-core mechanisms ported, the measured A/B against the Phase 10 loop, and the defaults it set |
| [User guide](user_guide.md) · [Runbook](operations/operator_runbook.md) · [API reference](api/api_reference.md) | using, operating and integrating |

