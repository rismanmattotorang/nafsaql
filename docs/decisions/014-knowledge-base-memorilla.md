# Decision 014: Knowledge Base on SurrealDB, GraphRAG, and Memorilla memory tokens (Phase 20)

Date 2026-11-07. Status: **adopted; Phases 0–2 implemented, 3–8 planned.** Research note:
[`docs/research/memorilla_knowledge.md`](../research/memorilla_knowledge.md). Specification:
[`docs/spec/knowledge_base_spec.md`](../spec/knowledge_base_spec.md). Prompt: [`docs/prompts/MEMORILLA_KB_PROMPT.md`](../prompts/MEMORILLA_KB_PROMPT.md).

## Context

The owner asked for a knowledge base (upload PDF, DOCX, TEX, PPTX, HTML and more; store as vectors and a graph in
SurrealDB), RAG and GraphRAG, and Memorilla (Stanford, arXiv 2610.21375), a learned memory that hands a frozen LLM 16
query-dependent soft tokens read from a whole document collection, trained and fine-tuned on uploaded knowledge and
served by Smiley Core, implemented with Candle.

## Decision

| Question | Decision | Why |
|---|---|---|
| Where does SurrealDB run? | Its own server process (3.3.5, `tools/surrealdb`, surrealkv storage, 127.0.0.1:8300), managed by `scripts/serve_surreal.sh`; the gateway calls its RPC over HTTP through `nafs-kb` | No new heavy dependency tree in the gateway; same process model as the engines; restartable on its own; verified spike |
| Wire format | SurrealDB RPC over HTTP in **CBOR**, bound variables only | its JSON decoding turns strings like `"Caption: Model ..."` into record ids (found in Phase 1, regression-tested) |
| Which version? | 3.3.5, not 4.0.0 | 4.0.0 is 4 days old, changes BM25 scoring and needs a one-way store upgrade; revisit after a few weeks |
| Parsers | `pdf_oxide` (PDF, pure Rust), our own ZIP + XML readers for DOCX and PPTX, `htmd` for HTML, our own LaTeX stripper, plain reads for Markdown and text | pure Rust, auditable, bounded against hostile archives; no PDFium, no Java, no Python |
| Chunking | `text-splitter` Markdown splitter, 1,200 characters (about 300 tokens), 150 overlap, heading path per chunk | semantic boundaries; Memorilla's benchmarks use 13–280-token chunks |
| Embeddings | Qwen3-Embedding-0.6B (1024-d): Smiley Core with a pooling head for serving, Candle as reference and for training | same family as the paper's encoder; `ort` no longer builds for Intel macOS, which rules out fastembed's ONNX models here |
| Graph | entities and relations extracted by System 1 under a JSON Schema, resolved by name and vector, Leiden communities; LightRAG-style local search first | uses the engines we have; one model call per answer |
| Memorilla in Rust | a new crate `nafs-memorilla` on Candle 0.11 for the module, encoder reference, dataset builder and trainer; Smiley Core gets one marked addition (caller-supplied input vectors for chosen rows) | Smiley Core's P1 principle forbids a tensor framework inside the engine |
| Released checkpoints | used only for the parity test | they project into Qwen3-8B's 4,096-wide space; ours are 2,048 (System 1) and 2,560 (System 2) |
| Training on this CPU | fine-tuning on a knowledge base, System 1 decoder, `retrieval_init`; Stage 1–2 pre-training is a GPU recipe | about 1.4 TFLOP per example through Qwen3-1.7B; 3.3 M pre-training rows would take months here |
| Chat integration | a knowledge mode per request: `off`, `rag`, `graph`, `memory`, `hybrid`, `auto`; retrieved material delimited as data; replies carry sources | the paper shows memory and retrieval answer different questions |

## Consequences

- Operators start one more service (`scripts/serve_surreal.sh start`; `make up` does it when `[knowledge].enabled`).
- SurrealDB is BSL 1.1 (self-hosting allowed); it is recorded in NOTICE as a separately downloaded binary, not
  redistributed.
- The knowledge API returns 503 `kb_unavailable` when SurrealDB is down; chat is unaffected.
- A Memorilla module trained here is tied to System 1's Qwen3-1.7B; changing System 1's model means retraining.

## Evidence so far

- `scripts/serve_surreal.sh spike` on the reference machine: HNSW KNN, BM25 (`@1,OR@`) and a two-hop graph traversal
  return the expected records (2026-11-07).
- Memorilla's parameter formula reproduces the paper's 194,123,776; ours is 31.5 M (System 1).
- Phase 1 ingestion timings: `benchmarks/kb_ingest.json`.
- Phase 2 (2026-11-07): Smiley Core's Q8_0 embeddings agree with the Candle f32 reference at cosine >= 0.9991; the Candle
  reference reproduces the model card's scores exactly. Hybrid search beats words and meaning alone (recall@5 0.90 vs
  0.85 and 0.55 on 20 questions), so it is the default; chat keeps passages with similarity >= 0.45. Indexing ran at 69
  tokens/s on a host at load 44–60 (`benchmarks/kb_retrieval.json`).
