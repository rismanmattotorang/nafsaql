# NAFS-AQL: KNOWLEDGE BASE, GRAPHRAG AND MEMORILLA PROMPT (v1)

Revised on 2026-11-07 from the owner's request so that it matches Nafs-Aql as it is built today (after Phase 19, commit
`8dc2df4`). Paste it into Claude Code at the repo root. The project-level name of this work is **Phase 20 (Knowledge)**;
inside this prompt its steps are called Phase 0 to Phase 8.

What changed from the original request, and why:

| Original request | Revised | Reason |
|---|---|---|
| "Memorilia" | **Memorilla** (Stanford SNAP, arXiv 2610.21375v3, `third_party/memorilla`) | the name of the paper, code and checkpoints |
| Integrate the trained model into LLM serving | Into **Smiley Core**, the default engine, through a minimal input-embedding hook; llama.cpp and vLLM are not changed | Smiley Core is the only engine we own the forward pass of |
| Implement Memorilla in Candle | Candle for the memory module, its training and the embedding encoder, in a **new crate**; Smiley Core keeps its one dependency | Smiley Core's principle P1 forbids a tensor framework inside the engine (see its `Cargo.toml`) |
| Train and fine-tune Memorilla on uploaded knowledge | **Fine-tune on this CPU** from a checkpoint; **pre-training (Stages 1 and 2) is a GPU job** with a recipe and a cost estimate | backprop through the frozen 1.7B decoder costs about 1.4 TFLOP per example; Stage 1 alone is 1.98 M examples |
| Use the released checkpoints | Only to **check the port** (parity with PyTorch). Serving needs a module trained for our decoder | every released checkpoint projects into Qwen3-8B (4,096 wide); System 1 is Qwen3-1.7B (2,048) and System 2 Qwen3-4B (2,560) |
| Install SurrealDB in the project | The **`surreal` server binary** (v3.3.5) in `tools/surrealdb/`, run as its own process like the engines; the gateway talks to it over HTTP | the embedded `surrealdb` crate pulls RocksDB and hundreds of crates into the gateway's build; one more server matches how the engines are run |
| Use RAG, GraphRAG and Memorilla | A **knowledge mode** per request (`off`, `rag`, `graph`, `memory`, `hybrid`), routed and measured like everything else; retrieved text enters the prompt **as data** | Memorilla and RAG answer different questions (paper, Figure 5); the existing retrieval memory (FR-061) already delimits retrieved text as data |

---

## 0. HOW YOU WORK

Same rules as `docs/prompts/COMPLETION_PROMPT.md`, section 0, plus these:

- **Checkpoint file:** `docs/status/PROGRESS.md`, a section "Phase 20 (Knowledge)". Update it after every unit of work.
- **Verify before you claim.** Every speed or quality claim needs a run on the reference machine, stored under
  `benchmarks/`, with interleaved A/B where two variants are compared. Label every run made while the owner's other
  workloads were busy as *shared machine*.
- **The live stack is the owner's.** The gateway (:8080) and both Smiley Core engines (:8201, :8202) may be running. Never
  `pkill -f 'port 82'`; kill test servers by saved PID; start test copies on other ports.
- **Ask before downloading model weights** (state the file, source and size). The SurrealDB binary is already approved.
- **Plain language in the UI** (`nafs-web` copy-lint test): no requirement ids, no "System 1/2" (say *quick* and
  *thorough*), no model jargon in default mode; details go behind **Advanced**.
- **Rust first.** No Python at run time. Python is allowed only for parity checks against the reference implementation
  and for the GPU pre-training recipe.

## 1. MISSION

Give Nafs-Aql a **Knowledge Base**: a person uploads documents (PDF, DOCX, PPTX, TEX, HTML, Markdown, plain text), Nafs-Aql
turns them into **chunks with vectors** and an **entity–relation graph** in SurrealDB, answers questions with **vector RAG**,
**GraphRAG** or **Memorilla's 16 memory tokens**, and can **train or fine-tune** a Memorilla memory module on that knowledge.
All of it runs on the CPU.

**Objectives, in priority order**
1. **Correct and grounded answers** from uploaded knowledge, with the passages or graph facts that support them shown to
   the user, and an honest *not in the knowledge base* when nothing supports an answer.
2. **Latency that stays usable on four cores**: ingestion runs in the background; a knowledge-grounded answer adds at most
   a few seconds to the quick path.
3. **Memorilla where it pays**: memory tokens instead of retrieved text only when a paired A/B on this machine shows it is
   no worse in quality and cheaper in prompt tokens or time.

**Definition of done (whole track)**
- Upload, index, search, delete and re-index work from the Knowledge page and the API; tests cover each parser.
- Chat can answer from a chosen knowledge base in each knowledge mode; replies carry their sources.
- A Memorilla module in Rust reproduces the PyTorch reference on the released checkpoint (parity test), and one trained
  for System 1 is served through Smiley Core.
- Fine-tuning on a knowledge base runs on this CPU with a measured time per example; Stage 1 and 2 pre-training has a GPU
  recipe and a cost estimate derived from measured per-example times.
- Each knowledge mode's quality and latency is measured on a held-out question set built from real uploaded documents.

## 2. GROUND TRUTH (do not re-derive)

### 2.1 Machine and stack
Intel Core i3-8100, 4 cores, AVX2 only, 32 GB, macOS 15.4.1, no GPU. Rust 1.96 workspace in `rust/` (11 crates).
Default engine **Smiley Core** (`rust/crates/smiley-core`, memmap2-only, GGUF, served by `nafs-smiley-server` on :8201
for System 1 Qwen3-1.7B Q4_K_M and :8202 for System 2 Qwen3-4B-Thinking-2507 Q4_K_M; 3 decode threads; prefix store on).
Gateway `nafs-gateway` (axum 0.8, :8080), state in SQLite (`nafs-memory`), web UI in `nafs-web/assets` (vanilla ES module,
no build step, embedded in the gateway). Existing retrieval memory (FR-061, `[retrieval]`, BM25 over prior verified
answers, off by default) delimits retrieved text as data.

### 2.2 Memorilla, from the paper and `third_party/memorilla`
- **Pipeline.** A frozen encoder embeds each document chunk once (store `Z_D`, N x h_e). Per query, a trainable memory
  module reads the whole store and returns **K = 16 soft tokens** in the decoder's input-embedding space; they replace the
  embeddings of 16 `<|memory|>` placeholders appended to the system prompt (`"\nMemory" + "<|memory|>"*16 + "Memory"`). The
  decoder is not changed; RoPE runs over the combined sequence.
- **Module** (`memorilla/memory.py`): 16 learned slots (Xavier) -> cross-attention to the one question vector (residual +
  LayerNorm) -> optional document self-attention (0 layers by default) -> LayerNorm over documents -> **2 pre-norm blocks**
  of cross-attention (8 heads, slots to documents, key padding mask) + GELU FFN (4x) -> LayerNorm -> Linear(h_e -> h_L).
  `retrieval_init` starts the slots from the top-16 documents by cosine instead (used for FactKG). Dropout 0.1.
- **Size.** 12·h_e² per block dominates. Paper configuration (h_e 2560, h_L 4096): **194,123,776** parameters (checked by
  formula). Ours with Qwen3-Embedding-0.6B (h_e 1024): **31.5 M** for System 1 (h_L 2048), **32.0 M** for System 2 (h_L 2560).
- **Objective.** Next-token loss on the answer tokens only; the decoder and encoder stay frozen, so gradients flow
  *through* the decoder into the module (no weight gradients for the decoder).
- **Curriculum.** Stage 1: 1.98 M Wikipedia paraphrase-reconstruction rows (chunks of about 128 tokens, 1.73 per row),
  lr 3e-4, 1 epoch. Stage 2: 1.34 M rows from 10 QA/summarisation sets, lr 3e-4, 1 epoch. Stage 3: the 8 benchmarks, lr 1e-4,
  5 epochs. AdamW (wd 0.01), linear warm-up then cosine to 10 %, bf16, clip 1.0, effective batch 128 / 128 / 64. About 19 h on
  4 H100s.
- **Data format for new knowledge** (`docs/data.md`): parquet rows `collection_id, question, answer, documents[,
  choices]`; rows sharing a `collection_id` share `documents`; each document entry is one chunk (benchmarks use 13–280
  tokens per chunk); embeddings stored as float16 safetensors (`index`, `documents-NNNNN`, `questions`).
- **Results that shape the design.** Best mean over 8 benchmarks at 16 tokens; FactKG 84.33 % vs 80.43 % full context;
  weaker than retrieval on entity-recall ("who", "which") and short collections (PubMedQA 59 vs 67–69 %); Memorilla and
  RAG solve different questions (15.9 % vs 10.0 % answered only by one on FactKG), so a **hybrid** is the stated future
  work. Larger encoder: at most 1.3 points. K = 128 at training: no gain.
- **Checkpoints** (`huggingface.co/memorilla/Memorilla-Qwen3-8B`): all for Qwen3-8B + Qwen3-Embedding-4B.
- **Encoder pooling.** The paper's Table 3 says "mean-pooled"; the code embeds with vLLM's `embed` task, which uses the
  model's own pooler (last token, normalised) and no instruction prefix for documents or questions. The port follows
  the code, since the checkpoints were trained with it.

### 2.3 SurrealDB
`tools/surrealdb/surreal` **v3.3.5** (darwin-amd64, sha256 of the release tarball
`a274451b30cf639df3b180b0e82ab8f7c52bc5244fa66aa65155a085f3eb6d31`). Verified on this machine on 2026-11-07: `FULLTEXT
ANALYZER ... BM25` indexes with `@n@` and `@n,OR@` matching and `search::score(n)`; `HNSW DIMENSION d DIST COSINE`
indexes with `<|k,ef|>` KNN and `vector::distance::knn()`; `TYPE RELATION` tables, `RELATE`, and multi-hop traversal
(`->mentions->entity->related->entity`). Storage engine `surrealkv://` (pure Rust, no RocksDB). v4.0.0 was released on
2026-11-03; stay on 3.3.x until it has been out for a few weeks, then re-run the spike.

## 3. TARGET ARCHITECTURE

```
 Browser: Knowledge page (upload, documents, search, graph, training)  ·  Chat (knowledge picker, sources)
                       │ HTTP + SSE
 nafs-gateway ── /v1/kb/* ──► nafs-kb ── ingest: parse → clean → chunk → embed → extract entities → store
      │                          │         query: BM25 + vector (RRF) · graph expansion · community summaries
      │                          └── HTTP ──► SurrealDB :8300 (surrealkv, ns nafs / db kb)
      │                                         tables: kb, document, chunk(HNSW, FULLTEXT), entity, mentions, related, community
      ├── orchestrator: knowledge mode → context block (as data) or memory tokens
      │
      ├── nafs-memorilla (Candle): encoder (Qwen3-Embedding-0.6B), memory module, trainer, dataset builder
      │        └── memory tokens [16 x h_L] ──► nafs-smiley-server  POST /v1/chat/completions {memory_embeds}
      └── nafs-smiley-server ──► smiley-core: forward_rows with an embedding override for marked positions
```

Crates added: `nafs-kb` (ingestion, SurrealDB client, retrieval, graph), `nafs-memorilla` (Candle: embedding, memory module,
training). Smiley Core gets one marked `// nafs-aql:` addition: rows whose input vector is supplied instead of looked up.

## 4. PHASES

Each phase ends with tests green, `PROGRESS.md` updated, and a short entry in `docs/roadmap/STATUS.md`. A phase that
compares variants stores its A/B under `benchmarks/`.

### Phase 0: Research, design, specification, SurrealDB
1. Research note `docs/research/memorilla_knowledge.md`: Memorilla's architecture read from code and paper; how to prepare
   datasets from uploaded documents; training and fine-tuning costs on this CPU (formula plus estimate, to be measured in
   Phase 6); the Candle port plan; Smiley Core integration; document-to-vector-and-graph crates in Rust; GraphRAG.
2. Decision record `docs/decisions/014-knowledge-base-memorilla.md` and ASSUMPTIONS entries.
3. Technical specification `docs/spec/knowledge_base_spec.md`: requirements (FR-K*, NFR-K*), data model (SurrealQL),
   API (`/v1/kb/*`), frontend (Knowledge page, chat integration), inference engine (embedding override, memory tokens),
   training (dataset builder, trainer, checkpoints), tests and acceptance per phase.
4. SurrealDB installed in `tools/surrealdb`, a service script `scripts/serve_surreal.sh` (start, stop, status; port 8300;
   root credentials from a secret file under `data/runtime/`), and a spike that proves vector, full-text and graph
   queries on this machine.

**Done when** the three documents exist and agree, and the spike passes.

### Phase 1: Knowledge Base foundation (ingestion, storage, search, Knowledge page)
1. `nafs-kb` crate: parsers for PDF, DOCX, PPTX, TEX, HTML, Markdown and text (pure Rust where possible); text cleanup;
   structure-aware chunking (headings, paragraphs, sentences; about 200 tokens, overlap 1 sentence); SurrealDB HTTP
   client; schema bootstrap (all tables and indexes of the spec, the vector index created when the embedder is known).
2. Gateway API: create, list, rename and delete knowledge bases; upload documents (multipart, size limit, type check);
   list and delete documents; chunk preview; search (BM25 in this phase). Ingestion is a background job with status.
3. Knowledge page in the UI: knowledge bases, drag-and-drop upload with progress, documents with status and chunk
   counts, a search box that shows ranked passages with their source, delete with confirmation.
4. Config `[knowledge]` (enabled, SurrealDB URL, namespace, database, upload limit, chunk size), `make up` starts SurrealDB
   when enabled, `/v1/system` shows its health.
5. Tests: parser fixtures for every format, chunker properties, SurrealDB integration test (skips when no server),
   gateway route tests, UI copy-lint.

**Done when** a PDF, a DOCX, a PPTX, a TEX and an HTML file upload from the page, show their chunks, and are found by
search; deleting a document removes its chunks.

### Phase 2: Embeddings and vector RAG
Candle Qwen3-Embedding-0.6B (last-token pooling, L2 normalised; documents without a prefix; a query gets the
instruction prefix for retrieval but **no prefix** when it feeds Memorilla, which was trained that way) in `nafs-memorilla`
(ask before downloading the weights, about 1.2 GB); embeddings stored on chunks; HNSW index; hybrid retrieval (BM25 + KNN,
reciprocal-rank fusion); chat knowledge mode `rag` with sources shown and the passages delimited as data. Measure embedding
throughput and retrieval quality (recall@5 on a question set built from the uploaded documents).

### Phase 3: Graph construction and GraphRAG
Entity and relation extraction per chunk with System 1 under a JSON Schema (verified by the existing schema verifier),
entity resolution (normalised name + embedding similarity), `mentions` and `related` edges, community detection (Leiden or
Louvain in Rust) and community summaries; knowledge mode `graph` (local search: entities of the question -> neighbourhood
-> chunks; global search: community summaries). Graph view on the Knowledge page.

### Phase 4: Memorilla in Candle (inference)
`MemoryModule` in Candle mirroring `memory.py` exactly (same parameter names); loader for `memory.pt` via a one-time
conversion to safetensors; a parity test against PyTorch on the released checkpoint with random inputs (max abs error
within bf16 tolerance); benchmarks of the module's forward time for N = 10 … 10,000 documents on this CPU.

### Phase 5: Memory tokens in Smiley Core
Smiley Core: a marked addition letting `forward_rows` take caller-supplied input vectors for chosen rows; the prefix store
must not reuse a cache across different memory vectors (key on a digest of the vectors). `nafs-smiley-server`: accept
`memory_embeds` (base64 f32, K x d_model) and a placeholder token. Gateway: knowledge mode `memory`. Test: an identity check
(supplying the token's own embedding gives bit-identical logits to the plain path).

### Phase 6: Datasets and training on uploaded knowledge
Dataset builder from a knowledge base (no external LLM needed for the first two): (a) reconstruction rows (restate a
chunk from memory, Stage-1 style); (b) graph-claim rows (true triples and corrupted ones, FactKG style, exact answers);
(c) synthetic question–answer rows written by System 2 from one or two chunks and kept only when System 1 with the gold
chunk answers them consistently. Parquet and safetensors in Memorilla's layout so the PyTorch trainer can read the same
data. Candle trainer (AdamW, warm-up + cosine, gradient clipping, answer-only loss, frozen f32 decoder, checkpoint per
epoch, resume). Measure seconds per example; derive the GPU-hours for Stages 1–2 for our decoder and write the recipe.
"Train" and "Fine-tune" buttons on the Knowledge page with progress and loss curve.

### Phase 7: Routing, hybrid and evaluation
Knowledge mode `hybrid` (memory tokens plus the top passages), an `auto` policy chosen by measured results, and an A/B on a
held-out question set from real documents: off vs rag vs graph vs memory vs hybrid, quality and latency, paired tests.

### Phase 8: Hardening and release
Security review of uploads (size, type sniffing, zip bombs in DOCX/PPTX, path traversal, HTML scripts), retention and
deletion of a knowledge base everywhere (SurrealDB, files, embeddings, checkpoints), operator runbook, report section, CI.

## 5. CONSTRAINTS AND TRAPS

- Retrieved text and graph facts are **data, not instructions**: delimit them and say so in the system prompt.
- Memory tokens are decoder-specific; a module trained for System 1 cannot be used on System 2.
- Qwen3's vocabulary has no `<|memory|>` token; Smiley Core does not need one: the placeholder rows take supplied vectors.
- The prefix store keys on token ids; rows with supplied vectors must change the key, or a cached prefix from another
  question will be reused.
- BM25 over a handful of chunks gives zero scores for terms present in half of them (IDF clamp); test with realistic sizes.
- Training through a decoder needs f32 weights in Candle (GGUF quantised matmuls are not differentiable); Qwen3-1.7B in
  f32 is about 6.9 GB.
- Debug-mode tests of heavy crates starve the live engines; use `cargo test --release` for them and not while
  benchmarking.

## 6. WHEN YOU STOP

Report what is done with evidence, what is not, and the next phase with its first three actions.
