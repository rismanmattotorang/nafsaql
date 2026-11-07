# Memorilla, a knowledge base, and GraphRAG for Nafs-Aql (Phase 20, research)

Date 2026-11-07. Sources: the paper (arXiv 2610.21375v3, 3 Nov 2026, kept as `papers/2610.21375v3.pdf`), the reference code
in `third_party/memorilla` (github.com/snap-stanford/memorilla @ `d8f1889`), the project site (memory.stanford.edu), and a
crates.io survey made on the same day. Decision: [`014-knowledge-base-memorilla.md`](../decisions/014-knowledge-base-memorilla.md).
Specification: [`knowledge_base_spec.md`](../spec/knowledge_base_spec.md). Prompt: [`MEMORILLA_KB_PROMPT.md`](../prompts/MEMORILLA_KB_PROMPT.md).

## 1. What Memorilla is

A memory system for a **frozen** LLM. Three operations:

1. **Encode once.** A frozen embedding model maps every document chunk d_i to z_i (h_e wide). The matrix Z_D (N x h_e) is the
   persistent store; only added or changed chunks are re-encoded. It is the same index dense retrieval builds, so it costs
   nothing extra over RAG.
2. **Read per query.** A trainable *memory module* reads all of Z_D, conditioned on the query embedding z_q, and returns
   K = 16 vectors in the **decoder's input-embedding space** (h_L wide).
3. **Inject.** The 16 vectors replace the input embeddings at 16 placeholder positions in the prompt. Nothing inside the
   decoder changes; RoPE runs over the whole sequence as usual.

The decoder therefore sees 16 "soft tokens" instead of thousands of document tokens, whatever N is: 1 or 512 documents
cost the same at decode time (paper, Table 14: 740 ms vs 739 ms end to end on an H100, while full context grows from 749 ms
to 19,932 ms).

## 2. The module, read from `memorilla/memory.py`

| Step | Operation | Shapes (batch b, slots K, documents N) |
|---|---|---|
| 1. Slots | `memory_queries` (1, K, h_e), Xavier-uniform; or, with `retrieval_init`, the K documents with the highest cosine to the query (zero-padded) | (b, K, h_e) |
| 2. Query conditioning | `slots = LN(slots + MHA(q=slots, k=v=z_q))`; one query vector shifts every slot alike, the residual keeps them apart | (b, K, h_e) |
| 3. Document encoder | optional `TransformerEncoder` over the documents (0 layers in every released recipe) | (b, N, h_e) |
| 4. Document norm | `docs = LN(Z_D)` | (b, N, h_e) |
| 5. Refinement, x2 | `slots += Dropout(MHA(LN(slots), docs, docs, key_padding_mask))`; `slots += FFN(LN(slots))`, FFN = Linear(h_e, 4h_e) GELU Dropout Linear(4h_e, h_e) Dropout | (b, K, h_e) |
| 6. Projection | `M = Linear(LN(slots))` to h_L | (b, K, h_L) |

Eight heads everywhere, dropout 0.1, `nn.MultiheadAttention` with a packed `in_proj_weight` (3h_e x h_e) and an `out_proj`.

**Parameter count** (no self-attention layers, L refinement blocks):
K·h_e + (4h_e² + 4h_e) + 4h_e + L·(12h_e² + 13h_e) + 2h_e + (h_e·h_L + h_L).
For the paper's configuration (h_e 2560, h_L 4096, K 16, L 2) this gives **194,123,776**, the number the paper reports. For
Nafs-Aql with Qwen3-Embedding-0.6B (h_e 1024): **31,512,576** for System 1 (Qwen3-1.7B, h_L 2048) and **32,037,376** for
System 2 (Qwen3-4B, h_L 2560). In f32 that is 126 MB; its forward pass is about 2·(N·K·h_e + 12·K·h_e²) FLOPs, under 0.5
GFLOP for N = 10,000, i.e. milliseconds on this CPU. The module is not the cost; the encoder and the decoder are.

**Prompt layout** (`memorilla/data.py`): the system prompt gets `"\nMemory" + "<|memory|>" * 16 + "Memory"` appended; the
user turn is the question; the chat template is applied with thinking off; prompts are left-padded in batches. At
inference the decoder's input embedding matrix is read from the safetensors, the placeholder rows are overwritten
(`inject_memory`), and vLLM is fed `prompt_embeds`.

**Encoder.** `load_encoder` runs Qwen3-Embedding-4B with vLLM's `embed` task in bf16: the model's own pooler (last token,
L2-normalised) and **no instruction prefix**, for documents and questions alike. The paper's Table 3 says "mean-pooled";
the code is what the checkpoints were trained with, so a port follows the code.

## 3. Training, read from `train.py`, the recipes and Appendix B

- **Objective:** cross-entropy on the answer tokens only (labels -100 elsewhere). Encoder and decoder frozen; the decoder
  runs in bf16 with `requires_grad_(False)`, but the loss is back-propagated **through** it into the module.
- **Optimiser:** AdamW, weight decay 0.01, linear warm-up from 1e-8 x lr, then cosine to 10 % of the peak, per step;
  gradient clipping 1.0; bf16 mixed precision; seed 23; Lightning DDP.
- **Curriculum** (Table 4):

| Stage | Data | Rows | Init | lr | Epochs | Max length | Effective batch |
|---|---|---|---|---|---|---|---|
| 1 | Wikipedia paraphrase reconstruction, chunks of about 128 tokens, 1.73 chunks per row | 1,979,809 | random | 3e-4 | 1 | 1024 | 128 |
| 2 | SQuAD v2, DROP, CoQA, PubMedQA, QuAIL, PwC, CNN/DM, SAMSum, DialogSum, MS MARCO | 1,343,993 | Stage 1 | 3e-4 | 1 | 1024 | 128 |
| 3 | the eight benchmarks' training splits | (per benchmark) | Stage 2 | 1e-4 | 5 | 2048 | 64 |
| single task / own data | one config | yours | Stage 2 | 1e-4 | 5 | 2048 | 2 x 2 x 4 GPUs |

About 6 + 4 + 9 hours on 4 H100s; a personalisation fine-tune takes about 25 minutes.

- **Own data** needs no code change: parquet rows `collection_id, question, answer, documents` (+ `choices` for
  multiple-choice scoring), one folder per config, `train` required, `validation` optional. Rows sharing a
  `collection_id` must share their `documents`; each document entry is embedded into **one** vector, so long texts must be
  split first. Chunk sizes used in the benchmarks: one KG triple (13 tokens) up to one chat session (280 tokens).

## 4. What the results mean for a knowledge base

| Finding | Number | Consequence for Nafs-Aql |
|---|---|---|
| Relational, multi-fact questions favour Memorilla | FactKG 84.33 % vs 80.43 % full context; vs RAG-50 +4.2 pp at 1 entity, +7.6 pp at 4+; RAG drops from 78.7 to 51.8 as composed facts go 1 -> 4 | a graph built from the documents is good *training data* for the module (true/false claims over triples) |
| Entity recall favours retrieval | NarrativeQA: RAG +3.45 ROUGE-L on "who", +2.93 on "which"; Memorilla +4.9 on "why", +7.0 on "how" | keep RAG; route by question type or combine |
| Short collections favour text | PubMedQA (324 tokens in all): 59 % vs 67–69 % | below a few thousand tokens, put the text in the prompt |
| They solve different questions | FactKG: 15.9 % only Memorilla, 10.0 % only RAG-50; PMv2: 27.1 % vs 9.2 % | a **hybrid** mode (memory tokens + top passages) is worth measuring |
| Capacity | K = 128 at training: no gain over 16; first-K at inference degrades smoothly | K = 16 |
| Encoder size | 4B -> 8B: within 1.3 points except PubMedQA (+3 questions) | a smaller encoder is a reasonable bet, but it is untested below 4B: measure |
| Fine detail is lost | stated limitation | answers that need an exact number or name should come from text |
| Transfer across decoders | untested | a module is tied to its decoder's embedding space and its training |

## 5. Preparing datasets from uploaded documents

A knowledge base has documents but no questions. Training rows must be made. In order of cost:

1. **Reconstruction (Stage 1 style, no model needed).** Collection = the chunks of one document (or a window of
   neighbouring chunks); question = "Restate the passage about <heading or first words>" or "Restate passage i";
   answer = the chunk text (cap 512 tokens). This teaches the module to carry content into the decoder.
2. **Graph claims (FactKG style, exact answers).** From the relation graph (Phase 3): a true triple as a claim, a
   corrupted one (swap the object for one of the same type) as a false claim; answer `True`/`False`; collection = the
   chunks that mention the subject, plus distractor chunks. Cheap and exactly scorable.
3. **Synthetic questions (Stage 2/3 style).** System 2 writes question–answer pairs from one chunk, or from two linked
   chunks for multi-hop questions; a pair is kept only when System 1, given the gold chunk as text, answers it the same
   way (a consistency filter that uses the existing verifier and judge). Collection = the whole knowledge base when it is
   small, otherwise the top-N chunks by embedding plus random distractors, so the module learns to find the needle.
4. **Validation split.** 10 % of the questions, never of the chunks: the store stays the same, the questions differ.

Layout written by the builder (so the PyTorch trainer can read it unchanged): `data/<config>/{train,validation}-00000-of-00001.parquet`
and `embeddings/<encoder-slug>/<config>/<split>/{index,documents-00000,questions}.safetensors` with float16 vectors.

## 6. Training on this CPU: the cost

Back-propagating to the module needs the decoder's activation gradients but not its weight gradients, so a step costs
about two forward passes: **≈ 4·P·T FLOPs** per example (P decoder parameters, T tokens of prompt + 16 + answer).

| Decoder | P | T | FLOPs per example | At 100–200 GFLOP/s (estimate, to be measured) |
|---|---|---|---|---|
| Qwen3-0.6B | 0.6 B | 200 | 0.48 T | 2.4–4.8 s |
| Qwen3-1.7B (System 1) | 1.7 B | 200 | 1.36 T | 7–14 s |
| Qwen3-4B (System 2) | 4.0 B | 200 | 3.2 T | 16–32 s |

So on this machine: **fine-tuning** a module on a knowledge base (a few thousand rows, a few epochs) is an overnight job
for System 1 — feasible. **Stage 1 + 2 pre-training** (3.3 M rows) would take about 300–600 days — not feasible; it is a
GPU job (estimated 10–20 GPU-hours on H100s for a 1.7B decoder, scaled from the paper's 10 hours on 4 H100s for 8B; to be
re-derived from measured per-example times in Phase 6). Gradients must flow through an f32 decoder in Candle (quantised
GGUF matmuls are not differentiable); Qwen3-1.7B in f32 is 6.9 GB, which fits in 32 GB.

Without a pre-trained module, a knowledge-base fine-tune starts from scratch; the paper's FactKG recipe does exactly
that (`retrieval_init`, 5 epochs, one GPU) and reaches its best result, so **`retrieval_init` from scratch is the CPU path**
until a GPU pre-train exists.

## 7. The Candle port

| Part | Candle | Notes |
|---|---|---|
| `MemoryModule` | `candle-nn` Linear, LayerNorm (eps 1e-5), a hand-written multi-head attention with the packed `in_proj` layout and key padding mask, GELU (erf) | parameter names identical to PyTorch so `memory.pt` converts 1:1 |
| Checkpoint | `memory.pt` -> safetensors once (Python, `torch.load(weights_only=True)`), `config.json` alongside | Candle cannot read pickles safely |
| Encoder | Qwen3-Embedding-0.6B: last-token pooling, L2 norm | in Candle for training and parity; at serving time Smiley Core can run it (Qwen3 architecture) with a pooling head, which reuses the engine's AVX2 kernels |
| Decoder for training | `candle-transformers` Qwen3 reworked to take `inputs_embeds`, f32 safetensors from `models/hf/Qwen3-1.7B` | autograd through a frozen model: `Var` only for the module |
| Trainer | AdamW from `candle-nn`, warm-up + cosine by hand, global-norm clipping by hand, answer-only cross-entropy | checkpoint per epoch, resume |
| Parity | random inputs through PyTorch and Candle on the released Stage 3 checkpoint | target: max abs error < 1e-3 in f32 |

Candle 0.11.1 is current. On x86 without MKL it uses the `gemm` crate: fine for the module, the bottleneck for training.

## 8. Serving memory tokens through Smiley Core

`Model::forward_rows` looks each token up with `self.embed(t, ...)`. The integration is one marked addition: an optional
slice of caller-supplied input vectors for chosen rows, used instead of the lookup. The server accepts
`memory_embeds` (K x d_model f32, base64) and puts them at 16 placeholder positions (any token id: the vector replaces it).
Two traps: the **prefix store** keys cached prefixes on token ids, so prompts with supplied vectors must carry a digest of
the vectors in the key; and a **batch** with several users must route each row's vectors to its own row.

## 9. Documents to vectors and a graph in Rust (crates.io survey, 2026-11-07)

| Need | Choice | Why | Rejected |
|---|---|---|---|
| Database | SurrealDB **3.3.5 server** (`tools/surrealdb`, surrealkv storage) over HTTP | vector (HNSW), full text (BM25) and graph in one store; no new crates in the gateway; same process model as the engines | embedded `surrealdb` crate (huge build; 4.0 drops RocksDB but is 4 days old); `rig-surrealdb` (brute-force cosine, no index, pins 3.x); swiftide (no SurrealDB); langchain-rust (heavy) |
| PDF | `pdf_oxide` 0.3.x (pure Rust, Markdown output), `pdf-extract` as fallback | quality close to PyMuPDF in its own benchmark | pdfium-render (native library), mupdf (AGPL), kreuzberg (bundles PDFium), extractous (GraalVM) |
| DOCX, PPTX | `zip` + `quick-xml`, our own reader of `word/document.xml` and `ppt/slides/slideN.xml` | small, auditable, bounded against zip bombs | office_oxide 0.1 (young), docx-rs (writer), dotext (dead) |
| HTML | `htmd` (HTML -> Markdown) after dropping `script`/`style`/`nav` | keeps headings for chunking | readability 0.3 (stale) |
| LaTeX | our own stripper (preamble and comments dropped, sectioning -> headings, formatting unwrapped, maths kept as `$..$`) | no maintained crate | pandoc wrapper (external binary) |
| Markdown, text | read directly | | |
| Chunking | `text-splitter` 0.33 `MarkdownSplitter` (headings, paragraphs, sentences), character sizer in Phase 1, Qwen tokenizer sizer later | semantic boundaries, overlap | |
| Embeddings | Qwen3-Embedding-0.6B (1024-d): Smiley Core pooling head for serving, Candle for training/parity | Qwen3 family, as the paper; `ort` dropped Intel macOS in 2.0.0-rc.11, so fastembed's ONNX models are out here | model2vec (fast, weaker; possible first tier) |
| Entities and relations | System 1 with a JSON Schema prompt, checked by the existing schema verifier, as a background job | uses the engine we have | gline-rs (pins an old `ort`, unverified on Intel), rust-bert (libtorch) |
| Communities | `leiden-rs` 0.8 or `graphrs` 0.12 | Leiden, weighted | |
| GraphRAG style | LightRAG-like local search first (entities of the question -> 1–2 hops -> chunks), GraphRAG global search over community summaries later | one LLM call per answer, not per community report | |

SurrealQL verified on this machine (`scripts/serve_surreal.sh spike`): `FULLTEXT ANALYZER a BM25` + `@n@` / `@n,OR@` +
`search::score(n)`; `HNSW DIMENSION d DIST COSINE` + `<|k,ef|>` + `vector::distance::knn()`; `TYPE RELATION`, `RELATE`,
`->edge->table->edge->table.field`. `ORDER BY` must name a selected alias (an expression there is a parse error).

**Trap found in Phase 1: never send document text to SurrealDB as JSON.** Its JSON decoding of RPC parameters turns any
string that *starts like a record id* into one and drops the rest: `"Caption: Model of the gate and more text"` arrives
as the record `Caption:Model` (measured on 3.3.5 with `type::of`). A SCHEMAFULL `string` field then rejects it (two of six
test documents failed), and a search query such as `"Note: memory"` would silently become a record. `nafs-kb` speaks the
RPC in **CBOR** (`ciborium`), as SurrealDB's own SDKs do: strings stay strings, record ids are tag 8. JSON `null` still
arrives as `NULL`, which `option<...>` fields refuse, so optional values are written as `$x ?? NONE`.
SurrealDB 3.x renamed `SEARCH ANALYZER` to `FULLTEXT ANALYZER` and removed MTREE; 4.0 changes BM25 scores and needs a
one-way upgrade of a 3.x store. Hybrid fusion is `search::rrf([$vs, $ft], k, 60)` (3.0+).

## 10. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| A module trained from scratch on one small knowledge base overfits or learns nothing | high | reconstruction + graph-claim rows first; validation loss and a held-out question set decide; RAG stays the default |
| Training time on four cores | high | System 1 decoder only; short sequences (T ≈ 200); measure before promising; GPU recipe for pre-training |
| Qwen3-Embedding-0.6B is weaker than the paper's 4B encoder | medium | the paper's 4B -> 8B ablation was flat; measure with retrieval recall@5 |
| Ingestion competes with chat for the four cores | high | background job with low priority, one at a time, pausable |
| SurrealDB is BSL 1.1 | low | self-hosted use is allowed; recorded in NOTICE |
| Prefix store reuses a cache across different memory vectors | certain if ignored | digest in the key, tested |

## 11. Phase 2 results (2026-11-07): embeddings, vector search, RAG in chat

All on the reference machine while the owner's other workloads ran (load average 44 to 60): **shared machine**.

| What | Result | Evidence |
|---|---|---|
| Candle reference (f32) on the model card's example | 0.7646, 0.1414 / 0.1355, 0.6000: the card's numbers exactly | `nafs-memorilla` parity test |
| Smiley Core Q8_0 against the Candle reference | cosine 0.99914 to 0.99959 on six texts (spec target 0.999) | same test |
| Pooling | the model's tokenizer appends `<|endoftext|>` (151643) and pools it; the port does the same | `tokenizer.json` post-processor |
| Indexing throughput | 379 passages, 66,978 tokens in 964 s (69 tokens/s; the server got about 70 % of one core) | `benchmarks/kb_retrieval.json` |
| Retrieval, 20 questions, k = 5 | words: hit@1 0.50, recall 0.85, MRR 0.660; meaning: 0.35, 0.55, 0.414; **hybrid: 0.55, 0.90, 0.672** | `scripts/kb_retrieval_eval.py` |
| HNSW against the exact scan | same top-1 on 20/20 questions, top-5 overlap 1.00; 111 vs 101 ms at 379 vectors | same file |
| Filtered HNSW (`WHERE kb = .. AND embedding <|k,ef|>`) | returns only the base's rows (it filters during the search), 457 ms on a 30-row base inside 3,000 rows at 64 dims | spike |
| Chat with the knowledge base (quick model) | FactKG question: correct numbers with a citation, against a hedged guess without the base; port question: correct; off-topic question: "not covered" note | gateway on :8091 |

What the numbers say. Meaning alone misses exact figures and names (the 0.6B encoder places "84.33" and "Qwen3-Embedding-4B"
passages below others), words alone miss paraphrases; fusing the two ranks wins on every measure, so **hybrid is the
default** for search and for chat. Off-topic passages scored 0.38 to 0.40 and relevant ones 0.45 to 0.77, so chat keeps
passages with similarity >= 0.45 (`rag_min_similarity`). Grounding is not perfect: on "How many training examples did
Memorilla use in its first stage?" System 1 read the right passage and still wrote "15 English Wikipedia articles" (the
page number), and the agreement samples marked it Cross-checked. Quality of grounded answers is measured in Phase 7.

## 12. Phase 3 results (2026-11-07): the knowledge graph and GraphRAG

Shared machine throughout (load average 16 to 86). Extraction model: System 1 (Qwen3-1.7B Q4_K_M on Smiley Core).

**The extraction prompt was measured, not guessed.** Four variants on the same two passages:

| Variant | Output tokens | Result |
|---|---|---|
| JSON objects with a description per entity and relation | 327 | faithful, valid; 60 s per passage at load 16–23 |
| `E | name | type` / `R | s | p | t` lines | 155–199 | format broken: literal `name`, predicates as entities |
| JSON arrays `[name, type]`, `[s, p, t]` | 15–231 | copied the template (`["name","type"]`), split into two objects |
| **JSON objects without descriptions, one worked example on an unrelated topic** | 138–139 | faithful, valid; adopted |

**First full build, and what it showed.** On the five small fixture documents (20 passages) the first build gave 91
entities, but the most mentioned were the document's own headings ("Architecture and Design", "Components"), relations
such as `architecture > Components`, and "LLVM", copied from the prompt's example into passages that never mention it.
Three changes, all in `graph.rs`: heading lines are removed from the extraction input; every word of an entity's name
must occur in the passage (grounding); generic words (`system`, `architecture`, ...) are not entities. The rebuild
(`?fresh=true`) kept 109 entities and 73 relations whose top items are the system's real parts (`nafs-core implements
The gate`, `SurrealDB uses HNSW`, `Ledger feeds dashboard`, `The gate computes features verifier outcome`). 18 of 20
replies passed the JSON Schema both times; 1 entity was merged by meaning in the first build, 0 in the second.

**Cost.** 17 to 28 seconds per passage (load 33 to 40), about 140 output tokens each. A 189-passage paper would take
about an hour, so building the graph is an explicit, resumable background job (`Build the graph` on the Knowledge
page), not part of every upload.

**Chat.** On "Which component implements the gate, and what features does the gate compute?" both `rag` and `graph`
modes answered correctly; graph mode put `nafs-core | implements | The gate` and the six `computes features` relations in
the prompt ahead of the passages, along with a few weaker facts from loosely matched entities. Whether graph mode helps
on balance is Phase 7's measurement.

**Known limits.** TeX macros defined outside the uploaded file stay as names (`\Sone` reads "Sone"); the file's own
`\newcommand`/`\def` macros are expanded. Types are coarse (eight words). Community summaries and global search are not
built (one System 2 call per community).

## 13. Phase 4 results (2026-11-07): the memory module in Candle

`nafs_memorilla::MemoryModule` mirrors `memory.py`: the same parameter names (packed `in_proj_weight`, `ffn_layers.i.0/.3`,
`encoder.layers.i.*`), so a converted state dict loads as is (`training/memorilla/convert_checkpoint.py`).

**Parity without downloading a checkpoint.** `training/memorilla/make_parity_fixtures.py` runs the unmodified reference
class (its helper module stubbed, so only torch is needed) with seeded random weights, LayerNorm parameters perturbed, and
one padded row, for three configurations; the fixtures (1 MB) are in the crate, so CI checks parity without Python.
Largest difference: 1.9e-6 (default), 9.5e-7 (`retrieval_init`), 7.2e-7 (document self-attention); target 1e-4. The
parameter formula matches the stored tensors and the paper (194,123,776; 190,190,080 for the 4B decoder). Changing the
values of padded documents does not change the memory.

**Speed on this CPU (load 31 to 34, shared machine).** Two findings:

1. Candle's portable matmul ran the module's 16-row products at about 7 GFLOP/s here (`examples/matmul_probe.rs`:
   15–25 ms for 16x1024 by 1024x4096). With Candle's `accelerate` feature (Apple's BLAS, macOS only) the same product
   takes 1.3–2.7 ms. Parity is unchanged.
2. Most of the forward pass does not depend on the question: the document encoder, the document norm and every block's
   keys and values. `prepare` computes them once per collection and `forward_prepared` answers each question from them.

| Module | N documents | full forward | prepare once | per question |
|---|---|---|---|---|
| Nafs-Aql (E 1024 -> 2048, 31.5 M) | 10 | 42 ms | 8 ms | 54 ms |
| | 100 | 88 ms | 11 ms | 49 ms |
| | 1,000 | 217 ms | 111 ms | 88 ms |
| | 10,000 | 1,429 ms | 877 ms | 436 ms (195 MB prepared) |
| Paper (E 2560 -> 4096, 194 M) | 10,000 | 4,837 ms | 3,093 ms | 533 ms (488 MB) |

NFR-K08 (< 50 ms at N = 10,000) is met up to about 100 documents, not at 10,000: the per-question attention over 10,000
keys and values is the remaining cost. Against the decoder (seconds per answer on this CPU) the module is a small share up
to a few thousand passages. `benchmarks/memorilla_forward.json`.

## 14. Phase 5 results (2026-11-07): memory tokens through Smiley Core

**Engine.** Three marked additions in `smiley-core`: `State::inputs` (vectors that replace chosen rows' token embeddings
in the next `forward_rows`), `Model::token_embedding`, and `Engine::admit_with_inputs` (vectors by prompt position,
applied during prefill; the positions must precede the prompt's last token). The shared prefix store never restores a
cache past the first supplied vector and never stores one beyond it, so a cache computed from one question's memory is
not reused for another's. Test on the real Qwen3-1.7B: supplying each row's own embedding gives exactly the plain
answer; nothing past the first vector is restored.

**Server.** `memory_embeds: {"vectors": [[f32; d_model]; K], "placeholder"?}` on `/v1/chat/completions`; the vectors go
to the K positions of the placeholder token (`<|fim_pad|>`, a special token chat never produces) in the prompt, and a
count mismatch is refused. On the same prompt with 16 placeholders: no vectors and zero vectors give the same answer up to
spelling, large vectors derail it, 15 vectors for 16 places is refused.

**Gateway.** Knowledge mode `memory`: the question is embedded **without** the retrieval instruction (as Memorilla was
trained), the module reads the base's prepared passage vectors (cached per base until its update time or vector count
changes), the system prompt gets Memorilla's `\nMemory<placeholder>x16Memory`, and every quick-model request of the turn
carries the 16 vectors. The request is held on the quick path: the module lives in System 1's embedding space, and System
2 would see placeholders with no vectors. Measured on 20 passages: prepare 56 ms once, read 26–28 ms per question.

**Not yet useful.** No module is trained for Qwen3-1.7B. The end-to-end run used an untrained module
(`examples/memory_init.rs`), whose tokens carry no knowledge: the model answered that it could not tell. `memory` mode is
therefore offered only in Advanced mode, labelled experimental, and only when `[knowledge] memory_checkpoint` names a
module. Phase 6 trains one.
