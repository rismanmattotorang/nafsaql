# Knowledge Base, GraphRAG and Memorilla: technical specification (Phase 20)

Version 1.9, 2026-11-08. Status: Phases 0–7 implemented (6: training works; 7: `hybrid`, `auto` and the measured
mode comparison), Phase 21's retrieval and graph upgrade (FR-K50 to FR-K59, below), Phase 22's two-lane
ingestion (FR-K61) and Phase 8's hardening (SR-K11 to SR-K13, FR-K62; decision 016). Research:
[`memorilla_knowledge.md`](../research/memorilla_knowledge.md). Decision: [`014`](../decisions/014-knowledge-base-memorilla.md).
Requirement ids: FR-K (functional), NFR-K (non-functional), SR-K (security), IF-K (interfaces).

## 1. Scope

A person creates knowledge bases, uploads documents to them, searches them, and asks the assistant questions answered
from them by vector RAG, GraphRAG, Memorilla memory tokens, or a hybrid. A Memorilla memory module can be trained or
fine-tuned on a knowledge base. Everything runs on the reference CPU machine.

Out of scope: OCR of scanned PDFs and images, spreadsheets, audio and video, multi-tenant permissions beyond the
gateway's existing token auth, and Stage 1–2 Memorilla pre-training on this machine (a GPU recipe is delivered instead).

## 2. Components

| Component | Crate / process | Phase |
|---|---|---|
| SurrealDB 3.3.5 server, surrealkv storage, :8300 | `tools/surrealdb/surreal`, `scripts/serve_surreal.sh` | 0 |
| Parsing, chunking, storage, search, ingestion jobs | `rust/crates/nafs-kb` | 1 |
| Knowledge API | `nafs-gateway` (`kb.rs`) | 1 |
| Knowledge page | `nafs-web/assets` (`#/knowledge`) | 1 |
| Embeddings (serving) | `nafs-smiley-server --embedding` with a pooling head, :8203 | 2 |
| Embeddings (training, parity), memory module, trainer, dataset builder | `rust/crates/nafs-memorilla` (Candle 0.11) | 2, 4, 6 |
| Graph extraction, entity resolution, communities | `nafs-kb::graph` | 3 |
| Memory tokens in the engine | `smiley-core` (marked addition), `nafs-smiley-server` | 5 |
| Knowledge modes in chat | `nafs-core` orchestrator, `nafs-types` | 2, 3, 5, 7 |

## 3. Functional requirements

### Knowledge bases and documents (Phase 1)
- **FR-K01** Create, list, rename, describe and delete knowledge bases. Deleting one removes its documents, chunks,
  vectors, graph, stored files and memory checkpoints.
- **FR-K02** Upload one or more files per request (multipart). Accepted kinds, detected by extension and checked by
  content sniffing: PDF (`%PDF-`), DOCX and PPTX (ZIP with `word/document.xml` / `ppt/presentation.xml`), TEX, HTML/HTM,
  Markdown, plain text. Anything else is refused with a reason.
- **FR-K03** An upload returns at once with one document record per file in state `queued`; ingestion runs in the
  background, one document at a time, through `parsing -> chunking -> storing -> ready` (Phase 2 adds `embedding`,
  Phase 3 `extracting`). A failure leaves the document in `failed` with a plain-language reason; it can be retried.
- **FR-K04** The same file (SHA-256) uploaded twice to one knowledge base is not ingested twice; the existing record is
  returned with `duplicate: true`.
- **FR-K05** List documents with title, kind, size, state, chunk and character counts, and time added; delete one
  document with its chunks; re-ingest one document from its stored original.
- **FR-K06** Show a document's chunks in order, with their heading path and page (PDF) where known.
- **FR-K07** Search a knowledge base: full-text BM25 (Phase 1), vector (Phase 2), hybrid by reciprocal-rank fusion
  (Phase 2). Results carry the chunk text, document title, heading path, score and rank.

### Parsing and chunking (Phase 1)
- **FR-K10** PDF: text per page in reading order (`pdf_oxide`, Markdown output); pages kept as boundaries; a PDF whose
  pages yield no text is reported as "no text layer (scanned?)".
- **FR-K11** DOCX: paragraphs from `word/document.xml`, headings from `Heading1..9` / `Title` paragraph styles, tables as
  rows of cells separated by ` | `; PPTX: one section per slide (`# Slide n`) with its text frames, then its notes.
- **FR-K12** HTML: `script`, `style`, `noscript`, `nav`, `footer`, `header` and `aside` dropped, then converted to
  Markdown (`htmd`); the `<title>` becomes the document title.
- **FR-K13** TEX: preamble before `\begin{document}` and `%` comments dropped; `\part`…`\subparagraph` become headings;
  `\title{}` the title; formatting commands unwrapped; environments `figure`, `table` keep their captions; maths kept as
  written between `$`.
- **FR-K14** Markdown and text are read as UTF-8 (lossy); text is normalised (NFC is not required; control characters
  removed, runs of blank lines collapsed, hyphenation at line ends joined).
- **FR-K15** Chunking with `text-splitter`'s Markdown splitter (headings, then paragraphs, sentences, words): target
  `chunk_chars` (default 1,200 characters, about 300 tokens), minimum 300, overlap `chunk_overlap` (default 150).
  Each chunk records its ordinal, heading path (`Intro > Method`), page (PDF) and character count.

### Retrieval and chat (Phases 2–3, 5, 7)
- **FR-K20** Chat requests take `knowledge: { kb, mode }` with mode `off | rag | graph | memory | hybrid | auto`.
- **FR-K21** `rag`: hybrid search for the top `k` chunks (default 4, within `rag_max_chars`, default 4,800 characters),
  keeping only passages whose similarity reaches `rag_min_similarity` (0.45, measured in Phase 2); the
  passages go into the prompt inside a delimited block that says they are reference data, not instructions; the reply
  carries `sources` (document, heading, chunk id, score).
- **FR-K22** `graph`: entities of the question (string match + entity-vector search) -> 1–2 hops over `related` ->
  chunks that mention them, ranked with the hybrid score; facts given as `subject — predicate — object` lines.
- **FR-K23** `memory`: question embedded without a prefix; the memory module turns the whole store into 16 vectors
  (prepared passage vectors cached per base); the engine receives them at 16 placeholder positions; the prompt keeps
  Memorilla's format; the turn is answered by the quick model only. Needs a module for that base: its own activated
  training run (Phase 6) or `[knowledge] memory_checkpoint`. Offered in Advanced mode as experimental, and only for a
  base that has one (`GET /v1/kb` reports `memory` per base).
- **FR-K24** `hybrid`: `memory` plus the top `[knowledge] hybrid_k` passages of `rag` (default 2), so the answer can
  quote what the memory recalls; quick model only, like `memory`. `auto`: the first mode of `[knowledge] auto_order`
  the base can serve — `memory`/`hybrid` need a module, `graph` needs relations in the graph, `rag` always works. The
  default order is `["rag"]`, from the Phase 7 measurement (§16 of the research note: on 20 held-out questions
  `graph` answered 15 against `rag`'s 18, winning none of the three discordant pairs, and took 28 % longer).
  When the memory cannot be read — no module, a width mismatch, an engine that does not take supplied input vectors —
  `memory` and `hybrid` fall back to `rag` and say so in `sources`, rather than answering with no knowledge at all.
- **FR-K25** When no chunk scores above `min_score` and mode is not `memory`, the assistant says the knowledge base
  does not cover the question, and answers from general knowledge only if the person asked for it.

### Graph (Phase 3)
- **FR-K30** On request (`POST /v1/kb/{kb}/graph/build`, a background job; `?fresh=true` starts over), System 1 reads
  each passage (heading lines removed) and replies `{entities:[{name,type}], relations:[{source,predicate,target}]}`
  (one worked example in the prompt; per-item descriptions were dropped: 2.4x the output tokens). The reply must pass a
  JSON Schema (`graph::extraction_schema`); a failing reply skips its passage. Entities are **grounded**: every word of
  a name must occur in the passage, and generic words (`system`, `architecture`, ...) are dropped.
- **FR-K31** Entity resolution: the key is the lower-case name without punctuation or a leading article; a new key is
  merged into an existing entity of the same type whose name vector has cosine >= `entity_merge_similarity` (0.92).
- **FR-K32** Edges: `chunk -mentions-> entity`, `entity -related-> entity` with predicate, weight (passages stating it)
  and evidence passages; a relation's id is a hash of (source, predicate, target), so repeats merge. Deleting or
  re-reading a document prunes evidence, relations and entities it alone supported.
- **FR-K33** Communities by Louvain (own implementation, deterministic) over `related`, labelled by their three most
  mentioned entities. Community summaries by System 2 and global search are deferred (one model call per community).
- **FR-K34** Graph view: the most mentioned entities or the two-hop neighbourhood of one, entity search by name, and
  for an entity its relations with the passage that states each.

### Memorilla (Phases 4–6)
- **FR-K40** Memory module in Candle with the parameter names and maths of `memorilla/memory.py`; loads converted
  safetensors with `config.json`; parity with PyTorch (§8). Implemented: `nafs_memorilla::MemoryModule`, with
  `prepare` (question-independent keys and values, once per collection) and `forward_prepared` (per question).
- **FR-K41** Dataset builder per knowledge base (`nafs_kb::dataset`): reconstruction rows and graph-claim rows (true and
  object-swapped false claims), every row reading the whole base, split 90/10 by a hash of the question; written as
  `dataset.json` + f32 safetensors, exported to Memorilla's parquet layout by `training/memorilla/export_dataset.py`.
  Synthetic QA rows by System 2 are **not built yet** (one model call per passage; Phase 7 decides if they pay).
- **FR-K42** Train (from scratch, `retrieval_init` on) and fine-tune (from the active checkpoint) a module for System 1:
  `memorilla-train`, a separate process started by `POST /v1/kb/{kb}/memory/runs`, through a frozen f32 Qwen3-1.7B
  (`nafs_memorilla::decoder`, differentiable throughout), answer-only loss, AdamW, warm-up + cosine, clipping, one example's
  graph at a time; progress file, stop, a checkpoint per epoch plus `best/` and `last/`; `activate` makes a run's module the
  base's. Initialisation: zero projection weights and the placeholder's embedding as bias (the untrained module returns
  the plain prompt). Not built: resume after a crash.
- **FR-K43** A GPU recipe (`training/memorilla/`) to pre-train Stages 1–2 for our encoder/decoder pair with the
  reference PyTorch code, with a cost estimate derived from measured per-example times.

Phase 21 (retrieval and graph upgrade; the gap analysis and the reasoning are in
[`graph_rag_upgrade_spec.md`](graph_rag_upgrade_spec.md)):

- **FR-K50** The graph **supplies candidates** to retrieval and never reorders it alone: passages a fact cites join
  the pool, and the search's rank and the graph's rank (facts' weight, then seed entities named, over the whole
  pool) are fused by reciprocal rank with `[knowledge] graph_weight` on the graph arm (1.0 = equal; 0 = facts only).
  Implemented: `Kb::fuse_graph`.
- **FR-K51** Contextual BM25: the words index is over `search_text` - the same text a passage is embedded as (title,
  heading path, context line, body). `reindex` fills it for passages stored before Phase 21.
- **FR-K52** The prompt block packs whole sentences under one character budget: each sentence scored by its own
  relevance to the question plus its passage's rank, both z-normalised; a unit that does not fit is skipped, not cut.
  Implemented: `nafs_core::knowledge::pack_units`.
- **FR-K53** A context line per passage, written by the graph extraction call the base already pays for, stored on
  the chunk, folded into `search_text` at once and into the vector on `reindex?fresh=true`. Never shown to a person.
- **FR-K54** The ontology is data: `GET/PUT /v1/kb/{kb}/ontology` (types with definitions, hashed; an empty list
  restores the default); the extraction prompt is built from it and a stray type is mapped onto its last type.
- **FR-K55** Every relation carries the sentence that states it, kept only if it occurs in the passage verbatim after
  whitespace normalisation; drops are counted per document (`quote_drops` in the log). **SR-K10**: the gate bounds
  what a document can put into a prompt through the graph.
- **FR-K56** Two-hop walk: hop 1 is every relation of a seed; hop 2 keeps `graph_beam` relations per reached entity,
  scored by weight (normalised to the heaviest first-hop relation) times `graph_decay` per hop; a second-hop fact
  carries `via`, and the prompt shows the chain.
- **FR-K59** Every knowledge reply carries a `sources` row naming the arms that ran and why any did not.
- **FR-K62** (Phase 8) The chat relevance filter keeps a passage whose meaning similarity reaches `rag_min_similarity`
  **or** whose words match strongly (`rag_lexical_rescue`, a BM25 score, default 10.0; 0 = off). Found by ranking the
  answering passages: vector search alone left 4 of 20 outside its top 20 on the paper base, and one of them, ranked
  first by the words, was dropped by the similarity test, so the question went unanswered. Off-topic questions scored
  at most 8.0 on that base, answering passages at least 10.8.
- **FR-K63** (Phase 8) A hybrid search fuses the words' ranking and the meaning ranking by reciprocal rank with the
  meaning arm weighted `[knowledge] vector_weight` (default 0.1) against the words' 1.0. Measured (`scripts/kb_retrieval_rank.py
  --fusion-sweep`, 50 questions that quote the passages and 20 heavily paraphrased ones, two bases of the paper): the
  words alone put the answering passage first for 86 to 88 % of the 50 and the meaning arm for 50 to 62 %, and an equal
  weight pulled the fused list below the words alone (mean reciprocal rank 0.885 and 0.854 against 0.913 and 0.895).
  Pooled over the 70 questions, weight 0.1 gave 0.844 and 0.779 against 0.800 and 0.748 at 1.0, and kept the meaning arm's
  recall (recall@8 0.98 on both bases against 0.96). Because the fusion is flat near the top (1/61 against 1/63), the
  small weight mostly breaks near-ties and reorders what the words rank far apart.
- **FR-K64** (Phase 8) English function words are left out of a words search (the words are OR-ed, so each one makes the
  store score every passage that holds it: at 50,000 passages a query with one very common word took 18 s against 1.4 s
  for the rare word alone); the query as typed is kept when nothing else is left. A words search is bounded by
  `[knowledge] search_timeout_s` (10): in a hybrid search the words are given up on and the answer is by meaning alone,
  with a note; a words-only search is told to try fewer or rarer words (429). Deleting a base removes its passages in
  batches of 500, in the background (one statement for 50,000 ran past the store client's 60-second limit; see SR-K12).
- **FR-K65** (Phase 8, NFR-K07) Optional: with `[knowledge] vector_cache_mb` above 0, a base of `exact_search_below` (5,000)
  vectors or more is searched exactly in memory. Its vectors (normalised, one contiguous array, 200 MB at 50,000
  passages) are loaded in the background on first use from the store document by document (164 to 296 s at 50,000 on the
  loaded reference machine) and searched by dot product: 18 to 22 ms median, 75 ms at worst, 100 of 100 planted vectors
  found first. The store's own HNSW index, the default, answered in 0.37 s at the median (0.93 s at the 95th percentile)
  once warm; its first queries after a cold start, and any made while other heavy queries ran, took 26 s to over 100 s.
  Until the copy is ready a search answers by words and says so. The copy is dropped when the base's vectors change (a
  document read again, vectors written, a fresh re-index, a delete) and checked against the base's vector count on every
  use. Off by default because the warm index is usable; for far larger bases or a store shared with other work.
- **FR-K66** (Phase 8, NFR-K07) In a base of 4,000 passages or more the words of a query that fill a capped probe (a fiftieth of
  the base, at least 400 passages, no scoring, about a second at worst) are left out of the words search when rarer words
  remain, and a words search is stopped by the store itself at `search_timeout_s`: with that, a rare word plus the
  commonest word ran in 4 s at the median instead of being stopped at 10 s in 7 of 10 cases, and a rare word plus a word in
  every passage's title line in 3 s instead of 5 of 5 stopped. A query of common words alone stays slow (13 s).
- **FR-K61** (Phase 22) Ingestion runs on two lanes: parsing, chunking and storing on a fast lane that sets the
  document `ready` (words searchable at once); vectors and the graph on a slow lane, one job at a time. A new upload
  never waits behind another base's embedding or extraction. Found by the live sweep: a 187-byte upload waited five
  minutes behind a 517-passage re-index. The queues are in memory, so a restart loses queued vector and graph jobs;
  `reindex` and `graph/build` restart them.
- **NFR-K11** `scripts/kb_modes_ab.py` gives every question exactly one fate (`answered`, `retrieved_not_used`,
  `not_retrieved`) and asserts the fates sum to the set.
- Specified, not shipped: **FR-K57** community summaries, **FR-K58** a reranker, **FR-K60** a confidence per relation
  with a review band, **NFR-K10** a measured ceiling, **NFR-K12** graph precision with an interval.

## 4. Non-functional requirements

| Id | Requirement | Measured how |
|---|---|---|
| NFR-K01 | Upload responds in < 1 s for files up to the limit (ingestion is asynchronous) | gateway test |
| NFR-K02 | Parsing + chunking a 20-page PDF in < 5 s on the reference machine (no embedding) | `benchmarks/kb_ingest.json` |
| NFR-K03 | BM25 search over 10,000 chunks in < 200 ms p95 | same |
| NFR-K04 | Embedding throughput reported (chunks/s); query embedding < 0.5 s | Phase 2 |
| NFR-K05 | `rag` adds < 3 s to the quick path's time to first token at 1,200 context tokens | Phase 2 A/B |
| NFR-K06 | Ingestion never runs more than one heavy step at a time and yields to chat (lower priority) | Phase 2 |
| NFR-K07 | A knowledge base of 1,000 documents / 50,000 chunks stays usable (list, search) | Phase 8, `benchmarks/kb_scale.json` (shared machine, load 20 to 130): listing 1,000 documents 0.2 to 0.4 s, counting vectors 0.06 to 0.1 s (was minutes), opening a document 0.2 to 0.5 s, search by meaning 0.37 s through the store's index once warm (22 ms from the optional in-memory copy), search by words 2 to 8 s for a rare word and about 4 s with a common word, **met for listing, counting and meaning; search by words is usable but slow under load (2 to 6 s), and a query of common words alone is stopped** |
| NFR-K08 | Memory module forward < 50 ms for N = 10,000 chunks on this CPU | Phase 4 bench: per question 49–54 ms up to 100 documents, 88 ms at 1,000, 436 ms at 10,000 (load 34); **not met at 10,000** |
| NFR-K09 | Training time per example measured and shown before a job starts (estimate for the whole run) | Phase 6: 36–94 s per question on Qwen3-1.7B (shared machine), 8–11 s on Qwen3-0.6B; shown per run on the Knowledge page |

## 5. Security requirements

- **SR-K01** Uploads are size-limited (`max_upload_mb`, default 50 per file) and count-limited (20 per request). Phase 8:
  a file is read in pieces and refused the moment it passes the limit (it is never held whole), and the request as a
  whole is limited by `max_request_mb` (default 200), not by files times file size (which let one request carry a
  gigabyte).
- **SR-K02** Files are stored under `data/runtime/kb/<kb>/<doc>.<ext>` with names the server generates; the client's
  file name is metadata only (no path traversal).
- **SR-K03** ZIP-based formats: at most 2,000 entries, at most 200 MB uncompressed in total, compression ratio per
  entry at most 200; only the XML parts needed are read. Phase 8: the headers' sizes can lie, so the bytes actually
  decompressed are summed and bounded as well (a deck of a thousand slides that each claim to be small cannot expand
  to a thousand times the cap); an XML entity bomb is not expanded; a LaTeX file reads no other file (`\input` and its
  kin are ignored); a reader that panics is that document's error.
- **SR-K04** HTML is converted to text; nothing from an uploaded document is ever rendered as HTML in the UI (escaped).
  Phase 8: scripts, styles, frames and objects are dropped, and a link whose target runs script (`javascript:`,
  `vbscript:`, `data:`) keeps its words and loses its target, so stored text is safe in a viewer that renders links.
- **SR-K05** Retrieved text and graph facts are delimited as data in the prompt (as FR-061); instructions inside documents
  are not followed.
- **SR-K06** SurrealDB listens on 127.0.0.1 only; root credentials in `data/runtime/surreal/root.secret` (mode 600),
  passed by environment, never logged; the gateway is the only client.
- **SR-K07** The knowledge API sits behind the gateway's token auth like every other `/v1` route.
- **SR-K11** (Phase 8) Resources are bounded: at most `concurrent_uploads` (2) requests are received at once, the next
  is told to try again (429, code `busy`); at most `max_queued_jobs` (500) jobs wait; a base holds at most
  `max_documents_per_base` (2,000) documents; a document makes at most `max_chunks_per_document` (6,000) passages
  (a larger one is refused with a request to split it); a parser that takes more than `parse_timeout_s` (180 s) is
  given up on; at most `max_concurrent_trainers` (1) training runs go at once across all bases.
- **SR-K12** (Phase 8) Deleting a base takes it out of use at once (it leaves lists and searches, and its name is freed) and removes
  it everywhere: its rows in SurrealDB (documents, passages, vectors, graph; in the background, in batches, resumed after
  a restart: about 16 passages a second on the loaded reference machine, so under an hour for 50,000),
  its folder (originals, training runs, activated modules), the prepared documents the memory reader keeps in
  memory, and the passages and graph quotes copied into the ledger's stored replies; a training run still going for it
  is stopped first. Deleting one document removes its originals, rows and its passages from stored replies (and the
  graph facts of that base, whose quotes cannot be traced to one document). The response reports how many replies
  were scrubbed and how many trainers stopped.
- **SR-K13** (Phase 8) What is deliberately not done: the answers a conversation received from a deleted base stay in the
  conversation (they are the user's own record, and a model's words cannot be traced to a passage); bases have no
  retention period of their own (they stay until deleted, like rooms); the upload folder is not encrypted at rest.

## 6. Data model (SurrealDB, namespace `nafs`, database `kb`)

Record ids are server-generated UUIDs without dashes (`kb:⟨32 hex⟩`). All tables are `SCHEMAFULL` except where noted.

```surql
DEFINE ANALYZER IF NOT EXISTS kb_text TOKENIZERS blank, class FILTERS lowercase, ascii, snowball(english);

DEFINE TABLE IF NOT EXISTS kb SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS name        ON kb TYPE string;
DEFINE FIELD IF NOT EXISTS description ON kb TYPE string DEFAULT "";
DEFINE FIELD IF NOT EXISTS created_at  ON kb TYPE datetime;
DEFINE FIELD IF NOT EXISTS updated_at  ON kb TYPE datetime;
DEFINE FIELD IF NOT EXISTS embedder    ON kb TYPE option<string>;   -- Phase 2: model id of the vectors
DEFINE FIELD IF NOT EXISTS dims        ON kb TYPE option<int>;

DEFINE TABLE IF NOT EXISTS document SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS kb         ON document TYPE record<kb>;
DEFINE FIELD IF NOT EXISTS title      ON document TYPE string;
DEFINE FIELD IF NOT EXISTS filename   ON document TYPE string;
DEFINE FIELD IF NOT EXISTS kind       ON document TYPE string;          -- pdf docx pptx tex html md txt
DEFINE FIELD IF NOT EXISTS bytes      ON document TYPE int;
DEFINE FIELD IF NOT EXISTS sha256     ON document TYPE string;
DEFINE FIELD IF NOT EXISTS path       ON document TYPE string;          -- stored original, relative to the repo
DEFINE FIELD IF NOT EXISTS status     ON document TYPE string;          -- queued parsing chunking storing embedding extracting ready failed
DEFINE FIELD IF NOT EXISTS error      ON document TYPE option<string>;
DEFINE FIELD IF NOT EXISTS pages      ON document TYPE option<int>;
DEFINE FIELD IF NOT EXISTS n_chunks   ON document TYPE int DEFAULT 0;
DEFINE FIELD IF NOT EXISTS n_chars    ON document TYPE int DEFAULT 0;
DEFINE FIELD IF NOT EXISTS parse_ms   ON document TYPE option<int>;
DEFINE FIELD IF NOT EXISTS created_at ON document TYPE datetime;
DEFINE FIELD IF NOT EXISTS updated_at ON document TYPE datetime;
DEFINE INDEX IF NOT EXISTS document_kb  ON document FIELDS kb;
DEFINE INDEX IF NOT EXISTS document_sha ON document FIELDS kb, sha256;

DEFINE TABLE IF NOT EXISTS chunk SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS kb        ON chunk TYPE record<kb>;
DEFINE FIELD IF NOT EXISTS doc       ON chunk TYPE record<document>;
DEFINE FIELD IF NOT EXISTS ord       ON chunk TYPE int;
DEFINE FIELD IF NOT EXISTS text      ON chunk TYPE string;
DEFINE FIELD IF NOT EXISTS heading   ON chunk TYPE string DEFAULT "";
DEFINE FIELD IF NOT EXISTS page      ON chunk TYPE option<int>;
DEFINE FIELD IF NOT EXISTS chars     ON chunk TYPE int;
DEFINE FIELD IF NOT EXISTS embedding ON chunk TYPE option<array<float>>;   -- Phase 2
DEFINE INDEX IF NOT EXISTS chunk_kb   ON chunk FIELDS kb;
DEFINE INDEX IF NOT EXISTS chunk_doc  ON chunk FIELDS doc;
DEFINE INDEX IF NOT EXISTS chunk_text ON chunk FIELDS text FULLTEXT ANALYZER kb_text BM25;
-- Phase 2, once the embedder is fixed (1024 for Qwen3-Embedding-0.6B):
-- DEFINE INDEX chunk_vec ON chunk FIELDS embedding HNSW DIMENSION 1024 DIST COSINE TYPE F32 EFC 150 M 12;

-- Phase 3
DEFINE TABLE IF NOT EXISTS entity SCHEMAFULL;      -- kb, name, key (normalised), type, description, embedding, degree
DEFINE TABLE IF NOT EXISTS mentions TYPE RELATION FROM chunk TO entity;
DEFINE TABLE IF NOT EXISTS related  TYPE RELATION FROM entity TO entity;   -- predicate, description, weight, evidence: array<record<chunk>>
DEFINE TABLE IF NOT EXISTS community SCHEMAFULL;   -- kb, level, members: array<record<entity>>, summary, embedding

-- Phase 6
DEFINE TABLE IF NOT EXISTS memory_run SCHEMAFULL;  -- kb, kind (train|finetune), decoder, encoder, status, epochs, rows, loss_curve, checkpoint, active
```

**Wire format (IF-K03).** The gateway calls SurrealDB's RPC `query` method over HTTP (`POST /rpc`, basic auth) with
bound variables, encoded as **CBOR** (`Content-Type` and `Accept: application/cbor`). JSON is not used: SurrealDB parses
JSON strings that start like a record id (`"Caption: Model ..."`) into record ids. Optional fields are written as
`$x ?? NONE` because a JSON/CBOR null is `NULL`, which `option<T>` refuses.

Search queries (Phase 1, verified against 3.3.5):

```surql
SELECT id, doc, ord, text, heading, page, search::score(1) AS score
  FROM chunk WHERE kb = $kb AND text @1,OR@ $q ORDER BY score DESC LIMIT $k;
```

Vector search (Phase 2): bases with fewer than `exact_search_below` (5,000) vectors are scanned exactly
(`vector::similarity::cosine`); larger ones use the HNSW index, where `WHERE kb = $kb AND embedding <|k,ef|> $v` was
measured to return only that base's rows. On 379 vectors both give the same top 5 on 20 of 20 questions. Hybrid search
fuses the BM25 and vector lists by reciprocal rank (k = 60) in `nafs-kb`.

## 7. Datasets for Memorilla (Phase 6)

| Kind | Collection (documents) | Question | Answer | Share |
|---|---|---|---|---|
| reconstruction | the chunks of one document (up to 64; longer documents in windows of 64) | `Restate the passage under "<heading>" that begins "<first 8 words>".` | the chunk text (<= 512 tokens) | 40 % |
| graph claim | chunks mentioning the subject + 8 random distractors | `Claim: <subject> <predicate> <object>. Is the claim supported?` | `True` / `False` (object swapped for one of the same type) | 30 % |
| synthetic QA | the knowledge base (<= 2,000 chunks) or top 256 by embedding + 64 random | written by System 2 from 1–2 linked chunks | short answer; kept if System 1 with the gold chunk agrees (judge >= 0.7 or exact match) | 30 % |

System prompt for all rows: "You are a helpful assistant." (Memorilla's default) + the memory placeholder. Validation:
10 % of questions per kind. Files: `data/runtime/kb/<kb>/memorilla/data/<kb>/{train,validation}-00000-of-00001.parquet` and
`.../embeddings/qwen3-embedding-0.6b/<kb>/<split>/{index,documents-00000,questions}.safetensors` (float16).

## 8. Inference engine (Phases 2, 4, 5)

**Embeddings (Phase 2, implemented).** `nafs-smiley-server --embedding` (:8203, `scripts/serve_smiley.sh embed`) loads
Qwen3-Embedding-0.6B (GGUF Q8_0) and serves `POST /v1/embeddings` (`{input: [..], instruction?: string}` ->
`{data: [{embedding: [..]}], usage, timings}`), computed as the final hidden state of the appended `<|endoftext|>` token
(after the output norm), L2-normalised, one text at a time in 64-token prefill chunks. Each passage is embedded with a
contextual header (`<document title> > <heading path>`, a blank line, the passage). Queries
for retrieval use `Instruct: Given a question, retrieve passages that answer it\nQuery: <q>`; documents and Memorilla
queries use no prefix. Candle's implementation of the same model is the reference (cosine >= 0.999 between the two on 32
fixed texts).

**Memory module (Phase 4, implemented).** Candle, f32, Apple Accelerate for matmuls on macOS; inputs `doc_embeds [B,N,E]`,
a padding mask, `question_embeds [B,E]`; output `[B,K,h_L]`. Parity: seeded random weights through the unmodified
reference class, three configurations with a padded row (fixtures in the crate); max abs difference 1.9e-6 (target
1e-4). The released checkpoint is optional (`convert_checkpoint.py`).

**Memory tokens in Smiley Core (Phase 5).** `Model::forward_rows` gains `inputs: Option<&[Option<&[f32]>]>`: for row `b`,
`Some(v)` (length `d_model`) replaces the token embedding. `Session` gains `generate_with_embeds` that places 16 vectors at
the positions of a placeholder token (Qwen3 `<|fim_pad|>`, id 151662, never produced by chat). The prefix store's key
includes a SHA-256 of the supplied vectors. `nafs-smiley-server`: `memory_embeds: {k: 16, d: 2048, b64: "<f32 LE>"}` on
`/v1/chat/completions`. Identity test: supplying each placeholder's own embedding row gives bit-identical logits to the
plain path.

## 9. Interfaces

### IF-K01 REST API (gateway, JSON unless stated; all behind token auth)

| Method and path | Request | Response | Phase |
|---|---|---|---|
| `GET /v1/kb` | | `{bases: [KbSummary]}` with document and chunk counts | 1 |
| `POST /v1/kb` | `{name, description?}` | `KbSummary` (201) | 1 |
| `GET /v1/kb/{kb}` | | `KbSummary` | 1 |
| `PATCH /v1/kb/{kb}` | `{name?, description?}` | `KbSummary` | 1 |
| `DELETE /v1/kb/{kb}` | | `{deleted: true, documents, chunks}` | 1 |
| `GET /v1/kb/{kb}/documents` | | `{documents: [Document]}` | 1 |
| `POST /v1/kb/{kb}/documents` | multipart, one or more `file` parts | `{documents: [Document], refused: [{filename, reason}]}` (202) | 1 |
| `GET /v1/kb/{kb}/documents/{doc}` | | `Document` | 1 |
| `DELETE /v1/kb/{kb}/documents/{doc}` | | `{deleted: true, chunks}` | 1 |
| `POST /v1/kb/{kb}/documents/{doc}/reingest` | | `Document` (202) | 1 |
| `GET /v1/kb/{kb}/documents/{doc}/chunks?offset&limit` | | `{chunks: [Chunk], total}` | 1 |
| `POST /v1/kb/{kb}/search` | `{query, k?: 8, mode?: "hybrid" \| "vector" \| "lexical"}` | `{results: [Hit], mode, took_ms, embed_ms, fallback?}` | 1 (lexical), 2 (vector, hybrid; default hybrid) |
| `POST /v1/kb/{kb}/reindex` | | `{queued}` (202): documents whose passages lack vectors | 2 |
| `GET /v1/kb/status` | | `{enabled, reachable, version, url, queue: {pending, running}}` | 1 |
| `GET /v1/kb/{kb}/graph?entity&depth&limit` | | `{nodes, edges, stats, enabled, progress}` | 3 |
| `POST /v1/kb/{kb}/graph/build?fresh` | | `{queued}` (202) | 3 |
| `GET /v1/kb/{kb}/graph/entities?q` | | `{entities}` | 3 |
| `GET /v1/kb/{kb}/graph/entities/{entity}` | | `{facts, passages}` | 3 |
| `POST /v1/kb/{kb}/memory/runs` | `{kind: "train"\|"finetune", epochs?, from?}` | `MemoryRun` (202) | 6 |
| `GET /v1/kb/{kb}/memory/runs` | | `{runs, active, trainer}` | 6 |
| `POST /v1/kb/{kb}/memory/runs/{run}/stop`, `.../activate` | | the run / `{active}` | 6 |

`Document`: `{id, kb, title, filename, kind, bytes, status, error?, pages?, n_chunks, n_chars, parse_ms?, created_at,
updated_at, duplicate?}`. `Hit`: `{chunk_id, doc_id, doc_title, ord, heading, page?, text, score, rank}`.
Errors: `{error: {code, message}}` with 400 (bad input), 404, 409 (name taken), 413 (too large), 415 (unsupported kind),
503 (`kb_unavailable`: SurrealDB down or knowledge disabled).

### IF-K02 Configuration (`[knowledge]` in `config/nafs-aql*.toml`)

```toml
[knowledge]
enabled = true
url = "http://127.0.0.1:8300"     # SurrealDB HTTP
namespace = "nafs"
database = "kb"
user = "root"
password_file = "data/runtime/surreal/root.secret"
files_dir = "data/runtime/kb"
max_upload_mb = 50
max_files_per_request = 20
chunk_chars = 1200
chunk_overlap = 150
search_k = 8
embed_url = "http://127.0.0.1:8203"   # Phase 2; "" = words only
embed_model = "Qwen3-Embedding-0.6B"
embed_dims = 1024
embed_batch = 8
exact_search_below = 5000
rag_k = 4
rag_max_chars = 4800
rag_min_similarity = 0.45
graph_llm_url = "http://127.0.0.1:8201/v1"   # Phase 3; "" = graph off
graph_model = "system1"
graph_max_tokens = 400
entity_merge_similarity = 0.92
graph_facts = 12
graph_weight = 1.0                     # FR-K50: the graph's rank as an arm beside the search's (0 = facts only)
graph_beam = 8                         # FR-K56: relations kept per entity at the second hop
graph_decay = 0.6                      # FR-K56: a path's score decays by this per hop
memory_checkpoint = ""                 # Phase 5; "" = memory mode off
memory_placeholder = "<|fim_pad|>"
memory_trainer = "rust/target/release/memorilla-train"   # Phase 6
memory_decoder = "models/hf/Qwen3-1.7B"
memory_gguf = "models/Qwen3-1.7B-Q4_K_M.gguf"
hybrid_k = 2                           # Phase 7: passages beside the memory tokens in "hybrid"
auto_order = ["rag"]                   # Phase 7: the modes "auto" tries, in order (measured)
```

## 10. Frontend (Knowledge page, `#/knowledge`)

Rail entry **Knowledge** (book icon) between Rooms and the Advanced group, visible in default mode.

- **Left column:** knowledge bases (name, document count), **New knowledge base** (name, optional description).
- **Header of the selected base:** name, description, counts (documents, passages, characters), Rename, Delete (dialog
  naming what is lost: documents, passages and stored files).
- **Upload:** a drop zone and a file button (multiple; accepts `.pdf .docx .pptx .tex .html .htm .md .markdown .txt`); per
  file a progress bar for the upload, then the document's state from polling (every 1.5 s while any document is not
  `ready`/`failed`); refused files listed with the reason.
- **Documents:** table of title (file name below), type badge, size, passages, state (spinner, Ready, or Failed with the
  reason on hover), added; row actions: View passages, Re-read, Delete.
- **Passages dialog:** the document's chunks in order with heading path and page, paged by 50.
- **Search:** a box, results as cards (passage text with the query terms highlighted after escaping, document title,
  heading, score), click opens the passage in its document.
- **Empty and error states:** knowledge disabled ("Knowledge is turned off in the configuration"), database unreachable
  ("The knowledge store is not running. Start it with `scripts/serve_surreal.sh start`"), empty base ("Drop documents here").
- **Phase 2+:** chat composer gets a knowledge picker (base + mode) and replies get a **Sources** card list.
- **Phase 3+:** a Graph tab; **Phase 6+:** a Memory tab (train, fine-tune, runs, loss chart, active checkpoint).

Plain language: passages (not chunks), knowledge base, "reading", "ready". Copy-lint must pass.

## 11. Tests and acceptance per phase

| Phase | Tests | Acceptance |
|---|---|---|
| 0 | `scripts/serve_surreal.sh spike` | PASS on this machine |
| 1 | unit: each parser on a fixture (PDF from `papers/`, generated DOCX/PPTX/HTML/TEX/MD), zip limits, chunker (bounds, order, heading path, overlap); integration (skips without a server): schema, create/upload/search/delete; gateway routes; UI copy-lint | a PDF, DOCX, PPTX, TEX and HTML file uploaded through the API are `ready`, searchable, and deleting removes their chunks |
| 2 | embedding parity Candle vs Smiley Core; hybrid search unit tests; chat `rag` end to end | recall@5 and latency stored; NFR-K04, K05 |
| 3 | extraction schema, resolution, traversal | graph built for the fixture set; `graph` answers with facts |
| 4 | parity vs PyTorch | max abs error < 1e-3 |
| 5 | identity logits test; prefix-store key test | memory mode answers end to end |
| 6 | dataset builder layout readable by `memorilla.data`; trainer loss decreases on a toy set | seconds per example measured; one module trained for a fixture base |
| 7 | paired A/B | per-mode quality and latency with CIs |
| 8 | security tests for SR-K01..K07 and SR-K11..K13 | `scripts/kb_security_sweep.py` (36 checks, `--chat` for the ledger checks) on its own gateway with small limits (`scripts/kb_security_gateway.sh`), unit tests for the parsers and the scrub; runbook; report |
