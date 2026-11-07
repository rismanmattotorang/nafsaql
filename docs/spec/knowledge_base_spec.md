# Knowledge Base, GraphRAG and Memorilla: technical specification (Phase 20)

Version 1.2, 2026-11-07. Status: Phases 0–3 implemented; Phases 4–8 specified. Research:
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
- **FR-K23** `memory`: question embedded without a prefix; the memory module turns the whole store into 16 vectors;
  the engine receives them at 16 placeholder positions; the prompt keeps Memorilla's format.
- **FR-K24** `hybrid`: `memory` plus the top 2 passages of `rag`. `auto`: chosen by the measured policy of Phase 7.
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
  safetensors with `config.json`; parity with PyTorch (§8).
- **FR-K41** Dataset builder per knowledge base: reconstruction rows, graph-claim rows, synthetic QA rows (§7), split
  90/10 by question; written in Memorilla's parquet + safetensors layout.
- **FR-K42** Train (from scratch, `retrieval_init` on) and fine-tune (from a checkpoint) a module for System 1, as a
  background job with progress, loss curve, cancel and resume; one checkpoint per epoch; the best by validation loss is
  marked active for that knowledge base.
- **FR-K43** A GPU recipe (`training/memorilla/`) to pre-train Stages 1–2 for our encoder/decoder pair with the
  reference PyTorch code, with a cost estimate derived from measured per-example times.

## 4. Non-functional requirements

| Id | Requirement | Measured how |
|---|---|---|
| NFR-K01 | Upload responds in < 1 s for files up to the limit (ingestion is asynchronous) | gateway test |
| NFR-K02 | Parsing + chunking a 20-page PDF in < 5 s on the reference machine (no embedding) | `benchmarks/kb_ingest.json` |
| NFR-K03 | BM25 search over 10,000 chunks in < 200 ms p95 | same |
| NFR-K04 | Embedding throughput reported (chunks/s); query embedding < 0.5 s | Phase 2 |
| NFR-K05 | `rag` adds < 3 s to the quick path's time to first token at 1,200 context tokens | Phase 2 A/B |
| NFR-K06 | Ingestion never runs more than one heavy step at a time and yields to chat (lower priority) | Phase 2 |
| NFR-K07 | A knowledge base of 1,000 documents / 50,000 chunks stays usable (list, search) | Phase 8 |
| NFR-K08 | Memory module forward < 50 ms for N = 10,000 chunks on this CPU | Phase 4 bench |
| NFR-K09 | Training time per example measured and shown before a job starts (estimate for the whole run) | Phase 6 |

## 5. Security requirements

- **SR-K01** Uploads are size-limited (`max_upload_mb`, default 50 per file) and count-limited (20 per request).
- **SR-K02** Files are stored under `data/runtime/kb/<kb>/<doc>.<ext>` with names the server generates; the client's
  file name is metadata only (no path traversal).
- **SR-K03** ZIP-based formats: at most 2,000 entries, at most 200 MB uncompressed in total, compression ratio per
  entry at most 200; only the XML parts needed are read.
- **SR-K04** HTML is converted to text; nothing from an uploaded document is ever rendered as HTML in the UI (escaped).
- **SR-K05** Retrieved text and graph facts are delimited as data in the prompt (as FR-061); instructions inside documents
  are not followed.
- **SR-K06** SurrealDB listens on 127.0.0.1 only; root credentials in `data/runtime/surreal/root.secret` (mode 600),
  passed by environment, never logged; the gateway is the only client.
- **SR-K07** The knowledge API sits behind the gateway's token auth like every other `/v1` route.

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

**Memory module (Phase 4).** Candle, f32; inputs `doc_embeds [1,N,1024]`, `question_embeds [1,1024]`; output `[1,16,h_L]`.
Parity: the released Stage 3 checkpoint (h_e 2560, h_L 4096) converted to safetensors; 4 random inputs with N in {1, 7,
64, 512}; max abs difference to PyTorch < 1e-3.

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
| `GET /v1/kb/{kb}/memory/runs`, `.../{run}`, `POST .../{run}/cancel` | | | 6 |

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
| 8 | security tests for SR-K01..K07 | runbook, report |
