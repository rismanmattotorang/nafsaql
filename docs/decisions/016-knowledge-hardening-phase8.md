# Decision 016: Hardening the knowledge base, closing the track (Phase 8 of the Knowledge track)

Date 2026-11-08. Status: **adopted; implemented and swept (36 of 36 security checks).** Specification:
[`docs/spec/knowledge_base_spec.md`](../spec/knowledge_base_spec.md) v1.9 (SR-K11 to SR-K13, FR-K62). Sweep:
`scripts/kb_security_sweep.py`, run by `scripts/kb_security_run.sh` on its own gateway with small limits.

## Context

Phase 8 of the track is "security review of uploads, retention and deletion of a knowledge base everywhere, operator
runbook, report section, CI". The review read the upload path (gateway handler, `Kb::upload`, the parsers), the
delete paths, the memory reader's caches, the training-run supervisor and the ledger, looking for what a hostile file,
a careless client or an ordinary deletion leaves behind. Most of SR-K01 to SR-K07 held (stored names are the content
hash, the API never shows a server path, the type is sniffed, ZIP parts are counted and ratio-checked). Eight things did
not hold or were missing.

## What the review found, and what was done

| # | Finding | Fix |
|---|---|---|
| 1 | The gateway buffered each uploaded file whole (`field.bytes()`) and only then compared it with the 50 MB limit; the request limit was files times file size, so one request could hold a gigabyte | Read in pieces and refuse at the limit; the request is limited by `max_request_mb` (200); `concurrent_uploads` (2) requests are received at once, the next gets 429 `busy` |
| 2 | ZIP expansion was checked against the headers' sizes, which can lie; a deck of many slides could expand far past 200 MB in total | The bytes actually read are summed and bounded (`Zip::read` against a budget) |
| 3 | A web page's `javascript:` link survived into the stored text as a Markdown link | `drop_unsafe_links` keeps the words and drops script targets |
| 4 | Nothing bounded the ingestion queue, the documents in a base, the passages one document makes (a 50 MB text file is 40,000 passages and hours of embedding), or a parser's running time | `max_queued_jobs`, `max_documents_per_base`, `max_chunks_per_document`, `parse_timeout_s`; a PDF reader's panic is that document's error |
| 5 | Deleting a base left its training run going for hours in a folder that no longer existed | `stop_trainers` asks every run of the base to stop and ends the trainer process (after checking its command line) |
| 6 | The memory reader kept the deleted base's prepared documents (vectors derived from its passages) in memory until restart | `Knowledge::forget` drops them |
| 7 | A stored reply keeps the passages and graph quotes it was answered from, so the ledger kept the text of a deleted base for the 90-day retention period (93 replies on the reference machine) | `Store::scrub_kb_sources` removes them from every reply of the base, or of one deleted document, and leaves a note |
| 8 | Nothing limited training runs across bases; each holds 8 to 11 GB | `max_concurrent_trainers` (1) |

Two further defects were found by the same work and are fixed: a vector write that met a transient SurrealDB
transaction conflict failed the document (`set_vectors` now retries, 6 attempts with a doubling wait), and the chat
relevance filter dropped a strong word match that the vector arm had missed (FR-K62, below).

## Decisions

| Question | Decision | Why |
|---|---|---|
| Scrub or delete the replies? | Scrub the `sources` only; leave the conversation and the model's words | The passages are copies of the user's documents; the answer is the user's record and cannot be traced to one passage |
| Retention for bases? | None of their own: they stay until deleted | Like rooms; the retention purge is for conversations |
| Encrypt the upload folder? | No | Out of scope on a single-user machine; the folder is `0700`-style by the OS user |
| The relevance filter | Keep a passage by similarity **or** by a strong word match (`rag_lexical_rescue`, 10.0) | On the paper base vector search alone left 4 of 20 answering passages outside its top 20; the words ranked one of them first and the similarity test dropped it. Off-topic questions scored at most 8.0 against at least 10.8 |
| A reranker (FR-K58) | Not built | On both paper bases the answering passage is in the hybrid top 4 for 19 of 20 questions (recall@1 0.90); a reranker could move one question of twenty, which no question set of this size can detect |

## Consequences

- Eight `[knowledge]` settings with defaults (`max_request_mb`, `concurrent_uploads`, `max_documents_per_base`,
  `max_chunks_per_document`, `max_queued_jobs`, `parse_timeout_s`, `max_concurrent_trainers`, `rag_lexical_rescue`),
  reported by `GET /v1/kb/status`; a new error code `busy` (HTTP 429).
- `DELETE /v1/kb/{kb}` and `DELETE /v1/kb/{kb}/documents/{doc}` report `replies_scrubbed` (and `trainers_stopped`).
- The queues are still in memory (a restart loses queued vector and graph jobs); that is a documented limit, not a
  hardening gap.
- Assumptions A-131 to A-134.
