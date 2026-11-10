# Knowledge module, Phase 25: in-process index, model-free graph, measured retrieval, interface (specification)

Version 1.0, 2026-11-10. Research and assessment: [`knowledge_sota_2026.md`](../research/knowledge_sota_2026.md).
Continues [`knowledge_base_spec.md`](knowledge_base_spec.md) (FR-K1 to FR-K66, NFR-K1 to NFR-K12, SR-K1 to SR-K13) and
[`graph_rag_upgrade_spec.md`](graph_rag_upgrade_spec.md). New ids: FR-K67 to FR-K82, NFR-K13 to NFR-K16, UX-1 to UX-21.

**Ground rule, unchanged:** four CPU cores, no GPU, one static binary, no model call added to a search, nothing leaves the
machine. **Claim rule:** "state of the art" is used only for a number measured here against a published number on the
same task. No public benchmark was downloaded for this phase (downloads need the owner's approval), so this document
claims *measured improvements over the module as merged at `380fc72`* on the project's own corpora, and says in §7 what
would be needed to claim more.

## 1. Strategy

The assessment (research note §1) found seven weaknesses, W1 to W7. The literature (§2 to §4 there) says where the
accuracy and the time of a retrieval system on a CPU are won:

1. **Take the store off the search path** (W1, W2). The store stays the source of truth; a search reads memory.
   BM25S reports a single-thread in-process index at about 2 ms a query where a server takes 50; here the server took
   seconds under load.
2. **Give every base a graph without a model** (W3, W4). The measured gain of graph retrieval comes from Personalized
   PageRank over a passage-and-entity graph (HippoRAG 2 ablations; LinearRAG; KET-RAG), not from typed relations. Typed
   relations with quotes stay as the optional, model-built layer a person reads.
3. **Decide fusion, field weights, feedback and the graph arm by measurement** on a question set large enough to
   separate two settings (W6, W7), with a harness that makes a sweep cost milliseconds.
4. **Never embed the same text twice** (W5).
5. **Fix what loses a person's work in the interface** before adding anything to it.

## 2. Requirements implemented: index and retrieval

### FR-K67: an in-process index per base

`nafs_kb::index::Index` holds, for each base in use, a `Base` made of one `Part` per document. A part holds the
document's passages (id, order, heading path, page, context line, text), an inverted index over their words, their
vectors (unit length, one row per passage) and the terms each passage names (FR-K71).

- **Source of truth.** SurrealDB. The index is built from `Store::index_rows` (one request for a base under 5,000
  passages, one per document above) and is never written to disk (FR-K77 specifies a file; not built).
- **Loading.** On first use, and for every base when the service starts. A base under 5,000 passages is read while the
  first search waits; a larger one in the background while searches use the store path and say so.
- **Memory.** `[knowledge] index_memory_mb` (default 512; 0 turns the index off). A base whose estimate
  (passages × (4 × dimensions + 4,000) bytes) exceeds the limit is never loaded and uses the store path. When the bases
  held exceed the limit the least recently used one is dropped.
- **Following the store.** Every write goes through the service, which updates the one document concerned: after a
  read or re-read (`refresh_doc`), after each batch of vectors (`apply_vectors`, no store read), after a graph pass
  wrote context lines, on deletion of a document or base, and on a fresh re-index (dropped, read again on next use). A
  `Base` is immutable; a change builds a new one that shares the untouched parts, so a search in flight keeps a
  consistent view.
- **Races.** Each base has a generation counter raised by every change. A load that started before a change does not
  install what it read; it reads again (three attempts).
- **Self-healing.** At most every 30 s per base, off the search path, the index's per-document passage and vector
  counts are compared with the store's; a document that differs is refreshed, one the store no longer has is removed.
  This covers writes by another process and any change the service missed.
- **Failure.** A failed refresh drops the base's index; the next search reads it again or uses the store path.

Tests: `index::tests` (10 unit tests), `tests/index_it.rs` (against a live store: same first hit as the store path,
upload, re-read, deletion and base deletion followed).

### FR-K68: analyzer and BM25F

- **Analyzer.** Lowercase, ASCII folding, English Snowball stems (`rust-stemmers` 1.2, pure Rust), function words
  dropped (the list of FR-K64): what the store's analyzer did. **Added:** an identifier is indexed whole beside its
  parts, so `vector_weight`, `Qwen3-1.7B`, `FR-K63` and `1.7` are terms of their own (the store's `class` tokenizer cut
  them into `vector`, `weight`, `qwen`, `3`, `1`, `7`, `b`, which match everywhere); CamelCase is split (`SurrealDB` →
  `surreal`, `db`, `surrealdb`); a possessive is its word.
- **Scoring.** BM25F (Robertson, Zaragoza, Taylor 2004) over four fields: document title, heading path, context line,
  body. Per term: `tf' = Σ_f w_f · tf_f / (1 − b + b · len_f / avglen_f)`, `score += idf · tf' · (k1 + 1) / (k1 + tf')`,
  `idf = ln(1 + (N − df + 0.5) / (df + 0.5))`, `k1 = 1.2`, `b = 0.5` (measured, §6.3; the textbook value is 0.75). Field term frequencies are kept in the postings,
  so weights are swept without rebuilding. `[knowledge] bm25_weights` sets them; the default is whatever §6 measured
  best.
- **Cost.** Document frequency is summed over the parts at query time (one hash lookup per part per term); scores
  accumulate in one array per base.

### FR-K69: the query path

- A search of an indexed base makes **no round trip to the store**: not for the vector count, not for common-term
  probes (FR-K66 is not needed in memory), not for the hits' text.
- The question's vector is remembered (256 questions, least recently used out). A repeated question embeds once, and
  graph mode no longer embeds the question a second time.
- Every hit carries both scores when they exist: the words' score and the cosine to the question, for *every* hit
  (the store path gave a hit only the score of the arm that found it). The chat filter (FR-K62) therefore tests the
  true similarity of a passage the words found.
- `POST /v1/kb/{kb}/search` takes `"engine": "store"` to answer through the store as before Phase 25 (for comparison and
  as the fallback), and every result says which engine answered (`"engine": "index" | "store"`). The search handler
  skips its existence check on the store when the base is indexed.
- `GET /v1/kb/status` reports the index: bases held, passages, megabytes, limit.

### FR-K70: fusion decided by measurement

Two fusions are implemented over the first `max(3k, 20)` candidates of each arm: weighted reciprocal rank (FR-K63, the
vector arm at `vector_weight`) and a convex combination of min-max normalised scores (Bruch et al. 2022, the vector arm's
share `fusion_alpha`). `[knowledge] fusion = "rrf" | "convex"`. The default is whatever §6 measured best.

### FR-K71: a passage-and-term graph with no model call

Built per base from the parts, in memory, when the base is loaded or changes (milliseconds; see §6).

- **Terms** are read from each passage's text and heading by rule: acronyms (`HNSW`, `BM25`), identifiers (a digit, an
  underscore, an inner capital, a dotted or hyphenated name with a capital or digit: `Qwen3-1.7B`, `vector_weight`,
  `SurrealDB`, `app.js`), runs of up to four capitalised words with an attached number (`Smiley Core`, `Phase 7`), and
  single capitalised words. A capitalised word that opens a sentence counts only if the base also has it capitalised
  mid-sentence; sentence openers and function words are never part of a name. Headings contribute acronyms and
  identifiers only (a heading is in title case throughout).
- **Nodes and edges.** A term is a node when at least two passages name it and at most a fifth of the base (20 at
  least) does. A passage is linked to each term it names, weighted by the term's rarity `ln(1 + N / df)`.
- **Why no typed relations.** HippoRAG 2's ablation loses 6.1 points of recall@5 without passage nodes and 0.7 without
  its model filter; LinearRAG and KET-RAG reach the same answers with no model at index time. The typed, quoted
  relations of FR-K30 to FR-K56 stay as the layer a person reads and the prompt cites; they are no longer the only way a
  base gets a graph.
- **View.** `GET /v1/kb/{kb}/graph` returns this graph (`"source": "text"`: the most named terms, linked when passages
  name both, the edge weight being the number of such passages and `evidence` one of them) when no model-built graph
  exists, or when asked with `?source=text`. `?entity=<term>` centres it on a term.

### FR-K72: Personalized PageRank as a retrieval arm

`FastGraph::ppr`: power iteration (20 rounds) over the bipartite graph. The walk starts at the best passages of the fused
search (rank-weighted, `seeds` = 10) and at the terms the question names (longest phrase first, weighted by rarity,
`entity_share` of the seed mass), follows passage-term links, and returns to the seeds with probability
`1 − damping`. Mass of a passage with no term returns to the seeds. The passages' visiting probabilities are one more
ranking, fused with the search's by rank at `[knowledge] graph_arm_weight` (0 = off). As FR-K50 found for the
model-built graph, the arm **supplies candidates**; it is never the order by itself.

In chat's `graph` mode, when the model-built graph names nothing in the question (or does not exist), the walk over the
text graph supplies the candidates instead of falling back to the plain search.

### FR-K73: a re-read keeps the vectors of unchanged passages

Before a re-read replaces a document's passages, the vectors of its embedded passages that have no context line are
read with the text they were embedded as (heading and body; the title must be unchanged). New passages whose text is
identical get their vector back at once; only changed passages go to the embedding server. Passages with a context line
are not reused (a re-read clears the line, so the embedded text differs).

### FR-K74: a self-check of the readers

`cargo run --release -p nafs-kb --example parse_agreement -- <file> <files...>` reads the same document from two formats
and reports the share of distinct words and of three-digit-or-longer numbers of each reading found in the other. No
public score exists for the PDF reader's reading order or for the TeX stripper; this is the check that needs no
download.

### FR-K75: an evaluation harness and question sets that can separate two settings

- `cargo run --release -p nafs-kb --example kb_eval -- --kb <id> --set name=file ... [--store] [--latency N] [--out f]`
  reads the base once into the index, embeds each question once (cached in `data/runtime/kb_eval_qvec.json`), and ranks
  every question under every setting: field weights, `k1`/`b`, both fusions over a grid, feedback terms, the graph arm
  over weight, damping and seed share. It reports recall at 1, 4, 8 and 20 and the mean reciprocal rank; for two-passage
  questions the mean over both passages and the share with *both* in the first 4, 8 and 20. `--store` ranks through the
  store path too; `--latency N` times N questions through both paths, interleaved.
- Question sets over the 847-passage paper base (`scripts/kb_p25_*.json`, checked by `scripts/kb_p25_check.py` against a
  snapshot of the passages): **60 quoted** (the document's own terms), **146 reworded** (mean share of the question's
  content words found in the answering passage 0.09), **50 two-passage** (two phrase lists that no single passage
  satisfies). Every accepted phrase is verbatim in at most three passages.
- One answer is 0.007 of recall on the reworded set (it was 0.05 on the 20-question set).

### FR-K76: feedback terms without a model (measured)

`Prf { passages, terms, weight }`: the best terms (frequency × rarity) of the first passages join the query at a
fraction of a query term's weight (RM3-like). Implemented in the index and swept by the harness; on or off by §6.

## 3. Requirements implemented: interface

The audit ranked 21 defects (research note §6). Each is a requirement; what was done is in §6.

| Id | Requirement |
|---|---|
| UX-1 | An answer in flight survives leaving Chat and coming back. |
| UX-2 | A streaming answer is rendered at most every 150 ms, with one full render at the end. |
| UX-3 | The Rooms access key is not sent in a URL (needs a gateway change; see §4). |
| UX-4 | An upload shows one row per file with its stage, can be cancelled, is checked for size and type before it is sent, and keeps its errors on screen with a retry. |
| UX-5 | `[n]` in an answer opens source *n* with a preview of the passage. |
| UX-6 | The documents table is updated row by row; typed text, results and the graph's focus survive a refresh. |
| UX-7 | A route change moves focus to the page heading and is announced; unknown and Advanced-only routes are handled. |
| UX-8 | Conversations in the list are buttons; deletion is reachable from the keyboard and can be undone. |
| UX-9 | Messages queue; an error stays until dismissed and does not look like a success. |
| UX-10 | No raw status code or exception text reaches a person. |
| UX-11 | Stop leaves a neutral note and keeps the partial answer; the message list is a log region. |
| UX-12 | Regenerate replaces the answer and keeps its settings. |
| UX-13 | The graph can be zoomed, panned and read: stable positions, a legend, "Showing N of M". |
| UX-14 | A search result opens its passage in context; "Ask about this" starts a chat on it. |
| UX-15 | Expensive actions ask first. |
| UX-16 | Text contrast meets 4.5:1. |
| UX-17 | The layout uses the dynamic viewport height and keeps its controls on a narrow screen. |
| UX-18 to UX-21 | Rooms roles and knowledge picker; Lab and Quality failure states; dialogs and hidden-tab polling; wording, admin-only commands, duplicate ids. |

Rules that stay: no build step, no external request, plain language (the copy lint), nothing from the server put into
the page unescaped.

## 4. Requirements specified and not built, with the reason

- **FR-K77: an index file per base.** Passages and vectors written beside the stored originals, validated by a stamp
  (document count, passage and vector counts, the base's last change), so a restart loads without reading the store.
  Not built: §6 measured the load from the store; at the sizes this machine holds it is seconds, and a second copy on
  disk is one more thing that can disagree with the store. Build when a base of 50,000 passages is in daily use.
- **FR-K78: questions written per passage at index time** (doc2query, HyPE). Targets the reworded-question gap at zero
  query cost. Not built: one model call per passage (17 to 28 s here), so 847 passages are 4 to 7 hours of the quick
  model on a machine at load 100; the measurement needs a quiet night. The graph pass already makes a call per passage
  and writes a context line; the questions would ride the same call.
- **FR-K79: a batched forward in the embedding server.** The server embeds one text at a time on one engine thread.
  Not built: an engine change (`smiley-core`) whose gain is at indexing time only; a re-read no longer embeds unchanged
  passages (FR-K73). The other half of this requirement, keeping the instruction's cache rows between questions, was
  built once §6.1 showed the embedding to be all that is left of a search (FR-K79a there).
- **FR-K80: an instant static-embedding tier, FR-K58: a reranker, learned sparse vectors.** Each needs model weights
  the owner has not approved.
- **FR-K81: public benchmarks** (NanoBEIR, SciFact, the HippoRAG 2 multi-hop subsets, MultiHop-RAG). The harness reads
  any `[question, [phrases]]` file; the corpora need a download the owner has not approved.
- **FR-K82: binary codes with rescoring** for bases above a few hundred thousand passages. Not needed below that
  (an exact scan of 100,000 × 1,024 took 20.8 ms).
- **UX-3** needs the Rooms WebSocket to take the key as its first frame: a protocol change with its own security review.

## 5. Non-functional requirements

| Id | Requirement | Verified by |
|---|---|---|
| NFR-K13 | A search of an indexed base of up to 5,000 passages costs under 5 ms beyond the question's embedding | `kb_eval --latency`, §6 |
| NFR-K14 | The indexes never exceed `index_memory_mb`; a base that does not fit is served by the store path | `Index::get`, `evict` |
| NFR-K15 | A deleted document or base is not served by the index after the delete call returns | `tests/index_it.rs` |
| NFR-K16 | Every number in §6 is reproducible with one command and labelled `SHARED MACHINE` | `benchmarks/kb_phase25/` |

## 6. Measurements and what they decided

All on the reference machine (4-core Intel i3-8100, no GPU), **SHARED MACHINE**, load average 6 to 8 unless a row says
otherwise (the machine had just restarted; the Phase 25 baseline in the research note was taken at load 24 to 129).
Base: the 847-passage paper base (9 documents, 847 vectors of 1,024 dimensions). Files: `benchmarks/kb_phase25/`.
Reproduce: the `kb_eval` command of FR-K75 with `--store --latency 40`.

### 6.1 Time

| What | Store path | Index | Note |
|---|---|---|---|
| Search by words, median (p90) | 177 ms (221) | **0.086 ms** (0.100) | 40 questions, interleaved |
| Hybrid search, median (p90), old embedding server | 740 ms (822) | 351 ms (427) | the index's 351 ms is 350 ms of question embedding |
| Hybrid search, median (p90), embedding server with the kept prefix (FR-K79a) | 578 ms (649) | **152 ms** (181) | 151 ms of it is the embedding |
| The hybrid search itself, without the embedding call | n/a | **0.17 ms** | 800 searches |
| The same with the graph arm (Personalized PageRank, 20 rounds) | n/a | 0.60 ms | |
| Question embedding, median, 30 questions, interleaved | 297 ms | **146 ms** | same vectors (cosine 1.000000 on all 30) |
| A repeated question | embeds again | 0 ms (remembered) | |
| Reading the base from the store and indexing it | n/a | 0.7 to 1.1 s | 847 passages, 5 MB held |
| Building the passage-and-term graph | 17 to 28 s **per passage** (model-built, Phase 21) | **under 1 ms for the base** | 1,306 terms, 5,975 links |

End to end, a hybrid search went from 740 ms to 152 ms at load 7 (4.9 times), and the part that is not the embedding
model from several hundred milliseconds to a fifth of a millisecond. Under the load of the baseline (24 to 129) the store
path took 5.3 s at the median; the index does not wait on the store, so its time under load is the embedding's.
NFR-K13 (under 5 ms beyond the embedding): met, 0.17 ms.

**FR-K79a, built after the measurement pointed at it.** With the store off the path, the question's embedding was 99.9 %
of a hybrid search. The embedder now keeps the previous text's tokens and cache rows and forwards only what differs
(`smiley-core/src/embed.rs`): every question starts with the same instruction, about half of a short question's tokens.
Attention is causal, so the kept rows are what the text would compute again; the vectors are identical (cosine 1.000000,
30 questions, and in mixed order with passages). The batched forward (FR-K79) stays unbuilt.

### 6.2 Accuracy (256 questions; R@k = answering passage within the first k; MRR = mean reciprocal rank)

| Set | Path | R@1 | R@8 | R@20 | MRR | both@8 |
|---|---|---|---|---|---|---|
| Quoted (60) | store, words | 0.867 | 1.000 | 1.000 | 0.917 | |
| | index, words | 0.883 | 1.000 | 1.000 | 0.926 | |
| | store, hybrid | 0.767 | 1.000 | 1.000 | 0.855 | |
| | **index, hybrid (default)** | **0.817** | 1.000 | 1.000 | **0.882** | |
| Reworded (146) | store, words | 0.048 | 0.199 | 0.322 | 0.105 | |
| | index, words | 0.062 | 0.205 | 0.315 | 0.112 | |
| | store, hybrid | 0.096 | 0.226 | 0.336 | 0.147 | |
| | **index, hybrid (default)** | **0.116** | **0.240** | 0.329 | **0.160** | |
| | index, vector alone | 0.062 | 0.137 | 0.192 | 0.088 | |
| Two-passage (50) | store, words | 0.330 | 0.810 | 0.890 | 0.510 | 0.64 |
| | index, words | 0.310 | 0.820 | 0.880 | 0.497 | 0.66 |
| | store, hybrid | 0.300 | 0.800 | 0.890 | 0.485 | 0.62 |
| | **index, hybrid (default)** | 0.260 | **0.830** | 0.890 | 0.460 | **0.68** |

Pooled over the 256 questions the hybrid MRR is 0.379 through the store and 0.388 through the index. One question is
0.017 of a 60-question set and 0.007 of the 146: **the index is as accurate as the store path within what these sets
resolve, and faster by the factors of §6.1.** It is not measurably more accurate.

### 6.3 What the sweep decided (59 settings, `eval_sweep.json`)

| Question | Result | Decision |
|---|---|---|
| Length normalisation `b` | 0.5 put the answer first for 53 of 60 quoted questions against 50 at 0.75; reworded and two-passage within one question | **`b = 0.5`** |
| `k1` | 0.9, 1.2, 1.6, 2.0 within two questions of each other | stays 1.2 |
| Field weights (title, heading, context from 0 to 3) | every setting within one question of flat weights on every set | **flat**; BM25F is kept (the weights cost nothing), the gain the literature reports did not appear on this base |
| Fusion: convex combination against reciprocal rank | convex α=0.3: quoted MRR 0.912, reworded 0.134, two-passage 0.493 (pooled 0.387); reciprocal rank w=0.1: 0.882, 0.160, 0.460 (pooled 0.388) | **not adopted.** Convex is better on quoted and two-passage, worse on reworded, equal pooled; the incumbent stays. `fusion = "convex"` is there for a base whose questions quote it |
| Vector weight in reciprocal-rank fusion | 0 → 0.1 → 0.3: quoted MRR 0.926, 0.882, 0.806; reworded 0.112, 0.160, 0.173 | **stays 0.1** (FR-K63 confirmed on a set seven times larger) |
| Feedback terms (FR-K76) | 5 passages, 8 terms, 0.3: quoted R@1 0.883 → 0.750, reworded MRR 0.112 → 0.120 | **off** |
| Graph arm (FR-K72) at weight 0.25, 0.5, 1.0 | first 8 unchanged on every set (both@8 0.68 with and without); reworded R@20 0.329 → 0.363 to 0.377 with passage seeds alone | **off by default** (`graph_arm_weight = 0`); used in `graph` mode. It supplies candidates below rank 8, which only a reranker would use (FR-K58) |
| The words' score scale (`rag_lexical_rescue` = 10.0) | 20 off-topic questions: best passage scores at most 8.9 (p90 8.2); answering passages of quoted questions score at least 15.6 at p10 | **stays 10.0** |

An earlier run at `b = 0.75` showed the graph arm raising both@8 on two-passage questions from 0.62 to 0.70. At `b = 0.5`
the plain search already finds those passages (0.68) and the arm adds nothing in the first 8: the gain was the
baseline's weakness, and is reported here so that it is not rediscovered.

### 6.4 The readers (FR-K74)

The technical report read as PDF (pdf_oxide, 1.36 s) and as its 34 TeX files (own stripper, 49 ms): **98.0 % of the
numbers and 91.6 % of the distinct words of the TeX reading are in the PDF reading.** The words that are not are
macro and label names the TeX stripper lets through inside mathematics and tables (`dfrac`, `cdot`, `allowbreak`,
`columnwidth`, table label names): a defect of the stripper, harmless to search (nobody asks for them) and recorded as
open. In the other direction 72 % of the PDF's words and 44 % of its numbers are in the TeX reading: the PDF also holds
the bibliography, the generated tables and the page furniture, which the TeX sources pull from other files.

### 6.5 Interface

| Item | State |
|---|---|
| UX-1, 2, 4 to 11, 13, 14, 20, 21 | done |
| UX-15 | done, except that a running graph build has no stop (needs a route) |
| UX-16 | done, except white on teal in the policy-map pills |
| UX-12 | partly: asking again keeps the mode and the knowledge base; it still adds a second answer (replacing in place needs a route) |
| UX-17 | partly: dynamic viewport height, safe-area padding, the documents table below 560 px; the theme switch on a phone is in Settings only |
| UX-18 | partly: list roles fixed, reasoning follows the setting; no knowledge picker in Rooms (needs a field in the room message) |
| UX-19 | mostly: failures offer "Try again", cards work from the keyboard, refresh waits while the tab is hidden; cards are still rebuilt |
| UX-3 | not done (§4) |

Added beyond the audit: a "No strong match" note when the best passage of a knowledge answer is below the chat filter's
own floor (similarity 0.45) or the answer has no sources; a note under the graph when it is the one read from the text.
Checked in a browser on the new build: an answer in flight survived leaving Chat and returning; `[4]` in an answer
opened its passage with "Open in the document"; search as you type answered in 39 ms and a result opened its passage
highlighted in context; no console error on Chat or Knowledge. The e2e sweep (68/68) loads every page. Uploads, Rooms,
Lab and Quality changes were not exercised by hand.

## 7. What this does and does not establish

- **Established here:** a search of an indexed base no longer waits on the store (three orders of magnitude on the
  words, 4.9 times end to end on a hybrid search at low load, more under load); every base has a graph in under a
  millisecond with no model call; a re-read embeds only changed passages; the settings in force were chosen on 256
  questions rather than 20 to 70, and six candidate improvements were measured and not adopted.
- **Not established:** that retrieval is more accurate than before (it is equal within the sets' resolution), or that
  any number here is state of the art against published systems. Published in-process BM25 (BM25S) reports about 1 to
  2 ms a query on 57,000 passages; 0.086 ms on 847 passages is consistent with that and is not a comparison on the same
  corpus.
- **The open problem is the reworded question.** The answering passage is first for 12 % of them and within the first
  8 for 24 %. The words cannot find a passage that shares no word with the question, and the 0.6B embedding model
  alone ranks the answering passage first for 6 %. What the literature offers needs the owner's go-ahead: a reranker
  or a stronger embedder (weights), questions written per passage at index time (FR-K78: hours of the quick model),
  and public benchmarks to compare on (FR-K81: downloads).
