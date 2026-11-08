# Knowledge retrieval and graph: the gaps, and the specification that closes them (Phase 21)

Version 1.0, 2026-11-08. Research: [`ontosmith_comparison.md`](../research/ontosmith_comparison.md). Builds on the
Phase 20 specification [`knowledge_base_spec.md`](knowledge_base_spec.md) (FR-K, NFR-K, SR-K); new requirement ids
continue in the same families (FR-K50…, NFR-K10…, SR-K10…).

This specification is written against two things: OntoSmith read in full (`third_party/ontosmith` at b1007d6), and the
2026 measured literature on retrieval. It closes the gaps those two expose, in the order their measured value
justifies, and it says where a gap is left open and why.

**The ground rule.** Every item here must be affordable on the reference machine: 4 cores, no GPU, with the owner's
other work sharing them. Anything needing a model call per candidate triple, or per retrieved passage, is specified
only if it fits inside a call we already make.

## 1. What we measured, and what it says

Phase 7 measured the modes against each other on 20 held-out questions over the fixture base
(`benchmarks/kb_modes_ab.json`): `rag` 18/20, `graph` 15/20 at 1.28x the time, closed-book 2/20. `graph` won **none**
of the three discordant pairs. The three it lost are passage-level facts whose passage was displaced from the prompt.

That is a mechanism, not noise, and §3.1 fixes it.

## 2. The gap table

| # | Gap | Evidence it matters | Closes in |
|---|---|---|---|
| **G-1** | The graph **re-ranks** retrieval instead of **supplying candidates** to it | Our own A/B: graph lost 3 of 3 discordant pairs, all passage-level facts | FR-K50 |
| **G-2** | The lexical index stores raw chunk text; the contextual header is used only for embedding | Anthropic: contextual BM25 takes top-20 failure 3.7 % → 2.9 % | FR-K51 |
| **G-3** | Passages reach the prompt truncated to `rag_max_chars / k` — the answering sentence can be the part cut | OntoSmith retrieves sentences; our budget is split blind | FR-K52 |
| **G-4** | No per-passage context line beyond title and heading | Anthropic's contextual retrieval writes one per chunk; we can get it free inside the extraction call we already make | FR-K53 |
| **G-5** | Entity types are a `[&str; 8]` array in `graph.rs`; domain content is code | OntoSmith packs; OMD-GraphRAG measures +15.15 % on one query type from ontology-guided extraction | FR-K54 |
| **G-6** | A relation stores the *passage* it came from, not the *sentence*, and nothing checks the sentence exists | OntoSmith's span gate; the single cheapest trust mechanism in their build | FR-K55 |
| **G-7** | The graph walk is one hop from at most five seeds, unweighted | OntoSmith's measured arm accumulates graph mass *per document*; their shipped beam has no path score and is the weaker design | FR-K56 |
| **G-8** | Communities are detected and never used | Global questions ("what does this base cover?") have no path | FR-K57 |
| **G-9** | No reranker | The single largest measured jump in the 2026 literature: 2.9 % → 1.9 % | FR-K58 |
| **G-10** | A reply does not say which retrieval arms ran | OntoSmith names every arm and writes a note for each that did not | FR-K59 |
| **G-11** | Retrieval is graded against our own question set as if it were complete | Ceiling-adjusted recall; our recall@5 has no measured ceiling | NFR-K10 |
| **G-12** | No per-triple confidence, no acceptance threshold, no review band | OntoSmith's judge; adapted to one call, not two | FR-K60 |

## 3. The specification

### 3.1 Retrieval

- **FR-K50 — the graph supplies candidates.** `graph` and `auto`-resolved graph retrieval must not reorder the
  passages the search returned. The graph's contribution is: (a) the passages its facts cite are **added** to the
  candidate pool if the search did not already find them, and (b) each candidate gains a graph signal — the number of
  seed entities it mentions, and the summed weight of the facts that cite it. The final order is one fused score over
  the retrieval rank and the graph signal, not a sort by the graph signal alone.

  Concretely, for candidate *c* with retrieval rank *r(c)* (absent ⇒ the depth bound) and graph mass
  *m(c)* = Σ weight of facts citing *c*, plus *e(c)* seed entities mentioned:

  ```
  score(c) = 1 / (60 + r(c))  +  w_graph · ( 1 / (60 + rank_by(m, e)(c)) )
  ```

  i.e. a second RRF list built from the graph signal, fused with the retrieval list by the same rule we already use.
  `w_graph` defaults to 1.0 (an equal arm) and is `[knowledge] graph_weight`. Rank fusion rather than score addition,
  because the two signals have no common scale — the same reason RRF is the production standard.

- **FR-K51 — contextual BM25.** A passage is indexed for full-text search as the same text it is embedded as: the
  document title, the heading path, then the passage body (and, with FR-K53, its context line). A stored
  `search_text` field carries it; the BM25 index moves to that field. Re-indexing an existing base backfills it.

- **FR-K52 — sentence units and a budget packer.** `prompt_block` stops splitting `rag_max_chars` evenly. Instead:
  1. every retrieved passage is split into sentences;
  2. each sentence is scored by its own relevance to the question **plus its passage's retrieval score**, both
     z-normalised over their own population, so the two compete on one scale without a tuned weight;
  3. units are filled greedily under one character budget for the whole block: whole sentences only, an over-budget
     unit **skipped rather than ending the fill**, a passage kept whole when its sentences would fill it anyway;
  4. a passage keeps its number and its source line, so citations still work.

  Sentence relevance is computed without a model call by default (term overlap with the question, idf-weighted over
  the retrieved set). When `[knowledge] sentence_embed` is on, the sentences of the retrieved passages are embedded in
  one batched call at query time — tens of sentences, not the whole base, so no new index is built and no ingestion
  cost is added.

- **FR-K53 — a context line per passage, at no extra model call.** The graph extraction call already reads every
  passage. It additionally returns `context`: one sentence, in the document's own terms, saying what the passage is
  about and what it belongs to. It is stored on the chunk, prepended to `search_text` (FR-K51) and to the embedded
  text, and shown nowhere to the person. A passage whose context line is missing or ungrounded falls back to the
  title-and-heading header, exactly as today.

- **FR-K58 — reranking (specified, not shipped in Phase 21).** A cross-encoder rerank of the fused candidate head,
  before the budget fill. The model is a reranker of the Qwen3 family served by Smiley Core in a new `--rerank` mode:
  the engine already reads logits for named tokens (the Memorilla trainer does it for `True`/`False`), so the server
  work is a prompt template and a two-token read, not a new kernel. It is **off by default**, named in the reply's
  arms when it runs and in the notes when it does not, and adopted only if a paired A/B on the question set shows a
  gain that justifies its latency. Deferred because it needs a model download and a measured adoption, not because it
  is hard.

- **FR-K59 — arms and notes.** Every knowledge reply states which arms ran (`lexical`, `vector`, `graph`, `memory`,
  `rerank`) and carries a note for each arm that did not, naming why. This replaces today's single `fallback` string.

### 3.2 The graph

- **FR-K54 — the ontology is data.** A knowledge base carries an ontology: entity types and relation types, each with
  a one-line definition and up to two boundary examples. It is stored with the base, editable, and versioned by a
  hash recorded on every extraction. The extraction prompt is **built from it**, and a candidate whose type is not in
  it is mapped onto one that is, or dropped. `graph.rs`'s `TYPES` array becomes the *default* ontology for a new base,
  not a constant in the code path. A base may adopt a different ontology and re-read its documents under it.

- **FR-K55 — evidence is a verified quote.** The extraction call returns, for every relation, the sentence of the
  passage that states it. A deterministic gate checks that the sentence occurs in the passage **verbatim** after
  whitespace normalisation; a relation whose quote fails the gate is dropped, and the count of drops is recorded on
  the document. The quote is stored on the relation and shown with the fact in the Graph card and in chat sources.

- **FR-K60 — a score per relation, and a review band.** The extraction call returns a confidence in `[0, 1]` per
  relation. Combined with the deterministic signals we already have (grounding, repeat count across passages, quote
  verification), it gives `p_true`. Relations with `p_true >= accept` are used in answers; those in
  `[review, accept)` are stored and shown in the Graph card as unconfirmed, and are **not** put into a prompt; below
  `review` they are dropped. Thresholds are `[knowledge] graph_accept` and `graph_review`, and their defaults are set
  from the measurement in §4, not chosen by feel.

- **FR-K56 — weighted multi-hop traversal.** `facts_for` grows a bounded walk: from the seed entities, up to
  `[knowledge] graph_hops` hops (default 2), keeping at each hop the best `graph_beam` relations (default 8 per
  frontier entity, 24 per hop) scored by

  ```
  path_score = Π over the path of ( weight_norm · p_true · decay^hop )
  ```

  with `decay` 0.6. A fact reaching the prompt carries its path, so a two-hop fact reads as the chain that produced
  it. This is deliberately **not** OntoSmith's beam, which prunes on cosine alone with no path score, no edge weight
  and no decay, and which discards the path when it turns a traversal into evidence.

- **FR-K57 — community summaries and a global route (specified, not shipped in Phase 21).** Each Louvain community
  gets a one-paragraph summary written by the quick model, stored and refreshed when the community changes. A
  question the gate classes as global ("what does this base cover", "summarise X") is answered from the summaries of
  the communities nearest the question rather than from passages. Deferred: one model call per community is the same
  order of cost as the graph build itself, and it needs its own measurement.

### 3.3 Measurement

- **NFR-K10 — a measured ceiling.** The retrieval question set is itself audited: a sample of its questions is checked
  for whether the base supports the expected answer at all, and recall is reported both raw and ceiling-adjusted,
  `matched / (total × support_rate)`, with the support rate's interval carried through. A question whose answer the
  base does not support is a property of the set, not a failure of retrieval.

- **NFR-K11 — a failure anatomy.** Every question of the set gets **exactly one** fate, and the fates sum to the set
  size, checked by assertion: `answered`, `retrieved_not_used` (the passage was in the prompt, the answer was not in
  the reply), `not_retrieved` (the passage exists and was not retrieved), `not_in_base`. This says which stage lost
  the question.

- **NFR-K12 — graph precision with an interval.** A sample of written relations is judged against its own evidence
  quote, blind to how it was written, with a control stratum of relations that repeat across passages. Precision is
  reported as a rate with a Wilson interval and the count of non-decisive items published beside it, never folded in.

- **SR-K10 — the quote gate is a security control too.** The verbatim check of FR-K55 bounds what a document can put
  into a prompt through the graph: a relation can only ever carry text that occurs in the passage it came from.

## 4. What Phase 21 implements, and what it defers

**Implemented now:** FR-K50, FR-K51, FR-K52, FR-K53, FR-K54, FR-K55, FR-K56, FR-K59, NFR-K11.

**Specified and deferred, with the reason:**

| Item | Why not now |
|---|---|
| FR-K58 reranker | Needs a model download and a measured adoption; the engine work is specified |
| FR-K57 community summaries | One model call per community, on a host where the graph build already costs 9.5 min for 20 passages |
| FR-K60 review band | Needs the measurement of §4 to set its thresholds; the confidence field and the storage ship now, the band's defaults follow |
| NFR-K10 ceiling, NFR-K12 graph precision | Both need an audit pass over a sample; the harness lands with the measurement, not before it |

## 5. Acceptance

Phase 21 is done when, on the fixture base and the same 20 held-out questions:

1. `graph` is **not worse** than `rag` (the Phase 7 regression is closed), measured paired with an exact McNemar test;
2. every relation in the graph carries a verified quote, and the count of quote-gate drops is recorded;
3. the ontology of a base is data that a reader can print, and changing it changes the extraction prompt;
4. the prompt block fills a budget with whole sentences and never truncates mid-sentence;
5. a reply names the arms that ran;
6. the failure anatomy accounts for every question exactly once;
7. the Rust suite, `fmt`, `clippy -D warnings`, `cargo deny` and the artefact checks are clean, and the knowledge
   sweep passes.
