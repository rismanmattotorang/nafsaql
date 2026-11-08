# OntoSmith read against Nafs-Aql: what to take, what we already do better, and the gaps worth closing

2026-11-08. Source: `third_party/ontosmith` at b1007d6 (MIT, Dongbum Lee), read in full — the pipeline, the retrieval
stack, the harness and the whitepaper. Written for Phase 21 of the Knowledge track. Companion specification:
[`graph_rag_upgrade_spec.md`](../spec/graph_rag_upgrade_spec.md).

This note is a comparison, not a review. OntoSmith solves a different problem from ours — it is a *governed* graph
builder for corpora that have to be audited, where we are an assistant that answers from a person's own documents on a
4-core CPU. Several of its choices would be wrong for us, and several of ours are already ahead of it. What follows
separates the three cases: what we should copy, what we should not, and where we are behind.

## 1. What OntoSmith is

An ontology-first knowledge-graph builder. The order of operations is the product:

1. **Profile** the corpus deterministically (no model calls), producing a census and the decisions a human should
   settle.
2. **Design** an ontology — entity and relation types with definitions and boundary examples — which a human approves
   and which ships as a hash-pinned **pack**: domain content as data, never as code branches.
3. **Build** under that pack through eight stages: ingest → NER/roster → candidates → gates → route → write → index →
   verify.
4. **Verify** with tooling shipped as part of the product: raw and audit-corrected scoring, ceiling-adjusted recall, a
   failure anatomy, and counted cost projections.
5. **Serve** from Neo4j: sentence-level evidence retrieval, and a chunk-level hybrid API underneath.

The invariant that organises all of it (whitepaper §3.2): **nothing is written on the proposing seat's say-so alone.**
A low-cost proposer extracts candidate triples with evidence spans and has no write authority; deterministic gates
check that the claimed quote occurs verbatim and that the form conforms; a pinned judge model decides; a routing stage
tiers what is written by threshold. Between the accept and reject thresholds sits a deliberate human review band.

Its published record is unusually honest, and worth reading as a model of how to report: corrected precision 0.9110
[0.8919–0.9275] against ceiling-adjusted recall 0.2028 [0.1856–0.2237] on Re-DocRED FULL-500, with a failure anatomy
that assigns each of 17,284 gold triples **exactly one fate** (matched 1,838; emission_blocked 1,759; none_judged
3,677; never_enumerated 6,340; roster_missed 3,670 — 57.9 % never enumerated at all). One pre-registered criterion is
recorded as FAILED rather than quietly dropped.

## 2. Where Nafs-Aql is already ahead

Read against the code rather than the prose, several of our choices are the stronger ones. This matters: the gap list
in §4 is shorter and sharper because of it.

| | OntoSmith | Nafs-Aql |
|---|---|---|
| Lexical arm | **No BM25 in the product.** The file store scores `Σ(1 + ln tf)` over distinct query terms and says in its own comment that it is "a monotone stand-in … not a claim of Lucene's scores". The only BM25 in the tree is in an evaluation harness. | Real BM25 in the store (`DEFINE INDEX … FULLTEXT ANALYZER kb_text BM25`), with a snowball-stemmed, ASCII-folded analyzer. |
| Chunking | `FixedWindowChunker(size=1000, min_chars=100)` — fixed 1000-**character** windows, **no overlap**, no structural boundary. A sentence straddling a window is split in two. | Heading-aware chunking with 1,200 characters and 150 of overlap, carrying the Markdown heading path and the page. |
| Vector index | A `sentence_embedding` vector index is created, populated — and **never queried**. Sentence search loads every sentence into memory and runs a full numpy matmul. No HNSW `M`/`ef*` is ever set; filtered vector search is a post-filter with a fixed ×5 over-fetch. | HNSW in SurrealDB (`EFC 150 M 12`, cosine), with an exact scan below `exact_search_below` so small bases are exact and large ones are approximate. |
| Fusion in the served path | **No RRF** in the sentence path: a two-layer z-score sum with a hard 2× weight on sentences. RRF appears only in the chunk-level fallback and in the lab. | RRF (k = 60) over the lexical and vector lists, the published-standard rank fusion. |
| Embedding | Hosted `text-embedding-3-large` only; `default_embedder()` refuses by name when the key is unset. No local path, by design. | Qwen3-Embedding-0.6B served locally by our own engine — no key, no network, and it runs on this CPU. |
| Served surface | A read-only, GET-only sample API on `127.0.0.1:8765`, **no auth, no TLS, no POST, no pagination, no rate limit**, and the sentence path is unreachable over HTTP. | The gateway's authenticated, rate-limited, TLS-capable API, with the same modes reachable from chat and from the API. |

Two further facts worth recording because they bound what their numbers mean. Their retrieval figures come from a
*lab* stack (`pipeline/query_stack.py`), not from the code that serves `ontosmith search`; the beam channel, the
z-score merge and the rerank that the served path uses **have no measured number anywhere in the tree**. And the
measurement is document-level recall@20 over one 116-question set that is not in the repository, so it is a record,
not something a reader can reproduce — their own `reproduce/README.md` classifies it that way.

## 3. What is genuinely better there, and why

### 3.1 The unit of retrieval is a sentence, not a passage

We retrieve 1,200-character passages and hand the model whole passages, each truncated to `rag_max_chars / k`
characters. With eight passages and the default 4,800-character budget that is 600 characters each — and the sentence
that answers the question may be the one cut off. OntoSmith retrieves **sentences**, and scores each by its own
cosine *plus its owning chunk's* cosine, so a sentence inherits the context that made its passage relevant without
spending the budget on the rest of the passage.

### 3.2 One scale, and a greedy packer instead of equal shares

Their merge is two z-score layers summed (`z_sentence + z_chunk`, sentences weighted 2×), then a fill under one token
budget for every query — whole units only, an over-budget unit **skipped rather than ending the scan**, and a chunk
dropped when every sentence of it has already been taken. Per-layer z-normalisation is the part worth copying: it lets
two scores on different scales compete without a tuned weight.

### 3.3 The graph supplies candidates; it does not re-rank them

This is the one that matters most to us. In OntoSmith, a triple's evidence quote is resolved to its owning chunk and
that chunk is **injected into the dense arm's candidate population** before scoring. The graph *grows the pool*; it
does not re-order someone else's result.

Ours does the opposite, and it is measurable. `Kb::graph_search` hard-sorts the retrieved passages by how many seed
entities each mentions, breaking ties by the search order. A passage that holds the answer but names none of the
question's entities falls below passages that name several. That is the mechanism behind our Phase 7 result: `graph`
answered 15 of 20 against `rag`'s 18, and all three discordant pairs went the same way — `graph` won none of them.
The three it lost ("how many cross-attention refinement blocks", "how many hops does local search walk", "which
database does the conversation store use") are exactly passage-level facts whose passage was displaced.

### 3.4 Evidence is a quote, and the quote is verified

Every written triple carries the sentence that states it, and a deterministic gate rejects the candidate if that
sentence does not occur verbatim in the document. We store the *passage id* a relation came from, which is enough to
find the evidence and not enough to check it. A verbatim span gate is cheap, needs no model, and turns "the extractor
said so" into "the document says so".

### 3.5 Domain content is data, not code

Our entity types are a `[&str; 8]` array in `graph.rs`, and anything the model writes is mapped onto them. OntoSmith's
types, their definitions and their boundary examples live in a versioned pack that is injected into the prompt and
enforced after it. This is also the one mechanism the 2026 literature measures a gain for: OMD-GraphRAG's
ontology-guided extraction injects the schema into the prompt and discards type-violating triples, and reports a
15.15 % improvement over LightRAG on one query type (arXiv 2603.25152 §3.1).

### 3.6 Failure anatomy, not just a score

Assigning each gold item exactly one fate — and reporting the counts — turns "recall is 0.20" into "57.9 % was never
enumerated, and here is the stage that lost it". We report recall@5 and leave the loss unattributed.

## 4. What we should *not* copy

- **The pinned judge model.** A second model call per candidate triple is unaffordable here: our graph build already
  costs one quick-model call per passage and took 9.5 minutes for 20 passages on a loaded host. The *idea* — a
  score per triple and a threshold — is worth having; a separate pinned model is not.
- **The human approval gate on the ontology.** Right for an audited corpus, wrong for a person dropping five files
  into a chat app. We should make the ontology data and let a person edit it, without blocking the build on approval.
- **Their beam traversal.** Per-hop cosine pruning with no cumulative path score, no edge weight and no decay — and
  the path structure is discarded when the traversal becomes evidence. We already carry edge weights, so copying this
  would be a downgrade; their own measured arm is the weighted PPR in the lab stack, not the beam they ship.
- **Dropping BM25 and the overlap in chunking.** Both of ours are better; the comparison table above says why.

## 5. The state of the art, measured

Searched 2026-11-08. The figures that bear on us:

- **Reranking is the single largest reported jump.** Anthropic's contextual-retrieval study reports top-20 retrieval
  failure falling 5.7 % → 3.7 % with contextual embeddings, → 2.9 % adding contextual BM25, → 1.9 % adding a
  cross-encoder reranker. We have the first, not the second, and not the third.
  ([Anthropic](https://www.anthropic.com/news/contextual-retrieval))
- **Contextual BM25 is half of that result and we are not taking it.** We build a contextual header (document title
  and heading path) and use it only when *embedding*; the BM25 index stores the raw chunk text.
- **Ontology-guided extraction is the measured graph-build lever** (OMD-GraphRAG, arXiv
  [2603.25152](https://arxiv.org/abs/2603.25152)); the same paper adds attribute-aware clustering and a query-complexity
  weight that fuses entity-level and community-level scores.
- **Query decomposition helps multi-hop, measured with a reranker**: +36.7 % MRR@10 and +11.6 % F1 on MultiHop-RAG and
  HotpotQA (arXiv [2507.00355](https://arxiv.org/abs/2507.00355)). The gain is reported for decomposition *and*
  reranking together, so it cannot be attributed to decomposition alone.
- **Rank fusion is the production baseline** and RRF is the usual choice because it works on ranks rather than scores
  ([hybrid-search reference](https://www.digitalapplied.com/blog/hybrid-search-bm25-vector-reranking-reference-2026)) —
  which we already do.
- A caution worth recording: one 2026 review reports that GraphRAG's win rates over vector RAG shrink after correcting
  for LLM-judge position and length artifacts. Our own Phase 7 measurement already points the same way, and OntoSmith's
  own scope note is blunter still — its graph advantage is +0.080 recall@20 on long contracts and **exactly zero** on
  short abstracts.

## 6. The measurement methodology, which is the strongest part of the project

Read separately, because it is transferable whatever we do with the graph. I replayed their keyless record
(`examples/demo/replay_scores.py`): it exits 0 and reproduces all 8 point estimates, 8 intervals and 5 fate rates
under both verdict sets. These are the practices worth taking:

1. **An A/B/C audit design.** Set A samples *unmatched predicted* records (is the system right where the answer key is
   silent?), set B samples *gold* records (do the key's own items exist in the text?), and **set C samples matched
   records as an honesty control on the panel** — "if the panel cannot confirm known matches, the panel, not the
   system, is the problem." Set C is the piece most harnesses omit, and it is what lets their record say "blind
   controls 37/40" instead of assuming the judge is competent.
2. **Quote verification as a verdict gate, with the failures published.** A `supported` verdict counts only when its
   quote occurs verbatim in the context; otherwise it is non-decisive, excluded from every denominator — and the
   exclusion count ships (11 of 740 under the ruled contract, 32 under the original). Deterministic, cheap, and it is
   what makes a machine verdict auditable at all.
3. **Correct for the reference, not only for the system.** Ceiling-adjusted recall is
   `matched / (gold_total x gold_support_rate)`, where the support rate of the answer key itself is *measured* by
   sampling it (0.5244 here). That turns "recall is 0.106" into "0.2028 against a measured ceiling of 0.52". We grade
   against our own answer keys as if they were complete.
4. **Localise disagreement; do not average it.** When a pre-registered cross-family agreement check failed its bar
   (0.815 against 0.95), they did not split the difference: the disagreements were one-directional and concentrated in
   six recurring surface situations, which were settled as written definitions, shipped verbatim as the judge's prompt,
   with the deciding rule recorded per verdict.
5. **Re-publish the worse number when a gate fails.** That ruling moved corrected precision from 0.9765 to 0.9110, and
   both are computed by the same estimator in the same script, the original kept as provenance. One pre-registered
   criterion (C-2, NONE-loss share <= 0.50, measured 0.5128) stands as FAIL in the record rather than being
   reinterpreted.
6. **Fail closed everywhere.** A missing metric is a gate failure, not a silent pass; an unpriced unit refuses the
   projection rather than costing $0; a drift audit that compared zero items fails rather than passes; an anatomy table
   whose counts do not sum refuses to render.
7. **A spend ledger with a double guard.** Append-only JSONL, one line per unit with a running total; refuse to *start*
   a unit whose estimate would cross the stop line, and raise if the actual cost crosses anyway, so a run "fails loud
   rather than recording spend off-book"; resume from the ledger by the ids that completed.
8. **Every headline in one of three categories** — recompute without a key, recompute with your own run, or record
   (measured once, the data ships, the run is history). Forcing the classification is what makes unbacked claims
   visible.

### 6.1 Three places where the prose outruns the code

Read against the code, three of OntoSmith's headline claims are not what ships. They are worth recording because two
of them are things this specification implements, so we would be copying a description rather than a system.

- **"Every written triple carries its evidence span and provenance" is true of the batch's JSONL artefacts, not of
  the served graph.** The registry's `Triple` has no `quote` or surface fields, and the Neo4j writer says so in its
  own comment; the graph schema *supports* a quote and the write path does not populate it. `inspect` reads the
  files, never the store. Our FR-K55 stores the verified quote on the relation itself, which is what the claim
  describes.
- **The judge never sees the ontology.** `validate_triple` is sent the document window, the subject, the relation
  name and the object — no definition, no boundary example, no endpoint signature. The pack reaches the proposer and
  the pair judge only, both on the proposer seat. Whitepaper §3.1 says the judge scores "against the pack's
  definitions"; it does not.
- **The flagship pack does not run the configuration its record describes.** The `open-domain` pack, which the
  flagship Re-DocRED record is about, declares no `pair_enumeration`, so a build under it takes the free-proposer
  branch rather than the roster-first sentence-window pair enumeration the record was measured with. Only
  `bio-medical` turns it on.

Also inert, and worth knowing before treating any number as the behaviour of the shipped code: `er_threshold = 0.40`
— quoted in the whitepaper as *the* entity-resolution threshold — is declared in three places and read nowhere; the
alias generator that the adopted operating point depends on exists only in the harness, with a docstring saying no
production module imports it.

### 6.2 The invariants worth copying whatever else we do

From the build pipeline, independent of its architecture:

- **Every drop is counted.** Duplicates, unattested surfaces, span voids each go to a counted sidecar, and an
  accounting identity (`proposed = items + off_anchor + evidence_missing`) is asserted.
- **Ambiguity is reported, never guessed** — at every level, from resolution to routing's review band.
- **Fail closed on every load**: an unpinned pack, an unknown payload key, a declared-but-missing artefact or a
  declared-but-unwired judge refuses the run. There is no degraded mode.
- **Cost is assignment, not accumulation**, so a resumed stage cannot double count, and an unknown cost stays unknown
  all the way to the total rather than becoming zero.

Their own practice falls short of this in places, and it is worth naming so we do not copy the gap: the flagship's
pre-registered gates ship as a Markdown table rather than as the `GateDefinition` data their own library defines, so
nothing evaluates them in CI even though three of the six metrics sit in a shipped JSON file. `verify project` does
not price anything despite two documents saying it does. And the baseline their headline "+0.080" rests on appears
only in prose, not in the conformance record.

## 7. Sources

- OntoSmith, `third_party/ontosmith` at b1007d6 — `docs/whitepaper.md`, `docs/serving.md`, `docs/PACK_DESIGN.md`,
  `docs/ADJUDICATION.md`, `pipeline/`, `harness/`, `reproduce/`.
- Anthropic, *Introducing Contextual Retrieval* — https://www.anthropic.com/news/contextual-retrieval
- OMD-GraphRAG — https://arxiv.org/abs/2603.25152
- *Question Decomposition for Retrieval-Augmented Generation* — https://arxiv.org/abs/2507.00355
- *Hybrid Search: BM25, Vector & Reranking Reference 2026* —
  https://www.digitalapplied.com/blog/hybrid-search-bm25-vector-reranking-reference-2026
- *RAG Chunking Strategies: A 2026 Retrieval Playbook* —
  https://www.digitalapplied.com/blog/rag-chunking-strategies-2026-retrieval-quality-playbook
