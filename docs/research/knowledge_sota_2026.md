# Knowledge module: where it stands and what the state of the art says (Phase 25)

2026-11-10. This note is the research the Phase 25 specification ([`knowledge_sota_spec.md`](../spec/knowledge_sota_spec.md))
rests on: (1) an assessment of the Knowledge module as merged at `380fc72`, (2) what the 2024–2026 literature and the
Rust ecosystem say about building vectors and graphs from documents, (3) the same for vector, hybrid and graph retrieval,
(4) an audit of the web interface. Marking: **V** = the source page was opened for this note, **S** = a search snippet
only, **R** = recalled and not re-checked, **M** = measured on this machine. Nothing here is a measurement of Nafs-Aql
unless it is marked M; the measurements of what was built are in the specification's acceptance table and in
`benchmarks/kb_phase25/`.

**Machine.** 4-core Intel i3-8100, no GPU, shared (load average 24 to 130 during the baseline, swap almost full): every
time is **SHARED MACHINE**.

## 1. Assessment of the module as merged

Pipeline: parse (pdf_oxide; own readers for DOCX, PPTX, HTML, TeX, Markdown) → Markdown-aware chunks of 1,200
characters with 150 overlap and a heading path → SurrealDB (BM25 full-text index over title + heading + context line +
body; HNSW over 1,024-dimension vectors of Qwen3-Embedding-0.6B) → hybrid search by weighted reciprocal-rank fusion
(vector weight 0.1), a lexical rescue and a similarity floor for chat, an optional graph arm (entities and typed
relations extracted by the quick model, two-hop weighted walk, Louvain communities), a sentence packer.

What is good and stays: accuracy work is measured (203-question gate set, 70-question fusion sweep, negative results
kept); every relation carries a verbatim quote; a document is searchable by its words seconds after upload; limits,
time-outs and a security sweep exist.

What the measurements below found weak:

| # | Finding | Evidence (M, SHARED MACHINE) |
|---|---|---|
| W1 | **Every search is several round trips to the store, and under load each costs seconds.** | 443-passage base, load 24–129: words 1,158 ms median (max 4.8 s); meaning 2,151 ms (query embedding 779 ms of it); hybrid 5,307 ms (max 11 s). At 50,000 passages (Phase 23) a words query took 2 to 8 s and needed a probe per term to drop common words. |
| W2 | **A hybrid search runs its three steps one after another** (count the vectors, embed the query, vector search, then words). | `Kb::search`, `lib.rs:1094`. |
| W3 | **The graph costs a model call per passage**, so it is opt-in and most bases never get one. | 17 to 28 s per passage (Phase 21); 9.5 minutes per 20 passages. |
| W4 | **The graph arm widens the pool by a fixed two-hop walk over typed relations**; it answered 15 of 20 against 18 for plain retrieval when it ranked alone, and is not in the default order. | Phase 7, Phase 21. |
| W5 | **The embedding server embeds one text at a time on one thread;** a re-read embeds every passage again even when its text did not change. | `nafs-smiley-server/src/embed.rs`, `Kb::ingest`. |
| W6 | **The paraphrase gap.** Words find a quoted fact (MRR 0.909) and miss a reworded one (0.450); the meaning arm alone is weak (R@1 0.25 on 20 paraphrased questions). | Phase 24, decision 017. |
| W7 | **Twenty paraphrased questions cannot separate two settings** (one answer is 0.05 of R@1), and there is no multi-hop set at all. | `scripts/kb_paper_questions_para.json`. |

## 2. Building the index: parsing, chunking, vectors

**Parsing.** pdf_oxide 0.3.78 (in use) reports 0.8 ms mean per document and a 100 % pass rate on 3,830 PDFs, against
2.8 ms / 95.1 % for unpdf and 4.08 ms / 91.5 % for pdf_extract (V, self-reported,
<https://github.com/yfedoseev/pdf_oxide>). It publishes no reading-order or table score. Two pure-Rust, model-free
readers do: edgeparse-core 0.3.2 (reading order 0.886, tables 0.564, headings 0.552, about 0.02 s per document; V,
<https://github.com/raphaelmansuy/edgeparse>) and pdf-inspector 1.25.2 (0.915 / 0.814 / 0.788; V,
<https://github.com/firecrawl/pdf-inspector>). Everything stronger needs PDFium or ONNX Runtime (pdfium-render, liteparse,
Xberg, ferrules), or is AGPL (mupdf): excluded by this project's rules (pure Rust, one static binary, permissive
licences). No maintained LaTeX-to-Markdown crate exists (tree-sitter-latex is generated C; texlab is GPL); no DOCX reader
publishes an accuracy figure. **Reading:** the readers stay; what can be added cheaply is a *self-check* of what they
produce (the same paper as PDF and as TeX must yield the same facts) and structure the chunker already uses.

**Chunking.** Fixed-size chunks beat semantic chunking on real (unstitched) documents: HotpotQA F1@5 90.59 against 87.37
and 84.79 (V, Qu et al., <https://arxiv.org/abs/2410.13070>). Chroma's study puts the best recall at 200 to 400 tokens and
finds no recall gain from overlap (V, <https://www.trychroma.com/research/evaluating-chunking>); ours are about 300 tokens.
Late chunking gives +1.8 nDCG@10 on the models it was built for and collapses on others (BGE-M3 0.246 → 0.070; V,
<https://arxiv.org/abs/2504.19754>); untested on last-token pooling, not adopted. Contextual retrieval (a context line per
passage in both arms) cuts top-20 failures from 5.7 % to 2.9 % (V, <https://www.anthropic.com/news/contextual-retrieval>);
already shipped as FR-K51/K53. **Reading:** the chunker is where the literature says it should be; chunk size was already
swept here (Phase 23, not adopted).

**Vectors.** Qwen3-Embedding-0.6B scores 61.83 on MTEB English v2 retrieval (V, model card). Static embedding tables
(potion-retrieval-32M 35.06, static-retrieval-mrl 34.95 on MTEB retrieval; V) are a hundred times faster and sit between
BM25 and MiniLM: a possible instant tier, but a 125 MB download that the owner has not approved, and a weaker arm than
the one fusion already weights at 0.1. Binary codes keep 96.45 % of retrieval quality with rescoring at 32× less memory
(V, <https://huggingface.co/blog/embedding-quantization>). An exact f32 scan of 100,000 × 1,024 took 20.8 ms here in numpy
(M); the project's own cache measured 18 to 22 ms at 50,000 (Phase 23). **Reading:** below a few hundred thousand passages
an exact in-memory scan is both the fastest and the most accurate index; an ANN library adds nothing at this scale.

**Words.** BM25S answers 953 queries a second on SciFact and 507 on FiQA (57,000 passages) on one thread, against 21 and
17 for Elasticsearch (V, <https://arxiv.org/abs/2407.03618>): about 2 ms where the store here takes seconds. BM25 variants
(Robertson, ATIRE, Lucene, BM25+, BM25L) differ by ±0.4 nDCG@10; stemming adds 1.3; stop words 0.1 (V, same). **Reading:**
an in-process inverted index is a latency change of three orders of magnitude and no accuracy risk if the analyzer is kept.

## 3. Retrieval: fusion, expansion, reranking

- **Fusion.** A convex combination of min-max-normalised scores beats reciprocal-rank fusion on all nine BEIR sets
  tested (SciFact 0.753 against 0.730, HotpotQA 0.699 against 0.675; V, Bruch et al.,
  <https://arxiv.org/abs/2210.11934>), needs one parameter, and that parameter is learned from a handful of queries;
  tuned RRF constants do not transfer between sets. Free at query time. **To measure here.**
- **Expansion with a model per query** (HyDE, query2doc, multi-query): one generation per query; measured here in
  Phase 23 with no gain. Not revisited.
- **Pseudo-relevance feedback without a model** (RM3): nDCG@10 0.506 → 0.518 and recall@1000 0.739 → 0.788 on DL19 (V,
  <https://arxiv.org/abs/2108.11044>); a recall tool, sometimes negative at the top. **To measure here** on the reworded
  questions, where recall is the limit.
- **Questions written per passage at index time** (doc2query--, HyPE): up to 16 % better effectiveness with filtering (V
  abstract, <https://arxiv.org/abs/2301.03266>); zero query cost, one model call per passage at index time. It targets W6
  directly, and the graph pass already makes a call per passage. **Specified, not built** (hours of the quick
  model for one base; spec FR-K78).
- **Rerankers.** Qwen3-Reranker-0.6B adds 4.0 points over its own first stage on MTEB-R (V, model card) at the price of
  one forward per candidate (seconds for 20 candidates here). A reranker was measured in Phase 23 and not adopted;
  deferred again (FR-K58).
- **Order in the prompt.** Evidence is used best when first or last (V abstract, <https://arxiv.org/abs/2307.03172>).

## 4. Graph retrieval

Recall@5 on multi-hop sets (V, HippoRAG 2, <https://arxiv.org/abs/2502.14802>): BM25 43.5 / 65.3 / 74.8 (MuSiQue, 2Wiki,
HotpotQA); a 7B embedder 69.7 / 76.5 / 94.5; HippoRAG 2 74.7 / 90.4 / 96.3. Its ablations locate the gain: without the
query-time model filter the average falls 0.7 points; without passage nodes 6.1; seeding from named entities alone 12.5.
LinearRAG builds the graph with **no model tokens** (entities, sentences and passages; Personalized PageRank) and reports
equal or better answers than HippoRAG 2 at 0.093 s a query against 1.69 s (V, self-reported,
<https://arxiv.org/abs/2510.10114>). KET-RAG's keyword–passage bipartite graph, also model-free, covers 50.8 % of MuSiQue
answers against 47.6 % for Microsoft GraphRAG at about one hundredth of the indexing cost (V,
<https://arxiv.org/abs/2502.09304>). GraphRAG-Bench and Han et al. agree on where a graph pays (V,
<https://arxiv.org/abs/2506.05690>, <https://arxiv.org/abs/2502.11371>): not on single-fact questions (plain retrieval
60.92 against 49.29 for GraphRAG), yes on multi-hop (HippoRAG 2 53.38 against 42.93), and community summaries are the
weakest arm on question answering (NQ 54.48 against 64.78). Query-time cost: plain retrieval 879 prompt tokens, HippoRAG 2
1,008, GraphRAG local 38,707, global 331,375, LightRAG 100,832.

**Reading:** the gain of a graph comes from *candidate supply through Personalized PageRank over a passage-and-entity
graph*, not from typed relations, and that graph can be built without a model. That is exactly the piece this module
lacks (W3, W4). Typed relations with quotes stay valuable for *showing* a person why two things are linked, so the
model-built graph becomes an optional second pass over the same structure.

## 5. What not to do, and why

| Not adopted | Reason |
|---|---|
| Microsoft GraphRAG local/global/DRIFT, LightRAG, LazyGraphRAG | 39,000 to 331,000 prompt tokens or 100+ model calls per query; minutes to hours a question on four cores |
| Community summaries, RAPTOR | a model call per community or node at index time; weakest arm on QA |
| Semantic, proposition or model-driven chunking; late chunking | no gain on real documents, or a model call per passage, or unvalidated for last-token pooling |
| An ANN library (usearch, hnsw_rs, lance, diskann) | exact scan is faster below a few hundred thousand passages and has no recall loss |
| tantivy for the words | a full search engine (segments, merges, its own files) for an index that is rebuilt from the store in seconds; the analyzer and scoring here are 300 lines and can weight fields |
| PDFium-, ONNX- or libtorch-based readers and entity models | break the single static binary; some are GPL/AGPL |
| A static-embedding tier, a reranker, learned sparse vectors | each needs model weights the owner has not approved; specified, not built |
| BM25+ / BM25L | within ±0.4 of each other |

## 6. Interface audit (summary)

All three files of the web interface were read (`app.js` 1,761 lines, `app.css` 602, `index.html`). Twenty-one defects
were ranked; the ones that lose work or block a user: leaving Chat aborts the answer in flight; the whole answer is
re-rendered on every frame while streaming; `[1]` in an answer is plain text rather than a link to its source; the
document table is rebuilt every 1.5 s and takes typed search text with it; an upload has one aggregate bar, no cancel and
errors that vanish; a route change moves no focus and announces nothing; chat rows cannot be reached or deleted from the
keyboard; errors and successes share one 2.6 s toast; raw `HTTP 500` reaches users; the graph cannot be zoomed, panned or
filtered and silently shows 36 of 271 things. Against what document assistants now offer (Kotaemon, RAGFlow, NotebookLM,
Onyx, Open WebUI, AnythingLLM, LM Studio, Obsidian's graph; V for the features cited from their own pages) the module
lacked clickable citations with a passage preview, opening a source at the cited passage, a per-file upload queue, a weak-
match warning and graph navigation. No cross-site scripting path was found.

## 7. Benchmarks available and what was used

Standard sets and their reference numbers: SciFact (BM25 0.679, SPLADE-v3 0.714, BGE-base 0.741 nDCG@10; V,
<https://castorini.github.io/pyserini/2cr/beir.html>), NanoBEIR (50 queries per subset, CC-BY-4.0), the HippoRAG 2
multi-hop subsets (1,000 queries each; table above), MultiHop-RAG, GraphRAG-Bench. **None was downloaded**: fetching
them needs the owner's approval, which was not asked for mid-run. Phase 25 is therefore measured on the project's own
corpora with a larger question set written for it (quoted, reworded and two-passage questions); the specification says
which claims that supports and which it does not.

## Sources

Bruch et al. 2022 (fusion) arXiv:2210.11934 · Lù 2024 (BM25S) arXiv:2407.03618 · Gutiérrez et al. 2025 (HippoRAG 2)
arXiv:2502.14802 · Zhuang et al. 2025 (LinearRAG) arXiv:2510.10114 · Huang et al. 2025 (KET-RAG) arXiv:2502.09304 ·
Xiang et al. 2025 (GraphRAG-Bench) arXiv:2506.05690 · Han et al. 2025 (RAG vs GraphRAG) arXiv:2502.11371 · Qu et al. 2024
(chunking) arXiv:2410.13070 · Günther et al. 2024 (late chunking) arXiv:2409.04701 · Gospodinov et al. 2023 (doc2query--)
arXiv:2301.03266 · Li et al. 2021 (PRF) arXiv:2108.11044 · Liu et al. 2023 (lost in the middle) arXiv:2307.03172 ·
Anthropic 2024 (contextual retrieval) · Hugging Face 2024 (embedding quantization) · the model cards and repositories
linked in the text.
