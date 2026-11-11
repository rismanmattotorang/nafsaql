# Where the time goes, and what makes it shorter: the engine, vectors, graphs and RAG (Phase 27)

Date 2026-11-11. The owner's request: assess Smiley Core, the vector and graph construction and RAG for speed, look
for ideas in papers and engineering notes, build and test the ones that hold up. Decision record:
[`020-speed-phase27.md`](../decisions/020-speed-phase27.md). Measurements: `benchmarks/engine_phase27/`,
`benchmarks/kb_phase27/`. Reference machine: Intel i3-8100 (4 cores, AVX2, no GPU), shared; load is stated with each
result. The rule is the same as before: a change becomes a default only when it wins an interleaved comparison on
this machine, and what does not win is recorded with its number.

## 1. Where the time goes

**An answer from documents** (quick model, about 1,100 prompt tokens, 64 answer tokens, measured through the server):

| Step | Time | Bound by |
|---|---|---|
| Search (words, vectors, fusion) | 0.2 ms | nothing worth optimising |
| Embedding the question | about 150 ms | the 0.6B encoder reading about 30 tokens |
| **Reading the prompt (prefill)** | **about 22 s at 50 tokens/s** | arithmetic: scales with cores |
| Writing the answer (decode) | about 4 s at 17 tokens/s | memory bandwidth: does not scale past three threads |

Four fifths of the wait is the model *reading* the passages. So the two levers are how fast a prompt is read and how
much prompt there is.

**The engine's prompt reading**, alone on the machine (`examples/prefill_bench`, Qwen3-1.7B Q4_K_M):

| Threads | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| Prompt tokens per second | 22.3 | 44.3 | 60.3 | 71.6 |

It scales almost linearly with cores, and 90 % of it is the weight products. The kernel does about ten multiply-adds
per cycle per core (`examples/kbench`: 54.5 ns for a 2,048-wide row against one activation of a batch of 32), which is
what its instruction mix allows on AVX2 without VNNI: the 4-bit format carries a scale every 32 weights, so every 32
weights need two widening multiplies. Decode, by contrast, is bound by memory and is 4 to 8 % *slower* with four
threads than with three (Phase 17), which is why the servers ran with three threads for everything.

**Vectors.** An exact scan of 847 passages takes 0.17 ms and of 100,000 passages 20.8 ms; the question's embedding
takes 150 ms. Below a few hundred thousand passages the scan is not where the time is.

**Graphs.** The rule-built graph of a base takes under a millisecond. The model-built graph takes 17 to 28 s a
passage, nearly all of it the quick model writing JSON.

## 2. What the literature offers

| Idea | Source | What it is | Verdict here |
|---|---|---|---|
| Separate thread counts for reading and writing | llama.cpp `--threads-batch` | prompt processing is compute-bound and takes more threads than generation | **built and adopted** (section 3) |
| Precomputed attention state per passage | TurboRAG (arXiv 2410.07590): 8.6x shorter time to first token | each chunk's KV cache is computed offline and concatenated at query time | not built: the result is not the same computation (chunks never attend to each other or to the instruction), the paper fine-tunes the model to recover accuracy, and the cache of one 1,200-character passage is about 34 MB at our precision (115 KB a token), 29 GB for the 847 passages of the test base |
| Repairing reused caches | CacheBlend (2.2 to 3.3x) | recompute a subset of tokens to restore cross-attention | not built: also inexact, and the selective recompute is engine work of the same size as the gain is uncertain on a 1.7B model |
| Exact reuse of a shared beginning | SGLang's radix cache; our prefix store | reuse the KV of the longest shared prefix | already built (Phase 17); a knowledge prompt shares only its instruction, because the passages differ per question |
| Query-dependent pruning of retrieved text | Provence (arXiv 2501.16214), RECOMP, LLMLingua | a trained cross-encoder keeps the sentences that bear on the question | the trained pruner is a further model; **the model-free counterpart is the sentence packer we have; its budget was measured, and the smaller prompt is built as an option, not the default** (section 4) |
| int8 and binary vector codes with rescoring | Sentence-Transformers and vendor notes; the Qwen3 embedding papers report int8 as near-lossless and binary as lossy | scan compact codes, rescore the best with full precision | deferred with the number: the scan is 0.1 % of a search at 847 passages and 12 % at 100,000; it matters for bases beyond the in-memory limit, which nobody here has |
| Matryoshka truncation | Qwen3-Embedding is trained for it | compare on the first dimensions only | same verdict as above |
| Graph extraction on the most central chunks first | KET-RAG (arXiv 2502.09304) | run the model only on the top fraction of chunks by PageRank | deferred: the rule-built graph already gives the PageRank; an ordering by it is a small change, but its value needs a graph-quality measurement we do not have |
| No model at indexing time | LazyGraphRAG, ContextRAG | noun-phrase graphs, model calls only at query time | this is the rule-built graph of Phase 25 |
| Speculative decoding, 1-bit weights, hedging | earlier phases | | measured and not adopted (Phases 7, 13, 19) |

## 3. The engine: read prompts with every core, write tokens with one fewer

`Engine::with_prefill_threads` gives the engine a second pool for prompt chunks. The decode pool keeps its width.
An idle pool's workers park after their spin window, so the two never compete; the weight products do not depend on
how rows are shared among workers. The server takes `--prefill-threads` (default: the same as `--threads`), and
`scripts/serve_smiley.sh` sets it to the number of cores for both models and for the embedding server, which only
reads.

Measured through two servers on spare ports, the same prompts to both, order alternating, nothing reused
(`scripts/prefill_threads_ab.py`):

| Model | Prompt tokens/s, 3 threads | with 4 for prompts | Decode tokens/s | Time to last token | Pairs faster |
|---|---|---|---|---|---|
| Qwen3-1.7B, 1,100-token prompts, 64 answer tokens | 49.7 | **61.0 (+23 %)** | 17.3 → 16.9 | 26.3 s → **22.2 s (−16 %)** | 7 of 8 |
| Qwen3-4B-Thinking, 740-token prompts, 32 answer tokens | 18.5 | **22.7 (+23 %)** | 8.4 → 8.3 | 43.7 s → **36.5 s (−16 %)** | 6 of 6 |

A first run on the reasoning model showed no gain (2 of 4 pairs); a build was using the fourth core at the time. The
table is the repeat with nothing else started by us. The point of the lesson: the fourth core helps only when it is
free, and on a machine whose fourth core is always busy `PREFILL_THREADS=3` restores the old behaviour.

The embedding server on four threads against three (`scripts/embed_threads_ab.py`, 24 texts of each kind,
alternating): a question in 252 ms against 293 ms (−14 %) and a 1,000-character passage in 1.08 s against 1.27 s
(−15 %), faster in 24 of 24 each. Reading documents in is therefore about a sixth faster.

**Measured and not adopted in the engine:**

- *Cache-blocking the prompt products* (a block of weight rows kept in the second-level cache while blocks of
  activations small enough for the first-level cache run against it): 62 to 65 tokens/s against 62 to 65 unblocked,
  once the benchmark had a warm-up pass. The wide kernel already decodes a row once per chunk, and blocking by
  activations repeats that decoding.
- *A prompt chunk of 32 tokens instead of 64*: 71 against 64 tokens/s in one interleaved run in one process, and no
  difference (64 to 65 both) in separate warmed processes. A difference that depends on where the allocator puts the
  buffers is not a default.
- Both were first "confirmed" by runs that timed the page faults of a freshly mapped model file. The benchmark now
  makes one untimed pass first.

## 4. RAG: the budget is what the model reads

The prompt carried the four best passages whole (`rag_k = 4`, `rag_max_chars = 4800`), so the sentence packer of
Phase 21 had nothing to choose. Three steps, each measured before the next.

**Step 1, no model: fewer characters of passage text.** `examples/rag_budget` measures, for each number of passages
searched and each budget, whether the answering phrase is in the packed prompt (four sets, 306 questions, 847
passages). Searching eight passages and keeping 1,800 characters of their best sentences put the answer in the prompt
for 178 questions against 174 for four whole passages, with about 30 % less passage text.

**Step 2, live: the saving was not there.** Fifty questions through two gateways that differed only in these two
settings, order alternating (`scripts/rag_budget_ab.py`, `rag_budget_ab_wider_search.json`): 45 right against 42 (four
discordant pairs to one, exact McNemar p = 0.38), but the prompt was only 10 % shorter (1,032 against 1,140 tokens),
and for the questions an arm was asked first, where the engine had nothing of the other arm's prompt to reuse, the
time was 22.6 s against 23.3 s. The reason is in the block the model actually reads: every source carries its
document's title, its heading path and its delimiters, about 185 characters, and eight short excerpts carry almost
twice as many headers as four passages. **A third of the block was headers** (3,204 characters for 1,803 of text
on the 50-question set).

**Step 3: budget the block, not the text.** The packer now charges a source's header the first time anything of
that passage is taken, so `rag_max_chars` is the size of what the model reads, and a document's title is written once
per run of its passages (`rag_compact_sources`, on). Offline, over the same 306 questions:

| Passages searched, block budget | Answer in the prompt (of 306) | Characters the model reads | Sources |
|---|---|---|---|
| 4, text budget 4,800, full headers (before) | 174 | 3,070 to 3,700 | 3.3 to 4.0 |
| 4, 2,400 | 170 | 2,270 to 2,570 | 3.1 to 3.7 |
| **6, 2,400** | **177** | **2,270 to 2,600** | 3.5 to 4.3 |
| 8, 2,400 | 176 | 2,270 to 2,590 | 3.6 to 4.7 |
| 6, 2,000 | 171 | 1,990 to 2,220 | 3.2 to 4.0 |
| 6, 3,600 | 179 | 3,010 to 3,710 | 4.1 to 5.5 |

Six passages searched and a block of 2,400 characters holds the answer for three more questions than before in
24 to 30 % fewer characters. Below 2,400 the answers start to leave the prompt.

**Step 4, live:**

Fifty questions, the old layout against six passages in a 2,400-character block
(`benchmarks/kb_phase27/rag_budget_ab.json`, load 6 to 8):

| | Before (4 passages whole) | 6 passages, block of 2,400 |
|---|---|---|
| Prompt tokens, median | 1,141 | **852 (−25 %)** |
| Time to the first token, asked first | 19.1 s | **14.2 s (−26 %)** |
| Time to the answer, asked first | 21.9 s | **17.5 s (−20 %)** |
| Faster in | | 42 of 50 pairs |
| **Right answers** | **42 of 50** | **39 of 50** (one discordant pair for it, four against; exact McNemar p = 0.38) |

The speed is real. The accuracy difference is not significant at fifty questions, but the four answers it lost are
of one kind: the quick model took a number from the wrong excerpt (18 entities for 109; "2.4 to 4.8 s" for 1.4 TFLOP)
or named the wrong subject, where the whole passage had kept the sentence next to its context. Excerpts make a
small model's reading worse in exactly the way that matters, and the support check does not catch it (the number
*is* in a passage).

**Decision: the default does not change** (four whole passages). The header-counting packer and the compact titles
are built, tested and **off** (`rag_compact_sources = false`). A deployment that prefers speed sets
`rag_k = 6`, `rag_max_chars = 2400`, `rag_compact_sources = true` and gets answers a fifth to a quarter sooner at
the accuracy measured above. Step 2's wider search (eight passages, 1,800 characters of text) is not adopted either:
no time saved.

The next candidate that keeps whole passages is **a stable passage order within a conversation**: when a follow-up
question retrieves passages the previous turn already had, putting those first in their earlier order lets the
engine reuse what it read (the second arm of every pair above was faster for this reason). It needs a multi-turn
measurement and is not built.

## 5. Not done, and why

- **A retrain or a new model file of any kind.** The embedding model in a 4-bit format would be read about a third
  faster by the kernel (the 8-bit kernel takes 80 ns where the 4-bit one takes 55), but every base would have to be
  embedded again and the retrieval quality measured; it needs the owner's word.
- **VNNI or AVX-512 kernels.** The reference CPU has neither. On a CPU with VNNI the two widening multiplies become
  one instruction; that is the largest kernel gain available and it cannot be measured here.
- **int8 vector codes, Matryoshka truncation, PageRank-ordered graph extraction, passage-level cache reuse**: see the
  table in section 2.
- **Public benchmarks.** None was downloaded, so "state of the art" is not claimed: every number is against this
  system's own earlier setting on this machine.

## Sources

- TurboRAG: <https://arxiv.org/abs/2410.07590>
- CacheBlend: <https://www.cs.princeton.edu/~ravian/COS597_F24/papers/cacheblend.pdf>
- Provence: <https://arxiv.org/abs/2501.16214>
- KET-RAG: <https://arxiv.org/abs/2502.09304>
- When to use graphs in RAG (LazyGraphRAG and others compared): <https://arxiv.org/abs/2506.05690>
- ContextRAG: <https://arxiv.org/abs/2605.19735>
- Embedding quantisation with rescoring: <https://sbert.net/examples/applications/embedding-quantization/README.html>
- Qwen3-VL-Embedding (int8 near-lossless, binary lossy): <https://arxiv.org/abs/2601.04720>
- llama.cpp `--threads-batch`: <https://manpages.debian.org/experimental/llama.cpp-examples/llama-parallel.1.en.html>
