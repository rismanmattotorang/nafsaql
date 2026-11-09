# Head to head: Nafs-Aql against vLLM Semantic Router and DeepSeek Harness (Phase 24)

2026-11-09. Sources: `third_party/semantic-router` at 029bf54ed (Go, Apache-2.0) and `third_party/deepseek-harness`
at fcc0ad4621 (`dsh`, TypeScript, MIT). Phase 22 read both for *mechanisms* ([`router_harness_comparison.md`](router_harness_comparison.md))
and ported eight of them. This note is the second pass the user asked for: what each project **is**, a feature-by-feature
table, a **measured** comparison where the same work can be run on both sides, the gaps that remain and the specification
that closes them ([`headtohead_spec.md`](../spec/headtohead_spec.md)).

**Scope note.** No project called "Semantic Router Duo" exists upstream or in the checkout; "Duo" is read as "the two
projects" (vLLM Semantic Router and DeepSeek Harness). Between the Phase 22 read and this one the upstream moved by five
commits in semantic-router (all bug fixes in signal extraction and cache keys, none changing a mechanism we ported) and
none in `dsh`.

**Machine.** 4-core Intel i3-8100, no GPU, shared with the user's other workloads: the load average was 24 to 29 (1
minute) while the five rounds below ran, so every time is **SHARED MACHINE** and absolute numbers are pessimistic for
both sides. The rounds interleave the two sides (Go benchmark, then Rust benchmark, five times) so that drift hits both.

## 1. What each project is

| | vLLM Semantic Router | DeepSeek Harness (`dsh`) | Nafs-Aql |
|---|---|---|---|
| One line | A decision layer in front of many models: signals, Boolean rules, a model-selection algorithm | A ReAct coding agent over an event-sourced session log | A CPU-only two-process assistant: a quick model, a reasoning model, an exact verifier between them, a knowledge base and an agent harness |
| Language / size | Go (+ Rust bindings for candle models, Python training), Envoy and Kubernetes integration | TypeScript on Cordis plugins, Node | Rust, 13 crates, one static gateway binary, no runtime dependency beyond SurrealDB and the two engines |
| What it needs | Envoy or its HTTP front, embedding/classifier models (0.3B and up), usually a GPU or 12+ vCPU | A DeepSeek/OpenAI-compatible endpoint and Node | Four CPU cores, 16 GB; no GPU |
| Decides correctness by | Model confidence, self-rating, peer consistency (no exact verifier in production) | The model's own `update_goal complete` | Exact verifiers: numeric/SymPy, unit tests in a sandbox, JSON Schema, Lean |
| Measures itself by | Harnesses, no committed results; paper tables without scripts | Host-overhead CI budgets, no task success | Interleaved A/Bs on verified task counts, negative results published |

## 2. Feature and capability table

`+` ahead, `=` comparable, `-` behind. Per row, the judgement of this repository's tests and the upstream source.

| Capability | Semantic Router | `dsh` | Nafs-Aql | Verdict |
|---|---|---|---|---|
| Routing rule | Lexicographic tier/priority + chosen algorithm | – | Expected-loss rule `L(ê₁−ê₂) > λΔc`, budget mode solves λ* | + |
| Certified acceptance | Clopper–Pearson per bin, α/B | – | Same (FR-R1), plus ledger-fed label stream (FR-R6) | = (ours has the data loop; theirs has a pinned artefact) |
| Unknown handling | Three-valued rules, `on_unknown` | – | `on_unknown` + declared missing features (FR-R2) | = |
| Reask / polarity guard | Yes | – | Yes (FR-R3, FR-R4) | = |
| Signal types | ~20 (keyword, embedding, domain, complexity, jailbreak, PII, fact-check, context length, ...) | – | domain, difficulty from draft features, reask, polarity, injection/PII patterns in the security layer | - (count) |
| Model-based jailbreak / PII classifiers | Yes (ModernBERT-class) | – | Pattern and structure checks; no classifier | - |
| Complexity prior from prototype banks | Yes | – | Specified (FR-R5), needs labels | - |
| Exact verification | No | No | Yes | + |
| Semantic cache | Yes, no write gate | – | Verified answers only | + |
| Extractive compression | BM25 line chunks | – | Sentence packer (`pack_units`) | = (and faster, §3) |
| Retrieval over documents | Cache + RAG plugin | Workspace search tools | Hybrid RRF over SurrealDB, GraphRAG, Memorilla, scale-tested to 50k passages | + |
| Agent loop | – | ReAct, plan mode, subagents (up to 1,000), browser and computer use | Best-first verifier-scored search, grammar-constrained patches, rollback, stall guard | + on verified tasks, - on breadth of tools |
| Event-sourced session replay | – | Yes (byte-for-byte prompt rebuild) | Append-only trace and ledger; prompt not rebuilt from them | - |
| Two-phase compaction, anchored meter, interrupted-step repair, repeat chain | – | Yes | Yes (FR-H1..H4) | = |
| Spill of long results to a file | – | Yes | Specified, not built (the model sees at most 930 bytes of an observation) | - |
| Sandbox | – | bwrap, no network unsharing, off-by-default read-before-edit | No network, limits, namespaces + seccomp / Seatbelt, 26 live escape probes | + |
| Deployment surface | Envoy ext_proc, Kubernetes operator, Helm, dashboard | CLI, TUI | One binary, systemd/launchd units, web UI, OpenAI-compatible API | - (Envoy/K8s), + (footprint) |
| Observability | Prometheus, tracing, dashboard | Session log | Prometheus, ledger, quality and gates pages, trace | = |
| Privacy | Optional cloud signal models | Uploads session log by default | Nothing leaves the machine | + |

**Where they remain ahead**, said plainly: the Envoy and Kubernetes integration (we are not an `ext_proc` filter), roughly
twenty signal types and trained jailbreak/PII classifiers (we have patterns), a management dashboard, an event-sourced
log that rebuilds any prompt, subagents at scale, and browser and computer-use tools. None of these is cheap on four
cores, and the first three need a GPU or a service we have decided not to run. They are recorded as *not adopted* in the
spec rather than as gaps to close.

## 3. Measurements

Method: the Go benchmarks of Semantic Router that **compile offline** (`GOPROXY=off`): `pkg/classification/lexical`
(BM25 and n-gram keyword signal, 30 rules), `pkg/promptcompression` and `pkg/embedding/vecmath`. Their `classification`,
`cache`, `looper` and `topiccontinuity` packages need modules that cannot be fetched here (no `go mod download` is
approved), so they are not run. For `dsh` the benchmarks need `pnpm install` (not approved); its **enforced CI budgets**
are quoted instead. The Rust counterparts are `rust/crates/nafs-bench/examples/h2h.rs`; each times whole batches so the
clock read is not the measurement (an earlier version timed the clock: 340 ns for a dot product). Five interleaved
rounds, median. Reproduce with `ROUNDS=5 benchmarks/headtohead/run.sh`; raw files in `benchmarks/headtohead/`.

| Workload | Semantic Router (Go) | Nafs-Aql (Rust) | Go ÷ Rust |
|---|---:|---:|---:|
| Routing signal on a 2 KB prompt: their 30-rule BM25 / our domain classifier | 197.7 µs | 2.2 µs | 90× |
| (their n-gram variant) | 3.65 ms | 2.2 µs | 1,654× |
| Extractive compression, 500 tokens | 745 µs | 310 µs | 2.4× |
| Extractive compression, 2K tokens | 5.07 ms | 1.16 ms | 4.4× |
| Extractive compression, 16K tokens | 50.3 ms | 8.96 ms | 5.6× |
| Dot product n=384, AVX2 + FMA | 70 ns | 90 ns | 0.78× |
| Dot product n=768, AVX2 + FMA | 99 ns | 145 ns | 0.68× |
| Dot product n=1024, AVX2 + FMA | 135 ns | 199 ns | 0.68× |
| Dot product n=384, portable | 440 ns | 349 ns | 1.26× |
| Dot product n=768, portable | 1.04 µs | 861 ns | 1.21× |
| Dot product n=1024, portable | 1.20 µs | 1.21 µs | 0.99× |
| Gate decision, fixed rule | none upstream | 7.4 µs | |
| Gate decision, calibrated + certified bins | none upstream | 2.4 µs | |
| Harness host overhead, 100 turns / 800 reads | their CI budget 900 ms | 1.75 ms | |

**How to read it, and what it does not say.**

- The routing row is *not* the same computation. Their BM25 scores 30 hand-written keyword rules; our classifier is a
  hashed bag of words against domain profiles. Both are "the cheap signal that runs on every request"; the ratio says what
  that signal costs, not that one is a better classifier. Neither is on the critical path: a model call is 10⁵ to 10⁷
  times longer.
- The compression rows *are* the same job (reduce a long text to a budget by relevance, no model). We are 2.4 to 5.6×
  faster, and the gap grows with length because their cost is superlinear. This was a **gap** at first (our packer
  recomputed each sentence's words twice per pass); `pack_units` now computes them once (G-B below). The first run, before the fix, took
  1.07 ms, 5.80 ms and 27.1 ms for the three sizes (a different load, so read it as the order of magnitude).
- The SIMD dot rows are the one place **they are ahead**: their hand-written AVX2 assembly is 1.3 to 1.5× faster than our
  intrinsics at n=768 and 1024 under this load. Our portable loop beats their portable loop. Before our AVX2 kernel
  existed the portable loop was the only path (1.9 µs at n=1024); it is now 199 ns. The remaining gap is 50–60 ns on a
  vector search that costs 18–22 ms per query (the loop over ~50,000 vectors), so a query-level effect is at most 0.2 %.
  We record it as measured, not as closed (G-A).
- The harness row compares host overhead only, which is all `dsh` measures. 1.75 ms for 100 turns and 800 reads against
  their 900 ms budget is a language difference (a compiled loop against Node plus its event system), and it says nothing
  about which agent solves more tasks; that comparison is the verified-task A/B in Phase 11 and 16.
- Load was 24 to 29 on four cores. The Go benchmarks ran with `GOMAXPROCS=4`; both sides were disturbed.

## 4. The gaps, ranked

| # | Gap | Evidence | Status |
|---|---|---|---|
| **G-A** | Vector dot product: our AVX2 kernel is 1.3–1.5× slower than their assembly | §3 | Open, bounded at 0.2 % of a query |
| **G-B** | Sentence packer recomputed words per pass | Compress 16K 27.1 → 8.96 ms (first run vs five-round median) | **Closed** |
| **G-C** | No exact in-memory vector search; every query went to the store's index | Cold HNSW 26–110 s, warm 0.37 s at 50k passages | **Closed, optional**: `vector_cache_mb` (exact, 18–22 ms/query, 164–296 s to load), default off because the warm index suffices |
| **G-D** | A stronger embedder for paraphrased questions | Qwen3-Embedding-4B vector-only R@1 0.25 → 0.45 | **Measured, not adopted** (decision 017) |
| **G-E** | Practice labels invisible in the UI | Quality page lacked the label stream | **Closed**: "Practice while idle" row |
| **G-F** | Signal breadth (~20 types) and model-based jailbreak/PII | Upstream source | Not adopted: needs a ≥ 0.3B classifier per request |
| **G-G** | Event-sourced replay of a prompt | `dsh` `deriveMessages()` | Specified (FR-H6), see spec §4 |
| **G-H** | Spill of long results to a file | `dsh` `maxInlineTokens` | Specified (FR-H5), deferred: the model sees ≤ 930 B of an observation, so a spill would not change what it sees without a tool to read it back |
| **G-I** | Difficulty prior from prototype banks | Upstream `complexity` signal | Specified (FR-R5), waits on labels (the label stream is filling the ledger) |
| **G-J** | Envoy `ext_proc` / Kubernetes | Upstream | Not adopted: this project is a single-binary assistant, not a mesh filter |

## 5. What was learned that goes against the table

1. **"Rust is faster" is not a result.** Two of five rows are slower or equal; the right reading is that cost on the
   per-request path was already negligible and the *accuracy* work (certified bins, exact verifiers) is where the two
   projects differ.
2. **A fair benchmark of the other side was possible for 3 of ~12 packages.** The rest need network; stating that is part
   of the result.
3. **The stronger embedder did not make retrieval better in the hybrid** (decision 017): vector-only improved, the hybrid
   did not, because the words already carry the quoted questions and the meaning only helps the paraphrased ones, where
   the 0.6B model at weight 0.2 is as good as the 4B at weight 1.0.
