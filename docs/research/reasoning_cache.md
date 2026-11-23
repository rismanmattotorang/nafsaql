# Reasoning cache: CPU-ARC on a CPU-only Semantic Duo (Phase 38)

Assessment written on 2026-11-23 for the owner's request to study NVIDIA's CPU-ARC, validate it for Nafs-Aql, and
build the parts that transfer. Measured results are in section 7 and `benchmarks/arc_phase38/`. The specification is
`docs/spec/reasoning_cache_spec.md`; the decision is `docs/decisions/028-reasoning-cache-phase38.md`.

## 1. What CPU-ARC is

Belcak, Heinrich, Muralidharan, Diao and Molchanov, *CPU-ARC: Agentic Reasoning Caching for on-CPU Reasoning
Reuse*, NVIDIA, arXiv 2611.09891 (6 Nov 2026). Two observations: once a problem has been reasoned through, executing
the solution needs little from a language model (O1); and the reasoning for one instance applies to every instance
that is *equirational* with it, the same reasoning with other wording or other parameter values (O2).

Roles: a source model S (the expensive reasoner), a builder B that reads S's transcript and writes the cache entry, a
small target model T that runs the entry, and judges J. In the paper S = B = Claude Opus 4.5 with 4,096 thinking
tokens, T = Claude Haiku 4.5 and then a fine-tuned Llama 3.2 1B, J = committees of two Opus instances, and an
EmbeddingGemma (300M, 768 dimensions) with a FAISS flat index for the semantic lookups.

**Semantic functions.** A Python callable `f(ctx) -> dict[str, Any]` that names its inputs, its outputs and its
callees by natural-language descriptions, not identifiers. Primitives: `load(ctx, description, d_type)` (vector search
over a stack of memory layers), `do(ctx, description, payloads)` (find another function by description, push a fresh
layer with the payloads, run it), `narrate` (the human-readable trace), `lm` and `extract` (calls to T), `convert` /
`smart_convert` (programmatic coercion, then T rewrites the value as a Python literal checked by `ast.literal_eval`),
`invoke_tool`.

**Construction (five phases).** (1) isolate the input and the expected output; (2) decompose the transcript into
function-sized logical operations, marking reusable ones; (3) recognise the parameters, write the template
q(.1, ..., .n) and the original values, with validity ranges; (4) plan the functions bottom-up at a chosen
granularity; (5) an agentic loop writes each function, searches the registry for an equivalent one first, checks
syntax and narration, runs it on the original values and has a judge compare with the transcript fragment; then the
root is tested end to end (correctness against the transcript and golden label; extraction robustness: T extracts the
parameters from the raw input N times).

**Invocation (IS1-IS4).** Embed the input and search the entry points (template and original) with a cosine
threshold; T extracts each parameter by its stored description; dispatch the root in a fresh context; return the
output dictionary and the narration. Only T and CPU run at invocation.

**Compilation.** Once the registry is stable, `do()` becomes a direct call and `load()` a dictionary lookup, giving a
standalone module with no embedder and no vector store; `lm()` and `extract()` survive.

**Results.** On MATH-500, AIME 2024-2026, GPQA Diamond, GSM8K, MMLU-Pro and SciBench: cache accuracy equal to or above
the source on 7 of 8 benchmarks (AIME '25: 93.3 vs 83.3), fidelity to the source above 98 % on 6 of 8, perturbation
pass rates of 90.8-99.3 % (phrasal, parametric, combined), building at 12-84x the source cost, invocation 16-291x
cheaper; break-even after about 13 hits (AIME) to 90 (GSM8K).

## 2. Is it relevant to Nafs-Aql?

Yes, for one measured reason: **on this machine the cost CPU-ARC removes is the largest cost there is.** An escalation
to System 2 (Qwen3-4B-Thinking on four AVX2 cores) takes 44 s to several minutes for a grade-school problem (decode
bound at about 5 tok/s with reasoning), and about a quarter of real chats escalate (Phase 30). The quick model's draft
takes 2-15 s and is wrong more often. A cache hit costs one sandboxed Python run (~0.1 s), plus, on the meaning tier,
one embedding (0.15-0.3 s) and one short System 1 reply. What Nafs-Aql already had does not cover this:

| Existing mechanism | What it reuses | Why it is not a reasoning cache |
|---|---|---|
| KV prefix store, ShadowPrefill (Phases 17, 30, 34) | attention state of a prompt prefix | monolithic: any change of wording or number misses after the first differing token, and decode (the dominant cost) is not reused at all |
| FR-061 retrieval of earlier verified answers | question/answer text, BM25 | a hint in the prompt ("reasoning-hint caching" in CPU-ARC's terms): the model still generates the whole answer, and a reused answer with other numbers is wrong |
| Recall memory, skills (Phases 26, 29) | notes and instructions | context for the model, not execution |
| Code agent and verifiers | tests and sandbox | run the model's code once; nothing is kept for the next question |

The gap is exactly CPU-ARC's hybrid: decomposed (robust to rewording and new numbers) and complete (no generation on
a hit).

## 3. What transfers to four CPU cores, and what does not

| CPU-ARC element | Nafs-Aql decision | Why |
|---|---|---|
| S = B = frontier model | **S = B = System 2** (Qwen3-4B-Thinking) | no other model on the machine; a probe showed it can write correct semantic functions from its own answer (321 s, 3 functions, right answer) |
| T = small model for `lm`/`extract` | **System 1** (Qwen3-1.7B) for extraction only, *before* dispatch | keeps the sandbox network-free and the code deterministic; one S1 call per hit instead of one per `extract` |
| Embedding search for every `load`/`do` | **resolved at registration**: the embedder links each `do()` description to a function id; the runtime matches by exact text, then the link table, then word overlap | the sandbox has no network; the 0.6B embedder is already running in the gateway; per-call vector search would cost ~0.2 s each |
| Entry search by cosine | **three tiers**: template as a pattern (no model), question shape with numbers masked (no model), then the meaning vector | a re-numbered question (CPU-ARC's parametric case) needs no model at all; this is new relative to the paper |
| T extracts parameters | System 1 extracts only on the meaning tier, and **every value must be written in the question and every number in the question must be used** | model-free guards against a small model inventing a value or a question that adds information the code ignores |
| Judges J (LLM committees) | **exact verifiers and the reference answer** at build; **shadow comparison with the models' own answers** before serving (after vCache and Krites) | no judge model to spare; agreement on live traffic is evidence a judge cannot give |
| Perturbation tests by a generator model | build-time **model-free numeric perturbation** (the code must run on other values), and an offline benchmark with hand-written variants and an independent oracle | a generator would be System 2 again, minutes per variant |
| `smart_convert` via T | programmatic conversion only; a failure is an invocation error and falls through | correctness first; a fall-through costs only the usual path |
| Compilation | **built**: `do()` -> direct call, `load(ctx)` -> parameter, `load(result)` -> literal key; the module also carries the templates, so it answers template matches with no model | the paper's compiled form still needs T for `extract`; ours does not, for re-numbered questions |
| Fine-tuning T on invocation logs | not built | no training on this machine (owner's rule since Phase 20) |
| Granularity setting | fixed (the builder is asked for reusable steps; reuse across entries is offered by description) | a 4B builder follows one instruction better than three |

## 4. Gaps CPU-ARC leaves open that matter here, and the remedies borrowed

| Gap | Remedy | Source |
|---|---|---|
| A cache hit is trusted as soon as the entry passes its build tests | **Trust after agreement**: an entry is a candidate until it agrees with the models' answer `trust_after` times on live questions; a candidate runs beside the models (shadow), costing ~0.1 s | Krites (Singh et al., arXiv 2602.13165), asynchronous verification before promotion |
| One global similarity threshold | **Per-entry floor**: starts at `serve_similarity`, lowers to the lowest similarity it agreed at, stays above the highest it disagreed at, never below `min_similarity` | vCache (Schroeder et al., ICLR 2026, arXiv 2502.03771), per-embedding learned thresholds |
| A wrong entry keeps answering | **Quarantine**: a disagreement on a checked answer (template or shape tier) stops it; two net disagreements stop it on the meaning tier | Dynamic Cheatsheet (Suzgun et al., arXiv 2504.07952): curate by test-time performance |
| Follow-up questions look like cached ones | **No cache for a question that leans on the previous turn** (`continuity::leans_on_previous`) | MeanCache (Gill et al., IPDPS 2025), context-dependence detection |
| Unbounded growth | **LFU eviction** at `max_entries`, keeping entries that others call | Kulkarni et al., arXiv 2608.20280 (LFU never beaten by more than 0.041) and TroVE (Wang et al., ICML 2024) trimming |
| Builder reuses functions only by its own search | registry search by the embedder offered to the builder; `do()` links to other entries' functions; eviction never removes a callee | CRAFT (Yuan et al., ICLR 2024), LATM (Cai et al., ICLR 2024) tool reuse |
| The builder rewords the template | **Template rebuilt from the question's own words** when the declared values are each written once | new here (found by the first probe) |
| Code that memorises the answer | the build reruns the code on other values of each numeric input and rejects one that fails | CREATOR (Qian et al., 2023): fix from execution feedback |

## 5. Related work (verified)

Verified on arXiv or publisher pages on 2026-11-23; BibTeX in `docs/research/refs_all.bib`.

- **Closest designs.** CacheSpec (Chen et al., EMNLP 2026, arXiv 2607.20507): Program-of-Thoughts programs as cache
  objects, one small model extracts variables on a hit and drafts for speculative decoding on a miss. Agentic Plan
  Caching (Zhang et al., NeurIPS 2025, arXiv 2506.14852): plan templates from agent logs, keyword match, a small model
  adapts. ReaComp (Naik et al., arXiv 2605.05485): reasoning traces compiled into symbolic solvers with zero model calls
  at test time. Program Learning with Verifiable Rewards (Bhat et al., arXiv 2610.19148): typed primitives from expert
  traces checked by deterministic verifiers. Generative Caching (Chakraborty et al., arXiv 2511.17565).
- **Verified semantic caching.** vCache, Krites, MeanCache, Category-Aware Semantic Caching (Wang et al., arXiv
  2510.26835), GPTCache (Bang, NLP-OSS 2023), eviction study (Kulkarni et al., arXiv 2608.20280).
- **Tool making and libraries.** LATM, TroVE, CRAFT, CREATOR, ReGAL (Stengel-Eskin et al., ICML 2024, arXiv
  2401.16467), LILO (Grand et al., ICLR 2024, arXiv 2310.19791), Voyager (Wang et al., arXiv 2305.16291).
- **Reasoning as code and as memory.** PAL (Gao et al., ICML 2023), Program of Thoughts (Chen et al., TMLR 2023),
  Buffer of Thoughts (Yang et al., NeurIPS 2024), Dynamic Cheatsheet, Agent Workflow Memory (Wang et al., ICML 2025),
  ReasoningBank (Ouyang et al., ICLR 2026). ReasonCACHE (Gupta et al., arXiv 2602.02366) and Crystal-KV (Wang et al.,
  arXiv 2601.16986) are cited by CPU-ARC; they cache KV prefixes, not programs.

## 6. Not built, and why

- **`lm()` inside semantic functions.** Would need a channel out of the sandbox to System 1 per call. Every case in the
  paper's appendices (unit classification, value extraction) moves to the extraction step. Revisit if entries for
  text classification are wanted.
- **Generator-model perturbation tests at build time.** Each would be a System 2 solve (minutes). The shadow
  comparison on live traffic gives the same evidence for free.
- **CacheSpec's miss path** (System 1 drafts for System 2 on a miss). Speculative decoding was measured in Phases 4 and
  17 (EAGLE-3, DSpark): no gain on these models on this CPU.
- **Merging similar functions (ReGAL).** Needs a builder pass over the whole library; worth it only once the cache
  holds hundreds of functions.
- **Caching open-domain or code questions.** Domains are configurable; math is the default because an exact
  comparison exists for it. Code questions already have the code agent and tests.

## 7. Measured on the reference machine

From `benchmarks/arc_phase38/results.json` (method in its README). Reference machine, load 5 to 17.

| | Result |
|---|---|
| Builds | 12 of 12 (7 in one round, 5 in two; each failed first round was refused by a check and fixed from its feedback); 461 s mean, 7.2x the 64 s System 2 answer; 20 functions, no cross-entry reuse (12 unrelated topics) |
| Original questions | 12 of 12 right (template) |
| Same words, other numbers | 24 of 24 right, template, 95 ms median, no model |
| Other words, same numbers | 13 of 14 right (the miss writes "Five" as a word) |
| Both changed | 4 of 12 right; 8 below a meaning similarity of 0.80 |
| Near misses | 6 of 14 answered, all wrong, with the extraction's own "same" field; 0 of 14 after the contrast words and the direct System 1 question (21 of 23 judgements right when measured alone) |
| Similarity floor | with the checks, 0.80 lost nothing on this set; 0.85 shipped |
| Chat time, other numbers | 0.12 s cache, 20.8 s usual path, 85.5 s System 2 (medians of 6, interleaved, all right) |
| Chat time, other words | 9.7 s cache arm (2 of 6 fell through), 18.1 s usual path, 81.8 s System 2 |
| Break-even | about 6 re-numbered questions bound for System 2, 23 for the usual path |
| Compiled module | 12 entries, 28 KB, 36 of 36 template questions right with no model in 78 ms; agrees with the runtime on 76 of 76 |

**On the live chat page.** The chat page sends no domain hint; a perimeter question was classified "open" and a
re-numbered question in a continued conversation was read as a follow-up, so neither reached the cache. Since the
template and shape lookups match only a whole stored question, they now run for every eligible question; a re-numbered
perimeter question in a continued conversation was then answered in 176 ms (129 ms of it the sandbox run), labelled
"Reused", with its working shown.

**What this says.** Against CPU-ARC's headline (frontier accuracy at 16-291x lower cost per call), the CPU pair gets
the large part of the gain where the question keeps its wording: a re-numbered problem costs a tenth of a second
instead of a minute and a half, with no model and an exact, readable derivation. Rewording is where a 1.7B model is
the weak link: it finds and confirms the same problem in most cases but not when words and numbers change together,
and its first, embedded "same" judgement was not enough to stop near misses. The checks added here stop them on this
set; more data is needed before the meaning tier's floor is lowered.
