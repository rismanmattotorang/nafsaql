# NAFS-AQL: REASONING CACHE PROMPT (v1)

Revised on 2026-11-23 from the owner's request so that it matches Nafs-Aql as it is built today (release 0.2.0,
commit `acd4d67`). The project-level name of this work is **Phase 38**.

What changed from the original request, and why:

| Original request | Revised | Reason |
|---|---|---|
| "NVIDIA's CPU-ARC" | **CPU-ARC: Agentic Reasoning Caching for on-CPU Reasoning Reuse**, Belcak et al., NVIDIA, arXiv 2611.09891 (6 Nov 2026), read in full | The paper is the specification to compare against; its cost and accuracy numbers come from Claude Opus 4.5 as source and builder and Llama 3.2 1B as target, not from a CPU pair |
| "System 2 turns a frontier model's trace into code" | Roles mapped onto the Semantic Duo: **source and builder = System 2** (Qwen3-4B-Thinking), **target = System 1** (Qwen3-1.7B) for parameter extraction, **judges = the exact verifiers** already in the gateway; the embedder (Qwen3-Embedding-0.6B) resolves descriptions | There is no frontier model on the reference machine; the paper's judge committees are replaced by verifiers that do not need a model |
| Implement the cache "in the harness engine" | In the **gateway's request path** (`nafs-core`), before the System 1 draft, with the build as an **idle-time worker** beside the audit and label workers | Every chat request passes through the orchestrator; the code agent is one caller among several |
| On a hit, System 1 extracts the parameters | Extraction is **tiered**: the stored template first (no model), then System 1; every extracted value must occur in the question | A template match costs microseconds and cannot invent a number; the model tier handles rewording |
| "Compiled into a standalone Python module" | `POST /v1/arc/compile` writes one standard-library Python file with descriptions resolved statically; the module answers without the gateway | As in the paper's section 2.2.3 |
| Find related ideas in other papers | Each borrowed idea is named with its source and either **built and measured or listed as not built** | No claim without a number |
| "The best Semantic Duo engine, a new state of the art" | Claims are **per measured property on the reference machine** (hit rate, correctness on hits, false hits, latency, build cost) | One machine and one model pair; no public leaderboard was run |
| Sweep test, update paper, README, website, push to main | Rust tests, `fmt`, `clippy`, the live sweep, then a commit on `main` and a push | The owner asked for main; CI is checked after the push |

---

## 0. How you work

Same rules as `docs/prompts/COMPLETION_PROMPT.md` section 0. Verify before you claim; label shared-machine timings;
never kill the owner's processes; ask before downloading model weights; plain language in the interface; Rust at run
time (Python only inside the sandbox, as for the verifiers).

## 1. Read

1. The CPU-ARC paper: construction phases 1-5, semantic functions and primitives (`load`, `do`, `narrate`, `lm`,
   `extract`, `convert`, `invoke_tool`), invocation steps IS1-IS4, compilation, choice of target, the perturbation
   tests (phrasal, parametric, combined) and the cost model (break-even `B / (S - C)`).
2. The Nafs-Aql request path (`nafs-core/src/orchestrator.rs`), the earlier-answer retrieval (FR-061), the verifier
   sandbox (`nafs-verify`), the embedder (`Knowledge::embed`), the idle workers (`nafs-gateway/src/api.rs`).
3. Related work on semantic and plan caches, tool making and reasoning templates; verify every reference.

## 2. Decide

Write `docs/research/reasoning_cache.md` (assessment, gaps, what transfers to four CPU cores and what does not) and
`docs/spec/reasoning_cache_spec.md` (data model, hit path, build path, trust, compile, API, configuration, tests,
measurements). Record the decision in `docs/decisions/028-reasoning-cache-phase38.md`.

## 3. Build

The cache, its configuration (`[arc]`), API (`/v1/arc*`), a page or panel to inspect and delete entries, unit tests
without a model, and a benchmark script with an independent oracle for every item.

## 4. Measure

On the reference machine: build success and build time per entry; on hits, correctness against the oracle for
reworded, re-numbered and both; false hits on near-miss questions that need different reasoning; hit latency
against the System 1 draft and the System 2 path; break-even number of hits.

## 5. Finish

Live sweep, paper section, README, website, release notes; commit to `main` and push; check CI.
