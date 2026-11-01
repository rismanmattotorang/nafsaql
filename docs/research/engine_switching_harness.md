# Engine flags, the model switch, and the harness cascade (Phase 14)

Date 2026-11-01. Goal set by the owner: assess the inference engine, the switch between the two models and the Codex-style
harness; research what concurrency, parallelism and current methods can still give on the target machine; implement what the
measurements justify. Decision record: [008](../decisions/008-engine-switching-harness.md). Report: section "Phase 14".
Target machine: Intel Core i3-8100, 4 cores, no SMT, AVX2 and FMA (no AVX-512, no VNNI), 32 GB DDR4, macOS 15; llama.cpp
797de63 built with `-march=native`, Accelerate BLAS, OpenMP, llamafile kernels and CPU repacking; Qwen3-1.7B (System 1) and
Qwen3-4B-Thinking-2507 (System 2), both GGUF Q4_K_M.

## 1. Assessment

### 1.1 The engine

Phases 7, 11 and 13 measured the things that change *what* the engine computes: speculative decoding (EAGLE-3, decision
P7-T04: no gain on System 1; DSpark, decision 004: skipped; n-gram, decision 007: 1.12x at best), weight formats (P7-T05: Q4_K_M
stays; Q4_0 and IQ4_NL with the AVX2 repack kernels are slower), KV-cache quantisation (decision 007: no gain at depth), batching
across slots (decision 007: 2.3-2.7x aggregate at short context, 1.5x at 1,600 tokens), and prefix caching (decision 006:
75-100 % of prompt tokens served from the cache). What was never varied is *how* the engine computes: the serving script has
run both servers with `-t 4`, default flash attention (`auto`), the default 512-token micro-batch, the default busy-wait level
(`--poll 50`) and normal priority since Phase 0. Section 3 measures those knobs (`scripts/engine_tuning.py`).

A second gap: the two servers each run four threads on four cores. One decoding alone owns the machine; when both decode (a
quick draft for one user beside a thorough answer for another, or a rescue batch beside thinking) eight busy threads share
four cores, and ggml's worker threads *spin* while waiting for each other (`ggml_graph_compute_poll_for_work`, `--poll`), so the
loss can exceed the arithmetic halving. The 100-prompt run shows the symptom: the model servers averaged 305 % CPU of 400 and
each of two users' streams ran at 4-7 tokens/s against 10-23 alone. Section 4 measures co-scheduling (`scripts/cosched_ab.py`).

### 1.2 The model switch

Two switches exist. **Inside a request**, the gate moves the work from System 1 to System 2: draft, exact check, rescue batch
(FR-096), decision, System 2. Both models stay resident (1.2 GB and 2.8 GB of 32 GB), so the switch is not a load; its cost is
the gate's own time, a slot on System 2, and the prefill of the System 2 prompt. The request's System 1 streams are dropped at
the decision (the rescue samples and the fused agreement samples are cancelled by closing the connection; llama-server frees a
slot when it next writes to a closed connection, at most one decode step later). Section 5 measures it end to end
(`scripts/handoff_latency.py`). **Between backends** (`POST /v1/system/backend`, llama.cpp against the vLLM profile) the
switch was a pointer swap after a health check: correct, but the first requests on the new pair paid a cold prefill of every
constant prefix (about 9.8 s to the first token against about 1 s warm, decision 005). Section 5 makes it warm-first.

### 1.3 The Codex harness

Phase 11 ported the mechanisms of Codex's agent core that survive a 1.7B/4B model on four cores (append-only transcript,
compaction, the `apply_patch` grammar as per-step GBNF, dispatch at item done, budget reminders, the turn diff, the cap inside
the call) and measured them (decision 006). What Codex has that Phase 11 did not take: **parallel tool calls** and **dispatch
while streaming** (`tools/parallel.rs`, `handle_output_item_done`) and **retries with backoff** of a failed sampling request
(`responses_retry.rs`). The first two do not pay here: one action per reply, and under the grammar the server stops at the end
of the action, so there is nothing left to overlap. The third is cheap and was missing: a stream that dropped mid-call ended
the run as `model_error`.

What the harness *does* leave on the table is Nafs-Aql's own idea. The agent runs on System 2 from its first step (decision 006:
a verified fix takes 33-59 s), while System 1 verified 7 of the same 10 tasks at about 11 s per step (Phase 11, round 2), and
four System 1 candidates in one request cost 1.7 times one (decision 007). A **cascade** inside the agent, the quick model
first with candidates judged by the verifier and the thorough model only when none passes, is the gate's logic applied to the
agent, and the exact verifier is the cascade's scorer (FrugalGPT's cascade with a learned scorer, Chen et al. 2023, here with
an exact one). Section 6 implements and measures it.

## 2. Research anchors

| Method | Source | Transfers to a 4-core AVX2 CPU with these models? |
|---|---|---|
| Iteration-level (continuous) batching | Orca, Yu et al., OSDI 2022; llama-server slots | In use; gain measured in Phase 13 |
| Paged KV cache, prefix sharing | vLLM, Kwon et al., SOSP 2023; SGLang RadixAttention, Zheng et al., NeurIPS 2024 | In use (`--cache-reuse`, host prompt cache; 75-100 % hits) |
| Speculative decoding (draft, EAGLE, DSpark, n-gram) | Leviathan et al., ICML 2023; Li et al., EAGLE-3 2025 | Measured earlier: no gain, the CPU is compute-bound in decode (decisions 004, 007) |
| Thread polling and oversubscription | ggml threadpool (`--poll`, `--threads-batch`) | **Measured here** (sections 3-4) |
| Flash attention on CPU, micro-batch size | ggml `-fa`, `-ub` | **Measured here** (section 3) |
| Low-bit table-lookup kernels (T-MAC), ternary models (BitNet b1.58, bitnet.cpp) | Wei et al., 2024; Ma et al., 2025 | Need other models or a trained ternary model: no Qwen3 variant exists; not adopted |
| Model cascades with a scorer | FrugalGPT, Chen, Zaharia, Zou, 2023; mixture-of-thought cascades, Yue et al., ICLR 2024 | **Adopted** in the agent with the exact verifier as the scorer (section 6) |
| Hedged requests, cancel the rest | Dean and Barroso, CACM 2013 | In use (rescue, rounds); the cascade's candidates are judged in order and the rest are not judged once one passes |
| Checkpoint loading for fast model switching | ServerlessLLM, Fu et al., OSDI 2024 | Not needed: both models stay resident; the switch is routing plus a warm prefix (section 5) |
| Retries with exponential backoff and jitter | Codex `responses_retry.rs`, `async-utils/backoff.rs` | **Ported** to the harness (section 6) |

## 3. Engine flags (`scripts/engine_tuning.py`, `benchmarks/engine_tuning.json`)

Method: `llama-bench` on the production GGUF files, `pp512` (prompt processing) and `tg64@d1024` (decode at the depth of a
repair prompt), one repetition per arm, arms cycled in three rounds, median over rounds; the host load before every row is
recorded. A first, block-ordered run (`benchmarks/engine_tuning_blocked.json`) was discarded: another workload of the owner's
started mid-way and turned single blocks by 0.4x. The interleaved run could not escape it either: the load before a row ranged
from 8 to 67 (median 14) against about 2 on an idle host.

| Flag | System 1 pp512 | System 1 tg64@d1024 | System 2 pp512 | System 2 tg64@d1024 |
|---|---|---|---|---|
| baseline (`-t 4`, fa auto, ub 512, poll 50) | 87.9 tok/s | 8.4 | 35.0 | 3.45 |
| `-fa off` | 0.79x | 0.66x | 0.88x | 0.25x |
| `-fa on` | 0.79x | 0.31x | 0.58x | 0.18x |
| `-ub 256` | 0.96x | 0.69x | 0.42x | 0.26x |
| `-ub 1024` | 0.96x | 0.94x | 0.55x | 0.32x |
| `-t 3` | 0.86x | 1.02x | 0.64x | 0.51x |
| `--poll 0` | 0.93x | 0.56x | 0.64x | 0.50x |
| `-lm mlock` | 1.18x | 0.70x | 0.73x | 0.35x |

(`--prio` is refused on macOS, where it needs SCHED_FIFO privileges; `-ub 128` was 0.6x on both models in the blocked run.)

**Verdict: inconclusive, flags unchanged.** The per-round ratios of the same flag against the same round's baseline swing from
0.07x to 3.8x (System 2 decode under `-fa on`: 0.15, 0.07, 1.04), which is the external load, not the flag: the baseline itself
is 2-4 times below its idle values (System 2 prefill 31 tok/s and decode 5-6 tok/s idle, decision 007). No flag can be adopted
on this evidence, and none is. What the phase leaves behind is the method: an interleaved, load-labelled matrix
(`scripts/engine_tuning.py`) to re-run on an idle host; the re-open rule in decision 008 says what counts as a win.

## 4. Co-scheduling the two servers (`scripts/cosched_ab.py`, `benchmarks/cosched_ab.json`)

Temporary servers per variant (ports 8011/8012, the production pair idle), an ~800-token cached context, 96 decoded tokens on
System 2, and System 1 decoding 64-token streams one after another beside it until System 2 ends; two interleaved reps per
variant on the shared host (load 6-11). "Keeps" is the rate with the other server busy over the rate alone.

| Variant (S1 flags \| S2 flags) | S2 alone | S2 + S1 | S2 keeps | S1 alone | S1 + S2 | S1 keeps | both, tokens/s |
|---|---|---|---|---|---|---|---|
| base (`-t 4` \| `-t 4`) | 3.62 | 2.29 | 63 % | 9.87 | 5.21 | 53 % | 7.50 |
| `--poll 0` on both | 3.67 | 2.37 | 65 % | 10.02 | 5.45 | 54 % | 7.82 |
| S1 `-t 2` | 4.33 | 2.02 | 47 % | 6.88 | 5.49 | 80 % | 7.52 |
| S1 `-t 3` | 4.11 | 2.00 | 49 % | 9.40 | 5.42 | 58 % | 7.42 |
| `--poll 0`, S1 `-t 2` | 6.37 | 1.69 | 27 % | 9.55 | 5.40 | 57 % | 7.09 |

**Verdict: nothing to arbitrate; serving unchanged.** When both servers decode, each keeps about half to two thirds of its
solo rate and the pair writes 7.1-7.8 tokens/s together, the same in every variant: the four cores are the bottleneck and the
flags only move the split (fewer System 1 threads give System 1 a larger share of a smaller total). Polling off does not help
(the spin is short-lived under the OS scheduler), and priorities are unavailable on macOS. "Freezing" the quick model while the
thorough one works would cut the pair's throughput and the other user's time to first token for no gain in the thorough
model's: not built. Phase 13's finding stands that the cores are best spent *before* System 2 (batched samples), and the
100-prompt run's finding that a second user shares the same throughput.

## 5. The model switch

### 5.1 Inside a request (`scripts/handoff_latency.py`, `benchmarks/handoff_latency.json`)

Four requests whose draft fails its exact check (a deliberately wrong `reference_answer`), the rescue on (4 samples) and off,
interleaved, reasoning streamed so System 2's first token is its first reasoning token, on the shared host (load 7-12):

| | rescue off (2 runs) | rescue on (2 runs) |
|---|---|---|
| draft end -> gate decision | 0.8-0.9 s | 26.9-34.5 s (the four-sample batch at this load; 14.5 s median idle, decision 007) |
| decision -> System 2's first token | 1.1-4.7 s | 1.5 s |
| System 1 slots busy after the decision | 0.0 s | 0.0 s |
| System 2 prompt: new / cached tokens | 15-27 / 392-1,148 | 5-13 / 262-974 |

The switch inside a request is already as fast as the engine allows: the gate's own work is under a second, System 2's
prompt is almost entirely served from its cache (the system prompt is warmed at start-up, and the question is short), and the
request's System 1 streams are gone before the decision (the rescue batch has ended; a fused sample cut early is dropped with
the draft's connection). There is nothing to freeze: an idle llama-server costs 0 % CPU (its threads spin only while a batch is
being computed). The 27-35 s of the rescue arm is the batch that Phase 13 measured to pay for itself (break-even 106 s per
avoided escalation against 250-580 s per escalation); it is not switch cost. What a prefetch of the System 2 prompt during the
draft could save is bounded by the 1.1-4.7 s above, at the price of contending with the draft; not worth building.

### 5.2 Between backends: warm-first

`POST /v1/system/backend` now (1) checks the target's health as before, (2) primes the target pair with every constant prefix
(the system prompt per domain on both models, the agent's prompt; `Orchestrator::warm_pair`), (3) swaps the active pair in one
atomic pointer exchange. Requests already in flight hold the pair they started on and finish there; new requests go to the new
pair with its caches warm. The response reports `warmed_prefixes`, `warm_ms` and `in_flight`. Measured on the shared host after a gateway restart: the warm-up of one pair (6 domains x 2 models + the agent prompt = 13
prefixes) is what start-up already did; a switch now pays it once, before the swap, instead of the first 13 requests each
paying a cold prefill.

## 6. The harness cascade (`scripts/harness_ab.py --arms s2,cascade`, `benchmarks/harness_ab_cascade.json`)

Design (`nafs_harness::agent`, `nafs_core::harness`): with `harness.cascade = true` the first step goes to System 1
(`Stage::Cheap`) as **one request with `cascade_samples` choices** (`Backend::stream_n`: the prompt is prefilled once, the
choices decode in one batch, at the rescue's temperature 0.7 so they differ). Each candidate's action is applied to its own copy
of the files and judged by the exact verifier, in order; the first that passes ends the run and the rest are not judged. When
none passes the best-scoring candidate (per-assert credit) becomes the state, its reply and observation stay in the transcript,
and System 2 continues from step 2 as before. A candidate is offered no `run` (it is judged by its files). A run asked for on
System 1 itself gets no cascade. Retries: a model call that fails for a reason other than the run's cap is retried up to
`harness.retries` (2) times with Codex's backoff (200 ms, doubling, jitter 0.9-1.1).

Interleaved live A/B on the ten harness tasks (`benchmarks/harness_ab_cascade.json`), System 2 as the run's model, production
defaults otherwise, on the shared host (load 5-16; wall times are not speed claims, the pairs are):

| Arm | Verified | Model calls | Steps on System 1 / System 2 | Wall time | Median per task | Malformed steps |
|---|---|---|---|---|---|---|
| Phase 11 default (System 2 from step 1) | 10/10 | 12 | 0 / 12 | 1,987 s | 96 s | 0 |
| cascade (4 System 1 candidates first) | 10/10 | 14 | 10 / 4 | **1,467 s (-26 %)** | 100 s | 0 |

| Task | Phase 11 default | cascade | first step |
|---|---|---|---|
| fix_binary_search | 78 s | **43 s** | System 1 passed |
| fix_fizzbuzz | 583 s | 535 s | failed, System 2 fixed it |
| fix_merge_intervals | 342 s | **131 s** | failed, System 2 fixed it |
| fix_palindrome_case | 93 s | 99 s | System 1 passed |
| fix_roman | 276 s | **25 s** | System 1 passed |
| fix_second_largest | 99 s | 289 s | failed, System 2 fixed it |
| fix_word_count | 322 s | **216 s** | failed, System 2 fixed it |
| write_caesar | 60 s | **18 s** | System 1 passed |
| write_flatten | 41 s | **10 s** | System 1 passed |
| write_rle | 94 s | 101 s | System 1 passed |

- **Six of ten tasks never reached System 2**: a four-candidate System 1 request verified them in 10-101 s (median 34 s against
  85 s for the same tasks on System 2). The other four paid one System 1 step before System 2's; two of those still ended
  faster (System 2's own time is random: the sequential arm took 342 s and 322 s on them), two slower (+190 s on
  `fix_second_largest`, where the Phase 11 arm's single System 2 call was a fast 99 s).
- The cascade was faster in 7 of 10 pairs (exact sign test p = 0.17: indicative, not significant at n = 10) and used 26 % less
  wall time in total; verified counts are equal, and no step was malformed in either arm (the grammar holds on System 1 too).
- The cascade decodes more tokens in total (5,958 against 4,534: four candidates per first step) but on the cheap model in one
  batch; what it saves is System 2 steps (4 against 12).
- Reporting gap: the server's timings for an `n`-choice request arrive after every choice has been handed on, so a cascade step
  that ends the run shows `prompt_tokens` and `cached_tokens` as 0; the counts of the System 2 steps are complete.

**Adopted: `harness.cascade = true`, `cascade_samples = 4`.** Re-open if a 30-task interleaved run on a quiet host shows the
cascade losing verified tasks, or its first step costing more than it saves on tasks System 1 cannot do.

## 7. What changed, and what did not

| Change | Evidence | Default |
|---|---|---|
| Harness cascade: System 1 candidates judged by the verifier first, System 2 only when none passes (FR-097) | 10/10 both arms; -26 % wall time, faster in 7 of 10 pairs; 6 of 10 tasks never reached System 2 | on (4 candidates) |
| Retries with Codex's backoff of a failed model call (FR-098) | unit tests; a dropped stream no longer ends the run | on (2) |
| Warm-first backend switch (FR-099) | the target pair is primed before the atomic swap; in-flight requests finish on the old pair | on |
| Engine flags (flash attention, micro-batch, threads, polling, mlock) | inconclusive on the shared host (per-round ratios 0.07x-3.8x) | unchanged |
| Co-scheduling of the two servers (polling, System 1 threads) | the pair writes 7.1-7.8 tokens/s together in every variant; nothing to arbitrate | unchanged |
| Prefetch of the System 2 prompt during the draft | the switch inside a request costs 1.1-4.7 s; no System 1 slot busy after the decision | not built |
| Parallel tool calls, dispatch while streaming (Codex) | nothing left to overlap under the grammar | not ported |
| Ternary models, table-lookup kernels, checkpoint-loading switches | change the model, or solve a GPU-memory problem this host does not have | not pursued |
