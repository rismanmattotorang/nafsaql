# Serving many users from Smiley Core on a commodity CPU (Phase 18)

Date 2026-11-03. The owner asked for research into how vLLM serves many users, then into the wider literature, and for the
best of it to be built into Smiley Core so that Nafs-Aql's default engine serves several people at once. Decision record:
[012](../decisions/012-multiuser-engine.md). Report: section "Phase 18". Scripts: `scripts/concurrency_ab.py`,
`crates/smiley-core/examples/batch_bench.rs`, `scripts/membw.c`. Records in `benchmarks/concurrency_ab.json`.

Everything here was measured on the reference machine (Intel i3-8100, 4 cores, AVX2, no AVX-512 and no AMX, 19.5 GB/s of
memory bandwidth) with Qwen3-1.7B Q4_K_M, each engine alone on the machine and the arms interleaved.

## 1. What vLLM does (read from the source)

Read at `third_party/vllm`, commit `9c778e9ff8` (2026-10-17), the V1 engine.

| Piece | What it is | Where |
|---|---|---|
| **Continuous batching** | one `step()` = schedule, execute, commit; a decode is simply a request with one new token, so prefill chunks and decode rows go through the same path | `vllm/v1/engine/core.py:657`, `vllm/v1/core/sched/scheduler.py:714` |
| **Token budget** | each step admits work up to `max_num_batched_tokens` (2,048 on CPU) and `max_num_seqs` (128 on CPU), running requests before waiting ones | `scheduler.py:641`, `arg_utils.py:3007` |
| **Chunked prefill** | a long prompt is fed a slice at a time so it never stops everyone else's decode; on by default | `scheduler.py:1184` |
| **PagedAttention** | the cache is blocks addressed through a per-request block table; a partial prefix hit copies the shared tail block | `kv_cache_utils.py`, `block_table.py:320` |
| **Prefix caching** | blocks hashed in a chain, reused across requests, least recently used evicted; on by default | `kv_cache_utils.py:707`, `block_pool.py:776` |
| **Preemption** | when the cache is full a request's blocks are freed and it is recomputed later; V1 has no swap | `scheduler.py:1649` |
| **The CPU path** | one fused attention call over a flattened batch, parallelised over `(request × KV split)` work items balanced by **total KV length over thread count**, not by request count | `csrc/cpu/cpu_attn_impl.hpp:408` |

Two details decided this design. First, the batch is **flattened and varlen**: tokens from every request sit in one tensor
with a cumulative-length array, and attention learns a token's sequence and position from that. Second, **paging exists
for video memory**: a GPU has little of it and fragmentation is fatal. Neither is true here, which is why the
implementation below keeps a contiguous cache per user and spends the complexity elsewhere.

## 2. What the literature and the machine say

The decisive number is not in a paper; it is the machine's roofline. This CPU peaks near 460 GFLOP/s and reads 19.5 GB/s,
so its ridge is about 24 FLOP per byte. Q4_K holds about 0.56 bytes per weight, so a batch of `b` rows has an arithmetic
intensity near `3.6 b`: the ridge is crossed near **seven rows**, after which the kernels are compute-bound and another
user buys nothing. Measured on this machine with llama.cpp's batched path, throughput rose 1.7x at 2 rows, 2.8x at 4 and
3.5x at 8, then flattened — the knee the roofline predicts. Smiley Core's own primitive (`examples/batch_bench.rs`)
measures 1.56x at 2 rows, 2.57x at 4 and 2.27x at 8.

| Technique | Verdict here |
|---|---|
| **Batch several users into one weight pass** | built: the one change that turns a memory-bound engine into a multi-user one |
| **Chunked prefill** | built: a 2,000-token prompt is about 20 s on this machine, and without chunking every other user stops for all of it |
| **Prefix sharing across users** | built: one shared store, which is also how a reasoning continuation avoids recomputing its own reasoning |
| **Paged cache with block tables** | **not built**: it exists for video memory. Here a slot's cache is ordinary memory (112-144 KB per position), the slot count is small, and a block table would add an indirection to the hottest loop for nothing |
| **int8 KV cache** | available (`--kv i8`), not the default: KV traffic does not amortise across a batch, so it is the next lever at long context; it needs its own A/B before it changes a default |
| **Speculative decoding while batching** | **not built**: verifying a draft multiplies rows per pass, and past the ridge there are no spare FLOPs. The literature agrees (2.7x at one row falling to 1.6x at eight) and Phase 7 measured no gain from EAGLE-3 here |
| **AMX / AVX-512 / oneDNN int8 kernels, PowerInfer, LUT kernels for 1-2 bit** | not applicable: this CPU has neither AVX-512 nor AMX, Qwen3 is dense, and the weights are 4-bit |

## 3. What was built

**The forward pass takes rows, not a sequence** (`model.rs`). Every matmul in it — Q, K, V, the output projection, the
feed-forward pair and the 50 MB output head — reads each weight row once for the whole batch and does one dot product per
row, so it never cared which sequence a row belonged to. Only three things did: the rotary embedding, the cache write and
attention. `forward_rows` now takes a position per row and a `Caches` describing where each row reads and writes:
`Shared` (one sequence over several positions: prefill, a speculative block) or `PerRow` (one position of each of several
sequences: several users). The old `forward` is a thin wrapper, so nothing else changed.

**A scheduler** (`serve.rs`, `Engine`). Slots hold a user each; `step()` is either a chunk of one user's prompt or one
token for every ready user, never both — a prefill chunk and a decode row would need two mutable borrows of one cache.
Admission refuses when every slot is busy or the prompt cannot fit, and hands the prompt back rather than truncating it.
A finished or cancelled slot leaves its cache in a store shared by every slot, so the next user who begins the same way —
the system prompt of a deployment, the transcript a code agent extends, the reasoning a budgeted turn continues — does not
prefill it again.

**The server serves from one engine thread** (`nafs-smiley-server`). It used to hold one session behind a mutex, so a
second request waited for the first to finish. Requests now go to a thread that owns the engine and batches them; it
sleeps when idle. `--slots` (4 by default) sets how many users share a pass, below the machine's ridge.
`[backends.smiley]` in the gateway stays at `max_in_flight = 1`, and that is a real limit rather than an oversight: the
gateway reads that number as both the concurrent-request width **and** the width of an `n`-choice request (the cascade's
four candidates, the rescue's samples), which this server still refuses with a 400. Raising it to 4 was tried and made
the code agent fail at once (`model_error` on the first step, caught by the harness sweep). The engine's concurrency is
therefore available to direct API clients today, and reaches the gateway when the server learns to answer `n > 1` — which
is now a small thing, since `n` candidates of one prompt are exactly `n` rows of one pass.

**What is pinned by tests.** That a user served with others gets *bit-identical* logits and the same text as alone
(`tests/batch.rs`, `serve::tests`); that a row reads its own cache and follows it when the caches are swapped; that a
user arriving mid-flight is served correctly; that a restored prefix changes nothing; that a full engine refuses and a
cancelled slot frees its room. Two real defects were caught this way: a slot whose prefix left nothing to prefill fed
token 0 instead of the prompt's last token, and a finished slot's cache was described by the wrong token sequence, so a
later request restored a cache that did not match its prompt.

## 4. What it achieves

Each engine alone on the machine, two rounds, the same eight questions with long answers (the regime Nafs-Aql's thorough
path is in: a short answer is mostly prefill, which neither engine batches, and there the gain is small).

| Users | Smiley Core, total | per user | Same text as alone | llama.cpp, total | per user | Same text as alone |
|---|---|---|---|---|---|---|
| 1 | 14.2 tok/s | 14.2 | 1/1 | 14.4 tok/s | 14.4 | 1/1 |
| 2 | **20.2 tok/s** | 10.1 | 2/2 | 14.4 tok/s | 7.2 | 2/2 |
| 4 | 24.9 tok/s | 6.2 | **4/4** | **32.4 tok/s** | 8.1 | **0/4** |

- **Before this phase Smiley Core had no concurrency at all**: a second request waited for the first. Two users now cost
  1.42x one user's time instead of 2x, and four cost 1.75x instead of 4x.
- **At two users it is 1.40x llama.cpp**; at one user they are level; **at four users llama.cpp is 1.30x faster**.
- **Every user's text was identical to what they would have been told alone, at every concurrency.** llama.cpp's was not
  at four users, in both rounds. Its answers are deterministic alone and identical at two users, so this is something
  four concurrent slots do to them; the cause was not investigated and is not claimed to be a defect.

**This is not state of the art against vLLM, and the comparison would be dishonest.** vLLM does not run on this machine
(its CPU backend wants AVX-512 and a Linux host; the Lima VM profile here runs a 0.6B model in FP32), so there is no
measurement to make. What can be said is bounded and measured: against llama.cpp, the most capable engine that does run
here, Smiley Core is level at one user, ahead at two, behind at four, and alone in keeping every user's answer identical.

## 5. Where the remaining gap is, and what would close it

At four users llama.cpp's time to first token is 0.18 s against 0.6-1.2 s here, and it is the same mechanism in both
places: **llama.cpp mixes prefill chunks into the same pass as decode rows, and this engine alternates them**. Four users
arriving together are prefilled one after another while nobody decodes. Closing it needs a `Caches` variant that maps
several rows to the same cache (the writes are already sequential, so only the borrow shape is in the way), after which a
step can carry one prefill chunk and every decode row at once — which is exactly vLLM's flattened batch. That is the
named next piece of work; the ridge measurement says it is worth more than any further kernel tuning.

Two smaller ones, in order: attention work is split across threads by `(row, head)` count, where vLLM's CPU kernel
balances by **total KV length** (with a long prompt beside short ones, one thread gets the heavy row); and the int8 KV
cache halves the per-user cache traffic that batching cannot amortise, which is what limits concurrency at long context.
