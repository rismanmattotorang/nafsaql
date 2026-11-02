# Smiley Core: engine speed, model switching and the harness's cost (Phase 17)

Date 2026-11-02. The owner asked for an assessment of Smiley Core's performance, of the switch from System 1 to System 2, and of whether
the new harness costs speed; for the engine, research into Rust concurrency and parallelism on this machine and other current methods;
for the switch, the fastest way to pass from one engine to the other. Decision record: [011](../decisions/011-engine-switching-costs.md).
Report: section "Phase 17". Scripts: `scripts/engine_profile.py`, `scripts/switch_cost.py`, `scripts/handoff_latency.py`;
records in `benchmarks/`. All numbers are from the shared reference machine (Intel i3-8100, 4 cores, DDR4, the owner's other workloads at
load 5-10), so absolute speeds are lower bounds and every comparison interleaves its arms.

## 1. How close is the engine to what the machine allows?

A multi-threaded streaming read (AVX2, 1 GiB, best of five; `scripts/membw.c`) reaches
**12.9 / 17.2 / 18.4 / 19.5 GB/s** with 1 / 2 / 3 / 4 threads: the fourth core adds 6 %. Decode reads every weight once per token, so
tokens per second is bandwidth over bytes per token (1,094 MB on System 1, 2,503 MB on System 2).

The engine's own phase split (`timings.smiley.phases_ms`, added to the server; median of three interleaved rounds, 96 greedy tokens):

| Model | Threads | Decode tok/s | GB/s | Matmul | Output head | Attention | Other | Sampling |
|---|---|---|---|---|---|---|---|---|
| System 1 | 1 | 6.4 | 7.0 | 73.6 % | 22.8 % | 1.3 % | 2.0 % | 0.2 % |
| | 2 | 10.2 | 11.2 | 72.9 % | 21.7 % | 2.0 % | 3.2 % | 0.3 % |
| | 3 | 11.3 | 12.4 | 73.2 % | 19.9 % | 2.2 % | 3.9 % | 0.3 % |
| | 4 | 10.8 | 11.9 | 75.6 % | 16.6 % | 2.2 % | 5.0 % | 0.3 % |
| System 2 | 1 | 2.9 | 7.4 | 84.1 % | 12.8 % | 1.3 % | 1.8 % | 0.1 % |
| | 2 | 4.5 | 11.3 | 83.3 % | 12.4 % | 1.6 % | 2.7 % | 0.1 % |
| | 3 | 5.0 | 12.6 | 82.6 % | 11.9 % | 2.1 % | 3.2 % | 0.1 % |
| | 4 | 4.6 | 11.6 | 83.0 % | 10.3 % | 2.7 % | 3.8 % | 0.1 % |

- **Decode is the matmuls and the output head** (86-97 % together): weights streamed once per token. Sampling is 0.1-0.3 % and attention
  1-3 % at these context lengths, so neither is worth optimising (a top-k fast path in the sampler would save under a percent).
- **The serial share grows with threads** (2 % at one thread, 5 % at four on System 1): norms, rotary embedding, residual adds and activation
  quantisation, the part Amdahl leaves.
- **Three threads beat four for decode in every one of the six interleaved pairs**: +4 % on System 1 and +8 % on System 2 at the median,
  while cold prefill (compute-bound) is 16-17 % slower at three (System 1 43 against 51 tokens/s, System 2 16.2 against 19.4). On a shared host
  the fourth worker is the one the scheduler preempts, and every matmul waits for the slowest worker. Decode dominates a request's time and most
  prompts are now cache hits (section 3), so the default is **3 threads** (`THREADS=4` for a quiet host), which also leaves a core for the
  verifier sandbox and the gateway.
- **The engine's own tuning is already near its ceiling.** Software prefetch distance (`SMILEY_CORE_PF`, interleaved, System 1, three rounds,
  paired against the default 1024): off -10.9 % in two of three rounds, 512 -3.3 %, 2048 +5.0 / +3.1 / -11.3 %, 4096 +5.2 / +1.9 / +1.3 %;
  park instead of spin -1.9 %. The default stays: the differences above 1024 are inside this host's noise (one 2048 round lost 11 %).

So parallelism is not where the next gain is: the engine sits within 60-95 % of the machine's bandwidth depending on thread count and load, and the
4th thread buys 6 % at most. What remains is doing less work per token or fewer bytes per token (section 4), and not redoing work (section 3).

## 2. What the literature offers, and what applies here

| Method | What it is | Verdict |
|---|---|---|
| Decode saturates memory early | decode throughput follows bandwidth over model bytes, and extra threads cost throughput (search results; one tool's measurement) | **confirmed above**: three threads |
| Prompt lookup / suffix decoding | draft tokens by matching the output's recent tokens against the prompt and past outputs; large gains where output copies input (code editing); SuffixDecoding reports up to 3.9x on SWE-bench (authors' number) | **not built**: the engine's speculation is greedy-only and Nafs-Aql samples at 0.3-0.7; it needs speculative sampling (accept a deterministic draft token with probability p(d), else sample the residual), which is lossless but is engine work with a bounded payoff: System 2's time is reasoning text, which does not copy the prompt. Phase 14 measured n-gram speculation at 1.12x at best |
| Prefix (radix) caching | keep the KV of earlier prompts and reuse the longest shared beginning (SGLang's RadixAttention) | **built** (section 3) |
| Lower-bit weights, BitNet | fewer bytes per token | not applicable without retraining or new model files |
| Batched decode | several sequences per weight pass | one stream per model by design (decision 009); llama.cpp keeps it |

## 3. Switching: the three costs, measured

**3.1 System 1 to System 2 inside a request** (`handoff_latency.py`, six requests): the gate's own time is **0.08-0.12 s**, the decision to
System 2's first reasoning token **0.4-1.4 s** (median 0.85 s; 3.0 s once, right after a restart), and **System 1 is idle at the decision (0.0 s
busy)**: each model is its own process with its own mapped weights, so there is nothing to freeze, copy or reload. (llama.cpp, Phase 14: median 1.5 s,
1.1-4.7 s.) The switch is the fastest it can be without prefetching System 2's prompt.

**3.2 Switching between kinds of request on one model.** The session held one cache, so a prompt only benefited when it followed a conversation with
the same beginning. Sending the same model a chat prompt, an agent prompt and an arithmetic prompt in turn (`switch_cost.py`, greedy, prefill of
the prompt only):

| Engine | Same prompt repeated | Agent after chat (alternating) | Rotating three kinds |
|---|---|---|---|
| Smiley Core before, System 1 | 1 token, 0.1 s | **266 tokens, 5.8 s** | 266 / 49 / 58 tokens, 5.8 / 1.0 / 1.3 s |
| Smiley Core before, System 2 | 1 token, 0.2 s | **262 tokens, 14.1 s** | 262 / 45 / 54 tokens, 14.1 / 2.4 / 2.7 s |
| llama.cpp (four slots) | 1 token, 0.1-0.2 s | 1 token, 0.1-0.2 s | 1 token, 0.1-0.3 s |
| **Smiley Core now, System 1** | 1 token, 0.1 s | **1 token, 0.1 s** | 1 token, 0.1 s |
| **Smiley Core now, System 2** | 1 token, 0.2 s | **1 token, 0.2 s** | 1 token, 0.2-0.3 s |

The fix is a **prefix store** in the engine (`smiley_core::prefix`, `KvCache::snapshot/restore`): when a conversation ends, its cache is kept as a
snapshot (about 114-147 KB per position; 768 MB per model by default, least recently used evicted, an entry that is the beginning of a longer one is
dropped), and a new prompt restores the snapshot that shares the longest beginning when that saves more than eight positions. A restore copies
about 30 MB for the agent prompt (milliseconds) against 5.8 s of prefill. The outputs are **byte-identical**: the same greedy answers (equal
SHA-256) with the store on and off, for both models and the three kinds (`switch_cost.py --verify`). It also makes the gateway's warm-up real: the
warm-first backend switch primes 13 prefixes, of which a one-prefix session could keep only the last.

**3.3 The reasoning budget's continuation re-prefilled the reasoning it had just computed.** At the budget the server closes the reasoning and
continues the turn with the reasoning so far as the assistant's prefix. The prefix was built as `<think>` + newline + the reasoning, and the
reasoning already begins with the newline the model wrote, so the first reasoning token became "two newlines" instead of "newline" and the cache
stopped matching at that position: **every reasoning token was prefilled again** (the engine's debug trace showed `match=60` of `kept=155`; at a
budget of 200 the continuation prefilled 108 tokens in 2.2 s, and on System 2 at 16-19 tokens/s that is 7-30 s for 128-512 reasoning tokens). The
prefix now repeats the generated text exactly (`continuation_prefix`, tested); the match covers the whole conversation (155 of 155, 259 of 259,
359 of 359) and the continuation prefills 5 tokens in 0.2 s. It also stops showing the model a prompt with an extra newline that it never wrote.
On the six-request handoff set: **cold requests prefill 350-390 tokens (22-24 s) instead of 700-1,160 (45-78 s)** and the mean time per
request fell from 182 s to 133 s (median 168 s to 133 s); the runs were not interleaved with the old build, but the prefill token counts are
deterministic.

**3.4 A check on the thread default with a long decode.** Acceptance test A3 (a System 2 answer within 240 s) took 245 s and 247 s at a load of 5-7, against 193-195 s
in the quiet runs of Phase 15. To separate the host from the setting, a 450-token System 2 decode was run at three and four threads, interleaved, three
rounds: **4.99 against 4.56 tokens/s (+9 %)**. At the host's present speed (System 2 at about 5 tokens/s, against 7.4 in the quiet A/B) a 1,024-token reasoning span cannot finish in 240 s whatever the
thread count; the test needs a quiet machine.

## 4. What was not built, and why

- **Speculative sampling for prompt-lookup drafts**: see section 2; it is the largest remaining engine item for code workloads and would be the next
  piece of work, with its own A/B.
- **Per-phase thread width** (four workers for prefill, three for decode): the pool's width is fixed for its lifetime and workers compute shares from
  it; changing that is a change to the engine's most delicate primitive, for at most the 17 % of prefill it would win back where the store has not
  already removed the prefill.
- **Sampler top-k fast path, attention tuning, KV int8**: sampling is 0.1-0.3 % and attention 1-3 % of decode; int8 KV halves the KV bytes, which are
  under 12 % of the bytes read even at 2,000 tokens of context.

## 5. The harness's cost

The guards cost nothing in the interleaved A/B over ten tasks (Phase 16): identical verified counts and model calls, wall times within the
host's noise. The compile check costs **37 ms** per edit, 35 ms of it Python's start-up, against steps of 5-230 s; the agent's scripted loop
(86 tests, many with eight steps) runs in 0.03 s in total, so the loop itself is microseconds. The prefix store and the continuation fix are
what the agent loop gains from this phase: its steps share a long beginning (the append-only transcript), which the store keeps and restores.
