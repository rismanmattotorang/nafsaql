# Smiley Core as the third inference backend (Phase 15)

Date 2026-11-01. The owner asked to integrate Smiley Core, the Rust CPU inference engine of their Smiley project
(`third_party/smiley-core`, mirrored from `~/Downloads/Smiley/smiley/engine`, revision 7cb6028, Apache-2.0), into Nafs-Aql as a
third backend beside llama.cpp and vLLM, reusing the code rather than re-implementing it. Decision record:
[009](../decisions/009-smiley-backend.md). Report: section "Phase 15".

## 1. What Smiley Core is, and what it can serve

A library crate of about 8,500 lines with one dependency (`memmap2`): a GGUF loader, a byte-level BPE tokenizer checked
against `llama-tokenize`, the `llama`, `qwen2` and `qwen3` forward passes with AVX2+FMA kernels, an F16 or int8 KV cache, a chat
session with its own templates (ChatML, Qwen3 with `enable_thinking`, Llama 3), prompt-lookup and DSpark speculative decoding,
and telemetry that reports the **bytes moved per token** beside the tokens per second, so a run can be checked against the
machine's memory roofline. It decodes **one stream per session**; there is no batching across requests, no constrained
(grammar) sampling, and no raw continuation endpoint. Its `compatibility()` probe refuses, from the header alone, any file it
cannot run exactly.

Both Nafs-Aql models pass the probe: Qwen3-1.7B and Qwen3-4B-Thinking-2507 in Q4_K_M (Q4_K and Q6_K tensors, `qwen3`
architecture, the Qwen3 template).

## 2. How it is integrated

| Piece | Where | What |
|---|---|---|
| The engine | `rust/crates/smiley-core` (vendored; `third_party/` is not in version control) | upstream source plus two marked additions: `SessionCfg::assistant_prefix` and `Emitted::logprob` (`README.md` there) |
| The server | `rust/crates/nafs-smiley-server` | one model per process behind the OpenAI chat API the gateway speaks: `POST /v1/chat/completions` (streamed or not, `usage`, llama-style `timings` plus a `smiley` block with the bytes moved), `GET /v1/models`, `/health`, `/metrics` |
| The backend | `config/nafs-aql.toml` `[backends.smiley]`, `kind = "smiley"` | the gateway's `Backend` needs no new code path: a kind that is not `llama` gets the standard chat path (no grammar, no raw continuation), as vLLM does; `max_in_flight = 1` per model |
| Serving | `scripts/serve_smiley.sh`, `make up` | System 1 on :8201, System 2 on :8202; the default backend (decision 009); `POST /v1/system/backend {"name": "llama"}` or `make up-llama` switches the gateway to llama.cpp (warm-first, decision 008) |

Request mapping in the server:

- `messages`: the system message becomes the session's system text, earlier user/assistant pairs are seeded as history, the last
  user message is the turn. The session is reset per request, and its cache prefix is reused when the new prompt shares it
  (the system prompt, the agent's prompt), which is what the gateway's warm-up primes.
- `chat_template_kwargs.enable_thinking` -> the session's `thinking` (Qwen3's empty reasoning block when off, as the model's own
  template does).
- `thinking_budget_tokens` -> the reasoning span is counted by token; at the budget the turn is cancelled and **replanned with
  the reasoning so far plus `</think>` as the assistant's prefix**, so the answer follows, as llama-server's budget forcing
  does. The cache prefix is reused, so only the forced tail is prefilled.
- `temperature`, `top_p`, `top_k` (default 40), `min_p` (default 0.05), `seed`: sampled as llama-server does by default (section 3.3).
- `logprobs` -> each content token's log-probability under the raw softmax (the gate's fixed rule reads the mean and the minimum).
- `n > 1`, `grammar`: refused with 400. The gateway's rescue (one `n`-choice request) and the harness grammar therefore do not
  run on this backend: a failed draft escalates without the rescue, and the agent's actions are parsed from free text with the
  stream cut at the end of the action, as the Phase 10 loop did.
- Reasoning is streamed as `reasoning_content`, recognised by the `<think>` / `</think>` token ids of the vocabulary.

## 3. Measurements

### 3.1 The engines alone (`scripts/smiley_ab.py`, `benchmarks/smiley_ab.json`)

Both engines driven through their own chat endpoint with identical requests (greedy; System 1 without thinking, 160 tokens;
System 2 with thinking and a 128-token budget, 320 tokens), four prompts (maths, code, open, factual), three rounds, the engine
order alternated per prompt; on the shared host (load 7-22). Medians:

| Model | Engine | Decode tok/s | Prefill tok/s | First token | Wall per answer | Bytes moved per token | Same answer as the other engine |
|---|---|---|---|---|---|---|---|
| System 1 (Qwen3-1.7B) | llama.cpp | 12.6 | 18.5 | 0.11 s | 5.7 s | - | 12/12 |
| System 1 | **Smiley Core** | **14.3** | 54.4 | 0.64 s | 5.2 s | 1,094 MB | 12/12 |
| System 2 (Qwen3-4B-Thinking) | llama.cpp | 6.6 | 20.6 | 1.85 s | 38.7 s | - | 9/12 |
| System 2 | **Smiley Core** | **7.4** | 19.2 | 1.60 s | 39.1 s | 2,503 MB | 9/12 |

Per pair (same prompt, same round) Smiley Core decodes faster in 10 of 12 System 1 pairs and 11 of 12 System 2 pairs, by a
median of **1.13x** and **1.14x**. The bytes moved per token (1.09 GB and 2.50 GB, the quantised weights once per token) put
both models on the memory roofline: at 14.3 and 7.4 tokens/s that is 15.6 and 18.5 GB/s, the DDR4 bandwidth Phase 13 inferred
for llama.cpp. So the gain is the engine's own overhead, not a different algorithm, and it is the gain one can expect from a
single-stream engine with no other work to do. Every System 1 answer and 9 of 12 System 2 answers are identical between the
engines under greedy decoding (the three differ after many reasoning tokens, where rounding orders diverge). System 1's first
token is later on Smiley Core (0.64 s against 0.11 s): the session is reset and the prompt replanned per request, and the
prefix match recovers the cache but not the tokenizer and planning work; System 2's is earlier (1.60 s against 1.85 s).

### 3.2 The whole service, one backend against the other (`scripts/backend_ab.py`, `benchmarks/backend_ab.json`)

The gateway switched between the backends before every request (`POST /v1/system/backend`, warm-first) and sent the same twelve
prompts in Auto mode, the order alternated per prompt: four maths questions with a reference answer, three code tasks with
hidden tests, two JSON-schema tasks, one factual and two open questions; then one code-agent task on each backend. Host
shared with the owner's other workload (median load 15.7 before the llama.cpp requests, 17.8 before Smiley Core's).

| | llama.cpp | Smiley Core |
|---|---|---|
| Checked answers (the 9 checkable prompts) | 9/9 | 9/9 |
| Requests that took the thorough path | 0 | 0 |
| Wall time, median per request | 17.9 s | **9.7 s** |
| Wall time, 12 requests | 303 s | **131 s** |
| First token, median | **0.97 s** | 1.50 s |
| Tokens written (answer plus reasoning) | 1,223 | 1,018 |

Per pair, Smiley Core was faster in **10 of 12** (one-sided sign test p = 0.02), by a median of **1.59x** (range 0.73x to
7.8x, the tails being load spikes that hit one arm). By kind (llama.cpp / Smiley Core, seconds): maths 176 / 64, code 69 / 32,
JSON 21 / 13, factual 5 / 3, open 33 / 20. Both backends answered every request on System 1 (the gate's decision is the same:
mode agreed in 12 of 12), so this is System 1 speed through the whole gateway; the thorough path was not exercised by this set.
The first token is later on Smiley Core for the reason given in 3.1 (the session is reset and the prompt replanned per
request). The warm-first switch took a median 7.65 s of warm-up per switch.

**The code agent** (a 4-line fix with hidden tests, `/v1/harness/code`, cascade on):

| Run | Backend | Result | Wall | Model calls |
|---|---|---|---|---|
| 1 (before the parser fix, below) | llama.cpp | verified, System 1 failed its four candidates, System 2 fixed it | 291 s | 2 |
| 1 | Smiley Core | **not verified**, 900 s deadline, every step ended "no action found" or in a model error | 900 s | 6 |
| 2 | llama.cpp | verified by one of System 1's four candidates | **53 s** | 1 |
| 2 | Smiley Core | verified: System 1's one candidate failed, System 2's patch passed | 263 s | 2 |

The first Smiley Core run exposed a defect that is not the engine's: without a grammar the harness reads the model's free text,
and Qwen3-1.7B writes the patch without its `*** Begin Patch` line (the same on llama.cpp when it is not constrained; replayed
with the same prompt on both servers), which the parser read as "no action". `find_patch` and the stream cut now accept a patch
that starts at its first file section, with a unit test. After the fix both backends verify the task. One run per backend
cannot rank them (llama.cpp's own two runs differ 5.5x); what is structural is that Smiley Core has one slot, so the cascade's
first step has one candidate instead of four, and a task System 1 can do in one of four tries needs System 2 on Smiley Core.


### 3.3 Sampling, and the acceptance suite on the default backend

llama-server samples with `top_k = 40` and `min_p = 0.05` unless told otherwise; Smiley Core had temperature and top-p only. The
first System 2 acceptance runs on Smiley Core wrote 1,025 reasoning tokens (the budget) against 274 on llama.cpp for the same
question, so the sampler was brought in line: `SessionCfg::top_k` / `min_p` (marked `// nafs-aql:`, off by default in the engine,
unit-tested), and the server's defaults are 40 and 0.05, overridable per request. Over three seeds on the first GSM8K question
(System 2, thinking budget 1,024, interleaved, shared host at load 5-8) llama.cpp wrote 952, 906 and 917 tokens in 174, 148 and
150 s; Smiley Core 1,079, 991 and 1,040 tokens in 146, 123 and 127 s: about 12 % more reasoning tokens and about 16 % less wall
time.

The MVP acceptance suite (9 tests) on Smiley Core: 7 of 9 passed on the first run, on a host still loaded by the preceding sweep.
A1 (first token within 2 s) took 9.0 s then and 1.7 s, 0.29 s and 0.06 s on quieter hosts; A3 (slow path within 240 s) took
309 s then, 193 s on the quiet re-run and 195 s after the sampler change (llama.cpp, same quiet run: 64 s, after reasoning 274 tokens); the record now reads 9 of 9 on Smiley Core. The 240 s budget is within reach
on Smiley Core only on a quiet host: System 2 decodes 7.4 tokens/s and a 1,024-token reasoning span is 138 s of it. A9 (degraded
mode) stops and restarts System 2 through the active engine's script (`stop-one` was added to `serve_smiley.sh`).

## 4. What it gives Nafs-Aql, and what it does not

**What it gives Nafs-Aql.** On the owner's machine, with one user at a time, the whole service answered the twelve-prompt set
in 43 % of llama.cpp's time with the same 9 of 9 checks passed, because the engine is 13 % faster on every token (3.1) and the
memory-bound decode is where System 1 spends its wall time. It is also a Rust engine in the same workspace (one `cargo build`,
no C++ toolchain, no third-party process to build or patch), and it reports the bytes moved per token.

**What it does not.** It decodes one stream per model. Therefore (a) a second concurrent request waits (`max_in_flight = 1`;
the runbook's two-session plan is for llama.cpp's four slots), (b) the `n`-choice rescue and the cascade's four candidates
become one sample, (c) there is no grammar, so the code agent depends on the free-text parser (hardened above) and a
malformed action costs a step, (d) there is no raw continuation, so the adaptive reasoning exit is off. llama.cpp keeps all
four and stays one command away (`make up-llama`, or the System page).

**Verdict.** On the criterion fixed before the run (time to a checked answer through the gateway), Smiley Core is the faster
backend for the single-user case and is the default (decision 009). llama.cpp is the choice when several people use the
system at once or when the code agent is the main use.

