# Peregrine beside Smiley Core: assessment and integration (Phase 39)

Date 2026-11-27. Decision [029](../decisions/029-two-engines-phase39.md). Measurements in `benchmarks/engines_phase39/`.

## 1. What Peregrine is

[Peregrine](https://github.com/RantAI-dev/peregrine) is RantAI's CPU inference engine for GGUF models (Apache-2.0,
v0.3.0, commit `194bfb7`, checked out in `third_party/peregrine`). It is pure Rust with one runtime dependency
(`memmap2`). It reports the bytes it moved per token beside tokens per second. Its kernels pick a SIMD tier at run time
(`x86_64-avx2` on the reference i3-8100). What matters to Nafs-Aql:

| | Peregrine | Smiley Core (ParagonCorp) |
|---|---|---|
| API | OpenAI chat, streamed or not, `usage`, llama-style `timings` | the same contract (`nafs-smiley-server`) |
| Qwen3 thinking, `reasoning_content`, `chat_template_kwargs` | yes | yes |
| Reasoning budget (`thinking_budget_tokens`) | yes (closes `</think>` and continues) | yes |
| Log-probabilities for the gate | yes (token and logprob only) | yes |
| Several users at once | one weight pass for every active row (batched slots) | slots, one pass per step |
| `n > 1` | yes, not streamed (400 with `stream`) | no |
| Constrained JSON (`response_format`) | yes, JSON Schema subset | no (ignored) |
| GBNF grammar, raw `/completion` | no (grammar is a 400) | no |
| ShadowPrefill, Memorilla memory tokens | no | yes (Nafs-Aql additions) |
| Embeddings, rerank, decisions | `/v1/embeddings`, `/v1/rerank`, `/v1/systemone` | `--embedding` only |
| Unseeded sampling | **seed 42 when none is sent** | a fresh seed |

The last row matters for the gate, so the client handles it (section 2).

## 2. Integration

- **A fourth engine kind**, `peregrine`, under `[backends.peregrine]` (System 1 on :8211, System 2 on :8212,
  `max_in_flight = 4`). `scripts/serve_peregrine.sh` builds it from `third_party/peregrine` and runs one process per
  model with 3 decode threads, like Smiley Core, so the comparison is like for like. `make up-peregrine`
  (`ROLE=system1|system2` for one role only), `make down-peregrine`.
- **One engine per role.** `[backend] system1` / `system2` (default: `active`), `NAFS_BACKEND=peregrine/smiley`,
  `POST /v1/system/backend {"system1": "peregrine"}`, and a picker on the System page. A selection is named
  `"<system1>/<system2>"`. The composed pair shares the per-engine `Backend` objects, so slot permits stay per server.
  Switching warms the new pair first and checks only the servers that move.
- **Fresh seeds.** Peregrine draws from (seed, turn, position) and reads an absent seed as 42. Two unseeded samples of
  one prompt would be the same text, and the gate's agreement feature would always see agreement. The client now sends
  a random seed on every unseeded Peregrine request with temperature above zero.
- **Streamed `n`.** Peregrine refuses `n` with `stream`, so `stream_n` sends `n` concurrent single streams on Peregrine
  (each with its own seed). The server batches them into one weight pass.
- **Engine-neutral query embeddings.** The knowledge client sent the retrieval instruction in a Smiley-only
  `instruction` field. Peregrine ignored it and embedded the bare question. The client now writes Qwen3-Embedding's
  query format (`Instruct: ...\nQuery:...`) into the text itself, so Smiley Core gets identical text and identical
  vectors.
- **Engine comparison in the product.** `POST /v1/bench/engines` (and *Speed → Compare engines*). It sends the same
  prompts to System 1 and System 2 of each running engine, greedy. Engines take turns inside each round and swap order
  between rounds. The run counts as a request in flight, so idle-only work (audits, labels, cache building) pauses
  during it. It reports time to first token, reading and writing speed, wall time, correctness and a paired sign test.
  `scripts/engine_compare.py` adds the end-to-end and several-users levels.

## 3. Measurements

Reference machine: Intel i3-8100, 4 cores, no GPU, shared with other work (load 5 to 26 during the runs, recorded per
request). Same GGUF files: Qwen3-1.7B Q4_K_M (System 1), Qwen3-4B-Thinking-2507 Q4_K_M (System 2),
Qwen3-Embedding-0.6B Q8_0. Arms interleaved throughout.

**Engine level** (`engine.json`; 3 rounds, greedy; System 2 with a 512-token reasoning budget):

| Model | Engine | Right | First token | Read tok/s | Write tok/s | Typical answer |
|---|---|---|---|---|---|---|
| System 1 | Smiley Core | 12/12 | 384 ms | 67 | 17.0 | 0.49 s |
| System 1 | Peregrine | 12/12 | 406 ms | 88 | 14.9 | 0.53 s |
| System 2 | Smiley Core | 9/9 | 215 ms | 23 | 10.0 | 51.2 s |
| System 2 | Peregrine | 9/9 | 244 ms | 21 | 9.0 | 62.7 s |

System 1: no difference in wall time (Peregrine faster in 7 of 12 pairs, median ratio 0.97, sign test p = 0.77).
Peregrine reads prompts 1.3x faster, and Smiley Core writes 1.14x faster. System 2: Smiley Core is faster in 9 of 9
pairs (median ratio 1.11, p = 0.004).

**Several users** (`conc.json`; 4 simultaneous quick-model requests, 2 rounds): Peregrine 39.4 tokens/s in total
against 31.6 for Smiley Core (1.25x), no errors.

**Knowledge-graph construction** (`graph_bench.json`; 12 passages of the vLLM paper, blocks of 4 in rotating order):

| Extraction | Passages/min | Replies that parse | Entities | Relations kept |
|---|---|---|---|---|
| Smiley Core, one at a time (the old default) | 3.00 | 11/12 | 89 | 39 |
| Peregrine, one at a time | 3.45 | 11/12 | 74 | 40 |
| Peregrine, 4 at once | 6.42 | 11/12 | 74 | 40 |
| Peregrine, 4 at once, schema-constrained | **7.61** | **12/12** | 73 | 35 |

Four at once gives the same graph as one at a time: the same 74 entities and 40 relations, because `buffered` keeps
passage order and decoding is greedy. On Smiley Core (`graph_bench_smiley_x4.json`, 8 passages) four at once is 5.54
against 3.39 passages a minute (1.63x), also with an identical graph. The engines extract different entity sets (mean
Jaccard 0.63): both pass the same grounding and quote checks, and which is better is not measured here.

**Embeddings** (`embed_ab.json`; 48 passages of about 1,000 characters, batches of 8, 2 rounds): Peregrine is faster
in 12 of 12 batches (median 1.41x; 0.72 passages/s). The vectors agree: cosine at least 0.9996 for passages and
0.9997 for queries, and the same top-5 passages for 93 % of questions. A base embedded on one engine can be searched
with the other.

**Through the whole service** (`e2e.json`): partial, see section 5.

## 4. What was adopted

- Smiley Core stays the default engine for both models. It is equal on System 1, faster on System 2, it is the only
  engine with ShadowPrefill and memory tokens, and it builds with `cargo build` from this repository.
- Peregrine is the choice for several users at once (1.25x total throughput), for faster indexing (embeddings 1.41x),
  and for System 1 prompt reading (1.3x). It is a selectable engine for either role.
- `graph_concurrency = 4` and `graph_constrain = true` are the new defaults. They build the graph 1.63x faster on
  Smiley Core and 2.5x faster on Peregrine against the old setting, and on Peregrine every reply parses.

## 5. End to end

**Partial: 2 of 24 planned requests** (`e2e.json`). The run was stopped to finish the release, because each question
escalated to System 2 takes about 2 minutes on this machine. On the first maths question (the train trip, 145
minutes), Smiley Core's System 1 drafted an answer that passed the check in 9.7 s. With System 1 on Peregrine, the
draft failed the check, and System 2 (Smiley Core) answered correctly after 120.6 s. That is one question, and greedy
drafts from the two engines differ because their arithmetic differs, so it says nothing about rates. The full run is
`python3 -I scripts/engine_compare.py e2e --rounds 2 --resume`. It resumes from the stored rows, and every selection
goes first equally often. Until it has run, the recommendation rests on the engine-level results above.

## 6. Further work

- Peregrine's `/v1/rerank` with a BERT reranker (bge-reranker-v2-m3) for the reworded-question gap, which is the
  largest open retrieval problem. It needs a model download, so it waits for the owner's approval.
- Peregrine's `--fast-prompt` (8-bit prompt weights; 1.05-1.08x on AVX2) and `peregrine tune` profiles, measured
  against accuracy on the ledger.
- `n`-choice rescue and the four-candidate cascade on Peregrine: batched choices without a shared prefill. Measure
  them against llama.cpp's forked-KV `n`.
- One Peregrine process for both models (`-m` twice) takes weight passes in turns instead of competing for cores.
  This may suit the mixed selection better than two processes.
