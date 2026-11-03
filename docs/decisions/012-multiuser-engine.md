# Decision 012: Smiley Core serves several users in one weight pass (Phase 18)

Date 2026-11-03. Status: **adopted.** Research note: [`docs/research/smiley_multiuser.md`](../research/smiley_multiuser.md).
Report: section "Phase 18".

## Context

Smiley Core has been Nafs-Aql's default engine since Phase 15, and it decoded one stream: a second request waited for the
first to finish (`max_in_flight = 1`). The owner asked for vLLM's serving design to be studied and the best of it built in
Rust, so the default engine serves many users. A decode step reads every weight once, so serving `n` users one after
another reads the model `n` times where one pass would read it once — on a memory-bound machine that is nearly the whole
difference.

## Decision

| What | Where | Default |
|---|---|---|
| The forward pass takes a position and a cache **per row** (`Caches::Shared` / `PerRow`), so rows of different users share one read of the weights | `smiley-core/src/model.rs` | always |
| A scheduler with slots, admission control and **chunked prefill**: a step is one chunk of one prompt or one token for every ready user | `smiley-core/src/serve.rs` | 4 slots |
| A **prefix store shared by all slots**: a finished slot's cache serves the next user who begins the same way | `smiley-core/src/serve.rs` | 768 MB per model |
| The server owns **one engine thread** instead of a session behind a mutex; `--slots` sets the width | `nafs-smiley-server` | 4 |
| `[backends.smiley] max_in_flight` | `config/nafs-aql*.toml` | 4 (was 1) |

## Evidence

Each engine alone on the machine, two rounds, long answers (research note, section 4): Smiley Core 14.2 / 20.2 / 24.9
tokens per second at 1 / 2 / 4 users against llama.cpp's 14.4 / 14.4 / 32.4. Two users now cost 1.42x one user's time
rather than 2x. Every user's text was identical to what they would have been told alone at every concurrency (llama.cpp's
was not, at four users). The engine primitive alone (`examples/batch_bench.rs`) gives 1.56x at 2 rows, 2.57x at 4, 2.27x
at 8 — the knee the machine's roofline predicts near seven rows (460 GFLOP/s over 19.5 GB/s against Q4_K's 0.56 bytes per
weight).

Pinned by tests: a batched user's logits are bit-identical to its solo logits; a row reads its own cache and follows it
when caches are swapped; a user may arrive mid-flight; a restored prefix changes no answer; a full engine refuses and
hands the prompt back. Two defects were caught by those tests during the work (a slot whose restored prefix left nothing
to prefill fed the wrong first token; a finished cache was stored under the wrong token sequence).

## Not adopted

- **A paged cache with block tables.** It exists because video memory is scarce and fragments; here a slot's cache is
  ordinary memory and the slot count is small, so it would add an indirection to the hottest loop for nothing.
- **Speculative decoding while batching.** Verifying a draft multiplies the rows per pass, and past the ridge there are
  no spare FLOPs; the literature shows the gain collapsing with batch size and Phase 7 measured none here.
- **int8 KV as the default.** It is available (`--kv i8`) and is the right next lever at long context, where the
  per-user cache traffic that batching cannot amortise dominates, but it has not had its own A/B.
- **Claiming parity with vLLM.** vLLM does not run on this machine (its CPU backend wants AVX-512 and Linux), so there is
  no measurement to make and none is claimed.

## Re-open conditions

- **Prefill mixed into the decode pass.** At four users llama.cpp reaches a first token in 0.18 s against 0.6-1.2 s here,
  because it mixes prefill chunks with decode rows and this engine alternates them. A `Caches` variant mapping several
  rows to one cache would close it; it is the next piece of work and is expected to be worth more than kernel tuning.
- Attention work is split by `(row, head)` count; vLLM's CPU kernel balances by total KV length, which matters when one
  long prompt shares a batch with short ones.
- A machine with more cores, AVX-512 or AMX: re-measure the ridge, and raise `--slots` above 4.
