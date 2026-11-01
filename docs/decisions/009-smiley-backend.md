# Decision 009: Smiley Core as the third inference backend (Phase 15)

Date 2026-11-01. Status: **adopted, and made the default backend after measuring it against llama.cpp.** Research note:
[`docs/research/smiley_backend.md`](../research/smiley_backend.md). Report: section "Phase 15".

## Context

The owner asked to integrate Smiley Core, the Rust CPU inference engine of their Smiley project (`third_party/smiley-core`,
revision 7cb6028, Apache-2.0), as a third backend beside llama.cpp and vLLM, reusing its code. The gateway already speaks one
OpenAI-compatible API to two kinds of server, so the integration is a server around the engine plus a configured pair.

## Decision

| What | Where | Default |
|---|---|---|
| The engine, vendored with two marked additions (`assistant_prefix`, `Emitted::logprob`) | `rust/crates/smiley-core` | builds with the workspace |
| `nafs-smiley-server`: one model per process behind `/v1/chat/completions` (stream, `usage`, llama-style `timings` plus the bytes moved), `/v1/models`, `/health`, `/metrics`; thinking on/off and the reasoning budget (continuation with `</think>`), `logprobs`; `n > 1` and `grammar` refused | `rust/crates/nafs-smiley-server` | — |
| `[backends.smiley]` (`kind = "smiley"`, :8201/:8202, `max_in_flight = 1`), `scripts/serve_smiley.sh`, `make up` | config, scripts, Makefile | **active**: `backend.active = "smiley"`; `make up` serves it, `make up-llama` switches to llama.cpp |

## Evidence

The owner asked for the winner to be the default. The criterion, fixed before the end-to-end run: time to a checked answer
through the gateway, with the answers equally checked. Two measurements (research note, section 3):

- **The engines alone** (`scripts/smiley_ab.py`, 12 pairs per model, interleaved): decode 14.3 vs 12.6 tokens/s on System 1
  and 7.4 vs 6.6 on System 2 (Smiley Core faster in 10 of 12 and 11 of 12 pairs, median 1.13x and 1.14x); first token
  0.64 vs 0.11 s on System 1 (llama.cpp earlier), 1.60 vs 1.85 s on System 2; identical answers 12/12 and 9/12.
- **The whole service** (`scripts/backend_ab.py`, the gateway switched between backends per request, 12 checkable and open
  prompts): 9/9 checked on both; 131 s against 303 s for the twelve (median 9.7 vs 17.9 s), Smiley Core faster in 10 of 12
  pairs (p = 0.02, median 1.59x). The code agent verified its task on both after a parser fix (263 s on Smiley Core; 53 s and
  291 s in llama.cpp's two runs: no ranking at n = 1).

Smiley Core wins the criterion for one user at a time. Its first run of the code agent failed (900 s deadline) and found a
harness defect that applies to every backend without a grammar: a patch missing its `*** Begin Patch` line was read as "no
action". Fixed in `find_patch` and `action_end`, unit-tested (`a_patch_without_its_begin_line_is_still_found`).

## Not adopted

- **llama.cpp as the default:** it keeps four slots, the `n`-choice rescue, the cascade's four candidates, the harness grammar
  and the raw-continuation exit, none of which Smiley Core has, so it serves several users and the code agent better; it is
  slower for one user. It stays one command away (`make up-llama`, or the System page) and is the choice for a shared
  deployment. Docker compose keeps vLLM (the container has no Smiley build).
- **In-process embedding of the engine in the gateway:** the gateway's backends are processes behind HTTP by design (a crash in
  an engine never takes the gateway down; the same `Backend` code serves all three); Smiley itself runs its third-party engines
  out of process for the same reason.
- **Batching, grammar and raw continuation on this backend:** the engine decodes one stream per session and has no constrained
  sampler; adding them would be engine work, not integration. The gateway degrades as it does for vLLM.

## Re-open conditions

- Smiley Core gains batched decode (several sequences per forward) or a constrained sampler: raise `max_in_flight`, enable the
  rescue and the grammar on it, and re-run the comparison.
- A shared deployment with more than two concurrent users, or a workload dominated by the code agent: set `backend.active =
  "llama"` (the measured costs above are what is given up).
- A repeated end-to-end run on a quiet host contradicts the 10-of-12 result.
