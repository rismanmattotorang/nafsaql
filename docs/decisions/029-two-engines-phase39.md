# Decision 029: Two inference engines, chosen per role (Phase 39)

Date 2026-11-27. Status: **adopted.** Assessment [`peregrine_assessment.md`](../research/peregrine_assessment.md);
measurements `benchmarks/engines_phase39/`.

## Context

Nafs-Aql ran on one CPU engine at a time: Smiley Core (ParagonCorp, the default), llama.cpp or vLLM. RantAI's Peregrine
is a second Rust CPU engine with the same OpenAI contract. It also has batched slots, `n > 1`, constrained JSON and a
faster embedding path. The owner asked for both engines to be selectable, System 1 and System 2 to run on either, a
benchmark inside the product, and then a pass over knowledge-graph and RAG construction and retrieval.

## Decisions

1. **Peregrine is a fourth engine kind**, `peregrine`, served by `scripts/serve_peregrine.sh` (:8211, :8212, embeddings
   :8213) from `third_party/peregrine`. It is not vendored: it is a separate checkout under its own Apache-2.0 licence
   (NOTICE).
2. **Each role picks its engine**: `[backend] system1` / `system2`, the selection `"<system1>/<system2>"`,
   `POST /v1/system/backend {"system1": ...}`, and the System page. One engine for both roles stays the plain name.
3. **The client hides engine differences.** A fresh seed goes on every unseeded Peregrine sample (its absent seed is
   42), streamed `n` becomes `n` concurrent streams, and query embeddings carry their instruction in the text.
4. **Smiley Core stays the default** (System 1 equal at p = 0.77, System 2 faster in 9 of 9 pairs). Peregrine is
   recommended for several users at once and for indexing.
5. **The engine comparison is a product feature** (`POST /v1/bench/engines`, *Speed → Compare engines*), interleaved
   and paired. It counts as work in flight, so idle-only jobs do not run on the same cores.
6. **Graph construction**: `graph_concurrency = 4` (ordered, so the graph is identical) and `graph_constrain = true`
   (JSON Schema `response_format` where the engine supports it).

## Consequences

- Graph builds are 1.63x faster on Smiley Core and 2.5x faster on Peregrine. On Peregrine no passage is skipped for
  an unparsable reply (12 of 12 against 11 of 12).
- Peregrine has no ShadowPrefill and no memory tokens. A role moved to it loses those features, and the Status page
  says which engine serves which role.
- A clean clone without `third_party/peregrine` runs as before; the Peregrine backend then shows as not running.
