# Decision 026: Move caches, claim blocks, read ahead, and let the answering model read between turns (Phase 34)

Date 2026-11-22. Status: **adopted.** Research [`freetoken_assessment.md`](../research/freetoken_assessment.md).
Measurements: `benchmarks/engine_phase34/` (shared machine, load 6 to over 20, every comparison interleaved). After
Yang et al., *FreeToken* (arXiv 2608.16157, 2026), whose mechanisms are for Mixture-of-Experts models on a GPU with
experts in host memory; what was taken is the reasoning behind them.

## Decisions

1. **The prefix store keeps the caches themselves** (`prefix::CacheStore`). A finished slot's cache moves into the
   store, and a request that continues that conversation takes it out. A request takes a cache when its prompt holds
   every kept token, or when the kept conversation ends inside one turn that the prompt has rewritten and gone past
   (the template's turn token tells which); otherwise it copies the shared positions, also from a request still being
   served. The newest conversation is always kept. An error returns the cache to the store.
   - Hand-off per request: 240 ms to 0 at 2,048 tokens on the 1.7B model, 457 ms to 0 on the 4B model.
2. **The first token is sampled from the pass that reads the end of the prompt** (`fuse_first_token`). One pass over
   the weights fewer per request: about 30 ms on the 1.7B model and 90 ms on the 4B model.
3. **Matmul rows and attention tasks are claimed in blocks** (`pool::Claims`): each worker's own share first, then
   from the back of the others', each owner keeping its last block.
   - Four threads: +19 % writing and +14 % reading on the 1.7B model (12 of 12 paired rounds each), +33 % and +10 % on
     the 4B model. Three threads: inside noise on the 1.7B model, +8 % and +15 % on the 4B model.
   - Taking from the front of a neighbour's share lost 5 to 10 % and was replaced.
4. **A started server reads its weights ahead**, in the order a pass reads them, while it already answers.
   - First request on an uncached model file: 9.2 s against 14.6 s.
5. **Between turns the model that answered reads the conversation too, and first** (`[shadow] answering_model`).
   - Later turns of a quick conversation: first token after 0.71 s instead of 4.25 s. Later turns answered by the
     reasoning model: 2.4 s instead of 18.6 s.
6. **The prefix store's budget can be changed and the store emptied while serving** (`GET`/`POST /v1/cache`), applied
   between two steps of the engine.
7. **Not adopted:** a paged cache with shared prefix pages, checkpoints at special tokens, a bandwidth-derived split
   of work, automatic thread counts, and a replayed reply that keeps its reasoning block (reasons in the research
   note, section 6).

## Switches

`--no-fuse-first-token`, `--no-read-ahead` and `SMILEY_CORE_CLAIM=0` on the engine server; `[shadow] answering_model`
and the request option `"shadow": "other"` on the gateway. Each restores the behaviour before this phase.

## Validation

The engine's 12 serving tests against the real model pass, six of them new (take, copy, copy from a live slot, the
fused first token at five prompt lengths, the budget); unit tests for the store, the cache and the claims. The
workspace suite, the live sweeps and the review are recorded in decision 027, which shipped with this one.
