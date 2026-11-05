# Decision 013: Smiley Core 06c2b88 and LittleBit (Phase 19)

Date 2026-11-05. Status: **adopted; LittleBit available, off by default.** Research note:
[`docs/research/smiley_core_update.md`](../research/smiley_core_update.md). Report: section "Phase 19".

## Context

The owner placed a new Smiley Core in `third_party/smiley-core` (revision 06c2b88 of 2026-11-04) and asked for it to replace
the vendored copy (`rust/crates/smiley-core`, from 7cb6028), for the features that depend on it to be adapted, and for the new
LittleBit capability (1-bit-class weights, arXiv 2506.13771) to be integrated and tested. The vendored copy carries Nafs-Aql's
own additions (a prefix store, a batching engine, a continuation prefix, log-probabilities, `top_k`/`min_p`), so this was a
three-way merge, not a copy.

## Decision

| What | Where | Default |
|---|---|---|
| Engine updated to 06c2b88: grouped online-softmax attention, Hadamard-rotated int8 KV, position-keyed sampling (lossless speculation at any temperature), vectorised math | `rust/crates/smiley-core` | always |
| Nafs-Aql additions re-applied on it: KV snapshot/restore (rotation-aware), `Caches::{Shared,PerRow}` and a per-row `attend`, `top_k`/`min_p` on the position-keyed sampler (`sample_filtered`), log-probabilities, `assistant_prefix`, prefix store, `reseed`, `serve.rs` | same | always |
| LittleBit sidecar support (`<model>.gguf.slb`, `lb_convert`, `lbbench`) | same | present |
| `nafs-smiley-server --littlebit`; `LITTLEBIT=1 scripts/serve_smiley.sh` | `nafs-smiley-server`, `scripts/` | **off** |
| The weights in use ("dense" / "littlebit") in each response's `timings.smiley`, in `/v1/models`, in the gateway's backend health and on the Status page | server, `nafs-core`, UI | always shown |
| `kbench` kept as an example: one thread, synthetic rows, every format, ns per row | `smiley-core/examples` | tool |

## Evidence

**LittleBit does not work yet without training.** Qwen3-1.7B (Q4_K_M) converted at 1.0 bit per weight with the engine's
Dual-SVID post-training converter: 196 projections, 172.7 MB of factors against 845.7 MB of source weights (4.9x), 0.980 bits
per weight, mean relative reconstruction error 0.752 (Q4_K itself is about 0.1), 13 minutes on four cores. On twelve
exact-answer prompts, greedy, interleaved on the same engine: **dense 9 of 12, LittleBit 0 of 12** (`benchmarks/littlebit_ab.json`;
the output is incoherent). Upstream's own measurement on SmolLM2-360M says the same, and says the paper's quality comes from
distillation-based quantisation-aware training, which needs a GPU. So the sidecar is opt-in, a missing or corrupt one refuses
to start, and a stale sidecar beside a model can never switch the weights silently.

**Two defects found and fixed while integrating.**

1. *LittleBit crashed the batching engine* (segmentation fault on the first request): `Session` sized the sidecar's scratch
   buffers and `Engine::new` did not. Fixed in `serve.rs`.
2. *A fourfold decode slowdown with no change to any kernel.* After the merge, Qwen3-1.7B decoded at 4.5-5.0 tokens per second
   against 19-20 for the previous commit and 16-18 for upstream's own build, interleaved on the same machine. The kernels were
   byte-identical after formatting and the profile put the time inside the Q4_K dot kernel. Bisecting by swapping files showed
   that adding the batching engine to the crate was enough; disassembly showed `core::array::try_from_fn` as a real call inside
   the kernel, once per super-block, in the slow build only. `std::array::from_fn` inside two `inline(always)` helpers
   (`q4k_scale_vecs`, `q6k_scales`) is inlined or not depending on how much unrelated code the crate holds. Fixed by spelling
   the eight-element arrays out. `kbench` (Q4_K, one activation, one thread): **679 ns per row before, 96 after**; the
   decode is back to 17-20 tokens per second. The previous scheme had the same exposure and had been lucky.

Pinned by tests: 110 library tests (including the new top-k / min-p test and the rotation-aware snapshot test), `tests/batch.rs`
(batched logits bit-identical to solo), `tests/engine.rs`, `tests/lbq.rs`, `tests/dspark.rs`; workspace fmt and clippy
`-D warnings` clean.

## Validation on the final code

370 Rust tests in release mode (30 suites, 0 failures), `cargo fmt --check` and `cargo clippy --workspace --all-targets -- -D warnings`
clean; live: contract tests, security 26/26 (0 open high-severity findings), Rooms 59/59, end-to-end 65/65, harness sweep 53/53,
MVP acceptance 9/9, on the fixed build (`scripts/live_sweep.sh`). The web interface was driven by hand after the merge: a chat
streamed, was cross-checked and routed, and the Status page named the engine in use.

## What is not claimed

- That LittleBit helps: it does not, here, until factors trained with QAT exist. The integration is ready for them (the sidecar
  format, the kernels, the loader, the label), not a speed or memory result.
- That the new engine is faster than the old one: it is level: median of five interleaved rounds 19.2 against 18.7 tokens per second, 9 of 12 against 9 of 12 on exact-answer prompts with the same three misses (`benchmarks/engine_update_ab.json`, shared host). The attention and sampling changes matter more at long context
  and with speculation, neither of which Nafs-Aql's default path uses.
- That the compiler-dependence is gone for good: the guard is `kbench` and this note. A kernel helper built with
  `std::array::from_fn` in a hot loop may regress the same way.
