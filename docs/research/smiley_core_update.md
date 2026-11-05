# Smiley Core 06c2b88 and LittleBit (Phase 19)

Date 2026-11-05. Decision: [013](../decisions/013-smiley-core-littlebit.md). Report: section "Phase 19".

## 1. The task

The owner replaced `third_party/smiley-core` with a new revision (06c2b88) and asked for the vendored engine to be updated,
the features that depend on it adapted, LittleBit integrated, everything re-tested, and the paper and README updated.

## 2. What changed upstream (7cb6028 to 06c2b88)

- `attn.rs`: grouped online-softmax attention (a key/value head's rows are read once for all the query heads that share it).
- `kv.rs`: an int8 cache rotated by a Hadamard transform (`KvCache::rotated`), with `decode_block`.
- `sample.rs`: sampling whose randomness is a hash of (seed, turn, position, token), so the token at a position is a pure
  function of the logits there. Speculation is then lossless at any temperature.
- `vmath.rs`: vectorised exp, max and sums for the softmax.
- `lbq.rs`, `examples/lb_convert.rs`, `examples/lbbench.rs`, `tests/lbq.rs`: LittleBit (arXiv 2506.13771). A weight matrix is
  approximated as `diag(h) U_sign diag(l) V_sign^T diag(g)` summed over stages; the factors live in a sidecar
  `<model>.gguf.slb` (magic `SLB1`, CRC32C per section, atomic publish, a stamp of the source file's size and a hash of its first
  64 KiB). The converter is post-training (Dual-SVID); the paper's quality comes from distillation-based QAT.

## 3. The merge

Our copy had diverged by: the batching engine (`serve.rs`) and its per-row forward pass, the prefix store (`prefix.rs`) and a
cache snapshot, `assistant_prefix`, per-token log-probabilities, `top_k`/`min_p`, `reseed`, and a shared prompt builder.

Procedure: `git merge-file` with base 7cb6028, ours, theirs, after running rustfmt over base and theirs (our copy is
rustfmt-formatted, upstream's is not): 44 conflicts became 14, all real. Then by hand:

- `kv.rs`: snapshot and restore record and check `rotated`; a restore needs `len <= snapshot.len` and `len <= max_ctx`.
- `model.rs`: `Caches::{Shared, PerRow}`, `forward_rows` (a position and a cache per row), a per-row `attend` that writes into
  row `row0` of the shared buffers. Upstream's `attn.rs` is untouched: for per-row caches attention is called once per user with
  one row, which is also what makes a user's logits bit-identical to a solo run's.
- `sample.rs`: `sample_filtered` adds `top_k` (candidates sorted most probable first, ties by id, cut to k) and `min_p`
  (floor at `min_p^(1/t)` times the best, since llama.cpp applies temperature last) around the nucleus, in llama.cpp's order.
  With both off the call is upstream's histogram nucleus.
- `serve.rs`: draws with `position_key(seed, 0, position)` per slot, so a user's text depends on their seed and position only.

## 4. Two defects the integration exposed

**The crash.** `Engine::new` did not size the LittleBit scratch (`State::set_lb_scratch(model.lb_scratch(max_batch))`); the first
sidecar matmul wrote through an empty `Vec`'s pointer: exit 139 on the first request.

**The slowdown.** After the merge, decode was 4.4-5.0 tok/s against 17-20 (previous build) and 16-18 (upstream's `run` example),
interleaved, same machine and flags. How it was found:

1. Upstream's `simd.rs`, `pool.rs`, `quant.rs`, `gguf.rs`, `attn.rs`, `vmath.rs`, `lbq.rs` equal ours after rustfmt, so the cause is
   not a kernel edit.
2. A single thread was 4x slower too (1.8 against 7.45 tok/s): not the pool, not the spin windows (`SMILEY_CORE_SPIN_US`).
3. Not codegen units (`codegen-units=1` same), not the 32-byte branch alignment erratum, not function and block alignment, not
   the stack offset (environment padding), not the control tables made `const`.
4. Bisecting by copying our files over a scratch copy of upstream: our `model.rs`, `kv.rs`, `session.rs`, `sample.rs` and
   `prefix.rs` were all fast; adding the full `serve.rs` made it slow; removing only `impl Engine` made it fast again, though
   nothing in the benchmark calls it.
5. `examples/kbench.rs` (upstream's, synthetic rows, one thread): Q4_K with one activation 679 ns per row in the slow crate and 97
   in the fast; Q6_K 676 against 122; the 32-activation kernel equal.
6. Disassembling `tile_q4_k`: the slow build has `callq core::array::try_from_fn` inside the loop; the fast one has none.
   `q4k_scale_vecs` and `q6k_scales` (both `#[inline(always)]`) build their eight vectors with `std::array::from_fn(|k| ...)`. Whether
   LLVM inlines it into the kernel depends on the size of the rest of the crate.

The fix is eight explicit elements instead of `from_fn` in those two helpers (`simd.rs`, marked `// nafs-aql:`): `kbench` Q4_K 96 ns
and Q6_K 112 ns per row, decode 17-20 tok/s. Upstream should take the same change; it has the same exposure.

## 5. LittleBit measured

`lb_convert models/Qwen3-1.7B-Q4_K_M.gguf --bpw 1.0 --stages 2`: 196 projections in 13 minutes on four cores, 172.7 MB of factors
for 845.7 MB of weights (4.9x), 0.980 bits per weight over the converted weights, worst tensor relative error 0.839
(`blk.0.attn_output`), **mean 0.752** (Q4_K itself is about 0.1).

`scripts/littlebit_ab.py` (twelve exact-answer prompts, greedy, `enable_thinking=false`, 48 tokens, dense and LittleBit servers
interleaved per prompt): **dense 9 of 12, LittleBit 0 of 12**. The text is incoherent ("icialicial//{{ylumрит..." for 17 * 23).
Decode with the sidecar: 20.1 tok/s median against 19.2 dense (+5 %): at this size the matmuls are not the whole token and the
factored path adds work of its own.

Reading: the integration works (loader, kernels, scratch, label, refusal on a bad sidecar, tests), the format is not yet useful
without trained factors, and a 1.7B model is far from the paper's regime. It stays off by default; turning it on needs a QAT run
(a GPU job, see the training runbook) and a re-run of the script above on the result.

## 6. Old against new

Qwen3-1.7B Q4_K_M, three threads, one slot, greedy, interleaved, shared host (load 5-8), `benchmarks/engine_update_ab.json`:

| | decode tok/s (median of 5 rounds) | exact-answer prompts |
|---|---|---|
| previous engine (7cb6028 + Phase 18) | 18.7 | 9 / 12 |
| new engine (06c2b88), after the fix | 19.2 | 9 / 12, the same three misses |
| new engine, before the fix | 4.4-5.0 | not measured |
| new engine with the 1.0-bit sidecar | 20.1 | 0 / 12 |

The three misses are the 1.7B model's: 3:40 pm plus 2 h 35 min (answered 5:45 pm), the formula for water, 15 % of 200. The first
was noticed in the web interface right after the merge and looked like a regression; it is not.

The new engine's gains are in attention at long context, the int8 cache and speculation at a temperature; Nafs-Aql's default
path (short prompts, f16 cache, sampling without speculation) uses none of them, so parity is the expected result. A long-context
or int8-KV measurement is the next thing to try, not something claimed here.

## 7. Re-validation

370 Rust tests in release mode (0 failures); fmt and clippy `-D warnings` clean; contract, security 26/26, Rooms 59/59, end-to-end
65/65, harness 53/53, MVP acceptance 9/9 (`scripts/live_sweep.sh`, on the fixed build); the web interface checked by hand.

## 8. Not done

- QAT-trained LittleBit factors (needs a GPU) and a re-run of the accuracy test on them.
- Long-context and int8-KV measurements of the new attention and cache.
- Linux builds were not run locally; CI covers them.
