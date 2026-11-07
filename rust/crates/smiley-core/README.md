# smiley-core (vendored)

Smiley Core is the owner's own CPU inference engine from the Smiley project (`~/Downloads/Smiley`, crate
`smiley/src-tauri/crates/smiley-core`; revision 06c2b88 of 2026-11-04; Apache-2.0). It is a Rust library with one dependency (`memmap2`)
that runs GGUF models of the `llama`, `qwen2` and `qwen3` architectures with F32, F16, Q5_0, Q8_0, Q4_K and Q6_K tensors,
and reports the bytes it moved per token beside the tokens per second.

It is vendored here, rather than referenced from `third_party/` (which is not in version control), so that the workspace and
CI build without the checkout. `nafs-smiley-server` wraps it in the OpenAI-compatible HTTP API the gateway already speaks, so
it serves as the third backend beside llama.cpp and vLLM (`[backends.smiley]`, Phase 15).

Upstream 06c2b88 adds, and this copy carries unchanged: grouped online-softmax attention (`attn.rs`), a Hadamard-rotated int8
KV cache, position-keyed sampling (`sample.rs`: speculation is lossless at any temperature), vectorised math (`vmath.rs`) and
LittleBit (`lbq.rs`, the `.slb` sidecar, the `lb_convert` converter and the `lbbench` benchmark; arXiv 2506.13771).

Modifications for Nafs-Aql, each marked `// nafs-aql:` in the source:

- `SessionCfg::assistant_prefix`: text the assistant has already written, appended after the template's assistant header
  when a turn is planned. The server uses it to continue a turn after the reasoning budget closes the think span, the way
  llama-server's `thinking_budget_tokens` does; the cache prefix is reused, so only the forced tail is prefilled.
- `Emitted::logprob` / `logprob_of`: the log-probability of each sampled token under the raw (temperature 1) softmax, which
  the gate's fixed rule reads as `logprobs`.
- `SessionCfg::top_k` / `min_p` and `sample::sample_filtered`: llama.cpp's cuts in its order (top-k, nucleus, min-p) on the
  position-keyed sampler. Both off takes upstream's histogram fast path; either on sorts the candidates.
- `KvCache::snapshot` / `restore` (rotation-aware) and `prefix.rs`: a store of finished conversations' caches so a prompt that
  shares a long beginning with an earlier one is not prefilled again.
- `Session::reseed`, and the shared prompt builder the server uses.
- `Caches::{Shared, PerRow}`, `Model::forward_rows` and a per-row `attend`: several users advance in one weight pass, each
  against its own cache (`serve.rs`, Phase 18). Attention runs one row at a time per user, so a user's logits equal a solo
  run's bit for bit; `attn.rs` itself is untouched.
- `embed.rs` (`Embedder`) and `Model::final_hidden`: last-token-pooled, L2-normalised embeddings from a decoder-only
  embedding model (Qwen3-Embedding), with the text followed by `<|endoftext|>`; the output head is never read (Phase 20).

The upstream tests are kept; those that need model files under `~/models` skip when the files are absent.
