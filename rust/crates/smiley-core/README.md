# smiley-core (vendored)

Smiley Core is the owner's own CPU inference engine from the Smiley project (`~/Downloads/Smiley/smiley/engine`, mirrored at
`third_party/smiley-core`; revision 7cb6028 of 2026-11-01; Apache-2.0). It is a Rust library with one dependency (`memmap2`)
that runs GGUF models of the `llama`, `qwen2` and `qwen3` architectures with F32, F16, Q5_0, Q8_0, Q4_K and Q6_K tensors,
and reports the bytes it moved per token beside the tokens per second.

It is vendored here, rather than referenced from `third_party/` (which is not in version control), so that the workspace and
CI build without the checkout. `nafs-smiley-server` wraps it in the OpenAI-compatible HTTP API the gateway already speaks, so
it serves as the third backend beside llama.cpp and vLLM (`[backends.smiley]`, Phase 15).

Modifications for Nafs-Aql, each marked `// nafs-aql:` in the source:

- `SessionCfg::assistant_prefix`: text the assistant has already written, appended after the template's assistant header
  when a turn is planned. The server uses it to continue a turn after the reasoning budget closes the think span, the way
  llama-server's `thinking_budget_tokens` does; the cache prefix is reused, so only the forced tail is prefilled.
- `Emitted::logprob`: the log-probability of each sampled token under the raw (temperature 1) softmax, which the gate's
  fixed rule reads as `logprobs`.

The upstream tests are kept; those that need model files under `~/models` skip when the files are absent.
