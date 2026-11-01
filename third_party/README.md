# third_party

Reference checkouts used to verify claims in the design documents. They are not part of the Nafs-Aql code and are excluded from version control (see `.gitignore`).

| Directory | Source | Used for |
|-----------|--------|----------|
| `vllm/` | github.com/vllm-project/vllm (tag v0.32.0 plus main of 2026-10-18) | reading CPU-backend docs and source (hybrid GDN path, supported models, env vars) |
| `unsloth/` | github.com/unslothai/unsloth (version 2026.10.6) | confirming that Core training needs a GPU (`unsloth/device_type.py`) |
| `llama.cpp/` | github.com/ggml-org/llama.cpp (commit 797de63, 2026-10-17), built CPU-only | benchmarks and the llama-server integration probe |
| `codex/` | github.com/openai/codex (revision 970111f6) | the agent-harness mechanisms assessed and ported in Phases 10, 11 and 14 |
| `smiley-core/` | the owner's Smiley project, `smiley/engine` (revision 7cb6028) | the Rust CPU engine vendored into `rust/crates/smiley-core` as the third backend (Phase 15); the copy in the workspace is the one that builds |

Rebuild llama.cpp (about 5 minutes on the reference machine):

```bash
cd third_party/llama.cpp
cmake -B build -DGGML_METAL=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 4 --target llama-bench llama-server llama-cli
```

Binaries land in `third_party/llama.cpp/build/bin/`. A Homebrew install of llama.cpp was tried and abandoned on this Intel Mac (it compiles the LLVM, Rust and Node dependency chain from source).
