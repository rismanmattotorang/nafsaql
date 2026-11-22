# third_party

Checkouts that Nafs-Aql still needs at run or build time. They are excluded from version control (see `.gitignore`).

| Directory | Source | Needed for |
|-----------|--------|------------|
| `llama.cpp/` | github.com/ggml-org/llama.cpp (commit 797de63, 2026-10-17), built CPU-only | the llama backend: `scripts/serve_llama.sh`, the launchd and systemd units, the benchmark scripts and the GGUF conversion all run its `build/bin/llama-server` |
| `memorilla/` | github.com/snap-stanford/memorilla (commit d8f1889, 2026-11) | reference for the Knowledge Base's Memorilla port (Phase 20): parity tests and the GPU pre-training recipe use its PyTorch code |
| `vllm/` | github.com/vllm-project/vllm (tag v0.32.0 plus main of 2026-10-18) | the Docker profile: `deploy/docker/docker-compose.yml` builds from it |

The Smiley Core engine is not here: it is vendored into the workspace as `rust/crates/smiley-core` (Phase 15) and builds with `cargo build`.

## Removed on 2026-11-01

Reference checkouts that nothing in the build, the tests, the scripts or the deployment files read. What was taken from them is
in the code and the research notes; to read the source again, clone it at the recorded revision.

| Was | Source and revision | What it was for |
|-----|---------------------|-----------------|
| `codex/` | github.com/openai/codex @ `970111f6` | the agent-harness mechanisms ported in Phases 10, 11 and 14 (now `rust/crates/nafs-harness`) |
| `unsloth/` | github.com/unslothai/unsloth @ `72d0d2dbf` (version 2026.10.6) | confirming that Core training needs a GPU |
| `OpenHarness/` | github.com/HKUDS/OpenHarness @ `9b2efd7` | a harness reference read during Phase 10 |
| `DeepInfer/` | github.com/deepseek-ai/DeepInfer @ `9bd9cf2` | the assessment in `docs/research/deepinfer_assessment.md` |
| `smiley-core/` | a copy of the owner's Smiley project, `smiley/src-tauri/crates/smiley-core` @ `7cb6028` | redundant: the workspace crate is the one that builds; the original is in `~/Downloads/Smiley` |

## Removed on 2026-11-22

Reference checkouts read during Phases 20 to 34. Each was an unmodified clone (checked before removal: no changed
tracked file), so the source is one `git clone` and `git checkout <revision>` away. About 2.8 GB.

| Was | Source and revision | What it was for |
|-----|---------------------|-----------------|
| `semantic-router/` | github.com/vllm-project/semantic-router @ `029bf54ed` | the comparison in `docs/research/router_memory_gaps.md` and the Go side of `benchmarks/headtohead/run.sh`, which needs this checkout back to run again (its results are stored) |
| `deepseek-harness/` | github.com/deepseek-ai/deepseek-harness @ `fcc0ad4621` | harness reference |
| `OpenViking/` | github.com/volcengine/OpenViking @ `13738af5e` | the context filesystem of Phase 29 (`docs/spec/context_fs_spec.md`) |
| `LMOps/` | github.com/microsoft/LMOps @ `6f53ef4` | prompt and distillation references |
| `EvolvingInfra/` | github.com/mit-han-lab/EvolvingInfra @ `20e8dc6` | engine reference |
| `ontosmith/` | github.com/Dongbumlee/ontosmith @ `b1007d6` | the graph and ontology work of Phase 21 |
| `liteparse/` | github.com/run-llama/liteparse @ `0548949` | the PDF layout rules ported in Phase 28 (`docs/research/liteparse_semtools_gaps.md`) |
| `FreeToken/` | github.com/FlashML-org/FreeToken @ `c84946b` | the assessment in `docs/research/freetoken_assessment.md` (Phase 34) |
| `semtools/` | github.com/run-llama/semtools @ `fe7f6d2` | search reference of Phase 28 |
| `smiley-core/` | a copy of the owner's Smiley project at `06c2b88` (identical to that commit's `smiley/src-tauri/crates/smiley-core/src`) | the engine merge of Phase 19; the original is in `~/Downloads/Smiley` |

Rebuild llama.cpp (about 5 minutes on the reference machine):

```bash
cd third_party/llama.cpp
cmake -B build -DGGML_METAL=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 4 --target llama-bench llama-server llama-cli
```

Binaries land in `third_party/llama.cpp/build/bin/`. A Homebrew install of llama.cpp was tried and abandoned on this Intel Mac (it compiles the LLVM, Rust and Node dependency chain from source).
