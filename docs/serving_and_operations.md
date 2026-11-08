# Serving profiles, configuration and operations

The full set-up notes: all three serving profiles, configuration, security and platform details. The [README](../README.md) has the three-command start.

## Quick start (full)

**Requirements:** Rust 1.96+, Python 3.10+, and a CPU build of llama.cpp (tested with commit `797de63`).

```bash
git clone https://github.com/rismanmattotorang/nafsaql.git && cd nafsaql

# 1. llama.cpp, built CPU-only into third_party/ (about 5 minutes)
git clone https://github.com/ggml-org/llama.cpp third_party/llama.cpp
git -C third_party/llama.cpp checkout 797de63
cmake -S third_party/llama.cpp -B third_party/llama.cpp/build -DGGML_METAL=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build third_party/llama.cpp/build -j4 --target llama-server llama-bench

# 2. Models (Qwen3-1.7B and Qwen3-4B-Thinking-2507, GGUF Q4_K_M, about 3.6 GB)
make models

# 3. Build the Rust gateway, start both model servers and the gateway
make up
```

Open **http://127.0.0.1:8080**. Try the *Verified math* example, then press **Think harder** on the answer to watch System 2 deliberate. `make down` stops everything.

```bash
make test        # Rust tests: gate modes, calibration, judge, audits, cancellation, verifiers, sandbox
scripts/live_sweep.sh   # every live suite against the running stack: contract, security, Rooms, end to end, harness, MVP acceptance (about 45 min)
make contract    # live SSE events and stored records against the JSON Schemas
make acceptance  # MVP acceptance tests A1–A9 against the running stack
python3 scripts/e2e_sweep.py   # every endpoint and every UI page, in headless Chrome
python3 -I scripts/harness_sweep.py   # Lean, code credit, validation and the code agent (add --no-models for the fast part)
```

<details>
<summary><b>vLLM backend in Docker</b> (second serving profile)</summary>

```bash
scripts/serve_vllm_docker.sh start && scripts/serve_vllm_docker.sh wait   # vLLM CPU image on :8101
python3 scripts/vllm_check.py                                              # drives every request path through the gateway
```

Switch with *Use this backend* on the System page. The recipe was built and debugged against a vLLM 0.30 x86 CPU image (override with `VLLM_IMAGE`). Facts worth knowing: on an AVX2-only VM oneDNN has no BF16 kernels, so the model runs in FP32; vLLM's three processes cost about 3 GB beyond the weights, so Qwen3-1.7B needs an 11–12 GiB Docker VM. The gateway-side check was **not yet completed** on the reference host because it was too short of memory (see `docs/roadmap/STATUS.md`, P7-T10).

</details>

<details>
<summary><b>Smiley Core backend</b> (the default serving profile since Phase 15)</summary>

```bash
make up                 # the default: builds nafs-smiley-server, serves Qwen3-1.7B on :8201 and Qwen3-4B-Thinking on :8202, starts the gateway
make up-llama           # starts llama-server on :8001/:8002 and switches the gateway to it (four slots, grammar, n-choice rescue)
make down-llama         # switches back to Smiley Core and stops llama-server
```

The engine is the owner's Smiley Core (vendored as `rust/crates/smiley-core` with two marked additions: a continuation prefix for the reasoning budget and per-token log-probabilities for the gate). It decodes one stream per model, so the gateway runs it without the grammar, the `n`-choice rescue and the raw-continuation exit (the standard path, as for vLLM). **It is the default backend:** the engine alone decodes 14.3 vs 12.6 tokens/s (System 1) and 7.4 vs 6.6 (System 2), and through the whole gateway the same twelve checked prompts took **131 s against 303 s** (9/9 checked on both, faster in 10 of 12 pairs, p = 0.02); the code agent verifies on both. The price is one stream per model: a second concurrent request waits, and the rescue and the cascade have one candidate. `make up-llama` switches to llama.cpp (four slots, grammar) for a shared deployment. Details: [docs/research/smiley_backend.md](docs/research/smiley_backend.md).

<details>
<summary><b>Configuration, security and deployment</b></summary>

- Config lives in `config/nafs-aql.toml`; `rust/target/release/nafs-gateway --check` validates it and the bind policy.
- The gateway refuses a non-loopback bind without tokens (`NAFS_AUTH_TOKENS` or `NAFS_AUTH_TOKENS_FILE`, 16+ characters each) unless `NAFS_ALLOW_INSECURE=1`. Browser requests must be same-origin.
- Code is verified in a network-denied sandbox with CPU, memory and wall-clock limits; `python3 -I scripts/security_tests.py` runs 26 live probes (sandbox escape, injection, auth, limits).
- **Platform notes.** macOS uses a `sandbox-exec` profile. Linux builds the same guarantees from namespaces and seccomp: new user, network, mount and PID namespaces; user data (`/home`, `/root`, `/tmp`, `/mnt`, `/media`, `/srv`, `/run/secrets` and others) replaced by empty mounts; every other mount read-only; no capabilities; a seccomp filter that denies `fork`, process spawning and mount-table changes; plus CPU, memory and wall-clock limits. The same escape tests run on both. What the Linux sandbox cannot hide is the gateway's own install, config and data directories (they stay readable, never writable), so keep those mode 0700 under a dedicated user (A-054). Linux needs unprivileged user namespaces (Ubuntu 24.04: `sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0`), `unshare` and `setpriv` from util-linux, and a python3 with ctypes; in Docker it also needs `seccomp=unconfined` and `apparmor=unconfined`, which the compose file sets. `nafs-gateway --check` runs the whole stack and reports whether it works.
- Recipes for docker compose, systemd, launchd, Caddy and nginx TLS, Prometheus alerts and a Grafana dashboard are in `deploy/`; `scripts/verify_deploy.sh --live` checks them. The operator runbook is [docs/operations/operator_runbook.md](docs/operations/operator_runbook.md).

</details>


## Architecture

A Rust workspace of nine crates:

| Crate | Role |
|---|---|
| `nafs-gateway` | axum HTTP/SSE/WebSocket API (53 documented endpoints, OpenAPI 3.1), auth, metrics, background workers |
| `nafs-core` | orchestrator and automaton, Aql Gate, calibration, drift, backends (llama.cpp, vLLM) |
| `nafs-verify` | exact verifiers (numeric, SymPy, code tests, JSON Schema, Lean) and the sandbox, rubric judge support, mutation harness |
| `nafs-harness` | derived from Open Math Harness and Codex: Lean protocol, scoring and candidate tree, `apply_patch` and its grammar (GBNF), truncation, the append-only transcript and compaction, the code agent loop |
| `nafs-memory` | SQLite ledger, traces, retrieval memory (BM25), preferences |
| `nafs-eval` | datasets, paired statistics, policy map, resumable evaluation runs |
| `nafs-bench` | load-test protocol, latency breakdown |
| `nafs-web` | the embedded web UI (vanilla JS, no build step) |
| `nafs-types` | shared types and the error taxonomy |

The mathematics behind the gate (threshold rule, verifier-bounded precision, Beta-prior stopping, sizes of the release gates) is in the [technical report](docs/report/nafs_aql_report.pdf); the 100 requirements and 45 tasks are in the [specification](docs/spec/nafs_aql_spec.pdf) and [roadmap](docs/roadmap/nafs_aql_roadmap.pdf).

## Where the project stands

Phases 0–13 of the roadmap are implemented on the CPU reference machine; GPU training and scale-out are deliberately deferred. Gates are computed from stored evidence, and today they say:

| Gate | Status | Why |
|:--:|:--:|---|
| G1 MVP and latency | ❌ | acceptance 8/9: the deliberate path took 354 s against the 240 s target while the host was shared |
| G4 evaluation suite | ❌ | NFR-007: median 4-session p95 is 2.15× the single-session value (target ≤ 2×) |
| G5 security and deployment | ✅ | 26/26 security probes, 0 open high findings, deployment checks pass |
| G0, G2, G3, G6 | ⏳ | need release-scale runs (thousands of items), the first GPU training cycle, and the scholar review |

**Next:** evidence at scale on a faster host, the first Tazkiya (distillation) cycle, adaptive early exit across more domains, a learned router built from the policy map and human preferences, and the scholar review. The details, with estimates, are in the [roadmap](docs/roadmap/nafs_aql_roadmap.pdf).

