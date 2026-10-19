# Nafs-Aql backlog (generated from data/tasks.csv and data/requirements.csv)

Estimates are person-weeks (PW) and are estimates, not commitments.

## Phase 0

| Task | Epic | Title | PW | Depends | Requirements | MoSCoW |
|---|---|---|---|---|---|---|
| P0-T01 | E0.1 Environment | Build llama.cpp and vLLM-CPU container; publish serving recipes for Profile A | 1.0 |  | MR-003, OR-001 | Must |
| P0-T02 | E0.1 Environment | Pin model revisions, GGUF hashes, chat-template hashes; write model manifest | 0.5 | P0-T01 | MR-001, NFR-012 | Must |
| P0-T03 | E0.2 Measurement | Implement minimal Eval Harness (dataset loaders, exact verifiers for GSM8K/HumanEval/schema, metrics) | 2.0 | P0-T01 | FR-050, ER-010 | Must |
| P0-T04 | E0.2 Measurement | Run fast-only and slow-only baselines on >= 4600 tasks; fit q, (r,phi), e1(s), e2(s), cost ratio, discordance | 1.5 | P0-T03 | ER-001 | Must |
| P0-T05 | E0.2 Measurement | Measure accuracy vs thinking budget per domain (H4) and acceptance-rate distribution | 1.0 | P0-T04 | FR-021 | Must |
| P0-T06 | E0.3 Hybrid gate | llama-bench Qwen3.5-2B/4B vs Qwen3-1.7B/4B on target CPU (H7); decide Profile C | 0.5 | P0-T01 | ER-011, MR-005 | Must |
| P0-T07 | E0.4 Workspace | Create Cargo workspace, CI (fmt, clippy, test on Linux and macOS), config schema | 1.0 |  | OR-001, NFR-023, OR-006 | Must |

## Phase 1

| Task | Epic | Title | PW | Depends | Requirements | MoSCoW |
|---|---|---|---|---|---|---|
| P1-T01 | E1.1 Gateway | axum gateway: /v1/chat, /v1/chat/stream (SSE), auth, rate limit, metrics, OpenAPI at /openapi.json | 2.0 | P0-T07 | IF-001, IF-002, IF-003, IF-005, NFR-021, NFR-022 | Must |
| P1-T02 | E1.2 Backends | Backend trait + llama-server and vLLM implementations: logprobs, parallel samples, thinking toggle, retry, semaphore | 2.0 | P0-T07 | IF-006, IF-007, IF-008, MR-002, NFR-011 | Must |
| P1-T03 | E1.3 Orchestrator | Trace-state automaton with fixed-rule gate, budget forcing, caps, decision record, assurance levels | 2.5 | P1-T02 | FR-012, FR-013, FR-023, FR-024, FR-026, FR-031, FR-002, NFR-001 | Must |
| P1-T04 | E1.4 Verifiers | Verifier Hub v1: code sandbox (subprocess limits), JSON Schema verifier | 2.0 | P0-T07 | FR-030, FR-035, SR-002 | Must |
| P1-T05 | E1.5 Memory | SQLite conversation store, JSONL traces, context cap and prompt-cache prefix preservation | 1.5 | P0-T07 | FR-003, FR-060, NFR-006, IF-010 | Must |
| P1-T06 | E1.6 Web UI | Chat room UI (askama + htmx + WebSocket), reasoning mode and assurance badges, controller event stream | 2.0 | P1-T01 | FR-001, FR-006, IF-004 | Must |
| P1-T07 | E1.7 MVP acceptance | Profile selection, MVP acceptance tests, latency targets measured on reference machine | 1.0 | P1-T03, P1-T04, P1-T05, P1-T06 | FR-005, NFR-002, NFR-003, NFR-004, NFR-005, ER-006 | Must |

## Phase 2

| Task | Epic | Title | PW | Depends | Requirements | MoSCoW |
|---|---|---|---|---|---|---|
| P2-T01 | E2.1 Calibrated gate | Feature extraction, logistic/isotonic estimator, cost-ratio rule, budget mode, recalibration job | 2.5 | P0-T04, P1-T03 | FR-010, FR-011, FR-014, FR-015, NFR-020 | Must |
| P2-T02 | E2.2 Verified loop | Sequential verified sampling, evidence-fed retries, caps from Beta fit, R<=1 on non-verifiable | 2.0 | P1-T03, P0-T05 | FR-020, FR-021, FR-022, FR-025 | Must |
| P2-T03 | E2.3 Verifier quality | Math verifier (numeric + SymPy sidecar), mutation-testing harness, (r,phi) publication, judge hard gate | 2.0 | P1-T04 | FR-030, FR-032, FR-034, ER-003, SR-003 | Must |
| P2-T04 | E2.4 Audit and ledger | 5 percent audit stream, capability table with unmeasured labels, adapter-hash keying | 1.5 | P2-T01 | FR-040, FR-041 | Must |
| P2-T05 | E2.5 Resilience | Backend reconnect, degraded mode, prompt-injection delimiting | 1.0 | P1-T02 | NFR-010, MR-004, SR-001 | Must |
| P2-T06 | E2.6 Gate G2 evaluation | Coupling test (>= 200 trials), calibration test (>= 2900), 3-point non-inferiority (>= 700 pairs) | 1.5 | P2-T01, P2-T02, P2-T03 | ER-002, ER-004, ER-005, FR-004 | Must |

## Phase 3

| Task | Epic | Title | PW | Depends | Requirements | MoSCoW |
|---|---|---|---|---|---|---|
| P3-T01 | E3.1 Data pipeline | Dataset builder from verified traces; schema; dedupe; domain cap; abstain and replay mixing; privacy redaction | 1.5 | P2-T04 | FR-070, TR-001, TR-002, TR-008, SR-004, FR-033 | Must |
| P3-T02 | E3.2 Synthetic data | System 2 synthetic generation on benchmark-like prompts with held-out verifier | 1.0 | P3-T01 | TR-003 | Should |
| P3-T03 | E3.3 Unsloth training | Unsloth LoRA scripts, GPU runbook (T4), checkpoint and adapter hashing, GGUF export | 1.5 | P3-T01 | TR-004, TR-005, TR-006 | Must |
| P3-T04 | E3.4 Promotion | Promotion gate with confidence bounds, regression suite, human sign-off, one-command rollback | 1.5 | P3-T03 | TR-007, ER-008, OR-005 | Must |

## Phase 4

| Task | Epic | Title | PW | Depends | Requirements | MoSCoW |
|---|---|---|---|---|---|---|
| P4-T01 | E4.1 Statistics | Paired tests, CIs, non-inferiority, Bonferroni in harness; report export | 1.5 | P2-T06 | FR-051 | Must |
| P4-T02 | E4.2 Performance console | goose load generator, bench runs API, latency quantiles, concurrency protocol | 2.0 | P1-T01 | FR-052, IF-009, ER-007, NFR-007 | Must |
| P4-T03 | E4.3 Dashboard | Evaluation dashboard: runs, calibration plots, routing accuracy, gate status | 2.0 | P4-T01 | FR-053 | Should |
| P4-T04 | E4.4 Drift | CUSUM drift monitor with negative control on ledger | 1.0 | P2-T04 | FR-042 | Should |

## Phase 5

| Task | Epic | Title | PW | Depends | Requirements | MoSCoW |
|---|---|---|---|---|---|---|
| P5-T01 | E5.1 Hardening | Security review, TLS and secrets, sandbox escape tests, packaging with manifest | 2.0 | P4-T02 | SR-006, OR-002, SR-002 | Must |
| P5-T02 | E5.2 Deployment | docker-compose (vLLM-CPU) and service files (llama-server); monitoring and alerts | 1.5 | P5-T01 | OR-003, OR-004 | Must |
| P5-T03 | E5.3 GPU scale-out | vLLM CUDA profile with both models resident, higher N and budget, speculative decoding | 1.5 | P5-T02 | MR-003 | Could |
| P5-T04 | E5.4 Upgrade decision | Profile C (Qwen3.5) evaluation on full suite; optional retrieval memory | 1.5 | P0-T06, P4-T01 | MR-005, FR-061 | Could |

## Phase 6

| Task | Epic | Title | PW | Depends | Requirements | MoSCoW |
|---|---|---|---|---|---|---|
| P6-T01 | E6.1 Release gate | 1-point non-inferiority on >= 6200 pairs; full regression; scholar review of analogy copy | 2.0 | P5-T01 | ER-002, SR-005 | Must |
| P6-T02 | E6.2 Documentation | Operator and user documentation; API reference; runbooks | 1.0 | P6-T01 | OR-003 | Must |
| P6-T03 | E6.3 Publication | Research write-up with measured results; artefact release | 1.5 | P6-T01 |  | Should |

## Phase 7

| Task | Epic | Title | PW | Depends | Requirements | MoSCoW |
|---|---|---|---|---|---|---|
| P7-T01 | E7.1 Observability | Per-stage latency with server prefill/decode/cache split; /v1/perf/stages; inspector waterfall and Performance breakdown | 1.0 | P4-T02 | NFR-030 | Must |
| P7-T02 | E7.2 Engine | Interleaved A/B harness and study of flash attention, BLAS prefill, KV-cache type and threads; verifier warm-up | 1.0 | P7-T01 | NFR-031 | Must |
| P7-T03 | E7.3 Cascade | Early escalation in the streaming draft (fixed rule) with decision-agreement study | 1.0 | P7-T01 | FR-080 | Should |
| P7-T04 | E7.2 Engine | EAGLE-3 heads for Qwen3-1.7B and Qwen3-4B (convert, serve, measure acceptance and speed-up) | 1.0 | P7-T02 | FR-081 | Could |
| P7-T05 | E7.2 Engine | Weight-format study: Q4_0 / IQ4_NL with AVX2 repack kernels against Q4_K_M (speed and accuracy on regression_500) | 1.0 | P7-T02 | NFR-031 | Could |
| P7-T06 | E7.4 Lab | Efficiency frontier in Lab insights: early escalation, saved tokens, routing agreement per domain | 1.0 | P7-T03 | ER-020 | Should |
| P7-T07 | E7.5 Release | End-to-end sweep, UI screenshots, technical report and slides refresh, forward roadmap | 1.0 | P7-T06 |  | Must |
| P7-T08 | E7.3 Cascade | Adaptive reasoning exit: probe-and-verify on the raw continuation path, with a token-saving study | 1.0 | P7-T03 | FR-082 | Should |
| P7-T09 | E7.3 Cascade | Judge on System 1: measure the small judge's (r, phi) with the rates harness and adopt it if the rates hold | 0.5 | P7-T01 | FR-034 | Could |
| P7-T10 | E7.6 Backends | vLLM CPU backend in Docker (carstenz/vllm-smoke image, BF16, one model in two modes) with an end-to-end check against the gateway | 1.0 | P5-T02 | MR-003 | Should |

## Requirements by phase

- **Phase 0**: FR-050, NFR-012, NFR-023, MR-001, MR-003, MR-005, OR-001, ER-001, ER-010, ER-011
- **Phase 1**: FR-001, FR-002, FR-003, FR-005, FR-006, FR-012, FR-013, FR-023, FR-024, FR-026, FR-030, FR-031, FR-035, FR-060, NFR-001, NFR-002, NFR-003, NFR-004, NFR-005, NFR-006, NFR-011, NFR-021, NFR-022, IF-001, IF-002, IF-003, IF-004, IF-005, IF-006, IF-007, IF-008, IF-010, MR-002, SR-002, OR-006, ER-006
- **Phase 2**: FR-004, FR-010, FR-011, FR-014, FR-015, FR-020, FR-021, FR-022, FR-025, FR-032, FR-034, FR-040, FR-041, NFR-010, NFR-020, MR-004, SR-001, SR-003, ER-002, ER-003, ER-004, ER-005
- **Phase 3**: FR-033, FR-070, TR-001, TR-002, TR-003, TR-004, TR-005, TR-006, TR-007, TR-008, SR-004, OR-005, ER-008
- **Phase 4**: FR-042, FR-051, FR-052, FR-053, NFR-007, IF-009, ER-007
- **Phase 5**: FR-061, SR-006, OR-002, OR-003, OR-004
- **Phase 6**: SR-005
- **Phase 7**: NFR-030, NFR-031, FR-080, FR-081, ER-020, FR-082
