# Nafs-Aql delivery status

Updated 2026-10-18 (night). Tracks roadmap Phases 0–7 (task IDs from `data/tasks.csv`) on the reference machine: Intel Core i3-8100, 4 cores, AVX2, 32 GB, macOS 15.4.1. Numbers are measurements unless marked otherwise.

All measurements after 14:00 are marked "shared machine" (ASSUMPTIONS A-027). They ran while the owner's other workloads were busy: an Ollama server, mistral.rs, and a second llama-server from another project at about 210% CPU, with load averages of 10–15 on 4 cores. Absolute speeds from that period are lower bounds.

## Summary

- **Phase 0:** done except the full-scale baseline (P0-T04) and budget curve (P0-T05). Each needs about 190 machine-hours on this CPU. The harness runs them, and the Lab page derives gate parameters from the results.
- **Phase 1:** done. All 31 review findings are fixed (`docs/research/code_review_phase1.md`). MVP acceptance: 8/9 on the latest run (21:26, load average 9–12). A3 returned a verified deliberate answer, but in 354 s against the 240 s target (NFR-004); it passed in 161 s under lighter load earlier in the day. Gate G1 therefore shows "fail" (see Known issues).
- **Phase 2:** all six tasks are implemented. The release gate G2 needs 700 pairs and 2,900 calibration samples, which this CPU cannot produce in a day, so only scaled runs are reported.
- **Phase 3:** the distillation pipeline is implemented and tested on the CPU host: dataset builder, synthetic generation, training script, promotion gate and rollback. Training itself needs a GPU host (TR-005), and the ledger holds only 2 verified System 2 traces against the roughly 5,000 the phase needs.
- **Phase 4:** all four tasks are implemented with unit tests on synthetic data:
  - statistics: exact McNemar tests, Bonferroni correction, per-domain comparisons, and Markdown/CSV export;
  - the load-test protocol at 1, 2 and 4 sessions with the NFR-007 verdict;
  - the G0–G4 gates dashboard;
  - CUSUM drift detection with a negative control.
- **Phase 5:** P5-T01, P5-T02 and P5-T04 are done; P5-T03 (GPU scale-out) is deferred, as the user asked.
  - Security review: 26/26 live security tests pass, with 0 open high-severity findings. One high-severity sandbox issue was found and fixed: sandboxed code could signal other processes.
  - TLS and secrets: a secure-by-default bind, tokens from a secret file, and TLS proxy configurations.
  - Packaging: binaries, configuration schema, model manifest and deployment recipes, with SHA-256 sums.
  - Deployment: docker compose (vLLM-CPU), systemd, launchd, Prometheus alerts and a Grafana dashboard. Deployment checks: 11 pass, 0 fail, 1 skipped (a clean Linux host is not available).
  - Alerts with a drill mode; retrieval memory.
  - Profile C decision (MR-005): **keep Profile A**. On 30 tasks, Qwen3.5-2B was more accurate (83% vs 67%, paired +17 points, 95% CI [+7, +30]), but 3.7× slower (88.9 s vs 24.0 s per task; 1.9 vs 4.3 tok/s under load). The adoption rule allows at most 1.3× (`benchmarks/profile_c_decision.json`). Gate G5 now passes.
- **Phase 6:** P6-T02 and P6-T03 are done; P6-T01 is implemented but its release criteria are not yet met.
  - Release gate G6 is on the dashboard: 1-point non-inferiority, a green regression run and scholar sign-off. A scholar-review packet is ready for a reviewer.
  - Documentation: operator runbook, user guide, and an API reference generated from OpenAPI 1.5 (50 paths).
  - Publication: a results write-up compiled to PDF, with every number generated from stored runs (`docs/publication/`).
- **Phase 7 (added, CPU inference performance):** P7-T01–T03 and P7-T07 done; P7-T04/T05 prepared; P7-T06 partly done. See "Phase 7" below.
- **UX:** the app is a research lab for System 1 / System 2 interaction (see "Lab and UX" below), including "Both" trials that collect human preferences between the two paths.

## Phase 0 and Phase 1

| Task | Status | Evidence |
|---|---|---|
| P0-T01 serving recipes | Done | `scripts/serve_llama.sh` (reads the model registry; `SPEC=1` opt-in speculative decoding), `scripts/serve_vllm.sh`, `deploy/lima/` |
| P0-T02 pins and hashes | Done | `config/models.manifest.json` |
| P0-T03 eval harness | Done | `nafs-eval`, `nafs-verify`; datasets GSM8K, HumanEval, **HumanEval+**, MBPP, **MBPP+**, schema, open, smoke, **regression_500** |
| P0-T04 baseline | Harness done, scaled only | `fast_only` / `slow_only`; `POST /v1/gate/derive` |
| P0-T05 budget curve | Harness done, scaled only | `slow_b<N>` systems, b* per domain (Prop. 9) |
| P0-T06 H7 | Done: passes (79–80%) | `benchmarks/reference-machine/llama_bench_h7_qwen35_vs_qwen3.md` |
| P0-T07 workspace and CI | Done | fmt and clippy `-D warnings` clean |
| P1-T01..T06 | Done | gateway, backends, orchestrator, verifiers, memory, UI |
| P1-T07 MVP acceptance | 8/9 on the latest run (A3 slow-path latency under load); 9/9 earlier | `benchmarks/mvp_acceptance.json`; A9 also checks that System 2 comes back |

## Phase 2

| Task | Status | Evidence |
|---|---|---|
| P2-T01 calibrated gate | Done | 5-fold held-out ECE with isotonic correction; promotion needs n ≥ 2,900, ≥ 30 per class and ECE ≤ 0.05; budget mode with DKW-sized λ*; recalibration from the ledger without double-counting evaluation items |
| P2-T02 verified loop | Done | delimited evidence-fed retries; R ≤ 1 on non-verifiable tasks; caps from a Beta-geometric fit (Prop. 3) |
| P2-T03 verifier quality | Done | SymPy math; mutation harness; (r, φ) on every verdict; rubric judge with JSON-schema-constrained output |
| P2-T04 audit and ledger | Done | idle-only audit worker; capability table per backend with the Prop. 5 estimator and human ratings |
| P2-T05 resilience | Done | degraded mode with a recovery check; SR-001 delimiting with tests |
| P2-T06 G2 evaluation | Harness done, scaled only | `nafs_blind`, `slow_conditioned`, λ variants, Pareto frontier |

### Verifier quality (shared machine)

| Verifier | Correct answers | r (true accept) | Mutants | φ (false accept) | φ 95% upper |
|---|---|---|---|---|---|
| Math (numeric + SymPy) | 2,000 | 1.000 | 2,800 | 0.000 | 0.0013 |
| JSON Schema | 24 | 1.000 | 53 | 0.000 | 0.067 |
| Code, EvalPlus tests (MBPP+, HumanEval+) | 364 | 0.995 | 754 | 0.109 | 0.133 |
| Code, original tests (MBPP, HumanEval) | 210 | 1.000 | 540 | 0.107 | 0.137 |
| Judge (rubric, JSON-constrained) | 23 | 1.000 | 22 | 0.045 | 0.23 |

Notes on the table:

- **Math** meets the ER-003 target (φ upper bound ≤ 0.01).
- **Code φ is an upper bound** inflated by equivalent mutants (A-030). The surviving mutants are boundary flips that also pass EvalPlus's far larger test sets.
- **Three verifier defects were found and fixed** by measuring on EvalPlus:
  - Python 3.10 rejected long lines with multi-byte text (tests are now compiled from bytes);
  - the sandbox memory limit was too low for some MBPP+ tests;
  - its CPU-time limit was too low for some tests.
- **The judge** now uses grammar-constrained JSON and returned a valid verdict on all 45 labelled items. The earlier free-text prompt failed on 14 of 45. Its one false accept is a knowledge error: it accepted "Ibn Rushd wrote *The Incoherence of the Philosophers*" (it was al-Ghazālī).

### Scaled Phase-2 evaluation (smoke_mixed, first 6 tasks × 5 systems, shared machine; indicative only)

| System | Accuracy (95% CI) | Escalated | Mean wall |
|---|---|---|---|
| Nafs-Aql, verifier visible | 6/6 (54–100%) | 33% | 96 s |
| Nafs-Aql, gate blind | 4/6 (22–96%) | 17% | 133 s |
| Fast only | 4/6 (22–96%) | 0% | 13 s |
| Deliberate only | 6/6 (54–100%) | 100% | 264 s |
| System 2 sees the draft | 6/6 (54–100%) | 100% | 319 s |

What the run shows:

- **With the verifier visible,** the gated system matched deliberate-only accuracy at **36% of its cost**, the H2 direction.
- **With the gate blind,** it kept two wrong math drafts. Its cheap samples agreed with the wrong answer (correlated errors), so routing accuracy was 50% and the answers were labelled "consensus". This is the case Theorem 6 warns about, and the case calibration must learn.
- **Coupling:** no wrong draft was adopted in either mode. With 2–3 wrong drafts per mode, this says nothing yet (ER-004 needs 200).

## Phase 3: Tazkiya (distillation)

| Task | Status | Evidence |
|---|---|---|
| P3-T01 data pipeline | Done | `training/tazkiya.py build` (details below) |
| P3-T02 synthetic data | Done | `tazkiya.py synthesize` (details below) |
| P3-T03 Unsloth training | Script and runbook done; GPU run deferred by the user ("skip GPU for now") | `training/train_unsloth.py` (TR-004 settings), `training/RUNBOOK.md` |
| P3-T04 promotion | Done | `tazkiya.py promote` / `rollback` (details below) |

What each piece does:

- **`build`** keeps only verified System 2 traces, plus consensus answers a person has approved (TR-002, FR-033). It redacts personal data (SR-004) and excludes opted-out conversations. It deduplicates, caps any domain at 30%, and adds 5% abstention and 20% replay examples (FR-070, TR-008). It writes a train/validation split and a manifest with hashes. On today's ledger it correctly reports "not ready": 2 examples against about 5,000 needed.
- **`synthesize`** runs System 2 on GSM8K and MBPP *training* splits and keeps an answer only when an exact verifier passes (TR-003).
- **`fetch-replay`** supplies the 20% replay share from UltraChat 200k (MIT licence). 400 redacted examples are in `data/tazkiya/replay.jsonl`, and more can be fetched.
- **`train_unsloth.py`** produces an adapter hash, GGUF in q4_k_m and q8_0, and a manifest (TR-006). It refuses to train without a CUDA GPU, and `--dry-run` validates the data.
- **`promote`** passes a candidate only when the paired gain has a lower 95% bound above 0 with a point estimate of at least 2 points, no domain's lower bound is below −1 point, the coupling test does not fail, and a person signs off (TR-007, ER-008 on `regression_500`). Promotion is a registry change that serving reads, and `rollback` undoes it in one command (OR-005).

Supporting features in the gateway and UI:

- per-conversation research opt-out;
- bounded retention, 90 days by default (OR-006), purged daily;
- a review queue for consensus answers;
- human ratings;
- the Tazkiya page.

There are 9 Python tests covering redaction, rationale length, the domain cap, promotion decisions and the rollback round trip.

## Phase 4: evaluation suite and performance console

| Task | Status | Evidence |
|---|---|---|
| P4-T01 statistics | Done | exact McNemar on discordant pairs; Bonferroni across comparisons and across domains (FR-051); per-domain paired Δ with bootstrap CIs; `GET /v1/eval/runs/{id}/export?format=md\|csv\|json` |
| P4-T02 performance console | Done | `POST /v1/bench/protocol`: the same prompt set at 1, 2 and 4 sessions with one uncounted warm-up per level; p50/p95/p99, throughput, escalation and cost per request; NFR-007 verdict (ER-007, FR-052) |
| P4-T03 dashboard | Done | `GET /v1/gates`: G0–G4 go/no-go from stored runs, acceptance, the H7 measurement, the model registry and protocol runs (FR-053), shown at the top of the Lab |
| P4-T04 drift | Done | two-sided CUSUM on calibration error and fast-path accuracy per domain with a shuffled negative control (FR-042), `GET /v1/drift`, shown on the Muhasaba page |

All four exit criteria have tests:

- **Statistics** have unit tests on synthetic data.
- **The CUSUM** detects every 2σ shift within 50 observations and stays silent on negative controls over 500 observations (A-035).
- **The dashboard** reports pending rather than fail when evidence is missing.

### Load-test protocol (shared machine, load average about 8)

| Sessions | p50 | p95 | p99 | Throughput |
|---|---|---|---|---|
| 1 | 1.1–2.0 s | 2.3–4.7 s | 2.3–4.7 s | 0.44–0.78 req/s |
| 2 | 1.7–2.4 s | 3.6–5.4 s | 3.9–5.9 s | 0.72–0.98 req/s |
| 4 | 3.2–3.3 s | 5.7–10.1 s | 7.0–10.7 s | 0.93–1.13 req/s |

Ranges span two runs. The second ran after System 1 was raised from 3 to 4 slots, which removes queueing at 4 sessions.

**NFR-007 fails:** p95 at 4 sessions is 2.1–2.5× the single-session value (target ≤ 2×). On four shared cores, run-to-run noise is as large as the effect. The roadmap fallback applies: document the single-session profile, and treat concurrency as a GPU-profile feature.

Gate G4 judges NFR-007 on the **median ratio over the five most recent eligible runs**, not on the latest run alone. A run is eligible when its single-session level has at least 8 requests; with fewer, "p95" is just the maximum. A later run with 4 requests per session gave 1.80× and is shown as indicative only. It was started from the UI, whose Run protocol button had sent 4 requests per session instead of the protocol's 8; that is fixed.

## Phase 5: hardening and deployment

| Task | Status | Evidence |
|---|---|---|
| P5-T01 security, TLS, secrets, packaging | Done | `scripts/security_tests.py` (live, 26 tests) writes `docs/security/security_review.json`. The gateway refuses non-loopback binds without tokens. `NAFS_AUTH_TOKENS_FILE` loads secrets; tokens must be at least 16 characters; secrets are redacted from logs. `deploy/tls/` holds Caddy and nginx configurations. `scripts/package_release.sh` builds the release with SHA256SUMS. `api/schemas/Config.schema.json` validates all three configurations |
| P5-T02 deployment and monitoring | Done (Linux host run pending) | `deploy/docker/docker-compose.yml` (vLLM-CPU built from the pinned checkout, Caddy, Prometheus, file secrets); `deploy/systemd/`, `deploy/launchd/`; `deploy/monitoring/alerts.yml` (7 rules, passes promtool); Grafana dashboard; `/v1/alerts` with a drill; `scripts/verify_deploy.sh` writes `deploy/verified.json` |
| P5-T03 GPU scale-out | Deferred by the user ("skip GPU for now") | none |
| P5-T04 Profile C and retrieval | Done | `scripts/profile_c_compare.py` writes `benchmarks/profile_c_decision.json`. Retrieval memory (FR-061): BM25 over verified answers, quoted as data with provenance, opted-out conversations excluded, injection test |

The security review found one high-severity issue. Sandboxed code could kill the gateway (`os.kill(os.getppid(), 9)`, probe ST-19), and any other process of the same user. On macOS, the sandbox now denies all signals except to the process itself. On Linux, a PID namespace (`unshare -p --fork --kill-child`) means no outside process can be addressed. A regression test covers it.

## Phase 6: release and publication

| Task | Status | Evidence |
|---|---|---|
| P6-T01 release gate | Gate implemented; criteria pending | G6 on the dashboard. 1-point non-inferiority needs 6,200 pairs (fallback: release with the 3-point result and a documented limitation). The regression baseline is running. Scholar review: `docs/release/scholar_review_packet.md` awaits a reviewer |
| P6-T02 documentation | Done | `docs/operations/operator_runbook.md`, `docs/user_guide.md`, `docs/api/api_reference.md` (generated from OpenAPI) |
| P6-T03 publication | Done (indicative results) | `docs/publication/nafs_aql_lab_results.pdf`; numbers come from `scripts/gen_publication.py` and name their run IDs |

## Phase 7: CPU inference performance (added 2026-10-18)

The roadmap ended at Phase 6. Phase 7 was defined in `data/tasks.csv` and `data/requirements.csv` (NFR-030, NFR-031, FR-080, FR-081, ER-020) for the goal of maximum speed on CPU, with GPU work skipped.

| Task | Status | Evidence |
|---|---|---|
| P7-T01 latency attribution | Done | `usage.stages` on every reply: queue, prefill, decode and KV-cache reuse from llama-server's own timings. `GET /v1/perf/stages`. Inspector waterfall and the "Where the time goes" card on the Performance page |
| P7-T02 engine A/B | Done | Interleaved runs for flash attention (no difference; A-044), BLAS prefill (inconclusive: spread 26–78 tok/s under load) and thread polling (no difference). Verifier warm-up: first verification 0.45 s → 3 ms. **Sequential, uncertainty-gated samples** (A-049): the two agreement samples had tripled the factual fast path (12.6 s median, TTFT 1.37 s); drawn only when the draft's confidence is within the band, the path is 4.3 s and 0.14 s (`benchmarks/samples_ab_factual.json`) |
| P7-T03 early escalation | Done; study below | Stops an unconfident open-ended draft under the fixed rule (`gate.early_escalation_tokens`, `gate.early_escalation_margin`); unit test; `scripts/early_escalation_study.py` → `benchmarks/early_escalation_study.json` |
| P7-T04 EAGLE-3 heads | Measured: **not adopted** (either system) | Heads converted to GGUF Q8_0 and pinned in `config/models.manifest.json` (`eagle3_heads`). Interleaved A/Bs (`scripts/eagle3_ab.py`): System 1 decode 1.04× at temperature 0 and 0.95× at the production temperature with 4 draft tokens (acceptance 49–52%), 0.86–0.93× with 8; System 2 0.99× with 4 (46%), 0.87–0.91× with 8 (`benchmarks/eagle3_ab_system{1,2}.json`). FR-081 (≥ 1.2×) not met (A-048). Opt-in: `EAGLE=1 scripts/serve_llama.sh start` |
| P7-T05 weight formats | Measured: **Q4_K_M stays** | HF → F16 → Q4_0 / IQ4_NL / Q4_K_M with `llama-quantize`; interleaved `llama-bench` (`logs/quant_ab.log`): decode 18.8 (production Q4_K_M), 16.9 (Q4_0), 18.0 (IQ4_NL), 17.7 (own Q4_K_M) tok/s; prefill 87 / 73 / 73 / 96. No gain from the repack formats on AVX2 (A-052) |
| P7-T06 efficiency frontier | Partly done | Lab policy map (per-domain best policy over λ); saved tokens and agreement in the early-escalation study |
| P7-T10 vLLM backend in Docker | Implemented; final gateway check blocked by host memory today | `scripts/serve_vllm_docker.sh` runs the `carstenz/vllm-smoke:0.1.0-cpu` image (vLLM 0.30, x86 CPU build) on :8101 as the gateway's `vllm` set; `scripts/vllm_check.py` drives every orchestrator request path against it (`benchmarks/vllm_check.json`). Found and fixed: `--disable-log-requests` is not a 0.30 flag; BF16 never returns a token on the AVX2-only VM (FP32 only); vLLM's three processes cost about 3 GB beyond the weights, so the 1.7B needs an 11–12 GiB VM and was OOM-killed at 10 GiB; the KV cache, context and prefill batch are sized explicitly. With the owner's workloads resident (35 GB swap in use) the VM would not start at 8 GiB; at 7 GiB it started and the 0.6B loaded (0.12 GiB left for the KV cache against 0.22 needed at 1,024 context); at 7.5 GiB, with 0.7 GB free on the host, the VM no longer started. The serving script, the gateway configuration and `scripts/vllm_check.py` are ready; the check needs a host with 7.5 GiB free for the stand-in or 12 GiB for Qwen3-1.7B (A-050) |
| P7-T09 judge on System 1 | Measured: **not adopted** | `judge.backend = "system1"` option added. Rates harness, 45 labelled items: System 1 judge r 0.78, φ 0.27 (upper 0.50) against System 2's r 1.00, φ 0.045 (upper 0.23). It accepts a quarter of wrong answers, so it is not a verifier (A-051) |
| P7-T08 adaptive reasoning exit (FR-082) | **Done; adopted** | System 2 reasons in chunks on llama-server's raw continuation endpoint (cached prefix); at reflection signals, with a 64-token gap and at the latest every 256 tokens, a short answer is probed. Math: the exact verifier ends the reasoning; factual: two agreeing probes (Dynasor-CoT). The final answer is generated after the reasoning as on the standard path. Per-request `adaptive_exit`, Lab system `slow_noexit`, preset "Early exit". Live smoke (shared machine, load 15): a GSM8K-style item verified after 3 probes at 385 of 1,024 reasoning tokens, probes 0.6–0.8 s each. **Paired Lab study** (run `e7607cbb`, 12 GSM8K items, `slow_only` with the exit vs `slow_noexit`, shared machine, load 6–7): 11 of 11 scored items correct in both arms (one item exceeded the wall-clock cap in both); mean wall 220 s → 123 s (−44%, cost ratio 1.80), median 222 s → 91 s; mean reasoning tokens 723 → 348 (−52%); paired Δ accuracy 0 (A-047) |
| P7-T07 sweep and documents | Done | `scripts/e2e_sweep.py` (37/37: 25 endpoints, chat/stream/verify/perf, 7 UI pages without JS errors). `scripts/capture_screenshots.py` (headless Chrome) → `docs/shared/screenshots/`. Report v2.0, slides, roadmap v2.0 and spec v1.1 recompiled with Tectonic |

**Where the time goes (live and lab replies with stage timings, shared machine).**

- **Fast path:** about 100% of wall time is the System 1 draft. The prompt cache serves 73–83% of prompt tokens, so prefill is small.
- **Decode is the bottleneck:** 2.6–4.3 tok/s in service against 22.9 tok/s in an idle llama-bench run. The other workloads on the four cores cost a factor of 6–9.
- **Engine levers that remain:** EAGLE-3 speculation and repacked 4-bit formats (P7-T04/T05).

**Early-escalation study (FR-080; 40 open-ended prompts, System 1 drafts with the production prompt, shared machine).** The full-draft fixed rule escalated 10 of 40 drafts. Results by margin:

| Setting | Agreement with full-draft decision | Escalations caught early | Wrong early escalations | Draft tokens saved |
|---|---|---|---|---|
| N = 48, margin 0.15 (first default) | 100% | 0 of 10 | 0 | 0 (never fires) |
| **N = 32, margin 0.04 (adopted)** | **100%** | **3 of 10** | **0** | **347 (about 116 per early escalation)** |
| N = 32, margin 0.02 | 95% | 5 of 10 | 2 | 642 |
| N = 32, margin 0 | 80% | 10 of 10 | 8 | 1,483 |

The adopted setting meets FR-080's 95% rule with no wrong escalation on this set. The drafts' mean log-probs cluster just above the threshold (−0.175 at the default λ), so small margins matter. With only 40 prompts, re-run the study (`--rescore` re-scores saved log-probs without new drafts) when the calibrated gate replaces the fixed rule (A-046).

## Defects fixed in the Phase 5–7 round (2026-10-18, evening)

- **Backend switch:** the gateway switched to a backend whose servers were not running, so every request returned 503 until someone switched back. It now refuses with a clear 503 unless `force` is set, logs every switch, and the System page shows "Not running" instead of the switch button. The end-to-end sweep checks this.
- **Alerts and lab traffic:** evaluation and load-test replies kept the fast-path latency alert firing while nobody was waiting. They are now tagged as lab traffic (`ephemeral = 2`, Prometheus label `traffic="lab"`) and excluded (A-043).
- **NFR-007 verdict:** gate G4 followed the latest protocol run alone. It now uses the median over recent runs with at least 8 requests per session, and the UI's Run protocol button sends 8 requests per session instead of 4.
- **UI:**
  - the Performance page never rendered its stage card (a missing `alive` argument);
  - a global `.dot` style changed the System page's health dots;
  - uncaught "Failed to fetch" errors appeared while the gateway restarted;
  - raw JSON showed in the Muhasaba and System pages (`Audits: {}`), and float noise in labels (λ = 0.05000000074505806, V ≥ 0.699999988079071);
  - the Insights table overflowed.
- **Found by running the web UI (2026-10-19):**
  - the Inspector's waterfall counted the adaptive exit's probe time twice (inside the deliberate stage and again as its own row); stage times now add up to the wall time, with a test;
  - a dropped stream (the browser suspended network I/O mid-answer; the gateway correctly cancelled the request and freed the slot) showed only "network error"; it now says the request was cancelled and to press Regenerate.
  - Known limitation: after a page reload, *Think harder* and *Regenerate* on an old turn no longer carry its reference answer (it is not stored with the conversation), so the new reply is judged, not exactly verified.
- **Release tooling:** `nafs-gateway --check` printed Rust `{"Ok": …}` wrappers; the config-schema generator needed Python 3.11.

## Phase 8: sequential evidence (added 2026-10-20)

Phase 8 asked what quantum theories of consciousness and the neuroscience of decision making contribute to Nafs-Aql's accuracy and speed, under four tests applied before any code: established, transferable to a classical CPU pipeline, a theorem with an engineering consequence, measurable here. The review is `docs/research/quantum_neuro_review.md`; the report's Section 8 carries the propositions (Q1–Q3, N1–N5).

| Task | Status | Evidence |
|---|---|---|
| P8-T04 review | Done | **Quantum consciousness: negative result.** Decoherence separates the cognitive timescale from any quantum one (Prop. Q1), and a classical machine cannot follow more than about 30 two-level systems or inherit a speed-up from a process being quantum (Prop. Q2), so nothing transfers as an algorithm (A-055). Quantum cognition's one structural lesson, order effects as anchoring (Prop. Q3), is already implemented by the draft-blind slow path. Wald's bound shows the mean-log-prob early escalation cannot be given an error guarantee at its 0.04 margin (≥ 260 tokens needed, drafts 47–291; Prop. N3, A-056). The probe gap of 64 tokens is within 1 % of the optimal inspection interval (Prop. N5, A-059) |
| P8-T01 confidence-signal study | Done (shared machine) | `scripts/confidence_signal_study.py` → `benchmarks/confidence_signal_study.json`: 150 GSM8K drafts, 31 wrong (0.793). AUROC for predicting a wrong draft: lowest token 0.65 [0.54, 0.75], top-1/top-2 margin 0.61, lowest 16-token window 0.58 [0.49, 0.70], mean log-prob 0.56. **The fixed rule's mean-log-prob threshold escalates 0 of 150 math drafts and catches 0 of 31 wrong ones** (math drafts sit at −0.03 whether right or wrong), so on unverified math chat every wrong draft was kept and the agreement samples were never drawn. Theorem 2 needs P(wrong | escalated) > 0.65 at the default price for mathematics; the best precision at ≥ 10 % escalation is 0.47. Offline cascade estimate (Prop. 15): 0.793 → 0.85 at 20 % escalation, which the default price rates as a loss |
| P8-T02 lowest-window signal and exact early escalation | Done | `nafs-core/src/sequential.rs`; `low_window_logprob` on every decision and in the calibration's logistic model; fixed-rule reason for `gate.fixed.low_window_domains`; the draft stream stops the moment a window crosses the threshold, with the decision the full draft would get (Prop. N2; unit test of the running-minimum property on 200 random drafts, orchestrator test with the mock backend). **Threshold left off** (`low_window_logprob_threshold` absent): the mechanism is exact, the measured signal is too weak to use it (A-057). Recorded as a measured negative result |
| P8-T05 agreement samples on unverified mathematics | **Done; adopted** | On the first 60 study drafts the two production samples were drawn: agreement AUROC 0.76 [0.62, 0.90] (0.82 with the lowest token as tie-breaker) against 0.72 for the lowest token and 0.61 for the window. "Any disagreement" escalates 25 %, catches 8 of 12 wrong drafts, expected cascade accuracy 0.80 → 0.91–0.93. `gate.samples_on_unverified_domains = ["math"]` draws the samples after every unverified math draft (A-060); orchestrator test |
| P8-T03 posterior exit rule | Done; **option, not default** (A-058) | `exit_rule = "posterior"`: Dirichlet(1) posterior over all probe answers, stop when P(majority) ≥ a bound collapsing from 0.85 to 0.70 over the budget (Prop. N4). Two agreeing probes stop (0.875) as before; one probe never does (0.75); a [3, 1] split stops after about 45 % of the budget. Lab systems `slow_exit_consistency` / `slow_exit_posterior`, preset "Exit rule". Paired run `73cd5bba` on 6 factual prompts: same factual content in both arms; reasoning tokens 357 (consistency) vs 382 (posterior), wall 93 s vs 101 s; where probes disagreed first, the posterior rule continued past a [2, 1] split as designed. No token saving on this set, so `consistency` stays the default (`benchmarks/exit_rule_run.json`) |

What was deliberately **not** changed: the mean-based early escalation (kept at its measured N = 32, margin 0.04 and labelled a heuristic), the probe schedule, and anything suggested by quantum-consciousness theories.

## Web UI test pass (2026-10-20)

The UI was driven in a real browser (every page, the chat with streaming, Inspector, search, theme toggle and shortcuts, Rooms over WebSocket, the alert drill, the judge and verifier playgrounds, the Lab presets) with a console and network error collector, plus API probes of every form the UI submits. No JavaScript error and no failed request occurred; the defects below were layout, accounting and error-handling faults. All are fixed and covered by the sweep (now 42 checks) or a unit test.

| Defect | Cause | Fix |
|---|---|---|
| The latency waterfall said "Gateway and other 8.3 s" on a 13.4 s reply: the samples' time was counted as overhead | Samples drawn after the draft (P7-T02, now on every unverified math draft) were still recorded as overlapping the draft | One `sample_after` stage with the group's wall time, counted in the totals; Inspector, breakdown API and publication script know it; unit test and sweep check that the stages add up |
| Tables in half-width cards (Run history, Muhasaba verifier quality, Tazkiya training data, System ledger) spilled outside their card | `.card-b` had no overflow handling | Cards that hold a table scroll horizontally |
| Run history printed raw ids (`slow_exit_posterior`) in its Other column | The label helper was not applied there | Uses the Lab's system labels |
| With the Inspector open at widths up to 1250 px, the composer, its controls and the Send button were hidden under the drawer | The drawer overlays the chat column | The thread reserves the drawer's width |
| Searching conversations with no match said "No conversations yet" | One message for two cases | "No conversation matches your search." |
| An invalid JSON schema in the verifier playground left the previous verdict on screen | The handler returned before clearing the result | The result is replaced by the error |
| A 9,000-word message returned HTTP 500 with the model server's internal message | NFR-006's prompt cap trimmed only history; the newest turn was sent whole | The newest turn is checked against the cap (HTTP 400 with the token count); a backend "exceeds the context size" reply also maps to 400 |
| Malformed request bodies (`max_items: -3`, invalid JSON) came back as plain text, an unknown dataset leaked an OS error | Framework rejections bypassed the error envelope | One outermost layer rewrites them to `{code, message}`; "unknown dataset 'x'" |

Checked and found correct: HTML in a room name is escaped; an empty room name creates nothing; an infinite loop in the playground stops at the 20 s CPU cap; empty verification input and out-of-range parameters are refused with clear messages; the alert drill and judge errors show inline. Not changed: rooms cannot be deleted (no endpoint; the acceptance test leaves `acceptance-*` rooms behind), and at half width the System ledger's last column needs a horizontal scroll.

## Linux sandbox: found by CI on the first push (2026-10-20)

GitHub-hosted Ubuntu 24.04 was the first clean machine the suite ran on. It showed two things:

- **Namespaces blocked by default.** AppArmor denies unprivileged user namespaces, so every code verification failed closed while `--check` claimed isolation was available. `isolation_available()` is now a functional probe, CI enables the sysctl, and the runbook documents it (A-053).
- **The Linux sandbox did not isolate the filesystem or `fork`.** This was already noted in the Phase-1 code review, but the sandbox tests asserted macOS-only guarantees on every platform, so they were scoped to macOS first. **Fixed (2026-10-20):** the Linux sandbox now has a mount namespace and a seccomp filter (A-054). User data directories are replaced by empty mounts, every mount is read-only except the scratch directory, capabilities are dropped, and `fork`, spawn and mount-table changes are denied. The file, spawn, fork-bomb, `/proc`, unmount and unshare attempts, plus a canary-file leak check, now run on macOS and Linux alike. `isolation_available()` runs the full stack as its probe.
- **Docker needs one more relaxation.** Run in a container (Ubuntu 24.04, default profiles) the probe showed `unshare` is denied by Docker's AppArmor profile as well as its seccomp profile, so the verifier failed closed; `docker-compose.yml` now sets `apparmor=unconfined` next to `seccomp=unconfined` (A-040). With both relaxed the sandbox tests pass in a container with and without the masked `/proc`.
- **Open item:** the gateway's own install, config and data directories (including the SQLite database) stay readable by sandboxed code on Linux, because the hide list is fixed. Passing the gateway's data and config paths into the hide list would close it.

## Speed

| Change | Effect |
|---|---|
| Skip the k cheap samples when an exact verifier decides (A-028) | System 1 draft median **7.9 s instead of 13.4 s** (about 40% faster; A/B on a shared machine, `benchmarks/reference-machine/samples_skip_ab_SHARED_MACHINE.txt`) |
| Skip samples on open-ended prompts while the fixed rule is in force | Same mechanism on most chat traffic |
| Cancellation reaches the backend | A stopped request frees its slot at once instead of generating up to 1,800 more tokens |
| Judge output constrained to the rubric JSON | About 60 answer tokens instead of free text; no lost calls from missing JSON |
| Per-domain thinking budgets by default in chat | Open questions reason within 512 tokens instead of 1,024 |
| Speculative decoding (Qwen3-0.6B draft, or n-gram lookup) | **Measured, no gain:** median decode was 7.3 tok/s baseline, 6.5 with the draft model and about 6.6 with n-gram lookup (2 alternating rounds, temperature 0.6, `logs/perf_spec.log`). The draft model competes for the same four cores and memory bandwidth, so it stays off (`SPEC=1` remains available) |
| Prompt-cache warm-up at startup | Each domain's system prompt is prefilled on both models when the gateway starts idle; a cold first request had 9.8 s TTFT, against about 1 s warm |
| System 1 slots 3 → 4 | Four sessions no longer queue behind three slots; batched decode makes the fourth slot cheap |
| Verifier warm-up at start-up (P7-T02) | First verification after a restart: 0.45 s → 3 ms |
| Early escalation (P7-T03) | N = 32, margin 0.04: 3 of 10 escalations decided early, about 116 draft tokens (8–40 s on this host) saved each, 100% agreement on 40 prompts |

Flash attention, a q8_0 KV cache and 3 instead of 4 threads showed no consistent gain beyond the run-to-run spread in two loaded runs, so the defaults stay. A sequential run had suggested flash attention off was 30% faster on System 1; an interleaved repeat (on, off, on, off) showed no difference, so the gap was an ordering effect of the shared load (A-044). `scripts/perf_tuning.sh bench|spec` repeats both studies on an idle machine.

## Lab and UX

| Area | Delivered |
|---|---|
| Chat | Verified memory (sources shown under the answer); **Both** path: System 1 and System 2 answer the same question side by side, and the person picks the better one (preference trials; A-037). Options popover keeps the composer simple. A one-line "why" under each answer gives the gate's reason. Automaton strip with stations (Draft *nafs* → Flagged *lawwāma* → Deliberate *ʿaql* → Accepted *muṭmaʾinna* / Abstain *tawaqquf*); 👍/👎 feedback; side-by-side compare of the System 1 draft and the final answer; regenerate, think harder, fast only; inspector with judge bars and verifier (r, φ); conversation search; thread bar with research opt-out and export; keyboard shortcuts (`?`); privacy notice; design-notes panel citing the sources |
| Lab | Insights: a pooled policy map of the best System 1 / System 2 policy per domain at each price of latency, with the value of deliberation and human preference; resumable runs; roadmap gates dashboard (G0–G6); six experiment presets framed as research questions (routing, λ frontier, coupling, budget curve, full G2, strong code tests); λ-variant gated systems; **accuracy–cost Pareto frontier**; confidence intervals and eligibility badges on every metric; a metrics glossary; parameter derivation from runs |
| Muhasaba (gate and ledger) | Alerts with a drill, over live traffic only (evaluation and load-test replies are tagged as lab traffic, A-043); value of deliberation from Both trials (preferred-S2 rate with CI per domain); drift monitor with sparklines; gate modes, promotion status, held-out calibration, parameter provenance, verifier quality with EvalPlus option, judge playground, audit stream, capability table with human ratings |
| Tazkiya | Training-data readiness, review queue, model registry, pipeline commands |

## Known issues and next steps

0. **NFR-007** fails on this shared 4-core host (see Phase 4). Gate G4 judges the median over eligible runs: 2.15 over 1.25, 2.15 and 2.48. Re-measure on an idle or larger machine before deciding on concurrency limits.
0. **NFR-004 under load:** the deliberate path decodes at about 2 tok/s while the host is shared, so an 820-token reasoning trace takes about 6 minutes. The roadmap's G1 fallback (reduce the budget to 512) should wait for the H4 budget curve, because it trades accuracy for speed. EAGLE-3 for System 2 is the speed lever that keeps accuracy.
0. **EAGLE-3 and repacked weights** (P7-T04/T05) need approval to run the GGUF converter on the downloaded heads and HF weights.

1. **Release-scale runs.** The 4,600-task baseline, G2 (700 pairs) and calibration (2,900 samples) need a faster host or nightly slices. Until then the fixed gate rule stays in force.
2. **Phase-3 training** needs about 5,000 verified traces (or synthetic examples), operator-supplied replay data (A-031) and a GPU host.
3. **Code φ** needs an equivalent-mutant filter before it can be compared with the ≤ 0.01 target (A-030).
4. **Shared machine.** Re-run the tuning and speculative-decoding studies when the host is idle.
5. **Judge cost** is 20–45 s per answer on this CPU (longer under load). No judge runs on the fast path (SR-003's lightweight check).
