# NAFS-AQL: COMPLETION PROMPT (v2)

Paste into Claude Code at the repo root. The facts in section 2 were verified against the live gateway and the repo on 2026-10-21. **At the start, re-check them (section 5, step 1). If the live gate output differs from section 2, the live output wins; note the difference.**

## 0. HOW YOU WORK

You run autonomously for hours and nobody is watching. Do reversible work without asking. Stop and ask only for a true blocker, in the format of section 9.

- **Keep a checkpoint file**, `docs/status/PROGRESS.md`: what is done (with run IDs and commit hashes), the next three actions, and open questions. Update it after every unit of work. After any context reset, read it before doing anything else.
- **Verify before you claim.** After each unit of work, recompute the gates and quote the result. Never describe a result you have not just observed.
- **Run long jobs in the background**, wait with a monitor or an until-loop, and never poll in a loop. macOS has no `timeout`; bash is 3.2 (an empty array under `set -u` aborts: use `${A[@]+"${A[@]}"}`). Scripts that read the TOML config need `python3.11 -I` (the default `python3` is 3.10).
- **Parallelize independent reads and tool calls.** Use subagents only for read-only research or for independent review of a diff. Never use them to run benchmarks: they would compete for the same four cores and ruin the measurement.
- **Keep your own output terse**: tables and conclusions, not logs.

## 1. MISSION

**Nafs-Aql** is a verifier-gated dual-process assistant on small Qwen models, CPU only. System 1 (Qwen3-1.7B) drafts at once; the Aql Gate escalates to System 2 (Qwen3-4B-Thinking-2507) only when expected error saved outweighs the added latency (`L·(ê₁−ê₂) > λ·Δc`); exact verifiers, not model agreement, decide what counts as correct. Both models run as GGUF Q4_K_M on llama.cpp, behind a Rust gateway (`rust/`), with a browser UI embedded in the gateway.

**Objectives, in priority order**
1. **Best CPU inference performance**: latency p50/p95, tokens/s, memory, behaviour under load.
2. **Most accurate S1↔S2 interaction**: calibrated gating, correct routing, quality non-inferior to always-S2, and coupling that demonstrably helps.

**Definition of done**
- Every gate whose evidence can be produced on this machine is computed from real, reproducible runs.
- Every gate that cannot be closed here stays `pending` or `fail`, with the exact missing evidence, the command to produce it elsewhere, and a machine-hour estimate **derived from per-item times you measured**.
- S2 latency and routing quality are each measurably better or measurably not improvable, with the paired A/B stored. A recorded negative result counts.
- G5 is still `pass`; CI is green on the PR.

**It is a correct outcome for G0, G2, G3 and G6 to still be `pending` when you finish.** A gate turned green by anything other than evidence is a failure of the whole task.

## 2. GROUND TRUTH (do not re-derive; do not redo)

### 2.1 Machine
Intel Core i3-8100: 4 cores, no SMT, **AVX2 only (no AVX-512, no AMX)**, 32 GB dual-channel DDR4-2400 (about 38 GB/s), macOS 15.4.1, no GPU. llama.cpp is pinned at `797de63` and built CPU-only in `third_party/llama.cpp/build/bin`. **The machine is shared with the owner's other workloads** (an Ollama server, mistral.rs, other llama-servers; load average often 5 to 15). These belong to the owner.

### 2.2 Live gate state (from `GET /v1/gates`)
| Gate | Status | Criteria and current evidence | Can it close here? |
|---|---|---|---|
| G0 | pending | **H7 pass** (Qwen3.5 decodes at 79 to 80% of dense Qwen3). **P0-T04**: fast-only and slow-only baselines on ≥4,600 tasks (the criterion text names an interim release at 2,000): 529 fast-only and 27 slow-only scored | Only by nightly resumable slices; the slow baseline is the cost |
| G1 | **FAIL** | MVP acceptance 8/9 (A3: slow path 354 s against ≤240 s, measured at load 9 to 12; the same test passed in 161 s under lighter load). Latency: fast TTFT 553 ms (≤2,000, ok), **slow path 354 s (>240)** | **Yes**: measure on an idle host first, then cut S2 tokens |
| G2 | pending | ECE ≤0.05 on ≥2,900 held-out samples (no promoted calibration); H1 routing accuracy ≥0.70 (6 blind-gate items, need ≥100); ER-002 3-point non-inferiority on ≥700 pairs (16); ER-004 wrong-draft adoption ≤5% upper bound on ≥200 trials (0 of 5) | Partly: see Track C |
| G3 | pending | TR-007 an adapter evaluated by the promotion gate (needs a GPU; the owner said "skip GPU for now"); OR-005 rollback recorded in the registry | OR-005 yes (a drill); TR-007 no |
| G4 | **FAIL** | NFR-007: p95 at 4 sessions ≤2× single session; median ratio **2.15** over runs 1.25, 2.15, 2.48. ER-007 load tests and P4 pass | Re-measure idle; Track D |
| G5 | pass | 0 open high findings; deployment recipes 11 pass, 0 fail, 1 skipped; Profile C decision recorded | Keep it passing |
| G6 | pending | ER-002 1-point on ≥6,200 pairs (16; the criterion names a fallback: release with the 3-point result and a documented limitation); ER-008 regression green (500 items, baseline 69.4%); SR-005 scholar sign-off | Pairs: no. Sign-off: human only |

Gate logic is `rust/crates/nafs-gateway/src/gates.rs` (FR-053); thresholds, evidence files and the rules for "pass", "fail" and "pending" are there. Read it, do not edit it.

### 2.3 Already measured on this CPU: do not repeat unless you have a new hypothesis and an idle host
- **Decode, idle**: System 1 22.9 tok/s, System 2 10.3 tok/s. In service under the owner's load: 2.6 to 4.3 tok/s (a factor of 6 to 9 slower).
- **No gain, measured with interleaved A/B**: EAGLE-3 heads (1.04× and 0.99× at about 50% acceptance), a Qwen3-0.6B draft model and n-gram speculation (6.5 against 7.3 tok/s), flash attention, q8_0 KV cache, thread polling and 3 threads, Q4_0 and IQ4_NL repacked weights (17 to 18 against 18.8 tok/s), vLLM on CPU (3.2 tok/s FP32, no BF16 kernels).
- **Adopted gains**: adaptive reasoning exit for System 2 (−44% mean wall time, −52% reasoning tokens, 11/11 correct in both arms; n=12 GSM8K, shared machine); cheap samples drawn only when the draft is uncertain (factual fast path 12.6 s → 4.3 s); samples skipped when an exact verifier decides; prompt-cache warm-up (first token 9.8 s → about 1 s); early escalation (N=32, margin 0.04).
- **Quality findings**: Qwen3.5-2B as System 1 is +17 points (95% CI +7 to +30, n=30) but 3.7× slower, so Profile A stays. System 1 as the rubric judge accepts 27% of wrong answers, so it stays on System 2. The fixed rule's mean-log-prob threshold escalated **0 of 31** wrong math drafts (token-confidence AUROC 0.49 to 0.65 across signals); agreement with two cheap samples has AUROC 0.76, so unverified math drafts now always draw them (about 15 to 20 s idle).

### 2.4 What the hardware allows
Decode is **memory-bandwidth bound**. A Q4_K_M 4B model is about 2.5 GB, so the ceiling is about 15 tok/s at 38 GB/s, and idle System 2 already reaches 10.3 (about 70%). Engine flags and kernels have little headroom left. The remaining levers are **fewer tokens and fewer calls**: tighter per-domain thinking budgets from the measured accuracy-versus-budget curve, adaptive exit in more domains, a better gate (fewer needless S2 calls), and not letting S1 and S2 fight over four cores.

### 2.5 Two ideas in earlier drafts that do not apply
- *Reusing System 1's prompt KV cache in System 2* is invalid: they are different models (1.7B and 4B), and KV caches do not transfer. It only applies to the single-model profile (the vLLM stand-in). Prefix caching inside each server is already on (73 to 83% of prompt tokens come from cache).
- *Evaluating ONNX Runtime, OpenVINO or CTranslate2* costs days of conversion on this host for Qwen3 with thinking. Do it only if a probe of at most an hour shows ≥1.3× decode on an idle host.

## 3. RULES

1. **Evidence integrity.** Gate status is computed from stored runs and records. Never edit gate logic, thresholds, stored records, fixtures or the registry's history to change a gate. If a threshold looks wrong, write a justified proposal (`docs/decisions/NNN-title.md`, plus an entry in `ASSUMPTIONS.md`) and list it in `docs/status/HUMAN-NEEDED.md`. The decision is the human's.
2. **No fabricated or extrapolated numbers.** A number needs a run ID, configuration, seed, commit, hardware and the load average at the time. Pin seeds, model revisions by hash (`config/models.manifest.json`), quantization and thread counts. A run that is smaller than a gate requires is reported as **scaled**, with n and a confidence interval. If a run cannot be done here, record it as blocked with the exact command.
3. **Measurement hygiene.** Before any timing, check `uptime` and `ps -Ao pid,pcpu,comm -r | head`. Label a result *idle* only if load1 <1.5 for five minutes and no foreign process uses >20% CPU; otherwise label it **SHARED MACHINE**. A performance claim needs an **interleaved** A/B (A B A B, at least three rounds): one sequential run once showed a fake 30% gain. **Never kill, renice or pause the owner's processes.** If the machine is never idle, run what is valid, label it, and write the idle-host command into HUMAN-NEEDED.
4. **Protect the owner's data and runtime.** Back up with `sqlite3 data/runtime/nafs.sqlite ".backup <file>"` before bulk or schema work. All test traffic must be ephemeral or lab-tagged. Check `inflight` in `GET /v1/system` before restarting the gateway or System 2 (the owner chats in the UI). Never delete the owner's conversations or rooms.
5. **G5 stays pass.** After any change, re-run the security and deployment checks. A new dependency, endpoint or model artifact needs re-verification and a recorded decision.
6. **Human-only items.** Prepare the scholar sign-off package and stop. Never mark SR-005 signed. Do not create the release tag unless every gate that is automatable here passes (expect it not to).
7. **Scope.** No new features, UI redesigns or refactors unless they serve a gate or one of the two objectives. No GPU spend. Do not train on CPU.
8. **Commits.** Small, tests green, conventional messages. Never force-push or rewrite history. No model weights, large binaries or secrets. Never skip hooks.
9. **Ambiguity.** Choose the conservative option, record it in `docs/decisions/`, continue.

## 4. COMPUTE ARITHMETIC (use it to plan, then replace with your own measurements)

One slow-only item takes about two minutes of System 2, so the 4,600-task baseline is about 190 machine-hours. **Do not run separate studies for each gate.** The same stratified, seeded item set, run once through `fast_only`, `slow_only` and `nafs_aql` in resumable nightly slices, feeds P0-T04 (G0), ER-002 (G2, G6) and the ablations. Labelled System 1 drafts for calibration need **no System 2 at all** (a System 1 draft takes seconds), so the ECE criterion may be closable in hours, not days. Measure per-item times in the first hour and write the real estimates into `gap-analysis.md`.

## 5. WORK PLAN, in order of value per machine-hour

**Step 1: orient (about 15 minutes).** Read `ASSUMPTIONS.md`, `docs/roadmap/STATUS.md`, `gates.rs`, and the pieces named below. Fetch `GET /v1/gates` and compare it with section 2.2. Write `docs/status/gap-analysis.md`: per gate and criterion, the exact rule, current evidence, the gap, the task that closes it, and **automatable here / needs hours / human-only**. Commit `docs: add gate gap analysis`.

**Track A: the truth about G1 and G4 (cheap, decides everything after it).**
Wait for or verify an idle host. Then run `make acceptance` (A9 restarts System 2: check `inflight`) and the load protocol three times (`POST /v1/bench/protocol`, 8 requests per session). Record CPU model, cores, RAM, instruction sets (`sysctl machdep.cpu`), thread counts and the load average. If the slow path is ≤240 s when idle, G1's latency criterion is closed by evidence and you move on. If not, go to Track B. If the idle NFR-007 ratio is ≤2, G4 closes; if not, Track D.

**Track B: deliberate-path latency (G1; objective 1).** Levers, in order:
1. *Tokens by domain.* Run the budget curve (`slow_b<N>` systems, paired, scaled) to get b* per domain. The roadmap's fallback of 512 tokens is acceptable only with a paired non-inferiority result.
2. *Adaptive exit beyond math and factual.* Where an exact verifier exists (code tests, JSON Schema) the verifier is an exact stop signal. Measure it paired against the current exit.
3. *Probe and answer stages.* Use the per-stage timings (`usage.stages`, `/v1/perf/stages`); the probe gap is already within 1% of the optimum, so look at answer length and the verifier round trip.
4. *Core sharing.* S1 uses 4 slots and S2 uses 2 on four cores: measure `-np` and thread splits.
5. *Only with an idle host and a ≥1.3× probe:* another runtime or quantization.
Adopt a change only if an interleaved paired A/B shows it non-inferior (pre-registered margin, 95% CI) and faster. Store negative results too.

**Track C: routing and coupling (G2; objective 2).**
- *Calibration.* Check how `recalibrate_from_ledger` selects items (`nafs-core/src/orchestrator.rs`) and produce exactly that evidence: labelled System 1 drafts on verifiable sets (`fast_only` runs) are the cheap source. Fit on a calibration split, report held-out ECE, Brier and AUROC of the gate score against "System 1 wrong" (the gate already reads `low_window_logprob`; agreement is the strong signal for math). Never calibrate on the test split. Promotion needs n ≥2,900, at least 30 right and 30 wrong drafts, and held-out ECE ≤0.05 (`POST /v1/gate/recalibrate`).
- *Routing.* H1 needs ≥100 blind-gate items; produce the cost-quality curve (`nafs_aql_l<NN>` λ variants, Pareto frontier in the Lab) and pick the operating point that is non-inferior at least latency. Unsure or out-of-distribution → System 2.
- *Non-inferiority.* Paired bootstrap, ≥1,000 resamples, fixed seed, per-domain results (hidden regressions). ER-002 at 3 points needs ≥700 pairs.
- *Coupling.* ER-004 needs ≥200 trials of the draft-conditioned ablation (`slow_conditioned`). Measure the **correction rate** (wrong drafts that System 2 fixes) and the **false-override rate** (right drafts it breaks). Test the failure paths: System 2 down (degraded mode), timeout, malformed draft, conflicting outputs; add unit or property tests where one is missing.
- *Ablations*, all stored: `fast_only`, `slow_only`, `nafs_blind`, `nafs_aql`, λ variants, `slow_conditioned`, random routing.
Report every sub-gate as scaled, with n and CI, until the gate's own n is reached.

**Track D: concurrency (G4, NFR-007).** If the idle ratio still exceeds 2, study slot counts and thread split. Because decode is bandwidth bound, extra sessions add little throughput on four cores; if that is what the data shows, follow the roadmap fallback (document the single-session profile and recommended concurrency limit) and put any threshold change in a decision proposal. Do not touch the threshold. A soak run (at least an hour of mixed S1 and S2 traffic, watching memory growth) is optional extra evidence: start it in the background only after the gate work, and only if the host allows.

**Track E: G3.** *Rollback (OR-005):* the registry and commands exist (`config/models.registry.json`, `training/tazkiya.py promote|rollback`). Run a real drill with a real second artifact (for example the already built `models/Qwen3-1.7B-Q4_K_M.own.gguf`): promote, run a regression subset, roll back, verify the restored hash and behaviour, store the record. **Label it honestly as a registry drill with a non-adapter artifact**, and write down what it does not show. *TR-007 stays pending:* training needs a GPU and the owner deferred it. Verify `training/train_unsloth.py --dry-run` and `training/RUNBOOK.md` are current and list the exact commands.

**Track F: G0.** Record the profile decision with its evidence (H7, the Profile C comparison, the idle numbers from Track A). P0-T04 progresses with the shared item set from section 4.

**Track G: re-verify G5** after any change: `python3 -I scripts/security_tests.py`, `scripts/verify_deploy.sh --live`, `scripts/validate_artifacts.sh`, the dependency and secrets audit, and the upgrade decisions for anything that changed.

**Track H: release package.** Recompute every gate and write `docs/release/RELEASE-CANDIDATE.md`: the gate table, evidence links, honest limitations. Update `docs/release/scholar_review_packet.md` (methodology, evaluation design, sample outputs per category including failures, review checklist) and leave sign-off pending. **No tag exists in the repository today; do not create `v0.1.0-rc1` unless all automatable gates pass.**

## 6. COMMANDS

```bash
make up | make down                       # llama-servers + gateway on :8080 (check `inflight` first)
curl -s localhost:8080/v1/gates | python3 -m json.tool
curl -s localhost:8080/v1/system           # inflight, backend, config hash
cd rust && cargo fmt --all --check && cargo clippy --workspace --all-targets --release -- -D warnings && cargo test --workspace --release
make test | make contract | make acceptance | make rooms-test
python3 -I scripts/e2e_sweep.py            # API + every UI page, desktop and 375 px
python3 -I scripts/security_tests.py ; scripts/verify_deploy.sh --live ; scripts/validate_artifacts.sh
rust/target/release/nafs-eval --benchmark gsm8k --max-items N --systems fast_only,slow_only,nafs_aql   # or POST /v1/eval/runs; resume: POST /v1/eval/runs/{id}/resume
scripts/perf_tuning.sh bench|spec          # interleaved engine A/B
python3 scripts/gen_tables.py ; (cd docs/roadmap && ../../tools/tectonic nafs_aql_roadmap.tex)
```

## 7. GIT AND CI

Work on branch `feat/complete-nafs-aql`. Other sessions push to `main` in this repository: `git fetch` before every push, rebase on `origin/main`, never force-push. Run format, lint and tests before each commit (the CI mirrors them: Ubuntu and macOS builds, OpenAPI and schema validation, `gen_tables.py`). When done, push, open a PR with `gh` (the description holds the final gate table and a changelog), and check the CI result with `gh run list`.

## 8. OPERATING LOOP

Pick the highest-value unclosed gap → implement → test → benchmark with an interleaved A/B → store evidence → recompute gates → commit → update `PROGRESS.md`. After each track, print the gate table in one compact block. If blocked, append to `docs/status/HUMAN-NEEDED.md` and move to the next track; stop entirely only when nothing remains that this machine can do.

## 9. WHAT TO ASK THE HUMAN, AND HOW

Only for: the scholar sign-off, a threshold proposal, an idle-host run, a GPU training run, a change that touches the owner's running processes. Each goes into `docs/status/HUMAN-NEEDED.md` as: *what*, *why it matters*, *the exact command or action*, *what evidence it will produce*, *which criterion it moves*.

## 10. FINAL REPORT (print at the end)

1. Gate table before and after, with the reason each remaining non-pass is not closed.
2. Headline CPU numbers (TTFT, p50 and p95 latency, tokens/s, RAM, load average), before and after, labelled idle or SHARED MACHINE, with hardware.
3. Routing: held-out ECE, % escalated, quality against always-S2 with CI (scaled or full), correction and false-override rates.
4. Every item that needs a human, with its exact command or action.
5. Commits, branch, PR link, CI status, tag state (expected: no tag).

Begin with Step 1 now.
