# Nafs-Aql operator runbook

This runbook is for whoever installs, runs and maintains a Nafs-Aql deployment. It covers the two supported profiles:

- **Reference machine:** llama-server on macOS or a Linux AVX2 host.
- **Linux vLLM-CPU:** docker compose.

## 1. Install

| Step | Reference machine (macOS or Linux AVX2) | Linux vLLM-CPU |
|---|---|---|
| Models | `scripts/fetch_models.sh` (GGUF, about 3.6 GB) | pulled by vLLM on first start (Hugging Face cache volume) |
| Build | `cd rust && cargo build --release` | `docker compose -f deploy/docker/docker-compose.yml build` |
| Secrets | `export NAFS_AUTH_TOKENS_FILE=/path/to/tokens` (one token of 16 or more characters per line) | write `deploy/docker/secrets/nafs_tokens` |
| Start | `make up`, or the launchd / systemd files in `deploy/launchd/` and `deploy/systemd/` | `NAFS_DOMAIN=your.host docker compose -f deploy/docker/docker-compose.yml up -d` |
| TLS | put `deploy/tls/Caddyfile` or `deploy/tls/nginx.conf` in front of 127.0.0.1:8080 | the `proxy` service (Caddy) terminates TLS |
| Verify | `scripts/verify_deploy.sh --live` and `rust/target/release/nafs-gateway --check` | the same, from the host |

**vLLM in Docker (reference machine).** `scripts/serve_vllm_docker.sh start && scripts/serve_vllm_docker.sh wait` serves Qwen3-1.7B in BF16 on :8101 from the `carstenz/vllm-smoke:0.1.0-cpu` image, as the gateway's `vllm` backend set expects; switch with *Use this backend* on the System page and check it with `python3 scripts/vllm_check.py`. The Docker VM needs about 6 GiB free (10 GiB allocated here). Expect a load time of several minutes and, on four shared cores, slower decoding than llama-server.

**Linux hosts and the code sandbox.** Verifying model-written code needs unprivileged user, network and PID namespaces. Stock Ubuntu 24.04 blocks them (`kernel.apparmor_restrict_unprivileged_userns=1`); `nafs-gateway --check` then reports `sandbox_isolation: false` and code verification fails closed. Enable them with `sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0` (persist it in `/etc/sysctl.d/`), or run the gateway in the provided compose file, which relaxes seccomp for the same reason (A-040).

The gateway refuses to listen on a non-loopback address without tokens (SR-006). Set `NAFS_ALLOW_INSECURE=1` only on an isolated lab network. Browsers must reach the gateway on the same origin. For any other host name, set `NAFS_ALLOWED_HOSTS` or enable tokens.

## 2. Daily operation

- **Health:** `GET /healthz` reports `ok`, or `degraded` when System 2 is down. In degraded mode, answers fall back to the unverified System 1 draft (MR-004).
- **Dashboards:**
  - Prometheus scrapes `/metrics`; the rules are in `deploy/monitoring/alerts.yml` and the dashboard in `deploy/monitoring/grafana-dashboard.json`.
  - Without Prometheus, the Muhasaba page shows the same five alert rules: fast-path latency, error rate, escalation rate, verifier acceptance and calibration drift.
  - The alerts describe live traffic only. Evaluation runs and load-test protocols started from the Lab or Performance pages are stored as lab traffic and do not trigger them (A-043).
- **Alert drill:** press *Run alert drill* on the Muhasaba page, or call `POST /v1/alerts/drill {"seconds": 120}`. Every rule reports firing for two minutes, so notification paths can be tested.
- **Retention:** conversations, replies, feedback, audits and the retrieval memory older than `storage.retention_days` (default 90) are deleted daily (SR-004).
- **Restarts:** evaluation runs interrupted by a restart resume automatically, keeping their completed items. You can also resume one with the Resume button or `POST /v1/eval/runs/{id}/resume`.
- **Shared hosts:** llama-server, vLLM and the gateway compete for the same cores. Run one backend set at a time, and check `uptime` before benchmarking.
- **Latency:** `GET /v1/perf/stages?hours=24` (or the Performance page) attributes wall time to stages and reports prefill and decode speed and KV-cache reuse per model stage. A falling decode rate with an unchanged cache-reuse share usually means CPU contention from other processes.
- **Speed changes:** adopt a serving flag or weight format only after an interleaved A/B (NFR-031); `scripts/perf_tuning.sh` and `scripts/early_escalation_study.py` record theirs. Early escalation is tuned with `gate.early_escalation_tokens` and `gate.early_escalation_margin` (0 tokens turns it off). The adaptive reasoning exit (`[deliberation.adaptive_exit]`) shortens System 2 reasoning by probing for the answer and verifying it; it needs llama-server (vLLM uses the standard path) and can be switched off per request with `"adaptive_exit": false`. EAGLE-3 heads are opt-in (`EAGLE=1`) and were not adopted on the reference CPU; re-measure with `scripts/eagle3_ab.py --system 1|2` on another host.

## 3. Alerts and what to do

| Alert | Likely cause | Action |
|---|---|---|
| Fast-path latency high | host overloaded, too many concurrent sessions, model swapped out | check `uptime` and memory pressure; reduce concurrency; NFR-007 is documented as failing above 1–2 sessions on a 4-core CPU |
| Error rate high | a backend down or timing out | `scripts/serve_llama.sh status`; restart the failing server; read `logs/system*.log` |
| Escalation rate high | gate too cautious (λ low), harder traffic, uncalibrated rule | review the Lab insights; raise λ or switch to budget mode once a calibration is promoted |
| Verifier acceptance low | model regression, prompt change, broken tests | run the regression suite; compare with the baseline run |
| Calibration drift | the traffic or the model changed | recalibrate (Muhasaba page); confirm on a fresh evaluation run |
| Degraded mode | System 2 unavailable | restart System 2; answers are labelled unverified meanwhile |

## 4. Changes and releases

1. Every model, adapter, template or configuration change runs the regression suite: `regression_500`, with fast_only for System 1 changes (ER-008).
2. Model promotion: `python3 training/tazkiya.py promote …`, then restart serving. Rollback: `python3 training/tazkiya.py rollback`, then restart (OR-005).
3. Release archive: `scripts/package_release.sh`, which produces binaries, schemas, the model manifest, deployment recipes and SHA256SUMS (OR-002).
4. Security: `python3 -I scripts/security_tests.py` must report 0 open high-severity findings (gate G5).
5. The gates dashboard (Lab page, `GET /v1/gates`) shows G0–G6 from stored evidence.

## 5. Backup and recovery

- **State.** All state is in `data/runtime/`:
  - `nafs.sqlite` holds conversations, replies, decisions, audits, runs and feedback;
  - `traces/trace.jsonl` is the append-only trace;
  - `gate_calibration.json` and `gate_params.json` hold the gate's calibration and parameters;
  - `verifier_rates.json` holds the measured verifier rates.
- **Backup.** Back up with `sqlite3 data/runtime/nafs.sqlite ".backup backup.sqlite"` while running; WAL mode makes this safe.
- **Restore.** Stop the gateway, replace the files, and start it again.
- **Models.** Model files are reproducible from `config/models.manifest.json`, which pins the revision and SHA-256 of each file.
