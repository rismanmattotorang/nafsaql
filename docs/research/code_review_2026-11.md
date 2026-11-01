# End-to-end code review, November 2026

Date 2026-11-01. Scope: every Rust crate of the workspace (except the vendored engine's internals), the scripts, the deployment
files and CI. Method: automated signals (clippy with the stricter lints, `cargo audit`, `cargo deny`), then four independent
read-only reviews (orchestrator and backends; gateway, web UI and the Smiley server; harness, patch parser and sandbox;
memory, evaluation, bench, types and the operational surface). Each finding was checked against its callers before a fix; a
claim that could not be substantiated was dropped. This note lists what was fixed, with the test that pins it where there is
one, and what was deliberately left, with the reason.

## Signals

- `cargo audit`: no vulnerable dependency (271 locked crates).
- `cargo deny`: there was no policy. `rust/deny.toml` now allows permissive licenses only, denies unknown registries and git
  sources and wildcard dependencies, and CI runs it (a new `supply-chain` job).
- About 150 production `unwrap()` calls, of which about 110 were `Mutex`/`RwLock` acquisitions (a poisoned lock turns one
  panic into a panic in every later request).

## Fixed

| Area | Defect | Fix |
|---|---|---|
| Prompt safety (SR-001) | `delimit()` found the closing tag in a Unicode-lowercased copy and sliced the original with those offsets: a user message such as twelve `İ` before `</verifier_evidence` panicked or let the real tag through and closed the data block | ASCII lowercasing (offsets preserved); regression test `delimit_neutralises_a_closing_tag_...` |
| Availability | `lock().unwrap()` on about 110 sites | `LockExt` / `RwLockExt` in `nafs-types` and `Store::db()` recover a poisoned guard |
| Availability | `param_provenance` held a read guard while calling methods that take the same lock (deadlock against a queued writer) | works on a snapshot |
| Availability | a patch that deleted `solution.py` panicked the request (`ws["solution.py"]`) | `get`, reported as a failed patch |
| Availability | a bench run with `duration_s = u64::MAX` panicked its task; the run stayed "running" and its cancel flag leaked; the same was true of every background run that panicked | `RunGuard` (flag removed, unfinished run marked failed); limits on `duration_s` and `requests` |
| Data lifecycle | `DELETE /v1/conversations/{id}` removed the conversation and messages only; replies, decisions, audits, feedback, reviews and the retrieval memory stayed (and stayed retrievable, bypassing the training opt-out); it also accepted a room's conversation and orphaned the room | `delete_conversation` purges everything (`purge_conversation_data`) and refuses a room's conversation; test `deleting_a_conversation_removes_its_ledger_rows_...` |
| Consistency | `save_reply` and `append_message` wrote two tables without a transaction; `purge_older_than` ran eight deletes unguarded and overflowed on a huge `days` | transactions; clamp |
| Consistency | `set_gate_state` assigned the mode and then rejected the budget (state changed by a 400) | validate, then assign |
| Consistency | a failed room delete left the room cancelled and tombstoned | flags restored on error |
| Consistency | a WebSocket that joined just after a room's deletion was never told | `is_deleted()` checked after joining |
| Robustness | an audit the backend rejected (context overflow) stayed `pending` and was retried forever, blocking all later audits | closed as `failed` (transient backend errors are still retried) |
| Robustness | a storage error after a 900 s generation discarded the answer | logged, answer returned |
| Robustness | the candidate calibration path used `replace(".json", ...)` (a path without the suffix overwrote the active file) | `Path::with_extension` |
| Security | the bearer token was accepted from the URL on every route (it lands in proxy access logs and history) | only on `/ws/`, where a browser cannot send a header |
| Security | token comparison was not constant-time, and failed authentication left no trace | constant-time scan over all tokens; a warning without the token |
| Security | CSP lacked `base-uri` and `form-action` | added (`connect-src` still allows `ws:`/`wss:` for older Safari) |
| Security | review and preference notes were stored up to the 2 MB body limit; a review could name a reply that does not exist | clipped to 1,000 characters; 404 for a missing reply |
| Security | Lean: `theorem` was pasted into commands unvalidated (a newline injected a `#eval`); `include_str`/`include_bytes` read files into the error text | identifier check (`is_lean_identifier`, tested); both words banned (tested) |
| Security | sandbox scratch directories were created 0755, cleaned up only on normal completion, and leaked when the model `chmod 000`'d inside | `Scratch`: mode 0700, removal on drop (also on cancellation) after restoring permissions |
| Privacy | the SQLite file and the trace were created with the default umask | 0600 after open; `UMask=0077` in the systemd unit |
| Smiley server | abandoned requests still ran a full prefill; the wait queue was unbounded; the `requests_processing` counter leaked on a panic | `is_closed()` check after taking the session, a bounded queue (503), a drop guard |
| Rooms | the per-connection reply channel was unbounded; the queue cap could be overshot by concurrent members | bounded channel (frames are advisory); `fetch_update` reserves the slot |
| Performance | the dashboards and insights did an N+1 SQLite read and bootstrap resampling on the async workers | moved into `spawn_blocking` (`blocking()` helper); one `eval_runs_full` |
| Performance | no indexes for the purge and dashboard lookups | 7 indexes; `busy_timeout` |
| Operations | `gateway.sh stop` ran `pkill -f` on a path fragment (any matching process on the host) | signals the pid-file process, then the listener on :8080, only if it is a gateway |
| Operations | `verify_deploy.sh` could leave a publicly known token file behind | `try/finally` |
| Operations | the release package omitted the default engine | `nafs-smiley-server` and `serve_smiley.sh` included |
| Supply chain | CI had no `permissions`, mutable action tags, no `--locked`, unpinned pip packages, no dependency policy | read-only token, actions pinned to commits, `--locked`, pinned packages, `cargo deny` job |
| Hygiene | unused dependencies and features (`tracing`, `uuid`, `fastrand`, `tower`, axum `macros`, tower-http `trace/cors/timeout`); crates publishable by accident | removed; `publish = false` everywhere |
| Readability | four copies of the background-run boilerplate, eight `spawn_blocking(...).map_err(...)`, dead `let _ = k;` / `let _ = evidence;`, a doc comment on the wrong function | `RunGuard`, `progress_saver`, `blocking`; removed; moved |

## Left, with the reason

These are real, and each needs more than a mechanical change or a platform that is not available here.

- **Linux sandbox read surface** (`linux_mountns.sh` hides a fixed denylist; model code can read `/app`, `/opt`, `/var/lib`, and
  unix sockets under `/run`): the fix is an allowlist root or a longer denylist, and it must be verified on Linux (the
  scratch directory itself may live under a path that would then be hidden). Not changed without a Linux host to test on.
- **Lean on Linux** runs with `unshare -rn` only (no mount or PID namespace, no seccomp). Same reason; the new word bans
  and the theorem check close the two concrete read paths found.
- **macOS seatbelt**: `process-info*` and `mach-lookup` are not denied (process arguments of other same-user processes may be
  readable). Needs a probe on macOS and a profile change that the sandbox tests then validate.
- **Verifier nonce** lives in the runner's globals, so adversarial candidate code can forge a passing line. A real fix moves
  the checker out of the candidate's process (a design change); the exact verifier remains sound against honest and
  merely wrong code, which is what the measurements use.
- **Authorization** is all-or-nothing (every token is an operator; rooms have no owner). A single-operator deployment is the
  documented model; multi-user use needs scopes and owners.
- **`/metrics` and `/healthz` are unauthenticated** (the sample nginx config blocks `/metrics`); documented, to be reached only
  through the proxy.
- **`audit: true` in a chat request** is a documented operator feature (FR-040) that a token holder can also use, which biases
  the self-model's audit sample; gate it behind a role when roles exist.
- **Unbounded growth by clients** (conversation count, history loaded per turn, retrieval index scan per request when
  retrieval is enabled, feedback rows): needs limits and an FTS index; retrieval is off by default.
- **Patch semantics** (a hunk of only added lines ignores its `@@` context and appends at EOF; whitespace-trimmed matching of
  Python lines; `Add File` over an existing path): each changes what a model's patch means and needs its own measured test.
- **Evaluation statistics** (errored items dropped instead of counted wrong; non-inferiority flags emitted at tiny n): additive
  fixes that change reported numbers, to be done with a re-run of the stored evaluations.
- **Docker compose and systemd hardening** beyond the directives added (`cap_drop`, `read_only`, `pids_limit`, narrower
  `ReadWritePaths`, `RestrictAddressFamilies` on the engine units): cannot be exercised here.
- **Model downloads** (`fetch_models.sh`) do not enforce the revision and SHA-256 that `config/models.manifest.json` records.
- **Function size**: `handle_with_cancel` (about 1,450 lines) and `deliberate_adaptive` (about 330) in `orchestrator.rs`, and
  `run_agent` (about 310) in `agent.rs`, are the largest readability debts. The review found the seams (context assembly,
  rescue, candidate launch and finish, tree scoring, judge branch, persistence; for the agent: turn building, call with
  retries, candidate pick, observation annotation, step recording) and replacing the `Candidate { out, err }` pair with a
  `Result`. They are refactors of the hottest, most intricate code, with many captured locals; they are best done one seam
  at a time, each with the existing integration tests (`tests/orchestrator.rs`, 40+) as the net.

## Verification

`cargo fmt --check`, `cargo clippy --workspace --release --all-targets -D warnings`, `cargo test --workspace --release`,
`cargo deny check`, and the live suites on the final build (contract, security, Rooms, end to end, harness sweep, MVP
acceptance); the results are in `docs/roadmap/STATUS.md`.
