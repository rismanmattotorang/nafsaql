//! The code agent (FR-093) bound to Nafs-Aql's backends and verifier sandbox. The loop is
//! `nafs_harness::agent` (Codex's turn loop with Open Math Harness's stop rule); this module supplies the
//! models, the sandboxed Python runner and the exact verifier.
//!
//! Phase 14: the cascade. The agent's first step can go to System 1 with several candidates in one `n`-choice
//! request ([`Backend::stream_n`]: the prompt is prefilled once, the choices decode in one batch), each judged by
//! the verifier; System 2 continues only when none passes. A failed model call is retried with Codex's backoff.
use crate::backend::{Backend, GenRequest};
use crate::orchestrator::{outcome_of, Orchestrator};
use futures::future::BoxFuture;
use nafs_harness::agent::{
    self, AgentConfig, AgentEnv, ContextMode, EditFormat, Reply, Stage, Turn, Usage, Verdict,
};
use nafs_harness::patch::Workspace;
use nafs_harness::truncate::format_exec_output;
use nafs_types::*;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};
use uuid::Uuid;

struct Env<'a> {
    orch: &'a Orchestrator,
    /// The run's model (`harness.model`).
    model: Arc<Backend>,
    thinking: bool,
    /// System 1, for the cascade's first step.
    cheap: Arc<Backend>,
    cheap_temperature: f32,
    thinking_budget: u32,
    /// Reasoning tokens on the first step (0 = `thinking_budget`).
    first_budget: u32,
    max_tokens: u32,
    inputs: VerificationInputs,
    calls: u64,
    /// Cut the stream after the first complete action.
    cut: bool,
}

impl Env<'_> {
    /// The request for `turn`: System 1 without reasoning on a cheap step, the run's model otherwise.
    fn request(
        &mut self,
        messages: Vec<(String, String)>,
        turn: &Turn,
    ) -> (Arc<Backend>, GenRequest, bool) {
        self.calls += 1;
        let (model, thinking) = match turn.stage {
            Stage::Cheap => (self.cheap.clone(), false),
            Stage::Main => (self.model.clone(), self.thinking),
        };
        let grammar = turn.grammar(thinking).filter(|_| model.supports_grammar());
        let constrained = grammar.is_some();
        let budget = if turn.step == 1 && self.first_budget > 0 {
            self.first_budget
        } else {
            self.thinking_budget
        };
        let temperature = match turn.stage {
            // Candidates must differ from each other: the rescue's temperature (0.7) rather than System 2's.
            Stage::Cheap => self.cheap_temperature,
            // A model that keeps reproducing judged files is sampled more diversely (0.3 per repeat, to 1.0).
            Stage::Main => (self.orch.cfg.deliberation.slow_temperature
                + 0.3 * turn.repeats as f32)
                .min(1.0_f32.max(self.orch.cfg.deliberation.slow_temperature)),
        };
        let req = GenRequest {
            memory_embeds: None,
            memory_placeholder: None,
            messages: messages
                .into_iter()
                .map(|(role, content)| Message { role, content })
                .collect(),
            max_tokens: self.max_tokens + if thinking { budget } else { 0 },
            temperature,
            top_p: 0.95,
            logprobs: false,
            thinking: Some(thinking),
            thinking_budget: thinking.then_some(budget),
            seed: Some(self.calls),
            cancel: None,
            response_format: None,
            early_stop: None,
            // Codex dispatches a tool call when its item is done, not when the response ends; here the
            // stream ends there. Under the grammar the server already stops at the end of the action, and
            // a cut would lose the final chunk that carries its timings.
            stop_after: (self.cut && grammar.is_none())
                .then_some(agent::action_end as fn(&str) -> Option<usize>),
            grammar,
        };
        (model, req, constrained)
    }
}

fn reply_of(out: crate::backend::GenOutput, constrained: bool) -> Reply {
    Reply {
        usage: Usage {
            prompt_tokens: out.server_prompt_n,
            cached_tokens: out.server_cache_n,
            prefill_ms: out.server_prefill_ms,
            decode_ms: out.server_decode_ms,
            completion_tokens: out.completion_tokens,
            reasoning_tokens: out.reasoning_tokens,
            finish_reason: out.finish_reason.clone(),
            constrained,
        },
        text: out.content,
    }
}

/// The run's cap holds inside a call too: dropping the stream stops generation on the server (the Phase 10
/// live run overran a 900 s cap by one 1,041 s call).
async fn capped<T>(
    deadline: Option<Instant>,
    call: impl std::future::Future<Output = nafs_types::Result<T>>,
) -> std::result::Result<T, String> {
    match deadline {
        Some(d) => tokio::time::timeout_at(d.into(), call).await.map_err(|_| {
            "the run's wall-clock cap was reached during the model call".to_string()
        })?,
        None => call.await,
    }
    .map_err(|e| e.to_string())
}

/// Compiles `candidate.py` and prints `NAFS_LINT_OK`, or `NAFS_LINT_ERR` with the error and the lines around it
/// (what SWE-agent's flake8 guardrail shows the agent). It never executes the candidate.
const LINT_RUNNER: &str = r#"import sys
try:
    src = open("candidate.py", encoding="utf-8").read()
    compile(src, "candidate.py", "exec")
except SyntaxError as e:
    lines = src.splitlines()
    n = e.lineno or 1
    shown = "\n".join(
        f"{i + 1:4d}| {lines[i]}" + ("   <-- here" if i + 1 == n else "")
        for i in range(max(0, n - 3), min(len(lines), n + 2))
    )
    print("NAFS_LINT_ERR")
    print(f"{type(e).__name__}: {e.msg} (line {n})")
    print(shown)
except (ValueError, RecursionError, MemoryError) as e:
    print("NAFS_LINT_ERR")
    print(f"{type(e).__name__}: {e}")
else:
    print("NAFS_LINT_OK")
"#;

impl AgentEnv for Env<'_> {
    fn complete(
        &mut self,
        messages: Vec<(String, String)>,
        turn: &Turn,
    ) -> BoxFuture<'_, std::result::Result<Reply, String>> {
        let (model, req, constrained) = self.request(messages, turn);
        let deadline = turn.deadline;
        Box::pin(async move {
            let mut sink = |_: &str, _: bool| {};
            let out = capped(deadline, model.stream(&req, &mut sink)).await?;
            Ok(reply_of(out, constrained))
        })
    }

    /// `n` candidates in one request. The width never exceeds the server's free slots (a wider request would
    /// queue behind itself); every choice that ended is returned in index order, and the call fails only when
    /// none did.
    fn complete_n(
        &mut self,
        messages: Vec<(String, String)>,
        turn: &Turn,
        n: usize,
    ) -> BoxFuture<'_, std::result::Result<Vec<Reply>, String>> {
        if n <= 1 {
            let one = self.complete(messages, turn);
            return Box::pin(async move { Ok(vec![one.await?]) });
        }
        let (model, req, constrained) = self.request(messages, turn);
        let deadline = turn.deadline;
        Box::pin(async move {
            let n = n.min(model.free_slots()).max(1);
            if n == 1 {
                let mut sink = |_: &str, _: bool| {};
                let out = capped(deadline, model.stream(&req, &mut sink)).await?;
                return Ok(vec![reply_of(out, constrained)]);
            }
            let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
            let driver = model.stream_n(&req, n, tx);
            let r = capped(deadline, driver).await;
            let mut outs: Vec<Option<Reply>> = (0..n).map(|_| None).collect();
            while let Ok((i, out)) = rx.try_recv() {
                if let (Some(slot), Ok(o)) = (outs.get_mut(i), out) {
                    *slot = Some(reply_of(o, constrained));
                }
            }
            let replies: Vec<Reply> = outs.into_iter().flatten().collect();
            match (replies.is_empty(), r) {
                (true, Err(e)) => Err(e),
                (true, Ok(_)) => Err("no candidate ended".into()),
                _ => Ok(replies),
            }
        })
    }

    fn backoff(&mut self, attempt: u32) -> BoxFuture<'_, ()> {
        // Codex: 200 ms initial delay, factor 2, jitter 0.9-1.1 (`async-utils/src/backoff.rs`).
        let ms =
            (200u64 << attempt.saturating_sub(1).min(4)) as f64 * (0.9 + 0.2 * fastrand::f64());
        Box::pin(async move { tokio::time::sleep(Duration::from_millis(ms as u64)).await })
    }

    fn run(&mut self, ws: &Workspace, file: &str) -> BoxFuture<'_, String> {
        let files: Vec<(String, String)> = ws.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        let file = file.to_string();
        let limits = self.orch.hub.limits.clone();
        Box::pin(async move {
            if let Some((bad, _)) = files.iter().find(|(k, _)| k.contains('/')) {
                return format!("Exit code: 1\nOutput:\nfiles in subdirectories cannot be run here ({bad}); keep the workspace flat");
            }
            let refs: Vec<(&str, &str)> = files
                .iter()
                .map(|(k, v)| (k.as_str(), v.as_str()))
                .collect();
            let t = Instant::now();
            match nafs_verify::sandbox::run(&refs, &file, None, &limits.python, &limits).await {
                Ok(o) => {
                    let timed_out = o
                        .killed
                        .as_deref()
                        .is_some_and(|k| k.contains("wall-clock"));
                    let code = if o.exit_ok {
                        0
                    } else if timed_out {
                        124
                    } else {
                        1
                    };
                    format_exec_output(code, t.elapsed(), &o.excerpt(4000), timed_out, 1800)
                }
                Err(e) => format!("Exit code: 1\nOutput:\nthe sandbox could not run {file}: {e}"),
            }
        })
    }

    /// The compile check, in the sandbox like everything that reads model-written code. Trouble in the sandbox
    /// itself is not the model's fault: it passes the edit on to the verifier.
    fn lint(&mut self, ws: &Workspace, file: &str) -> BoxFuture<'_, Option<String>> {
        let source = ws.get(file).cloned();
        let limits = self.orch.hub.limits.clone();
        let file = file.to_string();
        Box::pin(async move {
            let source = source?;
            let files = [
                ("candidate.py", source.as_str()),
                ("_nafs_lint.py", LINT_RUNNER),
            ];
            let out =
                nafs_verify::sandbox::run(&files, "_nafs_lint.py", None, &limits.python, &limits)
                    .await
                    .ok()?;
            out.stdout
                .strip_prefix("NAFS_LINT_ERR\n")
                .map(|report| report.trim_end().replace("candidate.py", &file))
        })
    }

    fn verify(&mut self, ws: &Workspace) -> BoxFuture<'_, Verdict> {
        let code = ws.get("solution.py").cloned().unwrap_or_default();
        Box::pin(async move {
            let v = self
                .orch
                .hub
                .with_rates(self.orch.hub.verify_code(&code, &self.inputs).await);
            let score = outcome_of(&v).score();
            Verdict {
                passed: v.passed,
                score,
                evidence: v.evidence,
            }
        })
    }
}

impl Orchestrator {
    pub fn validate_code_agent(&self, req: &CodeAgentRequest) -> std::result::Result<(), String> {
        let h = &self.cfg.harness;
        if !h.code_agent_enabled {
            return Err(
                "the code agent is disabled in this configuration (harness.code_agent_enabled)"
                    .into(),
            );
        }
        if req.task.trim().is_empty() || req.task.len() > 20_000 {
            return Err("task must be 1..20,000 characters".into());
        }
        if !req.verification.has_code_tests() {
            return Err(
                "verification must hold code tests (tests, or test_code with entry_point)".into(),
            );
        }
        if let Some(e) = req.verification.validation_error() {
            return Err(e);
        }
        if req.files.len() > 16 || req.files.values().map(String::len).sum::<usize>() > 200_000 {
            return Err("files: at most 16 files and 200,000 characters".into());
        }
        if let Some(bad) = req
            .files
            .keys()
            .find(|k| k.is_empty() || k.contains('/') || k.contains("..") || k.starts_with('.'))
        {
            return Err(format!(
                "files: '{}' must be a flat file name",
                bad.chars().take(40).collect::<String>()
            ));
        }
        if req.max_steps.is_some_and(|s| !(1..=12).contains(&s)) {
            return Err("max_steps must be in [1, 12]".into());
        }
        if req
            .model
            .as_deref()
            .is_some_and(|m| m != "system1" && m != "system2")
        {
            return Err("model must be \"system1\" or \"system2\"".into());
        }
        if req
            .context
            .as_deref()
            .is_some_and(|c| c != "append" && c != "reset")
        {
            return Err("context must be \"append\" or \"reset\"".into());
        }
        if req
            .edit_format
            .as_deref()
            .is_some_and(|f| !matches!(f, "auto" | "any" | "whole"))
        {
            return Err("edit_format must be \"auto\", \"any\" or \"whole\"".into());
        }
        if req
            .thinking_budget
            .is_some_and(|b| !(64..=4096).contains(&b))
        {
            return Err("thinking_budget must be in [64, 4096]".into());
        }
        if req
            .first_step_thinking_budget
            .is_some_and(|b| b != 0 && !(64..=4096).contains(&b))
        {
            return Err("first_step_thinking_budget must be 0 or in [64, 4096]".into());
        }
        if req.cascade_samples.is_some_and(|s| !(1..=4).contains(&s)) {
            return Err("cascade_samples must be in [1, 4]".into());
        }
        if req
            .stuck_after
            .is_some_and(|n| n != 0 && !(2..=12).contains(&n))
        {
            return Err("stuck_after must be 0 (never) or in [2, 12]".into());
        }
        if req.prune_over.is_some_and(|n| !(200..=4_000).contains(&n)) {
            return Err("prune_over must be in [200, 4000]".into());
        }
        if req
            .compact_at_tokens
            .is_some_and(|n| !(400..=20_000).contains(&n))
        {
            return Err("compact_at_tokens must be in [400, 20000]".into());
        }
        if req
            .guidance
            .as_deref()
            .is_some_and(|g| g.chars().count() > 2_000)
        {
            return Err("guidance must be at most 2,000 characters".into());
        }
        Ok(())
    }

    /// Run the code agent to completion (FR-093). Counts as an in-flight request, so the audit worker and
    /// the cache warm-up stay out of its way.
    pub async fn run_code_agent(
        &self,
        req: CodeAgentRequest,
    ) -> std::result::Result<CodeAgentReply, String> {
        self.validate_code_agent(&req)?;
        self.inflight.fetch_add(1, Ordering::SeqCst);
        struct Done<'a>(&'a Orchestrator);
        impl Drop for Done<'_> {
            fn drop(&mut self) {
                self.0.inflight.fetch_sub(1, Ordering::SeqCst);
            }
        }
        let _done = Done(self);
        let t0 = Instant::now();
        let h = &self.cfg.harness;
        let model = req.model.clone().unwrap_or_else(|| h.model.clone());
        let pair = self.active();
        let (backend, thinking) = if model == "system1" {
            (pair.s1.clone(), false)
        } else {
            (pair.s2.clone(), true)
        };
        // The cascade only makes sense in front of System 2: a run asked for on System 1 is already cheap.
        let cascade = model != "system1" && req.cascade.unwrap_or(h.cascade);
        let cascade_samples = if cascade {
            req.cascade_samples.unwrap_or(h.cascade_samples).clamp(1, 4) as usize
        } else {
            0
        };
        let context = match req.context.as_deref() {
            Some("reset") => ContextMode::Reset,
            Some(_) => ContextMode::Append,
            None => h.context,
        };
        let constrain = req.constrain.unwrap_or(h.constrain);
        let edit_format = match req.edit_format.as_deref() {
            Some("any") => EditFormat::Any,
            Some("whole") => EditFormat::Whole,
            Some(_) => EditFormat::Auto,
            None => h.edit_format,
        };
        let mut env = Env {
            orch: self,
            model: backend.clone(),
            thinking,
            cheap: pair.s1.clone(),
            cheap_temperature: h.cascade_temperature,
            thinking_budget: req.thinking_budget.unwrap_or(h.thinking_budget),
            first_budget: req
                .first_step_thinking_budget
                .unwrap_or(h.first_step_thinking_budget),
            max_tokens: h.step_max_tokens,
            inputs: req.verification.clone(),
            calls: 0,
            cut: req.stream_cut.unwrap_or(h.stream_cut),
        };
        let cfg = AgentConfig {
            max_steps: req.max_steps.unwrap_or(h.max_steps),
            context,
            keep_exchanges: h.keep_exchanges,
            compact_at_tokens: req
                .compact_at_tokens
                .map_or(h.compact_at_tokens, |n| n as usize),
            prune: req.prune.unwrap_or(true),
            prune_over: req
                .prune_over
                .map_or(nafs_harness::context::PRUNE_OVER, |n| n as usize),
            constrain,
            notes: h.notes,
            edit_format,
            whole_max_lines: h.whole_max_lines,
            unstick: req.unstick.unwrap_or(h.unstick),
            cascade_samples,
            retries: h.retries,
            lint: req.lint.unwrap_or(h.lint),
            rollback: req.rollback.unwrap_or(h.rollback),
            stuck_after: req.stuck_after.unwrap_or(h.stuck_after),
            guidance: req.guidance.clone(),
            deadline: Some(t0 + Duration::from_secs(h.wall_clock_s)),
            ..AgentConfig::default()
        };
        let out = agent::run_agent(&mut env, &req.task, req.files.clone(), &cfg).await;
        let passed = out.status == agent::Status::Verified;
        let status = serde_json::to_value(&out.status)
            .ok()
            .and_then(|v| v.as_str().map(String::from))
            .unwrap_or_default();
        metrics::counter!("nafs_code_agent_runs_total", "status" => status.clone()).increment(1);
        if cascade_samples > 0 {
            let first = out.steps.first();
            let result = match first.and_then(|s| s.verdict.as_ref()) {
                Some(v) if v.passed => "verified",
                Some(_) => "continued",
                None => "none",
            };
            metrics::counter!("nafs_code_agent_cascade_total", "result" => result).increment(1);
        }
        let usages = || out.steps.iter().filter_map(|s| s.usage.as_ref());
        let prefill_tokens = usages()
            .filter_map(|u| u.prompt_tokens)
            .map(u64::from)
            .sum();
        let cached_tokens = usages()
            .filter_map(|u| u.cached_tokens)
            .map(u64::from)
            .sum();
        let decoded_tokens = usages()
            .map(|u| u64::from(u.completion_tokens.max(u.reasoning_tokens)))
            .sum();
        metrics::counter!("nafs_code_agent_lint_rejections_total")
            .increment(u64::from(out.lint_rejections));
        metrics::counter!("nafs_code_agent_rollbacks_total").increment(u64::from(out.rollbacks));
        metrics::counter!("nafs_code_agent_prefill_tokens_total").increment(prefill_tokens);
        metrics::counter!("nafs_code_agent_cached_tokens_total").increment(cached_tokens);
        Ok(CodeAgentReply {
            run_id: Uuid::new_v4(),
            status,
            assurance: if passed {
                Assurance::Verified
            } else {
                Assurance::Unverified
            },
            files: out.workspace,
            passed,
            score: out.verdict.as_ref().map_or(0.0, |v| v.score),
            evidence: out.verdict.and_then(|v| v.evidence),
            steps: out
                .steps
                .into_iter()
                .map(|s| CodeAgentStep {
                    step: s.step,
                    action: s.action,
                    observation: s.observation,
                    passed: s.verdict.as_ref().map(|v| v.passed),
                    score: s.verdict.as_ref().map(|v| v.score),
                    ms: s.ms,
                    usage: s.usage.map(|u| CodeAgentUsage {
                        prompt_tokens: u.prompt_tokens,
                        cached_tokens: u.cached_tokens,
                        prefill_ms: u.prefill_ms,
                        decode_ms: u.decode_ms,
                        completion_tokens: u.completion_tokens,
                        reasoning_tokens: u.reasoning_tokens,
                        finish_reason: u.finish_reason,
                        constrained: u.constrained,
                    }),
                    compacted: s.compacted,
                    pruned: s.pruned,
                    model: match s.stage {
                        Stage::Cheap => "system1".into(),
                        Stage::Main => model.clone(),
                    },
                    samples: s.samples,
                    retries: s.retries,
                    rolled_back: s.rolled_back,
                })
                .collect(),
            model_calls: out.model_calls,
            model,
            wall_ms: t0.elapsed().as_millis() as u64,
            context: match context {
                ContextMode::Append => "append",
                ContextMode::Reset => "reset",
            }
            .into(),
            constrained: constrain && backend.supports_grammar(),
            edit_format: serde_json::to_value(edit_format)
                .ok()
                .and_then(|v| v.as_str().map(String::from))
                .unwrap_or_default(),
            verifier_calls: out.verifier_calls,
            compactions: out.compactions,
            prefill_tokens,
            cached_tokens,
            decoded_tokens,
            diff: out.diff,
            cascade: cascade_samples > 0,
            cascade_samples: cascade_samples as u32,
            lint_rejections: out.lint_rejections,
            rollbacks: out.rollbacks,
        })
    }
}
