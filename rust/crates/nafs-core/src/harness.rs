//! The code agent (FR-093) bound to Nafs-Aql's backends and verifier sandbox. The loop is
//! `nafs_harness::agent` (Codex's turn loop with Open Math Harness's stop rule); this module supplies the
//! model, the sandboxed Python runner and the exact verifier.
use crate::backend::{Backend, GenRequest};
use crate::orchestrator::{outcome_of, Orchestrator};
use futures::future::BoxFuture;
use nafs_harness::agent::{
    self, AgentConfig, AgentEnv, ContextMode, EditFormat, Reply, Turn, Usage, Verdict,
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
    model: Arc<Backend>,
    thinking: bool,
    thinking_budget: u32,
    /// Reasoning tokens on the first step (0 = `thinking_budget`).
    first_budget: u32,
    max_tokens: u32,
    inputs: VerificationInputs,
    calls: u64,
    /// Cut the stream after the first complete action.
    cut: bool,
}

impl AgentEnv for Env<'_> {
    fn complete(
        &mut self,
        messages: Vec<(String, String)>,
        turn: &Turn,
    ) -> BoxFuture<'_, std::result::Result<Reply, String>> {
        self.calls += 1;
        let grammar = turn
            .grammar(self.thinking)
            .filter(|_| self.model.supports_grammar());
        let constrained = grammar.is_some();
        let budget = if turn.step == 1 && self.first_budget > 0 {
            self.first_budget
        } else {
            self.thinking_budget
        };
        let req = GenRequest {
            messages: messages
                .into_iter()
                .map(|(role, content)| Message { role, content })
                .collect(),
            max_tokens: self.max_tokens + if self.thinking { budget } else { 0 },
            // A model that keeps reproducing judged files is sampled more diversely (0.3 per repeat, to 1.0).
            temperature: (self.orch.cfg.deliberation.slow_temperature + 0.3 * turn.repeats as f32)
                .min(1.0_f32.max(self.orch.cfg.deliberation.slow_temperature)),
            top_p: 0.95,
            logprobs: false,
            thinking: Some(self.thinking),
            thinking_budget: self.thinking.then_some(budget),
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
        let model = self.model.clone();
        let deadline = turn.deadline;
        Box::pin(async move {
            let mut sink = |_: &str, _: bool| {};
            let call = model.stream(&req, &mut sink);
            // The run's cap holds inside a call too: dropping the stream stops generation on the server
            // (the Phase 10 live run overran a 900 s cap by one 1,041 s call).
            let out = match deadline {
                Some(d) => tokio::time::timeout_at(d.into(), call).await.map_err(|_| {
                    "the run's wall-clock cap was reached during the model call".to_string()
                })?,
                None => call.await,
            }
            .map_err(|e| e.to_string())?;
            Ok(Reply {
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
            })
        })
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
            compact_at_tokens: h.compact_at_tokens,
            constrain,
            notes: h.notes,
            edit_format,
            whole_max_lines: h.whole_max_lines,
            unstick: req.unstick.unwrap_or(h.unstick),
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
        })
    }
}
