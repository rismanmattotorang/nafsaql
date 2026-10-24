//! The code agent (FR-093) bound to Nafs-Aql's backends and verifier sandbox. The loop is
//! `nafs_harness::agent` (Codex's turn loop with Open Math Harness's stop rule); this module supplies the
//! model, the sandboxed Python runner and the exact verifier.
use crate::backend::{Backend, GenRequest};
use crate::orchestrator::{outcome_of, Orchestrator};
use futures::future::BoxFuture;
use nafs_harness::agent::{self, AgentConfig, AgentEnv, Verdict};
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
    max_tokens: u32,
    inputs: VerificationInputs,
    calls: u64,
}

impl AgentEnv for Env<'_> {
    fn complete(
        &mut self,
        messages: Vec<(String, String)>,
    ) -> BoxFuture<'_, std::result::Result<String, String>> {
        self.calls += 1;
        let req = GenRequest {
            messages: messages
                .into_iter()
                .map(|(role, content)| Message { role, content })
                .collect(),
            max_tokens: self.max_tokens
                + if self.thinking {
                    self.thinking_budget
                } else {
                    0
                },
            temperature: self.orch.cfg.deliberation.slow_temperature,
            top_p: 0.95,
            logprobs: false,
            thinking: Some(self.thinking),
            thinking_budget: self.thinking.then_some(self.thinking_budget),
            seed: Some(self.calls),
            cancel: None,
            response_format: None,
            early_stop: None,
        };
        let model = self.model.clone();
        Box::pin(async move {
            let mut sink = |_: &str, _: bool| {};
            model
                .stream(&req, &mut sink)
                .await
                .map(|o| o.content)
                .map_err(|e| e.to_string())
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
        let mut env = Env {
            orch: self,
            model: backend,
            thinking,
            thinking_budget: h.thinking_budget,
            max_tokens: h.step_max_tokens,
            inputs: req.verification.clone(),
            calls: 0,
        };
        let cfg = AgentConfig {
            max_steps: req.max_steps.unwrap_or(h.max_steps),
            keep_exchanges: h.keep_exchanges,
            deadline: Some(t0 + Duration::from_secs(h.wall_clock_s)),
            ..AgentConfig::default()
        };
        let out = agent::run_agent(&mut env, &req.task, req.files.clone(), &cfg).await;
        let passed = out.status == agent::Status::Verified;
        metrics::counter!("nafs_code_agent_runs_total", "status" => format!("{:?}", out.status).to_lowercase()).increment(1);
        Ok(CodeAgentReply {
            run_id: Uuid::new_v4(),
            status: serde_json::to_value(&out.status)
                .ok()
                .and_then(|v| v.as_str().map(String::from))
                .unwrap_or_default(),
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
                })
                .collect(),
            model_calls: out.model_calls,
            model,
            wall_ms: t0.elapsed().as_millis() as u64,
        })
    }
}
