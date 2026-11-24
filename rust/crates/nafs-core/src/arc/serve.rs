//! The reasoning cache in the request path (Phase 38): lookup, extraction and run before the models; the reply when a
//! trusted entry answers; the comparison when a candidate ran beside the models; admission of System 2 answers to the
//! build queue; the idle-time builder.

use super::*;
use crate::orchestrator::{Emitter, Orchestrator};
use nafs_types::{
    Assurance, ChatReply, ChatRequest, DecisionRecord, Domain, GateFeatures, GateRule,
    ReasoningMode, StageTiming, StreamEvent, TraceRecord, TraceState, Usage, VerificationInputs,
    VerifierResult,
};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use uuid::Uuid;

/// What the cache did for one question.
pub enum ArcTry {
    Miss(String),
    /// A trusted entry answered (serve it).
    Serve(Hit, Outcome),
    /// A candidate answered: run the models too and compare.
    Shadow(Hit, Outcome),
}

impl ArcTry {
    pub fn note(&self) -> String {
        match self {
            ArcTry::Miss(why) => format!("miss: {why}"),
            ArcTry::Serve(h, o) | ArcTry::Shadow(h, o) => format!(
                "{} entry {} ({:?}, similarity {:.3}) answered {} in {} ms",
                if matches!(self, ArcTry::Serve(..)) {
                    "served:"
                } else {
                    "shadow:"
                },
                h.entry.id,
                h.tier,
                h.similarity,
                o.answer,
                o.ms
            ),
        }
    }
}

/// Whether this request may use the cache at all: the kind of question is cached, nothing the code cannot see (a
/// knowledge base, tests, a schema, a proof, memory tokens) shapes the answer, and the question stands on its own.
/// Template and shape lookups need no model and match only the stored problem itself, so they run whatever kind the
/// question was classified as (a perimeter question can be read as "open"); `eligible_for_meaning` adds the domain rule
/// for the lookup by meaning and for admission.
pub fn eligible(cfg: &crate::config::Config, req: &ChatRequest, v: &VerificationInputs) -> bool {
    cfg.arc.enabled
        && req.arc != Some(false)
        && req
            .knowledge
            .as_ref()
            .map(|k| k.mode.as_deref() == Some("off"))
            .unwrap_or(true)
        && v.lean.is_none()
        && v.tests.is_none()
        && v.test_code.is_none()
        && v.json_schema.is_none()
        && req.force_mode != Some(ReasoningMode::Slow)
}

/// Whether the meaning lookup (and building from the answer) applies: the kind of question is cached by meaning, and
/// the question does not lean on the turn before it (MeanCache's point: "and for 12?" depends on what the code does not
/// see). A template or shape match needs neither: it is the whole of a stored, self-contained question.
pub fn eligible_for_meaning(
    cfg: &crate::config::Config,
    dom: Domain,
    question: &str,
    has_history: bool,
) -> bool {
    cfg.arc.domains.iter().any(|d| d == dom.as_str())
        && !(has_history && crate::continuity::leans_on_previous(question))
}

fn content_of(dom: Domain, o: &Outcome) -> String {
    let body = o.summary.clone().unwrap_or_else(|| o.answer.clone());
    if dom == Domain::Math && !body.contains("\\boxed") {
        format!("{body}\n\n**Answer:** $\\boxed{{{}}}$", o.answer)
    } else {
        body
    }
}

fn reasoning_of(h: &Hit, o: &Outcome) -> String {
    format!(
        "Worked out by the reasoning cache: entry {} was built from an earlier answer of the reasoning model and has agreed with the models {} times.\n{}",
        h.entry.id,
        h.entry.stats.agreements,
        o.narration.join("\n")
    )
}

impl Orchestrator {
    fn arc_embedder(&self) -> Option<Arc<dyn crate::knowledge::Knowledge>> {
        self.knowledge
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Entry search, extraction and run. `honour_trust` false (the try endpoint) runs any live entry.
    pub async fn arc_try(
        &self,
        question: &str,
        honour_trust: bool,
        cancel: Option<CancelFlag>,
    ) -> ArcTry {
        self.arc_try_at(question, honour_trust, cancel, None, true)
            .await
    }

    /// As [`Self::arc_try`]; `floor` replaces every entry's meaning-tier floor (the try endpoint, for measurement), and
    /// `meaning` false keeps to the two lookups that need no model (the question's kind is not one cached by meaning).
    pub async fn arc_try_at(
        &self,
        question: &str,
        honour_trust: bool,
        cancel: Option<CancelFlag>,
        floor: Option<f32>,
        meaning: bool,
    ) -> ArcTry {
        let Some(store) = self.arc.clone() else {
            return ArcTry::Miss("the reasoning cache is off".into());
        };
        if store.is_empty() {
            return ArcTry::Miss("the cache is empty".into());
        }
        let cfg = &self.cfg.arc;
        let mut hit = store.lookup_text(question);
        let mut nearest = None;
        if hit.is_none() && !meaning {
            return ArcTry::Miss("no template or shape matches (the meaning lookup is for the listed kinds of question)".into());
        }
        if hit.is_none() {
            if let Some(k) = self.arc_embedder() {
                if let Some(qv) = k.embed(&normalize(question)).await {
                    hit = store.lookup_vector_at(&qv, cfg, floor);
                    nearest = store.nearest(&qv);
                }
            }
        }
        let Some(mut hit) = hit else {
            return ArcTry::Miss(match nearest {
                Some((id, s)) => format!("no entry is close enough (nearest {id} at {s:.3})"),
                None => "no entry is close".into(),
            });
        };
        let params = match hit.params.take() {
            Some(p) => p,
            None if cfg.model_extraction => {
                // The meaning tier finds questions on the same topic, not only the same problem (measured: 6 of 14
                // near misses, such as an area for a perimeter, passed the extraction's own "same" field). Two checks
                // follow: words that turn one question into another, without a model, then System 1 asked directly.
                if let Some((a, b)) = contrast_conflict(&hit.entry.question, question) {
                    metrics::counter!("nafs_arc_total", "result" => "contrast").increment(1);
                    return ArcTry::Miss(format!(
                        "entry {}: the question asks about \"{b}\" where the entry's asked about \"{a}\"",
                        hit.entry.id
                    ));
                }
                let s1 = self.active().s1.clone();
                if cfg.same_check {
                    match same_question(&s1, &hit.entry.question, question, cancel.clone()).await {
                        Ok(true) => {}
                        Ok(false) => {
                            metrics::counter!("nafs_arc_total", "result" => "not_same")
                                .increment(1);
                            return ArcTry::Miss(format!(
                                "entry {}: System 1 judged it a different question",
                                hit.entry.id
                            ));
                        }
                        Err(e) => return ArcTry::Miss(format!("entry {}: {e}", hit.entry.id)),
                    }
                }
                match extract_with(&s1, &hit.entry, question, cancel).await {
                    Ok(p) => p,
                    Err(e) => {
                        metrics::counter!("nafs_arc_total", "result" => "extract_miss")
                            .increment(1);
                        return ArcTry::Miss(format!("entry {}: {e}", hit.entry.id));
                    }
                }
            }
            None => {
                return ArcTry::Miss(
                    "only the meaning tier matched and model extraction is off".into(),
                )
            }
        };
        let unused = unused_numbers(&hit.entry, question, &params);
        if !unused.is_empty() {
            metrics::counter!("nafs_arc_total", "result" => "unused_number").increment(1);
            return ArcTry::Miss(format!(
                "entry {}: the question also gives {} which the entry does not use",
                hit.entry.id,
                unused.join(", ")
            ));
        }
        if let Some((d, v)) = params.iter().find(|(_, v)| !verbatim(question, v)) {
            return ArcTry::Miss(format!(
                "entry {}: \"{v}\" ({d}) is not in the question",
                hit.entry.id
            ));
        }
        match store.invoke(&hit.entry, &params).await {
            Ok(o) => {
                hit.params = Some(params);
                let serve = !honour_trust || (cfg.serve && hit.entry.status == Status::Trusted);
                // The try endpoint (trust not honoured) is a look, not a use: it is not counted.
                if honour_trust {
                    store.note_hit(&hit.entry.id, serve, false);
                }
                metrics::counter!("nafs_arc_total", "result" => if serve { "served" } else { "shadow" }).increment(1);
                if serve {
                    ArcTry::Serve(hit, o)
                } else {
                    ArcTry::Shadow(hit, o)
                }
            }
            Err(e) => {
                if honour_trust {
                    store.note_hit(&hit.entry.id, false, true);
                }
                metrics::counter!("nafs_arc_total", "result" => "error").increment(1);
                ArcTry::Miss(format!("entry {} could not run: {e}", hit.entry.id))
            }
        }
    }

    /// The reply for a served hit, or `None` when a reference answer the request carried disagrees (the entry is then
    /// recorded as wrong and the request goes to the models).
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn arc_reply(
        &self,
        req: &ChatRequest,
        user_content: &str,
        conv: Option<Uuid>,
        ephemeral: bool,
        request_id: &str,
        dom: Domain,
        t0: Instant,
        mut stages: Vec<StageTiming>,
        hit: &Hit,
        o: &Outcome,
        em: &Emitter,
    ) -> Option<ChatReply> {
        let cfg = self.cfg.clone();
        let store = self.arc.clone()?;
        let pair = self.active();
        let content = content_of(dom, o);
        let mut verdict: Option<VerifierResult> = None;
        let mut assurance = if hit.entry.stats.agreements > 0 {
            Assurance::Consensus
        } else {
            Assurance::Unverified
        };
        if let Some(r) = req
            .verification
            .as_ref()
            .and_then(|v| v.reference_answer.as_deref())
        {
            let v = self.hub.verify_math_full(&content, r).await;
            if !v.passed {
                store.record_comparison(
                    &hit.entry.id,
                    false,
                    true,
                    hit.tier,
                    hit.similarity,
                    &cfg.arc,
                );
                tracing::warn!(entry = %hit.entry.id, "a reasoning-cache answer failed its check; answering with the models");
                return None;
            }
            store.record_comparison(
                &hit.entry.id,
                true,
                true,
                hit.tier,
                hit.similarity,
                &cfg.arc,
            );
            assurance = Assurance::Verified;
            verdict = Some(v);
        }
        let model_calls = u32::from(hit.tier == Tier::Meaning);
        let explanation = format!(
            "Answered from the reasoning cache: entry {} matched by its {} (similarity {:.2}) and ran on CPU in {} ms; no model was asked to answer.",
            hit.entry.id,
            match hit.tier {
                Tier::Template => "template",
                Tier::Shape => "shape",
                Tier::Meaning => "meaning, values read by System 1",
            },
            hit.similarity,
            o.ms
        );
        let decision = DecisionRecord {
            decision_id: Uuid::new_v4(),
            request_id: request_id.to_string(),
            timestamp: nafs_types::now_utc(),
            config_hash: cfg.hash.clone(),
            model_profile: cfg.assistant.profile.clone(),
            backend: pair.name.clone(),
            calibration_version: None,
            features: GateFeatures::default(),
            e1_hat: 0.0,
            e2_hat: 0.0,
            lambda: req.lambda.unwrap_or(cfg.gate.lambda_default),
            delta_c: 0.0,
            escalate: false,
            rule: GateRule::ReasoningCache,
            audit: false,
            explanation,
        };
        em.send(StreamEvent::Decision(Box::new(decision.clone())));
        em.send(StreamEvent::Token {
            text: content.clone(),
            mode: ReasoningMode::Fast,
            reasoning: false,
            round: 0,
        });
        let wall = t0.elapsed().as_millis() as u64;
        stages.push(StageTiming {
            stage: "arc".into(),
            ms: o.ms,
            note: Some(format!("{:?} tier, entry {}", hit.tier, hit.entry.id).to_lowercase()),
            ..Default::default()
        });
        let usage = Usage {
            wall_ms: wall,
            ttft_ms: Some(wall),
            model_calls,
            verifier_calls: u32::from(verdict.is_some()),
            stages,
            ..Default::default()
        };
        let reply = ChatReply {
            reply_id: Uuid::new_v4(),
            conversation_id: conv,
            decision_id: decision.decision_id,
            reasoning_mode: ReasoningMode::Fast,
            assurance,
            trace_state: TraceState::T,
            states: vec![TraceState::C, TraceState::T],
            message: Message {
                role: "assistant".into(),
                content: content.clone(),
            },
            uncertainty_notice: None,
            verifier: verdict,
            usage: usage.clone(),
            degraded: false,
            draft: None,
            reasoning: Some(reasoning_of(hit, o)),
            decision: decision.clone(),
            domain: dom,
            rounds: vec![],
            draft_conditioned: false,
            retrieved: vec![],
            sources: vec![json!({"arc": {
                "entry": hit.entry.id, "tier": hit.tier, "similarity": hit.similarity, "ms": o.ms,
                "params": hit.params, "template": hit.entry.template, "built_from": hit.entry.question,
                "agreements": hit.entry.stats.agreements, "status": hit.entry.status,
                "functions": hit.entry.functions.iter().map(|f| f.description.clone()).collect::<Vec<_>>(),
            }})],
        };
        if let (false, Some(c)) = (ephemeral, conv) {
            let rewind = req.rewind_to.filter(|_| req.conversation_id.is_some());
            if let Err(e) =
                self.store
                    .append_turn(c, rewind, user_content, &content, reply.reply_id)
            {
                tracing::warn!(error = %e, conversation = %c, "could not store the turn");
            }
        }
        if let Err(e) = self.store.save_reply(&reply, ephemeral || req.lab, req.lab) {
            tracing::warn!(error = %e, reply = %reply.reply_id, "could not store the reply");
        }
        let _ = self.store.write_trace(&TraceRecord {
            schema_version: "1.0".into(),
            request_id: request_id.to_string(),
            decision_id: decision.decision_id,
            conversation_id: conv,
            timestamp: nafs_types::now_utc(),
            config_hash: cfg.hash.clone(),
            model_profile: cfg.assistant.profile.clone(),
            backend: pair.name.clone(),
            model_hash: format!("{}|{}", pair.s1.cfg.hf_id, pair.s2.cfg.hf_id),
            adapter_hash: "none".into(),
            domain: dom,
            path: ReasoningMode::Fast,
            states: vec![TraceState::C, TraceState::T],
            rounds: vec![],
            assurance,
            usage: usage.clone(),
            audit: false,
        });
        let traffic = if req.lab { "lab" } else { "live" };
        metrics::counter!("nafs_requests_total", "mode" => "cache", "assurance" => format!("{:?}", assurance).to_lowercase(), "domain" => dom.as_str(), "traffic" => traffic).increment(1);
        metrics::histogram!("nafs_latency_seconds", "mode" => "cache", "traffic" => traffic)
            .record(usage.wall_ms as f64 / 1000.0);
        em.send(StreamEvent::Done(Box::new(reply.clone())));
        Some(reply)
    }

    /// A candidate ran beside the models: what their answer (or the request's reference) says about it.
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn arc_compare(
        &self,
        hit: &Hit,
        o: &Outcome,
        dom: Domain,
        final_text: &str,
        assurance: Assurance,
        mode: ReasoningMode,
        reference: Option<&str>,
    ) {
        let Some(store) = self.arc.clone() else {
            return;
        };
        let (agree, checked) = match reference {
            Some(r) => (
                self.hub
                    .verify_math_full(&content_of(dom, o), r)
                    .await
                    .passed,
                true,
            ),
            None => {
                // An unchecked quick answer is no evidence either way.
                if !(assurance == Assurance::Verified
                    || assurance == Assurance::Consensus
                    || mode == ReasoningMode::Slow)
                    || assurance == Assurance::Abstained
                {
                    return;
                }
                (
                    same_answer(&o.answer, final_text),
                    assurance == Assurance::Verified,
                )
            }
        };
        if let Some(e) = store.record_comparison(
            &hit.entry.id,
            agree,
            checked,
            hit.tier,
            hit.similarity,
            &self.cfg.arc,
        ) {
            metrics::counter!("nafs_arc_shadow_total", "agree" => if agree { "yes" } else { "no" })
                .increment(1);
            tracing::info!(entry = %e.id, agree, checked, status = ?e.status, "reasoning cache compared with the models");
        }
    }

    /// A System 2 answer goes to the build queue when the configuration admits it.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn arc_admit(
        &self,
        question: &str,
        final_text: &str,
        reasoning: &str,
        reference: Option<&str>,
        dom: Domain,
        assurance: Assurance,
        reply_id: Uuid,
    ) {
        let Some(store) = self.arc.as_ref() else {
            return;
        };
        let cfg = &self.cfg.arc;
        let a = format!("{assurance:?}").to_lowercase();
        if !cfg.build || !cfg.admit.iter().any(|x| x == &a) || final_text.trim().is_empty() {
            return;
        }
        // Already answerable: nothing to build.
        if store.lookup_text(question).is_some() {
            return;
        }
        let added = store.enqueue(
            BuildJob {
                question: question.to_string(),
                answer: final_text.to_string(),
                reasoning: reasoning.to_string(),
                expected: reference.map(str::to_string),
                domain: dom.as_str().to_string(),
                source_reply: Some(reply_id.to_string()),
                attempts: 0,
            },
            cfg.queue_cap,
        );
        if added {
            metrics::counter!("nafs_arc_queued_total").increment(1);
        }
    }

    /// One build from the queue, run while nothing else is in flight; a request that arrives stops it and the job
    /// goes back to the front of the queue. `None` when the queue was empty.
    pub async fn run_one_arc_build(&self, force: bool) -> Option<BuildReport> {
        let store = self.arc.clone()?;
        let job = store.next_job()?;
        let inflight = self.inflight.clone();
        let cancel =
            (!force).then(|| CancelFlag(Arc::new(move || inflight.load(Ordering::SeqCst) > 0)));
        let pair = self.active();
        let k = self.arc_embedder();
        let embed = move |text: String| -> futures::future::BoxFuture<'static, Option<Vec<f32>>> {
            let k = k.clone();
            Box::pin(async move {
                match k {
                    Some(k) => k.embed(&text).await,
                    None => None,
                }
            })
        };
        let rep = build(
            &store,
            &pair.s2,
            Some(&pair.s1),
            &embed,
            &job,
            &self.cfg.arc,
            cancel,
        )
        .await;
        let cancelled = rep
            .reasons
            .iter()
            .any(|r| r.to_lowercase().contains("cancel"));
        if !rep.ok && cancelled && job.attempts < 3 {
            store.requeue(BuildJob {
                attempts: job.attempts + 1,
                ..job
            });
        }
        metrics::counter!("nafs_arc_builds_total", "ok" => if rep.ok { "yes" } else { "no" })
            .increment(1);
        tracing::info!(ok = rep.ok, entry = ?rep.entry, rounds = rep.rounds, ms = rep.ms, reasons = ?rep.reasons, "reasoning cache build");
        Some(rep)
    }

    /// Builds `job` now (the build endpoint with `wait`), without the idle rule.
    pub async fn arc_build_now(&self, job: BuildJob) -> Option<BuildReport> {
        let store = self.arc.clone()?;
        store.requeue(job);
        self.run_one_arc_build(true).await
    }
}
