//! nafs-eval: the Eval Harness (FR-050, FR-051, ER-*). Runs datasets through the orchestrator for the
//! system and its baselines, scores answers with the exact verifiers, and computes accuracy, routing
//! accuracy against the oracle rule of Theorem 2, calibration (ECE), cost, the coupling test (ER-004),
//! the accuracy-versus-budget curve (H4) and paired differences with bootstrap confidence intervals.
//!
//! Systems:
//! - `nafs_aql`: the full system; the gate sees the exact verifier's verdict on the draft.
//! - `nafs_blind`: the full system with the verifier hidden at run time, so the gate must route on its
//!   calibrated e1_hat; answers are scored afterwards. This is the honest routing / ECE test (review M9).
//! - `fast_only`, `slow_only`: the baselines.
//! - `slow_conditioned`: System 2 sees the System 1 draft (coupling ablation, ER-004).
//! - `slow_b<N>`: slow-only with thinking budget N (budget curve, P0-T05).
//! - `slow_noexit`: slow-only without the adaptive reasoning exit (P7-T08 study; pairs with `slow_only`).
use nafs_core::calibration::{ece, reliability};
use nafs_core::config::Caps;
use nafs_core::orchestrator::{DerivedParams, Emitter, Orchestrator};
use nafs_types::{ChatRequest, Domain, GateFeatures, Message, ReasoningMode, VerificationInputs};
use nafs_verify::mutation::clopper_pearson;
use nafs_verify::normalize_answer;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub mod rates;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: String,
    pub domain: String,
    pub prompt: String,
    #[serde(default)]
    pub verification: VerificationInputs,
    #[serde(default)]
    pub source: String,
    #[serde(default)]
    pub license: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalRunRequest {
    pub benchmark: String,
    #[serde(default = "d_systems")]
    pub systems: Vec<String>,
    #[serde(default)]
    pub max_items: Option<usize>,
    #[serde(default)]
    pub offset: Option<usize>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub lambda: Option<f32>,
    #[serde(default)]
    pub thinking_budget: Option<u32>,
    #[serde(default)]
    pub shuffle: Option<bool>,
}
fn d_systems() -> Vec<String> {
    vec!["nafs_aql".into(), "fast_only".into(), "slow_only".into()]
}

pub const FIXED_SYSTEMS: [&str; 8] = [
    "nafs_aql",
    "nafs_blind",
    "fast_only",
    "slow_only",
    "slow_conditioned",
    "slow_noexit",
    "slow_exit_consistency",
    "slow_exit_posterior",
];

/// Gated systems: `nafs_aql`, `nafs_blind`, and their lambda variants `nafs_aql_l<NN>` /
/// `nafs_blind_l<NN>` (lambda = NN/100), used to trace the accuracy-cost frontier (Prop. 2).
/// Returns (verifier hidden from the gate, lambda override).
pub fn gated(system: &str) -> Option<(bool, Option<f32>)> {
    let (base, lam) = match system.rsplit_once("_l") {
        Some((b, n)) if b == "nafs_aql" || b == "nafs_blind" => {
            let v: u32 = n.parse().ok().filter(|v| *v <= 100)?;
            (b, Some(v as f32 / 100.0))
        }
        _ => (system, None),
    };
    match base {
        "nafs_aql" => Some((false, lam)),
        "nafs_blind" => Some((true, lam)),
        _ => None,
    }
}

/// Thinking budget of a `slow_b<N>` system.
pub fn sweep_budget(system: &str) -> Option<u32> {
    system
        .strip_prefix("slow_b")
        .and_then(|n| n.parse().ok())
        .filter(|b| (16..=8192).contains(b))
}

impl EvalRunRequest {
    /// Reject unknown systems and out-of-range values before a run starts (review M5, M14).
    pub fn validate(&self) -> Result<(), String> {
        if self.systems.is_empty() || self.systems.len() > 12 {
            return Err("systems must list 1..12 systems".into());
        }
        for s in &self.systems {
            if !FIXED_SYSTEMS.contains(&s.as_str())
                && sweep_budget(s).is_none()
                && gated(s).is_none()
            {
                return Err(format!(
                    "unknown system '{}'; use {} or slow_b<16..8192>",
                    s.chars().take(40).collect::<String>(),
                    FIXED_SYSTEMS.join(", ")
                ));
            }
        }
        let mut seen = std::collections::HashSet::new();
        if !self.systems.iter().all(|s| seen.insert(s)) {
            return Err("systems must not repeat".into());
        }
        if let Some(l) = self.lambda {
            if !(0.0..=1.0).contains(&l) {
                return Err("lambda must be in [0, 1]".into());
            }
        }
        if let Some(b) = self.thinking_budget {
            if !(16..=8192).contains(&b) {
                return Err("thinking_budget must be in [16, 8192]".into());
            }
        }
        if self
            .max_items
            .map(|m| m == 0 || m > 20_000)
            .unwrap_or(false)
        {
            return Err("max_items must be in [1, 20000]".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ItemResult {
    pub task_id: String,
    pub system: String,
    pub domain: String,
    pub correct: Option<bool>,
    pub mode: String,
    pub assurance: String,
    pub escalated: bool,
    pub wall_ms: u64,
    pub ttft_ms: Option<u64>,
    pub model_calls: u32,
    pub completion_tokens: u32,
    pub reasoning_tokens: u32,
    /// Raw calibrated e1_hat(features) (never forced by a verifier).
    pub e1_hat: f32,
    pub features: GateFeatures,
    /// Exact label of the System 1 draft (verified after the fact for blind runs).
    pub draft_correct: Option<bool>,
    pub rule: String,
    pub decision_id: String,
    pub answer_preview: String,
    pub evidence: Option<String>,
    pub error: Option<String>,
    /// System 2 attempts and the first that passed (Beta fit, FR-021).
    #[serde(default)]
    pub attempts: u32,
    #[serde(default)]
    pub first_pass_attempt: Option<u32>,
    /// Normalised answer keys for the coupling test (ER-004).
    #[serde(default)]
    pub answer_key: String,
    #[serde(default)]
    pub draft_key: Option<String>,
    #[serde(default)]
    pub thinking_budget: Option<u32>,
}

pub fn list_datasets(dir: &str) -> Vec<Value> {
    let mut out = vec![];
    if let Ok(rd) = std::fs::read_dir(dir) {
        let mut names: Vec<_> = rd
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map(|x| x == "jsonl").unwrap_or(false))
            .collect();
        names.sort();
        for p in names {
            let text = std::fs::read_to_string(&p).unwrap_or_default();
            let tasks: Vec<Task> = text
                .lines()
                .filter_map(|l| serde_json::from_str(l).ok())
                .collect();
            let mut by_domain: BTreeMap<String, usize> = BTreeMap::new();
            for t in &tasks {
                *by_domain.entry(t.domain.clone()).or_default() += 1;
            }
            out.push(json!({"name": p.file_stem().unwrap().to_string_lossy(), "tasks": tasks.len(), "domains": by_domain,
                "license": tasks.first().map(|t| t.license.clone()), "source": tasks.first().map(|t| t.source.clone()),
                "verifiable": tasks.iter().filter(|t| t.verification.is_verifiable()).count()}));
        }
    }
    out
}

pub fn load_tasks(dir: &str, name: &str) -> anyhow::Result<Vec<Task>> {
    anyhow::ensure!(
        name.chars()
            .all(|c| c.is_alphanumeric() || c == '_' || c == '-'),
        "invalid dataset name"
    );
    let text = std::fs::read_to_string(format!("{dir}/{name}.jsonl"))?;
    Ok(text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(serde_json::from_str)
        .collect::<Result<Vec<Task>, _>>()?)
}

fn task_domain(t: &Task) -> Option<Domain> {
    Domain::all().into_iter().find(|d| d.as_str() == t.domain)
}

fn error_item(t: &Task, sys: &str, e: String) -> ItemResult {
    ItemResult {
        task_id: t.id.clone(),
        system: sys.into(),
        domain: t.domain.clone(),
        correct: None,
        mode: "error".into(),
        assurance: "error".into(),
        escalated: false,
        wall_ms: 0,
        ttft_ms: None,
        model_calls: 0,
        completion_tokens: 0,
        reasoning_tokens: 0,
        e1_hat: 0.0,
        features: GateFeatures::default(),
        draft_correct: None,
        rule: "".into(),
        decision_id: "".into(),
        answer_preview: "".into(),
        evidence: None,
        error: Some(e),
        attempts: 0,
        first_pass_attempt: None,
        answer_key: String::new(),
        draft_key: None,
        thinking_budget: None,
    }
}

fn enum_str<T: Serialize>(v: T) -> String {
    serde_json::to_value(v)
        .ok()
        .and_then(|x| x.as_str().map(String::from))
        .unwrap_or_default()
}

/// Run one item of one system.
pub async fn run_item(
    orch: &Orchestrator,
    t: &Task,
    sys: &str,
    req: &EvalRunRequest,
    cancel: Arc<AtomicBool>,
) -> ItemResult {
    let (blind, sys_lambda) = gated(sys).unwrap_or((false, None));
    let conditioned = sys == "slow_conditioned";
    let sweep = sweep_budget(sys);
    let creq = ChatRequest {
        messages: vec![Message {
            role: "user".into(),
            content: t.prompt.clone(),
        }],
        domain_hint: if blind { task_domain(t) } else { None },
        force_mode: match sys {
            "fast_only" => Some(ReasoningMode::Fast),
            "slow_only"
            | "slow_conditioned"
            | "slow_noexit"
            | "slow_exit_consistency"
            | "slow_exit_posterior" => Some(ReasoningMode::Slow),
            _ if sweep.is_some() => Some(ReasoningMode::Slow),
            _ => None,
        },
        lambda: sys_lambda.or(req.lambda),
        thinking_budget: sweep.or(req.thinking_budget),
        verification: if blind {
            None
        } else {
            Some(t.verification.clone())
        },
        ephemeral: Some(true),
        lab: true,
        show_reasoning: Some(false),
        audit: Some(false),
        ablation_draft_conditioned: conditioned.then_some(true),
        adaptive_exit: (sys == "slow_noexit").then_some(false),
        adaptive_exit_rule: match sys {
            "slow_exit_consistency" => Some("consistency".into()),
            "slow_exit_posterior" => Some("posterior".into()),
            _ => None,
        },
        ..Default::default()
    };
    // Same wall-clock cap as the HTTP path (review L5).
    let cap = Duration::from_secs(orch.cfg.server.request_wall_clock_cap_s + 30);
    let r = match tokio::time::timeout(
        cap,
        orch.handle_with_cancel(creq, Emitter(None), Some(cancel)),
    )
    .await
    {
        Ok(r) => r,
        Err(_) => return error_item(t, sys, "request wall-clock cap exceeded".into()),
    };
    let reply = match r {
        Ok(r) => r,
        Err(e) => return error_item(t, sys, e.to_string()),
    };
    let verifiable = t.verification.is_verifiable();
    let answer = &reply.message.content;
    let correct = if verifiable {
        orch.hub
            .verify(answer, &t.verification)
            .await
            .map(|v| v.passed)
    } else {
        None
    };
    // The draft actually produced in this run: shown separately when it differs from the answer.
    let had_draft = !reply.states.is_empty() && reply.states[0] == nafs_types::TraceState::A;
    let draft_text = if had_draft {
        Some(reply.draft.clone().unwrap_or_else(|| answer.clone()))
    } else {
        None
    };
    let draft_correct = match (reply.decision.features.draft_verified, &draft_text) {
        (Some(v), _) => Some(v),
        (None, Some(d)) if verifiable => {
            if *d == *answer {
                correct
            } else {
                orch.hub.verify(d, &t.verification).await.map(|v| v.passed)
            }
        }
        _ => None,
    };
    let first_pass = reply
        .rounds
        .iter()
        .position(|r| r.passed && r.verifier != "none")
        .map(|i| i as u32 + 1);
    ItemResult {
        task_id: t.id.clone(),
        system: sys.into(),
        domain: reply.domain.as_str().into(),
        correct,
        mode: if reply.reasoning_mode == ReasoningMode::Slow {
            "slow".into()
        } else {
            "fast".into()
        },
        assurance: enum_str(reply.assurance),
        escalated: reply.decision.escalate,
        wall_ms: reply.usage.wall_ms,
        ttft_ms: reply.usage.ttft_ms,
        model_calls: reply.usage.model_calls,
        completion_tokens: reply.usage.completion_tokens,
        reasoning_tokens: reply.usage.reasoning_tokens,
        e1_hat: reply.decision.e1_hat,
        features: reply.decision.features.clone(),
        draft_correct,
        rule: enum_str(reply.decision.rule),
        decision_id: reply.decision.decision_id.to_string(),
        answer_preview: answer.chars().take(400).collect(),
        evidence: reply
            .verifier
            .and_then(|v| v.evidence)
            .map(|e| e.chars().take(400).collect()),
        error: None,
        attempts: reply.rounds.len() as u32,
        first_pass_attempt: first_pass,
        answer_key: normalize_answer(answer),
        draft_key: draft_text.as_deref().map(normalize_answer),
        thinking_budget: reply.usage.thinking_budget,
    }
}

/// Run one evaluation. `progress` receives light snapshots (no bootstrap), at most every 2 s.
pub async fn run(
    orch: Arc<Orchestrator>,
    req: EvalRunRequest,
    cancel: Arc<AtomicBool>,
    progress: impl FnMut(&Value) + Send,
) -> anyhow::Result<Value> {
    run_with_prior(orch, req, cancel, progress, vec![]).await
}

/// Run (or resume) an evaluation: items already in `prior` (same task and system) are kept and skipped,
/// so a run interrupted by a restart continues where it stopped.
pub async fn run_with_prior(
    orch: Arc<Orchestrator>,
    req: EvalRunRequest,
    cancel: Arc<AtomicBool>,
    mut progress: impl FnMut(&Value) + Send,
    prior: Vec<ItemResult>,
) -> anyhow::Result<Value> {
    req.validate().map_err(anyhow::Error::msg)?;
    let dir = orch.cfg.eval.datasets_dir.clone();
    let mut tasks = load_tasks(&dir, &req.benchmark)?;
    if req.shuffle.unwrap_or(false) {
        let mut rng = fastrand::Rng::with_seed(req.seed.unwrap_or(0));
        rng.shuffle(&mut tasks);
    }
    let off = req.offset.unwrap_or(0).min(tasks.len());
    let tasks: Vec<Task> = tasks
        .into_iter()
        .skip(off)
        .take(req.max_items.unwrap_or(20))
        .collect();
    let started = chrono::Utc::now();
    let done: std::collections::HashSet<(String, String)> = prior
        .iter()
        .filter(|i| i.error.is_none())
        .map(|i| (i.task_id.clone(), i.system.clone()))
        .collect();
    let mut results: Vec<ItemResult> = prior.into_iter().filter(|i| i.error.is_none()).collect();
    let total = tasks.len() * req.systems.len();
    // Publish the total at once so the dashboard shows "0 / N" instead of "0 / 0".
    progress(&report(
        &req, &results, total, started, "running", &orch, false,
    ));
    let mut last = Instant::now();
    'tasks: for t in &tasks {
        for sys in &req.systems {
            if cancel.load(Ordering::Relaxed) {
                break 'tasks;
            }
            if done.contains(&(t.id.clone(), sys.clone())) {
                continue;
            }
            let item = run_item(&orch, t, sys, &req, cancel.clone()).await;
            if cancel.load(Ordering::Relaxed) && item.error.is_some() {
                break 'tasks; // the cancelled item is not a result
            }
            results.push(item);
            if last.elapsed() >= Duration::from_secs(2) || results.len() == total {
                last = Instant::now();
                progress(&report(
                    &req, &results, total, started, "running", &orch, false,
                ));
            }
        }
    }
    let status = if cancel.load(Ordering::Relaxed) {
        "cancelled"
    } else {
        "done"
    };
    let (o2, r2) = (orch.clone(), req.clone());
    let rep = tokio::task::spawn_blocking(move || {
        report(&r2, &results, total, started, status, &o2, true)
    })
    .await?;
    Ok(rep)
}

/// Exact two-sided McNemar test on the discordant pairs (b = only A correct, c = only B correct).
pub fn mcnemar_exact(b: u64, c: u64) -> f64 {
    let n = b + c;
    if n == 0 {
        return 1.0;
    }
    let k = b.min(c);
    // P(X <= k) for X ~ Bin(n, 1/2), summed in log space.
    let mut lc = 0.0f64;
    let mut cdf = 0.0f64;
    let ln2n = n as f64 * std::f64::consts::LN_2;
    for i in 0..=k {
        if i > 0 {
            lc += ((n - i + 1) as f64).ln() - (i as f64).ln();
        }
        cdf += (lc - ln2n).exp();
    }
    (2.0 * cdf).min(1.0)
}

fn bootstrap(diffs: &[f64], seed: u64, reps: usize) -> Option<(f64, f64)> {
    if diffs.is_empty() {
        return None;
    }
    let n = diffs.len();
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut boots: Vec<f64> = (0..reps)
        .map(|_| (0..n).map(|_| diffs[rng.usize(0..n)]).sum::<f64>() / n as f64)
        .collect();
    boots.sort_by(|x, y| x.partial_cmp(y).unwrap());
    Some((
        boots[(reps as f64 * 0.025) as usize],
        boots[(reps as f64 * 0.975) as usize - 1],
    ))
}

fn quantile(v: &mut [f64], q: f64) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some(v[((v.len() - 1) as f64 * q).round() as usize])
}
fn mean(v: &[f64]) -> Option<f64> {
    (!v.is_empty()).then(|| v.iter().sum::<f64>() / v.len() as f64)
}

/// Build the report: per-system metrics, paired differences, routing, calibration, coupling and the
/// budget curve (spec Section 10). `full = false` skips the bootstrap for cheap progress snapshots.
pub fn report(
    req: &EvalRunRequest,
    results: &[ItemResult],
    total: usize,
    started: chrono::DateTime<chrono::Utc>,
    status: &str,
    orch: &Orchestrator,
    full: bool,
) -> Value {
    let mut systems = serde_json::Map::new();
    for sys in &req.systems {
        let rs: Vec<&ItemResult> = results.iter().filter(|r| &r.system == sys).collect();
        let scored: Vec<&&ItemResult> = rs.iter().filter(|r| r.correct.is_some()).collect();
        let acc = (!scored.is_empty()).then(|| {
            scored.iter().filter(|r| r.correct == Some(true)).count() as f64 / scored.len() as f64
        });
        let answered: Vec<&&&ItemResult> = scored
            .iter()
            .filter(|r| r.assurance != "abstained")
            .collect();
        let sel = (!answered.is_empty()).then(|| {
            answered.iter().filter(|r| r.correct == Some(true)).count() as f64
                / answered.len() as f64
        });
        let ok: Vec<&&ItemResult> = rs.iter().filter(|r| r.error.is_none()).collect();
        let mut walls: Vec<f64> = ok.iter().map(|r| r.wall_ms as f64).collect();
        let mut ttfts: Vec<f64> = rs
            .iter()
            .filter_map(|r| r.ttft_ms.map(|x| x as f64))
            .collect();
        let mut assur: BTreeMap<String, usize> = BTreeMap::new();
        for r in &rs {
            *assur.entry(r.assurance.clone()).or_default() += 1;
        }
        let k_correct = scored.iter().filter(|r| r.correct == Some(true)).count() as u64;
        let ci = (!scored.is_empty()).then(|| clopper_pearson(k_correct, scored.len() as u64));
        systems.insert(sys.clone(), json!({"n": rs.len(), "n_scored": scored.len(), "accuracy": acc, "accuracy_ci95": ci.map(|(a, b)| [a, b]), "selective_accuracy": sel,
            "coverage": (!scored.is_empty()).then(|| answered.len() as f64 / scored.len() as f64),
            // Escalation from the gate's decision, so degraded escalations count (review L3).
            "escalation_rate": (!ok.is_empty()).then(|| ok.iter().filter(|r| r.escalated).count() as f64 / ok.len() as f64),
            "mean_wall_ms": mean(&walls), "p50_wall_ms": quantile(&mut walls.clone(), 0.5), "p95_wall_ms": quantile(&mut walls, 0.95),
            "mean_ttft_ms": mean(&ttfts), "p95_ttft_ms": quantile(&mut ttfts, 0.95),
            "mean_model_calls": mean(&ok.iter().map(|r| r.model_calls as f64).collect::<Vec<_>>()),
            "mean_reasoning_tokens": mean(&ok.iter().map(|r| r.reasoning_tokens as f64).collect::<Vec<_>>()),
            "errors": rs.iter().filter(|r| r.error.is_some()).count(), "assurance": assur, "thinking_budget": sweep_budget(sys)}));
    }
    let by = |sys: &str| -> BTreeMap<String, &ItemResult> {
        results
            .iter()
            .filter(|r| r.system == sys && r.correct.is_some())
            .map(|r| (r.task_id.clone(), r))
            .collect()
    };
    // Paired comparisons on items scored under both systems.
    let mut paired = vec![];
    let mut pairs: Vec<(String, String)> = vec![];
    for g in req.systems.iter().filter(|x| gated(x).is_some()) {
        pairs.push((g.clone(), "slow_only".into()));
        pairs.push((g.clone(), "fast_only".into()));
    }
    pairs.push(("fast_only".into(), "slow_only".into()));
    pairs.push(("slow_conditioned".into(), "slow_only".into()));
    pairs.push(("slow_noexit".into(), "slow_only".into()));
    pairs.push(("slow_exit_posterior".into(), "slow_exit_consistency".into()));
    for (a, b) in pairs.iter().map(|(a, b)| (a.as_str(), b.as_str())) {
        if !(req.systems.iter().any(|s| s == a) && req.systems.iter().any(|s| s == b)) {
            continue;
        }
        let (ma, mb) = (by(a), by(b));
        let pairs_ab: Vec<(String, f64)> = ma
            .iter()
            .filter_map(|(k, ra)| {
                mb.get(k).map(|rb| {
                    (
                        ra.domain.clone(),
                        (ra.correct == Some(true)) as i32 as f64
                            - (rb.correct == Some(true)) as i32 as f64,
                    )
                })
            })
            .collect();
        let diffs: Vec<f64> = pairs_ab.iter().map(|p| p.1).collect();
        let (only_a, only_b) = (
            diffs.iter().filter(|d| **d > 0.0).count() as u64,
            diffs.iter().filter(|d| **d < 0.0).count() as u64,
        );
        // FR-051: per-domain paired differences with Bonferroni correction across domains.
        let mut doms: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        for (d, x) in &pairs_ab {
            doms.entry(d.clone()).or_default().push(*x);
        }
        let m_dom = doms.len().max(1) as f64;
        let per_domain: Vec<Value> = doms
            .iter()
            .map(|(d, xs)| {
                let (pa, pb) = (xs.iter().filter(|v| **v > 0.0).count() as u64, xs.iter().filter(|v| **v < 0.0).count() as u64);
                let p = mcnemar_exact(pa, pb);
                json!({"domain": d, "n_pairs": xs.len(), "delta_accuracy": xs.iter().sum::<f64>() / xs.len() as f64,
                    "ci95": if full { bootstrap(xs, 11, 1000).map(|(l, h)| [l, h]) } else { None },
                    "mcnemar_p": p, "p_bonferroni": (p * m_dom).min(1.0)})
            })
            .collect();
        if diffs.is_empty() {
            continue;
        }
        let n = diffs.len();
        let m = diffs.iter().sum::<f64>() / n as f64;
        let ci = if full {
            let mut rng = fastrand::Rng::with_seed(req.seed.unwrap_or(7));
            let mut boots: Vec<f64> = (0..2000)
                .map(|_| (0..n).map(|_| diffs[rng.usize(0..n)]).sum::<f64>() / n as f64)
                .collect();
            boots.sort_by(|x, y| x.partial_cmp(y).unwrap());
            Some((boots[50], boots[1949]))
        } else {
            None
        };
        let cost = |s: &str| {
            mean(
                &results
                    .iter()
                    .filter(|r| r.system == s && r.error.is_none())
                    .map(|r| r.wall_ms as f64)
                    .collect::<Vec<_>>(),
            )
        };
        let cr = match (cost(a), cost(b)) {
            (Some(x), Some(y)) => Some(x / y.max(1.0)),
            _ => None,
        };
        let mcn = mcnemar_exact(only_a, only_b);
        paired.push(json!({"a": a, "b": b, "n_pairs": n, "delta_accuracy": m, "ci95": ci.map(|(l, h)| [l, h]),
            "discordant": {"only_a_correct": only_a, "only_b_correct": only_b}, "mcnemar_p": mcn, "per_domain": per_domain,
            "discordance": diffs.iter().filter(|d| **d != 0.0).count() as f64 / n as f64,
            "non_inferior_3pt": ci.map(|(l, _)| l > -0.03), "non_inferior_1pt": ci.map(|(l, _)| l > -0.01), "cost_ratio": cr,
            "gate_g2_eligible": n >= 700,
            "note": "Bootstrap 95% CI (2,000 resamples). Release gates need >= 700 pairs (3 pt, G2) and >= 6,200 pairs (1 pt, G6); smaller runs are indicative only."}));
    }
    // Bonferroni across the comparisons reported in this run (FR-051).
    let m_cmp = paired.len().max(1) as f64;
    for p in paired.iter_mut() {
        let adj = (p["mcnemar_p"].as_f64().unwrap_or(1.0) * m_cmp).min(1.0);
        p["p_bonferroni"] = json!(adj);
        p["significant_at_0_05"] = json!(adj < 0.05);
    }
    let cfg = &orch.cfg;
    let (fast, slow) = (by("fast_only"), by("slow_only"));
    // Routing against the oracle of Theorem 2, per gated system. Only nafs_blind measures the gate's
    // estimator; on nafs_aql the exact verifier decides verifiable items (review M9).
    let mut routing = serde_json::Map::new();
    for sys in req.systems.iter().filter(|x| gated(x).is_some()) {
        let (blind_sys, sys_lambda) = gated(sys).unwrap_or((false, None));
        let sys = sys.as_str();
        let items: Vec<&ItemResult> = results
            .iter()
            .filter(|r| r.system == sys && r.error.is_none())
            .collect();
        if items.is_empty() || fast.is_empty() || slow.is_empty() {
            continue;
        }
        let (mut agree, mut nn) = (0usize, 0usize);
        let mut confusion = json!({"escalate_needed": 0, "escalate_wasted": 0, "keep_correct": 0, "keep_missed": 0});
        for r in &items {
            let (Some(f), Some(s)) = (fast.get(&r.task_id), slow.get(&r.task_id)) else {
                continue;
            };
            let fw = match r.draft_correct {
                Some(c) => (!c) as i32 as f32,
                None => (f.correct == Some(false)) as i32 as f32,
            };
            let sw = (s.correct == Some(false)) as i32 as f32;
            let dom = Domain::all()
                .into_iter()
                .find(|d| d.as_str() == r.domain)
                .unwrap_or_default();
            let oracle = cfg.gate.loss_wrong * (fw - sw)
                > sys_lambda.or(req.lambda).unwrap_or(cfg.gate.lambda_default) * orch.delta_c(dom)
                    / cfg.gate.cost_unit_s;
            nn += 1;
            agree += (oracle == r.escalated) as usize;
            let key = match (r.escalated, oracle) {
                (true, true) => "escalate_needed",
                (true, false) => "escalate_wasted",
                (false, false) => "keep_correct",
                (false, true) => "keep_missed",
            };
            confusion[key] = json!(confusion[key].as_u64().unwrap_or(0) + 1);
        }
        if nn > 0 {
            routing.insert(sys.into(), json!({"n": nn, "routing_accuracy": agree as f64 / nn as f64, "confusion": confusion,
                "escalation_rate": items.iter().filter(|r| r.escalated).count() as f64 / items.len() as f64,
                "measures_gate": blind_sys, "lambda": sys_lambda.or(req.lambda).unwrap_or(cfg.gate.lambda_default),
                "note": if blind_sys { "Gate routed on e1_hat with the verifier hidden: this measures the gate (H1 target >= 0.70 at g <= 0.30)." } else { "The exact verifier decided verifiable items; this is not a measure of the gate's estimator." }}));
        }
    }
    // Calibration of the raw e1_hat on drafts with an exact label (gated systems).
    let lab: Vec<&ItemResult> = results
        .iter()
        .filter(|r| gated(&r.system).is_some() && r.draft_correct.is_some() && r.error.is_none())
        .collect();
    let calibration = if lab.is_empty() {
        Value::Null
    } else {
        let ps: Vec<f32> = lab.iter().map(|r| r.e1_hat).collect();
        let ys: Vec<bool> = lab.iter().map(|r| r.draft_correct == Some(false)).collect();
        json!({"n": ps.len(), "ece_e1": ece(&ps, &ys, 5), "brier_e1": ps.iter().zip(&ys).map(|(p, &y)| (p - y as i32 as f32).powi(2)).sum::<f32>() / ps.len() as f32,
            "draft_error_rate": ys.iter().filter(|&&y| y).count() as f64 / ys.len() as f64,
            "reliability": reliability(&ps, &ys, 5).iter().map(|(p, o, c)| json!({"predicted": p, "observed": o, "count": c})).collect::<Vec<_>>(),
            "calibration_in_use": orch.calibration().version, "er005_eligible": ps.len() >= cfg.gate.calibration_min_n,
            "note": "Raw e1_hat of the calibration in use, scored against exact draft labels. ER-005 needs >= 2,900 held-out samples."})
    };
    // Coupling test (ER-004): adoption of a wrong draft by System 2, blind vs conditioned.
    let coupling = {
        let fast_all: BTreeMap<String, &ItemResult> = results
            .iter()
            .filter(|r| r.system == "fast_only" && r.error.is_none())
            .map(|r| (r.task_id.clone(), r))
            .collect();
        let mut out = serde_json::Map::new();
        // Blind: slow_only vs the independent fast_only draft on the same task.
        let (mut kb, mut nb) = (0u64, 0u64);
        for s in results
            .iter()
            .filter(|r| r.system == "slow_only" && r.error.is_none())
        {
            if let Some(f) = fast_all.get(&s.task_id) {
                if f.correct == Some(false) {
                    nb += 1;
                    kb += (s.answer_key == f.answer_key && !s.answer_key.is_empty()) as u64;
                }
            }
        }
        let (mut kc, mut nc) = (0u64, 0u64);
        for s in results
            .iter()
            .filter(|r| r.system == "slow_conditioned" && r.error.is_none())
        {
            if s.draft_correct == Some(false) {
                nc += 1;
                kc += (Some(&s.answer_key) == s.draft_key.as_ref() && !s.answer_key.is_empty())
                    as u64;
            }
        }
        for (name, k, n) in [("draft_blind", kb, nb), ("draft_conditioned", kc, nc)] {
            if n > 0 {
                let (_, hi) = clopper_pearson(k, n);
                out.insert(name.into(), json!({"wrong_drafts": n, "adopted": k, "adoption_rate": k as f64 / n as f64, "upper95": hi, "h5_pass": hi <= 0.05, "er004_eligible": n >= 200}));
            }
        }
        if out.is_empty() {
            Value::Null
        } else {
            out.insert("note".into(), json!("Adoption = System 2's final answer equals a wrong System 1 draft. H5 needs the 95% upper bound <= 0.05; ER-004 needs >= 200 wrong-draft trials per mode."));
            Value::Object(out)
        }
    };
    let budget_curve = budget_curve(req, results, orch);
    let frontier = frontier(&systems);
    json!({"status": status, "request": req, "started_at": started, "progress": {"done": results.len(), "total": total},
        "systems": systems, "paired": paired, "routing": routing, "calibration": calibration, "coupling": coupling, "budget_curve": budget_curve, "frontier": frontier,
        "calibration_in_use": orch.calibration().version, "items": results})
}

/// Accuracy-cost frontier over the evaluated policies (Prop. 2): a policy is Pareto-optimal when no
/// other policy is at least as accurate and at most as costly, and strictly better in one of them.
fn frontier(systems: &serde_json::Map<String, Value>) -> Value {
    let pts: Vec<(String, f64, f64, u64)> = systems
        .iter()
        .filter_map(|(k, v)| {
            Some((
                k.clone(),
                v["accuracy"].as_f64()?,
                v["mean_wall_ms"].as_f64()? / 1000.0,
                v["n_scored"].as_u64().unwrap_or(0),
            ))
        })
        .collect();
    if pts.len() < 2 {
        return Value::Null;
    }
    let dominated = |i: usize| {
        pts.iter().enumerate().any(|(j, q)| {
            j != i && q.1 >= pts[i].1 && q.2 <= pts[i].2 && (q.1 > pts[i].1 || q.2 < pts[i].2)
        })
    };
    let slow = pts.iter().find(|p| p.0 == "slow_only").map(|p| p.2);
    json!(pts
        .iter()
        .enumerate()
        .map(
            |(i, p)| json!({"system": p.0, "accuracy": p.1, "mean_cost_s": p.2, "n": p.3,
        "cost_vs_slow": slow.map(|c| p.2 / c.max(1e-6)), "pareto": !dominated(i)})
        )
        .collect::<Vec<_>>())
}

/// Accuracy and cost per thinking budget and the Prop. 9 choice b* = argmin L(1-acc(b)) + lambda c(b).
fn budget_curve(req: &EvalRunRequest, results: &[ItemResult], orch: &Orchestrator) -> Value {
    let cfg = &orch.cfg;
    let lambda = req.lambda.unwrap_or(cfg.gate.lambda_default) as f64;
    // per domain -> per budget -> (n, correct, wall seconds, reasoning tokens)
    type Cell = (u64, u64, f64, f64);
    let mut doms: BTreeMap<String, BTreeMap<u32, Cell>> = BTreeMap::new();
    for r in results
        .iter()
        .filter(|r| r.error.is_none() && r.correct.is_some())
    {
        let b = match sweep_budget(&r.system) {
            Some(b) => b,
            None if r.system == "slow_only" => match r.thinking_budget {
                Some(b) => b,
                None => continue,
            },
            None => continue,
        };
        let e = doms
            .entry(r.domain.clone())
            .or_default()
            .entry(b)
            .or_insert((0, 0, 0.0, 0.0));
        e.0 += 1;
        e.1 += (r.correct == Some(true)) as u64;
        e.2 += r.wall_ms as f64 / 1000.0;
        e.3 += r.reasoning_tokens as f64;
    }
    if doms.values().all(|m| m.len() < 2) {
        return Value::Null;
    }
    json!(doms.into_iter().map(|(d, m)| {
        let pts: Vec<Value> = m.iter().map(|(b, (n, k, w, rt))| {
            let acc = *k as f64 / *n as f64;
            let wall = w / *n as f64;
            json!({"budget": b, "n": n, "accuracy": acc, "accuracy_ci95": clopper_pearson(*k, *n), "mean_wall_s": wall, "mean_reasoning_tokens": rt / *n as f64,
                "objective": cfg.gate.loss_wrong as f64 * (1.0 - acc) + lambda * wall / cfg.gate.cost_unit_s as f64})
        }).collect();
        let best = pts.iter().min_by(|a, b| a["objective"].as_f64().partial_cmp(&b["objective"].as_f64()).unwrap()).and_then(|p| p["budget"].as_u64());
        (d, json!({"points": pts, "b_star": best, "lambda": lambda}))
    }).collect::<serde_json::Map<_, _>>())
}

/// Recompute the report of a stored run from its items, so metric fixes apply to past runs.
pub fn rebuild(run: &Value, orch: &Orchestrator) -> Option<Value> {
    let req: EvalRunRequest = serde_json::from_value(run.get("request")?.clone()).ok()?;
    let items: Vec<ItemResult> = serde_json::from_value(run.get("items")?.clone()).ok()?;
    let started = run
        .get("started_at")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
        .unwrap_or_else(chrono::Utc::now);
    let total = run["progress"]["total"]
        .as_u64()
        .unwrap_or(items.len() as u64) as usize;
    let mut v = report(
        &req,
        &items,
        total,
        started,
        run["status"].as_str().unwrap_or("done"),
        orch,
        true,
    );
    for k in ["run_id", "kind", "finished_at", "calibration_in_use"] {
        if let Some(x) = run.get(k) {
            v[k] = x.clone();
        }
    }
    Some(v)
}

/// Labelled gate samples from a stored run: gated items with an exact draft label and k > 0 samples.
/// Returns (decision id, features, draft_wrong).
pub fn calibration_samples(run: &Value) -> anyhow::Result<Vec<(String, GateFeatures, bool)>> {
    let items: Vec<ItemResult> = serde_json::from_value(run["items"].clone())?;
    Ok(items
        .into_iter()
        .filter(|r| {
            gated(&r.system).is_some()
                && r.draft_correct.is_some()
                && r.features.k > 0
                && r.error.is_none()
        })
        .map(|r| (r.decision_id, r.features, r.draft_correct == Some(false)))
        .collect())
}

// ---------------- Phase-0 parameter derivation (P0-T04, P0-T05, FR-011, FR-021) ----------------

fn ln_beta(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}
/// Lanczos approximation of ln Gamma(x), x > 0.
fn ln_gamma(x: f64) -> f64 {
    const G: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        return std::f64::consts::PI.ln()
            - (std::f64::consts::PI * x).sin().ln()
            - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = G[0];
    let t = x + 7.5;
    for (i, g) in G.iter().enumerate().skip(1) {
        a += g / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// MLE of (alpha, beta) for the Beta-geometric model of System 2 attempts: a task's acceptance
/// probability a ~ Beta(alpha, beta); P(first pass at attempt j) = B(alpha+1, beta+j-1)/B(alpha, beta);
/// P(no pass in m attempts) = B(alpha, beta+m)/B(alpha, beta). `obs` = (first pass, attempts made).
pub fn fit_beta_geometric(obs: &[(Option<u32>, u32)]) -> Option<(f64, f64, f64)> {
    if obs.is_empty() {
        return None;
    }
    let ll = |a: f64, b: f64| -> f64 {
        let lb = ln_beta(a, b);
        obs.iter()
            .map(|(j, m)| match j {
                Some(j) => ln_beta(a + 1.0, b + *j as f64 - 1.0) - lb,
                None => ln_beta(a, b + *m as f64) - lb,
            })
            .sum()
    };
    let mut best = (f64::NEG_INFINITY, 1.0, 1.0);
    // Log-spaced grid, then a local refinement.
    for i in 0..60 {
        for k in 0..60 {
            let a = 10f64.powf(-1.5 + 3.5 * i as f64 / 59.0);
            let b = 10f64.powf(-1.5 + 3.5 * k as f64 / 59.0);
            let v = ll(a, b);
            if v > best.0 {
                best = (v, a, b);
            }
        }
    }
    let (mut a, mut b) = (best.1, best.2);
    let mut step = 0.1;
    for _ in 0..200 {
        let mut improved = false;
        for (da, db) in [
            (1.0, 1.0),
            (1.0, -1.0),
            (-1.0, 1.0),
            (-1.0, -1.0),
            (1.0, 0.0),
            (-1.0, 0.0),
            (0.0, 1.0),
            (0.0, -1.0),
        ] {
            let (na, nb) = (a * (1.0 + step * da), b * (1.0 + step * db));
            let v = ll(na, nb);
            if v > best.0 {
                best = (v, na, nb);
                a = na;
                b = nb;
                improved = true;
            }
        }
        if !improved {
            step /= 2.0;
            if step < 1e-4 {
                break;
            }
        }
    }
    Some((a, b, best.0))
}

/// Prop. 3: M* = ceil(alpha / (c_p/L_a) - alpha - beta), clipped to [1, max].
pub fn optimal_draws(alpha: f64, beta: f64, cost_ratio: f64, max: u32) -> u32 {
    let m = (alpha / cost_ratio.max(1e-6) - alpha - beta).ceil();
    (m.max(1.0) as u32).min(max)
}

/// Derive measured gate parameters from stored evaluation runs: e2_hat and delta_c per domain from
/// slow_only / fast_only pairs, N and R from the Beta fit of System 2 acceptance (Prop. 3), and the
/// thinking budget b* from the budget curve (Prop. 9). Domains with fewer than `min_n` items keep their
/// placeholders and are listed as such.
pub fn derive_params(
    runs: &[Value],
    orch: &Orchestrator,
    min_n: usize,
) -> anyhow::Result<DerivedParams> {
    let mut items: Vec<ItemResult> = vec![];
    let mut ids = vec![];
    for r in runs {
        items.extend(serde_json::from_value::<Vec<ItemResult>>(
            r["items"].clone(),
        )?);
        if let Some(id) = r["run_id"].as_str() {
            ids.push(id.to_string());
        }
    }
    anyhow::ensure!(
        !items.is_empty(),
        "no evaluation items in the selected runs"
    );
    let cfg = &orch.cfg;
    let mut p = DerivedParams {
        source_runs: ids,
        derived_at: Some(chrono::Utc::now().to_rfc3339()),
        ..Default::default()
    };
    let mut details = serde_json::Map::new();
    for d in Domain::all() {
        let ds = d.as_str();
        let slow: Vec<&ItemResult> = items
            .iter()
            .filter(|r| {
                r.domain == ds
                    && r.system == "slow_only"
                    && r.error.is_none()
                    && r.correct.is_some()
            })
            .collect();
        let fast: Vec<&ItemResult> = items
            .iter()
            .filter(|r| r.domain == ds && r.system == "fast_only" && r.error.is_none())
            .collect();
        let mut det = serde_json::Map::new();
        det.insert("n_slow".into(), json!(slow.len()));
        det.insert("n_fast".into(), json!(fast.len()));
        if slow.len() >= min_n {
            let wrong = slow.iter().filter(|r| r.correct == Some(false)).count();
            // Laplace-smoothed so a perfect small sample does not claim e2 = 0.
            let e2 = (wrong as f32 + 1.0) / (slow.len() as f32 + 2.0);
            p.e2_hat.insert(ds.into(), e2);
            det.insert("e2_hat".into(), json!({"value": e2, "wrong": wrong, "n": slow.len(), "ci95": clopper_pearson(wrong as u64, slow.len() as u64)}));
            if fast.len() >= min_n {
                let ms =
                    slow.iter().map(|r| r.wall_ms as f64).sum::<f64>() / slow.len() as f64 / 1000.0;
                let mf =
                    fast.iter().map(|r| r.wall_ms as f64).sum::<f64>() / fast.len() as f64 / 1000.0;
                let dc = (ms - mf).max(1.0) as f32;
                p.delta_c_s.insert(ds.into(), dc);
                det.insert(
                    "delta_c_s".into(),
                    json!({"value": dc, "mean_slow_s": ms, "mean_fast_s": mf}),
                );
            }
            // Beta fit of acceptance over System 2 attempts (verifiable domains only).
            let obs: Vec<(Option<u32>, u32)> = slow
                .iter()
                .filter(|r| r.attempts > 0 && r.rule == "forced_override")
                .map(|r| (r.first_pass_attempt, r.attempts))
                .collect();
            if obs.len() >= min_n && obs.iter().any(|o| o.0.is_some()) {
                if let Some((a, b, ll)) = fit_beta_geometric(&obs) {
                    let m = optimal_draws(
                        a,
                        b,
                        cfg.deliberation.cost_ratio as f64,
                        cfg.deliberation.max_rounds_cap + 1,
                    );
                    let base = orch.caps(d);
                    p.caps.insert(
                        ds.into(),
                        Caps {
                            n: 1,
                            r: m.saturating_sub(1),
                            budget: base.budget,
                        },
                    );
                    det.insert("beta_fit".into(), json!({"alpha": a, "beta": b, "mean_acceptance": a / (a + b), "log_likelihood": ll, "n_tasks": obs.len(),
                        "censored": obs.iter().filter(|o| o.0.is_none()).count(), "cost_ratio": cfg.deliberation.cost_ratio, "M_star": m, "N": 1, "R": m.saturating_sub(1),
                        "caveat": "Retries are evidence-fed, not independent draws; the Beta-geometric fit is the modelling choice of Prop. 3 and is validated as hypothesis H3."}));
                }
            }
        } else {
            det.insert(
                "status".into(),
                json!(format!(
                    "placeholder kept: {} slow_only items < {min_n}",
                    slow.len()
                )),
            );
        }
        details.insert(ds.into(), Value::Object(det));
    }
    // Budget curve: b* per domain.
    let req = EvalRunRequest {
        benchmark: String::new(),
        systems: vec![],
        max_items: None,
        offset: None,
        seed: None,
        lambda: None,
        thinking_budget: None,
        shuffle: None,
    };
    if let Value::Object(curve) = budget_curve(&req, &items, orch) {
        for (d, v) in &curve {
            let n_min = v["points"]
                .as_array()
                .map(|a| a.iter().filter_map(|p| p["n"].as_u64()).min().unwrap_or(0))
                .unwrap_or(0);
            if let (Some(b), true) = (v["b_star"].as_u64(), n_min as usize >= min_n) {
                p.budgets.insert(d.clone(), b as u32);
            }
        }
        details.insert("budget_curve".into(), Value::Object(curve));
    }
    p.details = Value::Object(details);
    Ok(p)
}

/// Report export (P4-T01): a self-contained Markdown summary of a stored run.
pub fn markdown_report(r: &Value) -> String {
    let f = |v: &Value, d: usize| {
        v.as_f64()
            .map(|x| format!("{x:.d$}"))
            .unwrap_or_else(|| "-".into())
    };
    let pc = |v: &Value| {
        v.as_f64()
            .map(|x| format!("{:.1}%", x * 100.0))
            .unwrap_or_else(|| "-".into())
    };
    let mut o = String::new();
    o += &format!(
        "# Nafs-Aql evaluation report\n\nRun `{}` on `{}` ({} items done of {}), status {}.\n\n",
        r["run_id"].as_str().unwrap_or("?"),
        r["request"]["benchmark"].as_str().unwrap_or("?"),
        r["progress"]["done"],
        r["progress"]["total"],
        r["status"].as_str().unwrap_or("?")
    );
    o += "## Systems\n\n| System | n | Accuracy | 95% CI | Escalated | Mean wall (s) | p95 wall (s) | Reasoning tok |\n|---|---|---|---|---|---|---|---|\n";
    if let Some(sys) = r["systems"].as_object() {
        for (k, v) in sys {
            let ci = v["accuracy_ci95"]
                .as_array()
                .map(|a| format!("{}–{}", pc(&a[0]), pc(&a[1])))
                .unwrap_or_else(|| "-".into());
            o += &format!(
                "| {k} | {} | {} | {ci} | {} | {} | {} | {} |\n",
                v["n_scored"],
                pc(&v["accuracy"]),
                pc(&v["escalation_rate"]),
                f(&json!(v["mean_wall_ms"].as_f64().map(|x| x / 1000.0)), 1),
                f(&json!(v["p95_wall_ms"].as_f64().map(|x| x / 1000.0)), 1),
                f(&v["mean_reasoning_tokens"], 0)
            );
        }
    }
    o += "\n## Paired comparisons (FR-051)\n\n| A − B | Pairs | Δ accuracy | 95% CI | McNemar p | Bonferroni p | Non-inferior (3 pt) | Cost ratio |\n|---|---|---|---|---|---|---|---|\n";
    for p in r["paired"].as_array().cloned().unwrap_or_default() {
        let ci = p["ci95"]
            .as_array()
            .map(|a| format!("[{}, {}]", f(&a[0], 3), f(&a[1], 3)))
            .unwrap_or_else(|| "-".into());
        o += &format!(
            "| {} − {} | {} | {} | {ci} | {} | {} | {} | {} |\n",
            p["a"].as_str().unwrap_or(""),
            p["b"].as_str().unwrap_or(""),
            p["n_pairs"],
            f(&p["delta_accuracy"], 3),
            f(&p["mcnemar_p"], 4),
            f(&p["p_bonferroni"], 4),
            p["non_inferior_3pt"],
            f(&p["cost_ratio"], 2)
        );
    }
    if let Some(rt) = r["routing"].as_object() {
        o += "\n## Routing against the Theorem 2 oracle\n\n| System | n | Routing accuracy | Escalation rate | Measures the gate |\n|---|---|---|---|---|\n";
        for (k, v) in rt {
            o += &format!(
                "| {k} | {} | {} | {} | {} |\n",
                v["n"],
                pc(&v["routing_accuracy"]),
                pc(&v["escalation_rate"]),
                v["measures_gate"]
            );
        }
    }
    if let Some(fr) = r["frontier"].as_array() {
        o += "\n## Accuracy–cost frontier (Prop. 2)\n\n| Policy | Accuracy | Mean cost (s) | vs deliberate-only | Pareto |\n|---|---|---|---|---|\n";
        for p in fr {
            o += &format!(
                "| {} | {} | {} | {} | {} |\n",
                p["system"].as_str().unwrap_or(""),
                pc(&p["accuracy"]),
                f(&p["mean_cost_s"], 1),
                f(&p["cost_vs_slow"], 2),
                p["pareto"]
            );
        }
    }
    if !r["calibration"].is_null() {
        let c = &r["calibration"];
        o += &format!(
            "\n## Calibration of e1_hat\n\nn = {}, ECE = {}, Brier = {}, draft error rate = {}.\n",
            c["n"],
            f(&c["ece_e1"], 3),
            f(&c["brier_e1"], 3),
            pc(&c["draft_error_rate"])
        );
    }
    if !r["coupling"].is_null() {
        o += "\n## Coupling test (ER-004)\n\n";
        for k in ["draft_blind", "draft_conditioned"] {
            let c = &r["coupling"][k];
            if !c.is_null() {
                o += &format!(
                    "- {k}: {} of {} wrong drafts adopted ({}; 95% upper bound {}).\n",
                    c["adopted"],
                    c["wrong_drafts"],
                    pc(&c["adoption_rate"]),
                    pc(&c["upper95"])
                );
            }
        }
    }
    o += "\nSmall runs are indicative only: G2 needs >= 700 pairs, ER-005 >= 2,900 calibration samples, ER-004 >= 200 wrong drafts per mode.\n";
    o
}

/// Item-level CSV export.
pub fn items_csv(r: &Value) -> String {
    let cols = [
        "task_id",
        "system",
        "domain",
        "correct",
        "mode",
        "assurance",
        "escalated",
        "wall_ms",
        "ttft_ms",
        "model_calls",
        "reasoning_tokens",
        "e1_hat",
        "draft_correct",
        "rule",
        "error",
    ];
    let mut o = cols.join(",") + "\n";
    for it in r["items"].as_array().cloned().unwrap_or_default() {
        let row: Vec<String> = cols
            .iter()
            .map(|c| {
                let v = &it[*c];
                let s = match v {
                    Value::String(x) => x.clone(),
                    Value::Null => String::new(),
                    x => x.to_string(),
                };
                if s.contains([',', '"', '\n']) {
                    format!("\"{}\"", s.replace('"', "\"\""))
                } else {
                    s
                }
            })
            .collect();
        o += &(row.join(",") + "\n");
    }
    o
}

/// Lab insights: pool every stored run to answer "which System 1 / System 2 policy is best, where, and at
/// what price of latency?". Policies: fast (System 1 only), slow (System 2 only), gated (the Aql Gate with
/// the verifier visible) and blind (the gate routing on its own estimate). For each domain: accuracy with
/// an exact interval, mean cost, the paired value of deliberation (slow minus fast on the same tasks), and
/// the policy that minimises L(1 - accuracy) + lambda * cost / cost_unit over a lambda grid (Prop. 2).
pub fn insights(
    runs: &[Value],
    orch: &Orchestrator,
    lambda: Option<f32>,
    prefs: &[Value],
) -> Value {
    let cfg = &orch.cfg;
    let policy = |sys: &str| -> Option<&'static str> {
        match sys {
            "fast_only" => Some("fast"),
            "slow_only" => Some("slow"),
            _ => match gated(sys) {
                Some((false, _)) => Some("gated"),
                Some((true, _)) => Some("blind"),
                None => None,
            },
        }
    };
    // (domain, policy) -> (n, correct, cost seconds)
    let mut cells: BTreeMap<(String, &'static str), (u64, u64, f64)> = BTreeMap::new();
    // domain -> paired (slow - fast) accuracy diffs and cost diffs
    let mut paired: BTreeMap<String, (Vec<f64>, Vec<f64>)> = BTreeMap::new();
    let mut n_runs = 0;
    for r in runs {
        let Ok(items) = serde_json::from_value::<Vec<ItemResult>>(r["items"].clone()) else {
            continue;
        };
        n_runs += 1;
        for it in items
            .iter()
            .filter(|i| i.error.is_none() && i.correct.is_some())
        {
            if let Some(p) = policy(&it.system) {
                let c = cells.entry((it.domain.clone(), p)).or_insert((0, 0, 0.0));
                c.0 += 1;
                c.1 += (it.correct == Some(true)) as u64;
                c.2 += it.wall_ms as f64 / 1000.0;
            }
        }
        let by = |s: &str| {
            items
                .iter()
                .filter(|i| i.system == s && i.correct.is_some() && i.error.is_none())
                .map(|i| (i.task_id.clone(), i))
                .collect::<BTreeMap<_, _>>()
        };
        let (f, sl) = (by("fast_only"), by("slow_only"));
        for (k, a) in &f {
            if let Some(b) = sl.get(k) {
                let e = paired.entry(a.domain.clone()).or_default();
                e.0.push(
                    (b.correct == Some(true)) as i32 as f64
                        - (a.correct == Some(true)) as i32 as f64,
                );
                e.1.push((b.wall_ms as f64 - a.wall_ms as f64) / 1000.0);
            }
        }
    }
    let grid = [0.0f32, 0.05, 0.1, 0.2, 0.4, 0.8];
    // f32 -> JSON goes through f64 (0.05 becomes 0.05000000074505806); round for display and clients.
    let r4 = |x: f32| (x as f64 * 1e4).round() / 1e4;
    let lam_now = lambda.unwrap_or(cfg.gate.lambda_default);
    let domains: std::collections::BTreeSet<String> = cells.keys().map(|k| k.0.clone()).collect();
    let rows: Vec<Value> = domains
        .iter()
        .map(|d| {
            let pol: Vec<Value> = ["fast", "gated", "blind", "slow"]
                .iter()
                .filter_map(|p| cells.get(&(d.clone(), *p)).map(|c| (p, c)))
                .map(|(p, (n, k, cost))| {
                    let (lo, hi) = clopper_pearson(*k, *n);
                    json!({"policy": p, "n": n, "accuracy": *k as f64 / *n as f64, "ci95": [lo, hi], "mean_cost_s": cost / *n as f64})
                })
                .collect();
            let best_at = |lam: f32| -> Option<Value> {
                pol.iter()
                    .filter(|p| p["n"].as_u64().unwrap_or(0) >= 5)
                    .map(|p| {
                        let loss = cfg.gate.loss_wrong as f64 * (1.0 - p["accuracy"].as_f64().unwrap_or(0.0)) + lam as f64 * p["mean_cost_s"].as_f64().unwrap_or(0.0) / cfg.gate.cost_unit_s as f64;
                        (loss, p)
                    })
                    .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
                    .map(|(loss, p)| json!({"lambda": r4(lam), "policy": p["policy"], "expected_loss": loss, "n": p["n"], "confidence": if p["n"].as_u64().unwrap_or(0) >= 30 { "moderate" } else { "low" }}))
            };
            let value = paired.get(d).map(|(da, dc)| {
                let m = da.iter().sum::<f64>() / da.len() as f64;
                let ci = bootstrap(da, 3, 1000);
                json!({"pairs": da.len(), "delta_accuracy": m, "ci95": ci.map(|(l, h)| [l, h]), "extra_cost_s": dc.iter().sum::<f64>() / dc.len() as f64})
            });
            let pref = prefs.iter().find(|p| p["domain"].as_str() == Some(d)).cloned();
            json!({"domain": d, "policies": pol, "value_of_deliberation": value, "human_preference": pref,
                "policy_map": grid.iter().filter_map(|l| best_at(*l)).collect::<Vec<_>>(), "recommended": best_at(lam_now)})
        })
        .collect();
    json!({"runs": n_runs, "lambda": lam_now, "lambda_grid": grid.iter().map(|l| r4(*l)).collect::<Vec<_>>(), "cost_unit_s": cfg.gate.cost_unit_s, "domains": rows,
        "note": "Pooled over every stored evaluation run. A policy needs at least 5 scored items to be recommended; 'low' confidence means fewer than 30. The policy map shows which interaction of System 1 and System 2 minimises L(1 - accuracy) + lambda * cost at each price of latency (Prop. 2)."})
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn beta_geometric_recovers_parameters() {
        // Simulate tasks with a ~ Beta(2, 3), up to 4 attempts each.
        let mut rng = fastrand::Rng::with_seed(5);
        let mut obs = vec![];
        for _ in 0..3000 {
            // Beta(2,3) as the 2nd smallest of 4 uniforms.
            let mut u: Vec<f64> = (0..4).map(|_| rng.f64()).collect();
            u.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let a = u[1];
            let mut first = None;
            for j in 1..=4 {
                if rng.f64() < a {
                    first = Some(j);
                    break;
                }
            }
            obs.push((first, 4));
        }
        let (a, b, _) = fit_beta_geometric(&obs).unwrap();
        assert!((a / (a + b) - 0.4).abs() < 0.05, "mean {}", a / (a + b));
        assert!(
            a > 1.0 && a < 4.0 && b > 1.5 && b < 6.0,
            "alpha {a} beta {b}"
        );
        // Prop. 3: alpha L / c - alpha - beta with c/L = 0.15 -> ceil(13.3 - 5) = 9, clipped at 5.
        assert_eq!(optimal_draws(2.0, 3.0, 0.15, 5), 5);
        assert_eq!(optimal_draws(2.0, 3.0, 0.5, 9), 1);
    }
    fn orch() -> (Orchestrator, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!("nafs-eval-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut cfg = nafs_core::Config::load(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../config/nafs-aql.toml"
        ))
        .unwrap();
        cfg.gate.calibration_file = dir.join("cal.json").display().to_string();
        cfg.gate.params_file = dir.join("params.json").display().to_string();
        let store = std::sync::Arc::new(
            nafs_memory::Store::open(&dir.join("db.sqlite"), &dir.join("t.jsonl")).unwrap(),
        );
        (Orchestrator::new(std::sync::Arc::new(cfg), store), dir)
    }
    fn item(
        task: &str,
        sys: &str,
        correct: bool,
        escalated: bool,
        draft_correct: Option<bool>,
        key: &str,
        draft_key: Option<&str>,
    ) -> ItemResult {
        let mut it = error_item(
            &Task {
                id: task.into(),
                domain: "math".into(),
                prompt: String::new(),
                verification: Default::default(),
                source: String::new(),
                license: String::new(),
            },
            sys,
            String::new(),
        );
        it.error = None;
        it.correct = Some(correct);
        it.escalated = escalated;
        it.mode = if escalated {
            "slow".into()
        } else {
            "fast".into()
        };
        it.draft_correct = draft_correct;
        it.answer_key = key.into();
        it.draft_key = draft_key.map(String::from);
        it.wall_ms = if escalated { 120_000 } else { 5_000 };
        it.e1_hat = if draft_correct == Some(false) {
            0.8
        } else {
            0.1
        };
        it
    }
    #[test]
    #[allow(clippy::vec_init_then_push)]
    fn report_measures_routing_coupling_and_budget_curve() {
        let (o, dir) = orch();
        let mut items = vec![];
        // t1: fast wrong, slow right -> escalation needed; t2: both right -> keep.
        items.push(item(
            "t1",
            "fast_only",
            false,
            false,
            Some(false),
            "num:3",
            Some("num:3"),
        ));
        items.push(item("t1", "slow_only", true, true, None, "num:4", None));
        items.push(item(
            "t2",
            "fast_only",
            true,
            false,
            Some(true),
            "num:7",
            Some("num:7"),
        ));
        items.push(item("t2", "slow_only", true, true, None, "num:7", None));
        // Blind gate: escalates t1 (correct), keeps t2 (correct).
        items.push(item(
            "t1",
            "nafs_blind",
            true,
            true,
            Some(false),
            "num:4",
            Some("num:3"),
        ));
        items.push(item(
            "t2",
            "nafs_blind",
            true,
            false,
            Some(true),
            "num:7",
            Some("num:7"),
        ));
        // Conditioned System 2 adopts the wrong draft on t1.
        items.push(item(
            "t1",
            "slow_conditioned",
            false,
            true,
            Some(false),
            "num:3",
            Some("num:3"),
        ));
        // Budget sweep on t1.
        items.push(item("t1", "slow_b256", false, true, None, "num:5", None));
        items.push(item("t1", "slow_b1024", true, true, None, "num:4", None));
        let req = EvalRunRequest {
            benchmark: "x".into(),
            systems: vec![
                "nafs_blind".into(),
                "fast_only".into(),
                "slow_only".into(),
                "slow_conditioned".into(),
                "slow_b256".into(),
                "slow_b1024".into(),
            ],
            max_items: None,
            offset: None,
            seed: Some(1),
            lambda: Some(0.2),
            thinking_budget: None,
            shuffle: None,
        };
        let r = report(
            &req,
            &items,
            items.len(),
            chrono::Utc::now(),
            "done",
            &o,
            true,
        );
        let rt = &r["routing"]["nafs_blind"];
        assert_eq!(rt["routing_accuracy"], 1.0);
        assert_eq!(rt["measures_gate"], true);
        assert_eq!(r["coupling"]["draft_blind"]["wrong_drafts"], 1);
        assert_eq!(r["coupling"]["draft_blind"]["adopted"], 0);
        assert_eq!(r["coupling"]["draft_conditioned"]["adopted"], 1);
        assert_eq!(r["calibration"]["n"], 2);
        let bc = &r["budget_curve"]["math"];
        assert_eq!(bc["points"].as_array().unwrap().len(), 2);
        assert_eq!(
            bc["b_star"], 1024,
            "the larger budget fixes the error at a small extra cost"
        );
        assert!(r["frontier"]
            .as_array()
            .unwrap()
            .iter()
            .any(|p| p["pareto"] == true));
        assert_eq!(gated("nafs_blind_l05"), Some((true, Some(0.05))));
        assert_eq!(gated("nafs_aql_l101"), None);
        assert!(r["paired"]
            .as_array()
            .unwrap()
            .iter()
            .any(|p| p["a"] == "nafs_blind" && p["b"] == "slow_only"));
        let _ = std::fs::remove_dir_all(dir);
    }
    #[test]
    fn mcnemar_matches_known_values() {
        assert!((mcnemar_exact(0, 0) - 1.0).abs() < 1e-12);
        // b = 1, c = 9: two-sided exact p = 2 * (1 + 10) / 1024 = 0.021484375
        assert!((mcnemar_exact(1, 9) - 0.021484375).abs() < 1e-9);
        assert!((mcnemar_exact(5, 5) - 1.0).abs() < 1e-9);
    }
    #[test]
    fn system_names_are_validated() {
        let mut r = EvalRunRequest {
            benchmark: "x".into(),
            systems: vec!["nafs_aql".into(), "slow_b256".into()],
            max_items: Some(3),
            offset: None,
            seed: None,
            lambda: None,
            thinking_budget: None,
            shuffle: None,
        };
        assert!(r.validate().is_ok());
        r.systems = vec!["<img src=x onerror=alert(1)>".into()];
        assert!(r.validate().is_err());
        r.systems = vec!["slow_b99999".into()];
        assert!(r.validate().is_err());
    }
}
