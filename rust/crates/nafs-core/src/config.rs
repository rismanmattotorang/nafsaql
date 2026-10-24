//! Configuration (OR-006): one TOML file, environment overrides, SHA-256 config hash in every trace.
use nafs_types::{Domain, NafsError, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub schema_version: String,
    pub server: ServerCfg,
    pub assistant: AssistantCfg,
    pub backend: BackendSelect,
    pub backends: BTreeMap<String, BackendSet>,
    #[serde(default)]
    pub backend_retry: RetryCfg,
    pub gate: GateCfg,
    pub deliberation: DeliberationCfg,
    #[serde(default)]
    pub verifiers: VerifierCfg,
    pub storage: StorageCfg,
    #[serde(default)]
    pub eval: EvalCfg,
    #[serde(default)]
    pub judge: JudgeCfg,
    #[serde(default)]
    pub audit: AuditCfg,
    #[serde(default)]
    pub alerts: AlertsCfg,
    #[serde(default)]
    pub retrieval: RetrievalCfg,
    /// Math and code harness (FR-090..FR-093): the code agent endpoint.
    #[serde(default)]
    pub harness: HarnessCfg,
    #[serde(skip)]
    pub hash: String,
}

/// The code agent (FR-093, `POST /v1/harness/code`): Codex's turn loop with Open Math Harness's stop rule,
/// on System 2 by default, in the verifier sandbox.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarnessCfg {
    #[serde(default = "d_true")]
    pub code_agent_enabled: bool,
    /// Model calls per run.
    #[serde(default = "d_agent_steps")]
    pub max_steps: u32,
    /// "system2" (reasoning) or "system1" (fast).
    #[serde(default = "d_agent_model")]
    pub model: String,
    /// Reasoning tokens per step on System 2.
    #[serde(default = "d_agent_think")]
    pub thinking_budget: u32,
    /// Answer tokens per step (a patch or a whole file).
    #[serde(default = "d_agent_tokens")]
    pub step_max_tokens: u32,
    /// Wall-clock cap of one run.
    #[serde(default = "d_agent_wall")]
    pub wall_clock_s: u64,
    /// Exchanges kept after each context reset (OMH keeps 4).
    #[serde(default = "d_agent_keep")]
    pub keep_exchanges: usize,
}
fn d_agent_steps() -> u32 {
    6
}
fn d_agent_model() -> String {
    "system2".into()
}
fn d_agent_think() -> u32 {
    512
}
fn d_agent_tokens() -> u32 {
    768
}
fn d_agent_wall() -> u64 {
    900
}
fn d_agent_keep() -> usize {
    3
}
impl Default for HarnessCfg {
    fn default() -> Self {
        Self {
            code_agent_enabled: true,
            max_steps: d_agent_steps(),
            model: d_agent_model(),
            thinking_budget: d_agent_think(),
            step_max_tokens: d_agent_tokens(),
            wall_clock_s: d_agent_wall(),
            keep_exchanges: d_agent_keep(),
        }
    }
}

/// Rubric judge (FR-034, SR-003): System 2 scores honesty h, harm-avoidance a, helpfulness u, grounding m.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgeCfg {
    pub enabled: bool,
    /// Reasoning tokens the (thinking-only) System 2 model may spend before scoring.
    pub thinking_budget: u32,
    pub max_tokens: u32,
    pub floor: f32,
    pub pass_v: f32,
    /// Judge-triggered retries on non-verifiable tasks (R <= 1, FR-022).
    pub max_retries: u32,
    /// Which model scores the rubric: "system2" (default) or "system1" (P7-T09: the judge is the largest
    /// fixed cost on open questions; the rates harness measures the smaller judge's r and phi).
    #[serde(default = "d_judge_backend")]
    pub backend: String,
}
fn d_judge_backend() -> String {
    "system2".into()
}
impl Default for JudgeCfg {
    fn default() -> Self {
        Self {
            enabled: true,
            thinking_budget: 64,
            max_tokens: 160,
            floor: 0.5,
            pass_v: 0.7,
            max_retries: 1,
            backend: d_judge_backend(),
        }
    }
}

/// Alert rules (OR-004), evaluated by the gateway over a recent window and exported as metrics; the
/// same thresholds appear in deploy/monitoring/alerts.yml for Prometheus.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertsCfg {
    pub window_min: i64,
    pub min_samples: usize,
    /// NFR-003: a 150-token fast-path reply within 12 s at p95.
    pub fast_p95_ms: f64,
    pub error_rate: f64,
    pub escalation_rate_max: f64,
    pub verifier_acceptance_min: f64,
}
impl Default for AlertsCfg {
    fn default() -> Self {
        Self {
            window_min: 60,
            min_samples: 10,
            fast_p95_ms: 12_000.0,
            error_rate: 0.05,
            escalation_rate_max: 0.60,
            verifier_acceptance_min: 0.30,
        }
    }
}

/// Optional retrieval memory (FR-061): prior verified answers, retrieved lexically (BM25) and
/// inserted as delimited data with provenance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalCfg {
    pub enabled: bool,
    pub top_k: usize,
    /// Minimum BM25 score relative to the best possible self-match (0..1).
    pub min_score: f64,
}
impl Default for RetrievalCfg {
    fn default() -> Self {
        Self {
            enabled: false,
            top_k: 2,
            min_score: 0.35,
        }
    }
}

/// Audit stream (FR-040) and self-model (Prop. 5).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditCfg {
    pub enabled: bool,
    /// Run audits only when no user request is in flight.
    pub idle_only: bool,
    /// Seconds between idle checks of the audit worker.
    pub poll_s: u64,
    /// Domains with fewer audits than this are reported as unmeasured (FR-041).
    pub min_audits_measured: u32,
}
impl Default for AuditCfg {
    fn default() -> Self {
        Self {
            enabled: true,
            idle_only: true,
            poll_s: 15,
            min_audits_measured: 30,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerCfg {
    pub bind: String,
    #[serde(default = "d_auth_env")]
    pub auth_tokens_env: String,
    #[serde(default = "d_rpm")]
    pub rate_limit_per_token_rpm: u32,
    #[serde(default = "d_cap")]
    pub request_wall_clock_cap_s: u64,
    #[serde(default = "d_prompt")]
    pub max_prompt_tokens: usize,
}
fn d_auth_env() -> String {
    "NAFS_AUTH_TOKENS".into()
}
fn d_rpm() -> u32 {
    120
}
fn d_cap() -> u64 {
    600
}
fn d_prompt() -> usize {
    4096
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistantCfg {
    pub system_prompt: String,
    #[serde(default = "d_profile")]
    pub profile: String,
}
fn d_profile() -> String {
    "A".into()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendSelect {
    pub active: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendSet {
    pub label: String,
    pub kind: String,
    pub system1: EndpointCfg,
    pub system2: EndpointCfg,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointCfg {
    pub url: String,
    pub model: String,
    pub hf_id: String,
    #[serde(default)]
    pub quantization: String,
    #[serde(default)]
    pub thinking: bool,
    #[serde(default = "d_inflight")]
    pub max_in_flight: usize,
}
fn d_inflight() -> usize {
    2
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryCfg {
    pub max_attempts: u32,
    pub base_backoff_ms: u64,
    pub max_backoff_ms: u64,
    pub connect_timeout_ms: u64,
    pub stream_idle_timeout_ms: u64,
    pub queue_timeout_ms: u64,
}
impl Default for RetryCfg {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            base_backoff_ms: 200,
            max_backoff_ms: 2000,
            connect_timeout_ms: 2000,
            stream_idle_timeout_ms: 120_000,
            queue_timeout_ms: 600_000,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateCfg {
    pub mode: String, // fixed | calibrated
    pub k_samples: u8,
    pub sample_max_tokens: u32,
    pub sample_temperature: f32,
    pub lambda_default: f32,
    pub loss_wrong: f32,
    /// Seconds of extra latency that count as one unit of cost (lambda is the price per unit).
    pub cost_unit_s: f32,
    pub calibration_file: String,
    pub calibration_max_ece: f32,
    pub e2_hat: BTreeMap<String, f32>,
    pub delta_c_s: BTreeMap<String, f32>,
    pub audit_fraction: f32,
    pub fixed: FixedRuleCfg,
    /// NFR-020 / ER-005: minimum held-out labelled samples before a calibration may be promoted.
    #[serde(default = "d_min_n")]
    pub calibration_min_n: usize,
    /// Minimum examples of each class (draft right / wrong) for a fit to be attempted (H1).
    #[serde(default = "d_min_class")]
    pub calibration_min_per_class: usize,
    /// FR-014 budget mode: target escalation rate B, and the DKW accuracy (epsilon, delta) of lambda*.
    #[serde(default = "d_budget")]
    pub escalation_budget: f32,
    #[serde(default = "d_eps")]
    pub budget_epsilon: f32,
    #[serde(default = "d_delta")]
    pub budget_delta: f32,
    /// FR-015: hours between automatic recalibrations from the ledger (0 = manual only).
    #[serde(default)]
    pub recalibrate_every_h: u64,
    /// Measured Phase-0 parameters (e2_hat, delta_c, caps, budgets); overrides the placeholders above.
    #[serde(default = "d_params")]
    pub params_file: String,
    /// Sample length for math and code, whose step-by-step answers exceed sample_max_tokens (M11).
    #[serde(default = "d_hard_tokens")]
    pub sample_max_tokens_hard: u32,
    /// Draw the k cheap samples even when an exact verifier will decide (costs S1 time; only useful
    /// to collect calibration features from live verifiable traffic).
    #[serde(default)]
    pub samples_on_verifiable: bool,
    /// P7-T02 sequential sampling: under the fixed rule, draw the k cheap samples only after the draft and
    /// only when the draft's mean log-prob lies within `uncertain_band` of the threshold. Confident drafts
    /// skip them (three sequences on four cores tripled the factual draft time, benchmarks/samples_ab_factual.json).
    #[serde(default = "d_true")]
    pub samples_when_uncertain: bool,
    #[serde(default = "d_band")]
    pub uncertain_band: f32,
    /// P7-T03 early escalation: under the fixed rule, stop an unverifiable draft once this many tokens
    /// have a running mean log-prob below the threshold minus `early_escalation_margin` (0 = off).
    #[serde(default = "d_early_tokens")]
    pub early_escalation_tokens: u32,
    #[serde(default = "d_early_margin")]
    pub early_escalation_margin: f32,
    /// Phase 8: window (tokens) of the lowest-window confidence signal (`low_window_logprob`).
    #[serde(default = "d_window")]
    pub confidence_window: u32,
    /// Phase 8: domains whose unverified drafts always get the k agreement samples (after the draft), whatever
    /// the draft's own confidence. Measured on mathematics: token confidence does not predict a wrong draft
    /// (AUROC 0.56-0.65) but disagreement with the cheap samples does (benchmarks/confidence_signal_study.json).
    #[serde(default = "d_samples_domains")]
    pub samples_on_unverified_domains: Vec<String>,
}
fn d_samples_domains() -> Vec<String> {
    vec!["math".into()]
}
fn d_window() -> u32 {
    16
}
fn d_band() -> f32 {
    0.10
}
fn d_early_tokens() -> u32 {
    32
}
fn d_early_margin() -> f32 {
    0.04
}
fn d_min_n() -> usize {
    2900
}
fn d_min_class() -> usize {
    30
}
fn d_budget() -> f32 {
    0.30
}
fn d_eps() -> f32 {
    0.025
}
fn d_delta() -> f32 {
    0.05
}
fn d_params() -> String {
    "data/runtime/gate_params.json".into()
}
fn d_hard_tokens() -> u32 {
    384
}
impl GateCfg {
    /// DKW sample size for estimating the escalation quantile to +-epsilon with prob. 1-delta (Prop. 6).
    pub fn dkw_n(&self) -> usize {
        ((2.0 / self.budget_delta as f64).ln() / (2.0 * (self.budget_epsilon as f64).powi(2)))
            .ceil() as usize
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixedRuleCfg {
    pub mean_logprob_threshold: f32,
    pub agreement_threshold: f32,
    pub escalate_open_domains: bool,
    /// Phase 8: escalate when the lowest `confidence_window`-token window of the draft has a mean log-prob
    /// below this (a local collapse of confidence). Off when absent. Because the signal is a running
    /// minimum, the draft is stopped the moment a window crosses it, with the same decision the full
    /// draft would get (Prop. N2).
    #[serde(default)]
    pub low_window_logprob_threshold: Option<f32>,
    /// Domains the window rule applies to. The signal's scale is domain-specific (open prose has a median
    /// lowest window of -0.32 against about -0.1 for math drafts), so the rule runs only where its
    /// threshold was measured.
    #[serde(default = "d_low_window_domains")]
    pub low_window_domains: Vec<String>,
}
fn d_low_window_domains() -> Vec<String> {
    vec!["math".into()]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeliberationCfg {
    pub draft_blind: bool,
    pub draft_max_tokens: u32,
    pub draft_temperature: f32,
    pub slow_temperature: f32,
    pub answer_max_tokens: u32,
    pub non_verifiable_max_rounds: u32,
    pub caps: BTreeMap<String, Caps>,
    #[serde(default)]
    pub caps_provenance: String,
    /// c_p / L_a: cost of one System 2 draw relative to the loss of failing to answer (Prop. 3, FR-021).
    #[serde(default = "d_cost_ratio")]
    pub cost_ratio: f32,
    /// Upper bound on derived R so a Beta fit cannot create unbounded loops (Lemma 1).
    #[serde(default = "d_max_r")]
    pub max_rounds_cap: u32,
    /// Permit the draft-conditioned ablation (ER-004). Never used for user traffic.
    #[serde(default = "d_true")]
    pub allow_draft_conditioned_ablation: bool,
    /// P7-T08: probe-and-verify early exit from System 2 reasoning.
    #[serde(default)]
    pub adaptive_exit: AdaptiveExitCfg,
    /// FR-086: the thinking budget of verified retry round r is `budget * decay^r`, never below
    /// `retry_min_budget`. The verifier's evidence makes a retry a repair, not a fresh search, so it
    /// needs less reasoning; 1.0 keeps the same budget on every round.
    #[serde(default = "d_retry_decay")]
    pub retry_budget_decay: f32,
    #[serde(default = "d_retry_min_budget")]
    pub retry_min_budget: u32,
    /// FR-091: which candidate a verified retry repairs. "best" keeps a candidate tree scored by the
    /// verifier (Open Math Harness's scoring protocol, per-assert credit on code) and repairs the best one;
    /// with all-or-nothing verdicts that is the latest, so "best" equals "last" unless credit differs.
    #[serde(default = "d_repair_policy")]
    pub repair_policy: String,
    /// FR-091: repairs of one candidate before the search moves on.
    #[serde(default = "d_repair_fanout")]
    pub repair_fanout: u32,
    /// FR-092: on code tasks a retry may answer with a Codex patch to the candidate instead of the whole
    /// program (fewer decode tokens). Whole code is always accepted. Off until measured (rule 3).
    #[serde(default)]
    pub patch_repair: bool,
}
fn d_repair_policy() -> String {
    "best".into()
}
fn d_repair_fanout() -> u32 {
    2
}
fn d_retry_decay() -> f32 {
    1.0
}
fn d_retry_min_budget() -> u32 {
    96
}
/// Adaptive reasoning exit (P7-T08, FR-082). System 2 reasons in chunks; at reflection signals ("Wait",
/// "Alternatively", ...) and at the latest every `probe_every_tokens`, the reasoning is closed provisionally
/// and a short answer is probed. On a verifiable task the exact verifier decides whether to stop; otherwise
/// `consistency` consecutive agreeing probes do (Dynasor-CoT). The final answer is then generated after the
/// shortened reasoning, exactly as on the standard path. llama-server only (raw completions with a cached prefix).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveExitCfg {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "d_probe_every")]
    pub probe_every_tokens: u32,
    #[serde(default = "d_min_gap")]
    pub min_gap_tokens: u32,
    #[serde(default = "d_probe_max")]
    pub probe_max_tokens: u32,
    #[serde(default = "d_consistency")]
    pub consistency: u32,
    #[serde(default = "d_ax_domains")]
    pub domains: Vec<String>,
    /// Phase 8: non-verifiable stopping rule, "consistency" (k consecutive agreeing probes, above) or
    /// "posterior": stop when the probability that the plurality probe answer is the majority answer
    /// (Dirichlet(1) posterior over all probes so far) is at least a bound that collapses linearly from
    /// `posterior_stop` to `posterior_stop_end` over the reasoning budget (Frazier-Yu deadline rule).
    #[serde(default = "d_exit_rule")]
    pub exit_rule: String,
    #[serde(default = "d_posterior_stop")]
    pub posterior_stop: f32,
    #[serde(default = "d_posterior_stop_end")]
    pub posterior_stop_end: f32,
    /// FR-085: on tasks whose exact verifier needs the whole answer (code tests, JSON Schema) the probe
    /// is a full answer; when it passes, it is the final answer and the answer stage is skipped.
    #[serde(default = "d_true")]
    pub reuse_probe_answer: bool,
    /// Full-answer probes cost an answer each, so they are rarer than short ones: at the latest every
    /// `full_probe_every_tokens` and never sooner than `full_probe_min_gap_tokens` after the previous one.
    #[serde(default = "d_full_probe_every")]
    pub full_probe_every_tokens: u32,
    #[serde(default = "d_full_probe_min_gap")]
    pub full_probe_min_gap_tokens: u32,
    /// Longest full-answer probe. A longer answer cannot pass a probe (it is only truncated), but the
    /// final answer stage still has `answer_max_tokens`; this bounds the probe overhead on items that
    /// never pass.
    #[serde(default = "d_full_probe_max")]
    pub full_probe_max_tokens: u32,
}
impl Default for AdaptiveExitCfg {
    fn default() -> Self {
        Self {
            enabled: false,
            probe_every_tokens: d_probe_every(),
            min_gap_tokens: d_min_gap(),
            probe_max_tokens: d_probe_max(),
            consistency: d_consistency(),
            domains: d_ax_domains(),
            exit_rule: d_exit_rule(),
            posterior_stop: d_posterior_stop(),
            posterior_stop_end: d_posterior_stop_end(),
            reuse_probe_answer: true,
            full_probe_every_tokens: d_full_probe_every(),
            full_probe_min_gap_tokens: d_full_probe_min_gap(),
            full_probe_max_tokens: d_full_probe_max(),
        }
    }
}
fn d_full_probe_every() -> u32 {
    512
}
fn d_full_probe_min_gap() -> u32 {
    192
}
fn d_full_probe_max() -> u32 {
    384
}
fn d_exit_rule() -> String {
    "consistency".into()
}
fn d_posterior_stop() -> f32 {
    0.85
}
fn d_posterior_stop_end() -> f32 {
    0.70
}
fn d_probe_every() -> u32 {
    256
}
fn d_min_gap() -> u32 {
    64
}
fn d_probe_max() -> u32 {
    24
}
fn d_consistency() -> u32 {
    2
}
fn d_ax_domains() -> Vec<String> {
    vec!["math".into(), "factual".into()]
}
fn d_cost_ratio() -> f32 {
    0.15
}
fn d_max_r() -> u32 {
    4
}
fn d_true() -> bool {
    true
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Caps {
    #[serde(rename = "N")]
    pub n: u32,
    #[serde(rename = "R")]
    pub r: u32,
    pub budget: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifierCfg {
    pub python: String,
    pub cpu_time_s: u64,
    pub wall_clock_s: u64,
    pub deny_network: bool,
    pub numeric_tolerance: f64,
    /// Address-space / RSS cap for verifier subprocesses, MiB (FR-033).
    #[serde(default = "d_mem")]
    pub memory_mb: u64,
    /// Interpreter with SymPy for symbolic math equivalence; empty disables the symbolic verifier.
    #[serde(default)]
    pub sympy_python: String,
    /// Mutation-measured (r, phi) per verifier (FR-035); loaded at startup when present.
    #[serde(default = "d_rates")]
    pub rates_file: String,
    /// FR-090: the `lean` executable (absolute path, e.g. ~/.elan/toolchains/<v>/bin/lean); empty disables
    /// Lean proof checking and requests with `verification.lean` are refused.
    #[serde(default)]
    pub lean: String,
    /// `LEAN_PATH` for a Mathlib build (empty: core Lean only).
    #[serde(default)]
    pub lean_path: String,
    #[serde(default = "d_lean_timeout")]
    pub lean_timeout_s: u64,
    #[serde(default = "d_lean_mem")]
    pub lean_memory_mb: u64,
}
fn d_lean_timeout() -> u64 {
    120
}
fn d_lean_mem() -> u64 {
    4096
}
fn d_mem() -> u64 {
    512
}
fn d_rates() -> String {
    "data/runtime/verifier_rates.json".into()
}
impl Default for VerifierCfg {
    fn default() -> Self {
        Self {
            python: "python3".into(),
            cpu_time_s: 10,
            wall_clock_s: 30,
            deny_network: true,
            numeric_tolerance: 1e-6,
            memory_mb: d_mem(),
            sympy_python: String::new(),
            rates_file: d_rates(),
            lean: String::new(),
            lean_path: String::new(),
            lean_timeout_s: d_lean_timeout(),
            lean_memory_mb: d_lean_mem(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalCfg {
    pub datasets_dir: String,
}
impl Default for EvalCfg {
    fn default() -> Self {
        Self {
            datasets_dir: "data/bench".into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageCfg {
    pub sqlite_path: String,
    pub trace_path: String,
    /// SR-004: conversations, replies and feedback older than this are deleted (0 = keep forever).
    #[serde(default = "d_retention")]
    pub retention_days: u32,
    /// Model registry written by the Tazkiya promotion tool (OR-005).
    #[serde(default = "d_registry")]
    pub registry_file: String,
}
fn d_retention() -> u32 {
    90
}
fn d_registry() -> String {
    "config/models.registry.json".into()
}

impl Config {
    pub fn load(path: &str) -> Result<Self> {
        let raw =
            std::fs::read_to_string(path).map_err(|e| NafsError::Config(format!("{path}: {e}")))?;
        let mut cfg: Config =
            toml::from_str(&raw).map_err(|e| NafsError::Config(format!("{path}: {e}")))?;
        if let Ok(b) = std::env::var("NAFS_BIND") {
            cfg.server.bind = b;
        }
        if let Ok(b) = std::env::var("NAFS_BACKEND") {
            cfg.backend.active = b;
        }
        if !cfg.backends.contains_key(&cfg.backend.active) {
            return Err(NafsError::Config(format!(
                "backend.active = '{}' is not defined under [backends]",
                cfg.backend.active
            )));
        }
        let mut h = Sha256::new();
        h.update(raw.as_bytes());
        cfg.hash = h
            .finalize()
            .iter()
            .take(8)
            .map(|b| format!("{b:02x}"))
            .collect();
        Ok(cfg)
    }
    pub fn caps(&self, d: Domain) -> Caps {
        self.deliberation
            .caps
            .get(d.as_str())
            .copied()
            .unwrap_or(Caps {
                n: 1,
                r: 1,
                budget: 512,
            })
    }
    pub fn e2(&self, d: Domain) -> f32 {
        *self.gate.e2_hat.get(d.as_str()).unwrap_or(&0.35)
    }
    pub fn delta_c(&self, d: Domain) -> f32 {
        *self.gate.delta_c_s.get(d.as_str()).unwrap_or(&90.0)
    }
}
