//! nafs-types: shared types and wire contracts (IF-005) for Nafs-Aql.
//! The JSON shapes mirror `api/openapi.yaml` and `api/schemas/*.schema.json`.
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

/// Path that produced a reply (IF-003).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningMode {
    Fast,
    Slow,
}

/// Assurance level attached to every reply (FR-031).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
#[serde(rename_all = "lowercase")]
pub enum Assurance {
    Verified,
    Consensus,
    Unverified,
    Abstained,
}

/// States of the request automaton (report Fig. 3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TraceState {
    A,
    L,
    M,
    T,
    X,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash, Default)]
#[serde(rename_all = "lowercase")]
pub enum Domain {
    Code,
    Math,
    Structured,
    Factual,
    Open,
    #[default]
    Other,
}

impl Domain {
    pub fn as_str(&self) -> &'static str {
        match self {
            Domain::Code => "code",
            Domain::Math => "math",
            Domain::Structured => "structured",
            Domain::Factual => "factual",
            Domain::Open => "open",
            Domain::Other => "other",
        }
    }
    pub fn all() -> [Domain; 6] {
        [
            Domain::Code,
            Domain::Math,
            Domain::Structured,
            Domain::Factual,
            Domain::Open,
            Domain::Other,
        ]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Message {
    pub role: String,
    pub content: String,
}

/// Optional inputs that make a task verifiable (FR-030).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VerificationInputs {
    /// Python statements (asserts) appended after the candidate's code.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tests: Option<Vec<String>>,
    /// HumanEval-style test program defining `check(candidate)`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub test_code: Option<String>,
    /// Function name passed to `check` (HumanEval).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entry_point: Option<String>,
    /// Code prefix (e.g. HumanEval prompt) prepended when the answer is only a body.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code_prefix: Option<String>,
    /// Reference answer for the math verifier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference_answer: Option<String>,
    /// JSON Schema the answer must satisfy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<Value>,
    /// A Lean 4 theorem whose proof Lean must accept (Open Math Harness protocol, FR-090).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lean: Option<LeanTask>,
}

/// A frozen Lean statement: a whole `.lean` file whose theorem is proved by `sorry`. The answer must
/// reproduce the statement byte for byte and prove it; Lean decides, after textual rules that refuse
/// `sorry`, `axiom`, `native_decide` and commands that run code (FR-090).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LeanTask {
    pub statement: String,
    /// The theorem's name; read from the statement when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub theorem: Option<String>,
}
impl VerificationInputs {
    /// Code tests that can actually run: plain asserts, or HumanEval `test_code` together with the
    /// `entry_point` that `check()` is called on (without it no assertion would execute; M1).
    pub fn has_code_tests(&self) -> bool {
        self.tests
            .as_ref()
            .map(|t| t.iter().any(|s| !s.trim().is_empty()))
            .unwrap_or(false)
            || (self.test_code.is_some()
                && self
                    .entry_point
                    .as_ref()
                    .map(|e| !e.trim().is_empty())
                    .unwrap_or(false))
    }
    pub fn is_verifiable(&self) -> bool {
        self.has_code_tests()
            || self
                .reference_answer
                .as_ref()
                .map(|r| !r.trim().is_empty())
                .unwrap_or(false)
            || self.json_schema.is_some()
            || self.lean.is_some()
    }
    /// Reason the inputs are malformed, if they are (used to reject requests with 400).
    pub fn validation_error(&self) -> Option<String> {
        if self.test_code.is_some()
            && self
                .entry_point
                .as_ref()
                .map(|e| e.trim().is_empty())
                .unwrap_or(true)
        {
            return Some("verification.test_code requires verification.entry_point (the function check() is called on)".into());
        }
        if let Some(ep) = &self.entry_point {
            if !ep.chars().all(|c| c.is_alphanumeric() || c == '_') {
                return Some("verification.entry_point must be a Python identifier".into());
            }
        }
        None
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChatRequest {
    #[serde(default)]
    pub conversation_id: Option<Uuid>,
    pub messages: Vec<Message>,
    #[serde(default)]
    pub domain_hint: Option<Domain>,
    /// Speed-care control: price of compute relative to the loss of a wrong answer (FR-004).
    #[serde(default)]
    pub lambda: Option<f32>,
    /// Operator override (experiments, baselines).
    #[serde(default)]
    pub force_mode: Option<ReasoningMode>,
    #[serde(default)]
    pub verification: Option<VerificationInputs>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub thinking_budget: Option<u32>,
    #[serde(default)]
    pub show_reasoning: Option<bool>,
    /// Do not persist the conversation (evaluation and benchmark traffic).
    #[serde(default)]
    pub ephemeral: Option<bool>,
    /// Force this request into the audit stream (FR-040); evaluation and operator use.
    #[serde(default)]
    pub audit: Option<bool>,
    /// Ablation only (coupling test, ER-004): System 2 sees the System 1 draft. Ignored unless enabled in config.
    #[serde(default)]
    pub ablation_draft_conditioned: Option<bool>,
    /// FR-061: use (true) or skip (false) retrieval of earlier verified answers; default from config.
    #[serde(default)]
    pub retrieve: Option<bool>,
    /// FR-082: use (true) or skip (false) the adaptive reasoning exit; default from config.
    #[serde(default)]
    pub adaptive_exit: Option<bool>,
    /// Phase 8: stopping rule of the adaptive exit on non-verifiable tasks, "consistency" (k consecutive
    /// agreeing probes) or "posterior" (majority posterior above a collapsing bound); default from config.
    #[serde(default)]
    pub adaptive_exit_rule: Option<String>,
    /// FR-085: domains the adaptive exit applies to for this request (overrides the configured list);
    /// `["all"]` means every domain. On code and structured tasks the probe is a full answer that the
    /// exact verifier judges, and a passing probe becomes the final answer.
    #[serde(default)]
    pub adaptive_exit_domains: Option<Vec<String>>,
    /// FR-086: factor applied to the thinking budget on each verified retry round (0.25..=1.0);
    /// default from configuration (1.0 = the same budget every round).
    #[serde(default)]
    pub retry_budget_decay: Option<f32>,
    /// FR-091: "best" (repair the best-scoring candidate, Open Math Harness search) or "last"; default
    /// from configuration.
    #[serde(default)]
    pub repair_policy: Option<String>,
    /// FR-092: allow a code retry to answer with a Codex patch to the candidate; default from configuration.
    #[serde(default)]
    pub patch_repair: Option<bool>,
    /// In-process lab traffic (evaluation and load-test runs). Never read from HTTP; such replies are stored
    /// as ephemeral and kept out of the operational alerts (OR-004), which describe the live service.
    #[serde(skip)]
    pub lab: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub reasoning_tokens: u32,
    #[serde(default)]
    pub thinking_budget: Option<u32>,
    pub wall_ms: u64,
    #[serde(default)]
    pub ttft_ms: Option<u64>,
    pub model_calls: u32,
    pub verifier_calls: u32,
    /// Where the time went (P7-T01): one entry per stage, in order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stages: Vec<StageTiming>,
}

/// One stage of a request's latency (P7-T01). Model stages carry the server's own split: queue wait in the
/// gateway, prompt evaluation (prefill) and generation (decode), and how much of the prompt came from the cache.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct StageTiming {
    /// retrieval, draft, verify_draft, deliberate, verify, judge, persist.
    pub stage: String,
    pub ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub queue_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_ms: Option<f64>,
    /// Prompt tokens evaluated, and prompt tokens reused from the KV cache.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_n: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_n: Option<u32>,
    /// Tokens generated (answer and reasoning).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens: Option<u32>,
    /// Free text, e.g. how an adaptive reasoning exit ended (P7-T08).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rates {
    /// True-accept rate (correct answers accepted).
    pub r: f64,
    /// False-accept rate (wrong answers accepted).
    pub phi: f64,
    pub n_mutants: u32,
    #[serde(default)]
    pub n_correct: u32,
    /// Clopper-Pearson 95% upper bound on phi.
    #[serde(default)]
    pub phi_upper95: Option<f64>,
    #[serde(default)]
    pub measured_at: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VerifierResult {
    pub verifier: String,
    pub passed: bool,
    #[serde(default)]
    pub score: Option<f32>,
    #[serde(default)]
    pub evidence: Option<String>,
    #[serde(default)]
    pub rates: Option<Rates>,
    #[serde(default)]
    pub duration_ms: u64,
    /// Exact verifiers accept at threshold 1.0 (Thm 1); soft ones never yield `verified`.
    #[serde(default)]
    pub exact: bool,
    /// Structured details, e.g. the judge's rubric scores.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GateFeatures {
    pub mean_logprob: Option<f32>,
    pub min_logprob: Option<f32>,
    /// Phase 8: lowest mean log-prob over any `gate.confidence_window` consecutive draft tokens
    /// ("lowest group confidence"); a running minimum, so an early crossing is the full draft's value.
    #[serde(default)]
    pub low_window_logprob: Option<f32>,
    pub k: u8,
    pub agreement: Option<f32>,
    pub domain: Domain,
    pub verifiable: bool,
    pub draft_verified: Option<bool>,
    pub prompt_tokens: u32,
    pub draft_tokens: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateRule {
    Fixed,
    Calibrated,
    Budget,
    ForcedFailedVerification,
    ForcedOverride,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionRecord {
    pub decision_id: Uuid,
    pub request_id: String,
    pub timestamp: DateTime<Utc>,
    pub config_hash: String,
    pub model_profile: String,
    pub backend: String,
    pub calibration_version: Option<String>,
    pub features: GateFeatures,
    pub e1_hat: f32,
    pub e2_hat: f32,
    pub lambda: f32,
    pub delta_c: f32,
    pub escalate: bool,
    pub rule: GateRule,
    pub audit: bool,
    pub explanation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatReply {
    pub reply_id: Uuid,
    pub conversation_id: Option<Uuid>,
    pub decision_id: Uuid,
    pub reasoning_mode: ReasoningMode,
    pub assurance: Assurance,
    pub trace_state: TraceState,
    pub states: Vec<TraceState>,
    pub message: Message,
    #[serde(default)]
    pub uncertainty_notice: Option<String>,
    #[serde(default)]
    pub verifier: Option<VerifierResult>,
    pub usage: Usage,
    #[serde(default)]
    pub degraded: bool,
    /// System 1 draft when the reply came from System 2.
    #[serde(default)]
    pub draft: Option<String>,
    #[serde(default)]
    pub reasoning: Option<String>,
    pub decision: DecisionRecord,
    pub domain: Domain,
    /// One record per System 2 attempt (empty on the fast path); used by the Phase-0 Beta fit (FR-021).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rounds: Vec<RoundRecord>,
    /// True when System 2 saw the System 1 draft (coupling ablation only, ER-004).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub draft_conditioned: bool,
    /// FR-061: earlier verified answers placed in the prompt as data (provenance and score).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub retrieved: Vec<Value>,
}

/// SSE / WebSocket stream events (FR-006).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event", content = "data", rename_all = "snake_case")]
pub enum StreamEvent {
    Start {
        request_id: String,
        conversation_id: Option<Uuid>,
        domain: Domain,
        backend: String,
    },
    Phase {
        state: TraceState,
        label: String,
    },
    Token {
        text: String,
        mode: ReasoningMode,
        reasoning: bool,
        round: u32,
    },
    Decision(Box<DecisionRecord>),
    Verification {
        round: u32,
        result: VerifierResult,
    },
    Retry {
        round: u32,
        of: u32,
        evidence: Option<String>,
    },
    Degraded {
        reason: String,
    },
    Done(Box<ChatReply>),
    Error {
        code: String,
        message: String,
    },
}
impl StreamEvent {
    pub fn name(&self) -> &'static str {
        match self {
            StreamEvent::Start { .. } => "start",
            StreamEvent::Phase { .. } => "phase",
            StreamEvent::Token { .. } => "token",
            StreamEvent::Decision(_) => "decision",
            StreamEvent::Verification { .. } => "verification",
            StreamEvent::Retry { .. } => "retry",
            StreamEvent::Degraded { .. } => "degraded",
            StreamEvent::Done(_) => "done",
            StreamEvent::Error { .. } => "error",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundRecord {
    pub round: u32,
    pub candidate_tokens: u32,
    pub reasoning_tokens: u32,
    pub verifier: String,
    pub passed: bool,
}

/// Newline-delimited trace record (IF-010).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceRecord {
    pub schema_version: String,
    pub request_id: String,
    pub decision_id: Uuid,
    pub conversation_id: Option<Uuid>,
    pub timestamp: DateTime<Utc>,
    pub config_hash: String,
    pub model_profile: String,
    pub backend: String,
    pub model_hash: String,
    pub adapter_hash: String,
    pub domain: Domain,
    pub path: ReasoningMode,
    pub states: Vec<TraceState>,
    pub rounds: Vec<RoundRecord>,
    pub assurance: Assurance,
    pub usage: Usage,
    pub audit: bool,
}

/// Error taxonomy (spec Section 8.7).
#[derive(Debug, thiserror::Error)]
pub enum NafsError {
    #[error("bad request: {0}")]
    BadRequest(String),
    #[error("unauthorized")]
    Unauthorized,
    #[error("rate limited")]
    RateLimited,
    #[error("queue timeout for backend {0}")]
    QueueTimeout(String),
    #[error("backend unavailable: {0}")]
    BackendUnavailable(String),
    #[error("backend timeout: {0}")]
    BackendTimeout(String),
    #[error("backend protocol error: {0}")]
    BackendProtocol(String),
    #[error("thinking budget exhausted")]
    BudgetExhausted,
    #[error("verifier unavailable: {0}")]
    VerifierUnavailable(String),
    #[error("sandbox limit: {0}")]
    SandboxLimit(String),
    #[error("storage error: {0}")]
    Storage(String),
    #[error("configuration error: {0}")]
    Config(String),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("internal error: {0}")]
    Internal(String),
    #[error("request cancelled by the client")]
    Cancelled,
    #[error("forbidden: {0}")]
    Forbidden(String),
    /// The request is well formed but clashes with existing state (for example a duplicate room name).
    #[error("conflict: {0}")]
    Conflict(String),
}
impl NafsError {
    pub fn code(&self) -> &'static str {
        match self {
            NafsError::BadRequest(_) => "E_BAD_REQUEST",
            NafsError::Unauthorized => "E_UNAUTHORIZED",
            NafsError::RateLimited => "E_RATE_LIMITED",
            NafsError::QueueTimeout(_) => "E_QUEUE_TIMEOUT",
            NafsError::BackendUnavailable(_) => "E_BACKEND_UNAVAILABLE",
            NafsError::BackendTimeout(_) => "E_BACKEND_TIMEOUT",
            NafsError::BackendProtocol(_) => "E_BACKEND_PROTOCOL",
            NafsError::BudgetExhausted => "E_BUDGET_EXHAUSTED",
            NafsError::VerifierUnavailable(_) => "E_VERIFIER_UNAVAILABLE",
            NafsError::SandboxLimit(_) => "E_SANDBOX_LIMIT",
            NafsError::Storage(_) => "E_STORAGE",
            NafsError::Config(_) => "E_CONFIG",
            NafsError::NotFound(_) => "E_NOT_FOUND",
            NafsError::Internal(_) => "E_INTERNAL",
            NafsError::Cancelled => "E_CANCELLED",
            NafsError::Forbidden(_) => "E_FORBIDDEN",
            NafsError::Conflict(_) => "E_CONFLICT",
        }
    }
    pub fn http_status(&self) -> u16 {
        match self {
            NafsError::BadRequest(_) => 400,
            NafsError::Unauthorized => 401,
            NafsError::RateLimited | NafsError::QueueTimeout(_) => 429,
            NafsError::NotFound(_) => 404,
            NafsError::Forbidden(_) => 403,
            NafsError::Conflict(_) => 409,
            NafsError::Cancelled => 499,
            NafsError::BackendUnavailable(_) => 503,
            NafsError::BackendTimeout(_) => 504,
            _ => 500,
        }
    }
}

pub type Result<T> = std::result::Result<T, NafsError>;

/// `POST /v1/harness/code` (FR-093): a bounded code agent over a small workspace.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CodeAgentRequest {
    /// What to build or fix.
    pub task: String,
    /// Starting files (flat names, e.g. `solution.py`); empty starts from nothing.
    #[serde(default)]
    pub files: std::collections::BTreeMap<String, String>,
    /// The hidden tests that end the run (`tests` asserts, or HumanEval `test_code` + `entry_point`); they
    /// run against `solution.py`.
    pub verification: VerificationInputs,
    /// Model calls (1..=12); default from configuration.
    #[serde(default)]
    pub max_steps: Option<u32>,
    /// "system2" or "system1"; default from configuration.
    #[serde(default)]
    pub model: Option<String>,
    /// Phase 11 A/B override of `harness.context`: "append" (Codex transcript) or "reset" (Phase 10 loop).
    #[serde(default)]
    pub context: Option<String>,
    /// Phase 11 A/B override of `harness.constrain`: sample actions under the action grammar.
    #[serde(default)]
    pub constrain: Option<bool>,
    /// Phase 11 A/B override of `harness.stream_cut`: end the stream when the action is complete.
    #[serde(default)]
    pub stream_cut: Option<bool>,
    /// Phase 11 A/B override of `harness.edit_format`: "auto", "any" or "whole".
    #[serde(default)]
    pub edit_format: Option<String>,
    /// Phase 11 A/B override of `harness.unstick`: break a repetition loop (compact, sample more diversely).
    #[serde(default)]
    pub unstick: Option<bool>,
    /// Override of `harness.thinking_budget` (System 2 reasoning tokens per step, 64..=4096).
    #[serde(default)]
    pub thinking_budget: Option<u32>,
    /// Override of `harness.first_step_thinking_budget` (0 = off, else 64..=4096).
    #[serde(default)]
    pub first_step_thinking_budget: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeAgentStep {
    pub step: u32,
    /// patch, write, run, submit, none or model_error.
    pub action: String,
    /// What the model was shown after the action.
    pub observation: String,
    /// The verifier's verdict after a change to the workspace.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub passed: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
    pub ms: u64,
    /// What the step's model call cost (absent when the call failed).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<CodeAgentUsage>,
    /// The transcript was compacted before this step's call.
    #[serde(default)]
    pub compacted: bool,
}

/// One model call of the code agent, as the server reported it (Phase 11).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CodeAgentUsage {
    /// Prompt tokens prefilled (not served from the server's cache).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u32>,
    /// Prompt tokens served from the cache.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_ms: Option<f64>,
    pub completion_tokens: u32,
    pub reasoning_tokens: u32,
    /// stop, length, or action_complete (the stream was cut after the action).
    pub finish_reason: String,
    /// Sampled under the action grammar.
    pub constrained: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeAgentReply {
    pub run_id: Uuid,
    /// verified, exhausted, deadline or model_error.
    pub status: String,
    /// `verified` only when the exact verifier accepted the returned files; otherwise `unverified`.
    pub assurance: Assurance,
    /// The verified files, or the best-scoring files seen.
    pub files: std::collections::BTreeMap<String, String>,
    pub passed: bool,
    pub score: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evidence: Option<String>,
    pub steps: Vec<CodeAgentStep>,
    pub model_calls: u32,
    pub model: String,
    pub wall_ms: u64,
    /// "append" or "reset".
    #[serde(default)]
    pub context: String,
    /// Whether actions were sampled under the action grammar.
    #[serde(default)]
    pub constrained: bool,
    /// "auto", "any" or "whole".
    #[serde(default)]
    pub edit_format: String,
    /// Sandbox runs of the verifier (a workspace already judged is not judged again).
    #[serde(default)]
    pub verifier_calls: u32,
    #[serde(default)]
    pub compactions: u32,
    /// Prompt tokens prefilled over the run, and those served from the server's cache.
    #[serde(default)]
    pub prefill_tokens: u64,
    #[serde(default)]
    pub cached_tokens: u64,
    /// Tokens decoded over the run (reasoning included).
    #[serde(default)]
    pub decoded_tokens: u64,
    /// The run's net change as a unified diff (returned files against the starting ones).
    #[serde(default)]
    pub diff: String,
}
