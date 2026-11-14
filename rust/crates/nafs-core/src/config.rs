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
    /// Labelled gate samples collected while the machine is idle (Phase 23).
    #[serde(default)]
    pub labels: LabelCfg,
    #[serde(default)]
    pub alerts: AlertsCfg,
    #[serde(default)]
    pub retrieval: RetrievalCfg,
    /// Memory of earlier conversations (Phase 26).
    #[serde(default)]
    pub recall: RecallCfg,
    /// ShadowPrefill (Phase 30): keep the other model of the pair warm.
    #[serde(default)]
    pub shadow: ShadowCfg,
    /// The context filesystem (Phase 29): skills in chat, saving quiet chats as files.
    #[serde(default)]
    pub context: ContextCfg,
    /// Math and code harness (FR-090..FR-093): the code agent endpoint.
    #[serde(default)]
    pub harness: HarnessCfg,
    /// Knowledge Base (Phase 20, FR-K*): documents in SurrealDB, searched and used in chat.
    #[serde(default)]
    pub knowledge: KnowledgeCfg,
    #[serde(skip)]
    pub hash: String,
}

/// `[knowledge]` (IF-K02): the SurrealDB connection, the upload limits and the chunker's settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct KnowledgeCfg {
    pub enabled: bool,
    /// SurrealDB HTTP endpoint (scripts/serve_surreal.sh serves 127.0.0.1:8300).
    pub url: String,
    pub namespace: String,
    pub database: String,
    pub user: String,
    /// File holding the password (written by scripts/serve_surreal.sh); never the password itself.
    pub password_file: String,
    /// Stored originals, one folder per knowledge base.
    pub files_dir: String,
    pub max_upload_mb: usize,
    pub max_files_per_request: usize,
    /// Phase 8 (SR-K11): the most one upload request may carry in all, so many large parts cannot be held in memory
    /// together; and how many uploads are received at once.
    pub max_request_mb: usize,
    pub concurrent_uploads: usize,
    /// Documents one base may hold, passages one document may make, jobs that may wait, and seconds a parser may take.
    pub max_documents_per_base: u64,
    pub max_chunks_per_document: usize,
    pub max_queued_jobs: usize,
    pub parse_timeout_s: u64,
    /// Seconds a search by words may take before a hybrid search answers by meaning alone (a words-only search is refused).
    pub search_timeout_s: u64,
    /// Training runs that may go at once across all bases (each needs 8 to 11 GB for the 1.7B decoder).
    pub max_concurrent_trainers: usize,
    /// Passage size and overlap in characters (FR-K15).
    pub chunk_chars: usize,
    pub chunk_overlap: usize,
    pub search_k: usize,
    /// Embedding server (`nafs-smiley-server --embedding`, Phase 2); empty turns vectors off (full-text only).
    pub embed_url: String,
    /// Model id stored with the vectors; changing it means re-embedding.
    pub embed_model: String,
    pub embed_dims: usize,
    /// Passages per embedding request.
    pub embed_batch: usize,
    /// Knowledge bases with fewer vectors than this are searched exactly; larger ones through the HNSW index.
    pub exact_search_below: u64,
    /// Chat: passages put in the prompt in `rag` mode, and the most characters they may take together.
    pub rag_k: usize,
    pub rag_max_chars: usize,
    /// Chat: a passage is used only when its hybrid rank fuses both lists or its similarity reaches this.
    pub rag_min_similarity: f64,
    /// A hybrid search fuses the words' ranking (weight 1) and the meaning ranking by reciprocal rank; this is the
    /// meaning's weight. 0.1 measured best on two bases of the paper (1.0 let the weaker arm push answers down).
    pub vector_weight: f64,
    /// The most memory (MB) the in-memory copies of large bases' vectors may use together (a base of 50,000 passages takes
    /// 200 MB). 0 (the default) turns the copies off and large bases use the store's index, which answered in 0.37 s at the
    /// median at 50,000 passages once warm; a copy answers exactly in 20 ms but takes minutes to load.
    pub vector_cache_mb: u64,
    /// Chat: a passage whose words match this strongly (a BM25 score) is used even if the meaning search missed it or
    /// scored it low; 0 turns it off. Off-topic questions scored at most 8.0, answering passages at least 10.8.
    pub rag_lexical_rescue: f64,
    /// Knowledge graph (Phase 3): the OpenAI-compatible server and model that extract entities and relations; empty
    /// turns the graph off.
    pub graph_llm_url: String,
    pub graph_model: String,
    pub graph_max_tokens: usize,
    /// Entities of one type whose name vectors are at least this similar are merged.
    pub entity_merge_similarity: f64,
    /// Chat in `graph` mode: facts put in the prompt.
    pub graph_facts: usize,
    /// FR-K50 (Phase 21): how much the graph's ranking of passages counts against the search's when the two are
    /// fused (1.0 = an equal arm, 0 = the graph adds facts but never moves a passage).
    pub graph_weight: f64,
    /// FR-K56 (Phase 21): relations kept per entity at the second hop of the graph walk, and the per-hop decay of a
    /// path's score (a second-hop fact must be well attested to outrank a first-hop one).
    pub graph_beam: usize,
    pub graph_decay: f64,
    /// Phase 25 (FR-K67): the most memory (MB) the in-process indexes of all bases may use (passages, their words,
    /// their vectors and the term graph: about 6 MB per 1,000 passages at 1,024 dimensions). A search of an indexed
    /// base makes no round trip to the store. 0 turns the index off; a base that does not fit uses the store.
    pub index_memory_mb: u64,
    /// FR-K68: BM25F weights of the document title, the heading path, the context line and the body.
    pub bm25_weights: [f32; 4],
    /// FR-K70: how the words and the meaning are fused in the index: "rrf" (by rank, the meaning weighted
    /// `vector_weight`) or "convex" (by normalised score, the meaning's share `fusion_alpha`).
    pub fusion: String,
    pub fusion_alpha: f64,
    /// FR-K72: how much the walk over the passage-and-term graph counts against the fused search (0 = off).
    pub graph_arm_weight: f64,
    /// Phase 26: a question word the base does not hold is replaced by the nearest word it does hold (a typing slip).
    pub fuzzy_terms: bool,
    /// Phase 28: target size (characters) of the sentence windows of a passage that get vectors of their own
    /// (semtools' small search unit); 0 = none. Costs about one more embedding pass per document.
    pub piece_chars: usize,
    /// Phase 28: how much a passage's best sentence window counts in its meaning score (0 to 1).
    pub piece_mix: f32,
    /// Phase 26, chat: among the passages that pass the relevance filter, those after the largest drop in meaning
    /// similarity are left out when that drop is at least this large (0 = keep all `rag_k`).
    pub rag_elbow_gap: f64,
    /// Phase 26, chat: a question that leans on the previous turn ("and how fast is it?") is searched together with
    /// that turn.
    pub followup_context: bool,
    /// Phase 26, chat: every sentence of an answer from documents is checked against the passages it was given.
    pub support_check: bool,
    /// The share of a sentence's content words a passage must hold for the sentence to count as supported.
    pub support_min_overlap: f64,
    /// A quick-model answer with fewer than this share of supported sentences goes to the reasoning model (0 = never).
    pub support_escalate_below: f64,
    /// Phase 27, chat: `rag_max_chars` is the size of the whole passage block (each source's header counted), and a
    /// document's title is written once, not on each of its passages. Off: the budget is the passage text alone.
    /// Off by default: with `rag_k = 6` and `rag_max_chars = 2400` it made prompts 25 % shorter and answers 20 to 26 %
    /// sooner, and 39 of 50 answers were right against 42 (wrong numbers taken from excerpts).
    pub rag_compact_sources: bool,
    /// Memorilla (Phase 5): a memory-module checkpoint (`config.json` + `weights.safetensors`) trained for System 1;
    /// empty turns `memory` mode off.
    pub memory_checkpoint: String,
    /// The token that marks the memory positions in the prompt (one token of the decoder's vocabulary).
    pub memory_placeholder: String,
    /// Phase 6: the trainer binary, the f32 decoder it trains through, and the GGUF whose tokenizer and template
    /// make the prompts (System 1's).
    pub memory_trainer: String,
    pub memory_decoder: String,
    pub memory_gguf: String,
    /// Phase 7, FR-K24: passages put in the prompt beside the memory tokens in `hybrid` mode.
    pub hybrid_k: usize,
    /// Phase 7, FR-K24: the order `auto` tries modes in; the first one this knowledge base can serve wins
    /// ("memory" and "hybrid" need a module, "graph" needs a built graph). The last entry should be one that
    /// always works ("rag").
    pub auto_order: Vec<String>,
}
impl Default for KnowledgeCfg {
    fn default() -> Self {
        Self {
            enabled: false,
            url: "http://127.0.0.1:8300".into(),
            namespace: "nafs".into(),
            database: "kb".into(),
            user: "root".into(),
            password_file: "data/runtime/surreal/root.secret".into(),
            files_dir: "data/runtime/kb".into(),
            max_upload_mb: 50,
            max_files_per_request: 20,
            max_request_mb: 200,
            concurrent_uploads: 2,
            max_documents_per_base: 2_000,
            max_chunks_per_document: 6_000,
            max_queued_jobs: 500,
            parse_timeout_s: 180,
            search_timeout_s: 10,
            max_concurrent_trainers: 1,
            chunk_chars: 1200,
            chunk_overlap: 150,
            search_k: 8,
            embed_url: String::new(),
            embed_model: "Qwen3-Embedding-0.6B".into(),
            embed_dims: 1024,
            embed_batch: 8,
            exact_search_below: 5000,
            rag_k: 4,
            rag_max_chars: 4800,
            rag_min_similarity: 0.45,
            vector_weight: 0.1,
            vector_cache_mb: 0,
            rag_lexical_rescue: 10.0,
            graph_llm_url: String::new(),
            graph_model: "system1".into(),
            graph_max_tokens: 400,
            entity_merge_similarity: 0.92,
            graph_facts: 12,
            graph_weight: 1.0,
            graph_beam: 8,
            graph_decay: 0.6,
            index_memory_mb: 512,
            bm25_weights: [1.0, 1.0, 1.0, 1.0],
            fusion: "rrf".into(),
            fusion_alpha: 0.2,
            graph_arm_weight: 0.0,
            fuzzy_terms: true,
            // Measured (Phase 28, 242 questions): quoted MRR 0.870 to 0.922, reworded 0.194 to 0.201, two-passage
            // 0.570 to 0.571, no set worse.
            piece_chars: 240,
            piece_mix: 0.5,
            rag_elbow_gap: 0.0,
            followup_context: true,
            support_check: true,
            support_min_overlap: 0.6,
            support_escalate_below: 0.0,
            rag_compact_sources: false,
            memory_checkpoint: String::new(),
            memory_placeholder: "<|fim_pad|>".into(),
            memory_trainer: "rust/target/release/memorilla-train".into(),
            memory_decoder: "models/hf/Qwen3-1.7B".into(),
            memory_gguf: "models/Qwen3-1.7B-Q4_K_M.gguf".into(),
            hybrid_k: 2,
            auto_order: vec!["rag".into()],
        }
    }
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
    /// Exchanges kept after each context reset (OMH keeps 4). Reset mode only.
    #[serde(default = "d_agent_keep")]
    pub keep_exchanges: usize,
    /// Phase 11: "append" keeps the transcript append-only so the server's prompt cache is reused (Codex);
    /// "reset" rebuilds it every step (Phase 10).
    #[serde(default)]
    pub context: nafs_harness::agent::ContextMode,
    /// Append mode: estimated transcript tokens above which it is compacted before the next call. Must leave
    /// room for the reply in the model's slot (System 1 slots hold 4,096 tokens on the reference setup).
    #[serde(default = "d_agent_compact")]
    pub compact_at_tokens: usize,
    /// Sample each action under the step's GBNF grammar (llama-server only; ignored elsewhere).
    #[serde(default = "d_true")]
    pub constrain: bool,
    /// End the model's stream as soon as its action is complete.
    #[serde(default = "d_true")]
    pub stream_cut: bool,
    /// Prose lines System 1 may write before its action under the grammar (System 2 plans while reasoning).
    #[serde(default = "d_agent_notes")]
    pub notes: usize,
    /// "auto" (whole file while solution.py is missing or short, patch or whole file above), "any" or "whole".
    #[serde(default)]
    pub edit_format: nafs_harness::agent::EditFormat,
    /// "auto": solution.py's length (lines) above which a patch is offered.
    #[serde(default = "d_agent_whole_lines")]
    pub whole_max_lines: usize,
    /// After a step that reproduced already-judged files: compact the transcript and sample more diversely.
    #[serde(default = "d_true")]
    pub unstick: bool,
    /// System 2 reasoning tokens on the first step only (0 = `thinking_budget` on every step): a cheap first
    /// attempt, the full budget after a failure (decision 006, round 3: same tasks verified in half the time).
    #[serde(default = "d_agent_first_think")]
    pub first_step_thinking_budget: u32,
    /// Phase 14: the first step goes to System 1 with `cascade_samples` candidates in one request, each judged
    /// by the verifier; System 2 continues from the best only when none passes. Off when `model` is system1.
    #[serde(default = "d_true")]
    pub cascade: bool,
    /// Candidates of the cascade's first step (1..=4, one `n`-choice request on System 1).
    #[serde(default = "d_agent_cascade_samples")]
    pub cascade_samples: u32,
    /// Sampling temperature of the cascade candidates (they must differ from each other).
    #[serde(default = "d_agent_cascade_temp")]
    pub cascade_temperature: f32,
    /// Retries of a model call that fails for a reason other than the run's cap (Codex's backoff).
    #[serde(default = "d_agent_retries")]
    pub retries: u32,
    /// Phase 16: refuse an edit whose Python file does not compile (SWE-agent's lint guardrail): the files stay,
    /// the model is shown the offending lines, and no verifier run is spent.
    #[serde(default = "d_true")]
    pub lint: bool,
    /// Phase 16: when a change scores below the best version so far, restore that version.
    #[serde(default = "d_true")]
    pub rollback: bool,
    /// Phase 16: stop after this many consecutive steps that fail the same way (0 = never); the second one
    /// already carries a nudge to change approach.
    #[serde(default = "d_agent_stuck_after")]
    pub stuck_after: u32,
}
fn d_max_risk() -> f32 {
    0.15
}
fn d_risk_alpha() -> f32 {
    0.05
}
fn d_on_unknown() -> String {
    "escalate".into()
}
fn d_reask_overlap() -> f32 {
    0.6
}
fn d_agent_stuck_after() -> u32 {
    4
}
fn d_agent_cascade_samples() -> u32 {
    4
}
fn d_agent_cascade_temp() -> f32 {
    0.7
}
fn d_agent_retries() -> u32 {
    2
}
fn d_agent_first_think() -> u32 {
    256
}
fn d_agent_whole_lines() -> usize {
    80
}
fn d_agent_compact() -> usize {
    2600
}
fn d_agent_notes() -> usize {
    2
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
            context: Default::default(),
            compact_at_tokens: d_agent_compact(),
            constrain: true,
            stream_cut: true,
            notes: d_agent_notes(),
            edit_format: Default::default(),
            whole_max_lines: d_agent_whole_lines(),
            unstick: true,
            first_step_thinking_budget: d_agent_first_think(),
            cascade: true,
            cascade_samples: d_agent_cascade_samples(),
            cascade_temperature: d_agent_cascade_temp(),
            retries: d_agent_retries(),
            lint: true,
            rollback: true,
            stuck_after: d_agent_stuck_after(),
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

/// Memory of earlier conversations (Phase 26): each finished turn of a stored conversation is kept as a short note,
/// and a question that speaks in the first person or points back ("my", "last time", "you said") is answered with
/// the notes that match it, quoted as data. No model call: words by BM25, meaning by the embedding server when it
/// runs. A private (`ephemeral`) request, a room and the Lab neither write to this memory nor read it.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RecallCfg {
    pub enabled: bool,
    /// Most notes put in one prompt.
    pub limit: usize,
    /// Lowest relevance (0 to 1) a note needs.
    pub min_score: f64,
    /// Days after which a note counts half.
    pub half_life_days: f64,
    /// Most characters of notes in one prompt; on a CPU every thousand characters is seconds of reading.
    pub budget_chars: usize,
    /// Give each note a meaning vector (one embedding call after the reply), so a question in other words finds it.
    pub vectors: bool,
    /// Most notes kept; the never-recalled oldest go first, and notes a person asked for are never removed (0 = no
    /// limit).
    pub max_episodes: usize,
}
impl Default for RecallCfg {
    fn default() -> Self {
        Self {
            enabled: true,
            limit: 3,
            min_score: 0.35,
            half_life_days: 30.0,
            budget_chars: 1500,
            vectors: true,
            max_episodes: 5000,
        }
    }
}

/// The context filesystem (Phase 29, `docs/spec/context_fs_spec.md`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ContextCfg {
    /// A question that asks how to do a task carries the saved skill that matches it.
    pub skills_in_chat: bool,
    /// Save a chat as a file (transcript and memories) once it has been quiet this many minutes (0 = only on request).
    pub commit_idle_minutes: u64,
}
impl Default for ContextCfg {
    fn default() -> Self {
        Self {
            skills_in_chat: true,
            commit_idle_minutes: 0,
        }
    }
}

/// ShadowPrefill (Phase 30, after Qiu et al. 2026): the model a conversation may switch to reads the conversation in
/// time that would otherwise go unused, so a switch reads only what is new. Smiley Core servers only.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ShadowCfg {
    /// After a reply, while nothing else runs, the model that did not answer reads the conversation so far.
    pub between_turns: bool,
    /// Wait this long after a reply, with no request in flight, before starting.
    pub idle_delay_ms: u64,
    /// While the quick model drafts, the reasoning model reads the same prompt (both share the cores).
    pub at_arrival: bool,
    /// Share of time an arrival shadow may compute (0.05 to 1).
    pub arrival_duty: f32,
    /// Keep what changes with each question (the domain instruction, retrieved passages, notes) out of the system
    /// prompt and put it before the question, so every turn's prompt begins with the previous turn's.
    pub stable_prefix: bool,
}
impl Default for ShadowCfg {
    fn default() -> Self {
        // Measured (Phase 30): stable layout + between-turns shadow cut the switch to the reasoning model from 36.6 s to
        // 2.0 s with the same right answers (42/50 both layouts); a shadow during the draft slowed every draft by 75 %.
        Self {
            between_turns: true,
            idle_delay_ms: 1500,
            at_arrival: false,
            arrival_duty: 0.25,
            stable_prefix: true,
        }
    }
}

/// The label stream (Phase 23): while no user request is in flight, the gateway answers tasks of a verifiable dataset
/// with the quick model alone (a draft and its agreement samples, no System 2 call), judges each draft with the exact
/// verifier, and so grows the ledger of labelled gate samples that calibration needs. The existing scheduled
/// recalibration (`[gate] recalibrate_every_h`) then fits and, if the held-out tests pass, promotes.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LabelCfg {
    pub enabled: bool,
    /// A dataset in `[eval] datasets_dir` whose tasks carry exact verification (`calib_set` is GSM8K train plus MBPP train).
    pub dataset: String,
    /// Seconds between idle checks.
    pub poll_s: u64,
    /// The most tasks answered in any hour, so the stream never takes the machine over.
    pub per_hour: u32,
    /// Orders the dataset (a hash of seed and task id); the position reached is kept in `cursor_file`.
    pub seed: u64,
    pub cursor_file: String,
}
impl Default for LabelCfg {
    fn default() -> Self {
        Self {
            enabled: false,
            dataset: "calib_set".into(),
            poll_s: 20,
            per_hour: 90,
            seed: 11,
            cursor_file: "data/runtime/label_cursor.json".into(),
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
    /// Tell the model how to write a chart or a diagram when the question asks for one (see `visual`).
    #[serde(default = "d_true")]
    pub visual_hints: bool,
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
    /// FR-R1 (Phase 22): the largest certified error rate at which a calibrated bin may keep a draft, and the
    /// confidence level of the one-sided Clopper-Pearson bound (Bonferroni over the bins).
    #[serde(default = "d_max_risk")]
    pub max_risk: f32,
    #[serde(default = "d_risk_alpha")]
    pub risk_alpha: f32,
    /// FR-R2: what the gate does when a feature it needs is missing: "escalate" (default), "keep" or "fixed".
    #[serde(default = "d_on_unknown")]
    pub on_unknown: String,
    /// FR-R3: content-word overlap with the previous user turn at or above which a question is a reask, and
    /// whether a reask of an unverified draft escalates under the fixed rule.
    #[serde(default = "d_reask_overlap")]
    pub reask_overlap: f32,
    #[serde(default = "d_true")]
    pub reask_escalates: bool,
    /// P7-T02 sequential sampling: under the fixed rule, draw the k cheap samples only after the draft and
    /// only when the draft's mean log-prob lies within `uncertain_band` of the threshold. Confident drafts
    /// skip them (three sequences on four cores tripled the factual draft time, benchmarks/samples_ab_factual.json).
    #[serde(default = "d_true")]
    pub samples_when_uncertain: bool,
    #[serde(default = "d_band")]
    pub uncertain_band: f32,
    /// FR-095: for domains whose samples are always drawn (`samples_on_unverified_domains`: math), draw them together
    /// with the draft in one batch rather than after it. Four streams cost 1.7 times one on this CPU, so draft plus two
    /// samples take about 1.8 draft times instead of 2.3; the samples are cancelled when the draft is stopped early.
    #[serde(default)]
    pub fuse_always_samples: bool,
    /// FR-096: when the draft fails its exact check, draw this many System 1 samples at once (one request, n choices) and
    /// verify them before escalating to System 2; 0 turns it off. At least 3 are needed to rescue anything on MBPP+
    /// (0, 0, 12 % and 19 % of failed drafts for 1 to 4 samples, `benchmarks/s1_rescue.json`).
    #[serde(default)]
    pub rescue_samples: u8,
    /// Sampling temperature of the rescue samples (the draft itself uses `deliberation.draft_temperature`).
    #[serde(default = "d_rescue_temp")]
    pub rescue_temperature: f32,
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
    /// FR-094: concurrent hedged attempts on System 2 for verifiable tasks.
    #[serde(default)]
    pub parallel: ParallelCfg,
}

/// FR-094 (`nafs_core::parallel`): width of a System 2 round on a verifiable task, from the measured cost of
/// batching concurrent streams and an online estimate of the one-attempt success probability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelCfg {
    #[serde(default)]
    pub enabled: bool,
    /// Most candidates in one round (at most the System 2 slots, `backends.*.system2.max_in_flight`).
    #[serde(default = "d_par_width")]
    pub max_width: u32,
    /// Measured wall-time factor of 1, 2, 3, ... concurrent streams at the depth of a repair prompt, about 1,600 tokens
    /// (`benchmarks/batch_scaling_system2_depth1200.json`). At short context the factors are 1.0, 1.5, 1.9, 1.8, but a repair
    /// prompt carries the earlier candidate and the verifier's evidence, and the batch gain fades with context depth.
    #[serde(default = "d_par_costs")]
    pub cost_factors: Vec<f32>,
    /// Widen only when verified answers per second improve by at least this fraction.
    #[serde(default = "d_par_gain")]
    pub min_gain: f32,
    /// Prior probability that one System 2 attempt passes on a task that reached it, per domain.
    #[serde(default = "d_par_prior")]
    pub p_prior: BTreeMap<String, f32>,
    /// The prior counts as this many attempts of evidence.
    #[serde(default = "d_par_weight")]
    pub prior_weight: f32,
    /// Run the first attempt alone and widen only after it fails. Measured on the 8 hard code items of the 100-prompt run
    /// (`benchmarks/parallel_ab.json`): four concurrent candidates from the start cost +44 % to +87 % on tasks that pass
    /// the first attempt and save only on failing ones, so the width is spent where the first attempt has failed.
    #[serde(default = "d_true")]
    pub hedge_after_failure: bool,
}
fn d_rescue_temp() -> f32 {
    0.7
}
fn d_par_width() -> u32 {
    4
}
fn d_par_costs() -> Vec<f32> {
    vec![1.0, 2.4, 3.1, 3.4]
}
fn d_par_gain() -> f32 {
    0.05
}
fn d_par_prior() -> BTreeMap<String, f32> {
    [("code", 0.35f32), ("math", 0.6), ("structured", 0.6)]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect()
}
fn d_par_weight() -> f32 {
    6.0
}
impl Default for ParallelCfg {
    fn default() -> Self {
        Self {
            enabled: false,
            max_width: d_par_width(),
            cost_factors: d_par_costs(),
            min_gain: d_par_gain(),
            p_prior: d_par_prior(),
            prior_weight: d_par_weight(),
            hedge_after_failure: true,
        }
    }
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
        // A test gateway can point at its own SurrealDB without editing the file.
        if let Ok(u) = std::env::var("NAFS_KB_URL") {
            cfg.knowledge.url = u;
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
