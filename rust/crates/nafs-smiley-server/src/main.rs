//! `nafs-smiley-server`: one GGUF model served by the vendored Smiley Core engine behind the subset of the
//! OpenAI chat API that `nafs_core::Backend` speaks. The gateway then treats it like llama-server or vLLM
//! (`[backends.smiley]`, Phase 15): `POST /v1/chat/completions` (streamed or not, with `usage` and llama-style
//! `timings`), `GET /v1/models` (the health check), `GET /health`, `GET /metrics`.
//!
//! What it maps, and how:
//! - `messages` -> a Smiley session: the system message is the session's system text, earlier user/assistant
//!   pairs are seeded as history, the last user message is the turn. The session is reset per request; its
//!   cache prefix is reused when the new prompt shares it (the system prompt, the agent's prompt).
//! - `chat_template_kwargs.enable_thinking` -> `SessionCfg::thinking` (Qwen3's empty reasoning block when off).
//! - `thinking_budget_tokens` -> the reasoning span is counted by token; at the budget the turn is cancelled and
//!   replanned with the reasoning so far plus `</think>` as the assistant's prefix (`SessionCfg::assistant_prefix`,
//!   a Nafs-Aql addition), so the answer follows as llama-server's budget forcing does.
//! - `logprobs` -> each content token's log-probability (`Emitted::logprob`, a Nafs-Aql addition), which the gate's
//!   fixed rule reads.
//! - `n > 1` and `grammar` are refused (400): the engine decodes one stream and has no constrained sampler; the
//!   gateway already treats a backend without them as it treats vLLM.
//!
//! - `memory_embeds` (Phase 20) -> Memorilla's memory tokens: the vectors replace the embeddings of the placeholder
//!   token's positions in the prompt (`Engine::admit_with_inputs`).
//!
//! One session per process (one stream at a time; requests queue): run one process per model, as with llama-server.
mod embed;

use axum::{
    extract::State,
    http::StatusCode,
    response::{sse::Event, IntoResponse, Response, Sse},
    routing::{get, post},
    Json, Router,
};
use clap::Parser;
use serde_json::{json, Value};
use smiley_core::model::Model;
use smiley_core::serve::{Emission, Engine, Finish, SlotCfg, SlotId};
use smiley_core::template::Template;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(about = "Serve a GGUF model with Smiley Core behind the OpenAI chat API")]
struct Args {
    /// The GGUF file (llama, qwen2 or qwen3; F32/F16/Q8_0/Q5_0/Q4_K/Q6_K tensors).
    #[arg(short = 'm', long)]
    model: std::path::PathBuf,
    /// The model id reported by /v1/models and expected in requests (any id is accepted).
    #[arg(long, default_value = "smiley")]
    alias: String,
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 8201)]
    port: u16,
    #[arg(short = 't', long)]
    threads: Option<usize>,
    /// Threads that read a prompt (prefill); default: the same as --threads. Reading a prompt scales with every
    /// core, writing tokens does not, so a machine that decodes best a thread short of its cores can still read
    /// prompts with all of them. For an embedding server, which only reads, this replaces --threads.
    #[arg(long = "prefill-threads")]
    prefill_threads: Option<usize>,
    /// Context length (clamped to the model's training context).
    #[arg(short = 'c', long, default_value_t = 8192)]
    ctx: usize,
    /// KV cache precision: f16 or i8.
    #[arg(long, default_value = "f16")]
    kv: String,
    /// A DSpark drafter directory (config.json + model.safetensors) trained for this model; greedy requests only.
    #[arg(long)]
    dspark: Option<std::path::PathBuf>,
    /// Prompt-lookup speculation on greedy requests (off: Nafs-Aql samples at a temperature, where it never applies).
    #[arg(long, default_value_t = false)]
    speculate: bool,
    /// Memory (MB) for snapshots of finished conversations' caches, so a prompt that shares a long beginning with an
    /// earlier one (not only the last) is not prefilled again; 0 = off. Not used with a DSpark drafter.
    #[arg(long = "prefix-cache-mb", default_value_t = 768)]
    prefix_cache_mb: usize,
    /// Users served at once, each in one row of the same weight pass. Above the machine's roofline ridge
    /// (about seven rows here) another row adds no throughput and every user waits longer for its own tokens.
    #[arg(long, default_value_t = smiley_core::serve::DEFAULT_MAX_SLOTS)]
    slots: usize,
    /// Run the projections a LittleBit sidecar (`<model>.gguf.slb`, built by `lb_convert`) covers from it. Off by
    /// default: post-training sidecars measured here produce wrong answers (docs/decisions/013), so a stale one
    /// beside a model must not switch the weights silently. A missing or corrupt sidecar refuses to start.
    #[arg(long, default_value_t = false)]
    littlebit: bool,
    /// Serve an embedding model (Qwen3-Embedding) behind `POST /v1/embeddings` instead of the chat API (Phase 20).
    #[arg(long, default_value_t = false)]
    embedding: bool,
    /// Longest text embedded, in tokens (embedding mode); longer texts are truncated.
    #[arg(long = "embed-max-tokens", default_value_t = 1024)]
    embed_max_tokens: usize,
}

struct App {
    /// The engine thread's queue: every request is served there, several at a time.
    jobs: std::sync::mpsc::Sender<EngineMsg>,
    engine: Arc<EngineShared>,
    model: Arc<Model>,
    template: Template,
    default_system: Option<&'static str>,
    /// Token ids that end a turn under this model's template.
    stop: Vec<u32>,
    max_ctx: usize,
    max_slots: usize,
    alias: String,
    think_open: Option<u32>,
    think_close: Option<u32>,
    /// Requests being served or waiting (llama-server's `requests_processing`).
    busy: AtomicUsize,
    requests: AtomicU64,
    tokens: AtomicU64,
    started: Instant,
    simd: &'static str,
    /// "dense" or "littlebit", reported in every response's timings.
    weights: &'static str,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();
    let a = Args::parse();
    let threads = a
        .threads
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get()));
    if a.embedding {
        embed::run(
            &a.model,
            a.alias.clone(),
            &a.host,
            a.port,
            a.prefill_threads.unwrap_or(threads),
            a.embed_max_tokens,
        )
        .await;
        return;
    }
    let t0 = Instant::now();
    let gguf = match smiley_core::gguf::Gguf::open(&a.model).and_then(|g| {
        smiley_core::check_compatible(&g)?;
        Ok(g)
    }) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("cannot open {}: {e}", a.model.display());
            std::process::exit(2);
        }
    };
    let template = match Template::detect(&gguf) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("cannot serve {}: {e}", a.model.display());
            std::process::exit(2);
        }
    };
    let model = match smiley_core::model::keep::open_with(&a.model, a.littlebit) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("cannot open {}: {e}", a.model.display());
            std::process::exit(2);
        }
    };
    let kv_prec = if a.kv == "i8" {
        smiley_core::kv::KvPrec::I8
    } else {
        smiley_core::kv::KvPrec::F16
    };
    let max_ctx = a.ctx.clamp(32, model.cfg.ctx_train);
    let prefill_threads = a.prefill_threads.unwrap_or(threads).max(1);
    let engine = Engine::with_prefill_threads(
        model.clone(),
        threads,
        prefill_threads,
        max_ctx,
        a.slots,
        kv_prec,
        a.prefix_cache_mb << 20,
    );
    let cache_mb = engine.cache_bytes_per_slot() >> 20;
    let (jobs, rx) = std::sync::mpsc::channel::<EngineMsg>();
    let shared = Arc::new(EngineShared::default());
    {
        let shared = shared.clone();
        std::thread::Builder::new()
            .name("smiley-engine".into())
            .spawn(move || engine_thread(engine, rx, shared))
            .expect("spawn the engine thread");
    }
    let app = Arc::new(App {
        think_open: model.tk.id_of("<think>"),
        think_close: model.tk.id_of("</think>"),
        stop: template.stop_ids(&model.tk),
        default_system: template.default_system(model.name()),
        template,
        max_ctx,
        max_slots: a.slots,
        alias: a.alias.clone(),
        busy: AtomicUsize::new(0),
        requests: AtomicU64::new(0),
        tokens: AtomicU64::new(0),
        started: Instant::now(),
        simd: smiley_core::simd_name(),
        weights: if model.lb_summary().is_some() {
            "littlebit"
        } else {
            "dense"
        },
        jobs,
        engine: shared,
        model,
    });
    let weights = match app.model.lb_summary() {
        Some((n, bytes)) => format!("LittleBit ({n} projections, {} MB of factors)", bytes >> 20),
        None => "dense".to_string(),
    };
    tracing::info!(
        "smiley-core: {} loaded in {:.1} s, {} threads ({} to read prompts), {}, ctx {}, {} slots ({} MB of cache each), {} weights, think tokens {:?}/{:?}",
        a.model.display(),
        t0.elapsed().as_secs_f64(),
        threads,
        prefill_threads,
        app.simd,
        max_ctx,
        a.slots,
        cache_mb,
        weights,
        app.think_open,
        app.think_close
    );
    let router = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/metrics", get(metrics))
        .route("/v1/chat/completions", post(chat))
        .with_state(app);
    let addr = format!("{}:{}", a.host, a.port);
    let listener = tokio::net::TcpListener::bind(&addr).await.expect("bind");
    tracing::info!("listening on http://{addr}");
    axum::serve(listener, router).await.expect("serve");
}

async fn health() -> Json<Value> {
    Json(json!({"status": "ok"}))
}

async fn models(State(s): State<Arc<App>>) -> Json<Value> {
    Json(
        json!({"object": "list", "data": [{"id": s.alias, "object": "model", "owned_by": "smiley-core", "weights": s.weights}]}),
    )
}

async fn metrics(State(s): State<Arc<App>>) -> String {
    format!(
        "# Smiley Core server\nllamacpp:requests_processing {}\nllamacpp:requests_deferred {}\nsmiley_slots_total {}\nsmiley_requests_total {}\nsmiley_tokens_predicted_total {}\nsmiley_decode_steps_total {}\nsmiley_decode_rows_total {}\nsmiley_uptime_seconds {}\n",
        s.engine.active.load(Ordering::SeqCst),
        s.engine.waiting.load(Ordering::SeqCst),
        s.max_slots,
        s.requests.load(Ordering::SeqCst),
        s.tokens.load(Ordering::SeqCst),
        s.engine.steps.load(Ordering::SeqCst),
        s.engine.rows.load(Ordering::SeqCst),
        s.started.elapsed().as_secs()
    )
}

fn bad(msg: impl Into<String>) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(
            json!({"error": {"message": msg.into(), "type": "invalid_request_error", "code": 400}}),
        ),
    )
        .into_response()
}

/// What one request asks for, read from the OpenAI-style body.
struct Plan {
    system: Option<String>,
    pairs: Vec<(String, String)>,
    user: String,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    top_k: usize,
    min_p: f32,
    seed: u64,
    thinking: bool,
    budget: Option<usize>,
    logprobs: bool,
    stream: bool,
    /// Memorilla's memory tokens (Phase 20): vectors for the placeholder token's positions in the prompt, in order.
    memory: Option<std::sync::Arc<Vec<Vec<f32>>>>,
    placeholder: String,
}

/// `memory_embeds: {"vectors": [[f32; d_model]; K], "placeholder"?: "<|fim_pad|>"}`: K vectors for the K places
/// the placeholder token takes in the prompt (Memorilla puts them in the system prompt).
/// Memory vectors (shared, as the request is planned twice when the reasoning budget closes) and the placeholder.
type Memory = (Option<std::sync::Arc<Vec<Vec<f32>>>>, String);

fn memory_of(b: &Value) -> Result<Memory, String> {
    let m = &b["memory_embeds"];
    let placeholder = m["placeholder"]
        .as_str()
        .unwrap_or("<|fim_pad|>")
        .to_string();
    if m.is_null() {
        return Ok((None, placeholder));
    }
    let rows = m["vectors"]
        .as_array()
        .ok_or("memory_embeds.vectors must be an array of vectors")?;
    if rows.is_empty() || rows.len() > 256 {
        return Err("memory_embeds.vectors must hold 1 to 256 vectors".into());
    }
    let vecs = rows
        .iter()
        .map(|r| {
            r.as_array()
                .ok_or("each memory vector must be an array of numbers")?
                .iter()
                .map(|x| {
                    x.as_f64()
                        .map(|v| v as f32)
                        .ok_or("each memory vector must be an array of numbers")
                })
                .collect::<Result<Vec<f32>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((Some(std::sync::Arc::new(vecs)), placeholder))
}

/// The prompt positions that take memory vectors: every occurrence of the placeholder token, which must match the
/// number of vectors.
fn memory_inputs(
    prompt: &[u32],
    placeholder: Option<u32>,
    vecs: &[Vec<f32>],
) -> Result<Vec<(usize, Vec<f32>)>, String> {
    let id =
        placeholder.ok_or("the placeholder is not a single token of this model's vocabulary")?;
    let at: Vec<usize> = prompt
        .iter()
        .enumerate()
        .filter(|(_, t)| **t == id)
        .map(|(i, _)| i)
        .collect();
    if at.len() != vecs.len() {
        return Err(format!(
            "the prompt holds {} memory placeholders for {} memory vectors",
            at.len(),
            vecs.len()
        ));
    }
    Ok(at.into_iter().zip(vecs.iter().cloned()).collect())
}

fn plan(b: &Value) -> Result<Plan, String> {
    if b["n"].as_u64().unwrap_or(1) > 1 {
        return Err("n > 1 is not supported: the engine decodes one stream".into());
    }
    if b.get("grammar").is_some_and(|g| !g.is_null())
        || b.get("json_schema").is_some_and(|g| !g.is_null())
    {
        return Err("grammar-constrained sampling is not supported by this backend".into());
    }
    let Some(msgs) = b["messages"].as_array() else {
        return Err("messages must be an array".into());
    };
    let text = |m: &Value| -> String {
        match &m["content"] {
            Value::String(s) => s.clone(),
            Value::Array(parts) => parts
                .iter()
                .filter_map(|p| p["text"].as_str())
                .collect::<Vec<_>>()
                .join(""),
            _ => String::new(),
        }
    };
    let mut system: Vec<String> = vec![];
    let mut pairs: Vec<(String, String)> = vec![];
    let mut pending_user: Option<String> = None;
    for m in msgs {
        match m["role"].as_str().unwrap_or("user") {
            "system" => system.push(text(m)),
            "user" => {
                if let Some(u) = pending_user.replace(text(m)) {
                    // Two user messages in a row: the earlier one got no answer.
                    pairs.push((u, String::new()));
                }
            }
            "assistant" => {
                let u = pending_user.take().unwrap_or_default();
                pairs.push((u, text(m)));
            }
            _ => {}
        }
    }
    let Some(user) = pending_user else {
        return Err("the last message must be from the user".into());
    };
    let kw = &b["chat_template_kwargs"];
    let (memory, placeholder) = memory_of(b)?;
    Ok(Plan {
        memory,
        placeholder,
        system: (!system.is_empty()).then(|| system.join("\n\n")),
        pairs,
        user,
        max_tokens: b["max_tokens"].as_u64().unwrap_or(256).clamp(1, 32_768) as usize,
        temperature: b["temperature"].as_f64().unwrap_or(0.8) as f32,
        top_p: b["top_p"].as_f64().unwrap_or(0.95) as f32,
        // llama-server's defaults, so the two engines sample alike unless a request says otherwise.
        top_k: b["top_k"].as_u64().unwrap_or(40) as usize,
        min_p: b["min_p"].as_f64().unwrap_or(0.05) as f32,
        seed: b["seed"].as_u64().unwrap_or(42),
        thinking: kw["enable_thinking"].as_bool().unwrap_or(true),
        budget: b["thinking_budget_tokens"]
            .as_u64()
            .or_else(|| b["reasoning_budget_tokens"].as_u64())
            .map(|v| v as usize),
        logprobs: b["logprobs"].as_bool().unwrap_or(false),
        stream: b["stream"].as_bool().unwrap_or(false),
    })
}

enum Msg {
    /// A piece of text: (text, is_reasoning, logprob).
    Delta(String, bool, f32),
    Done {
        finish: &'static str,
        usage: Value,
        timings: Value,
    },
    Err(String),
}

/// What one request produced, for its usage and timings block.
#[derive(Default)]
struct Run {
    in_think: bool,
    /// The model wrote the opening think tag itself (it is not part of `reasoning`).
    saw_open: bool,
    reasoning: String,
    content_tokens: u64,
    reasoning_tokens: u64,
    budget_hit: bool,
    prompt_n: u64,
    cache_n: u64,
    prompt_ms: f64,
    predicted_ms: f64,
    /// Decode seconds attributed to this request by the engine (its share of the shared steps).
    decode_ms: f64,
}

/// What the assistant has written when the reasoning budget is hit: the text that was generated, token for token, with
/// the think span closed. The model wrote `<think>` and then a reasoning that starts with its own newline; a second
/// newline here made the first reasoning token a different one, so the cache stopped matching at that position and every
/// reasoning token was prefilled again (measured: 108 tokens and 2.2 s at a budget of 200 on the 1.7B model, 5 and 0.2 s
/// after). A model that never wrote the opening tag gets it, with its newline.
fn continuation_prefix(saw_open: bool, reasoning: &str, closing: &str) -> String {
    if saw_open {
        format!("<think>{reasoning}{closing}")
    } else {
        format!("<think>\n{reasoning}{closing}")
    }
}

/// A request handed to the engine thread, and where its tokens go.
struct Job {
    prompt: Vec<u32>,
    cfg: SlotCfg,
    /// Memory vectors by prompt position (Memorilla, Phase 20).
    inputs: Vec<(usize, Vec<f32>)>,
    out: tokio::sync::mpsc::UnboundedSender<Emission>,
}

enum EngineMsg {
    Run(Box<Job>),
    Cancel(SlotId),
}

/// The engine thread: it owns the model and serves every request, several at a time in one pass over the weights.
///
/// One thread rather than a lock each request takes in turn, because the batching only exists if the steps of
/// different requests meet in the same place. It sleeps when there is nothing to serve.
fn engine_thread(
    mut engine: Engine,
    rx: std::sync::mpsc::Receiver<EngineMsg>,
    shared: Arc<EngineShared>,
) {
    let mut routes: HashMap<SlotId, tokio::sync::mpsc::UnboundedSender<Emission>> = HashMap::new();
    let mut waiting: VecDeque<Job> = VecDeque::new();
    let take = |msg,
                waiting: &mut VecDeque<Job>,
                engine: &mut Engine,
                routes: &mut HashMap<_, _>| match msg {
        EngineMsg::Run(j) => waiting.push_back(*j),
        EngineMsg::Cancel(id) => {
            engine.cancel(id);
            routes.remove(&id);
        }
    };
    loop {
        loop {
            match rx.try_recv() {
                Ok(m) => take(m, &mut waiting, &mut engine, &mut routes),
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => return,
            }
        }
        while engine.has_room() {
            let Some(j) = waiting.pop_front() else { break };
            // A caller that has gone is not worth a prefill.
            if j.out.is_closed() {
                continue;
            }
            match engine.admit_with_inputs(j.prompt, j.cfg, j.inputs) {
                Ok(id) => {
                    routes.insert(id, j.out);
                }
                Err((_, why)) => {
                    let _ = j.out.send(Emission {
                        slot: 0,
                        token: 0,
                        text: String::new(),
                        logprob: 0.0,
                        finish: Some(Finish::Error(why)),
                        cost: None,
                    });
                }
            }
        }
        shared.active.store(engine.active(), Ordering::SeqCst);
        shared.waiting.store(waiting.len(), Ordering::SeqCst);
        if engine.active() == 0 {
            // Nothing to serve: block until something arrives rather than spin on four cores.
            match rx.recv() {
                Ok(m) => take(m, &mut waiting, &mut engine, &mut routes),
                Err(_) => return,
            }
            continue;
        }
        for em in engine.step() {
            let done = em.finish.is_some();
            let slot = em.slot;
            if let Some(tx) = routes.get(&slot) {
                if tx.send(em).is_err() && !done {
                    // The caller has gone: stop generating for it.
                    engine.cancel(slot);
                    routes.remove(&slot);
                    continue;
                }
            }
            if done {
                routes.remove(&slot);
            }
        }
        shared.steps.store(engine.decode_steps, Ordering::SeqCst);
        shared.rows.store(engine.decode_rows, Ordering::SeqCst);
        let (entries, bytes, hits, saved) = engine.prefix_stats();
        shared.prefix.store(
            ((entries as u64) << 48)
                | ((bytes as u64 >> 20) << 32)
                | ((hits & 0xffff) << 16)
                | (saved & 0xffff),
            Ordering::SeqCst,
        );
        *shared.prefix_full.lock().unwrap_or_else(|e| e.into_inner()) =
            (entries, bytes, hits, saved);
        *shared.phases.lock().unwrap_or_else(|e| e.into_inner()) = engine.phases_ms();
    }
}

/// Counters the engine thread publishes for `/metrics` and the timings block.
#[derive(Default)]
struct EngineShared {
    active: AtomicUsize,
    waiting: AtomicUsize,
    steps: AtomicU64,
    rows: AtomicU64,
    prefix: AtomicU64,
    prefix_full: Mutex<(usize, usize, u64, u64)>,
    phases: Mutex<[f64; 6]>,
}

impl EngineShared {
    fn phases(&self) -> [f64; 6] {
        *self.phases.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// Serve one request: build its prompt, hand it to the engine, and turn the tokens it sends back into the
/// deltas the caller is streamed. The reasoning budget is enforced here, by replanning the turn with what the
/// model has written so far — the engine's prefix store then restores that reasoning instead of recomputing it.
async fn generate(app: &App, p: &Plan, tx: tokio::sync::mpsc::UnboundedSender<Msg>) {
    let t0 = Instant::now();
    let mut run = Run::default();
    let mut budget_left = p.max_tokens;
    let mut prefix: Option<String> = None;
    let mut finish = "stop";
    // At most two passes: the turn as asked, then the same turn continued with the reasoning closed.
    for pass in 0..2 {
        let thinking = p.thinking && pass == 0;
        let prompt_text = smiley_core::serve::chat_prompt(
            app.template,
            p.system.as_deref().or(app.default_system),
            &p.pairs,
            &p.user,
            thinking || prefix.is_some(),
            prefix.as_deref(),
        );
        let prompt = app.model.tk.encode(&prompt_text);
        let inputs = match &p.memory {
            Some(v) => match memory_inputs(&prompt, app.model.tk.id_of(&p.placeholder), v) {
                Ok(i) => i,
                Err(e) => {
                    let _ = tx.send(Msg::Err(e));
                    return;
                }
            },
            None => Vec::new(),
        };
        if prompt.len() + 1 >= app.max_ctx {
            let _ = tx.send(Msg::Err(format!(
                "the prompt is {} tokens; this model is served with {} positions",
                prompt.len(),
                app.max_ctx
            )));
            return;
        }
        let (etx, mut erx) = tokio::sync::mpsc::unbounded_channel::<Emission>();
        let cfg = SlotCfg {
            temperature: p.temperature,
            top_p: p.top_p,
            top_k: p.top_k,
            min_p: p.min_p,
            seed: p.seed,
            max_tokens: budget_left.max(1),
            stop: app.stop.clone(),
            logprobs: p.logprobs,
        };
        if app
            .jobs
            .send(EngineMsg::Run(Box::new(Job {
                prompt,
                cfg,
                inputs,
                out: etx,
            })))
            .is_err()
        {
            let _ = tx.send(Msg::Err("the engine is not running".into()));
            return;
        }
        let mut replan = false;
        let mut slot = 0;
        while let Some(em) = erx.recv().await {
            slot = em.slot;
            if let Some(c) = &em.cost {
                run.prompt_n += c.prefilled as u64;
                run.cache_n += (c.prompt_tokens - c.prefilled) as u64;
                run.prompt_ms += c.prefill_s * 1e3;
                run.decode_ms += c.decode_s * 1e3;
            }
            if let Some(f) = &em.finish {
                match f {
                    Finish::Error(e) => {
                        let _ = tx.send(Msg::Err(e.clone()));
                        return;
                    }
                    Finish::Stop => finish = "stop",
                    Finish::Length | Finish::Context => finish = "length",
                }
                if em.text.is_empty() {
                    break;
                }
            }
            // The think tags mark the span; they are not part of either text.
            if Some(em.token) == app.think_open && pass == 0 {
                run.in_think = true;
                run.saw_open = true;
                continue;
            }
            if Some(em.token) == app.think_close {
                run.in_think = false;
                continue;
            }
            if run.in_think {
                run.reasoning_tokens += 1;
                run.reasoning.push_str(&em.text);
                budget_left = budget_left.saturating_sub(1);
                if tx.send(Msg::Delta(em.text, true, em.logprob)).is_err() {
                    break;
                }
                if p.budget.is_some_and(|b| run.reasoning_tokens as usize >= b) {
                    run.budget_hit = true;
                    replan = true;
                    break;
                }
            } else {
                run.content_tokens += 1;
                budget_left = budget_left.saturating_sub(1);
                if tx.send(Msg::Delta(em.text, false, em.logprob)).is_err() {
                    break;
                }
            }
            if em.finish.is_some() {
                break;
            }
        }
        if !replan {
            break;
        }
        // Close the reasoning and continue the same turn. The cache of the pass just cancelled holds every
        // reasoning token, and the engine's prefix store hands it to the continuation.
        let _ = app.jobs.send(EngineMsg::Cancel(slot));
        let closing = "\n</think>\n\n";
        let _ = tx.send(Msg::Delta(closing.to_string(), true, 0.0));
        prefix = Some(continuation_prefix(run.saw_open, &run.reasoning, closing));
        run.in_think = false;
        if budget_left == 0 {
            finish = "length";
            break;
        }
    }
    let predicted_n = run.content_tokens + run.reasoning_tokens;
    app.tokens.fetch_add(predicted_n, Ordering::SeqCst);
    let (entries, bytes, hits, saved) = *app
        .engine
        .prefix_full
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let (steps, rows) = (
        app.engine.steps.load(Ordering::SeqCst),
        app.engine.rows.load(Ordering::SeqCst),
    );
    // The decode time attributed to this request (its share of the steps it was in), not the wall clock: with
    // other users in the same pass the wall clock includes their tokens too.
    if run.predicted_ms == 0.0 {
        run.predicted_ms = run.decode_ms;
    }
    let wall_ms = t0.elapsed().as_secs_f64() * 1e3;
    let _ = tx.send(Msg::Done {
        finish,
        usage: json!({"prompt_tokens": run.prompt_n + run.cache_n, "completion_tokens": predicted_n, "total_tokens": run.prompt_n + run.cache_n + predicted_n}),
        timings: json!({"prompt_n": run.prompt_n, "cache_n": run.cache_n, "prompt_ms": run.prompt_ms, "predicted_n": predicted_n, "predicted_ms": run.predicted_ms,
            "predicted_per_second": if run.predicted_ms > 0.0 { predicted_n as f64 / run.predicted_ms * 1e3 } else { 0.0 },
            "smiley": {"simd": app.simd, "weights": app.weights, "budget_forced": run.budget_hit, "wall_ms": wall_ms,
                "phases_ms": {"names": ["matmul", "head", "attention", "other", "sampling", "drafting"], "decode": app.engine.phases(), "prefill": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
                "concurrency": {"slots": app.max_slots, "active": app.engine.active.load(Ordering::SeqCst), "waiting": app.engine.waiting.load(Ordering::SeqCst),
                    "decode_steps": steps, "decode_rows": rows,
                    "rows_per_step": if steps > 0 { rows as f64 / steps as f64 } else { 0.0 }},
                "prefix_store": {"entries": entries, "mb": bytes >> 20, "restores": hits, "positions_saved": saved}}}),
    });
}

const MAX_WAITING: usize = 8;

/// Reserve one slot of `cap` on a counter: false when it is full. A compare-exchange loop rather than
/// `fetch_update`, which newer toolchains deprecate (renamed `try_update`) and older ones lack.
fn try_reserve(counter: &AtomicUsize, cap: usize) -> bool {
    let mut n = counter.load(Ordering::SeqCst);
    while n < cap {
        match counter.compare_exchange_weak(n, n + 1, Ordering::SeqCst, Ordering::SeqCst) {
            Ok(_) => return true,
            Err(now) => n = now,
        }
    }
    false
}

/// Counts a request as processing (the `requests_processing` metric) until dropped.
struct BusyGuard(Arc<App>);
impl Drop for BusyGuard {
    fn drop(&mut self) {
        self.0.busy.fetch_sub(1, Ordering::SeqCst);
    }
}

async fn chat(State(app): State<Arc<App>>, Json(body): Json<Value>) -> Response {
    let p = match plan(&body) {
        Ok(p) => p,
        Err(e) => return bad(e),
    };
    // One session decodes one request at a time; the others wait. A bounded wait queue keeps a burst of (possibly
    // abandoned) long prompts from piling up blocking threads and CPU-hours of prefill.
    if !try_reserve(&app.busy, MAX_WAITING) {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": {"message": "too many requests are waiting for this model", "type": "overloaded_error", "code": 503}})),
        )
            .into_response();
    }
    let busy = BusyGuard(app.clone());
    app.requests.fetch_add(1, Ordering::SeqCst);
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Msg>();
    let app2 = app.clone();
    let p = Arc::new(p);
    let p2 = p.clone();
    tokio::spawn(async move {
        let _busy = busy; // released when the generation ends, also if it panics
        generate(&app2, &p2, tx).await;
    });
    let id = format!("chatcmpl-{}", app.requests.load(Ordering::SeqCst));
    let model = app.alias.clone();
    if p.stream {
        let logprobs = p.logprobs;
        let stream = async_stream(rx, id, model, logprobs);
        return Sse::new(stream).into_response();
    }
    // Not streamed: collect everything.
    let mut rx = rx;
    let (mut content, mut reasoning, mut lps) = (String::new(), String::new(), vec![]);
    while let Some(m) = rx.recv().await {
        match m {
            Msg::Delta(t, r, lp) => {
                if r {
                    reasoning.push_str(&t)
                } else {
                    content.push_str(&t);
                    lps.push(json!({"token": t, "logprob": lp}));
                }
            }
            Msg::Done {
                finish,
                usage,
                timings,
            } => {
                let mut choice = json!({"index": 0, "message": {"role": "assistant", "content": content, "reasoning_content": reasoning}, "finish_reason": finish});
                if p.logprobs {
                    choice["logprobs"] = json!({"content": lps});
                }
                return Json(json!({"id": id, "object": "chat.completion", "model": model, "choices": [choice], "usage": usage, "timings": timings})).into_response();
            }
            Msg::Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({"error": {"message": e, "type": "server_error", "code": 500}})),
                )
                    .into_response()
            }
        }
    }
    (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": {"message": "the generation ended without a result", "type": "server_error", "code": 500}}))).into_response()
}

fn async_stream(
    rx: tokio::sync::mpsc::UnboundedReceiver<Msg>,
    id: String,
    model: String,
    logprobs: bool,
) -> impl futures::Stream<Item = Result<Event, std::convert::Infallible>> {
    use tokio_stream::StreamExt;
    tokio_stream::wrappers::UnboundedReceiverStream::new(rx).map(move |m| {
        let v = match m {
            Msg::Delta(t, r, lp) => {
                let mut choice = json!({"index": 0, "delta": if r { json!({"reasoning_content": t}) } else { json!({"content": t}) }, "finish_reason": Value::Null});
                if logprobs && !r {
                    choice["logprobs"] = json!({"content": [{"token": t, "logprob": lp}]});
                }
                json!({"id": id, "object": "chat.completion.chunk", "model": model, "choices": [choice]})
            }
            Msg::Done {
                finish,
                usage,
                timings,
            } => {
                json!({"id": id, "object": "chat.completion.chunk", "model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}], "usage": usage, "timings": timings})
            }
            Msg::Err(e) => {
                json!({"error": {"message": e}})
            }
        };
        Ok(Event::default().data(v.to_string()))
    })
    // `[DONE]` after the last chunk, as llama-server and vLLM send it.
    .chain(tokio_stream::once(Ok(Event::default().data("[DONE]"))))
}

#[cfg(test)]
mod tests {

    #[test]
    fn memory_vectors_go_to_the_placeholders_in_order() {
        use crate::{memory_inputs, plan};
        let b = serde_json::json!({"messages": [{"role": "user", "content": "q"}],
            "memory_embeds": {"vectors": [[1.0, 2.0], [3.0, 4.0]]}});
        let p = plan(&b).unwrap();
        let v = p.memory.unwrap();
        assert_eq!(p.placeholder, "<|fim_pad|>");
        let inputs = memory_inputs(&[5, 9, 7, 9, 2], Some(9), &v).unwrap();
        assert_eq!(inputs, vec![(1, vec![1.0, 2.0]), (3, vec![3.0, 4.0])]);
        assert!(memory_inputs(&[5, 9, 2], Some(9), &v)
            .unwrap_err()
            .contains("1 memory placeholders for 2"));
        assert!(memory_inputs(&[5, 9, 2], None, &v).is_err());
        let bad = serde_json::json!({"messages": [{"role": "user", "content": "q"}], "memory_embeds": {"vectors": [["x"]]}});
        assert!(plan(&bad).is_err());
    }

    use super::continuation_prefix;

    #[test]
    fn the_continuation_repeats_what_was_generated_without_an_extra_newline() {
        let closing = "\n</think>\n\n";
        // The model wrote <think>, then "\nOkay, ...": the prefix is those two pieces, not <think>\n + "\nOkay".
        assert_eq!(
            continuation_prefix(true, "\nOkay, let me think", closing),
            "<think>\nOkay, let me think\n</think>\n\n"
        );
        assert!(!continuation_prefix(true, "\nOkay", closing).contains("<think>\n\n"));
        // A model that never wrote the opening tag gets one.
        assert_eq!(
            continuation_prefix(false, "Okay", closing),
            "<think>\nOkay\n</think>\n\n"
        );
    }
}
