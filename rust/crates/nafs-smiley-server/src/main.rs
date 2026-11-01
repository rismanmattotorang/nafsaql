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
//! One session per process (one stream at a time; requests queue): run one process per model, as with llama-server.
use axum::{
    extract::State,
    http::StatusCode,
    response::{sse::Event, IntoResponse, Response, Sse},
    routing::{get, post},
    Json, Router,
};
use clap::Parser;
use serde_json::{json, Value};
use smiley_core::session::{Session, SessionCfg};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
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
}

struct App {
    sess: Mutex<Session>,
    alias: String,
    think_open: Option<u32>,
    think_close: Option<u32>,
    /// Requests being served or waiting for the session (llama-server's `requests_processing`).
    busy: AtomicUsize,
    requests: AtomicU64,
    tokens: AtomicU64,
    started: Instant,
    simd: &'static str,
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
    let cfg = SessionCfg {
        max_ctx: a.ctx,
        kv_prec: if a.kv == "i8" {
            smiley_core::kv::KvPrec::I8
        } else {
            smiley_core::kv::KvPrec::F16
        },
        threads,
        speculate: a.speculate,
        dspark: a.dspark.clone(),
        ..Default::default()
    };
    let t0 = Instant::now();
    let sess = match Session::open(&a.model, cfg) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("cannot open {}: {e}", a.model.display());
            std::process::exit(2);
        }
    };
    let tk = &sess.model.tk;
    let app = Arc::new(App {
        think_open: tk.id_of("<think>"),
        think_close: tk.id_of("</think>"),
        alias: a.alias.clone(),
        busy: AtomicUsize::new(0),
        requests: AtomicU64::new(0),
        tokens: AtomicU64::new(0),
        started: Instant::now(),
        simd: smiley_core::simd_name(),
        sess: Mutex::new(sess),
    });
    tracing::info!(
        "smiley-core: {} loaded in {:.1} s, {} threads, {}, ctx {}, think tokens {:?}/{:?}",
        a.model.display(),
        t0.elapsed().as_secs_f64(),
        threads,
        app.simd,
        a.ctx,
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
        json!({"object": "list", "data": [{"id": s.alias, "object": "model", "owned_by": "smiley-core"}]}),
    )
}

async fn metrics(State(s): State<Arc<App>>) -> String {
    format!(
        "# Smiley Core server\nllamacpp:requests_processing {}\nsmiley_requests_total {}\nsmiley_tokens_predicted_total {}\nsmiley_uptime_seconds {}\n",
        s.busy.load(Ordering::SeqCst),
        s.requests.load(Ordering::SeqCst),
        s.tokens.load(Ordering::SeqCst),
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
    Ok(Plan {
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

/// Counters and text of one request across its passes.
#[derive(Default)]
struct Run {
    in_think: bool,
    reasoning: String,
    content_tokens: u64,
    reasoning_tokens: u64,
    budget_hit: bool,
    prompt_n: u64,
    cache_n: u64,
    prompt_ms: f64,
    predicted_ms: f64,
    bytes: u64,
}

/// One pass of the session over the turn, streaming deltas; `phase2` continues after the reasoning budget.
#[allow(clippy::too_many_arguments)]
fn one_pass(
    sess: &mut Session,
    p: &Plan,
    tx: &tokio::sync::mpsc::UnboundedSender<Msg>,
    cancel: &AtomicBool,
    think: (Option<u32>, Option<u32>),
    phase2: bool,
    run: &mut Run,
) -> Result<(), String> {
    let budget = p.budget;
    let (_, telem) = sess.generate_with(&p.user, Some(cancel), |e| {
        if cancel.load(Ordering::Relaxed) {
            return;
        }
        if Some(e.id) == think.0 && !phase2 {
            run.in_think = true;
            return;
        }
        if Some(e.id) == think.1 {
            run.in_think = false;
            return;
        }
        if run.in_think {
            run.reasoning_tokens += 1;
            run.reasoning.push_str(e.text);
            if tx
                .send(Msg::Delta(e.text.to_string(), true, e.logprob))
                .is_err()
            {
                cancel.store(true, Ordering::Relaxed);
            }
            if budget.is_some_and(|b| run.reasoning_tokens as usize >= b) {
                run.budget_hit = true;
                cancel.store(true, Ordering::Relaxed);
            }
        } else {
            run.content_tokens += 1;
            if tx
                .send(Msg::Delta(e.text.to_string(), false, e.logprob))
                .is_err()
            {
                cancel.store(true, Ordering::Relaxed);
            }
        }
    })?;
    run.prompt_n += telem.prompt_tokens as u64;
    run.cache_n += telem.prefix_reused as u64;
    run.prompt_ms += telem.prefill_s * 1e3;
    run.predicted_ms += telem.decode_steps_s * 1e3;
    run.bytes += telem.bytes_total();
    Ok(())
}

/// Run one request on the session: results go through `tx`. Blocking; call from `spawn_blocking`.
fn generate(app: &App, p: &Plan, tx: tokio::sync::mpsc::UnboundedSender<Msg>) {
    let mut sess = match app.sess.lock() {
        Ok(s) => s,
        Err(poisoned) => poisoned.into_inner(),
    };
    // The client may have left while this request waited for the session: nobody wants the prefill.
    if tx.is_closed() {
        return;
    }
    let cancel = AtomicBool::new(false);
    sess.reset();
    sess.cfg.system = p.system.clone();
    sess.cfg.thinking = p.thinking;
    sess.cfg.temperature = p.temperature;
    sess.cfg.top_p = p.top_p;
    sess.cfg.top_k = p.top_k;
    sess.cfg.min_p = p.min_p;
    sess.cfg.max_tokens = p.max_tokens;
    sess.cfg.assistant_prefix = None;
    sess.reseed(p.seed);
    sess.seed_history(p.pairs.clone());
    let think = (app.think_open, app.think_close);
    let mut run = Run::default();
    // Phase 1: the turn as asked. Reasoning tokens are counted against the budget; at the budget the turn is
    // cancelled and continued in phase 2 with the think span closed.
    if let Err(e) = one_pass(&mut sess, p, &tx, &cancel, think, false, &mut run) {
        let _ = tx.send(Msg::Err(e));
        return;
    }
    let client_gone = cancel.load(Ordering::Relaxed) && !run.budget_hit;
    if run.budget_hit && !client_gone {
        // Phase 2: the same turn, with the reasoning so far and `</think>` as what the assistant has written.
        cancel.store(false, Ordering::Relaxed);
        let closing = "\n</think>\n\n";
        let _ = tx.send(Msg::Delta(closing.to_string(), true, 0.0));
        sess.reset();
        sess.seed_history(p.pairs.clone());
        sess.cfg.assistant_prefix = Some(format!("<think>\n{}{closing}", run.reasoning));
        sess.cfg.max_tokens = p
            .max_tokens
            .saturating_sub(run.reasoning_tokens as usize)
            .max(16);
        run.in_think = false;
        if let Err(e) = one_pass(&mut sess, p, &tx, &cancel, think, true, &mut run) {
            let _ = tx.send(Msg::Err(e));
            return;
        }
        sess.cfg.assistant_prefix = None;
    }
    let predicted_n = run.content_tokens + run.reasoning_tokens;
    let finish = if !client_gone && predicted_n as usize >= p.max_tokens {
        "length"
    } else {
        "stop"
    };
    app.tokens.fetch_add(predicted_n, Ordering::SeqCst);
    let (prompt_n, cache_n, bytes) = (run.prompt_n, run.cache_n, run.bytes);
    let _ = tx.send(Msg::Done {
        finish,
        usage: json!({"prompt_tokens": prompt_n + cache_n, "completion_tokens": predicted_n, "total_tokens": prompt_n + cache_n + predicted_n}),
        timings: json!({"prompt_n": prompt_n, "cache_n": cache_n, "prompt_ms": run.prompt_ms, "predicted_n": predicted_n, "predicted_ms": run.predicted_ms,
            "predicted_per_second": if run.predicted_ms > 0.0 { predicted_n as f64 / run.predicted_ms * 1e3 } else { 0.0 },
            "smiley": {"bytes_moved": bytes, "bytes_per_token": bytes.checked_div(predicted_n).unwrap_or(0), "simd": app.simd, "budget_forced": run.budget_hit}}),
    });
}

/// Requests that may be queued behind the running one (the gateway sends at most one per model at a time).
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
    tokio::task::spawn_blocking(move || {
        let _busy = busy; // released when the generation ends, also if it panics
        generate(&app2, &p2, tx);
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
