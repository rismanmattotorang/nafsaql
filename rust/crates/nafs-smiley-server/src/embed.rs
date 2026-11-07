//! `--embedding`: serve an embedding model (Qwen3-Embedding GGUF) behind OpenAI's `POST /v1/embeddings` instead of
//! the chat API (Phase 20, the Knowledge Base). One engine thread embeds one text at a time; requests queue.
//!
//! Request: `{"input": "text" | ["text", ...], "instruction"?: "task description"}`. With an instruction each text is
//! sent as Qwen3-Embedding's query format, `Instruct: {instruction}\nQuery:{text}`; documents go without one.
//! Response: `{"object": "list", "data": [{"object": "embedding", "index": i, "embedding": [...]}], "model", "usage",
//! "timings": {"ms", "per_text_ms"}}`.

use axum::{
    extract::State, http::StatusCode, response::IntoResponse, routing::get, routing::post, Json,
    Router,
};
use serde_json::{json, Value};
use smiley_core::embed::Embedder;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Texts per request; a knowledge-base ingestion sends batches well below this.
const MAX_INPUTS: usize = 256;

/// One text's vector, its token count, and whether it was truncated.
type Embedding = (Vec<f32>, usize, bool);

struct Job {
    texts: Vec<String>,
    reply: tokio::sync::oneshot::Sender<Result<Vec<Embedding>, String>>,
}

struct App {
    jobs: std::sync::mpsc::Sender<Job>,
    alias: String,
    dims: usize,
    max_tokens: usize,
    requests: AtomicU64,
    texts: AtomicU64,
    tokens: AtomicU64,
}

pub async fn run(
    model_path: &std::path::Path,
    alias: String,
    host: &str,
    port: u16,
    threads: usize,
    max_tokens: usize,
) {
    let t0 = Instant::now();
    let model = match smiley_core::model::keep::open_with(model_path, false) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("cannot open {}: {e}", model_path.display());
            std::process::exit(2);
        }
    };
    let mut embedder = Embedder::new(model, threads, max_tokens);
    let (dims, max_tokens) = (embedder.dims(), embedder.max_tokens);
    let (tx, rx) = std::sync::mpsc::channel::<Job>();
    std::thread::Builder::new()
        .name("smiley-embed".into())
        .spawn(move || {
            while let Ok(job) = rx.recv() {
                let out = job
                    .texts
                    .iter()
                    .map(|t| embedder.embed(t).map(|e| (e.vector, e.tokens, e.truncated)))
                    .collect::<Result<Vec<_>, String>>();
                let _ = job.reply.send(out);
            }
        })
        .expect("spawn the embedding thread");
    tracing::info!(
        "smiley-core embeddings: {} loaded in {:.1} s, {} threads, {} dims, up to {} tokens per text",
        model_path.display(),
        t0.elapsed().as_secs_f64(),
        threads,
        dims,
        max_tokens
    );
    let app = Arc::new(App {
        jobs: tx,
        alias,
        dims,
        max_tokens,
        requests: AtomicU64::new(0),
        texts: AtomicU64::new(0),
        tokens: AtomicU64::new(0),
    });
    let router = Router::new()
        .route("/health", get(|| async { Json(json!({"status": "ok"})) }))
        .route("/v1/models", get(models))
        .route("/metrics", get(metrics))
        .route("/v1/embeddings", post(embeddings))
        .with_state(app);
    let addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await.expect("bind");
    tracing::info!("listening on http://{addr} (embeddings)");
    axum::serve(listener, router).await.expect("serve");
}

async fn models(State(s): State<Arc<App>>) -> Json<Value> {
    Json(
        json!({"object": "list", "data": [{"id": s.alias, "object": "model", "owned_by": "smiley-core",
        "kind": "embedding", "dims": s.dims, "max_tokens": s.max_tokens}]}),
    )
}

async fn metrics(State(s): State<Arc<App>>) -> String {
    format!(
        "# Smiley Core embeddings\nsmiley_embed_requests_total {}\nsmiley_embed_texts_total {}\nsmiley_embed_tokens_total {}\n",
        s.requests.load(Ordering::Relaxed),
        s.texts.load(Ordering::Relaxed),
        s.tokens.load(Ordering::Relaxed)
    )
}

fn bad(msg: &str) -> axum::response::Response {
    (
        StatusCode::BAD_REQUEST,
        Json(json!({"error": {"message": msg, "type": "invalid_request_error"}})),
    )
        .into_response()
}

/// The texts as the model reads them: Qwen3-Embedding's query format when an instruction is given.
pub fn model_inputs(inputs: Vec<String>, instruction: Option<&str>) -> Vec<String> {
    match instruction.map(str::trim).filter(|i| !i.is_empty()) {
        Some(i) => inputs
            .into_iter()
            .map(|t| format!("Instruct: {i}\nQuery:{t}"))
            .collect(),
        None => inputs,
    }
}

async fn embeddings(
    State(s): State<Arc<App>>,
    Json(body): Json<Value>,
) -> axum::response::Response {
    let inputs: Vec<String> = match body.get("input") {
        Some(Value::String(t)) => vec![t.clone()],
        Some(Value::Array(a)) if a.iter().all(Value::is_string) => a
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect(),
        _ => return bad("`input` must be a string or an array of strings"),
    };
    if inputs.is_empty() || inputs.len() > MAX_INPUTS {
        return bad(&format!("between 1 and {MAX_INPUTS} inputs per request"));
    }
    let texts = model_inputs(inputs, body.get("instruction").and_then(Value::as_str));
    let t0 = Instant::now();
    let (reply, rx) = tokio::sync::oneshot::channel();
    let n = texts.len();
    if s.jobs.send(Job { texts, reply }).is_err() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": {"message": "the embedding thread stopped"}})),
        )
            .into_response();
    }
    let out = match rx.await {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": {"message": e}})),
            )
                .into_response()
        }
        Err(_) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"error": {"message": "the embedding thread stopped"}})),
            )
                .into_response()
        }
    };
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    let tokens: usize = out.iter().map(|o| o.1).sum();
    let truncated = out.iter().filter(|o| o.2).count();
    s.requests.fetch_add(1, Ordering::Relaxed);
    s.texts.fetch_add(n as u64, Ordering::Relaxed);
    s.tokens.fetch_add(tokens as u64, Ordering::Relaxed);
    let data: Vec<Value> = out
        .into_iter()
        .enumerate()
        .map(|(i, (v, _, _))| json!({"object": "embedding", "index": i, "embedding": v}))
        .collect();
    Json(json!({
        "object": "list", "data": data, "model": s.alias,
        "usage": {"prompt_tokens": tokens, "total_tokens": tokens, "truncated_inputs": truncated},
        "timings": {"ms": ms, "per_text_ms": ms / n as f64, "tokens_per_s": tokens as f64 / (ms / 1000.0).max(1e-9)},
    }))
    .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queries_get_the_instruction_format_documents_do_not() {
        let q = model_inputs(
            vec!["what is K?".into()],
            Some("Given a question, retrieve passages that answer it"),
        );
        assert_eq!(
            q[0],
            "Instruct: Given a question, retrieve passages that answer it\nQuery:what is K?"
        );
        assert_eq!(
            model_inputs(vec!["doc".into()], None),
            vec!["doc".to_string()]
        );
        assert_eq!(
            model_inputs(vec!["doc".into()], Some("  ")),
            vec!["doc".to_string()]
        );
    }
}
