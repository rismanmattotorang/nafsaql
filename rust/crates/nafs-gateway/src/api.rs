//! HTTP, SSE and WebSocket handlers.
use axum::extract::{Path, Query, Request, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::middleware::{self, Next};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::StreamExt;
use metrics_exporter_prometheus::PrometheusHandle;
use nafs_core::orchestrator::{Emitter, Orchestrator};
use nafs_core::Config;
use nafs_memory::Store;
use nafs_types::{
    ChatRequest, CodeAgentReply, CodeAgentRequest, NafsError, StreamEvent, VerificationInputs,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use uuid::Uuid;

const OPENAPI_YAML: &str = include_str!("../../../../api/openapi.yaml");
const OPENAPI_JSON: &str = include_str!("../../../../api/openapi.json");
pub struct AppStateInner {
    pub cfg: Arc<Config>,
    pub orch: Arc<Orchestrator>,
    pub store: Arc<Store>,
    pub prom: PrometheusHandle,
    pub tokens: Vec<String>,
    pub web_dir: Option<String>,
    pub started: Instant,
    pub allowed_hosts: Vec<String>,
    pub(crate) rooms: Mutex<HashMap<Uuid, Arc<crate::rooms::RoomHub>>>,
    rate: Mutex<HashMap<String, (Instant, u32)>>,
    cancels: Mutex<HashMap<Uuid, Arc<AtomicBool>>>,
    /// Recent request outcomes (time, ok) for the error-rate alert (OR-004).
    outcomes: Mutex<std::collections::VecDeque<(Instant, bool)>>,
    /// Alert drill: while set and in the future, every rule reports firing (P5 exit criterion).
    drill_until: Mutex<Option<Instant>>,
}
pub type AppState = Arc<AppStateInner>;

pub fn new_state(
    cfg: Arc<Config>,
    orch: Arc<Orchestrator>,
    store: Arc<Store>,
    prom: PrometheusHandle,
    tokens: Vec<String>,
    web_dir: Option<String>,
) -> AppState {
    let mut allowed_hosts: Vec<String> =
        vec!["localhost".into(), "127.0.0.1".into(), "[::1]".into()];
    if let Some(h) = cfg.server.bind.rsplit_once(':').map(|(h, _)| h.to_string()) {
        if h != "0.0.0.0" && h != "[::]" {
            allowed_hosts.push(h);
        }
    }
    if let Ok(extra) = std::env::var("NAFS_ALLOWED_HOSTS") {
        allowed_hosts.extend(
            extra
                .split(',')
                .map(|s| s.trim().to_lowercase())
                .filter(|s| !s.is_empty()),
        );
    }
    Arc::new(AppStateInner {
        cfg,
        orch,
        store,
        prom,
        tokens,
        web_dir,
        started: Instant::now(),
        allowed_hosts,
        rooms: Mutex::new(HashMap::new()),
        rate: Mutex::new(HashMap::new()),
        cancels: Mutex::new(HashMap::new()),
        outcomes: Mutex::new(std::collections::VecDeque::new()),
        drill_until: Mutex::new(None),
    })
}

pub struct ApiError(pub NafsError);
impl From<NafsError> for ApiError {
    fn from(e: NafsError) -> Self {
        ApiError(e)
    }
}
impl From<anyhow::Error> for ApiError {
    fn from(e: anyhow::Error) -> Self {
        ApiError(NafsError::BadRequest(e.to_string()))
    }
}
impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status =
            StatusCode::from_u16(self.0.http_status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        let mut r = (
            status,
            Json(json!({"code": self.0.code(), "message": self.0.to_string()})),
        )
            .into_response();
        if status == StatusCode::TOO_MANY_REQUESTS {
            r.headers_mut()
                .insert(header::RETRY_AFTER, "5".parse().unwrap());
        }
        r
    }
}
pub(crate) type ApiResult<T> = Result<T, ApiError>;

pub fn router(state: AppState) -> Router {
    let v1 = Router::new()
        .route("/v1/chat", post(chat))
        .route("/v1/chat/stream", post(chat_stream))
        .route("/v1/conversations", get(list_conversations))
        .route(
            "/v1/conversations/{id}",
            get(get_conversation)
                .patch(rename_conversation)
                .delete(delete_conversation),
        )
        .route("/v1/decisions/{id}", get(get_decision))
        .route(
            "/v1/rooms",
            get(crate::rooms::list_rooms).post(crate::rooms::create_room),
        )
        .route(
            "/v1/rooms/{id}",
            get(crate::rooms::get_room)
                .patch(crate::rooms::rename_room)
                .delete(crate::rooms::delete_room),
        )
        .route("/ws/rooms/{id}", get(crate::rooms::room_socket))
        .route(
            "/v1/bench/prompt-sets",
            get(|| async { Json(nafs_bench::prompt_sets()) }),
        )
        .route("/v1/bench/runs", get(list_bench).post(start_bench))
        .route("/v1/bench/runs/{id}", get(get_run))
        .route("/v1/bench/runs/{id}/cancel", post(cancel_run))
        .route("/v1/eval/datasets", get(list_datasets))
        .route("/v1/eval/datasets/{name}", get(preview_dataset))
        .route("/v1/eval/runs", get(list_eval).post(start_eval))
        .route("/v1/eval/runs/{id}", get(get_run))
        .route("/v1/eval/runs/{id}/cancel", post(cancel_run))
        .route("/v1/eval/runs/{id}/calibrate", post(calibrate))
        .route("/v1/eval/runs/{id}/resume", post(resume_eval))
        .route("/v1/verify", post(verify_adhoc))
        .route("/v1/harness", get(harness_info))
        .route("/v1/harness/code", post(code_agent))
        .route("/v1/judge", post(judge_adhoc))
        .route("/v1/verifiers", get(get_rates))
        .route("/v1/verifiers/runs", get(list_rate_runs).post(start_rates))
        .route("/v1/verifiers/runs/{id}", get(get_run))
        .route("/v1/verifiers/runs/{id}/cancel", post(cancel_run))
        .route("/v1/gate", get(get_gate).post(set_gate))
        .route("/v1/gate/recalibrate", post(recalibrate))
        .route("/v1/gate/derive", post(derive_params))
        .route("/v1/gate/params", get(get_params).delete(reset_params))
        .route("/v1/audits", get(list_audits))
        .route("/v1/audits/run-next", post(run_next_audit))
        .route("/v1/self-model", get(self_model))
        .route("/v1/replies/{id}/feedback", post(add_feedback))
        .route("/v1/tazkiya", get(tazkiya_summary))
        .route(
            "/v1/preferences",
            get(list_preferences).post(add_preference),
        )
        .route("/v1/drift", get(drift))
        .route("/v1/gates", get(gates_dashboard))
        .route("/v1/alerts", get(alerts))
        .route("/v1/perf/stages", get(perf_stages))
        .route("/v1/alerts/drill", post(alert_drill))
        .route("/v1/lab/insights", get(lab_insights))
        .route("/v1/bench/protocol", post(start_protocol))
        .route("/v1/eval/runs/{id}/export", get(export_run))
        .route("/v1/tazkiya/review", get(review_queue))
        .route("/v1/tazkiya/review/{id}", post(set_review))
        .route("/v1/system", get(system))
        .route("/v1/system/backend", post(set_backend))
        .route("/v1/system/activity", get(activity))
        .layer(middleware::from_fn_with_state(state.clone(), auth));
    Router::new()
        .route("/", get(|s: State<AppState>| serve_asset(s, "index.html")))
        .route(
            "/app.js",
            get(|s: State<AppState>| serve_asset(s, "app.js")),
        )
        .route(
            "/app.css",
            get(|s: State<AppState>| serve_asset(s, "app.css")),
        )
        .route(
            "/favicon.svg",
            get(|s: State<AppState>| serve_asset(s, "favicon.svg")),
        )
        .route("/healthz", get(healthz))
        .route(
            "/metrics",
            get(|State(s): State<AppState>| async move { s.prom.render() }),
        )
        .route(
            "/openapi.yaml",
            get(|| async { ([(header::CONTENT_TYPE, "application/yaml")], OPENAPI_YAML) }),
        )
        .route(
            "/openapi.json",
            get(|| async { ([(header::CONTENT_TYPE, "application/json")], OPENAPI_JSON) }),
        )
        .merge(v1)
        // No CORS layer: the UI is same-origin, and cross-origin callers are refused below (review H4).
        .layer(middleware::from_fn_with_state(state.clone(), origin_guard))
        .layer(tower_http::limit::RequestBodyLimitLayer::new(
            2 * 1024 * 1024,
        ))
        // Outermost: extractor rejections (malformed JSON, wrong field types, oversized bodies) arrive as plain
        // text from the framework; clients of this API get the same {code, message} envelope as every other error.
        .layer(middleware::from_fn(json_error_envelope))
        .with_state(state)
}

/// Rewrite a framework-generated plain-text 4xx rejection into the JSON error envelope.
async fn json_error_envelope(req: Request, next: Next) -> Response {
    let resp = next.run(req).await;
    let status = resp.status();
    let plain = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.starts_with("text/plain"));
    if !plain
        || !matches!(
            status,
            StatusCode::BAD_REQUEST
                | StatusCode::PAYLOAD_TOO_LARGE
                | StatusCode::UNSUPPORTED_MEDIA_TYPE
                | StatusCode::UNPROCESSABLE_ENTITY
        )
    {
        return resp;
    }
    let (parts, body) = resp.into_parts();
    let bytes = axum::body::to_bytes(body, 4096).await.unwrap_or_default();
    let text = String::from_utf8_lossy(&bytes);
    let msg = if text.trim().is_empty() {
        status
            .canonical_reason()
            .unwrap_or("bad request")
            .to_string()
    } else {
        text.trim().to_string()
    };
    let mut r = (
        status,
        Json(json!({"code": "E_BAD_REQUEST", "message": format!("bad request: {msg}")})),
    )
        .into_response();
    for (k, v) in parts.headers.iter() {
        if k != header::CONTENT_TYPE && k != header::CONTENT_LENGTH {
            r.headers_mut().insert(k.clone(), v.clone());
        }
    }
    r
}

async fn serve_asset(State(s): State<AppState>, name: &'static str) -> Response {
    match nafs_web::asset(name, s.web_dir.as_deref()) {
        Some((body, mime)) => (
            [
                (header::CONTENT_TYPE, mime),
                (header::CACHE_CONTROL, "no-cache"),
                (header::X_CONTENT_TYPE_OPTIONS, "nosniff"),
                (header::X_FRAME_OPTIONS, "DENY"),
                (header::REFERRER_POLICY, "no-referrer"),
                (
                    header::CONTENT_SECURITY_POLICY,
                    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self' ws: wss:; frame-ancestors 'none'",
                ),
            ],
            body,
        )
            .into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

fn host_only(h: &str) -> String {
    let h = h.trim().to_lowercase();
    if h.starts_with('[') {
        return h.split(']').next().map(|x| format!("{x}]")).unwrap_or(h);
    }
    h.rsplit_once(':').map(|(a, _)| a.to_string()).unwrap_or(h)
}

/// Same-origin and DNS-rebinding guard (review H4). A browser on another origin cannot call the API
/// (no CORS headers, and requests carrying a foreign Origin are refused, including WebSocket upgrades);
/// without auth tokens the Host header must also name this machine.
async fn origin_guard(State(s): State<AppState>, req: Request, next: Next) -> Response {
    let h = req.headers();
    let host = h
        .get(header::HOST)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    if s.tokens.is_empty() && !s.allowed_hosts.contains(&host_only(&host)) {
        return ApiError(NafsError::Forbidden(format!(
            "host '{}' not allowed (set NAFS_ALLOWED_HOSTS or enable auth)",
            host_only(&host)
        )))
        .into_response();
    }
    if let Some(origin) = h.get(header::ORIGIN).and_then(|v| v.to_str().ok()) {
        let o_host = origin
            .split_once("://")
            .map(|(_, r)| r.trim_end_matches('/').to_lowercase())
            .unwrap_or_default();
        if o_host != host.to_lowercase() {
            return ApiError(NafsError::Forbidden(
                "cross-origin requests are not accepted".into(),
            ))
            .into_response();
        }
    }
    next.run(req).await
}

fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'%' if i + 2 < b.len() => {
                match u8::from_str_radix(std::str::from_utf8(&b[i + 1..i + 3]).unwrap_or("zz"), 16)
                {
                    Ok(v) => {
                        out.push(v);
                        i += 3;
                        continue;
                    }
                    Err(_) => out.push(b'%'),
                }
            }
            b'+' => out.push(b' '),
            c => out.push(c),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Bearer-token auth and per-token rate limiting (NFR-022, NFR-011). WebSockets may pass ?token=
/// (percent-decoded, review M13).
async fn auth(State(s): State<AppState>, headers: HeaderMap, req: Request, next: Next) -> Response {
    if s.tokens.is_empty() {
        return next.run(req).await;
    }
    let from_header = headers
        .get(header::AUTHORIZATION)
        .and_then(|h| h.to_str().ok())
        .and_then(|h| h.strip_prefix("Bearer "))
        .map(String::from);
    let from_query = req.uri().query().and_then(|q| {
        q.split('&')
            .find_map(|kv| kv.strip_prefix("token=").map(percent_decode))
    });
    let Some(tok) = from_header.or(from_query).filter(|t| s.tokens.contains(t)) else {
        return ApiError(NafsError::Unauthorized).into_response();
    };
    if !rate_ok(&s, &tok) {
        return ApiError(NafsError::RateLimited).into_response();
    }
    next.run(req).await
}
fn rate_ok(s: &AppState, key: &str) -> bool {
    let mut rl = s.rate.lock().unwrap();
    let e = rl.entry(key.to_string()).or_insert((Instant::now(), 0));
    if e.0.elapsed() > Duration::from_secs(60) {
        *e = (Instant::now(), 0);
    }
    e.1 += 1;
    e.1 <= s.cfg.server.rate_limit_per_token_rpm
}

async fn healthz(State(s): State<AppState>) -> Json<Value> {
    let p = s.orch.active();
    let (a, b) = tokio::join!(p.s1.health(), p.s2.health());
    Json(
        json!({"status": if a.ok && b.ok { "ok" } else { "degraded" }, "backend": p.name,
        "backends": {"system1": if a.ok { "ready" } else { "unavailable" }, "system2": if b.ok { "ready" } else { "unavailable" }}}),
    )
}

pub(crate) async fn run_with_cap(
    s: &AppState,
    req: ChatRequest,
    em: Emitter,
    cancel: Option<Arc<AtomicBool>>,
) -> Result<nafs_types::ChatReply, NafsError> {
    let r = tokio::time::timeout(
        Duration::from_secs(s.cfg.server.request_wall_clock_cap_s + 30),
        s.orch.handle_with_cancel(req, em, cancel),
    )
    .await
    .map_err(|_| NafsError::BackendTimeout("request wall-clock cap exceeded".into()))
    .and_then(|r| r);
    // Outcomes feed the error-rate alert; client cancellations and bad requests are not failures.
    let ok = match &r {
        Ok(_) => Some(true),
        Err(NafsError::Cancelled) | Err(NafsError::BadRequest(_)) => None,
        Err(e) => {
            metrics::counter!("nafs_errors_total", "code" => e.code()).increment(1);
            tracing::warn!(
                "request failed: {}",
                crate::security::redact_secrets(&e.to_string(), &s.tokens)
            );
            Some(false)
        }
    };
    if let Some(ok) = ok {
        let mut o = s.outcomes.lock().unwrap();
        o.push_back((Instant::now(), ok));
        while o.len() > 5000 {
            o.pop_front();
        }
    }
    r
}

async fn chat(
    State(s): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> ApiResult<Json<nafs_types::ChatReply>> {
    Ok(Json(run_with_cap(&s, req, Emitter(None), None).await?))
}

async fn chat_stream(
    State(s): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> ApiResult<Sse<impl futures::Stream<Item = Result<Event, std::convert::Infallible>>>> {
    // Validation failures are HTTP 400 before the stream starts (review M14).
    s.orch.validate(&req)?;
    let (tx, rx) = mpsc::unbounded_channel::<StreamEvent>();
    let s2 = s.clone();
    let tx2 = tx.clone();
    let inner = tokio::spawn(async move { run_with_cap(&s2, req, Emitter(Some(tx2)), None).await });
    tokio::spawn(async move {
        // A panic or abort must still end the stream with an error event (review M10).
        let err = match inner.await {
            Ok(Ok(_)) => None,
            Ok(Err(NafsError::Cancelled)) => None,
            Ok(Err(e)) => Some((e.code().to_string(), e.to_string())),
            Err(j) => {
                tracing::error!("request task failed: {j}");
                Some((
                    "E_INTERNAL".into(),
                    "internal error: the request task stopped unexpectedly".into(),
                ))
            }
        };
        if let Some((code, message)) = err {
            let _ = tx.send(StreamEvent::Error { code, message });
        }
    });
    let stream = UnboundedReceiverStream::new(rx).map(|ev| {
        Ok(Event::default()
            .event(ev.name())
            .data(serde_json::to_string(&ev).unwrap_or_default()))
    });
    Ok(Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(10))))
}

async fn list_conversations(State(s): State<AppState>) -> ApiResult<Json<Value>> {
    Ok(Json(json!(s.store.list_conversations(200)?)))
}
async fn get_conversation(
    State(s): State<AppState>,
    Path(id): Path<Uuid>,
) -> ApiResult<Json<Value>> {
    Ok(Json(json!(s.store.get_conversation(id)?)))
}
#[derive(Deserialize)]
struct Rename {
    #[serde(default)]
    title: Option<String>,
    /// SR-004: exclude this conversation from evaluation exports and training.
    #[serde(default)]
    training_opt_out: Option<bool>,
}
async fn rename_conversation(
    State(s): State<AppState>,
    Path(id): Path<Uuid>,
    Json(b): Json<Rename>,
) -> ApiResult<StatusCode> {
    if b.title.is_none() && b.training_opt_out.is_none() {
        return Err(NafsError::BadRequest("set title and/or training_opt_out".into()).into());
    }
    if let Some(title) = b.title {
        let t: String = title.trim().chars().take(120).collect();
        if t.is_empty() {
            return Err(NafsError::BadRequest("title required".into()).into());
        }
        s.store.rename_conversation(id, &t)?;
    }
    if let Some(o) = b.training_opt_out {
        s.store.set_training_opt_out(id, o)?;
    }
    Ok(StatusCode::NO_CONTENT)
}

// ---------------- Human feedback and Tazkiya (FR-033, SR-004, Phase 3) ----------------
#[derive(Deserialize)]
struct FeedbackReq {
    rating: i64,
    #[serde(default)]
    note: Option<String>,
}
async fn add_feedback(
    State(s): State<AppState>,
    Path(id): Path<Uuid>,
    Json(b): Json<FeedbackReq>,
) -> ApiResult<StatusCode> {
    if b.rating != 1 && b.rating != -1 {
        return Err(NafsError::BadRequest("rating must be 1 or -1".into()).into());
    }
    let note: Option<String> = b.note.map(|n| n.chars().take(1000).collect());
    s.store.add_feedback(id, b.rating, note.as_deref())?;
    metrics::counter!("nafs_feedback_total", "rating" => if b.rating > 0 { "up" } else { "down" })
        .increment(1);
    Ok(StatusCode::NO_CONTENT)
}
async fn tazkiya_summary(State(s): State<AppState>) -> ApiResult<Json<Value>> {
    let registry: Value = std::fs::read_to_string(&s.cfg.storage.registry_file)
        .ok()
        .and_then(|x| serde_json::from_str(&x).ok())
        .unwrap_or(Value::Null);
    let p = s.orch.active();
    Ok(Json(
        json!({"ledger": s.store.tazkiya_summary()?, "registry": registry, "retention_days": s.cfg.storage.retention_days,
        "active_models": {"system1": p.s1.cfg.hf_id, "system2": p.s2.cfg.hf_id, "adapter_hash": "none"},
        "rules": {"FR-070": "verified System 2 traces, deduplicated, no domain above 30%, 5% abstention examples", "FR-033": "consensus answers only after human approval",
                  "TR-008": "20% replay of general instruction data", "SR-004": "personal data redacted; opted-out conversations excluded", "TR-005": "training runs on a GPU host, never on this machine"}}),
    ))
}
async fn review_queue(State(s): State<AppState>, Query(q): Query<Lim>) -> ApiResult<Json<Value>> {
    Ok(Json(json!(s
        .store
        .review_queue(q.limit.unwrap_or(50).min(500) as i64)?)))
}
#[derive(Deserialize)]
struct ReviewReq {
    approved: bool,
    #[serde(default)]
    reviewer: Option<String>,
    #[serde(default)]
    note: Option<String>,
}
async fn set_review(
    State(s): State<AppState>,
    Path(id): Path<Uuid>,
    Json(b): Json<ReviewReq>,
) -> ApiResult<StatusCode> {
    let reviewer: String = b
        .reviewer
        .unwrap_or_else(|| "operator".into())
        .chars()
        .filter(|c| c.is_alphanumeric() || " ._-@".contains(*c))
        .take(60)
        .collect();
    s.store
        .set_review(id, b.approved, &reviewer, b.note.as_deref())?;
    Ok(StatusCode::NO_CONTENT)
}
async fn delete_conversation(
    State(s): State<AppState>,
    Path(id): Path<Uuid>,
) -> ApiResult<StatusCode> {
    s.store.delete_conversation(id)?;
    Ok(StatusCode::NO_CONTENT)
}
async fn get_decision(State(s): State<AppState>, Path(id): Path<Uuid>) -> ApiResult<Json<Value>> {
    Ok(Json(json!(s.store.get_decision(id)?)))
}

// ---------------- Phase 4: drift, gates dashboard, load-test protocol, exports; lab preferences ----------------
#[derive(Deserialize)]
struct PrefReq {
    prompt: String,
    #[serde(default)]
    domain: Option<String>,
    fast_reply_id: Uuid,
    slow_reply_id: Uuid,
    preferred: String,
    #[serde(default)]
    note: Option<String>,
}
async fn add_preference(
    State(s): State<AppState>,
    Json(b): Json<PrefReq>,
) -> ApiResult<(StatusCode, Json<Value>)> {
    let dom: String = b
        .domain
        .unwrap_or_else(|| "other".into())
        .chars()
        .filter(|c| c.is_alphanumeric())
        .take(20)
        .collect();
    let prompt: String = b.prompt.chars().take(4000).collect();
    let id = s.store.add_preference(
        &dom,
        &prompt,
        b.fast_reply_id,
        b.slow_reply_id,
        &b.preferred,
        b.note.as_deref(),
    )?;
    metrics::counter!("nafs_preferences_total", "preferred" => b.preferred.clone()).increment(1);
    Ok((StatusCode::CREATED, Json(json!({"preference_id": id}))))
}
async fn list_preferences(State(s): State<AppState>) -> ApiResult<Json<Value>> {
    let rows = s.store.preference_stats()?;
    let rows: Vec<Value> = rows
        .into_iter()
        .map(|mut r| {
            let decided = r["slow"].as_u64().unwrap_or(0) + r["fast"].as_u64().unwrap_or(0);
            if decided > 0 {
                let k = r["slow"].as_u64().unwrap_or(0);
                let (lo, hi) = nafs_verify::mutation::clopper_pearson(k, decided);
                r["slow_preferred_rate"] = json!(k as f64 / decided as f64);
                r["ci95"] = json!([lo, hi]);
            }
            r
        })
        .collect();
    Ok(Json(
        json!({"domains": rows, "note": "Share of decided trials (ties and both-bad excluded) in which a person preferred the System 2 answer, with a Clopper-Pearson 95% interval: the value of deliberation where no exact verifier exists."}),
    ))
}
#[derive(Deserialize)]
struct DriftQ {
    #[serde(default)]
    k: Option<f64>,
    #[serde(default)]
    h: Option<f64>,
    #[serde(default)]
    reference: Option<usize>,
}
async fn drift(State(s): State<AppState>, Query(q): Query<DriftQ>) -> ApiResult<Json<Value>> {
    use nafs_core::drift::{cusum, negative_control, CusumParams};
    let d = CusumParams::default();
    let p = CusumParams {
        k: q.k.unwrap_or(d.k).clamp(0.0, 3.0),
        h: q.h.unwrap_or(d.h).clamp(1.0, 50.0),
        reference: q.reference.unwrap_or(d.reference).clamp(10, 1000),
    };
    let series = s.store.labelled_series()?;
    let mut by: BTreeMap<String, Vec<(String, f64, bool)>> = BTreeMap::new();
    for (t, dom, e1, wrong) in series {
        by.entry("all".into())
            .or_default()
            .push((t.clone(), e1, wrong));
        by.entry(dom).or_default().push((t, e1, wrong));
    }
    let thin = |v: &[f64]| -> Vec<f64> {
        let step = (v.len() / 120).max(1);
        v.iter().step_by(step).copied().collect()
    };
    let out: Vec<Value> = by
        .into_iter()
        .map(|(dom, obs)| {
            let calib: Vec<f64> = obs.iter().map(|(_, e1, w)| *w as i32 as f64 - e1).collect();
            let acc: Vec<f64> = obs.iter().map(|(_, _, w)| (!*w) as i32 as f64).collect();
            let chart = |name: &str, xs: &[f64]| match cusum(xs, p) {
                Some(r) => {
                    let nc = negative_control(xs, p, 17).map(|n| n.alarm_at.is_some());
                    json!({"series": name, "n": r.n, "alarm": r.alarm_at.is_some(), "alarm_at": r.alarm_at.map(|i| obs[i].0.clone()), "direction": r.direction,
                        "negative_control_alarm": nc, "reference_mean": r.reference_mean, "reference_sd": r.reference_sd, "s_hi": thin(&r.s_hi), "s_lo": thin(&r.s_lo)})
                }
                None => json!({"series": name, "n": xs.len(), "alarm": null, "note": format!("needs >= 20 labelled observations (has {})", xs.len())}),
            };
            json!({"domain": dom, "calibration_error": chart("calibration_error", &calib), "accuracy": chart("fast_path_accuracy", &acc)})
        })
        .collect();
    Ok(Json(json!({"params": p, "domains": out,
        "note": "Two-sided CUSUM on standardised series against a reference window, with a shuffled-order negative control (FR-042). Calibration error is (draft wrong) - e1_hat; accuracy is the fast-path pass rate."})))
}
async fn gates_dashboard(State(s): State<AppState>) -> ApiResult<Json<Value>> {
    let eval_runs: Vec<Value> = s
        .store
        .list_runs("eval", 500)?
        .into_iter()
        .filter_map(|r| r["run_id"].as_str().and_then(|id| Uuid::parse_str(id).ok()))
        .filter_map(|id| s.store.get_run(id).ok())
        .collect();
    let orch = s.orch.clone();
    let eval_runs: Vec<Value> = tokio::task::spawn_blocking(move || {
        eval_runs
            .iter()
            .map(|r| {
                if r["status"] != "running" {
                    nafs_eval::rebuild(r, &orch).unwrap_or_else(|| r.clone())
                } else {
                    r.clone()
                }
            })
            .collect()
    })
    .await
    .map_err(|e| NafsError::Internal(e.to_string()))?;
    let read = |p: &str| {
        std::fs::read_to_string(p)
            .ok()
            .and_then(|x| serde_json::from_str::<Value>(&x).ok())
    };
    let cal = s.orch.calibration();
    let ev = crate::gates::Evidence {
        eval_runs,
        bench_runs: s.store.list_runs("bench", 200)?,
        acceptance: read("benchmarks/mvp_acceptance.json"),
        h7: read("benchmarks/reference-machine/h7_result.json"),
        registry: read(&s.cfg.storage.registry_file),
        calibration_promoted: s.orch.promoted(),
        calibration_n: cal.n,
        calibration_ece: cal.ece,
        security: read("docs/security/security_review.json"),
        deploy: read("deploy/verified.json"),
        profile_c: read("benchmarks/profile_c_decision.json"),
        scholar: read("docs/release/scholar_review.json"),
    };
    Ok(Json(crate::gates::evaluate(&ev)))
}
async fn start_protocol(
    State(s): State<AppState>,
    Json(req): Json<nafs_bench::ProtocolRequest>,
) -> ApiResult<(StatusCode, Json<Value>)> {
    if req.levels.is_empty()
        || req.levels.len() > 6
        || req.levels.iter().any(|l| *l == 0 || *l > 16)
    {
        return Err(NafsError::BadRequest(
            "levels must list 1..6 concurrency values in 1..16".into(),
        )
        .into());
    }
    let id = Uuid::new_v4();
    let base = json!({"run_id": id, "kind": "protocol", "status": "running", "request": req, "created_at": chrono::Utc::now()});
    s.store.save_run("bench", id, "running", &base)?;
    let cancel = register_cancel(&s, id);
    let s2 = s.clone();
    tokio::spawn(async move {
        let s3 = s2.clone();
        let last = Arc::new(Mutex::new(Instant::now() - Duration::from_secs(5)));
        let mut v = nafs_bench::protocol(s2.orch.clone(), req, cancel, move |snap| {
            let mut l = last.lock().unwrap();
            if l.elapsed() < Duration::from_millis(1500) {
                return;
            }
            *l = Instant::now();
            let mut v = snap.clone();
            v["run_id"] = json!(id);
            let _ = s3.store.save_run("bench", id, "running", &v);
        })
        .await;
        v["run_id"] = json!(id);
        v["created_at"] = json!(chrono::Utc::now());
        let st = v["status"].as_str().unwrap_or("done").to_string();
        let _ = s2.store.save_run("bench", id, &st, &v);
        s2.cancels.lock().unwrap().remove(&id);
    });
    Ok((StatusCode::ACCEPTED, Json(base)))
}
#[derive(Deserialize)]
struct ExportQ {
    #[serde(default)]
    format: Option<String>,
}
async fn export_run(
    State(s): State<AppState>,
    Path(id): Path<Uuid>,
    Query(q): Query<ExportQ>,
) -> ApiResult<Response> {
    let run = s.store.get_run(id)?;
    let orch = s.orch.clone();
    let r = tokio::task::spawn_blocking(move || nafs_eval::rebuild(&run, &orch).unwrap_or(run))
        .await
        .map_err(|e| NafsError::Internal(e.to_string()))?;
    let short = id.to_string()[..8].to_string();
    Ok(match q.format.as_deref().unwrap_or("md") {
        "csv" => (
            [
                (header::CONTENT_TYPE, "text/csv; charset=utf-8".to_string()),
                (
                    header::CONTENT_DISPOSITION,
                    format!("attachment; filename=\"nafs-eval-{short}.csv\""),
                ),
            ],
            nafs_eval::items_csv(&r),
        )
            .into_response(),
        "json" => (
            [
                (header::CONTENT_TYPE, "application/json".to_string()),
                (
                    header::CONTENT_DISPOSITION,
                    format!("attachment; filename=\"nafs-eval-{short}.json\""),
                ),
            ],
            r.to_string(),
        )
            .into_response(),
        _ => (
            [
                (
                    header::CONTENT_TYPE,
                    "text/markdown; charset=utf-8".to_string(),
                ),
                (
                    header::CONTENT_DISPOSITION,
                    format!("attachment; filename=\"nafs-eval-{short}.md\""),
                ),
            ],
            nafs_eval::markdown_report(&r),
        )
            .into_response(),
    })
}

// ---------------- Monitoring and alerts (OR-004, P5-T02) ----------------
fn evaluate_alerts(s: &AppState) -> NafsResult<Value> {
    let a = &s.cfg.alerts;
    let rows = s.store.recent_window(a.window_min)?;
    let window = Duration::from_secs((a.window_min.max(1) as u64) * 60);
    let (n_out, n_err) = {
        let o = s.outcomes.lock().unwrap();
        let recent: Vec<&(Instant, bool)> =
            o.iter().filter(|(t, _)| t.elapsed() <= window).collect();
        (recent.len(), recent.iter().filter(|(_, ok)| !*ok).count())
    };
    let mut fast: Vec<f64> = rows
        .iter()
        .filter(|r| r.0 == "fast")
        .map(|r| r.1 as f64)
        .collect();
    fast.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let p95 = (!fast.is_empty()).then(|| fast[((fast.len() - 1) as f64 * 0.95).round() as usize]);
    let esc =
        (!rows.is_empty()).then(|| rows.iter().filter(|r| r.2).count() as f64 / rows.len() as f64);
    let exact: Vec<bool> = rows.iter().filter(|r| r.4).filter_map(|r| r.3).collect();
    let acc = (!exact.is_empty())
        .then(|| exact.iter().filter(|x| **x).count() as f64 / exact.len() as f64);
    // Drift: any CUSUM alarm on the pooled calibration-error series (FR-042).
    let series = s.store.labelled_series()?;
    let calib: Vec<f64> = series
        .iter()
        .map(|(_, _, e1, w)| *w as i32 as f64 - e1)
        .collect();
    let drift = nafs_core::drift::cusum(&calib, Default::default());
    let drill = s
        .drill_until
        .lock()
        .unwrap()
        .map(|t| t > Instant::now())
        .unwrap_or(false);
    let rule = |id: &str,
                desc: &str,
                value: Option<f64>,
                n: usize,
                breach: bool,
                threshold: String| {
        let enough = n >= a.min_samples;
        let firing = drill || (enough && breach);
        metrics::gauge!("nafs_alert_firing", "rule" => id.to_string()).set(firing as i32 as f64);
        json!({"rule": id, "description": desc, "value": value, "threshold": threshold, "n": n, "state": if firing { "firing" } else if enough { "ok" } else { "insufficient data" }, "drill": drill})
    };
    let rules = vec![
        rule(
            "fast_path_latency",
            "p95 fast-path latency (NFR-003)",
            p95,
            fast.len(),
            p95.map(|v| v > a.fast_p95_ms).unwrap_or(false),
            format!("> {} s", a.fast_p95_ms / 1000.0),
        ),
        rule(
            "error_rate",
            "share of failed requests",
            (n_out > 0).then(|| n_err as f64 / n_out as f64),
            n_out,
            n_out > 0 && n_err as f64 / n_out as f64 > a.error_rate,
            format!("> {:.0}%", a.error_rate * 100.0),
        ),
        rule(
            "escalation_rate",
            "share of requests escalated to System 2 (cost)",
            esc,
            rows.len(),
            esc.map(|v| v > a.escalation_rate_max).unwrap_or(false),
            format!("> {:.0}%", a.escalation_rate_max * 100.0),
        ),
        rule(
            "verifier_acceptance",
            "share of exact-verifier verdicts that pass",
            acc,
            exact.len(),
            acc.map(|v| v < a.verifier_acceptance_min).unwrap_or(false),
            format!("< {:.0}%", a.verifier_acceptance_min * 100.0),
        ),
        rule(
            "calibration_drift",
            "CUSUM alarm on calibration error (FR-042)",
            drift.as_ref().map(|d| d.alarm_at.is_some() as i32 as f64),
            calib.len(),
            drift
                .as_ref()
                .map(|d| d.alarm_at.is_some())
                .unwrap_or(false),
            "alarm".into(),
        ),
    ];
    let firing = rules.iter().filter(|r| r["state"] == "firing").count();
    Ok(
        json!({"window_min": a.window_min, "firing": firing, "drill": drill, "rules": rules,
        "note": "Evaluated over the recent window; rules with fewer than min_samples observations report insufficient data. deploy/monitoring/alerts.yml holds the same rules for Prometheus."}),
    )
}
type NafsResult<T> = Result<T, NafsError>;
#[derive(Deserialize)]
struct StagesQ {
    hours: Option<i64>,
    lab: Option<bool>,
}
/// Where the time goes (P7-T01): latency by stage, and the engine's prefill/decode speed and cache reuse.
async fn perf_stages(
    State(s): State<AppState>,
    Query(q): Query<StagesQ>,
) -> ApiResult<Json<Value>> {
    let hours = q.hours.unwrap_or(24).clamp(1, 24 * 90);
    let lab = q.lab.unwrap_or(false);
    let rows = s.store.stage_rows(hours, lab)?;
    let mut v = nafs_bench::latency_breakdown(&rows);
    v["hours"] = json!(hours);
    v["lab_included"] = json!(lab);
    Ok(Json(v))
}

async fn alerts(State(s): State<AppState>) -> ApiResult<Json<Value>> {
    let s2 = s.clone();
    Ok(Json(
        tokio::task::spawn_blocking(move || evaluate_alerts(&s2))
            .await
            .map_err(|e| NafsError::Internal(e.to_string()))??,
    ))
}
#[derive(Deserialize, Default)]
struct DrillReq {
    #[serde(default)]
    seconds: Option<u64>,
}
async fn alert_drill(
    State(s): State<AppState>,
    body: Option<Json<DrillReq>>,
) -> ApiResult<Json<Value>> {
    let secs = body
        .and_then(|b| b.0.seconds)
        .unwrap_or(120)
        .clamp(10, 3600);
    *s.drill_until.lock().unwrap() = Some(Instant::now() + Duration::from_secs(secs));
    tracing::warn!("alert drill started for {secs} s: every alert rule reports firing");
    let s2 = s.clone();
    let v = tokio::task::spawn_blocking(move || evaluate_alerts(&s2))
        .await
        .map_err(|e| NafsError::Internal(e.to_string()))??;
    Ok(Json(json!({"drill_seconds": secs, "alerts": v})))
}

// ---------------- Lab insights: which System 1 / System 2 policy is optimal where ----------------
#[derive(Deserialize)]
struct InsightQ {
    #[serde(default)]
    lambda: Option<f32>,
}
async fn lab_insights(
    State(s): State<AppState>,
    Query(q): Query<InsightQ>,
) -> ApiResult<Json<Value>> {
    let runs: Vec<Value> = s
        .store
        .list_runs("eval", 500)?
        .into_iter()
        .filter_map(|r| r["run_id"].as_str().and_then(|id| Uuid::parse_str(id).ok()))
        .filter_map(|id| s.store.get_run(id).ok())
        .collect();
    let prefs = s.store.preference_stats()?;
    let orch = s.orch.clone();
    let lam = q.lambda;
    let v = tokio::task::spawn_blocking(move || nafs_eval::insights(&runs, &orch, lam, &prefs))
        .await
        .map_err(|e| NafsError::Internal(e.to_string()))?;
    Ok(Json(v))
}

// ---------------- Bench, eval and verifier runs (IF-009) ----------------
fn register_cancel(s: &AppState, id: Uuid) -> Arc<AtomicBool> {
    let c = Arc::new(AtomicBool::new(false));
    s.cancels.lock().unwrap().insert(id, c.clone());
    c
}

async fn start_bench(
    State(s): State<AppState>,
    Json(req): Json<nafs_bench::BenchRunRequest>,
) -> ApiResult<(StatusCode, Json<Value>)> {
    if req.duration_s.is_none() && req.requests.is_none() {
        return Err(NafsError::BadRequest("set duration_s or requests".into()).into());
    }
    let id = Uuid::new_v4();
    let base = json!({"run_id": id, "kind": "bench", "status": "running", "request": req, "created_at": chrono::Utc::now()});
    s.store.save_run("bench", id, "running", &base)?;
    let cancel = register_cancel(&s, id);
    let (s2, last) = (
        s.clone(),
        Arc::new(Mutex::new(Instant::now() - Duration::from_secs(5))),
    );
    tokio::spawn(async move {
        let s3 = s2.clone();
        let last2 = last.clone();
        let c2 = cancel.clone();
        let fin = nafs_bench::run(s2.orch.clone(), req, cancel, move |v| {
            let mut l = last2.lock().unwrap();
            if l.elapsed() < Duration::from_millis(800) {
                return;
            }
            *l = Instant::now();
            let mut v = v.clone();
            v["run_id"] = json!(id);
            v["kind"] = json!("bench");
            let _ = s3.store.save_run("bench", id, "running", &v);
        })
        .await;
        let mut v = fin;
        let status = if c2.load(Ordering::Relaxed) {
            "cancelled"
        } else {
            "done"
        };
        v["run_id"] = json!(id);
        v["kind"] = json!("bench");
        v["status"] = json!(status);
        v["finished_at"] = json!(chrono::Utc::now());
        let _ = s2.store.save_run("bench", id, status, &v);
        s2.cancels.lock().unwrap().remove(&id);
    });
    Ok((StatusCode::ACCEPTED, Json(base)))
}
fn list_kind(s: &AppState, kind: &str) -> ApiResult<Json<Value>> {
    Ok(Json(json!(s
        .store
        .list_runs(kind, 50)?
        .into_iter()
        .map(strip_heavy)
        .collect::<Vec<_>>())))
}
async fn list_bench(State(s): State<AppState>) -> ApiResult<Json<Value>> {
    list_kind(&s, "bench")
}
async fn list_eval(State(s): State<AppState>) -> ApiResult<Json<Value>> {
    list_kind(&s, "eval")
}
async fn list_rate_runs(State(s): State<AppState>) -> ApiResult<Json<Value>> {
    list_kind(&s, "rates")
}
fn strip_heavy(mut v: Value) -> Value {
    if let Some(o) = v.as_object_mut() {
        o.remove("samples");
        o.remove("items");
        if let Some(vs) = o.get_mut("verifiers").and_then(|x| x.as_object_mut()) {
            for (_, x) in vs.iter_mut() {
                if let Some(xo) = x.as_object_mut() {
                    xo.remove("rows");
                    xo.remove("false_accepts");
                    xo.remove("false_rejects");
                }
            }
        }
    }
    v
}
async fn get_run(State(s): State<AppState>, Path(id): Path<Uuid>) -> ApiResult<Json<Value>> {
    let run = s.store.get_run(id)?;
    if run["kind"] == "eval" && run["status"] != "running" && run.get("items").is_some() {
        let orch = s.orch.clone();
        let r2 = run.clone();
        if let Ok(Some(v)) =
            tokio::task::spawn_blocking(move || nafs_eval::rebuild(&r2, &orch)).await
        {
            return Ok(Json(v));
        }
    }
    Ok(Json(run))
}
async fn cancel_run(State(s): State<AppState>, Path(id): Path<Uuid>) -> ApiResult<StatusCode> {
    match s.cancels.lock().unwrap().get(&id) {
        Some(c) => {
            c.store(true, Ordering::Relaxed);
            Ok(StatusCode::ACCEPTED)
        }
        None => Err(NafsError::NotFound(format!("active run {id}")).into()),
    }
}

async fn list_datasets(State(s): State<AppState>) -> Json<Value> {
    Json(json!(nafs_eval::list_datasets(&s.cfg.eval.datasets_dir)))
}
#[derive(Deserialize)]
struct Lim {
    #[serde(default)]
    limit: Option<usize>,
}
async fn preview_dataset(
    State(s): State<AppState>,
    Path(name): Path<String>,
    Query(q): Query<Lim>,
) -> ApiResult<Json<Value>> {
    let t = nafs_eval::load_tasks(&s.cfg.eval.datasets_dir, &name)
        .map_err(|e| NafsError::NotFound(e.to_string()))?;
    Ok(Json(json!(t
        .into_iter()
        .take(q.limit.unwrap_or(25).min(500))
        .collect::<Vec<_>>())))
}
async fn start_eval(
    State(s): State<AppState>,
    Json(req): Json<nafs_eval::EvalRunRequest>,
) -> ApiResult<(StatusCode, Json<Value>)> {
    req.validate().map_err(NafsError::BadRequest)?;
    nafs_eval::load_tasks(&s.cfg.eval.datasets_dir, &req.benchmark).map_err(|e| {
        let missing = e
            .downcast_ref::<std::io::Error>()
            .is_some_and(|io| io.kind() == std::io::ErrorKind::NotFound);
        NafsError::BadRequest(if missing {
            format!(
                "unknown dataset '{}'; see GET /v1/eval/datasets",
                req.benchmark.chars().take(40).collect::<String>()
            )
        } else {
            format!("dataset: {e}")
        })
    })?;
    let id = Uuid::new_v4();
    let base = json!({"run_id": id, "kind": "eval", "status": "running", "request": req, "created_at": chrono::Utc::now(), "progress": {"done": 0, "total": 0}});
    s.store.save_run("eval", id, "running", &base)?;
    spawn_eval(&s, id, req, vec![]);
    Ok((StatusCode::ACCEPTED, Json(base)))
}

/// Run an evaluation in the background; `prior` items (from an interrupted run) are kept and skipped.
fn spawn_eval(
    s: &AppState,
    id: Uuid,
    req: nafs_eval::EvalRunRequest,
    prior: Vec<nafs_eval::ItemResult>,
) {
    let cancel = register_cancel(s, id);
    let s2 = s.clone();
    tokio::spawn(async move {
        let s3 = s2.clone();
        let res = nafs_eval::run_with_prior(
            s2.orch.clone(),
            req.clone(),
            cancel,
            move |v| {
                let mut v = v.clone();
                v["run_id"] = json!(id);
                v["kind"] = json!("eval");
                let _ = s3.store.save_run("eval", id, "running", &v);
            },
            prior,
        )
        .await;
        let v = match res {
            Ok(mut v) => {
                v["run_id"] = json!(id);
                v["kind"] = json!("eval");
                v["finished_at"] = json!(chrono::Utc::now());
                v
            }
            Err(e) => {
                json!({"run_id": id, "kind": "eval", "status": "failed", "error": e.to_string(), "request": req})
            }
        };
        let _ = s2
            .store
            .save_run("eval", id, v["status"].as_str().unwrap_or("done"), &v);
        s2.cancels.lock().unwrap().remove(&id);
    });
}

/// Resume an interrupted or cancelled evaluation run in place (robustness against restarts).
async fn resume_eval(
    State(s): State<AppState>,
    Path(id): Path<Uuid>,
) -> ApiResult<(StatusCode, Json<Value>)> {
    let run = s.store.get_run(id)?;
    if run["kind"] != "eval"
        || !matches!(
            run["status"].as_str(),
            Some("interrupted") | Some("cancelled")
        )
    {
        return Err(NafsError::BadRequest(
            "only interrupted or cancelled evaluation runs can be resumed".into(),
        )
        .into());
    }
    let req: nafs_eval::EvalRunRequest = serde_json::from_value(run["request"].clone())
        .map_err(|e| NafsError::BadRequest(e.to_string()))?;
    let prior: Vec<nafs_eval::ItemResult> =
        serde_json::from_value(run["items"].clone()).unwrap_or_default();
    let mut v = run.clone();
    v["status"] = json!("running");
    s.store.save_run("eval", id, "running", &v)?;
    let n = prior.len();
    spawn_eval(&s, id, req, prior);
    Ok((
        StatusCode::ACCEPTED,
        Json(json!({"run_id": id, "status": "running", "kept_items": n})),
    ))
}
#[derive(Deserialize, Default)]
struct CalibrateOpts {
    #[serde(default)]
    include_ledger: bool,
}
async fn calibrate(
    State(s): State<AppState>,
    Path(id): Path<Uuid>,
    body: Option<Json<CalibrateOpts>>,
) -> ApiResult<Json<Value>> {
    let run = s.store.get_run(id)?;
    let samples = nafs_eval::calibration_samples(&run)?;
    let opts = body.map(|b| b.0).unwrap_or_default();
    let orch = s.orch.clone();
    let v = tokio::task::spawn_blocking(move || {
        if opts.include_ledger {
            orch.recalibrate_from_ledger(samples)
        } else {
            let (xs, ys): (Vec<_>, Vec<_>) = samples.into_iter().map(|(_, f, y)| (f, y)).unzip();
            orch.fit_and_maybe_promote(&xs, &ys, Some(id.to_string()))
        }
    })
    .await
    .map_err(|e| NafsError::Internal(e.to_string()))??;
    Ok(Json(v))
}

async fn start_rates(
    State(s): State<AppState>,
    Json(req): Json<nafs_eval::rates::RatesRequest>,
) -> ApiResult<(StatusCode, Json<Value>)> {
    let id = Uuid::new_v4();
    let base = json!({"run_id": id, "kind": "rates", "status": "running", "request": req, "created_at": chrono::Utc::now(), "progress": {"done": 0, "total": 0}});
    s.store.save_run("rates", id, "running", &base)?;
    let cancel = register_cancel(&s, id);
    let s2 = s.clone();
    tokio::spawn(async move {
        let s3 = s2.clone();
        let res = nafs_eval::rates::measure(s2.orch.clone(), req.clone(), cancel, move |v| {
            let mut v = v.clone();
            v["run_id"] = json!(id);
            v["kind"] = json!("rates");
            let _ = s3.store.save_run("rates", id, "running", &v);
        })
        .await;
        let mut v = res.unwrap_or_else(
            |e| json!({"status": "failed", "error": e.to_string(), "request": req}),
        );
        v["run_id"] = json!(id);
        v["kind"] = json!("rates");
        v["finished_at"] = json!(chrono::Utc::now());
        let _ = s2
            .store
            .save_run("rates", id, v["status"].as_str().unwrap_or("done"), &v);
        s2.cancels.lock().unwrap().remove(&id);
    });
    Ok((StatusCode::ACCEPTED, Json(base)))
}
async fn get_rates(State(s): State<AppState>) -> Json<Value> {
    let r = s.orch.hub.rates.read().unwrap().clone();
    Json(
        json!({"rates": r, "note": if r.is_empty() { "Not measured yet: start a measurement run (FR-032)." } else { "Mutation-measured (r, phi) per verifier, attached to every verdict." }}),
    )
}

#[derive(Deserialize)]
struct VerifyReq {
    answer: String,
    verification: VerificationInputs,
}
async fn verify_adhoc(
    State(s): State<AppState>,
    Json(b): Json<VerifyReq>,
) -> ApiResult<Json<Value>> {
    if let Some(e) = b.verification.validation_error() {
        return Err(NafsError::BadRequest(e).into());
    }
    s.orch.validate_lean(&b.verification)?;
    if b.answer.len() > 200_000 {
        return Err(NafsError::BadRequest("answer too long".into()).into());
    }
    match s.orch.hub.verify(&b.answer, &b.verification).await {
        Some(v) => Ok(Json(json!(v))),
        None => Err(NafsError::BadRequest("no verification inputs (tests, test_code with entry_point, reference_answer or json_schema)".into()).into()),
    }
}
// ---------------- Math and code harness (FR-090..FR-093) ----------------
/// What the integrated harness offers on this server, and where each part came from.
async fn harness_info(State(s): State<AppState>) -> Json<Value> {
    let c = &s.cfg;
    Json(json!({
        "lean": {
            "available": s.orch.hub.lean.is_some(),
            "executable": s.orch.hub.lean.as_ref().map(|l| l.exe.display().to_string()),
            "mathlib": !c.verifiers.lean_path.is_empty(),
            "protocol": "Open Math Harness verify: frozen statement and preamble, forbidden tokens, marker-fenced #print axioms and pp.all type check",
        },
        "repair": {
            "policy": c.deliberation.repair_policy,
            "fanout": c.deliberation.repair_fanout,
            "patch_repair": c.deliberation.patch_repair,
        },
        // FR-094 / FR-095: concurrent hedged rounds, the System 2 slots they can use, and fused agreement samples.
        "parallel": {
            "enabled": c.deliberation.parallel.enabled,
            "max_width": c.deliberation.parallel.max_width,
            "hedge_after_failure": c.deliberation.parallel.hedge_after_failure,
            "system2_slots": s.orch.active().s2.slots(),
            "cost_factors": c.deliberation.parallel.cost_factors,
            "fuse_always_samples": c.gate.fuse_always_samples,
        },
        // FR-096: System 1 samples drawn and verified before a failed draft is escalated.
        "rescue": { "samples": c.gate.rescue_samples, "temperature": c.gate.rescue_temperature },
        "code_agent": {
            "enabled": c.harness.code_agent_enabled,
            "model": c.harness.model,
            "max_steps": c.harness.max_steps,
            "wall_clock_s": c.harness.wall_clock_s,
            "context": c.harness.context,
            "compact_at_tokens": c.harness.compact_at_tokens,
            "constrain": c.harness.constrain,
            "grammar_supported": s.orch.active().s2.supports_grammar(),
            "stream_cut": c.harness.stream_cut,
            "edit_format": c.harness.edit_format,
            "whole_max_lines": c.harness.whole_max_lines,
            "unstick": c.harness.unstick,
            "thinking_budget": c.harness.thinking_budget,
            "first_step_thinking_budget": c.harness.first_step_thinking_budget,
        },
        "sources": [
            {"name": "Open Math Harness", "revision": "687f2a2", "license": "Apache-2.0", "parts": ["lean rules and verify protocol", "discovery scoring", "best-first candidate tree", "prover stop rule"]},
            {"name": "OpenAI Codex (codex-rs)", "revision": "970111f6", "license": "Apache-2.0", "parts": ["apply_patch parser and fuzzy matcher", "middle truncation", "exec output format", "seatbelt -D parameters", "turn loop", "append-only history and compaction", "freeform apply_patch grammar (as GBNF)", "dispatch at item done (stream cut)", "budget reminders", "turn diff"]},
        ],
    }))
}

/// FR-093: run the code agent to completion and return its files, verdict and steps.
async fn code_agent(
    State(s): State<AppState>,
    Json(b): Json<CodeAgentRequest>,
) -> ApiResult<Json<CodeAgentReply>> {
    s.orch
        .run_code_agent(b)
        .await
        .map(Json)
        .map_err(|e| NafsError::BadRequest(e).into())
}

#[derive(Deserialize)]
struct JudgeReq {
    question: String,
    answer: String,
}
async fn judge_adhoc(State(s): State<AppState>, Json(b): Json<JudgeReq>) -> ApiResult<Json<Value>> {
    if b.question.trim().is_empty() || b.answer.trim().is_empty() {
        return Err(NafsError::BadRequest("question and answer are required".into()).into());
    }
    Ok(Json(json!(
        s.orch.judge(&b.question, &b.answer, None).await?
    )))
}

// ---------------- Gate: mode, calibration, measured parameters (FR-011, FR-014, FR-015, FR-021) ----------------
fn gate_view(s: &AppState) -> Value {
    let cal = s.orch.calibration();
    let gs = s.orch.gate_state();
    let verdict = s.orch.promotion(&cal);
    let g = &s.cfg.gate;
    json!({
        "mode": gs.mode, "escalation_budget": gs.escalation_budget,
        "lambda_star": if gs.mode == "budget" { cal.lambda_star(gs.escalation_budget) } else { None },
        "effective_rule": if verdict.is_ok() && gs.mode != "fixed" { gs.mode.clone() } else { "fixed".into() },
        "promoted": verdict.is_ok(), "promotion_reason": verdict.err(),
        "calibration": cal,
        "requirements": {"min_n": g.calibration_min_n, "min_per_class": g.calibration_min_per_class, "max_ece": g.calibration_max_ece, "dkw_n": g.dkw_n(), "epsilon": g.budget_epsilon, "delta": g.budget_delta},
        "params": s.orch.param_provenance(), "derived": s.orch.params(),
        "recalibrate_every_h": g.recalibrate_every_h,
    })
}
async fn get_gate(State(s): State<AppState>) -> Json<Value> {
    Json(gate_view(&s))
}
#[derive(Deserialize)]
struct SetGate {
    #[serde(default)]
    mode: Option<String>,
    #[serde(default)]
    escalation_budget: Option<f32>,
}
async fn set_gate(State(s): State<AppState>, Json(b): Json<SetGate>) -> ApiResult<Json<Value>> {
    s.orch.set_gate_state(b.mode, b.escalation_budget)?;
    Ok(Json(gate_view(&s)))
}
#[derive(Deserialize, Default)]
struct RunIds {
    #[serde(default)]
    run_ids: Vec<Uuid>,
    #[serde(default)]
    min_n: Option<usize>,
    #[serde(default)]
    apply: bool,
}
fn load_runs(s: &AppState, ids: &[Uuid]) -> ApiResult<Vec<Value>> {
    if ids.len() > 50 {
        return Err(NafsError::BadRequest("at most 50 runs".into()).into());
    }
    ids.iter()
        .map(|id| s.store.get_run(*id).map_err(ApiError::from))
        .collect()
}
async fn recalibrate(
    State(s): State<AppState>,
    body: Option<Json<RunIds>>,
) -> ApiResult<Json<Value>> {
    let b = body.map(|b| b.0).unwrap_or_default();
    let mut extra = vec![];
    for r in load_runs(&s, &b.run_ids)? {
        extra.extend(nafs_eval::calibration_samples(&r)?);
    }
    let orch = s.orch.clone();
    let v = tokio::task::spawn_blocking(move || orch.recalibrate_from_ledger(extra))
        .await
        .map_err(|e| NafsError::Internal(e.to_string()))??;
    Ok(Json(v))
}
async fn derive_params(State(s): State<AppState>, Json(b): Json<RunIds>) -> ApiResult<Json<Value>> {
    if b.run_ids.is_empty() {
        return Err(NafsError::BadRequest(
            "run_ids: choose evaluation runs that include fast_only and slow_only".into(),
        )
        .into());
    }
    let runs = load_runs(&s, &b.run_ids)?;
    let p = nafs_eval::derive_params(&runs, &s.orch, b.min_n.unwrap_or(10).max(3))?;
    if b.apply {
        s.orch.set_params(p.clone())?;
    }
    Ok(Json(
        json!({"applied": b.apply, "params": p, "provenance": s.orch.param_provenance(),
        "note": if b.apply { "Measured values now replace the placeholders for the listed domains (recorded in the params file, FR-021)." } else { "Preview only; send apply = true to use these values." }}),
    ))
}
async fn get_params(State(s): State<AppState>) -> Json<Value> {
    Json(json!({"derived": s.orch.params(), "provenance": s.orch.param_provenance()}))
}
async fn reset_params(State(s): State<AppState>) -> ApiResult<Json<Value>> {
    s.orch.set_params(Default::default())?;
    Ok(Json(
        json!({"derived": s.orch.params(), "provenance": s.orch.param_provenance()}),
    ))
}

// ---------------- Audits and self-model (FR-040, FR-041, Prop. 5) ----------------
async fn list_audits(State(s): State<AppState>, Query(q): Query<Lim>) -> ApiResult<Json<Value>> {
    Ok(Json(
        json!({"counts": s.store.audit_counts()?, "audits": s.store.list_audits(q.limit.unwrap_or(100).min(1000) as i64)?,
        "worker": {"enabled": s.cfg.audit.enabled, "idle_only": s.cfg.audit.idle_only, "poll_s": s.cfg.audit.poll_s, "audit_fraction": s.cfg.gate.audit_fraction}}),
    ))
}
async fn run_next_audit(State(s): State<AppState>) -> ApiResult<Json<Value>> {
    let done = s.orch.run_one_audit().await?;
    Ok(Json(
        json!({"completed": done, "counts": s.store.audit_counts()?}),
    ))
}
async fn self_model(State(s): State<AppState>) -> ApiResult<Json<Value>> {
    let p = s.orch.active();
    let eps = s.cfg.gate.audit_fraction as f64;
    let mut domains = s
        .store
        .self_model(eps, s.cfg.audit.min_audits_measured as i64, &p.name)?;
    let fb = s.store.feedback_by_domain()?;
    for d in domains.iter_mut() {
        let (up, down) = fb
            .get(d["domain"].as_str().unwrap_or(""))
            .copied()
            .unwrap_or((0, 0));
        d["human_up"] = json!(up);
        d["human_down"] = json!(down);
    }
    Ok(Json(
        json!({"model_hash": format!("{}|{}", p.s1.cfg.hf_id, p.s2.cfg.hf_id), "adapter_hash": "none", "backend": p.name,
        "calibration": s.orch.calibration(), "gate": {"mode": s.orch.gate_state().mode, "promoted": s.orch.promoted()},
        "audit_fraction": eps, "audits": s.store.audit_counts()?, "domains": domains,
        "note": format!("Capability per domain uses the audit-weighted estimator of Prop. 5. Domains with fewer than {} completed audits are 'unmeasured' (FR-041); unknown values are null, never 0.", s.cfg.audit.min_audits_measured)}),
    ))
}

async fn system(State(s): State<AppState>) -> Json<Value> {
    let health = s.orch.health().await;
    let active = s.orch.active().name.clone();
    let c = &s.cfg;
    Json(json!({
        "version": env!("CARGO_PKG_VERSION"), "config_hash": c.hash, "profile": c.assistant.profile, "active_backend": active, "auth_enabled": !s.tokens.is_empty(),
        "security": {"auth_enabled": !s.tokens.is_empty(), "bind": c.server.bind, "loopback_only": crate::security::is_loopback(&c.server.bind),
            "sandbox_isolation": nafs_verify::sandbox::isolation_available(), "retention_days": c.storage.retention_days, "tls": "terminate at a reverse proxy (deploy/tls/)", "same_origin_only": true},
        "uptime_s": s.started.elapsed().as_secs(), "inflight": s.orch.inflight.load(Ordering::Relaxed),
        "host": {"os": std::env::consts::OS, "arch": std::env::consts::ARCH, "cpus": std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0)},
        "backends": health.into_iter().map(|(name, label, kind, a, b)| json!({"name": name, "label": label, "kind": kind, "active": name == active, "system1": a, "system2": b})).collect::<Vec<_>>(),
        "gate": {"k_samples": c.gate.k_samples, "lambda_default": c.gate.lambda_default, "cost_unit_s": c.gate.cost_unit_s, "loss_wrong": c.gate.loss_wrong,
                 "fixed": c.gate.fixed, "audit_fraction": c.gate.audit_fraction, "state": gate_view(&s)},
        "judge": c.judge, "audit": {"config": c.audit, "counts": s.store.audit_counts().unwrap_or(Value::Null)},
        "verifier_rates": s.orch.hub.rates.read().unwrap().clone(),
        "deliberation": c.deliberation, "server": {"bind": c.server.bind, "request_wall_clock_cap_s": c.server.request_wall_clock_cap_s, "max_prompt_tokens": c.server.max_prompt_tokens},
    }))
}
#[derive(Deserialize)]
struct SetBackend {
    name: String,
    /// Switch even when the backend's servers do not answer a health check.
    #[serde(default)]
    force: bool,
}
async fn set_backend(
    State(s): State<AppState>,
    Json(b): Json<SetBackend>,
) -> ApiResult<Json<Value>> {
    // Switching to a backend that is not running sends every request to dead servers (503), so it is
    // refused unless forced.
    if !b.force {
        let health = s.orch.health().await;
        if let Some((_, label, _, h1, h2)) = health.iter().find(|h| h.0 == b.name) {
            if !(h1.ok && h2.ok) {
                let why = [h1, h2]
                    .iter()
                    .filter(|h| !h.ok)
                    .map(|h| format!("{} at {} is not reachable", h.role, h.url))
                    .collect::<Vec<_>>()
                    .join("; ");
                return Err(NafsError::BackendUnavailable(format!(
                    "{label}: {why}. Start it first (see the System page), or pass \"force\": true."
                ))
                .into());
            }
        }
    }
    let before = s.orch.active().name.clone();
    s.orch.set_active(&b.name)?;
    tracing::info!("active backend switched from '{before}' to '{}'", b.name);
    Ok(Json(json!({"active_backend": b.name})))
}
async fn activity(State(s): State<AppState>) -> ApiResult<Json<Value>> {
    Ok(Json(json!(s.store.recent_replies(200)?)))
}

/// Background workers: the audit stream (FR-040) runs only while no request is in flight, and the
/// gate is recalibrated from the ledger on schedule (FR-015).
pub fn spawn_background(s: AppState) {
    {
        let s2 = s.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs(3)).await;
            let n = s2.orch.warm_prompt_cache().await;
            tracing::info!("prompt caches warmed ({n} prefixes)");
        });
    }
    if let Ok(ids) = s.store.mark_interrupted_runs() {
        if !ids.is_empty() {
            tracing::warn!("{} runs were interrupted by a restart", ids.len());
        }
        // Evaluation runs resume where they stopped (completed items are kept).
        for (kind, id) in ids.into_iter().filter(|(k, _)| k == "eval") {
            let _ = kind;
            if let Ok(run) = s.store.get_run(id) {
                if let Ok(req) =
                    serde_json::from_value::<nafs_eval::EvalRunRequest>(run["request"].clone())
                {
                    let prior: Vec<nafs_eval::ItemResult> =
                        serde_json::from_value(run["items"].clone()).unwrap_or_default();
                    tracing::info!(
                        "resuming evaluation run {id} with {} completed items",
                        prior.len()
                    );
                    let mut v = run.clone();
                    v["status"] = json!("running");
                    let _ = s.store.save_run("eval", id, "running", &v);
                    spawn_eval(&s, id, req, prior);
                }
            }
        }
    }
    if s.cfg.audit.enabled {
        let s2 = s.clone();
        tokio::spawn(async move {
            let mut backoff = 0u64;
            loop {
                tokio::time::sleep(Duration::from_secs(s2.cfg.audit.poll_s.max(1) + backoff)).await;
                let idle = s2.orch.inflight.load(Ordering::SeqCst) == 0;
                if s2.cfg.audit.idle_only && !idle {
                    continue;
                }
                match s2.orch.run_one_audit().await {
                    Ok(_) => backoff = 0,
                    Err(e) => {
                        backoff = (backoff * 2 + 30).min(600);
                        tracing::debug!(
                            "audit worker: {e}; retrying in {}s",
                            s2.cfg.audit.poll_s + backoff
                        );
                    }
                }
            }
        });
    }
    if s.cfg.storage.retention_days > 0 {
        let s2 = s.clone();
        tokio::spawn(async move {
            loop {
                match s2.store.purge_older_than(s2.cfg.storage.retention_days) {
                    Ok(n) if n > 0 => tracing::info!(
                        "retention: removed {n} conversations older than {} days (SR-004)",
                        s2.cfg.storage.retention_days
                    ),
                    Err(e) => tracing::warn!("retention purge failed: {e}"),
                    _ => {}
                }
                tokio::time::sleep(Duration::from_secs(24 * 3600)).await;
            }
        });
    }
    {
        // Keep the nafs_alert_firing gauges current for Prometheus (OR-004).
        let s2 = s.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(60)).await;
                let s3 = s2.clone();
                let _ = tokio::task::spawn_blocking(move || evaluate_alerts(&s3)).await;
            }
        });
    }
    let every = s.cfg.gate.recalibrate_every_h;
    if every > 0 {
        let s2 = s.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(every * 3600)).await;
                let orch = s2.orch.clone();
                match tokio::task::spawn_blocking(move || orch.recalibrate_from_ledger(vec![]))
                    .await
                {
                    Ok(Ok(v)) => tracing::info!(
                        "scheduled recalibration: promoted = {}, reason = {}",
                        v["promoted"],
                        v["reason"]
                    ),
                    Ok(Err(e)) => tracing::info!("scheduled recalibration skipped: {e}"),
                    Err(e) => tracing::warn!("scheduled recalibration failed: {e}"),
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::{host_only, percent_decode};
    #[test]
    fn token_query_is_percent_decoded() {
        assert_eq!(percent_decode("ab%2Bcd%2F%3D"), "ab+cd/=");
        assert_eq!(percent_decode("plain"), "plain");
        assert_eq!(percent_decode("bad%zz"), "bad%zz");
    }
    #[test]
    fn host_parsing() {
        assert_eq!(host_only("127.0.0.1:8080"), "127.0.0.1");
        assert_eq!(host_only("LOCALHOST"), "localhost");
        assert_eq!(host_only("[::1]:8080"), "[::1]");
        assert_eq!(host_only("evil.example:8080"), "evil.example");
    }
}
