//! Knowledge Base API (Phase 20, IF-K01): knowledge bases, document uploads, passages and search, backed by `nafs-kb`
//! and SurrealDB. Uploads have their own body limit; every other route keeps the gateway's 2 MB limit.

use axum::extract::{DefaultBodyLimit, Multipart, Path, Query, State};
use axum::http::StatusCode;
use axum::middleware;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use nafs_core::config::KnowledgeCfg;
use nafs_kb::{ChunkOpts, Kb, KbConfig, KbError, StoreConfig};
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::Arc;

use crate::api::AppState;

/// Build the service from `[knowledge]`; `None` when it is turned off. The password is read from its file now, so a
/// missing file shows up as "unavailable" with the reason rather than as a failed login later.
pub fn from_config(k: &KnowledgeCfg) -> Option<Arc<Kb>> {
    if !k.enabled {
        return None;
    }
    let password = std::fs::read_to_string(&k.password_file)
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    if password.is_empty() {
        tracing::warn!(file = %k.password_file, "knowledge: no SurrealDB password file; run scripts/serve_surreal.sh start");
    }
    Some(Kb::start(KbConfig {
        store: StoreConfig {
            url: k.url.clone(),
            namespace: k.namespace.clone(),
            database: k.database.clone(),
            user: k.user.clone(),
            password,
        },
        files_dir: k.files_dir.clone().into(),
        max_upload_bytes: k.max_upload_mb.max(1) * 1024 * 1024,
        max_files_per_request: k.max_files_per_request.max(1),
        chunk: ChunkOpts {
            chars: k.chunk_chars,
            overlap: k.chunk_overlap,
        },
        search_k: k.search_k.max(1),
        embed: (!k.embed_url.trim().is_empty()).then(|| nafs_kb::EmbedConfig {
            url: k.embed_url.clone(),
            model: k.embed_model.clone(),
            dims: k.embed_dims,
            batch: k.embed_batch.max(1),
        }),
        exact_search_below: k.exact_search_below,
        graph: (!k.graph_llm_url.trim().is_empty()).then(|| nafs_kb::GraphConfig {
            llm_url: k.graph_llm_url.clone(),
            model: k.graph_model.clone(),
            max_tokens: k.graph_max_tokens.max(64),
            merge_similarity: k.entity_merge_similarity,
        }),
        graph_weight: k.graph_weight,
        vector_weight: k.vector_weight,
        vector_cache_mb: k.vector_cache_mb,
        graph_beam: k.graph_beam.max(1),
        graph_decay: k.graph_decay,
        index: index_config(k),
        limits: nafs_kb::Limits {
            max_documents_per_base: k.max_documents_per_base.max(1),
            max_chunks_per_document: k.max_chunks_per_document.max(1),
            max_queued_jobs: k.max_queued_jobs.max(1),
            parse_timeout_s: k.parse_timeout_s.max(1),
            search_timeout_s: k.search_timeout_s.max(1),
        },
    }))
}

/// The index's settings from `[knowledge]` (Phase 25).
fn index_config(k: &nafs_core::config::KnowledgeCfg) -> nafs_kb::IndexConfig {
    use nafs_kb::index::{Bm25, Fusion, Params, Ppr};
    nafs_kb::IndexConfig {
        memory_mb: k.index_memory_mb,
        params: Params {
            bm25: Bm25 {
                weights: k.bm25_weights,
                ..Bm25::default()
            },
            fusion: if k.fusion == "convex" {
                Fusion::Convex {
                    alpha: k.fusion_alpha,
                }
            } else {
                Fusion::Rrf {
                    vector_weight: k.vector_weight,
                }
            },
            prf: None,
            fuzzy: k.fuzzy_terms,
            ppr: (k.graph_arm_weight > 0.0).then(|| Ppr {
                weight: k.graph_arm_weight,
                ..Ppr::default()
            }),
        },
    }
}

/// The routes, with authentication and the same-origin guard applied here because they sit outside the main
/// router's body limit.
pub fn routes(state: AppState) -> Router<AppState> {
    // SR-K11: the request as a whole is bounded by `max_request_mb`, not by files times file size (which let one
    // request carry a gigabyte), and never below one full file.
    let k = &state.cfg.knowledge;
    let upload_limit = k.max_request_mb.max(k.max_upload_mb).max(1) * 1024 * 1024;
    // Room for multipart framing on top of the files themselves.
    let upload_limit = upload_limit + 1024 * 1024;
    let uploads = Router::new()
        .route("/v1/kb/{kb}/documents", get(list_documents).post(upload))
        .layer(DefaultBodyLimit::max(upload_limit))
        .layer(tower_http::limit::RequestBodyLimitLayer::new(upload_limit));
    let rest = Router::new()
        .route("/v1/kb/status", get(status))
        .route("/v1/kb", get(list_kbs).post(create_kb))
        .route(
            "/v1/kb/{kb}",
            get(get_kb).patch(update_kb).delete(delete_kb),
        )
        .route(
            "/v1/kb/{kb}/documents/{doc}",
            get(get_document).delete(delete_document),
        )
        .route("/v1/kb/{kb}/documents/{doc}/reingest", post(reingest))
        .route("/v1/kb/{kb}/documents/{doc}/chunks", get(list_chunks))
        .route("/v1/kb/{kb}/search", post(search))
        .route("/v1/kb/{kb}/reindex", post(reindex))
        .route("/v1/kb/{kb}/graph", get(graph_view))
        .route("/v1/kb/{kb}/graph/build", post(graph_build))
        .route("/v1/kb/{kb}/graph/entities", get(graph_entities))
        .route("/v1/kb/{kb}/graph/entities/{entity}", get(graph_entity))
        .route("/v1/kb/{kb}/ontology", get(get_ontology).put(put_ontology))
        .route(
            "/v1/kb/{kb}/memory/runs",
            get(crate::memory_runs::list).post(crate::memory_runs::start),
        )
        .route(
            "/v1/kb/{kb}/memory/runs/{run}/stop",
            post(crate::memory_runs::stop),
        )
        .route(
            "/v1/kb/{kb}/memory/runs/{run}/activate",
            post(crate::memory_runs::activate),
        )
        .layer(tower_http::limit::RequestBodyLimitLayer::new(
            2 * 1024 * 1024,
        ));
    uploads
        .merge(rest)
        .layer(middleware::from_fn_with_state(
            state.clone(),
            crate::api::auth,
        ))
        .layer(middleware::from_fn_with_state(
            state,
            crate::api::origin_guard,
        ))
}

/// A knowledge-base error as the gateway's `{code, message}` envelope with the status of IF-K01.
pub struct KbApiError(KbError);

impl From<KbError> for KbApiError {
    fn from(e: KbError) -> Self {
        KbApiError(e)
    }
}

impl IntoResponse for KbApiError {
    fn into_response(self) -> Response {
        let status = match &self.0 {
            KbError::Unavailable(_) => StatusCode::SERVICE_UNAVAILABLE,
            KbError::NotFound(_) => StatusCode::NOT_FOUND,
            KbError::BadInput(_) => StatusCode::BAD_REQUEST,
            KbError::Conflict(_) => StatusCode::CONFLICT,
            KbError::TooLarge(_) => StatusCode::PAYLOAD_TOO_LARGE,
            KbError::Unsupported(_) => StatusCode::UNSUPPORTED_MEDIA_TYPE,
            KbError::Busy(_) => StatusCode::TOO_MANY_REQUESTS,
            KbError::Store(_) | KbError::Io(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };
        if status.is_server_error() {
            tracing::warn!(error = %self.0, "knowledge API error");
        }
        (
            status,
            Json(json!({ "code": self.0.code(), "message": self.0.to_string() })),
        )
            .into_response()
    }
}

type KbResult<T> = Result<T, KbApiError>;

fn service(s: &AppState) -> Result<&Arc<Kb>, KbApiError> {
    s.kb.as_ref().ok_or_else(|| {
        KbApiError(KbError::Unavailable(
            "Knowledge is turned off in the configuration ([knowledge] enabled = false)".into(),
        ))
    })
}

async fn ready(s: &AppState) -> Result<&Arc<Kb>, KbApiError> {
    let kb = service(s)?;
    kb.ready().await?;
    Ok(kb)
}

async fn status(State(s): State<AppState>) -> Json<Value> {
    let c = &s.cfg.knowledge;
    let Some(kb) = s.kb.as_ref() else {
        return Json(json!({ "enabled": false, "reachable": false, "url": c.url }));
    };
    let version = kb.store().version().await;
    let schema = match &version {
        Ok(_) => kb.ready().await.map_err(|e| e.to_string()),
        Err(_) => Ok(()),
    };
    Json(json!({
        "enabled": true,
        "reachable": version.is_ok() && schema.is_ok(),
        "version": version.as_ref().ok(),
        "error": version.as_ref().err().map(|e| e.to_string()).or(schema.err()),
        "url": c.url,
        "queue": kb.queue(),
        "index": { "memory_mb": c.index_memory_mb, "bases": kb.index_stats().0, "passages": kb.index_stats().1,
            "mb": kb.index_stats().2 / 1_048_576 },
        // Memory tokens need Smiley Core: no other engine takes caller-supplied input vectors, so the chat page
        // must not offer the mode while another one is active (FR-K23).
        "memory": s.knowledge_memory && s.orch.active().s1.kind == "smiley",
        "embedder": kb.embedder_status().await.map(|(model, reachable)| json!({ "model": model, "reachable": reachable, "url": c.embed_url })),
        "limits": { "max_upload_mb": c.max_upload_mb, "max_files_per_request": c.max_files_per_request,
            "max_request_mb": c.max_request_mb, "concurrent_uploads": c.concurrent_uploads,
            "max_documents_per_base": c.max_documents_per_base, "max_chunks_per_document": c.max_chunks_per_document,
            "max_queued_jobs": c.max_queued_jobs, "parse_timeout_s": c.parse_timeout_s, "search_timeout_s": c.search_timeout_s,
            "max_concurrent_trainers": c.max_concurrent_trainers, "vector_cache_mb": c.vector_cache_mb },
        "kinds": ["pdf", "docx", "pptx", "tex", "html", "md", "txt"],
    }))
}

async fn list_kbs(State(s): State<AppState>) -> KbResult<Json<Value>> {
    let kb = ready(&s).await?;
    // `memory` per base (Phase 6): chat may only offer memory mode for a base that has a module to read it with.
    let bases: Vec<Value> = kb
        .store()
        .list_kbs()
        .await?
        .into_iter()
        .map(|b| {
            let has = s.knowledge_memory_global
                || crate::memory_runs::active_checkpoint(&s.cfg.knowledge.files_dir, &b.id)
                    .is_some();
            let mut v = serde_json::to_value(b).unwrap_or_default();
            v["memory"] = json!(has);
            v
        })
        .collect();
    Ok(Json(json!({ "bases": bases })))
}

#[derive(Deserialize)]
struct NewKb {
    name: String,
    #[serde(default)]
    description: String,
}

async fn create_kb(
    State(s): State<AppState>,
    Json(b): Json<NewKb>,
) -> KbResult<(StatusCode, Json<Value>)> {
    let kb = ready(&s).await?;
    let created = kb.store().create_kb(&b.name, &b.description).await?;
    Ok((StatusCode::CREATED, Json(json!(created))))
}

async fn get_kb(State(s): State<AppState>, Path(id): Path<String>) -> KbResult<Json<Value>> {
    let kb = ready(&s).await?;
    Ok(Json(json!(kb.store().get_kb(&id).await?)))
}

#[derive(Deserialize)]
struct PatchKb {
    name: Option<String>,
    description: Option<String>,
}

async fn update_kb(
    State(s): State<AppState>,
    Path(id): Path<String>,
    Json(b): Json<PatchKb>,
) -> KbResult<Json<Value>> {
    let kb = ready(&s).await?;
    Ok(Json(json!(
        kb.store()
            .update_kb(&id, b.name.as_deref(), b.description.as_deref())
            .await?
    )))
}

async fn delete_kb(State(s): State<AppState>, Path(id): Path<String>) -> KbResult<Json<Value>> {
    let kb = ready(&s).await?;
    kb.store().get_kb(&id).await?;
    // Nothing may keep working on a base that is going away: a trainer would run on for hours into a folder that is
    // no longer there, and prepared documents would stay in memory.
    let trainers = crate::memory_runs::stop_trainers(&s.cfg.knowledge.files_dir, &id);
    let (documents, chunks) = kb.delete_kb(&id).await?;
    s.orch.forget_knowledge(&id);
    // The ledger kept the passages each reply was answered from: they go too.
    let store = s.store.clone();
    let key = id.clone();
    let replies = tokio::task::spawn_blocking(move || store.scrub_kb_sources(&key, None))
        .await
        .ok()
        .and_then(|r| r.ok())
        .unwrap_or(0);
    Ok(Json(
        json!({ "deleted": true, "documents": documents, "chunks": chunks,
            "trainers_stopped": trainers, "replies_scrubbed": replies }),
    ))
}

async fn list_documents(
    State(s): State<AppState>,
    Path(id): Path<String>,
) -> KbResult<Json<Value>> {
    let kb = ready(&s).await?;
    kb.store().get_kb(&id).await?;
    Ok(Json(
        json!({ "documents": kb.store().list_documents(&id).await? }),
    ))
}

/// FR-K02..K04: every `file` part becomes a document (or a refusal with its reason); ingestion is queued.
async fn upload(
    State(s): State<AppState>,
    Path(id): Path<String>,
    mut form: Multipart,
) -> KbResult<(StatusCode, Json<Value>)> {
    let kb = ready(&s).await?;
    kb.store().get_kb(&id).await?;
    let max_files = kb.config().max_files_per_request;
    let max_bytes = kb.config().max_upload_bytes;
    // At most `concurrent_uploads` requests are received at once: each is held in memory while it arrives.
    static GATE: std::sync::OnceLock<Arc<tokio::sync::Semaphore>> = std::sync::OnceLock::new();
    let gate = GATE
        .get_or_init(|| {
            Arc::new(tokio::sync::Semaphore::new(
                s.cfg.knowledge.concurrent_uploads.max(1),
            ))
        })
        .clone();
    let _slot = gate.try_acquire_owned().map_err(|_| {
        KbError::Busy("other uploads are being received; try again in a moment".into())
    })?;
    let mut documents = Vec::new();
    let mut refused = Vec::new();
    let mut seen = 0usize;
    while let Some(field) = form
        .next_field()
        .await
        .map_err(|e| KbError::BadInput(format!("the upload could not be read: {e}")))?
    {
        let Some(filename) = field.file_name().map(str::to_string) else {
            continue;
        };
        seen += 1;
        if seen > max_files {
            refused.push(json!({ "filename": filename, "reason": format!("at most {max_files} files per upload") }));
            continue;
        }
        // Read in pieces and stop at the limit: a file far over it is refused without being held whole.
        let mut buf: Vec<u8> = Vec::new();
        let mut too_big = false;
        let mut field = field;
        let received = loop {
            match field.chunk().await {
                Ok(Some(c)) => {
                    if buf.len() + c.len() > max_bytes {
                        too_big = true;
                        break Ok(());
                    }
                    buf.extend_from_slice(&c);
                }
                Ok(None) => break Ok(()),
                Err(e) => break Err(e),
            }
        };
        if too_big {
            refused.push(
                json!({ "filename": nafs_kb::display_name(&filename), "reason": format!(
                "the file is larger than the limit of {} MB", max_bytes / 1_048_576) }),
            );
            continue;
        }
        if let Err(e) = received {
            refused.push(json!({ "filename": filename, "reason": format!("the file could not be received: {e}") }));
            continue;
        }
        let bytes = buf;
        match kb.upload(&id, &filename, &bytes).await {
            Ok(d) => documents.push(d),
            Err(e @ (KbError::TooLarge(_) | KbError::Unsupported(_) | KbError::BadInput(_))) => {
                refused.push(json!({ "filename": nafs_kb::display_name(&filename), "reason": e.to_string() }))
            }
            Err(e) => return Err(e.into()),
        }
    }
    if seen == 0 {
        return Err(KbError::BadInput(
            "no files in the upload (send them as multipart parts named \"file\")".into(),
        )
        .into());
    }
    let status = if documents.is_empty() {
        StatusCode::UNPROCESSABLE_ENTITY
    } else {
        StatusCode::ACCEPTED
    };
    Ok((
        status,
        Json(json!({ "documents": documents, "refused": refused })),
    ))
}

async fn get_document(
    State(s): State<AppState>,
    Path((id, doc)): Path<(String, String)>,
) -> KbResult<Json<Value>> {
    let kb = ready(&s).await?;
    Ok(Json(json!(kb.store().get_document(&id, &doc).await?)))
}

async fn delete_document(
    State(s): State<AppState>,
    Path((id, doc)): Path<(String, String)>,
) -> KbResult<Json<Value>> {
    let kb = ready(&s).await?;
    let chunks = kb.delete_document(&id, &doc).await?;
    s.orch.forget_knowledge(&id);
    let store = s.store.clone();
    let (key, doc_key) = (id.clone(), doc.clone());
    let replies = tokio::task::spawn_blocking(move || store.scrub_kb_sources(&key, Some(&doc_key)))
        .await
        .ok()
        .and_then(|r| r.ok())
        .unwrap_or(0);
    Ok(Json(
        json!({ "deleted": true, "chunks": chunks, "replies_scrubbed": replies }),
    ))
}

async fn reingest(
    State(s): State<AppState>,
    Path((id, doc)): Path<(String, String)>,
) -> KbResult<(StatusCode, Json<Value>)> {
    let kb = ready(&s).await?;
    Ok((
        StatusCode::ACCEPTED,
        Json(json!(kb.reingest(&id, &doc).await?)),
    ))
}

#[derive(Deserialize)]
struct Page {
    #[serde(default)]
    offset: usize,
    #[serde(default = "d_limit")]
    limit: usize,
}
fn d_limit() -> usize {
    50
}

async fn list_chunks(
    State(s): State<AppState>,
    Path((id, doc)): Path<(String, String)>,
    Query(p): Query<Page>,
) -> KbResult<Json<Value>> {
    let kb = ready(&s).await?;
    kb.store().get_document(&id, &doc).await?;
    let (chunks, total) = kb.store().list_chunks(&id, &doc, p.offset, p.limit).await?;
    Ok(Json(
        json!({ "chunks": chunks, "total": total, "offset": p.offset }),
    ))
}

#[derive(Deserialize)]
struct SearchReq {
    query: String,
    k: Option<usize>,
    #[serde(default)]
    mode: Option<String>,
    /// "store" answers through the store as every search did before the in-process index; for comparing the two.
    #[serde(default)]
    engine: Option<String>,
}

async fn search(
    State(s): State<AppState>,
    Path(id): Path<String>,
    Json(b): Json<SearchReq>,
) -> KbResult<Json<Value>> {
    let kb = ready(&s).await?;
    let mode = match b.mode.as_deref() {
        None => nafs_kb::SearchMode::Hybrid,
        Some(m) => nafs_kb::SearchMode::parse(m).ok_or_else(|| {
            KbError::BadInput(format!(
                "unknown search mode \"{m}\" (lexical, vector or hybrid)"
            ))
        })?,
    };
    if b.query.chars().count() > 2000 {
        return Err(KbError::BadInput("the query is longer than 2,000 characters".into()).into());
    }
    let engine = match b.engine.as_deref() {
        None => nafs_kb::Engine::Auto,
        Some(e) => nafs_kb::Engine::parse(e)
            .ok_or_else(|| KbError::BadInput(format!("unknown engine \"{e}\" (index or store)")))?,
    };
    // An indexed base is known to exist; asking the store again would be the round trip the index removes.
    if engine == nafs_kb::Engine::Store || kb.index_of(&id).await?.is_none() {
        kb.store().get_kb(&id).await?;
    }
    let r = kb
        .search_with(
            &id,
            &b.query,
            b.k.unwrap_or(kb.config().search_k),
            mode,
            engine,
        )
        .await?;
    Ok(Json(json!(r)))
}

/// Give vectors to the documents of a knowledge base that lack them (stored before the embedder ran, or while it was
/// down).
#[derive(Deserialize)]
struct ReindexQuery {
    /// Embed every passage again, so the vectors carry the context lines the graph pass wrote (FR-K53).
    #[serde(default)]
    fresh: bool,
}

async fn reindex(
    State(s): State<AppState>,
    Path(id): Path<String>,
    Query(q): Query<ReindexQuery>,
) -> KbResult<(StatusCode, Json<Value>)> {
    let kb = ready(&s).await?;
    kb.store().get_kb(&id).await?;
    let queued = kb.reindex_fresh(Some(&id), q.fresh).await?;
    Ok((
        StatusCode::ACCEPTED,
        Json(json!({ "queued": queued, "fresh": q.fresh })),
    ))
}

/// Start reading the documents of a knowledge base for its graph (FR-K30); long-running, in the background.
#[derive(Deserialize)]
struct BuildQuery {
    /// Forget the graph and read every passage again.
    #[serde(default)]
    fresh: bool,
}

async fn graph_build(
    State(s): State<AppState>,
    Path(id): Path<String>,
    Query(q): Query<BuildQuery>,
) -> KbResult<(StatusCode, Json<Value>)> {
    let kb = ready(&s).await?;
    kb.store().get_kb(&id).await?;
    let queued = kb.build_graph(&id, q.fresh).await?;
    Ok((StatusCode::ACCEPTED, Json(json!({ "queued": queued }))))
}

#[derive(Deserialize)]
struct GraphQuery {
    entity: Option<String>,
    #[serde(default = "d_depth")]
    depth: usize,
    #[serde(default = "d_nodes")]
    limit: usize,
    /// "text" asks for the graph made from the text alone even when a model-built one exists.
    source: Option<String>,
}
fn d_depth() -> usize {
    1
}
fn d_nodes() -> usize {
    60
}

/// FR-K34: the graph around an entity, or the most mentioned entities, with counts and communities.
async fn graph_view(
    State(s): State<AppState>,
    Path(id): Path<String>,
    Query(q): Query<GraphQuery>,
) -> KbResult<Json<Value>> {
    let kb = ready(&s).await?;
    kb.store().get_kb(&id).await?;
    let mut view = kb
        .store()
        .graph_view(&id, q.entity.as_deref(), q.depth, q.limit)
        .await?;
    view["stats"] = kb.store().graph_stats(&id).await?;
    view["source"] = json!("model");
    // FR-K71: a base nobody has built a graph for still has the one made from its text.
    let empty = view["nodes"].as_array().is_none_or(Vec::is_empty);
    if q.source.as_deref() == Some("text") || (empty && q.entity.is_none()) {
        if let Some(text) = kb.text_graph_view(&id, q.entity.as_deref(), q.limit).await {
            if text["nodes"].as_array().is_some_and(|n| !n.is_empty()) {
                for key in ["nodes", "edges", "source", "terms"] {
                    view[key] = text[key].clone();
                }
            }
        }
    }
    view["enabled"] = json!(kb.graph_enabled());
    let docs = kb.store().list_documents(&id).await?;
    let (read, total): (u64, u64) = docs
        .iter()
        .fold((0, 0), |a, d| (a.0 + d.n_graphed, a.1 + d.n_chunks));
    view["progress"] = json!({ "passages_read": read, "passages": total,
        "building": docs.iter().any(|d| d.status == "extracting" || (d.status == "queued" && d.n_graphed < d.n_chunks)) });
    Ok(Json(view))
}

/// FR-K54: the ontology a base's graph is built under, as data a reader can print.
async fn get_ontology(State(s): State<AppState>, Path(id): Path<String>) -> KbResult<Json<Value>> {
    let kb = ready(&s).await?;
    let (o, is_default) = kb.ontology(&id).await?;
    Ok(Json(
        json!({ "types": o.types, "hash": o.hash(), "default": is_default }),
    ))
}

#[derive(Deserialize)]
struct PutOntology {
    #[serde(default)]
    types: Vec<nafs_kb::graph::OntoType>,
}

/// FR-K54: replace it (an empty list restores the default). Passages read from now on use it; a graph already
/// built is rebuilt with `graph/build?fresh=true`.
async fn put_ontology(
    State(s): State<AppState>,
    Path(id): Path<String>,
    Json(b): Json<PutOntology>,
) -> KbResult<Json<Value>> {
    let kb = ready(&s).await?;
    let o = (!b.types.is_empty()).then_some(nafs_kb::graph::Ontology { types: b.types });
    kb.set_ontology(&id, o).await?;
    let (o, is_default) = kb.ontology(&id).await?;
    Ok(Json(
        json!({ "types": o.types, "hash": o.hash(), "default": is_default,
        "note": "applies to passages read from now on; rebuild the graph with fresh=true to re-read the rest" }),
    ))
}

#[derive(Deserialize)]
struct EntityQuery {
    #[serde(default)]
    q: String,
}

/// Entities whose names match the words, or that are closest in meaning when the embedder runs.
async fn graph_entities(
    State(s): State<AppState>,
    Path(id): Path<String>,
    Query(q): Query<EntityQuery>,
) -> KbResult<Json<Value>> {
    let kb = ready(&s).await?;
    kb.store().get_kb(&id).await?;
    let found = kb.store().find_entities(&id, &q.q, None, 20).await?;
    Ok(Json(json!({ "entities": found })))
}

/// One entity: its relations, and the passage that states each.
async fn graph_entity(
    State(s): State<AppState>,
    Path((id, entity)): Path<(String, String)>,
) -> KbResult<Json<Value>> {
    let kb = ready(&s).await?;
    kb.store().get_kb(&id).await?;
    let facts = kb
        .store()
        .facts_for(&id, std::slice::from_ref(&entity), 50)
        .await?;
    let ev: Vec<String> = facts
        .iter()
        .filter_map(|f| {
            f["evidence"]
                .as_array()
                .and_then(|a| a.first())
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .take(20)
        .collect();
    let passages = kb.store().hits_by_id(&ev).await?;
    Ok(Json(json!({ "facts": facts, "passages": passages })))
}

/// Chat retrieval (FR-K20..K25) for the orchestrator: a hybrid search, then the relevance filter. A passage is kept
/// when its meaning similarity reaches `rag_min_similarity`; when the search fell back to words (no vectors, or the
/// embedding server is down), every word match is kept.
pub struct ChatKnowledge {
    kb: Arc<Kb>,
    min_similarity: f64,
    lexical_rescue: f64,
    /// Phase 26: the drop in similarity that cuts the passages after it (0 = off).
    elbow_gap: f64,
    graph_facts: usize,
    memory: Option<Arc<MemoryReader>>,
    /// FR-K24: the order `auto` tries modes in (`[knowledge] auto_order`).
    auto_order: Vec<String>,
}

/// One knowledge base's prepared documents: (stamp of the base when prepared, the preparation, passages read).
type Prepared = (String, Arc<nafs_memorilla::memory::PreparedDocs>, usize);

/// A checkpoint directory and the time its weights were last written.
type Key = (std::path::PathBuf, u128);

/// Memorilla's reader (FR-K23): the module, and per knowledge base its prepared documents, kept until the base's
/// vectors change (count or last update).
pub struct MemoryReader {
    /// `[knowledge] memory_checkpoint`: used for a base that has no module of its own.
    global: Option<(std::path::PathBuf, Arc<nafs_memorilla::MemoryModule>)>,
    files_dir: String,
    embed_dims: usize,
    placeholder: String,
    /// Modules loaded so far, by checkpoint directory and write time.
    loaded: std::sync::Mutex<std::collections::HashMap<Key, Arc<nafs_memorilla::MemoryModule>>>,
    prepared: std::sync::Mutex<std::collections::HashMap<String, Prepared>>,
    /// Held while a base's documents are prepared: the work costs seconds of CPU and the whole base's memory, so two
    /// questions arriving together must not each do it.
    preparing: tokio::sync::Mutex<()>,
    /// Phase 26: whether a module's memory says anything about the question (`true`), by checkpoint and write time,
    /// so the files are read (or an older module probed) once.
    verdicts: std::sync::Mutex<std::collections::HashMap<Key, bool>>,
}

/// Three unrelated questions an older module (saved without `metrics.json`) is asked once: a module whose memories
/// for them are the same vector says nothing about what was asked.
const MEMORY_PROBE_QUESTIONS: [&str; 3] = [
    "Who wrote this, and when was it published?",
    "What is the main result, in numbers?",
    "Which steps does the method take, in order?",
];

/// Why a module is passed over, in words that read inside the chat note "Memory could not be read (...); the
/// passages were used instead."
const MEMORY_UNINFORMATIVE: &str = "the memory of this knowledge base carries no information about the question, because it was trained too briefly";

/// Phase 26: whether a memory module may answer. What the trainer saved beside the weights decides; a module saved
/// before it wrote any is judged by the memories it gives the probe questions; with neither, nothing is known yet.
pub fn memory_usable(
    saved: Option<&nafs_memorilla::CheckpointMetrics>,
    probe: Option<&nafs_memorilla::CollapseStats>,
) -> Option<bool> {
    match (saved, probe) {
        (Some(m), _) => Some(!m.collapsed),
        (None, Some(p)) => Some(!p.collapsed()),
        (None, None) => None,
    }
}

/// A knowledge base's module with its documents prepared: what a question is read against.
struct ReadyMemory {
    reader: Arc<MemoryReader>,
    ckpt: Key,
    module: Arc<nafs_memorilla::MemoryModule>,
    kb_name: String,
    prepared: Arc<nafs_memorilla::memory::PreparedDocs>,
    documents: usize,
    prepare_ms: u64,
}

impl MemoryReader {
    /// The reader for this configuration (with embeddings on); the global checkpoint is loaded now if set.
    pub fn from_config(k: &KnowledgeCfg) -> Option<MemoryReader> {
        if k.embed_url.trim().is_empty() {
            return None;
        }
        let global = (!k.memory_checkpoint.trim().is_empty())
            .then(|| std::path::PathBuf::from(&k.memory_checkpoint))
            .and_then(|p| match nafs_memorilla::MemoryModule::load_cpu(&p) {
                Ok(m) if m.cfg.embedding_dim == k.embed_dims => {
                    tracing::info!(checkpoint = %p.display(), params = m.cfg.parameters(), "memory module loaded");
                    Some((p, Arc::new(m)))
                }
                Ok(m) => {
                    tracing::warn!(want = k.embed_dims, got = m.cfg.embedding_dim, "memory module reads vectors of another width; not used");
                    None
                }
                Err(e) => {
                    tracing::warn!(checkpoint = %p.display(), error = %e, "memory module not loaded");
                    None
                }
            });
        Some(MemoryReader {
            global,
            files_dir: k.files_dir.clone(),
            embed_dims: k.embed_dims,
            placeholder: k.memory_placeholder.clone(),
            loaded: Default::default(),
            prepared: Default::default(),
            preparing: Default::default(),
            verdicts: Default::default(),
        })
    }

    /// How a loaded module is named: its directory and the moment its weights were written. The time is part of the
    /// key because a path can be given new weights (the same run activated again after more training), and both this
    /// cache and the prepared-document cache would otherwise keep serving the old ones.
    fn stamp(p: &std::path::Path) -> Key {
        let t = std::fs::metadata(p.join("weights.safetensors"))
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_millis())
            .unwrap_or(0);
        (p.to_path_buf(), t)
    }

    /// The checkpoint a knowledge base would be answered from, without loading it: its activated run, else the
    /// global one.
    fn checkpoint_for(&self, kb: &str) -> Option<Key> {
        crate::memory_runs::active_checkpoint(&self.files_dir, kb)
            .or_else(|| self.global.as_ref().map(|(p, _)| p.clone()))
            .map(|p| Self::stamp(&p))
    }

    /// The verdict on a checkpoint that needs no model: a cached one, else what the trainer saved beside the weights.
    fn known_verdict(&self, ckpt: &Key) -> Option<bool> {
        if let Some(v) = self
            .verdicts
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(ckpt)
        {
            return Some(*v);
        }
        let saved = nafs_memorilla::CheckpointMetrics::read(&ckpt.0);
        let v = memory_usable(saved.as_ref(), None)?;
        self.remember_verdict(ckpt, v);
        Some(v)
    }

    fn remember_verdict(&self, ckpt: &Key, usable: bool) {
        let mut map = self.verdicts.lock().unwrap_or_else(|e| e.into_inner());
        // Only the current module of a base is ever asked about; old write times of the same path pile up otherwise.
        if map.len() >= 16 {
            map.clear();
        }
        map.insert(ckpt.clone(), usable);
        if !usable {
            tracing::warn!(checkpoint = %ckpt.0.display(), "memory module collapsed (the same memory for every question); not used");
        }
    }

    /// The module for a knowledge base: its activated run, else the global checkpoint. Loading one reads hundreds of
    /// megabytes, so it must not run on a runtime thread; the caller hands this to `spawn_blocking`.
    fn module_for(&self, kb: &str) -> Result<(Key, Arc<nafs_memorilla::MemoryModule>), String> {
        if let Some(p) = crate::memory_runs::active_checkpoint(&self.files_dir, kb) {
            let key = Self::stamp(&p);
            if let Some(m) = self
                .loaded
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .get(&key)
            {
                return Ok((key, m.clone()));
            }
            let m = nafs_memorilla::MemoryModule::load_cpu(&p)?;
            if m.cfg.embedding_dim != self.embed_dims {
                return Err(format!(
                    "the module reads {}-wide vectors; the passages are {}-wide",
                    m.cfg.embedding_dim, self.embed_dims
                ));
            }
            let m = Arc::new(m);
            let mut loaded = self.loaded.lock().unwrap_or_else(|e| e.into_inner());
            // A module is tens of megabytes and only the current one is ever asked for; keeping every module a long
            // run of activations produced would grow the gateway without bound.
            if loaded.len() >= 2 {
                loaded.clear();
            }
            loaded.insert(key.clone(), m.clone());
            return Ok((key, m));
        }
        self.global
            .clone()
            .map(|(p, m)| (Self::stamp(&p), m))
            .ok_or_else(|| {
                "no memory module for this knowledge base: train one on the Knowledge page"
                    .to_string()
            })
    }
}

impl ChatKnowledge {
    /// Phase 26: cut the kept passages at the largest drop in similarity of at least `gap` (`[knowledge]
    /// rag_elbow_gap`).
    pub fn with_elbow(mut self, gap: f64) -> ChatKnowledge {
        self.elbow_gap = gap.max(0.0);
        self
    }

    pub fn new(
        kb: Arc<Kb>,
        min_similarity: f64,
        lexical_rescue: f64,
        graph_facts: usize,
        memory: Option<MemoryReader>,
        auto_order: Vec<String>,
    ) -> ChatKnowledge {
        ChatKnowledge {
            kb,
            min_similarity,
            lexical_rescue,
            elbow_gap: 0.0,
            graph_facts: graph_facts.max(1),
            memory: memory.map(Arc::new),
            auto_order,
        }
    }

    /// Whether this knowledge base has a memory module to read (its own activated run, or the global checkpoint).
    /// Only the checkpoint's presence is checked: loading one takes seconds and `auto` runs on every request.
    fn has_memory(&self, kb: &str) -> bool {
        self.memory.as_ref().is_some_and(|m| {
            m.global.is_some() || crate::memory_runs::active_checkpoint(&m.files_dir, kb).is_some()
        })
    }

    /// Phase 26: whether `auto` may choose memory for this base: it has a module, and the module's memory depends on
    /// the question. A module saved with its metrics is judged from that file (no model is loaded); an older one is
    /// probed once, which loads it and prepares the base's passages, work a memory answer would do anyway. When the
    /// probe cannot run (the embedding server is down, say) the module is offered as before and the read decides.
    async fn has_usable_memory(&self, kb: &str) -> bool {
        if !self.has_memory(kb) {
            return false;
        }
        let Some(reader) = self.memory.as_ref() else {
            return false;
        };
        if let Some(v) = reader
            .checkpoint_for(kb)
            .and_then(|c| reader.known_verdict(&c))
        {
            return v;
        }
        match self.ready_memory(kb).await {
            Ok(r) => self.verdict(&r).await.unwrap_or(true),
            Err(_) => true,
        }
    }

    /// The verdict on a loaded module: cached or saved, else from the probe questions (once per module).
    async fn verdict(&self, r: &ReadyMemory) -> Result<bool, String> {
        if let Some(v) = r.reader.known_verdict(&r.ckpt) {
            return Ok(v);
        }
        let mut memories = Vec::with_capacity(MEMORY_PROBE_QUESTIONS.len());
        for question in MEMORY_PROBE_QUESTIONS {
            let q = self
                .kb
                .embed_plain(question)
                .await
                .map_err(|e| e.to_string())?;
            let (m, p) = (r.module.clone(), r.prepared.clone());
            memories.push(
                tokio::task::spawn_blocking(move || m.read_rows(&p, &q))
                    .await
                    .map_err(|e| e.to_string())??,
            );
        }
        let stats = nafs_memorilla::collapse_stats(&memories);
        tracing::info!(checkpoint = %r.ckpt.0.display(), slot_cosine = stats.slot_cosine, question_cosine = stats.question_cosine,
            memory_norm = stats.mean_norm, "memory module probed (no saved metrics)");
        let usable = memory_usable(None, Some(&stats)).unwrap_or(true);
        r.reader.remember_verdict(&r.ckpt, usable);
        Ok(usable)
    }

    /// The base's module, loaded, and its passages prepared for it (reused until the base or the module changes).
    async fn ready_memory(&self, kb: &str) -> Result<ReadyMemory, String> {
        let reader = self
            .memory
            .as_ref()
            .ok_or("searching by meaning is turned off, so memory cannot be read")?;
        // Loading a module reads hundreds of megabytes from disk, so it never runs on a runtime thread.
        let (ckpt, module) = {
            let (r, base) = (reader.clone(), kb.to_string());
            tokio::task::spawn_blocking(move || r.module_for(&base))
                .await
                .map_err(|e| e.to_string())??
        };
        let base = self
            .kb
            .store()
            .get_kb(kb)
            .await
            .map_err(|e| e.to_string())?;
        let dims = self
            .kb
            .embed_dims()
            .ok_or("searching by meaning is turned off")?;
        // Prepared documents are reused until the base changes (its update time) or its vector count does.
        let n_vec = self
            .kb
            .store()
            .vector_count(kb)
            .await
            .map_err(|e| e.to_string())? as usize;
        // The prepared keys and values come from the module's own weights, so the checkpoint's write time is
        // part of the stamp: re-activating a run after more training reuses the same directory.
        let stamp = format!(
            "{}|{n_vec}|{}|{}",
            base.updated_at,
            ckpt.0.display(),
            ckpt.1
        );
        let cached = reader
            .prepared
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(kb)
            .filter(|c| c.0 == stamp)
            .map(|c| (c.1.clone(), c.2));
        let t0 = std::time::Instant::now();
        let (prepared, documents, prepare_ms) = match cached {
            Some((p, n)) => (p, n, 0),
            None => {
                // One preparation at a time: it costs seconds of CPU and the whole base's vectors in memory, and
                // two questions arriving together would otherwise each do all of it.
                let _one_at_a_time = reader.preparing.lock().await;
                // The guard is dropped at the end of this statement, before any await below.
                let hit = reader
                    .prepared
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .get(kb)
                    .filter(|c| c.0 == stamp)
                    .map(|c| (c.1.clone(), c.2));
                if let Some((p, n)) = hit {
                    (p, n, 0)
                } else {
                    let (rows, n) = self
                        .kb
                        .store()
                        .vectors_of(kb, dims)
                        .await
                        .map_err(|e| e.to_string())?;
                    if n == 0 {
                        return Err("this knowledge base has no meaning index yet".into());
                    }
                    let m = module.clone();
                    let p = tokio::task::spawn_blocking(move || m.prepare_rows(&rows, n))
                        .await
                        .map_err(|e| e.to_string())??;
                    let p = Arc::new(p);
                    let mut map = reader.prepared.lock().unwrap_or_else(|e| e.into_inner());
                    // Only the base being asked about is worth keeping: each entry holds every passage's keys and
                    // values (about 20 KB per passage), so a few bases would be the gateway's whole memory.
                    map.clear();
                    map.insert(kb.to_string(), (stamp, p.clone(), n));
                    drop(map);
                    (p, n, t0.elapsed().as_millis() as u64)
                }
            }
        };
        Ok(ReadyMemory {
            reader: reader.clone(),
            ckpt,
            module,
            kb_name: base.name,
            prepared,
            documents,
            prepare_ms,
        })
    }

    /// Whether this knowledge base has a built graph to search.
    async fn has_graph(&self, kb: &str) -> bool {
        self.kb.graph_enabled()
            && self
                .kb
                .store()
                .graph_stats(kb)
                .await
                .is_ok_and(|v| v["relations"].as_u64().unwrap_or(0) > 0)
    }

    pub fn memory_enabled(&self) -> bool {
        self.memory.is_some()
    }

    /// Whether `[knowledge] memory_checkpoint` named a module that loaded: it serves any base without its own.
    pub fn memory_global(&self) -> bool {
        self.memory.as_ref().is_some_and(|m| m.global.is_some())
    }
}

/// FR-K24: the mode `auto` becomes, given the configured order and what the base can serve. Separate from the store
/// so it can be tested.
pub fn auto_mode(order: &[String], has_memory: bool, has_graph: bool) -> String {
    order
        .iter()
        .find(|m| match m.as_str() {
            "memory" | "hybrid" => has_memory,
            "graph" => has_graph,
            "rag" => true,
            _ => false,
        })
        .cloned()
        .unwrap_or_else(|| "rag".to_string())
}

/// The relevance filter without the elbow cut, as chat used it before Phase 26; the tests below pin its two rules.
///
/// A passage is kept when its meaning similarity reaches `min_similarity`, **or** when its words match strongly
/// (`lexical_rescue`, a BM25 score; 0 turns this off). The second rule exists because the vector arm misses some
/// passages the words find first (measured on the paper corpus: vector search alone left 4 of 20 answering passages
/// outside its top 20, and one of them, ranked first by the words, was dropped by the similarity test and the question
/// went unanswered). Off-topic questions scored at most 8.0 on the same base against at least 10.8 for the answering
/// passages, so the default sits between them.
#[cfg(test)]
fn keep_for_chat(
    hits: Vec<nafs_kb::Hit>,
    mode: nafs_kb::SearchMode,
    min_similarity: f64,
    lexical_rescue: f64,
    k: usize,
) -> Vec<nafs_kb::Hit> {
    nafs_kb::keep_for_chat(hits, mode, min_similarity, lexical_rescue, k, 0.0)
}

impl nafs_core::knowledge::Knowledge for ChatKnowledge {
    fn embed<'a>(&'a self, text: &'a str) -> nafs_core::knowledge::BoxFuture<'a, Option<Vec<f32>>> {
        Box::pin(async move { self.kb.embed_plain(text).await.ok() })
    }

    fn forget(&self, kb: &str) {
        if let Some(m) = &self.memory {
            // The prepared documents are vectors read from the base's passages: gone with the base.
            m.prepared
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .remove(kb);
        }
    }

    /// FR-K24: `auto` takes the first mode of `[knowledge] auto_order` this base can actually serve.
    fn resolve_mode<'a>(
        &'a self,
        kb: &'a str,
        asked: &'a str,
    ) -> nafs_core::knowledge::BoxFuture<'a, String> {
        Box::pin(async move {
            if asked != "auto" {
                return asked.to_string();
            }
            // Both checks are cheap (file tests and one count query; a module saved without metrics is probed the
            // first time only), and `graph` is only asked for when the order reaches it.
            // Phase 26: a module whose memory does not depend on the question does not count as memory.
            let memory = self.has_usable_memory(kb).await;
            let graph = self.auto_order.iter().any(|m| m == "graph") && self.has_graph(kb).await;
            auto_mode(&self.auto_order, memory, graph)
        })
    }

    fn retrieve<'a>(
        &'a self,
        kb: &'a str,
        query: &'a str,
        k: usize,
        mode: &'a str,
    ) -> nafs_core::knowledge::BoxFuture<'a, Result<nafs_core::knowledge::Retrieved, String>> {
        Box::pin(async move {
            let base = self
                .kb
                .store()
                .get_kb(kb)
                .await
                .map_err(|e| e.to_string())?;
            // Graph mode (FR-K22): facts around the question's entities, and passages that mention them first. A
            // passage the graph brought in (a fact's evidence) has no similarity score and is kept.
            let (facts, r, graph) = if mode == "graph" && self.kb.graph_enabled() {
                let (facts, r) = self
                    .kb
                    .graph_search(kb, query, (k * 2).max(8))
                    .await
                    .map_err(|e| e.to_string())?;
                (facts, r, true)
            } else {
                // Search deeper than k so the filter has room, then keep the best k that pass.
                let r = self
                    .kb
                    .search(kb, query, (k * 2).max(8), nafs_kb::SearchMode::Hybrid)
                    .await
                    .map_err(|e| e.to_string())?;
                (Vec::new(), r, false)
            };
            let hits = if graph {
                r.results
                    .into_iter()
                    .filter(|h| h.vector_score.is_none_or(|v| v >= self.min_similarity))
                    .take(k)
                    .collect()
            } else {
                nafs_kb::keep_for_chat(
                    r.results,
                    r.mode,
                    self.min_similarity,
                    self.lexical_rescue,
                    k,
                    self.elbow_gap,
                )
            };
            let facts: Vec<nafs_core::knowledge::Fact> = facts
                .iter()
                .take(self.graph_facts)
                .filter_map(|f| {
                    Some(nafs_core::knowledge::Fact {
                        source: f["source"].as_str()?.to_string(),
                        predicate: f["predicate"].as_str()?.to_string(),
                        target: f["target"].as_str()?.to_string(),
                        weight: f["weight"].as_u64().unwrap_or(1),
                        quote: f["quote"].as_str().unwrap_or("").to_string(),
                        hop: f["hop"].as_u64().unwrap_or(1) as u8,
                        via: f["via"].as_str().unwrap_or("").to_string(),
                    })
                })
                .collect();
            let passages = hits
                .into_iter()
                .map(|h| nafs_core::knowledge::Passage {
                    chunk_id: h.chunk_id,
                    doc_id: h.doc_id,
                    doc_title: h.doc_title,
                    heading: h.heading,
                    page: h.page,
                    text: h.text,
                    score: h.score,
                    lexical_score: h.lexical_score,
                    vector_score: h.vector_score,
                })
                .collect();
            // FR-K59: the arms that ran, and a note for each that did not, so a reply never implies a search it
            // did not make. The search names its own fallback; the graph arm is named only when it ran.
            let mut arms: Vec<String> = match r.mode {
                nafs_kb::SearchMode::Lexical => vec!["lexical".into()],
                nafs_kb::SearchMode::Vector => vec!["vector".into()],
                nafs_kb::SearchMode::Hybrid => vec!["lexical".into(), "vector".into()],
            };
            let mut notes: Vec<String> = Vec::new();
            if let Some(why) = &r.fallback {
                notes.push(format!("vector: {why}"));
            }
            if graph {
                if facts.is_empty() {
                    notes.push("graph: no entity of the question is in the graph".into());
                } else {
                    arms.push("graph".into());
                }
            } else if mode == "graph" {
                notes.push("graph: the knowledge graph is turned off on this server".into());
            }
            Ok(nafs_core::knowledge::Retrieved {
                kb_name: base.name,
                passages,
                mode: if graph {
                    "graph".into()
                } else {
                    format!("{:?}", r.mode).to_lowercase()
                },
                fallback: r.fallback,
                embed_ms: r.embed_ms,
                search_ms: r.took_ms,
                facts,
                arms,
                notes,
            })
        })
    }

    fn memory<'a>(
        &'a self,
        kb: &'a str,
        query: &'a str,
    ) -> nafs_core::knowledge::BoxFuture<'a, Result<nafs_core::knowledge::MemoryReadout, String>>
    {
        Box::pin(async move {
            // A module known to say nothing about the question is not even loaded.
            if let Some(reader) = &self.memory {
                if reader
                    .checkpoint_for(kb)
                    .and_then(|c| reader.known_verdict(&c))
                    == Some(false)
                {
                    return Err(MEMORY_UNINFORMATIVE.to_string());
                }
            }
            // Phase 26: a module whose memory is the same whatever is asked (older ones are probed here, once) is
            // not used; the caller then answers from passages and says so.
            let ready = self.ready_memory(kb).await?;
            if !self.verdict(&ready).await? {
                return Err(MEMORY_UNINFORMATIVE.to_string());
            }
            let ReadyMemory {
                reader,
                module,
                kb_name,
                prepared,
                documents,
                prepare_ms,
                ..
            } = ready;
            let q = self
                .kb
                .embed_plain(query)
                .await
                .map_err(|e| e.to_string())?;
            let t1 = std::time::Instant::now();
            let m = module.clone();
            let vectors = tokio::task::spawn_blocking(move || m.read_rows(&prepared, &q))
                .await
                .map_err(|e| e.to_string())??;
            Ok(nafs_core::knowledge::MemoryReadout {
                kb_name,
                vectors,
                placeholder: reader.placeholder.clone(),
                system: nafs_memorilla::train::SYSTEM_PROMPT.to_string(),
                documents,
                prepare_ms,
                read_ms: t1.elapsed().as_millis() as u64,
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nafs_kb::{Hit, SearchMode};

    #[test]
    fn auto_takes_the_first_mode_the_base_can_serve() {
        let order = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let o = order(&["hybrid", "graph", "rag"]);
        assert_eq!(auto_mode(&o, true, true), "hybrid");
        assert_eq!(auto_mode(&o, false, true), "graph");
        assert_eq!(auto_mode(&o, false, false), "rag");
        // A base with a module but no graph skips straight past graph.
        assert_eq!(
            auto_mode(&order(&["graph", "memory", "rag"]), true, false),
            "memory"
        );
        // Nothing configured, or nothing the base can serve: passages, which always work.
        assert_eq!(auto_mode(&[], false, false), "rag");
        assert_eq!(auto_mode(&order(&["memory"]), false, false), "rag");
        assert_eq!(auto_mode(&order(&["nonsense", "rag"]), true, true), "rag");
    }

    /// Phase 26: a module whose memory is the same for every question is not used, whether the trainer said so or
    /// (for a module saved before it did) the probe questions show it.
    #[test]
    fn a_module_that_says_nothing_about_the_question_is_not_used() {
        use nafs_memorilla::{collapse_stats, CheckpointMetrics};
        let saved = |collapsed| CheckpointMetrics {
            collapsed,
            ..Default::default()
        };
        // The collapsed Phase 20 module: one vector of norm 63 at all 16 slots for all three probe questions.
        let same: Vec<Vec<Vec<f32>>> = (0..3).map(|_| vec![vec![63.0f32 / 8.0; 64]; 16]).collect();
        let dead = collapse_stats(&same);
        let differing: Vec<Vec<Vec<f32>>> = (0..3)
            .map(|q| {
                (0..16)
                    .map(|s| {
                        (0..64)
                            .map(|i| (((q * 7919 + s * 104729 + i * 31) % 97) as f32 / 97.0) - 0.5)
                            .collect()
                    })
                    .collect()
            })
            .collect();
        let alive = collapse_stats(&differing);
        assert!(dead.collapsed() && !alive.collapsed());

        assert_eq!(memory_usable(Some(&saved(true)), None), Some(false));
        assert_eq!(memory_usable(Some(&saved(false)), None), Some(true));
        // What the trainer saved outranks a probe.
        assert_eq!(memory_usable(Some(&saved(false)), Some(&dead)), Some(true));
        assert_eq!(memory_usable(None, Some(&dead)), Some(false));
        assert_eq!(memory_usable(None, Some(&alive)), Some(true));
        assert_eq!(memory_usable(None, None), None);

        // `auto` then falls through to the next mode the base can serve.
        let order: Vec<String> = ["hybrid", "graph", "rag"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let usable = memory_usable(Some(&saved(true)), None).unwrap_or(true);
        assert_eq!(auto_mode(&order, usable, false), "rag");
        // The note chat shows is plain language.
        assert!(
            !MEMORY_UNINFORMATIVE.contains("collaps") && !MEMORY_UNINFORMATIVE.contains("cosine")
        );
    }

    /// The verdict is read from the checkpoint's own metrics file, once, and kept per checkpoint and write time.
    #[test]
    fn the_saved_verdict_is_read_without_loading_the_module() {
        let dir = std::env::temp_dir().join(format!("nafs-verdict-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("weights.safetensors"), b"not read").unwrap();
        let reader = MemoryReader {
            global: None,
            files_dir: String::new(),
            embed_dims: 8,
            placeholder: String::new(),
            loaded: Default::default(),
            prepared: Default::default(),
            preparing: Default::default(),
            verdicts: Default::default(),
        };
        let key = MemoryReader::stamp(&dir);
        // No metrics file (a module saved before Phase 26): nothing known until it is probed.
        assert_eq!(reader.known_verdict(&key), None);
        nafs_memorilla::CheckpointMetrics {
            collapsed: true,
            ..Default::default()
        }
        .write(&dir)
        .unwrap();
        assert_eq!(reader.known_verdict(&key), Some(false));
        // Cached: the file is not needed again.
        std::fs::remove_file(dir.join("metrics.json")).unwrap();
        assert_eq!(reader.known_verdict(&key), Some(false));
        // New weights under the same path are another module.
        let later = (key.0.clone(), key.1 + 1);
        assert_eq!(reader.known_verdict(&later), None);
        reader.remember_verdict(&later, true);
        assert_eq!(reader.known_verdict(&later), Some(true));
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn hit(id: &str, v: Option<f64>) -> Hit {
        Hit {
            chunk_id: id.into(),
            doc_id: "d".into(),
            doc_title: "t".into(),
            ord: 0,
            heading: String::new(),
            page: None,
            text: "x".into(),
            score: 0.03,
            rank: 1,
            lexical_score: Some(2.0),
            vector_score: v,
        }
    }

    #[test]
    fn chat_keeps_only_passages_close_in_meaning() {
        let hits = vec![
            hit("a", Some(0.62)),
            hit("b", None),
            hit("c", Some(0.2)),
            hit("d", Some(0.5)),
        ];
        let kept: Vec<String> = keep_for_chat(hits.clone(), SearchMode::Hybrid, 0.35, 10.0, 4)
            .into_iter()
            .map(|h| h.chunk_id)
            .collect();
        assert_eq!(kept, ["a", "d"]);
        assert_eq!(
            keep_for_chat(hits.clone(), SearchMode::Hybrid, 0.35, 10.0, 1).len(),
            1
        );
        // Words only (no vectors): every match stays.
        assert_eq!(
            keep_for_chat(hits, SearchMode::Lexical, 0.35, 10.0, 4).len(),
            4
        );
    }

    /// A strong word match is kept although the vector arm missed it or scored it low; a weak one is not.
    #[test]
    fn chat_keeps_a_strong_word_match_the_meaning_search_missed() {
        let mut strong = hit("s", None);
        strong.lexical_score = Some(15.0);
        let mut low_meaning = hit("m", Some(0.30));
        low_meaning.lexical_score = Some(27.0);
        let weak = hit("w", None);
        let hits = vec![weak.clone(), strong, low_meaning, weak];
        let kept: Vec<String> = keep_for_chat(hits.clone(), SearchMode::Hybrid, 0.45, 10.0, 8)
            .into_iter()
            .map(|h| h.chunk_id)
            .collect();
        assert_eq!(kept, ["s", "m"]);
        assert!(
            keep_for_chat(hits, SearchMode::Hybrid, 0.45, 0.0, 8).is_empty(),
            "0 turns the rescue off"
        );
    }
}
