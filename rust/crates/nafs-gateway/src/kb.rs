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
    }))
}

/// The routes, with authentication and the same-origin guard applied here because they sit outside the main
/// router's body limit.
pub fn routes(state: AppState) -> Router<AppState> {
    let upload_limit = state.cfg.knowledge.max_upload_mb.max(1)
        * 1024
        * 1024
        * state.cfg.knowledge.max_files_per_request.max(1);
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
        "embedder": kb.embedder_status().await.map(|(model, reachable)| json!({ "model": model, "reachable": reachable, "url": c.embed_url })),
        "limits": { "max_upload_mb": c.max_upload_mb, "max_files_per_request": c.max_files_per_request },
        "kinds": ["pdf", "docx", "pptx", "tex", "html", "md", "txt"],
    }))
}

async fn list_kbs(State(s): State<AppState>) -> KbResult<Json<Value>> {
    let kb = ready(&s).await?;
    Ok(Json(json!({ "bases": kb.store().list_kbs().await? })))
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
    let (documents, chunks) = kb.delete_kb(&id).await?;
    Ok(Json(
        json!({ "deleted": true, "documents": documents, "chunks": chunks }),
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
        let bytes = match field.bytes().await {
            Ok(b) => b,
            Err(e) => {
                refused.push(json!({ "filename": filename, "reason": format!("the file could not be received: {e}") }));
                continue;
            }
        };
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
    Ok(Json(json!({ "deleted": true, "chunks": chunks })))
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
    kb.store().get_kb(&id).await?;
    let r = kb
        .search(&id, &b.query, b.k.unwrap_or(kb.config().search_k), mode)
        .await?;
    Ok(Json(json!(r)))
}

/// Give vectors to the documents of a knowledge base that lack them (stored before the embedder ran, or while it was
/// down).
async fn reindex(
    State(s): State<AppState>,
    Path(id): Path<String>,
) -> KbResult<(StatusCode, Json<Value>)> {
    let kb = ready(&s).await?;
    kb.store().get_kb(&id).await?;
    let queued = kb.reindex(Some(&id)).await?;
    Ok((StatusCode::ACCEPTED, Json(json!({ "queued": queued }))))
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
    view["enabled"] = json!(kb.graph_enabled());
    let docs = kb.store().list_documents(&id).await?;
    let (read, total): (u64, u64) = docs
        .iter()
        .fold((0, 0), |a, d| (a.0 + d.n_graphed, a.1 + d.n_chunks));
    view["progress"] = json!({ "passages_read": read, "passages": total,
        "building": docs.iter().any(|d| d.status == "extracting" || (d.status == "queued" && d.n_graphed < d.n_chunks)) });
    Ok(Json(view))
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
    graph_facts: usize,
}

impl ChatKnowledge {
    pub fn new(kb: Arc<Kb>, min_similarity: f64, graph_facts: usize) -> ChatKnowledge {
        ChatKnowledge {
            kb,
            min_similarity,
            graph_facts: graph_facts.max(1),
        }
    }
}

/// The relevance filter, separate so it can be tested without a store.
pub fn keep_for_chat(
    hits: Vec<nafs_kb::Hit>,
    mode: nafs_kb::SearchMode,
    min_similarity: f64,
    k: usize,
) -> Vec<nafs_kb::Hit> {
    hits.into_iter()
        .filter(|h| {
            mode == nafs_kb::SearchMode::Lexical
                || h.vector_score.is_some_and(|v| v >= min_similarity)
        })
        .take(k)
        .collect()
}

impl nafs_core::knowledge::Knowledge for ChatKnowledge {
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
                keep_for_chat(r.results, r.mode, self.min_similarity, k)
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
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nafs_kb::{Hit, SearchMode};

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
        let kept: Vec<String> = keep_for_chat(hits.clone(), SearchMode::Hybrid, 0.35, 4)
            .into_iter()
            .map(|h| h.chunk_id)
            .collect();
        assert_eq!(kept, ["a", "d"]);
        assert_eq!(
            keep_for_chat(hits.clone(), SearchMode::Hybrid, 0.35, 1).len(),
            1
        );
        // Words only (no vectors): every match stays.
        assert_eq!(keep_for_chat(hits, SearchMode::Lexical, 0.35, 4).len(), 4);
    }
}
