//! `/v1/arc*`: the reasoning cache (Phase 38). List and read entries, delete or re-label them (the human review CPU-ARC
//! describes), queue or run a build, try a question against the cache alone, and compile it into one Python module.

use crate::api::{ApiResult, AppState};
use axum::extract::{Path, State};
use axum::Json;
use nafs_core::arc::serve::ArcTry;
use nafs_core::arc::{BuildJob, Status};
use nafs_types::NafsError;
use serde::Deserialize;
use serde_json::{json, Value};
use std::time::Instant;

fn store(s: &AppState) -> Result<std::sync::Arc<nafs_core::arc::ArcStore>, NafsError> {
    s.orch.arc.clone().ok_or_else(|| {
        NafsError::BadRequest("the reasoning cache is off ([arc] enabled = false)".into())
    })
}

pub async fn list(State(s): State<AppState>) -> ApiResult<Json<Value>> {
    let Some(st) = s.orch.arc.clone() else {
        return Ok(Json(json!({"enabled": false, "entries": [], "queue": []})));
    };
    let entries = st.list();
    let count = |x: Status| entries.iter().filter(|e| e.status == x).count();
    let hits: u64 = entries.iter().map(|e| e.stats.hits).sum();
    let served: u64 = entries.iter().map(|e| e.stats.served).sum();
    Ok(Json(json!({
        "enabled": true,
        "serve": s.cfg.arc.serve,
        "build": s.cfg.arc.build,
        "trust_after": s.cfg.arc.trust_after,
        "domains": s.cfg.arc.domains,
        "counts": {"trusted": count(Status::Trusted), "candidate": count(Status::Candidate), "quarantined": count(Status::Quarantined)},
        "hits": hits,
        "served": served,
        "functions": entries.iter().map(|e| e.functions.len()).sum::<usize>(),
        "entries": entries.iter().map(|e| e.summary()).collect::<Vec<_>>(),
        "queue": st.queued().iter().map(|j| json!({"question": j.question, "domain": j.domain, "attempts": j.attempts})).collect::<Vec<_>>(),
    })))
}

pub async fn get(State(s): State<AppState>, Path(id): Path<String>) -> ApiResult<Json<Value>> {
    let st = store(&s)?;
    let e = st
        .get(&id)
        .ok_or_else(|| NafsError::NotFound(format!("cache entry {id}")))?;
    let mut v = serde_json::to_value(&e).unwrap_or(Value::Null);
    v["floor"] = json!(nafs_core::arc::entry_floor(&e, &s.cfg.arc));
    Ok(Json(v))
}

pub async fn delete(State(s): State<AppState>, Path(id): Path<String>) -> ApiResult<Json<Value>> {
    let st = store(&s)?;
    if !st.remove(&id) {
        return Err(NafsError::NotFound(format!("cache entry {id}")).into());
    }
    Ok(Json(json!({"deleted": id})))
}

#[derive(Deserialize)]
pub struct StatusBody {
    status: Status,
}

pub async fn set_status(
    State(s): State<AppState>,
    Path(id): Path<String>,
    Json(b): Json<StatusBody>,
) -> ApiResult<Json<Value>> {
    let st = store(&s)?;
    let e = st
        .set_status(&id, b.status)
        .ok_or_else(|| NafsError::NotFound(format!("cache entry {id}")))?;
    Ok(Json(e.summary()))
}

#[derive(Deserialize)]
pub struct BuildBody {
    #[serde(default)]
    question: Option<String>,
    /// The answer to build from (System 2's reply text).
    #[serde(default)]
    answer: Option<String>,
    #[serde(default)]
    reasoning: Option<String>,
    /// A reference answer the code must reproduce (wins over the answer's own).
    #[serde(default)]
    expected: Option<String>,
    #[serde(default)]
    domain: Option<String>,
    /// Build from a stored reply instead (its question, answer and reasoning).
    #[serde(default)]
    reply_id: Option<String>,
    /// Build now and return the result, instead of queueing for an idle moment.
    #[serde(default)]
    wait: bool,
}

pub async fn build(State(s): State<AppState>, Json(b): Json<BuildBody>) -> ApiResult<Json<Value>> {
    let st = store(&s)?;
    let job = if let Some(id) = &b.reply_id {
        let r = s.store.reply_with_question(id)?;
        BuildJob {
            question: r["question"].as_str().unwrap_or_default().to_string(),
            answer: r["answer"].as_str().unwrap_or_default().to_string(),
            reasoning: r["reasoning"].as_str().unwrap_or_default().to_string(),
            expected: b.expected.clone(),
            domain: r["domain"].as_str().unwrap_or("math").to_string(),
            source_reply: Some(id.clone()),
            attempts: 0,
        }
    } else {
        let q = b.question.clone().filter(|q| !q.trim().is_empty());
        let a = b.answer.clone().or_else(|| b.expected.clone());
        match (q, a) {
            (Some(q), Some(a)) => BuildJob {
                question: q,
                answer: a,
                reasoning: b.reasoning.clone().unwrap_or_default(),
                expected: b.expected.clone(),
                domain: b.domain.clone().unwrap_or_else(|| "math".into()),
                source_reply: None,
                attempts: 0,
            },
            _ => {
                return Err(NafsError::BadRequest(
                    "give a question with an answer (or expected), or a reply_id".into(),
                )
                .into())
            }
        }
    };
    if job.question.trim().is_empty() || job.answer.trim().is_empty() {
        return Err(NafsError::BadRequest(
            "the reply has no question or answer to build from".into(),
        )
        .into());
    }
    if b.wait {
        let rep = s
            .orch
            .arc_build_now(job)
            .await
            .ok_or_else(|| NafsError::BadRequest("nothing was built".into()))?;
        let entry = rep.entry.as_ref().and_then(|id| st.get(id));
        return Ok(Json(json!({"report": rep, "entry": entry})));
    }
    let queued = st.enqueue(job, s.cfg.arc.queue_cap);
    Ok(Json(json!({"queued": queued, "queue": st.queued().len()})))
}

#[derive(Deserialize)]
pub struct TryBody {
    question: String,
    /// One meaning-tier floor for every entry, in place of their learned ones (for measuring thresholds).
    #[serde(default)]
    floor: Option<f32>,
}

/// The cache alone: no fallback to the models, trust ignored (quarantined entries are still skipped).
pub async fn try_question(
    State(s): State<AppState>,
    Json(b): Json<TryBody>,
) -> ApiResult<Json<Value>> {
    store(&s)?;
    let t = Instant::now();
    let r = s
        .orch
        .arc_try_at(&b.question, false, None, b.floor, true)
        .await;
    let ms = t.elapsed().as_millis() as u64;
    Ok(Json(match r {
        ArcTry::Miss(why) => json!({"hit": false, "why": why, "ms": ms}),
        ArcTry::Serve(h, o) | ArcTry::Shadow(h, o) => json!({
            "hit": true, "entry": h.entry.id, "status": h.entry.status, "tier": h.tier, "similarity": h.similarity,
            "params": h.params, "answer": o.answer, "summary": o.summary, "narration": o.narration,
            "run_ms": o.ms, "ms": ms,
        }),
    }))
}

#[derive(Deserialize, Default)]
pub struct CompileBody {
    /// Candidates too, not only trusted entries.
    #[serde(default)]
    all: bool,
}

pub async fn compile(
    State(s): State<AppState>,
    body: Option<Json<CompileBody>>,
) -> ApiResult<Json<Value>> {
    let st = store(&s)?;
    let all = body.map(|b| b.all).unwrap_or(false);
    let (module, n, skipped) = st.compile(all).await.map_err(NafsError::BadRequest)?;
    let path = std::path::Path::new(&s.cfg.arc.dir).join("arc_cache.py");
    let _ = std::fs::create_dir_all(&s.cfg.arc.dir);
    std::fs::write(&path, &module)
        .map_err(|e| NafsError::Internal(format!("could not write the module: {e}")))?;
    Ok(Json(
        json!({"path": path.display().to_string(), "entries": n, "skipped": skipped, "bytes": module.len(),
        "usage": "python3 arc_cache.py \"<question>\"  (answers questions that match a stored template, with no model)"}),
    ))
}
