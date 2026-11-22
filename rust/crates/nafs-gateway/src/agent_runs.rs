//! Code-agent runs a person can watch (Phase 35).
//!
//! `POST /v1/harness/code` answers when the run is over, up to fifteen minutes later, with nothing in between and no
//! way to stop it. These routes start a run and return at once; the run's steps are stored as they end, so the app
//! can show them, the run can be stopped, and an earlier run can be opened again.
//!
//! - `POST /v1/harness/runs` (the body of `/v1/harness/code`) → `202 {run_id}`; `409` while another run is going
//!   (the models serve one such run at a time).
//! - `GET /v1/harness/runs` → the newest runs, without their files.
//! - `GET /v1/harness/runs/{id}` → the run: status, steps so far, and the result when it has ended.
//! - `POST /v1/harness/runs/{id}/stop` → stops it after the step in progress (the model call is ended).

use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use nafs_types::{CodeAgentRequest, CodeAgentStep, NafsError};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use uuid::Uuid;

use crate::api::{ApiResult, AppState};

const KIND: &str = "agent";

/// Runs that are going, by id, each with the flag that stops it.
fn live() -> &'static Mutex<HashMap<Uuid, Arc<AtomicBool>>> {
    static LIVE: OnceLock<Mutex<HashMap<Uuid, Arc<AtomicBool>>>> = OnceLock::new();
    LIVE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Takes a run off the list of runs that are going when its task ends, however it ends: a panic inside the run must
/// not leave the list occupied, or every later run would be refused until the service restarts.
struct Going(Uuid);
impl Drop for Going {
    fn drop(&mut self) {
        live()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&self.0);
    }
}

/// How many agent runs are kept; older ones are deleted when a new one starts.
const KEEP_RUNS: i64 = 100;

fn save(s: &AppState, id: Uuid, status: &str, doc: &Value) {
    if let Err(e) = s.store.save_run(KIND, id, status, doc) {
        tracing::warn!(error = %e, run = %id, "the agent run could not be stored");
    }
}

pub async fn start(
    State(s): State<AppState>,
    Json(req): Json<CodeAgentRequest>,
) -> ApiResult<(StatusCode, Json<Value>)> {
    if !s.cfg.harness.code_agent_enabled {
        return Err(NafsError::BadRequest(
            "the code agent is turned off ([harness] code_agent_enabled)".into(),
        )
        .into());
    }
    s.orch
        .validate_code_agent(&req)
        .map_err(NafsError::BadRequest)?;
    let id = Uuid::new_v4();
    let stop = Arc::new(AtomicBool::new(false));
    {
        let mut l = live().lock().unwrap_or_else(|e| e.into_inner());
        if !l.is_empty() {
            return Err(NafsError::Conflict(
                "a run is already going: wait for it to end, or stop it".into(),
            )
            .into());
        }
        l.insert(id, stop.clone());
    }
    let tests = req.verification.tests.as_ref().map_or(0, Vec::len);
    let mut doc = json!({
        "id": id, "kind": KIND, "status": "running", "created_at": chrono::Utc::now().to_rfc3339(),
        "task": req.task, "file_names": req.files.keys().collect::<Vec<_>>(), "tests": tests,
        "model": req.model.clone().unwrap_or_else(|| s.cfg.harness.model.clone()),
        "max_steps": req.max_steps.unwrap_or(s.cfg.harness.max_steps),
        "request": { "task": req.task, "files": req.files, "verification": req.verification },
        "steps": [],
    });
    if let Err(e) = s.store.prune_runs(KIND, KEEP_RUNS) {
        tracing::warn!(error = %e, "old agent runs could not be removed");
    }
    save(&s, id, "running", &doc);
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<CodeAgentStep>();
    let s2 = s.clone();
    tokio::spawn(async move {
        let going = Going(id);
        let run = s2.orch.run_code_agent_watched(req, Some(tx), Some(stop));
        tokio::pin!(run);
        // Steps are stored as they end; the run's own result closes the record.
        let result = loop {
            tokio::select! {
                step = rx.recv() => {
                    if let Some(step) = step {
                        if let Some(a) = doc["steps"].as_array_mut() {
                            a.push(serde_json::to_value(&step).unwrap_or(Value::Null));
                        }
                        save(&s2, id, "running", &doc);
                    }
                }
                r = &mut run => break r,
            }
        };
        drop(going);
        let status = match result {
            Ok(reply) => {
                doc["status"] = json!(reply.status);
                doc["steps"] = serde_json::to_value(&reply.steps).unwrap_or(Value::Null);
                doc["result"] = serde_json::to_value(&reply).unwrap_or(Value::Null);
                if let Some(r) = doc["result"].as_object_mut() {
                    r.remove("steps");
                }
                "completed"
            }
            Err(e) => {
                doc["status"] = json!("failed");
                doc["error"] = json!(e);
                "failed"
            }
        };
        doc["ended_at"] = json!(chrono::Utc::now().to_rfc3339());
        save(&s2, id, status, &doc);
    });
    Ok((StatusCode::ACCEPTED, Json(json!({ "run_id": id }))))
}

pub async fn list(State(s): State<AppState>) -> ApiResult<Json<Value>> {
    let going: Vec<Uuid> = live()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .keys()
        .copied()
        .collect();
    let runs: Vec<Value> = s
        .store
        .list_runs(KIND, 30)?
        .into_iter()
        .map(|mut r| {
            // A run recorded as going whose task is no longer here ended with the service.
            let id = r["id"].as_str().and_then(|x| Uuid::parse_str(x).ok());
            if r["status"] == "running" && !id.is_some_and(|i| going.contains(&i)) {
                r["status"] = json!("interrupted");
            }
            json!({
                "id": r["id"], "status": r["status"], "created_at": r["created_at"], "task": r["task"],
                "model": r["model"], "steps": r["steps"].as_array().map_or(0, Vec::len),
                "passed": r["result"]["passed"], "wall_ms": r["result"]["wall_ms"],
            })
        })
        .collect();
    Ok(Json(json!({ "runs": runs, "running": !going.is_empty() })))
}

pub async fn get(State(s): State<AppState>, Path(id): Path<Uuid>) -> ApiResult<Json<Value>> {
    let mut r = s.store.get_run(id)?;
    if r["kind"] != KIND {
        return Err(NafsError::NotFound(format!("run {id}")).into());
    }
    let going = live()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .contains_key(&id);
    if r["status"] == "running" && !going {
        r["status"] = json!("interrupted");
    }
    Ok(Json(r))
}

pub async fn stop(Path(id): Path<Uuid>) -> ApiResult<Json<Value>> {
    let l = live().lock().unwrap_or_else(|e| e.into_inner());
    match l.get(&id) {
        Some(flag) => {
            flag.store(true, Ordering::Relaxed);
            Ok(Json(json!({ "stopping": true })))
        }
        None => Err(NafsError::NotFound(format!("run {id} is not going")).into()),
    }
}
