//! Memorilla training runs per knowledge base (Phase 6, FR-K42): build the dataset in the gateway, train in a separate
//! `memorilla-train` process (a CPU job of hours that must not share the gateway's threads), follow its progress file,
//! stop it, and make a run's best checkpoint the base's active module for chat's `memory` mode.
//!
//! On disk, under `<files_dir>/<kb>/memorilla/`: `runs/<id>/run.json` (what was asked, the process id), `runs/<id>/data/`
//! (the dataset), `runs/<id>/out/` (progress.jsonl, epoch-NN/, best/), and `active` (the checkpoint directory in use).

use std::path::{Path, PathBuf};

use axum::extract::{Path as UrlPath, State};
use axum::http::StatusCode;
use axum::Json;
use nafs_kb::KbError;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::api::AppState;
use crate::kb::KbApiError;

type R<T> = Result<T, KbApiError>;

fn base_dir(s: &AppState, kb: &str) -> PathBuf {
    let safe: String = kb.chars().filter(|c| c.is_ascii_alphanumeric()).collect();
    PathBuf::from(&s.cfg.knowledge.files_dir)
        .join(safe)
        .join("memorilla")
}

/// The checkpoint a knowledge base answers `memory` questions with: its own active run, if one was activated.
pub fn active_checkpoint(files_dir: &str, kb: &str) -> Option<PathBuf> {
    let safe: String = kb.chars().filter(|c| c.is_ascii_alphanumeric()).collect();
    let p = std::fs::read_to_string(
        PathBuf::from(files_dir)
            .join(safe)
            .join("memorilla")
            .join("active"),
    )
    .ok()?;
    let p = PathBuf::from(p.trim());
    p.join("weights.safetensors").exists().then_some(p)
}

fn alive(pid: u64) -> bool {
    // Signal 0 checks that the process exists without touching it.
    pid > 0 && unsafe { libc::kill(pid as i32, 0) } == 0
}

/// A run as the API reports it: what was asked, its state, and the progress so far.
fn describe(dir: &Path) -> Option<Value> {
    let mut run: Value = serde_json::from_slice(&std::fs::read(dir.join("run.json")).ok()?).ok()?;
    let lines: Vec<Value> = std::fs::read_to_string(dir.join("out").join("progress.jsonl"))
        .unwrap_or_default()
        .lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect();
    let last_of = |ev: &str| lines.iter().rev().find(|l| l["event"] == ev).cloned();
    let steps: Vec<Value> = lines
        .iter()
        .filter(|l| l["event"] == "step")
        .cloned()
        .collect();
    let epochs: Vec<Value> = lines
        .iter()
        .filter(|l| l["event"] == "epoch")
        .cloned()
        .collect();
    let pid = run["pid"].as_u64().unwrap_or(0);
    let building = run["state"] == "building";
    let state = if last_of("done").is_some() {
        if last_of("stopped").is_some() {
            "stopped"
        } else {
            "done"
        }
    } else if run["error"].is_string() {
        "failed"
    } else if alive(pid) {
        if steps.is_empty() {
            "preparing"
        } else {
            "training"
        }
    } else if building && alive(run["owner"].as_u64().unwrap_or(0)) {
        // Only while the gateway that builds the dataset is still running: a run left "building" by a restart is over.
        "building"
    } else {
        "failed"
    };
    let start = last_of("start");
    let sec_per_example: Vec<f64> = steps
        .iter()
        .filter_map(|s| s["seconds_per_example"].as_f64())
        .collect();
    run["state"] = json!(state);
    run["progress"] = json!({
        "step": steps.last().and_then(|s| s["step"].as_u64()).unwrap_or(0),
        "steps": start.as_ref().and_then(|s| s["steps"].as_u64()),
        "train_examples": start.as_ref().and_then(|s| s["train"].as_u64()),
        "validation_examples": start.as_ref().and_then(|s| s["validation"].as_u64()),
        "start_val_loss": start.as_ref().and_then(|s| s["val_loss"].as_f64()),
        "loss": steps.iter().map(|s| json!([s["step"], s["loss"]])).collect::<Vec<_>>(),
        "val_loss": epochs.iter().map(|e| json!([e["epoch"], e["val_loss"]])).collect::<Vec<_>>(),
        "seconds_per_example": if sec_per_example.is_empty() { Value::Null } else { json!(sec_per_example.iter().sum::<f64>() / sec_per_example.len() as f64) },
        "final_val_loss": last_of("done").and_then(|d| d["val_loss"].as_f64()),
    });
    run["has_checkpoint"] = json!(dir
        .join("out")
        .join("best")
        .join("weights.safetensors")
        .exists());
    if run["error"].is_null() && state == "failed" && building {
        run["error"] = json!("the server stopped while the questions were being prepared");
    }
    if run["error"].is_null() && state == "failed" {
        run["error"] = json!(std::fs::read_to_string(dir.join("trainer.log"))
            .ok()
            .and_then(|l| l
                .lines()
                .rev()
                .find(|x| x.contains("failed"))
                .map(str::to_string)));
    }
    Some(run)
}

pub async fn list(State(s): State<AppState>, UrlPath(kb): UrlPath<String>) -> R<Json<Value>> {
    let k = s
        .kb
        .as_ref()
        .ok_or_else(|| KbApiError::from(KbError::Unavailable("Knowledge is turned off".into())))?;
    k.ready().await?;
    k.store().get_kb(&kb).await?;
    let dir = base_dir(&s, &kb);
    let mut runs: Vec<Value> = std::fs::read_dir(dir.join("runs"))
        .map(|r| {
            r.filter_map(|e| e.ok())
                .filter_map(|e| describe(&e.path()))
                .collect()
        })
        .unwrap_or_default();
    runs.sort_by(|a, b| b["created_at"].as_str().cmp(&a["created_at"].as_str()));
    let active = active_checkpoint(&s.cfg.knowledge.files_dir, &kb);
    Ok(Json(
        json!({ "runs": runs, "active": active, "trainer": trainer_path(&s).exists() }),
    ))
}

fn trainer_path(s: &AppState) -> PathBuf {
    PathBuf::from(&s.cfg.knowledge.memory_trainer)
}

#[derive(Deserialize)]
pub struct NewRun {
    /// "train" (from scratch) or "finetune" (from the active checkpoint).
    #[serde(default = "d_kind")]
    kind: String,
    #[serde(default = "d_epochs")]
    epochs: usize,
    #[serde(default)]
    max_examples: usize,
    lr: Option<f64>,
}
fn d_kind() -> String {
    "train".into()
}
fn d_epochs() -> usize {
    3
}

/// FR-K42: build the dataset (reconstruction and graph claims) and start the trainer. One run at a time per base.
pub async fn start(
    State(s): State<AppState>,
    UrlPath(kb): UrlPath<String>,
    Json(b): Json<NewRun>,
) -> R<(StatusCode, Json<Value>)> {
    let k = s
        .kb
        .clone()
        .ok_or_else(|| KbApiError::from(KbError::Unavailable("Knowledge is turned off".into())))?;
    k.ready().await?;
    k.store().get_kb(&kb).await?;
    if !trainer_path(&s).exists() {
        return Err(KbError::Unavailable(format!(
            "the trainer is not built ({}): cargo build --release -p nafs-memorilla",
            s.cfg.knowledge.memory_trainer
        ))
        .into());
    }
    if b.kind != "train" && b.kind != "finetune" {
        return Err(KbError::BadInput("kind is \"train\" or \"finetune\"".into()).into());
    }
    let init = if b.kind == "finetune" {
        Some(
            active_checkpoint(&s.cfg.knowledge.files_dir, &kb).ok_or_else(|| {
                KbError::BadInput("fine-tuning needs an active module; train one first".into())
            })?,
        )
    } else {
        None
    };
    let dir = base_dir(&s, &kb);
    if let Ok(rd) = std::fs::read_dir(dir.join("runs")) {
        for e in rd.filter_map(|e| e.ok()) {
            if let Some(r) = describe(&e.path()) {
                if matches!(
                    r["state"].as_str(),
                    Some("building" | "preparing" | "training")
                ) {
                    return Err(KbError::Conflict(
                        "a training run is already in progress for this knowledge base".into(),
                    )
                    .into());
                }
            }
        }
    }
    let id = chrono::Utc::now().format("%Y%m%d-%H%M%S").to_string();
    let run_dir = dir.join("runs").join(&id);
    std::fs::create_dir_all(&run_dir).map_err(|e| KbError::Io(e.to_string()))?;
    let lr =
        b.lr.unwrap_or(if b.kind == "finetune" { 1e-4 } else { 3e-4 });
    let mut run = json!({ "id": id, "kb": kb, "kind": b.kind, "epochs": b.epochs.clamp(1, 20), "max_examples": b.max_examples, "lr": lr,
        "init": init, "created_at": chrono::Utc::now().to_rfc3339(), "state": "building", "pid": 0,
        // The gateway building the dataset: without it a run interrupted by a restart would stay "building" and block
        // every later run for this base.
        "owner": std::process::id() });
    std::fs::write(run_dir.join("run.json"), run.to_string())
        .map_err(|e| KbError::Io(e.to_string()))?;
    let trainer = trainer_path(&s);
    let decoder = s.cfg.knowledge.memory_decoder.clone();
    let gguf = s.cfg.knowledge.memory_gguf.clone();
    let mut started = run.clone();
    started["state"] = json!("building");
    tokio::spawn(async move {
        let data = run_dir.join("data");
        let res = async {
            let info = nafs_kb::dataset::build(&k, &kb, Default::default()).await?;
            nafs_kb::dataset::write(&k, &info, &data).await?;
            Ok::<_, KbError>(info.rows.len())
        }
        .await;
        match res {
            Err(e) => run["error"] = json!(format!("the dataset could not be built: {e}")),
            Ok(rows) => {
                run["rows"] = json!(rows);
                let log = std::fs::File::create(run_dir.join("trainer.log")).ok();
                let mut cmd = std::process::Command::new(&trainer);
                cmd.args([
                    "--data",
                    &data.to_string_lossy(),
                    "--out",
                    &run_dir.join("out").to_string_lossy(),
                ])
                .args(["--decoder", &decoder, "--gguf", &gguf])
                .args([
                    "--epochs",
                    &run["epochs"].to_string(),
                    "--lr",
                    &lr.to_string(),
                ])
                .args(["--max-examples", &run["max_examples"].to_string()]);
                if let Some(i) = run["init"].as_str() {
                    cmd.args(["--init", i]);
                }
                if let Some(l) = log {
                    cmd.stderr(l);
                }
                match cmd.stdout(std::process::Stdio::null()).spawn() {
                    Ok(child) => {
                        run["pid"] = json!(child.id());
                        run["state"] = json!("training");
                        // Reap it when it ends, so it does not stay a zombie that `kill -0` still finds.
                        std::thread::spawn(move || {
                            let mut c = child;
                            let _ = c.wait();
                        });
                    }
                    Err(e) => run["error"] = json!(format!("the trainer could not start: {e}")),
                }
            }
        }
        let _ = std::fs::write(run_dir.join("run.json"), run.to_string());
    });
    Ok((StatusCode::ACCEPTED, Json(started)))
}

/// End a run after its current step (it saves what it has as `stopped/`).
pub async fn stop(
    State(s): State<AppState>,
    UrlPath((kb, run)): UrlPath<(String, String)>,
) -> R<Json<Value>> {
    let dir = base_dir(&s, &kb)
        .join("runs")
        .join(run.replace(['/', '.'], ""));
    if !dir.join("run.json").exists() {
        return Err(KbError::NotFound("training run").into());
    }
    std::fs::create_dir_all(dir.join("out"))
        .and_then(|_| std::fs::write(dir.join("out").join("stop"), b""))
        .map_err(|e| KbError::Io(e.to_string()))?;
    Ok(Json(describe(&dir).unwrap_or(Value::Null)))
}

/// Make a run's best checkpoint the knowledge base's module for `memory` mode.
pub async fn activate(
    State(s): State<AppState>,
    UrlPath((kb, run)): UrlPath<(String, String)>,
) -> R<Json<Value>> {
    let base = base_dir(&s, &kb);
    let dir = base.join("runs").join(run.replace(['/', '.'], ""));
    let out = dir.join("out");
    let ckpt = ["best", "stopped"]
        .iter()
        .map(|d| out.join(d))
        .find(|p| p.join("weights.safetensors").exists())
        .ok_or_else(|| KbError::BadInput("this run has no checkpoint yet".into()))?;
    let abs = std::fs::canonicalize(&ckpt).map_err(|e| KbError::Io(e.to_string()))?;
    std::fs::write(base.join("active"), abs.to_string_lossy().as_bytes())
        .map_err(|e| KbError::Io(e.to_string()))?;
    Ok(Json(json!({ "active": abs })))
}
