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

/// Whether the trainer of this run is still running. The reaper writes `out/exit` the moment it ends, and that is
/// trusted over `kill(pid, 0)`: once a child is reaped the system is free to give its pid to anything else, and a
/// reused pid would leave the run "training" for ever and refuse every later run for the base.
fn running(dir: &Path, pid: u64) -> bool {
    !dir.join("out").join("exit").exists() && alive(pid)
}

/// How the trainer ended, as the reaper recorded it (`out/exit`): "ok", "exit 1", "signal 9".
fn exit_status(dir: &Path) -> Option<String> {
    std::fs::read_to_string(dir.join("out").join("exit"))
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// What a run's page says when its module learnt nothing about the question.
const UNUSABLE_NOTE: &str = "This run's memory is the same whatever is asked, so it would not help answer questions. Chat will use passages instead; train again, for longer or on more passages.";

/// A run as the API reports it: what was asked, its state, and the progress so far.
fn describe(dir: &Path) -> Option<Value> {
    let id = dir.file_name()?.to_string_lossy().to_string();
    let raw = std::fs::read(dir.join("run.json")).ok()?;
    // A run whose record cannot be read still exists on disk and may still have a trainer: say so rather than
    // letting it disappear from the list.
    let Ok(mut run) = serde_json::from_slice::<Value>(&raw) else {
        return Some(
            json!({ "id": id, "state": "failed", "created_at": "", "kind": "train", "has_checkpoint": false,
                "error": "this run's record could not be read" }),
        );
    };
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
    } else if running(dir, pid) {
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
        "question_cosine": epochs.iter().filter(|e| e["question_cosine"].is_number()).map(|e| json!([e["epoch"], e["question_cosine"]])).collect::<Vec<_>>(),
        "seconds_per_example": if sec_per_example.is_empty() { Value::Null } else { json!(sec_per_example.iter().sum::<f64>() / sec_per_example.len() as f64) },
        "final_val_loss": last_of("done").and_then(|d| d["val_loss"].as_f64()),
    });
    run["has_checkpoint"] = json!(dir
        .join("out")
        .join("best")
        .join("weights.safetensors")
        .exists());
    // Phase 26: what the trainer measured about the checkpoint that would be activated. `memory_usable` false means
    // its memory is the same whatever is asked, and chat will answer from passages instead.
    if let Some(m) = ["best", "stopped", "last"]
        .iter()
        .map(|d| dir.join("out").join(d))
        .find(|p| p.join("weights.safetensors").exists())
        .and_then(|p| nafs_memorilla::CheckpointMetrics::read(&p))
    {
        run["memory_usable"] = json!(!m.collapsed);
        run["checkpoint"] = json!(m);
        if m.collapsed {
            run["note"] = json!(UNUSABLE_NOTE);
        }
    }
    if run["error"].is_null() && state == "failed" && building {
        run["error"] = json!("the server stopped while the questions were being prepared");
    }
    if run["error"].is_null() && state == "failed" {
        // The trainer's own message if it printed one, else its last words, else how the system ended it: a run
        // killed for memory writes nothing at all, and "Failed" with no reason tells the operator nothing.
        let log = std::fs::read_to_string(dir.join("trainer.log")).unwrap_or_default();
        let said = log
            .lines()
            .rev()
            .find(|x| x.contains("failed") || x.contains("panicked"))
            .or_else(|| {
                log.lines()
                    .rev()
                    .find(|x| !x.trim().is_empty() && !x.trim_start().starts_with('{'))
            })
            .map(str::to_string);
        let how = exit_status(dir)
            .or_else(|| {
                // No exit marker and no owner: the gateway and the trainer went down together, which on this host
                // means the machine restarted. Seen for real on 2026-11-08, three steps into a run.
                (pid > 0 && !alive(run["owner"].as_u64().unwrap_or(0)))
                    .then(|| "no end".to_string())
            })
            .map(|s| match s.as_str() {
                "no end" => {
                    "the trainer stopped when the server did (the machine restarted?); start a new run".to_string()
                }
                "signal 9" => {
                    "the trainer was killed, most likely for memory (it needs about 8-11 GB free)"
                        .to_string()
                }
                other => format!("the trainer ended with {other}"),
            });
        run["error"] = json!(match (said, how) {
            (Some(a), Some(b)) => Some(format!("{b}: {a}")),
            (a, b) => a.or(b),
        });
    }
    Some(run)
}

/// The knowledge base, or the error that says why it cannot be reached.
async fn base(s: &AppState, kb: &str) -> R<()> {
    let k = s
        .kb
        .as_ref()
        .ok_or_else(|| KbApiError::from(KbError::Unavailable("Knowledge is turned off".into())))?;
    k.ready().await?;
    k.store().get_kb(kb).await?;
    Ok(())
}

pub async fn list(State(s): State<AppState>, UrlPath(kb): UrlPath<String>) -> R<Json<Value>> {
    base(&s, &kb).await?;
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

#[cfg(unix)]
fn signal_of(st: &std::process::ExitStatus) -> Option<i32> {
    std::os::unix::process::ExitStatusExt::signal(st)
}
#[cfg(not(unix))]
fn signal_of(_: &std::process::ExitStatus) -> Option<i32> {
    None
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
    /// Examples per optimiser step.
    accumulation: Option<usize>,
}
fn d_kind() -> String {
    "train".into()
}
/// The reference's from-scratch recipe (Phase 26): 5 epochs at 1e-4 with 16 examples per optimiser step. The first
/// recipe here (3 epochs at 3e-4, 4 examples a step) trained a module that gave every question the same memory.
fn d_epochs() -> usize {
    nafs_memorilla::train::DEFAULT_EPOCHS
}

/// Run directories, in every base or in one, whose trainer is being prepared or is going.
fn active_runs(files_dir: &str, only: Option<&str>) -> Vec<PathBuf> {
    let root = PathBuf::from(files_dir);
    let bases: Vec<PathBuf> = match only {
        Some(kb) => {
            let safe: String = kb.chars().filter(|c| c.is_ascii_alphanumeric()).collect();
            if safe.is_empty() {
                vec![]
            } else {
                vec![root.join(safe)]
            }
        }
        None => std::fs::read_dir(&root)
            .map(|r| r.filter_map(|e| e.ok()).map(|e| e.path()).collect())
            .unwrap_or_default(),
    };
    let mut out = vec![];
    for b in bases {
        let Ok(rd) = std::fs::read_dir(b.join("memorilla").join("runs")) else {
            continue;
        };
        for e in rd.filter_map(|e| e.ok()) {
            if let Some(r) = describe(&e.path()) {
                if matches!(
                    r["state"].as_str(),
                    Some("building" | "preparing" | "training")
                ) {
                    out.push(e.path());
                }
            }
        }
    }
    out
}

/// SR-K12: stop every run of a knowledge base that is going to be deleted. The run is asked to stop and its trainer
/// process, if it is one of ours (the command line is checked, since a pid can be reused), is ended. Returns how many
/// runs were stopped.
pub fn stop_trainers(files_dir: &str, kb: &str) -> usize {
    let mut n = 0;
    for dir in active_runs(files_dir, Some(kb)) {
        let _ = std::fs::create_dir_all(dir.join("out"))
            .and_then(|_| std::fs::write(dir.join("out").join("stop"), b""));
        let pid = std::fs::read(dir.join("run.json"))
            .ok()
            .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
            .and_then(|r| r["pid"].as_u64())
            .unwrap_or(0);
        if pid > 0 && running(&dir, pid) && is_trainer(pid) {
            #[cfg(unix)]
            unsafe {
                libc::kill(pid as i32, libc::SIGTERM);
            }
        }
        n += 1;
    }
    n
}

/// Whether the process is our trainer (not some other program that was given a dead trainer's pid).
fn is_trainer(pid: u64) -> bool {
    std::process::Command::new("ps")
        .args(["-p", &pid.to_string(), "-o", "command="])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).contains("memorilla-train"))
        .unwrap_or(false)
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
    // One run at a time per base. The check and the run's directory are taken together under one lock and with a
    // directory that must not already exist: the dataset takes minutes to build, and two requests arriving inside
    // that window would otherwise both pass the check and both start a trainer, writing into one directory.
    static STARTING: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let (id, run_dir) = {
        let _one_at_a_time = STARTING.lock().unwrap_or_else(|e| e.into_inner());
        // Each run holds 8 to 11 GB while it trains: only so many at once across all bases.
        let allowed = s.cfg.knowledge.max_concurrent_trainers.max(1);
        let going = active_runs(&s.cfg.knowledge.files_dir, None).len();
        if going >= allowed {
            return Err(KbError::Conflict(format!(
                "{going} training run(s) are already going on this server (the limit is {allowed}); wait for one to finish or stop it"
            ))
            .into());
        }
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
        std::fs::create_dir_all(dir.join("runs")).map_err(|e| KbError::Io(e.to_string()))?;
        let now = chrono::Utc::now();
        let mut taken = None;
        for n in 0..1000 {
            let id = match n {
                0 => now.format("%Y%m%d-%H%M%S").to_string(),
                _ => format!("{}-{n}", now.format("%Y%m%d-%H%M%S")),
            };
            let p = dir.join("runs").join(&id);
            if std::fs::create_dir(&p).is_ok() {
                taken = Some((id, p));
                break;
            }
        }
        taken.ok_or_else(|| KbError::Io("no free run directory".into()))?
    };
    let lr = b.lr.unwrap_or(nafs_memorilla::train::DEFAULT_LR);
    let accumulation = b
        .accumulation
        .unwrap_or(nafs_memorilla::train::DEFAULT_ACCUMULATION)
        .clamp(1, 64);
    let mut run = json!({ "id": id, "kb": kb, "kind": b.kind, "epochs": b.epochs.clamp(1, 20), "max_examples": b.max_examples, "lr": lr,
        "accumulation": accumulation,
        "init": init, "created_at": chrono::Utc::now().to_rfc3339(), "state": "building", "pid": 0,
        // The gateway building the dataset: without it a run interrupted by a restart would stay "building" and block
        // every later run for this base.
        "owner": std::process::id() });
    std::fs::write(run_dir.join("run.json"), run.to_string())
        .map_err(|e| KbError::Io(e.to_string()))?;
    let trainer = trainer_path(&s);
    let decoder = s.cfg.knowledge.memory_decoder.clone();
    let gguf = s.cfg.knowledge.memory_gguf.clone();
    // The trainer must put the memory tokens where the server will: the same placeholder token (FR-K42).
    let placeholder = s.cfg.knowledge.memory_placeholder.clone();
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
            Ok(rows) if run_dir.join("out").join("stop").exists() => {
                // Stopped while the questions were being prepared: the trainer is never started, so nothing spends
                // hours on a run the operator has already called off.
                run["rows"] = json!(rows);
                run["error"] = json!("the run was stopped before training began");
            }
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
                .args(["--placeholder", &placeholder])
                .args([
                    "--epochs",
                    &run["epochs"].to_string(),
                    "--lr",
                    &lr.to_string(),
                    "--accumulation",
                    &accumulation.to_string(),
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
                        // Written before anything else: a gateway killed in the next instant would otherwise leave a
                        // trainer running for hours that no run.json names, so nothing could report or stop it.
                        let _ = std::fs::write(run_dir.join("run.json"), run.to_string());
                        // Reap it when it ends, so it does not stay a zombie that `kill -0` still finds, and record
                        // how it ended: a run killed for memory says nothing on stderr.
                        let out = run_dir.join("out");
                        std::thread::spawn(move || {
                            let mut c = child;
                            let how = match c.wait() {
                                Ok(st) if st.success() => "ok".to_string(),
                                Ok(st) => match (st.code(), signal_of(&st)) {
                                    (Some(c), _) => format!("exit {c}"),
                                    (None, Some(s)) => format!("signal {s}"),
                                    _ => "an unknown end".to_string(),
                                },
                                Err(e) => format!("an unknown end ({e})"),
                            };
                            let _ = std::fs::create_dir_all(&out);
                            let _ = std::fs::write(out.join("exit"), how);
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
    base(&s, &kb).await?;
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

/// Make a run's best checkpoint the knowledge base's module for `memory` mode. The checkpoint is **copied** out of
/// the run: a run that is still training rewrites `out/best/` at every improving epoch, and the serving gateway
/// keys its loaded module on the path, so pointing at the live directory would both change the answers under the
/// reader and have it read a file as it is written.
pub async fn activate(
    State(s): State<AppState>,
    UrlPath((kb, run)): UrlPath<(String, String)>,
) -> R<Json<Value>> {
    base(&s, &kb).await?;
    let id = run.replace(['/', '.'], "");
    let home = base_dir(&s, &kb);
    let out = home.join("runs").join(&id).join("out");
    let ckpt = ["best", "stopped", "last"]
        .iter()
        .map(|d| out.join(d))
        .find(|p| p.join("weights.safetensors").exists())
        .ok_or_else(|| KbError::BadInput("this run has no checkpoint yet".into()))?;
    let io = |e: std::io::Error| KbError::Io(e.to_string());
    let dest = home.join("modules").join(&id);
    let tmp = home.join("modules").join(format!("{id}.part"));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).map_err(io)?;
    for f in ["config.json", "weights.safetensors"] {
        std::fs::copy(ckpt.join(f), tmp.join(f)).map_err(io)?;
    }
    // What the trainer measured goes with the weights: the reader decides from it whether the module may answer.
    let metrics = nafs_memorilla::CheckpointMetrics::read(&ckpt);
    if let Some(m) = &metrics {
        m.write(&tmp).map_err(KbError::Io)?;
    }
    let _ = std::fs::remove_dir_all(&dest);
    std::fs::rename(&tmp, &dest).map_err(io)?;
    let abs = std::fs::canonicalize(&dest).map_err(io)?;
    std::fs::write(home.join("active"), abs.to_string_lossy().as_bytes()).map_err(io)?;
    let mut out = json!({ "active": abs, "from": ckpt });
    if metrics.is_some_and(|m| m.collapsed) {
        out["memory_usable"] = json!(false);
        out["note"] = json!(UNUSABLE_NOTE);
    }
    Ok(Json(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_files_dir() -> PathBuf {
        let d = std::env::temp_dir().join(format!("nafs-runs-{}-{}", std::process::id(), line!()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    fn make_run(files: &Path, kb: &str, id: &str, state: &str) -> PathBuf {
        let dir = files.join(kb).join("memorilla").join("runs").join(id);
        std::fs::create_dir_all(dir.join("out")).unwrap();
        // "building" counts as going while the owner (this process) is alive; the pid is not a real trainer.
        let run = json!({ "id": id, "kind": "train", "state": state, "pid": 0, "owner": std::process::id() });
        std::fs::write(dir.join("run.json"), run.to_string()).unwrap();
        dir
    }

    /// SR-K12: the runs of a base about to be deleted are found and asked to stop; other bases' runs are not touched.
    #[test]
    fn the_runs_of_a_deleted_base_are_stopped_and_only_those() {
        let files = temp_files_dir();
        let mine = make_run(&files, "basea", "r1", "building");
        let other = make_run(&files, "baseb", "r2", "building");
        let done = make_run(&files, "basea", "r3", "failed");
        std::fs::write(
            done.join("run.json"),
            json!({"id": "r3", "state": "failed", "pid": 0, "owner": 0, "error": "x"}).to_string(),
        )
        .unwrap();
        assert_eq!(
            active_runs(files.to_str().unwrap(), None).len(),
            2,
            "both going runs are counted"
        );
        assert_eq!(active_runs(files.to_str().unwrap(), Some("basea")).len(), 1);
        assert_eq!(stop_trainers(files.to_str().unwrap(), "basea"), 1);
        assert!(
            mine.join("out").join("stop").exists(),
            "the run is asked to stop"
        );
        assert!(
            !other.join("out").join("stop").exists(),
            "another base's run is left alone"
        );
        assert!(!done.join("out").join("stop").exists());
        // An id that is not a key reaches nothing.
        assert_eq!(stop_trainers(files.to_str().unwrap(), "../.."), 0);
        assert_eq!(active_runs(files.to_str().unwrap(), Some("")).len(), 0);
        let _ = std::fs::remove_dir_all(&files);
    }
}
