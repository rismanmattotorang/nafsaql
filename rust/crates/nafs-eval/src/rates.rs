//! Verifier quality measurement (FR-032, ER-003, P2-T03): mutation testing of the exact verifiers on
//! the shipped datasets' reference solutions, and the judge's (r, phi) on a labelled set (FR-034).
use crate::load_tasks;
use nafs_core::orchestrator::Orchestrator;
use nafs_types::Rates;
use nafs_verify::mutation::{self, clopper_pearson};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RatesRequest {
    /// Tasks per dataset used to build cases (math, code, schema).
    #[serde(default = "d_items")]
    pub items_per_dataset: usize,
    #[serde(default)]
    pub include_judge: bool,
    #[serde(default = "d_conc")]
    pub concurrency: usize,
    /// Code datasets whose canonical solutions and tests build the code cases. The EvalPlus variants
    /// (humaneval_plus, mbpp_plus) carry far stronger tests, so they measure a lower phi.
    #[serde(default = "d_code")]
    pub code_datasets: Vec<String>,
    /// Skip the exact verifiers (measure the judge only).
    #[serde(default)]
    pub judge_only: bool,
}
fn d_code() -> Vec<String> {
    vec!["mbpp".into(), "humaneval".into()]
}
fn d_items() -> usize {
    60
}
fn d_conc() -> usize {
    3
}

fn read_jsonl(path: &str) -> Vec<Value> {
    std::fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect()
}

pub async fn measure(
    orch: Arc<Orchestrator>,
    req: RatesRequest,
    cancel: Arc<AtomicBool>,
    mut progress: impl FnMut(&Value) + Send,
) -> anyhow::Result<Value> {
    let dir = orch.cfg.eval.datasets_dir.clone();
    let n = req.items_per_dataset.clamp(1, 2000);
    // Math: numeric references from GSM8K.
    let refs: Vec<(String, f64)> = load_tasks(&dir, "gsm8k")?
        .into_iter()
        .filter_map(|t| {
            t.verification
                .reference_answer
                .as_deref()
                .and_then(|r| r.replace(',', "").trim().parse::<f64>().ok())
                .map(|v| (t.id, v))
        })
        .take(n)
        .collect();
    let mut cases = if req.judge_only {
        vec![]
    } else {
        mutation::math_cases(&refs)
    };
    // Code: canonical solutions joined to their tasks' tests.
    let mut code = vec![];
    let per = (n / req.code_datasets.len().max(1)).max(1);
    for ds in req
        .code_datasets
        .iter()
        .filter(|d| !req.judge_only && d.chars().all(|c| c.is_alphanumeric() || c == '_'))
    {
        let sols: HashMap<String, String> = read_jsonl(&format!("{dir}/_solutions/{ds}.jsonl"))
            .into_iter()
            .filter_map(|v| {
                Some((
                    v["id"].as_str()?.to_string(),
                    v["solution"].as_str()?.to_string(),
                ))
            })
            .collect();
        for t in load_tasks(&dir, ds)?
            .into_iter()
            .filter(|t| sols.contains_key(&t.id))
            .take(per)
        {
            let s = sols[&t.id].clone();
            code.push((t.id, t.verification, s));
        }
    }
    cases.extend(mutation::code_cases(&code));
    // Schema: valid instances.
    let valid: HashMap<String, Value> = read_jsonl(&format!("{dir}/_solutions/schema_tasks.jsonl"))
        .into_iter()
        .filter_map(|v| Some((v["id"].as_str()?.to_string(), v["solution"].clone())))
        .collect();
    let schema: Vec<(String, Value, Value)> = load_tasks(&dir, "schema_tasks")?
        .into_iter()
        .filter_map(|t| {
            Some((
                t.id.clone(),
                t.verification.json_schema.clone()?,
                valid.get(&t.id)?.clone(),
            ))
        })
        .collect();
    if !req.judge_only {
        cases.extend(mutation::schema_cases(&schema));
    }
    let total = cases.len();
    let started = chrono::Utc::now();
    let snap = |done: usize, stage: &str| json!({"status": "running", "stage": stage, "progress": {"done": done, "total": total}, "started_at": started, "request": req});
    progress(&snap(0, "exact verifiers"));
    let mut last = std::time::Instant::now();
    let mut out = mutation::run(&orch.hub, cases, req.concurrency.clamp(1, 8), |d, _| {
        if last.elapsed().as_secs() >= 2 {
            last = std::time::Instant::now();
            progress(&snap(d, "exact verifiers"));
        }
    })
    .await;
    if cancel.load(Ordering::Relaxed) {
        out["status"] = json!("cancelled");
        return Ok(out);
    }
    // Start from the published rates so a partial measurement (judge only, other datasets) updates
    // only what it measured.
    let mut rates: BTreeMap<String, Rates> =
        orch.hub.rates.read().unwrap().clone().into_iter().collect();
    let measured: BTreeMap<String, Rates> =
        serde_json::from_value(out["rates"].clone()).unwrap_or_default();
    rates.extend(measured);
    // Judge (FR-034): r on acceptable answers, phi on unacceptable ones.
    if req.include_judge {
        let set = read_jsonl(&format!("{dir}/_judge/judge_labelled.jsonl"));
        let mut rows = vec![];
        for (i, c) in set.iter().enumerate() {
            if cancel.load(Ordering::Relaxed) {
                break;
            }
            progress(
                &json!({"status": "running", "stage": "judge", "progress": {"done": i, "total": set.len()}, "started_at": started, "request": req}),
            );
            let (q, a, ok) = (
                c["question"].as_str().unwrap_or(""),
                c["answer"].as_str().unwrap_or(""),
                c["acceptable"].as_bool().unwrap_or(false),
            );
            match orch.judge(q, a, None).await {
                Ok(v) => rows.push(json!({"id": c["id"], "category": c["category"], "acceptable": ok, "passed": v.passed, "score": v.score, "details": v.details})),
                Err(e) => rows.push(json!({"id": c["id"], "category": c["category"], "acceptable": ok, "error": e.to_string()})),
            }
        }
        let scored: Vec<&Value> = rows.iter().filter(|r| r.get("passed").is_some()).collect();
        let nc = scored.iter().filter(|r| r["acceptable"] == true).count() as u64;
        let ac = scored
            .iter()
            .filter(|r| r["acceptable"] == true && r["passed"] == true)
            .count() as u64;
        let nw = scored.iter().filter(|r| r["acceptable"] == false).count() as u64;
        let aw = scored
            .iter()
            .filter(|r| r["acceptable"] == false && r["passed"] == true)
            .count() as u64;
        let (_, phi_hi) = clopper_pearson(aw, nw);
        let r = if nc > 0 { ac as f64 / nc as f64 } else { 0.0 };
        let phi = if nw > 0 { aw as f64 / nw as f64 } else { 0.0 };
        out["verifiers"]["judge"] = json!({"n_correct": nc, "accepted_correct": ac, "n_wrong": nw, "accepted_wrong": aw, "r": r, "phi": phi,
            "r_ci95": clopper_pearson(ac, nc), "phi_ci95": clopper_pearson(aw, nw), "errors": rows.len() - scored.len(), "rows": rows,
            "note": "Soft verifier: replies it accepts are labelled at most 'consensus' (Thm 1). The labelled set is small; the spec's release bar needs a larger set."});
        rates.insert(
            "judge".into(),
            Rates {
                r,
                phi,
                n_mutants: nw as u32,
                n_correct: nc as u32,
                phi_upper95: Some(phi_hi),
                measured_at: Some(chrono::Utc::now().to_rfc3339()),
            },
        );
    }
    // Publish (FR-032): attach to every verdict from now on and persist for restarts.
    let map: HashMap<String, Rates> = rates.clone().into_iter().collect();
    orch.hub.set_rates(map);
    let path = &orch.cfg.verifiers.rates_file;
    if let Some(p) = std::path::Path::new(path).parent() {
        let _ = std::fs::create_dir_all(p);
    }
    std::fs::write(path, serde_json::to_string_pretty(&rates)?)?;
    out["rates"] = json!(rates);
    out["status"] = json!("done");
    out["started_at"] = json!(started);
    out["progress"] = json!({"done": total, "total": total});
    out["request"] = json!(req);
    out["er003"] = json!(out["verifiers"].as_object().map(|m| m.iter().map(|(k, v)| (k.clone(), json!({
        "mutants": v["n_wrong"], "correct": v["n_correct"], "eligible": v["n_wrong"].as_u64().unwrap_or(0) >= 200 && v["n_correct"].as_u64().unwrap_or(0) >= 200,
        "phi_upper95": v["phi_ci95"][1], "target": if k == "json_schema" { "0 observed" } else if k == "judge" { "measured" } else { "<= 0.01" }}))).collect::<serde_json::Map<_, _>>()));
    Ok(out)
}
