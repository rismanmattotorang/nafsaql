//! Roadmap decision gates G0-G4 evaluated from stored evidence (FR-053, P4-T03). Each criterion is
//! "pass", "fail" or "pending" (not enough data to decide); a gate passes only when every criterion
//! passes. Pooled statistics combine items across stored evaluation runs, pairing items only within
//! the run that produced them.
use nafs_eval::{gated, ItemResult};
use nafs_verify::mutation::clopper_pearson;
use serde_json::{json, Value};
use std::collections::BTreeMap;

fn crit(id: &str, text: &str, status: &str, evidence: String) -> Value {
    json!({"id": id, "criterion": text, "status": status, "evidence": evidence})
}
fn gate(id: &str, title: &str, week: u32, criteria: Vec<Value>) -> Value {
    let st = |s: &str| criteria.iter().any(|c| c["status"] == s);
    let status = if st("fail") {
        "fail"
    } else if st("pending") {
        "pending"
    } else {
        "pass"
    };
    json!({"gate": id, "title": title, "week": week, "status": status, "criteria": criteria})
}

pub struct Evidence {
    pub eval_runs: Vec<Value>,
    pub bench_runs: Vec<Value>,
    pub acceptance: Option<Value>,
    pub h7: Option<Value>,
    pub registry: Option<Value>,
    pub calibration_promoted: bool,
    pub calibration_n: usize,
    pub calibration_ece: f32,
    /// Phase 5/6 evidence files (None when absent).
    pub security: Option<Value>,
    pub deploy: Option<Value>,
    pub profile_c: Option<Value>,
    pub scholar: Option<Value>,
}

pub fn evaluate(e: &Evidence) -> Value {
    // Items of all stored runs, grouped by run so pairs never cross runs.
    let runs: Vec<Vec<ItemResult>> = e
        .eval_runs
        .iter()
        .filter_map(|r| serde_json::from_value::<Vec<ItemResult>>(r["items"].clone()).ok())
        .collect();
    let all = || runs.iter().flatten();
    let n_fast = all()
        .filter(|i| i.system == "fast_only" && i.correct.is_some())
        .count();
    let n_slow = all()
        .filter(|i| i.system == "slow_only" && i.correct.is_some())
        .count();

    // ---- G0
    let h7 = match &e.h7 {
        Some(v) if v["pass"] == true => crit(
            "H7",
            "Qwen3.5 decode >= 70% of dense Qwen3 (profile decision)",
            "pass",
            v["summary"].as_str().unwrap_or("measured").to_string(),
        ),
        Some(v) => crit(
            "H7",
            "Qwen3.5 decode >= 70% of dense Qwen3 (profile decision)",
            "fail",
            format!("{}", v["summary"]),
        ),
        None => crit(
            "H7",
            "Qwen3.5 decode >= 70% of dense Qwen3 (profile decision)",
            "pending",
            "no stored measurement".into(),
        ),
    };
    let base_status = if n_fast >= 4600 && n_slow >= 4600 {
        "pass"
    } else {
        "pending"
    };
    let g0 = gate(
        "G0",
        "Phase-0 exit and profile decision",
        3,
        vec![
            h7,
            crit(
                "P0-T04",
                "Fast-only and slow-only baselines on >= 4,600 tasks (interim release at 2,000)",
                base_status,
                format!("{n_fast} fast-only and {n_slow} slow-only scored items stored"),
            ),
        ],
    );

    // ---- G1
    let acc = e
        .acceptance
        .as_ref()
        .and_then(|a| a["results"].as_array().cloned())
        .unwrap_or_default();
    let passed = acc.iter().filter(|r| r["pass"] == true).count();
    let mvp = if acc.is_empty() {
        "pending"
    } else if passed == acc.len() && acc.len() >= 9 {
        "pass"
    } else {
        "fail"
    };
    let detail = |id: &str| {
        acc.iter()
            .find(|r| r["id"] == id)
            .map(|r| r["detail"].as_str().unwrap_or("").to_string())
            .unwrap_or_default()
    };
    let ttft = detail("MVP-A1");
    let ttft_ms: Option<f64> = ttft
        .split("ttft=")
        .nth(1)
        .and_then(|x| x.split("ms").next())
        .and_then(|x| x.parse().ok());
    let wall: Option<f64> = detail("MVP-A3")
        .split("wall=")
        .nth(1)
        .and_then(|x| x.split('s').next())
        .and_then(|x| x.parse().ok());
    let lat = match (ttft_ms, wall) {
        (Some(t), Some(w)) => (
            if t <= 2000.0 && w <= 240.0 {
                "pass"
            } else {
                "fail"
            },
            format!("fast-path TTFT {t:.0} ms (<= 2,000), slow path {w:.0} s (<= 240)"),
        ),
        _ => ("pending", "acceptance run lacks A1/A3 timings".into()),
    };
    let g1 = gate(
        "G1",
        "MVP acceptance and latency targets",
        8,
        vec![
            crit(
                "MVP",
                "MVP acceptance tests pass",
                mvp,
                format!("{passed}/{} passed", acc.len()),
            ),
            crit(
                "NFR-002/004",
                "Latency targets on the reference machine",
                lat.0,
                lat.1,
            ),
        ],
    );

    // ---- G2: pooled over runs.
    let ece = if e.calibration_promoted {
        crit(
            "ECE",
            "Calibration ECE <= 0.05 on >= 2,900 held-out samples (NFR-020)",
            "pass",
            format!(
                "promoted: n {}, ECE {:.3}",
                e.calibration_n, e.calibration_ece
            ),
        )
    } else {
        crit(
            "ECE",
            "Calibration ECE <= 0.05 on >= 2,900 held-out samples (NFR-020)",
            "pending",
            format!(
                "no promoted calibration (current fit n {})",
                e.calibration_n
            ),
        )
    };
    // Routing accuracy of the blind gate against the oracle, from each run's own report.
    let (mut r_agree, mut r_n) = (0.0f64, 0u64);
    let (mut ni_diffs, mut cp_k, mut cp_n) = (vec![], 0u64, 0u64);
    for r in &e.eval_runs {
        if let Some(rt) = r["routing"].as_object() {
            for (k, v) in rt {
                if gated(k).map(|g| g.0).unwrap_or(false) {
                    let n = v["n"].as_u64().unwrap_or(0);
                    r_n += n;
                    r_agree += v["routing_accuracy"].as_f64().unwrap_or(0.0) * n as f64;
                }
            }
        }
        if let Some(c) = r["coupling"]["draft_blind"].as_object() {
            cp_k += c.get("adopted").and_then(|x| x.as_u64()).unwrap_or(0);
            cp_n += c.get("wrong_drafts").and_then(|x| x.as_u64()).unwrap_or(0);
        }
    }
    for items in &runs {
        let by = |s: &str| {
            items
                .iter()
                .filter(|i| i.system == s && i.correct.is_some())
                .map(|i| (i.task_id.clone(), i.correct == Some(true)))
                .collect::<BTreeMap<_, _>>()
        };
        let (a, b) = (by("nafs_aql"), by("slow_only"));
        for (k, x) in &a {
            if let Some(y) = b.get(k) {
                ni_diffs.push(*x as i32 as f64 - *y as i32 as f64);
            }
        }
    }
    let routing = if r_n >= 100 {
        let acc = r_agree / r_n as f64;
        crit(
            "H1",
            "Routing accuracy >= 0.70 (blind gate vs the Theorem 2 oracle)",
            if acc >= 0.70 { "pass" } else { "fail" },
            format!("{:.0}% on {r_n} items", acc * 100.0),
        )
    } else {
        crit(
            "H1",
            "Routing accuracy >= 0.70 (blind gate vs the Theorem 2 oracle)",
            "pending",
            format!("{r_n} blind-gate items (need >= 100 for a decision)"),
        )
    };
    let ni = if ni_diffs.len() >= 700 {
        let n = ni_diffs.len();
        let mut rng = fastrand::Rng::with_seed(7);
        let mut boots: Vec<f64> = (0..2000)
            .map(|_| (0..n).map(|_| ni_diffs[rng.usize(0..n)]).sum::<f64>() / n as f64)
            .collect();
        boots.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let lo = boots[50];
        crit(
            "ER-002",
            "3-point non-inferiority vs deliberate-only on >= 700 pairs",
            if lo > -0.03 { "pass" } else { "fail" },
            format!("{n} pairs, lower 95% bound {:+.1} points", lo * 100.0),
        )
    } else {
        crit(
            "ER-002",
            "3-point non-inferiority vs deliberate-only on >= 700 pairs",
            "pending",
            format!("{} pairs pooled over runs", ni_diffs.len()),
        )
    };
    let coupling = if cp_n >= 200 {
        let (_, hi) = clopper_pearson(cp_k, cp_n);
        crit(
            "ER-004",
            "Coupling: adoption of wrong drafts <= 5% (upper bound) on >= 200 trials",
            if hi <= 0.05 { "pass" } else { "fail" },
            format!("{cp_k}/{cp_n} adopted, upper bound {:.1}%", hi * 100.0),
        )
    } else {
        crit(
            "ER-004",
            "Coupling: adoption of wrong drafts <= 5% (upper bound) on >= 200 trials",
            "pending",
            format!("{cp_k}/{cp_n} wrong drafts adopted so far"),
        )
    };
    let g2 = gate(
        "G2",
        "Calibrated gate, routing, non-inferiority, coupling",
        13,
        vec![ece, routing, ni, coupling],
    );

    // ---- G3
    let hist = e
        .registry
        .as_ref()
        .and_then(|r| r["history"].as_array().cloned())
        .unwrap_or_default();
    let promoted = hist.iter().any(|h| h["action"] == "promote");
    let rolled = hist.iter().any(|h| h["action"] == "rollback");
    let g3 = gate(
        "G3",
        "First adapter trained and gated; rollback demonstrated",
        18,
        vec![
            crit(
                "TR-007",
                "An adapter evaluated by the promotion gate",
                if promoted { "pass" } else { "pending" },
                if promoted {
                    "promotion recorded in the registry".into()
                } else {
                    "no promotion yet (training needs a GPU host)".into()
                },
            ),
            crit(
                "OR-005",
                "Rollback demonstrated",
                if rolled { "pass" } else { "pending" },
                if rolled {
                    "rollback recorded in the registry".into()
                } else {
                    "no rollback recorded".into()
                },
            ),
        ],
    );

    // ---- G4
    let proto = e
        .bench_runs
        .iter()
        .find(|r| r["kind"] == "protocol" && r["status"] == "done");
    let load = match proto {
        Some(p) => crit(
            "ER-007",
            "Load tests at 1, 2 and 4 sessions with p50/p95/p99",
            "pass",
            format!("protocol run {}", p["run_id"].as_str().unwrap_or("?")),
        ),
        None => crit(
            "ER-007",
            "Load tests at 1, 2 and 4 sessions with p50/p95/p99",
            "pending",
            "no completed protocol run".into(),
        ),
    };
    // NFR-007 is judged on the median ratio over the most recent eligible runs, not on the latest run alone:
    // on a shared CPU single runs ranged from 1.8x to 2.5x. A run is eligible when the single-session level
    // has at least MIN_NFR_N fast-path samples (with fewer, "p95" is just the maximum).
    const MIN_NFR_N: u64 = 8;
    let eligible: Vec<(f64, &str)> = e
        .bench_runs
        .iter()
        .filter(|r| r["kind"] == "protocol" && r["status"] == "done")
        .filter(|r| r["nfr007"]["ratio"].is_f64())
        .filter(|r| {
            r["levels"]
                .as_array()
                .and_then(|l| l.first())
                .and_then(|l| l["results"]["latency_fast_ms"]["n"].as_u64())
                .unwrap_or(0)
                >= MIN_NFR_N
        })
        .take(5)
        .map(|r| {
            (
                r["nfr007"]["ratio"].as_f64().unwrap_or(f64::NAN),
                r["run_id"].as_str().unwrap_or("?"),
            )
        })
        .collect();
    let nfr = if eligible.is_empty() {
        crit(
            "NFR-007",
            "p95 at 4 sessions <= 2 x single session",
            "pending",
            match proto {
                Some(_) => {
                    format!("needs a protocol run with at least {MIN_NFR_N} requests per session")
                }
                None => "no protocol verdict".into(),
            },
        )
    } else {
        let mut rs: Vec<f64> = eligible.iter().map(|x| x.0).collect();
        rs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let m = rs.len();
        let median = if m % 2 == 1 {
            rs[m / 2]
        } else {
            (rs[m / 2 - 1] + rs[m / 2]) / 2.0
        };
        let each = eligible
            .iter()
            .map(|(r, _)| format!("{r:.2}"))
            .collect::<Vec<_>>()
            .join(", ");
        let detail = format!("median ratio {median:.2} over {m} run(s) ({each})");
        if median <= 2.0 {
            crit(
                "NFR-007",
                "p95 at 4 sessions <= 2 x single session",
                "pass",
                detail,
            )
        } else {
            crit(
                "NFR-007",
                "p95 at 4 sessions <= 2 x single session",
                "fail",
                format!("{detail}; fallback: document the single-session profile"),
            )
        }
    };
    let g4 = gate(
        "G4",
        "Evaluation suite complete; load tests reported",
        21,
        vec![
            crit(
                "P4",
                "Statistics, dashboard and drift monitor implemented with unit tests",
                "pass",
                "nafs-eval stats, gates dashboard, CUSUM drift monitor".into(),
            ),
            load,
            nfr,
        ],
    );
    // ---- G5
    let sec = match &e.security {
        Some(v)
            if v["open_high_or_above"].as_u64() == Some(0)
                && v["sandbox_escape_tests"] == "pass" =>
        {
            crit(
                "SR-006/SR-002",
                "Security review: no open high-severity findings; sandbox escape tests pass",
                "pass",
                format!(
                    "{} findings reviewed, 0 open high or above",
                    v["findings"].as_array().map(|a| a.len()).unwrap_or(0)
                ),
            )
        }
        Some(v) => crit(
            "SR-006/SR-002",
            "Security review: no open high-severity findings; sandbox escape tests pass",
            "fail",
            format!(
                "{} open high or above; escape tests {}",
                v["open_high_or_above"], v["sandbox_escape_tests"]
            ),
        ),
        None => crit(
            "SR-006/SR-002",
            "Security review: no open high-severity findings; sandbox escape tests pass",
            "pending",
            "no security review record".into(),
        ),
    };
    let dep = match &e.deploy {
        Some(v) if v["all_pass"] == true => crit(
            "OR-003",
            "Deployment recipes verified",
            "pass",
            v["summary"].as_str().unwrap_or("verified").to_string(),
        ),
        Some(v) => crit(
            "OR-003",
            "Deployment recipes verified",
            "fail",
            v["summary"]
                .as_str()
                .unwrap_or("verification failed")
                .to_string(),
        ),
        None => crit(
            "OR-003",
            "Deployment recipes verified",
            "pending",
            "run scripts/verify_deploy.sh".into(),
        ),
    };
    let pc = match &e.profile_c {
        Some(v) => crit(
            "MR-005",
            "GPU and Profile-C decisions recorded with data",
            "pass",
            v["decision"].as_str().unwrap_or("recorded").to_string(),
        ),
        None => crit(
            "MR-005",
            "GPU and Profile-C decisions recorded with data",
            "pending",
            "no Profile-C decision recorded".into(),
        ),
    };
    let g5 = gate(
        "G5",
        "Security review clean; deployment verified; upgrade decisions recorded",
        25,
        vec![sec, dep, pc],
    );

    // ---- G6
    let ni1 = if ni_diffs.len() >= 6200 {
        let n = ni_diffs.len();
        let mut rng = fastrand::Rng::with_seed(9);
        let mut boots: Vec<f64> = (0..2000)
            .map(|_| (0..n).map(|_| ni_diffs[rng.usize(0..n)]).sum::<f64>() / n as f64)
            .collect();
        boots.sort_by(|x, y| x.partial_cmp(y).unwrap());
        crit(
            "ER-002",
            "1-point non-inferiority vs deliberate-only on >= 6,200 pairs",
            if boots[50] > -0.01 { "pass" } else { "fail" },
            format!("{n} pairs, lower bound {:+.1} points", boots[50] * 100.0),
        )
    } else {
        crit(
            "ER-002",
            "1-point non-inferiority vs deliberate-only on >= 6,200 pairs",
            "pending",
            format!(
                "{} pairs (fallback: release with the 3-point result and a documented limitation)",
                ni_diffs.len()
            ),
        )
    };
    let reg_runs: Vec<&Value> = e
        .eval_runs
        .iter()
        .filter(|r| r["request"]["benchmark"] == "regression_500" && r["status"] == "done")
        .collect();
    let reg = match reg_runs.len() {
        0 => crit(
            "ER-008",
            "Regression suite green",
            "pending",
            "no completed regression_500 run".into(),
        ),
        1 => crit(
            "ER-008",
            "Regression suite green",
            "pass",
            format!(
                "baseline recorded: {} items, accuracy {:.1}%",
                reg_runs[0]["progress"]["done"],
                reg_runs[0]["systems"]["fast_only"]["accuracy"]
                    .as_f64()
                    .unwrap_or(0.0)
                    * 100.0
            ),
        ),
        _ => {
            let a: Vec<ItemResult> =
                serde_json::from_value(reg_runs[0]["items"].clone()).unwrap_or_default();
            let b: Vec<ItemResult> =
                serde_json::from_value(reg_runs[1]["items"].clone()).unwrap_or_default();
            let bm: BTreeMap<_, _> = b
                .iter()
                .filter(|i| i.correct.is_some())
                .map(|i| {
                    (
                        (i.task_id.clone(), i.system.clone()),
                        i.correct == Some(true),
                    )
                })
                .collect();
            let (mut worse, mut better) = (0u64, 0u64);
            for i in a.iter().filter(|i| i.correct.is_some()) {
                if let Some(prev) = bm.get(&(i.task_id.clone(), i.system.clone())) {
                    let now = i.correct == Some(true);
                    worse += (*prev && !now) as u64;
                    better += (!*prev && now) as u64;
                }
            }
            let p = nafs_eval::mcnemar_exact(worse, better);
            let drop = worse > better && p < 0.05;
            crit(
                "ER-008",
                "Regression suite green",
                if drop { "fail" } else { "pass" },
                format!(
                    "latest vs previous: {worse} items worse, {better} better (McNemar p = {p:.3})"
                ),
            )
        }
    };
    let sch = match &e.scholar {
        Some(v) if v["decision"] == "approved" => crit(
            "SR-005",
            "Scholar review of the analogy text signed off",
            "pass",
            format!(
                "{} on {}",
                v["reviewer"].as_str().unwrap_or("reviewer"),
                v["date"].as_str().unwrap_or("?")
            ),
        ),
        Some(v) if v["decision"] == "changes_requested" => crit(
            "SR-005",
            "Scholar review of the analogy text signed off",
            "fail",
            v["notes"]
                .as_str()
                .unwrap_or("changes requested")
                .to_string(),
        ),
        _ => crit(
            "SR-005",
            "Scholar review of the analogy text signed off",
            "pending",
            "review packet in docs/release/scholar_review_packet.md awaits a reviewer".into(),
        ),
    };
    let g6 = gate(
        "G6",
        "Release: 1-point non-inferiority, regression green, scholar sign-off",
        28,
        vec![ni1, reg, sch],
    );

    json!({"gates": [g0, g1, g2, g3, g4, g5, g6], "evaluated_at": chrono::Utc::now().to_rfc3339(),
        "note": "Pending means the stored evidence is not yet enough to decide; pass and fail follow the roadmap's go/no-go criteria."})
}

#[cfg(test)]
mod tests {
    use super::*;
    fn ev() -> Evidence {
        Evidence {
            eval_runs: vec![],
            bench_runs: vec![],
            acceptance: None,
            h7: None,
            registry: None,
            calibration_promoted: false,
            calibration_n: 0,
            calibration_ece: f32::NAN,
            security: None,
            deploy: None,
            profile_c: None,
            scholar: None,
        }
    }
    fn status<'a>(v: &'a Value, g: &str) -> &'a str {
        v["gates"]
            .as_array()
            .unwrap()
            .iter()
            .find(|x| x["gate"] == g)
            .unwrap()["status"]
            .as_str()
            .unwrap()
    }
    #[test]
    fn empty_evidence_is_pending_not_failed() {
        let v = evaluate(&ev());
        for g in ["G0", "G1", "G2", "G3", "G5", "G6"] {
            assert_eq!(status(&v, g), "pending", "{g}");
        }
    }
    #[test]
    fn gates_follow_their_criteria() {
        let mut e = ev();
        e.h7 = Some(json!({"pass": true, "summary": "79%"}));
        e.acceptance = Some(
            json!({"results": (1..=9).map(|i| json!({"id": format!("MVP-A{i}"), "pass": true,
            "detail": if i == 1 { "ttft=900ms" } else if i == 3 { "wall=161s" } else { "" }})).collect::<Vec<_>>()}),
        );
        e.registry = Some(json!({"history": [{"action": "promote"}, {"action": "rollback"}]}));
        e.bench_runs = vec![
            json!({"kind": "protocol", "status": "done", "run_id": "x", "nfr007": {"pass": false, "ratio": 2.6},
                   "levels": [{"results": {"latency_fast_ms": {"n": 8}}}]}),
        ];
        let v = evaluate(&e);
        assert_eq!(status(&v, "G1"), "pass");
        assert_eq!(status(&v, "G3"), "pass");
        assert_eq!(status(&v, "G4"), "fail", "NFR-007 failed");
        assert_eq!(status(&v, "G0"), "pending", "baseline not run at scale");
        e.acceptance = Some(json!({"results": [{"id": "MVP-A1", "pass": false, "detail": ""}]}));
        assert_eq!(status(&evaluate(&e), "G1"), "fail");
    }
    #[test]
    fn nfr007_uses_the_median_of_eligible_runs() {
        let run = |ratio: f64, n: u64| {
            json!({"kind": "protocol", "status": "done", "run_id": "r", "nfr007": {"pass": ratio <= 2.0, "ratio": ratio},
                   "levels": [{"results": {"latency_fast_ms": {"n": n}}}]})
        };
        let nfr = |e: &Evidence| {
            let v = evaluate(e);
            let g4 = v["gates"]
                .as_array()
                .unwrap()
                .iter()
                .find(|g| g["gate"] == "G4")
                .unwrap()
                .clone();
            let c = g4["criteria"]
                .as_array()
                .unwrap()
                .iter()
                .find(|c| c["id"] == "NFR-007")
                .unwrap()
                .clone();
            c["status"].as_str().unwrap().to_string()
        };
        let mut e = ev();
        // A lucky short run (4 samples, so p95 is the maximum) does not override two failing full runs.
        e.bench_runs = vec![run(1.8, 4), run(2.15, 8), run(2.48, 8)];
        assert_eq!(nfr(&e), "fail");
        e.bench_runs = vec![run(1.8, 8), run(2.15, 8), run(1.9, 8)];
        assert_eq!(nfr(&e), "pass");
        e.bench_runs = vec![run(1.8, 4)];
        assert_eq!(nfr(&e), "pending");
    }
}
