//! nafs-bench: closed-loop load generation for the performance console (FR-052, ER-007).
use nafs_core::orchestrator::{Emitter, Orchestrator};
use nafs_types::{ChatRequest, Message, ReasoningMode, StreamEvent};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchRunRequest {
    pub concurrency: usize,
    #[serde(default)]
    pub duration_s: Option<u64>,
    #[serde(default)]
    pub requests: Option<usize>,
    #[serde(default = "d_set")]
    pub prompt_set: String,
    #[serde(default)]
    pub mode: Option<String>, // auto | fast | slow
    #[serde(default)]
    pub lambda: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub thinking_budget: Option<u32>,
}
fn d_set() -> String {
    "chat-short".into()
}

pub fn prompt_sets() -> Value {
    json!({
      "chat-short": {"label": "Short chat (fast path)", "prompts": ["Say hello in one short sentence.", "What is 7 times 8? Reply with the number only.", "Name three primary colours.", "Give a synonym for 'quick'.", "What day comes after Friday?", "Translate 'thank you' into Indonesian."]},
      "mixed": {"label": "Mixed: math, code, factual, open", "prompts": ["A shop sells pens at 3 for $2. How much do 12 pens cost?", "Write a Python function is_palindrome(s) that ignores case and spaces.", "What is the capital of Japan?", "Give two tips for better sleep.", "If a train travels 180 km in 2.5 hours, what is its average speed in km/h?", "Return a JSON object with keys name and age for a 30-year-old named Sari."]},
      "math": {"label": "Word problems (escalation-heavy)", "prompts": ["Ali has 3 times as many marbles as Budi. Together they have 48. How many does Ali have?", "A rectangle has perimeter 50 cm and length 15 cm. What is its area?", "What is 15% of 240?", "If 4 workers build a wall in 6 days, how many days do 3 workers need?"]}
    })
}

#[derive(Debug, Clone, Serialize)]
pub struct Sample {
    pub t_start_ms: u64,
    pub ttft_ms: Option<u64>,
    pub latency_ms: u64,
    pub ok: bool,
    pub mode: String,
    pub tokens: u32,
    pub worker: usize,
    pub error: Option<String>,
}

fn q(v: &mut [f64], p: f64) -> Option<f64> {
    if v.is_empty() {
        None
    } else {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Some(v[((v.len() - 1) as f64 * p).round() as usize])
    }
}
fn quant(mut v: Vec<f64>) -> Value {
    json!({"p50": q(&mut v, 0.5), "p95": q(&mut v, 0.95), "p99": q(&mut v, 0.99), "mean": if v.is_empty() { None } else { Some(v.iter().sum::<f64>() / v.len() as f64) }, "n": v.len()})
}

pub fn summarize(
    req: &BenchRunRequest,
    samples: &[Sample],
    elapsed: Duration,
    done: bool,
) -> Value {
    let ok: Vec<&Sample> = samples.iter().filter(|s| s.ok).collect();
    let secs = elapsed.as_secs_f64().max(0.001);
    let by_mode = |m: &str| -> Value {
        quant(
            ok.iter()
                .filter(|s| s.mode == m)
                .map(|s| s.latency_ms as f64)
                .collect(),
        )
    };
    json!({"status": if done { "done" } else { "running" }, "request": req, "elapsed_s": secs,
        "results": {"requests": samples.len(), "ok": ok.len(), "error_rate": if samples.is_empty() { 0.0 } else { 1.0 - ok.len() as f64 / samples.len() as f64 },
            "throughput_rps": ok.len() as f64 / secs, "tokens_per_s": ok.iter().map(|s| s.tokens as f64).sum::<f64>() / secs,
            "escalation_rate": if ok.is_empty() { None } else { Some(ok.iter().filter(|s| s.mode == "slow").count() as f64 / ok.len() as f64) },
            "ttft_ms": quant(ok.iter().filter_map(|s| s.ttft_ms.map(|x| x as f64)).collect()), "latency_ms": quant(ok.iter().map(|s| s.latency_ms as f64).collect()),
            "latency_fast_ms": by_mode("fast"), "latency_slow_ms": by_mode("slow"), "cost_per_request_s": if ok.is_empty() { None } else { Some(ok.iter().map(|s| s.latency_ms as f64).sum::<f64>() / ok.len() as f64 / 1000.0) }},
        "samples": samples})
}

/// Closed-loop load test: `concurrency` workers issue requests until the duration or request count is reached.
pub async fn run(
    orch: Arc<Orchestrator>,
    req: BenchRunRequest,
    cancel: Arc<AtomicBool>,
    progress: impl Fn(&Value) + Send + Sync + 'static,
) -> Value {
    let sets = prompt_sets();
    let prompts: Vec<String> = sets[&req.prompt_set]["prompts"]
        .as_array()
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter_map(|p| p.as_str().map(String::from))
        .collect();
    let prompts = if prompts.is_empty() {
        vec!["Say hello.".to_string()]
    } else {
        prompts
    };
    let samples = Arc::new(Mutex::new(Vec::<Sample>::new()));
    let t0 = Instant::now();
    let deadline = req.duration_s.map(|d| t0 + Duration::from_secs(d));
    let max_req = req.requests.unwrap_or(usize::MAX);
    let issued = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let progress = Arc::new(progress);
    let mut handles = vec![];
    for w in 0..req.concurrency.clamp(1, 64) {
        let (orch, samples, prompts, issued, cancel, req, progress) = (
            orch.clone(),
            samples.clone(),
            prompts.clone(),
            issued.clone(),
            cancel.clone(),
            req.clone(),
            progress.clone(),
        );
        handles.push(tokio::spawn(async move {
            loop {
                if cancel.load(Ordering::Relaxed) {
                    break;
                }
                if let Some(d) = deadline {
                    if Instant::now() >= d {
                        break;
                    }
                }
                let i = issued.fetch_add(1, Ordering::SeqCst);
                if i >= max_req {
                    break;
                }
                let prompt = prompts[i % prompts.len()].clone();
                let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
                let start = Instant::now();
                let t_start_ms = (start - t0).as_millis() as u64;
                let force = match req.mode.as_deref() {
                    Some("fast") => Some(ReasoningMode::Fast),
                    Some("slow") => Some(ReasoningMode::Slow),
                    _ => None,
                };
                let creq = ChatRequest {
                    messages: vec![Message {
                        role: "user".into(),
                        content: prompt,
                    }],
                    force_mode: force,
                    lambda: req.lambda,
                    max_tokens: req.max_tokens,
                    thinking_budget: req.thinking_budget,
                    ephemeral: Some(true),
                    lab: true,
                    ..Default::default()
                };
                let o2 = orch.clone();
                let c2 = cancel.clone();
                let task = tokio::spawn(async move {
                    // Same wall-clock cap as the HTTP path, and cancellable mid-generation (review L5, M8).
                    let cap =
                        std::time::Duration::from_secs(o2.cfg.server.request_wall_clock_cap_s + 30);
                    match tokio::time::timeout(
                        cap,
                        o2.handle_with_cancel(creq, Emitter(Some(tx)), Some(c2)),
                    )
                    .await
                    {
                        Ok(r) => r,
                        Err(_) => Err(nafs_types::NafsError::BackendTimeout(
                            "request wall-clock cap exceeded".into(),
                        )),
                    }
                });
                let mut ttft = None;
                while let Some(ev) = rx.recv().await {
                    if let StreamEvent::Token {
                        reasoning: false, ..
                    } = ev
                    {
                        if ttft.is_none() {
                            ttft = Some(start.elapsed().as_millis() as u64);
                        }
                    }
                }
                let res = task.await;
                let latency_ms = start.elapsed().as_millis() as u64;
                let s = match res {
                    Ok(Ok(r)) => Sample {
                        t_start_ms,
                        ttft_ms: ttft,
                        latency_ms,
                        ok: true,
                        mode: if r.reasoning_mode == ReasoningMode::Slow {
                            "slow".into()
                        } else {
                            "fast".into()
                        },
                        tokens: r.usage.completion_tokens,
                        worker: w,
                        error: None,
                    },
                    Ok(Err(e)) => Sample {
                        t_start_ms,
                        ttft_ms: ttft,
                        latency_ms,
                        ok: false,
                        mode: "error".into(),
                        tokens: 0,
                        worker: w,
                        error: Some(e.to_string()),
                    },
                    Err(e) => Sample {
                        t_start_ms,
                        ttft_ms: ttft,
                        latency_ms,
                        ok: false,
                        mode: "error".into(),
                        tokens: 0,
                        worker: w,
                        error: Some(e.to_string()),
                    },
                };
                let snapshot = {
                    let mut g = samples.lock().unwrap();
                    g.push(s);
                    g.clone()
                };
                progress(&summarize(&req, &snapshot, t0.elapsed(), false));
            }
        }));
    }
    for h in handles {
        let _ = h.await;
    }
    let snapshot = samples.lock().unwrap().clone();
    summarize(&req, &snapshot, t0.elapsed(), true)
}

/// The recorded load-test protocol (ER-007): the same prompt set at 1, 2 and 4 concurrent sessions,
/// p50/p95/p99 per level, and the NFR-007 verdict (p95 fast-path latency at the recommended limit of 2
/// sessions at most twice the single-session value; decision 002). The load generator is in-process (closed-loop workers driving the full
/// orchestrator) rather than goose over HTTP; see ASSUMPTIONS A-036.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolRequest {
    #[serde(default = "d_set")]
    pub prompt_set: String,
    #[serde(default = "d_mode")]
    pub mode: String,
    #[serde(default = "d_levels")]
    pub levels: Vec<usize>,
    /// Requests per session at each level (so level c issues c * requests_per_session requests).
    #[serde(default = "d_rps")]
    pub requests_per_session: usize,
    #[serde(default)]
    pub max_tokens: Option<u32>,
}
fn d_mode() -> String {
    "fast".into()
}
fn d_levels() -> Vec<usize> {
    vec![1, 2, 4]
}
fn d_rps() -> usize {
    8
}

/// Recommended concurrency limit of the CPU profile (decision 002): NFR-007 is judged at this level.
pub const NFR007_SESSIONS: u64 = 2;

/// p95 fast-path latency of the level with `c` concurrent sessions.
pub fn p95_at(levels: &[Value], c: u64) -> Option<f64> {
    levels
        .iter()
        .find(|l| l["concurrency"].as_u64() == Some(c))
        .and_then(|l| {
            l["results"]["latency_fast_ms"]["p95"]
                .as_f64()
                .or(l["results"]["latency_ms"]["p95"].as_f64())
        })
}

/// NFR-007 verdict: p95 at the recommended concurrency limit (2 sessions, decision 002) at most twice
/// the single-session value. The 4-session ratio is kept in the record for comparison.
pub fn nfr007(levels: &[Value]) -> Value {
    match (p95_at(levels, 1), p95_at(levels, NFR007_SESSIONS)) {
        (Some(a), Some(b)) => {
            let mut v = json!({"p95_single_ms": a, "p95_at_limit_ms": b, "limit_sessions": NFR007_SESSIONS,
                "ratio": b / a.max(1.0), "pass": b <= 2.0 * a,
                "target": format!("p95 fast-path latency at {NFR007_SESSIONS} sessions (the recommended limit) <= 2 x single session (NFR-007, decision 002)")});
            if let Some(four) = p95_at(levels, 4) {
                v["p95_at_4_ms"] = json!(four);
                v["ratio_at_4"] = json!(four / a.max(1.0));
            }
            v
        }
        _ => json!({"pass": null, "note": format!("needs levels 1 and {NFR007_SESSIONS}")}),
    }
}

pub async fn protocol(
    orch: Arc<Orchestrator>,
    req: ProtocolRequest,
    cancel: Arc<AtomicBool>,
    progress: impl Fn(&Value) + Send + Sync + Clone + 'static,
) -> Value {
    let started = chrono::Utc::now().to_rfc3339();
    let mut levels: Vec<Value> = vec![];
    for (i, &c) in req.levels.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        let br = BenchRunRequest {
            concurrency: c.clamp(1, 16),
            duration_s: None,
            requests: Some(c.clamp(1, 16) * req.requests_per_session.clamp(1, 50)),
            prompt_set: req.prompt_set.clone(),
            mode: Some(req.mode.clone()),
            lambda: None,
            max_tokens: req.max_tokens,
            thinking_budget: None,
        };
        // One uncounted warm-up request per level, so a cold prompt cache does not become the p95.
        let warm = BenchRunRequest {
            concurrency: 1,
            requests: Some(1),
            ..br.clone()
        };
        let _ = run(orch.clone(), warm, cancel.clone(), |_| {}).await;
        let done_levels = levels.clone();
        let (p2, req2) = (progress.clone(), req.clone());
        let mut v = run(orch.clone(), br, cancel.clone(), move |snap| {
            let mut cur = snap.clone();
            if let Some(o) = cur.as_object_mut() {
                o.remove("samples");
            }
            p2(&json!({"status": "running", "kind": "protocol", "request": req2, "level_index": i, "levels": done_levels, "current": cur}));
        })
        .await;
        if let Some(o) = v.as_object_mut() {
            o.remove("samples");
        }
        v["concurrency"] = json!(c);
        levels.push(v);
    }
    let status = if cancel.load(Ordering::Relaxed) {
        "cancelled"
    } else {
        "done"
    };
    json!({"status": status, "kind": "protocol", "request": req, "started_at": started, "levels": levels, "nfr007": nfr007(&levels),
        "table": levels.iter().map(|l| json!({"concurrency": l["concurrency"], "requests": l["results"]["requests"], "ok": l["results"]["ok"],
            "p50_ms": l["results"]["latency_ms"]["p50"], "p95_ms": l["results"]["latency_ms"]["p95"], "p99_ms": l["results"]["latency_ms"]["p99"],
            "ttft_p95_ms": l["results"]["ttft_ms"]["p95"], "throughput_rps": l["results"]["throughput_rps"], "escalation_rate": l["results"]["escalation_rate"],
            "cost_per_request_s": l["results"]["cost_per_request_s"]})).collect::<Vec<_>>()})
}

/// One reply's latency record for the breakdown: reasoning mode ("fast" or "slow"), wall time and its stages.
pub type StageRow = (String, u64, Vec<nafs_types::StageTiming>);

fn qs(sorted: &[f64], p: f64) -> Option<f64> {
    (!sorted.is_empty()).then(|| sorted[((sorted.len() - 1) as f64 * p).round() as usize])
}

/// Where the time goes (P7-T01): per path, each stage's share of wall time with p50/p95, the
/// unattributed overhead, and the engine's prefill and decode speed, KV-cache reuse and queue wait per
/// model stage. "sample" stages overlap the draft and are excluded from the shares; "sample_after" (samples
/// drawn after the draft) adds to the wall time and is counted.
pub fn latency_breakdown(rows: &[StageRow]) -> Value {
    let mut paths = vec![];
    for mode in ["fast", "slow"] {
        let rs: Vec<&StageRow> = rows.iter().filter(|r| r.0 == mode).collect();
        if rs.is_empty() {
            continue;
        }
        let wall_total: f64 = rs.iter().map(|r| r.1 as f64).sum();
        let mut walls: Vec<f64> = rs.iter().map(|r| r.1 as f64).collect();
        walls.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut by: std::collections::BTreeMap<String, Vec<f64>> = Default::default();
        let mut attributed = 0.0;
        for r in &rs {
            for st in &r.2 {
                by.entry(st.stage.clone()).or_default().push(st.ms as f64);
                if st.stage != "sample" {
                    attributed += st.ms as f64;
                }
            }
        }
        let mut stages: Vec<Value> = by
            .into_iter()
            .map(|(k, mut v)| {
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let sum: f64 = v.iter().sum();
                json!({"stage": k, "n": v.len(), "p50_ms": qs(&v, 0.5), "p95_ms": qs(&v, 0.95),
                       "mean_ms": sum / v.len() as f64,
                       "share": if k == "sample" { Value::Null } else { json!(sum / wall_total.max(1.0)) },
                       "overlaps_draft": k == "sample"})
            })
            .collect();
        stages.sort_by(|a, b| {
            b["share"]
                .as_f64()
                .unwrap_or(-1.0)
                .partial_cmp(&a["share"].as_f64().unwrap_or(-1.0))
                .unwrap()
        });
        paths.push(json!({"mode": mode, "replies": rs.len(), "wall_p50_ms": qs(&walls, 0.5), "wall_p95_ms": qs(&walls, 0.95),
            "stages": stages, "overhead_share": ((wall_total - attributed) / wall_total.max(1.0)).max(0.0)}));
    }
    // Engine view per model stage.
    let mut engine = vec![];
    for stage in ["draft", "sample", "sample_after", "deliberate"] {
        let sts: Vec<&nafs_types::StageTiming> = rows
            .iter()
            .flat_map(|r| r.2.iter())
            .filter(|s| s.stage == stage)
            .collect();
        if sts.is_empty() {
            continue;
        }
        let (mut pn, mut pms, mut cn, mut dn, mut dms) = (0.0, 0.0, 0.0, 0.0, 0.0);
        let mut queues: Vec<f64> = vec![];
        let mut prefill: Vec<f64> = vec![];
        for s in &sts {
            if let (Some(n), Some(ms)) = (s.prompt_n, s.prefill_ms) {
                pn += n as f64;
                pms += ms;
                prefill.push(ms);
            }
            if let Some(c) = s.cache_n {
                cn += c as f64;
            }
            if let (Some(n), Some(ms)) = (s.tokens, s.decode_ms) {
                dn += n as f64;
                dms += ms;
            }
            if let Some(qm) = s.queue_ms {
                queues.push(qm as f64);
            }
        }
        queues.sort_by(|a, b| a.partial_cmp(b).unwrap());
        prefill.sort_by(|a, b| a.partial_cmp(b).unwrap());
        engine.push(json!({"stage": stage, "calls": sts.len(),
            "prefill_tok_s": (pms > 0.0).then(|| pn / pms * 1000.0),
            "decode_tok_s": (dms > 0.0).then(|| dn / dms * 1000.0),
            "cache_reuse": (pn + cn > 0.0).then(|| cn / (pn + cn)),
            "prefill_p50_ms": qs(&prefill, 0.5),
            "queue_p95_ms": qs(&queues, 0.95)}));
    }
    json!({"paths": paths, "engine": engine, "replies": rows.len()})
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nfr007_verdict() {
        let l = |c: u64, p95: f64| json!({"concurrency": c, "results": {"latency_fast_ms": {"p95": p95}, "latency_ms": {"p95": p95}}});
        let v = nfr007(&[l(1, 1000.0), l(2, 1500.0), l(4, 3900.0)]);
        assert_eq!(v["pass"], true, "judged at 2 sessions, not 4");
        assert_eq!(v["ratio_at_4"], 3.9);
        assert_eq!(nfr007(&[l(1, 1000.0), l(2, 2500.0)])["pass"], false);
        assert!(
            nfr007(&[l(1, 1000.0), l(4, 1900.0)])["pass"].is_null(),
            "needs level 2"
        );
        assert!(nfr007(&[l(2, 1.0)])["pass"].is_null());
    }

    #[test]
    fn breakdown_attributes_wall_time_to_stages() {
        use nafs_types::StageTiming;
        let st = |stage: &str, ms: u64| StageTiming {
            stage: stage.into(),
            ms,
            ..Default::default()
        };
        let model = |stage: &str, ms: u64| StageTiming {
            stage: stage.into(),
            ms,
            queue_ms: Some(0),
            prefill_ms: Some(100.0),
            decode_ms: Some(ms as f64 - 100.0),
            prompt_n: Some(50),
            cache_n: Some(150),
            tokens: Some(((ms - 100) / 100) as u32),
            note: None,
        };
        let rows: Vec<StageRow> = vec![
            ("fast".into(), 1_000, vec![model("draft", 900)]),
            (
                "slow".into(),
                10_000,
                vec![
                    model("draft", 1_000),
                    model("sample", 800),
                    model("deliberate", 8_000),
                    st("judge", 500),
                ],
            ),
        ];
        let b = latency_breakdown(&rows);
        let slow = &b["paths"][1];
        assert_eq!(
            slow["stages"][0]["stage"], "deliberate",
            "largest share first"
        );
        assert!((slow["stages"][0]["share"].as_f64().unwrap() - 0.8).abs() < 1e-9);
        assert!(
            (slow["overhead_share"].as_f64().unwrap() - 0.05).abs() < 1e-9,
            "samples overlap the draft"
        );
        let draft = b["engine"]
            .as_array()
            .unwrap()
            .iter()
            .find(|e| e["stage"] == "draft")
            .unwrap();
        assert!((draft["cache_reuse"].as_f64().unwrap() - 0.75).abs() < 1e-9);
        assert!((draft["prefill_tok_s"].as_f64().unwrap() - 500.0).abs() < 1e-9);
    }
}
