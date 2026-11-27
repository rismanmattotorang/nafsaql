//! Phase 39: engine comparison. The same prompts go to System 1 and System 2 of each engine (Smiley Core, Peregrine,
//! llama.cpp, ...) straight through the backend clients, without the gate, so the numbers are the engines' own:
//! time to first token, prompt (prefill) and generation (decode) speed, wall time and whether the answer holds the
//! expected value. Engines are interleaved inside every round and their order alternates between rounds (A B, B A, ...),
//! so a load change on the shared host hits every engine alike; the paired comparison counts per-prompt wins and
//! gives a two-sided sign-test p-value.
use nafs_core::backend::{Backend, GenOutput, GenRequest};
use nafs_core::orchestrator::Orchestrator;
use nafs_types::Message;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineCompareRequest {
    /// Engine names under `[backends]`; default: every configured engine whose two servers answer.
    #[serde(default)]
    pub engines: Vec<String>,
    /// `system1`, `system2` or both (default both).
    #[serde(default = "d_roles")]
    pub roles: Vec<String>,
    /// Rounds over the prompt set (default 3, at most 20).
    #[serde(default = "d_rounds")]
    pub rounds: usize,
    #[serde(default = "d_s1_tokens")]
    pub s1_max_tokens: u32,
    #[serde(default = "d_s2_tokens")]
    pub s2_max_tokens: u32,
    /// System 2's reasoning budget (tokens); the answer follows the forced `</think>`.
    #[serde(default = "d_budget")]
    pub s2_thinking_budget: u32,
}
fn d_roles() -> Vec<String> {
    vec!["system1".into(), "system2".into()]
}
fn d_rounds() -> usize {
    3
}
fn d_s1_tokens() -> u32 {
    160
}
fn d_s2_tokens() -> u32 {
    768
}
fn d_budget() -> u32 {
    512
}

/// One prompt of the comparison set: the text, the answer it must contain (lower-case, any of), and its kind.
pub struct EnginePrompt {
    pub id: &'static str,
    pub role: &'static str,
    pub text: &'static str,
    pub expect: &'static [&'static str],
}

const LONG_CONTEXT: &str = "Nafs-Aql routes each question to one of two language models. System 1 is a small model that answers at once; System 2 is a larger model that reasons before it answers. A gate estimates the chance that System 1's draft is wrong from cheap signals: whether a verifier accepts the draft, how often a few quick samples agree, and the draft's own token probabilities. When the expected loss of a wrong answer exceeds the price of the extra latency, the question is escalated to System 2. Verifiers check what can be checked exactly: arithmetic with a computer algebra system, code by running its tests in a sandbox, structured output against its schema. Everything else is judged by calibrated probabilities. The whole service runs on four CPU cores with no GPU, so every second of model time is visible to the person waiting.";

pub fn engine_prompts() -> Vec<EnginePrompt> {
    vec![
        EnginePrompt { id: "s1-arith", role: "system1", text: "What is 17 times 23? Reply with the number only.", expect: &["391"] },
        EnginePrompt { id: "s1-capital", role: "system1", text: "What is the capital city of Australia? Answer in one word.", expect: &["canberra"] },
        EnginePrompt { id: "s1-symbol", role: "system1", text: "What is the chemical symbol for sodium? Answer with the symbol only.", expect: &["na"] },
        EnginePrompt { id: "s1-long-read", role: "system1", text: "", expect: &["four", "4"] },
        // A decode-heavy request: the answers above are a few tokens, which says nothing about writing speed.
        EnginePrompt { id: "s1-long-write", role: "system1", text: "Explain in about 120 words why the sky is blue. Mention Rayleigh scattering.", expect: &["rayleigh"] },
        EnginePrompt { id: "s2-speed", role: "system2", text: "A train travels 120 km in 1.5 hours. What is its average speed in km/h? End with 'Answer: <number>'.", expect: &["80"] },
        EnginePrompt { id: "s2-linear", role: "system2", text: "Solve 3x + 7 = 25 for x. End with 'Answer: <number>'.", expect: &["6"] },
        EnginePrompt { id: "s2-percent", role: "system2", text: "What is 15% of 240? End with 'Answer: <number>'.", expect: &["36"] },
    ]
}

fn prompt_text(p: &EnginePrompt) -> String {
    if p.id == "s1-long-read" {
        // A prompt-heavy request: prefill dominates, decode is short.
        let ctx = std::iter::repeat_n(LONG_CONTEXT, 4)
            .collect::<Vec<_>>()
            .join("\n\n");
        format!("{ctx}\n\nOn how many CPU cores does the service run? Answer in one word.")
    } else {
        p.text.to_string()
    }
}

/// Whether `content` holds one of the expected values as a whole word (so "6" does not match "36").
pub fn answer_holds(content: &str, expect: &[&str]) -> bool {
    let low = content.to_lowercase();
    let words: Vec<&str> = low
        .split(|c: char| !(c.is_alphanumeric() || c == '.'))
        .map(|w| w.trim_matches('.'))
        .filter(|w| !w.is_empty())
        .collect();
    expect.iter().any(|e| words.iter().any(|w| w == e))
}

#[derive(Debug, Clone, Serialize)]
pub struct EngineSample {
    pub round: usize,
    pub prompt: String,
    pub role: String,
    pub engine: String,
    pub ok: bool,
    pub correct: bool,
    pub ttft_ms: Option<u64>,
    pub wall_ms: u64,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub prefill_tok_s: Option<f64>,
    pub decode_tok_s: Option<f64>,
    pub finish: String,
    pub load1: Option<f64>,
    pub error: Option<String>,
}

/// The host's one-minute load average (macOS and Linux), for labelling a run made on a busy machine.
pub fn load_average() -> Option<f64> {
    if let Ok(s) = std::fs::read_to_string("/proc/loadavg") {
        return s.split_whitespace().next()?.parse().ok();
    }
    let out = std::process::Command::new("sysctl")
        .args(["-n", "vm.loadavg"])
        .output()
        .ok()?;
    String::from_utf8_lossy(&out.stdout)
        .trim_matches(|c: char| c == '{' || c == '}' || c.is_whitespace())
        .split_whitespace()
        .next()?
        .parse()
        .ok()
}

fn speeds(o: &GenOutput) -> (Option<f64>, Option<f64>) {
    let prefill = match (o.server_prompt_n, o.server_prefill_ms) {
        (Some(n), Some(ms)) if n > 0 && ms > 0.0 => Some(n as f64 * 1000.0 / ms),
        _ => None,
    };
    let decode = match o.server_decode_ms {
        Some(ms) if ms > 0.0 && o.completion_tokens > 1 => {
            Some(o.completion_tokens as f64 * 1000.0 / ms)
        }
        _ => {
            let gen_ms = o.wall_ms.saturating_sub(o.ttft_ms.unwrap_or(0)) as f64;
            (gen_ms > 0.0 && o.completion_tokens > 1)
                .then(|| (o.completion_tokens - 1) as f64 * 1000.0 / gen_ms)
        }
    };
    (prefill, decode)
}

async fn one(
    b: &Backend,
    p: &EnginePrompt,
    req: &EngineCompareRequest,
) -> std::result::Result<GenOutput, String> {
    let s2 = p.role == "system2";
    let g = GenRequest {
        messages: vec![Message {
            role: "user".into(),
            content: prompt_text(p),
        }],
        max_tokens: if s2 {
            req.s2_max_tokens
        } else {
            req.s1_max_tokens
        },
        // Greedy: both engines then decode the same distribution's argmax, and a rerun is repeatable.
        temperature: 0.0,
        top_p: 1.0,
        thinking: Some(s2),
        thinking_budget: s2.then_some(req.s2_thinking_budget),
        seed: Some(7),
        ..Default::default()
    };
    let mut sink = |_: &str, _: bool| {};
    b.stream(&g, &mut sink).await.map_err(|e| e.to_string())
}

fn median(mut v: Vec<f64>) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    Some(if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    })
}

/// Two-sided sign test: probability of a split at least this uneven among `n` untied pairs under p = 1/2.
pub fn sign_test_p(wins: usize, losses: usize) -> Option<f64> {
    let n = wins + losses;
    if n == 0 {
        return None;
    }
    let k = wins.min(losses);
    let mut c = 1.0f64; // C(n, 0)
    let mut tail = 0.0;
    for i in 0..=k {
        if i > 0 {
            c = c * (n - i + 1) as f64 / i as f64;
        }
        tail += c;
    }
    Some((2.0 * tail / 2f64.powi(n as i32)).min(1.0))
}

pub fn summarize(
    req: &EngineCompareRequest,
    engines: &[String],
    samples: &[EngineSample],
    done: bool,
) -> Value {
    let mut per = vec![];
    for role in &req.roles {
        for e in engines {
            let rows: Vec<&EngineSample> = samples
                .iter()
                .filter(|s| &s.role == role && &s.engine == e)
                .collect();
            if rows.is_empty() {
                continue;
            }
            let ok: Vec<&&EngineSample> = rows.iter().filter(|s| s.ok).collect();
            per.push(json!({
                "role": role, "engine": e, "n": rows.len(), "ok": ok.len(),
                "correct": ok.iter().filter(|s| s.correct).count(),
                "ttft_ms_p50": median(ok.iter().filter_map(|s| s.ttft_ms.map(|x| x as f64)).collect()),
                "wall_ms_p50": median(ok.iter().map(|s| s.wall_ms as f64).collect()),
                "wall_s_total": ok.iter().map(|s| s.wall_ms as f64).sum::<f64>() / 1000.0,
                "prefill_tok_s_p50": median(ok.iter().filter_map(|s| s.prefill_tok_s).collect()),
                "decode_tok_s_p50": median(ok.iter().filter_map(|s| s.decode_tok_s).collect()),
                "completion_tokens_mean": if ok.is_empty() { None } else { Some(ok.iter().map(|s| s.completion_tokens as f64).sum::<f64>() / ok.len() as f64) },
            }));
        }
    }
    // Paired comparison of every engine against the first, per role: the same prompt in the same round.
    let mut pairs = vec![];
    if let Some(base) = engines.first() {
        for role in &req.roles {
            for other in engines.iter().skip(1) {
                let (mut wins, mut losses, mut ratios) = (0usize, 0usize, vec![]);
                for a in samples
                    .iter()
                    .filter(|s| &s.role == role && &s.engine == base && s.ok)
                {
                    if let Some(b) = samples.iter().find(|s| {
                        s.ok && &s.engine == other
                            && s.role == a.role
                            && s.round == a.round
                            && s.prompt == a.prompt
                    }) {
                        // Decode speed is the engine's own measure (tokens differ when the texts differ); wall is
                        // what a person waits for.
                        if b.wall_ms < a.wall_ms {
                            wins += 1;
                        } else if b.wall_ms > a.wall_ms {
                            losses += 1;
                        }
                        if a.wall_ms > 0 {
                            ratios.push(b.wall_ms as f64 / a.wall_ms as f64);
                        }
                    }
                }
                pairs.push(
                    json!({"role": role, "baseline": base, "engine": other, "pairs": ratios.len(),
                    "faster": wins, "slower": losses, "wall_ratio_median": median(ratios),
                    "sign_test_p": sign_test_p(wins, losses)}),
                );
            }
        }
    }
    let loads: Vec<f64> = samples.iter().filter_map(|s| s.load1).collect();
    json!({"status": if done { "done" } else { "running" }, "kind": "engines", "request": req, "engines": engines,
        "results": {"per_engine": per, "paired": pairs,
            "host_load": {"min": loads.iter().cloned().reduce(f64::min), "max": loads.iter().cloned().reduce(f64::max),
                "note": "one-minute load average at each request; above about 2 on four cores the host was shared and absolute speeds are lower"}},
        "samples": samples})
}

/// Run the comparison. Engines that are not configured are skipped with a note; unreachable servers give failed rows.
pub async fn compare(
    orch: Arc<Orchestrator>,
    mut req: EngineCompareRequest,
    cancel: Arc<AtomicBool>,
    progress: impl Fn(&Value) + Send + Sync + 'static,
) -> Value {
    // The comparison talks to the engines directly, so the gateway would think it idle and start its idle-only work
    // (audits, label collection, the reasoning-cache builder) on the same cores. Count the run as a request in flight.
    struct Busy(Arc<std::sync::atomic::AtomicUsize>);
    impl Drop for Busy {
        fn drop(&mut self) {
            self.0.fetch_sub(1, Ordering::SeqCst);
        }
    }
    orch.inflight.fetch_add(1, Ordering::SeqCst);
    let _busy = Busy(orch.inflight.clone());
    req.rounds = req.rounds.clamp(1, 20);
    req.roles.retain(|r| r == "system1" || r == "system2");
    if req.roles.is_empty() {
        req.roles = d_roles();
    }
    let mut engines: Vec<String> = vec![];
    if req.engines.is_empty() {
        for (name, _, _, a, b) in orch.health().await {
            if a.ok && b.ok {
                engines.push(name);
            }
        }
    } else {
        for e in &req.engines {
            if orch.pair(e).is_some() && !e.contains('/') {
                engines.push(e.clone());
            }
        }
    }
    let prompts: Vec<EnginePrompt> = engine_prompts()
        .into_iter()
        .filter(|p| req.roles.iter().any(|r| r == p.role))
        .collect();
    let mut samples: Vec<EngineSample> = vec![];
    let t0 = Instant::now();
    'outer: for round in 0..req.rounds {
        // A B, then B A: neither engine always runs first (after the other has warmed the page cache or heated the CPU).
        let mut order = engines.clone();
        if round % 2 == 1 {
            order.reverse();
        }
        for p in &prompts {
            for e in &order {
                if cancel.load(Ordering::Relaxed) {
                    break 'outer;
                }
                let Some(pair) = orch.pair(e) else { continue };
                let b = if p.role == "system1" {
                    &pair.s1
                } else {
                    &pair.s2
                };
                let load1 = load_average();
                let res = one(b, p, &req).await;
                let s = match res {
                    Ok(o) => {
                        let (prefill, decode) = speeds(&o);
                        EngineSample {
                            round,
                            prompt: p.id.into(),
                            role: p.role.into(),
                            engine: e.clone(),
                            ok: true,
                            correct: answer_holds(&o.content, p.expect),
                            ttft_ms: o.ttft_ms,
                            wall_ms: o.wall_ms,
                            prompt_tokens: o.server_prompt_n.unwrap_or(o.prompt_tokens),
                            completion_tokens: o.completion_tokens,
                            prefill_tok_s: prefill,
                            decode_tok_s: decode,
                            finish: o.finish_reason,
                            load1,
                            error: None,
                        }
                    }
                    Err(err) => EngineSample {
                        round,
                        prompt: p.id.into(),
                        role: p.role.into(),
                        engine: e.clone(),
                        ok: false,
                        correct: false,
                        ttft_ms: None,
                        wall_ms: 0,
                        prompt_tokens: 0,
                        completion_tokens: 0,
                        prefill_tok_s: None,
                        decode_tok_s: None,
                        finish: String::new(),
                        load1,
                        error: Some(err),
                    },
                };
                samples.push(s);
                let mut v = summarize(&req, &engines, &samples, false);
                v["elapsed_s"] = json!(t0.elapsed().as_secs_f64());
                progress(&v);
            }
        }
    }
    let mut v = summarize(&req, &engines, &samples, true);
    v["elapsed_s"] = json!(t0.elapsed().as_secs_f64());
    if cancel.load(Ordering::Relaxed) {
        v["status"] = json!("cancelled");
    }
    if engines.is_empty() {
        v["note"] = json!("no engine has both servers running: start them (scripts/serve_smiley.sh start, scripts/serve_peregrine.sh start)");
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn answers_match_whole_words_only() {
        assert!(answer_holds("Answer: 6", &["6"]));
        assert!(!answer_holds("Answer: 36", &["6"]));
        assert!(answer_holds("The speed is 80 km/h.", &["80"]));
        assert!(answer_holds("Canberra.", &["canberra"]));
        assert!(answer_holds("**Na**", &["na"]));
        assert!(!answer_holds("Banana", &["na"]));
    }

    #[test]
    fn sign_test_is_two_sided_and_bounded() {
        assert!((sign_test_p(10, 0).unwrap() - 2.0 / 1024.0).abs() < 1e-12);
        assert_eq!(sign_test_p(3, 3).unwrap(), 1.0);
        assert!(sign_test_p(0, 0).is_none());
        let p = sign_test_p(9, 3).unwrap();
        assert!(p > 0.1 && p < 0.2, "{p}");
    }

    #[test]
    fn summary_pairs_same_prompt_and_round() {
        let req = EngineCompareRequest {
            engines: vec![],
            roles: vec!["system1".into()],
            rounds: 1,
            s1_max_tokens: 1,
            s2_max_tokens: 1,
            s2_thinking_budget: 1,
        };
        let mk = |e: &str, wall: u64| EngineSample {
            round: 0,
            prompt: "p".into(),
            role: "system1".into(),
            engine: e.into(),
            ok: true,
            correct: true,
            ttft_ms: Some(10),
            wall_ms: wall,
            prompt_tokens: 5,
            completion_tokens: 10,
            prefill_tok_s: Some(50.0),
            decode_tok_s: Some(20.0),
            finish: "stop".into(),
            load1: Some(1.0),
            error: None,
        };
        let v = summarize(
            &req,
            &["a".into(), "b".into()],
            &[mk("a", 1000), mk("b", 500)],
            true,
        );
        let p = &v["results"]["paired"][0];
        assert_eq!(p["faster"], 1);
        assert_eq!(p["wall_ratio_median"], 0.5);
        assert_eq!(v["results"]["per_engine"].as_array().unwrap().len(), 2);
    }
}
