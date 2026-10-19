//! Mutation testing of the exact verifiers (FR-032, ER-003): known-correct answers measure the
//! true-accept rate r; deliberately wrong mutants measure the false-accept rate phi (Theorem 1).
//! Wrongness of every mutant is established by construction, never by the verifier under test.
use crate::VerifierHub;
use nafs_types::{Rates, VerificationInputs};
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize)]
pub struct Case {
    pub verifier: String,
    pub task: String,
    pub label: String,
    pub correct: bool,
    pub answer: String,
    #[serde(skip)]
    pub inputs: VerificationInputs,
}

fn fmt_num(v: f64) -> String {
    if (v - v.round()).abs() < 1e-9 {
        format!("{}", v.round() as i64)
    } else {
        format!("{}", (v * 1e6).round() / 1e6)
    }
}
fn commas(v: f64) -> String {
    let s = fmt_num(v);
    let (int, frac) = s
        .split_once('.')
        .map(|(a, b)| (a.to_string(), format!(".{b}")))
        .unwrap_or((s.clone(), String::new()));
    let neg = int.starts_with('-');
    let digits: Vec<char> = int.trim_start_matches('-').chars().collect();
    let mut out = String::new();
    for (i, c) in digits.iter().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(*c);
    }
    format!("{}{}{}", if neg { "-" } else { "" }, out, frac)
}

/// Math cases from (task id, numeric reference) pairs.
pub fn math_cases(refs: &[(String, f64)]) -> Vec<Case> {
    let mut out = vec![];
    for (id, v) in refs {
        let inputs = VerificationInputs {
            reference_answer: Some(fmt_num(*v)),
            ..Default::default()
        };
        let mk = |label: &str, correct: bool, answer: String| Case {
            verifier: "math".into(),
            task: id.clone(),
            label: label.into(),
            correct,
            answer,
            inputs: inputs.clone(),
        };
        let s = fmt_num(*v);
        out.push(mk(
            "answer_line",
            true,
            format!("Step 1: compute the parts.\nAnswer: {s}"),
        ));
        out.push(mk(
            "boxed",
            true,
            format!("We add 12 + 7 = 19 first, then finish.\nThe final answer is \\boxed{{{s}}}."),
        ));
        out.push(mk(
            "currency_commas",
            true,
            format!("So the total is ${}.\nAnswer: ${}", commas(*v), commas(*v)),
        ));
        out.push(mk(
            "equation_chain",
            true,
            format!("Answer: 2 * 3 + {s} - 6 = {s}"),
        ));
        out.push(mk("bare", true, s.clone()));
        let mut wrong = vec![
            ("plus_one", v + 1.0),
            ("minus_one", v - 1.0),
            ("times_ten", v * 10.0),
        ];
        if (v - v.round()).abs() < 1e-9 {
            wrong.push(("plus_half", v + 0.5));
        }
        if *v != 0.0 {
            wrong.push(("negated", -v));
            wrong.push(("halved", v / 2.0));
        }
        for (label, w) in wrong {
            if (w - v).abs() > 1e-9 {
                out.push(mk(
                    label,
                    false,
                    format!("Working it out step by step.\nAnswer: {}", fmt_num(w)),
                ));
            }
        }
        out.push(mk(
            "corrected_later_wrong",
            false,
            format!(
                "Answer: {s}\nWait, I made an error. Correcting it:\nAnswer: {}",
                fmt_num(v + 2.0)
            ),
        ));
    }
    out
}

fn mutate_code(sol: &str, entry: Option<&str>) -> Vec<(String, String)> {
    let mut m = vec![];
    for (label, from, to) in [
        ("plus_to_minus", " + ", " - "),
        ("minus_to_plus", " - ", " + "),
        ("lt_to_le", " < ", " <= "),
        ("le_to_lt", " <= ", " < "),
        ("gt_to_ge", " > ", " >= "),
        ("ge_to_gt", " >= ", " > "),
        ("eq_to_ne", " == ", " != "),
        ("ne_to_eq", " != ", " == "),
        ("true_to_false", "return True", "return False"),
        ("false_to_true", "return False", "return True"),
        ("range_shift", "range(", "range(1 + "),
    ] {
        // mutate the last occurrence (inside the solution body rather than the docstring/prompt)
        if let Some(i) = sol.rfind(from) {
            let mut s = sol.to_string();
            s.replace_range(i..i + from.len(), to);
            m.push((label.to_string(), s));
        }
    }
    if let Some(ep) = entry {
        let def = format!("def {ep}(");
        if let Some(i) = sol.find(&def) {
            let line_start = sol[..i].rfind('\n').map(|x| x + 1).unwrap_or(0);
            let indent: String = sol[line_start..i]
                .chars()
                .take_while(|c| c.is_whitespace())
                .collect();
            // end of the (possibly multi-line) signature: first ':' followed by newline after the def
            if let Some(rel) = sol[i..].find(":\n") {
                let at = i + rel + 2;
                let mut s = sol.to_string();
                s.insert_str(at, &format!("{indent}    return None\n"));
                m.push(("return_none".into(), s));
            }
        }
    }
    m
}

/// Code cases from (task id, inputs, canonical solution).
pub fn code_cases(items: &[(String, VerificationInputs, String)]) -> Vec<Case> {
    let mut out = vec![];
    for (id, inputs, sol) in items {
        let entry = inputs.entry_point.clone().or_else(|| {
            inputs
                .tests
                .as_ref()
                .and_then(|t| t.iter().find_map(|s| crate::target_name(s)))
        });
        let wrap = |code: &str| format!("```python\n{}\n```", code.trim_end());
        out.push(Case {
            verifier: "code_tests".into(),
            task: id.clone(),
            label: "canonical".into(),
            correct: true,
            answer: wrap(sol),
            inputs: inputs.clone(),
        });
        for (label, s) in mutate_code(sol, entry.as_deref()) {
            out.push(Case {
                verifier: "code_tests".into(),
                task: id.clone(),
                label,
                correct: false,
                answer: wrap(&s),
                inputs: inputs.clone(),
            });
        }
    }
    out
}

/// Schema cases from (task id, schema, a valid instance). Mutants violate constraints by construction.
pub fn schema_cases(items: &[(String, Value, Value)]) -> Vec<Case> {
    let mut out = vec![];
    for (id, schema, valid) in items {
        let inputs = VerificationInputs {
            json_schema: Some(schema.clone()),
            ..Default::default()
        };
        let mk = |label: &str, correct: bool, v: &Value, fenced: bool| Case {
            verifier: "json_schema".into(),
            task: id.clone(),
            label: label.into(),
            correct,
            answer: if fenced {
                format!(
                    "Here is the object:\n```json\n{}\n```",
                    serde_json::to_string_pretty(v).unwrap()
                )
            } else {
                v.to_string()
            },
            inputs: inputs.clone(),
        };
        out.push(mk("valid_fenced", true, valid, true));
        out.push(mk("valid_plain", true, valid, false));
        if let (Some(obj), Some(req)) = (valid.as_object(), schema["required"].as_array()) {
            for k in req.iter().filter_map(|k| k.as_str()).take(2) {
                let mut o = obj.clone();
                o.remove(k);
                out.push(mk(&format!("missing_{k}"), false, &Value::Object(o), true));
            }
            if let Some(k) = req.first().and_then(|k| k.as_str()) {
                let wrong = match &obj[k] {
                    Value::String(_) => json!(12345),
                    Value::Number(_) => json!("not a number"),
                    Value::Bool(_) => json!("yes"),
                    Value::Array(_) => json!({"x": 1}),
                    Value::Object(_) => json!([1]),
                    Value::Null => json!(1),
                };
                let mut o = obj.clone();
                o.insert(k.into(), wrong);
                out.push(mk(
                    &format!("wrong_type_{k}"),
                    false,
                    &Value::Object(o),
                    true,
                ));
            }
            if schema["additionalProperties"] == json!(false) {
                let mut o = obj.clone();
                o.insert("unexpected_field".into(), json!(true));
                out.push(mk("extra_field", false, &Value::Object(o), true));
            }
        }
        if let Some(arr) = valid.as_array() {
            let fixed_len = schema.get("const").is_some()
                || schema["minItems"].as_u64() == Some(arr.len() as u64);
            if fixed_len && !arr.is_empty() {
                let mut a = arr.clone();
                a.pop();
                out.push(mk("dropped_item", false, &Value::Array(a), true));
            }
            let mut a = arr.clone();
            a[0] = json!({"wrong": "type"});
            if schema.get("const").is_some()
                || schema["items"].get("type").is_some()
                || schema["items"].get("pattern").is_some()
            {
                out.push(mk("wrong_item_type", false, &Value::Array(a), true));
            }
        }
        out.push(
            mk(
                "not_json",
                false,
                &json!("I think the answer is an object with those fields."),
                false,
            )
            .with_answer("I think the answer is an object with those fields."),
        );
    }
    out
}
impl Case {
    fn with_answer(mut self, a: &str) -> Self {
        self.answer = a.into();
        self
    }
}

/// Exact two-sided Clopper-Pearson 95% interval for k successes in n trials.
pub fn clopper_pearson(k: u64, n: u64) -> (f64, f64) {
    if n == 0 {
        return (0.0, 1.0);
    }
    let cdf = |kk: i64, p: f64| -> f64 {
        // P(X <= kk), X ~ Bin(n, p), summed in log space
        if kk < 0 {
            return 0.0;
        }
        if kk as u64 >= n {
            return 1.0;
        }
        let (lp, lq) = (p.max(1e-300).ln(), (1.0 - p).max(1e-300).ln());
        let mut lc = 0.0f64;
        let mut sum = 0.0f64;
        for i in 0..=kk as u64 {
            if i > 0 {
                lc += ((n - i + 1) as f64).ln() - (i as f64).ln();
            }
            sum += (lc + i as f64 * lp + (n - i) as f64 * lq).exp();
        }
        sum.min(1.0)
    };
    // lower: largest p with P(X >= k) <= 0.025; upper: smallest p with P(X <= k) <= 0.025 (bisection)
    let lower = if k == 0 {
        0.0
    } else {
        let (mut lo, mut hi) = (0.0f64, 1.0f64);
        for _ in 0..80 {
            let mid = (lo + hi) / 2.0;
            if 1.0 - cdf(k as i64 - 1, mid) < 0.025 {
                lo = mid
            } else {
                hi = mid
            }
        }
        (lo + hi) / 2.0
    };
    let upper = if k == n {
        1.0
    } else {
        let mut lo = 0.0f64;
        let mut hi = 1.0f64;
        for _ in 0..80 {
            let mid = (lo + hi) / 2.0;
            if cdf(k as i64, mid) > 0.025 {
                lo = mid
            } else {
                hi = mid
            }
        }
        (lo + hi) / 2.0
    };
    (lower, upper)
}

/// Run cases with bounded concurrency; returns per-verifier rates and the raw outcomes.
pub async fn run(
    hub: &VerifierHub,
    cases: Vec<Case>,
    concurrency: usize,
    mut progress: impl FnMut(usize, usize),
) -> Value {
    use futures::future::join_all;
    let total = cases.len();
    let mut outcomes: Vec<(Case, bool, Option<String>)> = Vec::with_capacity(total);
    let mut done = 0usize;
    for chunk in cases.chunks(concurrency.max(1)) {
        let futs = chunk.iter().map(|c| async move {
            let v = hub.verify(&c.answer, &c.inputs).await;
            (
                c.clone(),
                v.as_ref().map(|r| r.passed).unwrap_or(false),
                v.and_then(|r| r.evidence),
            )
        });
        for o in join_all(futs).await {
            outcomes.push(o);
        }
        done += chunk.len();
        progress(done, total);
    }
    let mut per: BTreeMap<String, Value> = BTreeMap::new();
    let now = chrono_now();
    let mut rates: BTreeMap<String, Rates> = BTreeMap::new();
    for verifier in ["math", "code_tests", "json_schema"] {
        let rs: Vec<&(Case, bool, Option<String>)> = outcomes
            .iter()
            .filter(|o| o.0.verifier == verifier)
            .collect();
        if rs.is_empty() {
            continue;
        }
        let nc = rs.iter().filter(|o| o.0.correct).count() as u64;
        let ac = rs.iter().filter(|o| o.0.correct && o.1).count() as u64;
        let nw = rs.iter().filter(|o| !o.0.correct).count() as u64;
        let aw = rs.iter().filter(|o| !o.0.correct && o.1).count() as u64;
        let (r_lo, r_hi) = clopper_pearson(ac, nc);
        let (p_lo, p_hi) = clopper_pearson(aw, nw);
        let r = if nc > 0 { ac as f64 / nc as f64 } else { 0.0 };
        let phi = if nw > 0 { aw as f64 / nw as f64 } else { 0.0 };
        let mut by_label: BTreeMap<String, (u64, u64)> = BTreeMap::new();
        for o in &rs {
            let e = by_label.entry(o.0.label.clone()).or_default();
            e.0 += 1;
            e.1 += o.1 as u64;
        }
        per.insert(verifier.into(), json!({"n_correct": nc, "accepted_correct": ac, "n_wrong": nw, "accepted_wrong": aw, "r": r, "phi": phi,
            "r_ci95": [r_lo, r_hi], "phi_ci95": [p_lo, p_hi],
            "by_label": by_label.iter().map(|(k, (n, a))| (k.clone(), json!({"n": n, "accepted": a}))).collect::<BTreeMap<_, _>>(),
            "false_accepts": rs.iter().filter(|o| !o.0.correct && o.1).take(15).map(|o| json!({"task": o.0.task, "label": o.0.label, "answer": crate::tail_chars(&o.0.answer, 400)})).collect::<Vec<_>>(),
            "false_rejects": rs.iter().filter(|o| o.0.correct && !o.1).take(15).map(|o| json!({"task": o.0.task, "label": o.0.label, "evidence": o.2.as_deref().map(|e| crate::tail_chars(e, 400))})).collect::<Vec<_>>()}));
        let rec = Rates {
            r,
            phi,
            n_mutants: nw as u32,
            n_correct: nc as u32,
            phi_upper95: Some(p_hi),
            measured_at: Some(now.clone()),
        };
        // the math verifier reports under both names it can emit
        if verifier == "math" {
            rates.insert("math_numeric".into(), rec.clone());
            rates.insert("math_symbolic".into(), rec);
        } else {
            rates.insert(verifier.into(), rec);
        }
    }
    json!({"verifiers": per, "rates": rates, "total_cases": total, "measured_at": now})
}

fn chrono_now() -> String {
    let d = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("unix:{d}")
}
