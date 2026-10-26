//! nafs-verify: the Verifier Hub. Exact verifiers accept at threshold 1.0 (Theorem 1); a sandbox
//! failure (timeout, crash, limit) is a rejection with evidence, never a pass (spec 4.4).
use nafs_types::{Rates, VerificationInputs, VerifierResult};
use regex::Regex;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};
use std::time::Instant;

pub mod extract;
pub mod lean;
pub mod mutation;
pub mod sandbox;
pub use extract::{
    answer_similarity, extract_code, extract_final_answer, extract_json, latex_to_plain,
    normalize_answer,
};
pub use sandbox::{resolve_python, tail_chars, SandboxLimits, SandboxOutput};

#[derive(Debug, Clone, Default)]
pub struct VerifierHub {
    pub limits: SandboxLimits,
    pub numeric_tolerance: f64,
    /// Published (r, phi) per verifier from mutation testing (FR-032), attached to every verdict.
    pub rates: Arc<RwLock<HashMap<String, Rates>>>,
    /// Lean proof checking (FR-090); None when no Lean toolchain is configured.
    pub lean: Option<Arc<lean::LeanVerifier>>,
}

/// The sandboxed test runner. When an assert fails without a message (a bare `assert f(x) == y`), the
/// evidence adds the value the left side produced (`got 5`): concise execution feedback with the actual
/// value is what lets small models repair code (Self-Debugging, Chen et al. 2024). The verdict is unchanged.
const CODE_RUNNER: &str = r#"# -*- coding: utf-8 -*-
import sys, os, json, ast
_nafs_nonce = sys.stdin.readline().strip()
sys.stdin.close()
_d = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _d)
from solution import *
import solution as _nafs_solution
globals().update({k: v for k, v in vars(_nafs_solution).items() if not k.startswith('__')})
_items = json.load(open(os.path.join(_d, 'tests.json'), encoding='utf-8'))
def _nafs_got(src):
    try:
        t = ast.parse(src)
        if len(t.body) != 1 or not isinstance(t.body[0], ast.Assert):
            return ''
        e = t.body[0].test
        if isinstance(e, ast.Compare) and len(e.ops) == 1:
            e = e.left
        elif isinstance(e, ast.UnaryOp) and isinstance(e.op, ast.Not):
            e = e.operand
        return 'got %s' % repr(eval(compile(ast.Expression(e), 'got', 'eval'), globals()))[:160]
    except Exception:
        return ''
_passed, _fails = 0, []
for _i, _src in enumerate(_items):
    try:
        exec(compile(_src, 'test_%d' % (_i + 1), 'exec'), globals())
        _passed += 1
    except Exception as _e:
        _msg = str(_e)[:160] or (_nafs_got(_src) if isinstance(_e, AssertionError) else '')
        _fails.append('test %d: %s -> %s: %s' % (_i + 1, _src.strip()[:160], type(_e).__name__, _msg))
sys.stdout.write('\n%s credit %d/%d\n' % (_nafs_nonce, _passed, len(_items)))
sys.stdout.flush()
if _fails:
    sys.stderr.write('\n'.join(_fails[:6]) + '\n')
    sys.exit(1)
sys.stdout.write('\n' + _nafs_nonce + '\n')
sys.stdout.flush()
"#;

fn result(verifier: &str, passed: bool, evidence: Option<String>, t: Instant) -> VerifierResult {
    VerifierResult {
        verifier: verifier.into(),
        passed,
        score: Some(if passed { 1.0 } else { 0.0 }),
        evidence,
        rates: None,
        duration_ms: t.elapsed().as_millis() as u64,
        exact: true,
        details: None,
    }
}

impl VerifierHub {
    pub fn new(mut limits: SandboxLimits, numeric_tolerance: f64) -> Self {
        limits.python = resolve_python(&limits.python);
        limits.sympy_python = limits.sympy_python.as_deref().map(resolve_python);
        Self {
            limits,
            numeric_tolerance,
            rates: Arc::new(RwLock::new(HashMap::new())),
            lean: None,
        }
    }

    pub fn load_rates(&self, path: &str) {
        if let Ok(s) = std::fs::read_to_string(path) {
            if let Ok(m) = serde_json::from_str::<HashMap<String, Rates>>(&s) {
                *self.rates.write().unwrap() = m;
            }
        }
    }
    pub fn set_rates(&self, m: HashMap<String, Rates>) {
        *self.rates.write().unwrap() = m;
    }
    pub fn with_rates(&self, mut r: VerifierResult) -> VerifierResult {
        if r.rates.is_none() {
            r.rates = self.rates.read().unwrap().get(&r.verifier).cloned();
        }
        r
    }

    /// Run every exact verifier that the inputs enable; all must pass (conjunction).
    pub async fn verify(
        &self,
        answer: &str,
        inputs: &VerificationInputs,
    ) -> Option<VerifierResult> {
        let mut results = Vec::new();
        if inputs.has_code_tests() {
            results.push(self.with_rates(self.verify_code(answer, inputs).await));
        }
        if let Some(r) = inputs
            .reference_answer
            .as_deref()
            .filter(|r| !r.trim().is_empty())
        {
            results.push(self.with_rates(self.verify_math_full(answer, r).await));
        }
        if let Some(schema) = inputs.json_schema.as_ref() {
            results.push(self.with_rates(verify_json_schema(answer, schema)));
        }
        if let Some(task) = inputs.lean.as_ref() {
            results.push(self.with_rates(match &self.lean {
                Some(l) => l.verify(answer, task).await,
                // Fail closed: requests are refused earlier (validate); this is the last line.
                None => result(
                    "lean_proof",
                    false,
                    Some("No Lean toolchain is configured on this server.".into()),
                    Instant::now(),
                ),
            }));
        }
        if results.len() <= 1 {
            return results.pop();
        }
        let passed = results.iter().all(|r| r.passed);
        let evidence = results
            .iter()
            .filter(|r| !r.passed)
            .filter_map(|r| r.evidence.clone())
            .collect::<Vec<_>>()
            .join("\n");
        Some(VerifierResult {
            verifier: results
                .iter()
                .map(|r| r.verifier.as_str())
                .collect::<Vec<_>>()
                .join("+"),
            passed,
            score: Some(if passed { 1.0 } else { 0.0 }),
            evidence: if evidence.is_empty() {
                None
            } else {
                Some(evidence)
            },
            rates: None,
            duration_ms: results.iter().map(|r| r.duration_ms).sum(),
            exact: true,
            details: None,
        })
    }

    /// Numeric equality when both sides are plain numbers or simple fractions (GSM8K `#### 72` accepted);
    /// otherwise normalised string equality. Expressions are never reduced to their first digit.
    pub fn verify_math(&self, answer: &str, reference: &str) -> VerifierResult {
        let t = Instant::now();
        let got = extract_final_answer(answer).unwrap_or_default();
        let want = latex_to_plain(reference.rsplit("####").next().unwrap_or(reference).trim());
        match (plain_numeric(&got), plain_numeric(&want)) {
            (Some(a), Some(b)) => {
                let tol = self.numeric_tolerance.max(1e-9) * b.abs().max(1.0);
                let ok = (a - b).abs() <= tol;
                result("math_numeric", ok, (!ok).then(|| format!("The final answer {a} does not match the expected value. Re-check the computation and state the final answer as 'Answer: <number>'.")), t)
            }
            _ => {
                // Strict textual identity only (case, spaces, '*' and '$' ignored); equivalence of
                // different spellings is SymPy's job in verify_math_full.
                let canon = |x: &str| {
                    x.to_lowercase()
                        .chars()
                        .filter(|c| !c.is_whitespace() && *c != '*' && *c != '$')
                        .collect::<String>()
                        .trim_end_matches('.')
                        .to_string()
                };
                let ok = !got.is_empty() && canon(&got) == canon(&want);
                result("math_numeric", ok, (!ok).then(|| "The final answer does not match the expected answer. State the final answer on its own line as 'Answer: <value>'.".to_string()), t)
            }
        }
    }

    /// Numeric check first; when either side is not a plain number, SymPy equivalence (fractions,
    /// radicals, pi, simple expressions), run inside the sandbox because sympify evaluates code.
    pub async fn verify_math_full(&self, answer: &str, reference: &str) -> VerifierResult {
        let numeric = self.verify_math(answer, reference);
        let got = extract_final_answer(answer).unwrap_or_default();
        let want = latex_to_plain(reference.rsplit("####").next().unwrap_or(reference).trim());
        if numeric.passed
            || (plain_numeric(&got).is_some() && plain_numeric(&want).is_some())
            || got.is_empty()
        {
            return numeric;
        }
        let Some(py) = self.limits.sympy_python.clone() else {
            return numeric;
        };
        let t = Instant::now();
        match symbolic_equal(&got, &want, &py, &self.limits).await {
            Some(true) => result("math_symbolic", true, None, t),
            Some(false) => result("math_symbolic", false, Some("The final answer is not equivalent to the expected answer (symbolic check). Re-check and state it as 'Answer: <value>'.".into()), t),
            None => numeric,
        }
    }

    /// Execute the candidate's code against the tests in the sandbox. The runner reads a random nonce
    /// from stdin before importing the candidate and prints it only after every test passed (L10).
    pub async fn verify_code(&self, answer: &str, inputs: &VerificationInputs) -> VerifierResult {
        let t = Instant::now();
        let target = inputs.entry_point.clone().or_else(|| {
            inputs
                .tests
                .as_ref()
                .and_then(|ts| ts.iter().find_map(|s| target_name(s)))
        });
        let mut code =
            extract_code(answer, target.as_deref()).unwrap_or_else(|| answer.to_string());
        // M2: always prepend the HumanEval prefix (imports, helpers); a later model `def` overrides the stub.
        if let Some(prefix) = inputs.code_prefix.as_deref() {
            code = format!("{prefix}\n    pass\n\n{code}");
        }
        let nonce = format!("NAFS-OK-{}", uuid::Uuid::new_v4().simple());
        // The tests live in their own file and are compiled from bytes: Python 3.10's file tokenizer can
        // split a multi-byte character on very long lines (EvalPlus inputs) and report "Non-UTF-8 code".
        // Plain asserts (`tests`) run one by one, each compiled from its own string, so the verdict carries
        // per-assert credit for the repair search (Open Math Harness scores partial proofs the same way). The
        // credit line and the final line carry the run's nonce, which the candidate never sees on stdin.
        let runner = String::from(CODE_RUNNER);
        let mut items: Vec<String> = vec![];
        if let Some(tc) = inputs.test_code.as_deref() {
            let mut t = format!("{tc}\n");
            if let Some(ep) = inputs.entry_point.as_deref() {
                t.push_str(&format!("\ncheck({ep})\n"));
            }
            items.push(t);
        }
        if let Some(tests) = inputs.tests.as_ref() {
            items.extend(tests.iter().filter(|s| !s.trim().is_empty()).cloned());
        }
        let tests_json = serde_json::to_string(&items).unwrap_or_else(|_| "[]".into());
        let out = sandbox::run(
            &[
                ("solution.py", &code),
                ("runner.py", &runner),
                ("tests.json", &tests_json),
            ],
            "runner.py",
            Some(&format!("{nonce}\n")),
            &self.limits.python,
            &self.limits,
        )
        .await;
        let mut credit: Option<(u64, u64)> = None;
        let (passed, evidence) = match out {
            Ok(o) => {
                let ok = o.exit_ok
                    && o.stdout
                        .trim_end()
                        .lines()
                        .last()
                        .map(|l| l.trim() == nonce)
                        .unwrap_or(false);
                let tag = format!("{nonce} credit ");
                credit = o
                    .stdout
                    .lines()
                    .find_map(|l| l.trim().strip_prefix(tag.as_str()))
                    .and_then(|c| c.split_once('/'))
                    .and_then(|(p, t)| Some((p.trim().parse().ok()?, t.trim().parse().ok()?)));
                let mut ev = o
                    .excerpt(self.limits.max_output_bytes)
                    .lines()
                    .filter(|l| !l.trim().starts_with(tag.as_str()))
                    .collect::<Vec<_>>()
                    .join("\n")
                    .replace(&nonce, "");
                if let (false, Some((p, t))) = (ok, credit) {
                    ev = format!("{p} of {t} tests passed.\n{}", ev.trim());
                }
                (
                    ok,
                    (!ok).then(|| {
                        format!(
                            "Tests failed:\n{}",
                            if ev.trim().is_empty() {
                                "no output (a test did not complete)".into()
                            } else {
                                ev
                            }
                        )
                    }),
                )
            }
            Err(e) => (false, Some(format!("Tests could not run: {e}"))),
        };
        let mut r = result("code_tests", passed, evidence, t);
        if let Some((p, total)) = credit {
            r.details = Some(serde_json::json!({"tests_passed": p, "tests_total": total}));
        }
        r
    }
}

pub(crate) fn target_name(test: &str) -> Option<String> {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(
            r"assert\s+(?:not\s+)?(?:\(|set\(|math\.isclose\()?\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(",
        )
        .unwrap()
    })
    .captures(test)
    .map(|c| c[1].to_string())
    .filter(|n| {
        ![
            "abs",
            "set",
            "len",
            "sorted",
            "list",
            "tuple",
            "round",
            "str",
            "int",
            "float",
            "isinstance",
        ]
        .contains(&n.as_str())
    })
}

/// A plain number ("1,200", "$3.50", "-4", "25%") or a simple fraction ("3/4", "(1)/(2)"); else None.
pub fn plain_numeric(s: &str) -> Option<f64> {
    static NUM: OnceLock<Regex> = OnceLock::new();
    static FRAC: OnceLock<Regex> = OnceLock::new();
    let x = latex_to_plain(s)
        .replace(['$', ' '], "")
        .trim_end_matches(['.', '%'])
        .to_string();
    let num = NUM.get_or_init(|| Regex::new(r"^[-+]?\d[\d,]*(\.\d+)?$|^[-+]?\.\d+$").unwrap());
    if num.is_match(&x) {
        return x.replace(',', "").parse().ok();
    }
    let frac = FRAC.get_or_init(|| {
        Regex::new(r"^\(?([-+]?\d+(?:\.\d+)?)\)?/\(?([-+]?\d+(?:\.\d+)?)\)?$").unwrap()
    });
    let c = frac.captures(&x)?;
    let (a, b): (f64, f64) = (c[1].parse().ok()?, c[2].parse().ok()?);
    (b != 0.0).then(|| a / b)
}

async fn symbolic_equal(a: &str, b: &str, python: &str, limits: &SandboxLimits) -> Option<bool> {
    let payload = serde_json::json!({"a": a, "b": b}).to_string();
    let script = r#"
import sys, json, re
d = json.loads(sys.stdin.read())
from sympy import sympify, simplify, nsimplify
def norm(s):
    s = s.strip().strip('$').strip().rstrip('.')
    if '=' in s: s = s.split('=')[-1]
    s = s.replace('\\pi', 'pi').replace('π', 'pi').replace('√', 'sqrt').replace('^', '**').replace('×', '*').replace('·', '*')
    s = re.sub(r'(?<=\d),(?=\d{3}\b)', '', s)
    s = re.sub(r'(\d)\s*(pi|sqrt|\()', r'\1*\2', s)
    return s
try:
    x, y = sympify(norm(d['a'])), sympify(norm(d['b']))
    ok = bool(simplify(x - y) == 0) or abs(complex(x.evalf()) - complex(y.evalf())) < 1e-9
    print(json.dumps({'ok': ok}))
except Exception as e:
    print(json.dumps({'ok': None, 'error': str(e)[:200]}))
"#;
    let lim = SandboxLimits {
        wall_clock_s: 15,
        cpu_time_s: 10,
        memory_mb: 768,
        ..limits.clone()
    };
    let o = sandbox::run(
        &[("sym.py", script)],
        "sym.py",
        Some(&payload),
        python,
        &lim,
    )
    .await
    .ok()?;
    let v: serde_json::Value = serde_json::from_str(o.stdout.trim().lines().last()?).ok()?;
    v["ok"].as_bool()
}

pub fn verify_json_schema(answer: &str, schema: &serde_json::Value) -> VerifierResult {
    let t = Instant::now();
    let fail = |msg: String| result("json_schema", false, Some(msg), t);
    let Some(raw) = extract_json(answer) else {
        return fail(
            "No JSON value found in the answer. Reply with a single JSON value only.".into(),
        );
    };
    let instance: serde_json::Value = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(e) => return fail(format!("Invalid JSON: {e}")),
    };
    let validator = match jsonschema::validator_for(schema) {
        Ok(v) => v,
        Err(e) => return fail(format!("Invalid schema supplied: {e}")),
    };
    let errors: Vec<String> = validator
        .iter_errors(&instance)
        .take(3)
        .map(|e| format!("at '{}': {}", e.instance_path(), e))
        .collect();
    if errors.is_empty() {
        result("json_schema", true, None, t)
    } else {
        fail(format!("Schema validation failed: {}", errors.join("; ")))
    }
}

/// Parse a number from text: thousands separators, currency, percent, Unicode minus, simple fractions a/b.
pub fn parse_number(s: &str) -> Option<f64> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(r"-?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?|-?\.\d+").unwrap());
    let cleaned = latex_to_plain(s).replace(['$', '%', ' ', '(', ')'], "");
    if let Some((a, b)) = cleaned.split_once('/') {
        if let (Ok(a), Ok(b)) = (
            a.trim().replace(',', "").parse::<f64>(),
            b.trim().replace(',', "").parse::<f64>(),
        ) {
            if b != 0.0 {
                return Some(a / b);
            }
        }
    }
    let m = re.find(&cleaned)?;
    m.as_str().replace(',', "").parse::<f64>().ok()
}

#[cfg(test)]
mod tests;
