//! Domain classification (gate feature). Verification inputs win; then the client hint; then text heuristics.
use nafs_types::{Domain, VerificationInputs};
use regex::Regex;
use std::sync::OnceLock;

pub fn classify(text: &str, hint: Option<Domain>, v: Option<&VerificationInputs>) -> Domain {
    if let Some(v) = v {
        if v.json_schema.is_some() {
            return Domain::Structured;
        }
        if v.tests.as_ref().map(|t| !t.is_empty()).unwrap_or(false) || v.test_code.is_some() {
            return Domain::Code;
        }
        // FR-090: a formal proof is mathematics (the maths caps: retries and a larger budget).
        if v.lean.is_some() {
            return Domain::Math;
        }
        if v.reference_answer
            .as_ref()
            .map(|r| !r.trim().is_empty())
            .unwrap_or(false)
        {
            return Domain::Math;
        }
    }
    if let Some(h) = hint {
        return h;
    }
    static CODE: OnceLock<Regex> = OnceLock::new();
    static MATH: OnceLock<Regex> = OnceLock::new();
    static JSON: OnceLock<Regex> = OnceLock::new();
    static FACT: OnceLock<Regex> = OnceLock::new();
    let t = text.to_lowercase();
    let code = CODE.get_or_init(|| Regex::new(r"```|\bdef \w+\(|\bfunction\b|\bpython\b|\brust\b|\bjavascript\b|\bimplement\b|\bcode\b|\bbug\b|\bcompile|\bregex\b|\bsql\b|\balgorithm\b").unwrap());
    let math = MATH.get_or_init(|| Regex::new(r"\d+\s*[-+*/x×^=]\s*\d+|\bhow (many|much)\b|\bcalculate\b|\bcompute\b|\bsolve\b|\bequation\b|\bpercent|\bprobability\b|\bintegral\b|\bderivative\b|\bsum of\b|\bminutes?\b.*\d|\d.*\bminutes?\b").unwrap());
    let json = JSON.get_or_init(|| Regex::new(r"\bjson\b|\bschema\b|\byaml\b").unwrap());
    let fact = FACT.get_or_init(|| Regex::new(r"^(who|when|where|which|what) (is|was|are|were|did)\b|\bcapital of\b|\bpopulation of\b|\bin what year\b").unwrap());
    if json.is_match(&t) {
        Domain::Structured
    } else if code.is_match(&t) {
        Domain::Code
    } else if math.is_match(&t) {
        Domain::Math
    } else if fact.is_match(t.trim()) {
        Domain::Factual
    } else {
        Domain::Open
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn lean_tasks_are_mathematics() {
        let v = VerificationInputs {
            lean: Some(nafs_types::LeanTask {
                statement: "theorem t : True :=\nsorry".into(),
                theorem: None,
            }),
            ..Default::default()
        };
        // Whatever the wording, a Lean task gets the maths caps, not the open-question ones.
        assert_eq!(
            classify("Tell me about patience", None, Some(&v)),
            Domain::Math
        );
    }

    #[test]
    fn heuristics() {
        assert_eq!(
            classify("Write a python function to reverse a list", None, None),
            Domain::Code
        );
        assert_eq!(
            classify("How many minutes from 3:40 pm to 6:05 pm?", None, None),
            Domain::Math
        );
        assert_eq!(
            classify("What is the capital of Indonesia?", None, None),
            Domain::Factual
        );
        assert_eq!(
            classify("Give me a JSON object for a user", None, None),
            Domain::Structured
        );
        assert_eq!(classify("Tell me about patience", None, None), Domain::Open);
    }
}
