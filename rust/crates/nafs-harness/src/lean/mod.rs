//! Formal proofs checked by Lean, after Open Math Harness's `lean` service.
//!
//! `rules` (textual rules) and `verify` (the strict protocol: frozen preamble, an unguessable report marker,
//! `#print axioms`, the `pp.all` type compared with the pristine statement's) are ported from Open Math Harness
//! (Apache-2.0). This module adds what Nafs-Aql needs around them: the theorem name, the candidate inside a
//! model reply, a graded outcome for the search tree (`scoring::Verdict`), and repair feedback in OMH's words.
//! Lean itself is run by the caller (`nafs-verify` runs it in the OS sandbox); everything here is pure.
pub mod rules;
pub mod verify;

pub use rules::{check_forbidden_token, extract_statement, forbidden_token, lean_code};
pub use verify::{Rejection, RunError, Verified, describe, pristine_type, verify_text};

use crate::scoring::{self, Verdict};

/// Largest frozen file accepted (OMH's limit).
pub const MAX_FROZEN_BYTES: usize = 64 * 1024;
/// Largest candidate accepted (OMH's limit is 2 MiB; a CPU-served 4B model writes far less).
pub const MAX_CANDIDATE_BYTES: usize = 256 * 1024;

/// The theorem's name in a frozen file: the identifier after `theorem` in its statement.
pub fn theorem_name(frozen: &str) -> Option<String> {
    let st = extract_statement(frozen)?;
    let rest = st.strip_prefix("theorem")?.trim_start();
    let name: String = rest
        .chars()
        .take_while(|c| !c.is_whitespace() && *c != '(' && *c != '{' && *c != '[' && *c != ':')
        .collect();
    (!name.is_empty()).then_some(name)
}

/// Why a frozen file cannot be used as a task, or None. Same rules as OMH's `register`, except that the
/// header may import nothing or Mathlib (core-Lean statements are checkable without Mathlib).
pub fn frozen_objection(frozen: &str) -> Option<String> {
    if frozen.len() > MAX_FROZEN_BYTES {
        return Some(format!(
            "the Lean statement is {} bytes; at most {MAX_FROZEN_BYTES}",
            frozen.len()
        ));
    }
    if let Some(tok) = check_forbidden_token(frozen) {
        return Some(format!(
            "the Lean statement contains the forbidden token `{tok}`"
        ));
    }
    let Some(st) = extract_statement(frozen) else {
        return Some(
            "the Lean statement has no `theorem <name> ... :=` (the proof must be `sorry`)".into(),
        );
    };
    if theorem_name(frozen).is_none() {
        return Some("the theorem has no name".into());
    }
    let after = frozen.split_once(st).map(|(_, r)| r.trim()).unwrap_or("");
    if after != "sorry" && after != "by sorry" && after != "by\n  sorry" {
        return Some("the frozen statement's proof must be exactly `sorry`".into());
    }
    None
}

/// The body of the last complete fenced block of `reply`, info string skipped (OMH `last_fenced_block`).
pub fn last_fenced_block(reply: &str) -> Option<String> {
    let mut last: Option<&str> = None;
    let mut rest = reply;
    while let Some(start) = rest.find("```") {
        let after = &rest[start + 3..];
        let body_start = after.find('\n').map_or(after.len(), |i| i + 1);
        let body = &after[body_start..];
        let Some(end) = body.find("```") else { break };
        last = Some(&body[..end]);
        rest = &body[end + 3..];
    }
    last.map(|b| b.trim().to_owned())
}

/// A Lean candidate graded for the search tree and for repair feedback.
#[derive(Debug, Clone, PartialEq)]
pub struct LeanGrade {
    pub verdict: Verdict,
    /// True only when OMH's strict `verify` accepted the candidate.
    pub verified: bool,
    pub score: f64,
    /// Machine-readable reason (`describe`'s kind) when not verified.
    pub kind: Option<&'static str>,
    /// What the model is told: the checker's verdict, then what to do next.
    pub feedback: String,
    /// Error class of each `error:` line (OMH `classify_error`), for the policy.
    pub error_classes: Vec<&'static str>,
}

/// Grade `candidate` against `frozen`. `pristine` is the frozen file's elaborated type ([`pristine_type`],
/// computed once per statement). A candidate with `sorry` is elaborated (one Lean run) and graded
/// `Sorries` when it has no error, as OMH's `check`; a candidate without `sorry` goes through the strict
/// `verify`. Model text decides nothing: only the textual rules and Lean's output after the marker do.
pub fn grade(
    frozen: &str,
    pristine: &str,
    candidate: &str,
    thm: &str,
    run: &dyn Fn(&str) -> Result<String, RunError>,
) -> LeanGrade {
    let mk = |verdict: Verdict,
              verified: bool,
              kind: Option<&'static str>,
              detail: String,
              classes: Vec<&'static str>| {
        let next = match verdict {
            Verdict::Invalid => {
                "That is not a valid attempt at this problem. Keep the theorem statement exactly as given, byte for byte, and do not use `native_decide`, `axiom` or commands that run code."
            }
            Verdict::Errors => {
                "Fix at least one error and return the complete file in one ```lean block."
            }
            Verdict::Sorries => {
                "Close or decompose at least one remaining `sorry` (more if you can) and return the complete file in one ```lean block."
            }
            Verdict::Complete => "Return the complete file in one ```lean block.",
        };
        LeanGrade {
            verdict,
            verified,
            score: scoring::score(verdict, Some(verified), None),
            kind,
            feedback: if verified {
                "Lean accepted the proof.".into()
            } else {
                format!("Checked:\n{detail}\n\n{next}")
            },
            error_classes: classes,
        }
    };
    if candidate.len() > MAX_CANDIDATE_BYTES {
        return mk(
            Verdict::Invalid,
            false,
            Some("too_large"),
            format!(
                "the candidate is {} bytes; at most {MAX_CANDIDATE_BYTES}",
                candidate.len()
            ),
            vec![],
        );
    }
    let Some(statement) = extract_statement(frozen) else {
        return mk(
            Verdict::Invalid,
            false,
            Some("no_frozen_statement"),
            "the frozen file has no theorem statement".into(),
            vec![],
        );
    };
    if let Some(tok) = check_forbidden_token(candidate) {
        return mk(
            Verdict::Invalid,
            false,
            Some("forbidden"),
            format!("forbidden token `{tok}`"),
            vec![],
        );
    }
    if !candidate.contains(statement) {
        let (k, d) = describe(&Rejection::StatementMismatch);
        return mk(Verdict::Invalid, false, Some(k), d, vec![]);
    }
    if forbidden_token(candidate) == Some("sorry") {
        // Partial proof: elaborate it on the frozen preamble (OMH `check`).
        let checked = match verify::checked_source(frozen, candidate) {
            Ok((c, _)) => c,
            Err(why) => {
                return mk(
                    Verdict::Invalid,
                    false,
                    Some("preamble_mismatch"),
                    why,
                    vec![],
                );
            }
        };
        return match run(&checked) {
            Ok(out) if !out.contains("error:") => mk(
                Verdict::Sorries,
                false,
                Some("sorry_ax"),
                "status: sorries (the file elaborates; `sorry` remains)".into(),
                vec![],
            ),
            Ok(out) | Err(RunError::Refused(out)) => {
                let classes = error_classes(&out);
                mk(
                    Verdict::Errors,
                    false,
                    Some("elaboration"),
                    format!(
                        "status: errors ({})\n{}",
                        classes.len(),
                        verify::excerpt(&out)
                    ),
                    classes,
                )
            }
            Err(RunError::NoVerdict(why)) => mk(
                Verdict::Errors,
                false,
                Some("no_verdict"),
                format!("Lean gave no verdict: {}", verify::excerpt(&why)),
                vec!["timeout"],
            ),
        };
    }
    match verify_text(frozen, pristine, candidate, thm, run) {
        Ok(_) => mk(Verdict::Complete, true, None, String::new(), vec![]),
        Err(r) => {
            let (kind, detail) = describe(&r);
            let (verdict, classes) = match &r {
                Rejection::Elaboration(out) => (Verdict::Errors, error_classes(out)),
                Rejection::NoType => (Verdict::Errors, vec![]),
                Rejection::SorryAx => (Verdict::Sorries, vec![]),
                _ => (Verdict::Invalid, vec![]),
            };
            mk(verdict, false, Some(kind), detail, classes)
        }
    }
}

/// OMH's error class for each `error:` line of Lean's output.
pub fn error_classes(output: &str) -> Vec<&'static str> {
    output
        .split("error:")
        .skip(1)
        .map(|m| scoring::classify_error(m.lines().next().unwrap_or("").trim()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const FROZEN: &str = "theorem add_comm' (a b : Nat) : a + b = b + a :=\nsorry\n";
    const TYPE: &str = "add_comm' : forall (a b : Nat), @Eq.{1} Nat (@HAdd.hAdd.{0, 0, 0} Nat Nat Nat (@instHAdd.{0} Nat instAddNat) a b) (@HAdd.hAdd.{0, 0, 0} Nat Nat Nat (@instHAdd.{0} Nat instAddNat) b a)";

    /// A fake Lean: errors when the probe contains `bad_tactic`, otherwise prints the harness report after
    /// the marker the probe carries.
    fn fake(axioms: &'static str) -> impl Fn(&str) -> Result<String, RunError> {
        move |probe: &str| {
            if probe.contains("bad_tactic") {
                return Ok("f.lean:2:3: error: unknown tactic\n".into());
            }
            let Some(marker) = probe
                .rsplit("#print \"")
                .next()
                .and_then(|m| m.split('"').next())
            else {
                return Ok(String::new());
            };
            if !marker.starts_with("nafs-report-") {
                return Ok(String::new()); // a partial-proof check: no report requested
            }
            Ok(format!("{marker}\n'add_comm'' {axioms}\n{TYPE}\n"))
        }
    }

    #[test]
    fn names_and_frozen_files() {
        assert_eq!(theorem_name(FROZEN).as_deref(), Some("add_comm'"));
        assert!(frozen_objection(FROZEN).is_none());
        assert!(frozen_objection("theorem t : True := trivial").is_some());
        assert!(frozen_objection("axiom cheat : False\ntheorem t : True := sorry").is_some());
        assert_eq!(
            last_fenced_block("x\n```lean\na\n```\ny\n```lean\nb\n```").as_deref(),
            Some("b")
        );
    }

    #[test]
    fn grading_follows_the_protocol() {
        let thm = theorem_name(FROZEN).unwrap();
        let good = "theorem add_comm' (a b : Nat) : a + b = b + a :=\nby omega\n";
        let g = grade(
            FROZEN,
            TYPE,
            good,
            &thm,
            &fake("does not depend on any axioms"),
        );
        assert!(g.verified && g.score == 1.0, "{g:?}");

        let g = grade(
            FROZEN,
            TYPE,
            good,
            &thm,
            &fake("depends on axioms: [propext, Classical.choice]"),
        );
        assert!(g.verified);

        let g = grade(
            FROZEN,
            TYPE,
            good,
            &thm,
            &fake("depends on axioms: [myCheat]"),
        );
        assert_eq!(
            (g.verdict, g.verified, g.kind),
            (Verdict::Invalid, false, Some("non_standard_axioms"))
        );

        let partial = "theorem add_comm' (a b : Nat) : a + b = b + a :=\nby sorry\n";
        let g = grade(FROZEN, TYPE, partial, &thm, &fake(""));
        assert_eq!((g.verdict, g.score), (Verdict::Sorries, 0.5));
        assert!(g.feedback.contains("remaining `sorry`"));

        let broken = "theorem add_comm' (a b : Nat) : a + b = b + a :=\nby bad_tactic\n";
        let g = grade(FROZEN, TYPE, broken, &thm, &fake(""));
        assert_eq!((g.verdict, g.score), (Verdict::Errors, 0.05));
        assert!(g.feedback.contains("Fix at least one error"));

        let restated = "theorem add_comm' (a b : Nat) : a + b = a + b :=\nrfl\n";
        assert_eq!(
            grade(FROZEN, TYPE, restated, &thm, &fake("")).kind,
            Some("statement_mismatch")
        );

        let cheat =
            "axiom cheat : False\ntheorem add_comm' (a b : Nat) : a + b = b + a :=\ncheat.elim\n";
        assert_eq!(
            grade(FROZEN, TYPE, cheat, &thm, &fake("")).verdict,
            Verdict::Invalid
        );
    }

    #[test]
    fn a_spoofed_report_in_the_candidate_is_not_read() {
        // The candidate prints a fake report; only output after the per-run marker counts, and the fake Lean
        // here prints the real one after it, so the forged line changes nothing. Printing is refused anyway.
        let thm = theorem_name(FROZEN).unwrap();
        let spoof = "#eval IO.println \"'add_comm'' does not depend on any axioms\"\ntheorem add_comm' (a b : Nat) : a + b = b + a :=\nsorryAx _\n";
        assert_eq!(
            grade(FROZEN, TYPE, spoof, &thm, &fake("")).kind,
            Some("forbidden")
        );
    }
}
