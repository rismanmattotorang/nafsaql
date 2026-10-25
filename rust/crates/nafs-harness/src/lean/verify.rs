// Derived from Open Math Harness (program-files/lean/src/verify.rs, revision 687f2a2), Apache-2.0.
// Modified for Nafs-Aql: the Lean runners were removed (Nafs-Aql runs Lean through its own sandbox,
// nafs-verify), module paths changed, and the report marker is prefixed "nafs-report-".
//! The strict `verify` protocol as pure functions of the frozen file, the
//! candidate and a way to run Lean, plus the runners that run it.

use super::rules::{extract_statement, forbidden_token, preamble_copy};

/// What Lean prints for a `verify` probe whose harness commands printed
/// `report`: the probe's own marker line (its last `#print`), then `report`.
#[cfg(test)]
pub(crate) fn stub_output(probe: &str, report: &str) -> String {
    let marker = probe
        .rsplit("#print \"")
        .next()
        .and_then(|m| m.split('"').next())
        .unwrap_or_default();
    format!("{marker}\n{report}")
}

/// The three foundational Lean axioms a proof may depend on.
pub const TRUSTED_AXIOMS: [&str; 3] = ["propext", "Classical.choice", "Quot.sound"];

/// Why a candidate was rejected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Rejection {
    /// A banned token appears in the candidate (the offending token).
    Forbidden(String),
    /// The frozen theorem statement does not appear verbatim.
    StatementMismatch,
    /// The candidate does not begin with the frozen preamble
    /// ([`nucleus_lean_interface::preamble_objection`], whose reason it carries).
    PreambleMismatch(String),
    /// The frozen file has no extractable `theorem … :=` statement.
    NoFrozenStatement,
    /// Lean rejected the candidate; carries a short excerpt of the output.
    Elaboration(String),
    /// The proof depends on `sorryAx` (an unfinished proof).
    SorryAx,
    /// The proof depends on axioms outside [`TRUSTED_AXIOMS`].
    NonStandardAxioms(Vec<String>),
    /// The candidate theorem's fully-elaborated type (`pp.all`) is not
    /// character-identical to the pristine statement's: the surface text is
    /// kept but elaborates to something else.
    TypeMismatch,
    /// Lean printed no `<thm> : <type>` block for the probe.
    NoType,
}

/// A candidate that passed every check, with the axioms it depends on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Verified {
    pub axioms: Vec<String>,
}

pub type Verdict = Result<Verified, Rejection>;

/// Why a Lean run has no clean output.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RunError {
    /// Lean ran to the end and refused the file: Lean's own output, its
    /// verdict on the file.
    Refused(String),
    /// Lean gave no verdict: the probe could not be written, Lean could not
    /// be started or waited for, it outlived the deadline, or a signal ended
    /// it. Nothing is known about the file, and the run may be tried again.
    NoVerdict(String),
}

impl RunError {
    /// Lean's output, or why there is none.
    #[must_use]
    pub fn text(&self) -> &str {
        match self {
            Self::Refused(text) | Self::NoVerdict(text) => text,
        }
    }
}

/// Parse one `#print axioms` line: `(uses_sorry_ax, axioms)`.
///
/// The line looks like `'thm' depends on axioms: [propext, Classical.choice]`,
/// or Lean prints that a declaration does not depend on any axioms. Pass only
/// the line the harness's own command printed ([`harness_report`]), never
/// the whole output: the candidate's commands print before it.
#[must_use]
pub fn parse_axioms(output: &str) -> (bool, Vec<String>) {
    let uses_sorry = output.contains("sorryAx");
    let axioms = output
        .find("depends on axioms: [")
        .and_then(|i| {
            let rest = output.get(i + "depends on axioms: [".len()..)?;
            let close = rest.find(']')?;
            rest.get(..close)
        })
        .map(|inner| {
            inner
                .split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_owned)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    (uses_sorry, axioms)
}

/// Axioms used that are not trusted.
#[must_use]
pub fn non_standard(axioms: &[String]) -> Vec<String> {
    axioms
        .iter()
        .filter(|a| !TRUSTED_AXIOMS.contains(&a.as_str()))
        .cloned()
        .collect()
}

/// The `set_option pp.all true in #check @<thm>` line appended to probes. The
/// check is placed last so its multi-line output runs to the end of Lean's
/// output and can be sliced off without parsing message boundaries.
fn check_line(thm: &str) -> String {
    format!("set_option pp.all true in #check @{thm}\n")
}

/// Whether `text` starts with the head [`check_line`]'s output has:
/// `<thm> : `, prefixed by `@` when the statement has implicit binders and
/// with `.{<universes>}` after the name when it is universe-polymorphic.
/// Types are compared with their head, so a pristine and a candidate type
/// match only when Lean printed them the same way.
pub(crate) fn starts_with_type_head(text: &str, thm: &str) -> bool {
    let Some(rest) = text.strip_prefix('@').unwrap_or(text).strip_prefix(thm) else {
        return false;
    };
    let rest = match rest.strip_prefix(".{") {
        Some(levels) => match levels.split_once('}') {
            // Lean wraps a long universe list across lines.
            Some((names, after))
                if names
                    .chars()
                    .all(|c| c.is_alphanumeric() || c == '_' || c == ',' || c.is_whitespace()) =>
            {
                after
            }
            _ => return false,
        },
        None => rest,
    };
    rest.starts_with(" : ")
}

/// What `verify` appends to the file it elaborates, after the whole file:
/// the per-run `marker`, `#print axioms <thm>` and the `pp.all` check. Only
/// what follows the marker is read ([`harness_report`]).
fn report_commands(marker: &str, thm: &str) -> String {
    format!(
        "\n#print \"{marker}\"\n#print axioms {thm}\n{}",
        check_line(thm)
    )
}

/// The pristine statement's fully-elaborated type, computed **once per frozen
/// file**: one Lean run of the file `verify` builds with the frozen file as
/// its own candidate ([`checked_source`], proof still `sorry`), with
/// [`report_commands`] appended, exactly as `verify` runs a candidate. So the
/// frozen file's own commands cannot stand in for the report, and a file
/// `register` accepts is one `verify` can run: a preamble that cannot be
/// followed by the separator fails here. [`RunError::Refused`] carries Lean's
/// output (an error, a failed exit, or no well-formed report), or why
/// `verify` cannot read the file's preamble, and `register` refuses the file
/// on it; a [`RunError::NoVerdict`] from `run` is passed on. Pass the type
/// to [`verify_text`] for every candidate of that file.
pub fn pristine_type(
    frozen: &str,
    thm: &str,
    run: &dyn Fn(&str) -> Result<String, RunError>,
) -> Result<String, RunError> {
    let (checked, _) = checked_source(frozen, frozen).map_err(RunError::Refused)?;
    let marker = fresh_marker();
    let output = run(&format!("{checked}{}", report_commands(&marker, thm)))?;
    if output.contains("error:") {
        return Err(RunError::Refused(output));
    }
    match harness_report(&output, &marker, thm) {
        Some((_, pristine)) => Ok(pristine.to_owned()),
        None => Err(RunError::Refused(format!(
            "{output}\n[lean service] Lean printed no `#print axioms {thm}` and `#check @{thm}` report after the \
             file"
        ))),
    }
}

/// The complete command `verify` puts right after the frozen preamble. Lean
/// parses what follows it as a new command, so nothing a candidate writes
/// can continue the preamble's last declaration (a `protected` field added
/// to a `structure`, for one); it only prints a line.
pub const SEPARATOR: &str = "#print \"nucleus: the frozen preamble ends here\"";

/// The file `verify` elaborates for `candidate`, and the candidate's text
/// after its copy of the preamble: `frozen`'s own preamble through its last
/// character of code, [`SEPARATOR`] on that line, then the candidate from
/// the end of its copy on ([`preamble_copy`]). Blank lines follow the
/// separator when the candidate's copy is longer than the frozen text, so
/// Lean's line numbers are the candidate's own whenever the copy is not
/// shorter. `Err`: why the candidate does not keep the preamble.
pub fn checked_source<'a>(frozen: &str, candidate: &'a str) -> Result<(String, &'a str), String> {
    let Some(copy) = preamble_copy(frozen, candidate)? else {
        return Ok((candidate.to_owned(), candidate));
    };
    let (Some(preamble), Some(rest)) = (
        frozen.get(..copy.frozen_end),
        candidate.get(copy.candidate_end..),
    ) else {
        return Err("the frozen preamble could not be located".to_owned());
    };
    let lines = |text: &str| text.matches('\n').count();
    let padding = lines(&candidate[..copy.candidate_end]).saturating_sub(lines(preamble));
    Ok((
        format!("{preamble} {SEPARATOR}{}{rest}", "\n".repeat(padding)),
        rest,
    ))
}

/// Where [`SEPARATOR`] begins in every file [`checked_source`] builds from
/// `frozen`, as Lean reports a position (the 1-based line, the 0-based
/// column in characters): right after the preamble's last character of
/// code, which is also where Lean puts a parse error at the separator.
/// `None` when there is no separator.
#[allow(dead_code)]
pub(crate) fn separator_position(frozen: &str) -> Option<(usize, usize)> {
    let before = frozen.get(..preamble_copy(frozen, frozen).ok().flatten()?.frozen_end)?;
    let line_start = before.rfind('\n').map_or(0, |i| i + 1);
    Some((
        before.matches('\n').count() + 1,
        before[line_start..].chars().count(),
    ))
}

/// Decide a candidate, given the frozen file's [`pristine_type`] and a way
/// to run Lean. `run` receives a probe source ([`checked_source`], then
/// [`report_commands`]) and returns Lean's combined output after a clean
/// exit, else [`RunError`]. A run with no clean output rejects the
/// candidate, whatever the reason: a proof that elaborates past the deadline
/// is a failed attempt.
pub fn verify_text(
    frozen: &str,
    pristine: &str,
    candidate: &str,
    thm: &str,
    run: &dyn Fn(&str) -> Result<String, RunError>,
) -> Verdict {
    if let Some(tok) = forbidden_token(candidate) {
        return Err(Rejection::Forbidden(tok.to_owned()));
    }
    let Some(statement) = extract_statement(frozen) else {
        return Err(Rejection::NoFrozenStatement);
    };
    if !candidate.contains(statement) {
        return Err(Rejection::StatementMismatch);
    }
    let (checked, rest) = checked_source(frozen, candidate).map_err(Rejection::PreambleMismatch)?;
    if !rest.contains(statement) {
        return Err(Rejection::StatementMismatch);
    }
    if let Some(tok) = forbidden_token(&checked) {
        return Err(Rejection::Forbidden(tok.to_owned()));
    }
    let marker = fresh_marker();
    let probe = format!("{checked}{}", report_commands(&marker, thm));
    let output = match run(&probe) {
        Ok(out) => out,
        Err(why) => return Err(Rejection::Elaboration(excerpt(why.text()))),
    };
    if output.contains("error:") {
        return Err(Rejection::Elaboration(excerpt(&output)));
    }
    let Some((axiom_line, candidate_type)) = harness_report(&output, &marker, thm) else {
        return Err(Rejection::NoType);
    };
    let (uses_sorry, axioms) = parse_axioms(axiom_line);
    if uses_sorry {
        return Err(Rejection::SorryAx);
    }
    let extra = non_standard(&axioms);
    if !extra.is_empty() {
        return Err(Rejection::NonStandardAxioms(extra));
    }
    if candidate_type != pristine.trim() {
        return Err(Rejection::TypeMismatch);
    }
    Ok(Verified { axioms })
}

/// A per-run marker the candidate cannot predict: the probe prints it just
/// before the harness's own commands, so only what follows it is theirs.
/// `RandomState` keys come from the OS's randomness, so its hashes are
/// unguessable without a new dependency.
fn fresh_marker() -> String {
    use std::hash::{BuildHasher, Hasher};
    let mut marker = String::from("nafs-report-");
    for salt in 0..2u8 {
        let mut h = std::collections::hash_map::RandomState::new().build_hasher();
        h.write_u8(salt);
        h.write_u128(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos()),
        );
        marker.push_str(&format!("{:016x}", h.finish()));
    }
    marker
}

/// What the harness's own `#print axioms` and `#check` printed: the text
/// after the one line that is exactly `marker`, which must be one axioms line
/// for `'thm'` followed by the `#check` type block and nothing else. `None`
/// when the marker is missing or repeated or the report has any other shape:
/// output the candidate controls (its own commands run first) is never read.
pub fn harness_report<'a>(output: &'a str, marker: &str, thm: &str) -> Option<(&'a str, &'a str)> {
    let line = format!("{marker}\n");
    let start = if output.starts_with(&line) {
        0
    } else {
        output.find(&format!("\n{line}"))? + 1
    };
    if output.matches(marker).count() != 1 {
        return None;
    }
    let report = output.get(start + line.len()..)?;
    let (axiom_line, rest) = report.split_once('\n')?;
    let quoted = format!("'{thm}' ");
    let well_formed = axiom_line.strip_prefix(&quoted).is_some_and(|tail| {
        tail == "does not depend on any axioms"
            || tail
                .strip_prefix("depends on axioms: [")
                .is_some_and(|l| l.ends_with(']') && !l.contains('['))
    });
    if !well_formed || !starts_with_type_head(rest, thm) {
        return None;
    }
    Some((axiom_line, rest.trim()))
}

pub(crate) fn excerpt(s: &str) -> String {
    s.chars().take(800).collect()
}

/// Lean's combined `output` after it exited with `status`: clean only on a
/// success. A non-zero exit is a failed check even if no `error:` line was
/// captured, and never clean; it is Lean's verdict when Lean chose the exit
/// code, and none when a signal ended the process (the OOM killer, for one).
#[allow(dead_code)]
fn exit_verdict(status: std::process::ExitStatus, output: String) -> Result<String, RunError> {
    if status.success() {
        Ok(output)
    } else if status.code().is_some() {
        Err(RunError::Refused(format!(
            "lean exited with {status}:\n{output}"
        )))
    } else {
        Err(RunError::NoVerdict(format!(
            "lean was ended by {status}:\n{output}"
        )))
    }
}

/// A real Lean runner for the real-Lean tests: writes the probe to a temp file
/// and runs `lake env lean` in `pb_lean_dir` (the PutnamBench `lean4`
/// directory).
/// A stable machine-readable kind and a human/model-readable detail.
pub fn describe(r: &Rejection) -> (&'static str, String) {
    match r {
        Rejection::Forbidden(tok) => ("forbidden", format!("forbidden token `{tok}`")),
        Rejection::StatementMismatch => (
            "statement_mismatch",
            "the frozen theorem statement must appear verbatim; copy it from the problem file rather than \
             retyping it"
                .to_owned(),
        ),
        Rejection::PreambleMismatch(why) => ("preamble_mismatch", why.clone()),
        Rejection::NoFrozenStatement => {
            ("no_frozen_statement", "the frozen file has no theorem statement".to_owned())
        }
        Rejection::Elaboration(out) => ("elaboration", out.clone()),
        Rejection::SorryAx => ("sorry_ax", "the proof depends on sorryAx".to_owned()),
        Rejection::NonStandardAxioms(ax) => {
            ("non_standard_axioms", format!("non-standard axioms: {}", ax.join(", ")))
        }
        Rejection::TypeMismatch => {
            ("type_mismatch", "the elaborated type differs from the pristine statement's".to_owned())
        }
        Rejection::NoType => ("no_type", "Lean printed no type for the theorem".to_owned()),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::super::rules::{check_forbidden_token, lean_code};
    use super::*;

    const FROZEN: &str = "import Mathlib\n\ntheorem demo_thm (n : Nat) : n + 0 = n :=\nsorry\n";

    #[test]
    fn extract_statement_takes_theorem_up_to_the_proof_assignment() {
        assert_eq!(
            extract_statement(FROZEN),
            Some("theorem demo_thm (n : Nat) : n + 0 = n :=")
        );
        assert_eq!(extract_statement("no theorem here"), None);
    }

    #[test]
    fn extract_statement_ignores_the_word_theorem_inside_a_docstring() {
        // A docstring's prose must not anchor the statement.
        let frozen = "import Mathlib\n\n/--\nBy the fundamental theorem of algebra, show P.\n-/\ntheorem t (n : Nat) : n = n :=\nsorry\n";
        assert_eq!(
            extract_statement(frozen),
            Some("theorem t (n : Nat) : n = n :=")
        );
    }

    #[test]
    fn forbidden_tokens_are_caught() {
        assert_eq!(forbidden_token("... := sorry"), Some("sorry"));
        assert_eq!(forbidden_token("by native_decide"), Some("native_decide"));
        // Comments are not code: a proof may say so without being unfinished.
        assert_eq!(
            forbidden_token("theorem t : True := by\n  -- no sorry left\n  trivial"),
            None
        );
        assert_eq!(
            forbidden_token("/- was: sorry /- nested sorry -/ -/\ntheorem t : True := trivial"),
            None
        );
        assert_eq!(
            forbidden_token("/- comment -/ theorem t : True := sorry"),
            Some("sorry")
        );
        let text = "a -- b\nc /- d\ne -/ f \"g\"";
        assert_eq!(lean_code(text).unwrap().len(), text.len(), "nothing moves");
        // What `check` refuses (so `verify` too): a user axiom however it is
        // dressed, anything that runs code while Lean elaborates, and any
        // literal the textual check could read differently from Lean.
        let ambiguous = Some("string or character literal the checker cannot read unambiguously");
        let rows: &[(&str, Option<&str>)] = &[
            (
                "private axiom cheat : False\n#eval IO.println \"'demo_thm' depends on axioms: [propext]\"\n\
                 theorem demo_thm (n : Nat) : n + 1 = n := cheat.elim",
                Some("axiom"),
            ),
            ("axiom cheat : False", Some("axiom")),
            ("@[simp] axiom cheat : False", Some("axiom")),
            ("/-- doc -/ axiom cheat : False", Some("axiom")),
            (
                "noncomputable\n  protected axiom cheat : False",
                Some("axiom"),
            ),
            (
                "#eval IO.println \"'t' depends on axioms: [propext]\"",
                Some("#eval"),
            ),
            ("#eval! (1 : Nat)", Some("#eval")),
            ("run_cmd pure ()", Some("run_cmd")),
            ("elab \"x\" : term => pure default", Some("elab")),
            ("open Lean in\ntheorem t : True := trivial", Some("Lean")),
            ("def f := «IO».println", Some("IO")),
            (
                "set_option pp.explicit false in\ntheorem t : True := trivial",
                Some("set_option pp"),
            ),
            (
                "set_option debug.skipKernelTC true",
                Some("set_option debug"),
            ),
            ("def s := r\"x\"", ambiguous),
            ("def s := s!\"{1}\"", ambiguous),
            ("def c := '\"' #eval 1 -- \"", ambiguous),
            ("def c := '«' \"»\" #eval 1 \" + \" -- \"", ambiguous),
            ("def s := \"unterminated", ambiguous),
            // Legitimate proofs: the words in comments and strings, primes,
            // `∑'`, the header import, `set_option` of an ordinary option.
            (
                "theorem t : True := by\n  -- an axiom, #eval and elab in a comment\n  trivial",
                None,
            ),
            (
                "example : \"axiom #eval Lean\".length = 16 := by decide",
                None,
            ),
            (
                "import Mathlib\nimport Mathlib.Tactic\nopen Real\nset_option maxHeartbeats 400000 in\n",
                None,
            ),
            (
                "example (h' : 1 = 1) : ∑' _n : ℕ, (0 : ℝ) = 0 := by simp",
                None,
            ),
            ("theorem t : True := by sorry", None),
        ];
        for &(candidate, expected) in rows {
            assert_eq!(check_forbidden_token(candidate), expected, "{candidate}");
        }
    }

    #[test]
    fn axioms_are_parsed_and_classified() {
        let (s1, a1) = parse_axioms("'demo_thm' depends on axioms: [propext, Classical.choice]");
        assert!(!s1);
        assert_eq!(a1, ["propext", "Classical.choice"]);
        assert!(non_standard(&a1).is_empty());
        let (s2, a2) = parse_axioms("'t' depends on axioms: [sorryAx]");
        assert!(s2);
        assert_eq!(non_standard(&a2), ["sorryAx"]);
        let (_, a3) = parse_axioms("'t' does not depend on any axioms");
        assert!(a3.is_empty());
    }

    const TYPE: &str = "demo_thm : ∀ (n : Nat), @Eq.{1} Nat (@HAdd.hAdd.{0, 0, 0} Nat Nat Nat (@instHAdd.{0} Nat instAddNat) n (@OfNat.ofNat.{0} Nat (nat_lit 0) (instOfNatNat (nat_lit 0)))) n";

    /// Lean output for a clean candidate probe: axioms line, then the pp.all check.
    fn clean_output(probe: &str, ty: &str) -> String {
        stub_output(
            probe,
            &format!("'demo_thm' depends on axioms: [propext]\n{ty}\n"),
        )
    }

    #[test]
    fn pristine_type_is_the_report_after_the_frozen_file() {
        // The file's own output, a `#check` of the theorem included, comes
        // before the marker and is never read as the type.
        let run = |probe: &str| {
            assert!(probe.contains("theorem demo_thm") && probe.contains("#check @demo_thm"));
            let report = stub_output(
                probe,
                &format!("'demo_thm' depends on axioms: [sorryAx]\n{TYPE}\n"),
            );
            Ok(format!("demo_thm : ∀ (n : ℕ), n + 0 = n\n{report}"))
        };
        assert_eq!(pristine_type(FROZEN, "demo_thm", &run).as_deref(), Ok(TYPE));
        let err = |_: &str| Ok("x.lean:3:1: error: boom".to_owned());
        assert_eq!(
            pristine_type(FROZEN, "demo_thm", &err).unwrap_err(),
            RunError::Refused("x.lean:3:1: error: boom".into())
        );
        let unmarked = |_: &str| Ok(format!("declaration uses 'sorry'\n{TYPE}\n"));
        assert!(
            pristine_type(FROZEN, "demo_thm", &unmarked).is_err(),
            "no report after the marker"
        );
    }

    // The protocol cases, driven by a stubbed Lean runner (no Lean needed).
    #[test]
    fn sorry_candidate_is_rejected_before_lean_runs() {
        let run = |_: &str| -> Result<String, RunError> {
            panic!("Lean must not run for a sorry candidate")
        };
        assert_eq!(
            verify_text(FROZEN, TYPE, FROZEN, "demo_thm", &run),
            Err(Rejection::Forbidden("sorry".into()))
        );
    }

    #[test]
    fn tampered_statement_is_rejected() {
        let candidate = "import Mathlib\n\ntheorem demo_thm (n : Nat) : n + 0 = n ∨ True := by right; trivial\n";
        let run = |_: &str| -> Result<String, RunError> {
            panic!("Lean must not run on a statement mismatch")
        };
        assert_eq!(
            verify_text(FROZEN, TYPE, candidate, "demo_thm", &run),
            Err(Rejection::StatementMismatch)
        );
    }

    #[test]
    fn a_real_proof_with_the_identical_type_is_accepted() {
        let candidate = "import Mathlib\n\ntheorem demo_thm (n : Nat) : n + 0 = n :=\n by simp\n";
        let run = |probe: &str| Ok(clean_output(probe, TYPE));
        assert_eq!(
            verify_text(FROZEN, TYPE, candidate, "demo_thm", &run),
            Ok(Verified {
                axioms: vec!["propext".into()]
            })
        );
    }

    #[test]
    fn every_printed_type_head_is_read_and_must_match_the_pristine_one() {
        // Lean prefixes `@` for implicit binders and appends `.{u}` for
        // universe parameters; the head is part of the compared type.
        let candidate = "import Mathlib\n\ntheorem demo_thm (n : Nat) : n + 0 = n :=\n by simp\n";
        let body = " : ∀ (n : Nat), True";
        let rows: &[(&str, &str, Verdict)] = &[
            (
                "demo_thm",
                "demo_thm",
                Ok(Verified {
                    axioms: vec!["propext".into()],
                }),
            ),
            (
                "@demo_thm",
                "@demo_thm",
                Ok(Verified {
                    axioms: vec!["propext".into()],
                }),
            ),
            (
                "demo_thm.{u_1}",
                "demo_thm.{u_1}",
                Ok(Verified {
                    axioms: vec!["propext".into()],
                }),
            ),
            (
                "@demo_thm.{u_1,\n    u_2}",
                "@demo_thm.{u_1,\n    u_2}",
                Ok(Verified {
                    axioms: vec!["propext".into()],
                }),
            ),
            (
                "@demo_thm.{u_1}",
                "@demo_thm.{u_2}",
                Err(Rejection::TypeMismatch),
            ),
            ("@demo_thm", "demo_thm", Err(Rejection::TypeMismatch)),
        ];
        for (pristine_head, candidate_head, expected) in rows {
            let pristine_out = |probe: &str| {
                Ok(stub_output(
                    probe,
                    &format!("'demo_thm' depends on axioms: [sorryAx]\n{pristine_head}{body}\n"),
                ))
            };
            let pristine = pristine_type(FROZEN, "demo_thm", &pristine_out).unwrap();
            let run = |probe: &str| Ok(clean_output(probe, &format!("{candidate_head}{body}")));
            assert_eq!(
                &verify_text(FROZEN, &pristine, candidate, "demo_thm", &run),
                expected,
                "{candidate_head}"
            );
        }
    }

    #[test]
    fn a_rewritten_meaning_with_matching_surface_text_is_rejected_by_type() {
        // Surface statement byte-matches, but the elaborated type differs (e.g. a
        // with-solution `abbrev` was redefined): the type check must catch it.
        let candidate = "import Mathlib\n\ntheorem demo_thm (n : Nat) : n + 0 = n :=\n by simp\n";
        let run = |probe: &str| Ok(clean_output(probe, "demo_thm : ∀ (n : Nat), True"));
        assert_eq!(
            verify_text(FROZEN, TYPE, candidate, "demo_thm", &run),
            Err(Rejection::TypeMismatch)
        );
        let none = |probe: &str| {
            Ok(stub_output(
                probe,
                "'demo_thm' depends on axioms: [propext]\n",
            ))
        };
        assert_eq!(
            verify_text(FROZEN, TYPE, candidate, "demo_thm", &none),
            Err(Rejection::NoType)
        );
    }

    #[test]
    fn elaboration_errors_and_bad_axioms_are_rejected() {
        let candidate = "import Mathlib\n\ntheorem demo_thm (n : Nat) : n + 0 = n :=\n by bogus\n";
        let err = |_: &str| Ok("x.lean:4:3: error: unknown tactic".to_owned());
        assert!(matches!(
            verify_text(FROZEN, TYPE, candidate, "demo_thm", &err),
            Err(Rejection::Elaboration(_))
        ));
        let bad = |probe: &str| {
            Ok(stub_output(
                probe,
                &format!("'demo_thm' depends on axioms: [propext, Cheat.axiom]\n{TYPE}\n"),
            ))
        };
        assert_eq!(
            verify_text(FROZEN, TYPE, candidate, "demo_thm", &bad),
            Err(Rejection::NonStandardAxioms(vec!["Cheat.axiom".into()]))
        );
        // Whatever the candidate's own commands print comes before the
        // harness's marker and is never read: a clean-looking report printed
        // first does not hide the real one, and a report with anything
        // added after the marker is refused.
        let forged = |probe: &str| {
            let fake = format!("'demo_thm' depends on axioms: [propext]\n{TYPE}\n");
            let real = format!("'demo_thm' depends on axioms: [propext, Cheat.axiom]\n{TYPE}\n");
            Ok(format!("{fake}{}", stub_output(probe, &real)))
        };
        assert_eq!(
            verify_text(FROZEN, TYPE, candidate, "demo_thm", &forged),
            Err(Rejection::NonStandardAxioms(vec!["Cheat.axiom".into()]))
        );
        let padded = |probe: &str| {
            Ok(stub_output(
                probe,
                &format!("noise\n'demo_thm' depends on axioms: [propext]\n{TYPE}\n"),
            ))
        };
        assert_eq!(
            verify_text(FROZEN, TYPE, candidate, "demo_thm", &padded),
            Err(Rejection::NoType)
        );
    }
}
