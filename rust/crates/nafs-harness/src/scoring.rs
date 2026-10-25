// Derived from Open Math Harness (interfaces/discovery/src/scoring.rs, revision 687f2a2), Apache-2.0.
// Modified for Nafs-Aql: unchanged apart from this header; Nafs-Aql's own verdicts are mapped onto it in search.rs.
//! The task-scoring protocol, version 3: continuous credit for verified
//! sub-proofs, gated by Lean. The scalar `s_v` the discovery harness assigns
//! to a node from Lean's verdict on its file. Nothing a worker says enters
//! it; only what the `lean` service reported (`check`, `verify`, and for a
//! partial proof `analyze`, whose counts are recorded with the node).
//!
//! | verdict | `verify` | score |
//! | --- | --- | --- |
//! | `complete` | accepted | **1.0** (solved) |
//! | `complete` | rejected, or the verifier call failed | 0.9 |
//! | `sorries` (elaborates, `sorry` remains) | — | **0.5 + 0.4 · C / (C + L)** over the recorded [`Credit`]; **0.5** if any open leaf is hollow, if `C + L = 0`, or if no analysis was recorded |
//! | `errors` | — | 0.05 |
//! | `invalid` (banned token, statement not reproduced) | — | 0.0 |
//!
//! `C` is the number of closed sub-proofs that count and `L` the number of
//! open leaves (the `lean` interface's `SketchCounts::closed` and `::open`):
//! a closed sub-proof is a `have`/`obtain` step Lean accepted without `sorry`
//! that the fixed automation suite (plus a one-lemma library search) does
//! *not* prove outright and that the rest of the proof uses; an open leaf is
//! a remaining `sorry` goal, and it is **hollow** when the theorem's own goal
//! follows from the theorem's own hypotheses plus the leaf's target under the
//! same suite — a restatement, the `n = 0` disguise, a stronger statement the
//! suite can instantiate, "one easy piece plus the rest". A leaf whose
//! stripped statement the suite *refutes* is exempt: everything follows from
//! a false statement, so the test decides nothing there and the leaf counts
//! as open work (`untestable`; that is where a `by_contra` leaf lands). A
//! hollow leaf makes the node score the base whatever else it closed:
//! splitting earns nothing, only verified sub-proofs of a genuine
//! decomposition do.
//!
//! What the credit does *not* see is recorded beside it as observations for
//! the exploration policy ([`ErrorsObservation`], `delta_vs_parent`,
//! `auto_closable`, `false_leaf`): an `errors` node stays 0.05, and the
//! policy, not the scalar, tells a repairable error from a hard one.
//!
//! Every gate is a fixed Lean procedure under fixed heartbeat limits (the
//! `lean` interface's `ANALYSIS_SUITE_ID`), so the same file scores the same
//! anywhere; a gate that could not run leaves the base score and a recorded
//! reason. A node record says which version scored it (`scoring`).

use serde::{Deserialize, Serialize};

/// The protocol version a node record carries as `scoring`.
pub const VERSION: u32 = 3;

pub const SOLVED: f64 = 1.0;
/// A `complete` file that `verify` rejected or could not judge
/// (`verify_rejection`, `verifier_<code>` when the call failed): `check`
/// elaborates a candidate without `verify`'s import, axiom and type
/// checks, so `complete` there is not a solve. Just below one, the node
/// stays visible to the policy as the best unsolved node, and a worker
/// cannot win by fooling `check`.
pub const COMPLETE_UNVERIFIED: f64 = 0.9;
pub const SORRIES_BASE: f64 = 0.5;
pub const SORRIES_SPAN: f64 = 0.4;
pub const ERRORS: f64 = 0.05;
/// Below errors: a file with a banned token or without the frozen statement
/// verbatim is not an attempt at this problem at all. `check` decides it
/// without running Lean.
pub const INVALID: f64 = 0.0;
/// The most of the credit fraction a partial proof can earn, so that after
/// rounding it never reaches [`COMPLETE_UNVERIFIED`] (a sketch with ten
/// thousand counted steps and one open leaf is still not a complete proof).
pub const MAX_CREDIT_FRACTION: f64 = 0.999;
/// An `errors` node with at most this many errors is *repairable* for the
/// policy's purposes. A `timeout`-class error does not disqualify a node: a
/// heartbeat timeout is repaired by one `set_option maxHeartbeats` line.
pub const REPAIRABLE_MAX_ERRORS: u64 = 3;

/// Lean's verdict on a file, as the `lean` interface names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    Invalid,
    Errors,
    Sorries,
    Complete,
}

/// The recorded counts a `sorries` node's score is a pure function of: the
/// `lean` service's `SketchCounts::closed` and `::open`, and whether any
/// leaf was found hollow.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Credit {
    pub closed: u64,
    pub open: u64,
    pub hollow: bool,
}

impl Credit {
    /// `C / (C + L)`; 0 when hollow or when nothing counts.
    #[must_use]
    pub fn fraction(self) -> f64 {
        if self.hollow || self.closed + self.open == 0 {
            return 0.0;
        }
        self.closed as f64 / (self.closed + self.open) as f64
    }
}

/// The score of a node: `verdict` and, for a complete one, whether `verify`
/// accepted it; for a `sorries` node, the recorded credit (`None` when no
/// analysis was recorded, which scores the base). Rounded to four decimals
/// so recorded scores compare exactly.
#[must_use]
pub fn score(verdict: Verdict, verified: Option<bool>, credit: Option<Credit>) -> f64 {
    let raw = match verdict {
        Verdict::Complete => {
            if verified == Some(true) {
                SOLVED
            } else {
                COMPLETE_UNVERIFIED
            }
        }
        Verdict::Sorries => {
            SORRIES_BASE
                + SORRIES_SPAN
                    * credit
                        .map_or(0.0, Credit::fraction)
                        .min(MAX_CREDIT_FRACTION)
        }
        Verdict::Errors => ERRORS,
        Verdict::Invalid => INVALID,
    };
    (raw * 10_000.0).round() / 10_000.0
}

// ---- Observations beside the score ------------------------------------------

/// The classes an error message is sorted into (the first matching rule wins;
/// `other` when none does).
pub const ERROR_CLASSES: [&str; 11] = [
    "timeout",
    "syntax",
    "unsolved_goals",
    "unknown_name",
    "no_goals",
    "synth_failed",
    "rewrite_failed",
    "automation_failed",
    "type_mismatch",
    "tactic_misapplied",
    "other",
];

/// Whether some line of `message` starts with one of `heads`.
fn line_starts(message: &str, heads: &[&str]) -> bool {
    message
        .lines()
        .any(|l| heads.iter().any(|h| l.starts_with(h)))
}

fn contains_any(message: &str, needles: &[&str]) -> bool {
    needles.iter().any(|n| message.contains(n))
}

/// The class of one Lean error message (one of [`ERROR_CLASSES`]). A fixed
/// rule over the message text, so the same message always gets the same class.
#[must_use]
pub fn classify_error(message: &str) -> &'static str {
    if contains_any(
        message,
        &[
            "deterministic) timeout",
            "maximum recursion depth",
            "heartbeats",
        ],
    ) {
        return "timeout";
    }
    if line_starts(
        message,
        &[
            "unexpected token",
            "unexpected end of input",
            "expected token",
            "unexpected identifier",
            "expected '",
            "unexpected;",
        ],
    ) {
        return "syntax";
    }
    if line_starts(message, &["unsolved goals"]) {
        return "unsolved_goals";
    }
    if line_starts(
        message,
        &[
            "Unknown constant",
            "Unknown identifier",
            "unknown constant",
            "unknown identifier",
        ],
    ) || message.contains("The environment does not contain")
    {
        return "unknown_name";
    }
    if line_starts(message, &["No goals to be solved", "no goals"]) {
        return "no_goals";
    }
    if line_starts(message, &["failed to synthesize"])
        || message.contains("typeclass instance problem is stuck")
    {
        return "synth_failed";
    }
    if contains_any(
        message,
        &[
            "Did not find an occurrence of the pattern",
            "tactic failed, pattern",
            "Invalid rewrite argument",
            "motive is not type correct",
            "equality or iff proof",
        ],
    ) {
        return "rewrite_failed";
    }
    if line_starts(
        message,
        &[
            "omega could not prove",
            "linarith failed",
            "ring failed",
            "decide failed",
        ],
    ) || contains_any(
        message,
        &[
            "ring_nf made no progress",
            "made no progress",
            "exhaustive search",
            "interval_cases failed",
            "failed to prove",
            "positivity failed",
            "nlinarith failed",
            "norm_num failed",
            "could not close the goal",
        ],
    ) || automation_tactic_failed(message)
    {
        return "automation_failed";
    }
    if line_starts(
        message,
        &[
            "Type mismatch",
            "Application type mismatch",
            "Function expected",
            "Failed to infer type",
            "Invalid `",
            "Invalid projection",
            "Invalid argument name",
            "overloaded",
            "Ambiguous term",
            "Invalid field notation",
            "Invalid field",
            "unnecessary",
        ],
    ) || contains_any(
        message,
        &[
            "mod_cast has type",
            "expression has type",
            "must not contain metavariables",
            "Insufficient number of fields",
            "expected result type of cast",
        ],
    ) || invalid_notation(message)
    {
        return "type_mismatch";
    }
    if line_starts(
        message,
        &[
            "Tactic `",
            "Alternative",
            "invalid ",
            "Invalid target",
            "Invalid alternative",
        ],
    ) || contains_any(
        message,
        &[
            "tactic ",
            "no applicable constructor",
            "is not an inductive datatype",
            "introN",
        ],
    ) && (message.contains(" failed")
        || contains_any(
            message,
            &[
                "no applicable constructor",
                "is not an inductive datatype",
                "introN",
            ],
        ))
    {
        return "tactic_misapplied";
    }
    "other"
}

/// "Tactic `omega` failed", "Tactic `simp` failed", … for the automation
/// tactics.
fn automation_tactic_failed(message: &str) -> bool {
    const TACTICS: [&str; 13] = [
        "omega",
        "linarith",
        "nlinarith",
        "simp",
        "simp_all",
        "norm_num",
        "decide",
        "aesop",
        "tauto",
        "positivity",
        "ring",
        "field_simp",
        "exact?",
    ];
    TACTICS
        .iter()
        .any(|t| message.contains(&format!("Tactic `{t}` failed")))
}

/// "invalid `…` notation" anywhere in the message.
fn invalid_notation(message: &str) -> bool {
    message
        .match_indices("invalid `")
        .any(|(i, _)| message[i..].contains("` notation"))
}

/// What the harness records about an `errors` node beside `check`'s
/// diagnostics: the class multiset, where the first error falls in the proof
/// body, and the derived `repairable` flag the initial policy reads. Recorded
/// with the node as is.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ErrorsObservation {
    pub count: u64,
    /// One class per error, in [`ERROR_CLASSES`] order (a multiset).
    pub classes: Vec<String>,
    /// (line of the first error − first proof-body line) / body length;
    /// `None` when the body has no lines or no error has a line.
    pub first_error_fraction: Option<f64>,
    /// `count <= REPAIRABLE_MAX_ERRORS`.
    pub repairable: bool,
}

/// Builds the observation from `check`'s errors (`(line, message)` pairs)
/// and the proof body's line range `[body_start, body_end]` (1-based, the
/// line of the theorem's `:=` to the last non-empty line).
#[must_use]
pub fn errors_observation(
    errors: &[(u64, &str)],
    body_start: u64,
    body_end: u64,
) -> ErrorsObservation {
    let mut classes: Vec<String> = errors
        .iter()
        .map(|(_, m)| classify_error(m).to_owned())
        .collect();
    classes.sort_by_key(|c| {
        ERROR_CLASSES
            .iter()
            .position(|k| k == c)
            .unwrap_or(ERROR_CLASSES.len())
    });
    let first = errors.iter().map(|(line, _)| *line).min();
    let first_error_fraction = match first {
        Some(line) if body_end > body_start => {
            let at = line.saturating_sub(body_start) as f64 / (body_end - body_start) as f64;
            Some(((at.clamp(0.0, 1.0)) * 10_000.0).round() / 10_000.0)
        }
        _ => None,
    };
    let count = errors.len() as u64;
    let repairable = count <= REPAIRABLE_MAX_ERRORS;
    ErrorsObservation {
        count,
        classes,
        first_error_fraction,
        repairable,
    }
}

/// How a child's verdict moved against its parent's (recorded per node).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeltaVsParent {
    pub sorries: i64,
    pub errors: i64,
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::float_cmp
    )]
    use super::*;

    #[test]
    fn the_verdicts_other_than_sorries_score_fixed_values() {
        assert_eq!(score(Verdict::Complete, Some(true), None), 1.0);
        assert_eq!(score(Verdict::Complete, Some(false), None), 0.9);
        assert_eq!(
            score(Verdict::Complete, None, None),
            0.9,
            "no verifier's word, no solve"
        );
        let full = Some(Credit {
            closed: 9,
            open: 1,
            hollow: false,
        });
        assert_eq!(
            score(Verdict::Errors, None, full),
            0.05,
            "credit never lifts an errors node"
        );
        assert_eq!(score(Verdict::Invalid, None, full), 0.0);
        assert_eq!(score(Verdict::Complete, Some(true), full), 1.0);
    }

    #[test]
    fn a_sorries_node_earns_its_verified_share_of_the_sketch() {
        let credit = |closed, open, hollow| {
            Some(Credit {
                closed,
                open,
                hollow,
            })
        };
        // The root, a restatement, padding: nothing counts, or a leaf is hollow.
        assert_eq!(
            score(Verdict::Sorries, None, None),
            0.5,
            "no analysis recorded: the base"
        );
        assert_eq!(score(Verdict::Sorries, None, credit(0, 1, true)), 0.5);
        assert_eq!(score(Verdict::Sorries, None, credit(0, 1, false)), 0.5);
        assert_eq!(
            score(Verdict::Sorries, None, credit(0, 0, false)),
            0.5,
            "nothing counts"
        );
        assert_eq!(
            score(Verdict::Sorries, None, credit(4, 1, true)),
            0.5,
            "a hollow leaf spoils every closed step"
        );
        // Ten pieces, +0.04 each (0.4 over ten).
        assert_eq!(score(Verdict::Sorries, None, credit(1, 9, false)), 0.54);
        assert_eq!(score(Verdict::Sorries, None, credit(9, 1, false)), 0.86);
        // Genuine sketches: one of two, two of three.
        assert_eq!(score(Verdict::Sorries, None, credit(1, 1, false)), 0.7);
        assert_eq!(score(Verdict::Sorries, None, credit(2, 1, false)), 0.7667);
        // Never reaches a complete proof: L ≥ 1 for a sorries node.
        assert!(score(Verdict::Sorries, None, credit(10_000, 1, false)) < 0.9);
    }

    #[test]
    fn credit_is_a_fraction_of_counted_steps() {
        assert_eq!(
            Credit {
                closed: 1,
                open: 3,
                hollow: false
            }
            .fraction(),
            0.25
        );
        assert_eq!(
            Credit {
                closed: 1,
                open: 3,
                hollow: true
            }
            .fraction(),
            0.0
        );
        assert_eq!(Credit::default().fraction(), 0.0);
    }

    #[test]
    fn scores_are_rounded_to_four_decimals() {
        let s = score(
            Verdict::Sorries,
            None,
            Some(Credit {
                closed: 1,
                open: 2,
                hollow: false,
            }),
        ); // 0.5 + 0.4/3
        assert_eq!(s, 0.6333);
        assert_eq!(s.to_string(), "0.6333");
    }

    #[test]
    fn error_messages_fall_into_the_measured_classes() {
        assert_eq!(
            classify_error("(deterministic) timeout at `whnf`, maximum number of heartbeats"),
            "timeout"
        );
        assert_eq!(
            classify_error("unexpected token ':='; expected term"),
            "syntax"
        );
        assert_eq!(
            classify_error("unsolved goals\nn : ℕ\n⊢ n = n"),
            "unsolved_goals"
        );
        assert_eq!(
            classify_error("Unknown identifier `Nat.foo`"),
            "unknown_name"
        );
        assert_eq!(
            classify_error("unknown constant 'Real.bar'"),
            "unknown_name"
        );
        assert_eq!(classify_error("No goals to be solved"), "no_goals");
        assert_eq!(
            classify_error("failed to synthesize\n  Decidable p"),
            "synth_failed"
        );
        assert_eq!(
            classify_error("tactic 'rewrite' failed, Did not find an occurrence of the pattern"),
            "rewrite_failed"
        );
        assert_eq!(
            classify_error("linarith failed to find a contradiction"),
            "automation_failed"
        );
        assert_eq!(
            classify_error("Tactic `omega` failed: no contradiction"),
            "automation_failed"
        );
        assert_eq!(
            classify_error("Type mismatch\n  h\nhas type"),
            "type_mismatch"
        );
        assert_eq!(
            classify_error("Application type mismatch: the argument"),
            "type_mismatch"
        );
        assert_eq!(
            classify_error("Tactic `rcases` failed: x is not an inductive datatype"),
            "tactic_misapplied"
        );
        assert_eq!(classify_error("something nobody classified"), "other");
        assert!(ERROR_CLASSES.contains(&classify_error("anything")));
    }

    #[test]
    fn an_errors_observation_places_the_first_error_and_says_repairable() {
        let obs = errors_observation(&[(12, "unsolved goals"), (18, "Type mismatch")], 10, 20);
        assert_eq!(obs.count, 2);
        assert_eq!(obs.classes, ["unsolved_goals", "type_mismatch"]);
        assert_eq!(obs.first_error_fraction, Some(0.2));
        assert!(obs.repairable);
        let many = errors_observation(&[(1, "x"), (2, "x"), (3, "x"), (4, "x")], 1, 4);
        assert!(!many.repairable, "four errors are not repairable");
        let timeout = errors_observation(&[(3, "(deterministic) timeout")], 1, 4);
        assert!(
            timeout.repairable,
            "a heartbeat timeout is one `set_option maxHeartbeats` away: repairable"
        );
        assert_eq!(
            timeout.classes,
            ["timeout"],
            "classified, not disqualifying"
        );
        assert_eq!(
            errors_observation(&[], 1, 4),
            ErrorsObservation {
                repairable: true,
                ..Default::default()
            }
        );
        assert_eq!(
            errors_observation(&[(5, "x")], 5, 5).first_error_fraction,
            None,
            "an empty body"
        );
        assert_eq!(
            errors_observation(&[(3, "x")], 5, 9).first_error_fraction,
            Some(0.0),
            "before the body clamps"
        );
        assert_eq!(
            errors_observation(&[(30, "x")], 5, 9).first_error_fraction,
            Some(1.0),
            "after the body clamps"
        );
    }
}
