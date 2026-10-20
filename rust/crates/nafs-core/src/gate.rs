//! The Aql Gate (Theorem 2): escalate iff L (e1 - e2) > lambda * delta_c, with a calibrated e1.
//! Phase 1 also ships the placeholder fixed rule; every decision records which rule was used (FR-013).
use crate::backend::GenOutput;
use crate::calibration::Calibration;
use crate::config::Config;
use nafs_types::{Domain, GateFeatures, GateRule};
use nafs_verify::answer_similarity;

/// 0 = speed, 1 = care: how far lambda sits below twice its default.
fn care(cfg: &Config, lambda: f32) -> f32 {
    (1.0 - (lambda / (2.0 * cfg.gate.lambda_default.max(1e-3))).min(1.0)).max(0.0)
}

/// The fixed rule's mean log-prob threshold at this lambda (more care raises it).
pub fn fixed_logprob_threshold(cfg: &Config, lambda: f32) -> f32 {
    cfg.gate.fixed.mean_logprob_threshold + 0.25 * care(cfg, lambda)
}

/// Phase 8: the fixed rule's threshold on the lowest-window log-prob at this lambda, if the rule is on for
/// this domain (the signal's scale is domain-specific; see `FixedRuleCfg::low_window_domains`).
pub fn fixed_window_threshold(cfg: &Config, lambda: f32, domain: Domain) -> Option<f32> {
    let fx = &cfg.gate.fixed;
    if !fx.low_window_domains.iter().any(|d| d == domain.as_str()) {
        return None;
    }
    fx.low_window_logprob_threshold
        .map(|t| t + 0.25 * care(cfg, lambda))
}

pub fn features(
    draft: &GenOutput,
    samples: &[GenOutput],
    domain: Domain,
    verifiable: bool,
    draft_verified: Option<bool>,
    prompt_tokens: u32,
    window: usize,
) -> GateFeatures {
    // Agreement = mean similarity of each cheap sample to the draft (exact for numeric answers, word overlap for text).
    // Truncated samples (finish_reason "length") carry an intermediate number, not an answer (M11).
    let valid: Vec<&GenOutput> = samples
        .iter()
        .filter(|s| !s.content.is_empty() && s.finish_reason != "length")
        .collect();
    let agreement = if valid.is_empty() {
        None
    } else {
        Some(
            valid
                .iter()
                .map(|s| answer_similarity(&draft.content, &s.content).min(1.0))
                .sum::<f32>()
                / valid.len() as f32,
        )
    };
    GateFeatures {
        mean_logprob: draft.mean_logprob(),
        min_logprob: draft.min_logprob(),
        low_window_logprob: crate::sequential::low_window_mean(&draft.logprobs, window),
        k: valid.len() as u8,
        agreement,
        domain,
        verifiable,
        draft_verified,
        prompt_tokens,
        draft_tokens: draft.completion_tokens.max(draft.logprobs.len() as u32),
    }
}

#[derive(Debug, Clone)]
pub struct GateDecision {
    /// Raw calibrated estimate e1_hat(features), recorded even when a verifier decided (review M9).
    pub e1: f32,
    pub e2: f32,
    pub lambda: f32,
    pub delta_c: f32,
    pub escalate: bool,
    pub rule: GateRule,
    pub explanation: String,
}

/// Everything the gate needs besides the features; e2 and delta_c may come from Phase-0 measurements.
pub struct GateCtx<'a> {
    pub cfg: &'a Config,
    /// fixed | calibrated | budget
    pub mode: &'a str,
    pub lambda: f32,
    pub e2: f32,
    pub delta_c: f32,
    pub cal: &'a Calibration,
    /// Calibration passed the NFR-020 promotion test.
    pub promoted: bool,
    /// Escalation budget B for budget mode (FR-014).
    pub budget: f32,
}

/// Largest price lambda at which the calibrated rule would still escalate this sample (budget mode).
pub fn lambda_threshold(cfg: &Config, e1: f32, e2: f32, delta_c: f32) -> f32 {
    cfg.gate.loss_wrong * (e1 - e2) * cfg.gate.cost_unit_s / delta_c.max(1e-3)
}

pub fn decide(f: &GateFeatures, ctx: &GateCtx) -> GateDecision {
    let cfg = ctx.cfg;
    let cal = ctx.cal;
    let e1 = cal.e1(f);
    let (e2, delta_c) = (ctx.e2, ctx.delta_c);
    let use_cal = ctx.promoted && ctx.mode != "fixed";
    let lambda_star = if ctx.mode == "budget" && use_cal {
        cal.lambda_star(ctx.budget)
    } else {
        None
    };
    let (lambda, cal_rule) = match lambda_star {
        Some(ls) => (ls, GateRule::Budget),
        None => (ctx.lambda, GateRule::Calibrated),
    };
    let mk = |escalate: bool, rule: GateRule, explanation: String| GateDecision {
        e1,
        e2,
        lambda,
        delta_c,
        escalate,
        rule,
        explanation,
    };
    if f.draft_verified == Some(false) {
        return mk(
            true,
            GateRule::ForcedFailedVerification,
            "The draft failed an exact verifier, so it is never kept (FR-012).".into(),
        );
    }
    if f.draft_verified == Some(true) {
        return mk(
            false,
            if use_cal { cal_rule } else { GateRule::Fixed },
            format!("The draft passed an exact verifier (threshold 1.0), so deliberation cannot improve precision (Thm 1); e1_hat {e1:.2} recorded for calibration."),
        );
    }
    if use_cal {
        let cost = lambda * delta_c / cfg.gate.cost_unit_s;
        let gain = cfg.gate.loss_wrong * (e1 - e2);
        let escalate = gain > cost;
        let how = match lambda_star {
            Some(_) => format!(
                "budget mode B = {:.0}%, lambda* = {:.3}",
                ctx.budget * 100.0,
                lambda
            ),
            None => "calibrated".to_string(),
        };
        return mk(
            escalate,
            cal_rule,
            format!(
                "L(e1-e2) = {:.3} {} lambda*dc = {:.3} ({how}, held-out ECE {:.3}, n {}).",
                gain,
                if escalate { ">" } else { "<=" },
                cost,
                cal.ece,
                cal.n
            ),
        );
    }
    // Phase-1 placeholder fixed rule (spec 4.3): thresholds on log-probability and sample agreement,
    // made more conservative as lambda falls (more "care").
    let fx = &cfg.gate.fixed;
    let care = care(cfg, ctx.lambda);
    let lp_thr = fixed_logprob_threshold(cfg, ctx.lambda);
    let ag_thr = fx.agreement_threshold + 0.4 * care;
    let mut reasons = vec![];
    if let Some(m) = f.mean_logprob {
        if m < lp_thr {
            reasons.push(format!("mean log-prob {m:.2} < {lp_thr:.2}"));
        }
    }
    // Phase 8: a local collapse of confidence (lowest window) flags the draft even when its mean is fine.
    if let (Some(lw), Some(thr)) = (
        f.low_window_logprob,
        fixed_window_threshold(cfg, ctx.lambda, f.domain),
    ) {
        if lw < thr {
            reasons.push(format!(
                "lowest {}-token window log-prob {lw:.2} < {thr:.2}",
                cfg.gate.confidence_window
            ));
        }
    }
    // Agreement is evidence only where answers are canonical (a number, code, a structure, a short
    // fact). Open-ended replies legitimately differ in wording, so lexical agreement is not used there.
    let canonical = matches!(
        f.domain,
        Domain::Math | Domain::Code | Domain::Structured | Domain::Factual
    );
    if let (Some(a), true) = (f.agreement, canonical) {
        if a < ag_thr {
            reasons.push(format!(
                "sample agreement {:.0}% < {:.0}%",
                a * 100.0,
                ag_thr * 100.0
            ));
        }
    }
    if fx.escalate_open_domains
        && care > 0.9
        && matches!(f.domain, Domain::Math | Domain::Code)
        && !f.verifiable
    {
        reasons.push("maximum care on a hard domain".into());
    }
    let escalate = !reasons.is_empty();
    let why_fixed = if ctx.mode == "fixed" {
        "gate mode is fixed".to_string()
    } else {
        "no promoted calibration (NFR-020)".to_string()
    };
    mk(
        escalate,
        GateRule::Fixed,
        if escalate {
            format!("Fixed rule ({why_fixed}): {}.", reasons.join("; "))
        } else {
            format!("Fixed rule ({why_fixed}): draft confident (log-prob and agreement above thresholds); e1_hat {e1:.2} is not used.")
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    fn cfg() -> Config {
        let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../config/nafs-aql.toml");
        Config::load(p).expect("config loads")
    }
    fn f(mean: f32, agree: f32, domain: Domain, verified: Option<bool>) -> GateFeatures {
        GateFeatures {
            mean_logprob: Some(mean),
            min_logprob: Some(-2.0),
            low_window_logprob: Some(mean - 0.1),
            k: 2,
            agreement: Some(agree),
            domain,
            verifiable: verified.is_some(),
            draft_verified: verified,
            prompt_tokens: 40,
            draft_tokens: 60,
        }
    }
    fn ctx<'a>(
        c: &'a Config,
        cal: &'a Calibration,
        mode: &'a str,
        lambda: f32,
        promoted: bool,
        d: Domain,
    ) -> GateCtx<'a> {
        GateCtx {
            cfg: c,
            mode,
            lambda,
            e2: c.e2(d),
            delta_c: c.delta_c(d),
            cal,
            promoted,
            budget: c.gate.escalation_budget,
        }
    }
    #[test]
    fn failed_verification_always_escalates() {
        let c = cfg();
        let cal = Calibration::prior();
        let d = decide(
            &f(-0.01, 1.0, Domain::Math, Some(false)),
            &ctx(&c, &cal, "fixed", 5.0, false, Domain::Math),
        );
        assert!(d.escalate);
        assert_eq!(d.rule, GateRule::ForcedFailedVerification);
    }
    #[test]
    fn verified_draft_is_kept_with_raw_e1_and_honest_rule() {
        let c = cfg();
        let cal = Calibration::prior();
        let x = f(-2.0, 0.0, Domain::Code, Some(true));
        let d = decide(&x, &ctx(&c, &cal, "calibrated", 0.0, false, Domain::Code));
        assert!(!d.escalate);
        assert_eq!(
            d.rule,
            GateRule::Fixed,
            "unpromoted calibration must not be labelled calibrated (L1)"
        );
        assert!(
            (d.e1 - cal.e1(&x)).abs() < 1e-6,
            "e1 is the raw estimate, not 0 (M9)"
        );
    }
    #[test]
    fn confident_draft_kept_uncertain_escalated() {
        let c = cfg();
        let cal = Calibration::prior();
        let g = |m, a, l| {
            decide(
                &f(m, a, Domain::Factual, None),
                &ctx(&c, &cal, "fixed", l, false, Domain::Factual),
            )
            .escalate
        };
        assert!(!g(-0.05, 1.0, 0.2));
        assert!(g(-0.9, 1.0, 0.2));
        assert!(g(-0.05, 0.1, 0.2));
    }
    #[test]
    fn open_ended_wording_differences_do_not_escalate() {
        let c = cfg();
        let cal = Calibration::prior();
        // MVP-A1: "Say hello" with confident tokens but differently worded samples.
        assert!(
            !decide(
                &f(-0.14, 0.67, Domain::Open, None),
                &ctx(&c, &cal, "fixed", 0.2, false, Domain::Open)
            )
            .escalate
        );
        // A factual answer whose samples disagree still escalates.
        assert!(
            decide(
                &f(-0.14, 0.2, Domain::Factual, None),
                &ctx(&c, &cal, "fixed", 0.2, false, Domain::Factual)
            )
            .escalate
        );
    }
    #[test]
    fn more_care_escalates_more() {
        let c = cfg();
        let cal = Calibration::prior();
        let x = f(-0.3, 0.75, Domain::Factual, None);
        let speed = decide(&x, &ctx(&c, &cal, "fixed", 0.8, false, Domain::Factual)).escalate;
        let care = decide(&x, &ctx(&c, &cal, "fixed", 0.0, false, Domain::Factual)).escalate;
        assert!(
            care || !speed,
            "lowering lambda must never turn an escalation into a keep"
        );
        assert!(care);
    }
    #[test]
    fn calibrated_rule_is_threshold_on_cost() {
        let c = cfg();
        let cal = Calibration::prior();
        let x = f(-1.5, 0.0, Domain::Math, None); // high e1
        let e1 = cal.e1(&x);
        let lam_hi = 2.0 * lambda_threshold(&c, e1, c.e2(Domain::Math), c.delta_c(Domain::Math));
        assert!(decide(&x, &ctx(&c, &cal, "calibrated", 0.0, true, Domain::Math)).escalate);
        assert!(!decide(&x, &ctx(&c, &cal, "calibrated", lam_hi, true, Domain::Math)).escalate);
        assert_eq!(
            decide(&x, &ctx(&c, &cal, "calibrated", 0.0, true, Domain::Math)).rule,
            GateRule::Calibrated
        );
        // Unpromoted: fixed rule even in calibrated mode (H1).
        assert_eq!(
            decide(&x, &ctx(&c, &cal, "calibrated", 0.0, false, Domain::Math)).rule,
            GateRule::Fixed
        );
    }
    fn synthetic(n: usize, seed: u64) -> (Vec<GateFeatures>, Vec<bool>) {
        let mut rng = fastrand::Rng::with_seed(seed);
        let mut xs = vec![];
        let mut ys = vec![];
        for _ in 0..n {
            let m = -rng.f32() * 2.0;
            let p = 1.0 / (1.0 + (-(-2.0 - 3.0 * m)).exp()); // true P(wrong) rises as log-prob falls
            xs.push(f(
                m,
                if rng.bool() { 1.0 } else { 0.5 },
                if rng.bool() {
                    Domain::Math
                } else {
                    Domain::Open
                },
                None,
            ));
            ys.push(rng.f32() < p);
        }
        (xs, ys)
    }
    #[test]
    fn calibration_fit_is_held_out_and_well_calibrated() {
        let c = cfg();
        let (xs, ys) = synthetic(3000, 1);
        let th =
            |g: &GateFeatures, e: f32| lambda_threshold(&c, e, c.e2(g.domain), c.delta_c(g.domain));
        let cal = Calibration::fit(&xs, &ys, None, &th);
        assert_eq!(cal.folds, 5);
        assert!(
            cal.e1(&f(-1.8, 1.0, Domain::Math, None)) > cal.e1(&f(-0.1, 1.0, Domain::Math, None))
        );
        assert!(cal.ece < 0.05, "held-out ECE {}", cal.ece);
        assert!(cal.promotable(2900, 30, 0.05).is_ok());
        assert_eq!(cal.lambda_quantiles.len(), 101);
    }
    #[test]
    fn small_or_one_class_fits_are_never_promoted() {
        let c = cfg();
        let th =
            |g: &GateFeatures, e: f32| lambda_threshold(&c, e, c.e2(g.domain), c.delta_c(g.domain));
        // Review H1: 10 items, all drafts right.
        let xs: Vec<GateFeatures> = (0..10)
            .map(|i| f(-(i as f32) / 10.0, 1.0, Domain::Math, None))
            .collect();
        let ys = vec![false; 10];
        let cal = Calibration::fit(&xs, &ys, None, &th);
        assert!(cal.promotable(2900, 30, 0.05).is_err());
        assert!(cal.promotable(1, 30, 0.05).is_err(), "one class only");
    }
    #[test]
    fn budget_mode_meets_the_escalation_budget() {
        let c = cfg();
        let (xs, ys) = synthetic(3000, 2);
        let th =
            |g: &GateFeatures, e: f32| lambda_threshold(&c, e, c.e2(g.domain), c.delta_c(g.domain));
        let cal = Calibration::fit(&xs, &ys, None, &th);
        let (test_x, _) = synthetic(3000, 3);
        for b in [0.1f32, 0.3, 0.5] {
            let esc = test_x
                .iter()
                .filter(|x| {
                    let mut k = ctx(&c, &cal, "budget", 0.2, true, x.domain);
                    k.budget = b;
                    decide(x, &k).escalate
                })
                .count() as f32
                / test_x.len() as f32;
            assert!((esc - b).abs() < 0.05, "budget {b}: escalation rate {esc}");
        }
        assert!(c.gate.dkw_n() >= 2900);
    }
    #[test]
    fn truncated_samples_do_not_count_for_agreement() {
        let draft = GenOutput {
            content: "Answer: 145".into(),
            completion_tokens: 10,
            ..Default::default()
        };
        let cut = GenOutput {
            content: "Step 1: 60 + 25 = 85".into(),
            finish_reason: "length".into(),
            ..Default::default()
        };
        let ok = GenOutput {
            content: "So the answer is 145.\nAnswer: 145".into(),
            finish_reason: "stop".into(),
            ..Default::default()
        };
        let g = features(
            &draft,
            &[cut.clone(), ok],
            Domain::Math,
            false,
            None,
            10,
            16,
        );
        assert_eq!(g.k, 1);
        assert_eq!(g.agreement, Some(1.0));
        assert_eq!(
            features(&draft, &[cut], Domain::Math, false, None, 10, 16).agreement,
            None
        );
    }
    #[test]
    fn isotonic_is_monotone() {
        let ps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let ys = [false, true, false, false, true, true];
        let m = crate::calibration::isotonic_fit(&ps, &ys);
        let mut last = 0.0;
        for p in [0.0, 0.15, 0.25, 0.35, 0.45, 0.55, 0.7] {
            let v = crate::calibration::isotonic_apply(&m, p);
            assert!(v >= last - 1e-6);
            last = v;
        }
    }
}
