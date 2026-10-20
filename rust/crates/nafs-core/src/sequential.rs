//! Sequential evidence rules (Phase 8). Nafs-Aql reads streams of evidence (token log-probabilities of a
//! draft, answers probed from a reasoning trace) and must decide when enough has been seen. The rules here
//! come from sequential analysis and the accumulation-to-bound account of decision making; each one is a
//! proposition in `docs/research/quantum_neuro_review.md` and in the report's Phase-8 section.
//!
//! * `low_window_mean` / `WindowTracker`: the lowest mean log-probability over any `w` consecutive tokens
//!   ("lowest group confidence"). It is a running minimum, so a threshold crossing observed on a prefix is
//!   the full draft's decision as well (Prop. N2): early escalation on this signal is exact, not predictive.
//! * `ProbePosterior`: Dirichlet-multinomial posterior over the answers probed from System 2's reasoning,
//!   and the probability that the plurality answer is the majority (Adaptive-Consistency). With the
//!   collapsing bound of `collapsing_bound` (optimal under a deadline, Frazier & Yu 2007) it replaces the
//!   "k consecutive agreeing probes" rule of the adaptive exit (Prop. N4).
//! * `wald_expected_n_lower`: Wald's lower bound on the expected number of tokens any sequential test needs
//!   to separate two mean log-probabilities (Prop. N3); it says why the mean-based early escalation cannot
//!   be given an error guarantee at its working margin.
//! * `optimal_probe_interval`: the periodic inspection interval that minimises probe cost plus overshoot
//!   (Prop. N5); it validates the configured probe gap.
use std::collections::VecDeque;

/// Lowest mean log-probability over any window of `w` consecutive tokens. A draft shorter than `w` tokens is
/// one window. `None` for an empty draft.
pub fn low_window_mean(lps: &[f32], w: usize) -> Option<f32> {
    let w = w.max(1);
    if lps.is_empty() {
        return None;
    }
    if lps.len() <= w {
        return Some(lps.iter().sum::<f32>() / lps.len() as f32);
    }
    let mut s: f64 = lps[..w].iter().map(|&x| x as f64).sum();
    let mut best = s;
    for i in w..lps.len() {
        s += lps[i] as f64 - lps[i - w] as f64;
        if s < best {
            best = s;
        }
    }
    Some((best / w as f64) as f32)
}

/// Token-by-token tracker of the window signal for a stream. `push` returns the running minimum over the
/// complete windows seen so far, which is `None` until `w` tokens have arrived. For every prefix this value
/// is >= `low_window_mean` of the complete draft (a minimum over fewer windows), so a crossing seen here is
/// final.
#[derive(Debug, Clone)]
pub struct WindowTracker {
    w: usize,
    buf: VecDeque<f32>,
    sum: f64,
    min: Option<f32>,
}
impl WindowTracker {
    pub fn new(w: usize) -> Self {
        Self {
            w: w.max(1),
            buf: VecDeque::with_capacity(w.max(1) + 1),
            sum: 0.0,
            min: None,
        }
    }
    pub fn push(&mut self, lp: f32) -> Option<f32> {
        self.buf.push_back(lp);
        self.sum += lp as f64;
        if self.buf.len() > self.w {
            if let Some(old) = self.buf.pop_front() {
                self.sum -= old as f64;
            }
        }
        if self.buf.len() == self.w {
            let m = (self.sum / self.w as f64) as f32;
            self.min = Some(self.min.map_or(m, |x| x.min(m)));
        }
        self.min
    }
    pub fn value(&self) -> Option<f32> {
        self.min
    }
}

/// I_{1/2}(a, b) for integer a, b >= 1: the regularised incomplete beta function at one half, which equals
/// P[Binomial(a+b-1, 1/2) >= a] = 2^{-(a+b-1)} * sum_{j=a}^{a+b-1} C(a+b-1, j).
pub fn beta_half_cdf(a: u32, b: u32) -> f64 {
    let n = (a + b - 1) as u64;
    let mut c = 1f64; // C(n, 0)
    let mut tail = 0f64;
    for j in 0..=n {
        if j >= a as u64 {
            tail += c;
        }
        c = c * (n - j) as f64 / (j + 1) as f64;
    }
    tail / 2f64.powi(n as i32)
}

/// Dirichlet-multinomial posterior over probed answers with a uniform Dirichlet(1, ..., 1) prior
/// (Adaptive-Consistency, Aggarwal et al. 2023). Answers are clustered by a caller-supplied equivalence.
#[derive(Debug, Clone, Default)]
pub struct ProbePosterior {
    /// (representative answer, count), in order of first appearance.
    pub counts: Vec<(String, u32)>,
    pub total: u32,
}
impl ProbePosterior {
    pub fn observe(&mut self, answer: &str, same: impl Fn(&str, &str) -> bool) -> usize {
        self.total += 1;
        if let Some(i) = self.counts.iter().position(|(rep, _)| same(rep, answer)) {
            self.counts[i].1 += 1;
            return i;
        }
        self.counts.push((answer.to_string(), 1));
        self.counts.len() - 1
    }
    /// The plurality answer and its count.
    pub fn top(&self) -> Option<(&str, u32)> {
        self.counts
            .iter()
            .max_by_key(|(_, c)| *c)
            .map(|(s, c)| (s.as_str(), *c))
    }
    /// P[p_top > 1/2] under the Beta(v_top + 1, n - v_top + 1) marginal of the posterior: the probability that
    /// the plurality answer is the answer a majority of further probes would give.
    pub fn majority_probability(&self) -> f32 {
        match self.top() {
            None => 0.0,
            Some((_, v)) => (1.0 - beta_half_cdf(v + 1, self.total - v + 1)) as f32,
        }
    }
}

/// Stopping bound that collapses linearly from `start` (no budget used) to `end` (budget exhausted). Under a
/// deadline the Bayes-optimal threshold on the posterior is non-increasing in time (Frazier & Yu 2007); a
/// linear collapse is the simplest such schedule and is what the two parameters fix.
pub fn collapsing_bound(used: u32, horizon: u32, start: f32, end: f32) -> f32 {
    if horizon == 0 {
        return end.min(start);
    }
    let t = (used as f32 / horizon as f32).clamp(0.0, 1.0);
    start + (end - start) * t
}

/// Wald's lower bound on the expected number of i.i.d. observations under H1 for any sequential test of
/// H0: mu = tau against H1: mu = tau - delta with error probabilities (alpha, beta), when observations are
/// N(mu, sigma^2): E_1[N] >= [(1-beta) ln((1-beta)/alpha) + beta ln(beta/(1-alpha))] / (delta^2 / (2 sigma^2)).
pub fn wald_expected_n_lower(delta: f64, sigma: f64, alpha: f64, beta: f64) -> f64 {
    let kl = delta * delta / (2.0 * sigma * sigma);
    let num = (1.0 - beta) * ((1.0 - beta) / alpha).ln() + beta * (beta / (1.0 - alpha)).ln();
    num / kl
}

/// Interval (in reasoning tokens) between probes that minimises expected probe cost plus expected
/// overshoot when the token count at which the answer becomes available is exponential with mean
/// `mean_tokens_to_answer`: Delta* = sqrt(2 * c_p * v * m), with c_p the probe cost in seconds, v the
/// decode rate in tokens per second and m the mean. The cost curve is flat near the optimum:
/// cost(Delta)/cost(Delta*) = (Delta/Delta* + Delta*/Delta) / 2.
pub fn optimal_probe_interval(
    probe_cost_s: f64,
    decode_tok_s: f64,
    mean_tokens_to_answer: f64,
) -> f64 {
    (2.0 * probe_cost_s * decode_tok_s * mean_tokens_to_answer)
        .max(0.0)
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_signal_is_a_running_minimum_so_early_crossings_are_final() {
        let mut rng = fastrand::Rng::with_seed(3);
        for _ in 0..200 {
            let n = 1 + rng.usize(..300);
            let lps: Vec<f32> = (0..n).map(|_| -rng.f32() * 2.0).collect();
            let w = 1 + rng.usize(..40);
            let full = low_window_mean(&lps, w).unwrap();
            let mut tr = WindowTracker::new(w);
            for (i, &x) in lps.iter().enumerate() {
                let v = tr.push(x);
                if i + 1 < w {
                    assert!(v.is_none());
                } else {
                    assert!(
                        v.unwrap() >= full - 1e-4,
                        "prefix value below the full value"
                    );
                }
            }
            if n >= w {
                assert!((tr.value().unwrap() - full).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn short_drafts_are_one_window() {
        assert_eq!(low_window_mean(&[-1.0, -3.0], 16), Some(-2.0));
        assert_eq!(low_window_mean(&[], 16), None);
        assert_eq!(low_window_mean(&[0.0, 0.0, -4.0, 0.0], 2), Some(-2.0));
    }

    #[test]
    fn beta_half_cdf_matches_closed_forms() {
        assert!((beta_half_cdf(1, 1) - 0.5).abs() < 1e-12);
        assert!((beta_half_cdf(3, 1) - 0.125).abs() < 1e-12);
        assert!((beta_half_cdf(2, 1) - 0.25).abs() < 1e-12);
        assert!((beta_half_cdf(3, 2) - 0.3125).abs() < 1e-12);
        // Symmetry I_{1/2}(a, b) = 1 - I_{1/2}(b, a).
        for (a, b) in [(2, 5), (7, 3), (4, 4)] {
            assert!((beta_half_cdf(a, b) + beta_half_cdf(b, a) - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn posterior_reproduces_and_generalises_the_consistency_rule() {
        let same = |a: &str, b: &str| a == b;
        let mut p = ProbePosterior::default();
        p.observe("42", same);
        assert!((p.majority_probability() - 0.75).abs() < 1e-6, "one probe");
        p.observe("42", same);
        assert!(
            (p.majority_probability() - 0.875).abs() < 1e-6,
            "two agreeing probes"
        );
        // The consistency rule (two consecutive agreeing probes) is the bound 0.85 at the start of the budget.
        assert!(p.majority_probability() >= collapsing_bound(0, 512, 0.85, 0.70));
        let mut q = ProbePosterior::default();
        q.observe("42", same);
        q.observe("41", same);
        q.observe("42", same);
        // [2, 1]: the consistency rule would not stop (last two disagree... then agree only once); the
        // posterior is 0.6875, below both ends of the bound, so neither does this rule.
        assert!((q.majority_probability() - 0.6875).abs() < 1e-6);
        assert!(q.majority_probability() < collapsing_bound(512, 512, 0.85, 0.70));
        // [3, 1] stops only once enough of the budget is used (the bound has collapsed below 0.8125).
        q.observe("42", same);
        assert!((q.majority_probability() - 0.8125).abs() < 1e-6);
        assert!(q.majority_probability() < collapsing_bound(0, 512, 0.85, 0.70));
        assert!(q.majority_probability() >= collapsing_bound(400, 512, 0.85, 0.70));
        assert_eq!(q.top(), Some(("42", 3)));
    }

    #[test]
    fn collapsing_bound_is_monotone_and_hits_its_ends() {
        let mut last = f32::MAX;
        for t in 0..=512 {
            let b = collapsing_bound(t, 512, 0.85, 0.70);
            assert!(b <= last + 1e-6);
            last = b;
        }
        assert!((collapsing_bound(0, 512, 0.85, 0.70) - 0.85).abs() < 1e-6);
        assert!((collapsing_bound(512, 512, 0.85, 0.70) - 0.70).abs() < 1e-6);
        assert!((collapsing_bound(5, 0, 0.85, 0.70) - 0.70).abs() < 1e-6);
    }

    #[test]
    fn wald_bound_exceeds_a_draft_at_the_fixed_rule_margin() {
        // Measured on the 40 open-ended drafts of benchmarks/early_escalation_study.json: per-token
        // standard deviation about 0.28; the adopted margin is 0.04 nats; drafts are 47-291 tokens.
        let n = wald_expected_n_lower(0.04, 0.28, 0.05, 0.05);
        assert!(n > 250.0 && n < 300.0, "{n}");
        // Four times the margin needs a sixteenth of the tokens.
        assert!((wald_expected_n_lower(0.16, 0.28, 0.05, 0.05) * 16.0 - n).abs() < 1e-6);
    }

    #[test]
    fn probe_interval_matches_the_configured_gap_within_the_flat_region() {
        // Idle reference machine: probe about 0.7 s, System 2 decodes about 8 tok/s, answers appear after
        // about 300 reasoning tokens on average (benchmarks: mean 348 with the exit).
        let d = optimal_probe_interval(0.7, 8.0, 300.0);
        assert!(d > 50.0 && d < 70.0, "{d}");
        let ratio = |delta: f64| (delta / d + d / delta) / 2.0;
        assert!(ratio(64.0) < 1.01);
        assert!(ratio(256.0) > 2.0);
    }
}
