//! FR-094: hedged concurrent attempts on System 2, and when to use them.
//!
//! llama-server decodes the sequences of its slots in one batch. On the reference CPU that batch is far cheaper than
//! the same sequences one after the other (`benchmarks/batch_scaling_*.json`: four streams of Qwen3-4B take 1.78 times
//! one stream's wall time, so 2.25 times the tokens per second). A verifiable task can use that: sample several
//! candidates at once, verify each as it finishes, keep the first that passes and cancel the rest. This is the
//! "hedged request" of Dean and Barroso (2013) with an exact verifier as the arbiter, and repeated sampling with a
//! verifier (Brown et al., 2024) at the price of one batch instead of one sequence per sample.
//!
//! Width is a trade: more candidates per round raise the chance that a round passes, but a batch of N costs c(N) times
//! one stream. For one-stream success probability `p`, `rounds` repair rounds, and cost factor `c(N)`:
//!
//! ```text
//!   success(N)  = 1 - (1-p)^(N*rounds)
//!   E[time](N)  = c(N) * sum_{k<rounds} (1-p)^(N*k)          (in units of one stream's attempt)
//!   U(N)        = success(N) / E[time](N)                    verified answers per unit time
//! ```
//!
//! The policy picks the N with the largest U, and widens only when U(N) beats U(1) by `min_gain`. With the measured
//! costs it keeps N = 1 when p is high (a second stream costs more than it saves) and goes to 4 when p is low (the
//! escalated, hard tasks, where sequential retries are the dominant waste: 52 % of model time in the 100-prompt run).
use std::collections::BTreeMap;
use std::sync::Mutex;

/// Probability that at least one of `width * rounds` independent attempts passes.
pub fn success_prob(p: f64, width: usize, rounds: u32) -> f64 {
    1.0 - (1.0 - p.clamp(0.0, 1.0)).powi((width as u32 * rounds) as i32)
}

/// Expected wall time in units of one stream's attempt: a round of `width` costs `cost`, a later round is needed only
/// when every candidate of the earlier rounds failed.
pub fn expected_time(p: f64, width: usize, rounds: u32, cost: f64) -> f64 {
    let q = (1.0 - p.clamp(0.0, 1.0)).powi(width as i32);
    cost * (0..rounds).map(|k| q.powi(k as i32)).sum::<f64>()
}

/// Verified answers per unit of time.
pub fn utility(p: f64, width: usize, rounds: u32, cost: f64) -> f64 {
    success_prob(p, width, rounds) / expected_time(p, width, rounds, cost).max(1e-9)
}

/// The width for one request. `costs[i]` is the measured wall-time factor of `i + 1` concurrent streams;
/// `free_slots` is what the server can run now (a width above it would queue behind itself).
pub fn choose_width(
    p: f64,
    rounds: u32,
    costs: &[f64],
    max_width: usize,
    free_slots: usize,
    min_gain: f64,
) -> usize {
    let cap = max_width.min(costs.len()).min(free_slots.max(1)).max(1);
    let base = utility(p, 1, rounds, costs.first().copied().unwrap_or(1.0));
    let mut best = (1usize, base);
    for n in 2..=cap {
        let u = utility(p, n, rounds, costs[n - 1]);
        if u > best.1 * (1.0 + min_gain) && u > base * (1.0 + min_gain) {
            best = (n, u);
        }
    }
    best.0
}

/// Online estimate of the probability that one System 2 attempt on a task that reached it passes, per domain: a
/// Beta-Bernoulli posterior mean with the configured prior, weighted as `prior_weight` pseudo-attempts.
pub struct SuccessEstimator {
    prior: BTreeMap<String, f64>,
    weight: f64,
    counts: Mutex<BTreeMap<String, (f64, f64)>>,
}

impl SuccessEstimator {
    pub fn new(prior: BTreeMap<String, f32>, weight: f32) -> Self {
        Self {
            prior: prior.into_iter().map(|(k, v)| (k, v as f64)).collect(),
            weight: weight.max(0.0) as f64,
            counts: Mutex::new(BTreeMap::new()),
        }
    }
    pub fn p(&self, domain: &str) -> f64 {
        let prior = self.prior.get(domain).copied().unwrap_or(0.5);
        let (s, n) = self
            .counts
            .lock()
            .unwrap()
            .get(domain)
            .copied()
            .unwrap_or((0.0, 0.0));
        ((prior * self.weight + s) / (self.weight + n).max(1e-9)).clamp(0.01, 0.99)
    }
    pub fn record(&self, domain: &str, passed: bool) {
        let mut c = self.counts.lock().unwrap();
        let e = c.entry(domain.to_string()).or_insert((0.0, 0.0));
        e.0 += f64::from(u8::from(passed));
        e.1 += 1.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The costs measured on the reference machine for Qwen3-4B-Thinking (benchmarks/batch_scaling_system2.json).
    const C: [f64; 4] = [1.0, 1.5, 1.9, 1.78];

    #[test]
    fn hard_tasks_widen_and_easy_tasks_do_not() {
        // p = 0.25 (a hard escalated code task): four candidates win.
        assert_eq!(choose_width(0.25, 3, &C, 4, 4, 0.05), 4);
        // p = 0.8 (an easy one): a second stream costs more than it saves.
        assert_eq!(choose_width(0.8, 3, &C, 4, 4, 0.05), 1);
        // The cost curve is not monotone (c(4) < c(3)), so three is never chosen when four is available.
        for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6] {
            assert_ne!(choose_width(p, 3, &C, 4, 4, 0.0), 3, "p = {p}");
        }
    }

    /// The factors measured at the depth of a real repair prompt (about 1,600 tokens, `batch_scaling_system2_depth1200.json`):
    /// the batch gain fades with context and a round of four costs 3.4 single attempts. The policy then widens only when a
    /// single attempt almost never succeeds (p of 0.07 or less, tasks that are nearly hopeless, where more candidates raise
    /// the chance more than they cost), never for the success probabilities of tasks that reach System 2 (0.1 and up). At
    /// short context it widens for p up to 0.48. That is why concurrent hedging on System 2 is off by default on this CPU.
    #[test]
    fn at_repair_prompt_depth_the_width_stays_one_for_realistic_success_rates() {
        let deep = [1.0, 2.4, 3.1, 3.4];
        for i in 10..=99 {
            let p = i as f64 / 100.0;
            assert_eq!(choose_width(p, 3, &deep, 4, 4, 0.05), 1, "p = {p}");
        }
        for i in 1..=7 {
            let p = i as f64 / 100.0;
            assert_eq!(
                choose_width(p, 3, &deep, 4, 4, 0.05),
                4,
                "hopeless tasks widen: p = {p}"
            );
        }
        // At short context, where the batch is cheap, it widens for every p up to 0.48.
        assert_eq!(choose_width(0.25, 3, &C, 4, 4, 0.05), 4);
        assert_eq!(choose_width(0.6, 3, &C, 4, 4, 0.05), 1);
    }

    #[test]
    fn width_never_exceeds_the_free_slots_or_the_cap() {
        assert_eq!(choose_width(0.2, 3, &C, 4, 2, 0.05), 2);
        assert_eq!(choose_width(0.2, 3, &C, 4, 1, 0.05), 1);
        assert_eq!(
            choose_width(0.2, 3, &C, 4, 0, 0.05),
            1,
            "no free slot still runs one attempt"
        );
        assert_eq!(choose_width(0.2, 3, &C, 2, 4, 0.05), 2);
        assert_eq!(
            choose_width(0.2, 3, &[1.0], 4, 4, 0.05),
            1,
            "costs bound the width"
        );
    }

    #[test]
    fn the_formulas_match_hand_computation() {
        // p = 0.5, 3 rounds, N = 1: success 0.875, expected time 1 + 0.5 + 0.25.
        assert!((success_prob(0.5, 1, 3) - 0.875).abs() < 1e-12);
        assert!((expected_time(0.5, 1, 3, 1.0) - 1.75).abs() < 1e-12);
        // N = 4: success 1 - 0.5^12; time c * (1 + 0.5^4 + 0.5^8).
        assert!((success_prob(0.5, 4, 3) - (1.0 - 0.5f64.powi(12))).abs() < 1e-12);
        assert!(
            (expected_time(0.5, 4, 3, 1.78) - 1.78 * (1.0 + 0.0625 + 0.00390625)).abs() < 1e-12
        );
    }

    #[test]
    fn width_one_is_the_sequential_loop() {
        // With N = 1 and c = 1 the utility is the old loop's success per expected time.
        for p in [0.1, 0.5, 0.9] {
            assert!(
                (utility(p, 1, 3, 1.0) - success_prob(p, 1, 3) / expected_time(p, 1, 3, 1.0)).abs()
                    < 1e-12
            );
        }
    }

    #[test]
    fn the_estimator_moves_from_its_prior_to_the_observed_rate() {
        let e = SuccessEstimator::new([("code".to_string(), 0.4f32)].into_iter().collect(), 6.0);
        assert!((e.p("code") - 0.4).abs() < 1e-6);
        assert!(
            (e.p("other") - 0.5).abs() < 1e-9,
            "unknown domains start at 0.5"
        );
        for _ in 0..30 {
            e.record("code", false);
        }
        assert!(e.p("code") < 0.1, "{}", e.p("code"));
        for _ in 0..300 {
            e.record("code", true);
        }
        assert!(e.p("code") > 0.85);
    }
}
