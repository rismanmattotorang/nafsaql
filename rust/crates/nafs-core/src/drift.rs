//! Drift monitor (FR-042, P4-T04): two-sided tabular CUSUM on standardised series of the gate's
//! calibration error (y - e1_hat) and of fast-path accuracy, per domain, with a shuffled-label negative
//! control. The in-control mean and standard deviation come from a reference window at the start of
//! the series, so the chart asks "has the process changed since the reference period?".
//!
//! Thresholds: k is the allowance and h the decision interval, both in units of sigma. The spec's
//! k = 0.5 sigma with h = 5 sigma gives a two-sided in-control run length of only about 470 observations
//! (fewer when the reference mean and sigma are estimated), which contradicts the Phase-4 exit criterion
//! (no alarm on a 500-observation negative control, alpha = 0.01). The defaults use k = 1 sigma (the
//! optimal allowance for the 2 sigma shift the criterion names), h = 6 sigma and a 100-observation
//! reference window, which absorbs the error of estimating the in-control mean and sigma. Detection of
//! a 2 sigma shift takes about 7 observations (ASSUMPTIONS A-035).
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CusumParams {
    pub k: f64,
    pub h: f64,
    /// Observations used to estimate the in-control mean and standard deviation.
    pub reference: usize,
}
impl Default for CusumParams {
    fn default() -> Self {
        Self {
            k: 1.0,
            h: 6.0,
            reference: 100,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CusumResult {
    pub n: usize,
    pub reference_mean: f64,
    pub reference_sd: f64,
    /// First index (0-based, in the full series) at which either side crossed h.
    pub alarm_at: Option<usize>,
    /// Direction of the first alarm: "up" (mean increased) or "down".
    pub direction: Option<String>,
    pub s_hi: Vec<f64>,
    pub s_lo: Vec<f64>,
}

/// Run a two-sided CUSUM. Returns None when the series is too short for a reference window.
pub fn cusum(xs: &[f64], p: CusumParams) -> Option<CusumResult> {
    let r = p.reference.min(xs.len() / 2);
    if r < 10 {
        return None;
    }
    let refw = &xs[..r];
    let mean = refw.iter().sum::<f64>() / r as f64;
    let var = refw.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (r - 1) as f64;
    let sd = var.sqrt().max(1e-6);
    let (mut hi, mut lo) = (0.0f64, 0.0f64);
    let mut out = CusumResult {
        n: xs.len(),
        reference_mean: mean,
        reference_sd: sd,
        ..Default::default()
    };
    for (i, x) in xs.iter().enumerate() {
        let z = (x - mean) / sd;
        hi = (hi + z - p.k).max(0.0);
        lo = (lo - z - p.k).max(0.0);
        out.s_hi.push(hi);
        out.s_lo.push(lo);
        if out.alarm_at.is_none() && i >= r && (hi > p.h || lo > p.h) {
            out.alarm_at = Some(i);
            out.direction = Some(if hi > p.h { "up" } else { "down" }.into());
        }
    }
    Some(out)
}

/// Negative control: the same chart on the series with its order shuffled (a stationary series with
/// the same marginal distribution). An alarm here means the chart is too sensitive for this data.
pub fn negative_control(xs: &[f64], p: CusumParams, seed: u64) -> Option<CusumResult> {
    let mut v = xs.to_vec();
    fastrand::Rng::with_seed(seed).shuffle(&mut v);
    cusum(&v, p)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn gauss(rng: &mut fastrand::Rng) -> f64 {
        // Box-Muller
        let (u, v) = (rng.f64().max(1e-12), rng.f64());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
    #[test]
    fn detects_a_two_sigma_shift_within_50_observations() {
        for seed in 0..20 {
            let mut rng = fastrand::Rng::with_seed(seed);
            let mut xs: Vec<f64> = (0..100).map(|_| gauss(&mut rng)).collect();
            xs.extend((0..50).map(|_| 2.0 + gauss(&mut rng)));
            let r = cusum(&xs, CusumParams::default()).unwrap();
            let at = r.alarm_at.expect("alarm");
            assert!((100..150).contains(&at), "seed {seed}: alarm at {at}");
        }
    }
    #[test]
    fn negative_control_is_silent_over_500_observations() {
        // The exit criterion: no alarm on a 500-observation in-control series (checked over many seeds;
        // the expected false-alarm rate is below 1%).
        let mut alarms = 0;
        for seed in 0..200 {
            let mut rng = fastrand::Rng::with_seed(1000 + seed);
            let xs: Vec<f64> = (0..500).map(|_| gauss(&mut rng)).collect();
            alarms += negative_control(&xs, CusumParams::default(), seed)
                .unwrap()
                .alarm_at
                .is_some() as u32;
        }
        assert!(
            alarms <= 2,
            "{alarms} false alarms in 200 negative controls"
        );
    }
    #[test]
    fn spec_k_half_would_fail_the_exit_criterion() {
        // Documents A-035: with k = 0.5 sigma and h = 5 sigma, false alarms over 500 observations are common.
        let p = CusumParams {
            k: 0.5,
            h: 5.0,
            ..Default::default()
        };
        let mut alarms = 0;
        for seed in 0..100 {
            let mut rng = fastrand::Rng::with_seed(5000 + seed);
            let xs: Vec<f64> = (0..500).map(|_| gauss(&mut rng)).collect();
            alarms += cusum(&xs, p).unwrap().alarm_at.is_some() as u32;
        }
        assert!(alarms > 20, "{alarms}");
    }
    #[test]
    fn short_series_has_no_chart() {
        assert!(cusum(&[0.0; 12], CusumParams::default()).is_none());
    }
}
