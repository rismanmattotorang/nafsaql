//! Calibration of the fast-path error estimate e1_hat (Prop. 1): logistic regression on gate features
//! with isotonic post-correction. Promotion (NFR-020, ER-005) is judged on *out-of-fold* predictions:
//! every labelled sample is scored by a model that did not see it (k-fold cross-validation), so the
//! reported ECE is a held-out estimate. The fit also stores the escalation-threshold quantiles that the
//! budget mode (FR-014) turns into lambda*.
use nafs_types::GateFeatures;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Calibration {
    pub version: String,
    pub fitted_at: String,
    /// Labelled samples used (all are held out once under k-fold).
    pub n: usize,
    /// Held-out (out-of-fold) ECE, 5 equal-mass bins.
    pub ece: f32,
    /// FR-R1: the held-out bins of e1 with their certified error bounds; empty for a calibration fitted before
    /// Phase 22 (then no bin is certified and the mode stays fixed until a refit).
    #[serde(default)]
    pub risk_bins: Vec<RiskBin>,
    pub brier: f32,
    pub weights: Vec<f32>,
    pub feature_names: Vec<String>,
    #[serde(default)]
    pub source_run: Option<String>,
    /// Number of labelled drafts that were wrong / right.
    #[serde(default)]
    pub n_pos: usize,
    #[serde(default)]
    pub n_neg: usize,
    /// In-sample ECE, reported for comparison only (never used for promotion).
    #[serde(default)]
    pub ece_in_sample: Option<f32>,
    #[serde(default)]
    pub folds: usize,
    /// Isotonic map (x = logistic output, y = corrected probability), ascending x.
    #[serde(default)]
    pub isotonic: Vec<(f32, f32)>,
    /// Quantiles q = 0..=100 % of the per-sample escalation threshold lambda_i (FR-014, budget mode).
    #[serde(default)]
    pub lambda_quantiles: Vec<f32>,
    #[serde(default)]
    pub reliability: Vec<(f32, f32, usize)>,
}

pub const NF: usize = 8;
pub const FEATURE_NAMES: [&str; NF] = [
    "bias",
    "mean_logprob",
    "min_logprob",
    "disagreement",
    "log_draft_tokens",
    "domain_hard",
    "no_samples",
    "low_window_logprob",
];

pub fn feature_vector(f: &GateFeatures) -> [f32; NF] {
    let mean = f.mean_logprob.unwrap_or(-0.5);
    let min = f.min_logprob.unwrap_or(-5.0).max(-20.0);
    // Phase 8: the lowest-window signal; a draft without it is treated as uniformly confident at its mean.
    let low = f.low_window_logprob.unwrap_or(mean).max(-20.0);
    let dis = 1.0 - f.agreement.unwrap_or(1.0);
    let len = ((f.draft_tokens.max(1)) as f32).ln();
    let hard = matches!(
        f.domain,
        nafs_types::Domain::Math | nafs_types::Domain::Code
    ) as i32 as f32;
    let none = f.agreement.is_none() as i32 as f32;
    [1.0, mean, min, dis, len, hard, none, low]
}

fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

/// L2-regularised logistic regression by full-batch gradient descent on standardised features.
fn logistic(feats: &[[f32; NF]], ys: &[bool]) -> Vec<f32> {
    let n = feats.len().max(1) as f32;
    // Standardise columns 1.. for conditioning, then map the weights back to raw scale.
    let mut mu = [0f32; NF];
    let mut sd = [1f32; NF];
    for j in 1..NF {
        mu[j] = feats.iter().map(|x| x[j]).sum::<f32>() / n;
        let v = feats.iter().map(|x| (x[j] - mu[j]).powi(2)).sum::<f32>() / n;
        sd[j] = v.sqrt().max(1e-3);
    }
    let z: Vec<[f32; NF]> = feats
        .iter()
        .map(|x| {
            let mut o = [1f32; NF];
            for j in 1..NF {
                o[j] = (x[j] - mu[j]) / sd[j];
            }
            o
        })
        .collect();
    let mut w = [0f32; NF];
    let rate = ys.iter().filter(|&&y| y).count() as f32 / n;
    w[0] = (rate.clamp(0.01, 0.99) / (1.0 - rate.clamp(0.01, 0.99))).ln();
    let (lr, l2) = (0.5f32, 1e-2f32);
    for _ in 0..3000 {
        let mut g = [0f32; NF];
        for (x, &y) in z.iter().zip(ys) {
            let p = sigmoid(w.iter().zip(x).map(|(a, b)| a * b).sum());
            let e = p - y as i32 as f32;
            for j in 0..NF {
                g[j] += e * x[j];
            }
        }
        for j in 0..NF {
            let reg = if j == 0 { 0.0 } else { l2 * w[j] };
            w[j] -= lr * (g[j] / n + reg);
        }
    }
    let mut raw = vec![0f32; NF];
    raw[0] = w[0];
    for j in 1..NF {
        raw[j] = w[j] / sd[j];
        raw[0] -= w[j] * mu[j] / sd[j];
    }
    raw
}

/// Pool-adjacent-violators: non-decreasing map from score to probability.
pub fn isotonic_fit(ps: &[f32], ys: &[bool]) -> Vec<(f32, f32)> {
    let mut idx: Vec<usize> = (0..ps.len()).collect();
    idx.sort_by(|&a, &b| {
        ps[a]
            .partial_cmp(&ps[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    // blocks: (sum_y, count, min_x, max_x)
    let mut blocks: Vec<(f32, f32, f32, f32)> = vec![];
    for i in idx {
        blocks.push((ys[i] as i32 as f32, 1.0, ps[i], ps[i]));
        while blocks.len() >= 2 {
            let k = blocks.len();
            let (a, b) = (blocks[k - 2], blocks[k - 1]);
            if a.0 / a.1 > b.0 / b.1 {
                blocks.truncate(k - 2);
                blocks.push((a.0 + b.0, a.1 + b.1, a.2, b.3));
            } else {
                break;
            }
        }
    }
    let mut out = vec![];
    for (sy, c, lo, hi) in blocks {
        let m = ((sy + 0.5) / (c + 1.0)).clamp(0.001, 0.999); // light smoothing
        out.push((lo, m));
        if hi > lo {
            out.push((hi, m));
        }
    }
    out
}

pub fn isotonic_apply(map: &[(f32, f32)], p: f32) -> f32 {
    if map.is_empty() {
        return p;
    }
    if p <= map[0].0 {
        return map[0].1;
    }
    for w in map.windows(2) {
        let ((x0, y0), (x1, y1)) = (w[0], w[1]);
        if p <= x1 {
            return if x1 > x0 {
                y0 + (y1 - y0) * (p - x0) / (x1 - x0)
            } else {
                y1
            };
        }
    }
    map.last().unwrap().1
}

impl Calibration {
    /// Hand-set, *uncalibrated* prior: more uncertainty and disagreement imply a higher error probability.
    pub fn prior() -> Self {
        Self {
            version: "prior-v0 (uncalibrated)".into(),
            fitted_at: "-".into(),
            n: 0,
            ece: f32::NAN,
            risk_bins: vec![],
            brier: f32::NAN,
            weights: vec![-2.2, -3.0, -0.08, 2.5, 0.1, 0.6, 0.3, -1.0],
            feature_names: FEATURE_NAMES.iter().map(|s| s.to_string()).collect(),
            source_run: None,
            n_pos: 0,
            n_neg: 0,
            ece_in_sample: None,
            folds: 0,
            isotonic: vec![],
            lambda_quantiles: vec![],
            reliability: vec![],
        }
    }
    fn logit_p(&self, f: &GateFeatures) -> f32 {
        let x = feature_vector(f);
        // Files written before the 7th or 8th feature existed have fewer weights; zip stops at the shorter.
        sigmoid(self.weights.iter().zip(x.iter()).map(|(w, v)| w * v).sum())
    }
    /// Estimated probability that the fast draft is wrong.
    pub fn e1(&self, f: &GateFeatures) -> f32 {
        isotonic_apply(&self.isotonic, self.logit_p(f))
    }
    pub fn is_fitted(&self) -> bool {
        self.n > 0
    }

    /// FR-R1: the bin a draft's e1 falls in, and whether its certified error bound is within `max_risk`.
    /// `None` when the calibration has no bins (fitted before Phase 22).
    pub fn certified_for(&self, e1: f32, max_risk: f32) -> Option<(&RiskBin, bool)> {
        let bin = self
            .risk_bins
            .iter()
            .find(|b| e1 >= b.lo && e1 <= b.hi)
            .or_else(|| self.risk_bins.last())?;
        Some((bin, bin.upper <= max_risk))
    }
    /// NFR-020 promotion test: enough held-out samples of both classes and held-out ECE within target.
    pub fn promotable(
        &self,
        min_n: usize,
        min_per_class: usize,
        max_ece: f32,
        max_risk: f32,
    ) -> Result<(), String> {
        if !self.is_fitted() {
            return Err("no fitted calibration (uncalibrated prior)".into());
        }
        if self.n < min_n {
            return Err(format!(
                "n = {} held-out samples < {} required (NFR-020)",
                self.n, min_n
            ));
        }
        if self.n_pos < min_per_class || self.n_neg < min_per_class {
            return Err(format!(
                "class counts wrong/right = {}/{} < {} each",
                self.n_pos, self.n_neg, min_per_class
            ));
        }
        if self.ece.is_nan() || self.ece > max_ece {
            return Err(format!("held-out ECE {:.3} > {:.3}", self.ece, max_ece));
        }
        // FR-R1: a calibration none of whose bins can keep a draft at the allowed risk would escalate everything,
        // which the fixed rule does more cheaply.
        if !self.risk_bins.iter().any(|b| b.upper <= max_risk) {
            return Err(format!(
                "no e1 bin is certified at max_risk {max_risk:.2} (lowest bound {:.3})",
                self.risk_bins
                    .iter()
                    .map(|b| b.upper)
                    .fold(f32::INFINITY, f32::min)
            ));
        }
        Ok(())
    }
    /// lambda* for escalation budget B: the (1-B) quantile of the per-sample thresholds (FR-014).
    pub fn lambda_star(&self, budget: f32) -> Option<f32> {
        if self.lambda_quantiles.len() < 2 {
            return None;
        }
        let q = ((1.0 - budget.clamp(0.0, 1.0)) * (self.lambda_quantiles.len() - 1) as f32).round()
            as usize;
        Some(self.lambda_quantiles[q.min(self.lambda_quantiles.len() - 1)].max(0.0))
    }

    /// Fit with k-fold cross-validation. `y` = true if the fast draft was wrong. `threshold` maps a
    /// sample and its e1 to lambda_i, the largest price at which the gate would still escalate it.
    pub fn fit(
        xs: &[GateFeatures],
        ys: &[bool],
        source_run: Option<String>,
        threshold: &dyn Fn(&GateFeatures, f32) -> f32,
    ) -> Self {
        let feats: Vec<[f32; NF]> = xs.iter().map(feature_vector).collect();
        let n = xs.len();
        let folds = if n >= 50 { 5 } else { 2.min(n.max(1)) };
        let mut order: Vec<usize> = (0..n).collect();
        fastrand::Rng::with_seed(17).shuffle(&mut order);
        let fold_of: Vec<usize> = {
            let mut f = vec![0; n];
            for (rank, &i) in order.iter().enumerate() {
                f[i] = rank % folds.max(1);
            }
            f
        };
        // Out-of-fold logistic scores.
        let mut oof = vec![0f32; n];
        for k in 0..folds.max(1) {
            let (tr_x, tr_y): (Vec<[f32; NF]>, Vec<bool>) = (0..n)
                .filter(|&i| fold_of[i] != k || folds <= 1)
                .map(|i| (feats[i], ys[i]))
                .unzip();
            let w = logistic(&tr_x, &tr_y);
            for i in (0..n).filter(|&i| fold_of[i] == k) {
                oof[i] = sigmoid(w.iter().zip(feats[i].iter()).map(|(a, b)| a * b).sum());
            }
        }
        // Held-out estimate of the full pipeline: isotonic fitted on other folds' OOF scores.
        let mut held = vec![0f32; n];
        for k in 0..folds.max(1) {
            let (px, py): (Vec<f32>, Vec<bool>) = (0..n)
                .filter(|&i| fold_of[i] != k || folds <= 1)
                .map(|i| (oof[i], ys[i]))
                .unzip();
            let m = isotonic_fit(&px, &py);
            for i in (0..n).filter(|&i| fold_of[i] == k) {
                held[i] = isotonic_apply(&m, oof[i]);
            }
        }
        let weights = logistic(&feats, ys);
        let isotonic = isotonic_fit(&oof, ys);
        let mut cal = Self {
            version: format!("fit-{}", chrono::Utc::now().format("%Y%m%d-%H%M%S")),
            fitted_at: chrono::Utc::now().to_rfc3339(),
            n,
            ece: ece(&held, ys, 5),
            risk_bins: risk_bins(&held, ys, 5, 0.05),
            brier: held
                .iter()
                .zip(ys)
                .map(|(p, &y)| (p - y as i32 as f32).powi(2))
                .sum::<f32>()
                / n.max(1) as f32,
            weights,
            feature_names: FEATURE_NAMES.iter().map(|s| s.to_string()).collect(),
            source_run,
            n_pos: ys.iter().filter(|&&y| y).count(),
            n_neg: ys.iter().filter(|&&y| !y).count(),
            ece_in_sample: None,
            folds,
            isotonic,
            lambda_quantiles: vec![],
            reliability: reliability(&held, ys, 5),
        };
        let ins: Vec<f32> = xs.iter().map(|f| cal.e1(f)).collect();
        cal.ece_in_sample = Some(ece(&ins, ys, 5));
        let mut lam: Vec<f32> = xs.iter().zip(&ins).map(|(f, &e)| threshold(f, e)).collect();
        lam.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        if !lam.is_empty() {
            cal.lambda_quantiles = (0..=100)
                .map(|q| lam[((lam.len() - 1) as f32 * q as f32 / 100.0).round() as usize])
                .collect();
        }
        cal
    }
}

/// FR-R1: one held-out bin of e1 with its certified error bound.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RiskBin {
    /// The bin's range of held-out e1 (the last bin's `hi` is 1.0, the first's `lo` 0.0).
    pub lo: f32,
    pub hi: f32,
    pub n: usize,
    /// Drafts in the bin that were wrong.
    pub errors: usize,
    /// One-sided Clopper-Pearson upper bound on the bin's error rate at `alpha / bins`.
    pub upper: f32,
}

/// The regularised incomplete beta function I_x(a, b) by Lentz's continued fraction (Numerical Recipes 6.4).
fn betainc(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    // The continued fraction converges fast on one side of the mean; use the symmetry on the other.
    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - betainc(1.0 - x, b, a);
    }
    let ln_beta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
    let front = (a * x.ln() + b * (1.0 - x).ln() - ln_beta).exp();
    let tiny = 1e-300;
    let mut c = 1.0;
    let mut d = 1.0 - (a + b) * x / (a + 1.0);
    if d.abs() < tiny {
        d = tiny;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..300 {
        let m = m as f64;
        let aa = m * (b - m) * x / ((a + 2.0 * m - 1.0) * (a + 2.0 * m));
        d = 1.0 + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (a + b + m) * x / ((a + 2.0 * m) * (a + 2.0 * m + 1.0));
        d = 1.0 + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < 1e-12 {
            break;
        }
    }
    front * h / a
}

/// ln Gamma(x) (Lanczos, g = 7).
fn ln_gamma(x: f64) -> f64 {
    const G: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        return (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = G[0];
    let t = x + 7.5;
    for (i, g) in G.iter().enumerate().skip(1) {
        a += g / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// FR-R1: the one-sided Clopper-Pearson upper bound on a rate with `k` events in `n` trials at level `alpha`:
/// the smallest `u` with I_u(k + 1, n - k) >= 1 - alpha, by bisection; `1 - alpha^(1/n)` when `k = 0`; 1.0
/// when nothing was observed or everything failed.
pub fn clopper_pearson_upper(k: usize, n: usize, alpha: f64) -> f64 {
    if n == 0 || k >= n {
        return 1.0;
    }
    if k == 0 {
        return 1.0 - alpha.powf(1.0 / n as f64);
    }
    let (a, b) = ((k + 1) as f64, (n - k) as f64);
    let (mut lo, mut hi) = (0.0f64, 1.0f64);
    for _ in 0..64 {
        let mid = 0.5 * (lo + hi);
        if betainc(mid, a, b) >= 1.0 - alpha {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    hi
}

/// FR-R1: equal-mass bins of the held-out e1 with each bin's certified error bound at `alpha / bins`
/// (Bonferroni over the bins, which are declared before any is read).
pub fn risk_bins(ps: &[f32], ys: &[bool], bins: usize, alpha: f32) -> Vec<RiskBin> {
    if ps.is_empty() || bins == 0 {
        return vec![];
    }
    let mut idx: Vec<usize> = (0..ps.len()).collect();
    idx.sort_by(|&a, &b| {
        ps[a]
            .partial_cmp(&ps[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let n = ps.len();
    let level = alpha as f64 / bins as f64;
    let mut out: Vec<RiskBin> = vec![];
    for b in 0..bins {
        let lo = b * n / bins;
        let hi = ((b + 1) * n / bins).min(n);
        if hi <= lo {
            continue;
        }
        let sl = &idx[lo..hi];
        let errors = sl.iter().filter(|&&i| ys[i]).count();
        out.push(RiskBin {
            lo: if out.is_empty() { 0.0 } else { ps[sl[0]] },
            hi: if hi == n { 1.0 } else { ps[sl[sl.len() - 1]] },
            n: sl.len(),
            errors,
            upper: clopper_pearson_upper(errors, sl.len(), level) as f32,
        });
    }
    out
}

/// Expected calibration error with equal-mass bins (spec Section 10.1).
pub fn ece(ps: &[f32], ys: &[bool], bins: usize) -> f32 {
    if ps.is_empty() {
        return f32::NAN;
    }
    reliability(ps, ys, bins)
        .iter()
        .map(|(mp, my, c)| (*c as f32 / ps.len() as f32) * (mp - my).abs())
        .sum()
}

/// Reliability diagram points (mean predicted, observed, count) for the dashboard.
pub fn reliability(ps: &[f32], ys: &[bool], bins: usize) -> Vec<(f32, f32, usize)> {
    let mut idx: Vec<usize> = (0..ps.len()).collect();
    idx.sort_by(|&a, &b| {
        ps[a]
            .partial_cmp(&ps[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let n = ps.len();
    let mut out = vec![];
    for b in 0..bins {
        let lo = b * n / bins;
        let hi = ((b + 1) * n / bins).min(n);
        if hi <= lo {
            continue;
        }
        let sl = &idx[lo..hi];
        out.push((
            sl.iter().map(|&i| ps[i]).sum::<f32>() / sl.len() as f32,
            sl.iter().map(|&i| ys[i] as i32 as f32).sum::<f32>() / sl.len() as f32,
            sl.len(),
        ));
    }
    out
}

#[cfg(test)]
mod risk_tests {
    use super::*;

    /// Known values (scipy `beta.ppf(0.95, k + 1, n - k)`): 0 of 20 at alpha 0.05 is 1 - 0.05^(1/20) = 0.1391; 1 of
    /// 10 is 0.3942; 5 of 100 is 0.1023. The bound shrinks with n and grows with k.
    #[test]
    fn the_clopper_pearson_upper_bound_matches_known_values() {
        assert!((clopper_pearson_upper(0, 20, 0.05) - 0.1391).abs() < 1e-3);
        assert!((clopper_pearson_upper(1, 10, 0.05) - 0.3942).abs() < 2e-3);
        assert!((clopper_pearson_upper(5, 100, 0.05) - 0.1023).abs() < 2e-3);
        assert!(clopper_pearson_upper(0, 200, 0.05) < clopper_pearson_upper(0, 20, 0.05));
        assert_eq!(clopper_pearson_upper(0, 0, 0.05), 1.0);
        assert_eq!(clopper_pearson_upper(3, 3, 0.05), 1.0);
    }

    /// Bins are equal-mass over the held-out e1, each with its own bound, and a draft is certified only in a bin
    /// whose bound is within the risk.
    #[test]
    fn risk_bins_certify_only_the_safe_bins() {
        // 100 drafts: the lower half of e1 is always right, the upper half wrong half the time.
        let ps: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
        let ys: Vec<bool> = (0..100).map(|i| i >= 50 && i % 2 == 0).collect();
        let bins = risk_bins(&ps, &ys, 5, 0.05);
        assert_eq!(bins.len(), 5);
        assert_eq!(bins[0].lo, 0.0);
        assert_eq!(bins[4].hi, 1.0);
        assert!(bins[0].errors == 0 && bins[0].upper < 0.25, "{:?}", bins[0]);
        assert!(bins[4].errors == 10 && bins[4].upper > 0.5, "{:?}", bins[4]);
        let cal = Calibration {
            risk_bins: bins,
            ..Calibration::prior()
        };
        assert_eq!(cal.certified_for(0.05, 0.25).map(|x| x.1), Some(true));
        assert_eq!(cal.certified_for(0.95, 0.25).map(|x| x.1), Some(false));
        assert!(Calibration::prior().certified_for(0.5, 0.25).is_none());
    }
}
