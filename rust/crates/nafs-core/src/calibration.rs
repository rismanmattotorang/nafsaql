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
    /// NFR-020 promotion test: enough held-out samples of both classes and held-out ECE within target.
    pub fn promotable(
        &self,
        min_n: usize,
        min_per_class: usize,
        max_ece: f32,
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
