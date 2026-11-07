//! Live latency cost for the Aql Gate (docs/research/deepinfer_assessment.md, "Live latency cost for the
//! gate"). System 2's decode speed varies 6 to 9x with the owner's load on the shared host, while the gate's
//! latency cost delta_c is static. A rolling median of System 2 decode tok/s (the analogue of DeepInfer's
//! SamplerRuntimeEstimator) lets the gate scale delta_c by reference speed over current speed, clamped.
//! Off by default; never applied to lab (evaluation) requests, so paired runs and calibration stay reproducible.
use std::collections::VecDeque;

/// Deliberate stages shorter than this are too noisy to time decode speed (prefill and scheduling dominate).
pub const MIN_STAGE_TOKENS: u32 = 32;

/// Sliding-window median of System 2 decode speed (tok/s).
#[derive(Debug, Clone)]
pub struct RollingMedian {
    window: usize,
    min_samples: usize,
    xs: VecDeque<f32>,
}

impl RollingMedian {
    pub fn new(window: usize, min_samples: usize) -> Self {
        let window = window.max(1);
        Self {
            window,
            min_samples: min_samples.clamp(1, window),
            xs: VecDeque::with_capacity(window),
        }
    }
    /// Add one observation; non-finite and non-positive values are ignored.
    pub fn push(&mut self, x: f32) {
        if !x.is_finite() || x <= 0.0 {
            return;
        }
        if self.xs.len() == self.window {
            self.xs.pop_front();
        }
        self.xs.push_back(x);
    }
    /// Record a deliberate stage: `tokens` generated in `decode_ms` of server decode time.
    pub fn push_stage(&mut self, tokens: Option<u32>, decode_ms: Option<f64>) {
        if let (Some(t), Some(ms)) = (tokens, decode_ms) {
            if t >= MIN_STAGE_TOKENS && ms > 0.0 {
                self.push((t as f64 * 1000.0 / ms) as f32);
            }
        }
    }
    pub fn len(&self) -> usize {
        self.xs.len()
    }
    pub fn is_empty(&self) -> bool {
        self.xs.is_empty()
    }
    pub fn window(&self) -> usize {
        self.window
    }
    pub fn min_samples(&self) -> usize {
        self.min_samples
    }
    /// Median of the window, or None until `min_samples` observations have been seen.
    pub fn median(&self) -> Option<f32> {
        if self.xs.len() < self.min_samples {
            return None;
        }
        let mut v: Vec<f32> = self.xs.iter().copied().collect();
        v.sort_by(|a, b| a.total_cmp(b));
        let n = v.len();
        Some(if n % 2 == 1 {
            v[n / 2]
        } else {
            (v[n / 2 - 1] + v[n / 2]) / 2.0
        })
    }
}

/// Multiplier for delta_c: reference speed over current speed, clamped to [min_scale, max_scale].
/// A slower System 2 (busy host) raises the latency cost; a faster one lowers it.
pub fn scale(ref_tok_s: f32, current_tok_s: f32, min_scale: f32, max_scale: f32) -> f32 {
    if !(ref_tok_s > 0.0 && current_tok_s > 0.0) {
        return 1.0;
    }
    (ref_tok_s / current_tok_s).clamp(min_scale, max_scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_waits_for_min_samples() {
        let mut m = RollingMedian::new(8, 3);
        m.push(5.0);
        m.push(7.0);
        assert_eq!(m.median(), None);
        m.push(6.0);
        assert_eq!(m.median(), Some(6.0));
        m.push(9.0);
        assert_eq!(m.median(), Some(6.5));
    }

    #[test]
    fn window_slides_and_resists_outliers() {
        let mut m = RollingMedian::new(5, 3);
        for x in [10.0, 10.0, 10.0, 10.0, 10.0] {
            m.push(x);
        }
        // One stray fast or slow stage does not move the median.
        m.push(100.0);
        assert_eq!(m.median(), Some(10.0));
        // The host gets busy: once most of the window is slow, the median follows.
        for x in [3.0, 3.0, 3.0] {
            m.push(x);
        }
        assert_eq!(m.len(), 5);
        assert_eq!(m.median(), Some(3.0));
    }

    #[test]
    fn ignores_bad_observations() {
        let mut m = RollingMedian::new(4, 1);
        m.push(f32::NAN);
        m.push(f32::INFINITY);
        m.push(0.0);
        m.push(-2.0);
        assert!(m.is_empty());
        // Short stages and missing server timings are not samples.
        m.push_stage(Some(MIN_STAGE_TOKENS - 1), Some(1000.0));
        m.push_stage(Some(400), None);
        m.push_stage(None, Some(1000.0));
        m.push_stage(Some(400), Some(0.0));
        assert!(m.is_empty());
        m.push_stage(Some(400), Some(50_000.0));
        assert_eq!(m.median(), Some(8.0));
    }

    #[test]
    fn min_samples_is_bounded_by_window() {
        let m = RollingMedian::new(0, 10);
        assert_eq!((m.window(), m.min_samples()), (1, 1));
        let m = RollingMedian::new(4, 0);
        assert_eq!(m.min_samples(), 1);
    }

    #[test]
    fn scale_is_ref_over_current_and_clamped() {
        // Measured speeds on the shared host: 10.3 tok/s idle, 2.6 to 4.3 under load.
        assert!((scale(6.9, 6.9, 0.5, 3.0) - 1.0).abs() < 1e-6);
        assert!((scale(6.9, 4.6, 0.5, 3.0) - 1.5).abs() < 1e-6);
        assert!((scale(6.9, 10.3, 0.5, 3.0) - 6.9 / 10.3).abs() < 1e-6);
        assert_eq!(scale(6.9, 1.0, 0.5, 3.0), 3.0);
        assert_eq!(scale(6.9, 100.0, 0.5, 3.0), 0.5);
        // No usable speed: leave delta_c unchanged.
        assert_eq!(scale(6.9, 0.0, 0.5, 3.0), 1.0);
        assert_eq!(scale(0.0, 5.0, 0.5, 3.0), 1.0);
        assert_eq!(scale(6.9, f32::NAN, 0.5, 3.0), 1.0);
    }
}
