//! Telemetry: the reason this engine exists.
//!
//! Every engine in Smiley's catalogue reports tokens per second. None of them
//! reports bytes moved, which the performance model says is the quantity that
//! *causes* the tokens per second. An engine that reports only the consequence
//! cannot be checked against its own model; this one reports the cause, so a run
//! carries the evidence for or against the claim made about it.
//!
//! The important derived number is `roof_utilisation`: achieved bandwidth over the
//! machine's measured bandwidth. If it is low, this engine is compute-bound, and
//! every roofline statement it makes describes our own kernel rather than the
//! memory system. That is acceptance criterion A3, and it is the one number that
//! can invalidate the rest.

use std::time::Instant;

/// What one generated token cost.
#[derive(Debug, Clone, Copy, Default)]
pub struct TokenTelem {
    pub step: u32,
    pub dt_s: f64,
    pub bytes_weights: u64,
    pub bytes_kv: u64,
    pub macs: u64,
    pub barriers: u64,
    /// Tokens proposed by prompt-lookup for the step that produced this token.
    pub drafted: u8,
    /// Of those, how many survived verification.
    pub accepted: u8,
    /// True when this token came out of a batched verification rather than a GEMV.
    pub from_gemm: bool,
    /// True for the one token that prefill produced. It is a real output token and
    /// belongs in the text and in the count, but its cost is the prompt's, not a
    /// decode step's — so the bandwidth figures exclude it. Mixing it in would
    /// divide a whole prefill's bytes by one token and report a bandwidth several
    /// times the machine's.
    pub is_prefill: bool,
}

impl TokenTelem {
    pub fn bytes_total(&self) -> u64 {
        self.bytes_weights + self.bytes_kv
    }
    /// Bytes per second actually sustained across the memory interface.
    pub fn achieved_bandwidth(&self) -> f64 {
        if self.dt_s > 0.0 {
            self.bytes_total() as f64 / self.dt_s
        } else {
            0.0
        }
    }
    pub fn arithmetic_intensity(&self) -> f64 {
        let b = self.bytes_total();
        if b == 0 {
            0.0
        } else {
            2.0 * self.macs as f64 / b as f64
        }
    }
}

/// The machine's own roof, measured rather than looked up.
///
/// A planner decision or a utilisation figure that cannot be reproduced is not a
/// measurement, so both numbers go into the run manifest alongside the results.
#[derive(Debug, Clone, Copy)]
pub struct Machine {
    /// Sustained streaming-read bandwidth, bytes per second.
    pub bandwidth: f64,
    /// Sustained fused multiply-add throughput, FLOP per second.
    pub flops: f64,
    pub threads: usize,
    pub simd: &'static str,
}

impl Machine {
    /// Ridge point of the roofline: the arithmetic intensity at which a kernel stops
    /// being bandwidth-bound. Below this, more FLOPs are free.
    pub fn ridge(&self) -> f64 {
        if self.bandwidth > 0.0 {
            self.flops / self.bandwidth
        } else {
            0.0
        }
    }

    pub fn probe(threads: usize) -> Machine {
        Machine {
            bandwidth: probe_bandwidth(threads),
            flops: probe_flops(threads),
            threads,
            simd: crate::simd_name(),
        }
    }

    /// The probe, reused across model loads. It streams memory and spins FMA chains for
    /// 80–146 ms — 60–70 % of loading a small model — to measure two numbers that depend
    /// on the machine and the thread count, not on the model, so a switch between models
    /// pays for it once. Entries expire after ten minutes: a probe taken while the
    /// machine was busy would otherwise be the denominator of every roof-utilisation
    /// figure until the app restarted.
    pub fn cached(threads: usize) -> Machine {
        use std::sync::Mutex;
        use std::time::{Duration, Instant};
        static CACHE: Mutex<Vec<(Instant, Machine)>> = Mutex::new(Vec::new());
        const TTL: Duration = Duration::from_secs(600);
        {
            let c = CACHE.lock().unwrap_or_else(|e| e.into_inner());
            if let Some((_, m)) = c
                .iter()
                .find(|(t, m)| m.threads == threads && t.elapsed() < TTL)
            {
                return *m;
            }
        }
        // Probed outside the lock: two first loads at once probe twice rather than one
        // waiting 140 ms for the other, and the later write wins.
        let m = Machine::probe(threads);
        let mut c = CACHE.lock().unwrap_or_else(|e| e.into_inner());
        c.retain(|(t, e)| e.threads != threads && t.elapsed() < TTL);
        c.push((Instant::now(), m));
        m
    }
}

/// Streaming read bandwidth.
///
/// The buffer is far larger than any L3 on a desktop part, so this measures the path
/// to DRAM rather than the cache hierarchy — which is the path a decode step takes,
/// because at 267 MB of weights per token nothing stays resident between tokens.
fn probe_bandwidth(threads: usize) -> f64 {
    const MB: usize = 1 << 20;
    let n = 192 * MB / 8; // u64 elements
    let buf: Vec<u64> = (0..n as u64).collect();
    let mut best = 0.0f64;
    for _ in 0..3 {
        let t = Instant::now();
        let total: u64 = std::thread::scope(|s| {
            let chunk = n.div_ceil(threads);
            let hs: Vec<_> = buf
                .chunks(chunk)
                .map(|c| s.spawn(move || c.iter().fold(0u64, |a, &v| a ^ v)))
                .collect();
            hs.into_iter()
                .map(|h| h.join().unwrap_or(0))
                .fold(0, |a, v| a ^ v)
        });
        let dt = t.elapsed().as_secs_f64();
        // `total` is consumed so the read cannot be optimised away.
        std::hint::black_box(total);
        if dt > 0.0 {
            best = best.max((n * 8) as f64 / dt);
        }
    }
    best
}

/// Register-resident FMA throughput. Deliberately touches no memory, so it measures
/// the compute roof and not a second bandwidth figure.
///
/// The number of independent dependency chains is what this probe is about. An FMA
/// has a latency of about four cycles and a core retires two per cycle, so a core
/// needs eight or more accumulators in flight to reach its throughput; with fewer it
/// measures latency. Two earlier versions got this wrong. The first ran 3M
/// iterations of a scalar `mul_add` (a libm call without compile-time FMA) and
/// reported 3.5 GFLOP/s. The second ran eight scalar lanes, which the compiler packs
/// into *one* vector register — one chain — and reported 28 GFLOP/s on a part that
/// sustains well over 300: the batched prefill then "exceeded the machine roof" at
/// 65 GFLOP/s, which is how the defect showed. Where AVX2 and FMA are present the
/// probe now runs twelve independent 8-wide FMA chains per thread; elsewhere it runs
/// 64 scalar lanes, which a vectorising compiler turns into many chains.
fn probe_flops(threads: usize) -> f64 {
    const ITERS: u64 = 20_000_000;
    let mut best = 0.0f64;
    for _ in 0..3 {
        let t = Instant::now();
        let flop_per_thread = std::thread::scope(|s| {
            let hs: Vec<_> = (0..threads)
                .map(|w| s.spawn(move || fma_burst(w, ITERS)))
                .collect();
            hs.into_iter()
                .map(|h| h.join().unwrap_or(0.0))
                .fold(0.0, f64::max)
        });
        let dt = t.elapsed().as_secs_f64();
        if dt > 0.0 {
            best = best.max(flop_per_thread * threads as f64 / dt);
        }
    }
    best
}

/// One thread's share of the probe; returns the FLOP it performed.
fn fma_burst(seed: usize, iters: u64) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: the features the function is compiled for were just detected.
            return unsafe { fma_burst_avx2(seed, iters / 4) };
        }
    }
    const LANES: usize = 64;
    let mut a = [0.0f32; LANES];
    for (i, v) in a.iter_mut().enumerate() {
        *v = (seed + i) as f32 * 0.5 + 1.0;
    }
    let (b, c) = (1.0000001f32, 0.9999999f32);
    let n = iters / 8;
    for _ in 0..n {
        for v in a.iter_mut() {
            *v = *v * b + c;
        }
    }
    std::hint::black_box(a.iter().sum::<f32>());
    (n * LANES as u64 * 2) as f64
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn fma_burst_avx2(seed: usize, iters: u64) -> f64 {
    use std::arch::x86_64::*;
    const CHAINS: usize = 12;
    let b = _mm256_set1_ps(1.0000001);
    let c = _mm256_set1_ps(0.9999999);
    let mut acc = [_mm256_setzero_ps(); CHAINS];
    for (i, v) in acc.iter_mut().enumerate() {
        *v = _mm256_set1_ps((seed + i) as f32 * 0.5 + 1.0);
    }
    for _ in 0..iters {
        for v in acc.iter_mut() {
            *v = _mm256_fmadd_ps(*v, b, c);
        }
    }
    let mut sum = _mm256_setzero_ps();
    for v in acc {
        sum = _mm256_add_ps(sum, v);
    }
    let mut out = [0.0f32; 8];
    _mm256_storeu_ps(out.as_mut_ptr(), sum);
    std::hint::black_box(out);
    // Eight lanes, two FLOP per fused multiply-add.
    (iters * CHAINS as u64 * 8 * 2) as f64
}

/// Where wall time went, in seconds. The parts of one phase of a turn — prefill or
/// decode — and they sum to that phase's wall time to within loop overhead.
///
/// This exists because a roof utilisation of 25 % says *that* the engine is not
/// bandwidth-bound and nothing about *why*. Splitting a step into the matmuls that
/// stream weights, the output head, attention, the sampler and "everything else"
/// separates a kernel that is slow from a step that is serial: the first is a
/// property of the matmul rows, the second is Amdahl's law, and they have different
/// fixes.
#[derive(Debug, Clone, Copy, Default)]
pub struct Phases {
    /// Layer projections (Q, K, V, O, gate, up, down).
    pub matmul_s: f64,
    /// The output head's 49 152-row matmul.
    pub head_s: f64,
    pub attn_s: f64,
    /// Norms, rope, residuals, SwiGLU, activation quantisation, cache writes.
    pub other_s: f64,
    /// Picking the next token from the logits.
    pub sample_s: f64,
    /// Searching the context for a speculative draft.
    pub draft_s: f64,
}

impl Phases {
    pub fn total(&self) -> f64 {
        self.matmul_s + self.head_s + self.attn_s + self.other_s + self.sample_s + self.draft_s
    }
    /// Fraction of the phase spent outside the parallel matmuls, i.e. the serial part.
    pub fn serial_fraction(&self) -> f64 {
        let t = self.total();
        if t > 0.0 {
            (self.other_s + self.sample_s + self.draft_s) / t
        } else {
            0.0
        }
    }
    pub fn add_step(&mut self, s: &crate::model::StepStats) {
        self.matmul_s += s.ns_matmul as f64 / 1e9;
        self.head_s += s.ns_head as f64 / 1e9;
        self.attn_s += s.ns_attn as f64 / 1e9;
        self.other_s += s.ns_other() as f64 / 1e9;
    }
}

/// Everything one assistant turn cost, and the numbers derived from it.
#[derive(Debug, Clone, Default)]
pub struct TurnTelem {
    /// Wall-time split of the prompt pass and of the decode steps.
    pub prefill_phases: Phases,
    pub decode_phases: Phases,
    /// Positions in the cache when the turn ended, and the cache's capacity.
    pub ctx_used: usize,
    pub ctx_max: usize,
    /// Exchanges dropped from the front of the conversation to make this turn fit.
    pub dropped_exchanges: usize,
    /// Cache positions this turn kept from a previous conversation instead of
    /// prefilling them (after a reset, a seeded history or dropped exchanges).
    pub prefix_reused: usize,
    /// True when the caller cancelled the turn; the text and timings are what had been
    /// produced up to that point, not a failure.
    pub cancelled: bool,
    /// Multiply-accumulates of the prompt pass, for `prefill_gflops`.
    pub prefill_macs: u64,
    /// Forward passes over the weights during decode (one per step, speculative or not).
    pub passes: u64,
    /// Wall time of every decode step, including the last one when it produced only
    /// the end-of-turn token (which has no token row). The phases sum to this.
    pub decode_steps_s: f64,
    pub tokens: Vec<TokenTelem>,
    pub prefill_s: f64,
    pub prompt_tokens: usize,
    pub wall_s: f64,
    /// Bytes read during prefill, which are not part of the decode accounting.
    pub prefill_bytes: u64,
    /// DSpark drafting, when a drafter is attached: rounds drafted, the draft tokens
    /// verified (the chosen prefix), and the wall seconds spent in the drafter's
    /// block forward and in feeding it committed positions.
    pub dspark_rounds: u64,
    pub dspark_verified: u64,
    pub dspark_draft_s: f64,
    pub dspark_ctx_s: f64,
    /// Bytes the drafter read (its own weights), apart from the target's.
    pub dspark_bytes: u64,
}

impl TurnTelem {
    pub fn n(&self) -> usize {
        self.tokens.len()
    }
    pub fn ttft_s(&self) -> Option<f64> {
        self.tokens.first().map(|t| t.dt_s)
    }
    pub fn tok_per_s(&self) -> f64 {
        if self.wall_s > 0.0 {
            self.n() as f64 / self.wall_s
        } else {
            0.0
        }
    }
    fn decode_tokens(&self) -> impl Iterator<Item = &TokenTelem> {
        self.tokens.iter().filter(|t| !t.is_prefill)
    }
    pub fn bytes_weights(&self) -> u64 {
        self.decode_tokens().map(|t| t.bytes_weights).sum()
    }
    pub fn bytes_kv(&self) -> u64 {
        self.decode_tokens().map(|t| t.bytes_kv).sum()
    }
    /// Tokens produced by decode steps, i.e. excluding the prefill token. This is
    /// the denominator for any per-token cost figure.
    pub fn n_decode(&self) -> usize {
        self.decode_tokens().count()
    }
    pub fn bytes_total(&self) -> u64 {
        self.bytes_weights() + self.bytes_kv()
    }
    pub fn macs(&self) -> u64 {
        self.decode_tokens().map(|t| t.macs).sum()
    }
    pub fn barriers(&self) -> u64 {
        self.decode_tokens().map(|t| t.barriers).sum()
    }

    /// Decode-phase wall time: the sum of per-token deltas, excluding prefill. This
    /// is the denominator the bandwidth figure needs — mixing prefill into it would
    /// flatter the result, because prefill moves the same bytes for many tokens.
    pub fn decode_s(&self) -> f64 {
        self.decode_tokens().map(|t| t.dt_s).sum()
    }

    pub fn achieved_bandwidth(&self) -> f64 {
        let dt = self.decode_s();
        if dt > 0.0 {
            self.bytes_total() as f64 / dt
        } else {
            0.0
        }
    }

    /// **A3.** Fraction of the machine's measured streaming bandwidth this turn
    /// actually sustained. Below roughly a half, the engine is compute-bound and its
    /// roofline claims do not apply to itself.
    pub fn roof_utilisation(&self, m: &Machine) -> f64 {
        if m.bandwidth > 0.0 {
            self.achieved_bandwidth() / m.bandwidth
        } else {
            0.0
        }
    }

    pub fn arithmetic_intensity(&self) -> f64 {
        let b = self.bytes_total();
        if b == 0 {
            0.0
        } else {
            2.0 * self.macs() as f64 / b as f64
        }
    }

    /// Prompt throughput. The number that decides time-to-first-token on a long
    /// prompt, and the one the batched kernels exist to raise.
    pub fn prefill_tok_per_s(&self) -> f64 {
        if self.prefill_s > 0.0 {
            self.prompt_tokens as f64 / self.prefill_s
        } else {
            0.0
        }
    }

    /// Sustained arithmetic during prefill, in GFLOP/s, from counted multiply-adds.
    /// Prefill is the compute-bound phase, so this — not bandwidth — is its figure
    /// of merit; compare it against the machine's measured `flops`.
    pub fn prefill_gflops(&self) -> f64 {
        if self.prefill_s > 0.0 {
            2.0 * self.prefill_macs as f64 / self.prefill_s / 1e9
        } else {
            0.0
        }
    }

    /// Per-decode-token latency at the given quantile (0..=1), nearest-rank. A mean
    /// hides the spike a cold page or a contended core puts on one token; the tail is
    /// what a user watching the stream actually feels.
    pub fn latency_quantile(&self, q: f64) -> Option<f64> {
        let mut v: Vec<f64> = self.decode_tokens().map(|t| t.dt_s).collect();
        if v.is_empty() {
            return None;
        }
        v.sort_by(|a, b| a.total_cmp(b));
        let rank = ((q.clamp(0.0, 1.0) * v.len() as f64).ceil() as usize).clamp(1, v.len());
        Some(v[rank - 1])
    }

    /// Coefficient of variation of decode-token latency. Near zero on a steady
    /// engine; large when something else is competing for the cores.
    pub fn latency_cv(&self) -> Option<f64> {
        let v: Vec<f64> = self.decode_tokens().map(|t| t.dt_s).collect();
        if v.len() < 2 {
            return None;
        }
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        if mean <= 0.0 {
            return None;
        }
        let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (v.len() - 1) as f64;
        Some(var.sqrt() / mean)
    }

    /// Mean tokens emitted per forward pass over the weights. 1.0 without
    /// speculation; above 1.0 when drafts were accepted, which is the only way decode
    /// gets under one weight-read per token. `None` before any decode step ran.
    pub fn tokens_per_pass(&self) -> Option<f64> {
        (self.passes > 0).then(|| self.n_decode() as f64 / self.passes as f64)
    }

    pub fn drafted(&self) -> u64 {
        self.tokens.iter().map(|t| t.drafted as u64).sum()
    }
    pub fn accepted(&self) -> u64 {
        self.tokens.iter().map(|t| t.accepted as u64).sum()
    }
    /// Cumulative seconds per token, the shape `record_turn` stores. The first
    /// entry's cumulative time is the TTFT by that function's definition.
    pub fn cumulative(&self) -> Vec<(i64, i64, f64)> {
        let mut cum = 0.0;
        self.tokens
            .iter()
            .map(|t| {
                cum += t.dt_s;
                (t.step as i64, 0i64, cum)
            })
            .collect()
    }
    /// Fraction of drafted tokens that survived verification. `None` when nothing
    /// was drafted — never 0.0, which would read as "speculation failed" rather
    /// than "speculation was not attempted".
    pub fn acceptance_rate(&self) -> Option<f64> {
        let d = self.drafted();
        (d > 0).then(|| self.accepted() as f64 / d as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acceptance_is_none_when_nothing_was_drafted() {
        let mut t = TurnTelem::default();
        t.tokens.push(TokenTelem {
            step: 1,
            dt_s: 0.1,
            ..Default::default()
        });
        // Not 0.0: "not attempted" and "attempted and always rejected" are different
        // facts and the UI must not render them the same way.
        assert_eq!(t.acceptance_rate(), None);
        t.tokens.push(TokenTelem {
            step: 2,
            drafted: 4,
            accepted: 3,
            ..Default::default()
        });
        assert_eq!(t.acceptance_rate(), Some(0.75));
    }

    #[test]
    fn bandwidth_uses_decode_time_not_wall_time() {
        let t = TurnTelem {
            tokens: vec![
                // The prefill token: 5 s and a whole prompt's worth of bytes.
                TokenTelem {
                    dt_s: 5.0,
                    bytes_weights: 50_000_000,
                    is_prefill: true,
                    ..Default::default()
                },
                TokenTelem {
                    dt_s: 0.1,
                    bytes_weights: 1_000_000,
                    ..Default::default()
                },
                TokenTelem {
                    dt_s: 0.1,
                    bytes_weights: 1_000_000,
                    ..Default::default()
                },
            ],
            prefill_s: 5.0,
            wall_s: 5.2,
            ..Default::default()
        };
        assert_eq!(t.decode_s(), 0.2, "prefill must not be in decode time");
        assert_eq!(t.n(), 3, "the prefill token is still a real output token");
        assert_eq!(t.n_decode(), 2);
        assert_eq!(t.ttft_s(), Some(5.0));
        // 2 MB over 0.2 s. Including the 5 s prefill would report 0.4 MB/s and make
        // a fast engine look ten times slower than it is.
        assert_eq!(t.achieved_bandwidth(), 10_000_000.0);
    }

    #[test]
    fn roof_utilisation_is_a_fraction_of_measured_bandwidth() {
        let m = Machine {
            bandwidth: 20e9,
            flops: 100e9,
            threads: 4,
            simd: "test",
        };
        assert_eq!(m.ridge(), 5.0);
        let t = TurnTelem {
            tokens: vec![TokenTelem {
                dt_s: 1.0,
                bytes_weights: 10_000_000_000,
                ..Default::default()
            }],
            ..Default::default()
        };
        assert_eq!(t.roof_utilisation(&m), 0.5);
        // A machine that failed to probe must not produce a divide-by-zero or an
        // infinite utilisation that would pass an A3 assertion.
        let dead = Machine {
            bandwidth: 0.0,
            flops: 0.0,
            threads: 1,
            simd: "none",
        };
        assert_eq!(t.roof_utilisation(&dead), 0.0);
        assert_eq!(dead.ridge(), 0.0);
    }

    #[test]
    fn latency_quantiles_use_decode_tokens_only_and_nearest_rank() {
        let mut t = TurnTelem::default();
        // The prefill token's 5 s must never reach the tail statistics.
        t.tokens.push(TokenTelem {
            dt_s: 5.0,
            is_prefill: true,
            ..Default::default()
        });
        for i in 1..=100 {
            t.tokens.push(TokenTelem {
                dt_s: i as f64 / 1000.0,
                ..Default::default()
            });
        }
        assert_eq!(t.latency_quantile(0.5), Some(0.050));
        assert_eq!(t.latency_quantile(0.99), Some(0.099));
        assert_eq!(t.latency_quantile(1.0), Some(0.100));
        assert_eq!(TurnTelem::default().latency_quantile(0.5), None);
        assert!(t.latency_cv().unwrap() > 0.0);
        assert_eq!(TurnTelem::default().latency_cv(), None);
    }

    #[test]
    fn phases_sum_and_the_serial_fraction_excludes_the_parallel_matmuls() {
        let p = Phases {
            matmul_s: 6.0,
            head_s: 1.0,
            attn_s: 1.0,
            other_s: 1.0,
            sample_s: 0.5,
            draft_s: 0.5,
        };
        assert_eq!(p.total(), 10.0);
        // other + sample + draft are serial; attention and the matmuls are split
        // across the pool.
        assert_eq!(p.serial_fraction(), 0.2);
        assert_eq!(Phases::default().serial_fraction(), 0.0);
    }

    #[test]
    fn a_cached_probe_is_the_same_measurement_and_costs_nothing() {
        let a = Machine::cached(3);
        let t = std::time::Instant::now();
        let b = Machine::cached(3);
        assert!(
            t.elapsed() < std::time::Duration::from_millis(5),
            "{:?}",
            t.elapsed()
        );
        assert_eq!(
            (a.bandwidth, a.flops, a.threads),
            (b.bandwidth, b.flops, b.threads)
        );
        // Another thread count is another measurement, not the cached one.
        assert_eq!(Machine::cached(2).threads, 2);
        assert_eq!(
            Machine::cached(3).bandwidth,
            a.bandwidth,
            "and the first is still there"
        );
    }

    #[test]
    fn the_machine_probe_returns_plausible_numbers() {
        let m = Machine::probe(2);
        // Any real machine is above 1 GB/s and below 10 TB/s; outside that the probe
        // is measuring something other than memory and must not be trusted as the
        // denominator of A3.
        assert!(
            m.bandwidth > 1e9 && m.bandwidth < 1e13,
            "bandwidth {}",
            m.bandwidth
        );
        assert!(m.flops > 1e8 && m.flops < 1e14, "flops {}", m.flops);
        assert!(m.ridge() > 0.1, "ridge {}", m.ridge());
    }
}
