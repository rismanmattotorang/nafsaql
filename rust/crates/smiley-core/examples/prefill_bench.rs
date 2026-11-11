//! nafs-aql: how fast a prompt is read (prefill), by the number of tokens fed per pass over the weights.
//!
//! A prompt is fed in chunks; each chunk reads every weight once and multiplies each weight row with the chunk's
//! activations. A larger chunk reads the weights less often but walks more activation bytes per row. This prints
//! prompt tokens per second for each chunk size, arms interleaved over `rounds`, on a prompt of `tokens` tokens.
//!
//!     cargo run --release -p smiley-core --example prefill_bench -- <model.gguf> [threads] [tokens] [rounds] [chunks]

use smiley_core::kv::{KvCache, KvPrec};
use smiley_core::model::{State, StepStats};
use smiley_core::pool::Pool;
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = PathBuf::from(
        args.next()
            .expect("usage: prefill_bench <model.gguf> [threads] [tokens] [rounds] [chunks]"),
    );
    let threads: usize = args.next().and_then(|x| x.parse().ok()).unwrap_or(4);
    let tokens: usize = args.next().and_then(|x| x.parse().ok()).unwrap_or(256);
    let rounds: usize = args.next().and_then(|x| x.parse().ok()).unwrap_or(3);
    let chunks: Vec<usize> = args
        .next()
        .map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect())
        .unwrap_or_else(|| vec![64]);
    let model = smiley_core::model::keep::open(&path).expect("open model");
    let c = model.cfg;
    let pool = Pool::new(threads);
    let prompt: Vec<u32> = (0..tokens).map(|i| ((i * 37) % 1000 + 5) as u32).collect();
    println!(
        "{} on {} threads: a prompt of {} tokens, {} rounds",
        path.file_name().unwrap().to_string_lossy(),
        threads,
        tokens,
        rounds
    );
    // One pass that is not timed: the first walk over a freshly mapped model file pays for its page faults.
    {
        let mut kv = KvCache::new(KvPrec::F16, c.n_layer, c.n_kv_head, c.head_dim, 72);
        let mut st = State::new(&c, 64, 1, 72, threads, model.rope_factors.as_deref());
        let mut stats = StepStats::default();
        model
            .forward(
                &mut st,
                &mut kv,
                &pool,
                &prompt[..64.min(tokens)],
                0,
                &[],
                &mut stats,
            )
            .unwrap();
    }
    let mut best = vec![f64::MAX; chunks.len()];
    let mut split = vec![(0u64, 0u64, 0u64, 0u64); chunks.len()];
    let mut all: Vec<Vec<f64>> = vec![vec![]; chunks.len()];
    for round in 0..rounds {
        for k in 0..chunks.len() {
            // Rotate which arm runs first, so a drift in the machine's load reaches every arm.
            let i = (k + round) % chunks.len();
            let chunk = chunks[i];
            let mut kv = KvCache::new(KvPrec::F16, c.n_layer, c.n_kv_head, c.head_dim, tokens + 8);
            let mut st = State::new(
                &c,
                chunk,
                1,
                tokens + 8,
                threads,
                model.rope_factors.as_deref(),
            );
            let mut stats = StepStats::default();
            let t0 = Instant::now();
            let mut pos = 0;
            while pos < tokens {
                let take = chunk.min(tokens - pos);
                model
                    .forward(
                        &mut st,
                        &mut kv,
                        &pool,
                        &prompt[pos..pos + take],
                        pos,
                        &[],
                        &mut stats,
                    )
                    .unwrap();
                pos += take;
            }
            let s = t0.elapsed().as_secs_f64();
            if s < best[i] {
                split[i] = (
                    stats.ns_matmul,
                    stats.ns_attn,
                    stats.ns_head,
                    stats.ns_total,
                );
            }
            best[i] = best[i].min(s);
            all[i].push(tokens as f64 / s);
        }
    }
    for (i, chunk) in chunks.iter().enumerate() {
        let mut v = all[i].clone();
        v.sort_by(|a, b| a.total_cmp(b));
        let (mm, at, hd, tot) = split[i];
        let pct = |x: u64| 100.0 * x as f64 / tot.max(1) as f64;
        println!(
            "chunk {:>4}: best {:>6.1} tokens/s, median {:>6.1}  matmul {:.0} % attention {:.0} % head {:.0} % other {:.0} %  (rounds: {})",
            chunk,
            tokens as f64 / best[i],
            v[v.len() / 2],
            pct(mm),
            pct(at),
            pct(hd),
            pct(tot.saturating_sub(mm + at + hd)),
            all[i].iter().map(|x| format!("{x:.1}")).collect::<Vec<_>>().join(" ")
        );
    }
}
