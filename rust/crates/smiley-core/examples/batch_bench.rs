//! nafs-aql: what batching several users into one weight pass is worth on this machine.
//!
//! Decode reads every weight once per pass. Serving `n` users one after another reads them `n` times; serving
//! them in one pass reads them once and does `n` dot products per weight row. This measures both, on the same
//! model and machine, alternating the arms.
//!
//!     cargo run --release -p smiley-core --example batch_bench -- <model.gguf> [threads] [steps] [ctx]

use smiley_core::kv::{KvCache, KvPrec};
use smiley_core::model::{Caches, State, StepStats};
use smiley_core::pool::Pool;
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = PathBuf::from(
        args.next()
            .expect("usage: batch_bench <model.gguf> [threads] [steps] [ctx]"),
    );
    let threads: usize = args.next().and_then(|x| x.parse().ok()).unwrap_or(3);
    let steps: usize = args.next().and_then(|x| x.parse().ok()).unwrap_or(24);
    let ctx: usize = args.next().and_then(|x| x.parse().ok()).unwrap_or(512);
    let model = smiley_core::model::keep::open(&path).expect("open model");
    let c = model.cfg;
    let pool = Pool::new(threads);
    println!(
        "{} on {} threads: {} steps per user, {} positions of context\n",
        path.file_name().unwrap().to_string_lossy(),
        threads,
        steps,
        ctx
    );
    println!(
        "{:>5} {:>12} {:>12} {:>10} {:>12} {:>12}",
        "users", "one-by-one", "batched", "speed-up", "per-user b/s", "GB per token"
    );

    for n in [1usize, 2, 3, 4, 6, 8] {
        let mut best = (f64::MAX, f64::MAX);
        // Two rounds, alternating which arm runs first.
        for round in 0..2 {
            let mut times = [0.0f64; 2];
            for arm in if round == 0 { [0, 1] } else { [1, 0] } {
                let mut caches: Vec<KvCache> = (0..n)
                    .map(|_| KvCache::new(KvPrec::F16, c.n_layer, c.n_kv_head, c.head_dim, ctx))
                    .collect();
                // The scratch must hold the widest batch used here: the 64-token history prefill, or one row per user.
                let mut st = State::new(
                    &c,
                    64.max(n),
                    n.max(1),
                    ctx,
                    threads,
                    model.rope_factors.as_deref(),
                );
                let mut stats = StepStats::default();
                // Give every user the same amount of history, so the attention work is equal in both arms.
                let history: Vec<u32> = (0..64).map(|i| ((i * 37) % 1000 + 5) as u32).collect();
                for kv in caches.iter_mut() {
                    model
                        .forward(&mut st, kv, &pool, &history, 0, &[], &mut stats)
                        .unwrap();
                }
                let t0 = Instant::now();
                for s in 0..steps {
                    let pos = history.len() + s;
                    if arm == 0 {
                        // One user at a time: every user pays for its own read of the weights.
                        for kv in caches.iter_mut() {
                            model
                                .forward(&mut st, kv, &pool, &[7], pos, &[0], &mut stats)
                                .unwrap();
                        }
                    } else {
                        // Every user in one pass.
                        let mut refs: Vec<&mut KvCache> = caches.iter_mut().collect();
                        let tokens = vec![7u32; n];
                        let positions = vec![pos; n];
                        let want: Vec<usize> = (0..n).collect();
                        model
                            .forward_rows(
                                &mut st,
                                &mut Caches::PerRow(&mut refs),
                                &pool,
                                &tokens,
                                &positions,
                                &want,
                                &mut stats,
                            )
                            .unwrap();
                    }
                }
                times[arm] = t0.elapsed().as_secs_f64();
            }
            best = (best.0.min(times[0]), best.1.min(times[1]));
        }
        let (seq, bat) = best;
        let total = (n * steps) as f64;
        println!(
            "{:>5} {:>9.2} t/s {:>9.2} t/s {:>9.2}x {:>9.2} t/s {:>11.2}",
            n,
            total / seq,
            total / bat,
            seq / bat,
            total / bat / n as f64,
            (model.weight_bytes() as f64) / 1e9,
        );
    }
}
