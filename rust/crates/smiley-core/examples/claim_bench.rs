//! nafs-aql (Phase 34): fixed shares against claimed blocks, on a quiet machine and on one that is not the engine's
//! alone.
//!
//! A fork-join over fixed shares ends when its slowest worker ends, so a worker the system has descheduled holds
//! the whole pass back. With claimed blocks (`pool::Claims`) the workers that are running take what it has not
//! started. This measures both on the same model in one process, arms interleaved: tokens per second writing
//! (one row per pass) and reading a prompt (64 rows per pass), with `busy` other threads spinning at the engine's
//! own scheduling class, the way a second engine on the same cores does.
//!
//!     cargo run --release -p smiley-core --example claim_bench -- <model.gguf> [threads] [busy] [rounds] [decode_tokens] [prompt_tokens]
//!
//! `threads` and `busy` take comma-separated lists; every combination is run.

use smiley_core::kv::{KvCache, KvPrec};
use smiley_core::model::{State, StepStats};
use smiley_core::pool::{self, Pool};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

fn list(s: Option<String>, default: &[usize]) -> Vec<usize> {
    s.map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect())
        .unwrap_or_else(|| default.to_vec())
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if v.is_empty() {
        0.0
    } else {
        v[v.len() / 2]
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = PathBuf::from(args.next().expect(
        "usage: claim_bench <model.gguf> [threads] [busy] [rounds] [decode_tokens] [prompt_tokens]",
    ));
    let threads = list(args.next(), &[3, 4]);
    let busy = list(args.next(), &[0, 1]);
    let rounds: usize = args.next().and_then(|x| x.parse().ok()).unwrap_or(4);
    let n_decode: usize = args.next().and_then(|x| x.parse().ok()).unwrap_or(48);
    let n_prompt: usize = args.next().and_then(|x| x.parse().ok()).unwrap_or(192);
    let model = smiley_core::model::keep::open(&path).expect("open model");
    let c = model.cfg;
    let prompt: Vec<u32> = (0..n_prompt)
        .map(|i| ((i * 37) % 1000 + 5) as u32)
        .collect();
    let ctx = n_prompt + n_decode + 8;
    println!(
        "{}: {} rounds, {} tokens written at depth {}, a prompt of {} tokens read in chunks of 64",
        path.file_name().unwrap().to_string_lossy(),
        rounds,
        n_decode,
        n_prompt,
        n_prompt
    );
    println!("threads  busy  shares   write_tok_s  read_tok_s");
    for &t in &threads {
        let pool = Pool::new(t);
        let mut st = State::new(&c, 64, 1, ctx, t, model.rope_factors.as_deref());
        // One pass that is not timed: the first walk over a freshly mapped model file pays for its page faults.
        {
            let mut kv = KvCache::new(KvPrec::F16, c.n_layer, c.n_kv_head, c.head_dim, ctx);
            let mut stats = StepStats::default();
            model
                .forward(
                    &mut st,
                    &mut kv,
                    &pool,
                    &prompt[..64.min(n_prompt)],
                    0,
                    &[0],
                    &mut stats,
                )
                .unwrap();
        }
        for &b in &busy {
            let stop = Arc::new(AtomicBool::new(false));
            let hogs: Vec<_> = (0..b)
                .map(|_| {
                    let stop = stop.clone();
                    std::thread::spawn(move || {
                        pool::raise_priority();
                        let mut x = 1u64;
                        while !stop.load(Ordering::Relaxed) {
                            for _ in 0..4096 {
                                x = x
                                    .wrapping_mul(6364136223846793005)
                                    .wrapping_add(1442695040888963407);
                            }
                            std::hint::black_box(x);
                        }
                    })
                })
                .collect();
            let mut write = [vec![], vec![]];
            let mut read = [vec![], vec![]];
            for r in 0..rounds {
                for k in 0..2 {
                    let arm = (k + r) % 2;
                    pool::set_work_conserving(arm == 1);
                    let mut kv = KvCache::new(KvPrec::F16, c.n_layer, c.n_kv_head, c.head_dim, ctx);
                    let mut stats = StepStats::default();
                    let t0 = Instant::now();
                    let mut pos = 0;
                    while pos < n_prompt {
                        let take = 64.min(n_prompt - pos);
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
                    read[arm].push(n_prompt as f64 / t0.elapsed().as_secs_f64());
                    let t0 = Instant::now();
                    let mut tok = 17u32;
                    for i in 0..n_decode {
                        model
                            .forward(
                                &mut st,
                                &mut kv,
                                &pool,
                                &[tok],
                                n_prompt + i,
                                &[0],
                                &mut stats,
                            )
                            .unwrap();
                        // Greedy, so both arms write the same tokens.
                        let l = st.logits_row(0, c.n_vocab);
                        tok = (0..c.n_vocab)
                            .max_by(|&a, &b| l[a].partial_cmp(&l[b]).unwrap())
                            .unwrap() as u32;
                    }
                    write[arm].push(n_decode as f64 / t0.elapsed().as_secs_f64());
                }
            }
            stop.store(true, Ordering::Relaxed);
            for h in hogs {
                let _ = h.join();
            }
            // The machine's load drifts between arms, so the paired ratio per round is the figure to read: the
            // two arms of a round ran within seconds of each other.
            let ratio = |x: &[Vec<f64>; 2]| -> (f64, usize) {
                let mut r: Vec<f64> = x[1].iter().zip(&x[0]).map(|(c, f)| c / f).collect();
                let wins = r.iter().filter(|v| **v > 1.0).count();
                (median(&mut r), wins)
            };
            let (wr, ww) = ratio(&write);
            let (rr, rw) = ratio(&read);
            for (arm, name) in ["fixed", "claimed"].iter().enumerate() {
                println!(
                    "{t:7}  {b:4}  {name:7}  {:11.2}  {:10.1}",
                    median(&mut write[arm]),
                    median(&mut read[arm])
                );
            }
            println!(
                "{t:7}  {b:4}  claimed/fixed, median of {rounds} paired rounds: write {wr:.3} ({ww} of {rounds} faster), read {rr:.3} ({rw} of {rounds} faster)"
            );
        }
    }
    pool::set_work_conserving(true);
}
