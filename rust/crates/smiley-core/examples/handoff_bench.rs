//! nafs-aql: what a request pays to take a conversation's cache over from the prefix store and to give it back.
//!
//! A conversation of `depth` tokens is read once. Then `turns` requests each extend it by `suffix` tokens and write
//! `new` tokens, the way an agent's steps or a chat's turns extend a transcript. For every turn this prints the
//! time `admit` took (finding and taking the cache), the time to the first token, the time of an ordinary decode
//! step, and the time of the step that finished the request (which gives the cache back).
//!
//!     cargo run --release -p smiley-core --example handoff_bench -- <model.gguf> [threads] [depths] [turns] [suffix] [new]
//!
//! `HANDOFF_FUSE=0` gives the prompt's last token a pass of its own, as before Phase 34, so the two parts of the
//! change (no copy; the first token in the last chunk's pass) can be measured apart.

use smiley_core::kv::KvPrec;
use smiley_core::serve::{Engine, SlotCfg};
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = PathBuf::from(
        args.next()
            .expect("usage: handoff_bench <model.gguf> [threads] [depths] [turns] [suffix] [new]"),
    );
    let threads: usize = args.next().and_then(|x| x.parse().ok()).unwrap_or(3);
    let depths: Vec<usize> = args
        .next()
        .map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect())
        .unwrap_or_else(|| vec![512, 2048]);
    let turns: usize = args.next().and_then(|x| x.parse().ok()).unwrap_or(4);
    let suffix: usize = args.next().and_then(|x| x.parse().ok()).unwrap_or(8);
    let new: usize = args.next().and_then(|x| x.parse().ok()).unwrap_or(6);
    let model = smiley_core::model::keep::open(&path).expect("open model");
    let ctx = depths.iter().max().copied().unwrap_or(512) + turns * (suffix + new + 2) + 64;
    println!(
        "{} on {} threads, {} turns of +{} prompt tokens and {} new tokens",
        path.file_name().unwrap().to_string_lossy(),
        threads,
        turns,
        suffix,
        new
    );
    println!("depth  turn  admit_ms  first_token_ms  decode_step_ms  finish_step_ms  prefilled");
    for &depth in &depths {
        let mut e = Engine::new(model.clone(), threads, ctx, 1, KvPrec::F16, 2048 << 20);
        e.fuse_first_token = std::env::var("HANDOFF_FUSE").as_deref() != Ok("0");
        let cfg = SlotCfg {
            max_tokens: new,
            // No stop token: every turn writes exactly `new` tokens.
            stop: vec![],
            ..SlotCfg::default()
        };
        let mut convo: Vec<u32> = (0..depth).map(|i| ((i * 37) % 1000 + 5) as u32).collect();
        for turn in 0..=turns {
            if turn > 0 {
                convo.extend((0..suffix).map(|i| ((turn * 131 + i * 17) % 900 + 11) as u32));
            }
            let t0 = Instant::now();
            let id = e.admit(convo.clone(), cfg.clone()).expect("admit");
            let admit_ms = t0.elapsed().as_secs_f64() * 1e3;
            let (mut first_ms, mut steps) = (0.0, vec![]);
            let (finish_ms, prefilled) = loop {
                let ts = Instant::now();
                let out = e.step();
                let ms = ts.elapsed().as_secs_f64() * 1e3;
                let Some(em) = out.into_iter().find(|em| em.slot == id) else {
                    continue;
                };
                if first_ms == 0.0 {
                    first_ms = t0.elapsed().as_secs_f64() * 1e3;
                }
                convo.push(em.token);
                if let Some(c) = em.cost {
                    break (ms, c.prefilled);
                }
                steps.push(ms);
            };
            // The last sampled token was never fed: the next prompt holds it, the cache does not.
            steps.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mid = steps.get(steps.len() / 2).copied().unwrap_or(0.0);
            if turn > 0 {
                println!(
                    "{depth:5}  {turn:4}  {admit_ms:8.1}  {first_ms:14.1}  {mid:14.1}  {finish_ms:14.1}  {prefilled:9}"
                );
            }
        }
    }
}
