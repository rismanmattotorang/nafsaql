//! Convert a GGUF's projection matrices into a LittleBit sidecar (`<model>.slb`).
//!
//! ```text
//! cargo run --release --example lb_convert -- model.gguf [--bpw 1.0] [--stages 2]
//!     [--refine 1] [--kv-mult 1] [--head] [--rank N] [--out path.slb]
//! ```
//!
//! The method, the format and what PTQ-only quality honestly buys are in
//! `docs/LITTLEBIT-PLAN.md`. The next `Session::open` on the model picks the
//! sidecar up automatically; `SMILEY_CORE_LB=0` runs dense again.

use smiley_core::lbq::{convert_gguf, ConvertOpts};
use std::path::PathBuf;

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut path: Option<PathBuf> = None;
    let mut opts = ConvertOpts {
        verbose: true,
        ..Default::default()
    };
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut val = |what: &str| -> Result<&String, String> {
            it.next().ok_or(format!("{what} needs a value"))
        };
        match a.as_str() {
            "--bpw" => opts.bpw = val("--bpw")?.parse().map_err(|e| format!("--bpw: {e}"))?,
            "--stages" => {
                opts.stages = val("--stages")?
                    .parse()
                    .map_err(|e| format!("--stages: {e}"))?
            }
            "--refine" => {
                opts.refine = val("--refine")?
                    .parse()
                    .map_err(|e| format!("--refine: {e}"))?
            }
            "--kv-mult" => {
                opts.kv_mult = val("--kv-mult")?
                    .parse()
                    .map_err(|e| format!("--kv-mult: {e}"))?
            }
            "--rank" => {
                opts.rank_override =
                    Some(val("--rank")?.parse().map_err(|e| format!("--rank: {e}"))?)
            }
            "--head" => opts.head = true,
            "--out" => opts.out = Some(PathBuf::from(val("--out")?)),
            other if path.is_none() && !other.starts_with("--") => {
                path = Some(PathBuf::from(other))
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    let path = path.ok_or("usage: lb_convert model.gguf [--bpw 1.0] [--stages 2] …")?;
    if !(1..=8).contains(&opts.stages) {
        return Err("--stages must be 1..=8".into());
    }

    eprintln!(
        "converting {path:?}  bpw {}  stages {}  refine {}  threads {}",
        opts.bpw, opts.stages, opts.refine, opts.threads
    );
    let t0 = std::time::Instant::now();
    let rep = convert_gguf(&path, &opts)?;
    eprintln!(
        "wrote {:?}: {} tensors, {:.1} MB (source {:.1} MB, {:.1}×), {:.3} bpw over converted weights, {:.0}s",
        rep.out,
        rep.converted.len(),
        rep.data_bytes as f64 / 1e6,
        rep.source_bytes as f64 / 1e6,
        rep.source_bytes as f64 / rep.data_bytes.max(1) as f64,
        rep.bpw,
        t0.elapsed().as_secs_f64()
    );
    if !rep.skipped.is_empty() {
        eprintln!(
            "kept dense ({}): {}",
            rep.skipped.len(),
            rep.skipped.join(", ")
        );
    }
    let worst = rep
        .converted
        .iter()
        .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
    if let Some((name, r, err)) = worst {
        eprintln!("worst reconstruction: {name} (r {r}) rel err {err:.4}");
    }
    let mean = rep.converted.iter().map(|c| c.2).sum::<f64>() / rep.converted.len().max(1) as f64;
    eprintln!("mean reconstruction rel err {mean:.4}  (Q4_K itself is about 0.1)");
    if mean > 0.3 {
        eprintln!(
            "\nWARNING: this sidecar will not produce coherent text. PTQ-only LittleBit leaves most of \
             the weight energy unexplained (docs/LITTLEBIT-PLAN.md section 8); the paper's quality \
             comes from distillation QAT. Measured on SmolLM2-360M: gibberish at 1.0 and 3.0 bpw.\n\
             It will still be picked up automatically next to the model; remove it or set \
             SMILEY_CORE_LB=0 to run dense."
        );
    }
    Ok(())
}
