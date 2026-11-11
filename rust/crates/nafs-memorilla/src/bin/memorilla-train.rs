//! `memorilla-train`: train or fine-tune a Memorilla module on a knowledge-base dataset (Phase 6).
//!
//! memorilla-train --data <dataset dir> --out <run dir> [--decoder models/hf/Qwen3-1.7B]
//!   [--gguf models/Qwen3-1.7B-Q4_K_M.gguf] [--epochs 5] [--lr 1e-4] [--accumulation 16] [--dropout 0.1]
//!   [--max-examples 0] [--init <checkpoint dir>] [--seed 23] [--placeholder "<|fim_pad|>"]
//!
//! The defaults are the reference's from-scratch recipe (`third_party/memorilla/configs/factkg.yaml`).
use std::path::PathBuf;

use nafs_memorilla::train::{TrainOpts, Trainer, DEFAULT_ACCUMULATION, DEFAULT_EPOCHS, DEFAULT_LR};

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let get = |k: &str| {
        a.iter()
            .position(|x| x == k)
            .and_then(|i| a.get(i + 1))
            .cloned()
    };
    let num = |k: &str, d: f64| get(k).and_then(|v| v.parse().ok()).unwrap_or(d);
    let (Some(data), Some(out)) = (get("--data"), get("--out")) else {
        eprintln!("usage: memorilla-train --data <dataset dir> --out <run dir> [--epochs N] [--lr X] [--accumulation N] [--dropout P] [--max-examples N] [--init DIR]");
        std::process::exit(2);
    };
    let opts = TrainOpts {
        data: data.into(),
        out: out.into(),
        decoder: get("--decoder")
            .unwrap_or_else(|| "models/hf/Qwen3-1.7B".into())
            .into(),
        gguf: get("--gguf")
            .unwrap_or_else(|| "models/Qwen3-1.7B-Q4_K_M.gguf".into())
            .into(),
        epochs: num("--epochs", DEFAULT_EPOCHS as f64) as usize,
        lr: num("--lr", DEFAULT_LR),
        accumulation: num("--accumulation", DEFAULT_ACCUMULATION as f64) as usize,
        dropout: get("--dropout").and_then(|v| v.parse().ok()),
        max_examples: num("--max-examples", 0.0) as usize,
        init: get("--init").map(PathBuf::from),
        num_memories: num("--memories", 16.0) as usize,
        seed: num("--seed", 23.0) as u64,
        placeholder: get("--placeholder")
            .unwrap_or_else(|| nafs_memorilla::train::PLACEHOLDER.to_string()),
    };
    match Trainer::new(opts).and_then(|mut t| t.run()) {
        Ok(v) => println!("{{\"ok\": true, \"val_loss\": {v}}}"),
        Err(e) => {
            eprintln!("training failed: {e}");
            println!(
                "{{\"ok\": false, \"error\": {}}}",
                serde_json::Value::String(e)
            );
            std::process::exit(1);
        }
    }
}
