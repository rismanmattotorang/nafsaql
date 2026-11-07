//! Forward time of the memory module on this CPU (NFR-K08): random weights at the Nafs-Aql size (Qwen3-Embedding-0.6B
//! into Qwen3-1.7B: E = 1024, h_L = 2048) and the paper's (E = 2560, h_L = 4096), one question, N stored documents.
//! cargo run --release -p nafs-memorilla --example memory_bench > benchmarks/memorilla_forward.json

use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use nafs_memorilla::{MemoryConfig, MemoryModule};

fn main() {
    let dev = Device::Cpu;
    let mut rows = Vec::new();
    for (label, cfg) in [
        ("nafs-aql (E 1024 -> 2048)", MemoryConfig::paper(1024, 2048)),
        ("paper (E 2560 -> 4096)", MemoryConfig::paper(2560, 4096)),
    ] {
        let vm = VarMap::new();
        let m = MemoryModule::new(cfg, VarBuilder::from_varmap(&vm, DType::F32, &dev)).unwrap();
        let q = Tensor::randn(0f32, 1.0, (1, cfg.embedding_dim), &dev).unwrap();
        for n in [10usize, 100, 1000, 10_000] {
            let docs = Tensor::randn(0f32, 1.0, (1, n, cfg.embedding_dim), &dev).unwrap();
            let _ = m.forward(&docs, None, &q).unwrap(); // warm-up
            let time = |f: &dyn Fn() -> Tensor| -> f64 {
                let mut t: Vec<f64> = (0..5)
                    .map(|_| {
                        let s = Instant::now();
                        let _ = f().sum_all().unwrap().to_scalar::<f32>().unwrap();
                        s.elapsed().as_secs_f64() * 1000.0
                    })
                    .collect();
                t.sort_by(|a, b| a.total_cmp(b));
                (t[2] * 10.0).round() / 10.0
            };
            let full = time(&|| m.forward(&docs, None, &q).unwrap());
            let s = Instant::now();
            let prep = m.prepare(&docs, None).unwrap();
            let prepare_ms = (s.elapsed().as_secs_f64() * 10000.0).round() / 10.0;
            let per_question = time(&|| m.forward_prepared(&prep, &q).unwrap());
            let row = serde_json::json!({ "module": label, "parameters": cfg.parameters(), "documents": n,
                "full_forward_ms": full, "prepare_once_ms": prepare_ms, "per_question_ms": per_question,
                "prepared_mb": prep.elem_count() * 4 / 1_048_576 });
            eprintln!("{row}");
            rows.push(row);
        }
    }
    let load = std::process::Command::new("sysctl")
        .args(["-n", "vm.loadavg"])
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "what": "memory module forward, f32, Candle 0.11 CPU, batch 1, median of 5 after one warm-up; prepare_once = document keys/values computed once per collection, per_question = forward_prepared",
            "machine": "Intel Core i3-8100, 4 cores, AVX2, macOS 15.4.1",
            "host_load_average": load,
            "rows": rows,
        }))
        .unwrap()
    );
}
