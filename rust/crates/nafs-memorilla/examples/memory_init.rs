//! Write an **untrained** memory module (Candle's default initialisation) as a checkpoint, for exercising the serving
//! path before a trained module exists (Phase 5). Its memory tokens carry no knowledge.
//! cargo run --release -p nafs-memorilla --example memory_init -- <dir> [embedding_dim] [output_dim]
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use nafs_memorilla::{MemoryConfig, MemoryModule};

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let dir = std::path::PathBuf::from(
        a.first()
            .expect("usage: memory_init <dir> [embedding_dim] [output_dim]"),
    );
    let e = a.get(1).and_then(|x| x.parse().ok()).unwrap_or(1024);
    let h = a.get(2).and_then(|x| x.parse().ok()).unwrap_or(2048);
    let cfg = MemoryConfig::paper(e, h);
    let vm = VarMap::new();
    let _ = MemoryModule::new(cfg, VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu)).unwrap();
    MemoryModule::save(&vm, &cfg, &dir).unwrap();
    println!(
        "untrained module, {} parameters -> {}",
        cfg.parameters(),
        dir.display()
    );
}
