//! Probe: gradients of the memory module alone (a synthetic loss), to find non-finite ones.
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use nafs_memorilla::{MemoryConfig, MemoryModule};

fn main() {
    for retrieval in [true, false] {
        let cfg = MemoryConfig {
            retrieval_init: retrieval,
            num_memories: 4,
            ..MemoryConfig::paper(64, 32)
        };
        let vm = VarMap::new();
        let m =
            MemoryModule::new(cfg, VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu)).unwrap();
        let docs = Tensor::randn(0f32, 0.03, (1, 20, 64), &Device::Cpu).unwrap();
        let q = Tensor::randn(0f32, 0.03, (1, 64), &Device::Cpu).unwrap();
        let out = m.forward(&docs, None, &q).unwrap();
        let w = Tensor::randn(0f32, 1.0, out.dims(), &Device::Cpu).unwrap();
        let loss = (out * w).unwrap().sum_all().unwrap();
        let g = loss.backward().unwrap();
        let data = vm.data().lock().unwrap();
        let mut names: Vec<_> = data.keys().cloned().collect();
        names.sort();
        let mut bad = vec![];
        for n in &names {
            let v = &data[n];
            match g.get(v) {
                Some(t) => {
                    let s = t
                        .sqr()
                        .unwrap()
                        .sum_all()
                        .unwrap()
                        .to_scalar::<f32>()
                        .unwrap();
                    if !s.is_finite() {
                        bad.push(format!("{n}=NaN"));
                    }
                }
                None => bad.push(format!("{n}=none")),
            }
        }
        println!(
            "retrieval_init={retrieval}: loss {} ; problems: {:?}",
            loss.to_scalar::<f32>().unwrap(),
            bad
        );
        let init: Vec<String> = names
            .iter()
            .filter(|n| n.contains("in_proj_weight") || n.contains("memory_queries"))
            .map(|n| {
                format!(
                    "{n} std {:.4}",
                    data[n]
                        .as_tensor()
                        .sqr()
                        .unwrap()
                        .mean_all()
                        .unwrap()
                        .sqrt()
                        .unwrap()
                        .to_scalar::<f32>()
                        .unwrap()
                )
            })
            .collect();
        println!("  init: {init:?}");
    }
}
