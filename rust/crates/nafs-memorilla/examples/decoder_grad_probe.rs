//! Probe: is the gradient that reaches the memory rows through the decoder finite?
use candle_core::{DType, Device, Tensor, Var};
use nafs_memorilla::decoder::Decoder;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let d = Decoder::load(std::path::Path::new(&a[0])).unwrap();
    let g = smiley_core::gguf::Gguf::open(std::path::Path::new(&a[1])).unwrap();
    let tk = smiley_core::tokenizer::Tokenizer::from_gguf(&g).unwrap();
    let tpl = smiley_core::template::Template::detect(&g).unwrap();
    let sys = format!(
        "You are a helpful assistant.\nMemory{}Memory",
        "<|fim_pad|>".repeat(16)
    );
    let p = smiley_core::serve::chat_prompt(
        tpl,
        Some(&sys),
        &[],
        "Claim: vLLM uses PagedAttention. Is the claim supported? Answer True or False.",
        false,
        None,
    );
    let mut ids = tk.encode(&p);
    let from = ids.len();
    ids.extend(tk.encode("True"));
    ids.push(tk.id_of("<|im_end|>").unwrap());
    let ph = tk.id_of("<|fim_pad|>").unwrap();
    let at = ids.iter().position(|t| *t == ph).unwrap();
    for (label, init) in [("zeros", 0.0f32), ("small random", 0.02)] {
        let mem = if init == 0.0 {
            Var::zeros((16, d.cfg.hidden_size), DType::F32, &Device::Cpu).unwrap()
        } else {
            Var::randn(0f32, init, (16, d.cfg.hidden_size), &Device::Cpu).unwrap()
        };
        let x = Tensor::cat(
            &[
                &d.embed_ids(&ids[..at]).unwrap(),
                mem.as_tensor(),
                &d.embed_ids(&ids[at + 16..]).unwrap(),
            ],
            0,
        )
        .unwrap()
        .unsqueeze(0)
        .unwrap();
        let h = d.hidden(&x).unwrap().squeeze(0).unwrap();
        let n = ids.len() - from;
        let logits = d.logits(&h.narrow(0, from - 1, n).unwrap()).unwrap();
        let loss = candle_nn::loss::cross_entropy(
            &logits,
            &Tensor::new(&ids[from..], &Device::Cpu).unwrap(),
        )
        .unwrap();
        let gs = loss.backward().unwrap();
        let gm = gs.get(&mem).unwrap();
        let nrm = gm
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let nan_rows: Vec<usize> = (0..16)
            .filter(|r| {
                !gm.get(*r)
                    .unwrap()
                    .sqr()
                    .unwrap()
                    .sum_all()
                    .unwrap()
                    .to_scalar::<f32>()
                    .unwrap()
                    .is_finite()
            })
            .collect();
        println!(
            "{label}: loss {:.3}, grad norm² {nrm}, non-finite rows {nan_rows:?}",
            loss.to_scalar::<f32>().unwrap()
        );
    }
}
