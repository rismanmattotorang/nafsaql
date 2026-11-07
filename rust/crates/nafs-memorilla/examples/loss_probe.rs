//! Probe: the answer-token loss the trainer computes, on a plain chat prompt (no memory), for a small Qwen3.
//! cargo run --release -p nafs-memorilla --example loss_probe -- models/hf/Qwen3-0.6B models/Qwen3-0.6B-Q8_0.gguf
use candle_core::{IndexOp, Tensor, D};
use nafs_memorilla::decoder::Decoder;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let d = Decoder::load(std::path::Path::new(&a[0])).unwrap();
    let g = smiley_core::gguf::Gguf::open(std::path::Path::new(&a[1])).unwrap();
    let tk = smiley_core::tokenizer::Tokenizer::from_gguf(&g).unwrap();
    let tpl = smiley_core::template::Template::detect(&g).unwrap();
    let end = tk.id_of("<|im_end|>").unwrap();
    for (q, ans) in [
        (
            "What is the capital of France? Answer with one word.",
            "Paris",
        ),
        (
            "Is the sky blue on a clear day? Answer True or False.",
            "True",
        ),
    ] {
        let prompt = smiley_core::serve::chat_prompt(
            tpl,
            Some("You are a helpful assistant."),
            &[],
            q,
            false,
            None,
        );
        let mut ids = tk.encode(&prompt);
        let from = ids.len();
        ids.extend(tk.encode(ans));
        ids.push(end);
        let x = d.embed_ids(&ids).unwrap().unsqueeze(0).unwrap();
        let h = d.hidden(&x).unwrap().squeeze(0).unwrap();
        let n = ids.len() - from;
        let logits = d.logits(&h.narrow(0, from - 1, n).unwrap()).unwrap();
        let t = Tensor::new(&ids[from..], d.device()).unwrap();
        let loss = candle_nn::loss::cross_entropy(&logits, &t)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let top: Vec<u32> = (0..n)
            .map(|i| {
                logits
                    .i(i)
                    .unwrap()
                    .argmax(D::Minus1)
                    .unwrap()
                    .to_scalar::<u32>()
                    .unwrap()
            })
            .collect();
        println!(
            "{q:?} -> {ans:?}: loss {loss:.3}; argmax {:?} ({:?}); targets {:?}",
            top,
            tk.decode(&top),
            &ids[from..]
        );
        println!(
            "prompt tail: {:?}",
            &prompt[prompt.len().saturating_sub(60)..]
        );
        // Memorilla's system prompt: placeholders kept as tokens, replaced by zero vectors, or removed.
        let sys = format!(
            "You are a helpful assistant.\nMemory{}Memory",
            "<|fim_pad|>".repeat(16)
        );
        let p2 = smiley_core::serve::chat_prompt(tpl, Some(&sys), &[], q, false, None);
        let mut ids2 = tk.encode(&p2);
        let from2 = ids2.len();
        ids2.extend(tk.encode(ans));
        ids2.push(end);
        let ph = tk.id_of("<|fim_pad|>").unwrap();
        let at: Vec<usize> = ids2
            .iter()
            .enumerate()
            .filter(|(_, t)| **t == ph)
            .map(|(i, _)| i)
            .collect();
        let n2 = ids2.len() - from2;
        let t2 = Tensor::new(&ids2[from2..], d.device()).unwrap();
        let score = |x: Tensor, label: &str| {
            let h = d
                .hidden(&x.unsqueeze(0).unwrap())
                .unwrap()
                .squeeze(0)
                .unwrap();
            let logits = d.logits(&h.narrow(0, from2 - 1, n2).unwrap()).unwrap();
            let loss = candle_nn::loss::cross_entropy(&logits, &t2)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            let top: Vec<u32> = (0..n2)
                .map(|i| {
                    logits
                        .i(i)
                        .unwrap()
                        .argmax(D::Minus1)
                        .unwrap()
                        .to_scalar::<u32>()
                        .unwrap()
                })
                .collect();
            println!("  {label}: loss {loss:.3}, argmax {:?}", tk.decode(&top));
        };
        println!("  {} placeholders at {:?}..", at.len(), at.first());
        score(d.embed_ids(&ids2).unwrap(), "placeholder tokens");
        let before = d.embed_ids(&ids2[..at[0]]).unwrap();
        let after = d.embed_ids(&ids2[at[0] + 16..]).unwrap();
        let zeros =
            Tensor::zeros((16, d.cfg.hidden_size), candle_core::DType::F32, d.device()).unwrap();
        score(
            Tensor::cat(&[&before, &zeros, &after], 0).unwrap(),
            "zero vectors",
        );
        let mut no = ids2[..at[0]].to_vec();
        no.extend(&ids2[at[0] + 16..]);
        let _ = no;
    }
}
