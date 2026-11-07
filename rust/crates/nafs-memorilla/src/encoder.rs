//! Qwen3-Embedding on Candle (f32, CPU): token ids in, a unit vector out.
//!
//! Tokenisation is not done here. The caller supplies the ids (Smiley Core's GGUF tokenizer in Nafs-Aql), followed by
//! `<|endoftext|>` (`END_OF_TEXT`), which is the pooled position; this matches the model's own tokenizer, whose
//! post-processor appends that token.

use std::path::Path;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3::{Config, Model};

/// `<|endoftext|>` in the Qwen3 vocabulary.
pub const END_OF_TEXT: u32 = 151_643;

pub struct Encoder {
    /// A model with an empty cache; each text runs on a clone (the weights are shared, the cache is not).
    model: Model,
    device: Device,
    pub dims: usize,
}

impl Encoder {
    /// Load `config.json` and `model.safetensors` from a Hugging Face model directory.
    pub fn load(dir: &Path) -> Result<Encoder, String> {
        let cfg: Config = serde_json::from_slice(
            &std::fs::read(dir.join("config.json")).map_err(|e| format!("config.json: {e}"))?,
        )
        .map_err(|e| format!("config.json: {e}"))?;
        let device = Device::Cpu;
        let weights = dir.join("model.safetensors");
        // SAFETY: the file is memory-mapped read-only and not modified while the encoder lives.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, &device) }
            .map_err(|e| e.to_string())?;
        // The embedding checkpoint stores the bare decoder (`embed_tokens.weight`, `layers.0...`), without the
        // `model.` prefix of a causal-LM checkpoint; Candle's Qwen3 asks for the prefixed names.
        let vb = vb.rename_f(|name: &str| name.strip_prefix("model.").unwrap_or(name).to_string());
        let model = Model::new(&cfg, vb).map_err(|e| e.to_string())?;
        Ok(Encoder {
            model,
            device,
            dims: cfg.hidden_size,
        })
    }

    /// The normalised last-token embedding of `ids` (which must end with `END_OF_TEXT`).
    pub fn embed_ids(&self, ids: &[u32]) -> Result<Vec<f32>, String> {
        if ids.is_empty() {
            return Err("no tokens".into());
        }
        let mut model = self.model.clone();
        let input = Tensor::new(ids, &self.device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(|e| e.to_string())?;
        let hidden = model.forward(&input, 0).map_err(|e| e.to_string())?; // [1, len, dims], output-normed
        let last = hidden.i((0, ids.len() - 1)).map_err(|e| e.to_string())?;
        let mut v: Vec<f32> = last.to_vec1().map_err(|e| e.to_string())?;
        let norm = v
            .iter()
            .map(|x| (*x as f64) * (*x as f64))
            .sum::<f64>()
            .sqrt() as f32;
        if norm > 0.0 {
            v.iter_mut().for_each(|x| *x /= norm);
        }
        Ok(v)
    }
}

/// Cosine similarity of two vectors.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (*x as f64) * (*y as f64))
        .sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    (dot / (na * nb).max(1e-12)) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");

    /// Parity (spec §8): Smiley Core's Q8_0 embeddings against this f32 reference on fixed texts, and the model card's
    /// retrieval example ranked the same way. Skips when the weights are absent (`scripts/fetch_models.sh embed`).
    #[test]
    fn smiley_core_matches_the_candle_reference() {
        let hf = Path::new(ROOT).join("models/hf/Qwen3-Embedding-0.6B");
        let gguf = Path::new(ROOT).join("models/Qwen3-Embedding-0.6B-Q8_0.gguf");
        if !hf.join("model.safetensors").exists() || !gguf.exists() {
            eprintln!("skipped: embedding weights not downloaded");
            return;
        }
        let reference = Encoder::load(&hf).unwrap();
        let model = smiley_core::model::keep::open_with(&gguf, false).unwrap();
        let mut smiley = smiley_core::embed::Embedder::new(model.clone(), 3, 512);
        let instr = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:";
        let texts = [
            format!("{instr}What is the capital of China?"),
            format!("{instr}Explain gravity"),
            "The capital of China is Beijing.".to_string(),
            "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is \
             responsible for the movement of planets around the sun."
                .to_string(),
            "Caption: Model of the gate. Sixteen memory tokens replace the documents in the prompt.".to_string(),
            "SurrealDB stores passages with a BM25 index and an HNSW vector index; RELATE edges link entities.".to_string(),
        ];
        let mut refs = Vec::new();
        for t in &texts {
            let mut ids = model.tk.encode(t);
            ids.push(END_OF_TEXT);
            let r = reference.embed_ids(&ids).unwrap();
            let s = smiley.embed(t).unwrap();
            assert_eq!(s.tokens, ids.len());
            let c = cosine(&r, &s.vector);
            eprintln!("cosine {c:.5}  {}", &t[t.len().saturating_sub(40)..]);
            assert!(
                c >= 0.999,
                "Smiley Core drifts from the reference: cosine {c} for {t:?}"
            );
            refs.push(r);
        }
        // The model card's example: each query is closest to its own passage (reported 0.7646 / 0.6000 on the diagonal).
        let sim = |a: usize, b: usize| cosine(&refs[a], &refs[b]);
        eprintln!(
            "reference scores {:.4} {:.4} / {:.4} {:.4}",
            sim(0, 2),
            sim(0, 3),
            sim(1, 2),
            sim(1, 3)
        );
        assert!((sim(0, 2) - 0.7646).abs() < 0.01 && (sim(1, 3) - 0.6000).abs() < 0.01);
        assert!(sim(0, 2) > sim(0, 3) && sim(1, 3) > sim(1, 2));
    }
}
