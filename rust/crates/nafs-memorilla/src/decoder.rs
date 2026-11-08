//! A frozen Qwen3 decoder that takes **input embeddings** and is built only from operations Candle can differentiate
//! (Phase 6). The memory module is trained through it: its weights are plain tensors (no gradients), the memory
//! tokens injected into the input carry the gradient back to the module.
//!
//! Candle's own Qwen3 (`candle-transformers`) takes token ids and uses fused kernels (rotary embedding, RMS norm, fast
//! softmax) that have no backward pass; this one spells them out. `tests` checks it against that model's logits.

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::VarBuilder;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct DecoderConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

struct Layer {
    input_norm: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    o: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    post_norm: Tensor,
    gate: Tensor,
    up: Tensor,
    down: Tensor,
}

pub struct Decoder {
    pub cfg: DecoderConfig,
    embed: Tensor,
    head: Tensor,
    layers: Vec<Layer>,
    norm: Tensor,
    device: Device,
}

/// `x @ w^T` for `x` [.., in] and `w` [out, in].
fn lin(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    x.broadcast_matmul(&w.t()?)
}

/// RMS norm over the last dimension, from differentiable parts.
fn rms(x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor> {
    let inv = (x.sqr()?.mean_keepdim(D::Minus1)? + eps)?.sqrt()?.recip()?;
    x.broadcast_mul(&inv)?.broadcast_mul(w)
}

/// SiLU as x * sigmoid(x), with sigmoid(x) = (1 + tanh(x / 2)) / 2. Not x / (1 + exp(-x)): for a large negative
/// activation exp(-x) overflows to infinity, and although the forward value is still 0 the backward pass computes
/// 0 * inf = NaN (found in Phase 6: every memory gradient through Qwen3's MLP was NaN).
fn silu(x: &Tensor) -> Result<Tensor> {
    x * ((x * 0.5)?.tanh()? + 1.0)?.affine(0.5, 0.0)?
}

/// NeoX-style rotation (Qwen): `x * cos + rotate_half(x) * sin`, with cos/sin [T, head_dim].
fn rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let d = x.dim(D::Minus1)?;
    let x1 = x.narrow(D::Minus1, 0, d / 2)?;
    let x2 = x.narrow(D::Minus1, d / 2, d / 2)?;
    let rot = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;
    x.broadcast_mul(cos)? + rot.broadcast_mul(sin)?
}

impl Decoder {
    /// Load `config.json` and the safetensors shards of a Hugging Face Qwen3 directory, in f32.
    pub fn load(dir: &std::path::Path) -> std::result::Result<Decoder, String> {
        let cfg: DecoderConfig = serde_json::from_slice(
            &std::fs::read(dir.join("config.json")).map_err(|e| format!("config.json: {e}"))?,
        )
        .map_err(|e| format!("config.json: {e}"))?;
        let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
            .map_err(|e| e.to_string())?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        files.sort();
        let device = Device::Cpu;
        // SAFETY: the shards are memory-mapped read-only and not modified while the decoder lives.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, DType::F32, &device) }
            .map_err(|e| e.to_string())?;
        Decoder::new(cfg, vb, device).map_err(|e| e.to_string())
    }

    fn new(cfg: DecoderConfig, vb: VarBuilder, device: Device) -> Result<Decoder> {
        let (h, hd) = (cfg.hidden_size, cfg.head_dim);
        let (nq, nk, ff) = (
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.intermediate_size,
        );
        let m = vb.pp("model");
        let embed = m.get((cfg.vocab_size, h), "embed_tokens.weight")?;
        let head = if cfg.tie_word_embeddings {
            embed.clone()
        } else {
            vb.get((cfg.vocab_size, h), "lm_head.weight")?
        };
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| {
                let l = m.pp("layers").pp(i);
                Ok(Layer {
                    input_norm: l.get(h, "input_layernorm.weight")?,
                    q: l.get((nq * hd, h), "self_attn.q_proj.weight")?,
                    k: l.get((nk * hd, h), "self_attn.k_proj.weight")?,
                    v: l.get((nk * hd, h), "self_attn.v_proj.weight")?,
                    o: l.get((h, nq * hd), "self_attn.o_proj.weight")?,
                    q_norm: l.get(hd, "self_attn.q_norm.weight")?,
                    k_norm: l.get(hd, "self_attn.k_norm.weight")?,
                    post_norm: l.get(h, "post_attention_layernorm.weight")?,
                    gate: l.get((ff, h), "mlp.gate_proj.weight")?,
                    up: l.get((ff, h), "mlp.up_proj.weight")?,
                    down: l.get((h, ff), "mlp.down_proj.weight")?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let norm = m.get(h, "norm.weight")?;
        Ok(Decoder {
            cfg,
            embed,
            head,
            layers,
            norm,
            device,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Token embeddings [T, H] of `ids`.
    pub fn embed_ids(&self, ids: &[u32]) -> Result<Tensor> {
        let idx = Tensor::new(ids, &self.device)?;
        self.embed.index_select(&idx, 0)
    }

    /// Final hidden states [1, T, H] (after the output norm) for input embeddings [1, T, H], causal attention.
    pub fn hidden(&self, x: &Tensor) -> Result<Tensor> {
        let c = &self.cfg;
        let (b, t, _) = x.dims3()?;
        let (nq, nk, hd) = (c.num_attention_heads, c.num_key_value_heads, c.head_dim);
        // Rotary tables for positions 0..t.
        let inv: Vec<f32> = (0..hd / 2)
            .map(|i| 1.0 / (c.rope_theta as f32).powf(2.0 * i as f32 / hd as f32))
            .collect();
        let pos = Tensor::arange(0u32, t as u32, &self.device)?
            .to_dtype(DType::F32)?
            .reshape((t, 1))?;
        let freqs =
            pos.broadcast_mul(&Tensor::new(inv.as_slice(), &self.device)?.reshape((1, hd / 2))?)?;
        let freqs = Tensor::cat(&[&freqs, &freqs], 1)?;
        let (cos, sin) = (freqs.cos()?, freqs.sin()?);
        let mask: Vec<f32> = (0..t * t)
            .map(|i| if i % t > i / t { -1e9 } else { 0.0 })
            .collect();
        let mask = Tensor::from_vec(mask, (t, t), &self.device)?;
        let mut h = x.clone();
        for l in &self.layers {
            let a = rms(&h, &l.input_norm, c.rms_norm_eps)?;
            let q = lin(&a, &l.q)?.reshape((b, t, nq, hd))?;
            let k = lin(&a, &l.k)?.reshape((b, t, nk, hd))?;
            let v = lin(&a, &l.v)?.reshape((b, t, nk, hd))?;
            let q = rope(
                &rms(&q, &l.q_norm, c.rms_norm_eps)?.transpose(1, 2)?,
                &cos,
                &sin,
            )?;
            let k = rope(
                &rms(&k, &l.k_norm, c.rms_norm_eps)?.transpose(1, 2)?,
                &cos,
                &sin,
            )?;
            let v = v.transpose(1, 2)?;
            // Grouped-query attention: each key/value head serves nq/nk query heads.
            let rep = nq / nk;
            let k = k
                .unsqueeze(2)?
                .broadcast_as((b, nk, rep, t, hd))?
                .reshape((b, nq, t, hd))?;
            let v = v
                .unsqueeze(2)?
                .broadcast_as((b, nk, rep, t, hd))?
                .reshape((b, nq, t, hd))?;
            let scores = (q.contiguous()?.matmul(&k.t()?.contiguous()?)? / (hd as f64).sqrt())?
                .broadcast_add(&mask)?;
            let attn = candle_nn::ops::softmax(&scores, D::Minus1)?;
            let o = attn
                .matmul(&v.contiguous()?)?
                .transpose(1, 2)?
                .reshape((b, t, nq * hd))?;
            h = (h + lin(&o, &l.o)?)?;
            let m = rms(&h, &l.post_norm, c.rms_norm_eps)?;
            let f = lin(&(silu(&lin(&m, &l.gate)?)? * lin(&m, &l.up)?)?, &l.down)?;
            h = (h + f)?;
        }
        rms(&h, &self.norm, c.rms_norm_eps)
    }

    /// Logits [n, V] for hidden states [n, H].
    pub fn logits(&self, h: &Tensor) -> Result<Tensor> {
        lin(h, &self.head)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::IndexOp;

    /// The hand-written forward agrees with candle-transformers' Qwen3 on the real Qwen3-1.7B (skips without it).
    #[test]
    fn matches_candle_transformers_qwen3() {
        let dir = std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../models/hf/Qwen3-1.7B"
        ));
        if !dir.join("config.json").exists() {
            eprintln!("skipped: models/hf/Qwen3-1.7B absent");
            return;
        }
        let d = Decoder::load(dir).unwrap();
        let ids: Vec<u32> = vec![
            151644, 872, 198, 785, 6722, 315, 9625, 374, 12095, 13, 576, 6722, 315, 15344, 374,
        ];
        let ours = d
            .hidden(&d.embed_ids(&ids).unwrap().unsqueeze(0).unwrap())
            .unwrap();
        let ours = d
            .logits(&ours.i((0, ids.len() - 1)).unwrap().unsqueeze(0).unwrap())
            .unwrap()
            .squeeze(0)
            .unwrap();

        let cfg: candle_transformers::models::qwen3::Config =
            serde_json::from_slice(&std::fs::read(dir.join("config.json")).unwrap()).unwrap();
        let mut files: Vec<_> = std::fs::read_dir(dir)
            .unwrap()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        files.sort();
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, DType::F32, &Device::Cpu) }
            .unwrap();
        let mut reference =
            candle_transformers::models::qwen3::ModelForCausalLM::new(&cfg, vb).unwrap();
        let theirs = reference
            .forward(
                &Tensor::new(ids.as_slice(), &Device::Cpu)
                    .unwrap()
                    .unsqueeze(0)
                    .unwrap(),
                0,
            )
            .unwrap();
        let theirs = theirs.flatten_all().unwrap();
        let diff = (&ours - &theirs)
            .unwrap()
            .abs()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let scale = theirs
            .abs()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let (a, b) = (
            ours.argmax(0).unwrap().to_scalar::<u32>().unwrap(),
            theirs.argmax(0).unwrap().to_scalar::<u32>().unwrap(),
        );
        eprintln!("max |ours - candle| = {diff} (logit scale {scale}); argmax {a} vs {b}");
        assert_eq!(a, b);
        assert!(diff < 1e-2 * scale.max(1.0), "{diff}");
    }
}
