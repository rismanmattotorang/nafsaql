//! Memorilla's memory module in Candle (Phase 4, FR-K40): the maths and the parameter names of
//! `third_party/memorilla/memorilla/memory.py`, so a PyTorch state dict (converted to safetensors) loads as is.
//!
//! Forward, for documents `Z` [B, N, E], their padding mask, and the question `q` [B, E]:
//! 1. slots: learned `memory_queries` [1, K, E], or with `retrieval_init` the K documents most similar to `q`
//!    (cosine; zero rows when N < K);
//! 2. `slots = LN(slots + MHA(slots, q, q))` (question conditioning);
//! 3. optional post-norm Transformer encoder layers over the documents (ReLU, as PyTorch's default);
//! 4. `docs = LN(Z)`;
//! 5. per block: `slots += MHA(LN(slots), docs, docs, mask)`; `slots += FFN(LN(slots))`, FFN = Linear 4E, GELU, Linear;
//! 6. `M = Linear(LN(slots))` -> [B, K, output_dim].
//!
//! Dropout is the identity (inference; training adds it in Phase 6).

use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{layer_norm, linear, LayerNorm, LayerNormConfig, Linear, VarBuilder};
use serde::{Deserialize, Serialize};

/// Constructor arguments, as `config.json` beside a checkpoint stores them.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MemoryConfig {
    pub embedding_dim: usize,
    pub output_dim: usize,
    #[serde(default = "d_k")]
    pub num_memories: usize,
    #[serde(default = "d_heads")]
    pub num_heads: usize,
    #[serde(default)]
    pub num_self_attn_layers: usize,
    #[serde(default = "d_blocks")]
    pub num_cross_attn_layers: usize,
    #[serde(default = "d_dropout")]
    pub dropout: f64,
    #[serde(default)]
    pub retrieval_init: bool,
}
fn d_k() -> usize {
    16
}
fn d_heads() -> usize {
    8
}
fn d_blocks() -> usize {
    2
}
fn d_dropout() -> f64 {
    0.1
}

impl MemoryConfig {
    /// The paper's module for a given encoder and decoder width (K = 16, 8 heads, 2 blocks, no self-attention).
    pub fn paper(embedding_dim: usize, output_dim: usize) -> MemoryConfig {
        MemoryConfig {
            embedding_dim,
            output_dim,
            num_memories: 16,
            num_heads: 8,
            num_self_attn_layers: 0,
            num_cross_attn_layers: 2,
            dropout: 0.1,
            retrieval_init: false,
        }
    }

    /// Trainable parameters (the paper's formula; checked against PyTorch's count by the parity fixtures).
    pub fn parameters(&self) -> usize {
        let (e, h) = (self.embedding_dim, self.output_dim);
        let mha = 4 * e * e + 4 * e;
        let ln = 2 * e;
        let ffn = 8 * e * e + 5 * e;
        let enc = mha + 8 * e * e + 5 * e + 2 * ln;
        (if self.retrieval_init {
            0
        } else {
            self.num_memories * e
        }) + mha
            + ln
            + self.num_self_attn_layers * enc
            + ln
            + self.num_cross_attn_layers * (ln + mha + ln + ffn)
            + ln
            + e * h
            + h
    }
}

/// `nn.MultiheadAttention(batch_first=True)`: packed `in_proj_weight` [3E, E] and `in_proj_bias`, then `out_proj`.
struct Mha {
    wq: Tensor,
    wk: Tensor,
    wv: Tensor,
    bq: Tensor,
    bk: Tensor,
    bv: Tensor,
    out: Linear,
    heads: usize,
}

impl Mha {
    fn new(e: usize, heads: usize, vb: VarBuilder) -> Result<Mha> {
        let w = vb.get((3 * e, e), "in_proj_weight")?;
        let b = vb.get(3 * e, "in_proj_bias")?;
        Ok(Mha {
            wq: w.narrow(0, 0, e)?,
            wk: w.narrow(0, e, e)?,
            wv: w.narrow(0, 2 * e, e)?,
            bq: b.narrow(0, 0, e)?,
            bk: b.narrow(0, e, e)?,
            bv: b.narrow(0, 2 * e, e)?,
            out: linear(e, e, vb.pp("out_proj"))?,
            heads,
        })
    }

    /// Project `x` [B, L, E] with one third of the packed weights into heads [B, H, L, E/H].
    fn heads_of(&self, x: &Tensor, w: &Tensor, bias: &Tensor) -> Result<Tensor> {
        let (b, l, e) = x.dims3()?;
        x.broadcast_matmul(&w.t()?)?
            .broadcast_add(bias)?
            .reshape((b, l, self.heads, e / self.heads))?
            .transpose(1, 2)?
            .contiguous()
    }

    /// Keys and values of `kv` [B, Lk, E] in head layout: what does not depend on the queries.
    fn keys_values(&self, kv: &Tensor) -> Result<(Tensor, Tensor)> {
        Ok((
            self.heads_of(kv, &self.wk, &self.bk)?,
            self.heads_of(kv, &self.wv, &self.bv)?,
        ))
    }

    /// `q` [B, Lq, E], `kv` [B, Lk, E]; `mask` [B, Lk] with 1 at padding (keys to ignore).
    fn forward(&self, q: &Tensor, kv: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (kh, vh) = self.keys_values(kv)?;
        self.attend(q, &kh, &vh, mask)
    }

    /// Attention of `q` [B, Lq, E] over precomputed keys and values [B, H, Lk, E/H].
    fn attend(
        &self,
        q: &Tensor,
        kh: &Tensor,
        vh: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, lq, e) = q.dims3()?;
        let lk = kh.dim(2)?;
        let hd = e / self.heads;
        let qh = self.heads_of(q, &self.wq, &self.bq)?;
        let mut scores = (qh.matmul(&kh.t()?.contiguous()?)? / (hd as f64).sqrt())?; // [B, H, Lq, Lk]
        if let Some(m) = mask {
            // A large negative score where the key is padding, broadcast over heads and queries. Finite, not -inf, so a
            // row whose keys are all padding gives a uniform average instead of PyTorch's NaN.
            let neg = m.to_dtype(DType::F32)?.affine(-1e30, 0.0)?;
            let neg = neg.reshape((b, 1, 1, lk))?;
            scores = scores.broadcast_add(&neg)?;
        }
        let attn = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = attn
            .matmul(vh)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, lq, e))?;
        self.out.forward(&ctx)
    }
}

/// `nn.TransformerEncoderLayer(norm_first=False, activation=relu)`.
struct EncoderLayer {
    attn: Mha,
    l1: Linear,
    l2: Linear,
    n1: LayerNorm,
    n2: LayerNorm,
}

impl EncoderLayer {
    fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let x = self.n1.forward(&(x + self.attn.forward(x, x, mask)?)?)?;
        let ff = self.l2.forward(&self.l1.forward(&x)?.relu()?)?;
        self.n2.forward(&(x + ff)?)
    }
}

struct Block {
    attn_norm: LayerNorm,
    attn: Mha,
    ffn_norm: LayerNorm,
    ffn_in: Linear,
    ffn_out: Linear,
}

/// Documents prepared for many questions (see `MemoryModule::prepare`): the raw embeddings (for `retrieval_init`),
/// the padding mask, and per refinement block the keys and values in head layout.
pub struct PreparedDocs {
    docs: Tensor,
    padding: Option<Tensor>,
    kv: Vec<(Tensor, Tensor)>,
}

impl PreparedDocs {
    /// Floats held (keys and values of every block, plus the embeddings).
    pub fn elem_count(&self) -> usize {
        self.docs.elem_count()
            + self
                .kv
                .iter()
                .map(|(k, v)| k.elem_count() + v.elem_count())
                .sum::<usize>()
    }
}

pub struct MemoryModule {
    pub cfg: MemoryConfig,
    memory_queries: Option<Tensor>,
    query_cond_attn: Mha,
    query_cond_norm: LayerNorm,
    encoder: Vec<EncoderLayer>,
    doc_norm: LayerNorm,
    blocks: Vec<Block>,
    layer_norm: LayerNorm,
    projection: Linear,
}

fn ln(e: usize, vb: VarBuilder) -> Result<LayerNorm> {
    layer_norm(
        e,
        LayerNormConfig {
            eps: 1e-5,
            ..Default::default()
        },
        vb,
    )
}

impl MemoryModule {
    /// Build from a `VarBuilder`: safetensors for inference, a `VarMap` for training (fresh variables get Candle's
    /// default initialisation; Phase 6 sets the paper's Xavier initialisation).
    pub fn new(cfg: MemoryConfig, vb: VarBuilder) -> Result<MemoryModule> {
        let e = cfg.embedding_dim;
        if cfg.num_heads == 0 || !e.is_multiple_of(cfg.num_heads) {
            candle_core::bail!(
                "embedding_dim {e} is not divisible by num_heads {}",
                cfg.num_heads
            );
        }
        let memory_queries = if cfg.retrieval_init {
            None
        } else {
            Some(vb.get((1, cfg.num_memories, e), "memory_queries")?)
        };
        let encoder = (0..cfg.num_self_attn_layers)
            .map(|i| {
                let v = vb.pp("encoder.layers").pp(i);
                Ok(EncoderLayer {
                    attn: Mha::new(e, cfg.num_heads, v.pp("self_attn"))?,
                    l1: linear(e, 4 * e, v.pp("linear1"))?,
                    l2: linear(4 * e, e, v.pp("linear2"))?,
                    n1: ln(e, v.pp("norm1"))?,
                    n2: ln(e, v.pp("norm2"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let blocks = (0..cfg.num_cross_attn_layers)
            .map(|i| {
                Ok(Block {
                    attn_norm: ln(e, vb.pp("cross_attn_norms").pp(i))?,
                    attn: Mha::new(e, cfg.num_heads, vb.pp("cross_attn_layers").pp(i))?,
                    ffn_norm: ln(e, vb.pp("ffn_norms").pp(i))?,
                    ffn_in: linear(e, 4 * e, vb.pp("ffn_layers").pp(i).pp("0"))?,
                    ffn_out: linear(4 * e, e, vb.pp("ffn_layers").pp(i).pp("3"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(MemoryModule {
            memory_queries,
            query_cond_attn: Mha::new(e, cfg.num_heads, vb.pp("query_cond_attn"))?,
            query_cond_norm: ln(e, vb.pp("query_cond_norm"))?,
            encoder,
            doc_norm: ln(e, vb.pp("doc_norm"))?,
            blocks,
            layer_norm: ln(e, vb.pp("layer_norm"))?,
            projection: linear(e, cfg.output_dim, vb.pp("memory_projection"))?,
            cfg,
        })
    }

    /// Load `config.json` + `weights.safetensors` (or `memory.safetensors`) from a checkpoint directory, in f32.
    pub fn load(
        dir: &std::path::Path,
        device: &Device,
    ) -> std::result::Result<MemoryModule, String> {
        let cfg: MemoryConfig = serde_json::from_slice(
            &std::fs::read(dir.join("config.json")).map_err(|e| format!("config.json: {e}"))?,
        )
        .map_err(|e| format!("config.json: {e}"))?;
        let weights = ["weights.safetensors", "memory.safetensors"]
            .iter()
            .map(|f| dir.join(f))
            .find(|p| p.exists())
            .ok_or("no weights.safetensors or memory.safetensors in the checkpoint directory")?;
        // SAFETY: the file is memory-mapped read-only and not modified while the module lives.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, device) }
            .map_err(|e| e.to_string())?;
        MemoryModule::new(cfg, vb).map_err(|e| e.to_string())
    }

    /// Documents [B, N, E] (padding rows anywhere), `padding` [B, N] as u8/f32 with 1 at padding (or `None`),
    /// question [B, E] -> memory tokens [B, K, output_dim].
    pub fn forward(
        &self,
        docs: &Tensor,
        padding: Option<&Tensor>,
        question: &Tensor,
    ) -> Result<Tensor> {
        self.forward_prepared(&self.prepare(docs, padding)?, question)
    }

    /// The question-independent half of the forward pass: the document encoder, the document norm, and every
    /// block's keys and values. A store computes this once and reuses it for every question (the per-question cost
    /// then grows with K x N, not with N x E^2).
    pub fn prepare(&self, docs: &Tensor, padding: Option<&Tensor>) -> Result<PreparedDocs> {
        let mut d = docs.clone();
        for l in &self.encoder {
            d = l.forward(&d, padding)?;
        }
        let normed = self.doc_norm.forward(&d)?;
        let kv = self
            .blocks
            .iter()
            .map(|blk| blk.attn.keys_values(&normed))
            .collect::<Result<Vec<_>>>()?;
        Ok(PreparedDocs {
            docs: docs.clone(),
            padding: padding.cloned(),
            kv,
        })
    }

    /// The question-dependent half: memory tokens [B, K, output_dim] for `question` [B, E] over prepared documents.
    pub fn forward_prepared(&self, p: &PreparedDocs, question: &Tensor) -> Result<Tensor> {
        let (b, _, e) = p.docs.dims3()?;
        let q = question.reshape((b, 1, e))?;
        let slots = match &self.memory_queries {
            Some(mq) => mq
                .broadcast_as((b, self.cfg.num_memories, e))?
                .contiguous()?,
            None => self.retrieval_slots(&p.docs, p.padding.as_ref(), question)?,
        };
        let slots = self
            .query_cond_norm
            .forward(&(&slots + self.query_cond_attn.forward(&slots, &q, None)?)?)?;
        let mut s = slots;
        for (blk, (kh, vh)) in self.blocks.iter().zip(&p.kv) {
            s = (&s
                + blk
                    .attn
                    .attend(&blk.attn_norm.forward(&s)?, kh, vh, p.padding.as_ref())?)?;
            let f = blk
                .ffn_out
                .forward(&blk.ffn_in.forward(&blk.ffn_norm.forward(&s)?)?.gelu_erf()?)?;
            s = (s + f)?;
        }
        self.projection.forward(&self.layer_norm.forward(&s)?)
    }

    /// `retrieval_init`: per row, the K documents with the highest cosine to the question, in descending order
    /// (padding excluded), zero rows when there are fewer than K.
    fn retrieval_slots(
        &self,
        docs: &Tensor,
        padding: Option<&Tensor>,
        question: &Tensor,
    ) -> Result<Tensor> {
        let (b, n, e) = docs.dims3()?;
        let k = self.cfg.num_memories;
        let norm = |x: &Tensor| -> Result<Tensor> {
            x.broadcast_div(
                &x.sqr()?
                    .sum_keepdim(D::Minus1)?
                    .sqrt()?
                    .clamp(1e-12, f64::MAX)?,
            )
        };
        let scores = norm(docs)?
            .matmul(&norm(question)?.unsqueeze(2)?)?
            .squeeze(2)?; // [B, N]
        let scores: Vec<Vec<f32>> = scores.to_vec2()?;
        let pad: Option<Vec<Vec<f32>>> = match padding {
            Some(p) => Some(p.to_dtype(DType::F32)?.to_vec2()?),
            None => None,
        };
        let mut rows = Vec::with_capacity(b);
        for (i, row) in scores.iter().enumerate() {
            let mut idx: Vec<usize> = (0..n)
                .filter(|&j| pad.as_ref().is_none_or(|p| p[i][j] == 0.0))
                .collect();
            idx.sort_by(|&x, &y| row[y].total_cmp(&row[x]).then(x.cmp(&y)));
            idx.truncate(k);
            let mut parts = Vec::with_capacity(k);
            for &j in &idx {
                parts.push(docs.i((i, j..j + 1))?);
            }
            if idx.len() < k {
                parts.push(Tensor::zeros(
                    (k - idx.len(), e),
                    docs.dtype(),
                    docs.device(),
                )?);
            }
            rows.push(Tensor::cat(&parts, 0)?);
        }
        Tensor::stack(&rows, 0)
    }
}

/// Plain-slice entry points for callers without Candle (the gateway): documents as one row-major `n x E` buffer, the
/// question as `E` floats, memory tokens back as `K` vectors of `output_dim` floats.
impl MemoryModule {
    /// `load` on the CPU.
    pub fn load_cpu(dir: &std::path::Path) -> std::result::Result<MemoryModule, String> {
        MemoryModule::load(dir, &Device::Cpu)
    }

    pub fn prepare_rows(
        &self,
        docs: &[f32],
        n: usize,
    ) -> std::result::Result<PreparedDocs, String> {
        let e = self.cfg.embedding_dim;
        if n == 0 || docs.len() != n * e {
            return Err(format!(
                "{} document floats for {n} documents of width {e}",
                docs.len()
            ));
        }
        let t = Tensor::from_slice(docs, (1, n, e), &Device::Cpu).map_err(|x| x.to_string())?;
        self.prepare(&t, None).map_err(|x| x.to_string())
    }

    pub fn read_rows(
        &self,
        prepared: &PreparedDocs,
        question: &[f32],
    ) -> std::result::Result<Vec<Vec<f32>>, String> {
        let e = self.cfg.embedding_dim;
        if question.len() != e {
            return Err(format!(
                "question vector has {} values, the module needs {e}",
                question.len()
            ));
        }
        let q = Tensor::from_slice(question, (1, e), &Device::Cpu).map_err(|x| x.to_string())?;
        let m = self
            .forward_prepared(prepared, &q)
            .map_err(|x| x.to_string())?;
        m.squeeze(0)
            .and_then(|t| t.to_vec2::<f32>())
            .map_err(|x| x.to_string())
    }

    /// Write `config.json` and `weights.safetensors` (the format `load` reads).
    pub fn save(
        vm: &candle_nn::VarMap,
        cfg: &MemoryConfig,
        dir: &std::path::Path,
    ) -> std::result::Result<(), String> {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        std::fs::write(
            dir.join("config.json"),
            serde_json::to_vec_pretty(cfg).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        vm.save(dir.join("weights.safetensors"))
            .map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_paper_parameter_counts() {
        assert_eq!(MemoryConfig::paper(2560, 4096).parameters(), 194_123_776);
        assert_eq!(MemoryConfig::paper(2560, 2560).parameters(), 190_190_080);
        assert_eq!(MemoryConfig::paper(1024, 2048).parameters(), 31_512_576);
    }
}
