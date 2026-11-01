//! DSpark — DeepSeek's speculative drafter, for a Qwen3 target.
//!
//! DeepInfer (`vendors/DeepInfer`, `model_executor/models/mtp/dflash.py`) serves
//! DeepSeek-V4.1 with this method; DeepSeek also published stand-alone DSpark
//! drafters for small dense targets (`deepseek-ai/dspark_qwen3_4b_block7`, …), which
//! is what runs here. See `docs/DEEPINFER-PLAN.md`.
//!
//! # The drafter
//!
//! * **Context.** For every committed position the target's residual stream after a
//!   few chosen layers (`target_layer_ids`, five for Qwen3-4B) is concatenated,
//!   projected by `fc` and RMS-normed (`hidden_norm`). Each drafter layer turns that
//!   *shared* context row into a key (k_proj → per-head k_norm → RoPE at the row's
//!   absolute position) and a value (v_proj, not normed), cached per layer. Only the
//!   committed positions enter the context, so a rejected draft never needs undoing
//!   here — the cache is truncated by length like the target's.
//! * **Block.** A draft is one forward of `block_size` positions: the anchor (the
//!   token the target just produced, not yet fed) followed by `mask` tokens, at the
//!   positions right after the context. Queries come from the block; keys and values
//!   from `[context, block]`; attention is bidirectional within the block. Five Qwen3
//!   layers, a final norm, `lm_head`.
//! * **Markov head.** Each position's logits get a rank-256 bias from the token
//!   chosen at the previous position, `w2 · w1[prev]` — the only sequential step, and
//!   what makes a parallel block read like an autoregressive draft.
//! * **Confidence head.** `sigmoid(w · [hidden ; w1[prev]] + b)` estimates the chance
//!   each draft token survives verification *given its prefix did*; the product over
//!   a prefix is the chance the whole prefix survives. DeepInfer uses it to choose how
//!   much of a draft to verify (`csrc/engine/mtp_varlen.hpp`); so does `Session`.
//!
//! Predictions start at the anchor slot (DeepSpec's convention, `logits_start = 0`):
//! slot 0 embeds the known token and predicts the first draft token.
//!
//! Weights arrive as bf16 safetensors and are quantised at load — to Q4_K with a
//! port of llama.cpp's own quantiser (`quant::quantize_row_q4_k`), or Q8_0 — so every
//! matrix runs through the same AVX2 kernels as the target.

use crate::gguf::Format;
use crate::pool::{share, Pool};
use crate::quant::{
    dequant_row, dot_batch, quantize_row_q4_k, quantize_row_q8_0, row_bytes, QuantAct,
};
use crate::safetensors::{Json, SafeTensors};
use std::path::Path;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct DSparkConfig {
    pub hidden: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    pub n_ff: usize,
    pub n_vocab: usize,
    pub block_size: usize,
    pub mask_id: u32,
    pub markov_rank: usize,
    pub eps: f32,
    pub rope_theta: f64,
    pub target_layer_ids: Vec<usize>,
    pub confidence: bool,
    pub confidence_with_markov: bool,
    /// First block slot whose logits are a prediction (0 for DeepSpec heads).
    pub logits_start: usize,
}

impl DSparkConfig {
    pub fn from_json(text: &str) -> Result<DSparkConfig, String> {
        let j = Json::parse(text)?;
        let num = |k: &str| {
            j.get(k)
                .and_then(Json::as_num)
                .ok_or_else(|| format!("config.json: no {k}"))
        };
        let arch = j
            .get("architectures")
            .and_then(Json::as_arr)
            .and_then(|a| a.first())
            .and_then(Json::as_str)
            .unwrap_or("");
        if arch != "Qwen3DSparkModel" {
            return Err(format!(
                "not a Qwen3 DSpark drafter (architectures = {arch:?})"
            ));
        }
        let rope_theta = j
            .get("rope_parameters")
            .and_then(|r| r.get("rope_theta"))
            .and_then(Json::as_num)
            .or_else(|| j.get("rope_theta").and_then(Json::as_num))
            .unwrap_or(1e6);
        let taps = j
            .get("target_layer_ids")
            .and_then(Json::as_arr)
            .ok_or("config.json: no target_layer_ids")?
            .iter()
            .map(|v| v.as_num().map(|x| x as usize))
            .collect::<Option<Vec<_>>>()
            .ok_or("bad target_layer_ids")?;
        if let Some(lt) = j.get("layer_types").and_then(Json::as_arr) {
            if lt.iter().any(|t| t.as_str() != Some("full_attention")) {
                return Err("sliding-window DSpark heads are not implemented (all layers must be full_attention)".into());
            }
        }
        let c = DSparkConfig {
            hidden: num("hidden_size")? as usize,
            n_layer: num("num_hidden_layers")? as usize,
            n_head: num("num_attention_heads")? as usize,
            n_kv_head: num("num_key_value_heads")? as usize,
            head_dim: num("head_dim")? as usize,
            n_ff: num("intermediate_size")? as usize,
            n_vocab: num("vocab_size")? as usize,
            block_size: num("block_size")? as usize,
            mask_id: num("mask_token_id")? as u32,
            markov_rank: j.get("markov_rank").and_then(Json::as_num).unwrap_or(0.0) as usize,
            eps: j.get("rms_norm_eps").and_then(Json::as_num).unwrap_or(1e-6) as f32,
            rope_theta,
            target_layer_ids: taps,
            confidence: j
                .get("enable_confidence_head")
                .and_then(Json::as_bool)
                .unwrap_or(false),
            confidence_with_markov: j
                .get("confidence_head_with_markov")
                .and_then(Json::as_bool)
                .unwrap_or(false),
            logits_start: std::env::var("SMILEY_DSPARK_LOGITS_START")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
        };
        if !c.n_head.is_multiple_of(c.n_kv_head) || c.block_size == 0 || c.block_size > 32 {
            return Err(format!("implausible drafter shape: {c:?}"));
        }
        Ok(c)
    }
}

/// A quantised matrix, row-major: `rows` rows of `cols` weights, in Q4_K (the
/// default, 4.5 bits per weight: the drafter's bytes are half its cost) or Q8_0.
struct QW {
    fmt: Format,
    rows: usize,
    cols: usize,
    data: Vec<u8>,
}

/// `SMILEY_DSPARK_QUANT=q8` keeps the drafter at Q8_0 (the A/B switch Q4_K was
/// measured against).
fn drafter_format(cols: usize) -> Format {
    let q8 = std::env::var("SMILEY_DSPARK_QUANT")
        .map(|v| v == "q8")
        .unwrap_or(false);
    if !q8 && cols.is_multiple_of(256) {
        Format::Q4_K
    } else {
        Format::Q8_0
    }
}

impl QW {
    fn rb(&self) -> usize {
        row_bytes(self.fmt, self.cols)
    }
    fn row(&self, r: usize) -> &[u8] {
        let rb = self.rb();
        &self.data[r * rb..(r + 1) * rb]
    }
    /// bf16 rows from the file, quantised in parallel (1.4 B weights at load).
    fn load(st: &SafeTensors, name: &str, threads: usize) -> Result<QW, String> {
        let t = st.get(name)?;
        if t.shape.len() != 2 {
            return Err(format!("{name}: expected a matrix, shape {:?}", t.shape));
        }
        let (rows, cols) = (t.shape[0], t.shape[1]);
        if !cols.is_multiple_of(32) {
            return Err(format!("{name}: {cols} columns is not a multiple of 32"));
        }
        let fmt = drafter_format(cols);
        let rb = row_bytes(fmt, cols);
        let mut data = vec![0u8; rows * rb];
        let n = threads.max(1);
        std::thread::scope(|s| {
            for (w, chunk) in data.chunks_mut(rows.div_ceil(n) * rb).enumerate() {
                let r0 = w * rows.div_ceil(n);
                s.spawn(move || {
                    let mut x = vec![0.0f32; cols];
                    let mut q = Vec::with_capacity(rb);
                    for (i, out) in chunk.chunks_exact_mut(rb).enumerate() {
                        st.row_f32(t, r0 + i, &mut x);
                        q.clear();
                        if fmt == Format::Q4_K {
                            quantize_row_q4_k(&x, &mut q)
                        } else {
                            quantize_row_q8_0(&x, &mut q)
                        }
                        out.copy_from_slice(&q);
                    }
                });
            }
        });
        Ok(QW {
            fmt,
            rows,
            cols,
            data,
        })
    }
    fn bytes(&self) -> u64 {
        self.data.len() as u64
    }
}

struct DLayer {
    ln1: Vec<f32>,
    ln2: Vec<f32>,
    wq: QW,
    wk: QW,
    wv: QW,
    wo: QW,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    gate: QW,
    up: QW,
    down: QW,
    /// Context keys and values, `[position][n_kv_head * head_dim]`.
    kc: Vec<f32>,
    vc: Vec<f32>,
}

/// The quantised-matrix cache: `<dir>/smiley-dspark-<fmt>.bin`, keyed by the
/// safetensors file's size and modification time so a replaced checkpoint is never
/// served stale. Layout: magic, key, count, then per matrix name, format, rows,
/// cols, byte length and the bytes.
const CACHE_MAGIC: &[u8; 8] = b"SMDSPK01";

fn cache_path(dir: &Path) -> std::path::PathBuf {
    let tag = if drafter_format(256) == Format::Q4_K {
        "q4k"
    } else {
        "q8"
    };
    dir.join(format!("smiley-dspark-{tag}.bin"))
}

fn cache_key(dir: &Path) -> Option<[u64; 2]> {
    let m = std::fs::metadata(dir.join("model.safetensors")).ok()?;
    let t = m
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_secs();
    Some([m.len(), t])
}

fn save_cache(
    dir: &Path,
    names: &[String],
    mats: &std::collections::HashMap<String, QW>,
) -> std::io::Result<()> {
    use std::io::Write;
    let key = cache_key(dir).ok_or_else(|| std::io::Error::other("no key"))?;
    let tmp = cache_path(dir).with_extension("tmp");
    let mut f = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
    f.write_all(CACHE_MAGIC)?;
    f.write_all(&key[0].to_le_bytes())?;
    f.write_all(&key[1].to_le_bytes())?;
    f.write_all(&(names.len() as u64).to_le_bytes())?;
    for n in names {
        let w = &mats[n];
        f.write_all(&(n.len() as u64).to_le_bytes())?;
        f.write_all(n.as_bytes())?;
        f.write_all(&[if w.fmt == Format::Q4_K { 1 } else { 0 }])?;
        for x in [w.rows as u64, w.cols as u64, w.data.len() as u64] {
            f.write_all(&x.to_le_bytes())?;
        }
        f.write_all(&w.data)?;
    }
    f.flush()?;
    drop(f);
    std::fs::rename(tmp, cache_path(dir))
}

fn load_cache(dir: &Path, names: &[String]) -> Option<std::collections::HashMap<String, QW>> {
    let key = cache_key(dir)?;
    let b = std::fs::read(cache_path(dir)).ok()?;
    let mut i = 0usize;
    let mut take = |n: usize| -> Option<&[u8]> {
        let s = b.get(i..i + n)?;
        i += n;
        Some(s)
    };
    if take(8)? != CACHE_MAGIC {
        return None;
    }
    let u = |s: &[u8]| u64::from_le_bytes(s.try_into().unwrap());
    if u(take(8)?) != key[0] || u(take(8)?) != key[1] {
        return None;
    }
    let count = u(take(8)?) as usize;
    let mut out = std::collections::HashMap::new();
    for _ in 0..count {
        let nl = u(take(8)?) as usize;
        let name = String::from_utf8(take(nl)?.to_vec()).ok()?;
        let fmt = if take(1)?[0] == 1 {
            Format::Q4_K
        } else {
            Format::Q8_0
        };
        let (rows, cols, len) = (
            u(take(8)?) as usize,
            u(take(8)?) as usize,
            u(take(8)?) as usize,
        );
        if len != rows * row_bytes(fmt, cols) {
            return None;
        }
        out.insert(
            name,
            QW {
                fmt,
                rows,
                cols,
                data: take(len)?.to_vec(),
            },
        );
    }
    names.iter().all(|n| out.contains_key(n)).then_some(out)
}

/// What one draft cost, for the telemetry record.
#[derive(Debug, Default, Clone, Copy)]
pub struct DraftStats {
    pub ns: u64,
    pub bytes: u64,
    pub ctx_ns: u64,
}

pub struct Drafter {
    pub cfg: DSparkConfig,
    embed: QW,
    fc: QW,
    hidden_norm: Vec<f32>,
    layers: Vec<DLayer>,
    norm: Vec<f32>,
    lm_head: QW,
    markov_w1: Option<QW>,
    markov_w2: Option<QW>,
    conf_w: Vec<f32>,
    conf_b: f32,
    /// Committed positions held in every layer's context cache.
    pub ctx_len: usize,
    inv_freq: Vec<f64>,
    /// Confidence on the normed hidden state (the MLX port's reading of DeepSpec) or
    /// the pre-norm one (DeepInfer's V4.1 head). Settable for the calibration check.
    pub conf_on_normed: bool,
    pub stats: DraftStats,
    pub weight_bytes: u64,
}

fn rmsnorm(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let ss = x.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / x.len() as f64;
    let inv = (1.0 / (ss + eps as f64).sqrt()) as f32;
    for ((o, v), g) in out.iter_mut().zip(x).zip(w) {
        *o = v * inv * g;
    }
}

#[derive(Clone, Copy)]
struct Out(*mut f32);
unsafe impl Send for Out {}
unsafe impl Sync for Out {}
impl Out {
    /// # Safety
    /// `i` must be in bounds and written by one worker only.
    #[inline(always)]
    unsafe fn set(self, i: usize, v: f32) {
        *self.0.add(i) = v;
    }
}

/// `out[b * rows + r] = w[r] · acts[b]` for all rows, split across the pool.
fn matmul(w: &QW, acts: &[QuantAct], out: &mut [f32], pool: &Pool) {
    let nb = acts.len();
    debug_assert!(out.len() >= nb * w.rows);
    let dst = Out(out.as_mut_ptr());
    let rb = w.rb();
    pool.run(|wk, nw| {
        let (lo, hi) = share(w.rows, wk, nw);
        let mut tmp = vec![0.0f32; nb];
        for r in lo..hi {
            dot_batch(w.fmt, &w.data[r * rb..(r + 1) * rb], acts, &mut tmp[..nb]);
            for (b, &v) in tmp[..nb].iter().enumerate() {
                // SAFETY: rows are partitioned disjointly by `share`.
                unsafe { dst.set(b * w.rows + r, v) };
            }
        }
    });
}

fn quantise_rows(x: &[f32], width: usize, acts: &mut Vec<QuantAct>) {
    let n = x.len() / width;
    while acts.len() < n {
        acts.push(QuantAct::with_capacity(width));
    }
    for (a, row) in acts.iter_mut().zip(x.chunks_exact(width)) {
        a.fill(row);
    }
}

impl Drafter {
    /// Load a drafter directory (`config.json` + `model.safetensors`).
    pub fn load(dir: &Path, threads: usize) -> Result<Drafter, String> {
        let cfg_text = std::fs::read_to_string(dir.join("config.json"))
            .map_err(|e| format!("{}: {e}", dir.display()))?;
        let cfg = DSparkConfig::from_json(&cfg_text)?;
        let st = SafeTensors::open(&dir.join("model.safetensors"))?;
        // Every matrix by name: from the quantised cache beside the weights when it
        // matches this file and format, else quantised now (≈45 s for Q4_K on four
        // cores) and the cache written for next time.
        let mut names: Vec<String> = ["embed_tokens.weight", "fc.weight", "lm_head.weight"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        for i in 0..cfg.n_layer {
            for m in [
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            ] {
                names.push(format!("layers.{i}.{m}.weight"));
            }
        }
        if cfg.markov_rank > 0 {
            names.push("markov_head.markov_w1.weight".into());
            names.push("markov_head.markov_w2.weight".into());
        }
        let mut mats = load_cache(dir, &names).unwrap_or_default();
        if mats.len() != names.len() {
            mats.clear();
            for n in &names {
                mats.insert(n.clone(), QW::load(&st, n, threads)?);
            }
            let _ = save_cache(dir, &names, &mats);
        }
        let cell = std::cell::RefCell::new(mats);
        let q = |n: &str| {
            cell.borrow_mut()
                .remove(n)
                .ok_or_else(|| format!("{n}: missing after load"))
        };
        let v = |n: &str| st.vec_f32(n);
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for i in 0..cfg.n_layer {
            let p = |s: &str| format!("layers.{i}.{s}");
            layers.push(DLayer {
                ln1: v(&p("input_layernorm.weight"))?,
                ln2: v(&p("post_attention_layernorm.weight"))?,
                wq: q(&p("self_attn.q_proj.weight"))?,
                wk: q(&p("self_attn.k_proj.weight"))?,
                wv: q(&p("self_attn.v_proj.weight"))?,
                wo: q(&p("self_attn.o_proj.weight"))?,
                q_norm: v(&p("self_attn.q_norm.weight"))?,
                k_norm: v(&p("self_attn.k_norm.weight"))?,
                gate: q(&p("mlp.gate_proj.weight"))?,
                up: q(&p("mlp.up_proj.weight"))?,
                down: q(&p("mlp.down_proj.weight"))?,
                kc: Vec::new(),
                vc: Vec::new(),
            });
        }
        let (markov_w1, markov_w2) = if cfg.markov_rank > 0 {
            (
                Some(q("markov_head.markov_w1.weight")?),
                Some(q("markov_head.markov_w2.weight")?),
            )
        } else {
            (None, None)
        };
        let (conf_w, conf_b) = if cfg.confidence {
            (
                v("confidence_head.proj.weight")?,
                v("confidence_head.proj.bias")?
                    .first()
                    .copied()
                    .unwrap_or(0.0),
            )
        } else {
            (Vec::new(), 0.0)
        };
        let half = cfg.head_dim / 2;
        let inv_freq = (0..half)
            .map(|i| cfg.rope_theta.powf(-((2 * i) as f64) / cfg.head_dim as f64))
            .collect();
        let mut d = Drafter {
            embed: q("embed_tokens.weight")?,
            fc: q("fc.weight")?,
            hidden_norm: v("hidden_norm.weight")?,
            layers,
            norm: v("norm.weight")?,
            lm_head: q("lm_head.weight")?,
            markov_w1,
            markov_w2,
            conf_w,
            conf_b,
            ctx_len: 0,
            inv_freq,
            conf_on_normed: std::env::var("SMILEY_DSPARK_CONF_PRENORM")
                .map(|v| v != "1")
                .unwrap_or(true),
            stats: DraftStats::default(),
            weight_bytes: 0,
            cfg,
        };
        d.weight_bytes = d.embed.bytes()
            + d.fc.bytes()
            + d.lm_head.bytes()
            + d.layers
                .iter()
                .map(|l| {
                    [&l.wq, &l.wk, &l.wv, &l.wo, &l.gate, &l.up, &l.down]
                        .iter()
                        .map(|w| w.bytes())
                        .sum::<u64>()
                })
                .sum::<u64>()
            + d.markov_w1.as_ref().map_or(0, QW::bytes)
            + d.markov_w2.as_ref().map_or(0, QW::bytes);
        if d.fc.cols != d.cfg.target_layer_ids.len() * d.cfg.hidden {
            return Err(format!(
                "fc takes {} inputs, expected {} target layers × {}",
                d.fc.cols,
                d.cfg.target_layer_ids.len(),
                d.cfg.hidden
            ));
        }
        if d.conf_w.len()
            != d.cfg.hidden
                + if d.cfg.confidence_with_markov {
                    d.cfg.markov_rank
                } else {
                    0
                }
            && d.cfg.confidence
        {
            return Err(format!("confidence head takes {} features", d.conf_w.len()));
        }
        Ok(d)
    }

    /// Whether this drafter was trained for `target`: same vocabulary, same width, and
    /// every tapped layer exists.
    pub fn check_target(
        &self,
        n_vocab: usize,
        d_model: usize,
        n_layer: usize,
    ) -> Result<(), String> {
        let c = &self.cfg;
        if c.n_vocab != n_vocab {
            return Err(format!(
                "drafter vocabulary {} ≠ target {}",
                c.n_vocab, n_vocab
            ));
        }
        if c.hidden != d_model {
            return Err(format!(
                "drafter reads {}-wide hidden states; the target's are {}",
                c.hidden, d_model
            ));
        }
        if let Some(&l) = c.target_layer_ids.iter().find(|&&l| l >= n_layer) {
            return Err(format!("drafter taps layer {l}; the target has {n_layer}"));
        }
        Ok(())
    }

    fn rope(&self, v: &mut [f32], n_heads: usize, pos: usize) {
        let hd = self.cfg.head_dim;
        let half = hd / 2;
        for h in 0..n_heads {
            let head = &mut v[h * hd..(h + 1) * hd];
            for i in 0..half {
                let th = pos as f64 * self.inv_freq[i];
                let (s, c) = (th.sin() as f32, th.cos() as f32);
                let (x0, x1) = (head[i], head[i + half]);
                head[i] = x0 * c - x1 * s;
                head[i + half] = x0 * s + x1 * c;
            }
        }
    }

    fn head_norm(x: &mut [f32], w: &[f32], eps: f32) {
        for row in x.chunks_exact_mut(w.len()) {
            let ss = row.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / row.len() as f64;
            let inv = (1.0 / (ss + eps as f64).sqrt()) as f32;
            for (v, g) in row.iter_mut().zip(w) {
                *v = *v * inv * g;
            }
        }
    }

    /// Forget every context row at or after `len` (a rollback, or a new conversation).
    pub fn truncate(&mut self, len: usize) {
        let kvd = self.cfg.n_kv_head * self.cfg.head_dim;
        self.ctx_len = self.ctx_len.min(len);
        for l in &mut self.layers {
            l.kc.truncate(self.ctx_len * kvd);
            l.vc.truncate(self.ctx_len * kvd);
        }
    }

    /// Append committed positions `start..start + rows/width` to the context, from the
    /// target's tapped hidden states (`[row][tap][hidden]`).
    pub fn update_context(&mut self, taps: &[f32], start: usize, pool: &Pool) {
        let t0 = Instant::now();
        let c = self.cfg.clone();
        let width = c.target_layer_ids.len() * c.hidden;
        let m = taps.len() / width;
        if m == 0 {
            return;
        }
        self.truncate(start);
        assert_eq!(self.ctx_len, start, "drafter context must be contiguous");
        let mut acts = Vec::new();
        quantise_rows(taps, width, &mut acts);
        let mut fused = vec![0.0f32; m * c.hidden];
        matmul(&self.fc, &acts[..m], &mut fused, pool);
        let mut normed = vec![0.0f32; m * c.hidden];
        for (o, x) in normed
            .chunks_exact_mut(c.hidden)
            .zip(fused.chunks_exact(c.hidden))
        {
            rmsnorm(x, &self.hidden_norm, c.eps, o);
        }
        quantise_rows(&normed, c.hidden, &mut acts);
        let kvd = c.n_kv_head * c.head_dim;
        let mut k = vec![0.0f32; m * kvd];
        let mut v = vec![0.0f32; m * kvd];
        for li in 0..self.layers.len() {
            matmul(&self.layers[li].wk, &acts[..m], &mut k, pool);
            matmul(&self.layers[li].wv, &acts[..m], &mut v, pool);
            Self::head_norm(&mut k, &self.layers[li].k_norm, c.eps);
            for r in 0..m {
                self.rope(&mut k[r * kvd..(r + 1) * kvd], c.n_kv_head, start + r);
            }
            let l = &mut self.layers[li];
            l.kc.extend_from_slice(&k);
            l.vc.extend_from_slice(&v);
        }
        self.ctx_len = start + m;
        self.stats.ctx_ns += t0.elapsed().as_nanos() as u64;
        self.stats.bytes += self.fc.bytes()
            + self
                .layers
                .iter()
                .map(|l| l.wk.bytes() + l.wv.bytes())
                .sum::<u64>();
    }

    /// Draft up to `k` tokens after `anchor`, the token at position `ctx_len`. Returns
    /// the tokens and, when the head has one, each one's conditional survival
    /// probability from the confidence head.
    pub fn draft(&mut self, anchor: u32, k: usize, pool: &Pool) -> (Vec<u32>, Vec<f32>) {
        let t0 = Instant::now();
        let c = self.cfg.clone();
        let k = k.min(c.block_size - c.logits_start);
        // `SMILEY_DSPARK_WIDTH` runs a narrower block than the trained one (the draft
        // is speed-only — the target verifies every token — so this trades acceptance
        // for drafter compute, measured rather than assumed).
        let width = std::env::var("SMILEY_DSPARK_WIDTH")
            .ok()
            .and_then(|v| v.parse().ok())
            // Default: one slot past the last verified draft. Measured on Qwen3-4B:
            // width 4 for three drafts kept acceptance (54% vs 58% chat, equal on code)
            // and cut the block forward by ~30%.
            .unwrap_or(c.logits_start + k + 1)
            .clamp(c.logits_start + k.max(1), c.block_size);
        let (b, h, hd) = (width, c.hidden, c.head_dim);
        let (qd, kvd) = (c.n_head * hd, c.n_kv_head * hd);
        let off = self.ctx_len;
        let mut bytes = 0u64;

        // Embed [anchor, mask, mask, …].
        let mut x = vec![0.0f32; b * h];
        for (i, row) in x.chunks_exact_mut(h).enumerate() {
            let id = if i == 0 { anchor } else { c.mask_id } as usize;
            dequant_row(self.embed.fmt, self.embed.row(id), row);
        }
        let mut xn = vec![0.0f32; b * h];
        let mut q = vec![0.0f32; b * qd];
        let mut kb = vec![0.0f32; b * kvd];
        let mut vb = vec![0.0f32; b * kvd];
        let mut att = vec![0.0f32; b * qd];
        let mut tmp_h = vec![0.0f32; b * h];
        let mut g = vec![0.0f32; b * c.n_ff];
        let mut u = vec![0.0f32; b * c.n_ff];
        let mut acts = Vec::new();
        let scale = 1.0 / (hd as f32).sqrt();
        let gqa = c.n_head / c.n_kv_head;

        for li in 0..self.layers.len() {
            let l = &self.layers[li];
            for (o, r) in xn.chunks_exact_mut(h).zip(x.chunks_exact(h)) {
                rmsnorm(r, &l.ln1, c.eps, o);
            }
            quantise_rows(&xn, h, &mut acts);
            matmul(&l.wq, &acts[..b], &mut q, pool);
            matmul(&l.wk, &acts[..b], &mut kb, pool);
            matmul(&l.wv, &acts[..b], &mut vb, pool);
            bytes += l.wq.bytes() + l.wk.bytes() + l.wv.bytes();
            Self::head_norm(&mut q, &l.q_norm, c.eps);
            Self::head_norm(&mut kb, &l.k_norm, c.eps);
            for i in 0..b {
                self.rope(&mut q[i * qd..(i + 1) * qd], c.n_head, off + i);
                self.rope(&mut kb[i * kvd..(i + 1) * kvd], c.n_kv_head, off + i);
            }
            // Attention: every block query over [context, whole block], per head.
            let l = &self.layers[li];
            let n_keys = off + b;
            let out = Out(att.as_mut_ptr());
            let (kc, vc, q_ref, kb_ref, vb_ref) = (&l.kc, &l.vc, &q, &kb, &vb);
            pool.run(|wk, nw| {
                let (lo, hi) = share(b * c.n_head, wk, nw);
                let mut s = vec![0.0f32; n_keys];
                let mut acc = vec![0.0f32; hd];
                for t in lo..hi {
                    let (i, hh) = (t / c.n_head, t % c.n_head);
                    let kvh = hh / gqa;
                    let qv = &q_ref[i * qd + hh * hd..i * qd + (hh + 1) * hd];
                    let key = |p: usize| {
                        if p < off {
                            &kc[p * kvd + kvh * hd..p * kvd + (kvh + 1) * hd]
                        } else {
                            &kb_ref[(p - off) * kvd + kvh * hd..(p - off) * kvd + (kvh + 1) * hd]
                        }
                    };
                    let val = |p: usize| {
                        if p < off {
                            &vc[p * kvd + kvh * hd..p * kvd + (kvh + 1) * hd]
                        } else {
                            &vb_ref[(p - off) * kvd + kvh * hd..(p - off) * kvd + (kvh + 1) * hd]
                        }
                    };
                    let mut mx = f32::NEG_INFINITY;
                    for (p, sp) in s.iter_mut().enumerate() {
                        *sp = key(p).iter().zip(qv).map(|(a, b)| a * b).sum::<f32>() * scale;
                        mx = mx.max(*sp);
                    }
                    let mut sum = 0.0f32;
                    for sp in s.iter_mut() {
                        *sp = (*sp - mx).exp();
                        sum += *sp;
                    }
                    acc.fill(0.0);
                    for (p, sp) in s.iter().enumerate() {
                        let w = sp / sum;
                        for (a, vv) in acc.iter_mut().zip(val(p)) {
                            *a += w * vv;
                        }
                    }
                    for (d, a) in acc.iter().enumerate() {
                        // SAFETY: (row i, head hh) is owned by exactly one task.
                        unsafe { out.set(i * qd + hh * hd + d, *a) };
                    }
                }
            });
            quantise_rows(&att, qd, &mut acts);
            matmul(&l.wo, &acts[..b], &mut tmp_h, pool);
            for (a, o) in x.iter_mut().zip(&tmp_h) {
                *a += o;
            }
            for (o, r) in xn.chunks_exact_mut(h).zip(x.chunks_exact(h)) {
                rmsnorm(r, &l.ln2, c.eps, o);
            }
            quantise_rows(&xn, h, &mut acts);
            matmul(&l.gate, &acts[..b], &mut g, pool);
            matmul(&l.up, &acts[..b], &mut u, pool);
            for (gi, ui) in g.iter_mut().zip(&u) {
                *gi = *gi / (1.0 + (-*gi).exp()) * ui;
            }
            quantise_rows(&g, c.n_ff, &mut acts);
            matmul(&l.down, &acts[..b], &mut tmp_h, pool);
            for (a, o) in x.iter_mut().zip(&tmp_h) {
                *a += o;
            }
            bytes += l.wo.bytes() + l.gate.bytes() + l.up.bytes() + l.down.bytes();
        }

        let t_layers = t0.elapsed();
        // Final norm, then the head over the predicted slots only.
        for (o, r) in xn.chunks_exact_mut(h).zip(x.chunks_exact(h)) {
            rmsnorm(r, &self.norm, c.eps, o);
        }
        let s0 = c.logits_start;
        quantise_rows(&xn[s0 * h..(s0 + k) * h], h, &mut acts);
        let mut logits = vec![0.0f32; k * c.n_vocab];
        matmul(&self.lm_head, &acts[..k], &mut logits, pool);
        bytes += self.lm_head.bytes();
        let t_head = t0.elapsed();

        // Sequential Markov bias + confidence.
        let mut tokens = Vec::with_capacity(k);
        let mut conf = Vec::with_capacity(k);
        let mut prev = anchor;
        let mut w1 = vec![0.0f32; c.markov_rank];
        let mut bias = vec![0.0f32; c.n_vocab];
        for i in 0..k {
            let row = &mut logits[i * c.n_vocab..(i + 1) * c.n_vocab];
            if let (Some(m1), Some(m2)) = (&self.markov_w1, &self.markov_w2) {
                dequant_row(m1.fmt, m1.row(prev as usize), &mut w1);
                quantise_rows(&w1, c.markov_rank, &mut acts);
                matmul(m2, &acts[..1], &mut bias, pool);
                bytes += m2.bytes();
                for (lg, bb) in row.iter_mut().zip(&bias) {
                    *lg += bb;
                }
            }
            let tok = row
                .iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |a, (j, &v)| {
                    if v > a.1 {
                        (j, v)
                    } else {
                        a
                    }
                })
                .0 as u32;
            if c.confidence {
                let hid = if self.conf_on_normed {
                    &xn[(s0 + i) * h..(s0 + i + 1) * h]
                } else {
                    &x[(s0 + i) * h..(s0 + i + 1) * h]
                };
                let mut z = self.conf_b
                    + hid
                        .iter()
                        .zip(&self.conf_w)
                        .map(|(a, b)| a * b)
                        .sum::<f32>();
                if c.confidence_with_markov {
                    z += w1
                        .iter()
                        .zip(&self.conf_w[h..])
                        .map(|(a, b)| a * b)
                        .sum::<f32>();
                }
                conf.push(1.0 / (1.0 + (-z.clamp(-60.0, 60.0)).exp()));
            }
            tokens.push(tok);
            prev = tok;
        }
        if std::env::var_os("SMILEY_DSPARK_PROF").is_some() {
            let t = t0.elapsed();
            eprintln!(
                "[dspark] layers {:.1} ms, head {:.1} ms, markov+conf {:.1} ms (k={k}, ctx={off})",
                t_layers.as_secs_f64() * 1e3,
                (t_head - t_layers).as_secs_f64() * 1e3,
                (t - t_head).as_secs_f64() * 1e3
            );
        }
        self.stats.ns += t0.elapsed().as_nanos() as u64;
        self.stats.bytes += bytes;
        (tokens, conf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CFG: &str = r#"{"architectures":["Qwen3DSparkModel"],"block_size":7,"confidence_head_with_markov":true,
      "enable_confidence_head":true,"head_dim":128,"hidden_size":2560,"intermediate_size":9728,
      "layer_types":["full_attention","full_attention"],"markov_rank":256,"mask_token_id":151669,
      "num_attention_heads":32,"num_hidden_layers":2,"num_key_value_heads":8,"rms_norm_eps":1e-06,
      "rope_parameters":{"rope_theta":1000000,"rope_type":"default"},"target_layer_ids":[1,9,17,25,33],"vocab_size":151936}"#;

    #[test]
    fn reads_deepseeks_drafter_config() {
        let c = DSparkConfig::from_json(CFG).unwrap();
        assert_eq!(
            (c.hidden, c.n_layer, c.n_head, c.n_kv_head, c.head_dim),
            (2560, 2, 32, 8, 128)
        );
        assert_eq!((c.block_size, c.mask_id, c.markov_rank), (7, 151669, 256));
        assert_eq!(c.target_layer_ids, vec![1, 9, 17, 25, 33]);
        assert!(c.confidence && c.confidence_with_markov);
        assert_eq!(c.rope_theta, 1e6);
        assert_eq!(
            c.logits_start, 0,
            "DeepSpec heads predict from the anchor slot"
        );
        // A sliding-window head (DFlash lineage) is refused rather than run wrongly.
        assert!(DSparkConfig::from_json(&CFG.replace(
            "\"full_attention\",\"full_attention\"",
            "\"sliding_attention\""
        ))
        .is_err());
        assert!(
            DSparkConfig::from_json(&CFG.replace("Qwen3DSparkModel", "Qwen3Eagle3Model")).is_err()
        );
    }
}
