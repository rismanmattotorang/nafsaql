//! The Llama block, and the forward pass.
//!
//! # Why the matmul is batched even though decode is not
//!
//! `matmul` takes a *slice* of activations and reads each weight row once for the
//! whole slice. Plain decode passes a slice of one, which is the GEMV case; prefill
//! passes the whole prompt; a speculative verification passes the drafted block.
//! Writing it this way is not premature generality — it is the entire mechanism by
//! which prefill and speculation are cheap. A decode step reads 267 MB of weights to
//! produce one token; running twenty prompt tokens through that path one at a time
//! would read 5.3 GB, where the batched path reads 267 MB once. The arithmetic
//! intensity of the batched call is `b` times higher, which is precisely what the
//! planner in `planner.rs` decides on.
//!
//! # Byte accounting
//!
//! Every weight read goes through `matmul` or `embed`, and both add the exact
//! on-disk byte length of what they touched to `StepStats`. That total is the
//! number the engine reports as bytes moved, and it is a sum of tensor lengths
//! rather than an estimate from parameter count — the two differ by the block
//! metadata, which at Q4_K is 12.5% of the traffic.

use crate::gguf::{f16, Format, Gguf, TensorInfo};
use crate::kv::KvCache;
use crate::pool::{share, Pool};
use crate::quant::{dot_batch, row_bytes, QuantAct};
use crate::tokenizer::Tokenizer;
use std::time::Instant;

/// The transformer families this engine implements. Qwen2 is llama with biases on
/// the attention projections and NeoX-style rotary pairing; everything else — norms,
/// SwiGLU, grouped-query attention — is identical, so it shares the forward pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch {
    Llama,
    Qwen2,
    /// Qwen3: no attention biases, a per-head RMSNorm on queries and keys before
    /// rotation, and a head width independent of the model width (Qwen3-4B runs 32
    /// heads of 128 over a 2560-wide residual). The targets DeepSeek's DSpark
    /// drafters were trained against.
    Qwen3,
}

impl Arch {
    pub fn from_name(name: &str) -> Option<Arch> {
        Some(match name {
            "llama" => Arch::Llama,
            "qwen2" => Arch::Qwen2,
            "qwen3" => Arch::Qwen3,
            _ => return None,
        })
    }
    /// The metadata key prefix (`llama.block_count`, `qwen2.block_count`, …).
    pub fn key_prefix(self) -> &'static str {
        match self {
            Arch::Llama => "llama",
            Arch::Qwen2 => "qwen2",
            Arch::Qwen3 => "qwen3",
        }
    }
}

/// Architecture names this crate implements, for the compatibility probe's message.
pub const ARCHS: &[&str] = &["llama", "qwen2", "qwen3"];

#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub arch: Arch,
    /// NeoX rotary pairing (`i` with `i + rope_dim/2`) instead of adjacent pairs.
    pub rope_neox: bool,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    pub d_model: usize,
    pub n_ff: usize,
    pub n_vocab: usize,
    pub rope_dim: usize,
    pub eps: f32,
    pub rope_base: f32,
    pub ctx_train: usize,
}

impl Config {
    pub fn kv_dim(&self) -> usize {
        self.n_kv_head * self.head_dim
    }
    /// Width of the concatenated query heads — the attention output's width, which
    /// is `d_model` for llama and Qwen2 and not for Qwen3.
    pub fn q_dim(&self) -> usize {
        self.n_head * self.head_dim
    }
    /// Query heads per key/value head. 1 means multi-head, >1 is grouped-query.
    pub fn gqa(&self) -> usize {
        self.n_head / self.n_kv_head
    }
}

/// What one forward call touched. Summed into the per-token telemetry record.
#[derive(Default, Clone, Copy, Debug)]
pub struct StepStats {
    /// Exact bytes of weight tensors read.
    pub bytes_weights: u64,
    /// Exact bytes read from the KV cache — `κ(T)`.
    pub bytes_kv: u64,
    /// Multiply-accumulates, counted rather than estimated. Two FLOPs each.
    pub macs: u64,
    /// Fork-join barriers executed. Proposition 5 of the paper claims this is not
    /// the binding constraint; reporting it is what makes that checkable.
    pub barriers: u64,
    /// Wall nanoseconds inside the weight matmuls (every layer's projections).
    pub ns_matmul: u64,
    /// ... inside the output-head matmul, kept apart: at 49 152 rows it is a quarter
    /// of a token's bytes and behaves differently from the layer matmuls.
    pub ns_head: u64,
    /// ... inside attention: scores, softmax and the value mix.
    pub ns_attn: u64,
    /// Wall nanoseconds of the whole forward call. Everything that is none of the
    /// above — norms, rope, residuals, SwiGLU, activation quantisation, the cache
    /// writes — is `ns_total − ns_matmul − ns_head − ns_attn`, derived rather than
    /// timed so the parts always sum to the whole.
    pub ns_total: u64,
}

impl StepStats {
    /// Nanoseconds that are not matmul, head or attention.
    pub fn ns_other(&self) -> u64 {
        self.ns_total
            .saturating_sub(self.ns_matmul + self.ns_head + self.ns_attn)
    }
    pub fn bytes_total(&self) -> u64 {
        self.bytes_weights + self.bytes_kv
    }
    /// FLOP per byte — the x-axis of the roofline.
    pub fn arithmetic_intensity(&self) -> f64 {
        let b = self.bytes_total();
        if b == 0 {
            0.0
        } else {
            2.0 * self.macs as f64 / b as f64
        }
    }
    pub fn add(&mut self, o: &StepStats) {
        self.bytes_weights += o.bytes_weights;
        self.bytes_kv += o.bytes_kv;
        self.macs += o.macs;
        self.barriers += o.barriers;
        self.ns_matmul += o.ns_matmul;
        self.ns_head += o.ns_head;
        self.ns_attn += o.ns_attn;
        self.ns_total += o.ns_total;
    }
}

struct Layer {
    attn_norm: TensorInfo,
    wq: TensorInfo,
    wk: TensorInfo,
    wv: TensorInfo,
    /// Qwen2's attention biases, decoded once at load (a few KB per layer). `None`
    /// for llama, which has none — and a file that is missing one it should have
    /// fails at load, not by silently running without it.
    bq: Option<Vec<f32>>,
    bk: Option<Vec<f32>>,
    bv: Option<Vec<f32>>,
    /// Qwen3's per-head query and key RMSNorm weights (`head_dim` floats each).
    q_norm: Option<Vec<f32>>,
    k_norm: Option<Vec<f32>>,
    wo: TensorInfo,
    ffn_norm: TensorInfo,
    gate: TensorInfo,
    up: TensorInfo,
    down: TensorInfo,
}

pub struct Model {
    g: Gguf,
    pub cfg: Config,
    pub tk: Tokenizer,
    tok_embd: TensorInfo,
    out_norm: TensorInfo,
    /// `None` when the model ties the output projection to the embedding table.
    /// Detected by absence — assuming either way silently transposes the head.
    output: Option<TensorInfo>,
    /// Llama 3.x's per-frequency rotary factors (`rope_freqs.weight`, `rope_dim/2`
    /// floats): each angle is divided by its factor. Absent on every other model.
    pub rope_factors: Option<Vec<f32>>,
    layers: Vec<Layer>,
}

/// Reusable scratch. Allocated once per session so a decode step does no allocation.
pub struct State {
    pub x: Vec<f32>,
    xb: Vec<f32>,
    xb2: Vec<f32>,
    hb: Vec<f32>,
    hb2: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    /// One softmax row per worker, `max_ctx + 1` long: attention is split by head
    /// across the pool and each worker needs a private scratch row.
    att: Vec<f32>,
    att_stride: usize,
    /// `(cos, sin)` of every rotary angle for every position, computed once. Rope was
    /// recomputing `powf`, `sin` and `cos` for every pair of every head at every
    /// layer of every token — 20 000 transcendental calls per decode step.
    rope: Vec<f32>,
    acts: Vec<QuantAct>,
    pub logits: Vec<f32>,
    max_batch: usize,
    /// Layers whose output (the residual stream after the layer) `forward` copies
    /// out, in this order — what a DSpark drafter conditions on. Empty: no copy.
    tap_layers: Vec<usize>,
    /// `[batch row][tap][d_model]`, written by every `forward` while taps are set.
    pub tap: Vec<f32>,
}

impl State {
    /// `max_logit_rows` is sized separately from `max_batch` on purpose. A logits
    /// row is `n_vocab` floats — 192 KB for this model — so sizing the logits buffer
    /// by the prefill batch would allocate 39 MB to hold output that prefill never
    /// reads. Only speculation needs more than one row.
    pub fn new(
        cfg: &Config,
        max_batch: usize,
        max_logit_rows: usize,
        max_ctx: usize,
        threads: usize,
        rope_factors: Option<&[f32]>,
    ) -> State {
        let b = max_batch;
        let att_stride = max_ctx + 1;
        let half = cfg.rope_dim / 2;
        let mut rope = vec![0.0f32; max_ctx * half * 2];
        for pos in 0..max_ctx {
            for i in 0..half {
                // Same expression, same precision as the per-call version it replaces,
                // so the cached values are bit-identical to what was being computed.
                // With factors (Llama 3.x), ggml divides the angle by `factor[i]`.
                let freq = (cfg.rope_base as f64).powf(-((2 * i) as f64) / cfg.rope_dim as f64);
                let factor = rope_factors.and_then(|f| f.get(i)).copied().unwrap_or(1.0) as f64;
                let theta = pos as f64 * freq / factor;
                rope[(pos * half + i) * 2] = theta.cos() as f32;
                rope[(pos * half + i) * 2 + 1] = theta.sin() as f32;
            }
        }
        State {
            x: vec![0.0; b * cfg.d_model],
            // Holds both a normed residual row (d_model wide) and an attention output
            // row (q_dim wide); each phase indexes it with its own stride.
            xb: vec![0.0; b * cfg.d_model.max(cfg.q_dim())],
            xb2: vec![0.0; b * cfg.d_model],
            hb: vec![0.0; b * cfg.n_ff],
            hb2: vec![0.0; b * cfg.n_ff],
            q: vec![0.0; b * cfg.n_head * cfg.head_dim],
            k: vec![0.0; b * cfg.kv_dim()],
            v: vec![0.0; b * cfg.kv_dim()],
            att: vec![0.0; att_stride * threads.max(1)],
            att_stride,
            rope,
            acts: (0..b).map(|_| QuantAct::with_capacity(cfg.n_ff)).collect(),
            // One row per batch element, not one row total. Speculative verification
            // needs every drafted position's logits at once; a single shared row
            // would leave each verification reading the last row's distribution and
            // accepting drafts it should have rejected.
            logits: vec![0.0; max_logit_rows.max(1) * cfg.n_vocab],
            max_batch: b,
            tap_layers: Vec::new(),
            tap: Vec::new(),
        }
    }

    /// Copy the residual stream after each of `layers` out of every forward call,
    /// into `tap` as `[row][tap][d_model]` (the layout a DSpark `fc` reads).
    pub fn set_taps(&mut self, layers: &[usize], d_model: usize) {
        self.tap_layers = layers.to_vec();
        self.tap = vec![0.0; self.max_batch * layers.len() * d_model];
    }

    /// The tapped hidden states of batch row `b`: `n_taps × d_model` floats.
    pub fn tap_row(&self, b: usize, d_model: usize) -> &[f32] {
        let w = self.tap_layers.len() * d_model;
        &self.tap[b * w..(b + 1) * w]
    }

    /// Logits for the `i`-th row of the last `want_logits` request.
    pub fn logits_row(&self, i: usize, n_vocab: usize) -> &[f32] {
        &self.logits[i * n_vocab..(i + 1) * n_vocab]
    }
}

/// RMSNorm of one attention head in place, against its `head_dim` weights (Qwen3's
/// q/k norm). Summed in f64 like `rmsnorm_into`, so both norms round the same way.
fn head_rmsnorm(x: &mut [f32], w: &[f32], eps: f32) {
    let ss = x.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / x.len() as f64;
    let inv = (1.0 / (ss + eps as f64).sqrt()) as f32;
    for (v, g) in x.iter_mut().zip(w) {
        *v = *v * inv * g;
    }
}

/// A write cursor into the matmul's output, shared across workers.
///
/// Raw rather than a `&mut [f32]` split because the split is by row *stride*
/// (`b*rows + r`), not by a contiguous slice: with a batch, each worker writes an
/// interleaved set of ranges that `split_at_mut` cannot express. Disjointness is
/// guaranteed by `pool::share` instead, and tested there.
#[derive(Clone, Copy)]
struct OutPtr(*mut f32);
// SAFETY: workers write only indices inside their own share, which are disjoint.
unsafe impl Send for OutPtr {}
unsafe impl Sync for OutPtr {}

impl OutPtr {
    /// # Safety
    /// `i` must be inside the output buffer and must belong to the calling worker's
    /// share, so no two workers ever write the same index.
    ///
    /// This is a method rather than a bare field access on purpose: with edition
    /// 2021's disjoint closure capture, writing `dst.0` inside the worker closure
    /// captures the `*mut f32` field instead of the wrapper, and the closure then
    /// fails `Sync` however the wrapper is marked.
    #[inline(always)]
    unsafe fn set(self, i: usize, v: f32) {
        *self.0.add(i) = v;
    }
}

/// A shared base pointer from which workers carve out *disjoint* sub-slices: one
/// head's output, or one worker's softmax row. Same contract as `OutPtr`, for ranges.
#[derive(Clone, Copy)]
struct SharedBuf(*mut f32);
// SAFETY: every user takes sub-ranges no other worker takes (asserted by construction:
// ranges are keyed by the task or worker index).
unsafe impl Send for SharedBuf {}
unsafe impl Sync for SharedBuf {}

impl SharedBuf {
    /// # Safety
    /// `off..off+len` must be inside the buffer and used by exactly one worker.
    #[inline(always)]
    unsafe fn slice<'a>(self, off: usize, len: usize) -> &'a mut [f32] {
        std::slice::from_raw_parts_mut(self.0.add(off), len)
    }
}

/// Disjoint per-row access to the activation scratch, for the row-parallel prologues.
#[derive(Clone, Copy)]
struct SharedActs(*mut QuantAct);
// SAFETY: as `SharedBuf` — each index is taken by exactly one worker.
unsafe impl Send for SharedActs {}
unsafe impl Sync for SharedActs {}

impl SharedActs {
    /// # Safety
    /// `i` must be in bounds and used by exactly one worker.
    #[inline(always)]
    unsafe fn get<'a>(self, i: usize) -> &'a mut QuantAct {
        &mut *self.0.add(i)
    }
}

/// Batch rows at which the per-row work (norm, SwiGLU, quantisation) is split across
/// the pool. One decode row is a few microseconds — less than a fork–join.
const ROW_PARALLEL: usize = 4;

/// One matrix of a fused matmul group, and where its output goes.
struct MatJob<'a> {
    w: &'a TensorInfo,
    out: OutPtr,
}

// nafs-aql: where each row of a batch writes and reads its attention cache.
//
// Both shapes run the same forward pass, because every matmul in it is the same work whatever a row belongs to:
// one read of the weights serves the whole batch. `Shared` is one sequence over several positions (prefill, a
// speculative block). `PerRow` is one position of each of several sequences — several users in one weight pass,
// which is what makes concurrent serving on a memory-bound machine cost little more than serving one.
pub enum Caches<'a> {
    /// Every row writes to one cache, at its own position.
    Shared(&'a mut KvCache),
    /// Row `b` writes to cache `b`.
    PerRow(&'a mut [&'a mut KvCache]),
}

impl Caches<'_> {
    #[inline]
    fn get(&self, b: usize) -> &KvCache {
        match self {
            Caches::Shared(kv) => kv,
            Caches::PerRow(kvs) => kvs[b],
        }
    }
    #[inline]
    fn get_mut(&mut self, b: usize) -> &mut KvCache {
        match self {
            Caches::Shared(kv) => kv,
            Caches::PerRow(kvs) => kvs[b],
        }
    }
    /// Rows this holds caches for, or None when one cache serves every row.
    fn rows(&self) -> Option<usize> {
        match self {
            Caches::Shared(_) => None,
            Caches::PerRow(kvs) => Some(kvs.len()),
        }
    }
}

/// Largest batch one matmul call accepts. A stack array of this many scratch floats
/// holds one row's results; `PREFILL_CHUNK` and `draft_len + 1` are both far below it.
const MAX_BATCH: usize = 128;

#[inline]
fn f32_at(b: &[u8], i: usize) -> f32 {
    f32::from_le_bytes([b[i * 4], b[i * 4 + 1], b[i * 4 + 2], b[i * 4 + 3]])
}

impl Model {
    pub fn load(g: Gguf) -> Result<Model, String> {
        let arch_name = g.str("general.architecture")?;
        let arch = Arch::from_name(arch_name).ok_or_else(|| {
            format!(
                "Smiley Core implements {}; this model is {arch_name:?}",
                ARCHS.join(" and ")
            )
        })?;
        let k = |suffix: &str| format!("{}.{suffix}", arch.key_prefix());
        let n_layer = g.u32(&k("block_count"))? as usize;
        let d_model = g.u32(&k("embedding_length"))? as usize;
        let n_head = g.u32(&k("attention.head_count"))? as usize;
        let n_kv_head = g
            .u32(&k("attention.head_count_kv"))
            .unwrap_or(n_head as u32) as usize;
        let n_ff = g.u32(&k("feed_forward_length"))? as usize;
        // Rope scaling other than Llama 3's factor table is not implemented; a file
        // asking for it would run with the wrong angles and drift after a few tokens.
        if let Ok(t) = g.str(&k("rope.scaling.type")) {
            if t != "none" && t != "linear" {
                return Err(format!("rope scaling {t:?} is not implemented"));
            }
        }
        if n_head == 0 || n_kv_head == 0 || !n_head.is_multiple_of(n_kv_head) {
            return Err(format!(
                "implausible head configuration: {n_head} query heads over {n_kv_head} kv heads"
            ));
        }
        let head_dim = g
            .u32(&k("attention.key_length"))
            .map(|v| v as usize)
            .unwrap_or(d_model / n_head);
        // Qwen3 decouples the head width from the model width; the others must match.
        if arch != Arch::Qwen3 && head_dim * n_head != d_model {
            return Err(format!(
                "head dim {head_dim} × {n_head} heads ≠ embedding length {d_model}"
            ));
        }
        if !(head_dim * n_head).is_multiple_of(32) {
            return Err(format!(
                "query width {} is not a multiple of 32",
                head_dim * n_head
            ));
        }
        let rope_dim = g.u32(&k("rope.dimension_count")).unwrap_or(head_dim as u32) as usize;
        if rope_dim > head_dim || !rope_dim.is_multiple_of(2) {
            return Err(format!(
                "implausible rope dimension {rope_dim} for head dim {head_dim}"
            ));
        }
        // The activation quantiser works in blocks of 32, so both matmul widths must
        // be multiples of it. Checked here rather than tripping a debug assert deep
        // inside the first token.
        for (what, n) in [("embedding length", d_model), ("feed-forward length", n_ff)] {
            if !n.is_multiple_of(32) {
                return Err(format!("{what} {n} is not a multiple of 32"));
            }
        }

        let rope_scale = g
            .f32(&k("rope.scaling.factor"))
            .ok()
            .filter(|f| *f > 0.0)
            .map(|f| 1.0 / f);
        let cfg = Config {
            arch,
            rope_neox: arch != Arch::Llama,
            n_layer,
            n_head,
            n_kv_head,
            head_dim,
            d_model,
            n_ff,
            n_vocab: g.u32(&k("vocab_size")).unwrap_or(0) as usize,
            rope_dim,
            eps: g
                .f32(&k("attention.layer_norm_rms_epsilon"))
                .unwrap_or(1e-5),
            rope_base: g.f32(&k("rope.freq_base")).unwrap_or(10000.0),
            ctx_train: g.u32(&k("context_length")).unwrap_or(2048) as usize,
        };
        // Linear rope scaling folds into the base angle; it is expressed here as a
        // uniform factor table so one code path serves both it and Llama 3's.
        let mut rope_factors: Option<Vec<f32>> = rope_scale.map(|s| vec![1.0 / s; rope_dim / 2]);
        if let Ok(t) = g.tensor("rope_freqs.weight") {
            let n = t.numel() as usize;
            if n != rope_dim / 2 {
                return Err(format!(
                    "rope_freqs.weight has {n} entries, expected {}",
                    rope_dim / 2
                ));
            }
            let mut f = vec![0.0f32; n];
            crate::quant::dequant_row(t.kind, g.bytes(t), &mut f);
            if let Some(base) = &rope_factors {
                for (x, b) in f.iter_mut().zip(base) {
                    *x *= b;
                }
            }
            rope_factors = Some(f);
        }

        let get = |n: &str| -> Result<TensorInfo, String> { g.tensor(n).cloned() };
        // Biases are tiny and read on every token; decode them once rather than
        // dequantising a 128-float row per layer per step.
        let bias = |n: &str| -> Result<Option<Vec<f32>>, String> {
            match arch {
                Arch::Llama | Arch::Qwen3 => Ok(None),
                Arch::Qwen2 => {
                    let t = get(n)?;
                    let mut v = vec![0.0f32; t.numel() as usize];
                    crate::quant::dequant_row(t.kind, g.bytes(&t), &mut v);
                    Ok(Some(v))
                }
            }
        };
        // Qwen3's q/k norms: `head_dim` floats, decoded once like the biases.
        let head_norm = |n: &str| -> Result<Option<Vec<f32>>, String> {
            if arch != Arch::Qwen3 {
                return Ok(None);
            }
            let t = get(n)?;
            if t.numel() as usize != head_dim {
                return Err(format!(
                    "{n} has {} entries, expected the head dim {head_dim}",
                    t.numel()
                ));
            }
            let mut v = vec![0.0f32; head_dim];
            crate::quant::dequant_row(t.kind, g.bytes(&t), &mut v);
            Ok(Some(v))
        };
        let mut layers = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            layers.push(Layer {
                attn_norm: get(&format!("blk.{i}.attn_norm.weight"))?,
                wq: get(&format!("blk.{i}.attn_q.weight"))?,
                wk: get(&format!("blk.{i}.attn_k.weight"))?,
                wv: get(&format!("blk.{i}.attn_v.weight"))?,
                bq: bias(&format!("blk.{i}.attn_q.bias"))?,
                bk: bias(&format!("blk.{i}.attn_k.bias"))?,
                bv: bias(&format!("blk.{i}.attn_v.bias"))?,
                q_norm: head_norm(&format!("blk.{i}.attn_q_norm.weight"))?,
                k_norm: head_norm(&format!("blk.{i}.attn_k_norm.weight"))?,
                wo: get(&format!("blk.{i}.attn_output.weight"))?,
                ffn_norm: get(&format!("blk.{i}.ffn_norm.weight"))?,
                gate: get(&format!("blk.{i}.ffn_gate.weight"))?,
                up: get(&format!("blk.{i}.ffn_up.weight"))?,
                down: get(&format!("blk.{i}.ffn_down.weight"))?,
            });
        }
        let tok_embd = get("token_embd.weight")?;
        let out_norm = get("output_norm.weight")?;
        let output = g.tensor("output.weight").ok().cloned();

        let tk = Tokenizer::from_gguf(&g)?;
        let mut cfg = cfg;
        if cfg.n_vocab == 0 {
            cfg.n_vocab = tk.n_vocab;
        }
        if cfg.n_vocab != tk.n_vocab {
            return Err(format!(
                "metadata says {} vocabulary entries but the tokenizer has {}",
                cfg.n_vocab, tk.n_vocab
            ));
        }

        Ok(Model {
            g,
            cfg,
            tk,
            tok_embd,
            out_norm,
            output,
            rope_factors,
            layers,
        })
    }

    pub fn weight_bytes(&self) -> u64 {
        self.g.total_weight_bytes()
    }

    /// The file's metadata, for the template detector and the host's telemetry.
    pub fn gguf(&self) -> &Gguf {
        &self.g
    }

    /// `general.name`, or an empty string.
    pub fn name(&self) -> &str {
        self.g.str("general.name").unwrap_or("")
    }

    /// Formats present in this model and how many tensors use each — the input to
    /// the mixed-precision report.
    pub fn format_mix(&self) -> Vec<(Format, usize, u64)> {
        let mut out: Vec<(Format, usize, u64)> = Vec::new();
        for t in self.g.tensors.values() {
            match out.iter_mut().find(|e| e.0 == t.kind) {
                Some(e) => {
                    e.1 += 1;
                    e.2 += t.bytes();
                }
                None => out.push((t.kind, 1, t.bytes())),
            }
        }
        out.sort_by_key(|e| e.0.label());
        out
    }

    /// Bits per weight averaged over the whole model, weighted by tensor size.
    /// This is the `bpw` of the performance model, and it is measured, not quoted
    /// from the file's `general.file_type` — which for the reference model claims
    /// Q4_K_M while the tensors are mostly Q5_0.
    pub fn effective_bpw(&self) -> f64 {
        let bytes: u64 = self.g.tensors.values().map(|t| t.bytes()).sum();
        let params: u64 = self.g.tensors.values().map(|t| t.numel()).sum();
        if params == 0 {
            0.0
        } else {
            8.0 * bytes as f64 / params as f64
        }
    }

    /// One row of the embedding table, dequantised. The only place a weight is
    /// decoded into a buffer — it is a single row of 960, not a tensor.
    fn embed(&self, token: u32, out: &mut [f32], st: &mut StepStats) {
        let t = &self.tok_embd;
        let n = t.dims[0] as usize;
        let rb = row_bytes(t.kind, n);
        let all = self.g.bytes(t);
        let off = token as usize * rb;
        crate::quant::dequant_row(t.kind, &all[off..off + rb], &mut out[..n]);
        st.bytes_weights += rb as u64;
    }

    /// `out[b][row] = Σ w[row]·acts[b]` for every row of `w` and every activation.
    ///
    /// The row is read once and used for all `acts.len()` activations, so weight
    /// traffic is independent of batch size — and, since the batched kernels, so is
    /// the cost of *decoding* it. That is the whole point.
    ///
    /// Rows are split across the pool in contiguous, disjoint ranges — one fork–join
    /// for the whole matmul, which is the barrier the telemetry counts.
    fn matmul(
        &self,
        w: &TensorInfo,
        acts: &[QuantAct],
        nb: usize,
        out: &mut [f32],
        pool: &Pool,
        st: &mut StepStats,
    ) {
        debug_assert_eq!(out.len(), nb * w.dims[1] as usize);
        let jobs = [MatJob {
            w,
            out: OutPtr(out.as_mut_ptr()),
        }];
        self.matmul_group(&jobs, acts, nb, pool, st);
    }

    /// Several matrices that share one activation — Q, K and V; gate and up — in a
    /// single fork–join. Three barriers become one, and the workers run through the
    /// concatenated row space, so the split stays balanced even though K and V have a
    /// third of Q's rows. The byte, MAC and barrier accounting is unchanged except
    /// that the barrier count honestly falls.
    fn matmul_group(
        &self,
        jobs: &[MatJob],
        acts: &[QuantAct],
        nb: usize,
        pool: &Pool,
        st: &mut StepStats,
    ) {
        assert!(
            nb <= MAX_BATCH,
            "batch of {nb} exceeds the matmul scratch of {MAX_BATCH}"
        );
        let t0 = Instant::now();
        let total: usize = jobs.iter().map(|j| j.w.dims[1] as usize).sum();
        let acts = &acts[..nb];
        pool.run(|wk, nw| {
            let (lo, hi) = share(total, wk, nw);
            let mut tmp = [0.0f32; MAX_BATCH];
            let mut off = 0usize;
            for j in jobs {
                let n = j.w.dims[0] as usize;
                let rows = j.w.dims[1] as usize;
                let (a, z) = (lo.max(off), hi.min(off + rows));
                if a < z {
                    let rb = row_bytes(j.w.kind, n);
                    let bytes = self.g.bytes(j.w);
                    for r in (a - off)..(z - off) {
                        dot_batch(j.w.kind, &bytes[r * rb..(r + 1) * rb], acts, &mut tmp[..nb]);
                        for (b, &v) in tmp[..nb].iter().enumerate() {
                            // SAFETY: `share` partitions the concatenated row space into
                            // disjoint contiguous ranges, so `(job, r)` — hence the index
                            // `b*rows + r` within this job's output — is written by exactly
                            // one worker. The outputs outlive the call because `run` does
                            // not return until every worker has finished.
                            unsafe { j.out.set(b * rows + r, v) };
                        }
                    }
                }
                off += rows;
            }
        });
        // Counted once per matrix: the bytes crossed the memory interface once,
        // however many activations reused them.
        for j in jobs {
            let n = j.w.dims[0] as usize;
            let rows = j.w.dims[1] as usize;
            st.bytes_weights += (rows * row_bytes(j.w.kind, n)) as u64;
            st.macs += (rows * n * nb) as u64;
        }
        st.barriers += 1;
        st.ns_matmul += t0.elapsed().as_nanos() as u64;
    }

    fn rmsnorm(&self, x: &[f32], w: &TensorInfo, out: &mut [f32], st: &mut StepStats) {
        self.rmsnorm_into(x, w, out);
        st.bytes_weights += w.bytes();
    }

    fn rmsnorm_into(&self, x: &[f32], w: &TensorInfo, out: &mut [f32]) {
        let n = x.len();
        let wb = self.g.bytes(w);
        let ss = x.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / n as f64;
        let inv = (1.0 / (ss + self.cfg.eps as f64).sqrt()) as f32;
        match w.kind {
            Format::F32 => {
                for i in 0..n {
                    out[i] = x[i] * inv * f32_at(wb, i);
                }
            }
            Format::F16 => {
                for i in 0..n {
                    out[i] = x[i] * inv * f16(u16::from_le_bytes([wb[i * 2], wb[i * 2 + 1]]));
                }
            }
            other => {
                // A quantised norm vector would be a model we cannot read correctly;
                // refuse rather than approximate.
                panic!("norm weight in {} is not supported", other.label());
            }
        }
    }

    /// For every batch row: `xb[b] = rmsnorm(x[b])`, then quantise it into `acts[b]`.
    ///
    /// Rows are independent, so with a batch (prefill, speculative verification) they
    /// are split across the pool. This was a fifth of prefill time on one thread.
    fn norm_rows(
        &self,
        st8: &mut State,
        w: &TensorInfo,
        nb: usize,
        pool: &Pool,
        stats: &mut StepStats,
    ) {
        let d = self.cfg.d_model;
        let x: &[f32] = &st8.x;
        let xb = SharedBuf(st8.xb.as_mut_ptr());
        let acts = SharedActs(st8.acts.as_mut_ptr());
        let body = |wk: usize, nw: usize| {
            let (lo, hi) = share(nb, wk, nw);
            for b in lo..hi {
                // SAFETY: row `b` of `xb` and `acts[b]` belong to this worker alone.
                let out = unsafe { xb.slice(b * d, d) };
                self.rmsnorm_into(&x[b * d..(b + 1) * d], w, out);
                unsafe { acts.get(b) }.fill(out);
            }
        };
        if nb >= ROW_PARALLEL && pool.width() > 1 {
            pool.run(body);
            stats.barriers += 1;
        } else {
            body(0, 1);
        }
        // The norm vector is read once per row, as before.
        stats.bytes_weights += w.bytes() * nb as u64;
    }

    /// SwiGLU — `hb = silu(gate) ⊙ up` — and its quantisation, row-parallel like
    /// `norm_rows`.
    fn swiglu_rows(&self, st8: &mut State, nb: usize, pool: &Pool, stats: &mut StepStats) {
        let f = self.cfg.n_ff;
        let hb = SharedBuf(st8.hb.as_mut_ptr());
        let up: &[f32] = &st8.hb2;
        let acts = SharedActs(st8.acts.as_mut_ptr());
        let body = |wk: usize, nw: usize| {
            let (lo, hi) = share(nb, wk, nw);
            for b in lo..hi {
                // SAFETY: as `norm_rows`.
                let g = unsafe { hb.slice(b * f, f) };
                for (gi, &u) in g.iter_mut().zip(&up[b * f..(b + 1) * f]) {
                    *gi = (*gi / (1.0 + (-*gi).exp())) * u;
                }
                unsafe { acts.get(b) }.fill(g);
            }
        };
        if nb >= ROW_PARALLEL && pool.width() > 1 {
            pool.run(body);
            stats.barriers += 1;
        } else if nb == 1 && pool.width() > 1 && f >= 1024 {
            // Decode: one row, but thousands of `exp`s on the calling thread while
            // the workers spin — a tenth of a SmolLM2 step at four threads (round 13).
            // The activation is split across the pool for the gate (same arithmetic,
            // element by element), then quantised once, vectorised, on this thread.
            let g = SharedBuf(st8.hb.as_mut_ptr());
            pool.run(|wk, nw| {
                let (lo, hi) = share(f, wk, nw);
                // SAFETY: `share` gives each worker a disjoint range of the row.
                let gs = unsafe { g.slice(lo, hi - lo) };
                for (gi, &u) in gs.iter_mut().zip(&up[lo..hi]) {
                    *gi = (*gi / (1.0 + (-*gi).exp())) * u;
                }
            });
            stats.barriers += 1;
            st8.acts[0].fill(&st8.hb[..f]);
        } else {
            body(0, 1);
        }
    }

    /// Rotary embedding, ggml's "norm" layout: consecutive pairs `(2i, 2i+1)` are
    /// rotated by `pos · base^(-2i/rope_dim)`. NeoX-style models pair `i` with
    /// `i + dim/2` instead; llama does not, and using the wrong pairing produces a
    /// model that is coherent for a few tokens and then drifts.
    fn rope(&self, rope: &[f32], v: &mut [f32], n_heads: usize, pos: usize) {
        let hd = self.cfg.head_dim;
        let half = self.cfg.rope_dim / 2;
        let table = &rope[pos * half * 2..(pos + 1) * half * 2];
        for h in 0..n_heads {
            let head = &mut v[h * hd..(h + 1) * hd];
            if self.cfg.rope_neox {
                // Qwen2 (and GPT-NeoX): the two halves of the rotary span are paired.
                for i in 0..half {
                    let (c, s) = (table[i * 2], table[i * 2 + 1]);
                    let (x0, x1) = (head[i], head[i + half]);
                    head[i] = x0 * c - x1 * s;
                    head[i + half] = x0 * s + x1 * c;
                }
            } else {
                for i in 0..half {
                    let (c, s) = (table[i * 2], table[i * 2 + 1]);
                    let (x0, x1) = (head[2 * i], head[2 * i + 1]);
                    head[2 * i] = x0 * c - x1 * s;
                    head[2 * i + 1] = x0 * s + x1 * c;
                }
            }
        }
    }

    /// Run `tokens` at positions `start_pos..start_pos+tokens.len()`.
    ///
    /// Writes logits for the batch rows named in `want_logits`. Prefill asks for the
    /// last row only: computing all of them would cost an extra 47 M multiply-adds
    /// per prompt token for output nobody reads.
    // Eight arguments, and each is a separate axis: scratch, cache, pool, the tokens,
    // where they start, which rows want logits, and where the byte accounting goes.
    // Bundling them into a config struct would hide that `want_logits` is what makes
    // prefill cheap and speculation possible.
    #[allow(clippy::too_many_arguments)]
    /// One sequence over `tokens`, starting at `start_pos` (prefill, a decode step, a speculative block).
    pub fn forward(
        &self,
        st8: &mut State,
        kv: &mut KvCache,
        pool: &Pool,
        tokens: &[u32],
        start_pos: usize,
        want_logits: &[usize],
        stats: &mut StepStats,
    ) -> Result<(), String> {
        let positions: Vec<usize> = (start_pos..start_pos + tokens.len()).collect();
        self.forward_rows(
            st8,
            &mut Caches::Shared(kv),
            pool,
            tokens,
            &positions,
            want_logits,
            stats,
        )
    }

    // nafs-aql: the general pass. Row `b` carries `tokens[b]` at `positions[b]` of the sequence whose cache
    // `caches` gives for `b`; every row shares one read of the weights.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_rows(
        &self,
        st8: &mut State,
        caches: &mut Caches,
        pool: &Pool,
        tokens: &[u32],
        positions: &[usize],
        want_logits: &[usize],
        stats: &mut StepStats,
    ) -> Result<(), String> {
        let c = self.cfg;
        let nb = tokens.len();
        if nb == 0 {
            return Ok(());
        }
        // nafs-aql: the batch width is decided per step when users are served together, so an oversized batch is
        // a request to refuse, not a slice to overrun.
        if nb > st8.max_batch {
            return Err(format!(
                "batch of {nb} rows exceeds the scratch of {}",
                st8.max_batch
            ));
        }
        if positions.len() != nb || caches.rows().is_some_and(|r| r != nb) {
            return Err(format!(
                "batch of {nb} tokens with {} positions and {:?} caches",
                positions.len(),
                caches.rows()
            ));
        }
        let t_fwd = Instant::now();
        for b in 0..nb {
            let kv = caches.get(b);
            if positions[b] >= kv.max_ctx() {
                return Err(format!(
                    "context limit reached: position {} requested, cache holds {}",
                    positions[b],
                    kv.max_ctx()
                ));
            }
        }
        for (i, &t) in tokens.iter().enumerate() {
            if t as usize >= c.n_vocab {
                return Err(format!(
                    "token {t} at batch index {i} is outside the vocabulary"
                ));
            }
            self.embed(t, &mut st8.x[i * c.d_model..(i + 1) * c.d_model], stats);
        }

        let kvd = c.kv_dim();
        for (li, l) in self.layers.iter().enumerate() {
            // ---- attention ----
            self.norm_rows(st8, &l.attn_norm, nb, pool, stats);
            {
                // Q, K and V share one activation: one fork–join, not three.
                let acts = &st8.acts[..nb];
                let jobs = [
                    MatJob {
                        w: &l.wq,
                        out: OutPtr(st8.q.as_mut_ptr()),
                    },
                    MatJob {
                        w: &l.wk,
                        out: OutPtr(st8.k.as_mut_ptr()),
                    },
                    MatJob {
                        w: &l.wv,
                        out: OutPtr(st8.v.as_mut_ptr()),
                    },
                ];
                self.matmul_group(&jobs, acts, nb, pool, stats);
            }
            let qd = c.q_dim();
            for (buf, bias, width) in [
                (&mut st8.q, &l.bq, qd),
                (&mut st8.k, &l.bk, kvd),
                (&mut st8.v, &l.bv, kvd),
            ] {
                if let Some(bv) = bias {
                    for b in 0..nb {
                        for (x, y) in buf[b * width..(b + 1) * width].iter_mut().zip(bv) {
                            *x += *y;
                        }
                    }
                    stats.bytes_weights += (width * 4) as u64;
                }
            }

            // Qwen3: RMSNorm each head of the queries and keys before rotating them.
            if let (Some(qn), Some(kn)) = (&l.q_norm, &l.k_norm) {
                let hd = c.head_dim;
                for row in st8.q[..nb * qd].chunks_exact_mut(hd) {
                    head_rmsnorm(row, qn, c.eps);
                }
                for row in st8.k[..nb * kvd].chunks_exact_mut(hd) {
                    head_rmsnorm(row, kn, c.eps);
                }
                stats.bytes_weights += (2 * hd * 4) as u64;
            }

            for b in 0..nb {
                let pos = positions[b];
                self.rope(&st8.rope, &mut st8.q[b * qd..(b + 1) * qd], c.n_head, pos);
                self.rope(
                    &st8.rope,
                    &mut st8.k[b * kvd..(b + 1) * kvd],
                    c.n_kv_head,
                    pos,
                );
                let kv = caches.get_mut(b);
                for h in 0..c.n_kv_head {
                    let ks = b * kvd + h * c.head_dim;
                    kv.put(
                        li,
                        h,
                        pos,
                        &st8.k[ks..ks + c.head_dim],
                        &st8.v[ks..ks + c.head_dim],
                    );
                }
                // This position is now readable, so a later row of the batch can attend to an earlier one of the
                // same sequence — which is what makes prefill and speculative verification a single call.
                kv.advance(pos + 1);
            }

            let t_attn = Instant::now();
            self.attend(st8, caches, pool, li, nb, positions, stats);
            stats.ns_attn += t_attn.elapsed().as_nanos() as u64;

            for b in 0..nb {
                st8.acts[b].fill(&st8.xb[b * qd..(b + 1) * qd]);
            }
            let acts = &st8.acts[..nb];
            self.matmul(&l.wo, acts, nb, &mut st8.xb2[..nb * c.d_model], pool, stats);
            for i in 0..nb * c.d_model {
                st8.x[i] += st8.xb2[i];
            }

            // ---- feed-forward ----
            self.norm_rows(st8, &l.ffn_norm, nb, pool, stats);
            {
                let acts = &st8.acts[..nb];
                let jobs = [
                    MatJob {
                        w: &l.gate,
                        out: OutPtr(st8.hb.as_mut_ptr()),
                    },
                    MatJob {
                        w: &l.up,
                        out: OutPtr(st8.hb2.as_mut_ptr()),
                    },
                ];
                self.matmul_group(&jobs, acts, nb, pool, stats);
            }
            // SwiGLU: silu(gate) ⊙ up, quantised for the down projection.
            self.swiglu_rows(st8, nb, pool, stats);
            let acts = &st8.acts[..nb];
            self.matmul(
                &l.down,
                acts,
                nb,
                &mut st8.xb2[..nb * c.d_model],
                pool,
                stats,
            );
            for i in 0..nb * c.d_model {
                st8.x[i] += st8.xb2[i];
            }
            if let Some(t) = st8.tap_layers.iter().position(|&x| x == li) {
                let (d, nt) = (c.d_model, st8.tap_layers.len());
                for b in 0..nb {
                    let dst = (b * nt + t) * d;
                    st8.tap[dst..dst + d].copy_from_slice(&st8.x[b * d..(b + 1) * d]);
                }
            }
        }

        // ---- output head ----
        if want_logits.is_empty() {
            stats.ns_total += t_fwd.elapsed().as_nanos() as u64;
            return Ok(());
        }
        let head = self.output.as_ref().unwrap_or(&self.tok_embd);
        let nl = want_logits.len();
        if nl > st8.acts.len() || nl * c.n_vocab > st8.logits.len() {
            return Err(format!(
                "logits requested for {nl} rows; scratch holds {} activations and {} logit rows",
                st8.acts.len(),
                st8.logits.len() / c.n_vocab
            ));
        }
        for (i, &b) in want_logits.iter().enumerate() {
            if b >= nb {
                return Err(format!("logits requested for row {b} of a {nb}-row batch"));
            }
            let (xs, xe) = (b * c.d_model, (b + 1) * c.d_model);
            let (xv, xbv) = (&st8.x, &mut st8.xb);
            self.rmsnorm(&xv[xs..xe], &self.out_norm, &mut xbv[xs..xe], stats);
            st8.acts[i].fill(&st8.xb[xs..xe]);
        }
        // The output head is 50 MB — a quarter of all traffic for this model. One
        // batched call reads it once for every row; a loop would read it `nl` times,
        // which would make speculation cost more bandwidth than it saves.
        let before = stats.ns_matmul;
        self.matmul(
            head,
            &st8.acts[..nl],
            nl,
            &mut st8.logits[..nl * c.n_vocab],
            pool,
            stats,
        );
        stats.ns_head += stats.ns_matmul - before;
        stats.ns_matmul = before;
        stats.ns_total += t_fwd.elapsed().as_nanos() as u64;
        Ok(())
    }

    /// Attention for one layer over a batch: scores, softmax and the value mix for
    /// every `(batch row, query head)` pair, written into `st8.xb`.
    ///
    /// Split across the pool by pair when there is enough work to pay for the
    /// fork–join. It used to run on the calling thread alone while three cores idled,
    /// which is the term that grows with context and the one Amdahl's law punishes.
    #[allow(clippy::too_many_arguments)]
    fn attend(
        &self,
        st8: &mut State,
        caches: &Caches,
        pool: &Pool,
        li: usize,
        nb: usize,
        positions: &[usize],
        stats: &mut StepStats,
    ) {
        let c = self.cfg;
        let scale = 1.0 / (c.head_dim as f32).sqrt();
        let tasks = nb * c.n_head;
        // Work in multiply-accumulates: 2 passes over `n_ctx · head_dim` per task.
        let avg_ctx = positions.iter().sum::<usize>() / nb + 1;
        let work = tasks * avg_ctx * c.head_dim * 2;
        // Below this a fork–join (~20 µs) costs more than it saves.
        let parallel = pool.width() > 1 && work >= 1 << 19;
        let stride = st8.att_stride;
        let q: &[f32] = &st8.q;
        let att = SharedBuf(st8.att.as_mut_ptr());
        let out = SharedBuf(st8.xb.as_mut_ptr());
        let body = |wk: usize, nw: usize| {
            let (lo, hi) = share(tasks, wk, nw);
            // SAFETY: worker `wk` owns row `wk` of the scratch, and the output slice of
            // task `t` is `(t / n_head, t % n_head)`'s head, which no other task writes.
            let att = unsafe { att.slice(wk * stride, stride) };
            for t in lo..hi {
                let (b, h) = (t / c.n_head, t % c.n_head);
                let n_ctx = positions[b] + 1;
                let kv = caches.get(b);
                let kvh = h / c.gqa();
                let qo = b * c.q_dim() + h * c.head_dim;
                kv.scores(li, kvh, n_ctx, &q[qo..qo + c.head_dim], scale, att);
                // Softmax in place, max-shifted.
                let m = att[..n_ctx]
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for e in att[..n_ctx].iter_mut() {
                    *e = (*e - m).exp();
                    sum += *e;
                }
                let inv = 1.0 / sum;
                for e in att[..n_ctx].iter_mut() {
                    *e *= inv;
                }
                let o = unsafe { out.slice(qo, c.head_dim) };
                o.fill(0.0);
                kv.mix(li, kvh, n_ctx, att, o);
            }
        };
        if parallel {
            pool.run(body);
            stats.barriers += 1;
        } else {
            body(0, 1);
        }
        for b in 0..nb {
            let n_ctx = (positions[b] + 1) as u64;
            // Cache traffic for this row, *for this layer*: every position, K and V.
            // `bytes_read_at` covers all layers at once and would count the whole
            // cache 32 times over from inside the layer loop.
            stats.bytes_kv += caches.get(b).bytes_per_position_per_layer() * n_ctx;
            stats.macs += (c.n_head * c.head_dim) as u64 * n_ctx * 2;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_stats_intensity_is_flops_over_bytes() {
        let s = StepStats {
            bytes_weights: 100,
            bytes_kv: 100,
            macs: 400,
            barriers: 3,
            ..Default::default()
        };
        assert_eq!(s.bytes_total(), 200);
        // 2 FLOP per MAC, so 800 FLOP over 200 bytes.
        assert_eq!(s.arithmetic_intensity(), 4.0);
        // An empty step must not divide by zero.
        assert_eq!(StepStats::default().arithmetic_intensity(), 0.0);
    }

    #[test]
    fn step_stats_accumulate() {
        let mut a = StepStats {
            bytes_weights: 1,
            bytes_kv: 2,
            macs: 3,
            barriers: 4,
            ..Default::default()
        };
        a.add(&StepStats {
            bytes_weights: 10,
            bytes_kv: 20,
            macs: 30,
            barriers: 40,
            ..Default::default()
        });
        assert_eq!(
            (a.bytes_weights, a.bytes_kv, a.macs, a.barriers),
            (11, 22, 33, 44)
        );
    }

    #[test]
    fn gqa_ratio_is_derived_not_assumed() {
        let c = Config {
            n_layer: 32,
            n_head: 15,
            n_kv_head: 5,
            head_dim: 64,
            d_model: 960,
            n_ff: 2560,
            n_vocab: 49152,
            rope_dim: 64,
            eps: 1e-5,
            rope_base: 1e5,
            ctx_train: 8192,
            arch: Arch::Llama,
            rope_neox: false,
        };
        assert_eq!(c.gqa(), 3);
        assert_eq!(c.kv_dim(), 320);
        // Query head 14 must land on kv head 4, not 14.
        assert_eq!(14 / c.gqa(), 4);
    }
}

// ------------------------------------------------------------- the kept model --

/// The loaded model a host may keep for the next session on the same file.
///
/// A `Model` is a mapping plus the tensor directory: dropping it unmaps the weights,
/// and the next `Session::open` on the same file maps them again and minor-faults
/// every page on the first token (2.5 GB for Qwen3 4B). The pages were never gone —
/// the page cache had them — but the mapping was, and so were the TLB and the
/// faulted-in state. The host keeps one `Arc<Model>` here across a switch to the
/// *same* model (`residency.rs` says when: a release whose next session is this
/// model), and lets it go on a switch to another, so the rule that the previous
/// model's memory comes back before the next loads still holds.
pub mod keep {
    use super::Model;
    use crate::gguf::Gguf;
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, Mutex, OnceLock};
    use std::time::{Duration, Instant};

    /// How long an unclaimed model is kept before a `sweep` drops it.
    pub const TTL: Duration = Duration::from_secs(600);

    struct Slot {
        key: (PathBuf, u64, u64),
        model: Arc<Model>,
        since: Instant,
    }

    fn slot() -> &'static Mutex<Option<Slot>> {
        static S: OnceLock<Mutex<Option<Slot>>> = OnceLock::new();
        S.get_or_init(|| Mutex::new(None))
    }

    fn key(path: &Path) -> Result<(PathBuf, u64, u64), String> {
        let m = std::fs::metadata(path).map_err(|e| format!("cannot stat {path:?}: {e}"))?;
        let mtime = m
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);
        Ok((
            std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf()),
            m.len(),
            mtime,
        ))
    }

    /// The model at `path`: the kept one when it is the same file (same canonical
    /// path, size and modification time), else freshly loaded — and kept in place of
    /// whatever was there. `Err` is `Gguf::open`/`Model::load`'s.
    pub fn open(path: &Path) -> Result<Arc<Model>, String> {
        let k = key(path)?;
        let mut g = slot().lock().unwrap_or_else(|e| e.into_inner());
        if let Some(s) = g.as_mut() {
            if s.key == k {
                s.since = Instant::now();
                return Ok(Arc::clone(&s.model));
            }
        }
        let model = Arc::new(Model::load(Gguf::open(path)?)?);
        *g = Some(Slot {
            key: k,
            model: Arc::clone(&model),
            since: Instant::now(),
        });
        Ok(model)
    }

    /// Whether a load of `path` would find the kept model.
    pub fn has(path: &Path) -> bool {
        let Ok(k) = key(path) else { return false };
        slot()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .is_some_and(|s| s.key == k)
    }

    /// The kept model's path and how many sessions still hold it (besides the slot).
    pub fn status() -> Option<(PathBuf, usize, f64)> {
        slot()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .map(|s| {
                (
                    s.key.0.clone(),
                    Arc::strong_count(&s.model) - 1,
                    s.since.elapsed().as_secs_f64(),
                )
            })
    }

    /// Let the kept model go. Its mapping ends when the last session using it ends.
    pub fn drop_kept() -> bool {
        slot()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
            .is_some()
    }

    /// Drop the kept model when nothing uses it and it has sat for `TTL`.
    pub fn sweep() -> bool {
        let mut g = slot().lock().unwrap_or_else(|e| e.into_inner());
        if g.as_ref()
            .is_some_and(|s| Arc::strong_count(&s.model) == 1 && s.since.elapsed() >= TTL)
        {
            *g = None;
            return true;
        }
        false
    }
}
