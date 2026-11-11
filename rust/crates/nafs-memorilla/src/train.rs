//! Training a memory module on a knowledge base (Phase 6, FR-K42), as `train.py` does: the decoder and encoder frozen,
//! the module learns from the next-token loss on the answer tokens only, through the decoder.
//!
//! Each example: the module reads the base's passage vectors with the question vector and returns K memory tokens;
//! they replace the embeddings of the K placeholders of the prompt (Memorilla's system prompt, built by the same
//! `chat_prompt` the server uses, thinking off); the decoder runs over prompt + answer; cross-entropy on the answer and
//! what closes the assistant turn in the template (`<|im_end|>` and the newline after it, as the reference's labels
//! have), averaged over the answer tokens of an accumulation group, not per example. The reference's from-scratch
//! recipe (`configs/factkg.yaml`, `train.py`): AdamW (weight decay 0.01) at 1e-4, 16 examples per optimiser step, 5
//! epochs, linear warm-up from (almost) zero over the first 10 % of the steps then cosine to 10 % of the peak,
//! global-norm clipping at 1.0, dropout 0.1 in the module. `retrieval_init` from scratch (the paper's FactKG recipe;
//! no pre-trained module exists for Qwen3-1.7B), or a checkpoint to fine-tune, whose own K and widths then decide the
//! prompt. From scratch the memory projection starts at zero with the placeholder token's own embedding as its bias,
//! so the untrained module hands the decoder exactly the prompt it would read without memory.
//!
//! Phase 26, the collapse guard. The first module trained here (3e-4, 4 examples a step, 3 epochs, no dropout) ended
//! as one vector at all 16 slots for every question: the validation loss fell from 8.77 to 1.46 while claim accuracy
//! stayed where it was without memory, and `best/` was chosen by that loss. Every validation now also measures the
//! memories of the validation questions (`memory::collapse_stats`) and the claim accuracy, and `best/` is the
//! checkpoint that ranks first by: its memory differs between questions; then claim accuracy with memory minus
//! without (when the validation claims are of both kinds); then validation loss. Each checkpoint carries a
//! `metrics.json` with `collapsed` and the numbers, which the gateway reads before it serves the module.
//!
//! Progress goes to `<out>/progress.jsonl`; a file `<out>/stop` ends the run after the current step; every epoch is
//! saved under `<out>/epoch-NN/`, the best one (as above) under `<out>/best/`.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::{DType, Device, IndexOp, Tensor, Var};
use candle_nn::{VarBuilder, VarMap};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::decoder::Decoder;
use crate::memory::{
    collapse_stats, CheckpointMetrics, CollapseStats, Dropout, MemoryConfig, MemoryModule,
};

/// The reference's from-scratch recipe (`third_party/memorilla/configs/factkg.yaml`): the trainer's defaults.
pub const DEFAULT_LR: f64 = 1e-4;
/// Examples per optimiser step (the reference: batch 2 x accumulation 8).
pub const DEFAULT_ACCUMULATION: usize = 16;
pub const DEFAULT_EPOCHS: usize = 5;
/// The share of the optimiser steps spent warming up.
pub const WARMUP_RATIO: f64 = 0.1;
/// Where the warm-up starts, as a share of the peak (`LinearLR(start_factor=1e-8)`).
const WARMUP_START_FACTOR: f64 = 1e-8;

/// Memorilla's system prompt for every training and serving prompt (FR-K23).
pub const SYSTEM_PROMPT: &str = "You are a helpful assistant.";
pub const PLACEHOLDER: &str = "<|fim_pad|>";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainOpts {
    pub data: PathBuf,
    pub decoder: PathBuf,
    /// The GGUF the server serves: its tokenizer and chat template make the prompts.
    pub gguf: PathBuf,
    pub out: PathBuf,
    pub epochs: usize,
    pub lr: f64,
    pub accumulation: usize,
    /// Stop after this many training examples in all (0 = no limit), for a cost measurement.
    pub max_examples: usize,
    /// A checkpoint to fine-tune; `None` trains a `retrieval_init` module from scratch.
    pub init: Option<PathBuf>,
    pub num_memories: usize,
    pub seed: u64,
    /// The token the memory positions carry. Must be the one the server puts there (`[knowledge]
    /// memory_placeholder`): the prompt around the memory block has to match between training and serving.
    #[serde(default = "default_placeholder")]
    pub placeholder: String,
    /// Dropout probability while training; `None` takes the module's own (`config.json`, 0.1 as the reference).
    #[serde(default)]
    pub dropout: Option<f64>,
}

fn default_placeholder() -> String {
    PLACEHOLDER.to_string()
}

#[derive(Debug, Deserialize)]
struct Row {
    kind: String,
    question: String,
    answer: String,
    split: String,
}

#[derive(Debug, Deserialize)]
struct DatasetFile {
    kb_name: String,
    dims: usize,
    rows: Vec<Row>,
}

/// One tokenised example: prompt ids with the placeholder run at `mem_at..mem_at+K`, then the answer ids.
struct Example {
    ids: Vec<u32>,
    mem_at: usize,
    answer_from: usize,
    question: usize,
    kind: String,
}

fn load_matrix(path: &Path) -> Result<Tensor, String> {
    let t = candle_core::safetensors::load(path, &Device::Cpu)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    t.get("embeddings")
        .cloned()
        .ok_or_else(|| format!("{}: no `embeddings` tensor", path.display()))
}

/// The warm-up steps of a run of `total` optimiser steps (`int(warmup_ratio * total)`; none for a very short run).
pub fn warmup_steps(total: usize) -> usize {
    (WARMUP_RATIO * total as f64) as usize
}

/// The learning rate of optimiser step `step` (from 0): linear from almost zero to the peak over the first `warm`
/// steps, then cosine from the peak to 10 % of it at `total`. As the reference's `LinearLR` then `CosineAnnealingLR`.
pub fn lr_at(step: usize, total: usize, warm: usize, peak: f64) -> f64 {
    let warm = warm.min(total);
    if step < warm {
        let f = WARMUP_START_FACTOR + (1.0 - WARMUP_START_FACTOR) * step as f64 / warm as f64;
        return peak * f;
    }
    let p = (step - warm) as f64 / (total - warm).max(1) as f64;
    let min = 0.1 * peak;
    min + (peak - min) * 0.5 * (1.0 + (std::f64::consts::PI * p.min(1.0)).cos())
}

/// AdamW (decoupled weight decay 0.01, betas 0.9 / 0.999, eps 1e-8) over the module's variables, from gradients the
/// caller has accumulated (`None`: no gradient this step).
struct Adam {
    m: Vec<Tensor>,
    v: Vec<Tensor>,
    t: i32,
}

impl Adam {
    fn new(vars: &[Var]) -> candle_core::Result<Adam> {
        let z = |v: &Var| v.as_tensor().zeros_like();
        Ok(Adam {
            m: vars.iter().map(z).collect::<candle_core::Result<_>>()?,
            v: vars.iter().map(z).collect::<candle_core::Result<_>>()?,
            t: 0,
        })
    }

    fn step(
        &mut self,
        vars: &[Var],
        grads: &[Option<Tensor>],
        lr: f64,
        scale: f64,
    ) -> candle_core::Result<()> {
        let (b1, b2, eps, wd) = (0.9f64, 0.999f64, 1e-8, 0.01);
        self.t += 1;
        let (c1, c2) = (1.0 - b1.powi(self.t), 1.0 - b2.powi(self.t));
        for (k, v) in vars.iter().enumerate() {
            let Some(g) = &grads[k] else { continue };
            let g = (g * scale)?;
            self.m[k] = ((&self.m[k] * b1)? + (&g * (1.0 - b1))?)?;
            self.v[k] = ((&self.v[k] * b2)? + (g.sqr()? * (1.0 - b2))?)?;
            let upd = ((&self.m[k] / c1)? / ((&self.v[k] / c2)?.sqrt()? + eps)?)?;
            let theta = v.as_tensor();
            let next = ((theta * (1.0 - lr * wd))? - (upd * lr)?)?;
            v.set(&next)?;
        }
        Ok(())
    }
}

pub struct Trainer {
    /// The ids of "True" and "False" (claim accuracy), and the placeholder's embedding (the no-memory baseline).
    true_false: (u32, u32),
    plain: Tensor,
    decoder: Decoder,
    module: MemoryModule,
    vars: VarMap,
    docs: Tensor,
    questions: Tensor,
    train: Vec<Example>,
    val: Vec<Example>,
    opts: TrainOpts,
    kb_name: String,
    /// True and false claims in the dataset: all-true claims make `claim_accuracy` meaningless.
    claims: (usize, usize),
    /// Training-time dropout in the module (never in validation).
    dropout: Dropout,
}

/// One validation pass.
struct Eval {
    /// Mean loss per answer token.
    val_loss: f64,
    /// Share of the validation claims answered right with the module's memory, and how many there are.
    claim_accuracy: f64,
    claims: usize,
    stats: CollapseStats,
}

/// Whether checkpoint `new` should replace `old` as `best/`: a memory that differs between questions first, then the
/// larger gain in claim accuracy over no memory (when both were measured), then the lower validation loss.
pub fn better(new: &CheckpointMetrics, old: &CheckpointMetrics) -> bool {
    if new.collapsed != old.collapsed {
        return !new.collapsed;
    }
    let gain = |m: &CheckpointMetrics| Some(m.claim_accuracy? - m.claim_accuracy_without_memory?);
    if let (Some(a), Some(b)) = (gain(new), gain(old)) {
        if (a - b).abs() > 1e-9 {
            return a > b;
        }
    }
    match (new.val_loss, old.val_loss) {
        (Some(a), Some(b)) => a < b,
        (Some(_), None) => true,
        _ => false,
    }
}

impl Trainer {
    pub fn new(opts: TrainOpts) -> Result<Trainer, String> {
        let t0 = Instant::now();
        let file: DatasetFile = serde_json::from_slice(
            &std::fs::read(opts.data.join("dataset.json"))
                .map_err(|e| format!("dataset.json: {e}"))?,
        )
        .map_err(|e| format!("dataset.json: {e}"))?;
        let docs = load_matrix(&opts.data.join("documents.safetensors"))?
            .unsqueeze(0)
            .map_err(|e| e.to_string())?;
        let questions = load_matrix(&opts.data.join("questions.safetensors"))?;
        let g = smiley_core::gguf::Gguf::open(&opts.gguf)?;
        let tk = smiley_core::tokenizer::Tokenizer::from_gguf(&g)?;
        let template = smiley_core::template::Template::detect(&g)?;
        let ph = tk
            .id_of(&opts.placeholder)
            .ok_or("the placeholder is not a token of this model")?;
        // What closes an assistant message in this template: `<|im_end|>` and a newline for Qwen3 / ChatML, exactly
        // what the reference's labels supervise after the answer (`apply_chat_template(add_generation_prompt=False)`)
        // and what the server writes after an earlier answer of a conversation. The server stops at the end-of-turn
        // token, so the newline never changes what is generated; it keeps the loss per token the reference's.
        let mut close = tk.encode(&template.assistant_turn(""));
        if close.is_empty() {
            close.push(tk.id_of("<|im_end|>").unwrap_or(tk.eos));
        }
        let decoder = Decoder::load(&opts.decoder)?;
        // The module decides how many memory positions the prompt has. When fine-tuning, that is the checkpoint's
        // own K, not the command line's: a mismatch would train against a prompt the server never sends (K smaller)
        // or slice past the end of an example (K larger).
        let cfg = match &opts.init {
            Some(dir) => serde_json::from_slice::<MemoryConfig>(
                &std::fs::read(dir.join("config.json")).map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?,
            None => MemoryConfig {
                retrieval_init: true,
                num_memories: opts.num_memories,
                ..MemoryConfig::paper(file.dims, decoder.cfg.hidden_size)
            },
        };
        if cfg.embedding_dim != file.dims {
            return Err(format!(
                "the module reads {}-wide vectors; this knowledge base's are {}-wide",
                cfg.embedding_dim, file.dims
            ));
        }
        if cfg.output_dim != decoder.cfg.hidden_size {
            return Err(format!(
                "the module writes {}-wide memory tokens; {} takes {}",
                cfg.output_dim,
                opts.decoder.display(),
                decoder.cfg.hidden_size
            ));
        }
        let k = cfg.num_memories;
        let system = format!(
            "{SYSTEM_PROMPT}\nMemory{}Memory",
            opts.placeholder.repeat(k)
        );
        let mut train = Vec::new();
        let mut val = Vec::new();
        // SR-K05: a passage mentioning the placeholder token (an ML knowledge base easily does) would add memory
        // positions the module does not fill. The server cleans the same text at serve time, so cleaning it here
        // keeps the two prompts identical as well as keeping the run alive.
        let clean = |t: &str| t.replace(&opts.placeholder, "[memory placeholder]");
        let mut claims = (0usize, 0usize);
        for (i, r) in file.rows.iter().enumerate() {
            let (question, answer) = (clean(&r.question), clean(&r.answer));
            if r.kind == "claim" {
                match answer.trim() {
                    "True" => claims.0 += 1,
                    _ => claims.1 += 1,
                }
            }
            let prompt = smiley_core::serve::chat_prompt(
                template,
                Some(&system),
                &[],
                &question,
                false,
                None,
            );
            let mut ids = tk.encode(&prompt);
            let at: Vec<usize> = ids
                .iter()
                .enumerate()
                .filter(|(_, t)| **t == ph)
                .map(|(j, _)| j)
                .collect();
            if at.len() != k || at.windows(2).any(|w| w[1] != w[0] + 1) {
                return Err(format!(
                    "row {i}: {} placeholder tokens, expected {k} in a row",
                    at.len()
                ));
            }
            let answer_from = ids.len();
            ids.extend(tk.encode(&answer));
            ids.extend_from_slice(&close);
            let ex = Example {
                ids,
                mem_at: at[0],
                answer_from,
                question: i,
                kind: r.kind.clone(),
            };
            if r.split == "validation" {
                val.push(ex);
            } else {
                train.push(ex);
            }
        }
        // The module: from a checkpoint, or retrieval_init from scratch (its config was read above).
        let mut vars = VarMap::new();
        let module = MemoryModule::new(
            cfg,
            VarBuilder::from_varmap(&vars, DType::F32, &Device::Cpu),
        )
        .map_err(|e| e.to_string())?;
        match &opts.init {
            Some(dir) => {
                vars.load(dir.join("weights.safetensors"))
                    .map_err(|e| e.to_string())?;
            }
            None => {
                // Zero weights and the placeholder token's own embedding as bias: the untrained module returns exactly
                // the placeholder embeddings, so training starts from the plain prompt (the model reads ordinary
                // tokens there) with gradients of ordinary size. Measured in Phase 6: random vectors at the embedding
                // scale derailed the 1.7B model (first check loss 14.7 nats), and exact zeros put every memory row at
                // the RMS norm's singular point (gradient norm around 1e7).
                let w = Tensor::zeros(
                    (cfg.output_dim, cfg.embedding_dim),
                    DType::F32,
                    &Device::Cpu,
                )
                .map_err(|e| e.to_string())?;
                vars.set_one("memory_projection.weight", w)
                    .map_err(|e| e.to_string())?;
                let b = decoder
                    .embed_ids(&[ph])
                    .and_then(|t| t.squeeze(0))
                    .map_err(|e| e.to_string())?;
                vars.set_one("memory_projection.bias", b)
                    .map_err(|e| e.to_string())?;
            }
        }
        eprintln!(
            "loaded in {:.0} s: {} train / {} validation examples over {} passages; module {} parameters",
            t0.elapsed().as_secs_f64(),
            train.len(),
            val.len(),
            docs.dim(1).unwrap_or(0),
            cfg.parameters()
        );
        let tf = (tk.encode("True"), tk.encode("False"));
        let true_false = match (tf.0.as_slice(), tf.1.as_slice()) {
            ([t], [f]) => (*t, *f),
            _ => return Err("\"True\" or \"False\" is not one token for this tokenizer".into()),
        };
        let plain = decoder.embed_ids(&vec![ph; k]).map_err(|e| e.to_string())?;
        let dropout = Dropout::new(opts.dropout.unwrap_or(cfg.dropout), opts.seed ^ 0xD80F);
        Ok(Trainer {
            dropout,
            true_false,
            plain,
            decoder,
            module,
            vars,
            docs,
            questions,
            train,
            val,
            opts,
            kb_name: file.kb_name,
            claims,
        })
    }

    /// The memory tokens [K, H] the module gives for an example's question (no dropout: validation and serving).
    fn memory(&self, ex: &Example) -> candle_core::Result<Tensor> {
        let q = self.questions.i(ex.question)?.unsqueeze(0)?;
        self.module.forward(&self.docs, None, &q)?.squeeze(0)
    }

    /// The same with dropout: what a training step learns through.
    fn memory_train(&self, ex: &Example) -> candle_core::Result<Tensor> {
        let q = self.questions.i(ex.question)?.unsqueeze(0)?;
        self.module
            .forward_train(&self.docs, None, &q, &self.dropout)?
            .squeeze(0)
    }

    /// Logits [n, V] at the positions that predict an example's `n` target tokens (the answer and the turn's close),
    /// with the given memory tokens in place. Row 0 predicts the first answer token.
    fn target_logits(&self, ex: &Example, mem: &Tensor) -> candle_core::Result<Tensor> {
        let k = mem.dim(0)?;
        let before = self.decoder.embed_ids(&ex.ids[..ex.mem_at])?;
        let after = self.decoder.embed_ids(&ex.ids[ex.mem_at + k..])?;
        let x = Tensor::cat(&[&before, mem, &after], 0)?.unsqueeze(0)?;
        let h = self.decoder.hidden(&x)?.squeeze(0)?;
        // Position j predicts token j+1: the targets are predicted from answer_from-1 .. len-2.
        let n = ex.ids.len() - ex.answer_from;
        self.decoder.logits(&h.narrow(0, ex.answer_from - 1, n)?)
    }

    /// The **summed** cross-entropy over an example's target tokens (a scalar tensor in the autograd graph) and how
    /// many they are. Summed, so a group of examples is averaged over its tokens: the reference's loss is the mean
    /// over the supervised tokens of a batch, in which a long answer weighs more than "True".
    fn loss_sum(&self, ex: &Example, mem: &Tensor) -> candle_core::Result<(Tensor, usize)> {
        let logits = self.target_logits(ex, mem)?;
        let n = ex.ids.len() - ex.answer_from;
        let targets = Tensor::new(&ex.ids[ex.answer_from..], self.decoder.device())?;
        Ok((
            (candle_nn::loss::cross_entropy(&logits, &targets)? * n as f64)?,
            n,
        ))
    }

    /// Whether the model prefers the right one of "True" / "False" as the first answer token of a claim.
    fn claim_right(&self, ex: &Example, first: &Tensor) -> candle_core::Result<bool> {
        let (t, f) = self.true_false;
        let l: Vec<f32> = first.to_vec1()?;
        Ok((l[t as usize] > l[f as usize]) == (ex.ids[ex.answer_from] == t))
    }

    /// Claim accuracy on the validation claims with the placeholders' own embeddings: the model without memory. It
    /// does not depend on the module, so it is measured once.
    fn claim_accuracy_without_memory(&self) -> candle_core::Result<f64> {
        let (mut hits, mut n) = (0usize, 0usize);
        for ex in self.val.iter().filter(|e| e.kind == "claim") {
            let first = self.target_logits(ex, &self.plain)?.i(0)?;
            hits += self.claim_right(ex, &first)? as usize;
            n += 1;
        }
        Ok(hits as f64 / n.max(1) as f64)
    }

    /// One pass over the validation examples: the loss per answer token, claim accuracy with the module's memory, and
    /// whether the memories of the validation questions differ. The memory is detached first: nothing before the
    /// decoder is tracked, so validation does not hold a 1.7B model's activations (about 1 GB) for numbers it only
    /// reads. The claim's answer is read off the logits the loss already needs.
    fn evaluate(&self) -> candle_core::Result<Eval> {
        let (mut sum, mut tokens) = (0.0f64, 0usize);
        let (mut hits, mut claims) = (0usize, 0usize);
        let mut memories: Vec<Vec<Vec<f32>>> = Vec::with_capacity(self.val.len());
        for ex in &self.val {
            let mem = self.memory(ex)?.detach();
            memories.push(mem.to_vec2()?);
            let logits = self.target_logits(ex, &mem)?;
            let n = ex.ids.len() - ex.answer_from;
            let targets = Tensor::new(&ex.ids[ex.answer_from..], self.decoder.device())?;
            sum += candle_nn::loss::cross_entropy(&logits, &targets)?.to_scalar::<f32>()? as f64
                * n as f64;
            tokens += n;
            if ex.kind == "claim" {
                hits += self.claim_right(ex, &logits.i(0)?)? as usize;
                claims += 1;
            }
        }
        Ok(Eval {
            val_loss: if tokens == 0 {
                f64::NAN
            } else {
                sum / tokens as f64
            },
            claim_accuracy: hits as f64 / claims.max(1) as f64,
            claims,
            stats: collapse_stats(&memories),
        })
    }

    /// What is written beside a checkpoint. Claim accuracy is left out when it cannot rank anything: no validation
    /// claims, or claims of one kind only (the model that always says "True" would score 1.0).
    fn metrics(&self, ev: &Eval, without: f64) -> CheckpointMetrics {
        let usable = ev.claims > 0 && self.claims.0 > 0 && self.claims.1 > 0;
        CheckpointMetrics {
            collapsed: ev.stats.collapsed(),
            stats: ev.stats,
            val_loss: ev.val_loss.is_finite().then_some(ev.val_loss),
            claim_accuracy: usable.then_some(ev.claim_accuracy),
            claim_accuracy_without_memory: usable.then_some(without),
            claims: ev.claims,
        }
    }

    /// A checkpoint and its metrics. The metrics go first, so a reader that finds new weights finds their verdict.
    fn save(&self, dir: &Path, m: &CheckpointMetrics) -> Result<(), String> {
        m.write(dir)?;
        MemoryModule::save(&self.vars, &self.module.cfg, dir)
    }

    fn log(&self, v: serde_json::Value) {
        eprintln!("{v}");
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.opts.out.join("progress.jsonl"))
        {
            let _ = writeln!(f, "{v}");
        }
    }

    pub fn run(&mut self) -> Result<f64, String> {
        std::fs::create_dir_all(&self.opts.out).map_err(|e| e.to_string())?;
        std::fs::write(
            self.opts.out.join("options.json"),
            serde_json::to_vec_pretty(&self.opts).unwrap_or_default(),
        )
        .map_err(|e| e.to_string())?;
        let acc = self.opts.accumulation.max(1);
        let per_epoch = self.train.len().div_ceil(acc);
        let mut total = per_epoch * self.opts.epochs;
        if self.opts.max_examples > 0 {
            total = total.min(self.opts.max_examples.div_ceil(acc));
        }
        let warm = warmup_steps(total);
        let vars: Vec<Var> = self.vars.all_vars();
        let mut adam = Adam::new(&vars).map_err(|e| e.to_string())?;
        let mut rng = self.opts.seed.max(1);
        let mut step = 0usize;
        let mut seen = 0usize;
        let mut best: Option<CheckpointMetrics> = None;
        let e0 = self.evaluate().map_err(|e| e.to_string())?;
        let v0 = e0.val_loss;
        let without = self
            .claim_accuracy_without_memory()
            .map_err(|e| e.to_string())?;
        self.log(json!({"event": "start", "claim_accuracy": e0.claim_accuracy, "claim_accuracy_without_memory": without, "claims": e0.claims,
            "claims_true": self.claims.0, "claims_false": self.claims.1, "kb_name": self.kb_name, "train": self.train.len(),
            "validation": self.val.len(), "steps": total, "warmup_steps": warm, "val_loss": v0, "lr": self.opts.lr,
            "accumulation": acc, "epochs": self.opts.epochs, "dropout": self.dropout.probability(),
            "slot_cosine": e0.stats.slot_cosine, "question_cosine": e0.stats.question_cosine, "memory_norm": e0.stats.mean_norm}));
        if self.claims.1 == 0 && self.claims.0 > 0 {
            self.log(json!({"event": "warning", "message": "every claim in this dataset is true (the graph had no second entity of a type to swap in), so claim accuracy only says how often the model answers True; the best checkpoint is chosen by validation loss among those whose memory differs between questions"}));
        }
        'epochs: for epoch in 0..self.opts.epochs {
            // A seeded shuffle per epoch.
            let mut order: Vec<usize> = (0..self.train.len()).collect();
            for i in (1..order.len()).rev() {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                order.swap(i, (rng % (i as u64 + 1)) as usize);
            }
            for batch in order.chunks(acc) {
                if step >= total || self.opts.out.join("stop").exists() {
                    break 'epochs;
                }
                let t = Instant::now();
                let lr = lr_at(step, total, warm, self.opts.lr);

                // One example's graph at a time (a 1.7B decoder's activations are about 1 GB per example). Only the
                // module's gradients are kept: Candle's GradStore holds one for every tensor of the graph.
                // The group's loss is the mean over its answer tokens, so each example's summed loss is divided by
                // the group's token count before its backward pass.
                let tokens: usize = batch
                    .iter()
                    .map(|&i| self.train[i].ids.len() - self.train[i].answer_from)
                    .sum();
                let mut acc: Vec<Option<Tensor>> = vec![None; vars.len()];
                let mut total_loss = 0.0f64;
                for &i in batch {
                    let ex = &self.train[i];
                    let mem = self.memory_train(ex).map_err(|e| e.to_string())?;
                    let (sum, _) = self.loss_sum(ex, &mem).map_err(|e| e.to_string())?;
                    let l = (sum / tokens.max(1) as f64).map_err(|e| e.to_string())?;
                    total_loss += l.to_scalar::<f32>().map_err(|e| e.to_string())? as f64;
                    let g = l.backward().map_err(|e| e.to_string())?;
                    for (k, v) in vars.iter().enumerate() {
                        if let Some(gi) = g.get(v) {
                            acc[k] = Some(match acc[k].take() {
                                Some(a) => (a + gi).map_err(|e| e.to_string())?,
                                None => gi.clone(),
                            });
                        }
                    }
                }
                // Global-norm clipping at 1.0.
                let mut sq = 0.0f64;
                for g in acc.iter().flatten() {
                    sq += g
                        .sqr()
                        .and_then(|x| x.sum_all())
                        .and_then(|x| x.to_scalar::<f32>())
                        .map_err(|e| e.to_string())? as f64;
                }
                let norm = sq.sqrt();
                if !norm.is_finite() {
                    return Err(format!("the gradient is not finite at step {}", step + 1));
                }
                let scale = if norm > 1.0 { 1.0 / norm } else { 1.0 };
                adam.step(&vars, &acc, lr, scale)
                    .map_err(|e| e.to_string())?;
                step += 1;
                seen += batch.len();
                let l = total_loss;
                let kinds: Vec<&str> = batch.iter().map(|&i| self.train[i].kind.as_str()).collect();
                self.log(json!({"event": "step", "epoch": epoch, "step": step, "of": total, "examples": seen, "loss": l, "lr": lr,
                    "grad_norm": norm, "tokens": tokens, "seconds": t.elapsed().as_secs_f64(), "seconds_per_example": t.elapsed().as_secs_f64() / batch.len() as f64, "kinds": kinds}));
            }
            let ev = self.evaluate().map_err(|e| e.to_string())?;
            let m = self.metrics(&ev, without);
            let dir = self.opts.out.join(format!("epoch-{epoch:02}"));
            self.save(&dir, &m)?;
            let is_best = m.val_loss.is_some() && best.as_ref().is_none_or(|b| better(&m, b));
            if is_best {
                self.save(&self.opts.out.join("best"), &m)?;
                best = Some(m.clone());
            }
            self.log(json!({"event": "epoch", "epoch": epoch, "val_loss": ev.val_loss, "best": best.as_ref().and_then(|b| b.val_loss),
                "is_best": is_best, "collapsed": m.collapsed, "slot_cosine": ev.stats.slot_cosine,
                "question_cosine": ev.stats.question_cosine, "memory_norm": ev.stats.mean_norm,
                "claim_accuracy": ev.claim_accuracy, "claim_accuracy_without_memory": without, "claims": ev.claims, "saved": dir}));
        }
        if step == 0 {
            // Stopped before the first step: the module is still its initialisation and must not be offered as a
            // checkpoint, or it would be activated and answer every question with memory that learnt nothing.
            self.log(json!({"event": "stopped", "step": 0}));
            self.log(json!({"event": "done", "steps": 0, "examples": 0, "val_loss": v0, "start_val_loss": v0,
                "error": "the run was stopped before it had trained on anything, so no module was saved"}));
            return Ok(v0);
        }
        let ev = self.evaluate().map_err(|e| e.to_string())?;
        let m = self.metrics(&ev, without);
        if step < total && self.opts.out.join("stop").exists() {
            self.save(&self.opts.out.join("stopped"), &m)?;
            self.log(json!({"event": "stopped", "step": step}));
        }
        // The module as it ends, always (a run cut short by --max-examples never reaches an epoch's end).
        self.save(&self.opts.out.join("last"), &m)?;
        if best.as_ref().is_none_or(|b| better(&m, b)) {
            self.save(&self.opts.out.join("best"), &m)?;
            best = Some(m.clone());
        }
        let b = best.unwrap_or(m);
        self.log(json!({"event": "done", "steps": step, "examples": seen, "val_loss": ev.val_loss, "start_val_loss": v0,
            "claim_accuracy": ev.claim_accuracy, "claim_accuracy_without_memory": without, "claims": ev.claims,
            "collapsed": ev.stats.collapsed(), "slot_cosine": ev.stats.slot_cosine, "question_cosine": ev.stats.question_cosine,
            "memory_norm": ev.stats.mean_norm,
            "best": {"collapsed": b.collapsed, "val_loss": b.val_loss, "claim_accuracy": b.claim_accuracy,
                "claim_accuracy_without_memory": b.claim_accuracy_without_memory, "slot_cosine": b.stats.slot_cosine,
                "question_cosine": b.stats.question_cosine, "memory_norm": b.stats.mean_norm}}));
        if b.collapsed {
            eprintln!("warning: the best checkpoint's memory is the same for every question (question cosine {:.6}); it will not be used to answer", b.stats.question_cosine);
        }
        Ok(ev.val_loss)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_warms_up_from_zero_then_decays_to_a_tenth() {
        // The reference: LinearLR(start_factor=1e-8) over int(0.1 * total) steps, then cosine to 10 % of the peak.
        assert_eq!(warmup_steps(100), 10);
        assert_eq!(warmup_steps(125), 12);
        assert_eq!(warmup_steps(5), 0);
        assert!(lr_at(0, 100, 10, 1.0) < 1e-7);
        assert!((lr_at(5, 100, 10, 1.0) - 0.5).abs() < 1e-7);
        assert!((lr_at(9, 100, 10, 1.0) - 0.9).abs() < 1e-7);
        assert!((lr_at(10, 100, 10, 1.0) - 1.0).abs() < 1e-9);
        assert!((lr_at(100, 100, 10, 1.0) - 0.1).abs() < 1e-9);
        assert!(lr_at(55, 100, 10, 1.0) < 0.6 && lr_at(55, 100, 10, 1.0) > 0.5);
        // No warm-up for a run too short to have one: cosine from the peak.
        assert!((lr_at(0, 5, 0, 1.0) - 1.0).abs() < 1e-9);
        // Monotone up, then monotone down.
        for s in 1..10 {
            assert!(lr_at(s, 100, 10, 1.0) > lr_at(s - 1, 100, 10, 1.0));
        }
        for s in 11..=100 {
            assert!(lr_at(s, 100, 10, 1.0) < lr_at(s - 1, 100, 10, 1.0));
        }
    }

    #[test]
    fn the_defaults_are_the_reference_recipe() {
        assert_eq!(DEFAULT_LR, 1e-4);
        assert_eq!(DEFAULT_ACCUMULATION, 16);
        assert_eq!(DEFAULT_EPOCHS, 5);
        assert_eq!(MemoryConfig::paper(8, 8).dropout, 0.1);
    }

    fn ckpt(collapsed: bool, with: Option<f64>, val: f64) -> CheckpointMetrics {
        CheckpointMetrics {
            collapsed,
            val_loss: Some(val),
            claim_accuracy: with,
            claim_accuracy_without_memory: with.map(|_| 0.55),
            claims: 20,
            ..Default::default()
        }
    }

    /// The Phase 20 failure: the collapsed checkpoint had the lowest validation loss and was kept as best.
    #[test]
    fn best_prefers_a_memory_that_differs_then_claim_gain_then_loss() {
        let collapsed_low_loss = ckpt(true, Some(0.55), 1.46);
        let alive_high_loss = ckpt(false, Some(0.60), 3.2);
        assert!(better(&alive_high_loss, &collapsed_low_loss));
        assert!(!better(&collapsed_low_loss, &alive_high_loss));
        // Among live ones: the larger gain over no memory wins, whatever the loss.
        assert!(better(
            &ckpt(false, Some(0.70), 3.0),
            &ckpt(false, Some(0.60), 2.0)
        ));
        assert!(!better(
            &ckpt(false, Some(0.60), 2.0),
            &ckpt(false, Some(0.70), 3.0)
        ));
        // The same gain, or none measured: the lower loss.
        assert!(better(
            &ckpt(false, Some(0.60), 1.9),
            &ckpt(false, Some(0.60), 2.0)
        ));
        assert!(better(&ckpt(false, None, 1.9), &ckpt(false, None, 2.0)));
        assert!(!better(&ckpt(false, None, 2.0), &ckpt(false, None, 2.0)));
        // Among collapsed ones the order is the same; the gateway refuses them all.
        assert!(better(&ckpt(true, None, 1.0), &ckpt(true, None, 2.0)));
    }
}
