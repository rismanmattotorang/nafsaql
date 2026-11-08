//! Training a memory module on a knowledge base (Phase 6, FR-K42), as `train.py` does: the decoder and encoder frozen,
//! the module learns from the next-token loss on the answer tokens only, through the decoder.
//!
//! Each example: the module reads the base's passage vectors with the question vector and returns K memory tokens;
//! they replace the embeddings of the K placeholders of the prompt (Memorilla's system prompt, built by the same
//! `chat_prompt` the server uses, thinking off); the decoder runs over prompt + answer; cross-entropy on the answer and
//! its end-of-turn token. AdamW (weight decay 0.01), linear warm-up over 10 % of the steps then cosine to 10 % of the
//! peak, global-norm clipping at 1.0, gradient accumulation. `retrieval_init` from scratch (the paper's FactKG recipe;
//! no pre-trained module exists for Qwen3-1.7B), or a checkpoint to fine-tune, whose own K and widths then decide the
//! prompt. From scratch the memory projection starts at zero with the placeholder token's own embedding as its bias,
//! so the untrained module hands the decoder exactly the prompt it would read without memory.
//!
//! Progress goes to `<out>/progress.jsonl`; a file `<out>/stop` ends the run after the current step; every epoch is
//! saved under `<out>/epoch-NN/`, the lowest validation loss under `<out>/best/`.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::{DType, Device, IndexOp, Tensor, Var};
use candle_nn::{VarBuilder, VarMap};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::decoder::Decoder;
use crate::memory::{MemoryConfig, MemoryModule};

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

/// Linear warm-up over the first `warm` steps, then cosine from the peak to 10 % of it.
pub fn lr_at(step: usize, total: usize, warm: usize, peak: f64) -> f64 {
    let warm = warm.clamp(1, total.max(1));
    if step < warm {
        return peak * (step + 1) as f64 / warm as f64;
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
        let end = tk.id_of("<|im_end|>").unwrap_or(tk.eos);
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
            ids.push(end);
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
        Ok(Trainer {
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

    /// The memory tokens [K, H] the module gives for an example's question.
    fn memory(&self, ex: &Example) -> candle_core::Result<Tensor> {
        let q = self.questions.i(ex.question)?.unsqueeze(0)?;
        self.module.forward(&self.docs, None, &q)?.squeeze(0)
    }

    /// Final hidden states [T, H] of an example with the given memory tokens in place.
    fn hidden_with(&self, ex: &Example, mem: &Tensor) -> candle_core::Result<Tensor> {
        let k = mem.dim(0)?;
        let before = self.decoder.embed_ids(&ex.ids[..ex.mem_at])?;
        let after = self.decoder.embed_ids(&ex.ids[ex.mem_at + k..])?;
        self.decoder
            .hidden(&Tensor::cat(&[&before, mem, &after], 0)?.unsqueeze(0)?)?
            .squeeze(0)
    }

    /// Accuracy on the validation claims: is P(True) > P(False) right at the answer position? With the module's
    /// memory, and with the placeholders' own embeddings (the model without memory).
    fn claim_accuracy(&self) -> candle_core::Result<(f64, f64, usize)> {
        let (t, f) = self.true_false;
        let (mut with, mut without, mut n) = (0usize, 0usize, 0usize);
        for ex in self.val.iter().filter(|e| e.kind == "claim") {
            let truth = ex.ids[ex.answer_from] == t;
            for (mem, hits) in [
                (self.memory(ex)?.detach(), &mut with),
                (self.plain.clone(), &mut without),
            ] {
                let h = self
                    .hidden_with(ex, &mem)?
                    .i(ex.answer_from - 1)?
                    .unsqueeze(0)?;
                let l: Vec<f32> = self.decoder.logits(&h)?.squeeze(0)?.to_vec1()?;
                if (l[t as usize] > l[f as usize]) == truth {
                    *hits += 1;
                }
            }
            n += 1;
        }
        Ok((
            with as f64 / n.max(1) as f64,
            without as f64 / n.max(1) as f64,
            n,
        ))
    }

    /// The answer-token loss of one example (a scalar tensor in the autograd graph).
    fn loss(&self, ex: &Example) -> candle_core::Result<Tensor> {
        self.loss_from(ex, self.memory(ex)?)
    }

    /// The same loss with the memory detached first: nothing before the decoder is tracked, so validation does not
    /// hold a 1.7B model's activations (about 1 GB) for a number it only reads.
    fn eval_loss(&self, ex: &Example) -> candle_core::Result<Tensor> {
        self.loss_from(ex, self.memory(ex)?.detach())
    }

    fn loss_from(&self, ex: &Example, mem: Tensor) -> candle_core::Result<Tensor> {
        let k = mem.dim(0)?;
        let before = self.decoder.embed_ids(&ex.ids[..ex.mem_at])?;
        let after = self.decoder.embed_ids(&ex.ids[ex.mem_at + k..])?;
        let x = Tensor::cat(&[&before, &mem, &after], 0)?.unsqueeze(0)?;
        let h = self.decoder.hidden(&x)?.squeeze(0)?;
        // Position j predicts token j+1: the answer tokens are predicted from answer_from-1 .. len-2.
        let n = ex.ids.len() - ex.answer_from;
        let hs = h.narrow(0, ex.answer_from - 1, n)?;
        let logits = self.decoder.logits(&hs)?;
        let targets = Tensor::new(&ex.ids[ex.answer_from..], self.decoder.device())?;
        candle_nn::loss::cross_entropy(&logits, &targets)
    }

    fn validate(&self) -> candle_core::Result<f64> {
        if self.val.is_empty() {
            return Ok(f64::NAN);
        }
        let mut s = 0.0;
        for ex in &self.val {
            s += self.eval_loss(ex)?.to_scalar::<f32>()? as f64;
        }
        Ok(s / self.val.len() as f64)
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
        let warm = (total / 10).max(1);
        let vars: Vec<Var> = self.vars.all_vars();
        let mut adam = Adam::new(&vars).map_err(|e| e.to_string())?;
        let mut rng = self.opts.seed.max(1);
        let mut step = 0usize;
        let mut seen = 0usize;
        let mut best = f64::INFINITY;
        let v0 = self.validate().map_err(|e| e.to_string())?;
        let a0 = self.claim_accuracy().map_err(|e| e.to_string())?;
        self.log(json!({"event": "start", "claim_accuracy": a0.0, "claim_accuracy_without_memory": a0.1, "claims": a0.2,
            "claims_true": self.claims.0, "claims_false": self.claims.1, "kb_name": self.kb_name, "train": self.train.len(),
            "validation": self.val.len(), "steps": total, "val_loss": v0}));
        if self.claims.1 == 0 && self.claims.0 > 0 {
            self.log(json!({"event": "warning", "message": "every claim in this dataset is true (the graph had no                 second entity of a type to swap in), so claim accuracy only says how often the model answers True"}));
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
                let mut acc: Vec<Option<Tensor>> = vec![None; vars.len()];
                let mut total_loss = 0.0f64;
                for &i in batch {
                    let l = (self.loss(&self.train[i]).map_err(|e| e.to_string())?
                        / batch.len() as f64)
                        .map_err(|e| e.to_string())?;
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
                    "grad_norm": norm, "seconds": t.elapsed().as_secs_f64(), "seconds_per_example": t.elapsed().as_secs_f64() / batch.len() as f64, "kinds": kinds}));
            }
            let v = self.validate().map_err(|e| e.to_string())?;
            let dir = self.opts.out.join(format!("epoch-{epoch:02}"));
            MemoryModule::save(&self.vars, &self.module.cfg, &dir)?;
            if v < best && v.is_finite() {
                best = v;
                MemoryModule::save(&self.vars, &self.module.cfg, &self.opts.out.join("best"))?;
            }
            self.log(json!({"event": "epoch", "epoch": epoch, "val_loss": v, "best": best, "saved": dir}));
        }
        if step == 0 {
            // Stopped before the first step: the module is still its initialisation and must not be offered as a
            // checkpoint, or it would be activated and answer every question with memory that learnt nothing.
            self.log(json!({"event": "stopped", "step": 0}));
            self.log(json!({"event": "done", "steps": 0, "examples": 0, "val_loss": v0, "start_val_loss": v0,
                "error": "the run was stopped before it had trained on anything, so no module was saved"}));
            return Ok(v0);
        }
        if step < total && self.opts.out.join("stop").exists() {
            MemoryModule::save(&self.vars, &self.module.cfg, &self.opts.out.join("stopped"))?;
            self.log(json!({"event": "stopped", "step": step}));
        }
        let v = self.validate().map_err(|e| e.to_string())?;
        // The module as it ends, always (a run cut short by --max-examples never reaches an epoch's end).
        MemoryModule::save(&self.vars, &self.module.cfg, &self.opts.out.join("last"))?;
        if !self.opts.out.join("best").exists() || (v < best && v.is_finite()) {
            MemoryModule::save(&self.vars, &self.module.cfg, &self.opts.out.join("best"))?;
        }
        let a = self.claim_accuracy().map_err(|e| e.to_string())?;
        self.log(json!({"event": "done", "steps": step, "examples": seen, "val_loss": v, "start_val_loss": v0,
            "claim_accuracy": a.0, "claim_accuracy_without_memory": a.1, "claims": a.2}));
        Ok(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_warms_up_then_decays_to_a_tenth() {
        assert!((lr_at(0, 100, 10, 1.0) - 0.1).abs() < 1e-9);
        assert!((lr_at(9, 100, 10, 1.0) - 1.0).abs() < 1e-9);
        assert!((lr_at(10, 100, 10, 1.0) - 1.0).abs() < 1e-9);
        assert!((lr_at(100, 100, 10, 1.0) - 0.1).abs() < 1e-9);
        assert!(lr_at(55, 100, 10, 1.0) < 0.6 && lr_at(55, 100, 10, 1.0) > 0.5);
    }
}
