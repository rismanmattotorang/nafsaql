// nafs-aql: text embeddings from a decoder-only embedding model (Qwen3-Embedding), Phase 20 of Nafs-Aql.
//
// Qwen3-Embedding is a Qwen3 decoder whose sentence vector is the final hidden state of the last token, after the
// output norm, L2-normalised ("last-token pooling"). The text is followed by `<|endoftext|>`, which is the pooled
// position. No chat template, no logits: the output head is never read, so a prefill is the whole cost.

use std::time::Instant;

use crate::kv::{KvCache, KvPrec};
use crate::model::{Model, State, StepStats};
use crate::pool::Pool;

/// Prompt tokens per forward call, as in a session's prefill.
const CHUNK: usize = 64;

pub struct Embedder {
    pub model: std::sync::Arc<Model>,
    st: State,
    kv: KvCache,
    pool: Pool,
    /// The token appended to every text and pooled (`<|endoftext|>`, else the tokenizer's eos).
    end: u32,
    /// Longest input in tokens, the end token included; longer texts are truncated.
    pub max_tokens: usize,
}

/// One embedding and what it cost.
pub struct Embedded {
    pub vector: Vec<f32>,
    pub tokens: usize,
    pub truncated: bool,
    pub ms: f64,
}

impl Embedder {
    pub fn new(model: std::sync::Arc<Model>, threads: usize, max_tokens: usize) -> Embedder {
        let c = model.cfg;
        let max_tokens = max_tokens.clamp(8, c.ctx_train);
        let mut st = State::new(
            &c,
            CHUNK,
            1,
            max_tokens,
            threads.max(1),
            model.rope_factors.as_deref(),
        );
        let (lx, lt) = model.lb_scratch(CHUNK);
        st.set_lb_scratch(lx, lt);
        let kv = KvCache::new(KvPrec::F16, c.n_layer, c.n_kv_head, c.head_dim, max_tokens);
        let end = model.tk.id_of("<|endoftext|>").unwrap_or(model.tk.eos);
        Embedder {
            st,
            kv,
            pool: Pool::new(threads.max(1)),
            end,
            model,
            max_tokens,
        }
    }

    /// Width of the vectors (the model's hidden size).
    pub fn dims(&self) -> usize {
        self.model.cfg.d_model
    }

    /// Embed one text: unit-length vector of `dims()` floats.
    pub fn embed(&mut self, text: &str) -> Result<Embedded, String> {
        let t0 = Instant::now();
        let mut toks = self.model.tk.encode(text);
        let truncated = toks.len() + 1 > self.max_tokens;
        toks.truncate(self.max_tokens - 1);
        toks.push(self.end);
        self.kv.truncate(0);
        let mut stats = StepStats::default();
        let mut start = 0;
        let mut last_rows = 0;
        for chunk in toks.chunks(CHUNK) {
            self.model.forward(
                &mut self.st,
                &mut self.kv,
                &self.pool,
                chunk,
                start,
                &[],
                &mut stats,
            )?;
            start += chunk.len();
            last_rows = chunk.len();
        }
        let mut v = self.model.final_hidden(&self.st, last_rows - 1);
        let norm = v
            .iter()
            .map(|x| (*x as f64) * (*x as f64))
            .sum::<f64>()
            .sqrt() as f32;
        if norm > 0.0 {
            v.iter_mut().for_each(|x| *x /= norm);
        }
        Ok(Embedded {
            vector: v,
            tokens: toks.len(),
            truncated,
            ms: t0.elapsed().as_secs_f64() * 1000.0,
        })
    }
}
