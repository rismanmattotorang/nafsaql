//! The session API: load a model, hold a conversation, report what it cost.
//!
//! # Suffix-indexed speculation
//!
//! The paper proposed speculative decode with a draft model. This implements the
//! variant that needs no second model at all: match the longest suffix of what has
//! been generated against everything earlier in the context (`suffix.rs`: an n-gram
//! index, a vote across occurrences, a confidence table learned from what
//! verification accepted) and propose whatever followed. How many tokens to verify
//! is priced against the measured cost of a verification pass of that width
//! (`choose_k`), so a doubtful draft is not verified at all. The draft costs a few
//! hash lookups — microseconds against a decode step measured in tens of
//! milliseconds — and a few bytes per context token.
//!
//! The reason it works here is the workload. ParagonCorp's tasks are *input-grounded*:
//! catalogue normalisation, INCI extraction, invoice parsing, review scoring. Most
//! output tokens are copied or lightly transformed from the prompt, so the context
//! already contains the continuation. On free-form generation the acceptance rate
//! falls to near zero and the mechanism costs one extra batched forward per step —
//! which is why `acceptance_rate` is reported rather than assumed.
//!
//! Verification is exact: a drafted token is accepted only when it equals the
//! argmax the model would have produced anyway, so greedy-with-speculation emits
//! *precisely* the greedy sequence (acceptance criterion A6). Because that guarantee
//! only holds for argmax, speculation is disabled automatically when the temperature
//! is non-zero rather than quietly changing the sampling distribution.

use crate::dspark::Drafter;
use crate::gguf::Gguf;
use crate::kv::{KvCache, KvPrec};
use crate::model::{Model, State, StepStats};
use crate::pool::Pool;
use crate::suffix::{Cell, SuffixIndex};
use crate::telem::{Machine, TokenTelem, TurnTelem};
use crate::template::{today_llama3, Template};
use crate::tokenizer::Utf8Stream;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

/// Prompt tokens per forward call. Bounds scratch memory while still reading the
/// weights once per chunk instead of once per prompt token.
const PREFILL_CHUNK: usize = 64;

#[derive(Debug, Clone)]
pub struct SessionCfg {
    pub max_ctx: usize,
    pub kv_prec: KvPrec,
    /// 0 means greedy. Speculation is only enabled at 0 (see module docs).
    pub temperature: f32,
    pub top_p: f32,
    // nafs-aql: keep only the `top_k` most probable tokens (0 = all) and those with at least `min_p` times the
    // best token's probability (0 = off); llama-server's defaults are 40 and 0.05.
    pub top_k: usize,
    pub min_p: f32,
    pub seed: u64,
    pub max_tokens: usize,
    pub threads: usize,
    pub speculate: bool,
    /// Tokens proposed per speculative step.
    pub draft_len: usize,
    /// Shortest suffix (in tokens) accepted as a match in the context; 0 disables
    /// suffix drafting. Longer matches are preferred whenever they exist.
    pub ngram: usize,
    /// Prepended once per conversation. `None` uses the model's own default.
    pub system: Option<String>,
    /// Let a reasoning model think (Qwen3's `enable_thinking`). Off writes the
    /// template's empty reasoning block, so the reply starts with the answer.
    pub thinking: bool,
    /// A DSpark drafter directory (`config.json` + `model.safetensors`) trained for
    /// this target. When set, greedy decoding drafts with it instead of prompt lookup.
    pub dspark: Option<std::path::PathBuf>,
    /// Most draft tokens verified per round (≤ the drafter's block); the policy may
    /// verify fewer. `SMILEY_DSPARK_K` pins an exact count for measurement.
    pub dspark_max: usize,
    // nafs-aql: text the assistant has already written for the turn being planned, appended
    // after the template's assistant header (and after the empty reasoning block when thinking
    // is off). A continuation: the server closes a reasoning span at its budget by replanning
    // the turn with the reasoning so far plus `</think>` here, and the cache prefix is reused.
    pub assistant_prefix: Option<String>,
}

impl Default for SessionCfg {
    fn default() -> Self {
        SessionCfg {
            max_ctx: 2048,
            kv_prec: KvPrec::F16,
            temperature: 0.0,
            top_p: 0.95,
            top_k: 0,
            min_p: 0.0,
            seed: 0,
            max_tokens: 256,
            threads: std::thread::available_parallelism().map_or(4, |n| n.get()),
            speculate: true,
            draft_len: 12,
            ngram: 2,
            system: None,
            thinking: true,
            dspark: None,
            dspark_max: 3,
            assistant_prefix: None,
        }
    }
}

pub struct Session {
    pub model: std::sync::Arc<Model>,
    kv: KvCache,
    st: State,
    /// Created once per session. Its workers park between matmuls rather than
    /// being respawned 224 times per token.
    pool: Pool,
    pub cfg: SessionCfg,
    pub machine: Machine,
    /// Every token fed to the model so far. Also the corpus prompt-lookup searches.
    ctx: Vec<u32>,
    rng: u64,
    started: bool,
    /// The chat template the file declares, and every token id that ends a turn
    /// under it (plus the tokenizer's own eos).
    template: Template,
    stops: Vec<u32>,
    default_system: Option<&'static str>,
    /// The n-gram index over `ctx` and the confidence it has learned, and the cells
    /// the last draft's tokens are booked to once verification has decided them.
    suffix: SuffixIndex,
    draft_cells: Vec<Cell>,
    /// Consecutive speculative steps that accepted nothing (reported, not acted on).
    spec_misses: u32,
    steps: u64,
    /// The conversation as text, (user, assistant) per exchange. The cache holds the
    /// same thing as positions; this copy is what lets the oldest exchanges be dropped
    /// and the rest re-encoded when the context fills, instead of failing the turn.
    history: Vec<(String, String)>,
    /// Scratch for nucleus sampling, reused so a sampled token allocates nothing.
    cand: Vec<(u32, f32)>,
    /// The attached DSpark drafter, fed the target's tapped hidden states.
    pub dspark: Option<Drafter>,
    /// Measured verify cost per batch width, for choosing how much to verify.
    vcost: VerifyCost,
    /// Plain steps left before drafting is probed again, and the measured token
    /// rate of drafting rounds (draft + verify), against which a plain step is priced.
    park: u32,
    draft_rate: Option<f64>,
    drafted_rounds: u64,
    /// Committed positions not yet fed to the drafter (while parked): fed in one
    /// batched update before the next draft, which costs far less than one per step.
    pending_taps: Vec<f32>,
    pending_start: usize,
    /// The tokens the cache still holds after a `reset` or `seed_history`: the next
    /// prompt is matched against them and only its divergent tail is prefilled
    /// (`reuse_prefix`). Empty while a conversation is live.
    kept: Vec<u32>,
    /// Positions the last turn took from the cache instead of prefilling them.
    prefix_reused: usize,
}

/// Plain steps to take after drafting is measured to lose, before probing again.
const PARK_STEPS: u32 = 12;

/// What one extra row costs a verification pass, as a fraction of a plain step.
/// Measured with `examples/vcurve` (width 1..16, median of 9): SmolLM2-360M 0.47,
/// Llama-3.2-1B 0.40, Qwen2.5-0.5B 0.42. Verification on this CPU is compute-bound,
/// not bandwidth-bound — a width-8 pass costs about 4 steps, not 1 — which is why
/// the ceiling on any lossless speculation here is near 1/0.42 ≈ 2.4×. The engine
/// priced a row at 0.12 before this was measured, and drafted far too short.
const ROW_COST: f64 = 0.42;

/// Seconds a verification forward of width `w` takes, learned online. A width seen
/// twice is its own measurement (an EMA; the first reading of a width is a cold one
/// and the second can only lower it). Any other width is the plain-step time scaled
/// by `1 + β·(w−1)`, with β the mean slope of the widths measured so far, or
/// `ROW_COST` before any. Extrapolating a line through two or three noisy points
/// out to width 12 — what this did before — priced a 12-token draft at 85 ms
/// against a true 77 ms-or-less and refused every long draft.
#[derive(Default)]
struct VerifyCost {
    ema: [Option<f64>; 17],
    seen: [u8; 17],
}

impl VerifyCost {
    fn observe(&mut self, w: usize, secs: f64) {
        if w == 0 || w >= self.ema.len() {
            return;
        }
        self.seen[w] = self.seen[w].saturating_add(1);
        self.ema[w] = Some(match (self.ema[w], self.seen[w]) {
            (Some(e), 2) => e.min(secs),
            (Some(e), _) => 0.8 * e + 0.2 * secs,
            (None, _) => secs,
        });
    }
    /// Mean extra-row cost, in plain steps, over the widths measured at least twice.
    fn slope(&self) -> f64 {
        let Some(t1) = self.ema[1].filter(|_| self.seen[1] >= 2) else {
            return ROW_COST;
        };
        let s: Vec<f64> = (2..self.ema.len())
            .filter(|&w| self.seen[w] >= 2)
            .filter_map(|w| {
                self.ema[w].map(|t| ((t / t1 - 1.0) / (w as f64 - 1.0)).clamp(0.15, 0.9))
            })
            .collect();
        if s.is_empty() {
            ROW_COST
        } else {
            s.iter().sum::<f64>() / s.len() as f64
        }
    }
    fn estimate(&self, w: usize) -> Option<f64> {
        if let (Some(t), true) = (
            self.ema.get(w).copied().flatten(),
            self.seen.get(w).is_some_and(|&n| n >= 2),
        ) {
            return Some(t);
        }
        let t1 = self.ema[1]?;
        Some(t1 * (1.0 + self.slope() * (w as f64 - 1.0)))
    }
    /// `estimate`, or the prior curve while not even a plain step has been timed.
    fn cost(&self, w: usize) -> f64 {
        self.estimate(w)
            .unwrap_or(1.0 + ROW_COST * (w as f64 - 1.0))
    }
}

/// How many of `conf.len()` drafted tokens to verify: the prefix length `k` that
/// maximises expected committed tokens per second of verification, where
/// `conf[j]` is the head's probability that draft `j` survives given its prefix did.
/// The prefix survives with the running product, so expected committed tokens are
/// `1 + Σ_{j<k} Π_{i≤j} conf[i]` (the `1` is the target's own token). DeepInfer's
/// batch version of the same rule is `mtp_varlen.hpp`; at batch 1 it is this.
fn choose_k(conf: &[f32], cost: impl Fn(usize) -> f64) -> usize {
    let (mut best_k, mut best) = (0usize, 1.0 / cost(1));
    let (mut surv, mut expect) = (1.0f64, 1.0f64);
    for (j, &c) in conf.iter().enumerate() {
        surv *= c as f64;
        expect += surv;
        let rate = expect / cost(j + 2);
        if rate > best {
            best = rate;
            best_k = j + 1;
        }
    }
    best_k
}

/// Reserve this many positions for the answer when deciding whether a prompt fits.
/// A prompt that leaves no room to answer is not a fit.
fn answer_reserve(cfg: &SessionCfg) -> usize {
    cfg.max_tokens.min(cfg.max_ctx / 4).max(16)
}

/// Argmax, or temperature + nucleus sampling, over one row of logits.
///
/// A free function over borrowed logits: the method version needed `&mut self` for
/// the generator, so every call copied a 49 152-float row (192 KB) out of the
/// scratch first. Nucleus sampling no longer sorts the whole vocabulary either: the
/// softmax is summed over all of it (exactness needs the full normaliser), but only
/// the most probable candidates are sorted, and the full sort is the fallback when
/// they do not reach `top_p` — which keeps the distribution exactly the specified one.
fn sample_row(
    logits: &[f32],
    temperature: f32,
    top_p: f32,
    top_k: usize,
    min_p: f32,
    rng: &mut u64,
    cand: &mut Vec<(u32, f32)>,
) -> u32 {
    if temperature <= 0.0 {
        let mut best = 0usize;
        let mut bv = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > bv {
                bv = v;
                best = i;
            }
        }
        return best as u32;
    }
    let t = temperature;
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    cand.clear();
    let mut sum = 0.0f32;
    for (i, &v) in logits.iter().enumerate() {
        let e = ((v - max) / t).exp();
        sum += e;
        cand.push((i as u32, e));
    }
    // Most of the mass is almost always in the first few hundred tokens: partition
    // those to the front and sort only them. When they do not reach `top_p`, sort
    // everything — rare, and it keeps the nucleus exactly the specified one.
    const FAST_K: usize = 512;
    let desc = |a: &(u32, f32), b: &(u32, f32)| b.1.total_cmp(&a.1);
    let mut k = FAST_K.min(cand.len());
    if k < cand.len() {
        cand.select_nth_unstable_by(k - 1, desc);
    }
    cand[..k].sort_unstable_by(desc);
    if cand[..k].iter().map(|e| e.1).sum::<f32>() / sum < top_p && k < cand.len() {
        cand.sort_unstable_by(desc);
        k = cand.len();
    }
    // nafs-aql: top-k first (llama.cpp's sampler order: top-k, top-p, min-p); the nucleus is then taken over
    // the renormalised survivors. Only the sorted head is cut, so k above FAST_K means no cut.
    let mut prefix = &cand[..k];
    if top_k > 0 && top_k < prefix.len() {
        prefix = &prefix[..top_k];
        sum = prefix.iter().map(|e| e.1).sum();
    }
    // Nucleus: the smallest prefix whose mass reaches top_p.
    let mut cum = 0.0f32;
    let mut keep = prefix.len();
    for (i, e) in prefix.iter().enumerate() {
        cum += e.1 / sum;
        if cum >= top_p {
            keep = i + 1;
            break;
        }
    }
    // nafs-aql: min-p on the probabilities at temperature 1, as llama.cpp (it applies temperature last):
    // logit - max >= ln(min_p)  <=>  e >= min_p^(1/t) * e_max for e = exp((logit - max) / t).
    if min_p > 0.0 {
        let floor = min_p.powf(1.0 / t) * prefix[0].1;
        keep = prefix[..keep]
            .iter()
            .take_while(|e| e.1 >= floor)
            .count()
            .max(1);
    }
    let p = &prefix[..keep];
    let total: f32 = p.iter().map(|e| e.1).sum();
    let mut x = *rng;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *rng = x;
    let r = (x >> 11) as f32 / (1u64 << 53) as f32 * total;
    let mut acc = 0.0f32;
    for e in p {
        acc += e.1;
        if r <= acc {
            return e.0;
        }
    }
    p.last().map(|e| e.0).unwrap_or(0)
}

/// What the caller learns about each emitted token.
pub struct Emitted<'a> {
    pub id: u32,
    pub text: &'a str,
    pub telem: &'a TokenTelem,
    // nafs-aql: log-probability of `id` under the softmax of the raw logits (temperature 1).
    pub logprob: f32,
}

// nafs-aql: the log-probability of token `id` in one row of logits, with the full normaliser.
fn logprob_of(logits: &[f32], id: u32) -> f32 {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = logits.iter().map(|&v| (v - max).exp()).sum();
    logits
        .get(id as usize)
        .map_or(f32::NEG_INFINITY, |&v| v - max - sum.ln())
}

impl Session {
    pub fn open(path: &Path, cfg: SessionCfg) -> Result<Session, String> {
        crate::check_compatible(&Gguf::open(path)?)?;
        let model = crate::model::keep::open(path)?;
        let c = model.cfg;

        let max_ctx = cfg.max_ctx.clamp(32, c.ctx_train);
        let threads = cfg.threads.max(1);
        let dspark = match &cfg.dspark {
            Some(dir) => {
                let d = Drafter::load(dir, threads)?;
                d.check_target(c.n_vocab, c.d_model, c.n_layer)?;
                Some(d)
            }
            None => None,
        };
        let dspark_k = dspark.as_ref().map_or(0, |d| {
            cfg.dspark_max
                .clamp(1, d.cfg.block_size - d.cfg.logits_start)
        });
        let logit_rows = cfg.draft_len.max(dspark_k) + 1;
        let max_batch = PREFILL_CHUNK.max(logit_rows);
        let kv = KvCache::new(cfg.kv_prec, c.n_layer, c.n_kv_head, c.head_dim, max_ctx);
        let mut st = State::new(
            &c,
            max_batch,
            logit_rows,
            max_ctx,
            threads,
            model.rope_factors.as_deref(),
        );
        // Probe with the same width the engine will actually run at: a bandwidth
        // figure measured single-threaded is not the denominator a four-thread
        // decode step should be judged against.
        if let Some(d) = &dspark {
            st.set_taps(&d.cfg.target_layer_ids, c.d_model);
        }
        let machine = Machine::cached(threads);
        let pool = Pool::new(threads);
        let template = Template::detect(model.gguf())?;
        let stops = template.stop_ids(&model.tk);
        let default_system = template.default_system(model.name());
        let seed = if cfg.seed == 0 {
            0x2545_F491_4F6C_DD1D
        } else {
            cfg.seed
        };

        Ok(Session {
            model,
            kv,
            st,
            pool,
            cfg: SessionCfg {
                max_ctx,
                threads,
                dspark_max: dspark_k,
                ..cfg
            },
            machine,
            ctx: Vec::new(),
            rng: seed,
            started: false,
            template,
            stops,
            default_system,
            suffix: SuffixIndex::new(),
            draft_cells: Vec::new(),
            spec_misses: 0,
            steps: 0,
            history: Vec::new(),
            cand: Vec::new(),
            dspark,
            vcost: VerifyCost::default(),
            park: 0,
            draft_rate: None,
            drafted_rounds: 0,
            pending_taps: Vec::new(),
            pending_start: 0,
            kept: Vec::new(),
            prefix_reused: 0,
        })
    }

    /// Seconds for one verification forward of each width in `widths` (median of
    /// `reps`), at the end of the current context. The positions written are past
    /// `ctx` and are cut again, exactly as a rejected draft's are. Used to measure
    /// the batch→cost curve the draft length is priced against; call after a turn.
    pub fn probe_verify_curve(
        &mut self,
        widths: &[usize],
        reps: usize,
    ) -> Result<Vec<(usize, f64)>, String> {
        let start = self.ctx.len();
        let tok = self.ctx.last().copied().unwrap_or(0);
        let mut out = Vec::new();
        for &w in widths {
            let batch = vec![tok; w];
            let want: Vec<usize> = (0..w).collect();
            let mut times = Vec::new();
            for _ in 0..reps + 1 {
                let mut ss = StepStats::default();
                let t = Instant::now();
                let Session {
                    model,
                    st,
                    kv,
                    pool,
                    ..
                } = self;
                model.forward(st, kv, pool, &batch, start, &want, &mut ss)?;
                times.push(t.elapsed().as_secs_f64());
                self.kv.truncate(start);
            }
            times.remove(0); // the first touches pages the rest find warm
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            out.push((w, times[times.len() / 2]));
        }
        Ok(out)
    }

    /// Feed the drafter every committed position it has not seen yet.
    fn flush_pending(&mut self) {
        if let (Some(d), false) = (&mut self.dspark, self.pending_taps.is_empty()) {
            d.update_context(&self.pending_taps, self.pending_start, &self.pool);
        }
        self.pending_taps.clear();
    }

    /// Forget the conversation. The cache is *not* cleared: its positions stay valid
    /// for the tokens that were fed (`kept`), and the next prompt reuses whatever
    /// prefix it shares with them — the system block at least, when the system
    /// prompt is unchanged, which is what the quality lab's forty probes share, and
    /// everything up to the edited message on a regenerate or an edit
    /// (FreeToken's radix prefix cache and semantic anchors, in their one-sequence
    /// form; `docs/FREETOKEN-PLAN.md`).
    pub fn reset(&mut self) {
        self.flush_pending();
        if !self.ctx.is_empty() {
            self.kept = std::mem::take(&mut self.ctx);
        } else {
            self.ctx.clear();
        }
        self.pending_taps.clear();
        self.suffix.clear();
        self.started = false;
        self.spec_misses = 0;
        self.steps = 0;
        self.history.clear();
    }

    /// After a `reset`/`seed_history`, keep the cache positions the new prompt `toks`
    /// shares with what was fed before and return how many; the caller prefills the
    /// rest. The match stops one token short of the whole prompt, because a turn
    /// needs at least one new position to produce logits.
    fn reuse_prefix(&mut self, toks: &[u32]) -> usize {
        let kept = std::mem::take(&mut self.kept);
        let limit = toks.len().saturating_sub(1).min(kept.len());
        let l = toks
            .iter()
            .zip(kept.iter())
            .take(limit)
            .take_while(|(a, b)| a == b)
            .count();
        self.kv.truncate(l);
        if let Some(d) = &mut self.dspark {
            d.truncate(l);
        }
        self.ctx.clear();
        self.ctx.extend_from_slice(&kept[..l]);
        self.suffix.clear();
        l
    }

    /// Exchanges currently held in the conversation.
    // nafs-aql: re-seed the sampler for the next turn (`seed` is otherwise read once at open).
    pub fn reseed(&mut self, seed: u64) {
        self.cfg.seed = seed;
        self.rng = seed.max(1).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    }

    pub fn exchanges(&self) -> usize {
        self.history.len()
    }

    /// Resume an earlier conversation: the exchanges are replayed from an empty
    /// cache on the next turn (`plan_prompt` re-encodes the full transcript when
    /// nothing is started), so a reopened chat continues where it left off.
    pub fn seed_history(&mut self, exchanges: Vec<(String, String)>) {
        self.reset();
        self.history = exchanges;
    }

    pub fn context_used(&self) -> usize {
        self.ctx.len()
    }

    pub fn kv_resident_bytes(&self) -> u64 {
        self.kv.resident_bytes()
    }

    /// The system message in force: the caller's, else the one the model's own
    /// template would insert (SmolLM and Qwen name themselves; Llama 3 has none).
    fn system_text(&self) -> Option<&str> {
        self.cfg.system.as_deref().or(self.default_system)
    }

    /// The template the file declares (`chatml` or `llama3`).
    pub fn template(&self) -> Template {
        self.template
    }

    /// The text the model's own chat template produces for one more user message.
    ///
    /// On a continuing conversation the text begins by *closing the previous
    /// assistant message*. A turn ends when the model samples the end-of-turn token,
    /// and that token is emitted but never fed — so without this the cache held
    /// `…Paris.<|im_start|>user` and the model saw its own answer run straight into
    /// the next question, a transcript its template never produces.
    fn build_prompt(&self, user: &str) -> String {
        let mut s = String::new();
        if !self.started {
            s.push_str(
                &self
                    .template
                    .system_block(self.system_text(), &today_llama3()),
            );
        } else {
            s.push_str(self.template.close_previous());
        }
        s.push_str(&self.template.user_turn(user));
        s.push_str(self.template.assistant_open(self.cfg.thinking));
        if let Some(p) = &self.cfg.assistant_prefix {
            s.push_str(p); // nafs-aql
        }
        s
    }

    /// The whole conversation from the start: system, the kept exchanges, then the
    /// new user message. Used after dropping the oldest exchanges.
    fn full_prompt(&self, kept: &[(String, String)], user: &str) -> String {
        let mut s = self
            .template
            .system_block(self.system_text(), &today_llama3());
        for (u, a) in kept {
            s.push_str(&self.template.user_turn(u));
            s.push_str(&self.template.assistant_turn(a));
        }
        s.push_str(&self.template.user_turn(user));
        s.push_str(self.template.assistant_open(self.cfg.thinking));
        if let Some(p) = &self.cfg.assistant_prefix {
            s.push_str(p); // nafs-aql
        }
        s
    }

    /// The tokens to prefill for `user`, and how many old exchanges had to go.
    ///
    /// The incremental prompt is used whenever it fits with room left to answer.
    /// Otherwise the oldest exchanges are dropped one at a time and the remainder is
    /// re-encoded from an empty cache — the same policy the Colibri and Foundry
    /// adapters apply, so a long conversation degrades by forgetting its beginning
    /// rather than by refusing to answer. Only a single message that cannot fit even
    /// alone is an error.
    fn plan_prompt(&mut self, user: &str) -> Result<(Vec<u32>, usize), String> {
        let reserve = answer_reserve(&self.cfg);
        let max = self.cfg.max_ctx;
        self.prefix_reused = 0;
        let inc = if !self.started && !self.history.is_empty() {
            // Seeded history: the whole transcript is the prompt, less what the cache
            // still holds of it.
            let toks = self
                .model
                .tk
                .encode(&self.full_prompt(&self.history.clone(), user));
            let l = self.reuse_prefix(&toks);
            self.prefix_reused = l;
            toks[l..].to_vec()
        } else if !self.started {
            // A fresh or reset conversation: the system block may be in the cache.
            let toks = self.model.tk.encode(&self.build_prompt(user));
            let l = self.reuse_prefix(&toks);
            self.prefix_reused = l;
            toks[l..].to_vec()
        } else {
            self.model.tk.encode(&self.build_prompt(user))
        };
        if self.ctx.len() + inc.len() + reserve <= max {
            return Ok((inc, 0));
        }
        for drop in 0..=self.history.len() {
            let toks = self
                .model
                .tk
                .encode(&self.full_prompt(&self.history[drop..], user));
            let last = drop == self.history.len();
            if toks.len() + reserve <= max || (last && toks.len() + 1 < max) {
                self.history.drain(..drop);
                // The re-encoded transcript shares its system block (and, when few
                // exchanges went, more) with the cache: keep that, prefill the rest.
                self.pending_taps.clear();
                if self.kept.is_empty() {
                    self.kept = std::mem::take(&mut self.ctx);
                }
                let l = self.reuse_prefix(&toks);
                self.prefix_reused = l;
                self.started = false;
                self.spec_misses = 0;
                return Ok((toks[l..].to_vec(), drop));
            }
        }
        let alone = self.model.tk.encode(&self.full_prompt(&[], user)).len();
        Err(format!(
            "this message needs {alone} positions on its own, more than the {max}-token \
             context holds — raise the context length in Settings"
        ))
    }

    /// Propose tokens to follow `next` from the context, and how many of them to
    /// verify.
    ///
    /// `next` is the crucial argument. It is the token just emitted but *not yet
    /// fed*, so it is not in `self.ctx` — and the drafted block will be placed
    /// immediately after it. Matching on `ctx`'s last n tokens alone searches for
    /// an n-gram that ends one position too early, and the proposed continuation is
    /// then offset by one from what the model is about to predict. That version
    /// accepted 0 of 168 drafts on a verbatim copy task, which is the single case
    /// prompt-lookup exists to win. (`SuffixIndex::propose` takes `next` for the
    /// same reason.)
    ///
    /// The count is `choose_k` over the learned per-token confidences and the
    /// measured verify cost. When that says "verify nothing" the step is a plain
    /// one — except every `PROBE_EVERY`th step, which verifies a couple of tokens
    /// anyway, because a confidence that is never tested never recovers.
    fn suffix_draft(&mut self, next: u32) -> Vec<u32> {
        const PROBE_EVERY: u64 = 16;
        self.draft_cells.clear();
        let min = self.cfg.ngram;
        if min == 0 || self.cfg.draft_len == 0 {
            return Vec::new();
        }
        self.suffix.sync(&self.ctx, min);
        let d = self
            .suffix
            .propose(&self.ctx, next, self.cfg.draft_len, min);
        if d.toks.is_empty() {
            self.suffix.cool();
            return Vec::new();
        }
        let conf: Vec<f32> = d.cells.iter().map(|&c| self.suffix.conf(c)).collect();
        let vcost = &self.vcost;
        let mut k = choose_k(&conf, |w| vcost.cost(w));
        if k == 0 && self.steps.is_multiple_of(PROBE_EVERY) {
            k = d.toks.len().min(2);
        }
        if k == 0 {
            self.suffix.cool();
        }
        let mut toks = d.toks;
        toks.truncate(k);
        self.draft_cells = d.cells;
        self.draft_cells.truncate(k);
        toks
    }

    /// Run one conversational turn. `on_token` sees text as it is produced.
    pub fn generate<F: FnMut(Emitted)>(
        &mut self,
        user: &str,
        on_token: F,
    ) -> Result<(String, TurnTelem), String> {
        self.generate_with(user, None, on_token)
    }

    /// As `generate`, stopping cleanly as soon as `cancel` is set. The flag is
    /// checked between prefill chunks and before every decode step, so a Stop takes
    /// effect within one step instead of after the whole answer — which at 1 024
    /// tokens was most of a minute of a core-saturating thread nobody was reading.
    /// A cancelled turn returns what it produced, marked `cancelled`, and the
    /// conversation stays consistent for the next message.
    pub fn generate_with<F: FnMut(Emitted)>(
        &mut self,
        user: &str,
        cancel: Option<&AtomicBool>,
        mut on_token: F,
    ) -> Result<(String, TurnTelem), String> {
        let n_vocab = self.model.cfg.n_vocab;
        let turn_start = Instant::now();
        let mut telem = TurnTelem {
            ctx_max: self.cfg.max_ctx,
            ..Default::default()
        };
        let stopped = |c: Option<&AtomicBool>| c.is_some_and(|f| f.load(Ordering::Relaxed));

        // ---- prefill ----
        self.flush_pending();
        let (ptoks, dropped) = self.plan_prompt(user)?;
        telem.dropped_exchanges = dropped;
        telem.prefix_reused = self.prefix_reused;
        self.started = true;
        telem.prompt_tokens = ptoks.len();

        let mut pre = StepStats::default();
        let t0 = Instant::now();
        let mut fed = 0usize;
        while fed < ptoks.len() {
            if stopped(cancel) {
                // Nothing of this turn is in the conversation yet; roll the cache back
                // to where the turn started so the next message sees a clean context.
                self.kv.truncate(self.ctx.len());
                let keep = self.ctx.len();
                if let Some(d) = &mut self.dspark {
                    d.truncate(keep);
                }
                telem.cancelled = true;
                telem.prefill_s = t0.elapsed().as_secs_f64();
                telem.wall_s = turn_start.elapsed().as_secs_f64();
                telem.ctx_used = self.ctx.len();
                return Ok((String::new(), telem));
            }
            let take = PREFILL_CHUNK.min(ptoks.len() - fed);
            let last = fed + take == ptoks.len();
            let start = self.ctx.len() + fed;
            // Only the final chunk's last row produces the logits we sample from.
            let want: &[usize] = if last { &[take - 1] } else { &[] };
            let Session {
                model,
                st,
                kv,
                pool,
                dspark,
                ..
            } = self;
            model.forward(st, kv, pool, &ptoks[fed..fed + take], start, want, &mut pre)?;
            if let Some(d) = dspark {
                // Every prompt position joins the drafter's context as it is computed.
                let w = d.cfg.target_layer_ids.len() * model.cfg.d_model;
                let tc = Instant::now();
                d.update_context(&st.tap[..take * w], start, pool);
                telem.dspark_ctx_s += tc.elapsed().as_secs_f64();
                dump_prefill(&ptoks[fed..fed + take], &st.tap[..take * w], start);
            }
            fed += take;
        }
        self.ctx.extend_from_slice(&ptoks);
        telem.prefill_bytes = pre.bytes_total();
        telem.prefill_macs = pre.macs;
        telem.prefill_phases.add_step(&pre);

        // ---- decode ----
        //
        // The token sampled from the prefill logits is the turn's *first output
        // token*. An earlier version fed it and emitted only what came after, which
        // silently dropped the first word of every reply — the model said "The
        // capital of France is Paris" and the user saw "capital of France is Paris".
        // It is emitted here, with prefill's cost, and fed at the top of the loop.
        let mut out = String::new();
        let mut utf8 = Utf8Stream::default();
        let ts = Instant::now();
        let mut next = {
            let Session {
                st, cfg, rng, cand, ..
            } = self;
            sample_row(
                &st.logits[..n_vocab],
                cfg.temperature,
                cfg.top_p,
                cfg.top_k,
                cfg.min_p,
                rng,
                cand,
            )
        };
        telem.prefill_phases.sample_s += ts.elapsed().as_secs_f64();
        telem.prefill_s = t0.elapsed().as_secs_f64();
        let trace = std::env::var_os("SMILEY_CORE_TRACE").is_some();
        let speculating =
            (self.cfg.speculate || self.dspark.is_some()) && self.cfg.temperature <= 0.0;
        let fixed_k: Option<usize> = std::env::var("SMILEY_DSPARK_K")
            .ok()
            .and_then(|v| v.parse().ok());
        let mut step = 0u32;

        let stops = self.stops.clone();
        let ends = |id: u32| stops.contains(&id);
        if !ends(next) {
            step = 1;
            let tt = TokenTelem {
                step,
                dt_s: telem.prefill_s,
                bytes_weights: pre.bytes_weights,
                bytes_kv: pre.bytes_kv,
                macs: pre.macs,
                barriers: pre.barriers,
                from_gemm: ptoks.len() > 1,
                is_prefill: true,
                ..Default::default()
            };
            let text = utf8.push(&self.model.tk.token_bytes(next));
            out.push_str(&text);
            telem.tokens.push(tt);
            let logprob = logprob_of(&self.st.logits[..n_vocab], next); // nafs-aql
            on_token(Emitted {
                id: next,
                text: &text,
                telem: telem.tokens.last().unwrap(),
                logprob,
            });
        }

        while step < self.cfg.max_tokens as u32 && !ends(next) {
            if stopped(cancel) {
                telem.cancelled = true;
                break;
            }
            let t = Instant::now();
            let mut ss = StepStats::default();

            // `next` has been emitted but not yet fed. Feed it together with whatever
            // the context suggests will follow it.
            let td = Instant::now();
            if self.park > 0 {
                self.park -= 1;
            }
            let drafting = speculating && self.dspark.is_some() && self.park == 0;
            if drafting {
                self.flush_pending();
            }
            let drafted = if drafting {
                let kmax = self.cfg.dspark_max;
                let Session {
                    dspark,
                    pool,
                    vcost,
                    ..
                } = self;
                let d = dspark.as_mut().unwrap();
                let (toks, conf) = d.draft(next, kmax, pool);
                let k = match fixed_k {
                    Some(k) => k.min(toks.len()),
                    None if conf.len() == toks.len() && !conf.is_empty() => {
                        choose_k(&conf, |w| vcost.cost(w))
                    }
                    None => toks.len(),
                };
                dump_draft(next, &toks, &conf, k);
                telem.dspark_rounds += 1;
                telem.dspark_verified += k as u64;
                telem.dspark_draft_s += td.elapsed().as_secs_f64();
                toks[..k].to_vec()
            } else if speculating && self.dspark.is_none() {
                self.suffix_draft(next)
            } else {
                Vec::new()
            };
            telem.decode_phases.draft_s += td.elapsed().as_secs_f64();
            self.steps += 1;
            let mut batch = Vec::with_capacity(drafted.len() + 1);
            batch.push(next);
            batch.extend_from_slice(&drafted);
            let start = self.ctx.len();
            if start + batch.len() >= self.cfg.max_ctx {
                break;
            }
            let want: Vec<usize> = (0..batch.len()).collect();
            let tv = Instant::now();
            {
                let Session {
                    model,
                    st,
                    kv,
                    pool,
                    ..
                } = self;
                model.forward(st, kv, pool, &batch, start, &want, &mut ss)?;
            }
            let verify_s = tv.elapsed().as_secs_f64();
            self.vcost.observe(batch.len(), verify_s);

            // Accept a drafted token only when it equals what the model would have
            // produced anyway, so the emitted sequence is bit-identical to greedy
            // decoding without speculation (A6). `emitted` always holds at least one
            // token: the argmax after `next`.
            let ts = Instant::now();
            let mut emitted = Vec::with_capacity(batch.len());
            let mut accepted = 0usize;
            for j in 0..batch.len() {
                let pick = {
                    let Session {
                        st, cfg, rng, cand, ..
                    } = self;
                    sample_row(
                        st.logits_row(j, n_vocab),
                        cfg.temperature,
                        cfg.top_p,
                        cfg.top_k,
                        cfg.min_p,
                        rng,
                        cand,
                    )
                };
                if trace {
                    let r = self.st.logits_row(j, n_vocab);
                    let (mut m1, mut m2) = (f32::MIN, f32::MIN);
                    for &v in r {
                        if v > m1 {
                            m2 = m1;
                            m1 = v
                        } else if v > m2 {
                            m2 = v
                        }
                    }
                    if m1 - m2 < 0.25 {
                        eprintln!(
                            "near-tie  token {} of this turn: top-2 logit gap {:.4}",
                            step as usize + j + 1,
                            m1 - m2
                        );
                    }
                }
                emitted.push(pick);
                if j < drafted.len() && pick == drafted[j] {
                    accepted += 1;
                } else {
                    break;
                }
            }
            telem.decode_phases.sample_s += ts.elapsed().as_secs_f64();
            telem.decode_phases.add_step(&ss);
            let step_before = step;
            // Positions to keep: `next` at `start`, plus each accepted draft. The
            // rejected tail was written to the cache before verification and must not
            // remain visible to the next step.
            self.ctx.push(next);
            for &d in drafted.iter().take(accepted) {
                self.ctx.push(d);
            }
            self.kv.truncate(self.ctx.len());
            if let Some(d) = &mut self.dspark {
                // The committed positions — `next` and the accepted drafts — join the
                // drafter's context; the rejected tail's hidden states are dropped.
                // While parked they wait, and go in as one batch before the next draft.
                let w = d.cfg.target_layer_ids.len() * self.model.cfg.d_model;
                if self.pending_taps.is_empty() {
                    self.pending_start = start;
                }
                self.pending_taps
                    .extend_from_slice(&self.st.tap[..(1 + accepted) * w]);
                if drafting {
                    let tc = Instant::now();
                    self.flush_pending();
                    telem.dspark_ctx_s += tc.elapsed().as_secs_f64();
                    // Price the round against a plain step: a draft that does not pay
                    // for itself parks drafting for a while (DeepInfer's varlen rule
                    // prices the verify width; this prices the draft too).
                    let round_s = td.elapsed().as_secs_f64();
                    let rate = (1 + accepted) as f64 / round_s.max(1e-9);
                    self.draft_rate = Some(match self.draft_rate {
                        Some(r) => 0.7 * r + 0.3 * rate,
                        None => rate,
                    });
                    self.drafted_rounds += 1;
                    // Only against a plain step actually measured (an extrapolated one
                    // under a fluctuating load parked chat that drafting was winning),
                    // and only when drafting is clearly slower.
                    let plain = self.vcost.ema[1].map(|t| 1.0 / t);
                    if fixed_k.is_none() && self.drafted_rounds >= 8 {
                        if let (Some(dr), Some(pr)) = (self.draft_rate, plain) {
                            if dr < 0.8 * pr {
                                self.park = PARK_STEPS;
                                self.draft_rate = None;
                            }
                        }
                    }
                }
            }
            if !drafted.is_empty() {
                if self.dspark.is_none() {
                    self.suffix.observe(&self.draft_cells, accepted);
                }
                if accepted > 0 {
                    self.spec_misses = 0;
                } else {
                    self.spec_misses = self.spec_misses.saturating_add(1);
                }
            }

            let dt = t.elapsed().as_secs_f64();
            telem.decode_steps_s += dt;
            // One step produced `n_em` tokens for one pass over the weights. The cost
            // is divided across them, because that is what actually happened: a
            // speculative step that yields four tokens really did move the bytes once.
            let n_em = emitted.len().max(1);
            for (i, &id) in emitted.iter().enumerate() {
                if ends(id) {
                    next = id;
                    break;
                }
                step += 1;
                let tt = TokenTelem {
                    step,
                    dt_s: dt / n_em as f64,
                    bytes_weights: ss.bytes_weights / n_em as u64,
                    bytes_kv: ss.bytes_kv / n_em as u64,
                    macs: ss.macs / n_em as u64,
                    barriers: ss.barriers / n_em as u64,
                    drafted: if i == 0 { drafted.len() as u8 } else { 0 },
                    accepted: if i == 0 { accepted as u8 } else { 0 },
                    from_gemm: batch.len() > 1,
                    is_prefill: false,
                };
                let text = utf8.push(&self.model.tk.token_bytes(id));
                out.push_str(&text);
                telem.tokens.push(tt);
                let logprob = logprob_of(self.st.logits_row(i, n_vocab), id); // nafs-aql
                on_token(Emitted {
                    id,
                    text: &text,
                    telem: telem.tokens.last().unwrap(),
                    logprob,
                });
                next = id;
                if step >= self.cfg.max_tokens as u32 {
                    break;
                }
            }
            // A pass counts toward tokens-per-pass only when it emitted a token; the
            // final pass that samples only the end-of-turn token emits nothing.
            if step > step_before {
                telem.passes += 1;
            }
        }
        let tail = utf8.finish();
        out.push_str(&tail);
        if std::env::var_os("SMILEY_CORE_TRACE").is_some() {
            let ms: Vec<String> = self
                .vcost
                .ema
                .iter()
                .enumerate()
                .filter_map(|(w, t)| t.map(|t| format!("{w}:{:.1}", t * 1e3)))
                .collect();
            eprintln!("verify ms by width  {}", ms.join("  "));
        }

        self.history.push((user.to_string(), out.clone()));
        if let Some(d) = &mut self.dspark {
            telem.dspark_bytes = std::mem::take(&mut d.stats.bytes);
        }
        telem.ctx_used = self.ctx.len();
        telem.wall_s = turn_start.elapsed().as_secs_f64();
        Ok((out, telem))
    }
}

/// `SMILEY_DSPARK_DUMP=<dir>`: write the prompt's tapped hidden states and the first
/// draft to `<dir>`, so the PyTorch reference (`tests/dspark_reference.py`) can replay
/// the drafter on exactly the inputs this engine gave it.
fn dump_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("SMILEY_DSPARK_DUMP").map(std::path::PathBuf::from)
}

fn dump_prefill(toks: &[u32], taps: &[f32], start: usize) {
    let Some(dir) = dump_dir() else { return };
    let _ = std::fs::create_dir_all(&dir);
    use std::io::Write;
    let open = |n: &str| {
        std::fs::OpenOptions::new()
            .create(true)
            .append(start > 0)
            .write(true)
            .truncate(start == 0)
            .open(dir.join(n))
    };
    if let Ok(mut f) = open("taps.f32") {
        let _ = f.write_all(
            &taps
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
        );
    }
    if let Ok(mut f) = open("tokens.txt") {
        let _ = writeln!(
            f,
            "{}",
            toks.iter()
                .map(|t| t.to_string())
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
}

fn dump_draft(anchor: u32, toks: &[u32], conf: &[f32], k: usize) {
    let Some(dir) = dump_dir() else { return };
    let p = dir.join("draft.json");
    if p.exists() {
        return; // the first draft only
    }
    let _ = std::fs::write(
        p,
        format!(
            "{{\"anchor\": {anchor}, \"tokens\": {:?}, \"confidence\": {:?}, \"chosen\": {k}}}\n",
            toks, conf
        ),
    );
}

#[cfg(test)]
mod tests {

    // nafs-aql: top-k and min-p cut the tail before sampling.
    #[test]
    fn top_k_and_min_p_cut_the_tail() {
        // Four tokens at temperature 1: probabilities proportional to e^3, e^2, e^1, e^0 (token 0 is likeliest).
        let logits = [3.0f32, 2.0, 1.0, 0.0];
        let mut cand = Vec::new();
        let mut draw = |top_k, min_p, top_p, seed: u64| {
            // Seeds are scrambled as `Session::reseed` does: xorshift from a tiny seed starts with tiny draws.
            let mut rng = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
            super::sample_row(&logits, 1.0, top_p, top_k, min_p, &mut rng, &mut cand)
        };
        // top-k 1 is greedy whatever the seed.
        assert!((1..200u64).all(|s| draw(1, 0.0, 1.0, s) == 0));
        // top-k 2 never returns the two least likely tokens, and does return both of the others.
        let picks: Vec<u32> = (1..400u64).map(|s| draw(2, 0.0, 1.0, s)).collect();
        assert!(picks.iter().all(|&t| t < 2));
        assert!(picks.contains(&0) && picks.contains(&1));
        // min-p 0.3 drops tokens below 0.3 x the best probability: e^-1 = 0.37 stays, e^-2 = 0.14 goes.
        let picks: Vec<u32> = (1..400u64).map(|s| draw(0, 0.3, 1.0, s)).collect();
        assert!(picks.iter().all(|&t| t < 2));
        // Off by default: with no cut every token can be drawn.
        let picks: Vec<u32> = (1..2000u64).map(|s| draw(0, 0.0, 1.0, s)).collect();
        assert!((0..4).all(|t| picks.contains(&t)));
    }

    #[test]
    fn choose_k_prices_expected_tokens_against_verify_cost() {
        let flat = |_w: usize| 1.0;
        // Free verification: verify everything with any chance of surviving.
        assert_eq!(super::choose_k(&[0.9, 0.8, 0.7], flat), 3);
        // Each extra row costs 30% of a step: only drafts likely to survive pay.
        let steep = |w: usize| 1.0 + 0.3 * (w as f64 - 1.0);
        assert_eq!(super::choose_k(&[0.95, 0.9, 0.2], steep), 2);
        assert_eq!(super::choose_k(&[0.1, 0.9, 0.9], steep), 0);
        assert_eq!(super::choose_k(&[], flat), 0);
    }

    #[test]
    fn verify_cost_scales_the_plain_step_by_the_measured_row_cost() {
        let mut v = super::VerifyCost::default();
        assert_eq!(v.estimate(2), None);
        assert!(
            (v.cost(5) - (1.0 + 4.0 * super::ROW_COST)).abs() < 1e-12,
            "the prior before any measurement"
        );
        v.observe(1, 10.0); // a cold first step
        v.observe(1, 4.0);
        v.observe(1, 4.0);
        // Only the plain step is known: prior slope.
        assert!((v.estimate(3).unwrap() - 4.0 * (1.0 + 2.0 * super::ROW_COST)).abs() < 1e-9);
        // Width 2 measured twice: a cold 9.0 then 6.0 → min → 6.0, which is its own value
        // and moves the slope (6/4 − 1 = 0.5) used for unmeasured widths.
        v.observe(2, 9.0);
        v.observe(2, 6.0);
        assert!((v.estimate(2).unwrap() - 6.0).abs() < 1e-9);
        assert!((v.estimate(5).unwrap() - 4.0 * (1.0 + 4.0 * 0.5)).abs() < 1e-9);
        // A width seen once is not trusted: it is still the model.
        v.observe(8, 100.0);
        assert!((v.estimate(8).unwrap() - 4.0 * (1.0 + 7.0 * 0.5)).abs() < 1e-9);
    }
}
