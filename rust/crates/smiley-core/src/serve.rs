// nafs-aql: serving several users from one model, one weight pass at a time.
//
// # Why this exists
//
// A decode step reads every weight once to produce one token. Serving `n` users one after another reads the
// whole model `n` times; serving them in one pass reads it once and does `n` dot products per weight row. On a
// memory-bound machine that is nearly the whole difference between the two. Measured on the reference machine
// (4 cores, 19.5 GB/s, Qwen3-1.7B Q4_K_M, `examples/batch_bench.rs`): 2 users 1.56x, 4 users 2.57x, 8 users
// 2.27x the tokens per second of serving them one at a time. The roofline says why it stops there: Q4_K holds
// about 0.56 bytes per weight, so a batch of `b` rows has an arithmetic intensity near `3.6 b` FLOP/byte, and
// this machine's ridge (460 GFLOP/s over 19.5 GB/s) is near 24 — crossed at about seven rows, after which the
// kernels are compute-bound and another user buys nothing. `DEFAULT_MAX_SLOTS` sits below that knee.
//
// # The loop
//
// One `step` is either a chunk of one user's prompt or one token for every user that is ready, never both:
// a prefill chunk and a decode row would need two mutable borrows of the same cache. Chunking is what keeps a
// long prompt from stopping everyone else (Sarathi-Serve's argument, which matters more here than on a GPU:
// a 2,000-token prompt is about 20 seconds of this machine, and `PREFILL_CHUNK` bounds the stall to about one
// second).
//
// # What it does not do
//
// No paged cache: each slot owns a contiguous cache. Paging exists on GPUs because video memory is scarce and
// fragmentation is fatal; here a slot's cache is 112-144 KB per position of ordinary memory, the slot count is
// small, and a block table would cost an indirection on the hottest loop in the engine for nothing. No
// speculation while batching: verifying a draft multiplies the rows per pass, which past the ridge above is
// exactly where there are no spare FLOPs (measured elsewhere: 2.7x at one row falling to 1.6x at eight).

use crate::kv::{KvCache, KvPrec};
use crate::model::{Caches, Model, State, StepStats};
use crate::pool::Pool;
use crate::prefix::{common_prefix, supersedes, CacheStore, MIN_PREFIX};
use crate::tokenizer::Utf8Stream;
use std::sync::Arc;

/// Prompt tokens fed in one step. The stall this imposes on everyone else's decode is one chunk's prefill,
/// about a second on the reference machine's 4B model.
pub const PREFILL_CHUNK: usize = 64;

/// Users served at once by default. Above the machine's roofline ridge (about seven rows) another user adds no
/// throughput, and every user's own tokens arrive more slowly, so this is deliberately below it.
pub const DEFAULT_MAX_SLOTS: usize = 4;

/// Why a slot stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Finish {
    /// A stop token, or the end of the turn.
    Stop,
    /// The request's own token limit.
    Length,
    /// The cache has no room left for this conversation.
    Context,
    /// The engine could not run the step.
    Error(String),
}

/// What one request asks of the sampler and when it should stop.
#[derive(Debug, Clone)]
pub struct SlotCfg {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub min_p: f32,
    pub seed: u64,
    pub max_tokens: usize,
    /// Token ids that end the turn.
    pub stop: Vec<u32>,
    /// Report each token's log-probability (the gate's rule reads it).
    pub logprobs: bool,
    // nafs-aql (ShadowPrefill, Phase 30): a shadow request only reads its prompt into the cache, for a request that
    // may come later (the other model of a pair warming the conversation it may be switched to). It is the lowest
    // priority: its chunks run only when no other prompt is waiting, it is never decoded, and a real request that
    // shares its beginning takes over what it has read so far.
    pub shadow: bool,
}

impl Default for SlotCfg {
    fn default() -> Self {
        SlotCfg {
            temperature: 0.0,
            top_p: 0.95,
            top_k: 40,
            min_p: 0.05,
            seed: 42,
            max_tokens: 256,
            stop: vec![],
            logprobs: false,
            shadow: false,
        }
    }
}

/// What one request cost, reported with its last token.
#[derive(Debug, Clone, Default)]
pub struct SlotCost {
    /// Tokens in the prompt, and how many of them had to be prefilled (the rest came from a restored prefix).
    pub prompt_tokens: usize,
    pub prefilled: usize,
    pub produced: usize,
    pub prefill_s: f64,
    /// Decode seconds attributed to this request: each step's wall time divided among the rows that shared it.
    pub decode_s: f64,
}

/// One token of one request, or the end of it.
#[derive(Debug, Clone)]
pub struct Emission {
    pub slot: SlotId,
    pub token: u32,
    /// The token's text, held back when it is a partial UTF-8 sequence.
    pub text: String,
    pub logprob: f32,
    /// Set on the last emission of a request.
    pub finish: Option<Finish>,
    /// Set with `finish`: what the request cost.
    pub cost: Option<SlotCost>,
}

/// A request's handle while it is being served.
pub type SlotId = u64;

struct Slot {
    id: SlotId,
    kv: KvCache,
    cfg: SlotCfg,
    /// The whole prompt. Everything but its last token is prefilled; the last token is the first thing decoded.
    prompt: Vec<u32>,
    /// Prompt tokens already written to the cache.
    fed: usize,
    /// Position the next token will take.
    pos: usize,
    /// The token to feed on the next decode step.
    next: u32,
    produced: usize,
    /// Tokens written to the cache after the prompt (every sampled token is fed back on the next step), so a
    /// finished slot's cache can be described by the token sequence it holds.
    fed_back: Vec<u32>,
    seed: u64,
    utf8: Utf8Stream,
    /// Prompt tokens, how many of them were prefilled (the rest were restored), and the time that took.
    prompt_tokens: usize,
    prefilled: usize,
    prefill_s: f64,
    decode_s: f64,
    // nafs-aql: input vectors for prompt positions (Memorilla's memory tokens), sorted by position. The cache
    // beyond the first of them depends on the vectors, so prefixes are neither restored nor kept past it.
    inputs: Vec<(usize, Vec<f32>)>,
}

impl Slot {
    /// The prompt is in the cache and the slot is ready to decode.
    fn ready(&self) -> bool {
        self.fed + 1 >= self.prompt.len()
    }

    /// The tokens this slot's cache holds, in order: the prompt positions that were prefilled (`fed` of them)
    /// followed by the tokens fed back during decode. Describing the cache any other way keeps it under the wrong
    /// tokens, and a later request reuses a cache that does not match its prompt.
    fn held(&self) -> Vec<u32> {
        let mut tokens: Vec<u32> = self.prompt[..self.fed.min(self.prompt.len())].to_vec();
        tokens.extend(&self.fed_back);
        // nafs-aql: a cache computed from supplied vectors is described by its tokens only up to the first vector.
        if let Some((p, _)) = self.inputs.first() {
            tokens.truncate(*p);
        }
        tokens.truncate(self.kv.filled());
        tokens
    }
}

/// The prompt text of one chat turn: the system block, the earlier exchanges, this turn's message, and the
/// assistant's opening (plus whatever the assistant has already written, for a continuation).
///
/// One builder for both paths, so a prompt served by the batching engine is the same string a single session
/// would have built, token for token — and so a continuation restores the cache of the turn it continues.
pub fn chat_prompt(
    template: crate::template::Template,
    system: Option<&str>,
    pairs: &[(String, String)],
    user: &str,
    thinking: bool,
    assistant_prefix: Option<&str>,
) -> String {
    let mut s = template.system_block(system, &crate::template::today_llama3());
    for (u, a) in pairs {
        s.push_str(&template.user_turn(u));
        s.push_str(&template.assistant_turn(a));
    }
    s.push_str(&template.user_turn(user));
    s.push_str(template.assistant_open(thinking));
    if let Some(p) = assistant_prefix {
        s.push_str(p);
    }
    s
}

/// Several users, one model, one weight pass per step.
pub struct Engine {
    model: Arc<Model>,
    pool: Pool,
    /// nafs-aql: a wider pool for reading prompts, when one was asked for (`with_prefill_threads`). Writing a
    /// token is bound by memory and is fastest a thread short of the cores; reading a prompt is bound by
    /// arithmetic and scales with every core. An idle pool's workers are parked, so the two do not compete.
    prefill_pool: Option<Pool>,
    state: State,
    slots: Vec<Slot>,
    max_slots: usize,
    max_ctx: usize,
    kv_prec: KvPrec,
    /// Caches of finished requests, so a prompt that shares a beginning with one of them (the system prompt
    /// every user of a deployment sends, the transcript a code agent extends) is not prefilled again. Shared
    /// across slots, which is where it pays most: one user's work serves the next.
    // nafs-aql (Phase 34): the caches themselves, moved in when a slot ends and out when a request continues the
    // conversation; see `prefix::CacheStore`.
    prefixes: CacheStore,
    /// Token ids that open a turn under the model's chat template (`<|im_start|>`). With them the engine can tell a
    /// conversation that moved past its last turn (the request takes the cache) from another conversation or another
    /// sample (the request copies from it). Empty: a request takes a cache only when its prompt holds all of it.
    pub turn_starts: Vec<u32>,
    /// Feed the prompt's last token with the last chunk of the prompt and sample the first token from that pass,
    /// instead of in a pass of its own. One pass over the weights fewer per request (measured: 0.05 s on the 1.7B
    /// model, 0.1 s on the 4B). On by default; a switch so the two can be measured against each other.
    pub fuse_first_token: bool,
    next_id: SlotId,
    /// Engine-wide totals, for the server's telemetry.
    pub stats: StepStats,
    /// Decode steps run, and the rows they carried: their ratio is the batching actually achieved.
    pub decode_steps: u64,
    pub decode_rows: u64,
    samp: crate::sample::Scratch,
    // nafs-aql (ShadowPrefill): while real requests are decoding, one shadow chunk runs every `shadow_every` steps
    // (0: shadows run only when nothing else is being served); `since_shadow` counts the steps since the last one.
    pub shadow_every: usize,
    since_shadow: usize,
    /// Shadow requests read to the end, taken over by a real request, and positions they read.
    pub shadow_done: u64,
    pub shadow_taken: u64,
    pub shadow_positions: u64,
}

impl Engine {
    pub fn new(
        model: Arc<Model>,
        threads: usize,
        max_ctx: usize,
        max_slots: usize,
        kv_prec: KvPrec,
        prefix_cache_bytes: usize,
    ) -> Engine {
        Engine::with_prefill_threads(
            model,
            threads,
            threads,
            max_ctx,
            max_slots,
            kv_prec,
            prefix_cache_bytes,
        )
    }

    /// `new`, with prompts read by `prefill_threads` workers and tokens written by `threads`. Measured on four
    /// cores (Qwen3-1.7B, Q4_K_M): a prompt is read at 60 tokens/s by three threads and 72 by four, while
    /// decode is 4 to 8 % faster with three than with four.
    pub fn with_prefill_threads(
        model: Arc<Model>,
        threads: usize,
        prefill_threads: usize,
        max_ctx: usize,
        max_slots: usize,
        kv_prec: KvPrec,
        prefix_cache_bytes: usize,
    ) -> Engine {
        let prefill_threads = prefill_threads.max(1);
        let c = model.cfg;
        let max_slots = max_slots.max(1);
        let max_ctx = max_ctx.clamp(32, c.ctx_train);
        // The widest batch is a prefill chunk or one row per slot, whichever is larger; every decode row wants
        // logits, a prefill chunk wants none.
        let max_batch = PREFILL_CHUNK.max(max_slots);
        let mut state = State::new(
            &c,
            max_batch,
            max_slots,
            max_ctx,
            // The per-worker scratch must serve the wider of the two pools.
            threads.max(prefill_threads),
            model.rope_factors.as_deref(),
        );
        // A LittleBit sidecar's projections need scratch for their intermediates; without it the first one
        // writes through an empty buffer.
        let (lx, lt) = model.lb_scratch(max_batch);
        state.set_lb_scratch(lx, lt);
        Engine {
            model,
            pool: Pool::new(threads),
            prefill_pool: (prefill_threads != threads).then(|| Pool::new(prefill_threads)),
            state,
            slots: Vec::with_capacity(max_slots),
            max_slots,
            max_ctx,
            kv_prec,
            prefixes: CacheStore::new(prefix_cache_bytes),
            turn_starts: Vec::new(),
            fuse_first_token: true,
            next_id: 1,
            stats: StepStats::default(),
            decode_steps: 0,
            decode_rows: 0,
            samp: crate::sample::Scratch::default(),
            shadow_every: 0,
            since_shadow: 0,
            shadow_done: 0,
            shadow_taken: 0,
            shadow_positions: 0,
        }
    }

    pub fn max_slots(&self) -> usize {
        self.max_slots
    }
    pub fn active(&self) -> usize {
        self.slots.len()
    }
    pub fn has_room(&self) -> bool {
        self.slots.len() < self.max_slots
    }
    pub fn max_ctx(&self) -> usize {
        self.max_ctx
    }
    /// Bytes one slot's cache holds, for sizing a deployment.
    pub fn cache_bytes_per_slot(&self) -> u64 {
        let c = self.model.cfg;
        KvCache::new(self.kv_prec, c.n_layer, c.n_kv_head, c.head_dim, 1).bytes_per_position()
            * self.max_ctx as u64
    }

    /// Take a request, or give it back when every slot is busy or the prompt cannot fit.
    pub fn admit(&mut self, prompt: Vec<u32>, cfg: SlotCfg) -> Result<SlotId, (Vec<u32>, String)> {
        self.admit_with_inputs(prompt, cfg, Vec::new())
    }

    // nafs-aql: a request whose prompt positions in `inputs` take supplied vectors instead of their tokens'
    // embeddings (Memorilla, Phase 20). The positions must lie before the prompt's last token, which is decoded,
    // not prefilled.
    pub fn admit_with_inputs(
        &mut self,
        prompt: Vec<u32>,
        cfg: SlotCfg,
        mut inputs: Vec<(usize, Vec<f32>)>,
    ) -> Result<SlotId, (Vec<u32>, String)> {
        inputs.sort_by_key(|x| x.0);
        let d = self.model.cfg.d_model;
        if let Some((p, v)) = inputs
            .iter()
            .find(|(p, v)| *p + 1 >= prompt.len() || v.len() != d)
        {
            let msg = format!(
                "input vector at position {p} ({} values) does not fit a {}-token prompt of a {d}-wide model",
                v.len(),
                prompt.len()
            );
            return Err((prompt, msg));
        }
        // nafs-aql (ShadowPrefill): a real request never waits for a shadow. A shadow that has read part of this
        // prompt hands it over (cancelling keeps what it read, and the lookup below restores it); a full engine
        // gives up its oldest shadow to make room.
        if !cfg.shadow {
            let sharing: Vec<SlotId> = self
                .slots
                .iter()
                .filter(|s| {
                    s.cfg.shadow
                        && common_prefix(&s.prompt[..s.fed.min(s.prompt.len())], &prompt)
                            >= MIN_PREFIX
                })
                .map(|s| s.id)
                .collect();
            for id in sharing {
                self.shadow_taken += 1;
                self.cancel(id);
            }
            if !self.has_room() {
                if let Some(id) = self.slots.iter().find(|s| s.cfg.shadow).map(|s| s.id) {
                    self.cancel(id);
                }
            }
        }
        if !self.has_room() {
            return Err((prompt, "every slot is busy".into()));
        }
        if prompt.is_empty() {
            return Err((prompt, "the prompt is empty".into()));
        }
        if prompt.len() + 1 > self.max_ctx {
            let n = prompt.len();
            return Err((
                prompt,
                format!(
                    "prompt of {n} tokens does not fit in {} positions",
                    self.max_ctx
                ),
            ));
        }
        let c = self.model.cfg;
        let id = self.next_id;
        self.next_id += 1;
        let fresh =
            |e: &Engine| KvCache::new(e.kv_prec, c.n_layer, c.n_kv_head, c.head_dim, e.max_ctx);
        // The last prompt token is never prefilled (it is fed by the pass the first token is sampled from), so a
        // prefix may cover everything before it.
        // nafs-aql: nothing past the first supplied vector comes from the store (it was computed from other vectors).
        let cap = inputs
            .first()
            .map_or(prompt.len() - 1, |x| x.0.min(prompt.len() - 1));
        // nafs-aql (Phase 34): the cache this request starts from. A kept conversation it continues is taken whole
        // (no copy); one somebody else may still want is copied from; and so is the cache of a request still being
        // served (two users under one system prompt who arrive together: the positions a slot has already written
        // never change, so they can be read while it goes on).
        let kept = self.prefixes.find(&prompt, cap);
        let live = self
            .slots
            .iter()
            .enumerate()
            .map(|(i, s)| (i, common_prefix(&s.held(), &prompt).min(cap)))
            .max_by_key(|&(_, l)| l)
            .filter(|&(_, l)| l >= MIN_PREFIX);
        let (kv, fed) = match (kept, live) {
            (Some((i, l)), live) if live.is_none_or(|(_, ll)| l >= ll) => {
                // A prompt with supplied vectors never takes a cache: its own stops at the first vector.
                if inputs.is_empty()
                    && supersedes(self.prefixes.tokens(i), &prompt, &self.turn_starts)
                {
                    let mut kv = self.prefixes.take(i);
                    kv.truncate(l);
                    self.prefixes.note_hit(l, true);
                    (kv, l)
                } else {
                    let mut kv = fresh(self);
                    if kv.copy_prefix_from(self.prefixes.cache(i), l) {
                        self.prefixes.note_hit(l, false);
                        (kv, l)
                    } else {
                        (kv, 0)
                    }
                }
            }
            (_, Some((i, l))) => {
                let mut kv = fresh(self);
                if kv.copy_prefix_from(&self.slots[i].kv, l) {
                    self.prefixes.note_hit(l, false);
                    (kv, l)
                } else {
                    (kv, 0)
                }
            }
            _ => (fresh(self), 0),
        };
        // A reused prefix can leave nothing to prefill, in which case the first decode step must still feed
        // the prompt's last token: `prefill_chunk`, which normally feeds or sets it, never runs.
        let next = if fed + 1 >= prompt.len() {
            prompt[prompt.len() - 1]
        } else {
            0
        };
        self.slots.push(Slot {
            id,
            kv,
            // The draw is keyed by (seed, position), so a user's tokens do not depend on who else is being served.
            seed: cfg.seed.max(1),
            cfg,
            prompt_tokens: prompt.len(),
            prefilled: 0,
            prompt,
            fed,
            pos: fed,
            next,
            produced: 0,
            fed_back: vec![],
            utf8: Utf8Stream::default(),
            prefill_s: 0.0,
            decode_s: 0.0,
            inputs,
        });
        Ok(id)
    }

    /// Drop a request (its caller has gone, or it is being replanned); its cache is kept for whoever shares its
    /// beginning next.
    pub fn cancel(&mut self, id: SlotId) {
        if let Some(i) = self.slots.iter().position(|s| s.id == id) {
            let s = self.slots.remove(i);
            self.keep_prefix(s);
        }
    }

    /// Keep what a slot computed: the prompt positions that are in its cache, and the tokens it produced. The
    /// cache itself goes to the store; nothing is copied.
    fn keep_prefix(&mut self, s: Slot) {
        let tokens = s.held();
        if tokens.len() < MIN_PREFIX || self.prefixes.covers(&tokens) {
            return;
        }
        let mut kv = s.kv;
        kv.truncate(tokens.len());
        self.prefixes.insert(tokens, kv);
    }

    // nafs-aql (Phase 34): how reuses were served: requests that took a kept cache, requests that copied from one,
    // and caches evicted for the budget.
    pub fn handoff_stats(&self) -> (u64, u64, u64) {
        (
            self.prefixes.taken,
            self.prefixes.copied,
            self.prefixes.evicted,
        )
    }

    /// The prefix store's budget in bytes.
    pub fn prefix_budget(&self) -> usize {
        self.prefixes.budget()
    }

    // nafs-aql (Phase 34, after FreeToken's runtime cache reconfiguration): change the memory the prefix store may
    // hold while the engine is serving. Call it between steps (the engine thread does); requests being served keep
    // their own caches and are not touched. Shrinking evicts at once, least recently used first.
    pub fn set_prefix_budget(&mut self, bytes: usize) {
        self.prefixes.set_budget(bytes);
    }

    /// Drop every kept cache (the memory goes back to the system); requests being served are not touched.
    pub fn flush_prefixes(&mut self) {
        self.prefixes.clear();
    }

    /// Where the engine's time has gone, in milliseconds: matmul, output head, attention, other, sampling,
    /// drafting. Engine-wide rather than per request: a decode step is shared by every row in it.
    pub fn phases_ms(&self) -> [f64; 6] {
        let p = &self.stats;
        [p.ns_matmul, p.ns_head, p.ns_attn, p.ns_other(), 0, 0].map(|ns| ns as f64 / 1e6)
    }

    /// (entries, bytes, restores, positions not prefilled) of the shared prefix store.
    pub fn prefix_stats(&self) -> (usize, usize, u64, u64) {
        (
            self.prefixes.entries(),
            self.prefixes.bytes(),
            self.prefixes.hits,
            self.prefixes.positions_saved,
        )
    }

    /// One scheduler iteration: a chunk of one prompt, or one token for every ready user. Returns what to send
    /// to each caller. An empty result with no active slots means there is nothing to do.
    pub fn step(&mut self) -> Vec<Emission> {
        if self.slots.is_empty() {
            return vec![];
        }
        // nafs-aql (ShadowPrefill): a shadow that has read its whole prompt is done; its cache goes to the store.
        if let Some(i) = self.slots.iter().position(|s| s.cfg.shadow && s.ready()) {
            let s = self.slots.remove(i);
            self.shadow_done += 1;
            self.shadow_positions += s.prefilled as u64;
            let done = Emission {
                slot: s.id,
                token: 0,
                text: String::new(),
                logprob: 0.0,
                finish: Some(Finish::Stop),
                cost: Some(SlotCost {
                    prompt_tokens: s.prompt_tokens,
                    prefilled: s.prefilled,
                    produced: 0,
                    prefill_s: s.prefill_s,
                    decode_s: 0.0,
                }),
            };
            self.keep_prefix(s);
            return vec![done];
        }
        let real = self.slots.iter().any(|s| !s.cfg.shadow);
        // A real prompt that is not yet in the cache comes first, oldest slot first: a user waiting for their first
        // token is the one most sensitive to delay, and one chunk is a bounded interruption for the others.
        let pending = self.slots.iter().position(|s| !s.ready() && !s.cfg.shadow);
        // A shadow chunk runs when nothing else is served, or once every `shadow_every` steps of decoding.
        let shadow = self.slots.iter().position(|s| !s.ready() && s.cfg.shadow);
        let shadow_turn = match (real, shadow) {
            (_, None) => false,
            (false, Some(_)) => true,
            (true, Some(_)) => {
                self.since_shadow += 1;
                pending.is_none() && self.shadow_every > 0 && self.since_shadow >= self.shadow_every
            }
        };
        if shadow_turn {
            self.since_shadow = 0;
        }
        if let Some(i) = pending.or(shadow.filter(|_| shadow_turn)) {
            return match self.prefill_chunk(i) {
                Err(e) => {
                    // A pass is refused before it writes anything, or fails after writing whole positions the slot
                    // has not booked; either way the cache still holds what the slot says it holds, and since the
                    // request may have taken that cache from the store, it goes back there.
                    let s = self.slots.remove(i);
                    let id = s.id;
                    self.keep_prefix(s);
                    vec![Emission {
                        slot: id,
                        token: 0,
                        text: String::new(),
                        logprob: 0.0,
                        finish: Some(Finish::Error(e)),
                        cost: None,
                    }]
                }
                Ok(None) => vec![],
                // The chunk reached the end of the prompt and carried the first token with it.
                Ok(Some(em)) => {
                    if em.finish.is_some() {
                        let s = self.slots.remove(i);
                        self.keep_prefix(s);
                    }
                    vec![em]
                }
            };
        }
        if !real {
            return vec![];
        }
        // Shadows are never decoded: they step aside while the real rows take their step.
        if self.slots.iter().any(|s| s.cfg.shadow) {
            let (parked, rest): (Vec<Slot>, Vec<Slot>) = std::mem::take(&mut self.slots)
                .into_iter()
                .partition(|s| s.cfg.shadow);
            self.slots = rest;
            let out = self.decode();
            self.slots.extend(parked);
            return out;
        }
        self.decode()
    }

    // nafs-aql (ShadowPrefill): whether the engine holds work other than shadows, and whether it holds a shadow.
    pub fn real_active(&self) -> usize {
        self.slots.iter().filter(|s| !s.cfg.shadow).count()
    }
    pub fn shadow_active(&self) -> usize {
        self.slots.iter().filter(|s| s.cfg.shadow).count()
    }

    /// Feed the next chunk of slot `i`'s prompt.
    ///
    // nafs-aql (Phase 34): the chunk that reaches the end of the prompt carries the prompt's last token as well and
    // asks for that row's logits, so the first token is sampled from the same pass over the weights (`Some`). Before,
    // the last token always had a pass of its own: a request with a 7-token suffix read the weights twice. A chunk
    // that does not reach the end, a shadow (never decoded) and a full chunk leave the last token for a decode step.
    fn prefill_chunk(&mut self, i: usize) -> Result<Option<Emission>, String> {
        let t0 = std::time::Instant::now();
        let s = &mut self.slots[i];
        let last = s.prompt.len() - 1;
        let left = last - s.fed;
        let take = PREFILL_CHUNK.min(left);
        let fuse = self.fuse_first_token && !s.cfg.shadow && left < PREFILL_CHUNK;
        let rows = take + fuse as usize;
        let chunk: Vec<u32> = s.prompt[s.fed..s.fed + rows].to_vec();
        let start = s.fed;
        // nafs-aql: this chunk's supplied vectors, by row (never the last token's: `admit_with_inputs` refuses that).
        self.state.inputs = s
            .inputs
            .iter()
            .filter(|(p, _)| *p >= start && *p < start + take)
            .map(|(p, v)| (p - start, v.clone()))
            .collect();
        let want = [take];
        self.model.forward(
            &mut self.state,
            &mut self.slots[i].kv,
            self.prefill_pool.as_ref().unwrap_or(&self.pool),
            &chunk,
            start,
            if fuse { &want } else { &[] },
            &mut self.stats,
        )?;
        let dt = t0.elapsed().as_secs_f64();
        let s = &mut self.slots[i];
        s.fed += take;
        s.prefilled += take;
        s.pos = s.fed;
        if !fuse {
            s.prefill_s += dt;
            if s.ready() {
                s.next = s.prompt[last];
            }
            return Ok(None);
        }
        // The pass's time is shared by its rows; one of them was the first decode step.
        let share = dt / rows as f64;
        s.prefill_s += dt - share;
        let fed = s.prompt[last];
        Ok(Some(self.emit(i, 0, fed, share)))
    }

    /// Sample slot `i`'s next token from logits row `row` of the pass that just fed `fed` at the slot's position,
    /// and book it: the emission its caller is sent, with `finish` and the request's cost when it ends here.
    fn emit(&mut self, i: usize, row: usize, fed: u32, share: f64) -> Emission {
        let n_vocab = self.model.cfg.n_vocab;
        let s = &mut self.slots[i];
        let logits = self.state.logits_row(row, n_vocab);
        let id = crate::sample::sample_filtered(
            logits,
            s.cfg.temperature,
            s.cfg.top_p,
            s.cfg.top_k,
            s.cfg.min_p,
            crate::sample::position_key(s.seed, 0, s.pos + 1),
            &mut self.samp,
        );
        let logprob = if s.cfg.logprobs {
            crate::session::logprob_of(logits, id)
        } else {
            0.0
        };
        // The token just fed sat at `pos`; after the prompt's own positions it is part of what the cache
        // now describes.
        if s.pos >= s.prompt.len() - 1 {
            s.fed_back.push(fed);
        }
        s.pos += 1;
        s.next = id;
        s.produced += 1;
        s.decode_s += share;
        let stop = s.cfg.stop.contains(&id);
        // The stop token itself is not part of the answer.
        let text = if stop {
            String::new()
        } else {
            s.utf8.push(&self.model.tk.token_bytes(id))
        };
        let finish = if stop {
            Some(Finish::Stop)
        } else if s.produced >= s.cfg.max_tokens {
            Some(Finish::Length)
        } else if s.pos + 1 >= self.max_ctx {
            Some(Finish::Context)
        } else {
            None
        };
        let cost = finish.is_some().then_some(SlotCost {
            prompt_tokens: s.prompt_tokens,
            prefilled: s.prefilled,
            produced: s.produced,
            prefill_s: s.prefill_s,
            decode_s: s.decode_s,
        });
        Emission {
            slot: s.id,
            token: id,
            text,
            logprob,
            finish,
            cost,
        }
    }

    /// One token for every ready slot, in one pass over the weights.
    fn decode(&mut self) -> Vec<Emission> {
        let t0 = std::time::Instant::now();
        let n = self.slots.len();
        let tokens: Vec<u32> = self.slots.iter().map(|s| s.next).collect();
        let positions: Vec<usize> = self.slots.iter().map(|s| s.pos).collect();
        let want: Vec<usize> = (0..n).collect();
        let mut caches: Vec<&mut KvCache> = self.slots.iter_mut().map(|s| &mut s.kv).collect();
        let r = self.model.forward_rows(
            &mut self.state,
            &mut Caches::PerRow(&mut caches),
            &self.pool,
            &tokens,
            &positions,
            &want,
            &mut self.stats,
        );
        if let Err(e) = r {
            // A failed step is the whole batch's failure: every caller is told, and the slots go.
            let out = self
                .slots
                .iter()
                .map(|s| Emission {
                    slot: s.id,
                    token: 0,
                    text: String::new(),
                    logprob: 0.0,
                    finish: Some(Finish::Error(e.clone())),
                    cost: None,
                })
                .collect();
            // Their caches go back to the store (see `step`): a request that took a conversation's cache must not
            // lose it to an error.
            for s in std::mem::take(&mut self.slots) {
                self.keep_prefix(s);
            }
            return out;
        }
        self.decode_steps += 1;
        self.decode_rows += n as u64;
        let share = t0.elapsed().as_secs_f64() / n as f64;
        let out: Vec<Emission> = (0..n).map(|b| self.emit(b, b, tokens[b], share)).collect();
        let finished: Vec<SlotId> = out
            .iter()
            .filter(|em| em.finish.is_some())
            .map(|em| em.slot)
            .collect();
        if !finished.is_empty() {
            let (gone, keep): (Vec<Slot>, Vec<Slot>) = std::mem::take(&mut self.slots)
                .into_iter()
                .partition(|s| finished.contains(&s.id));
            self.slots = keep;
            for s in gone {
                self.keep_prefix(s);
            }
        }
        out
    }

    /// What one request cost: prompt tokens, tokens produced, and the seconds spent prefilling and decoding it
    /// (a decode step's time divided among the rows that shared it).
    pub fn slot_cost(&self, id: SlotId) -> Option<(usize, usize, f64, f64)> {
        self.slots
            .iter()
            .find(|s| s.id == id)
            .map(|s| (s.prompt_tokens, s.produced, s.prefill_s, s.decode_s))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(max_tokens: usize) -> SlotCfg {
        SlotCfg {
            max_tokens,
            ..SlotCfg::default()
        }
    }

    /// The model these tests run against: `SMILEY_CORE_MODEL`, else the repository's own System 1 weights.
    fn model() -> Option<Arc<Model>> {
        let p = match std::env::var_os("SMILEY_CORE_MODEL") {
            Some(p) => std::path::PathBuf::from(p),
            None => std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../../models/Qwen3-1.7B-Q4_K_M.gguf"),
        };
        p.exists().then(|| crate::model::keep::open(&p).ok())?
    }

    /// Two users served together must each get exactly the text they would get alone.
    #[test]
    fn users_served_together_get_what_they_would_get_alone() {
        let Some(m) = model() else {
            eprintln!("skipped: set SMILEY_CORE_MODEL");
            return;
        };
        let prompts = [
            vec![1u32, 450, 29871, 13, 5618],
            vec![1u32, 910, 338, 263, 1243],
        ];
        let steps = 8;
        let solo: Vec<String> = prompts
            .iter()
            .map(|p| {
                let mut e = Engine::new(m.clone(), 3, 256, 1, KvPrec::F16, 0);
                e.admit(p.clone(), cfg(steps)).unwrap();
                let mut text = String::new();
                while e.active() > 0 {
                    for em in e.step() {
                        text.push_str(&em.text);
                    }
                }
                text
            })
            .collect();

        let mut e = Engine::new(m.clone(), 3, 256, 2, KvPrec::F16, 1 << 26);
        let a = e.admit(prompts[0].clone(), cfg(steps)).unwrap();
        let b = e.admit(prompts[1].clone(), cfg(steps)).unwrap();
        let (mut ta, mut tb) = (String::new(), String::new());
        while e.active() > 0 {
            for em in e.step() {
                if em.slot == a {
                    ta.push_str(&em.text);
                } else if em.slot == b {
                    tb.push_str(&em.text);
                }
            }
        }
        assert_eq!(ta, solo[0], "user A's text changed when sharing the engine");
        assert_eq!(tb, solo[1], "user B's text changed when sharing the engine");
        assert!(
            e.decode_rows > e.decode_steps,
            "the users never shared a pass"
        );
    }

    /// nafs-aql: a prompt read by a wider pool gives the answer it gives when read by the decode pool. The
    /// weight products do not depend on how the rows are shared out; attention's partial sums do, in the last
    /// bits, so this compares the greedy text of a prompt long enough to be read in chunks.
    #[test]
    fn a_prompt_read_by_a_wider_pool_gets_the_same_answer() {
        let Some(m) = model() else {
            eprintln!("skipped: set SMILEY_CORE_MODEL");
            return;
        };
        let prompt: Vec<u32> = (0..150u32).map(|i| (i * 37) % 1000 + 5).collect();
        let run = |mut e: Engine| {
            e.admit(prompt.clone(), cfg(8)).unwrap();
            let mut text = String::new();
            while e.active() > 0 {
                for em in e.step() {
                    text.push_str(&em.text);
                }
            }
            text
        };
        let narrow = run(Engine::new(m.clone(), 3, 256, 1, KvPrec::F16, 0));
        let wide = Engine::with_prefill_threads(m.clone(), 3, 4, 256, 1, KvPrec::F16, 0);
        assert_eq!(wide.prefill_pool.as_ref().map(Pool::width), Some(4));
        assert_eq!(wide.pool.width(), 3);
        assert_eq!(run(wide), narrow);
        // The same width for both makes no second pool.
        assert!(
            Engine::with_prefill_threads(m, 3, 3, 256, 1, KvPrec::F16, 0)
                .prefill_pool
                .is_none()
        );
    }

    /// A user that arrives while another is already decoding joins the batch and is served correctly.
    #[test]
    fn a_user_can_arrive_mid_flight() {
        let Some(m) = model() else {
            eprintln!("skipped: set SMILEY_CORE_MODEL");
            return;
        };
        let late = vec![1u32, 910, 338, 263, 1243];
        let mut solo = Engine::new(m.clone(), 3, 256, 1, KvPrec::F16, 0);
        solo.admit(late.clone(), cfg(6)).unwrap();
        let mut want = String::new();
        while solo.active() > 0 {
            for em in solo.step() {
                want.push_str(&em.text);
            }
        }

        let mut e = Engine::new(m.clone(), 3, 256, 2, KvPrec::F16, 1 << 26);
        let first = e.admit(vec![1u32, 450, 29871, 13, 5618], cfg(20)).unwrap();
        for _ in 0..6 {
            e.step();
        }
        let second = e.admit(late, cfg(6)).unwrap();
        let mut got = String::new();
        while e.active() > 0 {
            for em in e.step() {
                if em.slot == second {
                    got.push_str(&em.text);
                }
            }
        }
        assert_eq!(
            got, want,
            "a user that arrived later was served differently"
        );
        assert!(first != second);
    }

    /// A second user whose prompt begins like the first one's does not prefill that beginning again.
    #[test]
    fn a_shared_beginning_is_prefilled_once_for_everyone() {
        let Some(m) = model() else {
            eprintln!("skipped: set SMILEY_CORE_MODEL");
            return;
        };
        // A long shared opening (a system prompt), then a different question each.
        let opening: Vec<u32> = (0..80).map(|i| ((i * 13) % 900 + 11) as u32).collect();
        let run = |e: &mut Engine, tail: u32| {
            let mut p = opening.clone();
            p.push(tail);
            let id = e.admit(p, cfg(3)).unwrap();
            let mut text = String::new();
            while e.active() > 0 {
                for em in e.step() {
                    if em.slot == id {
                        text.push_str(&em.text);
                    }
                }
            }
            text
        };
        let mut e = Engine::new(m.clone(), 3, 256, 1, KvPrec::F16, 1 << 26);
        run(&mut e, 5618);
        let before = e.stats.bytes_weights;
        let second = run(&mut e, 6324);
        let (entries, _, hits, saved) = e.prefix_stats();
        assert!(
            entries > 0 && hits > 0,
            "the second user did not restore a prefix"
        );
        assert!(
            saved as usize >= opening.len() - 1,
            "only {saved} positions were reused"
        );
        // With the opening already in the cache, the second user's prefill is a chunk of weights, not three.
        let second_cost = e.stats.bytes_weights - before;
        assert!(
            second_cost < before,
            "the second user read as many weights as the first ({second_cost} vs {before})"
        );
        // And it is still the right answer: the same question on a cold engine gives the same text.
        let mut cold = Engine::new(m, 3, 256, 1, KvPrec::F16, 0);
        assert_eq!(
            run(&mut cold, 6324),
            second,
            "a restored prefix changed the answer"
        );
    }

    #[test]
    fn supplied_input_vectors_are_used_as_given() {
        // nafs-aql: giving rows their own token embeddings changes nothing; giving them other vectors changes the
        // answer, and such a prompt is never restored from (or stored as) a token prefix past the first vector.
        let Some(m) = model() else { return };
        let mut e = Engine::new(m.clone(), 2, 512, 2, KvPrec::F16, 64 << 20);
        let prompt = m.tk.encode(
            "The capital of France is Paris. The capital of Italy is Rome. The capital of Spain is",
        );
        let run = |e: &mut Engine, inputs: Vec<(usize, Vec<f32>)>| -> Vec<u32> {
            let id = e.admit_with_inputs(prompt.clone(), cfg(8), inputs).unwrap();
            let mut out = vec![];
            loop {
                for em in e.step() {
                    assert_eq!(em.slot, id);
                    out.push(em.token);
                    if em.finish.is_some() {
                        return out;
                    }
                }
            }
        };
        let plain = run(&mut e, vec![]);
        let own: Vec<(usize, Vec<f32>)> =
            (3..9).map(|p| (p, m.token_embedding(prompt[p]))).collect();
        assert_eq!(
            run(&mut e, own),
            plain,
            "a row's own embedding must give the same tokens"
        );
        let (_, _, hits_before, saved_before) = e.prefix_stats();
        let noise: Vec<(usize, Vec<f32>)> =
            (3..9).map(|p| (p, vec![0.05; m.cfg.d_model])).collect();
        let _ = run(&mut e, noise);
        let (_, _, hits, saved) = e.prefix_stats();
        // Whatever was restored stops before position 3.
        assert!(
            saved - saved_before <= 3 * (hits - hits_before),
            "restored past the first supplied vector"
        );
        assert!(e
            .admit_with_inputs(
                prompt.clone(),
                cfg(8),
                vec![(prompt.len() - 1, vec![0.0; m.cfg.d_model])]
            )
            .is_err());
        assert!(e
            .admit_with_inputs(prompt.clone(), cfg(8), vec![(2, vec![0.0; 7])])
            .is_err());
    }

    // nafs-aql (ShadowPrefill): a shadow reads its prompt only when nothing else is waiting, is never decoded, and a
    // real request with the same beginning takes over what it read; the real request's tokens are what a cold
    // engine produces.
    #[test]
    fn a_shadow_warms_the_cache_and_a_real_request_takes_it_over() {
        let Some(m) = model() else {
            eprintln!("skipped: set SMILEY_CORE_MODEL");
            return;
        };
        let prompt: Vec<u32> = (0..300u32).map(|i| 300 + (i * 7) % 900).collect();
        let mut real = prompt.clone();
        real.extend([450u32, 29871, 1576]);
        let run_to_end = |e: &mut Engine, id: SlotId| -> Vec<u32> {
            let mut out = vec![];
            for _ in 0..200 {
                for em in e.step() {
                    if em.slot == id {
                        if em.finish.is_none() || em.token != 0 {
                            out.push(em.token);
                        }
                        if em.finish.is_some() {
                            return out;
                        }
                    }
                }
            }
            out
        };
        // Cold reference.
        let mut cold = Engine::new(m.clone(), 2, 512, 1, KvPrec::F16, 64 << 20);
        let id = cold.admit(real.clone(), cfg(6)).unwrap();
        let want = run_to_end(&mut cold, id);
        // A shadow alone reads to the end, then a real request restores it.
        let mut e = Engine::new(m.clone(), 2, 512, 1, KvPrec::F16, 64 << 20);
        let sh = e
            .admit(
                prompt.clone(),
                SlotCfg {
                    shadow: true,
                    ..cfg(6)
                },
            )
            .unwrap();
        let mut done = None;
        for _ in 0..64 {
            for em in e.step() {
                if em.slot == sh && em.finish.is_some() {
                    done = em.cost;
                }
            }
            if done.is_some() {
                break;
            }
        }
        let cost = done.expect("the shadow finished");
        assert_eq!(
            cost.prefilled,
            prompt.len() - 1,
            "a shadow reads every position but its last"
        );
        assert_eq!(cost.produced, 0, "a shadow is never decoded");
        assert_eq!(e.shadow_done, 1);
        let (_, _, hits0, _) = e.prefix_stats();
        let id = e.admit(real.clone(), cfg(6)).unwrap();
        assert_eq!(
            run_to_end(&mut e, id),
            want,
            "a warm request answers as a cold one"
        );
        let (_, _, hits1, saved) = e.prefix_stats();
        assert!(
            hits1 > hits0 && saved as usize >= prompt.len() - 2,
            "the shadow's cache was not restored"
        );
        // A shadow cut short: the real request arrives after one chunk and takes over the part read.
        let mut f = Engine::new(m, 2, 512, 1, KvPrec::F16, 64 << 20);
        f.admit(
            prompt.clone(),
            SlotCfg {
                shadow: true,
                ..cfg(6)
            },
        )
        .unwrap();
        let _ = f.step();
        let id = f.admit(real.clone(), cfg(6)).unwrap();
        assert_eq!(
            f.shadow_taken, 1,
            "the real request did not take the shadow over"
        );
        assert_eq!(f.shadow_active(), 0);
        let (_, _, hits, _) = f.prefix_stats();
        assert_eq!(hits, 1, "the partial shadow was not restored");
        assert_eq!(run_to_end(&mut f, id), want);
    }

    #[test]
    fn a_full_engine_refuses_and_a_cancelled_slot_frees_its_room() {
        let Some(m) = model() else {
            eprintln!("skipped: set SMILEY_CORE_MODEL");
            return;
        };
        let mut e = Engine::new(m, 2, 128, 1, KvPrec::F16, 0);
        let a = e.admit(vec![1u32, 450, 29871], cfg(4)).unwrap();
        let refused = e.admit(vec![1u32, 450], cfg(4));
        assert!(refused.is_err(), "a full engine admitted a second request");
        assert_eq!(
            refused.unwrap_err().0,
            vec![1u32, 450],
            "the prompt is handed back"
        );
        e.cancel(a);
        assert_eq!(e.active(), 0);
        assert!(
            e.admit(vec![1u32, 450], cfg(4)).is_ok(),
            "cancelling did not free the slot"
        );
        // A prompt longer than the context is refused rather than truncated.
        let mut small = Engine::new(e.model.clone(), 2, 32, 1, KvPrec::F16, 0);
        assert!(small.admit(vec![7u32; 64], cfg(4)).is_err());
    }

    // ---- nafs-aql (Phase 34): the cache store that moves caches, and the first token from the last prompt chunk ----

    /// Serve `prompt` to the end on `e`: the tokens written, and how many prompt tokens were prefilled.
    fn serve(e: &mut Engine, prompt: &[u32], max_tokens: usize) -> (Vec<u32>, usize) {
        let id = e.admit(prompt.to_vec(), cfg(max_tokens)).unwrap();
        let mut out = vec![];
        loop {
            for em in e.step() {
                if em.slot != id {
                    continue;
                }
                out.push(em.token);
                if let Some(c) = em.cost {
                    return (out, c.prefilled);
                }
            }
        }
    }

    fn opening(n: usize) -> Vec<u32> {
        (0..n).map(|i| ((i * 13) % 900 + 11) as u32).collect()
    }

    /// The next step of a transcript takes the kept cache itself: nothing is copied, only the new tokens are read,
    /// and the answer is the one a cold engine gives.
    #[test]
    fn a_continued_conversation_takes_its_cache_without_a_copy() {
        let Some(m) = model() else {
            eprintln!("skipped: set SMILEY_CORE_MODEL");
            return;
        };
        let mut e = Engine::new(m.clone(), 3, 512, 1, KvPrec::F16, 64 << 20);
        let first = opening(80);
        let (written, read) = serve(&mut e, &first, 4);
        assert_eq!(
            read, 79,
            "a cold prompt is read whole, but for its last token"
        );
        // What the transcript is now: the prompt, what was written, and two new tokens.
        let mut next = first.clone();
        next.extend(&written);
        next.extend([450u32, 29871]);
        let (taken, copied, _) = e.handoff_stats();
        let (got, read) = serve(&mut e, &next, 4);
        assert_eq!(
            e.handoff_stats(),
            (taken + 1, copied, 0),
            "the continuation did not take the cache"
        );
        // The cache held the prompt and every written token but the last, which was sampled and never fed.
        assert_eq!(
            read, 2,
            "only the last written token and one new token are read"
        );
        assert_eq!(e.prefix_stats().0, 1, "the cache went back to the store");
        let mut cold = Engine::new(m, 3, 512, 1, KvPrec::F16, 0);
        assert_eq!(
            serve(&mut cold, &next, 4).0,
            got,
            "a taken cache changed the answer"
        );
    }

    /// A second sample of a prompt copies from the first one's cache and leaves it; a prompt that rewrote the kept
    /// conversation's last turn and went past it takes the cache; both answer as a cold engine does.
    #[test]
    fn a_rewritten_last_turn_is_taken_and_another_sample_copies() {
        let Some(m) = model() else {
            eprintln!("skipped: set SMILEY_CORE_MODEL");
            return;
        };
        let Some(open) = m.tk.id_of("<|im_start|>") else {
            eprintln!("skipped: the model has no <|im_start|> token");
            return;
        };
        // system and user turns, then the assistant's header.
        let mut prompt = vec![open];
        prompt.extend(opening(30));
        prompt.push(open);
        prompt.extend(opening(20).iter().map(|t| t + 1000));
        prompt.extend([open, 77]);
        let mut e = Engine::new(m.clone(), 3, 512, 1, KvPrec::F16, 64 << 20);
        e.turn_starts = vec![open];
        let (reply, _) = serve(&mut e, &prompt, 5);
        if reply.contains(&open) {
            eprintln!("skipped: the model opened a turn by itself");
            return;
        }
        // Another sample of the same prompt: the first reply may be the one the conversation continues from.
        let (again, read) = serve(&mut e, &prompt, 5);
        assert_eq!(e.handoff_stats(), (0, 1, 0), "a second sample must copy");
        assert_eq!(read, 0, "the whole prompt was reused");
        assert_eq!(again, reply, "the same seed writes the same reply");
        assert_eq!(e.prefix_stats().0, 1);
        // The next turn replays the reply in another form (here: other tokens) and asks again.
        let mut next = prompt.clone();
        next.extend([if reply[0] == 900 { 901 } else { 900 }, 902]);
        next.extend([open, 5, 6, 7, open, 77]);
        let (got, read) = serve(&mut e, &next, 4);
        assert_eq!(
            e.handoff_stats(),
            (1, 1, 0),
            "the rewritten turn's cache was not taken"
        );
        assert_eq!(read, next.len() - 1 - prompt.len());
        let mut cold = Engine::new(m.clone(), 3, 512, 1, KvPrec::F16, 0);
        assert_eq!(serve(&mut cold, &next, 4).0, got);
        // Without the turn token the engine cannot tell an edit from another conversation, and copies.
        let mut plain = Engine::new(m, 3, 512, 1, KvPrec::F16, 64 << 20);
        serve(&mut plain, &prompt, 5);
        serve(&mut plain, &next, 4);
        assert_eq!(plain.handoff_stats(), (0, 1, 0));
    }

    /// A request that arrives while another with the same beginning is still being served reads that beginning
    /// from the other's cache.
    #[test]
    fn a_request_copies_a_beginning_from_one_still_being_served() {
        let Some(m) = model() else {
            eprintln!("skipped: set SMILEY_CORE_MODEL");
            return;
        };
        let (mut pa, mut pb) = (opening(80), opening(80));
        pa.push(5618);
        pb.push(6324);
        let mut cold = Engine::new(m.clone(), 3, 512, 1, KvPrec::F16, 0);
        let want = serve(&mut cold, &pb, 3).0;
        let mut e = Engine::new(m, 3, 512, 2, KvPrec::F16, 64 << 20);
        let a = e.admit(pa, cfg(40)).unwrap();
        // Until the first user has written a token: its prompt is in its cache.
        while !e.step().iter().any(|em| em.slot == a) {}
        let b = e.admit(pb, cfg(3)).unwrap();
        assert_eq!(e.handoff_stats(), (0, 1, 0), "the beginning was not copied");
        assert_eq!(e.prefix_stats().3, 80);
        let mut got = vec![];
        'serve: loop {
            for em in e.step() {
                if em.slot == b {
                    got.push(em.token);
                    if let Some(c) = em.cost {
                        assert_eq!(
                            c.prefilled, 0,
                            "nothing of the shared prompt was read again"
                        );
                        break 'serve;
                    }
                }
            }
        }
        assert_eq!(got, want);
        assert_eq!(e.active(), 1, "the first user is still being served");
    }

    /// The first token is sampled from the pass that reads the end of the prompt: the same tokens as with a pass
    /// of its own, for one pass over the weights fewer, at every prompt length around a chunk boundary.
    #[test]
    fn the_first_token_comes_from_the_last_prompt_chunk() {
        let Some(m) = model() else {
            eprintln!("skipped: set SMILEY_CORE_MODEL");
            return;
        };
        for n in [
            5usize,
            PREFILL_CHUNK,
            PREFILL_CHUNK + 1,
            PREFILL_CHUNK + 2,
            2 * PREFILL_CHUNK + 9,
        ] {
            let prompt = opening(n);
            let mut fused = Engine::new(m.clone(), 3, 256, 1, KvPrec::F16, 0);
            let mut apart = Engine::new(m.clone(), 3, 256, 1, KvPrec::F16, 0);
            apart.fuse_first_token = false;
            let (a, read_a) = serve(&mut fused, &prompt, 3);
            let (b, read_b) = serve(&mut apart, &prompt, 3);
            assert_eq!(
                a, b,
                "{n} tokens: the first token's pass changed the answer"
            );
            assert_eq!((read_a, read_b), (n - 1, n - 1));
            // A prompt whose last chunk is full leaves the last token for a decode step either way.
            let saves = (n - 1) % PREFILL_CHUNK != 0;
            assert_eq!(
                fused.stats.bytes_weights < apart.stats.bytes_weights,
                saves,
                "{n} tokens: {} against {} weight bytes",
                fused.stats.bytes_weights,
                apart.stats.bytes_weights
            );
            assert_eq!(fused.decode_steps + saves as u64, apart.decode_steps);
        }
    }

    /// A request that ends on its first token, a conversation larger than the budget, and a budget changed while
    /// the engine runs.
    #[test]
    fn the_store_keeps_the_newest_conversation_and_follows_its_budget() {
        let Some(m) = model() else {
            eprintln!("skipped: set SMILEY_CORE_MODEL");
            return;
        };
        // One byte of budget: far less than any conversation.
        let mut e = Engine::new(m.clone(), 3, 256, 1, KvPrec::F16, 1);
        let prompt = opening(40);
        let (one, read) = serve(&mut e, &prompt, 1);
        assert_eq!((one.len(), read), (1, 39));
        let (entries, bytes, _, _) = e.prefix_stats();
        assert_eq!(
            entries, 1,
            "the newest conversation is kept whatever the budget"
        );
        assert_eq!(bytes as u64, 40 * e.cache_bytes_per_slot() / 256);
        // The same prompt again: the cache holds all of it, so nothing is read and the token is the same.
        let (again, read) = serve(&mut e, &prompt, 1);
        assert_eq!((again, read), (one, 0));
        assert_eq!(e.handoff_stats().0, 1);
        // Another conversation replaces it: the budget has room for one.
        let other: Vec<u32> = opening(40).iter().map(|t| t + 2000).collect();
        serve(&mut e, &other, 1);
        assert_eq!(e.prefix_stats().0, 1);
        assert_eq!(e.handoff_stats().2, 1, "the older conversation was evicted");
        // A larger budget keeps both; shrinking it evicts at once; zero keeps nothing.
        e.set_prefix_budget(64 << 20);
        serve(&mut e, &prompt, 1);
        assert_eq!(e.prefix_stats().0, 2);
        e.set_prefix_budget(1);
        assert_eq!(e.prefix_stats().0, 1);
        e.set_prefix_budget(0);
        assert_eq!((e.prefix_stats().0, e.prefix_stats().1), (0, 0));
        serve(&mut e, &prompt, 1);
        assert_eq!(
            e.prefix_stats().0,
            0,
            "a store with no budget keeps nothing"
        );
        e.set_prefix_budget(64 << 20);
        serve(&mut e, &prompt, 1);
        e.flush_prefixes();
        assert_eq!(e.prefix_stats().0, 0);
    }
}
