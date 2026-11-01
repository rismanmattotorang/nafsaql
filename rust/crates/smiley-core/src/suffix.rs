//! Suffix-indexed drafting: model-free speculation that prices its own confidence.
//!
//! Prompt-lookup decoding scans the context backwards for the last n tokens, takes
//! the first hit and always proposes the same number of tokens. SuffixDecoding
//! (Oliaro et al., NeurIPS 2025) and SAM Decoding show what that leaves out: match
//! the *longest* suffix available, consider *every* earlier occurrence rather than
//! the newest, and speculate as far as the evidence supports rather than a fixed
//! count. This module does the three on a CPU budget.
//!
//! * **Index, not scan.** Every n-gram of every order in `ORDERS` is hashed into a
//!   table as the context grows, so finding a match is `ORDERS.len()` lookups no
//!   matter how long the context is. The scan it replaces was O(context) per step.
//! * **Vote.** The newest `WAYS` occurrences of the longest matching suffix each
//!   propose a continuation; the plurality token wins at every depth and the draft
//!   stops where the survivors disagree to nothing.
//! * **Learned confidence.** The probability that draft token `j` is accepted is
//!   not guessed: a small table indexed by (match order, whether the occurrences
//!   agreed, depth, whether the previous draft was accepted whole) is updated from
//!   what verification actually accepted. The last feature is the one that matters
//!   in a copy: mid-run, acceptance is ~100% at every depth, at the start of one it
//!   is not, and only the history tells the two apart. The session
//!   turns those probabilities into a draft length with the measured verify cost
//!   (`choose_k`, DeepInfer's varlen rule), so a doubtful draft costs nothing.
//!
//! Drafting never changes the output: a drafted token is committed only when it
//! equals what the model would have chosen itself (see `session.rs`).

use std::collections::HashMap;

/// Suffix lengths tried, longest first when matching.
pub const ORDERS: [usize; 6] = [2, 3, 4, 6, 8, 12];
/// Earlier occurrences remembered per n-gram (newest first).
const WAYS: usize = 4;
/// Depth classes in the confidence table: depth 0, 1, 2, and 3-or-more.
const DEPTHS: usize = 4;
/// Memory of the confidence table: older outcomes count 2% less each step.
const FORGET: f32 = 0.98;

/// One cell of the confidence table that a drafted token's acceptance is booked to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cell {
    order: u8,
    agree: u8,
    depth: u8,
    hot: u8,
}

#[derive(Clone, Copy)]
struct Slot {
    pos: [u32; WAYS],
    n: u8,
}

/// A proposed continuation with the cell each token's acceptance will be booked to.
#[derive(Debug, Default)]
pub struct Draft {
    pub toks: Vec<u32>,
    pub cells: Vec<Cell>,
    /// Length of the suffix that matched.
    pub order: usize,
}

pub struct SuffixIndex {
    maps: Vec<HashMap<u64, Slot>>,
    /// `ctx[..indexed]` is in the tables.
    indexed: usize,
    /// (successes, trials) per cell, decayed.
    table: Vec<(f32, f32)>,
    /// The last draft was accepted whole: a copy run is under way.
    hot: bool,
}

fn key(toks: impl Iterator<Item = u32>) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for t in toks {
        h = (h ^ (t as u64 + 1)).wrapping_mul(0x0000_0100_0000_01b3);
        h ^= h >> 29;
    }
    h
}

fn cell_at(c: Cell) -> usize {
    ((c.order as usize * 2 + c.agree as usize) * DEPTHS + c.depth as usize) * 2 + c.hot as usize
}

impl SuffixIndex {
    pub fn new() -> Self {
        // Optimistic priors, worth two observations: a cell that starts pessimistic
        // never clears the break-even for a draft, so it is never tried and never
        // learns. A longer match is likelier to continue; a copy run is likelier
        // still. The first real outcomes replace these within a few drafts.
        let prior = [0.45f32, 0.55, 0.65, 0.78, 0.88, 0.94];
        let mut table = vec![(0.0, 0.0); ORDERS.len() * 2 * DEPTHS * 2];
        for (o, p) in prior.iter().enumerate() {
            for agree in 0..2 {
                for d in 0..DEPTHS {
                    for hot in 0..2 {
                        let q = (p * if agree == 1 { 1.0 } else { 0.8 }
                            + if hot == 1 { 0.5 * (1.0 - p) } else { 0.0 })
                        .clamp(0.02, 0.98);
                        table[cell_at(Cell {
                            order: o as u8,
                            agree,
                            depth: d as u8,
                            hot,
                        })] = (2.0 * q, 2.0);
                    }
                }
            }
        }
        SuffixIndex {
            maps: ORDERS.iter().map(|_| HashMap::new()).collect(),
            indexed: 0,
            table,
            hot: false,
        }
    }

    /// Forget the corpus (the conversation was reset or rebuilt) but keep what was
    /// learned about how reliable each kind of match is.
    pub fn clear(&mut self) {
        for m in &mut self.maps {
            m.clear();
        }
        self.indexed = 0;
        self.hot = false;
    }

    /// A plain step ran: whatever run was under way is over.
    pub fn cool(&mut self) {
        self.hot = false;
    }

    /// Index every position of `ctx` not yet indexed. An n-gram is entered when its
    /// last token arrives, so it can only propose tokens that already exist.
    pub fn sync(&mut self, ctx: &[u32], min_order: usize) {
        if ctx.len() < self.indexed {
            self.clear();
        }
        for end in self.indexed + 1..=ctx.len() {
            for (oi, &n) in ORDERS.iter().enumerate() {
                if n < min_order || n > end {
                    continue;
                }
                let start = end - n;
                let slot = self.maps[oi]
                    .entry(key(ctx[start..end].iter().copied()))
                    .or_insert(Slot {
                        pos: [0; WAYS],
                        n: 0,
                    });
                let keep = (slot.n as usize).min(WAYS - 1);
                slot.pos.copy_within(0..keep, 1);
                slot.pos[0] = start as u32;
                slot.n = (keep + 1) as u8;
            }
        }
        self.indexed = ctx.len();
    }

    /// Probability that the token at `c` is accepted, given the ones before it were,
    /// with an optimism bonus that shrinks as the cell is tested (an upper
    /// confidence bound: a draft worth trying is worth one more look before the
    /// table is believed).
    pub fn conf(&self, c: Cell) -> f32 {
        let (s, t) = self.table[cell_at(c)];
        let t = t.max(1e-3);
        (s / t + 0.3 / t.sqrt()).clamp(0.01, 0.99)
    }

    /// Propose up to `k` tokens to follow `next`, the token just chosen but not yet
    /// in `ctx`. The longest suffix of `ctx + next` that occurred before wins.
    pub fn propose(&self, ctx: &[u32], next: u32, k: usize, min_order: usize) -> Draft {
        if k == 0 || ctx.is_empty() {
            return Draft::default();
        }
        for oi in (0..ORDERS.len()).rev() {
            let n = ORDERS[oi];
            if n < min_order || ctx.len() + 1 < n {
                continue;
            }
            let tail = ctx[ctx.len() + 1 - n..]
                .iter()
                .copied()
                .chain(std::iter::once(next));
            let Some(slot) = self.maps[oi].get(&key(tail.clone())) else {
                continue;
            };
            let tail: Vec<u32> = tail.collect();
            // Genuine matches that have something after them (hash collisions out).
            let mut alive: Vec<usize> = slot.pos[..slot.n as usize]
                .iter()
                .map(|&p| p as usize)
                .filter(|&p| p + n < ctx.len() && ctx[p..p + n] == tail[..])
                .collect();
            if alive.is_empty() {
                continue;
            }
            let mut d = Draft {
                order: n,
                ..Default::default()
            };
            for depth in 0..k {
                // Plurality of the survivors' next tokens; the newest wins a tie.
                let mut best: Option<(u32, usize)> = None;
                let mut alive_here = 0;
                for &p in alive.iter() {
                    let at = p + n + depth;
                    if at >= ctx.len() {
                        continue;
                    }
                    alive_here += 1;
                    let t = ctx[at];
                    let votes = alive
                        .iter()
                        .filter(|&&q| q + n + depth < ctx.len() && ctx[q + n + depth] == t)
                        .count();
                    if best.map_or(true, |(_, v)| votes > v) {
                        best = Some((t, votes));
                    }
                }
                let Some((tok, votes)) = best else { break };
                d.toks.push(tok);
                d.cells.push(Cell {
                    order: oi as u8,
                    agree: (votes == alive_here) as u8,
                    depth: depth.min(DEPTHS - 1) as u8,
                    hot: self.hot as u8,
                });
                alive.retain(|&p| p + n + depth < ctx.len() && ctx[p + n + depth] == tok);
            }
            if !d.toks.is_empty() {
                return d;
            }
        }
        Draft::default()
    }

    /// Book what verification decided: `accepted` of the drafted tokens survived, so
    /// tokens `0..accepted` succeeded, token `accepted` (if any) failed, and the rest
    /// were never tested.
    pub fn observe(&mut self, cells: &[Cell], accepted: usize) {
        self.hot = !cells.is_empty() && accepted == cells.len();
        for (j, c) in cells.iter().enumerate().take(accepted + 1) {
            let cell = &mut self.table[cell_at(*c)];
            cell.0 = cell.0 * FORGET + (j < accepted) as u8 as f32;
            cell.1 = cell.1 * FORGET + 1.0;
        }
    }
}

impl Default for SuffixIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn built(ctx: &[u32]) -> SuffixIndex {
        let mut s = SuffixIndex::new();
        s.sync(ctx, 2);
        s
    }

    #[test]
    fn matches_the_suffix_ending_at_the_unfed_token() {
        // ... 1 2 3 4 5 6 ... then "1 2" with 3 about to be fed: propose 4 5 6.
        let ctx = [9, 1, 2, 3, 4, 5, 6, 7, 1, 2];
        let d = built(&ctx).propose(&ctx, 3, 3, 2);
        assert_eq!(d.toks, vec![4, 5, 6]);
    }

    #[test]
    fn the_longest_suffix_beats_a_newer_shorter_one() {
        // "7 1 2 3" occurred once, followed by 4; "1 2 3" occurred more recently
        // followed by 9. The query ends 7 1 2 3, so the longer match answers.
        let ctx = [7, 1, 2, 3, 4, 8, 1, 2, 3, 9, 5, 7, 1, 2];
        let d = built(&ctx).propose(&ctx, 3, 1, 2);
        assert_eq!(d.toks, vec![4]);
        assert_eq!(d.order, 4);
    }

    #[test]
    fn occurrences_vote_and_the_newest_breaks_a_tie() {
        // "1 2" was followed by 5 twice and by 6 once: 5 wins at depth 0.
        let ctx = [1, 2, 5, 0, 1, 2, 6, 0, 1, 2, 5, 0, 7, 1];
        let d = built(&ctx).propose(&ctx, 2, 1, 2);
        assert_eq!(d.toks, vec![5]);
        assert_eq!(d.cells[0].agree, 0, "the occurrences disagreed");
        // Tie: one vote each, the newest (…2 5) wins.
        let ctx = [1, 2, 6, 0, 1, 2, 5, 0, 7, 1];
        let d = built(&ctx).propose(&ctx, 2, 1, 2);
        assert_eq!(d.toks, vec![5]);
    }

    #[test]
    fn no_match_and_no_continuation_propose_nothing() {
        let ctx = [1, 2, 3, 4, 5];
        assert!(built(&ctx).propose(&ctx, 9, 4, 2).toks.is_empty());
        // The only match is the context's own tail, which has nothing after it.
        let ctx = [5, 6, 1, 2];
        assert!(built(&ctx).propose(&ctx, 7, 4, 2).toks.is_empty());
        assert!(built(&ctx).propose(&ctx, 7, 0, 2).toks.is_empty());
    }

    #[test]
    fn a_draft_never_runs_past_the_context() {
        let ctx = [1, 2, 3, 4, 0, 1, 2];
        let d = built(&ctx).propose(&ctx, 3, 8, 2);
        assert_eq!(d.toks, vec![4, 0, 1, 2]);
    }

    #[test]
    fn syncing_incrementally_equals_syncing_at_once() {
        let ctx: Vec<u32> = (0..200u32).map(|i| i % 17).collect();
        for cut in [60usize, 120, 199] {
            let once = built(&ctx[..cut]);
            let mut inc = SuffixIndex::new();
            for end in [10, 11, cut / 2, cut] {
                inc.sync(&ctx[..end], 2);
            }
            let (a, b) = (
                inc.propose(&ctx[..cut], ctx[cut], 6, 2),
                once.propose(&ctx[..cut], ctx[cut], 6, 2),
            );
            assert!(
                !b.toks.is_empty(),
                "cut {cut}: the periodic corpus should always match"
            );
            assert_eq!(a.toks, b.toks, "cut {cut}");
            assert_eq!(a.order, b.order, "cut {cut}");
        }
    }

    #[test]
    fn a_shrinking_context_rebuilds_the_index() {
        let long = [1, 2, 3, 4, 1, 2];
        let mut s = SuffixIndex::new();
        s.sync(&long, 2);
        let fresh = [7, 8, 9, 7, 8];
        s.sync(&fresh, 2);
        assert_eq!(s.propose(&fresh, 9, 2, 2).toks, vec![7, 8]);
        assert!(
            s.propose(&fresh, 3, 2, 2).toks.is_empty(),
            "the old corpus is gone"
        );
    }

    #[test]
    fn confidence_is_learned_from_outcomes() {
        let ctx = [1, 2, 3, 4, 5, 1, 2];
        let mut s = built(&ctx);
        let d = s.propose(&ctx, 3, 3, 2);
        let before = s.conf(d.cells[0]);
        for _ in 0..40 {
            s.observe(&d.cells, 0); // always rejected at the first token
        }
        let cold = Cell {
            hot: 0,
            ..d.cells[0]
        };
        assert!(
            s.conf(cold) < before * 0.5,
            "{} -> {}",
            before,
            s.conf(cold)
        );
        let after_reject = s.conf(cold);
        for _ in 0..80 {
            s.observe(&d.cells, d.cells.len()); // always fully accepted
        }
        // A fully accepted draft makes the next one a "hot" cell: its confidence is
        // booked apart from the cold cell the rejections taught.
        assert!(s.hot);
        assert!(
            s.conf(Cell {
                hot: 1,
                ..d.cells[0]
            }) > 0.9
        );
        assert!(
            s.conf(Cell {
                hot: 0,
                ..d.cells[0]
            }) > after_reject + 0.3
        );
        // Tokens after the first failure were never tested and are not booked.
        let c2 = d.cells[2];
        let t_before = s.table[cell_at(c2)].1;
        s.observe(&d.cells, 0);
        assert!((s.table[cell_at(c2)].1 - t_before).abs() < 1e-6);
        s.cool();
        assert!(!s.hot);
    }
}
