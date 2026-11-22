// nafs-aql: a few finished conversations' caches, kept so that a prompt that shares a long prefix with one of them does not
// recompute it.
//
// The session holds one cache, so a prompt only benefits when it follows a conversation with the same beginning. A server that
// alternates kinds of request (chat, agent, arithmetic) prefilled the whole prompt every time. This keeps the last few
// conversations as snapshots and restores the one that shares the longest prefix (the small form of a radix prefix cache).

use crate::kv::KvSnapshot;

/// Prefixes shorter than this are cheaper to recompute than to look up and copy.
pub const MIN_PREFIX: usize = 16;

struct Entry {
    tokens: Vec<u32>,
    snap: KvSnapshot,
    used: u64,
}

/// Snapshots of finished conversations under a byte budget, least recently used evicted first.
pub struct PrefixStore {
    entries: Vec<Entry>,
    budget: usize,
    bytes: usize,
    tick: u64,
    /// Prompts that restored a prefix, and the positions they did not have to prefill.
    pub hits: u64,
    pub positions_saved: u64,
}

/// Length of the common beginning of two token lists.
pub fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

impl PrefixStore {
    pub fn new(budget_bytes: usize) -> PrefixStore {
        PrefixStore {
            entries: Vec::new(),
            budget: budget_bytes,
            bytes: 0,
            tick: 0,
            hits: 0,
            positions_saved: 0,
        }
    }

    pub fn entries(&self) -> usize {
        self.entries.len()
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Whether some entry already holds all of `tokens` as its beginning (nothing new to keep).
    pub fn covers(&self, tokens: &[u32]) -> bool {
        self.entries
            .iter()
            .any(|e| e.tokens.len() >= tokens.len() && e.tokens[..tokens.len()] == *tokens)
    }

    /// Keep `snap`, which holds the cache of `tokens`. An entry that is only the beginning of `tokens` is dropped (this one
    /// carries all of it); old entries are evicted to stay within the budget. A snapshot larger than the budget is not kept.
    pub fn insert(&mut self, tokens: Vec<u32>, snap: KvSnapshot) {
        debug_assert_eq!(tokens.len(), snap.len());
        if tokens.len() < MIN_PREFIX || snap.bytes() > self.budget {
            return;
        }
        let mut i = 0;
        while i < self.entries.len() {
            let e = &self.entries[i];
            if e.tokens.len() <= tokens.len() && tokens[..e.tokens.len()] == *e.tokens {
                self.bytes -= self.entries.swap_remove(i).snap.bytes();
            } else {
                i += 1;
            }
        }
        while self.bytes + snap.bytes() > self.budget {
            let Some(oldest) = (0..self.entries.len()).min_by_key(|&j| self.entries[j].used) else {
                break;
            };
            self.bytes -= self.entries.swap_remove(oldest).snap.bytes();
        }
        self.tick += 1;
        self.bytes += snap.bytes();
        self.entries.push(Entry {
            tokens,
            snap,
            used: self.tick,
        });
    }

    /// The entry that shares the longest beginning with `prompt` (at most `limit` tokens of it, at least [`MIN_PREFIX`]),
    /// with that length.
    pub fn best(&mut self, prompt: &[u32], limit: usize) -> Option<(&KvSnapshot, usize)> {
        let (idx, lcp) = self
            .entries
            .iter()
            .enumerate()
            .map(|(i, e)| (i, common_prefix(prompt, &e.tokens).min(limit)))
            .max_by_key(|&(_, l)| l)
            .filter(|&(_, l)| l >= MIN_PREFIX)?;
        self.tick += 1;
        self.entries[idx].used = self.tick;
        Some((&self.entries[idx].snap, lcp))
    }

    /// Book a restore that skipped `positions` positions of prefill.
    pub fn note_hit(&mut self, positions: usize) {
        self.hits += 1;
        self.positions_saved += positions as u64;
    }
}

// nafs-aql (Phase 34, after FreeToken): the batching engine's store. It keeps the caches themselves, not copies.
//
// # Why
//
// `PrefixStore` above copies a cache out when a conversation ends and copies it back in when the next request shares
// its beginning. Measured on the reference machine at a 2,048-token conversation, that is 55 ms on the way in and
// 165 ms on the way out for the 1.7B model (72 and 205 ms for the 4B), on every request, to move bytes that were
// already where they had to be. FreeToken keeps one residency for prefill and decode and re-points its page table
// when a request's pages join the prefix tree; the form of that here is ownership: a finished slot's cache moves
// into the store, and a request that continues that conversation takes the cache out again. Nothing is copied.
//
// # When a request may take a cache, and when it must copy
//
// Taking a cache removes it from the store, so it is right only when nobody else can want what the request does not
// keep. Two cases (`supersedes`):
// - the request's prompt holds every token of the kept conversation (the next step of an agent, a continued turn, a
//   prompt a shadow request has read): nothing is lost;
// - the kept conversation ends inside one turn that the prompt has rewritten and gone past. This is what a chat
//   template does to the turn before the last: Qwen3 writes the reply after a reasoning block and replays it without
//   one, so the next prompt stops matching at the start of that reply, and the reply as it was generated will never
//   be sent again. Turn boundaries are the template's own special tokens, which is the observation FreeToken's
//   semantic anchors rest on: a harness edits whole blocks, and the blocks begin at special tokens.
// In every other case (another conversation that shares a system prompt, a second sample of the same prompt) the kept
// cache stays and the shared positions are copied, as before.

struct Held {
    tokens: Vec<u32>,
    cache: crate::kv::KvCache,
    used: u64,
}

/// Whether a request for `prompt` leaves nothing worth keeping in a kept cache of `held`: see the note above.
/// `turn_starts` are the token ids that open a turn under the model's chat template (none: only the first case).
pub fn supersedes(held: &[u32], prompt: &[u32], turn_starts: &[u32]) -> bool {
    let shared = common_prefix(held, prompt);
    if shared >= held.len() {
        return true;
    }
    let opens = |t: &u32| turn_starts.contains(t);
    !held[shared..].iter().any(opens) && prompt[shared..].iter().any(opens)
}

/// Caches of finished conversations under a byte budget, least recently used evicted first.
///
/// The budget bounds what is kept beside the newest conversation, which is always kept: its cache was a slot's own
/// memory a moment ago, so keeping it adds nothing, and dropping it makes the next step of a long agent transcript
/// read the whole transcript again (what the copying store did to a conversation larger than its budget). A budget
/// of zero keeps nothing.
pub struct CacheStore {
    entries: Vec<Held>,
    budget: usize,
    tick: u64,
    /// Requests that reused a kept cache, and the positions they did not have to prefill.
    pub hits: u64,
    pub positions_saved: u64,
    /// Of those, the requests that took the cache itself, and the ones that copied from it.
    pub taken: u64,
    pub copied: u64,
    /// Caches dropped to stay within the budget.
    pub evicted: u64,
}

impl CacheStore {
    pub fn new(budget_bytes: usize) -> CacheStore {
        CacheStore {
            entries: Vec::new(),
            budget: budget_bytes,
            tick: 0,
            hits: 0,
            positions_saved: 0,
            taken: 0,
            copied: 0,
            evicted: 0,
        }
    }

    pub fn entries(&self) -> usize {
        self.entries.len()
    }

    /// Bytes of memory the kept caches occupy.
    pub fn bytes(&self) -> usize {
        self.entries.iter().map(|e| e.cache.occupied_bytes()).sum()
    }

    pub fn budget(&self) -> usize {
        self.budget
    }

    /// Whether some entry already holds all of `tokens` as its beginning (nothing new to keep).
    pub fn covers(&self, tokens: &[u32]) -> bool {
        self.entries
            .iter()
            .any(|e| e.tokens.len() >= tokens.len() && e.tokens[..tokens.len()] == *tokens)
    }

    /// The entry that shares the longest beginning with `prompt` (at most `limit` tokens of it, at least
    /// [`MIN_PREFIX`]): its index and that length.
    pub fn find(&self, prompt: &[u32], limit: usize) -> Option<(usize, usize)> {
        self.entries
            .iter()
            .enumerate()
            .map(|(i, e)| (i, common_prefix(prompt, &e.tokens).min(limit)))
            // The longest; among equals the most recently used, so a tie does not disturb an older conversation.
            .max_by_key(|&(i, l)| (l, self.entries[i].used))
            .filter(|&(_, l)| l >= MIN_PREFIX)
    }

    /// The tokens entry `i` holds the cache of.
    pub fn tokens(&self, i: usize) -> &[u32] {
        &self.entries[i].tokens
    }

    /// Entry `i`'s cache, to copy from; the entry counts as used.
    pub fn cache(&mut self, i: usize) -> &crate::kv::KvCache {
        self.tick += 1;
        self.entries[i].used = self.tick;
        &self.entries[i].cache
    }

    /// Remove entry `i` and hand its cache over.
    pub fn take(&mut self, i: usize) -> crate::kv::KvCache {
        self.entries.swap_remove(i).cache
    }

    /// Keep `cache`, which holds exactly `tokens`. An entry that is only the beginning of `tokens` is dropped (this
    /// one carries all of it); the least recently used entries go until the rest fits the budget.
    pub fn insert(&mut self, tokens: Vec<u32>, cache: crate::kv::KvCache) {
        debug_assert_eq!(tokens.len(), cache.filled());
        if self.budget == 0 || tokens.len() < MIN_PREFIX {
            return;
        }
        self.entries
            .retain(|e| !(e.tokens.len() <= tokens.len() && tokens[..e.tokens.len()] == *e.tokens));
        self.tick += 1;
        self.entries.push(Held {
            tokens,
            cache,
            used: self.tick,
        });
        self.fit();
    }

    /// Evict the least recently used entries until the store fits its budget, never the newest one.
    fn fit(&mut self) {
        while self.entries.len() > 1 && self.bytes() > self.budget {
            let oldest = (0..self.entries.len())
                .min_by_key(|&j| self.entries[j].used)
                .unwrap_or(0);
            self.entries.swap_remove(oldest);
            self.evicted += 1;
        }
    }

    /// Change the budget while serving (the memory a personal machine can spare changes while the engine runs).
    /// Shrinking evicts at once; zero keeps nothing.
    pub fn set_budget(&mut self, budget_bytes: usize) {
        self.budget = budget_bytes;
        if budget_bytes == 0 {
            self.evicted += self.entries.len() as u64;
            self.entries.clear();
        } else {
            self.fit();
        }
    }

    /// Drop every kept cache.
    pub fn clear(&mut self) {
        self.evicted += self.entries.len() as u64;
        self.entries.clear();
    }

    /// Book a reuse that skipped `positions` positions of prefill, by taking the cache or by copying from it.
    pub fn note_hit(&mut self, positions: usize, took: bool) {
        self.hits += 1;
        self.positions_saved += positions as u64;
        if took {
            self.taken += 1;
        } else {
            self.copied += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv::{KvCache, KvPrec};

    fn snap_of(tokens: &[u32]) -> KvSnapshot {
        let mut c = KvCache::new(KvPrec::F16, 1, 1, 8, 128);
        c.advance(tokens.len());
        c.snapshot(tokens.len())
    }

    fn toks(start: u32, n: usize) -> Vec<u32> {
        (start..start + n as u32).collect()
    }

    #[test]
    fn the_entry_with_the_longest_shared_beginning_is_found() {
        let mut s = PrefixStore::new(1 << 20);
        let agent = toks(100, 40);
        let chat = toks(500, 30);
        s.insert(agent.clone(), snap_of(&agent));
        s.insert(chat.clone(), snap_of(&chat));
        let mut prompt = agent[..35].to_vec();
        prompt.extend([9000, 9001]);
        let (_, l) = s.best(&prompt, prompt.len() - 1).unwrap();
        assert_eq!(l, 35);
        let (_, l) = s.best(&chat, chat.len() - 1).unwrap();
        assert_eq!(
            l, 29,
            "capped at the limit: the last prompt token needs its own position"
        );
        assert!(s.best(&toks(7000, 40), 39).is_none(), "nothing shared");
        // A shared beginning shorter than MIN_PREFIX is not worth a restore.
        let mut short = agent[..MIN_PREFIX - 1].to_vec();
        short.extend(toks(8000, 20));
        assert!(s.best(&short, short.len() - 1).is_none());
    }

    #[test]
    fn a_longer_conversation_replaces_the_one_it_extends() {
        let mut s = PrefixStore::new(1 << 20);
        let a = toks(1, 20);
        s.insert(a.clone(), snap_of(&a));
        assert!(s.covers(&a[..10]));
        let longer = toks(1, 50);
        assert!(!s.covers(&longer));
        s.insert(longer.clone(), snap_of(&longer));
        assert_eq!(
            s.entries(),
            1,
            "the shorter one is the beginning of the longer"
        );
        assert!(s.covers(&a) && s.covers(&longer));
    }

    #[test]
    fn the_least_recently_used_entry_goes_first_when_the_budget_is_full() {
        let one = snap_of(&toks(0, 20)).bytes();
        let mut s = PrefixStore::new(one * 2 + one / 2);
        let (a, b, c) = (toks(1000, 20), toks(2000, 20), toks(3000, 20));
        s.insert(a.clone(), snap_of(&a));
        s.insert(b.clone(), snap_of(&b));
        s.best(&a, 19).unwrap(); // a is used, so b is the oldest
        s.insert(c.clone(), snap_of(&c));
        assert_eq!(s.entries(), 2);
        assert!(s.covers(&a) && s.covers(&c) && !s.covers(&b));
        assert!(s.bytes() <= one * 2 + one / 2);
        // A snapshot that cannot fit at all is not kept.
        let mut tiny = PrefixStore::new(one / 2);
        tiny.insert(a.clone(), snap_of(&a));
        assert_eq!(tiny.entries(), 0);
    }

    fn cache_of(tokens: &[u32]) -> KvCache {
        let mut c = KvCache::new(KvPrec::F16, 1, 1, 8, 128);
        c.advance(tokens.len());
        c
    }

    // nafs-aql: turn boundaries decide whether a request may take a kept cache or must leave it for somebody else.
    #[test]
    fn a_request_takes_a_cache_only_when_nothing_else_can_want_it() {
        const OPEN: u32 = 9; // the token that opens a turn
        let turn = |role: u32, body: &[u32]| {
            let mut t = vec![OPEN, role];
            t.extend(body);
            t
        };
        let system = turn(1, &[100, 101, 102]);
        let user1 = turn(2, &[200, 201]);
        // The reply as it was generated (after a reasoning block) and as the template replays it (without one).
        let generated = [3u32, 70, 71, 72, 300, 301];
        let replayed = [3u32, 300, 301];
        let mut held = [system.clone(), user1.clone()].concat();
        held.push(OPEN);
        held.extend(generated);
        // The next step of the same transcript: every kept token is in the prompt.
        let mut next = held.clone();
        next.extend([400, 401]);
        assert!(supersedes(&held, &next, &[OPEN]));
        assert!(
            supersedes(&held, &next, &[]),
            "the first case needs no template"
        );
        // The next turn: the reply is replayed without its reasoning, and a new question follows.
        let mut edited = [system.clone(), user1.clone()].concat();
        edited.push(OPEN);
        edited.extend(replayed);
        edited.extend(turn(2, &[500]));
        assert!(supersedes(&held, &edited, &[OPEN]));
        assert!(
            !supersedes(&held, &edited, &[]),
            "without turn tokens the cache is left alone"
        );
        // A second sample of the same prompt: the first sample's reply may be the one the conversation continues from.
        let again = [system.clone(), user1.clone(), vec![OPEN, 3]].concat();
        assert!(!supersedes(&held, &again, &[OPEN]));
        // Another conversation under the same system prompt: the kept one holds whole turns the prompt does not.
        let other = [system.clone(), turn(2, &[600, 601]), vec![OPEN, 3]].concat();
        assert!(!supersedes(&held, &other, &[OPEN]));
        // A kept conversation of several turns, and a prompt that rewrites an early one.
        let mut long = held.clone();
        long.extend(turn(2, &[700]));
        let mut rewrite = [system, user1].concat();
        rewrite.push(OPEN);
        rewrite.extend(replayed);
        rewrite.extend(turn(2, &[800]));
        assert!(!supersedes(&long, &rewrite, &[OPEN]));
    }

    #[test]
    fn the_cache_store_finds_takes_and_evicts() {
        let one = cache_of(&toks(0, 20)).occupied_bytes();
        let mut s = CacheStore::new(one * 2 + one / 2);
        let (a, b, c) = (toks(1000, 20), toks(2000, 20), toks(3000, 20));
        s.insert(a.clone(), cache_of(&a));
        s.insert(b.clone(), cache_of(&b));
        assert_eq!(s.bytes(), 2 * one);
        // The longest shared beginning wins, capped at the limit.
        let mut prompt = a[..18].to_vec();
        prompt.extend([9000, 9001]);
        let (i, l) = s.find(&prompt, prompt.len() - 1).unwrap();
        assert_eq!((s.tokens(i), l), (&a[..], 18));
        assert_eq!(s.find(&a, 19).unwrap().1, 19);
        assert!(s.find(&toks(7000, 40), 39).is_none());
        // Using a makes b the oldest, so b goes when c does not fit.
        let _ = s.cache(i);
        s.insert(c.clone(), cache_of(&c));
        assert_eq!(s.entries(), 2);
        assert!(s.covers(&a) && s.covers(&c) && !s.covers(&b));
        assert_eq!(s.evicted, 1);
        // Taking removes the entry; a longer conversation replaces the one it extends.
        let (i, _) = s.find(&a, 19).unwrap();
        let cache = s.take(i);
        assert_eq!(cache.filled(), 20);
        assert!(!s.covers(&a));
        let longer = toks(3000, 50);
        s.insert(longer.clone(), cache_of(&longer));
        assert_eq!(s.entries(), 1);
        assert!(s.covers(&c) && s.covers(&longer));
    }

    #[test]
    fn the_newest_conversation_is_kept_whatever_the_budget_and_the_budget_can_change() {
        let one = cache_of(&toks(0, 20)).occupied_bytes();
        // Smaller than one conversation: the newest is kept anyway, alone.
        let mut s = CacheStore::new(one / 2);
        let (a, b) = (toks(1000, 20), toks(2000, 20));
        s.insert(a.clone(), cache_of(&a));
        assert_eq!(s.entries(), 1);
        s.insert(b.clone(), cache_of(&b));
        assert_eq!(s.entries(), 1);
        assert!(s.covers(&b) && !s.covers(&a));
        // Growing the budget lets more stay; shrinking evicts at once; zero keeps nothing.
        s.set_budget(one * 3);
        s.insert(a.clone(), cache_of(&a));
        assert_eq!(s.entries(), 2);
        s.set_budget(one);
        assert_eq!(s.entries(), 1);
        assert!(s.covers(&a), "the most recently used stays");
        s.set_budget(0);
        assert_eq!((s.entries(), s.bytes()), (0, 0));
        s.insert(a.clone(), cache_of(&a));
        assert_eq!(s.entries(), 0, "a store with no budget keeps nothing");
        // A conversation shorter than MIN_PREFIX is not worth keeping.
        let mut t = CacheStore::new(one * 4);
        let short = toks(1, MIN_PREFIX - 1);
        t.insert(short.clone(), cache_of(&short));
        assert_eq!(t.entries(), 0);
        t.insert(a.clone(), cache_of(&a));
        t.clear();
        assert_eq!(t.entries(), 0);
    }
}
