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
}
