//! An in-memory copy of a large knowledge base's vectors, searched exactly (Phase 8, NFR-K07).
//!
//! SurrealDB's HNSW index answered in 0.37 s at the median at 50,000 passages of 1,024 dimensions once it was warm (the
//! first queries after a cold start, and any made while other heavy queries ran, took 26 s to over 100 s). The vectors of
//! such a base are 200 MB; held in memory (normalised, in one contiguous array) an exact search over all of them is 51
//! million multiply-adds: 18 to 22 ms, exact, and independent of what else the store is doing. It is off by default
//! (`vector_cache_mb = 0`) because the warm index is already usable; it is for bases of hundreds of thousands of passages
//! or hosts whose store is shared.
//!
//! The copy is loaded once, in the background, from the store in pages; until it is ready a search that would have used
//! it answers by words alone and says so. It is dropped when the base changes (the base's vector count is part of its
//! identity, and the service drops it on deletes and re-indexes), and bases whose vectors would not fit in
//! `[knowledge] vector_cache_mb` keep using the store's index.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::store::Store;

/// One base's vectors: ids in store order, vectors normalised to unit length, row-major.
pub struct Loaded {
    /// Vectors held when this copy was made (the identity the service checks against the store's own count).
    pub count: u64,
    pub dims: usize,
    ids: Vec<String>,
    data: Vec<f32>,
}

impl Loaded {
    pub fn from_rows(rows: Vec<(String, Vec<f32>)>, dims: usize) -> Loaded {
        let mut ids = Vec::with_capacity(rows.len());
        let mut data = Vec::with_capacity(rows.len() * dims);
        for (id, mut v) in rows {
            if v.len() != dims {
                continue;
            }
            normalise(&mut v);
            ids.push(id);
            data.extend_from_slice(&v);
        }
        Loaded {
            count: ids.len() as u64,
            dims,
            ids,
            data,
        }
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// The `k` most similar vectors to `q` (cosine), best first: (id, similarity).
    pub fn top(&self, q: &[f32], k: usize) -> Vec<(String, f32)> {
        if q.len() != self.dims || self.is_empty() {
            return vec![];
        }
        let mut q = q.to_vec();
        normalise(&mut q);
        let mut scored: Vec<(f32, usize)> = self
            .data
            .chunks_exact(self.dims)
            .enumerate()
            .map(|(i, row)| (dot(row, &q), i))
            .collect();
        let k = k.min(scored.len());
        if k < scored.len() {
            scored.select_nth_unstable_by(k, |a, b| b.0.total_cmp(&a.0));
            scored.truncate(k);
        }
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        scored
            .into_iter()
            .map(|(s, i)| (self.ids[i].clone(), s))
            .collect()
    }
}

fn normalise(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        v.iter_mut().for_each(|x| *x /= n);
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    // Four partial sums let the compiler vectorise without changing the result's meaning.
    let mut s = [0f32; 4];
    for (x, y) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        for i in 0..4 {
            s[i] += x[i] * y[i];
        }
    }
    s.iter().sum::<f32>()
        + a.chunks_exact(4)
            .remainder()
            .iter()
            .zip(b.chunks_exact(4).remainder())
            .map(|(x, y)| x * y)
            .sum::<f32>()
}

enum State {
    Loading,
    Ready(Arc<Loaded>),
}

/// What a lookup found.
pub enum Lookup {
    /// The copy, current for the count given.
    Ready(Arc<Loaded>),
    /// A load is under way (or has just been started): not yet.
    Loading,
    /// This base's vectors would not fit in the cache: use the store's index.
    TooBig,
}

pub struct VectorCache {
    cap_bytes: u64,
    map: Arc<Mutex<HashMap<String, State>>>,
}

impl VectorCache {
    pub fn new(cap_mb: u64) -> VectorCache {
        VectorCache {
            cap_bytes: cap_mb * 1024 * 1024,
            map: Default::default(),
        }
    }

    /// The vectors of `kb` if a copy of exactly `count` vectors is held; otherwise a background load is started (once)
    /// and `Loading` returned, or `TooBig` when the base is over the cap.
    pub fn lookup(&self, store: &Store, kb: &str, count: u64, dims: usize) -> Lookup {
        if count * dims as u64 * 4 > self.cap_bytes {
            return Lookup::TooBig;
        }
        let mut map = self.map.lock().unwrap_or_else(|e| e.into_inner());
        match map.get(kb) {
            Some(State::Ready(l)) if l.count == count => return Lookup::Ready(l.clone()),
            Some(State::Loading) => return Lookup::Loading,
            _ => {}
        }
        map.insert(kb.to_string(), State::Loading);
        drop(map);
        let (store, kb_id, shared) = (store.clone(), kb.to_string(), self.map.clone());
        tokio::spawn(async move {
            let loaded = load(&store, &kb_id, dims).await;
            let mut map = shared.lock().unwrap_or_else(|e| e.into_inner());
            // A base dropped while loading (deleted, re-indexed) is no longer marked Loading: the copy is discarded.
            if !matches!(map.get(&kb_id), Some(State::Loading)) {
                return;
            }
            match loaded {
                Ok(l) => {
                    tracing::info!(kb = %kb_id, vectors = l.len(), "vectors of a large base loaded into memory");
                    map.insert(kb_id, State::Ready(Arc::new(l)));
                }
                Err(e) => {
                    tracing::warn!(kb = %kb_id, error = %e, "the vectors of a large base could not be loaded");
                    map.remove(&kb_id);
                }
            }
        });
        Lookup::Loading
    }

    /// Forget a base's copy (it changed, or it is gone).
    pub fn drop_kb(&self, kb: &str) {
        self.map
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(kb);
    }
}

async fn load(store: &Store, kb: &str, dims: usize) -> Result<Loaded, crate::KbError> {
    let docs: Vec<String> = store
        .list_documents(kb)
        .await?
        .into_iter()
        .map(|d| d.id)
        .collect();
    let mut rows: Vec<(String, Vec<f32>)> = Vec::new();
    for doc in &docs {
        rows.extend(store.vectors_of_doc(doc).await?);
    }
    Ok(Loaded::from_rows(rows, dims))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vec_of(seed: u32, dims: usize) -> Vec<f32> {
        let mut x = seed.wrapping_mul(2654435761).wrapping_add(1);
        (0..dims)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                (x % 2000) as f32 / 1000.0 - 1.0
            })
            .collect()
    }

    /// The in-memory search is exact: it returns what a brute-force cosine ranking returns.
    #[test]
    fn the_cached_search_is_exact() {
        let dims = 37; // not a multiple of four: the remainder path is used
        let rows: Vec<(String, Vec<f32>)> = (0..500)
            .map(|i| (format!("c{i}"), vec_of(i, dims)))
            .collect();
        let l = Loaded::from_rows(rows.clone(), dims);
        assert_eq!(l.len(), 500);
        for qs in [3u32, 77, 401, 9999] {
            let q = vec_of(qs, dims);
            let mut brute: Vec<(f32, &str)> = rows
                .iter()
                .map(|(id, v)| {
                    let (mut a, mut b) = (v.clone(), q.clone());
                    normalise(&mut a);
                    normalise(&mut b);
                    (a.iter().zip(&b).map(|(x, y)| x * y).sum(), id.as_str())
                })
                .collect();
            brute.sort_by(|a, b| b.0.total_cmp(&a.0));
            let got = l.top(&q, 8);
            assert_eq!(got.len(), 8);
            for (g, b) in got.iter().zip(&brute) {
                assert_eq!(g.0, b.1);
                assert!((g.1 - b.0).abs() < 1e-5);
            }
        }
    }

    /// A vector of the wrong width is skipped, a query of the wrong width finds nothing, and k beyond the base is the base.
    #[test]
    fn odd_inputs_are_handled() {
        let l = Loaded::from_rows(
            vec![
                ("a".into(), vec![1.0, 0.0]),
                ("b".into(), vec![1.0, 0.0, 0.0]),
            ],
            2,
        );
        assert_eq!(l.len(), 1);
        assert!(l.top(&[1.0, 0.0, 0.0], 3).is_empty());
        assert_eq!(l.top(&[0.0, 1.0], 10).len(), 1);
        assert!(Loaded::from_rows(vec![], 2).top(&[1.0, 0.0], 3).is_empty());
    }
}
