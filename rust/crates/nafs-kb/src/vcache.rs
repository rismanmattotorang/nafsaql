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

/// Dot product of two vectors, on AVX2 and FMA when the machine has them (four independent accumulators, 32 floats a
/// step: about 90 ns for 384 dimensions and 150 ns for 1,024 here), else the portable sum. The head-to-head benchmark
/// against vLLM Semantic Router's `vecmath` (`benchmarks/headtohead/`) found the portable one 3 to 5 times slower.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        // One cached answer rather than two feature probes on every call.
        static FAST: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *FAST.get_or_init(|| is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"))
        {
            // SAFETY: both features were detected; the kernel reads only `min(a.len(), b.len())` floats.
            return unsafe { dot_avx2(a, b) };
        }
    }
    dot_portable(a, b)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len().min(b.len());
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut i = 0;
    let (mut s0, mut s1, mut s2, mut s3) = (
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
    );
    let (mut s4, mut s5, mut s6, mut s7) = (
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
    );
    // 64 floats a step with eight independent accumulators: an FMA has a latency of four cycles and two issue per cycle, so
    // eight chains keep both ports busy.
    while i + 64 <= n {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), s0);
        s1 = _mm256_fmadd_ps(
            _mm256_loadu_ps(pa.add(i + 8)),
            _mm256_loadu_ps(pb.add(i + 8)),
            s1,
        );
        s2 = _mm256_fmadd_ps(
            _mm256_loadu_ps(pa.add(i + 16)),
            _mm256_loadu_ps(pb.add(i + 16)),
            s2,
        );
        s3 = _mm256_fmadd_ps(
            _mm256_loadu_ps(pa.add(i + 24)),
            _mm256_loadu_ps(pb.add(i + 24)),
            s3,
        );
        s4 = _mm256_fmadd_ps(
            _mm256_loadu_ps(pa.add(i + 32)),
            _mm256_loadu_ps(pb.add(i + 32)),
            s4,
        );
        s5 = _mm256_fmadd_ps(
            _mm256_loadu_ps(pa.add(i + 40)),
            _mm256_loadu_ps(pb.add(i + 40)),
            s5,
        );
        s6 = _mm256_fmadd_ps(
            _mm256_loadu_ps(pa.add(i + 48)),
            _mm256_loadu_ps(pb.add(i + 48)),
            s6,
        );
        s7 = _mm256_fmadd_ps(
            _mm256_loadu_ps(pa.add(i + 56)),
            _mm256_loadu_ps(pb.add(i + 56)),
            s7,
        );
        i += 64;
    }
    s0 = _mm256_add_ps(s0, s4);
    s1 = _mm256_add_ps(s1, s5);
    s2 = _mm256_add_ps(s2, s6);
    s3 = _mm256_add_ps(s3, s7);
    while i + 32 <= n {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), s0);
        s1 = _mm256_fmadd_ps(
            _mm256_loadu_ps(pa.add(i + 8)),
            _mm256_loadu_ps(pb.add(i + 8)),
            s1,
        );
        s2 = _mm256_fmadd_ps(
            _mm256_loadu_ps(pa.add(i + 16)),
            _mm256_loadu_ps(pb.add(i + 16)),
            s2,
        );
        s3 = _mm256_fmadd_ps(
            _mm256_loadu_ps(pa.add(i + 24)),
            _mm256_loadu_ps(pb.add(i + 24)),
            s3,
        );
        i += 32;
    }
    while i + 8 <= n {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), s0);
        i += 8;
    }
    let v = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
    // Horizontal sum in registers: 8 lanes to 4 to 2 to 1.
    let q = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    let q = _mm_hadd_ps(q, q);
    let q = _mm_hadd_ps(q, q);
    let mut sum = _mm_cvtss_f32(q);
    while i < n {
        sum += *pa.add(i) * *pb.add(i);
        i += 1;
    }
    sum
}

/// The portable dot product (kept for the benchmark and for machines without AVX2).
pub fn dot_portable(a: &[f32], b: &[f32]) -> f32 {
    // Four partial sums let the compiler vectorise without changing the result's meaning.
    let mut s = [0f32; 4];
    let ((ca, ra), (cb, rb)) = (a.as_chunks::<4>(), b.as_chunks::<4>());
    for (x, y) in ca.iter().zip(cb) {
        for i in 0..4 {
            s[i] += x[i] * y[i];
        }
    }
    s.iter().sum::<f32>() + ra.iter().zip(rb).map(|(x, y)| x * y).sum::<f32>()
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

    /// The AVX2 kernel and the portable one agree (to rounding) on every length, including ones that are not multiples of 8.
    #[test]
    fn the_fast_dot_agrees_with_the_portable_one() {
        for n in [0usize, 1, 7, 8, 9, 31, 32, 33, 37, 64, 384, 768, 1000, 1024] {
            let a = vec_of(11, n.max(1))[..n].to_vec();
            let b = vec_of(23, n.max(1))[..n].to_vec();
            let (x, y) = (dot(&a, &b), dot_portable(&a, &b));
            assert!((x - y).abs() <= 1e-3 * (1.0 + y.abs()), "n={n}: {x} vs {y}");
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
