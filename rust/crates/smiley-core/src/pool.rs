//! A persistent worker pool with exactly one fork–join per matmul.
//!
//! # Why not rayon, and why not `thread::scope`
//!
//! rayon is excluded by the spec: its work-stealing gives neither a stable thread
//! count nor a predictable barrier count, and this engine *reports* both as measured
//! quantities. A figure that changes with the scheduler's mood is not a measurement.
//!
//! `std::thread::scope` is safe and would need no unsafe code at all, but it spawns
//! and joins on every call. A single decode step runs seven matmuls in each of 32
//! layers — 224 fork–joins per token. At the ~25 µs it costs macOS to create a
//! thread that is 5.6 ms per token against a step budget of roughly 30 ms: a fifth
//! of the time spent creating threads that do 130 µs of work each. The workers here
//! are created once per session and park on a condvar between matmuls.
//!
//! # Partitioning
//!
//! Output rows are split into `n` contiguous ranges, each worker runs its range to
//! completion, and the call returns when all have. There is no stealing and no
//! per-row synchronisation: the ranges are disjoint by construction, which is what
//! makes the raw-pointer writes in `matmul` sound.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// A type-erased `Fn(worker_index, worker_count)`.
///
/// Erased through a monomorphised trampoline rather than by transmuting a `&dyn` fat
/// pointer into two words — the trampoline is a plain function pointer and makes no
/// assumption about trait-object layout.
#[derive(Clone, Copy)]
struct Job {
    data: *const (),
    call: unsafe fn(*const (), usize, usize),
}

unsafe fn trampoline<F: Fn(usize, usize) + Sync>(p: *const (), w: usize, n: usize) {
    (*(p as *const F))(w, n)
}

/// How long a worker spins on the epoch before parking on the condvar.
///
/// # Why spin at all
///
/// A decode step is ~160 fork–joins (Q/K/V, O, gate/up and down in each of 32
/// layers, attention, the head), each a few tens of microseconds of work. Parking
/// every worker on a condvar between them made every fork–join pay a kernel wake-up
/// — measured as the layer matmuls sustaining 10 GB/s while the one large matmul
/// (the 49 152-row head) reached 18 GB/s on the same kernels. Workers now spin for a
/// short window first, so back-to-back matmuls within a token never sleep, and park
/// only when the session goes quiet (between turns), so an idle engine costs no CPU.
///
/// Spinning is also what makes an oversubscribed machine worse: four sessions each
/// spinning four workers on four cores starve one another (the throughput test
/// failed only when the suite ran its tests in parallel). So a waiter spins with a
/// pause instruction for `SPIN`, then yields its core to anything runnable until
/// `YIELD`, and only then parks.
const SPIN_DEFAULT_US: u64 = 50;
const YIELD_DEFAULT_US: u64 = 0;

/// The spin and yield windows. `SMILEY_CORE_SPIN_US` / `SMILEY_CORE_YIELD_US`
/// override them (0 and 0 park at once, which is the original condvar pool), so the
/// trade-off can be measured on a given machine instead of assumed.
fn windows() -> (Duration, Duration) {
    use std::sync::OnceLock;
    static W: OnceLock<(Duration, Duration)> = OnceLock::new();
    *W.get_or_init(|| {
        let env = |k: &str, d: u64| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(d)
        };
        (
            Duration::from_micros(env("SMILEY_CORE_SPIN_US", SPIN_DEFAULT_US)),
            Duration::from_micros(env("SMILEY_CORE_YIELD_US", YIELD_DEFAULT_US)),
        )
    })
}

/// Wait until `done()` holds: spin, then yield, then (if `park` is given) block.
#[inline]
fn backoff(t0: Instant, spins: &mut u32) -> Phase {
    *spins = spins.wrapping_add(1);
    if !spins.is_multiple_of(64) {
        std::hint::spin_loop();
        return Phase::Spin;
    }
    let e = t0.elapsed();
    let (spin, yld) = windows();
    if e < spin {
        std::hint::spin_loop();
        Phase::Spin
    } else if e < yld {
        std::thread::yield_now();
        Phase::Yield
    } else {
        Phase::Park
    }
}

#[derive(PartialEq)]
enum Phase {
    Spin,
    Yield,
    Park,
}

struct Inner {
    /// Bumped once per `run`; a worker runs the job when it sees a new value. Workers
    /// compare epochs rather than read a flag, so a worker that was slow to notice a
    /// job cannot run it twice or miss the next one.
    epoch: AtomicU64,
    /// Workers still running the current job.
    remaining: AtomicUsize,
    shutdown: AtomicBool,
    /// Set when a job panicked on a worker. The worker still counts itself done —
    /// otherwise the caller would spin forever — and `run` re-raises on its own
    /// thread, where the session's `catch_unwind` turns it into an error status.
    poisoned: AtomicBool,
    /// Written by `run` before the epoch is published (Release) and read by workers
    /// after they observe it (Acquire).
    job: UnsafeCell<Option<Job>>,
    /// Parking for idle workers. `sleepers` is only touched under the mutex, which is
    /// what makes "publish then notify if anyone sleeps" race-free.
    park: Mutex<usize>,
    wake: Condvar,
    /// The caller sleeps here once its spin window has passed, and the last worker
    /// to finish wakes it. Yielding in a loop instead was measured to *lose* to the
    /// condvar pool under load: on macOS a yielding thread keeps competing with the
    /// very worker it is waiting for.
    done_mu: Mutex<()>,
    done_cv: Condvar,
    /// Fixed for the pool's lifetime. Held here rather than derived from the job
    /// count: `remaining` decrements as workers finish, so a worker that read the
    /// width late would be told there were fewer workers than there are and would
    /// compute a share overlapping someone else's.
    n: usize,
}

// SAFETY: `job` is written only by the thread inside `run`, before it publishes a new
// epoch with Release ordering and while no worker is running a job (the previous
// `run` waited for `remaining == 0`); workers read it only after an Acquire load of
// the new epoch. The pointer it holds is valid until `run` returns, which is after
// every worker has finished.
unsafe impl Sync for Inner {}
unsafe impl Send for Inner {}

pub struct Pool {
    n: usize,
    inner: Arc<Inner>,
    handles: Vec<JoinHandle<()>>,
}

/// Ask the scheduler to treat the current thread as work a user is waiting on.
///
/// On macOS a thread's QoS class decides how it competes with everything else on
/// the machine. The default for a GUI app's threads is fine while the window is in
/// front; when it is not, the system demotes the process (App Nap) and its threads
/// ran at priority 4 against 31 for a terminal process. A decode step is about 130
/// fork-joins, each waiting on its slowest worker, so one demoted worker
/// behind a busy `mediaanalysisd` made the app 2.5x slower than the same engine
/// run from a shell. `USER_INITIATED` is the class Apple documents for exactly
/// this: work started by the user whose result they are waiting for.
/// Elsewhere this is a no-op; Linux and Windows do not demote a backgrounded app.
pub fn raise_priority() {
    #[cfg(target_os = "macos")]
    {
        const QOS_CLASS_USER_INITIATED: u32 = 0x19;
        extern "C" {
            fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
        }
        // SAFETY: a libSystem call that only changes the calling thread's own
        // scheduling class; it takes no pointers. A failure leaves the class as it was.
        unsafe {
            pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
        }
    }
}

impl Pool {
    pub fn new(n: usize) -> Pool {
        let n = n.max(1);
        // Worker 0 is the calling thread, so it is raised along with the others.
        raise_priority();
        let inner = Arc::new(Inner {
            epoch: AtomicU64::new(0),
            remaining: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
            poisoned: AtomicBool::new(false),
            job: UnsafeCell::new(None),
            park: Mutex::new(0),
            wake: Condvar::new(),
            done_mu: Mutex::new(()),
            done_cv: Condvar::new(),
            n,
        });
        let mut handles = Vec::with_capacity(n - 1);
        // Worker 0 is the calling thread, so only n-1 are spawned. That keeps the
        // caller busy instead of blocked, and makes `n == 1` allocate nothing.
        for w in 1..n {
            let inner = Arc::clone(&inner);
            handles.push(std::thread::spawn(move || {
                raise_priority();
                let mut seen = 0u64;
                loop {
                    // Spin, then park.
                    let t0 = Instant::now();
                    let mut spins = 0u32;
                    loop {
                        if inner.epoch.load(Ordering::Acquire) != seen
                            || inner.shutdown.load(Ordering::Acquire)
                        {
                            break;
                        }
                        if backoff(t0, &mut spins) == Phase::Park {
                            let mut g = inner.park.lock().unwrap_or_else(|e| e.into_inner());
                            *g += 1;
                            while inner.epoch.load(Ordering::Acquire) == seen
                                && !inner.shutdown.load(Ordering::Acquire)
                            {
                                g = inner.wake.wait(g).unwrap_or_else(|e| e.into_inner());
                            }
                            *g -= 1;
                            break;
                        }
                    }
                    if inner.shutdown.load(Ordering::Acquire) {
                        return;
                    }
                    seen = inner.epoch.load(Ordering::Acquire);
                    // SAFETY: see the `Sync` impl — published before the epoch.
                    let job = unsafe { *inner.job.get() };
                    if let Some(j) = job {
                        // SAFETY: `j.data` points at the closure passed to `run`,
                        // which spins until `remaining` reaches zero — so the closure
                        // is still alive here.
                        let ok =
                            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
                                (j.call)(j.data, w, inner.n)
                            }));
                        if ok.is_err() {
                            inner.poisoned.store(true, Ordering::Release);
                        }
                    }
                    if inner.remaining.fetch_sub(1, Ordering::AcqRel) == 1 {
                        // Last one out: wake the caller if it went to sleep. Taking the
                        // lock orders this after the caller's check-then-wait.
                        let _g = inner.done_mu.lock().unwrap_or_else(|e| e.into_inner());
                        inner.done_cv.notify_one();
                    }
                }
            }));
        }
        Pool { n, inner, handles }
    }

    pub fn width(&self) -> usize {
        self.n
    }

    /// Run `f(worker, n)` on every worker and return once all have finished.
    ///
    /// A panic in any share — the caller's or a worker's — is re-raised on the
    /// caller only after every worker has finished with `f`: `f` and whatever it
    /// borrows live on the caller's stack, so unwinding earlier would free them under
    /// the workers (and leave `remaining` counting into the next job).
    pub fn run<F: Fn(usize, usize) + Sync>(&self, f: F) {
        if self.n == 1 {
            f(0, 1);
            return;
        }
        let job = Job {
            data: &f as *const F as *const (),
            call: trampoline::<F>,
        };
        // SAFETY: no worker is running a job (the previous call waited for
        // `remaining == 0`), and none reads `job` until it sees the new epoch.
        unsafe { *self.inner.job.get() = Some(job) };
        self.inner.remaining.store(self.n - 1, Ordering::Release);
        self.inner.epoch.fetch_add(1, Ordering::AcqRel);
        {
            let g = self.inner.park.lock().unwrap_or_else(|e| e.into_inner());
            if *g > 0 {
                self.inner.wake.notify_all();
            }
        }
        // The caller takes share 0 rather than blocking immediately.
        let mine = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(0, self.n)));
        // Spin for the stragglers: the remaining work is microseconds, and sleeping
        // here would hand the wake-up latency straight back.
        let (t0, mut spins) = (Instant::now(), 0u32);
        let (spin, _) = windows();
        while self.inner.remaining.load(Ordering::Acquire) > 0 {
            if t0.elapsed() < spin {
                std::hint::spin_loop();
                spins = spins.wrapping_add(1);
                continue;
            }
            let mut g = self.inner.done_mu.lock().unwrap_or_else(|e| e.into_inner());
            while self.inner.remaining.load(Ordering::Acquire) > 0 {
                g = self
                    .inner
                    .done_cv
                    .wait(g)
                    .unwrap_or_else(|e| e.into_inner());
            }
        }
        let _ = spins;
        let poisoned = self.inner.poisoned.swap(false, Ordering::AcqRel);
        if let Err(p) = mine {
            std::panic::resume_unwind(p);
        }
        if poisoned {
            panic!("a pool worker panicked while running a job");
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.inner.shutdown.store(true, Ordering::Release);
        {
            let _g = self.inner.park.lock().unwrap_or_else(|e| e.into_inner());
            self.inner.wake.notify_all();
        }
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

/// Contiguous share of `total` items belonging to worker `w` of `n`.
///
/// Balanced to within one item, and every index belongs to exactly one worker —
/// which is the precondition the disjoint writes in `matmul` rely on.
#[inline]
pub fn share(total: usize, w: usize, n: usize) -> (usize, usize) {
    let lo = total * w / n;
    let hi = total * (w + 1) / n;
    (lo, hi)
}

#[cfg(test)]
mod tests {

    #[test]
    fn a_caller_panic_waits_for_the_workers_and_the_pool_survives() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let pool = Pool::new(4);
        let touched = AtomicUsize::new(0);
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool.run(|w, _| {
                if w == 0 {
                    panic!("caller share");
                }
                // Workers outlive the caller's panic; their captured state must too.
                std::thread::sleep(std::time::Duration::from_millis(5));
                touched.fetch_add(1, Ordering::SeqCst);
            })
        }));
        assert!(r.is_err(), "the caller's panic was swallowed");
        assert_eq!(
            touched.load(Ordering::SeqCst),
            3,
            "run returned before the workers finished"
        );
        // The next job starts from a clean count.
        let n = AtomicUsize::new(0);
        pool.run(|_, _| {
            n.fetch_add(1, Ordering::SeqCst);
        });
        assert_eq!(n.load(Ordering::SeqCst), 4);
    }
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn shares_partition_exactly_and_evenly() {
        for total in [0usize, 1, 7, 960, 2560, 49152] {
            for n in 1..=8 {
                let mut covered = vec![0u8; total];
                let mut sizes = Vec::new();
                for w in 0..n {
                    let (lo, hi) = share(total, w, n);
                    assert!(lo <= hi && hi <= total);
                    sizes.push(hi - lo);
                    for c in covered.iter_mut().take(hi).skip(lo) {
                        *c += 1;
                    }
                }
                assert!(
                    covered.iter().all(|&c| c == 1),
                    "total={total} n={n}: not a partition"
                );
                let (min, max) = (
                    sizes.iter().min().copied().unwrap(),
                    sizes.iter().max().copied().unwrap(),
                );
                assert!(max - min <= 1, "total={total} n={n}: unbalanced {sizes:?}");
            }
        }
    }

    #[test]
    fn every_worker_runs_exactly_once_per_call() {
        for n in [1usize, 2, 4, 8] {
            let p = Pool::new(n);
            assert_eq!(p.width(), n);
            for _ in 0..50 {
                let hits: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
                p.run(|w, nw| {
                    assert_eq!(nw, n, "worker told the wrong width");
                    hits[w].fetch_add(1, Ordering::SeqCst);
                });
                for (w, h) in hits.iter().enumerate() {
                    assert_eq!(h.load(Ordering::SeqCst), 1, "n={n} worker {w}");
                }
            }
        }
    }

    /// The property the raw-pointer writes in `matmul` depend on: workers only ever
    /// touch their own share, so concurrent writes never overlap.
    #[test]
    fn disjoint_shares_allow_concurrent_writes() {
        let n = 4;
        let p = Pool::new(n);
        let total = 10_000usize;
        let out: Vec<AtomicUsize> = (0..total).map(|_| AtomicUsize::new(0)).collect();
        p.run(|w, nw| {
            let (lo, hi) = share(total, w, nw);
            for (i, o) in out.iter().enumerate().take(hi).skip(lo) {
                o.store(i + 1, Ordering::Relaxed);
            }
        });
        for (i, o) in out.iter().enumerate() {
            assert_eq!(o.load(Ordering::Relaxed), i + 1);
        }
    }

    /// A panic on a spawned worker must surface on the caller, not hang it.
    #[test]
    fn a_worker_panic_is_raised_on_the_caller_and_the_pool_survives() {
        let p = Pool::new(3);
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            p.run(|w, _| {
                if w == 2 {
                    panic!("boom");
                }
            })
        }));
        assert!(r.is_err(), "the worker's panic must reach the caller");
        let c = AtomicUsize::new(0);
        p.run(|_, _| {
            c.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(
            c.load(Ordering::Relaxed),
            3,
            "the pool must still work afterwards"
        );
    }

    /// Workers park after the spin window and must still wake for the next job —
    /// the lost-wake-up case the publish-then-notify-under-the-lock order exists for.
    #[test]
    fn parked_workers_wake_for_the_next_job() {
        let p = Pool::new(4);
        for _ in 0..5 {
            std::thread::sleep(Duration::from_millis(3)); // well past SPIN: all parked
            let c = AtomicUsize::new(0);
            p.run(|_, _| {
                c.fetch_add(1, Ordering::Relaxed);
            });
            assert_eq!(c.load(Ordering::Relaxed), 4);
        }
    }

    #[test]
    fn a_pool_can_be_reused_and_dropped_cleanly() {
        let p = Pool::new(4);
        let c = AtomicUsize::new(0);
        for _ in 0..1000 {
            p.run(|_, _| {
                c.fetch_add(1, Ordering::Relaxed);
            });
        }
        assert_eq!(c.load(Ordering::Relaxed), 4000);
        drop(p); // must join every worker rather than hang
    }
}
