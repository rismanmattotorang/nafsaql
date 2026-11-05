//! LittleBit kernel micro-benchmark: one thread, synthetic rows, L2-resident.
//!
//! ```text
//! cargo run --release --example lbbench -- [n] [rows] [reps]
//! ```
//!
//! Times the f32 sign-flip kernel (`lb_dot_batch`) against the int8-activation
//! kernel (`lb_dot_i8`) on the same bit rows, in ns per row and weights per ns.

use smiley_core::lbq::{lb_dot_batch, lb_dot_i8_batch, quantize_i8_row};
use std::time::Instant;

fn main() {
    let a: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|s| s.parse().ok())
        .collect();
    let (n, rows, reps) = (
        *a.first().unwrap_or(&960),
        *a.get(1).unwrap_or(&512),
        *a.get(2).unwrap_or(&400),
    );
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    let mut rnd = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let rb = n / 8;
    let bits: Vec<u8> = (0..rows * rb).map(|_| (rnd() >> 32) as u8).collect();
    let x: Vec<f32> = (0..n)
        .map(|_| ((rnd() >> 40) as f32 / (1u64 << 23) as f32) - 1.0)
        .collect();
    let mut q = vec![0.0f32; n];
    quantize_i8_row(&mut q, n, |i| x[i]);
    let mut sink = 0.0f32;
    for round in 0..3 {
        let t = Instant::now();
        let mut o = [0.0f32; 1];
        for _ in 0..reps {
            for r in 0..rows {
                lb_dot_batch(&bits[r * rb..(r + 1) * rb], n, &x, n, 1, &mut o);
                sink += o[0];
            }
        }
        let f = t.elapsed().as_nanos() as f64 / (reps * rows) as f64;
        let t = Instant::now();
        for _ in 0..reps {
            for r in 0..rows {
                let mut o = [0.0f32; 1];
                lb_dot_i8_batch(&bits[r * rb..(r + 1) * rb], n, &q, n, 1, &mut o);
                sink += o[0];
            }
        }
        let i = t.elapsed().as_nanos() as f64 / (reps * rows) as f64;
        println!(
            "round {round}: n {n}  f32 {f:.0} ns/row ({:.1} w/ns)   int8 {i:.0} ns/row ({:.1} w/ns)   x{:.2}",
            n as f64 / f,
            n as f64 / i,
            f / i
        );
    }
    std::hint::black_box(sink);
    // The fork-join itself: an empty `pool.run`, the floor under every fused matmul.
    let pool = smiley_core::pool::Pool::new(4);
    for _ in 0..3 {
        let t = Instant::now();
        let reps = 20_000;
        for _ in 0..reps {
            pool.run(|_, _| {});
        }
        println!(
            "empty pool.run (4 workers): {:.2} us",
            t.elapsed().as_nanos() as f64 / reps as f64 / 1e3
        );
    }
}
