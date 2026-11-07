//! Probe: Candle CPU matmul cost at the memory module's shapes (contiguous vs transposed weight, 2-D vs batched).
use candle_core::{Device, Tensor};
use std::time::Instant;
fn main() {
    let dev = Device::Cpu;
    let x = Tensor::randn(0f32, 1.0, (16, 1024), &dev).unwrap();
    let x3 = x.unsqueeze(0).unwrap();
    let w = Tensor::randn(0f32, 1.0, (4096, 1024), &dev).unwrap();
    let wt = w.t().unwrap().contiguous().unwrap();
    let t = |label: &str, f: &dyn Fn() -> Tensor| {
        let _ = f();
        let s = Instant::now();
        for _ in 0..20 {
            let _ = f();
        }
        eprintln!(
            "{label}: {:.2} ms",
            s.elapsed().as_secs_f64() * 1000.0 / 20.0
        );
    };
    t("x[16,1024] @ w.t() (strided)", &|| {
        x.matmul(&w.t().unwrap()).unwrap()
    });
    t("x[16,1024] @ wt (contiguous)", &|| x.matmul(&wt).unwrap());
    t("x[1,16,1024] broadcast_matmul w.t()", &|| {
        x3.broadcast_matmul(&w.t().unwrap()).unwrap()
    });
    t("x[1,16,1024] broadcast_matmul wt", &|| {
        x3.broadcast_matmul(&wt).unwrap()
    });
    let lin = candle_nn::Linear::new(w.clone(), None);
    t("candle_nn::Linear on [1,16,1024]", &|| {
        candle_core::Module::forward(&lin, &x3).unwrap()
    });
}
