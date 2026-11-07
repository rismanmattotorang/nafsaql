//! Probe: which Candle op's backward gives a non-finite gradient (with the macOS Accelerate build)?
use candle_core::{Device, Tensor, Var, D};
fn chk(name: &str, f: &dyn Fn(&Tensor) -> Tensor, shape: (usize, usize, usize)) {
    let x = Var::randn(0f32, 1.0, shape, &Device::Cpu).unwrap();
    let y = f(x.as_tensor());
    let w = Tensor::randn(0f32, 1.0, y.dims(), &Device::Cpu).unwrap();
    let g = (y * w).unwrap().sum_all().unwrap().backward().unwrap();
    let gx = g.get(&x).unwrap();
    let s = gx
        .sqr()
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    println!("{name:28} grad norm² {s}");
}
fn main() {
    let dev = Device::Cpu;
    let w = Tensor::randn(0f32, 0.1, (24, 16), &dev).unwrap();
    let mask: Vec<f32> = (0..36)
        .map(|i| if i % 6 > i / 6 { -1e9 } else { 0.0 })
        .collect();
    let mask = Tensor::from_vec(mask, (6, 6), &dev).unwrap();
    chk(
        "broadcast_matmul w.t()",
        &|x| x.broadcast_matmul(&w.t().unwrap()).unwrap(),
        (1, 6, 16),
    );
    chk(
        "matmul contiguous w.t()",
        &|x| {
            x.squeeze(0)
                .unwrap()
                .matmul(&w.t().unwrap().contiguous().unwrap())
                .unwrap()
        },
        (1, 6, 16),
    );
    chk(
        "softmax",
        &|x| candle_nn::ops::softmax(x, D::Minus1).unwrap(),
        (1, 6, 6),
    );
    chk(
        "softmax + causal -1e9",
        &|x| candle_nn::ops::softmax(&x.broadcast_add(&mask).unwrap(), D::Minus1).unwrap(),
        (1, 6, 6),
    );
    chk(
        "rms (sqrt recip)",
        &|x| {
            x.broadcast_mul(
                &(x.sqr().unwrap().mean_keepdim(D::Minus1).unwrap() + 1e-6)
                    .unwrap()
                    .sqrt()
                    .unwrap()
                    .recip()
                    .unwrap(),
            )
            .unwrap()
        },
        (1, 6, 16),
    );
    chk(
        "broadcast_as reshape (gqa)",
        &|x| {
            x.unsqueeze(1)
                .unwrap()
                .broadcast_as((1, 2, 6, 16))
                .unwrap()
                .reshape((1, 12, 16))
                .unwrap()
        },
        (1, 6, 16),
    );
    chk(
        "narrow + cat + neg",
        &|x| {
            Tensor::cat(
                &[
                    &x.narrow(2, 8, 8).unwrap().neg().unwrap(),
                    &x.narrow(2, 0, 8).unwrap(),
                ],
                2,
            )
            .unwrap()
        },
        (1, 6, 16),
    );
    chk(
        "transpose matmul",
        &|x| {
            let q = x
                .reshape((1, 6, 2, 8))
                .unwrap()
                .transpose(1, 2)
                .unwrap()
                .contiguous()
                .unwrap();
            q.matmul(&q.t().unwrap().contiguous().unwrap()).unwrap()
        },
        (1, 6, 16),
    );
    chk(
        "silu composed",
        &|x| {
            (x * (x.neg().unwrap().exp().unwrap() + 1.0)
                .unwrap()
                .recip()
                .unwrap())
            .unwrap()
        },
        (1, 6, 16),
    );
    let tgt = Tensor::new(&[1u32, 3, 5], &dev).unwrap();
    chk(
        "cross_entropy",
        &|x| {
            candle_nn::loss::cross_entropy(&x.squeeze(0).unwrap(), &tgt)
                .unwrap()
                .reshape((1, 1, 1))
                .unwrap()
        },
        (1, 3, 10),
    );
}
