//! Certified gate bins on labelled samples (Phase 8 follow-up, FR-R1): fit the calibration from stored `fast_probe`
//! (or `nafs_blind`) evaluation runs, print its reliability bins with their Clopper-Pearson bounds, then check the
//! certificate out of sample by repeated random splits: fit on one part, bin the other part by the fitted e1, and count
//! how often a bin's held-out errors are significantly more than its certified bound allows (a binomial tail of 0.01 or less).
//!
//!     cargo run --release -p nafs-eval --example gate_report -- config/nafs-aql.toml run1.json [run2.json ...] [--splits 300]
//!
//! The run files are `GET /v1/eval/runs/{id}` responses.

use nafs_core::calibration::Calibration;
use nafs_core::config::Config;
use nafs_types::GateFeatures;

/// P(X >= k) for X ~ Binomial(n, p): the chance of seeing k or more errors in n held-out drafts if the bin's true error
/// rate were no higher than its certified bound p. A violation of the certificate is a tail at or below 0.01; a plain
/// "observed rate above the bound" comparison would flag a bin of ten drafts for one unlucky error.
fn tail(k: usize, n: usize, p: f64) -> f64 {
    if k == 0 {
        return 1.0;
    }
    let ln_c = |n: usize, i: usize| {
        (1..=i)
            .map(|j| ((n - i + j) as f64 / j as f64).ln())
            .sum::<f64>()
    };
    let p = p.clamp(1e-12, 1.0 - 1e-12);
    (k..=n)
        .map(|i| (ln_c(n, i) + i as f64 * p.ln() + (n - i) as f64 * (1.0 - p).ln()).exp())
        .sum::<f64>()
        .min(1.0)
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cfg = Config::load(&args[0])?;
    let splits: usize = args
        .iter()
        .position(|a| a == "--splits")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(300);
    let mut samples: Vec<(String, GateFeatures, bool)> = vec![];
    for p in args[1..].iter().filter(|a| a.ends_with(".json")) {
        let run: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(p)?)?;
        samples.extend(nafs_eval::calibration_samples(&run)?);
    }
    let mut seen = std::collections::HashSet::new();
    samples.retain(|s| seen.insert(s.0.clone()));
    let n = samples.len();
    let wrong = samples.iter().filter(|s| s.2).count();
    println!(
        "{n} labelled samples, {wrong} drafts wrong ({:.1} %)",
        100.0 * wrong as f64 / n as f64
    );
    let g = &cfg.gate;
    let th = |_: &GateFeatures, e1: f32| nafs_core::gate::lambda_threshold(&cfg, e1, 0.25, 90.0);
    let xs: Vec<GateFeatures> = samples.iter().map(|s| s.1.clone()).collect();
    let ys: Vec<bool> = samples.iter().map(|s| s.2).collect();

    // ---- the full fit
    let cal = Calibration::fit(&xs, &ys, Some("gate_report".into()), &th);
    println!(
        "\nfull fit: held-out ECE {:.3}, Brier {:.3}",
        cal.ece, cal.brier
    );
    println!(
        "{:>8} {:>8} {:>5} {:>7} {:>8}  certified at max_risk {:.2}",
        "e1 from", "e1 to", "n", "wrong", "bound", g.max_risk
    );
    for b in &cal.risk_bins {
        println!(
            "{:8.3} {:8.3} {:5} {:7} {:8.3}  {}",
            b.lo,
            b.hi,
            b.n,
            b.errors,
            b.upper,
            if b.upper <= g.max_risk { "yes" } else { "no" }
        );
    }
    match cal.promotable(
        g.calibration_min_n,
        g.calibration_min_per_class,
        g.calibration_max_ece,
        g.max_risk,
    ) {
        Ok(()) => println!("promotable under the configured tests"),
        Err(e) => println!("not promotable: {e}"),
    }

    // ---- what the fixed rule (the one in force today) does with the same drafts
    {
        let prior = Calibration::prior();
        let (mut kept, mut kept_wrong) = (0usize, 0usize);
        for (f, &y) in xs.iter().zip(&ys) {
            let ctx = nafs_core::gate::GateCtx {
                cfg: &cfg,
                mode: "fixed",
                lambda: g.lambda_default,
                e2: cfg.e2(f.domain),
                delta_c: cfg.delta_c(f.domain),
                cal: &prior,
                promoted: false,
                budget: g.escalation_budget,
            };
            if !nafs_core::gate::decide(f, &ctx).escalate {
                kept += 1;
                kept_wrong += y as usize;
            }
        }
        println!(
            "\nthe fixed rule on the same drafts: keeps {kept} of {n} ({:.1} %), wrong among them {kept_wrong} ({:.1} %)",
            100.0 * kept as f64 / n as f64,
            100.0 * kept_wrong as f64 / kept.max(1) as f64
        );
    }

    // ---- the certificate out of sample
    let (mut viol, mut tested, mut in_cert, mut in_cert_wrong, mut total_test, mut promotable_fits) =
        (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
    let mut per_risk: Vec<(f32, usize, usize, usize)> = [0.10f32, 0.15, 0.20, 0.25, 0.30, 0.40]
        .iter()
        .map(|&r| (r, 0, 0, 0))
        .collect();
    // For each risk, the bins certified at it that had held-out items, and how many of those broke their bound.
    let mut per_risk_viol: Vec<(usize, usize)> = vec![(0, 0); per_risk.len()];
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    for _ in 0..splits {
        let mut idx: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            idx.swap(i, (rng.next() % (i as u64 + 1)) as usize);
        }
        let cut = n * 6 / 10;
        let (tr, te) = idx.split_at(cut);
        let (txs, tys): (Vec<GateFeatures>, Vec<bool>) =
            tr.iter().map(|&i| (xs[i].clone(), ys[i])).unzip();
        if tys.iter().filter(|&&y| y).count() < 2 || tys.iter().filter(|&&y| !y).count() < 2 {
            continue;
        }
        let c = Calibration::fit(&txs, &tys, None, &th);
        if c.promotable(
            g.calibration_min_n,
            g.calibration_min_per_class,
            g.calibration_max_ece,
            g.max_risk,
        )
        .is_ok()
        {
            promotable_fits += 1;
        }
        // Per bin of the fitted calibration: the held-out items that fall in it.
        let mut bin_n = vec![0usize; c.risk_bins.len()];
        let mut bin_w = vec![0usize; c.risk_bins.len()];
        for &i in te {
            let e1 = c.e1(&xs[i]);
            let Some((bin, _)) = c.certified_for(e1, g.max_risk) else {
                continue;
            };
            let k = c
                .risk_bins
                .iter()
                .position(|b| std::ptr::eq(b, bin))
                .unwrap();
            bin_n[k] += 1;
            bin_w[k] += ys[i] as usize;
            total_test += 1;
        }
        for (k, b) in c.risk_bins.iter().enumerate() {
            if bin_n[k] == 0 {
                continue;
            }
            if b.upper <= g.max_risk {
                tested += 1;
                // A violation: the held-out error rate in a certified bin is above the bound it was certified at.
                if tail(bin_w[k], bin_n[k], b.upper as f64) <= 0.01 {
                    viol += 1;
                }
                in_cert += bin_n[k];
                in_cert_wrong += bin_w[k];
            }
            for (ri, r) in per_risk.iter_mut().enumerate() {
                if b.upper <= r.0 {
                    r.1 += bin_n[k];
                    r.2 += bin_w[k];
                    per_risk_viol[ri].0 += 1;
                    if tail(bin_w[k], bin_n[k], b.upper as f64) <= 0.01 {
                        per_risk_viol[ri].1 += 1;
                    }
                }
            }
        }
        for r in per_risk.iter_mut() {
            r.3 += te.len();
        }
    }
    println!(
        "\nout of sample, {splits} random 60/40 splits ({} of them promotable at max_risk {:.2}):",
        promotable_fits, g.max_risk
    );
    println!("  certified bins tested {tested}; held-out errors significantly above the bound in {viol} ({:.1} %)", 100.0 * viol as f64 / tested.max(1) as f64);
    println!("  held-out drafts in certified bins: {in_cert} of {total_test} binned; wrong among them {in_cert_wrong} ({:.1} %)", 100.0 * in_cert_wrong as f64 / in_cert.max(1) as f64);
    println!("\n  coverage and selective error by the risk a bin must be certified at (held-out):");
    println!(
        "  {:>9} {:>10} {:>16} {:>22}",
        "max_risk", "coverage", "error among kept", "bins significantly over"
    );
    for (ri, (r, kept, wrong, all)) in per_risk.into_iter().enumerate() {
        let (bins, over) = per_risk_viol[ri];
        println!(
            "  {:9.2} {:9.1}% {:15.1}% {:>14} of {:<6}",
            r,
            100.0 * kept as f64 / all.max(1) as f64,
            100.0 * wrong as f64 / kept.max(1) as f64,
            over,
            bins
        );
    }
    println!(
        "  (keeping every draft: error {:.1} %)",
        100.0 * wrong as f64 / n as f64
    );

    // ---- the rules side by side on the same held-out drafts: what each keeps and how often a kept draft is wrong
    {
        let rules: Vec<(&str, &str, f32)> = vec![
            ("fixed rule (in force today)", "fixed", 1.0),
            ("calibrated, expected loss only", "calibrated", 1.0),
            (
                "calibrated + certified bins, max_risk 0.40",
                "calibrated",
                0.40,
            ),
            (
                "calibrated + certified bins, max_risk 0.25",
                "calibrated",
                0.25,
            ),
            (
                "calibrated + certified bins, max_risk 0.15",
                "calibrated",
                0.15,
            ),
        ];
        let mut tot: Vec<(usize, usize, usize)> = vec![(0, 0, 0); rules.len()];
        let mut rng = Rng(0x0dd_ba11_5eed_1234);
        for _ in 0..splits {
            let mut idx: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                idx.swap(i, (rng.next() % (i as u64 + 1)) as usize);
            }
            let (tr, te) = idx.split_at(n * 6 / 10);
            let (txs, tys): (Vec<GateFeatures>, Vec<bool>) =
                tr.iter().map(|&i| (xs[i].clone(), ys[i])).unzip();
            if tys.iter().filter(|&&y| y).count() < 2 || tys.iter().filter(|&&y| !y).count() < 2 {
                continue;
            }
            let c = Calibration::fit(&txs, &tys, None, &th);
            for (ri, (_, mode, max_risk)) in rules.iter().enumerate() {
                let mut cfg2 = cfg.clone();
                cfg2.gate.max_risk = *max_risk;
                for &i in te {
                    let f = &xs[i];
                    let ctx = nafs_core::gate::GateCtx {
                        cfg: &cfg2,
                        mode,
                        lambda: g.lambda_default,
                        e2: cfg.e2(f.domain),
                        delta_c: cfg.delta_c(f.domain),
                        cal: &c,
                        promoted: *mode == "calibrated",
                        budget: g.escalation_budget,
                    };
                    tot[ri].2 += 1;
                    if !nafs_core::gate::decide(f, &ctx).escalate {
                        tot[ri].0 += 1;
                        tot[ri].1 += ys[i] as usize;
                    }
                }
            }
        }
        println!("\nthe rules on the same held-out drafts ({splits} random 60/40 splits; the calibration is fitted on the 60 %):");
        println!(
            "  {:46} {:>10} {:>18} {:>18}",
            "rule", "kept", "wrong among kept", "escalated"
        );
        for (ri, (name, _, _)) in rules.iter().enumerate() {
            let (kept, wr, all) = tot[ri];
            println!(
                "  {:46} {:9.1}% {:17.1}% {:17.1}%",
                name,
                100.0 * kept as f64 / all.max(1) as f64,
                100.0 * wr as f64 / kept.max(1) as f64,
                100.0 * (all - kept) as f64 / all.max(1) as f64
            );
        }
    }

    // ---- the same check for a design where the bins are counted on data the model never saw: fit on one part, put the
    // bin edges at quantiles of e1 over a second part (no label read), count errors per bin on that part, test on a third.
    let mut per_risk: Vec<(f32, usize, usize, usize)> = [0.15f32, 0.20, 0.25, 0.30, 0.40, 0.50]
        .iter()
        .map(|&r| (r, 0, 0, 0))
        .collect();
    let mut per_risk_viol: Vec<(usize, usize)> = vec![(0, 0); per_risk.len()];
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let mut used = 0usize;
    for _ in 0..splits {
        let mut idx: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            idx.swap(i, (rng.next() % (i as u64 + 1)) as usize);
        }
        let (a, rest) = idx.split_at(n * 5 / 10);
        let (b, te) = rest.split_at(n * 25 / 100);
        let (axs, ays): (Vec<GateFeatures>, Vec<bool>) =
            a.iter().map(|&i| (xs[i].clone(), ys[i])).unzip();
        if ays.iter().filter(|&&y| y).count() < 2 || ays.iter().filter(|&&y| !y).count() < 2 {
            continue;
        }
        let mut c = Calibration::fit(&axs, &ays, None, &th);
        let ps: Vec<f32> = b.iter().map(|&i| c.e1(&xs[i])).collect();
        let by: Vec<bool> = b.iter().map(|&i| ys[i]).collect();
        c.risk_bins = nafs_core::calibration::risk_bins(&ps, &by, 5, g.risk_alpha);
        used += 1;
        let mut bin_n = vec![0usize; c.risk_bins.len()];
        let mut bin_w = vec![0usize; c.risk_bins.len()];
        for &i in te {
            let e1 = c.e1(&xs[i]);
            let Some((bin, _)) = c.certified_for(e1, 1.0) else {
                continue;
            };
            let k = c
                .risk_bins
                .iter()
                .position(|x| std::ptr::eq(x, bin))
                .unwrap();
            bin_n[k] += 1;
            bin_w[k] += ys[i] as usize;
        }
        for (k, bn) in c.risk_bins.iter().enumerate() {
            for (ri, r) in per_risk.iter_mut().enumerate() {
                if bn.upper <= r.0 && bin_n[k] > 0 {
                    r.1 += bin_n[k];
                    r.2 += bin_w[k];
                    per_risk_viol[ri].0 += 1;
                    if tail(bin_w[k], bin_n[k], bn.upper as f64) <= 0.01 {
                        per_risk_viol[ri].1 += 1;
                    }
                }
            }
        }
        for r in per_risk.iter_mut() {
            r.3 += te.len();
        }
    }
    println!("\nbins counted on a part the model never saw ({used} splits: fit 50 %, count 25 %, test 25 %):");
    println!(
        "  {:>9} {:>10} {:>16} {:>22}",
        "max_risk", "coverage", "error among kept", "bins significantly over"
    );
    for (ri, (r, kept, wr, all)) in per_risk.into_iter().enumerate() {
        let (bins, over) = per_risk_viol[ri];
        println!(
            "  {:9.2} {:9.1}% {:15.1}% {:>14} of {:<6}",
            r,
            100.0 * kept as f64 / all.max(1) as f64,
            100.0 * wr as f64 / kept.max(1) as f64,
            over,
            bins
        );
    }
    Ok(())
}
