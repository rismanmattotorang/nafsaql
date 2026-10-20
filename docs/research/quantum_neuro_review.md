# Quantum consciousness, neuroscience and Nafs-Aql: what transfers, what does not (Phase 8)

Written 2026-10-20. Companion artefacts: `rust/crates/nafs-core/src/sequential.rs` (the rules), `scripts/confidence_signal_study.py` and `benchmarks/confidence_signal_study.json` (the measurement), `docs/report/sections/phase8.tex` (the same results in the technical report), ASSUMPTIONS A-054 to A-058.

## 0. Method

The question asked was: can ideas from quantum theories of consciousness and from the neuroscience of decision making improve the inference speed or the reasoning accuracy of Nafs-Aql? Every candidate idea had to pass four tests, in order, before any code was written:

1. **Established?** Is the mechanism demonstrated, or a hypothesis? (A hypothesis may still yield a useful mathematical structure, but it cannot be cited as a reason.)
2. **Transferable?** Does a classical, CPU-only inference pipeline inherit anything from it, or only from its classical description?
3. **A theorem with a consequence?** Can the idea be stated as a result whose proof changes a parameter or an algorithm of Nafs-Aql, in the style of Section 3 of the technical report (every result ends with an engineering consequence)?
4. **Measurable here?** Can the change be validated on the reference machine with the existing exact verifiers, so that the claim rests on evidence and not on the analogy?

Ideas that failed a test are recorded with the reason, in the same spirit as the "tried and not adopted" table of the README. Labels follow the report: **[exact]** holds under the stated model; **[strong]** needs an assumption that must be checked empirically; **[literature]** is a quantitative estimate taken from the cited source, not proved here.

## 1. Quantum consciousness

### 1.1 The claims and their status

The orchestrated objective reduction theory (Orch OR) holds that consciousness arises from quantum computations in neuronal microtubules, terminated by gravitational self-collapse (Hameroff and Penrose, *Physics of Life Reviews* 11(1):39–78, 2014). Tegmark (*Physical Review E* 61:4194–4206, 2000) estimated the decoherence times of the relevant superpositions (ion positions in neurons, kink excitations in microtubules) at 10⁻²⁰ to 10⁻¹³ s against dynamical timescales of 10⁻³ to 10⁻¹ s, and concluded that the cognitively relevant parts of the brain behave classically. Hagan, Hameroff and Tuszyński (*Physical Review E* 65:061901, 2002) argued the estimate is too pessimistic by several orders of magnitude; even their corrected values are shorter than the 25 ms of a conscious moment the theory needs.

Recent experiments are suggestive but not decisive. Babcock et al. (*Journal of Physical Chemistry B* 128:4035–4046, 2024) predicted and measured collective ultraviolet superradiance of tryptophan networks in microtubules; the directly measured effect is a modest increase of fluorescence quantum yield with network size (about 1.4×), and the superradiant lifetimes are modelled rather than measured. Kalra et al. (*ACS Central Science* 9:352–361, 2023) reported excitation energy migrating over about 6.6 nm along microtubules. Wiest's 2025 review argues that microtubules are plausible targets of inhalational anaesthetics. None of these demonstrates a quantum computation in a neuron, and the 2013 Bandyopadhyay resonance results still lack independent replication.

### 1.2 First-principles results

**Proposition Q1 (decoherence separation) [literature].** With decoherence times τ_dec ≤ 10⁻¹³ s (Tegmark) or ≤ 10⁻⁴ s (the rebuttal's most favourable estimate) and dynamical timescales τ_dyn ≥ 10⁻³ s, the ratio τ_dyn / τ_dec is at least 10 and more probably 10¹⁰. Over any interval on which cognition changes, a superposition of neural states has decohered into a mixture.

*Consequence.* Whatever the physical substrate, the description of cognition at the timescale of a decision is a classical stochastic process: a probability distribution over classical states evolving under a Markov kernel. That is exactly the object Section 3 of the report already models (a stochastic fast policy π₁, a stochastic slow policy π₂, a finite automaton over trace states). Quantum consciousness therefore cannot add a state variable that Nafs-Aql lacks.

**Proposition Q2 (no transferable algorithm) [exact].** Let a mechanism act on n two-level systems. A classical machine that follows the mechanism's state exactly stores 2ⁿ complex amplitudes; in double precision the reference machine's 32 GB hold at most n = 30. Moreover, no theorem states that BQP ⊄ BPP, so no speed-up can be *inherited* by a classical algorithm from the mere fact that a process is quantum.

*Proof.* The state space of n qubits is ℂ^{2ⁿ}; 2³⁰ × 16 bytes = 17 GB. The second statement is the open status of the complexity-class separation. ∎

*Consequence.* Even if Orch OR were established, nothing of it would be computable inside a CPU inference loop, and nothing of it would transfer as an algorithm. Quantum consciousness is recorded as a **negative result** for Nafs-Aql (A-054). This is a stronger statement than "unproven": it fails test 2 regardless of test 1.

### 1.3 Quantum cognition: the one structure that transfers

Quantum cognition uses the mathematics of Hilbert-space probability as a *descriptive* model of human judgement, without any claim about the brain's physics (Busemeyer and Bruza, *Quantum Models of Cognition and Decision*, Cambridge University Press, 2012; Pothos and Busemeyer, *Behavioral and Brain Sciences* 36:255–274, 2013). Its best-tested prediction concerns question-order effects: Wang and Busemeyer (*Topics in Cognitive Science* 5:689–710, 2013) derived the parameter-free QQ equality, confirmed on about 70 national surveys (Wang, Solloway, Shiffrin and Busemeyer, *PNAS* 111:9431–9436, 2014). The structure behind it is non-commutativity: measuring A then B does not give the distribution of measuring B then A.

**Proposition Q3 (order effects are anchoring) [exact].** Let the draft be a "measurement" of the task and the deliberate answer a second one. If deliberation is conditioned on the draft, the joint law of (draft, answer) is in general order-dependent; if deliberation is draft-blind, the two laws factorise and the only shared error is the collision term Σ_{y wrong} π₁(y|x) π₂(y|x) of Proposition 9 of the report.

*Proof.* Conditioning changes the second law by construction; independence removes the conditioning and leaves the product measure, whose joint-error mass is the collision term. ∎

*Consequence.* Nafs-Aql's draft-blind slow path (FR-023) and its coupling test (ER-004) already implement the only engineering consequence of quantum cognition. A 2026 preprint (arXiv 2607.17219) audits question-order effects in language models with the QQ equality and finds them, which is empirical support for keeping the slow path blind. **No change.**

### 1.4 Quantum-inspired algorithms

Dequantised linear-algebra algorithms (Tang, STOC 2019) and tensor-network compression need low-rank structure that a dense 4-bit decoder on AVX2 does not expose at inference time; the repacked weight formats measured in P7-T05 showed no gain, and these methods would require retraining or a different runtime. **Not pursued** (fails test 4 on this machine).

### 1.5 Verdict table

| Idea | Test failed | Record |
|---|---|---|
| Orch OR as a computational principle | 1 (unestablished) and 2 (not transferable, Prop. Q2) | A-054, negative result |
| Decoherence-time arguments | pass; they *justify* the classical model (Prop. Q1) | no change |
| Quantum-cognition order effects | pass; consequence already implemented (Prop. Q3) | no change, cited as support for FR-023 |
| Quantum Zeno / measurement cost | the honest classical form is the inspection problem (Prop. N5) | validates the probe gap, no change |
| Dequantised and tensor-network methods | 4 (not measurable here, no runtime) | not pursued |

## 2. Neuroscience of decision making

### 2.1 What is established

Three results are textbook neuroscience and have exact mathematical forms:

- **Accumulation to a bound.** Perceptual decisions are made by integrating noisy evidence until a bound is reached; neurons in the lateral intraparietal area ramp with the accumulated evidence and reach a common level at the time of the decision (Gold and Shadlen, *Annual Review of Neuroscience* 30:535–574, 2007). The optimal form of this process is Wald's sequential probability ratio test (SPRT): among all tests with the same error probabilities it needs the fewest observations on average (Wald, *Annals of Mathematical Statistics* 16:117–186, 1945; Wald and Wolfowitz, *Annals of Mathematical Statistics* 19:326–339, 1948).
- **Collapsing bounds under a cost of time.** When evidence has a cost or a deadline, the Bayes-optimal bound is not constant: it decreases with elapsed time (Frazier and Yu, *NIPS* 2007, dynamic programming under a stochastic deadline; Drugowitsch, Moreno-Bote, Churchland, Shadlen and Pouget, *Journal of Neuroscience* 32:3612–3628, 2012, who estimated the cost of accumulating evidence in monkeys and humans).
- **Metacognitive sensitivity.** The quality of a confidence signal is the association between confidence and accuracy, measured bias-free with ROC-type statistics (Fleming and Lau, *Frontiers in Human Neuroscience* 8:443, 2014). The all-or-none "ignition" of a global workspace when evidence crosses a threshold (Dehaene and Changeux, *Neuron* 70:200–227, 2011) is the same threshold picture at the level of a whole system.

Nafs-Aql already contains the one-step form of this family: the Aql Gate is Chow's reject rule and the expected-value-of-control decision (report, Theorem 2). What it lacked was the *sequential* form: three of its mechanisms read a stream of evidence and stop on a bound, and each had a hand-set rule where a theorem exists.

| Mechanism | Evidence stream | Bound (before Phase 8) | Sequential result |
|---|---|---|---|
| Early escalation of a draft (P7-T03) | per-token log-probs | running mean below threshold − margin after N = 32 tokens | Prop. N2 (exact window rule), Prop. N3 (Wald bound on the mean rule) |
| Gate's confidence signal | per-token log-probs | mean log-prob (and min, in the calibration) | Prop. N1 (ROC dominance), the confidence-signal study |
| Adaptive exit on non-verifiable tasks (P7-T08) | probed answers | k = 2 consecutive agreeing probes | Prop. N4 (posterior with a collapsing bound) |
| Adaptive exit probing schedule | reflection signals, gap 64, at most 256 | fixed gap | Prop. N5 (optimal inspection interval) |

In the design vocabulary: *lawwāma* (self-reproach) is the bound crossing that flags the first impulse; *tawaqquf* (withholding judgement) is the continuation region; *muhasaba* is the accounting of evidence that decides which. The language-model literature has reached the same rules independently: Adaptive-Consistency stops sampling on a Dirichlet posterior (Aggarwal, Madaan, Yang and Mausam, EMNLP 2023), Early-Stopping Self-Consistency (Li et al., ICLR 2024) and DeepConf's "lowest group confidence" (Fu et al., arXiv 2508.15260, 2025) stop traces on local confidence.

### 2.2 Results

**Proposition N1 (ROC dominance) [exact].** Let s and s′ be scalar confidence signals of a draft and let their ROC curves for predicting "draft wrong" be R and R′ (recall of wrong drafts as a function of the escalation rate g). If R′(g) ≥ R(g) for every g, then for every loss L, price λ and cost Δc the best threshold gate on s′ has an objective J no larger than the best threshold gate on s.

*Proof.* At a fixed escalation rate g the compute term λ Δc g of the objective (report, eq. 1) is the same for both gates and the error term is L · (number of wrong drafts not escalated) = L · W · (1 − R(g)); dominance makes it no larger for s′. Minimising over g preserves the inequality. ∎

*Consequence.* The gate should read the signal with the dominant ROC, measured against exact labels; AUROC summarises it, the ROC at the operating rate decides it. This is what `scripts/confidence_signal_study.py` measures (Section 3.1).

**Proposition N2 (exact early escalation on a running minimum) [exact].** Let ℓ₁, ℓ₂, … be the token log-probs of a draft and define the lowest-window signal w(n) = min over complete windows of W consecutive tokens among the first n tokens (a draft shorter than W is one window). Then w(n) is non-increasing in n. Hence, for a gate rule "escalate iff w(N_final) < θ", the stopping time T = min{n : w(n) < θ} satisfies: (i) stopping at T gives exactly the decision the full draft would receive; (ii) no rule that observes the same stream can escalate such a draft earlier without a non-zero disagreement rate.

*Proof.* w(n) is a minimum over a growing set. (i) If w(T) < θ then w(N_final) ≤ w(T) < θ. (ii) Before T every window seen is ≥ θ, so any earlier escalation contradicts the rule on a draft whose later windows all stay ≥ θ. ∎

*Consequence.* Early escalation on the lowest-window signal is a decision, not a prediction: it needs no margin, no minimum token count and no agreement study. This replaces the P7-T03 mechanism wherever the window rule is in force (`gate.fixed.low_window_logprob_threshold`, `gate.confidence_window`). The mean rule is kept only as the configured heuristic where no window threshold exists.

**Proposition N3 (Wald's bound on the mean rule) [exact under an i.i.d. Gaussian model].** If token log-probs are i.i.d. N(μ, σ²), any sequential test of μ = τ against μ = τ − δ with error probabilities (α, β) has expected sample size under the alternative
E₁[N] ≥ [(1 − β) ln((1 − β)/α) + β ln(β/(1 − α))] / (δ² / 2σ²).

*Proof.* Wald's 1945 lower bound with the Kullback–Leibler divergence of two Gaussians with equal variance. ∎

*Measured values.* On the 40 open-ended drafts of `benchmarks/early_escalation_study.json` the per-token standard deviation is about 0.28 (interquartile 0.23–0.31) and the drafts' final means lie within 0.02–0.10 of the threshold −0.175; the adopted margin is δ = 0.04. With α = β = 0.05 the bound is E₁[N] ≥ 260 tokens, while the drafts are 47–291 tokens long (median 133). The early tokens of a draft are also systematically more confident than the rest (prefix means at 16 tokens are 0.05–0.10 above the final mean), which violates the model in the direction of *more* false early escalations.

*Consequence.* No sequential rule on the running mean can decide the fixed rule's own question within a draft at a controlled error rate. The measured "100 % agreement" of the N = 32, margin 0.04 setting rests on 3 early escalations (Clopper–Pearson lower bound 0.37 at 95 %). The mean rule is therefore labelled an *empirical heuristic* (A-055) and kept at its measured setting; it is not made more aggressive, and the window rule of Prop. N2 is the mechanism with a guarantee.

**Proposition N4 (posterior stopping with a collapsing bound) [exact for the stopping rule; strong for optimality].** Let the probes of System 2's reasoning on a non-verifiable task return answers clustered into categories, with counts v after k probes and a uniform Dirichlet(1, …, 1) prior. The posterior probability that the plurality answer is the majority answer is
P(p_top > ½ | v) = 1 − I_{½}(v_top + 1, k − v_top + 1) = 1 − 2^{−k−1} Σ_{j = v_top + 1}^{k + 1} C(k + 1, j),
where I is the regularised incomplete beta function. Under a per-probe cost and a hard reasoning budget H, the Bayes-optimal rule stops at the first k with P(p_top > ½) ≥ θ(t_k) where θ is non-increasing in the budget used t_k (Frazier and Yu 2007, Theorem 1; the deadline here is deterministic, the simplest case of their class).

*Proof.* The marginal of a Dirichlet component is Beta(v_top + 1, k − v_top + 1), and I_x(a, b) for integer a, b equals P[Binomial(a + b − 1, x) ≥ a]; at x = ½ this is the binomial tail sum. Monotonicity of the optimal threshold is the cited theorem: with fewer probes left, the continuation value falls, so the stopping region grows. ∎

*Values.* One probe gives 0.75; two agreeing probes 0.875; [2, 1] gives 0.6875; [3, 1] gives 0.8125; [4, 1] gives 0.89. With θ(0) = 0.85 and θ(H) = 0.70 the rule reproduces the Dynasor-style "two consecutive agreeing probes" rule at the start of the budget, never stops on a single probe, uses *all* probes rather than the last two (a plurality, Theorem 6 of the report), and lets a [3, 1] split stop once about 45 % of the budget is used.

*Consequence.* `deliberation.adaptive_exit.exit_rule = "posterior"` with `posterior_stop` and `posterior_stop_end`; per request `adaptive_exit_rule`; Lab systems `slow_exit_posterior` and `slow_exit_consistency` for a paired comparison. Verifiable tasks are untouched: the exact verifier is the stopping signal (Theorem 1).

**Proposition N5 (optimal probe interval) [exact under an exponential model].** If the reasoning token count at which the answer becomes available is exponential with mean m, a probe costs c_p seconds and decoding runs at v tokens per second, the periodic probing interval that minimises expected probe cost plus expected overshoot is Δ* = √(2 c_p v m) tokens, and cost(Δ)/cost(Δ*) = (Δ/Δ* + Δ*/Δ)/2.

*Proof.* Expected probes ≈ m/Δ + 1, expected overshoot Δ/2 tokens; minimise c_p m/Δ + Δ/(2v). ∎

*Values.* Idle reference machine: c_p ≈ 0.7 s, v ≈ 8 tok/s, m ≈ 300 (mean 348 reasoning tokens with the exit in run e7607cbb): Δ* ≈ 58 tokens. The configured minimum gap of 64 costs 0.5 % more than the optimum; under the shared-machine load (c_p ≈ 3 s, v ≈ 3 tok/s) Δ* ≈ 73 and the gap costs 1 %. The 256-token fallback is far from optimal (cost ratio 2.3) but only fires when no reflection signal appears.

*Consequence.* The configured schedule is validated; **no change**. This is also the honest classical content of the "quantum Zeno" intuition that measuring too often is costly: here a probe has no back-action (the server keeps the prefix cached), only a price.

## 3. Measurements

All measurements were taken on the shared reference machine (load average 5–8, other sessions compiling) and are labelled accordingly; accuracy and log-probabilities do not depend on load, wall-clock numbers do.

### 3.1 Confidence-signal study (GSM8K, System 1 drafts, exact labels)

150 GSM8K test items (shuffled, seed 20261020), drafted by System 1 under the production prompt and sampling (temperature 0.2, top-p 0.8, no thinking, up to 512 tokens), scored by the exact numeric verifier. **31 of 150 drafts are wrong (accuracy 0.793).** Signals are scored so that a lower value flags a wrong draft; AUROC is the probability that a wrong draft scores below a right one, with a 400-resample bootstrap interval. Recall and precision are reported when the least-confident 10 % and 20 % of drafts are escalated.

| Signal | AUROC [95 % CI] | recall / precision at 10 % | at 20 % |
|---|---|---|---|
| min log-prob (lowest token) | **0.646** [0.539, 0.746] | 0.23 / 0.47 | 0.35 / 0.37 |
| mean top-1/top-2 margin | 0.614 [0.493, 0.729] | 0.23 / 0.47 | 0.39 / 0.40 |
| lowest 16-token window of the margin | 0.611 [0.510, 0.715] | 0.06 / 0.13 | 0.26 / 0.27 |
| lowest 16-token window log-prob (DeepConf) | 0.581 [0.488, 0.698] | 0.13 / 0.27 | 0.29 / 0.30 |
| negative mean top-5 entropy | 0.573 [0.453, 0.679] | 0.16 / 0.33 | 0.26 / 0.27 |
| mean log-prob (the fixed rule's signal) | 0.564 [0.449, 0.672] | 0.13 / 0.27 | 0.26 / 0.27 |
| bottom-10 % tokens mean log-prob | 0.555 [0.459, 0.675] | 0.10 / 0.20 | 0.23 / 0.23 |
| tail (last 32 tokens) mean log-prob | 0.492 [0.366, 0.631] | 0.19 / 0.40 | 0.23 / 0.23 |

Three facts follow.

1. **The fixed rule is blind on mathematics.** At the production threshold (−0.175 at the default λ) the mean-log-prob rule escalates 0 of 150 drafts and catches 0 of the 31 wrong ones: math drafts have a mean log-prob of about −0.03 whether right or wrong. On the unverified mathematics path (a math question in chat without a reference answer) every wrong System 1 draft was therefore kept, and because such drafts never fall into the uncertainty band, the cheap agreement samples were never drawn either.
2. **Token confidence is a weak metacognitive signal for this model on this task.** The best signal is the single lowest token (AUROC 0.65); the DeepConf-style window signal is 0.58 and its interval includes chance. No ROC dominates: the window signal, which Proposition N2 makes exact to act on, is not the signal worth acting on.
3. **Theorem 2 forbids escalating on any of them at the default price.** Escalation pays only if P(wrong | escalated) > ê₂ + λΔc/(L · cost unit) = 0.15 + 0.2 · 150/60 = 0.65 for mathematics. The highest precision reached at any escalation rate of 10 % or more is 0.47. Only at high care (λ ≤ 0.05, bar 0.275) would the lowest-token signal at 20 % escalation (precision 0.37) qualify, and the fixed rule already escalates every unverified hard-domain draft at maximum care.

**Decision.** The lowest-window signal is recorded on every decision and the calibrated gate reads it as a feature, but the fixed-rule window threshold stays **off** (`low_window_logprob_threshold` absent): the mechanism is exact (Prop. N2) and tested, the measured signal is not strong enough to use it (A-057). This is a measured negative result, kept next to the positive ones.

### 3.1b Agreement with cheap samples: the signal the plurality theorem supports

The production gate has a second signal, agreement between the draft and k = 2 cheap System 1 samples (temperature 0.8), but under the fixed rule the samples are drawn only when the draft's mean log-prob lies within 0.10 of the threshold, which on mathematics never happens (Section 3.1). For the first 60 drafts of the study the two samples were drawn exactly as production draws them (seeds 1000 and 1001, `sample_max_tokens_hard`) and compared with the draft's final number, as `answer_similarity` does. 12 of the 60 drafts are wrong (accuracy 0.80); the two samples took 54 s per item on the loaded machine (about 15–20 s idle).

| Signal (first 60 drafts) | AUROC [95 % CI] |
|---|---|
| agreement with the two samples (Theorem 6) | **0.759** [0.618, 0.901] |
| agreement, ties broken by the lowest token | **0.823** [0.690, 0.940] |
| lowest token log-prob | 0.720 [0.553, 0.886] |
| mean log-prob | 0.623 [0.431, 0.811] |
| lowest 16-token window | 0.611 [0.438, 0.792] |

| Rule on agreement | escalation | recall of wrong drafts | precision | expected cascade accuracy (p₂ = 1.0 / 0.9) |
|---|---|---|---|---|
| fewer than half agree (the configured 0.5 at zero care) | 7 % | 0.17 | 0.50 | 0.833 / 0.827 |
| any disagreement (the configured 0.7 at the default care) | 25 % | **0.67** | 0.53 | **0.933 / 0.908** |

Agreement is the only signal whose ROC rises clearly above chance on this task, as Theorem 6 predicts: a wrong draft at temperature 0.2 is usually a draft whose answer does not hold a majority at temperature 0.8. Its precision (0.53) is still below the 0.65 that Theorem 2 demands at the default price for a *calibrated* rule; the fixed rule in force escalates on it regardless, and the measured outcome of doing so is +11 to +13 points on unverified mathematics for one escalation in four plus the samples' own cost.

**Decision (implemented).** `gate.samples_on_unverified_domains = ["math"]`: an unverified math draft always gets its two samples after the draft, whatever its token confidence says (A-060). The fixed rule's existing agreement threshold then does the rest. This is the one accuracy change of Phase 8 that the measurements support, and it comes from the plurality theorem, not from token confidence.

### 3.2 Scale of the window signal across domains

On the 40 open-ended drafts of the early-escalation study the lowest 16-token window has median −0.32 (quantiles 10/90 %: −0.44 / −0.18); on the GSM8K drafts it is about −0.1. A threshold set on mathematics would escalate most open-ended prompts. The window rule therefore applies only to the domains where its threshold was measured (`gate.fixed.low_window_domains = ["math"]`, A-057).

### 3.3 Expected effect on the unverified mathematics path

By Proposition 15 of the report (cascade difference), the expected accuracy of a gated system that escalates a fraction g of these drafts is (1 − g)(1 − ρ₁) + g p₂, where ρ₁ is the error among kept drafts and p₂ System 2's accuracy (11 of 11 in run e7607cbb; 0.9 taken as a conservative bound). Offline, on the 150 drafts:

| Rule | escalation | expected accuracy (p₂ = 1.0 / 0.9) |
|---|---|---|
| fast only (today's unverified-math path) | 0 % | 0.793 |
| lowest-token log-prob < −1.19 | 10 % | 0.840 / 0.830 |
| lowest-token log-prob < −1.03 | 20 % | 0.867 / 0.847 |
| lowest-window log-prob < −0.21 | 10 % | 0.820 / 0.810 |
| lowest-window log-prob < −0.18 | 20 % | 0.853 / 0.833 |

A 20 % escalation rate costs about 0.2 × 150 s = 30 s of mean latency per math question for about +5 points; at the default price (λ = 0.2 per minute) the gate's objective rates that as a loss, which is the same conclusion as Section 3.1 in cost terms.

### 3.4 Adaptive exit rules on factual prompts

Paired Lab run `73cd5bba` (`benchmarks/exit_rule_run.json`): six factual prompts from `factual_probe`, systems `slow_exit_consistency` and `slow_exit_posterior`, 512-token budget, shared machine (load 6–9). Both arms answered every prompt with the same factual content (Jakarta on Java; al-Ghazālī; 1969; 212 °F; Jupiter with 95 moons; the Tigris), with different wording because the final answer is sampled after the reasoning. Reasoning tokens: consistency 357 mean (183, 511, 223, 307, 408, 511), posterior 382 mean (318, 511, 238, 203, 511, 511); mean wall 93 s against 101 s. Two prompts exhausted the budget under both rules. On the one prompt where the rules diverged most (`open-00`), the probes disagreed first and then agreed: the consistency rule stopped on the two consecutive agreeing probes at 183 tokens, the posterior rule saw a [2, 1] split (0.69 < 0.80) and continued to a [3, 1] split at 318 tokens, exactly as Prop. N4 says it should. That caution is the rule's point, and on this set it costs tokens rather than saving them.

**Decision.** By the criterion fixed in advance (A-058: default only with unchanged answers *and* no token increase) the posterior rule stays an **option** (`exit_rule = "posterior"`, per request `adaptive_exit_rule`) and `consistency` remains the default. The comparison is indicative (n = 6, no exact labels); the Lab preset "Exit rule" repeats it on more prompts.

## 4. What changed in the code

- `nafs-core/src/sequential.rs`: `low_window_mean`, `WindowTracker`, `ProbePosterior`, `beta_half_cdf`, `collapsing_bound`, `wald_expected_n_lower`, `optimal_probe_interval`, with tests of the monotonicity (Prop. N2), the closed forms (Prop. N4), the Wald numbers (Prop. N3) and the interval (Prop. N5).
- Gate features gain `low_window_logprob` (recorded on every decision; the calibration's logistic model reads it as its eighth feature). The fixed rule gains the window reason for the domains in `low_window_domains`.
- The draft stream stops exactly when a window crosses the threshold (`EarlyStop { below_window }`), with the reason in the decision's explanation.
- The adaptive exit gains the posterior rule with the collapsing bound; Lab systems `slow_exit_posterior` / `slow_exit_consistency`.
- `gate.samples_on_unverified_domains`: unverified math drafts always get the two agreement samples (Section 3.1b), the one accuracy change the measurements support.
- Nothing was added for the quantum-consciousness ideas; Propositions Q1–Q3 are the record of why.

## 5. Sources

Opened and checked on 2026-10-20 (title, authors, venue, year from the linked page):

- Tegmark, M. (2000). Importance of quantum decoherence in brain processes. *Physical Review E* 61(4):4194–4206. https://arxiv.org/abs/quant-ph/9907009 ; https://pubmed.ncbi.nlm.nih.gov/11088215/
- Hameroff, S., Penrose, R. (2014). Consciousness in the universe: A review of the 'Orch OR' theory. *Physics of Life Reviews* 11(1):39–78. https://pubmed.ncbi.nlm.nih.gov/24070914/
- Babcock, N. S. et al. (2024). Ultraviolet superradiance from mega-networks of tryptophan in biological architectures. *Journal of Physical Chemistry B* 128(17):4035–4046. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11075083/
- Busemeyer, J. R., Bruza, P. D. (2012). *Quantum Models of Cognition and Decision*. Cambridge University Press.
- Pothos, E. M., Busemeyer, J. R. (2013). Can quantum probability provide a new direction for cognitive modeling? *Behavioral and Brain Sciences* 36(3):255–274. https://www.cambridge.org/core/product/6F36FF614A8B06BDC0F2BB878E0C9105
- Wang, Z., Busemeyer, J. R. (2013). A quantum question order model supported by empirical tests of an a priori and precise prediction. *Topics in Cognitive Science* 5(4):689–710. https://jbusemey.pages.iu.edu/quantum/QuestOrdEff.pdf
- Wang, Z., Solloway, T., Shiffrin, R. M., Busemeyer, J. R. (2014). Context effects produced by question orders reveal quantum nature of human judgments. *PNAS* 111(26):9431–9436.
- Auditing question-order effects in large language models with the QQ equality (2026). arXiv:2607.17219. https://arxiv.org/pdf/2607.17219
- Wald, A. (1945). Sequential tests of statistical hypotheses. *Annals of Mathematical Statistics* 16(2):117–186.
- Wald, A., Wolfowitz, J. (1948). Optimum character of the sequential probability ratio test. *Annals of Mathematical Statistics* 19(3):326–339. https://www.projecteuclid.org/euclid.aoms/1177730197
- Gold, J. I., Shadlen, M. N. (2007). The neural basis of decision making. *Annual Review of Neuroscience* 30:535–574.
- Frazier, P. I., Yu, A. J. (2007). Sequential hypothesis testing under stochastic deadlines. *NIPS* 20:465–472. https://proceedings.neurips.cc/paper/2007/hash/9c82c7143c102b71c593d98d96093fde-Abstract.html
- Drugowitsch, J., Moreno-Bote, R., Churchland, A. K., Shadlen, M. N., Pouget, A. (2012). The cost of accumulating evidence in perceptual decision making. *Journal of Neuroscience* 32(11):3612–3628. https://repository.cshl.edu/id/eprint/25448/
- Fleming, S. M., Lau, H. C. (2014). How to measure metacognition. *Frontiers in Human Neuroscience* 8:443. https://www.frontiersin.org/articles/10.3389/fnhum.2014.00443
- Dehaene, S., Changeux, J.-P. (2011). Experimental and theoretical approaches to conscious processing. *Neuron* 70(2):200–227.
- Howard, S. R., Ramdas, A., McAuliffe, J., Sekhon, J. (2021). Time-uniform, nonparametric, nonasymptotic confidence sequences. *Annals of Statistics* 49(2):1055–1080. https://projecteuclid.org/journals/annals-of-statistics/volume-49/issue-2/Time-uniform-nonparametric-nonasymptotic-confidence-sequences/10.1214/20-AOS1991.full
- Aggarwal, P., Madaan, A., Yang, Y., Mausam (2023). Let's sample step by step: Adaptive-Consistency for efficient reasoning and coding with LLMs. *EMNLP* 2023:12375–12396. https://aclanthology.org/2023.emnlp-main.761
- Li, Y. et al. (2024). Escape sky-high cost: Early-stopping self-consistency for multi-step reasoning. *ICLR* 2024. https://proceedings.iclr.cc/paper_files/paper/2024/hash/3fe2a777282299ecb4f9e7ebb531f0ab-Abstract-Conference.html
- Fu, Y. et al. (2025). Deep think with confidence. arXiv:2508.15260. https://arxiv.org/abs/2508.15260
- Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience* 11:127–138. https://www.nature.com/articles/nrn2787
