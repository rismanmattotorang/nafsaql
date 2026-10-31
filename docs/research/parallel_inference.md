# Parallel and concurrent inference on four CPU cores (Phase 13)

Date 2026-10-30. Goal set by the owner: assess inference and gate performance and push Nafs-Aql towards the best
performance a four-core CPU can give, using the cores in parallel. Decision record:
[007](../decisions/007-parallel-inference.md). Report: section "Phase 13".

## 1. Assessment: where the time goes

From the 100-prompt run on an idle host (`docs/research/perf_100.md`, `benchmarks/perf_100.json`):

| Finding | Number |
|---|---|
| Thinking on the thorough model | 77.6 % of model time (243 s per call) |
| Quick drafts | 17.9 % (15 s per call, 98 calls) |
| Exact checks, gate arithmetic, progress probes | under 2.5 % together (97 checks took 24 s) |
| Six prompts (4 abstained code, 2 timeouts) | 52 % of all model time, no checked answer |
| Thorough slot queueing | 476 s in total (17 s per thorough call): System 2 had 2 slots |
| Two users | each stream 4-7 tok/s against 10.3 alone |

So the gate and the verifiers are not the bottleneck (the gate is arithmetic on a few floats; checks are 0.25 s).
Decode is, and decode on this CPU is limited by two things: memory bandwidth for one stream, and compute for many.

## 2. The measurement that decides everything: batch scaling

llama-server decodes the sequences of its slots in one batch. `scripts/batch_scaling.py` sends N identical-length
requests at once and measures the wall time (median of 3 interleaved reps, prompt prefill included, so these are
conservative):

| Streams N | System 1 (Qwen3-1.7B) wall factor c(N) / aggregate speed-up | System 2 (Qwen3-4B-Thinking) c(N) / speed-up |
|---|---|---|
| 1 | 1.00 / 1.00x (15.8 tok/s) | 1.00 / 1.00x (6.7 tok/s) |
| 2 | 1.33 / 1.50x | 1.49-1.50 / 1.33-1.34x |
| 3 | 1.78 / 1.69x | 1.90-2.02 / 1.49-1.58x |
| 4 | 1.71 / **2.34x** (36.9 tok/s) | 1.78-1.87 / **2.13-2.25x** (15.0 tok/s) |

(System 2: temporary 4-slot server and then the production server after the change; `batch_scaling_system2*.json`.)

Reading: a single stream is bandwidth-bound (about 16 GB/s of weights per second at both model sizes, consistent
with a DDR4 roofline), so more streams are cheap until compute saturates at about 2.2x. N = 3 is consistently worse
than N = 4 (an awkward split over four cores), so the policy never picks 3. System 2 had **two** slots in production:
at most a 1.33x aggregate. Four slots give 2.13x. This is the one change that helps every multi-user case for free.

## 3. Research anchors (what the literature offers, and what transfers)

| Idea | Source | Verdict for a 4-core CPU |
|---|---|---|
| Repeated sampling plus an exact verifier: coverage grows log-linearly with samples; unit tests and proof checkers turn coverage into accuracy | Brown et al., 2024 (arXiv 2407.21787) | **Adopted** for code: Nafs-Aql has exactly such verifiers |
| Hedged requests: send the same request to several replicas, take the first answer, cancel the rest | Dean and Barroso, CACM 2013 | **Adopted** as the shape of a concurrent round (first verified wins) |
| Parallel scaling: P streams act like O(log P) times the parameters, at much lower memory and latency cost than parameter scaling | Chen et al., 2025 (arXiv 2505.10475) | Motivation only: needs trained transformations; here the streams differ by sampling seed |
| Confidence-based early stopping of parallel reasoning traces | Fu et al., 2025 (DeepConf, arXiv 2508.15260) | The window rule of Phase 8 already does this for drafts; not extended to System 2 here |
| Concurrent attention over a shared cache, workers cooperating | Rodionov et al., 2025 (Hogwild!, arXiv 2504.06261) | Not adopted: small models (1.7B/4B) are reported to struggle to cooperate |
| Model-free n-gram / suffix speculative decoding | Oliaro et al., 2024 (arXiv 2411.04975); llama.cpp `--spec-type ngram-*` | **Measured, not adopted** (section 4) |
| Low-bit table-lookup kernels for CPUs | Wei et al., 2024 (T-MAC, arXiv 2407.00088) | Out of scope: needs 2-3 bit models and a different runtime; accuracy cost on 1.7B/4B unknown |

## 4. Self-speculative (n-gram) decoding: measured, no gain

`scripts/ngram_ab.py`, Qwen3-1.7B, greedy, four prompt kinds, configurations cycled twice
(`benchmarks/ngram_ab_system1.json`). Decode speed relative to no speculation:

| Mode | code repair | code write | JSON | maths |
|---|---|---|---|---|
| ngram-simple | 1.05x | 0.94x | **0.84x** | 1.01x |
| ngram-map-k | 1.09x | 0.96x | 0.96x | 0.98x |
| ngram-mod | **1.12x** (98 of 138 drafted tokens accepted) | 0.98x | 1.00x | 0.98x |

The best case, copying a function back in code repair with 71 % of drafted tokens accepted, gains 12 %; the
worst loses 16 %. Verifying k drafted tokens costs about k decode steps on this compute-bound CPU (decision 004),
so a draft pays only above about 50 % acceptance across the whole output. Not adopted.

## 5. Fusing the agreement samples with the draft (FR-095): adopted

For an unverified maths draft the gate always draws k = 2 cheap samples (A-060). They used to run *after* the draft:
about 1 + c(2) = 2.33 draft times. Fused they share the draft's batch: c(3) = 1.78. Interleaved A/B
(`scripts/fuse_ab.py`, 14 GSM8K prompts, 2 repetitions, stream closed at the gate's decision event so no System 2 run
is started, idle host):

| | sequential | fused |
|---|---|---|
| median time to the gate's decision | 23.1 s | **17.9 s** (-24 %) |
| faster in | | 23 of 28 pairs (exact sign test p = 0.0005) |
| escalations | 7 of 28 | 5 of 28; discordant pairs 4 to 2 (McNemar p = 0.69) |
| median first token | 0.11 s | 0.17 s |

The escalation rates are not distinguishable, and the decision flips as often between two repeats of the *same*
arm (6 of 28) as between arms (6 of 28): the flips are sampling noise. A draft stopped early is going to System 2
whatever the samples say, so the fused samples are dropped (cancelled) at that point.

## 6. Batching fades with context depth

Section 2 measured short prompts. A repair prompt is not short: it carries the earlier candidate and the verifier's
evidence, and the reasoning that follows adds up to 1,024 tokens. `batch_scaling.py --depth` repeats the measurement with a
shared context, prompt cached, decode-only throughput (System 2):

| Context | 1 stream | 4 streams | gain | wall factor c(4) |
|---|---|---|---|---|
| short | 5.12 tok/s | 13.78 | **2.69x** | 1.60 |
| ~800 tokens (600 words) | 5.03 | 9.17 | 1.82x | 2.24 |
| ~1,600 tokens (1,200 words) | 4.09 | 6.14 | **1.50x** | 3.40 |

The gain a batch gives is gone by the depth where System 2 actually works. A quantised KV cache with flash attention
(`-ctk q8_0 -ctv q8_0 --flash-attn on`) does not bring it back: 7.94 tok/s at 4 streams and 800 tokens (against 9.17 with the
f16 cache; one stream is slower too, 3.68 against 5.03) and 6.36 at 1,600 tokens (against 6.14): not adopted, consistent with
decision 006.

## 7. One request with `n` choices shares the prefill

A round of candidates sent as separate requests makes every slot prefill the whole prompt, and prefill (31 tok/s on
System 2) is compute-bound, so it gains nothing from batching. `scripts/prefill_share.py`, a 703-token prompt and a 24-token
answer, cache off, median of 3:

| One request | Four separate concurrent requests | One request with n = 4 |
|---|---|---|
| 31.7 s | **124.5 s (3.9x)** | **39.1 s (1.23x)** |

llama-server prefills once and forks the KV cache for `n` choices. `Backend::stream_n` sends a round as one such request and
hands each choice on as soon as it ends, so each is verified while the others decode. This is what made the first live
hedging runs slow: separate slots cost 3.4 to 3.9 single attempts per round.

## 8. Concurrent hedged rounds on System 2 (FR-094): implemented, off by default

Design (`nafs_core::parallel`, the System 2 loop): a round launches W candidates; each is generated, patched and verified in
its own future; they are consumed in completion order; the first to pass wins and the rest are cancelled by dropping their
stream. Width comes from the measured cost curve and an online success estimate (`U(N) = success / E[time]`), never beyond the
free slots; `repair_rounds` lets a round replace serial repairs; `hedge_after_failure` runs the first attempt alone. A round
has its own deadline 5 s inside the request cap and degrades to the best candidate.

Live A/B on the 8 MBPP+ code items that went to System 2 in the 100-prompt run (4 solved by the first attempt, 4 failed all
three; `scripts/parallel_ab.py`, `benchmarks/parallel_ab.json`, idle host, System 2 forced):

| Arm | Verified | Wall time | Note |
|---|---|---|---|
| seq (1 attempt + 2 serial repairs), run 1 / run 2 | 4/8 / 3/8 | 4,052 s / 4,523 s | item 239 passed on the 3rd attempt once and not the other time |
| four fresh candidates at once, separate requests | 4/8 | 3,308 s | easy items **+44 % to +87 %**; failing items -11 % to -45 % |
| first attempt alone, then four, separate requests | 4/5 | | 3 runs ended at the request cap (HTTP 504): the round overran |
| first attempt alone, then four, one `n = 4` request | 4/7 | 3,558 s on 7 | 1 run ended at the cap before the deadline fix; failing items 872-930 s against 636-901 s sequential |

Outcomes do not differ beyond sampling noise (the two sequential runs differ from each other), and the times show no gain:
a round of four at repair depth costs 2.2 to 3.4 single attempts (section 6), so it is no faster than the two serial repairs
it replaces. The width policy agrees: with the measured deep cost factors `[1.0, 2.4, 3.1, 3.4]` it keeps width 1 for every
success probability from 0.10 up and widens only for p of 0.07 or less (hopeless tasks); with the short-context factors it
widens up to p = 0.48. So concurrent hedging on System 2 is **not** adopted on this CPU: it stays implemented, tested and
off (`deliberation.parallel.enabled = false`).

## 9. Cheap concurrent samples before escalating (FR-096): adopted

Where batching is cheap (System 1, short context: four streams cost 1.7 times one) and the verifier is exact, the cores are
better spent before System 2 than inside it. A failed draft on a checkable task now draws `k = 4` System 1 samples in one
request, verifies each as it finishes, and takes the first that passes.

Offline (`scripts/s1_rescue.py`, 60 MBPP+ items, draft at production settings, samples at 0.7): draft pass@1 46.7 %; of 32
failed drafts 0, 0, 4 and 6 are rescued by 1, 2, 3 and 4 samples (19 % with four); pass@5 56.7 %; four samples in one
request take 17.5 s against 10.4 s for the draft (1.68x).

Live, interleaved (`scripts/rescue_ab.py`, 40 items, same request with rescue off and on, stream closed at the gate's
decision):

| | rescue off | rescue on (4) |
|---|---|---|
| escalations to System 2 | 22 | 20 (3 avoided, 1 the other way: the draft itself is random) |
| extra time when the rescue ran (20 items) | | median 14.5 s |
| total extra time | | 317 s |

Break-even is 106 s per avoided escalation; an escalation on this CPU costs 250 to 580 s (the 8 code items of the 100-prompt run:
median 289 s, mean 580 s), so the rescue nets +10.8 to +35.6 s per code item. Rescued answers pass the exact verifier, so no
accuracy is traded. The sample is small (40 items, 3 avoided escalations) and the offline study agrees on the rate; both are
reported as they are. On by default (`gate.rescue_samples = 4`).

## 10. What changed, and what did not

| Change | Evidence | Default |
|---|---|---|
| Fused agreement samples (FR-095) | -24 % time to decision, p = 0.0005 | on |
| System 1 rescue before escalating (FR-096) | 3 of 19 escalations avoided live; break-even 106 s against 250-580 s | on (4 samples) |
| System 2 with four slots | 2.7x decode throughput at short context, 1.5x at 1,600 tokens | on |
| Concurrent hedged rounds on System 2 (FR-094) | no width pays at repair depth; separate slots prefill the prompt W times | off |
| `n`-choice requests (`stream_n`) | prefill shared: 39.1 s against 124.5 s | used by rounds and the rescue |
| n-gram self-speculation | 1.12x best, 0.84x worst | not adopted |
| q8_0 KV cache with flash attention | no gain at depth | not adopted |
