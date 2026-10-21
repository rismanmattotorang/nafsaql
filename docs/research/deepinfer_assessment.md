# DeepInfer: can Nafs-Aql use it on CPU?

Assessment of `third_party/DeepInfer` (github.com/deepseek-ai/DeepInfer, commit `9bd9cf2`, Apache-2.0), 2026-10-21. Read against the measured limits of the reference machine (Intel Core i3-8100, 4 cores, AVX2 only, 32 GB DDR4-2400, no GPU) and the pinned llama.cpp `797de63`. No benchmark was run for this note: the host was busy with evaluation run `553a4eff`, and a timing taken beside it would be noise.

## Verdict

| Question | Answer |
|---|---|
| Adopt DeepInfer as the inference engine? | **No.** It has no CPU backend, supports one model, and needs a toolchain this Mac cannot run. |
| Port its kernels? | **No.** They are CUDA (SM100), Ascend C and TileLang written for tensor cores and FP8/MXFP8. |
| Use its techniques? | **One is worth a probe, one is worth building, the rest do not apply.** |

## Why the engine itself is out

1. **Hardware.** `deepinfer/platform.py` defines exactly two platforms, `GPU` and `NPU`. `setup.py` builds either a `CUDAExtension` (links `libcuda`) or an Ascend `NpuExtension`; there is no CPU build. The README lists NVIDIA SM100 GPUs and Huawei Ascend 950 NPUs only. `deep_cluster/hardware/cpu.py` is only NUMA topology parsing for pinning host threads next to GPUs.
2. **Model.** The only supported model is DeepSeek-V4.1-Flash, a mixture-of-experts model whose published benchmarks use 4 GPUs for prefill and 64 for decode. It does not fit in 32 GB, and Qwen3 is not supported.
3. **Toolchain.** Linux, GCC 13, Python 3.12, `torch==2.13`, CUDA 13 or CANN 9.2. The last macOS x86_64 PyTorch wheel is 2.2.2; this is the same wall that ruled out native vLLM here (`docs/roadmap/STATUS.md`, P7-T10). DeepInfer was itself built on vLLM, whose CPU path measured 3.2 tok/s on this host.
4. **Kernels.** FlashMLA, DeepGEMM, DeepEP, DeepSelect and the TileLang ops target tensor cores, FP8 and multi-GPU communication. None has an AVX2 path, and the bottleneck they solve (keeping thousands of GPU cores busy) is not ours: our decode is memory-bandwidth bound at batch 1 and compute-bound at batch 4.

## Technique by technique

| DeepInfer feature | What it does | Fit for Nafs-Aql on this CPU |
|---|---|---|
| **DSpark speculative decoding** (DFlash block drafter + Markov head + confidence head) | A small drafter emits a block of up to 7 tokens in one forward pass; a confidence head predicts each token's acceptance probability | **Probe it.** Already implemented in our pinned llama.cpp (`--spec-type draft-dspark`, `--spec-draft-p-min` truncates the block at low confidence). See below |
| **Confidence-scheduled (variable-length) verification** (`csrc/engine/mtp_varlen.hpp`) | Sorts draft tokens by predicted acceptance and verifies the prefix that maximises expected accepted tokens per millisecond, against a profiled step-time curve corrected online by rolling medians | **The idea transfers, the code does not.** Its value grows with how steeply verify cost rises with draft length, and on this CPU it rises steeply. llama.cpp's `--spec-draft-p-min` is the single-sequence version of the same rule |
| **Online runtime estimators** (`SamplerRuntimeEstimator`, `ForwardRuntimeEstimator`: sliding-window and two-level medians) | Keeps the cost model honest as load and context length change | **Build the analogue in the gate.** See "Live latency cost" below |
| SWA bounded replay (YOCO prefill truncation) | Recomputes only the last 128 tokens of a sliding-window cache after a prefix hit | Not applicable: Qwen3 dense models have no sliding-window layers. Prefix caching already serves 73 to 83% of prompt tokens |
| Zero-overhead (asynchronous) scheduling | Overlaps CPU-side scheduling with GPU execution | Not applicable: on a CPU-only host the CPU is the device; there is nothing to overlap |
| Batch-invariant inference, deterministic sampling | Identical results across batch compositions | NVIDIA-only in DeepInfer. Our paired A/Bs already pin seeds and interleave rounds; not worth porting |
| Structured outputs (xgrammar) | Regex, JSON Schema and EBNF constraints | Already covered: llama.cpp grammars and JSON Schema constraints are in use (the judge) |
| Expert, data and context parallelism, EPLB, prefill/decode disaggregation | Multi-GPU scale-out | Not applicable to one 4-core host |
| Evaluation harness (`eval/`: AIME, GPQA, HLE, pass@k, judge) | Benchmark scripts | Nafs-Aql's harness already covers its domains with exact verifiers |

## DSpark: the one candidate with a real chance

**What exists.**

- llama.cpp `797de63` ships DFlash and DSpark drafting; the built `llama-server` lists `draft-dspark` among its `--spec-type` values.
- `deepseek-ai/dspark_qwen3_4b_block7` (1.39 B parameters, 2.8 GB safetensors, includes embeddings) is trained for `Qwen/Qwen3-4B`.
- Nothing exists for Qwen3-1.7B, so **System 1 is out of scope**; System 1 is already fast.

**Why it might beat EAGLE-3, which gave no gain here.** EAGLE-3 drafts one token per drafter step, so an 8-token draft costs 8 sequential drafter passes. DSpark drafts a whole block in one pass. The confidence head also lets llama.cpp stop the block early, which matters on a steep verify-cost curve.

**What the stored EAGLE-3 run says about that curve** (`benchmarks/eagle3_ab_system2.json`, System 2, shared machine, load 6.6). Tokens per step are one plus drafts times acceptance; the step cost follows from the measured speed-up.

| Draft length | Acceptance per drafted token | Tokens per step | Measured speed-up | Implied step cost, in single decode steps |
|---|---|---|---|---|
| 4 | 0.46 | 2.83 | 0.99x | 2.9 |
| 8 | 0.37 | 3.93 | 0.87x | 4.5 |

Read this as: drafting and verifying 8 tokens costs about 4.5 ordinary decode steps on four AVX2 cores. On a GPU the same verification is nearly free, which is why DSpark reports large gains there.

**Break-even arithmetic for DSpark, block 7.** Assume one drafter pass plus an 8-position verify costs 3.5 to 4.2 decode steps (cheaper than EAGLE-3 at 8 because drafting is one pass, not eight).

| Accepted tokens per block | Tokens per step | Speed-up at step cost 3.5 | At step cost 4.2 |
|---|---|---|---|
| 2.5 | 3.5 | 1.00x | 0.83x |
| 3.5 | 4.5 | 1.29x | 1.07x |
| 4.5 | 5.5 | 1.57x | 1.31x |

FR-081 asks for 1.2x. That needs about 3.5 to 4 accepted tokens per block of 7, on this CPU, against our checkpoint.

**The open risk.** The drafter was trained on hidden states of `Qwen/Qwen3-4B`. Our System 2 is `Qwen3-4B-Thinking-2507`, a further-trained checkpoint with the same architecture and tokenizer. DFlash-family drafters read the target's hidden states, so the mismatch will lower acceptance by an amount only a measurement can give. Long thinking traces on math are repetitive and favourable to drafting; that cuts the other way.

**Probe, at most an hour of machine time, on an idle host** (the completion plan's rule for engine changes: adopt only at >= 1.3x on a probe, then an interleaved paired A/B):

```bash
# 1. Fetch the drafter (2.8 GB; needs the owner's approval). The System 2 safetensors are not on the
#    Mac's disk (models/hf/Qwen3-4B-Thinking-2507/STAGED_IN_VM.txt): restore them first (the vllm target fetches both HF checkpoints, about 11.5 GB).
huggingface-cli download deepseek-ai/dspark_qwen3_4b_block7 --local-dir models/hf/dspark_qwen3_4b_block7
scripts/fetch_models.sh vllm
# 2. Convert against our System 2 checkpoint (inherits its tokenizer and embeddings), with the
#    conversion venv used for the EAGLE-3 heads
tools/venv-convert/bin/python third_party/llama.cpp/convert_hf_to_gguf.py models/hf/dspark_qwen3_4b_block7 \
    --target-model-dir models/hf/Qwen3-4B-Thinking-2507 --outtype q8_0 --outfile models/Qwen3-4B-DSpark-Q8_0.gguf
# 3. Measure the verify-cost curve directly (the number every speculative scheme depends on)
third_party/llama.cpp/build/bin/llama-bench -m models/Qwen3-4B-Thinking-2507-Q4_K_M.gguf -t 4 -r 3 -n 0 -d 1024 -p 1,2,4,8
# 4. Interleaved A/B on System 2 with the eagle3_ab.py protocol: plain vs draft-dspark at n-max 7 and p-min 0, 0.3, 0.5
```

Adopt only if the paired A/B shows >= 1.2x decode at the production temperature with acceptance logged, and greedy outputs identical to the plain server. Record a negative result otherwise; it would close the speculative-decoding question for this hardware.

## Live latency cost for the gate

DeepInfer keeps its cost model honest with online rolling medians. The Aql Gate does not: its latency cost per domain (`Orchestrator::delta_c`) comes from the configuration or from derived parameters, and it is static. On this shared machine System 2's decode speed varies by a factor of 6 to 9 with the owner's load (10.3 tok/s idle, 2.6 to 4.3 under load, 6.9 in service today). The gate therefore prices deliberation the same at load 2 and at load 12, so it escalates too readily when the host is busy.

The analogue is small and needs no new dependency:

- keep a rolling median of System 2 decode tok/s from the stage timings the gateway already records (`usage.stages`, `/v1/perf/stages`);
- scale the latency cost as reference speed over current speed, clamped to a sane range;
- leave it off for evaluation runs, so paired comparisons and calibration stay reproducible;
- adopt it only after a paired evaluation shows routing quality non-inferior under load and fewer needless escalations.

This serves objective 2 (correct routing) more than objective 1, and it is the part of DeepInfer's design that matters most for a shared CPU host.

## Not worth further time

The kernels, the parallelism stack, SWA replay, asynchronous scheduling and batch invariance solve GPU-cluster problems. None should be revisited unless Nafs-Aql gains a GPU profile (P5-T03, deferred by the owner).
