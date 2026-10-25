# Decision 004: DSpark speculative decoding for System 2: skip

Date 2026-10-21. Status: **skip (do not adopt); re-open on the conditions below.** No change to serving, configuration or the manifest.

## What was assessed

llama.cpp 797de63 already supports DSpark (`--spec-type draft-dspark`). The only published drafter for a Qwen3 4B target is `deepseek-ai/dspark_qwen3_4b_block7` (1.39 B parameters, 1.07 GB as Q8_0), trained for base **Qwen3-4B**, not the **Qwen3-4B-Thinking-2507** that serves as System 2. Nothing exists for System 1 (Qwen3-1.7B). Adoption rule (FR-081): at least 1.2x decode speed at the production temperature, with greedy outputs identical.

## Evidence (raw files in `benchmarks/dspark_probe/`)

**1. Cost of verifying a block** (`llama-bench`, Qwen3-4B-Thinking-2507 Q4_K_M, 4 threads, context depth 1,024, load average about 6, so noisy):

| Tokens verified in one step | Time per step | Relative to one decode step |
|---|---|---|
| 1 | 148 ms | 1.00 |
| 2 | 214 ms | 1.44 |
| 4 | 271 ms | 1.82 |
| 8 (a full block of 7 plus the anchor) | 567 ms | 3.82 |

**2. Acceptance of the drafter on Thinking-2507** (greedy, 160 tokens, block 7, no confidence cut-off; acceptance does not depend on host speed):

| Domain | Drafted | Accepted | Per drafted token | Accepted per block of 7 |
|---|---|---|---|---|
| math | 266 | 121 | 45% | 3.2 |
| code | 363 | 107 | 29% | 2.1 |
| factual | 462 | 91 | 20% | 1.4 |
| all three | 1,091 | 319 | 29% | 2.0 |

**3. Resulting bound** (full-block mode, ignoring the drafter's own cost, which only lowers it): tokens per step is 1 plus the accepted tokens, against a step that costs 3.82 decode steps: math 1.10x, code 0.80x, factual 0.62x, overall 0.80x. The 1.2x target needs about 4 accepted tokens per block; the drafter delivers 1.4 to 3.2. Greedy outputs matched the plain server on 2 of 3 prompts (one factual output differed).

**4. What could not be measured.** Real decode speed. The probe server ran at 0.4 tokens/s, but the host was swap-bound at the time (swap 21 to 29 GB of 30 GB, 52,000 pages swapped in per 15 s), and the plain production System 2 dropped to 0.5 tokens/s on one prompt in the same window, so those speeds are discarded. A first attempt to start four test servers at once pushed swap from 21 to 26 GB and was stopped.

## Why skip

1. **The valid numbers point the wrong way.** A full 7-token block loses on code and factual prompts and gains at most 10% on math. Confidence truncation could help math (a shorter block verifies at 1.8x instead of 3.8x) but cannot help factual or code, whose acceptance is 20 to 30%, and the speed of that mode was not measurable here.
2. **Where it could help, a better lever is already adopted.** The adaptive reasoning exit cuts System 2 math reasoning by half (STATUS P7-T08); whether drafting still multiplies the remaining tokens was not shown, and the exit's probes force prefix re-evaluations that drafting does not speed up.
3. **Memory.** The reference host is already swap-bound with the owner's other workloads. The drafter adds 1.07 GB and its own compute buffers to the production System 2 permanently.
4. **Licence.** The drafter repository declares no licence (no model card at revision 3457dff). A new model artefact needs a recorded licence under G5 (rule 5 of the completion plan).
5. **Mismatch.** Trained for the base model; acceptance on Thinking-2507 is the measured 29% overall.

## Re-open when all of these hold

- a drafter trained for (or fine-tuned on) the Thinking-2507 checkpoint, with a declared licence, and a measured acceptance of at least 4 per block on math;
- a host with at least 2 GB of free RAM beyond System 1, System 2 and the owner's workloads, or the CPU profile moved to a larger machine;
- the interleaved A/B (`scripts/dspark_ab.py` in the probe session's branch, run with `--port-base`; macOS reserves port 8021 for its FTP proxy) on an idle host, showing at least 1.2x at the production temperature.

## Housekeeping

- The probe session's branch (`claude/priceless-solomon-0e04b3`: a manifest entry and `scripts/dspark_ab.py`) is **not merged**. `scripts/dspark_accept_probe.py` (this decision's probe) is committed for reproduction.
- The converted drafter `models/Qwen3-4B-dspark-block7-Q8_0.gguf` (1.07 GB) is untracked and can be deleted with `rm models/Qwen3-4B-dspark-block7-Q8_0.gguf`.
