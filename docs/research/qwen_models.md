# Survey: small Qwen language models for CPU-only inference

Verified on 2026-10-18. Every model row below was checked by opening the Hugging Face page (or the HF `/api/models` listing, where stated). Rows that could not be opened are marked **UNVERIFIED**. Secondary sources (news sites, blogs) are labelled as such and never used for parameters or licences.

Method note: `qwen.ai/blog?id=...` pages are JavaScript-rendered and return an empty shell to a plain fetch; they were read through the `r.jina.ai` text proxy, which is noted in the verification log. The `qwenlm.github.io/blog/` index now stops at September 2025 (Qwen3Guard); all 2026 releases are announced on `qwen.ai/blog`.

---

## 1. Qwen generations (as of October 2026)

| Generation | Release date(s) | Open-weight members | Official announcement / report | Notes |
|---|---|---|---|---|
| **Qwen3** | 2025-04-29 (initial); "2507" refresh Jul–Aug 2025 (Instruct-2507 / Thinking-2507 checkpoints) | 0.6B, 1.7B, 4B, 8B, 14B, 32B dense; 30B-A3B, 235B-A22B MoE; + `-Base`, `-Instruct-2507`, `-Thinking-2507` | Technical report: arXiv:2505.09388 (submitted 2025-05-14), https://arxiv.org/abs/2505.09388 | Standard GQA transformer (`Qwen3ForCausalLM`). Apache-2.0. Single checkpoint with `enable_thinking` switch; 2507 refresh splits Instruct / Thinking. Text-only. |
| **Qwen3-Next** | 2025-09-11 | Qwen3-Next-80B-A3B-Instruct / -Thinking | Blog `qwen.ai/blog?id=qwen3-next` (link taken from QwenLM GitHub README; not opened) | First Qwen with hybrid Gated DeltaNet + gated attention + ultra-sparse MoE. No small variants. Precursor of the 3.5 architecture. |
| **Qwen3.5** | 2026-02-16 (397B-A17B); 2026-02-24 (122B-A10B, 35B-A3B, 27B); 2026-03-02 (9B, 4B, 2B, 0.8B) | 0.8B, 2B, 4B, 9B, 27B dense; 35B-A3B, 122B-A10B, 397B-A17B MoE; `-Base` for all but 27B/122B/397B | Blog "Qwen3.5: Towards Native Multimodal Agents", `https://qwen.ai/blog?id=qwen3.5` (page dated 2026-02-14; read via proxy). No standalone technical report; blog promises an "upcoming technical report". Separate: Qwen3.5-Omni Technical Report arXiv:2604.15804 (2026-04-17). | `Qwen3_5ForConditionalGeneration`; hybrid 3:1 Gated DeltaNet : Gated Attention; vision encoder built in (pipeline `image-text-to-text`); MTP head; 248,320-token vocab; Apache-2.0. API-only sibling: Qwen3.5-Plus. |
| **Qwen3.6** | 2026-04-15/16 (35B-A3B); 2026-04-22 (27B) | Qwen3.6-35B-A3B, Qwen3.6-27B (+ FP8 of each) | "Qwen3.6-35B-A3B: Agentic Coding Power, Now Open to All" `https://qwen.ai/blog?id=qwen3.6-35b-a3b` (2026-04-15); "Qwen3.6-27B: Flagship-Level Coding in a 27B Dense Model" `https://qwen.ai/blog?id=qwen3.6-27b` (2026-04-22). No tech report. | Same `qwen3_5` / `qwen3_5_moe` architecture as 3.5. API-only: Qwen3.6-Plus, Qwen3.6-Max-Preview (2026-04-20), Qwen3.6-Flash (= 35B-A3B hosted). **No ≤9B open weights.** Blog says the open family "keeps expanding ... stay tuned", but nothing smaller had appeared by 2026-10-18. |
| **Qwen3.7** | Preview on Arena 2026-05-14; announced Apsara Summit 2026-05-20 (secondary sources) | **None.** HF API search `author=Qwen&search=Qwen3.7` returns an empty list. | Official URL reported by deeplearning.ai as `https://qwen.ai/blog?id=qwen3.7` — could not be loaded (proxy returned unrelated content). Secondary: deeplearning.ai The Batch (2026-06-05), KuCoin news (2026-05-19). | Qwen3.7-Max (text) and Qwen3.7-Plus(-Preview) (multimodal) are API-only; "its weights are not open" (The Batch). Treat as a closed generation. |
| **Qwen3.8** | Blog 2026-08-03; weights 2026-08-12 (2.4T-A95B), 2026-08-14 (27B), 2026-08-27 (Flash-Next) | Qwen3.8-2.4T-A95B, Qwen3.8-27B, Qwen3.8-Flash-Next (180B) (+ FP8 of each) | "Qwen3.8-Max: A New Bar for Coding and Cowork", `https://qwen.ai/blog?id=qwen3.8` (2026-08-03, read via proxy). No tech report. | Qwen3.8-27B is `Qwen3_5ForConditionalGeneration` (same hybrid arch, 64 layers). Qwen3.8-Flash-Next is a **new** architecture (`qwen4_exp`: hyper-connections, GDN, MoE, sparse "QSA" indexer, n-gram embeddings). **No ≤9B open weights.** |

Source for the timeline: https://github.com/QwenLM/Qwen3.8 README "News" section (the repo covers 3.5, 3.6 and 3.8) and the HF collections `Qwen/qwen35`, `Qwen/qwen36`, `Qwen/qwen38`.

**Bottom line for a CPU design document:** the only generations with open text LLMs ≤9B are **Qwen3** (0.6B/1.7B/4B/8B, plus the 4B 2507 refresh) and **Qwen3.5** (0.8B/2B/4B/9B). Qwen3.6 and Qwen3.8 bottom out at 27B dense / 35B-A3B MoE; Qwen3.7 has no open weights at all.

---

## 2. All text-LLM variants ≤9B, per generation

### 2.1 Qwen3 (April 2025) — `Qwen3ForCausalLM`, `model_type: qwen3`, standard GQA

Common to all rows: Apache-2.0; text-only (not multimodal); vocab 151,936; config verified on `Qwen/Qwen3-4B/raw/main/config.json` (`architectures: ["Qwen3ForCausalLM"]`, `max_position_embeddings: 40960`, `tie_word_embeddings: true`). Thinking control: `enable_thinking` kwarg in `apply_chat_template` (default **on**), plus `/think` and `/no_think` soft switches inside user turns.

| HF repo (opened) | Total / non-embedding params | Layers | Attention (Q/KV heads) | Context | Thinking | Notes |
|---|---|---|---|---|---|---|
| `Qwen/Qwen3-0.6B` | 0.6B / 0.44B (HF tag 0.8B) | 28 | GQA 16/8 | 32,768 | switchable (`enable_thinking`) | Smallest; `KeyError: 'qwen3'` on transformers < 4.51 |
| `Qwen/Qwen3-1.7B` | 1.7B / 1.4B | 28 | GQA 16/8 | 32,768 | switchable | |
| `Qwen/Qwen3-4B` | 4.0B / 3.6B | 36 | GQA 32/8 | 32,768 native; 131,072 with YaRN | switchable | hidden 2560 |
| `Qwen/Qwen3-8B` | 8.2B / 6.95B | 36 | GQA 32/8 | 32,768 native; 131,072 with YaRN | switchable | |
| `Qwen/Qwen3-4B-Instruct-2507` | 4.0B / 3.6B | 36 | GQA 32/8 | **262,144** native | **non-thinking only** (no `<think>` blocks; `enable_thinking=False` not needed) | 2507 refresh |
| `Qwen/Qwen3-4B-Thinking-2507` | 4.0B / 3.6B | 36 | GQA 32/8 | **262,144** native | **thinking only** (template always emits `<think>`) | 2507 refresh |
| `Qwen/Qwen3-0.6B-Base`, `Qwen/Qwen3-1.7B-Base`, `Qwen/Qwen3-4B-Base` | as above | | | | pretrained only | Listed in HF API org query; pages not opened individually |

There is no 0.6B/1.7B/8B "-2507" refresh in the Qwen org listing (only 4B, 30B-A3B and 235B-A22B got 2507 checkpoints).

### 2.2 Qwen3.5 (Feb–Mar 2026) — `Qwen3_5ForConditionalGeneration`, `model_type: qwen3_5` (text sub-config `qwen3_5_text`), hybrid Gated DeltaNet

Common to all rows (verified on `Qwen/Qwen3.5-0.8B/raw/main/config.json` and the four model cards):
- `layer_types`: repeating `[linear_attention, linear_attention, linear_attention, full_attention]` (`full_attention_interval: 4`). "linear_attention" = **Gated DeltaNet** (`linear_conv_kernel_dim: 4`, head dim 128); "full_attention" = **Gated Attention** with GQA and `partial_rotary_factor 0.25`, head dim 256.
- `vision_config` present → **multimodal by default** (pipeline tag `image-text-to-text`). Text-only serving is possible (vLLM `--language-model-only`; llama.cpp simply omits the `mmproj`).
- Multi-token-prediction (MTP) head shipped in the weights; vocab 248,320 (padded); tied embeddings on the small models.
- Context 262,144 native (4B/9B cards: extendable to ~1,010,000 with YaRN).
- Licence Apache-2.0 (all four).
- Thinking control: `enable_thinking` passed via `chat_template_kwargs` (Transformers / vLLM / SGLang). **Qwen3.5 does not support the `/think` `/nothink` soft switch.** Single checkpoint per size (no separate Instruct/Thinking files).
- No tech report; cards cite the blog (`qwen.ai/blog?id=qwen3.5`, "Qwen3.5: Towards Native Multimodal Agents", Feb 2026).
- Dense FFN in all four (MoE only from 35B-A3B upwards).

| HF repo (opened) | Params (card / HF tag) | Layers, hidden, FFN | Gated DeltaNet heads (V / QK) | Gated Attention (Q / KV) | Thinking default | Notes |
|---|---|---|---|---|---|---|
| `Qwen/Qwen3.5-0.8B` | 0.8B / 0.9B | 24, 1024, 3584 | 16 / 16 | 8 / 2 | **non-thinking** (card); card warns the 0.8B is "more prone to thinking loops" in thinking mode | `-Base` exists |
| `Qwen/Qwen3.5-2B` | 2B / 2B | 24, 2048, 6144 | 16 / 16 | 8 / 2 | **non-thinking** (card); same loop warning | `-Base` exists |
| `Qwen/Qwen3.5-4B` | 4B / 5B | 32, 2560, 9216 | (card: "most layers GDN") | 16 / 4 | **thinking on** (card: pass `enable_thinking: False` to disable) | `-Base` exists |
| `Qwen/Qwen3.5-9B` | 9B / 10B | 32, 4096, 12288 | 32 / 16 | 16 / 4 | **thinking on** (card) | `-Base` exists; SAE repos `Qwen/SAE-Res-Qwen3.5-9B-Base-*` also exist |

Discrepancy to flag: the Unsloth Qwen3.5 guide (`unsloth.ai/docs/models/qwen3.5`) says reasoning is *disabled by default for 0.8B/2B/4B/9B* and must be enabled with `--chat-template-kwargs '{"enable_thinking":true}'`, whereas the official 4B and 9B cards say thinking is on by default. The behaviour is template-driven, so check the chat template in the GGUF you ship.

Not in scope but relevant for "3B-active" CPU use: `Qwen/Qwen3.5-35B-A3B` (36B total, 3B active, `qwen3_5_moe`) — >9B total, not listed above.

### 2.3 Qwen3.6 (April 2026) — none ≤9B

HF API `author=Qwen&search=Qwen3.6` returns exactly four repos: `Qwen/Qwen3.6-35B-A3B`, `-35B-A3B-FP8`, `Qwen/Qwen3.6-27B`, `-27B-FP8`. For reference (pages opened):

| HF repo | Params | Architecture | Context | Licence | Thinking | Multimodal |
|---|---|---|---|---|---|---|
| `Qwen/Qwen3.6-35B-A3B` | 35B total / ~3B active; 40 layers; 256 experts, 8 routed + 1 shared | `qwen3_5_moe` (GGUF arch `qwen35moe`); hybrid GDN + Gated Attention + MoE | 262,144 (YaRN to 1,010,000) | Apache-2.0 | on by default; `enable_thinking`; `preserve_thinking` option; no soft switch | yes |
| `Qwen/Qwen3.6-27B` | 27B (HF tag 28B); 64 layers, hidden 5120; 48 linear-attention + 16 full-attention layers | `Qwen3_5ForConditionalGeneration` (config.json opened) | 262,144 | Apache-2.0 | same as above | yes |

### 2.4 Qwen3.7 — no open weights (see §1).

### 2.5 Qwen3.8 (August 2026) — none ≤9B

HF API `author=Qwen&search=Qwen3.8` returns six repos: `Qwen3.8-27B`, `-27B-FP8`, `Qwen3.8-2.4T-A95B`, `-2.4T-A95B-FP8`, `Qwen3.8-Flash-Next`, `-Flash-Next-FP8`. For reference:

| HF repo (opened) | Params | Architecture | Context | Licence | Thinking | Multimodal |
|---|---|---|---|---|---|---|
| `Qwen/Qwen3.8-27B` | 27B (HF tag 28B); 64 layers, hidden 5120; 48 `linear_attention` + 16 `full_attention`; GDN 48 V / 16 K heads; attention 24 Q / 4 KV | `Qwen3_5ForConditionalGeneration`, `model_type qwen3_5` (config.json opened) | 262,144 (YaRN to 1,000,000) | Apache-2.0 | on by default; `enable_thinking`; `reasoning_effort` xhigh (default) / medium / low; `preserve_thinking` default on | yes |
| `Qwen/Qwen3.8-Flash-Next` | 180B (HF tag) | `qwen4_exp` (new; see llama.cpp §4) | — | — | — | yes (not opened; from org listing) |

The Qwen3.8-27B card names no smaller or planned Qwen3.8 models.

---

## 3. Quantised versions usable on CPU

CPU-only inference in practice means **GGUF + llama.cpp** (or its wrappers: Ollama, LM Studio, llama-cpp-python). AWQ / GPTQ-Int4 / FP8 repos are GPU formats for vLLM/SGLang and are listed only for completeness.

### 3.1 Official Qwen-org quantisations

| Repo (status) | Format | Quant types shown on card | Licence | Note |
|---|---|---|---|---|
| `Qwen/Qwen3-0.6B-GGUF` (opened) | GGUF | Q8_0 (639 MB) only | Apache-2.0 | card gives `llama-server -hf Qwen/Qwen3-0.6B-GGUF:Q8_0` |
| `Qwen/Qwen3-1.7B-GGUF` (opened) | GGUF | Q8_0 (1.83 GB) (+1 unnamed variant) | Apache-2.0 | |
| `Qwen/Qwen3-4B-GGUF` (opened) | GGUF | Q4_K_M 2.5 GB, Q5_0, Q5_K_M, Q6_K, Q8_0 4.28 GB | Apache-2.0 | "We follow the latest version of llama.cpp"; YaRN flags documented |
| `Qwen/Qwen3-8B-GGUF` (opened) | GGUF | Q4_K_M 5.03 GB, Q5_0, Q5_K_M, Q6_K, Q8_0 8.71 GB | Apache-2.0 | |
| `Qwen/Qwen3-Embedding-0.6B-GGUF` (opened) | GGUF | Q8_0 639 MB, F16 1.2 GB | Apache-2.0 | `llama-embedding ... --pooling last` |
| `Qwen/Qwen3-Embedding-4B-GGUF`, `Qwen/Qwen3-Embedding-8B-GGUF` | GGUF | — | — | in org listing; not opened |
| `Qwen/Qwen3-4B-AWQ` (opened) | AWQ 4-bit | — | Apache-2.0 | GPU |
| `Qwen/Qwen3-8B-AWQ` | AWQ | — | — | in org listing; not opened |
| `Qwen/Qwen3-0.6B-GPTQ-Int8` (opened) | GPTQ 8-bit | — | Apache-2.0 | GPU |
| `Qwen/Qwen3-1.7B-GPTQ-Int8` | GPTQ 8-bit | — | — | in org listing; not opened |
| `Qwen/Qwen3-0.6B-FP8`, `-1.7B-FP8`, `-4B-FP8`, `-8B-FP8` | FP8 | — | — | in org listing; not opened |
| `Qwen/Qwen3-4B-Instruct-2507-FP8` (opened) | FP8 block-128 | — | Apache-2.0 | GPU |
| `Qwen/Qwen3-4B-Thinking-2507-FP8` | FP8 | — | — | in org listing; not opened |
| `Qwen/Qwen3-{0.6B,1.7B,4B,8B}-MLX-{4bit,6bit,8bit,bf16}` | MLX | — | — | Apple Silicon only; in org listing |
| `Qwen/Qwen3-Next-80B-A3B-Instruct-GGUF` (opened) | GGUF | Q4_K_M 48.4 GB … Q8_0 | Apache-2.0 | proves official llama.cpp support for `qwen3next`; >9B |

**Important negative findings (HF API `author=Qwen&search=GGUF`, 54 repos):**
- There is **no official Qwen GGUF for any Qwen3.5, Qwen3.6 or Qwen3.8 model**, and none for `Qwen3-4B-Instruct-2507` / `Qwen3-4B-Thinking-2507`. The Qwen3.5 small-model cards only link to the community "Browse Quantizations" list.
- Qwen's own GPTQ-Int4 / FP8 for Qwen3.5 exist only for 27B, 35B-A3B, 122B-A10B, 397B-A17B; **nothing official for 0.8B–9B**.

### 3.2 Unsloth GGUF repos (all opened; all Apache-2.0; "UD-" = Unsloth Dynamic 2.0 imatrix quants)

| Repo | GGUF arch tag | Quant types available |
|---|---|---|
| `unsloth/Qwen3-0.6B-GGUF` | qwen3 | UD-IQ1_S, UD-IQ1_M, UD-IQ2_XXS, Q2_K, UD-IQ2_M, Q2_K_L, UD-Q2_K_XL, UD-IQ3_XXS, Q3_K_S, Q3_K_M, UD-Q3_K_XL, IQ4_XS, Q4_K_S, IQ4_NL, Q4_0, Q4_1, Q4_K_M, UD-Q4_K_XL, Q5_K_S, Q5_K_M, UD-Q5_K_XL, Q6_K, UD-Q6_K_XL, Q8_0, UD-Q8_K_XL, BF16 |
| `unsloth/Qwen3-1.7B-GGUF` | qwen3 | same 26 types as above |
| `unsloth/Qwen3-4B-GGUF` | qwen3 | same set (Q4_K_M 2.5 GB, UD-Q4_K_XL 2.55 GB, Q8_0 4.28 GB, BF16 8.05 GB) |
| `unsloth/Qwen3-8B-GGUF` | qwen3 | same set minus Q4_0 |
| `unsloth/Qwen3-4B-Instruct-2507-GGUF` | qwen3 | same set; 16-bit is F16 |
| `unsloth/Qwen3-4B-Thinking-2507-GGUF` | qwen3 | same set; F16 8.05 GB |
| `unsloth/Qwen3.5-0.8B-GGUF` | **qwen35** | UD-IQ2_XXS 338 MB, UD-IQ2_M, UD-Q2_K_XL, UD-IQ3_XXS, Q3_K_S, Q3_K_M, UD-Q3_K_XL, IQ4_XS, Q4_K_S, IQ4_NL, Q4_0, Q4_1, Q4_K_M 533 MB, UD-Q4_K_XL 559 MB, Q5_K_S, Q5_K_M, UD-Q5_K_XL, Q6_K, UD-Q6_K_XL, Q8_0 812 MB, UD-Q8_K_XL, BF16 1.52 GB |
| `unsloth/Qwen3.5-2B-GGUF` | qwen35 | same set (Q4_K_M 1.28 GB, UD-Q4_K_XL 1.34 GB, Q8_0 2.01 GB, BF16 3.78 GB) |
| `unsloth/Qwen3.5-4B-GGUF` | qwen35 | same set (Q4_K_M 2.74 GB, UD-Q4_K_XL 2.91 GB, Q8_0 4.48 GB, BF16 8.42 GB) |
| `unsloth/Qwen3.5-9B-GGUF` | qwen35 | same set (Q4_K_M 5.68 GB, UD-Q4_K_XL 5.97 GB, Q8_0 9.53 GB, BF16 17.9 GB) |
| `unsloth/Qwen3.6-35B-A3B-GGUF` (>9B, reference) | qwen35moe | UD-IQ1_M … UD-Q4_K_XL 22.4 GB, MXFP4_MOE, … Q8_0 36.9 GB, BF16 69.4 GB |
| `unsloth/Qwen3.6-27B-GGUF` (>9B, reference) | qwen35 | UD-IQ2_XXS … Q4_K_M 16.8 GB, UD-Q4_K_XL 17.6 GB, Q8_0 28.6 GB, BF16 53.8 GB |
| `unsloth/Qwen3.8-27B-GGUF` (>9B, reference) | qwen35 | UD-IQ1_S … UD-Q4_K_M 16.5 GB, UD-Q4_K_XL 17.6 GB … Q8_0 29 GB, BF16 54.7 GB; **plus a separate MTP Q4_0 tensor file (1.37 GB)** |

Unsloth Qwen3.5 guide notes (unsloth.ai/docs/models/qwen3.5): build llama.cpp from latest master (`-DGGML_CUDA=OFF` for CPU); vision needs the `mmproj-F16` file via `--mmproj`; the guide states that Qwen3.5 GGUFs did not run in Ollama because of the separate mmproj (status as of the doc's March 2026 update — re-check). Memory guidance (RAM+VRAM): 0.8B/2B ≈ 3.5 GB at 4-bit; 4B ≈ 5.5 GB; 9B ≈ 6.5 GB.

Other community GGUF uploaders seen in listings (not opened): bartowski (`bartowski/Qwen_Qwen3-4B-GGUF`), lmstudio-community, mradermacher.

---

## 4. llama.cpp support status per architecture

| Architecture (GGUF arch) | Models | llama.cpp status | Evidence |
|---|---|---|---|
| `qwen3` / `qwen3moe` (standard GQA) | Qwen3 0.6B–235B, 2507 refresh | **Fully supported; mature CPU path** (dense GEMM/GEMV kernels, flash-attn, KV cache quant, YaRN) | Official `Qwen/Qwen3-*-GGUF` cards with llama.cpp instructions |
| `qwen3next` (GDN + gated attn + MoE) | Qwen3-Next-80B-A3B, Qwen3-Coder-Next | Supported since **PR #16095 merged 2025-11-28** ("focused on CORRECTNESS ONLY"); CPU was initially slow (see §7); native GDN kernels landed Mar 2026 | https://github.com/ggml-org/llama.cpp/pull/16095 ; issue #19480 closed 2026-03-15 |
| **`qwen35` / `qwen35moe`** (Qwen3.5 hybrid GDN) | Qwen3.5 0.8B–397B, **Qwen3.6-27B / 35B-A3B, Qwen3.8-27B** (all share `qwen3_5`) | **Supported.** First PR #19435 merged 2026-02-08, reverted next day (#19453); replacement **PR #19468 "[MODEL] support qwen3.5 series" merged 2026-02-10** incl. vision (mmproj, Qwen3-VL encoder reused; DeepStack omitted). `ggml: add GATED_DELTA_NET op` (CPU/CUDA) PR #19504 merged 2026-03-07; chunked fused GDN path PR #20340; Metal/Vulkan/SYCL/WebGPU/Hexagon GDN kernels followed. | https://github.com/ggml-org/llama.cpp/pull/19468 ; GitHub API PR search |
| MTP (multi-token prediction) for qwen35 | Qwen3.5/3.6/3.8 | Early PRs #19937, #20533, #20700 (Feb–Mar 2026) were **closed unmerged**. However PR #21845 (merged 2026-06-05) "benchmarks MTP speculative decoding on Qwen3.6-27B" and `unsloth/Qwen3.8-27B-GGUF` ships a separate MTP Q4_0 file, which indicates MTP-based speculative decoding landed by mid-2026. **The merging PR was not identified — treat MTP support as UNVERIFIED and check `llama-server --help` for a `--spec-type mtp`-style option.** | GitHub API PR search (truncated at 100k chars) |
| `qwen4exp` (Qwen3.8-Flash-Next) | Qwen3.8-Flash-Next 180B | Added in **PR #27742** (release b10660, ~Sep 2026); "support is still being refined" (follow-up fixes for caching, state save/restore, quantisation) | newreleases.io b10660 notes |
| Diffusion (`llama-diffusion-cli`) | Dream, LLaDA, RND1 | Supported architectures: **Dream, LLaDA, RND1** only (PRs #14644, #14771) | `examples/diffusion/README.md` on master |

**What this means for CPU-only deployment today (2026-10-18):**
- **Qwen3 0.6B/1.7B/4B/8B (+2507)**: run in llama.cpp on CPU with the fully-optimised dense path. Lowest risk.
- **Qwen3.5 0.8B/2B/4B/9B**: run in llama.cpp on CPU (text-only without mmproj). The GDN linear-attention layers use the `GATED_DELTA_NET` op, which has had a native CPU kernel since March 2026; per-token recurrent state means long contexts do not grow a KV cache for 3/4 of the layers. Open issues still exist around the hybrid path (e.g. #28879 perplexity non-monotonic across precisions on `qwen3_5`, Sept 2026; #29473 Hexagon NPU correctness, Sept 2026) and no systematic x86 CPU benchmark Qwen3-4B vs Qwen3.5-4B on identical hardware was found. Expect the hybrid path to be somewhat slower per token than a same-size dense GQA model at short context and competitive-or-better at long context; verify with `llama-bench` on the target CPU.
- **Qwen3.6-27B / Qwen3.8-27B**: run on the same `qwen35` path (unsloth GGUFs exist), but at 27B they are outside this survey's CPU budget (~16 GB at Q4).
- **Non-AR / diffusion Qwen derivatives**: only Dream-family models run in llama.cpp (`llama-diffusion-cli`); SDAR/TraDo/Fast-dLLM/DiffuCoder do not.

---

## 5. Qwen-based non-autoregressive / diffusion / parallel-decoding models

**Official status: no non-autoregressive Qwen from Alibaba exists.** Evidence: HF API `author=Qwen&search=diffusion` returns an empty list; the Qwen org's only diffusion models are image generators (Qwen-Image, MMDiT); the Qwen3.5/3.6/3.8 blogs and cards describe only autoregressive decoding (the MTP head is a speculative-decoding aid, still AR). The newest "dQwen3.5" paper (Sept 2026) is academic, not from the Qwen team. Everything below is community/third-party.

| Model | HF repo(s) (status) | Base Qwen model | Paper | Size | Licence | llama.cpp | vLLM / other |
|---|---|---|---|---|---|---|---|
| **Dream 7B** (HKU NLP) | `Dream-org/Dream-v0-Instruct-7B` (opened), `Dream-org/Dream-v0-Base-7B` (listed in GitHub README) | Qwen2.5-7B ("initialized from Qwen2.5 weights", paper §; ~580B tokens) | arXiv:2508.15487 (2025-08-21) | 7B (HF tag 8B) | Apache-2.0 | **Yes** — Dream arch in `llama-diffusion-cli` (`--diffusion-algorithm 3`); community GGUFs: `bartowski/Dream-org_Dream-v0-Instruct-7B-GGUF`, `mradermacher/Dream-v0-Instruct-7B-GGUF`, `diffuse-cpp/Dream-v0-Instruct-7B-GGUF` (listed, not opened) | card shows `vllm serve` and SGLang; Transformers `trust_remote_code` |
| **Dream-Coder 7B** | `Dream-org/Dream-Coder-v0-Instruct-7B` (opened) | Qwen2.5-Coder-7B base (blog: "adapted from Qwen2.5-Coder 7B base using 322B training tokens") | arXiv:2509.01142 (2025-09-01) | 7B (tag 8B) | Apache-2.0 | Dream arch → should run; GGUFs `mradermacher/Dream-Coder-v0-Instruct-7B-GGUF`, `DevQuasar/...` listed | vLLM, SGLang per card |
| **DiffuCoder** (Apple + HKU) | `apple/DiffuCoder-7B-Base`, `apple/DiffuCoder-7B-Instruct` (HF API opened: `license: apple-amlr`, base `apple/DiffuCoder-7B-Base`), `apple/DiffuCoder-7B-cpGRPO` (GitHub README). `apple/DiffuCoder-7B-cpSFT` → **HTTP 401, UNVERIFIED** | Qwen2.5-Coder-7B ("We adapt our model from Qwen-2.5-Coder", paper §3) | arXiv:2506.20639 (2025-06-25) | 7B | **apple-amlr** (Apple ML research licence, not Apache) | No | Transformers `trust_remote_code` + custom `diffusion_generate()`; MLX in progress |
| **SDAR** (JetAstra / Shanghai AI Lab) | `JetLM/SDAR-1.7B-Chat` (opened), `SDAR-4B-Chat`, `SDAR-8B-Chat`, `SDAR-30B-A3B-Chat`, `SDAR-30B-A3B-Sci`, each also in `-b8/-b16/-b32/-b64` block-size variants (collection opened; 21 repos) | **Qwen3**-1.7B-Base / 4B-Base / 8B-Base / 30B-A3B-Base (GitHub README) | arXiv:2510.06303 (2025-10-07) | 1.7B, 4B, 8B, 30B-A3B | HF card: Apache-2.0; GitHub: MIT (discrepancy) | No (no SDAR arch; `mradermacher/SDAR-4B-Chat-GGUF` exists but block-diffusion decoding is not implemented in llama.cpp) | `generate.py` (transformers ≥4.52.4), **JetEngine** (nano-vllm based), LMDeploy; HF card also lists vLLM/SGLang |
| **TraDo** (Gen-Verse, Princeton) | `Gen-Verse/TraDo-8B-Instruct` (opened), `TraDo-4B-Instruct`, `TraDo-8B-Thinking` (API listed) | SDAR-4B/8B-Chat ("starting from the SDAR base model", paper) → Qwen3-4B/8B-Base | arXiv:2509.06949 (2025-09-08; ICLR 2026) | 4B, 8B | HF card: MIT; GitHub: Apache-2.0 (discrepancy) | No | Transformers `trust_remote_code` + `block_diffusion_generate`; jetengine |
| **Fast-dLLM v2** (NVIDIA) | `Efficient-Large-Model/Fast_dLLM_v2_1.5B` (opened), `Efficient-Large-Model/Fast_dLLM_v2_7B` (opened) | Qwen2.5-1.5B-Instruct / Qwen2.5-7B-Instruct | arXiv:2509.26328 (2025-09-30) | 1.54B (1.31B non-emb), 7B | Apache-2.0 | No | Transformers `trust_remote_code`, custom generate; block size 32 |
| **dllm-hub** small adapters | `dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1`, `dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1` (opened), `dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-{mdlm,bd3lm}-v0.1` | Qwen3-0.6B-Base; Qwen2.5-Coder-0.5B-Instruct | dLLM framework arXiv:2602.22661; BD3LM arXiv:2503.09573 | 0.6B / 0.5B | Apache-2.0 | No | Transformers `trust_remote_code`; card lists vLLM/SGLang |
| **dQwen3.5** (academic) | **No checkpoints found** (arXiv page has no HF/GitHub links) — UNVERIFIED as a usable model | Qwen3.5 0.8B/2B/4B/9B (hybrid backbone made bidirectional) | arXiv:2609.20751 (2026-09-17; Xue, Rout, Akella, Klivans, Sanghavi, Shakkottai) | — | — | No | — |
| **DFlash draft** (z-lab / Modal) | `z-lab/Qwen3.5-35B-A3B-DFlash` (opened) | target `Qwen/Qwen3.5-35B-A3B` | arXiv:2602.06036 | 0.4B | Apache-2.0 | No | SGLang `--speculative-algorithm DFLASH`; vLLM PR #40898 pending. **Not a standalone LM** — a block-diffusion draft for speculative decoding |
| NVIDIA NeMo AutoModel "Qwen3-8B I-DLM" | training recipe only; no checkpoint repo identified | Qwen3-8B | — | — | — | — | — |

Not Qwen-based but supported by `llama-diffusion-cli`: LLaDA-8B, RND1. `diffuse-cpp` is a separate CPU engine with its own Dream/LLaDA GGUFs.

---

## 6. Qwen3-Embedding / Reranker small variants and newer embedding generation

| Repo (opened) | Params | Base | Context | Output dim | Licence | Paper | CPU path |
|---|---|---|---|---|---|---|---|
| `Qwen/Qwen3-Embedding-0.6B` | 0.6B | Qwen3-0.6B-Base | 32K | 1024 (MRL: user-defined 32–1024) | Apache-2.0 | arXiv:2506.05176 (2025-06-05) | `Qwen/Qwen3-Embedding-0.6B-GGUF` (Q8_0, F16) via `llama-embedding --pooling last` or `llama-server --embedding --pooling last` |
| `Qwen/Qwen3-Reranker-0.6B` | 0.6B | Qwen3-0.6B-Base | 32K | — (yes/no logit scoring) | Apache-2.0 | arXiv:2506.05176 | No official GGUF; community quants only (not verified) |
| Larger siblings (listed) | `Qwen3-Embedding-4B` (2560-d), `-8B` (4096-d), `Qwen3-Reranker-4B`, `-8B`; GGUF for Embedding-4B/8B | | | | | | |
| **Newer generation:** `Qwen/Qwen3-VL-Embedding-2B` (opened) | 2B | Qwen3-VL-2B-Instruct | 32K | up to 2048 (64–2048) | Apache-2.0 | arXiv:2601.04720 (2026-01-08) | multimodal (text/image/video/screenshot); no official GGUF |
| `Qwen/Qwen3-VL-Embedding-8B`, `Qwen/Qwen3-VL-Reranker-2B`, `-8B` | listed in org | | | | | | |

No "Qwen3.5-Embedding" or newer text-only embedding exists: HF API `author=Qwen&search=Embedding` returns exactly 8 repos (Qwen3-Embedding-0.6B/4B/8B, their GGUFs, Qwen3-VL-Embedding-2B/8B).

---

## 7. Published CPU throughput numbers (tokens/s)

Only measurements with a stated source are listed. Figures marked *estimate* are not measurements.

| Source | CPU / RAM | Model & quant | Tool / build, threads | pp (t/s) | tg (t/s) |
|---|---|---|---|---|---|
| https://huggingface.co/b4ph/qwen3-4b-lowram-bench (opened) | "16-core x86-64 CPU (AVX-512 + VNNI)", 61.6 GB RAM, model not named; `-ngl 0` | **Qwen3-4B** Q4_K_M / Q5_K_M / Q6_K / Q8_0 / F16 | llama.cpp b10516 (`b95502ba9`), `-t 16`, pp512/tg128, 5 repeats | Q4_K_M 2064.8; Q8_0 1355.9; F16 357.6 | **Q4_K_M 6.50; Q5_K_M 5.64; Q6_K 5.93; Q8_0 5.04; F16 2.52** (Q3_K_M 6.56) |
| https://github.com/ggml-org/llama.cpp/pull/16095 (opened; user benchmarks in thread, Nov 2025) | AMD EPYC 9454P, 24 threads | Qwen3-Next-80B-A3B Q4_0 (qwen3next, pre-optimisation) | llama.cpp PR build | ~74 | ~11.8 (Qwen3-30B-A3B same setup ≈63 tg) |
| same PR thread | AMD Ryzen AI Max+ 395, pure CPU | Qwen3-Next-80B-A3B | | ~102 | ~17 |
| https://github.com/ggml-org/llama.cpp/issues/19480 (opened; 2026-02-10, closed 2026-03-15 after native GDN kernels) | AMD Ryzen AI 9 HX PRO 370 (12C/24T), 96 GB DDR5-5600 dual-channel | Qwen3-Coder-Next (80B-A3B) Q4_K_M (~51 GB) | Ollama 0.15.6 / llama.cpp build 7973, CPU-only | — | 7.74 (dense qwen2.5-coder:32b: 3.54 on same box) |
| https://github.com/ggml-org/llama.cpp/issues/29473 (opened; Sept 2026) — **ARM, not x86** | Snapdragon 7 Gen 4 (SM7750), 4 threads | **Qwen3.5-4B Q4_0** (qwen35) | llama.cpp 0.5.0-dev commit 86a24a1 | 46.3 (pp512) | 9.1 (tg64) ; Llama-3.2-1B Q4_0 on same device: 216.3 / 34.4 |
| https://itsfoss.com/testing-local-llms-without-gpu (opened; 2026-05-15) | older Intel Core i5 laptop, ~12 GB RAM, CPU-only | Qwen3 0.6B Q4_K_M (`qwen3:0.6b`) | Ollama `--verbose` | — | 34–36 |
| https://mail.openbenchmarking.org/performance/test/pts/llama-cpp/32347b8d... (opened; 75 public results Feb–Apr 2026) | various | **Qwen3-8B Q8_0, Prompt Processing 1024 only** | llama.cpp b8121, CPU BLAS | Ryzen 9 9950X3D 121; Ryzen 7 9850X3D 99; 2× Xeon 6980P 122; EPYC 9655P 118; Threadripper 7980X 79; TR PRO 5965WX 54; Xeon Silver 4216 34; Core Ultra X7 358H 27 | (not reported) |
| https://www.techplained.com/run-qwen-3-5-9b-64gb-ram (opened; 2026-04-09, upd. 2026-08-31) — **low confidence: tool, build, threads, method not stated** | Ryzen 9 7950X / 64 GB DDR5; i7-13700K / 64 GB DDR5; Ryzen 7 5800X / 64 GB DDR4 | Qwen3.5-9B Q4_K_M, 32K ctx | not stated | 280 / 240 / 180 | 24 / 20 / 15 |
| bestllmfor.com CPU guide (via search summary; *estimates*, ±20 %) | i5-12400 DDR4-3200; i7-13700K DDR5-5600; Ryzen 7 7700X DDR5-6000 | Qwen3.5-4B / 9B Q4_K_M, Ollama | — | — | ~15 / ~6 ; ~23 / ~12 ; ~24 / ~11 |
| https://hn.svelte.dev/item/47320244 (opened) | HP Z440 workstation (CPU not named) | "latest Qwen3.5 models", quant not stated | ik_llama.cpp + optimisation flags | — | "5.5× speedup", no absolute numbers |

**Not found:** a controlled x86 `llama-bench` comparison of Qwen3-4B vs Qwen3.5-4B (same CPU, same quant), or any x86 CPU measurement for Qwen3.5-0.8B/2B. A `llama-bench -t <physical cores> -p 512 -n 128` run on the target hardware is required before committing to numbers in the design document.

---

## 8. BibTeX

```bibtex
@article{qwen3_2025,
  title   = {Qwen3 Technical Report},
  author  = {An Yang and Anfeng Li and Baosong Yang and others},
  journal = {arXiv preprint arXiv:2505.09388},
  year    = {2025},
  url     = {https://arxiv.org/abs/2505.09388}
}

@article{qwen3_embedding_2025,
  title   = {Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models},
  author  = {Yanzhao Zhang and Mingxin Li and Dingkun Long and others},
  journal = {arXiv preprint arXiv:2506.05176},
  year    = {2025},
  url     = {https://arxiv.org/abs/2506.05176}
}

@article{qwen3_vl_embedding_2026,
  title   = {Qwen3-VL-Embedding and Qwen3-VL-Reranker: A Unified Framework for State-of-the-Art Multimodal Retrieval and Ranking},
  author  = {Mingxin Li and others},
  journal = {arXiv preprint arXiv:2601.04720},
  year    = {2026},
  url     = {https://arxiv.org/abs/2601.04720}
}

@misc{qwen3_5_blog_2026,
  title        = {Qwen3.5: Towards Native Multimodal Agents},
  author       = {{Qwen Team}},
  howpublished = {\url{https://qwen.ai/blog?id=qwen3.5}},
  month        = feb,
  year         = {2026},
  note         = {No standalone technical report as of 2026-10-18; model cards cite this post}
}

@article{qwen3_5_omni_2026,
  title   = {Qwen3.5-Omni Technical Report},
  author  = {{Qwen Team}},
  journal = {arXiv preprint arXiv:2604.15804},
  year    = {2026},
  url     = {https://arxiv.org/abs/2604.15804}
}

@misc{qwen3_6_35b_blog_2026,
  title        = {Qwen3.6-35B-A3B: Agentic Coding Power, Now Open to All},
  author       = {{Qwen Team}},
  howpublished = {\url{https://qwen.ai/blog?id=qwen3.6-35b-a3b}},
  month        = apr,
  year         = {2026}
}

@misc{qwen3_6_27b_blog_2026,
  title        = {Qwen3.6-27B: Flagship-Level Coding in a 27B Dense Model},
  author       = {{Qwen Team}},
  howpublished = {\url{https://qwen.ai/blog?id=qwen3.6-27b}},
  month        = apr,
  year         = {2026}
}

@misc{qwen3_8_blog_2026,
  title        = {Qwen3.8-Max: A New Bar for Coding and Cowork},
  author       = {{Qwen Team}},
  howpublished = {\url{https://qwen.ai/blog?id=qwen3.8}},
  month        = aug,
  year         = {2026},
  note         = {Cited by the Qwen3.8-27B model card; no technical report}
}

@article{dream7b_2025,
  title   = {Dream 7B: Diffusion Large Language Models},
  author  = {Jiacheng Ye and others},
  journal = {arXiv preprint arXiv:2508.15487},
  year    = {2025},
  url     = {https://arxiv.org/abs/2508.15487}
}

@article{dreamcoder_2025,
  title   = {Dream-Coder 7B: An Open Diffusion Language Model for Code},
  author  = {Zhihui Xie and others},
  journal = {arXiv preprint arXiv:2509.01142},
  year    = {2025},
  url     = {https://arxiv.org/abs/2509.01142}
}

@article{diffucoder_2025,
  title   = {DiffuCoder: Understanding and Improving Masked Diffusion Models for Code Generation},
  author  = {Shansan Gong and Ruixiang Zhang and Huangjie Zheng and Jiatao Gu and Navdeep Jaitly and Lingpeng Kong and Yizhe Zhang},
  journal = {arXiv preprint arXiv:2506.20639},
  year    = {2025},
  url     = {https://arxiv.org/abs/2506.20639}
}

@article{sdar_2025,
  title   = {SDAR: A Synergistic Diffusion-AutoRegression Paradigm for Scalable Sequence Generation},
  author  = {Shuang Cheng and others},
  journal = {arXiv preprint arXiv:2510.06303},
  year    = {2025},
  url     = {https://arxiv.org/abs/2510.06303}
}

@article{trado_2025,
  title   = {Revolutionizing Reinforcement Learning Framework for Diffusion Large Language Models},
  author  = {Yinjie Wang and Ling Yang and Bowen Li and Ye Tian and Ke Shen and Mengdi Wang},
  journal = {arXiv preprint arXiv:2509.06949},
  year    = {2025},
  note    = {ICLR 2026},
  url     = {https://arxiv.org/abs/2509.06949}
}

@article{fastdllm_v2_2025,
  title   = {Fast-dLLM v2: Efficient Block-Diffusion LLM},
  author  = {Chengyue Wu and others},
  journal = {arXiv preprint arXiv:2509.26328},
  year    = {2025},
  url     = {https://arxiv.org/abs/2509.26328}
}

@article{dqwen3_5_2026,
  title   = {dQwen3.5: Hybrid-Attention Diffusion Language Models},
  author  = {Anton Xue and Litu Rout and Aditya Akella and Adam Klivans and Sujay Sanghavi and Sanjay Shakkottai},
  journal = {arXiv preprint arXiv:2609.20751},
  year    = {2026},
  url     = {https://arxiv.org/abs/2609.20751}
}
```

---

## 9. Verification log (every URL opened, in order)

Status: OK = loaded and used; PROXY = loaded via `r.jina.ai` text proxy; PARTIAL = loaded but truncated (first 100k chars); FAIL = not loaded.

| URL | Status | Used for |
|---|---|---|
| https://huggingface.co/Qwen | OK | org front page (only Qwen3.8 collections visible) |
| https://github.com/QwenLM/Qwen3.8 | OK | release timeline 3.5/3.6/3.8, collection links |
| https://qwenlm.github.io/blog/ | OK | index stops at Sept 2025 |
| https://huggingface.co/collections/Qwen/qwen35 | OK | 21 Qwen3.5 repos |
| https://huggingface.co/collections/Qwen/qwen36 | OK | 4 repos |
| https://huggingface.co/collections/Qwen/qwen38 | OK | 4 repos |
| https://huggingface.co/Qwen?search_models=Qwen3.5 / 3.6 / 3.7 / 3.8 | OK (search param ignored by page) | superseded by API queries |
| https://huggingface.co/api/models?author=Qwen&search=Qwen3.6&limit=100 | OK | 4 repos |
| https://huggingface.co/api/models?author=Qwen&search=Qwen3.8&limit=100 | OK | 6 repos |
| https://huggingface.co/api/models?author=Qwen&search=Qwen3.5&limit=100 | OK | 29 repos |
| https://huggingface.co/api/models?author=Qwen&search=Qwen3.7&limit=100 | OK | EMPTY |
| https://huggingface.co/api/models?author=Qwen&search=GGUF&limit=200 | OK | 54 official GGUF repos |
| https://huggingface.co/api/models?author=Qwen&search=Qwen3-&limit=200 | OK (capped at 200) | AWQ/GPTQ/FP8/MLX/Base listings |
| https://huggingface.co/api/models?author=Qwen&search=Embedding&limit=100 | OK | 8 repos |
| https://huggingface.co/api/models?author=Qwen&search=diffusion&limit=50 | OK | EMPTY |
| https://qwen.ai/blog?id=qwen3.5 / qwen3.8 / qwen3.6-35b-a3b / qwen3.6-27b (direct) | FAIL (JS shell, body = "Qwen") | — |
| https://r.jina.ai/https://qwen.ai/blog?id=qwen3.5 | PROXY, PARTIAL | Qwen3.5 blog |
| https://r.jina.ai/https://qwen.ai/blog?id=qwen3.8 | PROXY | Qwen3.8 blog |
| https://r.jina.ai/https://qwen.ai/blog?id=qwen3.6-35b-a3b | PROXY | Qwen3.6-35B-A3B blog |
| https://r.jina.ai/https://qwen.ai/blog?id=qwen3.6-27b | PROXY | Qwen3.6-27B blog |
| https://r.jina.ai/https://qwen.ai/blog?id=qwen3.7 (two encodings) | FAIL (proxy returned an unrelated PDF) | Qwen3.7 blog not verified |
| https://www.deeplearning.ai/the-batch/qwen3-7-max-adds-speed-and-power | OK (secondary) | Qwen3.7 closed-weights statement |
| https://www.kucoin.com/news/flash/alibaba-launches-qwen3-7-preview-... | OK (secondary) | Qwen3.7 preview date |
| https://arxiv.org/abs/2505.09388 | OK | Qwen3 tech report |
| https://arxiv.org/abs/2506.05176 | OK | Qwen3 Embedding report |
| https://arxiv.org/abs/2601.04720 | OK | Qwen3-VL-Embedding report |
| https://arxiv.org/abs/2604.15804 | OK | Qwen3.5-Omni report |
| https://huggingface.co/Qwen/Qwen3.5-0.8B | OK | model row |
| https://huggingface.co/Qwen/Qwen3.5-2B | OK | model row |
| https://huggingface.co/Qwen/Qwen3.5-4B | OK | model row |
| https://huggingface.co/Qwen/Qwen3.5-9B | OK | model row |
| https://huggingface.co/Qwen/Qwen3.5-0.8B/raw/main/config.json | OK | architecture fields |
| https://huggingface.co/Qwen/Qwen3.6-35B-A3B | OK | reference row |
| https://huggingface.co/Qwen/Qwen3.6-27B/raw/main/config.json | OK | architecture |
| https://huggingface.co/Qwen/Qwen3.8-27B | OK | reference row |
| https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json | OK | architecture |
| https://huggingface.co/Qwen/Qwen3-0.6B | OK | model row |
| https://huggingface.co/Qwen/Qwen3-1.7B | OK | model row |
| https://huggingface.co/Qwen/Qwen3-4B | OK | model row |
| https://huggingface.co/Qwen/Qwen3-4B/raw/main/config.json | OK | `Qwen3ForCausalLM` |
| https://huggingface.co/Qwen/Qwen3-8B | OK | model row |
| https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507 | OK | model row |
| https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507 | OK | model row |
| https://huggingface.co/Qwen/Qwen3-Embedding-0.6B | OK | §6 |
| https://huggingface.co/Qwen/Qwen3-Reranker-0.6B | OK | §6 |
| https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B | OK | §6 |
| https://huggingface.co/Qwen/Qwen3-4B-GGUF | OK | §3 |
| https://huggingface.co/Qwen/Qwen3-0.6B-GGUF | OK | §3 |
| https://huggingface.co/Qwen/Qwen3-1.7B-GGUF | OK | §3 |
| https://huggingface.co/Qwen/Qwen3-8B-GGUF | OK | §3 |
| https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF | OK | §3/§6 |
| https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct-GGUF | OK | §3/§4 |
| https://huggingface.co/Qwen/Qwen3-4B-AWQ | OK | §3 |
| https://huggingface.co/Qwen/Qwen3-0.6B-GPTQ-Int8 | OK | §3 |
| https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507-FP8 | OK | §3 |
| https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF | OK | §3 |
| https://huggingface.co/unsloth/Qwen3.5-2B-GGUF | OK | §3 |
| https://huggingface.co/unsloth/Qwen3.5-4B-GGUF | OK | §3 |
| https://huggingface.co/unsloth/Qwen3.5-9B-GGUF | OK | §3 |
| https://huggingface.co/unsloth/Qwen3-0.6B-GGUF | OK | §3 |
| https://huggingface.co/unsloth/Qwen3-1.7B-GGUF | OK | §3 |
| https://huggingface.co/unsloth/Qwen3-4B-GGUF | OK | §3 |
| https://huggingface.co/unsloth/Qwen3-8B-GGUF | OK | §3 |
| https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF | OK | §3 |
| https://huggingface.co/unsloth/Qwen3-4B-Thinking-2507-GGUF | OK | §3 |
| https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF | OK | §3 (reference) |
| https://huggingface.co/unsloth/Qwen3.6-27B-GGUF | OK | §3 (reference) |
| https://huggingface.co/unsloth/Qwen3.8-27B-GGUF | OK | §3 (reference; MTP file) |
| https://docs.unsloth.ai/models/qwen3.5 → https://unsloth.ai/docs/models/qwen3.5 | redirect, then OK | §3 notes |
| https://api.github.com/search/issues?q=repo:ggml-org/llama.cpp+Qwen3.5+is:pr+is:merged... | PARTIAL | PRs #19435, #19453, #19468 |
| https://api.github.com/search/issues?q=repo:ggml-org/llama.cpp+qwen3next+is:pr+is:merged... | PARTIAL | PR #16095 etc. |
| https://api.github.com/search/issues?q=repo:ggml-org/llama.cpp+"gated delta"+CPU+is:issue... | PARTIAL | issues #29473, #28879, #28019, #27998 |
| https://api.github.com/search/issues?q=repo:ggml-org/llama.cpp+Qwen3.6+is:pr... | PARTIAL | no dedicated Qwen3.6 PR (same arch) |
| https://api.github.com/search/issues?q=repo:ggml-org/llama.cpp+Qwen3.8+is:pr... | PARTIAL | no dedicated Qwen3.8-27B PR |
| https://api.github.com/search/issues?q=repo:ggml-org/llama.cpp+GATED_DELTA_NET+cpu+is:pr... | PARTIAL | PR #19504 |
| https://api.github.com/search/issues?q=repo:ggml-org/llama.cpp+MTP+qwen3.5+is:pr... | PARTIAL | PRs #19937, #20533, #20700 closed; #21845 merged |
| https://api.github.com/search/issues?q=repo:ggml-org/llama.cpp+Qwen3-Coder-Next+CPU+... (two queries) | PARTIAL | issue list |
| https://github.com/ggml-org/llama.cpp/pull/19468 | OK | Qwen3.5 support PR |
| https://github.com/ggml-org/llama.cpp/pull/16095 | PARTIAL | Qwen3-Next PR + CPU numbers |
| https://github.com/ggml-org/llama.cpp/issues/19480 | OK | CPU slowness issue |
| https://github.com/ggml-org/llama.cpp/issues/29473 | OK | Qwen3.5-4B CPU baseline (ARM) |
| https://raw.githubusercontent.com/ggml-org/llama.cpp/master/examples/diffusion/README.md | OK | diffusion CLI archs |
| https://newreleases.io/project/github/ggml-org/llama.cpp/release/b10660 | OK (mirror of GitHub release notes) | qwen4exp |
| https://buttondown.com/weekly-project-news/archive/weekly-github-report-for-llamacpp-february-08-6955/ | PARTIAL (secondary) | pointer to issue #19480 |
| https://huggingface.co/Dream-org/Dream-v0-Instruct-7B | OK | §5 |
| https://huggingface.co/Dream-org/Dream-Coder-v0-Instruct-7B | OK | §5 |
| https://github.com/HKUNLP/Dream | OK | §5 |
| https://hkunlp.github.io/blog/2025/dream-coder/ | OK | §5 base model |
| https://arxiv.org/abs/2508.15487 ; https://arxiv.org/html/2508.15487 | OK | Dream paper (Qwen2.5 init) |
| https://arxiv.org/abs/2509.01142 | OK | Dream-Coder paper |
| https://huggingface.co/apple/DiffuCoder-7B-cpSFT | FAIL (HTTP 401) | UNVERIFIED |
| https://huggingface.co/api/models/apple/DiffuCoder-7B-cpSFT | FAIL (HTTP 401) | UNVERIFIED |
| https://huggingface.co/api/models/apple/DiffuCoder-7B-Instruct | OK | licence apple-amlr |
| https://github.com/apple/ml-diffucoder | OK | checkpoints list |
| https://arxiv.org/abs/2506.20639 ; https://arxiv.org/html/2506.20639 | OK / PARTIAL | DiffuCoder paper |
| https://huggingface.co/JetLM/SDAR-1.7B-Chat | OK | §5 |
| https://huggingface.co/collections/JetLM/sdar-689b1b6d392a4eeb2664f8ff | OK | 21 SDAR repos |
| https://github.com/JetAstra/SDAR | OK | base models |
| https://arxiv.org/abs/2510.06303 | OK | SDAR paper |
| https://huggingface.co/Gen-Verse/TraDo-8B-Instruct | OK | §5 |
| https://huggingface.co/api/models?author=Gen-Verse&search=TraDo | OK | 3 repos |
| https://github.com/Gen-Verse/dLLM-RL | OK | §5 |
| https://arxiv.org/abs/2509.06949 ; https://arxiv.org/html/2509.06949 | OK / PARTIAL | TraDo paper |
| https://huggingface.co/Efficient-Large-Model/Fast_dLLM_v2_7B | OK | §5 |
| https://huggingface.co/Efficient-Large-Model/Fast_dLLM_v2_1.5B | OK | §5 |
| https://huggingface.co/api/models?author=Efficient-Large-Model&search=Fast_dLLM | OK | 2 repos |
| https://arxiv.org/abs/2509.26328 | OK | Fast-dLLM v2 paper |
| https://huggingface.co/dllm-hub | OK | 6 repos |
| https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1 | OK | §5 |
| https://arxiv.org/abs/2609.20751 | OK | dQwen3.5 paper |
| https://huggingface.co/z-lab/Qwen3.5-35B-A3B-DFlash | OK | §5 |
| https://docs.nvidia.com/nemo/automodel/nightly/model-coverage/dllm/qwen/qwen3-idlm.md | OK (page is an index; no checkpoint) | §5 |
| https://huggingface.co/api/models?search=Dream-v0-Instruct-7B GGUF | OK | 17 GGUF repos |
| https://huggingface.co/api/models?search=SDAR GGUF | OK | 2 repos |
| https://huggingface.co/b4ph/qwen3-4b-lowram-bench | OK | §7 |
| https://hn.svelte.dev/item/47320244 | OK | §7 |
| https://itsfoss.com/testing-local-llms-without-gpu | OK | §7 |
| https://www.techplained.com/run-qwen-3-5-9b-64gb-ram | OK (low confidence) | §7 |
| https://mail.openbenchmarking.org/performance/test/pts/llama-cpp/32347b8d61630305a0be85e4a3e9a06b256f9857 | OK | §7 |
| https://openbenchmarking.org/test/pts/llama-cpp | FAIL (HTTP 403) | — |
| https://www.mgroeber.de/benchmarking-llamacpp-on-legacy-hardware-2.html | OK (no Qwen data) | — |

Web searches run (WebSearch tool, 2026-10-18): Qwen3.5 release blog; Qwen3.6 release; Qwen3.8 release; Qwen3.7 release; llama.cpp Qwen3.5 support PR; llama.cpp Qwen3-Next PR; llama.cpp Qwen3.5 CPU slow; Qwen official diffusion LLM; llama-diffusion-cli supported models; Qwen3-4B GGUF CPU tokens/s; Qwen3.5 technical report; Qwen3.5-9B/4B CPU t/s; Qwen3-Next CPU slow issue; Qwen3/3.5 diffusion models 2026; Qwen3.5-4B llama-bench Ryzen; Qwen3-4B-Instruct-2507 tg128 CPU; Dream-Coder arXiv.
