# Stack feasibility: CPU-only serving + GPU fine-tuning (verified 2026-10-18)

Scope: vLLM, Unsloth, llama.cpp, Ollama, CPU throughput, Rust crate ecosystem, eval datasets/harnesses.
Method: every claim below cites the URL (or local file) it was read from. Items that could not be opened or confirmed are marked **UNVERIFIED**. Dates are only stated where the source showed them (GitHub API `published_at`, crates.io `created_at`, PyPI `upload_time`, or local `git log`). Target hardware for the estimates in section 5: Intel i3-8100 (4C/4T, AVX2, no AVX-512, DDR4-2400 dual channel).

Local primary sources used (read-only): `vendors/vllm` (git HEAD `9c778e9ff8`, committed 2026-10-17 18:18 -0500; tag `v0.32.0` present = commit `5df63a7203`, 2026-10-17 11:57 -0700, **not** an ancestor of HEAD) and `vendors/unsloth` (`unsloth/device_type.py`).

---

## 0. Headline findings

| # | Finding | Confidence |
|---|---------|-----------|
| 1 | **Small (≤9B) models only exist in the Qwen3.5 generation.** The `Qwen` HF org has Qwen3.5-0.8B/2B/4B/9B (+27B, 35B-A3B, 122B-A10B, 397B-A17B); Qwen3.6 has only 27B and 35B-A3B; Qwen3.8 has only 27B, 2.4T-A95B and Flash-Next. For a CPU-only box the candidate set is Qwen3.5-0.8B/2B/4B/9B (or older Qwen3-0.6B/1.7B/4B/8B dense). | Verified (HF API) |
| 2 | **Qwen3.5/3.6/3.8-27B all share `model_type: qwen3_5`, `architectures: ["Qwen3_5ForConditionalGeneration"]`, hybrid `linear_attention`×3 : `full_attention`×1 layer layout.** So "Qwen3.5 support" in a runtime covers Qwen3.6-27B and Qwen3.8-27B too. Qwen3.8-Flash-Next is a different arch (`qwen4_exp`). | Verified (HF config.json) |
| 3 | **vLLM v0.32.0 is a git tag (2026-10-17) but not yet a published GitHub Release nor a PyPI release.** GitHub "Latest" and PyPI latest are both v0.31.0 (2026-10-05). | Verified |
| 4 | **vLLM CPU backend does run hybrid GDN (Qwen3.5-family) models, but the accelerated kernels require AMX or AVX-512BF16.** On an AVX2-only CPU the code takes the generic torch fallback path. Qwen3.5 is *not* on the CPU "Validated models" page. AVX2 is officially "Limited features". | Verified (source + docs) |
| 5 | **Speculative decoding on the vLLM CPU backend exists in code (EAGLE, draft model, DFlash-for-GDN, dynamic SD) but is undocumented on the CPU page.** n-gram on CPU: no explicit evidence either way. | Partially verified |
| 6 | **Unsloth Core training requires a GPU** (`device_type.py` raises "You need a GPU" unless `UNSLOTH_ALLOW_CPU=1`, which is commented "Test-only CPU fallback"). The README's "CPUs" support refers to Unsloth Studio chat (GGUF inference via llama.cpp) and Data Recipes, per the requirements page. | Verified |
| 7 | **llama.cpp supports the whole Qwen3.5 family (PR #19468, merged 2026-02-10) with a native CPU `GATED_DELTA_NET` op (PR #19504, 2026-03-07; chunked fused path PR #20340, 2026-03-11).** Qwen3.8-Flash-Next (`qwen4exp`) merged 2026-08-27 (PR #27742). CPU speed for hybrid models is below dense models of the same active size (see §3, §5). | Verified |
| 8 | **Ollama v0.40.6 (2026-10-16) serves `qwen3.5:0.8b/2b/4b/9b`; `qwen3.6` and `qwen3.8` library entries start at 27B.** Ollama's OpenAI-compatible API does **not** support `logprobs`; use the native `/api/chat` or llama-server/vLLM for an uncertainty router. | Verified |
| 9 | **Bandwidth-bound decode ceiling on i3-8100 (38.4 GB/s theoretical, ~22 GB/s observed-effective on similar Skylake/Kaby-Lake parts): Qwen3.5-4B Q4_K_M ≈ 9–16 tok/s; 9B Q4_K_M ≈ 4–7 tok/s; 0.8B Q8_0 ≈ 25–45 tok/s (likely compute-bound lower).** Hybrid-GDN CPU kernels add overhead beyond this; the only AVX2-class Qwen3.5-4B data points are 2 tok/s (Core 2 Duo, DDR2) and ~11 tok/s (Ryzen 5 7540U, DDR5, self-reported). | Estimate |
| 10 | Rust toolchain: **1.99.1 (2026-10-16)** is current; local 1.96.0 is three minors behind. All requested crates resolved on crates.io (table in §6). | Verified |

---

## 1. vLLM

### 1.1 Release status of v0.32.0

| Item | Value | Source |
|------|-------|--------|
| Tag `v0.32.0` | commit `5df63a7203cdb9328d824a27b682e77c7b9928be`, tagger date 2026-10-17 11:57 -0700 (GitHub tag page shows "17 Oct 18:57", no year) | local `git log -1 v0.32.0`; https://github.com/vllm-project/vllm/releases/tag/v0.32.0 |
| Pre-release tags | `v0.32.0rc0`…`rc3` exist | GitHub tags API `https://api.github.com/repos/vllm-project/vllm/tags` |
| GitHub Release object | **None** for v0.32.0 (API `releases/tags/v0.32.0` → 404); releases list shows **v0.31.0 "Latest", 2026-10-05T06:44:55Z**; v0.30.0 2026-09-22; v0.29.0 2026-09-09; v0.28.0 2026-08-26 | https://api.github.com/repos/vllm-project/vllm/releases ; https://github.com/vllm-project/vllm/releases |
| PyPI | latest `vllm` **0.31.0** (2026-10-05); 0.32.0 not on PyPI | https://pypi.org/pypi/vllm/json |
| Release notes content | Tag page shows only one truncated item ("[Bugfix][Rust Frontend] Fix macOS build…"); no CPU/Qwen/spec-decode highlights visible | https://github.com/vllm-project/vllm/releases/tag/v0.32.0 |

Conclusion: treat v0.32.0 as "tagged, release pending" as of 2026-10-18. Pin to v0.31.0 wheels/images for reproducibility unless building from the tag.

### 1.2 CPU backend facts (docs, latest)

Source: https://docs.vllm.ai/en/latest/getting_started/installation/cpu.html (and local `docs/getting_started/installation/cpu.x86.inc.md`).

- x86: Linux; CPU flags "`avx512f` (Recommended), `avx2` (Limited features)"; dtypes FP32/FP16/BF16; FAQ says torch-CPU float16 is unstable, prefer BF16.
- "Pre-built vLLM wheels for x86 with AVX512/AVX2 are available since version 0.17.0." Nightly CPU wheels at `https://wheels.vllm.ai/nightly/cpu`.
- Docker: `vllm/vllm-openai-cpu` (`latest-x86_64`, `latest-arm64`); AMD Zen target `vllm-openai-zen`.
- Quantisation: AWQ, GPTQ (x86/s390x/Power); compressed-tensors INT8 W8A8 (x86/s390x).
- Env: `VLLM_CPU_KVCACHE_SPACE` (GiB, default 0, described as legacy — prefer `--kv-cache-memory-bytes`), `VLLM_CPU_OMP_THREADS_BIND` (`auto` default | core list | `nobind`), `VLLM_CPU_NUM_OF_RESERVED_CPU`, `CPU_VISIBLE_MEMORY_NODES`, `VLLM_ZENTORCH_WEIGHT_PREPACK`.
- Apple silicon: "users must build from source"; FP32/FP16 only; no pre-built wheels. (Not relevant to an Intel Mac — vLLM CPU is Linux-only on x86.)
- Tuning tip: "Use multiples of 32 as `--block-size`, which is 128 by default."
- The CPU page does **not** mention speculative decoding, hybrid/Mamba/GDN, or Qwen3.5.

### 1.3 Does the CPU backend support Qwen3.5/3.6/3.8 hybrid (Gated DeltaNet) models?

**Docs:** The CPU validated-models page lists only `GptOssForCausalLM, LlamaForCausalLM, GraniteForCausalLM, Qwen3ForCausalLM, Qwen3MoeForCausalLM, Qwen2ForCausalLM, GLMForCausalLM, GemmaForCausalLM, Phi3ForCausalLM, MistralForCausalLM` (+ multimodal incl. Gemma3/Gemma4/Qwen2-VL/Qwen3-VL-MoE). `Qwen3_5ForConditionalGeneration` and `Qwen3NextForCausalLM` are **absent**; validated hardware is Xeon 5/6. Legend: "✅ Runs and optimized." Source: https://docs.vllm.ai/en/latest/models/hardware_supported_models/cpu/ and `https://raw.githubusercontent.com/vllm-project/vllm/v0.32.0/docs/models/hardware_supported_models/cpu.md`.

**Supported-models page (all hardware):** `Qwen3_5ForConditionalGeneration` ("Qwen3.5", e.g. `Qwen/Qwen3.5-9B-Instruct`), `Qwen3_5MoeForConditionalGeneration`, `Qwen3NextForCausalLM` are listed; `Qwen3.6`/`Qwen3.8` strings do not appear (they resolve to the same architecture class, see §0). Tip: `--language-model-only` runs hybrid-only models text-only. Source: https://docs.vllm.ai/en/latest/models/supported_models.html

**Source code (local `vendors/vllm`, HEAD 2026-10-17):**
- `vllm/platforms/cpu.py`: `_cpu_mamba_backend()` detects `layer_types` containing `"linear_attention"` and returns `"gdn"`; comment "Accelerated GDN uses AMX tiles or AVX-512BF16 VDPBF16PS." Guard: `if torch.cpu._is_avx512_bf16_supported() and cache_config.mamba_ssm_cache_dtype != "float32": …` else SSM cache reset to float32. `support_hybrid_kv_cache()` returns `True`; `_align_hybrid_block_size` reconciles attention/mamba page sizes.
- `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`: `elif current_platform.is_cpu(): register_cpu_gdn_attention_ops(); self._forward_method = self.forward_cpu`.
- `vllm/model_executor/layers/mamba/ops/cpu/gdn_attention.py`: native C++ conv path only when `torch.cpu._is_amx_tile_supported() or is_arm_bf16()` (`can_use_native_conv`), otherwise `_unpacked_conv_weight(layer)` torch path; comment: "C++ conv … on x86 uses VDPBF16PS … so it runs on any AVX-512BF16 CPU."
- Merged PR trail (GitHub search API, repo vllm-project/vllm): #42707 "[CPU] Add fused GDN support for AMX CPU platform" (closed 2026-05-15); #43534 "[CPU][Perf] Enable fused kernels for GDN's gated delta rules" (2026-06-02); #46090 "[CPU][Spec Decode] Support DFlash speculative decoding for GDN models on CPU" (2026-06-18); #48073 "[CPU] Fix Qwen-Next SSM type for AMX GDN" (2026-07-09); #48577 "[CPU][Spec Decode] Optimize GDN conv path for speculative decoding" (2026-07-14); #49219 "[CPU][Perf] Fuse GDN decode core" (2026-07-20); #49640/#49688 "GDN attention falls back to slow torch conv1d on non-AMX AVX-512BF16 CPUs" (fixed 2026-07-24); #52191 "[CPU] Support FP16/BF16 persisted GDN state on AMX" (2026-08-13). Still open: #60051 "[CPU] Support FP16 in the GDN attention kernels" (2026-10-05), #60405 "[CPU][GDN] Route FP16 attention through Triton/FLA" (2026-10-07), #48212 CPU-tuned FLA Triton autotune configs.

**Verdict:** Qwen3.5-family on vLLM-CPU is *implemented* and actively optimised **for AMX / AVX-512BF16 Xeons**. On an **AVX2-only i3-8100** it should load and run via the unaccelerated torch path, but there is no documentation, no validation entry, and no public benchmark; expect it to be slow relative to llama.cpp. **No GitHub issue was found that states "hybrid/GDN unsupported on CPU"** (searches: https://github.com/vllm-project/vllm/issues?q=is%3Aissue+Qwen3.5+CPU and the API query `CPU GDN in:title`). Qwen3-dense (`Qwen3ForCausalLM`, 1.7B/4B/8B/14B) *is* on the CPU validated list with Xeon 6 recipes.

### 1.4 Speculative decoding on the CPU backend

- Docs page https://docs.vllm.ai/en/latest/features/speculative_decoding/ lists N-gram (`prompt_lookup_min/max`, default 5), EAGLE/EAGLE3, draft model (`use_heterogeneous_vocab`), MTP, suffix decoding, PARD, DFlash, DSpark, dynamic SD, adaptive verification; configured via `--speculative-config '{...}'`. **The page says nothing about hardware/backends** (only "PP not composable as of vllm<=0.15.0", "draft model not supported in vllm<=0.10.0").
- Local source: `vllm/v1/worker/cpu_model_runner.py` line 32 comment "speculative decoding is now supported on CPU with C++ native impls"; it logs "EAGLE drafter KV cache initialized for CPU backend" / "Draft model KV cache initialized for CPU backend". `cpu_worker.py` warns: "Speculative decoding on CPU without Intel OpenMP in LD_PRELOAD will cause significant performance loss."
- PRs: #46090 (DFlash for GDN on CPU, 2026-06-18), #47012 "[CPU][Spec Decode] Fix Dynamic Speculative Decoding on CPU backend" (closed 2026-06-29; issue #47014 still open).
- **n-gram on CPU:** no explicit statement found; the method is drafter-only/model-free so there is no CPU-specific kernel to gate it — **UNVERIFIED on CPU**, test empirically.

### 1.5 Reasoning parser and thinking toggle for newest Qwen

- Registered parsers (local `vllm/reasoning/__init__.py`): the only Qwen entry is `name="qwen3"` → `qwen3_engine_reasoning_parser`. There is **no** `qwen3_5`/`qwen3_8` parser. Docs page https://docs.vllm.ai/en/latest/features/reasoning_outputs.html lists `qwen3` for "Qwen3 series" (json/regex structured output + tool calling) and does not mention Qwen3.5/3.6/3.8.
- HF card `Qwen/Qwen3.5-4B` (https://huggingface.co/Qwen/Qwen3.5-4B): `vllm serve Qwen/Qwen3.5-4B --port 8000 --tensor-parallel-size 1 --max-model-len 262144 --reasoning-parser qwen3`; "vLLM from the main branch … is required"; thinking on by default; disable with `"chat_template_kwargs": {"enable_thinking": False}`; layout "8 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))".
- HF card `Qwen/Qwen3.8-27B` (https://huggingface.co/Qwen/Qwen3.8-27B): `vllm serve "Qwen/Qwen3.8-27B"` (no parser flag shown); `chat_template_kwargs`: `enable_thinking` (bool), `reasoning_effort` ∈ {`xhigh` (default), `medium`, `low`}, `preserve_thinking` (default on). Unsloth's Qwen3.8 page additionally lists `none` and gives `--chat-template-kwargs '{"reasoning_effort":"medium"}'` for llama.cpp (https://unsloth.ai/docs/models/qwen3.8-how-to-run-and-fine-tune).
- vLLM docs: `--default-chat-template-kwargs '{"enable_thinking": false}'` sets a server default; "Request-level `chat_template_kwargs` always take priority over server defaults." (https://docs.vllm.ai/en/latest/features/reasoning_outputs.html)
- Discrepancy to note: Unsloth's Qwen3.5 page says Qwen3.5 *Small* models have "reasoning … disabled by default" (https://unsloth.ai/docs/models/qwen3.5-how-to-run-and-fine-tune), while the HF Qwen3.5-4B card says thinking is on by default. Set `enable_thinking` explicitly per request.
- Sampling (HF cards, both): thinking general temp 1.0/top_p 0.95/top_k 20/min_p 0/presence 1.5 (Qwen3.5) or 0.0 (Qwen3.8); non-thinking temp 0.7/top_p 0.8/top_k 20/presence 1.5.

### 1.6 Per-token logprobs in the OpenAI-compatible server

Source: https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/ and local `vllm/entrypoints/openai/chat_completion/protocol.py`, `vllm/config/model.py`.

- Chat Completions: standard OpenAI `logprobs: bool` + `top_logprobs: int` (not re-documented, assumed OpenAI semantics). Extra sampling params: `prompt_logprobs: int | None` (also in Completions) and `logprob_token_ids: list[int]` ("Specific vocab token IDs to return logprobs for at each generated position … Requires `logprobs=True`"; "More efficient than `top_logprobs=-1` when only a small fixed label set is needed").
- Extra chat params include `chat_template_kwargs` ("Additional keyword args to pass to the template renderer"), `return_tokens_as_token_ids` ("tokens are represented as strings of the form 'token_id:{token_id}'"), `return_token_ids`, `structured_outputs`, `cache_salt`, `priority`.
- Server limit: `ModelConfig.max_logprobs: int = Field(default=20, ge=-1)` (local source) → CLI `--max-logprobs`; the value `-1` is accepted by the field validator (semantics not documented on the page read — UNVERIFIED).
- Endpoints (index page https://docs.vllm.ai/en/latest/serving/online_serving/): `/v1/completions`, `/v1/chat/completions`, `/v1/chat/completions/batch`, `/v1/responses`, `/v1/embeddings`, `/v1/decisions`, `/v1/messages` (Anthropic), `/tokenize`, `/detokenize`, `/metrics`, `/health`, `/v1/models`.

---

## 2. Unsloth

| Item | Finding | Source |
|------|---------|--------|
| Version | PyPI `unsloth` **2026.10.6**, uploaded 2026-10-15 (matches local) | https://pypi.org/pypi/unsloth/json |
| Core training needs GPU | Requirements page: "Unsloth Core: Works on Linux and Windows"; "Supports NVIDIA GPUs since 2018+ … Minimum CUDA Capability 7.0"; "Unsloth works on AMD and Intel GPUs"; "Unsloth still works without a GPU, for Chat + Data Recipes"; "Unsloth Studio supports CPU devices for Chat for GGUF models and Data Recipes". Training is listed only for "NVIDIA, AMD, MLX, Intel devices". Local `device_type.py`: `UNSLOTH_ALLOW_CPU=1` → "Test-only CPU fallback: report 'cuda'…"; otherwise `raise NotImplementedError("Unsloth cannot find any torch accelerator? You need a GPU.")` | https://unsloth.ai/docs/get-started/beginner-start-here/unsloth-requirements ; `vendors/unsloth/unsloth/device_type.py` |
| What README "CPUs" means | README: "We support Multi GPU setups, NVIDIA, AMD, Intel GPUs, CPUs and the Vulkan backend"; `UNSLOTH_LLAMA_CPP_BACKEND=vulkan # or cpu, cuda, rocm, auto`; `UNSLOTH_CPU_THREADS=8 unsloth studio …`; `UNSLOTH_NO_TORCH=1` "GGUF-only mode". Combined with the requirements page → CPU = Studio GGUF chat (llama.cpp) + Data Recipes, **not** training. Studio page: "Powered by llama.cpp + Hugging Face". | https://github.com/unslothai/unsloth ; https://unsloth.ai/docs/new/studio.md |
| Apple/MLX | Requirements page Mac section: "Training, MLX and GGUF inference are ALL supported" (macOS 12+, "Intel or Apple Silicon" for the Studio/Desktop app); Core section says "Apple/Silicon/MLX is in the works" (internal inconsistency). Mac install page: "install Unsloth Desktop to run and train MLX or GGUF models on your Mac". `device_type.py` returns `"mlx"` when `unsloth_zoo.mlx.is_mlx_available()`. MLX is Apple-silicon-only → not applicable to an Intel Mac. | https://unsloth.ai/docs/get-started/beginner-start-here/unsloth-requirements ; https://unsloth.ai/docs/get-started/install/mac.md |
| VRAM table (QLoRA 4-bit / LoRA 16-bit, "absolute minimum") | 3B: 3.5 / 8 GB; 7B: 5 / 19; 8B: 6 / 22; 9B: 6.5 / 24; 14B: 8.5 / 33; 27B: 22 / 64 | https://unsloth.ai/docs/get-started/beginner-start-here/unsloth-requirements |
| Qwen3.5 fine-tune guide | Sizes 0.8B–122B-A10B. "Qwen3.5 bf16 LoRA VRAM use: 0.8B: 3GB • 2B: 5GB • 4B: 10GB • 9B: 22GB • 27B: 56GB"; 35B-A3B bf16 LoRA 74GB. **"It is not recommended to do QLoRA (4-bit) training on the Qwen3.5 models"**. "Please use `transformers v5` for Qwen3.5". Kernel compile "can take longer than normal, especially on T4 GPUs" (implies T4 is a supported target for small sizes; no explicit "fits on free T4" claim). SFT example: `r=16, lora_alpha=16, lora_dropout=0, max_steps=100, optim="adamw_8bit"`. GGUF: `model.save_pretrained_gguf("directory", tokenizer, quantization_method="q4_k_m")`. Colab notebooks: `Qwen3_5_(0_8B)_Vision.ipynb`, `Qwen3_5_(2B)_Vision.ipynb`, `Qwen3_5_(4B)_Vision.ipynb`, `Qwen3_5_(4B)_Vision_GRPO.ipynb`, `Qwen_3_5_27B_A100(80GB).ipynb`, `Qwen3_5_MoE.ipynb`. | https://unsloth.ai/docs/models/qwen3.5/fine-tune.md ; https://github.com/unslothai/notebooks |
| Free Colab T4 (16 GB) sufficiency | Per the bf16-LoRA table: 0.8B (3 GB), 2B (5 GB), 4B (10 GB) fit a 16 GB T4; **9B (22 GB) does not** in bf16 LoRA, and QLoRA is "not recommended" for Qwen3.5. Studio page: Colab "T4 GPUs … train and run most models up to 22B parameters" (generic claim, not Qwen3.5-specific). | same + https://unsloth.ai/docs/new/studio.md |
| Qwen3.6 | Run-only guide exists (`models/qwen3.6.md`: "Qwen3.6-27B and 35B-A3B"); no fine-tune page; notebook `Qwen3_6_MoE.ipynb` exists. `models/qwen3.6-how-to-run-and-fine-tune` → 404. | https://unsloth.ai/docs/sitemap.md ; https://github.com/unslothai/notebooks |
| Qwen3.8 | Train guide: "Qwen3.8-27B can now be fine-tuned"; "QLoRA works with 24GB", "LoRA needs >36GB"; RL "free or on 24GB VRAM"; Kaggle "2× Tesla T4" notebook `Kaggle-Qwen3.8_(27B)-Conversational.ipynb`; example `r=16, lora_alpha=16, learning_rate=2e-4, max_steps=100`; `model.save_pretrained_gguf("qwen38_gguf", tokenizer, quantization_method="q4_k_m")`. | https://unsloth.ai/docs/models/qwen3.8/train.md |
| Hyperparameter defaults (general guide) | rank "Choose 16 or 32" ("Suggested 8, 16, 32, 64, 128"); alpha "`r` (standard) or `r * 2`"; LR "2e-4 as a starting point" (RL 5e-6); "Recommended: 1-3 epochs"; effective batch 16 (2 × grad-accum 8); dropout 0 ("= 0 is optimized"); targets `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`. | https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide |
| GGUF export API | `model.save_pretrained_gguf("directory", tokenizer, quantization_method="q4_k_m"|"q8_0"|"f16")`; `model.push_to_hub_gguf(...)`; `ALLOWED_QUANTS` includes `not_quantized, fast_quantized, quantized, f32, f16, q8_0, q4_k_m, q5_k_m, q2_k, q3_k_l/m/s, q4_0, q4_1, q4_k_s, q4_k, q5_k, q5_0, q5_1, q5_k_s, q6_k, iq2_xxs, iq2_xs, iq3_xxs, q3_k_xs`; manual path = `save_pretrained_merged(..., save_method="merged_16bit")` → build llama.cpp → `convert_hf_to_gguf.py … --outtype f16`. | https://unsloth.ai/docs/basics/running-and-saving-models/saving-to-gguf |

Note: `docs.unsloth.ai/...` URLs 301-redirect to `unsloth.ai/docs/...`.

---

## 3. llama.cpp

### 3.1 Releases

| Item | Value | Source |
|------|-------|--------|
| Latest build tag | **b11716**, 2026-10-17T21:11:34Z (marked pre-release in API; "hexagon: add support for Q1_0 and Q2_0" #30122) | https://api.github.com/repos/ggml-org/llama.cpp/releases ; https://github.com/ggml-org/llama.cpp/releases |
| Latest named release | **v0.7.0**, 2026-10-15T17:13:19Z (GitHub "latest"); bNNNN tags are the per-commit builds, vX.Y.Z are curated releases (page links a "Releases and versioning" discussion; scheme not explained on the page itself) | https://github.com/ggml-org/llama.cpp/releases/tag/v0.7.0 ; https://api.github.com/repos/ggml-org/llama.cpp/releases/latest |
| v0.7.0 highlights | server default port → **9931** (#30159); LiliCorr draft selection on DFlash (#30506); GLM5-Next MTP; "Qwen3.8 Flash Next accepts image embeddings" (#30397); AVX2 Q2_0 dot product (#30447); Qwen3.5 embedding models (#27920) | same |

### 3.2 `llama-server` flags and endpoints (master README, read 2026-10-18)

Source: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md (raw copy grepped) and `tools/server/server-common.cpp`, `server-task.cpp` (raw).

- Endpoints: `POST /v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/embeddings`, `/v1/models`, `/v1/chat/completions/control` (realtime reasoning control), `/v1/chat/completions/input_tokens`, Anthropic `/v1/messages`; native `/completion`, `/tokenize`, `/health`, `/metrics`, `/slots`.
- Streaming: SSE via `stream: true` (note: browser `EventSource` unusable because it needs POST).
- **Logprobs:** native `n_probs` ("If greater than 0, the response also contains the probabilities of top N tokens for each generated token") and `post_sampling_probs`; response field `completion_probabilities[].top_logprobs`. **OpenAI chat `logprobs`/`top_logprobs` are mapped in source**: `server-common.cpp` L1458-1466: `if (json_value(body, "logprobs", false)) { … llama_params["n_probs"] = json_value(body, "top_logprobs", 20); } else if (body.contains("top_logprobs") …) throw "top_logprobs requires logprobs to be set to true"`; `server-task.cpp` emits `choice["logprobs"]` in chat responses and streaming chunks. (Not documented in the README's chat section.) Prompt logprobs in OpenAI format: not found — **UNVERIFIED**.
- Reasoning: `--reasoning-format FORMAT` = `none` | `deepseek` (→ `message.reasoning_content`) | `deepseek-legacy` (default: `auto`); `--reasoning-budget N` (-1 unrestricted, 0 immediate end); `--reasoning-budget-message`; `--jinja, --no-jinja` "(default: enabled)"; `--chat-template-kwargs` is used by Unsloth docs for `enable_thinking`/`reasoning_effort` (flag shown in Unsloth's commands; not grepped in README table).
- Speculative decoding: `--spec-draft-model, -md, --model-draft FNAME`; `--spec-draft-n-max N` (default 3) and `--spec-draft-n-min` (old `--draft-max`/`--draft-min` "removed"); `--spec-draft-p-min`; **`--spec-type none,draft-simple,draft-eagle3,draft-mtp,draft-dflash,draft-dspark,ngram-simple,ngram-map-k,ngram-map-k4v,ngram-mod,ngram-cache`** "(default: none)"; n-gram params `--spec-ngram-simple-size-n` (12) / `-size-m` (48) / `-min-hits` (1), `--spec-ngram-mod-n-min` (48) / `-n-max` (64) / lookup 24; draft threads `--spec-draft-threads/-td`.
- CPU threading: `-t, --threads N` "(default: -1)"; `-tb, --threads-batch N`; `-C, --cpu-mask M`; `--cpu-range`; `--cpu-strict <0|1>`; `--prio N`; `--numa distribute|isolate|numactl`; `--threads-http`; `-np, --parallel N` (-1 auto); `-cb, --cont-batching` (default on); `--cache-reuse N`; `-c, --ctx-size` (0 = from model).

### 3.3 Qwen3.5/3.6/3.8 support and CPU performance

| Item | Evidence | Source |
|------|----------|--------|
| Qwen3.5 arch | PR #19468 "[MODEL] support qwen3.5 series", merged **2026-02-10** | https://github.com/ggml-org/llama.cpp/pull/19468 (API) |
| GDN op (CPU+CUDA) | PR #19504 "ggml: add GATED_DELTA_NET op", merged **2026-03-07**; author described the CPU version as a basic vector reference impl; chunked fused path enabled by #20340 "llama : enable chunked fused GDN path" (2026-03-11); "graph : remove redundant GDN state transposes" #20443 (2026-03-13). `ggml/src/ggml-cpu/ops.cpp` contains `ggml_compute_forward_gated_delta_net_one_chunk/_f32`. Later GDN PRs are for Vulkan (#20334 2026-03-12, #20662, #24581), SYCL, WebGPU, Hexagon, CUDA — **no further CPU-specific GDN optimisation commit found after 2026-03-13** in `ops.cpp` history. | GitHub commits API for `ggml/src/ggml-cpu/ops.cpp`; https://github.com/ggml-org/llama.cpp/pull/19504 |
| Qwen3.6-27B / Qwen3.8-27B | Same `qwen3_5` HF model_type (see §0) → covered by Qwen3.5 support; Unsloth publishes GGUFs and llama-server commands for both | HF config.json; https://unsloth.ai/docs/models/qwen3.8-how-to-run-and-fine-tune |
| Qwen3.8-Flash-Next | PR #27742 "model: add Qwen3.8-Flash-Next (qwen4exp)", merged **2026-08-27** | API |
| CPU perf of hybrid models | Issue #17936 (EPYC 9454P, 24 threads, build 7315, Q4_0): Qwen3-Next-80B-A3B **tg128 11.76 t/s** vs Qwen3-30B-A3B **63.10 t/s** (same ~3B active) — closed stale 2026-01-28, pre-GDN-op. Issue #19480 (Ryzen AI 9 HX370, DDR5-5600, Qwen3-Coder-Next Q4_K_M): **7.74 t/s** CPU-only; closed 2026-03-15 after "Native GDN kernels have landed across all backends". PR #20334 (Strix Point, March 2026): Qwen3.5-9B Q4_K_M Vulkan TG 12 t/s and "Vulkan vs CPU on 9B: 1.7x PP, 7.5x TG" → CPU ≈ 1.6 t/s TG on that laptop at that build. Discussion #21136: Qwen3.5-4B Q4_K_M on Core 2 Duo E8400 / 8 GB DDR2, `-t 2`: **~2 tok/s**. | https://github.com/ggml-org/llama.cpp/issues/17936 ; https://github.com/ggml-org/llama.cpp/issues/19480 ; https://github.com/ggml-org/llama.cpp/pull/20334 ; https://github.com/ggml-org/llama.cpp/discussions/21136 |
| Assessment | Hybrid GDN models run on CPU but the CPU GDN path is a reference kernel; expect generation below the bandwidth ceiling and below a dense Qwen3 of equal size. For a 4-core AVX2 box, benchmark Qwen3.5-4B against Qwen3-4B (dense) before committing. | — |

### 3.4 `llama-diffusion-cli`

Source: https://github.com/ggml-org/llama.cpp/tree/master/examples/diffusion — "This directory contains implementations for Diffusion LLMs (DLLMs)". Supported/exampled: **Dream** (`dream7b.gguf`), **LLaDA** (`llada-8b.gguf`), **RND1** (`RND1-Base-0910.gguf`). Flags: `--diffusion-steps` (default 256), `--diffusion-algorithm 0..4` (origin/entropy/margin/random/confidence; default 4), `--diffusion-visual`, scheduling `--diffusion-eps` or `--diffusion-block-length`, plus `--temp/--top-k/--top-p/--seed`, `-m/-p/-ub/-c/-b`. Files: `CMakeLists.txt, README.md, diffusion-cli.cpp, diffusion.cpp, diffusion.h`. Background PRs #14644, #14771. It is a CLI example, not served by `llama-server`.

---

## 4. Ollama

| Item | Finding | Source |
|------|---------|--------|
| Current version | **v0.40.6**, 2026-10-16T22:00:46Z (local 0.40.3 was 2026-10-11). 0.40.6: q2_0 GGUF tensor/file types; 0.40.5 (2026-10-14): Nemotron 3.5 Super, OpenAI Responses fix; 0.40.0 (2026-09-25): MLX runtime default on Apple Silicon, cites `qwen3.8`, `qwen3.6`, `qwen3.5` | https://api.github.com/repos/ollama/ollama/releases/latest ; https://github.com/ollama/ollama/releases |
| Qwen3.5 small models | Library `qwen3.5`: tags `0.8b` (1.2–1.3 GB), `2b` (2.7–3.1 GB), `4b` (3.3–4.0 GB), `9b` (6.6–7.6 GB), `27b`, `35b`, `122b`; 256K ctx; badges vision/tools/thinking; "Updated 4 days ago"; quant level not stated on page | https://ollama.com/library/qwen3.5 |
| Qwen3.6 / Qwen3.8 | `qwen3.6`: 27b, 35b only. `qwen3.8`: 27b only (18 GB) + `27b-mlx`. No small variants. | https://ollama.com/library/qwen3.6 ; https://ollama.com/library/qwen3.8 |
| Thinking toggle | `/api/generate` and `/api/chat`: `think` — "(for thinking models) should the model think before responding? Can be a boolean or a thinking level" (`"low"`, `"medium"`, `"high"`, `"max"`); response `message.thinking`. | https://github.com/ollama/ollama/blob/main/docs/api.md |
| OpenAI-compatible API | `/v1/chat/completions` (streaming, JSON mode, tools, vision; `reasoning_effort`/`reasoning.effort` control thinking), `/v1/completions`, `/v1/embeddings`, `/v1/responses` (non-stateful only; supports `think`), `/v1/models`. **Unsupported: `logprobs`**, `tool_choice`, `logit_bias`, `n`, `user`. | https://docs.ollama.com/api/openai-compatibility |
| Perf caveat | Issue #14579 (open, 2026-03-03): Qwen3.5 in Ollama much slower than llama.cpp on GPU (e.g. 15–20 vs ~100 t/s for 35B-A3B); no CPU-only numbers. | https://github.com/ollama/ollama/issues/14579 |

Implication for the uncertainty router: Ollama cannot return logprobs → use `llama-server` (`logprobs`/`top_logprobs` → `n_probs`) or vLLM.

---

## 5. CPU inference throughput (published) and i3-8100 estimate

### 5.1 Verified data points (CPU-only unless noted)

| Source | CPU / memory | Model, quant | Threads | pp t/s | tg t/s |
|--------|--------------|--------------|---------|--------|--------|
| https://computingforgeeks.com/run-local-llm-llama-cpp/ (July 2026, build b10075 series) | Intel i7-6700 (Skylake 4C/8T, AVX2, DDR4-2133), RAM n/s | Qwen2.5-3B-Instruct Q4_K_M | 4 | 43 (pp512) | **11.8** (tg128) |
| https://www.mgroeber.de/benchmarking-llamacpp-on-legacy-hardware-2.html (build 6878, `-DGGML_AVX2=1`) | i7-8550U (Kaby Lake-R, 4C/8T, AVX2), 16 GB | Llama 7B Q4_0 (3.56 GiB) | 4 / 8 | 13.40 / 16.29 | **5.64 / 6.37** |
| https://github.com/ggml-org/llama.cpp/discussions/21136 | Core 2 Duo E8400, 8 GB DDR2 | Qwen3.5-4B Q4_K_M (Unsloth) | 2 | n/s | **~2** |
| https://huggingface.co/bartowski/huihui-ai_Qwen3-14B-abliterated-GGUF (deprecated Q4_0_X_X section) | EPYC 7702 (Zen 2, AVX2) | qwen2 3B Q4_K_M | 64 | 301 (pp512) | 18.8 (tg128) / 36.3 (tg512) |
| https://github.com/ggml-org/llama.cpp/issues/17936 (build 7315) | EPYC 9454P (Zen 4, AVX-512, 12ch DDR5-4800) | Qwen3-30B-A3B Q4_0 / Qwen3-Next-80B-A3B Q4_0 | 24 | 236 / 74 | 63.1 / 11.8 |
| https://github.com/ggml-org/llama.cpp/issues/19480 | Ryzen AI 9 HX370 (Zen 5, DDR5-5600 ≈ 65 GB/s real) | Qwen3-Coder-Next 80B-A3B Q4_K_M | 12 | n/s | 7.74 |
| https://github.com/ggml-org/llama.cpp/pull/20334 (Mar 2026, Vulkan PR; CPU ratio quoted) | Strix Point laptop | Qwen3.5-0.8B / 4B / 9B Q4_K_M (Vulkan iGPU) | — | 1337 / 346 / 249 | 85 / 21 / 12; CPU "7.5x" slower TG on 9B |
| https://x.com/TeksEdge/status/2093927202488455498 (self-reported, no bench output) | Ryzen 5 7540U (6C/12T, DDR5) | Qwen3.5-4B Q4_K_M, q4_0 KV | 9 | n/s | ~11 |
| https://lenovopress.lenovo.com/lp2463-power-efficient-llm-inference-with-intel-xeon-6-processors (2026-07-02) | Xeon 6972P/6960P (vLLM 0.21.0+cpu) | Llama 3(.1) 8B BF16 | — | TTFT <200 ms | "approaching 30 tokens/s/user" at 1–8 users |

Not usable / not found: llama.cpp discussion #4167 is Apple-Silicon-only (title "Performance of llama.cpp on Apple Silicon M-series"; no x86 CPU rows) — https://github.com/ggml-org/llama.cpp/discussions/4167. vLLM CPU perf dashboard numbers for Qwen3-1.7B/4B/8B on Xeon: the recipes pages (https://recipes.vllm.ai/Qwen/Qwen3-8B etc.) give configs, not throughput; openbenchmarking `pts/vllm-cpu` returned HTTP 403 — **UNVERIFIED**. Intel's Qwen3 article has Xeon 6 numbers only as an image (https://www.intel.com/content/www/us/en/developer/articles/technical/accelerate-qwen3-large-language-models.html) — **UNVERIFIED**. The kunalganglani.com benchmark page has no CPU rows.

### 5.2 Bandwidth-bound decode ceiling for i3-8100 (estimate)

Formula: `tok/s ≈ effective_memory_bandwidth / bytes_of_weights_read_per_token` (single-stream decode, weights dominate; KV/GDN state adds a few %). Theoretical DDR4-2400 dual channel = 2 ch × 8 B × 2400 MT/s = **38.4 GB/s**. Calibration from the two Skylake-class rows above: i7-6700 (1.79–1.9 GiB model → 11.8 t/s ⇒ ~22 GB/s effective) and i7-8550U (3.56 GiB → 5.6–6.4 t/s ⇒ ~21–23 GB/s effective). So assume **~22 GB/s effective (≈60% of theoretical)** for a 4-core Coffee Lake desktop.

Weight bytes/token ≈ params × bits-per-weight/8 (Q4_K_M ≈ 4.8 bpw incl. metadata; Q8_0 ≈ 8.5 bpw):

| Model | Q4_K_M size (approx) | Ceiling @38.4 GB/s | Estimate @22 GB/s | Q8_0 size | Estimate @22 GB/s |
|-------|----------------------|--------------------|-------------------|-----------|-------------------|
| Qwen3-0.6B / Qwen3.5-0.8B | 0.36 / 0.48 GB | 100+ / 80 | 60 / 45 (**compute-bound; realistic 25–40**) | 0.64 / 0.85 GB | 34 / 26 |
| Qwen3-1.7B / Qwen3.5-2B | 1.0 / 1.2 GB | 38 / 32 | 22 / 18 | 1.8 / 2.1 GB | 12 / 10 |
| Qwen3-4B / Qwen3.5-4B | 2.4 / 2.4 GB | 16 | **~9** | 4.3 GB | ~5 |
| Qwen3-8B / Qwen3.5-9B | 4.8 / 5.4 GB | 8 / 7 | **~4.5 / ~4** | 8.5 / 9.6 GB | ~2.6 / ~2.3 |

Caveats (explicitly estimates): (1) Qwen3.5-family hybrid layers run through llama.cpp's reference CPU `GATED_DELTA_NET` kernel; measured ratios (issue #17936: 5.4× slower than an equal-active-size MoE; PR #20334: CPU 7.5× slower than iGPU) suggest **discounting Qwen3.5 numbers by roughly 1.5–3× vs dense Qwen3 until measured** — the Core 2 Duo 2 t/s point is consistent with that. (2) Prompt processing on 4 AVX2 cores is compute-bound: expect ~30–60 t/s pp for 3–4B Q4 (i7-6700: 43 t/s), i.e. a 1k-token prompt costs ~20–30 s. (3) Use `-t 4` (physical cores; i3-8100 has no HT). (4) Q8_0 roughly halves decode speed vs Q4_K_M; Q4_K_M is the practical default. (5) vLLM-CPU on AVX2 will not beat these numbers for single-stream decode and its GDN path is unaccelerated there; vLLM-CPU's advantage (batched throughput on many-core AVX-512/AMX Xeons) does not apply to a 4-core i3.

---

## 6. Rust ecosystem (crates.io, queried 2026-10-18)

Toolchain: **Rust 1.99.1, 2026-10-16T04:47:44Z** (1.99.0 2026-10-01; 1.98.1 2026-09-03; 1.98.0 2026-08-20; 1.97.1 2026-07-16; local 1.96.0 is behind) — https://api.github.com/repos/rust-lang/rust/releases, https://github.com/rust-lang/rust/releases.

htmx (JS): npm `htmx.org` dist-tag `latest` = **2.0.11** (2026-09-22), `next` = **4.0.0** (2026-08-28); GitHub "latest" release v4.0.0 (2026-08-28); licence 0BSD — https://registry.npmjs.org/htmx.org, https://api.github.com/repos/bigskysoftware/htmx/releases/latest. (htmx 4 is a major rewrite; pin 2.0.x unless you adopt 4 deliberately.)

Crate data from `https://crates.io/api/v1/crates/<name>` (`max_stable_version`, that version's `created_at`, `license`):

| Crate | Version | Date | Licence | Purpose |
|-------|---------|------|---------|---------|
| axum | 0.8.9 | 2026-04-14 | MIT | HTTP router/web framework on tower+hyper |
| axum-extra | 0.12.6 | 2026-04-14 | MIT | Extra extractors/responses for axum (typed headers, SSE helpers) |
| tokio | 1.53.3 | 2026-10-13 | MIT | Async runtime |
| tokio-stream | 0.1.19 | 2026-07-22 | MIT | Stream utilities for tokio |
| tokio-tungstenite | 0.30.0 | 2026-07-11 | MIT | WebSocket client/server on tokio |
| tower | 0.5.3 | 2026-01-12 | MIT | Service/middleware abstraction |
| tower-http | 0.7.1 | 2026-08-31 | MIT | HTTP middleware (trace, cors, compression, timeout) |
| tower-sessions | 0.15.0 | 2026-02-01 | MIT | Session middleware for tower/axum |
| hyper | 1.12.0 | 2026-10-06 | MIT | Low-level HTTP/1+2 implementation |
| reqwest | 0.13.5 | 2026-09-08 | MIT OR Apache-2.0 | HTTP client |
| reqwest-eventsource | 0.6.0 | 2024-03-29 | MIT OR Apache-2.0 | SSE client built on reqwest (stale; check compat with reqwest 0.13) |
| eventsource-stream | 0.2.3 | 2022-02-17 | MIT OR Apache-2.0 | SSE parser for byte streams (stale but tiny) |
| async-stream | 0.3.6 | 2024-10-01 | MIT | `stream!` macro for async generators |
| futures | 0.3.34 | 2026-08-11 | MIT OR Apache-2.0 | Futures/streams combinators |
| serde | 1.0.229 | 2026-07-18 | MIT OR Apache-2.0 | Serialization framework |
| serde_json | 1.0.154 | 2026-10-11 | MIT OR Apache-2.0 | JSON for serde |
| schemars | 1.2.3 | 2026-10-12 | MIT | JSON Schema generation from Rust types |
| jsonschema | 0.59.0 | 2026-10-16 | MIT | JSON Schema validation |
| utoipa | 6.0.0 | 2026-09-22 | MIT OR Apache-2.0 | Compile-time OpenAPI docs |
| utoipa-swagger-ui | 10.0.1 | 2026-09-23 | MIT OR Apache-2.0 | Swagger UI for utoipa |
| tracing | 0.1.44 | 2025-12-18 | MIT | Structured logging/tracing facade |
| tracing-subscriber | 0.3.23 | 2026-03-13 | MIT | Subscribers/filters/formatters for tracing |
| tracing-opentelemetry | 0.34.0 | 2026-09-23 | MIT | Bridge tracing → OpenTelemetry |
| opentelemetry | 0.33.1 | 2026-10-09 | Apache-2.0 | OpenTelemetry API |
| metrics | 0.24.6 | 2026-05-13 | MIT | Metrics facade |
| metrics-exporter-prometheus | 0.18.3 | 2026-04-30 | MIT AND Apache-2.0 | Prometheus exporter for `metrics` |
| sqlx | 0.9.0 | 2026-05-21 | MIT OR Apache-2.0 | Async SQL (compile-time checked), SQLite/PG/MySQL |
| rusqlite | 0.40.2 | 2026-08-08 | MIT | Sync SQLite bindings |
| askama | 0.16.1 | 2026-09-04 | MIT OR Apache-2.0 | Compile-time Jinja-like templates |
| maud | 0.27.0 | 2025-02-02 | MIT OR Apache-2.0 | Compile-time HTML macro |
| anyhow | 1.0.104 | 2026-07-18 | MIT OR Apache-2.0 | Flexible error type |
| thiserror | 2.0.21 | 2026-09-23 | MIT OR Apache-2.0 | `derive(Error)` |
| clap | 4.6.7 | 2026-09-14 | MIT OR Apache-2.0 | CLI argument parser |
| figment | 0.10.19 | 2024-05-17 | MIT OR Apache-2.0 | Layered config (toml/env) |
| config | 0.15.27 | 2026-09-30 | MIT OR Apache-2.0 | Layered configuration |
| uuid | 1.28.0 | 2026-10-10 | Apache-2.0 OR MIT | UUID generation/parsing |
| chrono | 0.4.45 | 2026-06-04 | MIT OR Apache-2.0 | Date/time |
| criterion | 0.8.2 | 2026-02-04 | Apache-2.0 OR MIT | Statistics-driven micro-benchmarks |
| goose | 0.18.1 | 2025-08-14 | Apache-2.0 | Load-testing framework (Locust-like) |
| async-openai | 0.42.2 | 2026-10-10 | MIT | OpenAI-API client (works against OpenAI-compatible servers) |
| leptos | 0.8.22 | 2026-10-05 | MIT | Full-stack reactive web framework |
| dioxus | 0.7.10 | 2026-07-30 | MIT OR Apache-2.0 | Cross-platform UI framework |

Compatibility notes to check at build time (not verified here): `reqwest-eventsource 0.6.0` (2024) against `reqwest 0.13`; `axum 0.8` vs `tower-http 0.7`; `utoipa 6` vs `axum 0.8`.

---

## 7. Evaluation datasets, licences, and harnesses

Licence read from the HF dataset API `cardData.license`/tags (`https://huggingface.co/api/datasets/<id>`) or GitHub repo licence API, 2026-10-18.

| Benchmark | Canonical source | Licence (as published) | Notes |
|-----------|------------------|------------------------|-------|
| GSM8K | `openai/gsm8k` (HF) | MIT | lastModified 2026-03-23 |
| MATH | `hendrycks/competition_math` (HF); `EleutherAI/hendrycks_math` | MIT / MIT | |
| HumanEval | `openai/openai_humaneval` (HF) | MIT | |
| MBPP | `google-research-datasets/mbpp` (HF) | CC-BY-4.0 | |
| MMLU | `cais/mmlu` (HF) | MIT | |
| MMLU-Pro | `TIGER-Lab/MMLU-Pro` (HF) | MIT | lastModified 2026-05-02 |
| GPQA | `Idavidrein/gpqa` (HF) | CC-BY-4.0 | **gated=auto** (accept terms); lastModified 2026-09-21 |
| BIG-Bench Hard | `lukaemon/bbh` (HF, no licence tag); github.com/suzgunmirac/BIG-Bench-Hard | MIT (GitHub) | HF mirror carries no licence metadata |
| IFEval | `google/IFEval` (HF) | Apache-2.0 | |
| TruthfulQA | `truthfulqa/truthful_qa` (HF) | Apache-2.0 | |
| SimpleQA | github.com/openai/simple-evals (MIT); mirror `basicv8vc/SimpleQA` (HF) | MIT / MIT | |
| LiveCodeBench | `livecodebench/code_generation_lite` (HF) tag `license:cc` (unspecified CC variant); github.com/LiveCodeBench/LiveCodeBench | MIT (code, GitHub); data "cc" — **variant UNVERIFIED** | check dataset card for exact CC terms |
| AIME 2024 | `HuggingFaceH4/aime_2024` (no licence tag); `Maxwell-Jia/AIME_2024` (MIT); `AI-MO/aimo-validation-aime` (Apache-2.0) | varies | problems are AMC/MAA-authored; mirrors' licences cover packaging only |
| AIME 2025 | `math-ai/aime25` (Apache-2.0, 2026-01-19); `opencompass/AIME2025` (MIT); `yentinglin/aime_2025` (none) | varies | same caveat |
| ARC-Challenge | `allenai/ai2_arc` (HF) | CC-BY-SA-4.0 | |
| HellaSwag | `Rowan/hellaswag` (HF; README "Licensing Information: MIT https://github.com/rowanz/hellaswag/blob/master/LICENSE") | MIT | GitHub repo API returned "Repository access blocked"; licence taken from HF README |
| RouteLLM routing data | `routellm/gpt4_dataset` (HF) Apache-2.0; `lmarena-ai/arena-human-preference-55k` (HF) Apache-2.0; code github.com/lm-sys/RouteLLM Apache-2.0 | Apache-2.0 | `routellm/lmsys-arena-human-preference-55k` → not accessible via API (private/moved) |
| "Hybrid LLM" (Ding et al., ICLR 2024) routing data | Paper https://arxiv.org/abs/2404.14618 | — | **No public dataset release found** (searches returned only the paper and MixInstruct, a separate benchmark by Jiang et al. 2023). UNVERIFIED as public. |

Harnesses:

| Harness | Version | Date | Licence | Source |
|---------|---------|------|---------|--------|
| lm-evaluation-harness (`lm_eval`) | 0.4.13 | 2026-08-31 | MIT | https://pypi.org/pypi/lm_eval/json ; https://github.com/EleutherAI/lm-evaluation-harness/releases/tag/v0.4.13 |
| lighteval | 0.13.0 | 2025-11-24 | MIT | https://pypi.org/pypi/lighteval/json ; https://github.com/huggingface/lighteval/releases/tag/v0.13.0 (no release in ~11 months) |
| inspect-ai | 0.3.281 | 2026-10-15 | MIT | https://pypi.org/pypi/inspect-ai/json ; https://github.com/UKGovernmentBEIS/inspect_ai/releases/tag/0.3.281 |

All three can target an OpenAI-compatible endpoint (llama-server/vLLM); not re-verified per harness here.

---

## 8. Recommendations (derived from the above)

1. **Serving on the i3-8100:** llama.cpp `llama-server` (v0.7.0 / b117xx), `-t 4`, Q4_K_M. It gives SSE streaming, OpenAI `logprobs`/`top_logprobs`, `--reasoning-format deepseek`, `--chat-template-kwargs` for `enable_thinking`/`reasoning_effort`, and n-gram speculation (`--spec-type ngram-mod` or `ngram-simple`) with no draft model. vLLM-CPU is Linux-only, AVX2 = "limited features", and its GDN path is unaccelerated without AVX-512BF16/AMX; keep it for AVX-512 Xeon deployments or for batched eval, not for this box.
2. **Model family:** the only sub-10B models from the newest generations are Qwen3.5-0.8B/2B/4B/9B. Benchmark Qwen3.5-4B against dense Qwen3-4B on the i3 — the hybrid CPU kernel may erase Qwen3.5's quality edge with a 1.5–3× speed penalty.
3. **Fine-tuning:** Unsloth on a free Colab T4 covers Qwen3.5-0.8B/2B/4B in bf16 LoRA (3/5/10 GB stated); Qwen3.5-9B needs ~22 GB (not T4); Unsloth advises against QLoRA for Qwen3.5. Export with `save_pretrained_gguf(..., "q4_k_m")`. No CPU training path exists.
4. **Uncertainty router:** avoid Ollama's OpenAI endpoint (no logprobs). vLLM additionally offers `logprob_token_ids` for a fixed label set and `prompt_logprobs`.
5. **Pin:** vLLM 0.31.0 (0.32.0 tag unreleased), Rust 1.99.1, htmx 2.0.11 (or consciously 4.0.0), crates per §6.

---

## Verification log

Status: OK = opened and used; REDIRECT = followed to new URL; PARTIAL = opened but did not contain the requested detail; FAIL = could not open (UNVERIFIED).

### vLLM
- OK https://github.com/vllm-project/vllm/releases/tag/v0.32.0 (date "17 Oct 18:57", no year)
- OK https://github.com/vllm-project/vllm/releases (v0.31.0 "Latest" 05 Oct; v0.32.0 absent)
- OK https://api.github.com/repos/vllm-project/vllm/releases ; …/releases/latest ; …/tags ; …/git/refs/tags/v0.32.0 (releases/tags/v0.32.0 → 404 = no Release object)
- OK https://pypi.org/pypi/vllm/json (0.31.0, 2026-10-05)
- OK https://docs.vllm.ai/en/latest/getting_started/installation/cpu.html
- OK https://docs.vllm.ai/en/latest/models/hardware_supported_models/cpu/ and raw `docs/models/hardware_supported_models/cpu.md@v0.32.0`
- OK https://docs.vllm.ai/en/latest/models/supported_models.html
- OK https://docs.vllm.ai/en/latest/features/reasoning_outputs.html
- REDIRECT→OK https://docs.vllm.ai/en/latest/features/spec_decode.html → https://docs.vllm.ai/en/latest/features/speculative_decoding/
- REDIRECT→OK https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html → https://docs.vllm.ai/en/latest/serving/online_serving/ → https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/
- FAIL (404) https://docs.vllm.ai/en/latest/features/compatibility_matrix/ ; no `compatibility_matrix*.md` in local docs either → feature×hardware matrix UNVERIFIED
- OK https://github.com/vllm-project/vllm/issues?q=is%3Aissue+Qwen3.5+CPU (page 1 of 324; no CPU-inference-unsupported statement)
- PARTIAL https://github.com/vllm-project/vllm/issues?q=is%3Aissue+%22CPU%22+%28mamba+OR+%22Qwen3-Next%22+OR+hybrid+OR+GDN%29+backend (page 1 only)
- OK GitHub search API `repo:vllm-project/vllm CPU GDN in:title` and `speculative CPU backend in:title`
- OK local `vendors/vllm` files: `vllm/platforms/cpu.py`, `vllm/v1/worker/cpu_model_runner.py`, `vllm/v1/worker/cpu_worker.py`, `vllm/model_executor/layers/mamba/ops/cpu/gdn_attention.py`, `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`, `vllm/reasoning/__init__.py`, `vllm/entrypoints/openai/chat_completion/protocol.py`, `vllm/config/model.py`, `docs/getting_started/installation/cpu.x86.inc.md`
- OK https://huggingface.co/Qwen/Qwen3.5-4B ; https://huggingface.co/Qwen/Qwen3.8-27B ; HF model-list API (`author=Qwen&search=Qwen3.5|3.6|3.8`) ; raw `config.json` for Qwen3.5-4B, Qwen3.5-9B, Qwen3.6-27B, Qwen3.8-27B
- WebSearch result pages (not opened): https://github.com/vllm-project/vllm/issues/35391, /35344, /36275, /34948, /40934, /47639, /35924, https://github.com/vllm-project/vllm-neuron/issues/53, https://github.com/vllm-project/vllm-xpu-kernels/issues/172

### Unsloth
- REDIRECT (docs.unsloth.ai → unsloth.ai/docs) then OK: https://unsloth.ai/docs/get-started/beginner-start-here/unsloth-requirements ; https://unsloth.ai/docs/models/qwen3.5-how-to-run-and-fine-tune ; https://unsloth.ai/docs/models/qwen3.8-how-to-run-and-fine-tune ; https://unsloth.ai/docs/basics/running-and-saving-models/saving-to-gguf ; https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide
- FAIL (404) https://unsloth.ai/docs/models/qwen3.6-how-to-run-and-fine-tune (run-only page is https://unsloth.ai/docs/models/qwen3.6.md per sitemap — not opened)
- OK https://unsloth.ai/docs/sitemap.md ; https://unsloth.ai/docs/models/qwen3.5/fine-tune.md ; https://unsloth.ai/docs/models/qwen3.8/train.md ; https://unsloth.ai/docs/get-started/install/mac.md ; https://unsloth.ai/docs/new/studio.md
- OK https://github.com/unslothai/unsloth (README) ; https://github.com/unslothai/notebooks (first 100k chars)
- OK https://pypi.org/pypi/unsloth/json ; local `vendors/unsloth/unsloth/device_type.py`, `README.md`
- PARTIAL WebSearch "unsloth MLX" (only third-party forks + Studio pages surfaced)

### llama.cpp
- OK https://github.com/ggml-org/llama.cpp/releases ; https://api.github.com/repos/ggml-org/llama.cpp/releases(?per_page=8, /latest) ; https://github.com/ggml-org/llama.cpp/releases/tag/v0.7.0
- OK https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md (+ raw) ; raw `tools/server/server-common.cpp`, `server-task.cpp`, `server-context.cpp`, `server-http.cpp` ; raw `ggml/src/ggml-cpu/ops.cpp`
- OK https://github.com/ggml-org/llama.cpp/tree/master/examples/diffusion
- OK GitHub API: pulls/19468, /19504, /20334, /27742 ; commits?path=ggml/src/ggml-cpu/ops.cpp ; search `gated_delta_net in:title is:merged`
- OK https://github.com/ggml-org/llama.cpp/issues/19480 ; /issues/17936 ; /discussions/21136 ; /discussions/4167 (Apple-only)
- PARTIAL raw `convert_hf_to_gguf.py` grep for `Qwen3_5|Qwen4` returned nothing (class naming differs; arch support established via PR #19468 instead)
- WebSearch-only (not opened): PR #20340, #20443, #26001, #30207, issues #27623, #19345, #15940, #22967, #24712, #27162

### Ollama
- OK https://github.com/ollama/ollama/releases ; https://api.github.com/repos/ollama/ollama/releases/latest
- OK https://ollama.com/library/qwen3.5 ; https://ollama.com/library/qwen3.6 ; https://ollama.com/library/qwen3.8
- OK https://github.com/ollama/ollama/blob/main/docs/api.md ; https://docs.ollama.com/api/openai-compatibility ; https://github.com/ollama/ollama/issues/14579

### CPU throughput
- OK https://computingforgeeks.com/run-local-llm-llama-cpp/ ; https://www.mgroeber.de/benchmarking-llamacpp-on-legacy-hardware-2.html ; https://huggingface.co/bartowski/huihui-ai_Qwen3-14B-abliterated-GGUF ; https://lenovopress.lenovo.com/lp2463-power-efficient-llm-inference-with-intel-xeon-6-processors
- PARTIAL https://www.intel.com/content/www/us/en/developer/articles/technical/accelerate-qwen3-large-language-models.html (numbers only in an image) ; https://github.com/nandan2003/edge-cpu-inference (table in CSV not shown) ; https://www.kunalganglani.com/llm-benchmarks (no CPU rows) ; https://qwen.readthedocs.io/en/latest/getting_started/speed_benchmark.html (GPU only)
- FAIL (403) https://openbenchmarking.org/test/pts/vllm-cpu ; https://openbenchmarking.org/result/2606062-SHIF-PROJECT21
- WebSearch-only, low-trust: https://x.com/TeksEdge/status/2093927202488455498 (Ryzen 5 7540U ~11 t/s), markaicode/willitrunai/llm-bench (modelled estimates, not used)

### Rust
- OK https://crates.io/api/v1/crates/<crate> for all 42 crates listed ; https://registry.npmjs.org/htmx.org ; https://api.github.com/repos/bigskysoftware/htmx/releases/latest ; https://github.com/rust-lang/rust/releases ; https://api.github.com/repos/rust-lang/rust/releases

### Datasets / harnesses
- OK HF dataset API for: openai/gsm8k, hendrycks/competition_math, EleutherAI/hendrycks_math, openai/openai_humaneval, google-research-datasets/mbpp, cais/mmlu, TIGER-Lab/MMLU-Pro, Idavidrein/gpqa, lukaemon/bbh, google/IFEval, truthfulqa/truthful_qa, basicv8vc/SimpleQA, livecodebench/code_generation_lite, livecodebench/code_generation, HuggingFaceH4/aime_2024, Maxwell-Jia/AIME_2024, AI-MO/aimo-validation-aime, math-ai/aime25, opencompass/AIME2025, yentinglin/aime_2025, allenai/ai2_arc, Rowan/hellaswag (+ raw README), routellm/gpt4_dataset, lmarena-ai/arena-human-preference-55k
- FAIL HF API: routellm/lmsys-arena-human-preference-55k, simplescaling/aime_2024 ("Invalid username or password" = private/absent); lmsys/lmsys-arena-human-preference-55k (non-JSON response)
- OK GitHub licence API: suzgunmirac/BIG-Bench-Hard (MIT), LiveCodeBench/LiveCodeBench (MIT), lm-sys/RouteLLM (Apache-2.0), openai/simple-evals (MIT), EleutherAI/lm-evaluation-harness (MIT), huggingface/lighteval (MIT), UKGovernmentBEIS/inspect_ai (MIT)
- FAIL GitHub: rowanz/hellaswag ("Repository access blocked"), m-a-p/Hybrid-LLM (Not Found)
- OK PyPI JSON: lm_eval, lighteval, inspect-ai ; GitHub releases/latest for the three harness repos
- WebSearch-only: https://arxiv.org/abs/2404.14618 (Hybrid LLM paper), https://lmsys.org/blog/2024-07-01-routellm/
