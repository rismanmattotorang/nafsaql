# Literature verification notes — System 1 / System 2 hybrid LLM paper

Verified 2026-10-18. Companion file: `lit_s1s2.bib`.

Rule applied: an entry is in the `.bib` only if its arXiv abstract page, venue proceedings page, or DOI/Crossref record was opened and the title/authors/year confirmed from that page. For each entry the URL(s) opened are listed. Venue fields were added only when the arXiv "Comments"/journal-ref line or a venue proceedings page confirmed them. Where a venue was reported in the task brief but could not be confirmed from a venue page, the entry is kept as an arXiv preprint and the discrepancy is flagged below.

Legend: **[V]** verified from page; **[V-venue]** venue additionally confirmed from a venue/proceedings page; **[flag]** something the design paper should be careful about.

---

## Part 1a — Test-time compute, verifiers, self-correction

| key | URL(s) opened | status / notes |
|---|---|---|
| snell2024scaling | https://arxiv.org/abs/2408.03314 ; https://proceedings.iclr.cc/paper_files/paper/2025/hash/1b623663fd9b874366f3ce019fdfdd44-Abstract-Conference.html | [V-venue] ICLR 2025 (oral). **[flag]** ICLR title differs from arXiv title ("...Scaling Parameters for Reasoning"). |
| brown2024monkeys | https://arxiv.org/abs/2407.21787 | [V] arXiv only; no venue found on ICLR/OpenReview domains. |
| wang2023selfconsistency | https://arxiv.org/abs/2203.11171 | [V] Comments: "Published at ICLR 2023". |
| wei2022cot | https://arxiv.org/abs/2201.11903 ; https://proceedings.neurips.cc/paper/2022/hash/9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html | [V-venue] NeurIPS 2022 (vol. 35). |
| lightman2024verify | https://arxiv.org/abs/2305.20050 ; https://proceedings.iclr.cc/paper_files/paper/2024/hash/aca97732e30bcf1303bc22ac3924fd16-Abstract-Conference.html | [V-venue] ICLR 2024. |
| cobbe2021gsm8k | https://arxiv.org/abs/2110.14168 | [V] arXiv only. |
| madaan2023selfrefine | https://arxiv.org/abs/2303.17651 ; https://neurips.cc/virtual/2023/poster/71632 | [V-venue] NeurIPS 2023 poster. |
| shinn2023reflexion | https://arxiv.org/abs/2303.11366 ; https://proceedings.neurips.cc/paper_files/paper/2023/hash/1b44b878bb782e6954cd888628510e90-Abstract-Conference.html | [V-venue] NeurIPS 2023. **[flag]** arXiv lists 6 authors (incl. Edward Berman); NeurIPS proceedings lists 5. Bib uses the proceedings list and notes the difference. |
| huang2024selfcorrect | https://arxiv.org/abs/2310.01798 | [V] Comments: "ICLR 2024". |
| kamoi2024survey | https://arxiv.org/abs/2406.01297 | [V] journal-ref TACL 12:1417-1440 (2024), DOI 10.1162/tacl_a_00713. |
| muennighoff2025s1 | https://arxiv.org/abs/2501.19393 | [V] arXiv only (no venue in Comments). |
| deepseek2025r1 | https://arxiv.org/abs/2501.12948 | [V] arXiv page carries journal-ref "Nature 645, 633-638 (2025)" and DOI 10.1038/s41586-025-09422-z; v2 revised 4 Jan 2026. |
| guo2025deepseekr1nature | https://www.nature.com/articles/s41586-025-09422-z (reached via idp.nature.com cookie redirect) | [V] Title "DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning", first author Daya Guo, Nature 645:633-638, published 17 Sep 2025. |
| openai2024o1 | https://arxiv.org/abs/2412.16720 | [V] "OpenAI o1 System Card", author entity OpenAI + ~263 named authors (first: Aaron Jaech). v1 21 Dec 2024. |
| qwen2025qwen3 | https://arxiv.org/abs/2505.09388 | [V] 60 authors, first An Yang; 14 May 2025. |
| yang2024qwen25 | https://arxiv.org/abs/2412.15115 | [V] "Qwen" + An Yang et al. (42 authors); v1 19 Dec 2024. |

## Part 1b — Speculative decoding, routing, cascades, fast/slow

| key | URL(s) opened | status / notes |
|---|---|---|
| leviathan2023speculative | https://arxiv.org/abs/2211.17192 | [V] Comments: "ICML 2023 Oral". |
| chen2023speculative | https://arxiv.org/abs/2302.01318 | [V] arXiv only. |
| cai2024medusa | https://arxiv.org/abs/2401.10774 ; https://proceedings.mlr.press/v235/cai24b.html | [V-venue] ICML 2024, PMLR 235:5209-5235. |
| li2024eagle | https://arxiv.org/abs/2401.15077 ; https://proceedings.mlr.press/v235/li24bt.html | [V-venue] ICML 2024, PMLR 235:28935-28948. |
| chen2023frugalgpt | https://arxiv.org/abs/2305.05176 | [V] arXiv only. |
| ong2025routellm | https://arxiv.org/abs/2406.18665 ; https://proceedings.iclr.cc/paper_files/paper/2025/hash/5503a7c69d48a2f86fc00b3dc09de686-Abstract-Conference.html | [V-venue] ICLR 2025. **[flag]** ICLR title says "...LLMs **from** Preference Data"; arXiv says "with". Bib uses ICLR title. |
| ding2024hybridllm | https://arxiv.org/abs/2404.14618 | [V] Comments: "Accepted to ICLR 2024 (main conference)". |
| yue2024cascades | https://arxiv.org/abs/2310.03094 | [V] Comments: "Accepted to ICLR 2024". |
| saha2025system1x | https://arxiv.org/abs/2407.14414 | [V] Comments: "ICLR 2025 (Camera-Ready)". |
| su2025dualformer | https://arxiv.org/abs/2410.09918 ; https://proceedings.iclr.cc/paper_files/paper/2025/hash/ed45d6a03de84cc650cae0655f699356-Abstract-Conference.html | [V-venue] ICLR 2025. |
| booch2021thinking | https://arxiv.org/abs/2010.06002 | [V] journal-ref: Proc. AAAI 2021, 35(17):15042-15046. |
| bengio2019system2 | https://neurips.cc/virtual/2019/invited-talk/15488 | [V] NeurIPS 2019 Invited Talk (Posner Lecture). No paper/proceedings record exists; cite as @misc talk. |
| graves2016act | https://arxiv.org/abs/1603.08983 | [V] arXiv only. |
| raposo2024mod | https://arxiv.org/abs/2404.02258 | [V] arXiv only. |

## Part 1c — Adaptive thinking / when to think

| key | URL(s) opened | status / notes |
|---|---|---|
| zhang2025adaptthink | https://arxiv.org/abs/2505.13417 | [V] arXiv only (no venue in Comments). |
| fang2025thinkless | https://arxiv.org/abs/2505.13379 | [V] arXiv only. |
| he2025selfroute | https://arxiv.org/abs/2505.20664 | [V] "Self-Route" exists; He, Ding, et al. (HIT). arXiv only. |
| arora2025efficient | https://arxiv.org/abs/2502.04463 | [V] "Training Language Models to Reason Efficiently" (this is the "Learning to Reason Efficiently" item). Comments: "NeurIPS 2025". |
| hou2025thinkprune | https://arxiv.org/abs/2504.01296 | [V] arXiv only. |
| yang2025deer | https://arxiv.org/abs/2504.15895 | [V] arXiv only. |
| chen2024overthinking | https://arxiv.org/abs/2412.21187 | [V] arXiv only. |

## Part 1d — Serving / PEFT / quantization

| key | URL(s) opened | status / notes |
|---|---|---|
| kwon2023pagedattention | https://arxiv.org/abs/2309.06180 | [V] Comments: "SOSP 2023". |
| hu2022lora | https://arxiv.org/abs/2106.09685 ; https://iclr.cc/virtual/2022/poster/6319 | [V-venue] ICLR 2022. |
| dettmers2023qlora | https://arxiv.org/abs/2305.14314 ; https://proceedings.neurips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html | [V-venue] NeurIPS 2023 (vol. 36). |
| frantar2023gptq | https://arxiv.org/abs/2210.17323 | [V] Comments: "ICLR 2023". |
| lin2024awq | https://arxiv.org/abs/2306.00978 | [V] Comments: "MLSys 2024 Best Paper Award". |

## Part 1e — Calibration / selective prediction / deferral / conformal

| key | URL(s) opened | status / notes |
|---|---|---|
| guo2017calibration | https://arxiv.org/abs/1706.04599 | [V] Comments: "ICML 2017". |
| kadavath2022know | https://arxiv.org/abs/2207.05221 | [V] 36 authors; arXiv only. |
| tian2023calibration | https://arxiv.org/abs/2305.14975 | [V] Comments: "EMNLP 2023 Camera Ready". |
| chow1970reject | https://api.crossref.org/works/10.1109/TIT.1970.1054406 (IEEE Xplore page returned empty) | [V] IEEE Trans. Inf. Theory 16(1):41-46, Jan 1970. |
| geifman2017selective | https://arxiv.org/abs/1705.08500 ; https://proceedings.neurips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html | [V-venue] NIPS 2017 (vol. 30). |
| madras2018defer | https://arxiv.org/abs/1711.06664 | [V] Comments: "Accepted as a conference paper at NeurIPS 2018". |
| mozannar2020defer | https://arxiv.org/abs/2006.01862 | [V] Comments: "ICML 2020". |
| angelopoulos2023conformal | https://arxiv.org/abs/2107.07511 ; https://api.crossref.org/works/10.1561/2200000101 | [V] Published as "Conformal Prediction: A Gentle Introduction", Found. Trends ML 16(4):494-591, 2023. **[flag]** Published title differs from arXiv title; bib uses journal title and notes arXiv title. |

## Part 1f — Classical statistics

| key | URL(s) opened | status / notes |
|---|---|---|
| massart1990dkw | https://api.crossref.org/works/10.1214/aop/1176990746 (projecteuclid page returned empty) | [V] Ann. Probab. 18(3):1269-1283, 1990. |
| hoeffding1963 | https://api.crossref.org/works/10.1080/01621459.1963.10500830 (tandfonline 403) | [V] JASA 58(301):13-30, 1963. |
| clopper1934 | https://academic.oup.com/biomet/article-lookup/doi/10.1093/biomet/26.4.404 | [V] Biometrika 26(4):404-413, Dec 1934. |
| page1954cusum | https://academic.oup.com/biomet/article-lookup/doi/10.1093/biomet/41.1-2.100 | [V] Biometrika 41(1-2):100-115, 1954. |

## Part 1g — Cognitive science / metareasoning

| key | URL(s) opened | status / notes |
|---|---|---|
| shenhav2013evc | https://api.crossref.org/works/10.1016/j.neuron.2013.07.007 (Elsevier page is a JS redirect) | [V] Neuron 79(2):217-240, 2013. |
| lieder2020resource | https://www.cambridge.org/core/product/identifier/S0140525X1900061X/type/journal_article ; Crossref | [V] BBS 43:e1, 2020. |
| russell1991metareasoning | https://api.crossref.org/works/10.1016/0004-3702(91)90015-C | [V] Artif. Intell. 49(1-3):361-395, 1991. |
| kahneman2011thinking | https://openlibrary.org/isbn/9780374275631.json | [V, partial] **[flag]** Publisher page (us.macmillan.com) returned 403 and the LoC catalog is JS-only; verified via Open Library record (Farrar, Straus and Giroux, 2011, ISBN 9780374275631). Facts are uncontroversial but not confirmed from the publisher's own page. |
| evans2013dual | https://api.crossref.org/works/10.1177/1745691612460685 (SAGE page 403) | [V] Perspect. Psychol. Sci. 8(3):223-241, 2013. |
| sloman1996two | https://api.crossref.org/works/10.1037/0033-2909.119.1.3 (PsycNet is JS-only) | [V] Psychol. Bull. 119(1):3-22, 1996. |
| stanovich2000individual | https://www.cambridge.org/core/product/identifier/S0140525X00003435/type/journal_article ; Crossref | [V] BBS 23(5):645-665, 2000. |

## Part 1h — Benchmarks

| key | URL(s) opened | status / notes |
|---|---|---|
| hendrycks2021math | https://arxiv.org/abs/2103.03874 | [V] Comments: "NeurIPS 2021" (Datasets & Benchmarks track). |
| chen2021humaneval | https://arxiv.org/abs/2107.03374 | [V] 58 authors; arXiv only. |
| rein2024gpqa | https://arxiv.org/abs/2311.12022 ; https://colmweb.org/2024/AcceptedPapers.html | [V-venue] COLM 2024 (spotlight). |
| hendrycks2021mmlu | https://arxiv.org/abs/2009.03300 | [V] Comments: "ICLR 2021". |
| zheng2023mtbench | https://arxiv.org/abs/2306.05685 | [V] Comments: "NeurIPS 2023 Datasets and Benchmarks Track". |
| suzgun2022bbh | https://arxiv.org/abs/2210.09261 | [V] arXiv only (the ACL 2023 Findings version was not checked; cite as arXiv). |
| austin2021mbpp | https://arxiv.org/abs/2108.07732 | [V] arXiv only. |
| jain2025livecodebench | https://arxiv.org/abs/2403.07974 ; https://iclr.cc/virtual/2025/poster/29033 | [V-venue] ICLR 2025. |
| wei2024simpleqa | https://arxiv.org/abs/2411.04368 | [V] arXiv only. Second author appears on arXiv as "Nguyen Karina". |
| lin2022truthfulqa | https://arxiv.org/abs/2109.07958 | [V] Comments: "ACL 2022 (main conference)". |

## Part 1i — Security

| key | URL(s) opened | status / notes |
|---|---|---|
| perez2022ignore | https://arxiv.org/abs/2211.09527 | [V] Comments: "ML Safety Workshop NeurIPS 2022". |
| greshake2023injection | https://arxiv.org/abs/2302.12173 ; https://api.crossref.org/works/10.1145/3605764.3623985 | [V] AISec '23 (16th ACM Workshop on AI and Security, at CCS 2023), pp. 79-90. |
| zou2023universal | https://arxiv.org/abs/2307.15043 | [V] arXiv only. |
| wei2023jailbroken | https://arxiv.org/abs/2307.02483 ; https://neurips.cc/virtual/2023/poster/70702 | [V-venue] NeurIPS 2023 (oral). |

## Part 1j — Faithfulness / hallucination

| key | URL(s) opened | status / notes |
|---|---|---|
| turpin2023unfaithful | https://arxiv.org/abs/2305.04388 | [V] Comments: "NeurIPS 2023". |
| lanham2023faithfulness | https://arxiv.org/abs/2307.13702 | [V] 30 authors; arXiv only. |
| ji2023hallucination | https://arxiv.org/abs/2202.03629 ; https://api.crossref.org/works/10.1145/3571730 | [V] ACM Comput. Surv. 55(12):1-38 (article 248), 2023. **[flag]** Crossref author list (10) is shorter than arXiv v7 (13); bib uses the published list. |

## Part 2 — Diffusion / NAR / parallel decoding

| key | URL(s) opened | status / notes |
|---|---|---|
| nie2025llada | https://arxiv.org/abs/2502.09992 | [V] arXiv only. **[flag]** Task brief implied no venue; search found only an "ICLR 2025 DeLTa Workshop Oral" label on OpenReview, which could not be fetched (bot wall). No ICML 2025 record found. Cite as arXiv. |
| ye2025dream | https://arxiv.org/abs/2508.15487 | [V] arXiv ID 2508.15487 confirmed; 21 Aug 2025; 8 authors, first Jiacheng Ye. |
| gong2025diffucoder | https://arxiv.org/abs/2506.20639 | [V] arXiv only. **[flag]** Citing papers list it as ICLR 2026 but the OpenReview page is bot-walled and no proceedings page was fetched; kept as arXiv. |
| arriola2025blockdiffusion | https://arxiv.org/abs/2503.09573 | [V] Comments: "ICLR 2025 Oral". |
| wu2025fastdllm | https://arxiv.org/abs/2505.22618 | [V] arXiv only. |
| wu2026fastdllmv2 | https://arxiv.org/abs/2509.26328 ; https://proceedings.iclr.cc/paper_files/paper/2026/hash/d0865cbe51d35ec322f9af9db7806fc7-Abstract-Conference.html | [V-venue] ICLR 2026. |
| inception2025mercury | https://arxiv.org/abs/2506.17298 | [V] arXiv only; author entity "Inception Labs" + 12 named authors. |
| cheng2026sdar | https://arxiv.org/abs/2510.06303 ; https://aclanthology.org/2026.findings-acl.1110/ | [V-venue] Findings of ACL 2026, pp. 22058-22075. **[flag]** ACL version has 13 authors (adds Yuhua Jiang, Yihao Liu); arXiv v3 has 11. Bib uses ACL list. |
| wang2026trado | https://arxiv.org/abs/2509.06949 ; https://proceedings.iclr.cc/paper_files/paper/2026/hash/a61023ce36d21010f1423304f8ec49af-Abstract-Conference.html | [V-venue] ICLR 2026. Paper title is "Revolutionizing Reinforcement Learning Framework for Diffusion Large Language Models"; TraDo is the model family name, TraceRL the method. |
| gu2018nat | https://arxiv.org/abs/1711.02281 | [V] Comments: "Accepted by ICLR 2018". |
| ghazvininejad2019maskpredict | https://arxiv.org/abs/1904.09324 | [V] Comments: "EMNLP 2019". |
| austin2021d3pm | https://arxiv.org/abs/2107.03006 ; https://proceedings.neurips.cc/paper/2021/hash/958c530554f78bcd8e97125b70e6973d-Abstract.html | [V-venue] NeurIPS 2021 (vol. 34). |
| lou2024sedd | https://arxiv.org/abs/2310.16834 | [V] Comments: "ICML 2024 Oral". |
| sahoo2024mdlm | https://arxiv.org/abs/2406.07524 | [V] Comments: "NeurIPS 2024". |
| santilli2023parallel | https://arxiv.org/abs/2305.10427 | [V] Comments: "Accepted at ACL 2023 main conference"; DOI 10.18653/v1/2023.acl-long.689. |
| kou2024cllms | https://arxiv.org/abs/2403.00835 | [V] Comments: ICML 2024. |
| gloeckle2024mtp | https://arxiv.org/abs/2404.19737 ; https://proceedings.mlr.press/v235/gloeckle24a.html | [V-venue] ICML 2024, PMLR 235:15706-15734. |
| li2025dlmsurvey | https://arxiv.org/abs/2508.10875 | [V] "A Survey on Diffusion Language Models", Li, Chen, Guo, Shen; v1 Aug 2025, v3 Jun 2026. |

## Part 2c — 2026 when-to-think / adaptive-budget papers (6 added)

| key | URL(s) opened | status / notes |
|---|---|---|
| lin2026planbudget | https://arxiv.org/abs/2505.16122 | [V] Comments: "accepted to the ICLR 2026". |
| guo2026thinkwhenneeded | https://arxiv.org/abs/2601.18146 | [V] Comments: "Accepted by SIGIR2026". |
| xu2026stableadaptive | https://arxiv.org/abs/2602.22556 | [V] arXiv (Feb 2026). |
| fan2026thinkinghard | https://arxiv.org/abs/2608.07968 | [V] arXiv (Aug 2026). |
| zhou2026bet | https://arxiv.org/abs/2605.11625 | [V] arXiv (May 2026). |
| zhai2026adaptive | https://arxiv.org/abs/2604.14853 | [V] arXiv (Apr 2026). |

Verified but **not added** to the .bib (cap of six): "The Shadow Price of Reasoning" (Wan et al., https://arxiv.org/abs/2606.03092, Jun 2026; search results say ICML 2026 poster, not confirmed from icml.cc); "Think Fast and Slow: Step-Level Cognitive Depth Adaptation for LLM Agents / CogRouter" (Yang et al., https://arxiv.org/abs/2602.12662, Feb 2026 — **[flag]** the arXiv author list as rendered ends with the single token "Linus" and contains "Xiang We", which looks truncated; re-check before citing); "The Zero-Step Thinking" (Tan et al., https://arxiv.org/abs/2510.19176, NeurIPS'25 Efficient Reasoning Workshop).

---

## Could not verify / left out

- **Bengio 2019 "From System 1 Deep Learning to System 2 Deep Learning"** — no paper or proceedings record exists. The NeurIPS 2019 invited-talk page (https://neurips.cc/virtual/2019/invited-talk/15488) confirms the title, speaker and that it was the Posner Lecture. Included as `@misc` only.
- **Kahneman 2011** — included, but publisher page could not be fetched (403); see flag above.
- **"Botvinick & Cohen", "Dehghani/Kool"** — skipped per the brief.
- Nothing else from the brief was dropped: every requested item resolved to a real page.

## Author-list / title discrepancies the paper should respect

1. Snell et al.: cite the ICLR 2025 title if citing the venue.
2. RouteLLM: ICLR title uses "from Preference Data".
3. Angelopoulos & Bates: journal title is "Conformal Prediction: A Gentle Introduction".
4. Reflexion: 5 authors in proceedings vs 6 on arXiv.
5. SDAR: 13 authors in ACL Findings vs 11 on arXiv.
6. Ji et al. survey: 10 authors in the ACM record vs 13 on arXiv v7.
7. Arora & Zanette: the paper is titled "Training Language Models to Reason Efficiently" (not "Learning to...").

---

## What the key papers actually found (for accurate citation)

### Test-time compute
- **snell2024scaling** — Studies two test-time axes: search against a process reward model and adaptive revision of the output distribution. Effectiveness depends strongly on prompt difficulty. A per-prompt "compute-optimal" allocation improves test-time-compute efficiency by >4x over best-of-N; in a FLOPs-matched comparison, on problems where the small model has non-trivial success rates, test-time compute can beat a 14x larger model.
- **brown2024monkeys** — Coverage (fraction of problems with at least one correct sample) grows roughly log-linearly with number of samples across four orders of magnitude (exponentiated power law). On SWE-bench Lite, DeepSeek-Coder-V2-Instruct goes from 15.9% (1 sample) to 56% (250 samples) vs 43% single-sample SOTA. Without automatic verifiers, majority voting and reward models plateau after a few hundred samples.
- **wang2023selfconsistency** — Sample multiple reasoning paths and take the majority answer. Gains over greedy CoT: GSM8K +17.9, SVAMP +11.0, AQuA +12.2, StrategyQA +6.4, ARC-challenge +3.9 points.
- **wei2022cot** — Few-shot CoT exemplars elicit multi-step reasoning in large models; PaLM 540B with 8 exemplars reaches SOTA on GSM8K, surpassing finetuned GPT-3 with a verifier. Abstract reports no percentage gains.
- **lightman2024verify** — Process supervision (step-level labels) beats outcome supervision on MATH; the process-supervised RM solves 78% of a representative MATH test subset; active learning improves data efficiency; releases PRM800K (800K step labels).
- **cobbe2021gsm8k** — Introduces GSM8K (8.5K problems). Training a verifier and sampling many candidates then selecting the top-ranked one improves accuracy and scales better with data than finetuning alone. No specific accuracies in the abstract.
- **muennighoff2025s1** — s1K: 1,000 curated reasoning examples; "budget forcing" (truncate, or append "Wait" to extend). s1-32B (Qwen2.5-32B-Instruct SFT) exceeds o1-preview on competition math by up to 27% (MATH, AIME24); budget forcing lifts AIME24 from 50% to 57% without further training.
- **deepseek2025r1 / guo2025deepseekr1nature** — Pure RL (GRPO) on DeepSeek-V3-Base without SFT yields reasoning behaviours (self-reflection, verification, "aha moment"). Nature version: AIME 2024 pass@1 rises 15.6% -> 77.9% during R1-Zero training, 86.7% with self-consistency. R1 adds cold-start SFT, rejection sampling and a second RL stage; distilled small models released. Limitations noted: no tool use, prompt sensitivity, overthinking simple questions, language mixing, reward-hacking risk.
- **qwen2025qwen3** — Single model family (0.6B-235B, dense and MoE) with a switchable thinking / non-thinking mode and a "thinking budget" that lets users trade latency against accuracy per query; 119 languages; Apache 2.0.
- **arora2025efficient** — RL objective that penalises length proportional to a single hyperparameter, producing a family of models at different efficiency levels; large inference-cost reductions with most accuracy retained on two open-weight reasoning models. No numbers in abstract.

### Self-correction
- **madaan2023selfrefine** — Same LLM generates, critiques and refines; ~20% absolute average improvement across 7 tasks with GPT-3.5/ChatGPT/GPT-4, no training. (Authors' own slides flag math/GSM8K as a weak spot where gains need oracle feedback.)
- **shinn2023reflexion** — Verbal self-reflection stored in episodic memory; 91% pass@1 on HumanEval vs 80% for GPT-4.
- **huang2024selfcorrect** — Intrinsic self-correction (no external feedback) does not improve reasoning and can degrade it. Use this to motivate external verifiers rather than self-critique.
- **kamoi2024survey** — Prompted LLMs have not been shown to self-correct except on unusually suitable tasks; self-correction works with reliable external feedback or large-scale fine-tuning.
- **chen2024overthinking** — o1-like models spend excess tokens on trivial problems; proposes outcome/process efficiency metrics and a self-training recipe that cuts overhead on GSM8K/MATH500/GPQA/AIME while preserving accuracy.

### Learning when to think (System 1 / System 2 switching)
- **zhang2025adaptthink** — "NoThinking" beats thinking on easy problems; RL with a constrained objective and importance sampling to choose mode; on DeepSeek-R1-Distill-Qwen-1.5B, -53% response length, +2.4% accuracy on three math sets.
- **fang2025thinkless** — `<short>`/`<think>` control tokens trained with Decoupled GRPO; cuts long-CoT usage by 50-90% on Minerva Algebra, MATH-500, GSM8K.
- **he2025selfroute** — Linear probe on hidden states predicts solvability pre-inference; Gradient-10K training set; accuracy comparable to reasoning mode with 30-55% fewer tokens.
- **hou2025thinkprune** — RL with a hard token limit (zero reward past limit), iteratively tightened; halves reasoning length on AIME24 for R1-Distill-Qwen-1.5B with ~2% drop.
- **yang2025deer** — Training-free early exit when a trial answer is high-confidence at reasoning transition points; across 10 benchmarks / 11 models, CoT length -19.1% to -80.1%, accuracy +0.3 to +5.0.
- **saha2025system1x** — Controller splits a planning problem into easy/hard sub-goals governed by a user hybridisation factor x; outperforms pure System-1, pure System-2 (A*-imitating) and symbolic A* on Maze and Blocksworld.
- **su2025dualformer** — One transformer trained on randomised (partially dropped) traces; on 30x30 mazes: slow mode 97.6% optimal (Searchformer 93.3%) with 45.5% fewer steps; fast mode 80% (solution-only 30%); auto mode 96.6% with 59.9% fewer steps.
- **lin2026planbudget** — Decompose query into sub-questions and allocate token budget per sub-question by estimated complexity; up to +70% accuracy, -39% tokens, +193.8% on their E3 efficiency metric; DS-Qwen-32B matches DS-LLaMA-70B efficiency.
- **guo2026thinkwhenneeded** — Pre-generation router head picks Think/Non-Think per ranking instance; e.g. +6.3% NDCG@10 with -49.5% tokens (MovieLens, Qwen3-4B).
- **xu2026stableadaptive** — Hybrid fine-tuning then adaptive RL with correctness-preserving advantage shaping and length-aware gradient regulation; up to +3.7/+3.6 points with -40.6%/-43.9% tokens on Qwen2.5-1.5B/7B.
- **fan2026thinkinghard** — When sharing one budget across many questions, reasoning models behave as greedy sequential solvers, front-load effort and ignore question value; planning prompts spread compute but do not prioritise by difficulty. Global budget allocation is a distinct, unmet capability.
- **zhou2026bet** — Budget-Efficient Thinking: cold start + GRPO with investment-cost-aware reward; learns short-solve / early-abstain ("nice fold") / persist ("hero call"); -54% reasoning tokens, up to +3.2% accuracy across 7 benchmarks and 3 base models.
- **zhai2026adaptive** — Lagrangian relaxation of a global budget into per-instance allocations, imitated by a lightweight classifier; up to +12.8% relative on MATH at matched budget; classifier >91% agreement with oracle.

### Routing and cascades
- **chen2023frugalgpt** — LLM cascade learned over API price tiers; matches GPT-4 at up to 98% lower cost, or +4% accuracy at equal cost.
- **ong2025routellm** — Routers trained from human preference data (Chatbot Arena) with augmentation; >2x cost reduction without quality loss; routers transfer to unseen strong/weak model pairs.
- **ding2024hybridllm** — Difficulty-predicting router between small (edge) and large (cloud) models with a tunable quality knob; up to 40% fewer large-model calls at unchanged quality.
- **yue2024cascades** — Weak model's answer consistency (across CoT and Program-of-Thought) as the difficulty signal; GPT-3.5-turbo -> GPT-4 cascade matches GPT-4 at ~40% of its cost on six reasoning benchmarks.
- **chow1970reject / geifman2017selective / madras2018defer / mozannar2020defer** — Reject-option and learning-to-defer foundations: Chow's error/reject trade-off; Geifman & El-Yaniv's post-hoc selective classifier with risk guarantees (e.g., 2% top-5 ImageNet error at ~60% coverage); Madras et al. learning to defer to a downstream decision-maker; Mozannar & Sontag's consistent surrogate loss for joint classifier+rejector.

### Speculative / parallel decoding
- **leviathan2023speculative** — Draft with a cheap model, verify in parallel with the target; exact target distribution; 2-3x on T5-XXL.
- **chen2023speculative** — Same idea with modified rejection sampling; 2-2.5x on Chinchilla-70B in a distributed setting.
- **cai2024medusa** — Extra decoding heads + tree attention; Medusa-1 >2.2x, Medusa-2 2.3-3.6x (abstract) / 2.3-2.8x (PMLR abstract wording).
- **li2024eagle** — Feature-level draft autoregression with one-step-ahead token input; 2.7-3.5x latency speedup on LLaMA2-Chat 70B, distribution preserved.
- **santilli2023parallel / kou2024cllms** — Jacobi/Gauss-Seidel fixed-point decoding (up to 38% speedup, ~2x with parallel hardware, no retraining); CLLMs fine-tune the model to converge in fewer Jacobi iterations, 2.4-3.4x.
- **gloeckle2024mtp** — n independent heads predict n future tokens; 13B models +12% HumanEval, +17% MBPP; up to 3x faster inference with 4-token prediction.
- **raposo2024mod / graves2016act** — Learned per-token compute allocation: MoD caps tokens per layer via top-k routing (matches baselines at equal FLOPs, up to 50% faster sampling steps); ACT lets an RNN learn how many steps to take per input.

### Diffusion language models
- **nie2025llada** — LLaDA 8B masked-diffusion LM trained from scratch (2.3T tokens per full text); comparable to self-built AR baselines, competitive with LLaMA3 8B on in-context learning; beats GPT-4o on reversal poem completion. Abstract has no benchmark numbers.
- **ye2025dream** — Dream 7B: AR-initialised, context-adaptive token-level noise rescheduling; claims strongest open diffusion LM; supports arbitrary-order generation, infilling, quality-speed trade-off. No numbers in abstract.
- **gong2025diffucoder** — 7B code dLLM trained on 130B tokens; coupled-GRPO (diffusion-native RL) improves EvalPlus by +4.4%; shows dLLMs can modulate how AR-like their decoding order is; temperature diversifies generation order.
- **arriola2025blockdiffusion** — Block-wise diffusion interpolating AR and diffusion: variable-length generation, KV caching and parallel sampling within blocks; SOTA likelihood among diffusion LMs.
- **wu2025fastdllm** — Training-free: block-wise approximate KV cache + confidence-threshold parallel decoding; up to 27.6x throughput on LLaDA/Dream with minimal accuracy loss.
- **wu2026fastdllmv2** — Converts a pretrained AR LLM into a block-diffusion model with ~1B fine-tuning tokens (vs 580B for Dream, i.e., ~500x less); two-level cache; up to 2.5x over AR decoding at matched quality.
- **cheng2026sdar** — Lightweight AR -> blockwise-diffusion conversion; larger models more robust to block size / thresholds; 30B MoE surpasses its AR counterpart on GPQA and ChemBench. (ACL version adds 1.7B-30B range and 2.3x wall-clock speedup on H200 in latency-critical regimes.)
- **wang2026trado** — TraceRL: trajectory-aware RL for dLLMs with a diffusion-based value model; TraDo-8B-Instruct +6.1% relative over Qwen2.5-7B-Instruct and +51.3% over Llama3.1-8B-Instruct on math (arXiv abstract; ICLR abstract reports +4.5% MATH500 / +6.6% LiveCodeBench-V2 instead — cite the version you quote); first long-CoT dLLM via curriculum (+18.1% relative on MATH500).
- **inception2025mercury** — Mercury Coder Mini/Small: 1109 / 737 tokens/s on H100 (Artificial Analysis), up to ~10x faster than speed-optimised frontier models at comparable quality; 2nd on Copilot Arena quality, fastest overall.
- **lou2024sedd / sahoo2024mdlm / austin2021d3pm** — Foundations: D3PM structured discrete transition matrices; SEDD score-entropy loss (beats GPT-2 at matched size, fewer NFEs); MDLM Rao-Blackwellised masked objective reducing to weighted MLM losses, with semi-AR sampling.
- **li2025dlmsurvey** — Survey organising dLLMs by relation to AR/MLMs, training (pre/post), inference (parallel decoding, caching), multimodal extensions, open challenges (efficiency, long sequences, infrastructure).

### Calibration / uncertainty (brief)
- **guo2017calibration** — Modern deep nets are miscalibrated; temperature scaling is a strong post-hoc fix.
- **kadavath2022know** — Large models are well-calibrated on MC/true-false formats; P(True) self-evaluation works and improves with multiple samples; P(IK) trained predictors generalise partially.
- **tian2023calibration** — Verbalised confidence from RLHF models (ChatGPT, GPT-4, Claude) is better calibrated than token probabilities; ~50% relative ECE reduction on TriviaQA/SciQ/TruthfulQA.
- **angelopoulos2023conformal** — Tutorial on split conformal prediction giving distribution-free marginal coverage guarantees.
- **massart1990dkw / hoeffding1963 / clopper1934 / page1954cusum** — DKW with tight constant 2; Hoeffding bound; exact binomial CI; CUSUM change detection.
