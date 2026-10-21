# Requirements traceability table (generated from data/requirements.csv)

| Req ID | Statement (short) | Motivating result / decision | Component | Verifying test | Phase |
|---|---|---|---|---|---|
| FR-001 | Chat rooms | D-01 product scope | nafs-gateway, nafs-web | IT-010 | P1 |
| FR-002 | Reply metadata | Thm gate; Thm plurality | nafs-core | UT-020 | P1 |
| FR-003 | Conversation context | Prop roofline | nafs-memory | UT-030 | P1 |
| FR-004 | Speed-care control | Prop frontier | nafs-core, nafs-web | UT-021 | P2 |
| FR-005 | Model profiles | D-02 roster | nafs-core, config | IT-011 | P1 |
| FR-006 | Controller events | D-03 transparency | nafs-gateway | IT-012 | P1 |
| FR-010 | Gate features | Thm gate | nafs-core | UT-040 | P2 |
| FR-011 | Derived gate rule | Thm gate | nafs-core | UT-041, ER-005 | P2 |
| FR-012 | Failed verification escalates | Draft defect 10; Thm precision | nafs-core | UT-042 | P1 |
| FR-013 | Decision record | Prop audit | nafs-core, nafs-memory | UT-043 | P1 |
| FR-014 | Budget mode | Thm gate (ii); Prop samples (i) | nafs-core | UT-044 | P2 |
| FR-015 | Gate recalibration | Prop miscal | nafs-core, ledger | ER-005 | P2 |
| FR-020 | Sequential verified sampling | Thm coverage; Lemma termination | nafs-core | UT-050 | P2 |
| FR-021 | Caps from Phase-0 fit | Prop stopping | config, nafs-eval | UT-051 | P2 |
| FR-022 | No unverified refinement | Thm selfrefine | nafs-core | UT-052 | P2 |
| FR-023 | Draft-blind deliberation | Prop collision | nafs-core | UT-053, ER-004 | P1 |
| FR-024 | Thinking budget and forcing | Prop budget | nafs-core | UT-054 | P1 |
| FR-025 | Evidence-fed retry | Thm gatedloop | nafs-core | UT-055 | P2 |
| FR-026 | Bounded loop | Lemma termination | nafs-core | UT-056 | P1 |
| FR-030 | Exact verifiers | Thm precision | nafs-verify | UT-060 | P1 |
| FR-031 | Assurance levels | Thm plurality | nafs-core | UT-061 | P1 |
| FR-032 | Verifier rates | Thm precision | nafs-verify | ER-003 | P2 |
| FR-033 | Consensus not training data | Thm plurality; Prop distill | distillation | UT-070 | P3 |
| FR-034 | Judge hard gate | Draft defect 3 (T6) | nafs-verify | UT-062 | P2 |
| FR-035 | Code sandbox | SR threat model | nafs-verify | UT-063, ST-001 | P1 |
| FR-040 | Audit stream | Prop audit | nafs-core, ledger | UT-080 | P2 |
| FR-041 | Capability table | Prop audit | ledger | UT-081 | P2 |
| FR-042 | Drift monitoring | Prop samples; draft defect 7 | ledger | UT-082 | P4 |
| FR-050 | Evaluation harness | Prop samples; Prop cascade | nafs-eval | IT-090 | P0 |
| FR-051 | Paired statistics | Prop samples (ii) | nafs-eval | UT-091 | P4 |
| FR-052 | Performance console | Prop roofline | nafs-bench, nafs-web | IT-092 | P4 |
| FR-053 | Gate dashboard | Roadmap gates | nafs-web | IT-093 | P4 |
| FR-060 | Persistent store | D-04 operations | nafs-memory | UT-031 | P1 |
| FR-061 | Retrieval memory | SR-001 | nafs-memory | UT-032 | P5 |
| FR-070 | Distillation dataset builder | Prop distill | distillation | UT-071 | P3 |
| NFR-001 | Hard latency bound | Lemma termination | nafs-core | PT-001 | P1 |
| NFR-002 | Fast-path first token | Prop roofline; measured 94 tok/s prefill | nafs-core, backends | PT-002 | P1 |
| NFR-003 | Fast-path reply latency | Measured 15.5 tok/s decode | backends | PT-003 | P1 |
| NFR-004 | Slow-path latency | Measured 6.7 tok/s decode | backends | PT-004 | P1 |
| NFR-005 | Memory budget | Prop roofline | backends, config | PT-005 | P1 |
| NFR-006 | Context cap | Prop roofline | nafs-memory, backends | UT-033 | P1 |
| NFR-007 | Concurrency | Prop roofline | nafs-gateway | PT-006 | P4 |
| NFR-010 | Backend resilience | D-04 operations | nafs-gateway | IT-013 | P2 |
| NFR-011 | Backpressure | IF-008 | nafs-gateway | IT-014 | P1 |
| NFR-012 | Pinned model identity | Draft risk: version drift | config, nafs-memory | UT-034 | P0 |
| NFR-020 | Gate calibration | Prop miscal; Prop samples | nafs-core, nafs-eval | ER-005 | P2 |
| NFR-021 | Observability | D-04 operations | nafs-gateway | IT-015 | P1 |
| NFR-022 | Access control | SR threat model | nafs-gateway | ST-002 | P1 |
| NFR-023 | Build reproducibility | D-04 operations | workspace, CI | CI-001 | P0 |
| IF-001 | OpenAPI document | D-03 interfaces | nafs-gateway | CT-001 | P1 |
| IF-002 | Chat endpoints | FR-006 | nafs-gateway | CT-002 | P1 |
| IF-003 | Reasoning mode in replies | FR-002 | nafs-gateway | CT-003 | P1 |
| IF-004 | Chat-room WebSocket | FR-001 | nafs-gateway | CT-004 | P1 |
| IF-005 | JSON Schemas | D-03 interfaces | nafs-types | CT-005 | P1 |
| IF-006 | Backend contract | D-02 stack | nafs-core (backend trait) | CT-006 | P1 |
| IF-007 | Retry policy | D-04 operations | nafs-core | UT-035 | P1 |
| IF-008 | Backend backpressure | Prop roofline | nafs-core | UT-036 | P1 |
| IF-009 | Bench and eval endpoints | FR-050, FR-052 | nafs-gateway | CT-007 | P4 |
| IF-010 | Telemetry format | FR-013 | nafs-memory | CT-008 | P1 |
| MR-001 | Pinned models | D-02 roster | config | UT-037 | P0 |
| MR-002 | Thinking toggle | D-02 stack | nafs-core | UT-038 | P1 |
| MR-003 | Serving configurations | A-004 | deployment | IT-016 | P0 |
| MR-004 | Degraded mode | NFR-010 | nafs-core | IT-017 | P2 |
| MR-005 | Profile C adoption | H7 | config | ER-011 | P0 |
| TR-001 | SFT data schema | Prop distill | distillation | UT-072 | P3 |
| TR-002 | Verified-only training data | Prop distill (i) | distillation | UT-073 | P3 |
| TR-003 | Synthetic data | Prop distill (ii) | distillation | UT-074 | P3 |
| TR-004 | Unsloth configuration | Unsloth docs | distillation | IT-094 | P3 |
| TR-005 | GPU path | A-003 | distillation | IT-094 | P3 |
| TR-006 | Checkpoints and export | NFR-012 | distillation | UT-075 | P3 |
| TR-007 | Promotion gate | Prop distill (iii) | distillation, nafs-eval | ER-008 | P3 |
| TR-008 | Replay and abstention | Prop distill (i) | distillation | UT-076 | P3 |
| SR-001 | Prompt-injection handling | Threat model | nafs-core, nafs-memory | ST-003 | P2 |
| SR-002 | Sandbox isolation | Threat model | nafs-verify | ST-001 | P1 |
| SR-003 | Content safety | FR-034 | nafs-verify | ST-004 | P2 |
| SR-004 | Privacy | Ethics | nafs-memory, distillation | ST-005 | P3 |
| SR-005 | Religious content | Sec. background rules | nafs-web, docs | RV-001 | P6 |
| SR-006 | Transport and secrets | Threat model | deployment | ST-006 | P5 |
| OR-001 | Build and CI | D-04 operations | workspace, CI | CI-001 | P0 |
| OR-002 | Packaging | NFR-012 | deployment | CI-002 | P5 |
| OR-003 | Deployment recipes | MR-003 | deployment | IT-016 | P5 |
| OR-004 | Monitoring and alerts | FR-042 | deployment | IT-018 | P5 |
| OR-005 | Rollback | TR-007 | deployment, config | IT-019 | P3 |
| OR-006 | Configuration management | NFR-012 | config | UT-039 | P1 |
| ER-001 | Phase-0 baseline size | Prop samples (i) | nafs-eval | ER-001 | P0 |
| ER-002 | Non-inferiority gates | Prop samples (ii); Prop cascade | nafs-eval | ER-002 | P2 |
| ER-003 | Verifier mutation tests | Thm precision | nafs-verify, nafs-eval | ER-003 | P2 |
| ER-004 | Coupling test | Prop collision | nafs-eval | ER-004 | P2 |
| ER-005 | Calibration test | Prop samples; draft defect 7 | nafs-eval | ER-005 | P2 |
| ER-006 | Unit coverage | Lemma termination | workspace | CI-003 | P1 |
| ER-007 | Load test protocol | NFR-007 | nafs-bench | PT-006 | P4 |
| ER-008 | Regression suite | Draft risk: version drift | nafs-eval, CI | ER-008 | P3 |
| ER-010 | Metric definitions | Eval plan | nafs-eval | UT-092 | P0 |
| ER-011 | Hybrid-model benchmark | H7 | nafs-bench | ER-011 | P0 |
| NFR-030 | Latency attribution | D-07 CPU speed goal | nafs-core, nafs-gateway, nafs-web | UT-110 | P7 |
| NFR-031 | Interleaved tuning evidence | A-044 ordering effect | nafs-bench, scripts | PT-010 | P7 |
| FR-080 | Early escalation | Thm gate; D-07 | nafs-core | UT-111 | P7 |
| FR-081 | Speculative decoding heads | D-07; Phase-5 draft-model result | scripts, serving | PT-011 | P7 |
| ER-020 | Efficiency frontier | Lab mission | nafs-eval, nafs-web | ER-020 | P7 |
| FR-082 | Adaptive reasoning exit | Thm 1; DTSR/Dynasor-CoT 2026; D-07 | nafs-core | UT-112 | P7 |
| FR-083 | Lowest-window confidence | Prop. N2; DeepConf 2025 | nafs-core | UT-120 | P8 |
| FR-084 | Posterior exit rule | Prop. N4; Frazier-Yu 2007; Adaptive-Consistency 2023 | nafs-core | UT-121 | P8 |
| FR-085 | Agreement samples on unverified drafts | Thm plurality; confidence-signal study | nafs-core | UT-122 | P8 |
| ER-021 | Confidence-signal study | Prop. N1; Fleming and Lau 2014 | scripts, benchmarks | ER-021 | P8 |
| FR-090 | Room lifecycle | Rooms review (2026-10-20) | nafs-gateway, nafs-memory | UT-130 | P9 |
| FR-091 | Room join snapshot | Rooms review (2026-10-20) | nafs-gateway | UT-131 | P9 |
| FR-092 | Room resilience | Rooms review (2026-10-20) | nafs-gateway | UT-132 | P9 |
| FR-093 | Rooms interface | Rooms review (2026-10-20) | nafs-web | UT-133 | P9 |
| ER-022 | Rooms verification | Rooms review (2026-10-20) | scripts | ER-022 | P9 |
| FR-085 | Exact-verifier exit on whole-answer tasks | Thm 1; DEER (ICLR 2026); S* (EMNLP 2025 Findings); run 553a4eff | nafs-core | UT-130 | P7 |
| FR-086 | Retry budget decay and time guard | NFR-001; S* (EMNLP 2025 Findings); run 553a4eff | nafs-core | UT-131 | P7 |
