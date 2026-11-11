<div align="center">

# Nafs-Aql

### Fast when it can. Careful when it must.

**An AI assistant that runs on four CPU cores, answers from your own documents, and tells you when it is not sure.**

Built by the [Dirac](https://dirac.id) R&D Team.

[![CI](https://github.com/rismanmattotorang/nafsaql/actions/workflows/ci.yml/badge.svg)](https://github.com/rismanmattotorang/nafsaql/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Runs on CPU](https://img.shields.io/badge/runs%20on-CPU-teal.svg)](docs/features_and_results.md)
[![Rust](https://img.shields.io/badge/built%20with-Rust-b7410e.svg)](rust/)

[The problem](#the-problem) · [The product](#the-product) · [Proof](#proof) · [Quick start](#quick-start) · [How it works](#how-it-works) · [Docs](docs/index.md)

<br>

<img src="docs/shared/screenshots/chat.png" alt="Nafs-Aql answering a question and showing how the answer was checked" width="920">

</div>

---

## The problem

Small models are fast and wrong too often. Large models are careful and need a GPU or someone else's servers.
Neither tells you which of its answers to trust.

## The product

**01 — Checked, or "Not sure".** A small model answers at once. A reasoning model steps in only when the question is worth the wait. Where an answer can be checked (maths, code against tests, data against a schema), a checker decides, not the model. Every answer is labelled *Checked*, *Cross-checked* or *Not checked*.

**02 — Your documents, with sources.** Drop in PDF, Word, PowerPoint, LaTeX, web pages or notes. Ask in plain language. Every claim points at the passage that states it, the answer says which of its sentences are *not* in those passages, and a map shows what your documents talk about.

**03 — It remembers, and shows what it remembers.** Tell it something in one chat and ask in another. It reads its notes only when your question is about you or points back ("my", "last time", "you said"), so other questions are not slowed down. Every note can be read and deleted; private chats and rooms are never remembered.

**04 — Four CPU cores.** No GPU. No API key. Nothing leaves your machine.

## Proof

Measured on one shared 4-core Intel i3-8100. Every figure comes from a stored run in [`benchmarks/`](benchmarks/).

| | |
|---|---|
| **36 %** | of the cost of always reasoning, at the same accuracy on hard questions |
| **0.15 s** | to search 847 passages by words and meaning together; the search itself takes 0.2 ms, the rest is reading the question |
| **18 of 20** | held-out questions answered with passages, against 2 of 20 without |
| **38 of 44** | answers made up without reading are flagged by the check against the passages; it flags 1 of 42 right ones, and it cannot catch a wrong number taken from the right passage |
| **10×** | as often does a follow-up question ("and how fast is it?") find its passage first, when it is read with the question before it |
| **14 of 14** | live checks of the memory pass: a fact is found from another chat by its words or its meaning, and is gone when you forget it |

The rest, including what did not work, is in [features and measured results](docs/features_and_results.md).

## Quick start

You need Rust 1.96+, Python 3.10+ and about 4 GB of disk.

```bash
git clone https://github.com/rismanmattotorang/nafsaql.git && cd nafsaql
```

```bash
make models
```

```bash
make up
```

Open **http://127.0.0.1:8080**. `make down` stops everything. Other ways to serve it: [serving and operations](docs/serving_and_operations.md).

## How it works

```mermaid
flowchart LR
  Q["Your question"] --> S1["Small model<br/>answers at once"]
  S1 --> C{"Checker"}
  C -->|"passes"| A["Answer<br/>checked"]
  C -->|"fails, or the gate is unsure"| S2["Reasoning model<br/>thinks it through"]
  S2 --> C2{"Checker or judge"}
  C2 -->|"passes"| A
  C2 -->|"nothing earned acceptance"| X["Says it is not sure"]
  K[("Your documents")] -.->|"passages and graph"| S1
```

The gate escalates only when the error it expects to save is worth the waiting time. Where a checker exists its verdict is final; agreement between samples is a hint, never proof.

## Ask your documents

<table>
<tr>
<td width="50%"><img src="docs/shared/screenshots/knowledge.png" alt="The Knowledge page: documents, passages and search"><br><sub><b>Your library.</b> Searchable by its words seconds after upload, by meaning soon after.</sub></td>
<td width="50%"><img src="docs/shared/screenshots/knowledge_chat.png" alt="A chat answer with numbered sources"><br><sub><b>Answers with sources.</b> Click a number to read the passage.</sub></td>
</tr>
<tr>
<td colspan="2"><img src="docs/shared/screenshots/knowledge_graph.png" alt="The map of a set of documents" width="100%"><br><sub><b>A map of your documents.</b> Drawn from the text the moment a file is read; a deeper pass adds relations, each with the sentence that states it.</sub></td>
</tr>
</table>

Search runs in memory, beside the store that keeps your files, so it does not slow down when the machine is busy. A typing slip in a question is matched to the nearest word your documents contain, and a follow-up such as "and how fast is it?" is searched together with the question before it. Reading a file again only re-reads what changed. Uploads are capped and checked, and deleting a library removes what was derived from it. Details: [Knowledge specification](docs/spec/knowledge_base_spec.md), its [search update](docs/spec/knowledge_sota_spec.md), and [how it compares with vLLM Semantic Router and Memorilla](docs/research/router_memory_gaps.md).

## Status

A research preview, built in the open, running end to end on a CPU today.

- **Open:** questions worded differently from the document are found in the first eight passages only a quarter of the time. The remedies need larger models than this release ships.
- **Not done:** the release-scale evaluation, the first distillation cycle and the pre-training of the memory-token module need a faster machine. The learned gate waits for its ledger: 1,294 of the 2,900 checked answers it asks for.
- **Did not work, and published:** speculative decoding, 1-bit weights, concurrent hedging, a larger embedding model, score-based fusion, a graph arm in everyday search, guessing a question's difficulty before answering it.

Current state: [docs/roadmap/STATUS.md](docs/roadmap/STATUS.md).

## Documentation

| | |
|---|---|
| [User guide](docs/user_guide.md) | the app, rooms, the lab, your documents |
| [Paper](docs/report/nafs_aql_report.pdf) | the formal model and its proofs, the architecture, every measurement |
| [Features and measured results](docs/features_and_results.md) | everything built, with evidence, and how it compares |
| [Serving](docs/serving_and_operations.md) · [Runbook](docs/operations/operator_runbook.md) · [API](docs/api/api_reference.md) | running it |
| [All documents](docs/index.md) | specifications, research notes, decisions |

## Contributing

Issues and pull requests are welcome. Run `make test`, keep `cargo fmt` and `cargo clippy -D warnings` clean, back a performance claim with an interleaved comparison, and record a judgement call in `ASSUMPTIONS.md`.

## Cite

```bibtex
@techreport{nafsaql2026,
  title       = {Nafs-Aql: A Verifier-Gated Dual-Process Architecture on Small Qwen Models for CPU-Only Deployment},
  author      = {{Dirac R\&D Team}},
  institution = {Dirac},
  year        = {2026},
  url         = {https://github.com/rismanmattotorang/nafsaql}
}
```

## License

[Apache License 2.0](LICENSE). Qwen3 model weights, llama.cpp, vLLM and the evaluation datasets keep their own licences; see [NOTICE](NOTICE).

---

<div align="center">

**About Dirac.** [Dirac](https://dirac.id) is an Indonesian deep-tech startup focused on quantum computing.

</div>
