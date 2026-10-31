# Production UX pass: plain language by default

Date 2026-10-26. Scope: `rust/crates/nafs-web/assets` (index.html, app.js, app.css), `scripts/e2e_sweep.py`,
`docs/user_guide.md`. Builds on the OpenUI pass (`docs/research/ui_ux_plan.md`).

## Problem

The Web UI spoke the research report's language to everyone: about 70 requirement ids (FR-053, NFR-007, ER-004,
…), 30 theorem and hypothesis references, about 90 model symbols (ê₁, ê₂, λ, Δc, φ), statistical method names,
and "System 1 / System 2" throughout. A simple maths answer carried its routing trace. Seven pages of equal
weight mixed the two user-facing pages (Chat, Rooms) with five research and operator consoles. On a phone the
composer controls took about 40% of the screen and the bottom bar truncated its labels.

## Principles

1. **Plain language first.** Say what something means for the person ("Checked", "Thought it through"), not how
   it works inside.
2. **One primary action per screen.** Chat is for asking; settings, technical detail and consoles stay out of the
   way until wanted.
3. **Expert detail on demand, never removed.** Everything the research needs is still there, behind Advanced
   mode (Settings → Advanced) and collapsed sections.

## What changed

- **Navigation:** Chat and Rooms; with Advanced mode, an *Insights* group of Quality (muḥāsaba), Learning
  (tazkiya), Experiments, Speed and Status. The Arabic names are subtitles. Phones show Chat, Rooms and *More*.
  Routes (`#/gate`, `#/tazkiya`, `#/eval`, `#/bench`, `#/system`) are unchanged, so links keep working.
- **Composer:** one mode control (Auto · Quick · Thorough; *Compare* in Advanced mode) and *Add a check*. The
  speed–care balance, thinking time, reasoning display, memory, theme and access key moved to **Settings**. The
  research opt-out moved to the chat's ⋯ menu as *Don't use to improve answers*.
- **Answer card:** *Quick/Thorough*, *Checked / Cross-checked / Not sure* and the time. "Not checked" is shown only
  in Advanced mode, since open questions have no exact check. A *Why?* button explains the path in one sentence
  (`whyText`, written in the UI, not the server's technical explanation). Duplicate actions removed.
- **Status pill:** *Ready*, *Quick answers only* or *Offline*, with a plain tooltip.
- **Insights pages:** each opens with the question it answers ("How reliable are the answers?"), Quality opens
  with four summary numbers, and every label, header and hint is in plain words. Operator commands sit in a
  collapsed *For operators* section.
- **Server-written text** (routing explanations, notes, warning names, milestone titles) passes through
  `plain()` in app.js: reference tags and sentences that are only about a requirement id are dropped, and
  "System 1/2", "verifier", "deliberation" and "gate" become the product's words. Known identifiers (warning
  rules, milestones, prompt sets, model servers) have fixed plain names. The server keeps its precise wording for
  the API and the research record.
- **Loading:** pages show a skeleton until their data arrives, instead of a blank view.

## Terminology map

| Internal | In the UI |
|---|---|
| System 1 / fast path / S1 | quick model, quick answer, *Quick* |
| System 2 / deliberate / S2 | thorough model, thorough answer, *Thorough* |
| Aql Gate, gate, escalate | Auto, routing, think longer, *sent to thorough* |
| verified / consensus / unverified / abstained | Checked / Cross-checked / (no badge) / Not sure |
| verifier, exact verifier | check, exact check |
| λ, price of latency | speed weight; Settings: "Balance between speed and care" |
| ê₁ / ê₂ | predicted chance the quick answer is wrong / chance after thinking |
| Δc | extra time |
| r / φ | accepts right answers / fooled by wrong answers |
| ECE | calibration error |
| CUSUM drift monitor | "Is quality changing?" |
| ledger, audit | reliability by topic, spot check |
| roadmap gates G0–G6 | Release readiness, milestones 1–7 |
| p50 / p95 / p99, TTFT | typical / slowest 5% / slowest 1%, first words |
| prefill / decode / KV-cache reuse | reading / writing speed, reused |
| Both trial | Compare |
| Inspector | Details |
| API token | access key |

## Guardrails

- `cargo test -p nafs-web`: `ui_copy_has_no_internal_references` fails on requirement ids, theorem references,
  model symbols or "System 1/2" in app.js or index.html outside comments. A line holding a pattern rather than
  copy is marked `copy-lint: allow`.
- `scripts/e2e_sweep.py` (headless Chrome) checks the rendered text of Chat and Rooms as a first-time visitor sees
  them and of every page in Advanced mode, besides the existing error and phone-width checks.
  `--ui-only` runs just the UI part (writes `benchmarks/e2e_sweep_ui.json`).
