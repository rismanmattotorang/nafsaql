# Web UI: benchmark against OpenUI and the plan applied

Date 2026-10-21. Scope: every page (Chat, Lab, Muhasaba, Tazkiya, Rooms, Performance, System), with Chat and Rooms as the chat surfaces that must feel world-class.

## What OpenUI is, and what we take from it

[OpenUI](https://github.com/thesysdev/openui) (Thesys, MIT) is a Generative UI framework: a model emits a compact streaming language (OpenUI Lang) that a renderer turns into live UI from a registered component library. Its React `react-ui` package ships `AgentInterface` (a chat surface with thread history, streaming and a model-renderable message area, conversation `starters`), `ToolCall` / `ToolResult`, `Steps`, `Callout` / `TextCallout`, `FollowUpBlock` / `FollowUpItem`, `CodeBlock`, `MarkDownRenderer`, cards, tabs, tables and charts, with a `ThemeProvider` for light and dark themes. A `browser-bundle` exists for no-build embeds (3.2 MB of script, 900 KB gzipped, loaded from a CDN).

We do **not** embed the bundle:

- the gateway's Content-Security-Policy is `script-src 'self'` and the UI is served offline from the binary; a CDN script would be a new dependency with security review (rule 5 of the completion plan) and 900 KB on first load for a page that is a few tens of kilobytes today;
- OpenUI's value is letting the *model* lay out the UI; Nafs-Aql's 1.7B and 4B models are budgeted for answering, and asking them to emit OpenUI Lang would spend tokens the latency budget does not have, on the very path we just made faster;
- the current UI is a dependency-free ES module, which keeps the Rust build, CI and the security tests simple.

What we adopt is OpenUI's **component vocabulary and interaction model**, implemented in the existing vanilla stack:

| OpenUI | Nafs-Aql equivalent (this change) |
|---|---|
| `AgentInterface` with `starters` | Chat welcome with starter cards and a "how an answer is made" strip; thread with grouped history; composer with explicit path, care and verification controls |
| `Steps` | The answer's journey (draft, flagged, deliberate, accepted or abstain) as a step strip with state icons, live stage label and elapsed time |
| `ToolCall` / `ToolResult` | Verifier runs and the gate decision as tool cards: name, status, duration, expandable evidence |
| `Callout` / `TextCallout` | Uncertainty notices, degraded mode and errors as callouts with a kind (info, warning, danger, success) |
| `FollowUpBlock` / `FollowUpItem` | Follow-up chips after every answer: think harder, fast only, explain step by step, check with a reference, compare draft and answer |
| `CodeBlock`, `MarkDownRenderer` | Existing safe Markdown with code blocks: language label, copy, wrap |
| `ThemeProvider` | Design tokens on `:root` with a dark theme; spacing and type scales made consistent |
| Cards, tabs, tables | Page intros ("what this page is for" and the primary action), consistent card headers and hints, empty states that say what to do next |

## Principles applied on every page

1. **Say what the page is for** in one line at the top, with the primary action next to it.
2. **One idea per card**, a heading that names the question the card answers, a hint that names the source of the numbers.
3. **Every number has a unit and a comparison** (target, baseline or interval).
4. **Empty states are instructions**, not apologies.
5. **Streaming is legible**: what is happening now, for how long, and what will happen next.
6. **Keyboard and screen reader**: focus rings, labels, live regions for streaming status, reduced motion respected.
7. **Phone width works**: drawers for lists, no sideways scroll, controls wrap.

## Chat and Rooms specifics

- Journey steps replace the automaton pills; the live step shows the stage label ("System 2 deliberating, attempt 1 of 3, budget 1,024 tokens") and elapsed time in an `aria-live` region.
- Tool cards for each verifier verdict and for the gate decision (collapsed by default, one line: `code_tests · passed · 0.4 s`), evidence on expand.
- Follow-up chips appear under a finished answer and are context-aware (a fast unverified math answer offers "Think harder" and "Check with a reference answer").
- Conversations grouped by day; Rooms messages grouped by author within two minutes; presence, typing and unread markers kept.
- Composer: the path control explains each option on hover, the care slider has end labels and a numeric readout, verification is a labelled attachment, shortcuts are listed under `?`.

The analogy vocabulary (nafs, lawwāma, ʿaql, muṭmaʾinna, tawaqquf) stays where it was (tooltips and the design notes); it is never the only label of a state (SR-005 packet, section 2).
