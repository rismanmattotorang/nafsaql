# Decision 025: Nivo charts in answers, and how the models are asked for them (Phase 33)

Date 2026-11-14. Status: **adopted.** Code: `rust/crates/nafs-web/assets/{render.js,app.js,app.css,nivo.js}`,
`tools/nivo-bundle/`, `rust/crates/nafs-core/src/visual.rs`. Tests: `node scripts/render_test.mjs` (197 checks),
`cargo test -p nafs-core visual`. Sweep: `scripts/visual_sweep.py`, scored by `scripts/visual_score.mjs`; raw runs in
`benchmarks/visual_sweep_*.json`.

## Decisions

1. **Real Nivo (https://nivo.rocks), vendored once.** Nivo 0.99 with React 18 and d3 is bundled by esbuild into one
   committed file (`nivo.js`, 696 kB, 218 kB gzip; pinned in `tools/nivo-bundle/package-lock.json`, rebuilt with
   `tools/nivo-bundle/build.sh`). The page loads it on first use only, so a chat without a chart never fetches it, and
   it works offline afterwards. The earlier "no library" rule (decision 024) still holds for maths and Mermaid.
2. **A `chart` block of JSON is the contract.** Ten kinds: bar (grouped, stacked, horizontal), line, area, pie, donut,
   radar, scatter, heatmap, treemap, funnel, sankey, radial bar. `chartSpec()` in `render.js` checks the JSON and puts it in
   one canonical shape without touching the DOM (limits on points, series and tree depth; colour schemes from an allow
   list; every string escaped by React). Chart.js-style `labels`/`datasets`, a plain map, a flat table, a list of
   links for a sankey, an index or keys that are not in the rows, and JSON a bracket or two short are all accepted. Anything
   else stays a code block.
3. **Lenient about the fence.** A small model forgets the tag, writes `json`, or writes no fence. A complete, valid chart
   spec or Mermaid diagram is drawn from a `chart`, `json`, untagged or `python`-under-a-comment block, or bare at the end of
   the text; a half-written one stays text while it streams.
4. **Look.** Nivo's theme is read from the page's own colours, so charts follow light and dark (and redraw when the
   theme changes); labels are skipped when crowded, long category names rotate, the legend moves under the plot on a phone,
   a heat map picks black or white text per cell. Each chart has Source, Copy source, Save CSV, Save SVG, Save PNG (3x,
   styles inlined, waits for the animation to end), Expand, and a "Show the data" table that is also the fallback when the
   library cannot load.
5. **Asking the model (`visual.rs`).** A question that asks for a chart or diagram (words checked in the question; not
   for code, Python, JSON or matplotlib questions, not when the caller pins a domain) is treated as an open question, not
   maths, and gets one instruction in place of the domain's: reply with one sentence and a `chart` or `mermaid` block,
   shown with **one example of the kind asked for** (bar, line, pie, donut, radar, scatter, heatmap, treemap, funnel,
   sankey; flowchart, sequence, state, pie) with `<placeholders>`. Switch off with `assistant.visual_hints = false`.
6. **Saved diagrams carry their styles.** Diagram colours come from the page style sheet, so a saved SVG now inlines them
   (the Phase 32 fix applies to charts too).

## Sweep of the models (quick model, Qwen3-1.7B, 12 chart and 10 diagram questions, data given in the question)

Scored on what the page would draw: a figure of the expected type with every number and label, or a diagram of the
expected kind with enough nodes and the words asked for. The machine was shared (load 6 to 16), which slows time, not
accuracy.

| Arm | Charts right (drawn) | Diagrams right (drawn) | Note |
|---|---|---|---|
| A: nothing added | 0/12 (0) | 0/10 (0) | prose, a numbered list, or the diagram as text |
| B: long, all shapes | 7/12 (12) | 5/10 (8) | the example values are copied |
| C: short, one example | 10/12 (10) | 7/10 (10) | best of the same-for-every-question wordings |
| D: C plus kinds and placeholders | 7/12 (10) | 6/10 (9) | longer text confuses the small model |
| E1: gateway, generic example (C) | 11/12 (11) | 4/10 (9) | every sequence or state question became a flowchart |
| E2: gateway, example of the kind asked for, copied values | 6/12 (12) | 7/10 (10) | copies the example's two rows |
| **E3: gateway, kind example with placeholders (adopted)** | **8/12 (11)** | **8/10 (10)** | |

Findings: (1) with no instruction a small model draws nothing; (2) it copies the shape and the size of the example, so the
example must match the kind asked for and carry placeholders, not numbers; (3) a longer instruction is worse; (4) what is
left is multi-series reasoning (it writes the quarters as series, or a diagonal table) and one JSON slip; (5) the page's
leniency about fences turned round 1's drawn rate from 0% for the exact fence into 9-10 of 12 (re-scored on the same answers:
chart B 6/12, C 9/12; diagram B 5/10, C 7/10). With 12 and 10 questions per arm, one question is 8 to 10 points: read the
arms as ordered, not as exact.

**Careful model (Qwen3-4B, thinking), 4 chart and 4 diagram questions** (bar, line, pie, sankey; tea, web, traffic, CI),
`benchmarks/visual_sweep_slow.json`: nothing added 0/8 (four diagrams were written as `python` blocks or plain text, four
charts as prose); with the gateway's instruction **8/8** right, all in the exact fence, about 100 s per answer on the shared
machine. The careful model needs the instruction as much as the quick one, and follows it better.
