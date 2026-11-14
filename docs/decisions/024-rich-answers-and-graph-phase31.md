# Decision 024: Draw answers as people read them, and the knowledge graph as a map (Phase 31)

Date 2026-11-14. Status: **adopted.** Code: `rust/crates/nafs-web/assets/render.js`, `app.js`, `app.css`. Tests:
`node scripts/render_test.mjs` (142 checks).

## Decisions

1. **One renderer for every answer** (`render.js`, a pure string-to-string module, no build step, no dependency). It
   draws Markdown with tables (alignment, numbers right-aligned by themselves, sticky header, copy as TSV), task lists,
   callouts (`> [!tip]`), highlighted code (copy, collapse above 24 lines), LaTeX as MathML, and Mermaid as inline SVG.
2. **Maths without a library.** LaTeX becomes MathML Core, which current browsers draw natively: fractions, roots,
   sums and integrals with limits, matrices, cases, aligned blocks, accents, `\text`, `\boxed`. Plain brackets are not
   stretched. Dollar amounts such as "$5 and $6" are not read as formulas.
3. **Diagrams without a library.** A Mermaid subset (flowchart and graph in four directions with seven node shapes,
   subgraphs, edge labels and styles; sequence diagrams with loops, alternatives and notes; pie charts) is laid out in
   layers, long edges routed through the layers they cross. Anything else, and anything that does not parse, stays a
   code block. Each diagram can show its source or be saved as SVG.
4. **Safe while streaming.** All model text is escaped; no HTML, no script, no `javascript:` link, no style from
   the model except a checked colour. An unclosed fence, formula or diagram stays text until it closes; every prefix of
   a rich answer renders without error (tested).
5. **Why not KaTeX and Mermaid.** Both are downloads that need approval, would need the page policy relaxed or vendoring,
   and would not work offline on first use. The hand-written subset covers what the model writes in practice and falls
   back to code for the rest. If a future answer needs more, vendoring them stays open.
6. **The graph is a map.** Force layout with collision and communities, curved edges, node size by mentions, colour by
   group, hover lights the neighbourhood and names the thing, a tap chooses it (the inspector shows the sentences that
   state each relation), a double tap opens what is around it, dragging moves it, scroll zooms, labels never overlap,
   legend chips by group, full-screen. A tap is handled on pointer-up because pointer capture sends the click to the SVG.

## Also in Phase 31

A command palette (Ctrl/Cmd+K, any page: pages, new chat, settings, theme, Advanced mode; arrow keys, Enter, Esc; new chat moves to Ctrl/Cmd+Shift+O), a Latest button when the chat is scrolled more than 240 px from the newest text, and a shorter welcome screen (a question, four examples, one line of help). Checked in the browser; no automated UI tests for these.

## Phase 32 addendum (2026-11-14): saving diagrams, state diagrams, copy LaTeX

- **Save SVG was unstyled.** Diagram colours come from the page style sheet, so the saved file drew black shapes. Saving now
  copies the computed styles inline, adds the namespace, size and a background. **Save PNG** draws that SVG at up to 3x on
  a canvas (checked in the browser: valid PNG, 1365x402 for a four-node flowchart, nodes and labels styled).
- **Diagram bar:** Source, Copy source, Save SVG, Save PNG, Expand (full-window view, Esc closes).
- **State diagrams** (`stateDiagram` / `-v2`: `[*]` start and end dots, labelled transitions, descriptions, composite
  states as frames, choice/fork) are read as flowcharts. Two arrows between the same pair are drawn side by side.
- **Display maths** shows a "Copy LaTeX" button on hover (always on touch screens).
- Still code blocks: class, ER, gantt, mindmap, journey and other Mermaid kinds.
- Tests: `node scripts/render_test.mjs` (148 checks).
