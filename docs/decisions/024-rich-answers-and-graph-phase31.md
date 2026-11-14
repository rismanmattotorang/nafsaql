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

## Not done

Command palette, scroll-to-latest button and a simpler welcome screen were considered and left for a later phase.
