"""Hint variants for the chart and diagram sweep. B and C are candidates; the one kept is `visual_instruction` in
rust/crates/nafs-core/src/visual.rs (keep the two in step)."""
CHART_B = (
    "The user wants a chart. Reply with one short sentence, then a ```chart block that holds only JSON, using exactly the numbers the user gave. "
    "Never draw with text characters. Layout:\n```chart\n"
    '{"type":"bar","title":"Sales","indexBy":"label","keys":["value"],"data":[{"label":"Jan","value":120}]}\n```\n'
    "Other shapes (type: bar, line, pie, donut, radar, scatter, heatmap, treemap, funnel, sankey):\n"
    '- bar with several series: "keys":["revenue","cost"] and one value per key in each row; horizontal: "layout":"horizontal"; stacked: "stacked":true\n'
    '- line: {"type":"line","series":[{"id":"Units","data":[{"x":"Jan","y":120},{"x":"Feb","y":150}]}]}\n'
    '- pie or donut: {"type":"pie","data":[{"label":"Chrome","value":65}]}\n'
    '- radar: {"type":"radar","indexBy":"metric","keys":["A","B"],"data":[{"metric":"Speed","A":8,"B":6}]}\n'
    '- scatter: {"type":"scatter","series":[{"id":"Points","data":[{"x":150,"y":50}]}]}\n'
    '- heatmap: {"type":"heatmap","data":[{"id":"Store A","data":[{"x":"Mon","y":5}]}]}\n'
    '- treemap: {"type":"treemap","data":{"name":"all","children":[{"name":"Photos","value":40}]}}\n'
    '- funnel: {"type":"funnel","data":[{"id":"visits","label":"Visits","value":1000}]}\n'
    '- sankey: {"type":"sankey","nodes":[{"id":"Coal"},{"id":"Power"}],"links":[{"source":"Coal","target":"Power","value":30}]}'
)
CHART_C = (
    "The user wants a chart. Reply with one short sentence, then a ```chart block with only JSON, using exactly the numbers given. Never draw with text characters. "
    "Types: bar, line, pie, donut, radar, scatter, heatmap, treemap, funnel, sankey. Layout:\n```chart\n"
    '{"type":"bar","title":"Sales","indexBy":"label","keys":["value"],"data":[{"label":"Jan","value":120}]}\n```\n'
    'Line: "series":[{"id":"Units","data":[{"x":"Jan","y":120}]}]. Pie: "data":[{"label":"Chrome","value":65}].'
)
DIAGRAM_B = (
    "The user wants a diagram. Reply with one short sentence, then a ```mermaid block. Layout:\n```mermaid\n"
    "flowchart TD\n  A[Start] --> B{Ready?}\n  B -->|yes| C[Go]\n  B -->|no| A\n```\n"
    "Start with `flowchart TD` (or `flowchart LR`, `sequenceDiagram`, `stateDiagram-v2`, `pie`). Give every node a short letter id and its text in brackets; a decision in braces; text on an arrow between bars. "
    "Groups: subgraph Name ... end. Sequence: participant B as Browser, then B->>S: request and S-->>B: reply. State: [*] --> Idle, Idle --> Busy: start. Pie: pie title Time, then \"Work\" : 8. "
    "No styling, no HTML."
)
DIAGRAM_C = (
    "The user wants a diagram. Reply with one short sentence, then a ```mermaid block. Layout:\n```mermaid\n"
    "flowchart TD\n  A[Start] --> B{Ready?}\n  B -->|yes| C[Go]\n  B -->|no| A\n```\n"
    "Use `flowchart TD` (or `sequenceDiagram`, `stateDiagram-v2`, `pie`), short letter ids with the text in brackets, no styling."
)
CHART_D = (
    "The user wants a chart. Reply with one short sentence, then a ```chart block with only JSON, using exactly the numbers the user gave. Never draw with text characters. "
    "Types: bar, line, pie, donut, radar, scatter, heatmap, treemap, funnel, sankey. Layout:\n```chart\n"
    '{"type":"bar","title":"<title>","indexBy":"label","keys":["value"],"data":[{"label":"<name>","value":<number>}]}\n```\n'
    'Several series: put each name in "keys" and give every row a number for each. '
    'Line and scatter: "series":[{"id":"<name>","data":[{"x":<x>,"y":<y>}]}]. Pie, donut and funnel: "data":[{"label":"<name>","value":<number>}]. '
    'Treemap: "data":{"name":"all","children":[{"name":"<name>","value":<number>}]}. '
    'Sankey: "nodes":[{"id":"<name>"}],"links":[{"source":"<name>","target":"<name>","value":<number>}]. '
    'Heatmap: "data":[{"id":"<row>","data":[{"x":"<column>","y":<number>}]}].'
)
DIAGRAM_D = (
    "The user wants a diagram. Reply with one short sentence, then a ```mermaid block whose first line is the kind. Layout:\n```mermaid\n"
    "flowchart TD\n  A[Start] --> B{Ready?}\n  B -->|yes| C[Go]\n  B -->|no| A\n```\n"
    "Use `sequenceDiagram` when parties send messages to each other (participant C as Customer, then C->>B: request and B-->>C: reply), "
    "`stateDiagram-v2` for states ([*] --> Idle, Idle --> Busy: start), `pie` for shares, and `flowchart TD` for steps. "
    "Give flowchart nodes short letter ids with the text in brackets. No styling, no comments."
)
HINTS = {"chart": {"B": CHART_B, "C": CHART_C, "D": CHART_D}, "diagram": {"B": DIAGRAM_B, "C": DIAGRAM_C, "D": DIAGRAM_D}}
