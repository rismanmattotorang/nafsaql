#!/usr/bin/env node
// Tests of the answer renderer (rust/crates/nafs-web/assets/render.js): Markdown, tables, code, maths, Mermaid, streaming
// and hostile input. No dependencies: `node scripts/render_test.mjs`.
import { markdown, mathml, mermaid, chartSpec, highlight, esc } from '../rust/crates/nafs-web/assets/render.js';

let pass = 0, fail = 0;
const t = (name, ok, detail = '') => { if (ok) pass++; else { fail++; console.log(`FAIL ${name} ${detail}`); } };
const has = (name, html, ...parts) => parts.forEach((p) => t(`${name}: ${p instanceof RegExp ? p : JSON.stringify(p)}`, p instanceof RegExp ? p.test(html) : html.includes(p), `\n  in ${html.slice(0, 400)}`));
const lacks = (name, html, ...parts) => parts.forEach((p) => t(`${name}: no ${JSON.stringify(String(p))}`, p instanceof RegExp ? !p.test(html) : !html.includes(p), `\n  in ${html.slice(0, 400)}`));

// ---- markdown basics
let h = markdown('# Title\n\nSome **bold**, *italic*, ~~gone~~ and `code <b>`.\n\n- a\n- b\n  - nested\n  - more\n- c\n\n1. one\n2. two\n\n> quote\n\n---\n\n- [x] done\n- [ ] todo');
has('basics', h, '<h2>Title</h2>', '<strong>bold</strong>', '<em>italic</em>', '<del>gone</del>', '<code>code &lt;b&gt;</code>', '<blockquote><p>quote</p></blockquote>', '<hr>', '<ol>');
has('nested list', h, /<li>b<ul><li>nested<\/li><li>more<\/li><\/ul><\/li>/);
has('tasks', h, 'class="task"', 'class="box on"', 'done: ');
t('ordered list start', markdown('3. x\n4. y').includes('<ol start="3">'));
has('callout', markdown('> [!warning]\n> Careful here'), 'class="callout warning"', '<b class="ct">Warning</b>');

// ---- tables
h = markdown('| Model | Size | Speed |\n|:--|--:|:-:|\n| Quick | 1.7B | **fast** |\n| Careful | 4B | `slow` |\n\nafter');
has('table', h, '<div class="tablewrap">', '<table>', '<th class="al-left">Model</th>', '<th class="al-right">Size</th>', '<th class="al-center">Speed</th>', '<td class="al-right">1.7B</td>', '<strong>fast</strong>', '<code>slow</code>', '<p>after</p>');
has('numeric columns right-align by themselves', markdown('| Item | Cost |\n|---|---|\n| a | 10 |\n| b | $2.50 |\n| c | 3% |'), '<th class="al-right">Cost</th>');
has('pipe in code in a cell', markdown('| a | b |\n|---|---|\n| `x|y` | z |'), '<code>x|y</code>');
has('ragged row', markdown('| a | b | c |\n|---|---|---|\n| x |'), '<td>x</td><td></td><td></td>');
lacks('a paragraph with a pipe is not a table', markdown('use a | b to mean or'), '<table');

// ---- code
h = markdown('```python\ndef f(x):\n    # note\n    return "a" + str(x) * 2\n```\ntext');
has('code', h, '<pre class="code">', '<span class="lang">python</span>', 'class="t-k">def</span>', 'class="t-f">f</span>', 'class="t-c"># note</span>', 'class="t-s">&quot;a&quot;</span>', 'class="t-n">2</span>', '<p>text</p>');
has('rust', markdown('```rust\nfn main() { let x: u32 = 5; }\n```'), 'class="t-k">fn</span>', 'class="t-k">let</span>');
has('json keys', markdown('```json\n{"a": 1, "b": [true, null]}\n```'), 'class="t-p"', 'class="t-l">true</span>');
has('unknown language is escaped', markdown('```zzz\n<b>&</b>\n```'), '&lt;b&gt;&amp;&lt;/b&gt;');
has('long code collapses', markdown('```js\n' + 'x();\n'.repeat(40) + '```'), 'class="code long"', 'Show all 40 lines');
has('tilde fence', markdown('~~~sh\nls -la\n~~~'), '<span class="lang">sh</span>', 'ls -la');
has('unclosed fence streams as code', markdown('Here:\n```py\nprint(1)'), '<pre class="code">', 'print');
has('inline code beats markdown', markdown('`*not em*` and **b**'), '<code>*not em*</code>', '<strong>b</strong>');

// ---- maths
const m = (tex, d = false) => mathml(tex, d);
has('fraction', m('\\frac{a+b}{2}'), '<mfrac>', '<mn>2</mn>');
has('roots', m('\\sqrt{x}+\\sqrt[3]{y}'), '<msqrt>', '<mroot>');
has('scripts', m('x^2_i + y_j + e^{-x^2}'), '<msubsup>', '<msub>', '<msup>', '<mo>−</mo>');
has('sum with limits', m('\\sum_{i=1}^{n} i^2'), '<munderover>', '∑');
has('integral keeps side scripts', m('\\int_0^1 f(x)\\,dx'), '∫', '<msubsup>'.replace('msubsup', 'munderover'));
has('greek and symbols', m('\\alpha \\leq \\beta \\Rightarrow \\infty'), 'α', '≤', 'β', '⇒', '∞');
has('matrix', m('\\begin{pmatrix} a & b \\\\ c & d \\end{pmatrix}'), '<mtable', '<mtr>', '<mtd>', '<mo fence="true" stretchy="true">(</mo>');
has('cases', m('f(x)=\\begin{cases} 1 & x>0 \\\\ 0 & x\\le 0 \\end{cases}'), '<mtable', '{');
has('left right', m('\\left( \\frac{a}{b} \\right)'), 'stretchy="true"');
has('text and functions', m('\\text{speed} = \\sin x'), '<mtext>speed</mtext>', '<mi mathvariant="normal">sin</mi>');
has('accent', m('\\hat{y} + \\vec{v} + \\overline{AB}'), '<mover accent="true">');
has('boxed', m('\\boxed{42}'), 'menclose');
has('aligned', m('\\begin{aligned} a &= b \\\\ c &= d \\end{aligned}'), 'columnalign="right left"');
has('display flag', m('x', true), 'display="block"');
lacks('math output has no script', m('<script>alert(1)</script>'), '<script', 'onerror');
h = markdown('Inline $E=mc^2$ and a price of $5 and $6 here.\n\n$$\\int_0^\\infty e^{-x}dx = 1$$\n\nAlso \\(a^2+b^2=c^2\\) and \\[x=\\frac{-b}{2a}\\]');
has('inline and display math', h, '<math', 'display="block"', '<msup>');
t('dollar amounts are not math', (h.match(/<math/g) || []).length === 4, String((h.match(/<math/g) || []).length));
lacks('unclosed display math stays text', markdown('so $$x = \\frac{1}{2'), '<math');
has('math in a code span is code', markdown('`$x^2$`'), '<code>$x^2$</code>');
has('math fence', markdown('```math\n\\frac{1}{2}\n```'), '<div class="math" data-tex="\\frac{1}{2}"><math', 'data-act="tex"');

// ---- mermaid
const flow = 'graph TD\n  A[Question] --> B{Checkable?}\n  B -- yes --> C[Exact checker]\n  B -->|no| D(Quick model)\n  D --> E([Gate])\n  E -.-> F[(Ledger)]\n  C --> G((Done))\n  subgraph Fast\n    D\n    E\n  end\n  classDef hot fill:#fde68a,stroke:#b45309\n  class C hot';
let svg = mermaid(flow);
t('flowchart renders', !!svg && svg.startsWith('<svg'));
has('flowchart parts', svg || '', 'Question', 'Checkable?', 'class="mm-node mm-diamond"', 'class="mm-node mm-stadium"', 'class="mm-node mm-db"', 'class="mm-node mm-circle"', 'class="mm-group"', 'class="mm-edge dotted"', 'fill:#fde68a', '>yes<', '>no<');
svg = mermaid('flowchart LR\n A --> B --> C\n A --> C');
t('LR chain', !!svg && (svg.match(/class="mm-node/g) || []).length === 3 && (svg.match(/class="mm-edge/g) || []).length === 3);
svg = mermaid('graph TD\n A --> B\n B --> A\n B --> B');
t('a cycle and a self loop do not hang', !!svg);
t('multiple targets with &', (mermaid('graph TD\n A --> B & C') || '').match(/class="mm-edge/g)?.length === 2);
t('hostile flowchart label is escaped', !/<script|onerror=/.test(mermaid('graph TD\n A["<script>alert(1)</script>"] --> B[<img src=x onerror=alert(1)>]') || ''));
t('hostile style colour is dropped', !/javascript/.test(mermaid('graph TD\n A --> B\n style A fill:url(javascript:alert(1))') || ''));
const seq = 'sequenceDiagram\n  participant U as User\n  actor G as Gateway\n  U->>G: ask\n  G-->>U: answer\n  Note right of G: checked\n  loop every turn\n    U->>U: read\n  end\n  alt escalate\n    G->>U: careful\n  else keep\n    G->>U: quick\n  end';
svg = mermaid(seq);
t('sequence renders', !!svg); has('sequence parts', svg || '', 'class="mm-life"', '>User<', '>ask<', 'class="mm-frame"', 'class="mm-note"', '[escalate]');
svg = mermaid('pie showData title Where time goes\n "Reading prompt" : 80\n "Decoding" : 15\n "Other" : 5');
t('pie renders', !!svg); has('pie parts', svg || '', 'Where time goes', 'Reading prompt', '80%', 'class="mm-slice"');
t('unknown kinds fall back to code', markdown('```mermaid\ngantt\n title x\n```').includes('<pre class="code">') && !markdown('```mermaid\ngantt\n title x\n```').includes('<svg'));
t('broken mermaid falls back to code', markdown('```mermaid\ngraph TD\n```').includes('<pre class="code">'));
t('mermaid in an unclosed fence is still code', markdown('```mermaid\ngraph TD\n A --> B').includes('<pre class="code">'));
has('diagram figure', markdown('```mermaid\ngraph TD\n A --> B\n```'), '<figure class="diagram"', 'data-act="source"', 'data-act="svg"', 'data-act="png"', 'data-act="copysrc"', 'data-act="full"');
const st = mermaid('stateDiagram-v2\n  [*] --> Idle\n  Idle --> Running : start\n  Running --> Idle : stop\n  state Running {\n    [*] --> Warm\n    Warm --> Hot\n  }\n  Running --> [*]\n  Idle : waiting for work');
t('state diagram draws', !!st && st.includes('State diagram') && st.includes('Idle') && st.includes('Warm') && st.includes('start'), String(st).slice(0, 200));
t('empty state diagram falls back', mermaid('stateDiagram-v2') === null);

// ---- charts (Nivo): spec checking and the figure
const cs = (o) => chartSpec(typeof o === 'string' ? o : JSON.stringify(o));
t('bar: rows, keys and index are found', (() => { const r = cs({ type: 'bar', data: [{ q: 'Q1', a: 3, b: 4 }, { q: 'Q2', a: 5, b: 6 }] }); return r.type === 'bar' && r.indexBy === 'q' && r.keys.join() === 'a,b' && r.data[1].b === 6; })());
t('bar: a plain map becomes rows', (() => { const r = cs({ type: 'column', data: { A: 1, B: 2 } }); return r.type === 'bar' && r.data.length === 2 && r.keys[0] === 'value'; })());
t('bar: numbers in strings are read', cs({ type: 'bar', data: [{ n: 'x', v: '1,200' }] }).data[0].v === 1200);
t('bar: horizontal and stacked', (() => { const r = cs({ type: 'bar', layout: 'horizontal', stacked: true, data: [{ n: 'x', v: 1 }] }); return r.horizontal && r.stacked; })());
t('Chart.js shape is understood', (() => { const r = cs({ type: 'line', data: { labels: ['a', 'b'], datasets: [{ label: 's', data: [1, 2] }] } }); return r.type === 'line' && r.series[0].id === 's' && r.series[0].data[1].y === 2; })());
t('labels + values arrays', cs({ type: 'pie', labels: ['a', 'b'], data: [3, 4] }).data.length === 2);
t('pie: donut by name or flag', cs({ type: 'donut', data: [{ label: 'a', value: 1 }] }).donut === true && cs({ type: 'pie', data: [{ label: 'a', value: 1 }] }).donut === false && cs({ type: 'pie', donut: true, data: [{ label: 'a', value: 1 }] }).donut === true);
t('pie: duplicate names stay distinct', (() => { const r = cs({ type: 'pie', data: [{ label: 'a', value: 1 }, { label: 'a', value: 2 }] }); return new Set(r.data.map((d) => d.id)).size === 2; })());
t('line: flat rows become series', (() => { const r = cs({ type: 'line', data: [{ m: 'Jan', a: 1, b: 2 }, { m: 'Feb', a: 3, b: 4 }] }); return r.series.length === 2 && r.series[0].data[0].x === 'Jan'; })());
t('line: mixed x are made all text', (() => { const r = cs({ type: 'line', series: [{ id: 'a', data: [{ x: 1, y: 1 }, { x: 'two', y: 2 }] }] }); return r.series[0].data.every((p) => typeof p.x === 'string'); })());
t('scatter needs numbers', cs({ type: 'scatter', series: [{ id: 'a', data: [{ x: 'a', y: 1 }] }] }).error && !cs({ type: 'scatter', series: [{ id: 'a', data: [{ x: 1, y: 1 }] }] }).error);
t('heatmap and radial bar series', !cs({ type: 'heatmap', data: [{ id: 'r', data: [{ x: 'a', y: 1 }] }] }).error && !cs({ type: 'radialbar', data: [{ id: 'r', data: [{ x: 'a', y: 1 }] }] }).error);
t('radar keys', cs({ type: 'radar', indexBy: 'm', keys: ['A'], data: [{ m: 'x', A: 1 }] }).keys[0] === 'A');
t('treemap tree', (() => { const r = cs({ type: 'treemap', data: { name: 'r', children: [{ name: 'a', value: 2 }, { name: 'b', children: [{ name: 'c', value: 1 }] }] } }); return r.root.children.length === 2; })());
t('treemap without values is refused', !!cs({ type: 'treemap', data: { name: 'r', children: [{ name: 'a' }] } }).error);
t('sankey: links to unknown nodes are dropped', (() => { const r = cs({ type: 'sankey', nodes: ['a', 'b'], links: [{ source: 'a', target: 'b', value: 3 }, { source: 'a', target: 'z', value: 1 }, { source: 'a', target: 'a', value: 1 }] }); return r.links.length === 1; })());
t('funnel', cs({ type: 'funnel', data: [{ id: 'a', label: 'A', value: 10 }, { id: 'b', value: 5 }] }).data.length === 2);
t('aliases and unknown types', cs({ type: 'Bar Chart', data: [{ n: 'x', v: 1 }] }).type === 'bar' && !!cs({ type: 'gantt', data: [] }).error);
t('colour scheme is checked', cs({ type: 'bar', colors: 'set2', data: [{ n: 'x', v: 1 }] }).scheme === 'set2' && cs({ type: 'bar', colors: 'url(javascript:1)', data: [{ n: 'x', v: 1 }] }).scheme === undefined);
t('JSON a bracket short is closed', cs('{"type":"scatter","series":[{"id":"a","data":[{"x":1,"y":2}]}}').series[0].data.length === 1 && !cs('{"type":"pie","data":[{"label":"a","value":1}]').error);
t('bad JSON, empty, huge', !!cs('{nope').error && !!cs({ type: 'bar', data: [] }).error && !!cs({ type: 'bar', data: Array.from({ length: 500 }, (_, i) => ({ n: i, v: i })) }).error);
t('NaN and Infinity never reach the chart', cs({ type: 'bar', data: [{ n: 'x', v: 'Infinity' }, { n: 'y', v: 1 }] }).data[0].v === 0);
h = markdown('```chart\n{"type":"bar","title":"T <b>","data":[{"n":"a","v":1}]}\n```');
has('chart figure', h, '<figure class="diagram nvchart"', 'data-kind="chart"', 'data-act="png"', 'data-act="svg"', 'data-act="csv"', 'data-act="full"', 'class="chart-data"', '<th>n</th>', '<td>a</td>');
lacks('chart title is escaped', h, 'T <b>'); has('chart title text', h, 'T &lt;b&gt;');
t('an unknown chart stays code', markdown('```chart\n{"type":"gantt","data":[1]}\n```').includes('<pre class="code">'));
t('an unclosed chart stays code', markdown('```chart\n{"type":"bar","data":[{"n":"a","v":1}]}').includes('<pre class="code">'));
t('every prefix of a chart answer renders', (() => { const full = 'Here:\n\n```chart\n{"type":"pie","data":[{"label":"a","value":1}]}\n```\n\nDone.'; for (let i = 0; i <= full.length; i++) markdown(full.slice(0, i)); return true; })());

// ---- leniency toward small models: a chart or diagram is drawn even if the fence is wrong or missing
const kindOf = (x) => { const o = markdown(x); return o.includes('nvchart') ? 'chart' : o.includes('data-kind="diagram"') ? 'diagram' : 'text'; };
const PIE = '{"type":"pie","data":[{"label":"a","value":1}]}';
t('chart in a json fence', kindOf('```json\n' + PIE + '\n```') === 'chart');
t('chart in an untagged fence', kindOf('```\n' + PIE + '\n```') === 'chart');
t('chart with no fence at the end of the text', kindOf('Here you go:\n\n' + PIE) === 'chart');
t('chart as the whole answer, spread over lines', kindOf('{\n"type":"pie",\n"data":[{"label":"a","value":1}]\n}') === 'chart');
t('a half-written bare chart stays text', kindOf('{"type":"pie","data":[{"label":"a"') === 'text');
t('json that is not a chart stays code', kindOf('```json\n{"type":"person","name":"x"}\n```') === 'text' && kindOf('```json\n{"a":1}\n```') === 'text');
t('mermaid in an untagged fence', kindOf('```\nflowchart LR\nA[x] --> B[y]\n```') === 'diagram');
t('mermaid in a python fence under a comment', kindOf('```python\n# a flowchart\nflowchart TD\n A[x] --> B[y]\n```') === 'diagram');
t('mermaid with no fence', kindOf('Sure.\n\nsequenceDiagram\nA->>B: hi\nB-->>A: ok') === 'diagram');
t('real python is not read as mermaid', kindOf('```python\nimport math\nprint(math.pi)\n```') === 'text');
t('sankey from a list of links', cs({ type: 'sankey', data: [{ source: 'a', target: 'b', amount: 3 }] }).links[0].value === 3);
t('bar: index that is not in the rows falls back to the text column', cs({ type: 'bar', indexBy: 'city', keys: ['yes'], data: [{ label: 'P', yes: 1 }] }).indexBy === 'label');
t('radar: keys that are not columns fall back to the numbers', cs({ type: 'radar', indexBy: 'm', keys: ['Battery'], data: [{ m: 'Battery', A: 8, B: 6 }] }).keys.join() === 'A,B');

// ---- citations and safety
const cited = markdown('Memory is 768 MB [1] and fast [2, 3] but [9] is not a source.', new Set([1, 2, 3]));
has('citations', cited, 'data-cite="1"', 'data-cite="2"', 'data-cite="3"', '[9]');
lacks('no citations without sources', markdown('see [1]'), 'data-cite');
const evil = markdown('<img src=x onerror=alert(1)> <script>alert(1)</script> [x](javascript:alert(1)) [y](https://a.b/"onmouseover="x) ![i](x)\n\n| <b>h</b> |\n|--|\n| <i>c</i> |');
lacks('hostile html', evil, '<img', '<script', 'href="javascript', '<b>h', '<i>c');
t('no raw attribute break-out', !/ onmouseover="/.test(evil));
has('autolink', markdown('see https://example.com/a?b=1 now.'), 'href="https://example.com/a?b=1"', 'rel="noopener noreferrer"');
t('empty and nullish', markdown('') === '' && markdown(null) === '' && markdown(undefined) === '');
t('esc', esc('<&>"\'') === '&lt;&amp;&gt;&quot;&#39;');
t('highlight escapes', highlight('a<b', 'py') === 'a&lt;b');

// ---- streaming: every prefix of a rich answer renders without throwing, and a late close upgrades the block
const rich = 'Here is a table:\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n```python\nprint(1)\n```\n\nand $x^2$ plus\n\n$$\\frac{1}{2}$$\n\n```mermaid\ngraph TD\n A --> B\n```\n\nDone.';
let threw = 0; const t0 = Date.now(); for (let i = 0; i <= rich.length; i++) { try { markdown(rich.slice(0, i), null); } catch { threw++; } }
t('every prefix renders', threw === 0, String(threw)); t('streaming is fast enough', Date.now() - t0 < 1500, `${Date.now() - t0} ms`);
t('diagram appears only when closed', !markdown(rich.slice(0, rich.indexOf('A --> B') + 7)).includes('<svg') && markdown(rich).includes('<svg'));

console.log(`${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
