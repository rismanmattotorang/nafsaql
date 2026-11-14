#!/usr/bin/env node
// Tests of the answer renderer (rust/crates/nafs-web/assets/render.js): Markdown, tables, code, maths, Mermaid, streaming
// and hostile input. No dependencies: `node scripts/render_test.mjs`.
import { markdown, mathml, mermaid, highlight, esc } from '../rust/crates/nafs-web/assets/render.js';

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
has('math fence', markdown('```math\n\\frac{1}{2}\n```'), '<div class="math"><math');

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
has('diagram figure', markdown('```mermaid\ngraph TD\n A --> B\n```'), '<figure class="diagram"', 'data-act="source"', 'data-act="svg"');

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
