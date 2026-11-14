// Nafs-Aql answer renderer (Phase 31). Pure string-to-string, no DOM, no network, no build step, so it runs in the
// browser and under Node (scripts/render_test.mjs). It turns what a model writes into safe HTML:
//   - Markdown: headings, paragraphs, nested lists, task lists, quotes, rules, links, emphasis, strikethrough, tables
//     with column alignment, fenced code with light syntax highlighting;
//   - maths: LaTeX in $...$, $$...$$, \(...\), \[...\] typeset as MathML (fractions, roots, scripts, big operators
//     with limits, matrices, accents, Greek and the common symbols) rather than approximated in plain text;
//   - diagrams: Mermaid flowcharts, sequence diagrams and pie charts drawn as inline SVG; other Mermaid kinds are
//     shown as code, never lost;
//   - citations: [1] and [2, 3] become source buttons when the caller says which numbers are sources.
// Everything that comes from the model is escaped. Unclosed fences, formulas and diagrams (an answer still streaming)
// degrade to plain text and are drawn properly once they close.

export const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

// ============================== maths: LaTeX subset to MathML ==============================
const GREEK = { alpha: 'α', beta: 'β', gamma: 'γ', delta: 'δ', epsilon: 'ϵ', varepsilon: 'ε', zeta: 'ζ', eta: 'η', theta: 'θ', vartheta: 'ϑ', iota: 'ι', kappa: 'κ', lambda: '\u03bb', mu: 'μ', nu: 'ν', xi: 'ξ', pi: 'π', varpi: 'ϖ', rho: 'ρ', varrho: 'ϱ', sigma: 'σ', varsigma: 'ς', tau: 'τ', upsilon: 'υ', phi: 'ϕ', varphi: 'φ', chi: 'χ', psi: 'ψ', omega: 'ω', Gamma: 'Γ', Delta: 'Δ', Theta: 'Θ', Lambda: 'Λ', Xi: 'Ξ', Pi: 'Π', Sigma: 'Σ', Upsilon: 'Υ', Phi: 'Φ', Psi: 'Ψ', Omega: 'Ω' };
const SYMBOL = { times: '×', cdot: '⋅', div: '÷', pm: '±', mp: '∓', ast: '∗', star: '⋆', circ: '∘', bullet: '•', le: '≤', leq: '≤', ge: '≥', geq: '≥', ne: '≠', neq: '≠', approx: '≈', sim: '∼', simeq: '≃', equiv: '≡', cong: '≅', propto: '∝', ll: '≪', gg: '≫', infty: '∞', partial: '∂', nabla: '∇', forall: '∀', exists: '∃', nexists: '∄', in: '∈', notin: '∉', ni: '∋', subset: '⊂', supset: '⊃', subseteq: '⊆', supseteq: '⊇', cup: '∪', cap: '∩', emptyset: '∅', varnothing: '∅', setminus: '∖', land: '∧', wedge: '∧', lor: '∨', vee: '∨', neg: '¬', lnot: '¬', to: '→', rightarrow: '→', leftarrow: '←', gets: '←', leftrightarrow: '↔', Rightarrow: '⇒', Leftarrow: '⇐', Leftrightarrow: '⇔', implies: '⟹', iff: '⟺', mapsto: '↦', uparrow: '↑', downarrow: '↓', ldots: '…', dots: '…', cdots: '⋯', vdots: '⋮', ddots: '⋱', degree: '°', prime: '′', angle: '∠', perp: '⊥', parallel: '∥', mid: '∣', therefore: '∴', because: '∵', hbar: 'ℏ', ell: 'ℓ', Re: 'ℜ', Im: 'ℑ', aleph: 'ℵ', langle: '⟨', rangle: '⟩', lfloor: '⌊', rfloor: '⌋', lceil: '⌈', rceil: '⌉', quad: ' ', qquad: ' ', '%': '%', '$': '$', '&': '&', '#': '#', '_': '_', '{': '{', '}': '}' };
const BIGOP = { sum: '∑', prod: '∏', coprod: '∐', int: '∫', iint: '∬', iiint: '∭', oint: '∮', bigcup: '⋃', bigcap: '⋂', bigoplus: '⨁', bigotimes: '⨂', lim: 'lim', max: 'max', min: 'min', sup: 'sup', inf: 'inf', limsup: 'lim sup', liminf: 'lim inf', argmax: 'arg max', argmin: 'arg min' };
const FUNCS = new Set(['sin', 'cos', 'tan', 'cot', 'sec', 'csc', 'arcsin', 'arccos', 'arctan', 'sinh', 'cosh', 'tanh', 'log', 'ln', 'lg', 'exp', 'det', 'dim', 'gcd', 'lcm', 'ker', 'deg', 'arg', 'Pr', 'mod', 'bmod', 'tr', 'rank', 'softmax', 'sign']);
const ACCENT = { hat: '^', widehat: '^', bar: '¯', overline: '¯', vec: '→', tilde: '~', widetilde: '~', dot: '˙', ddot: '¨', check: 'ˇ', acute: '´', grave: '`', breve: '˘' };
const OPEN = { '(': '(', '[': '[', '\\{': '{', '\\langle': '⟨', '\\lfloor': '⌊', '\\lceil': '⌈', '|': '|', '\\|': '‖', '.': '' };
const CLOSE = { ')': ')', ']': ']', '\\}': '}', '\\rangle': '⟩', '\\rfloor': '⌋', '\\rceil': '⌉', '|': '|', '\\|': '‖', '.': '' };
const ENV = { matrix: ['', ''], pmatrix: ['(', ')'], bmatrix: ['[', ']'], Bmatrix: ['{', '}'], vmatrix: ['|', '|'], Vmatrix: ['‖', '‖'], cases: ['{', ''], array: ['', ''], aligned: ['', ''], align: ['', ''], gathered: ['', ''], split: ['', ''] };

function tokenise(src) {
  const out = []; let i = 0;
  const push = (tok, from, to) => { tok.s = from; tok.e = to; out.push(tok); };
  while (i < src.length) {
    const c = src[i]; const from = i;
    if (c === '\\') {
      const m = /^\\([a-zA-Z]+)\*?/.exec(src.slice(i));
      if (m) { i += m[0].length; push({ t: 'cmd', v: m[1] }, from, i); } else { i += 2; push({ t: 'cmd', v: src[from + 1] ?? '' }, from, i); }
    } else if (c === '{' || c === '}' || c === '^' || c === '_' || c === '&') { i++; push({ t: c }, from, i); }
    else if (/\s/.test(c)) { i++; }
    else if (/\d/.test(c)) { const m = /^\d+(?:\.\d+)?/.exec(src.slice(i)); i += m[0].length; push({ t: 'num', v: m[0] }, from, i); }
    else if (/\p{L}/u.test(c)) { i++; push({ t: 'id', v: c }, from, i); }
    else { i++; push({ t: 'op', v: c }, from, i); }
  }
  return out;
}

class TeX {
  constructor(src) { this.src = src; this.k = tokenise(src); this.p = 0; }
  peek() { return this.k[this.p]; }
  next() { return this.k[this.p++]; }
  // A group `{...}` or a single token, as an <mrow> or one element.
  arg() {
    const t = this.peek(); if (!t) return '<mrow></mrow>';
    if (t.t === '{') { this.next(); const r = this.row((x) => x.t === '}'); if (this.peek()?.t === '}') this.next(); return `<mrow>${r}</mrow>`; }
    return this.atom() || '<mrow></mrow>';
  }
  rawArg() { // the source text inside `{...}` (spaces kept), for \text and environment names
    const t = this.peek(); if (!t || t.t !== '{') return this.next()?.v ?? '';
    this.next(); let depth = 1; const from = t.e; let to = from;
    while (this.p < this.k.length) { const x = this.next(); if (x.t === '{') depth++; else if (x.t === '}') { if (--depth === 0) { to = x.s; break; } } to = x.e; }
    return this.src.slice(from, to).replace(/\\([a-zA-Z]+)/g, (m, w) => SYMBOL[w] ?? GREEK[w] ?? w).replace(/\s+/g, ' ').trim();
  }
  row(stop) {
    let out = '';
    while (this.p < this.k.length && !(stop && stop(this.peek()))) {
      const t = this.peek();
      if (t.t === '}') { this.next(); continue; }
      let base = this.atom(); if (base == null) continue;
      // Scripts, in either order. A big operator keeps them as limits.
      let sub = null, sup = null;
      for (let n = 0; n < 2; n++) {
        if (this.peek()?.t === '_' && sub == null) { this.next(); sub = this.arg(); } else if (this.peek()?.t === '^' && sup == null) { this.next(); sup = this.arg(); } else break;
      }
      if (sub != null || sup != null) {
        const big = base.startsWith('<mo largeop') || base.startsWith('<mo data-big');
        if (sub != null && sup != null) base = `<${big ? 'munderover' : 'msubsup'}>${base}${sub}${sup}</${big ? 'munderover' : 'msubsup'}>`;
        else if (sub != null) base = `<${big ? 'munder' : 'msub'}>${base}${sub}</${big ? 'munder' : 'msub'}>`;
        else base = `<${big ? 'mover' : 'msup'}>${base}${sup}</${big ? 'mover' : 'msup'}>`;
      }
      out += base;
    }
    return out;
  }
  atom() {
    const t = this.next(); if (!t) return null;
    switch (t.t) {
      case 'num': return `<mn>${esc(t.v)}</mn>`;
      case 'id': return `<mi>${esc(t.v)}</mi>`;
      case 'op': return `<mo${/[()[\]|]/.test(t.v) ? ' stretchy="false"' : ''}>${esc(t.v === '-' ? '−' : t.v === '*' ? '∗' : t.v)}</mo>`;
      case '{': { const r = this.row((x) => x.t === '}'); if (this.peek()?.t === '}') this.next(); return `<mrow>${r}</mrow>`; }
      case '^': case '_': { this.arg(); return null; }
      case '&': return '<mo data-col="1"></mo>';
      case 'cmd': return this.command(t.v);
      default: return null;
    }
  }
  command(c) {
    if (GREEK[c]) return `<mi>${GREEK[c]}</mi>`;
    if (BIGOP[c]) { const named = /^[a-z ]+$/.test(BIGOP[c]); return named ? `<mo data-big="1" movablelimits="true">${BIGOP[c]}</mo>` : `<mo largeop="true" data-big="1" movablelimits="${c === 'int' || c === 'iint' || c === 'iiint' || c === 'oint' ? 'false' : 'true'}">${BIGOP[c]}</mo>`; }
    if (FUNCS.has(c)) return `<mi mathvariant="normal">${c}</mi><mo>&#x2061;</mo>`;
    if (SYMBOL[c] != null) { const s = SYMBOL[c]; return s === ' ' ? '<mspace width="1em"></mspace>' : /[A-Za-z0-9%$&#_{}]/.test(s) ? `<mi>${esc(s)}</mi>` : `<mo>${esc(s)}</mo>`; }
    switch (c) {
      case 'frac': case 'dfrac': case 'tfrac': { const a = this.arg(), b = this.arg(); return `<mfrac>${a}${b}</mfrac>`; }
      case 'binom': { const a = this.arg(), b = this.arg(); return `<mrow><mo>(</mo><mfrac linethickness="0">${a}${b}</mfrac><mo>)</mo></mrow>`; }
      case 'sqrt': { let idx = null; if (this.peek()?.t === 'op' && this.peek().v === '[') { this.next(); idx = this.row((x) => x.t === 'op' && x.v === ']'); this.next(); } const a = this.arg(); return idx != null ? `<mroot>${a}<mrow>${idx}</mrow></mroot>` : `<msqrt>${a}</msqrt>`; }
      case 'text': case 'mathrm': case 'textrm': case 'operatorname': case 'mbox': case 'mathit': case 'textit': case 'textbf': case 'mathbf': case 'mathsf': case 'mathtt': case 'mathcal': case 'mathbb': case 'mathfrak': case 'boldsymbol': case 'bm': {
        const raw = this.rawArg(); const text = ['text', 'textrm', 'mbox', 'textit', 'textbf'].includes(c);
        const variant = { mathbf: 'bold', textbf: 'bold', boldsymbol: 'bold', bm: 'bold', mathit: 'italic', textit: 'italic', mathsf: 'sans-serif', mathtt: 'monospace', mathcal: 'script', mathbb: 'double-struck', mathfrak: 'fraktur' }[c];
        if (text) return `<mtext${variant ? ` mathvariant="${variant}"` : ''}>${esc(raw)}</mtext>`;
        return `<mi mathvariant="${variant || 'normal'}">${esc(raw)}</mi>`;
      }
      case 'overbrace': case 'underbrace': { const a = this.arg(); return `<${c === 'overbrace' ? 'mover' : 'munder'}>${a}<mo>${c === 'overbrace' ? '⏞' : '⏟'}</mo></${c === 'overbrace' ? 'mover' : 'munder'}>`; }
      case 'overset': case 'stackrel': { const top = this.arg(), base = this.arg(); return `<mover>${base}${top}</mover>`; }
      case 'underset': { const bot = this.arg(), base = this.arg(); return `<munder>${base}${bot}</munder>`; }
      case 'boxed': { const a = this.arg(); return `<menclose notation="box"><mrow>${a}</mrow></menclose>`; }
      case 'cancel': { const a = this.arg(); return `<menclose notation="updiagonalstrike">${a}</menclose>`; }
      case 'left': { const d = this.delim(); const r = this.row((x) => x.t === 'cmd' && x.v === 'right'); let close = ''; if (this.peek()?.t === 'cmd' && this.peek().v === 'right') { this.next(); close = this.delim(true); } return `<mrow><mo fence="true" stretchy="true">${esc(OPEN[d] ?? d)}</mo>${r}<mo fence="true" stretchy="true">${esc(CLOSE[close] ?? close)}</mo></mrow>`; }
      case 'right': return null;
      case 'big': case 'Big': case 'bigg': case 'Bigg': case 'bigl': case 'bigr': case 'Bigl': case 'Bigr': { const d = this.delim(); return `<mo stretchy="true">${esc(OPEN[d] ?? CLOSE[d] ?? d)}</mo>`; }
      case 'begin': return this.environment(this.rawArg());
      case 'end': this.rawArg(); return null;
      case ',': case ';': case ':': case ' ': return '<mspace width="0.25em"></mspace>';
      case '!': return '<mspace width="-0.15em"></mspace>';
      case '\\': return '<mo data-row="1"></mo>';
      case 'limits': case 'nolimits': case 'displaystyle': case 'textstyle': case 'scriptstyle': case 'nonumber': case 'notag': case 'label': return null;
      case 'not': { const a = this.atom(); return `<mrow>${a}<mo≯</mo></mrow>`; }
      case 'ce': return `<mtext>${esc(this.rawArg())}</mtext>`;
      case 'mathop': return this.arg();
      case 'tag': { return `<mtext>(${esc(this.rawArg())})</mtext>`; }
      case 'color': case 'textcolor': { this.rawArg(); return c === 'textcolor' ? this.arg() : null; }
      default:
        if (ACCENT[c]) { const a = this.arg(); const wide = /^wide|overline/.test(c); return `<mover accent="true">${a}<mo${wide ? ' stretchy="true"' : ''}>${ACCENT[c]}</mo></mover>`; }
        return `<mi mathvariant="normal">${esc(c)}</mi>`;
    }
  }
  delim(forClose = false) { const t = this.next(); if (!t) return ''; if (t.t === 'cmd') return '\\' + t.v; if (t.t === 'op' || t.t === 'id') return t.v; if (t.t === '{' || t.t === '}') return forClose ? ')' : '('; return ''; }
  environment(name) {
    const [open, close] = ENV[name.replace('*', '')] ?? ['', ''];
    if (name === 'array' && this.peek()?.t === '{') this.rawArg(); // column spec
    const rows = [[[]]]; let cur = '';
    const flush = () => { rows[rows.length - 1][rows[rows.length - 1].length - 1] = cur; cur = ''; };
    while (this.p < this.k.length) {
      const t = this.peek();
      if (t.t === 'cmd' && t.v === 'end') { this.next(); this.rawArg(); break; }
      if (t.t === '&') { this.next(); flush(); rows[rows.length - 1].push(''); continue; }
      if (t.t === 'cmd' && t.v === '\\') { this.next(); flush(); rows.push(['']); continue; }
      const before = this.p; const one = this.row((x) => x.t === '&' || (x.t === 'cmd' && (x.v === '\\' || x.v === 'end')));
      cur += one; if (this.p === before) this.next();
    }
    flush();
    const align = name === 'cases' ? 'left' : name === 'aligned' || name === 'align' || name === 'split' ? 'alternating' : 'center';
    const table = `<mtable columnalign="${align === 'alternating' ? 'right left' : align}" rowspacing="0.4em" columnspacing="${align === 'alternating' ? '0' : '1em'}">${rows.filter((r) => r.some((c) => c)).map((r) => `<mtr>${r.map((c) => `<mtd><mrow>${c}</mrow></mtd>`).join('')}</mtr>`).join('')}</mtable>`;
    if (!open && !close) return table;
    return `<mrow><mo fence="true" stretchy="true">${esc(open)}</mo>${table}<mo fence="true" stretchy="true">${esc(close)}</mo></mrow>`;
  }
}

/// LaTeX to a MathML element. `display` centres it on its own line.
export function mathml(tex, display = false) {
  try {
    const p = new TeX(tex); const body = p.row(null);
    return `<math xmlns="http://www.w3.org/1998/Math/MathML"${display ? ' display="block"' : ''}>${body}</math>`;
  } catch { return `<code class="m">${esc(tex)}</code>`; }
}

// ============================== code: light syntax highlighting ==============================
const KW = {
  py: 'and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield match case',
  js: 'async await break case catch class const continue debugger default delete do else export extends finally for from function if import in instanceof let new of return static super switch this throw try typeof var void while with yield interface type enum implements readonly public private protected namespace',
  rs: 'as async await break const continue crate dyn else enum extern fn for if impl in let loop match mod move mut pub ref return self Self static struct super trait type unsafe use where while',
  go: 'break case chan const continue default defer else fallthrough for func go goto if import interface map package range return select struct switch type var',
  java: 'abstract assert break case catch class const continue default do else enum extends final finally for if implements import instanceof interface native new package private protected public return static super switch synchronized this throw throws try void volatile while',
  c: 'auto break case const continue default do else enum extern for goto if inline register return sizeof static struct switch typedef union volatile while class namespace template typename using virtual public private protected new delete',
  sh: 'if then else elif fi for while do done case esac in function return export local echo cd exit set unset source alias',
  sql: 'select from where group by order having insert into values update set delete create table alter drop index join inner left right outer on as and or not null is in like limit offset union distinct case when then else end primary key foreign references default',
  json: '', yaml: '', toml: '', html: '', css: '',
};
const LIT = new Set('True False None true false null nil undefined NaN Infinity self this'.split(' '));
const ALIAS = { python: 'py', python3: 'py', javascript: 'js', typescript: 'js', ts: 'js', tsx: 'js', jsx: 'js', mjs: 'js', node: 'js', rust: 'rs', golang: 'go', bash: 'sh', shell: 'sh', zsh: 'sh', console: 'sh', 'c++': 'c', cpp: 'c', cc: 'c', h: 'c', hpp: 'c', cs: 'java', kotlin: 'java', swift: 'java', scala: 'java', yml: 'yaml', jsonc: 'json', postgres: 'sql', sqlite: 'sql', mysql: 'sql' };
const COMMENT = { py: '#', sh: '#', yaml: '#', toml: '#', sql: '--', js: '//', rs: '//', go: '//', java: '//', c: '//', css: null };

/// Highlight one code block. Returns HTML of spans; unknown languages are returned escaped.
export function highlight(code, lang = '') {
  const l = ALIAS[lang.toLowerCase()] || lang.toLowerCase();
  if (!(l in KW) && !COMMENT[l]) return esc(code);
  const kw = new Set((KW[l] || '').split(' ').filter(Boolean)); const cm = COMMENT[l];
  const sqlish = l === 'sql';
  let out = ''; let i = 0; const n = code.length;
  const span = (cls, s) => `<span class="t-${cls}">${esc(s)}</span>`;
  while (i < n) {
    const c = code[i]; const rest = code.slice(i, i + 3);
    if (l !== 'json' && (c === '/' && code[i + 1] === '*')) { const j = code.indexOf('*/', i + 2); const e = j < 0 ? n : j + 2; out += span('c', code.slice(i, e)); i = e; continue; }
    if (cm && code.startsWith(cm, i) && !(cm === '#' && l === 'sh' && code[i - 1] === '$')) { const j = code.indexOf('\n', i); const e = j < 0 ? n : j; out += span('c', code.slice(i, e)); i = e; continue; }
    if (c === '"' || c === "'" || c === '`') {
      const triple = (c === '"' || c === "'") && rest === c.repeat(3) && l === 'py';
      const q = triple ? c.repeat(3) : c; let j = i + q.length;
      while (j < n && !code.startsWith(q, j)) { if (code[j] === '\\') j++; else if (!triple && c !== '`' && code[j] === '\n') break; j++; }
      const e = Math.min(n, j + q.length); out += span('s', code.slice(i, e)); i = e; continue;
    }
    if (/\d/.test(c) && !/[\w$]/.test(code[i - 1] || '')) { const m = /^(0x[\da-fA-F_]+|\d[\d_]*(\.\d+)?([eE][+-]?\d+)?)/.exec(code.slice(i)); out += span('n', m[0]); i += m[0].length; continue; }
    if (/[A-Za-z_$]/.test(c)) {
      const m = /^[A-Za-z_$][\w$]*/.exec(code.slice(i)); const w = m[0]; const nxt = code.slice(i + w.length).match(/^\s*(.)/)?.[1];
      const cls = (sqlish ? kw.has(w.toLowerCase()) : kw.has(w)) ? 'k' : LIT.has(w) ? 'l' : /^[A-Z][A-Za-z0-9]*$/.test(w) && w.length > 1 && l !== 'sh' ? 'y' : nxt === '(' ? 'f' : null;
      out += cls ? span(cls, w) : esc(w); i += w.length; continue;
    }
    if (l === 'json' && c === ':') { out += esc(c); i++; continue; }
    out += esc(c); i++;
  }
  if (l === 'json') out = out.replace(/(<span class="t-s">&quot;[^<]*&quot;<\/span>)(\s*:)/g, '<span class="t-p">$1</span>$2');
  return out;
}

// ============================== diagrams: Mermaid subset to SVG ==============================
const NODE_W_MIN = 64, NODE_H = 38, PAD_X = 22, GAP_X = 36, GAP_Y = 52;
const textW = (s, size = 13) => Math.ceil([...s].reduce((a, ch) => a + (/[A-Z0-9MW@%]/.test(ch) ? 0.68 : /[ilj.,:;|!' ]/.test(ch) ? 0.3 : /[\u0100-\uffff]/.test(ch) ? 1 : 0.55), 0) * size);
const cleanLabel = (s) => String(s ?? '').replace(/<br\s*\/?>/gi, '\n').replace(/^["'`]|["'`]$/g, '').replace(/<[^>]+>/g, '').trim();

// ---- flowchart ----
function parseFlow(lines) {
  const head = lines[0].trim().match(/^(?:graph|flowchart)\s*(TB|TD|BT|LR|RL)?/i);
  const dir = (head?.[1] || 'TD').toUpperCase().replace('TB', 'TD');
  const nodes = new Map(); const edges = []; const classes = new Map(); const styles = new Map(); const groups = [];
  const stack = [];
  const SHAPES = [['\\(\\(', '\\)\\)', 'circle'], ['\\(\\[', '\\]\\)', 'stadium'], ['\\[\\[', '\\]\\]', 'sub'], ['\\[\\(', '\\)\\]', 'db'], ['\\{\\{', '\\}\\}', 'hex'], ['\\[/', '/\\]', 'para'], ['\\[\\\\', '\\\\\\]', 'para'], ['\\(', '\\)', 'round'], ['\\[', '\\]', 'rect'], ['\\{', '\\}', 'diamond'], ['>', '\\]', 'flag']];
  const nodeRe = (() => { const alts = SHAPES.map(([o, c]) => `${o}\\s*("[^"]*"|[^\\]\\)\\}]*?)\\s*${c}`); return new RegExp(`^([A-Za-z_][\\w.-]*)\\s*(?:${alts.join('|')})?`); })();
  const shapeOf = (m) => { for (let i = 0; i < SHAPES.length; i++) if (m[i + 2] !== undefined) return [SHAPES[i][2], m[i + 2]]; return [null, null]; };
  const addNode = (id, shape, label) => { let n = nodes.get(id); if (!n) { n = { id, shape: 'rect', label: id, group: null }; nodes.set(id, n); } if (stack.length) n.group = stack[stack.length - 1]; if (shape) { n.shape = shape; n.label = cleanLabel(label); } return n; };
  const parseNodes = (s) => { // `A --> B & C`: a chain of node groups joined by arrows
    const ids = []; let rest = s.trim();
    while (rest) {
      const m = nodeRe.exec(rest); if (!m) break;
      const [shape, label] = shapeOf(m); addNode(m[1], shape, label); ids.push(m[1]); rest = rest.slice(m[0].length).trim();
      if (rest.startsWith('&')) { rest = rest.slice(1).trim(); continue; } break;
    }
    return { ids, rest };
  };
  const ARROW = /^(<?)(-{2,}|={2,}|-\.+-?|-\.-)(>?|x|o)?(?:\|([^|]*)\|)?\s*/;
  for (const raw of lines.slice(1)) {
    let line = raw.replace(/%%.*$/, '').trim(); if (!line || /^direction\b/i.test(line)) continue;
    let cm = line.match(/^subgraph\s+([^\s\[]+)(?:\s*\[(.*)\])?\s*(.*)$/i);
    if (cm) { const id = cm[1]; const g = { id, label: cleanLabel(cm[2] ?? cm[3] ?? id) || id, parent: stack[stack.length - 1] ?? null }; groups.push(g); stack.push(id); continue; }
    if (/^end$/i.test(line)) { stack.pop(); continue; }
    cm = line.match(/^classDef\s+(\S+)\s+(.*)$/i); if (cm) { classes.set(cm[1], cm[2]); continue; }
    cm = line.match(/^class\s+([\w,.-]+)\s+(\S+)/i); if (cm) { for (const id of cm[1].split(',')) addNode(id.trim()).cls = cm[2]; continue; }
    cm = line.match(/^style\s+(\S+)\s+(.*)$/i); if (cm) { styles.set(cm[1], cm[2]); continue; }
    if (/^(click|linkStyle|accTitle|accDescr)\b/i.test(line)) continue;
    let { ids: from, rest } = parseNodes(line); if (!from.length) continue;
    let cur = from;
    while (rest) {
      const am = ARROW.exec(rest); if (!am) break;
      let label = am[4]; rest = rest.slice(am[0].length); let headCh = am[3];
      // `A -- text --> B` and `A == text ==> B`: the text sits between two arrow halves.
      if (label == null && am[2].length === 2 && !headCh && /^(--|==)$/.test(am[2])) {
        const tm = /^([^-=>|\[\]{}()]+?)\s*(-{2,}|={2,})(>?)\s*/.exec(rest);
        if (tm) { label = tm[1]; headCh = tm[3]; rest = rest.slice(tm[0].length); }
      }
      const { ids: to, rest: after } = parseNodes(rest); if (!to.length) break;
      const dotted = /\./.test(am[2]), thick = am[2].startsWith('=');
      const head = headCh === '>' ? 'arrow' : headCh === 'x' ? 'cross' : headCh === 'o' ? 'dot' : 'none'; const tail = am[1] === '<' ? 'arrow' : 'none';
      for (const a of cur) for (const b of to) edges.push({ from: a, to: b, label: label ? cleanLabel(label) : '', dotted, thick, head, tail });
      cur = to; rest = after;
    }
  }
  return { dir, nodes: [...nodes.values()], edges, classes, styles, groups };
}

function layoutFlow(g) {
  const horiz = g.dir === 'LR' || g.dir === 'RL';
  const sz = new Map();
  for (const n of g.nodes) {
    const lines = n.label.split('\n'); const w = Math.max(...lines.map((l) => textW(l))) + PAD_X * 2; const h = Math.max(NODE_H, lines.length * 18 + 20);
    const extra = n.shape === 'diamond' ? [1.5, 1.7] : n.shape === 'hex' ? [1.25, 1] : n.shape === 'circle' ? [1.1, 1.1] : [1, 1];
    let W = Math.max(NODE_W_MIN, w * extra[0]), H = h * extra[1]; if (n.shape === 'circle') { W = H = Math.max(W, H); } if (n.dot) { W = H = n.dot === 'end' ? 24 : 18; } if (n.shape === 'db') H += 16;
    sz.set(n.id, { w: W, h: H, lines });
  }
  // Layers by longest path, ignoring edges that would close a cycle.
  const out = new Map(g.nodes.map((n) => [n.id, []])), indeg = new Map(g.nodes.map((n) => [n.id, 0]));
  const colour = new Map(); const back = new Set();
  const dfs = (u) => { colour.set(u, 1); for (const e of g.edges.filter((x) => x.from === u)) { const c = colour.get(e.to) || 0; if (c === 1) back.add(e); else if (c === 0) dfs(e.to); } colour.set(u, 2); };
  for (const n of g.nodes) if (!colour.get(n.id)) dfs(n.id);
  const dag = g.edges.filter((e) => !back.has(e) && e.from !== e.to);
  for (const e of dag) { out.get(e.from).push(e.to); indeg.set(e.to, indeg.get(e.to) + 1); }
  const layer = new Map(); const q = g.nodes.filter((n) => !indeg.get(n.id)).map((n) => n.id);
  q.forEach((id) => layer.set(id, 0));
  const deg = new Map(indeg);
  while (q.length) { const u = q.shift(); for (const v of out.get(u)) { layer.set(v, Math.max(layer.get(v) ?? 0, layer.get(u) + 1)); deg.set(v, deg.get(v) - 1); if (deg.get(v) === 0) q.push(v); } }
  for (const n of g.nodes) if (!layer.has(n.id)) layer.set(n.id, 0);
  // An edge that spans several layers gets a small invisible node in each layer it crosses, so it is routed beside
  // the nodes in between rather than through them.
  let dn = 0; const path = new Map();
  for (const e of dag) {
    const a = layer.get(e.from), b = layer.get(e.to); const ids = [e.from];
    for (let l = a + 1; l < b; l++) { const id = `\u0000d${dn++}`; layer.set(id, l); sz.set(id, { w: 10, h: 10, lines: [], dummy: true }); ids.push(id); }
    ids.push(e.to); path.set(e, ids);
  }
  const ids = [...g.nodes.map((n) => n.id), ...[...sz.keys()].filter((k) => sz.get(k).dummy)];
  const L = Math.max(0, ...layer.values()) + 1; const rows = Array.from({ length: L }, () => []);
  for (const id of ids) rows[layer.get(id)].push(id);
  // Order within a layer by the average position of the neighbours in the next layer up or down, a few sweeps.
  const pos = new Map(); rows.forEach((r) => r.forEach((id, i) => pos.set(id, i)));
  const parents = new Map(ids.map((id) => [id, []])), kids = new Map(ids.map((id) => [id, []]));
  for (const ps of path.values()) for (let i = 0; i + 1 < ps.length; i++) { parents.get(ps[i + 1]).push(ps[i]); kids.get(ps[i]).push(ps[i + 1]); }
  const gkey = (id) => g.nodes.find((n) => n.id === id)?.group ?? '';
  for (let sweep = 0; sweep < 6; sweep++) {
    const down = sweep % 2 === 0;
    for (let li = down ? 1 : L - 2; down ? li < L : li >= 0; li += down ? 1 : -1) {
      const ref = down ? parents : kids;
      const bary = (id) => { const p = ref.get(id); return p.length ? p.reduce((a, x) => a + pos.get(x), 0) / p.length : pos.get(id); };
      rows[li].sort((a, b) => gkey(a).localeCompare(gkey(b)) || bary(a) - bary(b) || pos.get(a) - pos.get(b)); rows[li].forEach((id, i) => pos.set(id, i));
    }
  }
  // Coordinates: layers along the main axis; across it, each node is pulled towards the middle of its neighbours and
  // then nudged apart so that nothing overlaps.
  const main = (id) => (horiz ? sz.get(id).w : sz.get(id).h), cross = (id) => (horiz ? sz.get(id).h : sz.get(id).w);
  const gapMain = horiz ? GAP_X + 28 : GAP_Y, gapCross = horiz ? 22 : 30;
  const layerMain = rows.map((r) => Math.max(0, ...r.map(main)));
  const c = new Map(); // centre across the axis
  rows.forEach((r) => { let x = 0; for (const id of r) { c.set(id, x + cross(id) / 2); x += cross(id) + gapCross; } });
  const spread = (r) => { for (let i = 1; i < r.length; i++) { const need = c.get(r[i - 1]) + cross(r[i - 1]) / 2 + gapCross + cross(r[i]) / 2; if (c.get(r[i]) < need) c.set(r[i], need); } for (let i = r.length - 2; i >= 0; i--) { const need = c.get(r[i + 1]) - cross(r[i + 1]) / 2 - gapCross - cross(r[i]) / 2; if (c.get(r[i]) > need && i < r.length - 1 && false) c.set(r[i], need); } };
  for (let sweep = 0; sweep < 8; sweep++) {
    const down = sweep % 2 === 0;
    for (let li = down ? 1 : L - 2; down ? li < L : li >= 0; li += down ? 1 : -1) {
      const ref = down ? parents : kids;
      for (const id of rows[li]) { const nb = ref.get(id); if (nb.length) c.set(id, nb.reduce((a, x) => a + c.get(x), 0) / nb.length); }
      spread(rows[li]);
    }
  }
  const lo = Math.min(...ids.map((id) => c.get(id) - cross(id) / 2));
  const place = new Map(); let m0 = 0;
  rows.forEach((r, li) => { for (const id of r) { const cc = c.get(id) - lo; const mm = m0 + layerMain[li] / 2; place.set(id, horiz ? { x: mm, y: cc } : { x: cc, y: mm }); } m0 += layerMain[li] + gapMain; });
  if (g.dir === 'BT' || g.dir === 'RL') { const tot = m0 - gapMain; for (const p of place.values()) { if (g.dir === 'BT') p.y = tot - p.y; else p.x = tot - p.x; } }
  return { place, sz, back, path };
}

const FLOW_STYLE = (cls, classes, styles, id) => { const s = [classes.get(cls) || '', styles.get(id) || ''].join(','); const get = (k) => new RegExp(`${k}\\s*:\\s*([^,;]+)`).exec(s)?.[1]?.trim(); return { fill: get('fill'), stroke: get('stroke'), color: get('color'), w: get('stroke-width') }; };
const safeColour = (c) => (c && /^(#[0-9a-fA-F]{3,8}|[a-zA-Z]+|rgba?\([\d\s,.%]+\))$/.test(c) ? c : null);

function flowSvg(g, kind = 'Flowchart') {
  const { place, sz, path } = layoutFlow(g);
  if (!g.nodes.length) return null;
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const n of g.nodes) { const p = place.get(n.id), s = sz.get(n.id); x0 = Math.min(x0, p.x - s.w / 2); x1 = Math.max(x1, p.x + s.w / 2); y0 = Math.min(y0, p.y - s.h / 2); y1 = Math.max(y1, p.y + s.h / 2); }
  for (const [id, sc] of sz) { if (!sc.dummy) continue; const p = place.get(id); x0 = Math.min(x0, p.x - 6); x1 = Math.max(x1, p.x + 6); y0 = Math.min(y0, p.y - 6); y1 = Math.max(y1, p.y + 6); }
  // Subgraph frames.
  const frames = g.groups.map((gr) => { const members = g.nodes.filter((n) => { let k = n.group; while (k) { if (k === gr.id) return true; k = g.groups.find((z) => z.id === k)?.parent; } return false; }); if (!members.length) return null;
    let a = Infinity, b = Infinity, c = -Infinity, d = -Infinity; for (const n of members) { const p = place.get(n.id), s = sz.get(n.id); a = Math.min(a, p.x - s.w / 2); c = Math.max(c, p.x + s.w / 2); b = Math.min(b, p.y - s.h / 2); d = Math.max(d, p.y + s.h / 2); }
    const depth = (() => { let k = gr.parent, n = 0; while (k) { n++; k = g.groups.find((z) => z.id === k)?.parent; } return n; })();
    const pad = 16 - depth * 3; return { gr, x: a - pad, y: b - pad - 14, w: c - a + pad * 2, h: d - b + pad * 2 + 14 }; }).filter(Boolean);
  for (const f of frames) { x0 = Math.min(x0, f.x); y0 = Math.min(y0, f.y); x1 = Math.max(x1, f.x + f.w); y1 = Math.max(y1, f.y + f.h); }
  const M = 18; const W = x1 - x0 + M * 2, H = y1 - y0 + M * 2; const ox = M - x0, oy = M - y0;
  let s = '';
  for (const f of frames) s += `<g class="mm-group"><rect x="${f.x + ox}" y="${f.y + oy}" width="${f.w}" height="${f.h}" rx="10"/><text x="${f.x + ox + 10}" y="${f.y + oy + 15}">${esc(f.gr.label)}</text></g>`;
  // Edges, drawn under the nodes. An edge follows the invisible nodes the layout gave it, as one smooth curve.
  const edgeEnd = (a, b) => { const pa = place.get(a), sa = sz.get(a), pb = place.get(b); const dx = pb.x - pa.x, dy = pb.y - pa.y; if (!dx && !dy) return { x: pa.x, y: pa.y };
    const t = Math.min(dx ? (sa.w / 2) / Math.abs(dx) : Infinity, dy ? (sa.h / 2) / Math.abs(dy) : Infinity); return { x: pa.x + dx * t, y: pa.y + dy * t }; };
  const labelBoxes = []; const horiz = g.dir === 'LR' || g.dir === 'RL';
  g.edges.forEach((e) => {
    if (e.from === e.to) { const p = place.get(e.from), sc = sz.get(e.from); const x = p.x + ox + sc.w / 2, y = p.y + oy; s += `<path class="mm-edge" d="M${x} ${y - 8} q26 -8 26 8 q0 16 -26 8" fill="none" marker-end="url(#mm-arrow)"/>`; return; }
    const ids = path.get(e) || [e.from, e.to];
    const pts = ids.map((id) => place.get(id)); const A = edgeEnd(ids[0], ids[1]), B = edgeEnd(ids[ids.length - 1], ids[ids.length - 2]);
    let P = [A, ...pts.slice(1, -1), B].map((p) => ({ x: p.x + ox, y: p.y + oy }));
    // Two states that point at each other: draw the two arrows side by side, not on top of one another.
    const twin = g.edges.some((o) => o !== e && o.from === e.to && o.to === e.from); let nrm = null;
    if (twin) { const a = P[0], b = P[P.length - 1]; const len = Math.hypot(b.x - a.x, b.y - a.y) || 1; nrm = { x: -(b.y - a.y) / len, y: (b.x - a.x) / len }; P = P.map((p) => ({ x: p.x + nrm.x * 8, y: p.y + nrm.y * 8 })); }
    let d = `M${P[0].x} ${P[0].y}`;
    for (let i = 1; i < P.length; i++) { const a = P[i - 1], b = P[i]; d += horiz ? ` C${(a.x + b.x) / 2} ${a.y} ${(a.x + b.x) / 2} ${b.y} ${b.x} ${b.y}` : ` C${a.x} ${(a.y + b.y) / 2} ${b.x} ${(a.y + b.y) / 2} ${b.x} ${b.y}`; }
    const mk = (t) => (t === 'arrow' ? 'url(#mm-arrow)' : t === 'cross' ? 'url(#mm-cross)' : t === 'dot' ? 'url(#mm-dot)' : null);
    s += `<path class="mm-edge${e.dotted ? ' dotted' : ''}${e.thick ? ' thick' : ''}" d="${d}" fill="none"${mk(e.head) ? ` marker-end="${mk(e.head)}"` : ''}${mk(e.tail) ? ` marker-start="${mk(e.tail)}"` : ''}/>`;
    if (e.label) { const k = Math.floor((P.length - 1) / 2); const a = P[k], b = P[k + 1]; const w = textW(e.label, 12) + 12; const sh = nrm ? Math.abs(nrm.x) * (w / 2 + 2) + Math.abs(nrm.y) * 12 : 0; const mx = (a.x + b.x) / 2 + (nrm ? nrm.x * sh : 0), my = (a.y + b.y) / 2 + (nrm ? nrm.y * sh : 0);
      labelBoxes.push(`<g class="mm-elabel"><rect x="${mx - w / 2}" y="${my - 10}" width="${w}" height="20" rx="5"/><text x="${mx}" y="${my + 4}" text-anchor="middle">${esc(e.label)}</text></g>`); }
  });
  s += labelBoxes.join('');
  for (const n of g.nodes) {
    const p = place.get(n.id), c = sz.get(n.id); const cx = p.x + ox, cy = p.y + oy, w = c.w, h = c.h; const st = FLOW_STYLE(n.cls, g.classes, g.styles, n.id);
    const sty = [safeColour(st.fill) && `fill:${safeColour(st.fill)}`, safeColour(st.stroke) && `stroke:${safeColour(st.stroke)}`, st.w && /^[\d.]+(px)?$/.test(st.w) && `stroke-width:${st.w}`].filter(Boolean).join(';');
    const shape = {
      rect: `<rect x="${cx - w / 2}" y="${cy - h / 2}" width="${w}" height="${h}" rx="6"/>`,
      round: `<rect x="${cx - w / 2}" y="${cy - h / 2}" width="${w}" height="${h}" rx="14"/>`,
      stadium: `<rect x="${cx - w / 2}" y="${cy - h / 2}" width="${w}" height="${h}" rx="${h / 2}"/>`,
      sub: `<rect x="${cx - w / 2}" y="${cy - h / 2}" width="${w}" height="${h}" rx="3"/><line x1="${cx - w / 2 + 8}" y1="${cy - h / 2}" x2="${cx - w / 2 + 8}" y2="${cy + h / 2}"/><line x1="${cx + w / 2 - 8}" y1="${cy - h / 2}" x2="${cx + w / 2 - 8}" y2="${cy + h / 2}"/>`,
      circle: n.dot ? `<circle cx="${cx}" cy="${cy}" r="${w / 2 - (n.dot === 'end' ? 3 : 0)}" class="mm-solid"/>${n.dot === 'end' ? `<circle cx="${cx}" cy="${cy}" r="${w / 2}" class="mm-ring"/>` : ''}` : `<circle cx="${cx}" cy="${cy}" r="${w / 2}"/>`,
      diamond: `<polygon points="${cx},${cy - h / 2} ${cx + w / 2},${cy} ${cx},${cy + h / 2} ${cx - w / 2},${cy}"/>`,
      hex: `<polygon points="${cx - w / 2},${cy} ${cx - w / 2 + 16},${cy - h / 2} ${cx + w / 2 - 16},${cy - h / 2} ${cx + w / 2},${cy} ${cx + w / 2 - 16},${cy + h / 2} ${cx - w / 2 + 16},${cy + h / 2}"/>`,
      para: `<polygon points="${cx - w / 2 + 12},${cy - h / 2} ${cx + w / 2},${cy - h / 2} ${cx + w / 2 - 12},${cy + h / 2} ${cx - w / 2},${cy + h / 2}"/>`,
      flag: `<polygon points="${cx - w / 2},${cy - h / 2} ${cx + w / 2},${cy - h / 2} ${cx + w / 2},${cy + h / 2} ${cx - w / 2},${cy + h / 2} ${cx - w / 2 + 12},${cy}"/>`,
      db: `<path d="M${cx - w / 2} ${cy - h / 2 + 8} v${h - 16} a${w / 2} 8 0 0 0 ${w} 0 v${-(h - 16)} a${w / 2} 8 0 0 0 ${-w} 0 a${w / 2} 8 0 0 0 ${w} 0"/>`,
    }[n.shape] || '';
    const ty = cy - ((c.lines.length - 1) * 9) + 4.5 + (n.shape === 'db' ? 6 : 0);
    const label = c.lines.map((l, i) => `<tspan x="${cx}" y="${ty + i * 18}">${esc(l)}</tspan>`).join('');
    s += `<g class="mm-node mm-${n.shape}"${sty ? ` style="${sty}"` : ''}>${shape}<text text-anchor="middle"${safeColour(st.color) ? ` style="fill:${safeColour(st.color)}"` : ''}>${label}</text></g>`;
  }
  const defs = '<defs><marker id="mm-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0 0 10 5 0 10z" class="mm-head"/></marker><marker id="mm-cross" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M1 1 9 9M9 1 1 9" class="mm-x"/></marker><marker id="mm-dot" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="7" markerHeight="7"><circle cx="5" cy="5" r="4" class="mm-head"/></marker></defs>';
  return `<svg class="mm-svg" viewBox="0 0 ${Math.ceil(W)} ${Math.ceil(H)}" width="${Math.ceil(W)}" style="--w:${Math.ceil(W)}px" role="img" aria-label="${kind} with ${g.nodes.length} ${kind === 'Flowchart' ? 'steps' : 'states'}">${defs}${s}</svg>`;
}

// ---- sequence diagram ----
function sequenceSvg(lines) {
  const parts = []; const pid = new Map(); const add = (id, label, actor) => { if (!pid.has(id)) { pid.set(id, parts.length); parts.push({ id, label: label || id, actor: !!actor }); } else if (label) parts[pid.get(id)].label = label; };
  const items = []; let autonum = false, n = 0;
  for (const raw of lines.slice(1)) {
    const line = raw.replace(/%%.*$/, '').trim(); if (!line) continue;
    let m;
    if ((m = line.match(/^(participant|actor)\s+(\S+)(?:\s+as\s+(.+))?$/i))) { add(m[2], m[3] && cleanLabel(m[3]), m[1].toLowerCase() === 'actor'); continue; }
    if (/^autonumber/i.test(line)) { autonum = true; continue; }
    if ((m = line.match(/^(loop|alt|opt|par|critical|break|rect)\b\s*(.*)$/i))) { items.push({ k: 'open', kind: m[1].toLowerCase(), label: cleanLabel(m[2]) }); continue; }
    if ((m = line.match(/^(else|and|option)\b\s*(.*)$/i))) { items.push({ k: 'else', label: cleanLabel(m[2]) }); continue; }
    if (/^end$/i.test(line)) { items.push({ k: 'close' }); continue; }
    if ((m = line.match(/^Note\s+(left of|right of|over)\s+([^:]+):\s*(.*)$/i))) { const who = m[2].split(',').map((x) => x.trim()); who.forEach((w) => add(w)); items.push({ k: 'note', side: m[1].toLowerCase(), who, text: cleanLabel(m[3]) }); continue; }
    if ((m = line.match(/^(activate|deactivate)\s+(\S+)/i))) { items.push({ k: m[1].toLowerCase(), who: m[2] }); continue; }
    if ((m = line.match(/^(\S+?)\s*(--?>>|--?>|--?x|--?\)|<<--?>>)\s*([+-]?)(\S+?)\s*:\s*(.*)$/))) {
      add(m[1]); add(m[4]); items.push({ k: 'msg', from: m[1], to: m[4], dashed: m[2].startsWith('--'), arrow: /x$/.test(m[2]) ? 'cross' : />>$/.test(m[2]) || m[2].endsWith('>') ? 'arrow' : 'open', text: cleanLabel(m[5]), n: autonum ? ++n : 0 }); continue;
    }
  }
  if (parts.length < 1) return null;
  const colW = Math.max(120, ...parts.map((p) => textW(p.label) + 36)); const x = (i) => 30 + colW / 2 + i * colW;
  const headH = 40; let y = headH + 26; let body = ''; const lifeEnd = []; const frames = [];
  const open = [];
  for (const it of items) {
    if (it.k === 'msg') {
      const a = x(pid.get(it.from)), b = x(pid.get(it.to)); const self = a === b; const text = (it.n ? `${it.n}. ` : '') + it.text;
      if (self) { body += `<path class="mm-edge${it.dashed ? ' dotted' : ''}" d="M${a} ${y} h30 v22 h-30" fill="none" marker-end="url(#mm-arrow)"/><text x="${a + 36}" y="${y + 4}" class="mm-msg">${esc(text)}</text>`; y += 44; }
      else { const lx = Math.min(a, b) + (Math.abs(b - a)) / 2; body += `<text x="${lx}" y="${y - 7}" text-anchor="middle" class="mm-msg">${esc(text)}</text><path class="mm-edge${it.dashed ? ' dotted' : ''}" d="M${a} ${y} L${b} ${y}" fill="none" marker-end="${it.arrow === 'arrow' ? 'url(#mm-arrow)' : it.arrow === 'cross' ? 'url(#mm-cross)' : 'url(#mm-open)'}"/>`; y += 36; }
    } else if (it.k === 'note') {
      const ids = it.who.map((w) => pid.get(w)); const cx = x(ids[0]); const w = Math.max(100, textW(it.text, 12) + 20); const nx = it.side === 'left of' ? cx - w - 10 : it.side === 'right of' ? cx + 10 : (ids.length > 1 ? (x(ids[0]) + x(ids[ids.length - 1])) / 2 - w / 2 : cx - w / 2);
      body += `<g class="mm-note"><rect x="${nx}" y="${y - 12}" width="${w}" height="28" rx="4"/><text x="${nx + w / 2}" y="${y + 6}" text-anchor="middle">${esc(it.text)}</text></g>`; y += 40;
    } else if (it.k === 'open') { open.push({ ...it, y0: y - 14 }); y += 22; }
    else if (it.k === 'else') { const o = open[open.length - 1]; if (o) { (o.splits ||= []).push({ y, label: it.label }); y += 22; } }
    else if (it.k === 'close') { const o = open.pop(); if (o) frames.push({ ...o, y1: y - 6 }); y += 10; }
  }
  const W = 60 + colW * parts.length; const H = y + 18;
  let s = '';
  parts.forEach((p, i) => { s += `<line class="mm-life" x1="${x(i)}" y1="${headH}" x2="${x(i)}" y2="${H - 16}"/>`; });
  for (const f of frames) { const w = W - 36; s += `<g class="mm-frame"><rect x="18" y="${f.y0}" width="${w}" height="${f.y1 - f.y0}" rx="4"/><path class="tag" d="M18 ${f.y0} h${textW(f.kind) + 20} v16 l-8 6 h${-(textW(f.kind) + 12)}z"/><text x="26" y="${f.y0 + 14}" class="kind">${esc(f.kind)}</text><text x="${textW(f.kind) + 44}" y="${f.y0 + 15}">${esc(f.label ? `[${f.label}]` : '')}</text>${(f.splits || []).map((sp) => `<line x1="18" x2="${18 + w}" y1="${sp.y - 6}" y2="${sp.y - 6}" stroke-dasharray="5 4"/><text x="${textW(f.kind) + 44}" y="${sp.y + 8}">${esc(sp.label ? `[${sp.label}]` : '')}</text>`).join('')}</g>`; }
  s += body;
  parts.forEach((p, i) => { const w = colW - 20; s += `<g class="mm-node mm-part${p.actor ? ' actor' : ''}"><rect x="${x(i) - w / 2}" y="2" width="${w}" height="${headH - 6}" rx="${p.actor ? 18 : 6}"/><text x="${x(i)}" y="${headH / 2 + 3}" text-anchor="middle">${esc(p.label)}</text></g>`; });
  const defs = '<defs><marker id="mm-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0 0 10 5 0 10z" class="mm-head"/></marker><marker id="mm-open" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M1 1 9 5 1 9" class="mm-x" fill="none"/></marker><marker id="mm-cross" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M1 1 9 9M9 1 1 9" class="mm-x"/></marker></defs>';
  void lifeEnd;
  return `<svg class="mm-svg" viewBox="0 0 ${Math.ceil(W)} ${Math.ceil(H)}" width="${Math.ceil(W)}" style="--w:${Math.ceil(W)}px" role="img" aria-label="Sequence diagram with ${parts.length} participants">${defs}${s}</svg>`;
}

// ---- pie chart ----
const PIE = ['#0f766e', '#2563eb', '#d97706', '#7c3aed', '#db2777', '#059669', '#dc2626', '#0891b2', '#65a30d', '#9333ea'];
function pieSvg(lines) {
  let title = cleanLabel(lines[0].match(/\btitle\s+(.*)$/i)?.[1] ?? ''); const sl = []; const showData = /showData/i.test(lines[0]);
  for (const raw of lines.slice(1)) { const line = raw.trim(); let m; if ((m = line.match(/^title\s+(.*)$/i))) title = cleanLabel(m[1]); else if ((m = line.match(/^"?([^":]+?)"?\s*:\s*(-?[\d.]+)\s*$/))) { const v = +m[2]; if (v > 0) sl.push({ label: cleanLabel(m[1]), v }); } }
  const total = sl.reduce((a, b) => a + b.v, 0); if (!sl.length || !total) return null;
  const R = 92, cx = 116, cy = 116 + (title ? 18 : 0); let a0 = -Math.PI / 2; let s = '';
  sl.forEach((p, i) => { const a1 = a0 + (p.v / total) * Math.PI * 2; const large = a1 - a0 > Math.PI ? 1 : 0; const f = (a) => [cx + R * Math.cos(a), cy + R * Math.sin(a)];
    const [x0, y0] = f(a0), [x1, y1] = f(a1 - (sl.length === 1 ? 0.0001 : 0)); const mid = (a0 + a1) / 2;
    s += `<path d="M${cx} ${cy} L${x0} ${y0} A${R} ${R} 0 ${large} 1 ${x1} ${y1}z" fill="${PIE[i % PIE.length]}" class="mm-slice"><title>${esc(p.label)}: ${p.v} (${(p.v / total * 100).toFixed(1)}%)</title></path>`;
    if (p.v / total > 0.05) s += `<text x="${cx + R * 0.62 * Math.cos(mid)}" y="${cy + R * 0.62 * Math.sin(mid) + 4}" text-anchor="middle" class="mm-pct">${(p.v / total * 100).toFixed(0)}%</text>`; a0 = a1; });
  const lx = cx + R + 28; const lw = Math.max(...sl.map((p) => textW(p.label + (showData ? ` ${p.v}` : '')) )) + 34;
  sl.forEach((p, i) => { const ly = cy - (sl.length * 22) / 2 + i * 22 + 11; s += `<rect x="${lx}" y="${ly - 9}" width="12" height="12" rx="3" fill="${PIE[i % PIE.length]}"/><text x="${lx + 20}" y="${ly + 1}" class="mm-legend">${esc(p.label)}${showData ? ` <tspan class="faint">${p.v}</tspan>` : ''}</text>`; });
  const W = lx + lw + 6, H = cy + R + 18; if (title) s += `<text x="${W / 2}" y="20" text-anchor="middle" class="mm-title">${esc(title)}</text>`;
  return `<svg class="mm-svg" viewBox="0 0 ${Math.ceil(W)} ${Math.ceil(H)}" width="${Math.ceil(W)}" style="--w:${Math.ceil(W)}px" role="img" aria-label="Pie chart${title ? ': ' + esc(title) : ''}">${s}</svg>`;
}

// ---- state diagram: read as a flowchart (states are rounded boxes, [*] a dot, composites are frames) ----
function stateToFlow(lines) {
  const out = ['flowchart TD']; const labels = new Map(); const defs = []; const body = []; const stack = []; let k = 0;
  const q = (t) => `"${String(t).replace(/"/g, "'")}"`; const scope = () => stack[stack.length - 1] || 'root';
  const node = (raw, role) => { const id = raw.trim(); if (id === '[*]') { const nid = `${role}_${scope().replace(/\W/g, '_')}`; if (!labels.has(nid)) { labels.set(nid, 1); defs.push([stack.slice(), `${nid}${role === 'start' ? '((" "))' : '((" "))'}`]); } return nid; } return id.replace(/\W/g, '_') || id; };
  for (const raw of lines.slice(1)) {
    const line = raw.replace(/%%.*$/, '').trim(); if (!line || /^(note|end note|accTitle|accDescr|classDef|class |style )/i.test(line)) continue;
    let m;
    if ((m = line.match(/^direction\s+(TB|TD|BT|LR|RL)/i))) { out[0] = `flowchart ${m[1].toUpperCase()}`; continue; }
    if ((m = line.match(/^state\s+"([^"]+)"\s+as\s+(\w+)/i))) { defs.push([stack.slice(), `${m[2]}(${q(m[1])})`]); continue; }
    if ((m = line.match(/^state\s+(\w+)\s*<<(choice|fork|join)>>/i))) { defs.push([stack.slice(), m[2] === 'choice' ? `${m[1]}{" "}` : `${m[1]}[" "]`]); continue; }
    if ((m = line.match(/^state\s+"?([^"{]+?)"?\s*\{$/i))) { const id = m[1].trim().replace(/\W/g, '_'); stack.push(id); body.push(`subgraph ${id}[${q(m[1].trim())}]`); continue; }
    if (line === '}') { stack.pop(); body.push('end'); continue; }
    if ((m = line.match(/^(\[\*\]|[\w.-]+)\s*(-->|<--)\s*(\[\*\]|[\w.-]+)\s*(?::\s*(.*))?$/))) {
      const rev = m[2] === '<--'; const a = node(rev ? m[3] : m[1], 'start'), b = node(rev ? m[1] : m[3], 'end');
      body.push(`${a} -->${m[4] ? `|${q(m[4].trim())}|` : ''} ${b}`); k++; continue;
    }
    if ((m = line.match(/^(\w+)\s*:\s*(.+)$/))) { labels.set(m[1], (labels.get(m[1]) ? labels.get(m[1]) + '<br>' : '') + m[2].trim()); continue; }
    if ((m = line.match(/^(\w+)$/))) { defs.push([stack.slice(), `${m[1]}(${q(m[1])})`]); }
  }
  if (!k && !defs.length) return null;
  // Plain states become rounded boxes; described ones carry their text.
  const seen = new Set(); const stated = []; const declare = (id) => { if (seen.has(id)) return; seen.add(id); if (!/^(start|end)_/.test(id)) stated.push(`${id}(${q(labels.get(id) && typeof labels.get(id) === 'string' ? `${id}<br>${labels.get(id)}` : id)})`); };
  for (const l of body) { const mm = l.match(/^(\S+) -->(?:\|"[^"]*"\|)? (\S+)$/); if (mm) { declare(mm[1]); declare(mm[2]); } }
  return [...out, ...stated, ...defs.map((d) => d[1]), ...body];
}

/// Mermaid text to SVG, or null when the kind or the text is not understood (the caller shows the code).
export function mermaid(src) {
  const lines = String(src).replace(/\r/g, '').split('\n').filter((l) => l.trim() && !/^\s*%%/.test(l));
  if (!lines.length) return null;
  const first = lines[0].trim();
  try {
    if (/^(graph|flowchart)\b/i.test(first)) { const g = parseFlow(lines.flatMap((l) => l.split(/;\s*(?=\S)/))); return g.nodes.length > 80 ? null : flowSvg(g); }
    if (/^sequenceDiagram\b/i.test(first)) return sequenceSvg(lines);
    if (/^pie\b/i.test(first)) return pieSvg(lines);
    if (/^stateDiagram(-v2)?\b/i.test(first)) { const fl = stateToFlow(lines); if (!fl) return null; const g = parseFlow(fl); for (const n of g.nodes) if (/^(start|end)_/.test(n.id)) n.dot = n.id.slice(0, n.id.indexOf('_')); return g.nodes.length > 80 ? null : flowSvg(g, 'State diagram'); }
  } catch { return null; }
  return null;
}

// ============================== Markdown ==============================
const BLOCK = '\u0000'; const MATH = '\u0001'; const CODE = '\u0002';

function inline(t, ctx) {
  // Inline code and inline maths are set aside first, so nothing inside them is read as Markdown.
  const keep = []; const hold = (html) => { keep.push(html); return `${CODE}${keep.length - 1}${CODE}`; };
  let s = t.replace(/`([^`\n]+)`/g, (_, c) => hold(`<code>${esc(c)}</code>`));
  s = s.replace(/\\\(([\s\S]+?)\\\)/g, (_, m) => hold(mathml(m.trim(), false)))
    .replace(/(^|[^\w$\\])\$([^$\n]{1,160}?)\$(?![\w$])/g, (m, pre, math) => (/^\s|\s$/.test(math) || /^\d[\d,.]*$/.test(math) ? m : pre + hold(mathml(math.trim(), false))));
  s = esc(s);
  s = s.replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, (_, a, u) => `<a href="${u}" target="_blank" rel="noopener noreferrer">${a}</a>`)
    .replace(/(^|[\s(])(https?:\/\/[^\s<)]+[^\s<).,;:!?])/g, (_, p, u) => `${p}<a href="${u}" target="_blank" rel="noopener noreferrer">${u.replace(/^https?:\/\//, '')}</a>`)
    .replace(/\*\*\*([^*]+)\*\*\*/g, '<strong><em>$1</em></strong>').replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>').replace(/__([^_]+)__/g, '<strong>$1</strong>')
    .replace(/(^|[^*\w])\*([^*\n]+)\*(?!\w)/g, '$1<em>$2</em>').replace(/(^|[^\w])_([^_\n]+)_(?!\w)/g, '$1<em>$2</em>')
    .replace(/~~([^~]+)~~/g, '<del>$1</del>').replace(/==([^=]+)==/g, '<mark>$1</mark>')
    .replace(/&lt;br\s*\/?&gt;/g, '<br>').replace(/&lt;(sub|sup|kbd)&gt;([^<&]{1,40})&lt;\/\1&gt;/g, '<$1>$2</$1>');
  if (ctx.cites) s = s.replace(/\[(\d{1,2}(?:\s*[,;]\s*\d{1,2})*)\]/g, (m, ns) => ns.split(/\s*[,;]\s*/).map((n) => (ctx.cites.has(+n) ? `<button type="button" class="cite" data-cite="${+n}" aria-haspopup="dialog" aria-expanded="false" aria-label="Source ${+n}: show the passage">${+n}</button>` : `[${+n}]`)).join(''));
  return s.replace(new RegExp(`${CODE}(\\d+)${CODE}`, 'g'), (_, i) => keep[+i]);
}

const isTableRow = (l) => /^\s*\|.*\|\s*$/.test(l) || (/\|/.test(l) && /^[^|]+\|[^|]*/.test(l) && !/^\s*[-*+]\s/.test(l) && l.split('|').length > 2);
const isDelim = (l) => /^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$/.test(l || '') && /\|/.test(l || '');
const splitRow = (l) => { const cells = []; let cur = ''; let tick = false; const t = l.trim().replace(/^\||\|$/g, ''); for (let i = 0; i < t.length; i++) { const c = t[i]; if (c === '`') tick = !tick; if (c === '\\' && t[i + 1] === '|') { cur += '|'; i++; continue; } if (c === '|' && !tick) { cells.push(cur.trim()); cur = ''; } else cur += c; } cells.push(cur.trim()); return cells; };

const mathBlock = (tex) => `<div class="math" data-tex="${esc(tex.trim())}">${mathml(tex, true)}<button type="button" class="math-copy" data-act="tex" aria-label="Copy the formula as LaTeX">Copy LaTeX</button></div>`;

function codeBlock(lang, code, ctx) {
  const body = code.replace(/\n$/, '');
  const l = (lang || '').trim().split(/\s+/)[0].toLowerCase();
  if (l === 'mermaid' && ctx.closed) {
    const svg = mermaid(body);
    if (svg) return `<figure class="diagram" data-kind="diagram"><div class="fig-bar"><span class="lang">diagram</span><span class="grow"></span><button type="button" class="fig-btn" data-act="source">Source</button><button type="button" class="fig-btn" data-act="copysrc">Copy source</button><button type="button" class="fig-btn" data-act="svg">Save SVG</button><button type="button" class="fig-btn" data-act="png">Save PNG</button><button type="button" class="fig-btn" data-act="full" aria-label="Open the diagram larger">Expand</button></div><div class="fig-body">${svg}</div><pre class="fig-source" hidden><code>${esc(body)}</code></pre></figure>`;
  }
  if ((l === 'math' || l === 'latex' || l === 'tex' || l === 'katex') && ctx.closed) return mathBlock(body);
  if ((l === 'svg') && ctx.closed && /^\s*<svg[\s>]/i.test(body)) return `<pre class="code"><span class="lang">svg</span><button class="copy" type="button">copy</button><code>${esc(body)}</code></pre>`;
  const lines = body.split('\n').length; const long = lines > 28;
  return `<pre class="code${long ? ' long' : ''}"><span class="lang">${esc(l || 'text')}</span><button class="copy" type="button">copy</button><code>${highlight(body, l)}</code>${long ? `<button type="button" class="more" data-act="expand">Show all ${lines} lines</button>` : ''}</pre>`;
}

function listItems(lines, start, ctx) {
  // Nested lists by indentation; an item continues on indented lines. Returns [html, nextIndex].
  const indentOf = (l) => l.match(/^\s*/)[0].replace(/\t/g, '    ').length;
  const re = /^(\s*)([-*+]|\d+[.)])\s+(.*)$/;
  const base = indentOf(lines[start]); let i = start; let html = ''; let type = null; let first = null;
  while (i < lines.length) {
    const m = re.exec(lines[i]); if (!m) break; const ind = indentOf(lines[i]); if (ind < base) break;
    if (ind > base) { // a deeper list belongs to the previous item
      const [sub, next] = listItems(lines, i, ctx); html = html.replace(/<\/li>$/, `${sub}</li>`); i = next; continue;
    }
    const t = /\d/.test(m[2]) ? 'ol' : 'ul'; if (type && t !== type) break; type = t; if (first == null && t === 'ol') first = parseInt(m[2], 10);
    let text = m[3]; i++;
    while (i < lines.length && lines[i].trim() && !re.test(lines[i]) && indentOf(lines[i]) > base && !/^\s*(```|\|)/.test(lines[i])) { text += ' ' + lines[i].trim(); i++; }
    const task = /^\[([ xX])\]\s+(.*)$/.exec(text);
    html += task ? `<li class="task"><span class="box${task[1] === ' ' ? '' : ' on'}" aria-hidden="true"></span><span class="sr-only">${task[1] === ' ' ? 'to do: ' : 'done: '}</span>${inline(task[2], ctx)}</li>` : `<li>${inline(text, ctx)}</li>`;
    // A blank line between items keeps the same list when the next line is another item of this level.
    if (i < lines.length && !lines[i].trim()) { const nx = lines.slice(i + 1).find((l) => l.trim()); const nm = nx && re.exec(nx); if (nm && indentOf(nx) >= base && (indentOf(nx) > base || (/\d/.test(nm[2]) ? 'ol' : 'ul') === type)) { while (i < lines.length && !lines[i].trim()) i++; } }
  }
  return [`<${type}${type === 'ol' && first > 1 ? ` start="${first}"` : ''}>${html}</${type}>`, i];
}

/// Markdown to HTML. `cites`, when given, is the set of source numbers [n] may turn into citation buttons.
export function markdown(src, cites = null) {
  if (!src) return '';
  const ctx = { cites, closed: true };
  let s = String(src).replace(/\r/g, '');
  const blocks = [];
  // Fenced code first (``` or ~~~), by lines, so nothing inside it is read as Markdown. A fence that never closes
  // (an answer still streaming) runs to the end and is drawn as plain code until it closes.
  {
    const src2 = s.split('\n'); const keepLines = []; let k = 0;
    while (k < src2.length) {
      const fm = /^([ \t]*)(`{3,}|~{3,})\s*([^`\n]*)$/.exec(src2[k]);
      if (!fm) { keepLines.push(src2[k]); k++; continue; }
      const [, ind, fence, lang] = fm; const body = []; let j = k + 1; let closed = false;
      const closeRe = new RegExp(`^[ \\t]*${fence[0]}{${fence.length},}[ \\t]*$`);
      while (j < src2.length) { if (closeRe.test(src2[j])) { closed = true; break; } body.push(ind ? src2[j].replace(new RegExp(`^${ind}`), '') : src2[j]); j++; }
      blocks.push({ lang, code: body.join('\n'), closed }); keepLines.push(`${BLOCK}${blocks.length - 1}${BLOCK}`); k = closed ? j + 1 : j;
    }
    s = keepLines.join('\n');
  }
  // Display maths. An unclosed $$ stays text until it closes.
  s = s.replace(/\$\$([\s\S]+?)\$\$/g, (_, m) => { blocks.push({ math: m.trim() }); return `\n${BLOCK}${blocks.length - 1}${BLOCK}\n`; })
    .replace(/\\\[([\s\S]+?)\\\]/g, (_, m) => { blocks.push({ math: m.trim() }); return `\n${BLOCK}${blocks.length - 1}${BLOCK}\n`; });
  const lines = s.split('\n'); const out = []; let para = [];
  const flush = () => { if (para.length) { out.push(`<p>${inline(para.join('\n'), ctx).replace(/\n/g, '<br>')}</p>`); para = []; } };
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const bm = line.match(new RegExp(`^${BLOCK}(\\d+)${BLOCK}$`));
    if (bm) { flush(); const b = blocks[+bm[1]]; out.push(b.math != null ? mathBlock(b.math) : codeBlock(b.lang, b.code, { ...ctx, closed: b.closed })); continue; }
    if (!line.trim()) { flush(); continue; }
    if (/^ {0,3}([-*_])(\s*\1){2,}\s*$/.test(line)) { flush(); out.push('<hr>'); continue; }
    const hm = line.match(/^(#{1,6})\s+(.*?)\s*#*\s*$/); if (hm) { flush(); const lv = Math.min(5, hm[1].length + 1); out.push(`<h${lv}>${inline(hm[2], ctx)}</h${lv}>`); continue; }
    if (/^\s*>/.test(line)) { flush(); const q = []; while (i < lines.length && /^\s*>/.test(lines[i])) { q.push(lines[i].replace(/^\s*>\s?/, '')); i++; } i--;
      const inner = markdown(q.join('\n'), cites); const call = /^<p>\s*\[!(note|tip|important|warning|caution)\]\s*(<br>)?/i.exec(inner);
      out.push(call ? `<aside class="callout ${call[1].toLowerCase()}"><b class="ct">${call[1][0].toUpperCase() + call[1].slice(1).toLowerCase()}</b>${inner.replace(call[0], '<p>')}</aside>` : `<blockquote>${inner}</blockquote>`); continue; }
    if (isTableRow(line) && isDelim(lines[i + 1])) {
      flush(); const head = splitRow(line); const al = splitRow(lines[i + 1]).map((c) => (/^:-+:$/.test(c) ? 'center' : /-:$/.test(c) ? 'right' : /^:-/.test(c) ? 'left' : ''));
      i += 2; const body = []; while (i < lines.length && lines[i].trim() && isTableRow(lines[i]) && !/^\s*\u0000/.test(lines[i])) { body.push(splitRow(lines[i])); i++; } i--;
      const cell = (tag, c, k) => `<${tag}${al[k] ? ` class="al-${al[k]}"` : ''}>${inline(c, ctx)}</${tag}>`;
      const isNum = (c) => /^[-+−]?[$€£]?[\d,.]+(\s?[%a-zA-Zµ]{0,4})?$/.test(c.replace(/\*\*/g, '').trim());
      const num = (k) => body.some((r) => r[k]) && body.every((r) => !r[k] || isNum(r[k]));
      const hh = head.map((c, k) => { if (!al[k] && num(k)) al[k] = 'right'; return cell('th', c, k); }).join('');
      out.push(`<div class="tablewrap"><button type="button" class="tbl-copy" data-act="table">Copy table</button><table><thead><tr>${hh}</tr></thead><tbody>${body.map((r) => `<tr>${head.map((_, k) => cell('td', r[k] ?? '', k)).join('')}</tr>`).join('')}</tbody></table></div>`); continue;
    }
    if (/^(\s*)([-*+]|\d+[.)])\s+/.test(line)) { flush(); const [html, next] = listItems(lines, i, ctx); out.push(html); i = next - 1; continue; }
    para.push(line);
  }
  flush();
  return out.join('');
}

// Plain text of an answer, for copying: Markdown kept, the model's own text.
export const answerPlain = (src) => String(src ?? '');
