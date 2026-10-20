// Nafs-Aql web UI. Vanilla ES module, no build step. Served by nafs-gateway.
// Sections: core utilities, markdown, charts, chat, rooms, performance, evaluation, system.

// ============================== core ==============================
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
// Safe lookup for code that runs after an await: the page may have changed, so a missing element
// becomes a detached placeholder instead of a TypeError.
const $s = (s) => document.querySelector(s) || document.createElement('div');
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const h = (html) => { const t = document.createElement('template'); t.innerHTML = html.trim(); return t.content.firstElementChild; };
const store = { get(k, d) { try { const v = localStorage.getItem('nafs.' + k); return v === null ? d : JSON.parse(v); } catch { return d; } }, set(k, v) { try { localStorage.setItem('nafs.' + k, JSON.stringify(v)); } catch {} } };
const fmtMs = (ms) => ms == null ? '–' : ms < 1000 ? `${Math.round(ms)} ms` : ms < 60000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.floor(ms / 60000)}m ${Math.round((ms % 60000) / 1000)}s`;
const pct = (x, d = 1) => x == null || Number.isNaN(x) ? '–' : `${(x * 100).toFixed(d)}%`;
const r4 = (v) => typeof v === 'number' ? +v.toPrecision(4) : v;
const num = (x, d = 2) => x == null || Number.isNaN(x) ? '–' : Number(x).toFixed(d);
const ago = (iso) => { const s = (Date.now() - new Date(iso).getTime()) / 1000; return s < 60 ? 'just now' : s < 3600 ? `${Math.floor(s / 60)}m ago` : s < 86400 ? `${Math.floor(s / 3600)}h ago` : new Date(iso).toLocaleDateString(); };
const icon = {
  bolt: '<svg viewBox="0 0 24 24"><path d="M13 2 4 14h7l-1 8 9-12h-7z"/></svg>',
  brain: '<svg viewBox="0 0 24 24"><path d="M9 4a3 3 0 0 0-3 3 3 3 0 0 0-2 5 3 3 0 0 0 2 5 3 3 0 0 0 3 3h1V4z"/><path d="M15 4a3 3 0 0 1 3 3 3 3 0 0 1 2 5 3 3 0 0 1-2 5 3 3 0 0 1-3 3h-1V4z"/></svg>',
  check: '<svg viewBox="0 0 24 24"><path d="M5 12l5 5 9-10"/></svg>',
  x: '<svg viewBox="0 0 24 24"><path d="M6 6l12 12M18 6 6 18"/></svg>',
  users: '<svg viewBox="0 0 24 24"><circle cx="9" cy="8" r="3"/><path d="M3 19c.6-3 3-5 6-5s5.4 2 6 5"/><path d="M16 4a3 3 0 0 1 0 6M21 19c-.4-2.2-1.7-3.8-3.5-4.5"/></svg>',
  alert: '<svg viewBox="0 0 24 24"><path d="M12 3 2 20h20z"/><path d="M12 10v4M12 17h.01"/></svg>',
  send: '<svg viewBox="0 0 24 24"><path d="M4 12 20 4l-6 16-3-7z"/></svg>',
  stop: '<svg viewBox="0 0 24 24"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>',
  copy: '<svg viewBox="0 0 24 24"><rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V5a1 1 0 0 0-1-1H5a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h3"/></svg>',
  search: '<svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="6"/><path d="m20 20-4-4"/></svg>',
  plus: '<svg viewBox="0 0 24 24"><path d="M12 5v14M5 12h14"/></svg>',
  retry: '<svg viewBox="0 0 24 24"><path d="M4 12a8 8 0 1 0 3-6.3"/><path d="M4 4v5h5"/></svg>',
  shield: '<svg viewBox="0 0 24 24"><path d="M12 3 5 6v6c0 4.4 3 7.6 7 9 4-1.4 7-4.6 7-9V6z"/><path d="m9 12 2 2 4-4"/></svg>',
  play: '<svg viewBox="0 0 24 24"><path d="M7 5v14l11-7z"/></svg>',
  up: '<svg viewBox="0 0 24 24"><path d="M7 11v9H4v-9zM7 11l4-8c1.5 0 2.5 1 2.3 2.6L13 9h5.6a2 2 0 0 1 2 2.4l-1.4 7A2 2 0 0 1 17.2 20H7"/></svg>',
  down: '<svg viewBox="0 0 24 24"><path d="M17 13V4h3v9zM17 13l-4 8c-1.5 0-2.5-1-2.3-2.6L11 15H5.4a2 2 0 0 1-2-2.4l1.4-7A2 2 0 0 1 6.8 4H17"/></svg>',
  compare: '<svg viewBox="0 0 24 24"><rect x="3" y="4" width="7.5" height="16" rx="1.5"/><rect x="13.5" y="4" width="7.5" height="16" rx="1.5"/></svg>',
  book: '<svg viewBox="0 0 24 24"><path d="M4 5a2 2 0 0 1 2-2h13v16H6a2 2 0 0 0-2 2z"/><path d="M4 21V5"/></svg>',
  download: '<svg viewBox="0 0 24 24"><path d="M12 4v11M7 10l5 5 5-5M5 20h14"/></svg>',
  leaf: '<svg viewBox="0 0 24 24"><path d="M5 19c0-8 5-14 15-15-1 10-7 15-15 15z"/><path d="M5 19 13 11"/></svg>',
};

// ============================== design vocabulary (analogy, not theology: ASSUMPTIONS A-010) ==============================
const STATIONS = {
  A: { en: 'Draft', ar: 'nafs', tip: 'System 1 answers at once: the first, unexamined response.' },
  L: { en: 'Flagged', ar: 'lawwāma', tip: 'Self-reproach (Q 75:2): the gate or a verifier finds the draft wanting.' },
  M: { en: 'Deliberate', ar: '\u02bfaql', tip: 'Reflection: System 2 reasons within a budget, without seeing the draft.' },
  T: { en: 'Accepted', ar: 'muṭmaʾinna', tip: 'At rest (Q 89:27): the answer is accepted with a stated assurance.' },
  X: { en: 'Abstain', ar: 'tawaqquf', tip: 'Judgement withheld: no candidate earned acceptance within the budget.' },
};
function designNotes() {
  modal(`<h3 style="margin-top:0">${icon.book} Design notes: nafs and \u02bfaql</h3>
    <p class="small">Nafs-Aql borrows the classical vocabulary of the self as a <b>design analogy</b>, not as a theological claim. The engineering names are primary; scholar review is a release gate (SR-005).</p>
    <table class="t small"><tr><th>Term</th><th>Source</th><th>In Nafs-Aql</th></tr>
      <tr><td><i>nafs</i> (self; first impulse)</td><td>Q 12:53 (al-nafs al-ammāra)</td><td>System 1: the fast draft</td></tr>
      <tr><td><i>lawwāma</i> (self-reproaching)</td><td>Q 75:2</td><td>State L: the draft is flagged by the gate or a verifier</td></tr>
      <tr><td><i>\u02bfaql</i> (intellect, restraint)</td><td>al-Ghazālī, <i>Iḥyāʾ</i> Book 21: the counsellor of the heart</td><td>The Aql Gate and System 2 deliberation</td></tr>
      <tr><td><i>muṭmaʾinna</i> (at rest)</td><td>Q 89:27</td><td>State T: accepted with an assurance level</td></tr>
      <tr><td><i>tawaqquf</i> (withholding judgement)</td><td>usage in jurisprudence and theology</td><td>State X: abstention with a notice</td></tr>
      <tr><td><i>muḥāsaba</i> (self-accounting)</td><td>Sufi practice of examining the self</td><td>The ledger, audits and capability table</td></tr>
      <tr><td><i>tazkiya</i> (purification)</td><td>Q 91:9</td><td>Distillation: verified System 2 traces improve System 1</td></tr></table>
    <p class="tiny muted">Rūḥ, ilhām and the Active Intellect are deliberately not mapped. Sources and verification notes: docs/research/islamic_sources.md.</p>`);
}

function toast(msg, ms = 2600) { const t = $('#toast'); t.textContent = msg; t.classList.add('show'); clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.remove('show'), ms); }
// A page loader that loses the gateway mid-request (restart, network) must not leave an uncaught error:
// show one notice instead; the status pill and the next navigation recover on their own.
window.addEventListener('unhandledrejection', (e) => { const m = String(e.reason?.message || e.reason || '');
  if (/Failed to fetch|NetworkError|Load failed/i.test(m)) { e.preventDefault(); if (!toast._net || Date.now() - toast._net > 10000) { toast._net = Date.now(); toast('Gateway unreachable; retrying when it is back', 4000); } } });
function modal(html) { $('#modalContent').innerHTML = html; $('#modal').showModal(); }

const api = {
  token: () => store.get('token', ''),
  headers() { const h = { 'Content-Type': 'application/json' }; const t = this.token(); if (t) h.Authorization = `Bearer ${t}`; return h; },
  async req(method, path, body) {
    const r = await fetch(path, { method, headers: this.headers(), body: body === undefined ? undefined : JSON.stringify(body) });
    if (r.status === 204) return null;
    const text = await r.text(); let data; try { data = text ? JSON.parse(text) : null; } catch { data = text; }
    if (!r.ok) { const e = new Error((data && data.message) || `HTTP ${r.status}`); e.status = r.status; e.data = data; if (r.status === 401) toast('Unauthorized: set an API token (key icon, bottom left).', 4000); throw e; }
    return data;
  },
  get(p) { return this.req('GET', p); }, post(p, b) { return this.req('POST', p, b ?? {}); }, patch(p, b) { return this.req('PATCH', p, b); }, del(p) { return this.req('DELETE', p); },
  // POST + Server-Sent Events reader (EventSource cannot POST).
  async stream(path, body, onEvent, signal) {
    const r = await fetch(path, { method: 'POST', headers: this.headers(), body: JSON.stringify(body), signal });
    if (!r.ok) { let m = `HTTP ${r.status}`; try { m = (await r.json()).message || m; } catch {} throw new Error(m); }
    const reader = r.body.getReader(); const dec = new TextDecoder(); let buf = '';
    for (;;) {
      const { value, done } = await reader.read(); if (done) break;
      buf += dec.decode(value, { stream: true });
      let i; while ((i = buf.indexOf('\n\n')) >= 0) {
        const chunk = buf.slice(0, i); buf = buf.slice(i + 2);
        let ev = 'message', data = '';
        for (const line of chunk.split('\n')) { if (line.startsWith('event:')) ev = line.slice(6).trim(); else if (line.startsWith('data:')) data += line.slice(5).trim(); }
        if (data) { try { onEvent(ev, JSON.parse(data)); } catch (e) { console.warn('bad event', e, data); } }
      }
    }
  },
};

// ============================== markdown (safe subset) ==============================
// Make common LaTeX from the models readable without a math renderer.
function prettyMath(t) {
  const sym = { times: '×', cdot: '·', div: '÷', le: '≤', leq: '≤', ge: '≥', geq: '≥', neq: '≠', ne: '≠', approx: '≈', pm: '±', infty: '∞', to: '→', rightarrow: '→', Rightarrow: '⇒', implies: '⇒', quad: ' ', qquad: '  ', ldots: '…', dots: '…', pi: 'π', alpha: 'α', beta: 'β', theta: 'θ', lambda: 'λ', sqrt: '√', degree: '°', circ: '°', '%': '%', '$': '$' };
  let x = t;
  for (let i = 0; i < 3; i++) x = x.replace(/\\(?:d|t)?frac\{([^{}]*)\}\{([^{}]*)\}/g, '($1)/($2)');
  x = x.replace(/\\boxed\{([^{}]*)\}/g, '**$1**').replace(/\\(?:text|mathrm|mathbf|textbf|operatorname)\{([^{}]*)\}/g, '$1').replace(/\\sqrt\{([^{}]*)\}/g, '√($1)');
  x = x.replace(/\\left|\\right/g, '').replace(/\\[,;:! ]/g, ' ').replace(/\\([a-zA-Z]+|[%$])/g, (m, w) => sym[w] ?? m);
  x = x.replace(/\^\{([^{}]*)\}/g, '^$1').replace(/\\\(|\\\)/g, '');
  return x;
}

function md(src) {
  if (!src) return '';
  const blocks = []; let s = String(src).replace(/\r/g, '');
  // Display math ($$...$$ or \[...\]) renders as a centred formula line; inline $...$ as code.
  s = s.replace(/```([\w+-]*)\n?([\s\S]*?)(```|$)/g, (_, lang, code) => { blocks.push({ lang, code }); return `\u0000${blocks.length - 1}\u0000`; });
  s = prettyMath(s); // after code blocks are set aside, so code is never rewritten
  s = s.replace(/\$\$\s*([\s\S]*?)\s*\$\$/g, (_, m) => `\u0001${m.trim().replace(/\n/g, ' ')}\u0001`).replace(/\\\[\s*([\s\S]*?)\s*\\\]/g, (_, m) => `\u0001${m.trim().replace(/\n/g, ' ')}\u0001`);
  const inline = (t) => esc(t)
    .replace(/\u0001([^\u0001]*)\u0001/g, '<div class="math">$1</div>')
    .replace(/(^|[^\w$])\$([^$\n]{1,80}?)\$(?![\w$])/g, '$1<code class="m">$2</code>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|[^*])\*([^*\n]+)\*/g, '$1<em>$2</em>')
    .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  const out = []; const lines = s.split('\n'); let list = null; let para = [];
  const flushPara = () => { if (para.length) { out.push(`<p>${inline(para.join('\n')).replace(/\n/g, '<br>')}</p>`); para = []; } };
  const flushList = () => { if (list) { out.push(`<${list.t}${list.t === 'ol' && list.start > 1 ? ` start="${list.start}"` : ''}>${list.items.map(i => `<li>${inline(i)}</li>`).join('')}</${list.t}>`); list = null; } };
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const cb = line.match(/^\u0000(\d+)\u0000$/);
    if (cb) { flushPara(); flushList(); const b = blocks[+cb[1]]; out.push(`<pre><span class="lang">${esc(b.lang)}</span><button class="copy" type="button">copy</button><code>${esc(b.code.replace(/\n$/, ''))}</code></pre>`); continue; }
    const dm = line.trim().match(/^\u0001([^\u0001]*)\u0001$/); if (dm) { flushPara(); flushList(); out.push(`<div class="math">${esc(dm[1])}</div>`); continue; }
    const hm = line.match(/^(#{1,4})\s+(.*)$/); if (hm) { flushPara(); flushList(); out.push(`<h${hm[1].length + 1}>${inline(hm[2])}</h${hm[1].length + 1}>`); continue; }
    if (/^\s*\|.*\|\s*$/.test(line) && /^\s*\|?\s*:?-{2,}/.test(lines[i + 1] || '')) {
      flushPara(); flushList(); const row = (l) => l.trim().replace(/^\||\|$/g, '').split('|').map(c => c.trim());
      const head = row(line); i += 1; const body = []; while (i + 1 < lines.length && /^\s*\|.*\|\s*$/.test(lines[i + 1])) { body.push(row(lines[++i])); }
      out.push(`<table><thead><tr>${head.map(c => `<th>${inline(c)}</th>`).join('')}</tr></thead><tbody>${body.map(r => `<tr>${r.map(c => `<td>${inline(c)}</td>`).join('')}</tr>`).join('')}</tbody></table>`); continue;
    }
    const ul = line.match(/^\s*[-*+]\s+(.*)$/), ol = line.match(/^\s*(\d+)[.)]\s+(.*)$/);
    // A numbered list split by blank lines continues its numbering (models often separate items).
    if (ul || ol) { flushPara(); const t = ul ? 'ul' : 'ol'; if (!list || list.t !== t) { flushList(); list = { t, items: [], start: ol ? +ol[1] : 1 }; } list.items.push(ul ? ul[1] : ol[2]); continue; }
    if (!line.trim()) { flushPara(); const nx = lines.slice(i + 1).find(l => l.trim()); const same = nx && list && (list.t === 'ol' ? /^\s*\d+[.)]\s+/.test(nx) : /^\s*[-*+]\s+/.test(nx)); if (!same) flushList(); continue; }
    if (list && /^\s{2,}\S/.test(line)) { list.items[list.items.length - 1] += ' ' + line.trim(); continue; }
    flushList(); para.push(line);
  }
  flushPara(); flushList();
  return out.join('');
}
document.addEventListener('click', (e) => {
  const b = e.target.closest('pre .copy'); if (!b) return;
  navigator.clipboard?.writeText(b.parentElement.querySelector('code').textContent).then(() => { b.textContent = 'copied'; setTimeout(() => (b.textContent = 'copy'), 1200); });
});

// ============================== charts (inline SVG) ==============================
const chart = {
  scatter(points, { w = 640, hgt = 220, xLabel = '', yLabel = '', lines = [] } = {}) {
    if (!points.length) return '<div class="empty small">No data yet.</div>';
    const pad = { l: 48, r: 12, t: 10, b: 30 };
    const xs = points.map(p => p.x), ys = points.map(p => p.y).concat(lines.map(l => l.y));
    const xmax = Math.max(1, ...xs), ymax = Math.max(1, ...ys) * 1.08;
    const X = x => pad.l + (x / xmax) * (w - pad.l - pad.r), Y = y => hgt - pad.b - (y / ymax) * (hgt - pad.t - pad.b);
    const ticks = (m) => [0, .25, .5, .75, 1].map(f => f * m);
    let g = ticks(ymax).map(t => `<line class="gridl" x1="${pad.l}" x2="${w - pad.r}" y1="${Y(t)}" y2="${Y(t)}"/><text x="${pad.l - 6}" y="${Y(t) + 3}" text-anchor="end">${t >= 1000 ? (t / 1000).toFixed(0) + 's' : Math.round(t)}</text>`).join('');
    g += ticks(xmax).map(t => `<text x="${X(t)}" y="${hgt - 12}" text-anchor="middle">${t.toFixed(0)}</text>`).join('');
    g += lines.map(l => `<line x1="${pad.l}" x2="${w - pad.r}" y1="${Y(l.y)}" y2="${Y(l.y)}" stroke="${l.color}" stroke-dasharray="4 3"/><text x="${w - pad.r}" y="${Y(l.y) - 4}" text-anchor="end" style="fill:${l.color}">${esc(l.label)}</text>`).join('');
    g += points.map(p => `<circle cx="${X(p.x)}" cy="${Y(p.y)}" r="3.6" fill="${p.color}" fill-opacity=".8"><title>${esc(p.title || '')}</title></circle>`).join('');
    return `<div class="chart"><svg viewBox="0 0 ${w} ${hgt}" role="img"><line class="axis" x1="${pad.l}" x2="${w - pad.r}" y1="${hgt - pad.b}" y2="${hgt - pad.b}"/>${g}<text x="${(w + pad.l) / 2}" y="${hgt - 1}" text-anchor="middle">${esc(xLabel)}</text><text x="10" y="${hgt / 2}" transform="rotate(-90 10 ${hgt / 2})" text-anchor="middle">${esc(yLabel)}</text></svg></div>`;
  },
  bars(items, { w = 640, hgt = 200, max = null, fmt = v => v } = {}) {
    if (!items.length) return '<div class="empty small">No data yet.</div>';
    const pad = { l: 40, r: 10, t: 12, b: 34 }; const m = max ?? Math.max(1e-9, ...items.map(i => i.v ?? 0));
    const bw = (w - pad.l - pad.r) / items.length;
    const g = items.map((it, i) => { const v = it.v ?? 0; const bh = (v / m) * (hgt - pad.t - pad.b); const x = pad.l + i * bw + bw * .18;
      return `<rect x="${x}" y="${hgt - pad.b - bh}" width="${bw * .64}" height="${Math.max(0, bh)}" rx="4" fill="${it.color || 'var(--accent)'}"><title>${esc(it.label)}: ${esc(fmt(it.v))}</title></rect>
              <text x="${x + bw * .32}" y="${hgt - pad.b - bh - 4}" text-anchor="middle">${esc(it.v == null ? '–' : fmt(it.v))}</text><text x="${x + bw * .32}" y="${hgt - 14}" text-anchor="middle">${esc(it.label)}</text>`; }).join('');
    return `<div class="chart"><svg viewBox="0 0 ${w} ${hgt}"><line class="axis" x1="${pad.l}" x2="${w - pad.r}" y1="${hgt - pad.b}" y2="${hgt - pad.b}"/>${g}</svg></div>`;
  },
  line(series, { w = 640, hgt = 220, xLabel = '', yLabel = '', yMax = 1, yFmt = v => pct(v, 0), xLog = false } = {}) {
    const all = series.flatMap(s => s.points); if (!all.length) return '<div class="empty small">No data yet.</div>';
    const pad = { l: 48, r: 14, t: 12, b: 32 }; const xv = (x) => xLog ? Math.log2(x) : x;
    const xs = all.map(p => xv(p.x)); const x0 = Math.min(...xs), x1 = Math.max(...xs) === x0 ? x0 + 1 : Math.max(...xs);
    const X = x => pad.l + ((xv(x) - x0) / (x1 - x0)) * (w - pad.l - pad.r), Y = y => hgt - pad.b - (y / yMax) * (hgt - pad.t - pad.b);
    let g = [0, .25, .5, .75, 1].map(f => `<line class="gridl" x1="${pad.l}" x2="${w - pad.r}" y1="${Y(f * yMax)}" y2="${Y(f * yMax)}"/><text x="${pad.l - 6}" y="${Y(f * yMax) + 3}" text-anchor="end">${esc(yFmt(f * yMax))}</text>`).join('');
    g += [...new Set(all.map(p => p.x))].map(x => `<text x="${X(x)}" y="${hgt - 14}" text-anchor="middle">${esc(String(x))}</text>`).join('');
    g += series.map(s => { const pts = [...s.points].sort((a, b) => a.x - b.x); return `<polyline fill="none" stroke="${s.color}" stroke-width="2" points="${pts.map(p => `${X(p.x)},${Y(p.y)}`).join(' ')}"/>` + pts.map(p => `<circle cx="${X(p.x)}" cy="${Y(p.y)}" r="4" fill="${s.color}"><title>${esc(s.label)} · ${esc(p.title || '')}</title></circle>`).join(''); }).join('');
    return `<div class="chart"><svg viewBox="0 0 ${w} ${hgt}" role="img"><line class="axis" x1="${pad.l}" x2="${w - pad.r}" y1="${hgt - pad.b}" y2="${hgt - pad.b}"/>${g}<text x="${(w + pad.l) / 2}" y="${hgt - 1}" text-anchor="middle">${esc(xLabel)}</text><text x="10" y="${hgt / 2}" transform="rotate(-90 10 ${hgt / 2})" text-anchor="middle">${esc(yLabel)}</text></svg></div>`;
  },
  frontier(pts, { w = 640, hgt = 260 } = {}) {
    if (!pts?.length) return '<div class="empty small">No data yet.</div>';
    const pad = { l: 48, r: 16, t: 14, b: 34 }; const xmax = Math.max(1, ...pts.map(p => p.mean_cost_s)) * 1.1;
    const X = x => pad.l + (x / xmax) * (w - pad.l - pad.r), Y = y => hgt - pad.b - y * (hgt - pad.t - pad.b);
    const eff = pts.filter(p => p.pareto).sort((a, b) => a.mean_cost_s - b.mean_cost_s);
    let g = [0, .25, .5, .75, 1].map(f => `<line class="gridl" x1="${pad.l}" x2="${w - pad.r}" y1="${Y(f)}" y2="${Y(f)}"/><text x="${pad.l - 6}" y="${Y(f) + 3}" text-anchor="end">${f * 100}%</text>`).join('');
    g += [0, .25, .5, .75, 1].map(f => `<text x="${X(f * xmax)}" y="${hgt - 16}" text-anchor="middle">${Math.round(f * xmax)}s</text>`).join('');
    if (eff.length > 1) g += `<polyline fill="none" stroke="var(--accent)" stroke-width="2" stroke-dasharray="5 3" points="${eff.map(p => `${X(p.mean_cost_s)},${Y(p.accuracy)}`).join(' ')}"/>`;
    g += pts.map((p, i) => `<g><circle cx="${X(p.mean_cost_s)}" cy="${Y(p.accuracy)}" r="6" fill="${p.pareto ? sysColor(p.system) : 'var(--surface)'}" stroke="${sysColor(p.system)}" stroke-width="2"><title>${esc(sysLabel(p.system))}: ${pct(p.accuracy, 0)} at ${num(p.mean_cost_s, 0)} s (n ${p.n})</title></circle>${(() => { const right = X(p.mean_cost_s) > w * 0.7; return `<text x="${X(p.mean_cost_s) + (right ? -9 : 9)}" y="${Y(p.accuracy) + (i % 2 ? 16 : -8)}" text-anchor="${right ? 'end' : 'start'}">${esc(shortLabel(p.system))}</text>`; })()}</g>`).join('');
    return `<div class="chart"><svg viewBox="0 0 ${w} ${hgt}" role="img" aria-label="Accuracy against mean cost per policy"><line class="axis" x1="${pad.l}" x2="${w - pad.r}" y1="${hgt - pad.b}" y2="${hgt - pad.b}"/>${g}<text x="${(w + pad.l) / 2}" y="${hgt - 1}" text-anchor="middle">mean wall time per task (cost)</text><text x="10" y="${hgt / 2}" transform="rotate(-90 10 ${hgt / 2})" text-anchor="middle">accuracy</text></svg></div>`;
  },
  reliability(pts, { w = 300, hgt = 260 } = {}) {
    const pad = 34; const S = (v) => pad + v * (w - pad - 10), Yv = (v) => hgt - pad - v * (hgt - pad - 10);
    const g = (pts || []).map(p => `<circle cx="${S(p.predicted)}" cy="${Yv(p.observed)}" r="${3 + Math.min(8, Math.sqrt(p.count))}" fill="var(--accent)" fill-opacity=".7"><title>pred ${num(p.predicted)} · obs ${num(p.observed)} · n=${p.count}</title></circle>`).join('');
    return `<div class="chart"><svg viewBox="0 0 ${w} ${hgt}"><rect x="${pad}" y="10" width="${w - pad - 10}" height="${hgt - pad - 10}" fill="none" class="axis" stroke="var(--border)"/>
      <line x1="${S(0)}" y1="${Yv(0)}" x2="${S(1)}" y2="${Yv(1)}" stroke="var(--faint)" stroke-dasharray="4 3"/>${g}
      <text x="${w / 2 + 12}" y="${hgt - 6}" text-anchor="middle">predicted ê₁ (fast-path error)</text><text x="10" y="${hgt / 2}" transform="rotate(-90 10 ${hgt / 2})" text-anchor="middle">observed error</text>
      ${[0, .5, 1].map(t => `<text x="${S(t)}" y="${hgt - 20}" text-anchor="middle">${t}</text><text x="${pad - 5}" y="${Yv(t) + 3}" text-anchor="end">${t}</text>`).join('')}</svg></div>`;
  },
};

// ============================== shell, router, status ==============================
const routes = {};
let cleanup = null;
function setTheme(t) { if (t) document.documentElement.dataset.theme = t; else delete document.documentElement.dataset.theme; store.set('theme', t || null); }
setTheme(store.get('theme', null));
$('#themeBtn').onclick = () => { const cur = document.documentElement.dataset.theme || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'); setTheme(cur === 'dark' ? 'light' : 'dark'); };
$('#tokenBtn').onclick = () => {
  modal(`<h3 style="margin-top:0">API token</h3><p class="muted small">Needed only when the gateway runs with <code>NAFS_AUTH_TOKENS</code>. Stored in this browser only.</p>
    <input id="tokIn" type="password" style="width:100%" placeholder="Bearer token" value="${esc(api.token())}"><div class="row" style="margin-top:10px"><button class="btn sm" id="tokSave" type="button">Save</button></div>`);
  $('#tokSave').onclick = () => { store.set('token', $('#tokIn').value.trim()); $('#modal').close(); toast('Token saved'); refreshStatus(); };
};
$('#backendPill').onclick = () => (location.hash = '#/system');
$('.brand').style.cursor = 'pointer'; $('.brand').title = 'Design notes: nafs and \u02bfaql'; $('.brand').onclick = designNotes;
async function refreshStatus() {
  try {
    const s = await api.get('/healthz');
    const pill = $('#backendPill'); const ok = s.status === 'ok';
    pill.querySelector('.dot').className = 'dot ' + (ok ? 'ok' : (s.backends.system1 === 'ready' ? 'warn' : 'bad'));
    pill.querySelector('.txt').textContent = `${s.backend === 'vllm' ? 'vLLM' : 'llama.cpp'} · S1 ${s.backends.system1 === 'ready' ? '●' : '○'} S2 ${s.backends.system2 === 'ready' ? '●' : '○'}`;
    pill.title = `Active backend: ${s.backend}. System 1 ${s.backends.system1}, System 2 ${s.backends.system2}. Click for details.`;
  } catch { const pill = $('#backendPill'); pill.querySelector('.dot').className = 'dot bad'; pill.querySelector('.txt').textContent = 'gateway offline'; }
}
setInterval(refreshStatus, 10000); refreshStatus();
let routeGen = 0;
async function route() {
  const name = (location.hash.replace(/^#\//, '').split('/')[0]) || 'chat';
  const my = ++routeGen; // a slower, older render must not overwrite a newer page (review L8)
  if (cleanup) { try { cleanup(); } catch {} cleanup = null; }
  $$('.rail a').forEach(a => a.classList.toggle('active', a.dataset.route === name));
  const r = routes[name] || routes.chat;
  $('#pageTitle').textContent = r.title; document.title = `${r.title} · Nafs-Aql`;
  const view = h('<section class="view"></section>'); view.id = 'view'; $('#view').replaceWith(view);
  const alive = () => my === routeGen;
  let c = null; try { c = await r.render(view, alive) || null; } catch (e) { if (alive()) view.innerHTML = `<div class="page"><div class="notice bad">${icon.alert}<span>${esc(e.message)}</span></div></div>`; }
  if (alive()) cleanup = c; else if (c) { try { c(); } catch {} }
}
window.addEventListener('hashchange', route);

// ============================== chat: one assistant turn ==============================
const STATE_NAMES = Object.fromEntries(Object.entries(STATIONS).map(([k, v]) => [k, v.en]));
const modeBadge = (m) => m === 'slow' ? `<span class="badge b-slow" title="System 2: deliberate reasoning">${icon.brain}Deliberate · System 2</span>` : `<span class="badge b-fast" title="System 1: fast draft">${icon.bolt}Fast · System 1</span>`;
const ASSURE = { verified: ['b-verified', icon.check, 'Verified', 'An exact verifier (tests, numeric reference or JSON Schema) accepted this answer.'],
  consensus: ['b-consensus', icon.users, 'Consensus', 'Independent samples agree, or the rubric judge accepted the answer. A soft check, not a proof (Theorems 1 and 6).'],
  unverified: ['b-unverified', '', 'Unverified', 'No exact verifier applies to this answer.'],
  abstained: ['b-abstained', icon.alert, 'Abstained', 'No candidate passed verification or the judge within the budget.'] };
const assureBadge = (a) => { const [c, i, l, t] = ASSURE[a] || ASSURE.unverified; return `<span class="badge ${c}" title="${esc(t)}">${i}${l}</span>`; };
const AVATAR = '<div class="avatar" aria-hidden="true"><svg viewBox="0 0 64 64"><circle cx="24" cy="32" r="9"/><circle cx="40" cy="32" r="9"/></svg></div>';

class Turn {
  constructor(container, { onSelect, onRetry, prompt, request } = {}) {
    this.t0 = performance.now(); this.events = []; this.fast = ''; this.slow = ''; this.reason = ''; this.states = ['A']; this.cur = 'A'; this.label = 'System 1 drafting';
    this.verifs = []; this.decision = null; this.reply = null; this.escalated = false; this.domain = null; this.error = null; this.prompt = prompt; this.request = request; this.onSelect = onSelect; this.onRetry = onRetry;
    this.el = h(`<div class="msg bot">${AVATAR}<div class="body"><div class="meta"></div><div class="automaton"></div><div class="why"></div><div class="folds"></div><div class="verifs"></div><div class="answer"></div><div class="notes"></div>
      <div class="actions"><button data-a="up" class="fb" title="Good answer">${icon.up}</button><button data-a="down" class="fb" title="Poor answer">${icon.down}</button><span class="sep"></span>
      <button data-a="inspect" title="Show the gate decision and trace">${icon.search}Inspect</button><button data-a="compare" title="Compare the System 1 draft with the final answer" hidden>${icon.compare}Compare</button><button data-a="copy">${icon.copy}Copy</button>
      <button data-a="regen" title="Ask the same question again">${icon.retry}Regenerate</button><button data-a="slow" title="Ask again with deliberation forced">${icon.brain}Think harder</button><button data-a="fast" title="Ask again with the fast path only">${icon.bolt}Fast only</button></div></div></div>`);
    this.el.querySelector('.actions').onclick = (e) => { const a = e.target.closest('button')?.dataset.a; if (!a) return;
      if (a === 'inspect') this.onSelect?.(this, true); if (a === 'copy') navigator.clipboard?.writeText(this.reply?.message?.content || this.answerText()).then(() => toast('Copied'));
      if (a === 'slow' || a === 'fast') this.onRetry?.(this.prompt, a, this.request);
      if (a === 'regen') this.onRetry?.(this.prompt, this.request?.force_mode || 'auto', this.request);
      if (a === 'compare') compareModal(this);
      if (a === 'up' || a === 'down') this.rate(a === 'up' ? 1 : -1); };
    this.el.querySelector('.body').addEventListener('dblclick', () => this.onSelect?.(this, true));
    container.appendChild(this.el); this.scheduled = false; this.render();
    this.ticker = setInterval(() => { if (this.reply || this.error || !this.el.isConnected) clearInterval(this.ticker); else this.schedule(); }, 500);
  }
  dispose() { clearInterval(this.ticker); this.onUpdate = null; }
  async rate(v) {
    if (!this.reply?.reply_id) return; const prev = this.feedback; this.feedback = prev === v ? prev : v; this.render();
    if (prev === v) return;
    try { await api.post(`/v1/replies/${this.reply.reply_id}/feedback`, { rating: v }); toast(v > 0 ? 'Thanks: marked as a good answer' : 'Thanks: marked as a poor answer. The ledger records it for review.'); }
    catch (e) { this.feedback = prev; this.render(); toast(e.message); }
  }
  log(kind, text) { this.events.push({ t: performance.now() - this.t0, kind, text }); }
  answerText() { return this.escalated ? this.slow : this.fast; }
  on(ev, data) {
    const d = data.data;
    switch (ev) {
      case 'start': this.domain = d.domain; this.backend = d.backend; this.log('start', `request on ${d.backend}, domain ${d.domain}`); break;
      case 'phase': this.cur = d.state; this.label = d.label; if (this.states[this.states.length - 1] !== d.state) this.states.push(d.state); this.log('phase', `${d.state}: ${d.label}`); break;
      case 'token': if (d.mode === 'fast') this.fast += d.text; else if (d.reasoning) this.reason += d.text; else this.slow += d.text; if (!this.firstToken) { this.firstToken = performance.now() - this.t0; this.log('token', 'first token'); } break;
      case 'decision': this.decision = d; this.escalated = d.escalate; this.log('decision', `${d.escalate ? 'escalate' : 'keep'} (${d.rule}) — ${d.explanation}`); break;
      case 'verification': d.kind = this.decision ? `attempt ${(this.attempts = (this.attempts || 0) + 1)}` : 'draft'; this.verifs.push(d); this.log('verify', `${d.result.verifier}: ${d.result.passed ? 'pass' : 'fail'}${d.result.evidence ? ' — ' + d.result.evidence.split('\n')[0] : ''}`); break;
      case 'retry': this.log('retry', `attempt rejected; retrying${d.evidence ? ' with feedback' : ''}`); if (this.slow) this.reason += `\n[previous candidate rejected]\n`; this.slow = ''; break;
      case 'degraded': this.log('degraded', d.reason); this.degraded = d.reason; break;
      case 'done': this.reply = d; this.states = d.states; this.cur = d.trace_state; this.decision = d.decision; this.escalated = d.reasoning_mode === 'slow'; this.log('done', `${d.reasoning_mode} · ${d.assurance} · ${fmtMs(d.usage.wall_ms)}`); break;
      case 'error': this.error = d.message; this.log('error', d.message); break;
    }
    this.schedule();
  }
  schedule() { if (this.scheduled) return; this.scheduled = true; requestAnimationFrame(() => { this.scheduled = false; this.render(); }); }
  render() {
    const r = this.reply, el = this.el; const live = !r && !this.error;
    const mode = r ? r.reasoning_mode : (this.escalated ? 'slow' : 'fast');
    el.querySelector('.meta').innerHTML = [modeBadge(mode), r ? assureBadge(r.assurance) : '', (r?.domain || this.domain) ? `<span class="badge b-domain">${esc(r?.domain || this.domain)}</span>` : '',
      r?.degraded ? `<span class="badge b-degraded">${icon.alert}Degraded</span>` : '',
      `<span class="time">${r ? `${fmtMs(r.usage.ttft_ms)} to first token · ${fmtMs(r.usage.wall_ms)} total` : live ? `<span class="spin"></span> ${fmtMs(performance.now() - this.t0)}` : ''}</span>`].join('');
    const order = ['A', 'L', 'M', 'T']; const final = r ? r.trace_state : null;
    const seq = (final === 'X' ? ['A', 'L', 'M', 'X'] : this.escalated || this.states.includes('L') ? order : ['A', 'T']);
    el.querySelector('.automaton').innerHTML = seq.map((s, i) => { const visited = this.states.includes(s); const cls = live && s === this.cur ? 'cur' : visited ? 'done' : ''; const st = STATIONS[s];
      return `${i ? '<span class="arrow">→</span>' : ''}<span class="s ${cls}" title="${esc(st.tip)}"><b>${s}</b>${st.en}<i class="ar">${st.ar}</i></span>`; }).join('') + (live ? `<span class="lbl">${esc(this.label)}</span>` : '');
    const d0 = this.decision; el.querySelector('.why').innerHTML = d0 && !live ? `<span title="${esc(d0.explanation)}">${d0.escalate ? icon.brain : icon.bolt} ${esc(d0.rule === 'forced_override' ? 'You chose the path.' : d0.explanation.length > 150 ? d0.explanation.slice(0, 150) + '…' : d0.explanation)}</span>` : '';
    const folds = el.querySelector('.folds'); let html = '';
    const draft = r ? r.draft : (this.escalated ? this.fast : null);
    if (draft) html += `<details class="fold draft"><summary>System 1 draft · ${r?.draft_conditioned ? 'shown to System 2 (coupling ablation)' : 'not used'}${this.decision ? ' — ' + esc(this.decision.explanation.slice(0, 90)) : ''}</summary><div class="fb">${esc(draft)}</div></details>`;
    const reasoning = r ? r.reasoning : this.reason;
    if (reasoning) { const ntok = r ? r.usage.reasoning_tokens : reasoning.length / 4 | 0; const wasOpen = folds.querySelector('.reason')?.open;
      html += `<details class="fold reason" ${wasOpen ?? (live) ? 'open' : ''}><summary>${icon.brain} Reasoning · ${r ? ntok + ' tokens' : '~' + ntok + ' tokens so far'}${r?.usage?.thinking_budget ? ` (budget ${r.usage.thinking_budget})` : ''}</summary><div class="fb">${esc(reasoning)}</div></details>`; }
    if (folds.innerHTML !== html) { const fb = folds.querySelector('.reason .fb'); const atBottom = fb && fb.scrollTop + fb.clientHeight >= fb.scrollHeight - 20; folds.innerHTML = html; const nfb = folds.querySelector('.reason .fb'); if (nfb && (atBottom || live)) nfb.scrollTop = nfb.scrollHeight; }
    el.querySelector('.verifs').innerHTML = this.verifs.map(v => `<span class="verif ${v.result.passed ? 'pass' : 'fail'}" title="${esc(v.result.evidence || 'passed')}">${v.result.passed ? icon.check : icon.x} ${esc(v.result.verifier)}${v.result.verifier === 'judge' && v.result.score != null ? ' V ' + num(v.result.score, 2) : ''} · ${v.kind || 'final'}</span>`).join('');
    const text = r ? r.message.content : this.answerText();
    const ans = el.querySelector('.answer');
    if (!text && live) ans.innerHTML = this.escalated ? `<span class="muted small"><span class="spin"></span> System 2 is deliberating…${this.reason ? '' : ' (reasoning hidden; enable "Show reasoning" to watch)'}</span>` : '<span class="caret"></span>';
    else ans.innerHTML = md(text) + (live ? '<span class="caret"></span>' : '');
    let notes = '';
    if (r?.retrieved?.length) notes += `<div class="sources">${icon.book}<span>Used ${r.retrieved.length} earlier verified answer${r.retrieved.length > 1 ? 's' : ''} as reference: ${r.retrieved.map(x => `<span class="src" title="${esc(x.question || '')}">${esc((x.question || '').slice(0, 60))}${(x.question || '').length > 60 ? '…' : ''} <span class="faint">(${pct(x.score, 0)} match)</span></span>`).join(' · ')}</span></div>`;
    if (r?.uncertainty_notice) notes += `<div class="notice ${r.assurance === 'abstained' ? 'bad' : ''}">${icon.alert}<span>${esc(r.uncertainty_notice)}</span></div>`;
    if (this.error) notes += `<div class="notice bad">${icon.alert}<span>${esc(this.error)}</span></div>`;
    el.querySelector('.notes').innerHTML = notes;
    el.querySelector('.actions').style.display = live ? 'none' : '';
    const acts = el.querySelector('.actions'); acts.querySelector('[data-a=compare]').hidden = !(r && r.draft);
    acts.querySelector('[data-a=up]').classList.toggle('on', this.feedback === 1); acts.querySelector('[data-a=down]').classList.toggle('on', this.feedback === -1);
    acts.querySelectorAll('.fb').forEach(b => { b.hidden = !r?.reply_id || !!r?.conversation_id === false; });
    this.onUpdate?.(this);
  }
  static fromStored(container, reply, opts, feedback) { const t = new Turn(container, opts); t.feedback = feedback ?? null; t.reply = reply; t.decision = reply.decision; t.escalated = reply.reasoning_mode === 'slow'; t.states = reply.states; t.domain = reply.domain; if (reply.verifier) t.verifs = [{ round: 0, result: reply.verifier }]; t.render(); return t; }
}

function compareModal(t) {
  const r = t.reply; if (!r?.draft) return;
  const words = (x) => new Set(String(x).toLowerCase().match(/[\p{L}\p{N}]+/gu) || []);
  const a = words(r.draft), b = words(r.message.content); const inter = [...a].filter(w => b.has(w)).length; const jac = inter / Math.max(1, new Set([...a, ...b]).size);
  const dv = r.decision?.features?.draft_verified;
  modal(`<h3 style="margin-top:0">${icon.compare} System 1 draft and System 2 answer</h3>
    <p class="small muted">Word overlap ${pct(jac, 0)} · draft ${dv == null ? 'not checked by an exact verifier' : dv ? 'passed its verifier' : 'failed its verifier'} · final ${esc(r.assurance)}${r.draft_conditioned ? ' · System 2 saw this draft (coupling ablation)' : ' · System 2 never saw the draft (FR-023)'}</p>
    <div class="cmp"><div><div class="cmp-h">${icon.bolt} Draft · <i>nafs</i></div><div class="answer">${md(r.draft)}</div></div>
    <div><div class="cmp-h">${icon.brain} Final · <i>\u02bfaql</i></div><div class="answer">${md(r.message.content)}</div></div></div>`);
}
function judgeBars(d) {
  const row = (k, l, v, floor) => `<div class="gauge"><div class="lbl"><span>${l}</span><span>${num(v, 2)}${floor != null ? ` <span class="faint">(floor ${floor})</span>` : ''}</span></div><div class="bar"><i style="width:${Math.min(100, (v || 0) * 100)}%;background:${floor != null && v < floor ? 'var(--bad)' : 'var(--ok)'}"></i>${floor != null ? `<span class="thr" style="left:${floor * 100}%"></span>` : ''}</div></div>`;
  return `<div style="margin-top:6px">${row('h', 'Honesty', d.honesty, 0.5)}${row('a', 'Harm-avoidance', d.harm_avoidance, 0.5)}${row('u', 'Helpfulness', d.helpfulness, 0.5)}${row('m', 'Grounding', d.grounding)}${row('V', 'V = ½·min(h,a) + ¼u + ¼m', d.V, 0.7)}${d.reason ? `<p class="tiny muted">${esc(d.reason)}</p>` : ''}</div>`;
}
document.addEventListener('click', async (e) => {
  const dl = e.target.closest('a[data-dl]');
  if (dl) { e.preventDefault(); try { const r = await fetch(dl.getAttribute('href'), { headers: api.headers() }); if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const name = (r.headers.get('content-disposition') || '').match(/filename="([^"]+)"/)?.[1] || 'export'; const a = document.createElement('a'); a.href = URL.createObjectURL(await r.blob()); a.download = name; a.click(); } catch (err) { toast(err.message); } return; }
  const b = e.target.closest('[data-decision]'); if (!b) return; e.preventDefault();
  try { const d = await api.get('/v1/decisions/' + b.dataset.decision); modal(`<h3 style="margin-top:0">Decision record</h3><div class="codebox" style="max-height:60vh;overflow:auto">${esc(JSON.stringify(d, null, 2))}</div>`); } catch (err) { toast(err.message); }
});
// P7-T01: where the time went. Stage colours and a stacked bar (samples drawn with the draft overlap it, so they are listed, not stacked; samples drawn after it, sample_after, are stacked).
const STAGE = { retrieval: ['Retrieval', 'var(--faint)'], draft: ['System 1 draft', 'var(--fast)'], sample: ['Cheap samples', 'var(--leaf)'], sample_after: ['Cheap samples', 'var(--leaf)'], verify_draft: ['Verify draft', 'var(--navy)'], probe: ['Answer probe', 'var(--violet)'],
  deliberate: ['System 2 deliberation', 'var(--slow)'], verify: ['Verify', 'var(--navy)'], judge: ['Judge', 'var(--amber)'], overhead: ['Gateway and other', 'var(--surface-3)'] };
const stageBar = (parts) => { const tot = parts.reduce((a, p) => a + p.v, 0) || 1;
  return `<div class="stagebar">${parts.filter(p => p.v > 0).map(p => `<i style="width:${(p.v / tot * 100).toFixed(2)}%;background:${STAGE[p.k]?.[1] || 'var(--slate)'}" title="${esc(STAGE[p.k]?.[0] || p.k)} · ${p.label}"></i>`).join('')}</div>`; };
const engineLine = (s) => s.prefill_ms == null ? '' : `prefill ${fmtMs(s.prefill_ms)} for ${s.prompt_n ?? '?'} tok${s.cache_n ? ` (${s.cache_n} cached)` : ''} · decode ${s.decode_ms && s.tokens ? num(s.tokens / s.decode_ms * 1000, 1) + ' tok/s' : fmtMs(s.decode_ms)}${s.queue_ms ? ` · queued ${fmtMs(s.queue_ms)}` : ''}`;
function stagesHTML(u) { const S = (u?.stages || []); if (!S.length) return '';
  const main = S.filter(s => s.stage !== 'sample'); const sum = main.reduce((a, s) => a + s.ms, 0); const other = Math.max(0, (u.wall_ms || 0) - sum);
  const parts = [...main.map(s => ({ k: s.stage, v: s.ms, label: fmtMs(s.ms) })), { k: 'overhead', v: other, label: fmtMs(other) }];
  return `<div class="sec"><h3>Where the time went</h3>${stageBar(parts)}<table class="t mini-t">${S.map(s => `<tr><td><i class="dot" style="background:${STAGE[s.stage]?.[1] || 'var(--slate)'}"></i>${esc(STAGE[s.stage]?.[0] || s.stage)}${s.stage === 'sample' ? ' <span class="faint">(parallel)</span>' : ''}<div class="tiny faint">${engineLine(s)}${s.note ? `${engineLine(s) ? ' · ' : ''}${esc(s.note)}` : ''}</div></td><td class="num">${fmtMs(s.ms)}</td></tr>`).join('')}
    ${other > 0 ? `<tr><td><i class="dot" style="background:var(--surface-3)"></i>Gateway and other</td><td class="num">${fmtMs(other)}</td></tr>` : ''}</table></div>`; }
// A dropped stream (network suspended, gateway restarted) cancels the request on the server, so say so and what to do.
const streamError = (e) => /network|fetch|load failed|suspended|terminated/i.test(String(e?.message || e)) ? 'The connection to the gateway was interrupted, so this request was cancelled and its slot freed. Press Regenerate to ask again.' : (e?.message || String(e));
function inspectorHTML(t) {
  if (!t) return `<div class="sec"><h3>Inspector</h3><p class="muted small">Send a message or click <b>Inspect</b> on an answer to see how the Aql Gate decided, what each verifier said, and how the request moved through the automaton A → L → M → T/X.</p>
    <dl class="kv small"><dt>A</dt><dd>System 1 drafts (fast, no reasoning)</dd><dt>L</dt><dd>Gate or verifier flags the draft</dd><dt>M</dt><dd>System 2 deliberates, draft-blind, with a thinking budget</dd><dt>T</dt><dd>Accepted with an assurance level</dd><dt>X</dt><dd>Abstain with an uncertainty notice</dd></dl></div>`;
  const d = t.decision, r = t.reply, f = d?.features || {};
  const gainBar = d ? (() => { const gain = Math.max(0, (d.e1_hat - d.e2_hat)); const cost = d.lambda * d.delta_c / 60; const m = Math.max(.2, gain, cost) * 1.15;
    return `<div class="gauge"><div class="lbl"><span>expected benefit L·(ê₁−ê₂) = ${num(gain, 3)}</span><span>cost λ·Δc = ${num(cost, 3)}</span></div>
      <div class="bar"><i style="width:${gain / m * 100}%;background:${d.escalate ? 'var(--slow)' : 'var(--fast)'}"></i><span class="thr" style="left:${cost / m * 100}%" title="threshold"></span></div></div>`; })() : '';
  const bar = (label, v, color) => `<div class="gauge"><div class="lbl"><span>${label}</span><span>${v == null ? '–' : num(v, 3)}</span></div><div class="bar"><i style="width:${Math.min(100, (v || 0) * 100)}%;background:${color}"></i></div></div>`;
  return `<div class="sec"><h3>Decision</h3>${d ? `<div class="row" style="margin-bottom:6px">${d.escalate ? `<span class="badge b-slow">${icon.brain}Escalated</span>` : `<span class="badge b-fast">${icon.bolt}Kept</span>`}<span class="badge b-domain">rule: ${esc(d.rule)}</span></div>
      <p class="small" style="margin:6px 0 10px">${esc(d.explanation)}</p>${f.draft_tokens ? bar('ê₁ fast-path error (estimate)', d.e1_hat, 'var(--crimson)') : '<p class="tiny faint">ê₁: n/a (no System 1 draft on this request)</p>'}${bar('ê₂ error after deliberation', d.e2_hat, 'var(--teal)')}${gainBar}
      <p class="tiny faint">Theorem 2: escalate iff L·(ê₁−ê₂) &gt; λ·Δc. Calibration: ${esc(d.calibration_version || '–')}.</p>` : '<p class="muted small">Waiting for the gate…</p>'}</div>
    <div class="sec"><h3>Gate features</h3><dl class="kv small"><dt>Domain</dt><dd>${esc(f.domain || t.domain || '–')}</dd><dt>Mean log-prob</dt><dd>${num(f.mean_logprob, 3)}</dd><dt>Min log-prob</dt><dd>${num(f.min_logprob, 2)}</dd><dt>Lowest window</dt><dd>${num(f.low_window_logprob, 3)}</dd>
      <dt>Sample agreement</dt><dd>${f.agreement == null ? '–' : pct(f.agreement, 0)} (k = ${f.k ?? 0})</dd><dt>Verifiable</dt><dd>${f.verifiable ? 'yes' : 'no'}${f.draft_verified != null ? ` · draft ${f.draft_verified ? 'passed' : 'failed'}` : ''}</dd>
      <dt>Draft tokens</dt><dd>${f.draft_tokens ?? '–'}</dd><dt>λ (speed–care)</dt><dd>${d ? num(d.lambda, 2) : '–'}</dd><dt>Δc (expected extra)</dt><dd>${d ? r4(d.delta_c) + ' s' : '–'}</dd></dl></div>
    ${r ? `<div class="sec"><h3>Usage</h3><dl class="kv small"><dt>Wall time</dt><dd>${fmtMs(r.usage.wall_ms)}</dd><dt>First token</dt><dd>${fmtMs(r.usage.ttft_ms)}</dd><dt>Model calls</dt><dd>${r.usage.model_calls} <span class="faint">(Lemma 1 bound)</span></dd>
      <dt>Verifier calls</dt><dd>${r.usage.verifier_calls}</dd><dt>Tokens</dt><dd>${r.usage.prompt_tokens} in · ${r.usage.completion_tokens} out</dd><dt>Reasoning</dt><dd>${r.usage.reasoning_tokens}${r.usage.thinking_budget ? ' / ' + r.usage.thinking_budget + ' budget' : ''}</dd>
      <dt>States</dt><dd class="mono">${esc(r.states.join(' → '))}</dd><dt>Backend</dt><dd>${esc(d?.backend || '')}</dd><dt>Decision</dt><dd><button class="linkbtn mono tiny" data-decision="${esc(r.decision_id)}">${esc(r.decision_id.slice(0, 8))}…</button></dd></dl></div>${stagesHTML(r.usage)}` : ''}
    ${t.verifs.length ? `<div class="sec"><h3>Verifier results</h3>${t.verifs.map((v) => `<div style="margin-bottom:8px"><span class="verif ${v.result.passed ? 'pass' : 'fail'}">${v.result.passed ? icon.check : icon.x} ${esc(v.result.verifier)} · ${v.kind || 'final'}</span>${v.result.rates ? `<span class="tiny faint"> r ${num(v.result.rates.r, 3)} · φ ${num(v.result.rates.phi, 3)}${v.result.rates.phi_upper95 != null ? ' (≤ ' + num(v.result.rates.phi_upper95, 3) + ')' : ''}</span>` : ''}
      ${v.result.verifier === 'judge' && v.result.details ? judgeBars(v.result.details) : ''}${v.result.evidence ? `<div class="codebox" style="margin-top:4px">${esc(v.result.evidence)}</div>` : ''}</div>`).join('')}</div>` : ''}
    <div class="sec"><h3>Timeline</h3><ul class="timeline">${t.events.map(e => `<li><span class="tt">${(e.t / 1000).toFixed(1)}s</span><span>${esc(e.text)}</span></li>`).join('') || '<li><span></span><span class="muted">Loaded from history.</span></li>'}</ul></div>`;
}

// ============================== chat page ==============================
const EXAMPLES = [
  { t: 'Verified math', d: 'Word problem with a numeric reference: watch System 1 get checked.', prompt: 'A train leaves at 3:40 pm and arrives at 6:05 pm the same day. How many minutes is the trip?', v: { reference_answer: '145' } },
  { t: 'Code with tests', d: 'Python with unit tests run in the sandbox.', prompt: 'Write a Python function `run_length_encode(s)` that compresses consecutive repeats, e.g. "aaabcc" -> "a3b1c2".', v: { tests: ['assert run_length_encode("aaabcc") == "a3b1c2"', 'assert run_length_encode("") == ""', 'assert run_length_encode("x") == "x1"'] } },
  { t: 'JSON with a schema', d: 'Structured output validated against JSON Schema.', prompt: 'Create a JSON object for a library book with title, author, year (integer 1450-2026) and available (boolean).', v: { json_schema: { type: 'object', required: ['title', 'author', 'year', 'available'], properties: { title: { type: 'string' }, author: { type: 'string' }, year: { type: 'integer', minimum: 1450, maximum: 2026 }, available: { type: 'boolean' } }, additionalProperties: false } } },
  { t: 'Quick fact', d: 'Easy question: the gate should keep the fast draft.', prompt: 'What is the capital of Indonesia?' },
  { t: 'Trick question', d: 'Classic intuition trap with a reference answer.', prompt: 'A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost in dollars?', v: { reference_answer: '0.05' } },
  { t: 'Open question', d: 'No exact verifier applies; the answer stays unverified.', prompt: 'Give three practical tips for writing clear technical documentation.' },
];

function settingsBar() {
  const s = store.get('chatSettings', { mode: 'auto', lambda: 0.2, budget: 0, reason: true });
  return { s, save() { store.set('chatSettings', s); } };
}

routes.chat = { title: 'Chat', async render(view) {
  const cfg = settingsBar(); const S = cfg.s;
  view.innerHTML = `<div class="chat noinspect" id="chatGrid">
    <aside class="convs"><div class="head" style="flex-direction:column;align-items:stretch;gap:8px"><button class="btn" id="newConv" style="justify-content:center" title="New chat (Ctrl/Cmd+K)">${icon.plus}New chat</button>
      <input id="convSearch" type="search" placeholder="Search conversations" aria-label="Search conversations"></div><div class="list" id="convList"></div></aside>
    <div class="thread"><div class="mbar"><button class="btn sm ghost" id="convsToggle" type="button" aria-label="Show conversations"><svg viewBox="0 0 24 24"><path d="M4 6h16M4 12h16M4 18h16"/></svg>Chats</button><span class="sp"></span><button class="btn sm" id="mbNew" type="button">${icon.plus}New chat</button></div><div class="threadbar" id="threadBar" hidden><b id="tbTitle" class="t1"></b><span class="sp"></span>
        <label class="toggle small" title="SR-004: when on, this conversation may be used for evaluation and, after redaction and review, for training"><input type="checkbox" id="tbResearch"> Use for research</label>
        <button class="btn sm ghost" id="tbExport" title="Download this conversation as JSON">${icon.download}Export</button></div>
      <div class="msgs" id="msgs"><div class="inner" id="inner"></div></div>
      <div class="composer"><div class="inner">
        <div id="verifyPanel" class="verify-panel" hidden>
          <div class="row" style="margin-bottom:8px"><b class="small">Verification inputs</b><span class="hint">Make the task verifiable: the answer is checked by an exact verifier (Theorem 1).</span><span class="sp"></span><button class="btn ghost sm" id="vClear" type="button">Clear</button></div>
          <div class="seg" id="vTabs"><button data-t="ref" class="on" type="button">Reference answer</button><button data-t="tests" type="button">Python tests</button><button data-t="schema" type="button">JSON Schema</button></div>
          <div style="margin-top:8px"><input id="vRef" type="text" placeholder="e.g. 145" style="width:100%"><textarea id="vTests" rows="4" placeholder="assert f(2) == 4&#10;assert f(0) == 0" hidden></textarea><textarea id="vSchema" rows="5" placeholder='{"type":"object","required":["name"]}' hidden></textarea></div>
        </div>
        <div class="box"><textarea id="input" rows="1" placeholder="Ask anything… (Enter to send, Shift+Enter for a new line)" aria-label="Message"></textarea>
          <button class="btn send" id="sendBtn" title="Send">${icon.send}</button></div>
        <div class="controls">
          <div class="grp"><span>Path</span><div class="seg" id="modeSeg"><button data-m="auto" title="The Aql Gate decides">Auto</button><button data-m="fast" title="System 1 only">Fast</button><button data-m="slow" title="Always deliberate">Deliberate</button><button data-m="both" title="Lab trial: System 1 and System 2 answer the same question side by side, and you choose the better answer">Both</button></div></div>
          <div class="grp slider" title="λ: price of extra latency. Lower = more care (escalate more), higher = more speed."><span>Care</span><input type="range" id="lam" min="0" max="0.8" step="0.05"><span>Speed</span><span class="mono" id="lamV"></span></div>
          <div class="grp opts"><button class="btn ghost sm" id="optBtn" type="button" aria-haspopup="true" aria-expanded="false" title="Thinking budget and reasoning display">Options ▾</button>
            <div class="pop" id="optPop" hidden><label class="f">Thinking budget<select id="budget" title="Thinking budget for System 2. 'Per domain' uses the configured or measured budget for the question's domain."><option value="0">per domain</option><option value="256">256</option><option value="512">512</option><option value="1024">1024</option><option value="2048">2048</option></select></label>
              <label class="toggle"><input type="checkbox" id="showReason"> Show System 2 reasoning</label>
              <label class="toggle" title="FR-061: reuse earlier verified answers, quoted as data with their source"><input type="checkbox" id="useMemory"> Use verified memory</label><p class="tiny faint" style="margin:6px 0 0">Care (λ) is the price of extra latency: lower means the gate deliberates more often.</p></div></div>
          <span class="chip" id="verifyChip" title="Attach a reference answer, tests or a JSON Schema">${icon.shield}<span>Verify</span></span>
          <span class="sp"></span><button class="btn ghost sm" id="inspBtn" title="Toggle inspector (Ctrl/Cmd+I)">${icon.search}Inspector</button>
        </div><div class="privacy">Conversations may be used, after redaction, for evaluation and training. Turn off <b>Use for research</b> to exclude one. Press <kbd>?</kbd> for shortcuts.</div></div></div></div>
    <aside class="inspector" id="insp"></aside></div>`;
  const inner = $('#inner'), msgs = $('#msgs'), input = $('#input'), grid = $('#chatGrid');
  let convId = store.get('conv', null); let busy = null; let selected = null; let turns = [];
  const insp = $('#insp'); const narrow = () => window.innerWidth < 1250;
  const showInspector = (on, persist = true) => { grid.classList.toggle('noinspect', !on); grid.classList.toggle('showinspect', on && narrow()); if (persist && !narrow()) store.set('inspector', on); };
  showInspector(!narrow() && store.get('inspector', true), false);
  const select = (t, explicit = false) => { selected = t; turns.forEach(x => x.el.classList.toggle('selected', x === t)); insp.innerHTML = inspectorHTML(t); if (explicit && narrow()) showInspector(true, false); };
  $('#inspBtn').onclick = () => showInspector(grid.classList.contains('noinspect'));
  // Phones: the conversation list is a drawer behind the "Chats" button; picking a conversation closes it.
  const closeConvs = () => grid.classList.remove('convs-open');
  $('#convsToggle').onclick = () => grid.classList.toggle('convs-open');
  $('#mbNew').onclick = () => { closeConvs(); newConv(); };
  grid.addEventListener('click', (e) => { if (e.target === grid || e.target.closest('#convList .conv, #newConv')) setTimeout(closeConvs, 0); });
  // settings
  const segBtns = $$('#modeSeg button'); const setMode = (m) => { S.mode = m; segBtns.forEach(b => b.classList.toggle('on', b.dataset.m === m)); cfg.save(); };
  segBtns.forEach(b => b.onclick = () => setMode(b.dataset.m)); setMode(S.mode);
  $('#lam').value = S.lambda; $('#lamV').textContent = num(S.lambda, 2); $('#lam').oninput = (e) => { S.lambda = +e.target.value; $('#lamV').textContent = num(S.lambda, 2); cfg.save(); };
  $('#budget').value = String(S.budget); $('#budget').onchange = (e) => { S.budget = +e.target.value; cfg.save(); };
  $('#showReason').checked = S.reason; $('#showReason').onchange = (e) => { S.reason = e.target.checked; cfg.save(); };
  $('#useMemory').checked = !!S.memory; $('#useMemory').onchange = (e) => { S.memory = e.target.checked; cfg.save(); };
  const optPop = $('#optPop'); $('#optBtn').onclick = (e) => { e.stopPropagation(); optPop.hidden = !optPop.hidden; $('#optBtn').setAttribute('aria-expanded', String(!optPop.hidden)); };
  optPop.onclick = (e) => e.stopPropagation(); const closePop = () => { optPop.hidden = true; $('#optBtn')?.setAttribute('aria-expanded', 'false'); }; document.addEventListener('click', closePop);
  // verification panel
  let vtab = 'ref'; const vp = $('#verifyPanel');
  const setTab = (t) => { vtab = t; $$('#vTabs button').forEach(b => b.classList.toggle('on', b.dataset.t === t)); $('#vRef').hidden = t !== 'ref'; $('#vTests').hidden = t !== 'tests'; $('#vSchema').hidden = t !== 'schema'; };
  $$('#vTabs button').forEach(b => b.onclick = () => setTab(b.dataset.t));
  const verification = () => { const v = {}; const ref = $('#vRef').value.trim(); if (ref) v.reference_answer = ref; const tests = $('#vTests').value.split('\n').map(x => x.trimEnd()).filter(Boolean); if (tests.length) v.tests = tests;
    const sc = $('#vSchema').value.trim(); if (sc) { try { v.json_schema = JSON.parse(sc); } catch { throw new Error('JSON Schema is not valid JSON'); } } return Object.keys(v).length ? v : null; };
  const updateChip = () => { let v = null; try { v = verification(); } catch {} const c = $('#verifyChip'); c.classList.toggle('on', !!v); c.querySelector('span').textContent = v ? `Verifiable: ${Object.keys(v).map(k => ({ reference_answer: 'reference', tests: 'tests', json_schema: 'schema' }[k])).join(' + ')}` : 'Verify'; };
  $('#verifyChip').onclick = () => { vp.hidden = !vp.hidden; }; ['vRef', 'vTests', 'vSchema'].forEach(id => $('#' + id).addEventListener('input', updateChip));
  $('#vClear').onclick = () => { $('#vRef').value = ''; $('#vTests').value = ''; $('#vSchema').value = ''; updateChip(); };
  const fillVerification = (v) => { $('#vRef').value = v?.reference_answer || ''; $('#vTests').value = (v?.tests || []).join('\n'); $('#vSchema').value = v?.json_schema ? JSON.stringify(v.json_schema, null, 1) : '';
    if (v) { vp.hidden = false; setTab(v.reference_answer ? 'ref' : v.tests ? 'tests' : 'schema'); } updateChip(); };
  // input
  const autosize = () => { input.style.height = 'auto'; input.style.height = Math.min(220, input.scrollHeight) + 'px'; };
  input.addEventListener('input', autosize);
  input.addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(); } });
  $('#sendBtn').onclick = () => busy ? busy.abort() : send();
  const setBusy = (ctrl) => { busy = ctrl; const b = $s('#sendBtn'); b.innerHTML = ctrl ? icon.stop : icon.send; b.title = ctrl ? 'Stop' : 'Send'; b.classList.toggle('danger', !!ctrl); };
  const scrollDown = () => { msgs.scrollTop = msgs.scrollHeight; };
  const welcome = () => { inner.innerHTML = `<div class="welcome"><h2>Fast when it can, careful when it must.</h2><p>System 1 drafts instantly. The Aql Gate decides whether System 2 should deliberate. Exact verifiers decide what counts as verified.</p>
      <div class="examples">${EXAMPLES.map((x, i) => `<div class="ex" data-i="${i}" tabindex="0"><b>${esc(x.t)}</b><span>${esc(x.d)}</span></div>`).join('')}</div>
      <div class="how">${['A', 'L', 'M', 'T'].map(k => `<span class="st" title="${esc(STATIONS[k].tip)}"><b>${k}</b> ${STATIONS[k].en.toLowerCase()} <i>${STATIONS[k].ar}</i></span>`).join('→')}<span class="faint">or</span><span class="st" title="${esc(STATIONS.X.tip)}"><b>X</b> abstain <i>${STATIONS.X.ar}</i></span></div>
      <p class="tiny faint" style="margin-top:10px"><button class="linkbtn" id="dnBtn">Design notes</button> · the vocabulary of the self is used as an analogy</p></div>`;
    $('#dnBtn') && ($('#dnBtn').onclick = designNotes);
    $$('.ex', inner).forEach(el => { const go = () => { const x = EXAMPLES[+el.dataset.i]; input.value = x.prompt; autosize(); fillVerification(x.v || null); input.focus(); }; el.onclick = go; el.onkeydown = (e) => e.key === 'Enter' && go(); }); };
  const addUser = (text, who) => { const el = h(`<div class="msg user">${who ? `<div class="who">${esc(who)}</div>` : ''}</div>`); el.appendChild(document.createTextNode(text)); inner.appendChild(el); };
  let convCache = [];
  $('#convSearch').addEventListener('input', () => renderConvs());
  async function loadConvs() { try { convCache = await api.get('/v1/conversations'); renderConvs(); } catch {} }
  function renderConvs() {
    { const q = ($('#convSearch')?.value || '').toLowerCase(); const list = convCache.filter(c => !q || c.title.toLowerCase().includes(q)); const L = $('#convList');
      L.innerHTML = list.length ? list.map(c => `<div class="conv ${c.conversation_id === convId ? 'on' : ''}" data-id="${c.conversation_id}"><span class="t1" title="${esc(c.title)}">${esc(c.title)}</span><span class="tiny faint">${ago(c.updated_at)}</span><button class="x" title="Export as JSON" data-exp="${c.conversation_id}">⤓</button><button class="x" title="Delete" data-del="${c.conversation_id}">×</button></div>`).join('') : `<div class="empty small">${q ? 'No conversation matches your search.' : 'No conversations yet.'}</div>`;
      $$('.conv', L).forEach(el => el.onclick = async (e) => { if (e.target.dataset.exp) { const c = await api.get('/v1/conversations/' + e.target.dataset.exp);
          const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([JSON.stringify(c, null, 2)], { type: 'application/json' })); a.download = `nafs-aql-${c.conversation_id.slice(0, 8)}.json`; a.click(); return; }
        if (e.target.dataset.del) { const id = e.target.dataset.del; const title = el.querySelector('.t1')?.textContent || 'this conversation';
          if (await confirmDialog({ title: 'Delete this conversation?', ok: 'Delete', body: `<p>“${esc(title)}” and its messages are removed for good. This cannot be undone.</p><p class="small muted">Use the export button first to keep a copy.</p>` })) { await api.del('/v1/conversations/' + id); if (convId === id) newConv(); loadConvs(); } return; } openConv(el.dataset.id); });
    }
  }
  const showBar = (c) => { const bar = $s('#threadBar'); if (!c) { bar.hidden = true; return; } bar.hidden = false; $s('#tbTitle').textContent = c.title; $s('#tbResearch').checked = !c.training_opt_out; };
  $('#tbResearch').onchange = async (e) => { if (!convId) return; try { await api.patch('/v1/conversations/' + convId, { training_opt_out: !e.target.checked }); toast(e.target.checked ? 'This conversation may be used for research' : 'Excluded from evaluation exports and training'); } catch (x) { toast(x.message); e.target.checked = !e.target.checked; } };
  $('#tbExport').onclick = async () => { if (!convId) return; const c = await api.get('/v1/conversations/' + convId); const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([JSON.stringify(c, null, 2)], { type: 'application/json' })); a.download = `nafs-aql-${c.conversation_id.slice(0, 8)}.json`; a.click(); };
  function newConv() { convId = null; store.set('conv', null); turns.forEach(t => t.dispose()); turns = []; select(null); welcome(); showBar(null); $$('.conv').forEach(c => c.classList.remove('on')); input.focus(); }
  async function openConv(id) {
    try { const c = await api.get('/v1/conversations/' + id); convId = id; store.set('conv', id); inner.innerHTML = ''; turns = [];
      let lastUser = null;
      for (const m of c.messages) { if (m.role === 'user') { addUser(m.content); lastUser = m.content; } else if (m.reply) { const t = Turn.fromStored(inner, m.reply, { onSelect: select, onRetry: retry, prompt: lastUser, request: { verification: null } }, m.feedback); turns.push(t); } }
      if (!c.messages.length) welcome(); select(turns[turns.length - 1] || null); showBar(c); scrollDown(); loadConvs();
    } catch (e) { toast('Could not open conversation: ' + e.message); newConv(); }
  }
  $('#newConv').onclick = newConv;
  async function send(textOverride, modeOverride, reqOverride) {
    if (busy) return; const text = (textOverride ?? input.value).trim(); if (!text) return;
    let v; try { v = reqOverride ? reqOverride.verification : verification(); } catch (e) { toast(e.message); return; }
    if (inner.querySelector('.welcome')) inner.innerHTML = '';
    addUser(text); if (!textOverride) { input.value = ''; autosize(); if (v) { $('#vRef').value = ''; $('#vTests').value = ''; $('#vSchema').value = ''; vp.hidden = true; updateChip(); } }
    const mode = modeOverride || S.mode;
    if (mode === 'both') return trial(text, v);
    const body = { conversation_id: convId, messages: [{ role: 'user', content: text }], lambda: S.lambda, thinking_budget: S.budget > 0 ? S.budget : undefined, show_reasoning: S.reason,
      force_mode: mode === 'auto' ? null : mode, verification: v || undefined, retrieve: S.memory ? true : undefined };
    const turn = new Turn(inner, { onSelect: select, onRetry: retry, prompt: text, request: body }); turns.push(turn); select(turn);
    turn.onUpdate = (t) => { if (selected === t) insp.innerHTML = inspectorHTML(t); const near = msgs.scrollHeight - msgs.scrollTop - msgs.clientHeight < 160; if (near) scrollDown(); };
    scrollDown(); const ctrl = new AbortController(); setBusy(ctrl);
    try { await api.stream('/v1/chat/stream', body, (ev, data) => { if (ev === 'start' && data.data.conversation_id && !convId) { convId = data.data.conversation_id; store.set('conv', convId); showBar({ title: text.slice(0, 60), training_opt_out: false }); } turn.on(ev, data); }, ctrl.signal);
      if (!turn.reply && !turn.error) turn.on('error', { data: { message: 'The stream ended unexpectedly; the answer may still be saved, so reopen the conversation to check.' } }); }
    catch (e) { if (e.name === 'AbortError') { turn.on('error', { data: { message: 'Stopped.' } }); } else turn.on('error', { data: { message: streamError(e) } }); }
    finally { setBusy(null); loadConvs(); input.focus(); }
  }
  // Lab trial: both paths answer the same prompt (sequentially, to share the CPU fairly); the person
  // chooses the better answer, which estimates the value of deliberation where no verifier exists.
  async function trial(text, v) {
    const box = h(`<div class="trial"><div class="trial-h">${icon.compare}<b>Lab trial</b><span class="muted small">Both paths answer the same question. Choose the better answer to help measure when deliberation is worth its cost.</span></div>
      <div class="trial-cols"><div class="col" data-c="fast"><div class="col-h">${icon.bolt} System 1 · <i>nafs</i></div></div><div class="col" data-c="slow"><div class="col-h">${icon.brain} System 2 · <i>\u02bfaql</i></div></div></div><div class="vote"></div></div>`);
    inner.appendChild(box); scrollDown();
    const ctrl = new AbortController(); setBusy(ctrl); const replies = {};
    const run = async (m) => {
      const body = { messages: [{ role: 'user', content: text }], lambda: S.lambda, thinking_budget: S.budget > 0 ? S.budget : undefined, show_reasoning: S.reason, force_mode: m, verification: v || undefined, ephemeral: true, audit: false };
      const t = new Turn(box.querySelector(`[data-c=${m}]`), { onSelect: select, onRetry: retry, prompt: text, request: body }); turns.push(t);
      t.onUpdate = (x) => { if (selected === x) insp.innerHTML = inspectorHTML(x); };
      try { await api.stream('/v1/chat/stream', body, (ev, data) => t.on(ev, data), ctrl.signal); if (!t.reply && !t.error) t.on('error', { data: { message: 'The stream ended unexpectedly.' } }); }
      catch (e) { t.on('error', { data: { message: e.name === 'AbortError' ? 'Stopped.' : streamError(e) } }); }
      replies[m] = t.reply;
    };
    try { await run('fast'); if (!ctrl.signal.aborted) await run('slow'); } finally { setBusy(null); input.focus(); }
    const f = replies.fast, sl = replies.slow; const vote = box.querySelector('.vote');
    if (!f || !sl) { vote.innerHTML = '<span class="muted small">The trial did not complete, so there is nothing to compare.</span>'; return; }
    const extra = (sl.usage.wall_ms - f.usage.wall_ms) / 1000;
    const exact = f.verifier?.exact && sl.verifier?.exact ? ` · exact verifier: System 1 ${f.verifier.passed ? 'passed' : 'failed'}, System 2 ${sl.verifier.passed ? 'passed' : 'failed'}` : '';
    vote.innerHTML = `<span class="small">Which answer is better? <span class="faint">System 2 took ${extra >= 0 ? '+' : ''}${extra.toFixed(0)} s${esc(exact)}</span></span><span class="sp"></span>
      ${[['fast', 'System 1'], ['slow', 'System 2'], ['tie', 'About the same'], ['both_bad', 'Both poor']].map(([k, l]) => `<button class="btn sm ${k === 'tie' || k === 'both_bad' ? 'ghost' : ''}" data-p="${k}">${l}</button>`).join('')}`;
    vote.querySelectorAll('[data-p]').forEach(b => b.onclick = async () => {
      try { await api.post('/v1/preferences', { prompt: text, domain: sl.domain, fast_reply_id: f.reply_id, slow_reply_id: sl.reply_id, preferred: b.dataset.p });
        vote.innerHTML = `<span class="small">${icon.check} Recorded: ${esc(b.textContent)}. See the value of deliberation per domain on the Muhasaba page.</span>`; } catch (e) { toast(e.message); } });
  }
  function retry(prompt, mode, req) { if (!prompt) return toast('Original prompt not available'); send(prompt, mode, req || { verification: null }); }
  const keys = (e) => {
    const mod = e.metaKey || e.ctrlKey; const typing = /INPUT|TEXTAREA|SELECT/.test(document.activeElement?.tagName || '');
    if (mod && e.key.toLowerCase() === 'k') { e.preventDefault(); newConv(); }
    else if (mod && e.key.toLowerCase() === 'i') { e.preventDefault(); showInspector(grid.classList.contains('noinspect')); }
    else if (e.key === 'Escape' && busy) { busy.abort(); }
    else if (!typing && e.key === '/') { e.preventDefault(); input.focus(); }
    else if (!typing && e.key === '?') { modal(`<h3 style="margin-top:0">Keyboard shortcuts</h3><table class="t small"><tr><td><kbd>Enter</kbd></td><td>Send</td></tr><tr><td><kbd>Shift</kbd>+<kbd>Enter</kbd></td><td>New line</td></tr><tr><td><kbd>Esc</kbd></td><td>Stop the answer</td></tr><tr><td><kbd>/</kbd></td><td>Focus the message box</td></tr><tr><td><kbd>Ctrl/Cmd</kbd>+<kbd>K</kbd></td><td>New chat</td></tr><tr><td><kbd>Ctrl/Cmd</kbd>+<kbd>I</kbd></td><td>Toggle the inspector</td></tr></table>`); }
  };
  document.addEventListener('keydown', keys);
  if (convId) await openConv(convId); else { welcome(); select(null); }
  loadConvs(); input.focus();
  // Navigating away stops the live view; the request is cancelled on the server when the stream closes.
  return () => { document.removeEventListener('keydown', keys); document.removeEventListener('click', closePop); turns.forEach(t => t.dispose()); if (busy) busy.abort(); };
} };

// ============================== rooms ==============================
// Shared chats over WebSocket (IF-004). The page keeps one connection to one room at a time. The server's
// `welcome` frame carries a snapshot (`cursor`, `pending`) so that, after any join or reconnect, the transcript
// is: the first `cursor` stored messages, then the pending questions, then live frames, with no overlap.
const ROOM_COLORS = ['#1B8A8F', '#6A4C9C', '#C97D10', '#B3263E', '#3A8F4B', '#2F6FD0', '#8A5A44', '#5B6770'];
const roomColor = (n) => { let x = 0; for (const c of String(n).toLowerCase()) x = (x * 31 + c.codePointAt(0)) >>> 0; return ROOM_COLORS[x % ROOM_COLORS.length]; };
const roomInitials = (n) => { const p = String(n).trim().split(/\s+/); return ((p[0]?.[0] || '?') + (p.length > 1 ? p[1][0] : '')).toUpperCase(); };
const roomAvatar = (n, me) => `<span class="rav${me ? ' me' : ''}" style="background:${roomColor(n)}" title="${esc(n)}${me ? ' (you)' : ''}">${esc(roomInitials(n))}</span>`;
const clock = (iso) => { const d = new Date(iso); return Number.isNaN(+d) ? '' : d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }); };
const dayLabel = (iso) => { const d = new Date(iso), t = new Date(), y = new Date(t); y.setDate(t.getDate() - 1);
  return d.toDateString() === t.toDateString() ? 'Today' : d.toDateString() === y.toDateString() ? 'Yesterday' : d.toLocaleDateString([], { weekday: 'short', day: 'numeric', month: 'short', year: d.getFullYear() === t.getFullYear() ? undefined : 'numeric' }); };
const cleanRoomName = (s) => String(s).replace(/[\u0000-\u001f\u007f]/g, ' ').replace(/\s+/g, ' ').trim().slice(0, 80);
const cleanUserName = (s) => String(s).replace(/[^\p{L}\p{N}_\- ]/gu, '').replace(/\s+/g, ' ').trim().slice(0, 32) || 'guest';
const plural = (n, w) => `${n} ${w}${n === 1 ? '' : 's'}`;
const people = (n) => n === 1 ? '1 person' : `${n} people`;
const downloadText = (name, text, type) => { const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([text], { type })); a.download = name; document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 2000); };
function confirmDialog({ title, body, ok = 'Delete', danger = true }) {
  return new Promise((resolve) => {
    modal(`<h3 style="margin-top:0">${esc(title)}</h3>${body}<div class="row" style="justify-content:flex-end;margin-top:14px"><button class="btn ghost" id="cdNo" type="button">Cancel</button><button class="btn ${danger ? 'danger' : ''}" id="cdYes" type="button">${esc(ok)}</button></div>`);
    const dlg = $('#modal'); dlg.classList.add('confirm'); let done = false; const fin = (v) => { if (done) return; done = true; if (dlg.open) dlg.close(); dlg.classList.remove('confirm'); resolve(v); };
    $('#cdYes').onclick = () => fin(true); $('#cdNo').onclick = () => fin(false); dlg.addEventListener('close', () => fin(false), { once: true }); $('#cdNo').focus();
  });
}
// Stored messages -> display items. A room question is stored as "[Name] text".
function roomItems(messages) {
  return messages.map((m) => { if (m.role === 'user') { const mm = m.content.match(/^\[([^\]]+)\] ([\s\S]*)$/); return { kind: 'q', who: mm ? mm[1] : '', text: mm ? mm[2] : m.content, ts: m.created_at }; }
    return m.reply ? { kind: 'a', reply: m.reply, ts: m.created_at, feedback: m.feedback } : null; }).filter(Boolean);
}
function roomMarkdown(room, conv) {
  const items = roomItems(conv.messages); let out = `# ${room.name}\n\nExported ${new Date().toLocaleString()} · ${plural(items.filter(i => i.kind === 'q').length, 'question')}\n`; let day = '';
  for (const i of items) { const d = new Date(i.ts).toDateString(); if (d !== day) { day = d; out += `\n---\n\n### ${dayLabel(i.ts)}\n`; }
    if (i.kind === 'q') out += `\n**${i.who || 'Someone'}** (${clock(i.ts)}):\n\n${i.text}\n`;
    else { const r = i.reply; out += `\n**Assistant** (${clock(i.ts)} · ${r.reasoning_mode === 'slow' ? 'deliberate, System 2' : 'fast, System 1'} · ${r.assurance}):\n\n${r.message?.content || ''}\n`; } }
  return out;
}

routes.rooms = { title: 'Rooms', async render(view, alive) {
  const stored = store.get('user', '') || `user-${Math.random().toString(36).slice(2, 6)}`; store.set('user', cleanUserName(stored));
  view.innerHTML = `<div class="rooms" id="rooms">
    <aside class="rooms-side" id="roomsSide" aria-label="Rooms">
      <div class="rooms-head">
        <div class="row"><b style="font-size:15px">Rooms</b><span class="tiny faint" id="roomCount"></span><span class="sp"></span><button class="btn sm" id="newRoomBtn" type="button">${icon.plus}New room</button></div>
        <form id="newRoomForm" class="rnew" hidden autocomplete="off"><div class="row" style="flex-wrap:nowrap"><input id="roomName" type="text" maxlength="80" placeholder="Room name" aria-label="New room name" style="flex:1"><button class="btn sm" type="submit" id="mkRoom">Create</button><button class="btn sm ghost" type="button" id="cancelNew" aria-label="Cancel">${icon.x}</button></div>
          <div class="rerr" id="roomNameErr" role="alert"></div></form>
        <input id="roomSearch" type="search" class="rsearch" placeholder="Search rooms" aria-label="Search rooms">
      </div>
      <div class="rooms-list" id="roomList" role="list"><div class="empty small">Loading rooms…</div></div>
      <div class="rooms-me"><label class="f">Your name<input id="userName" type="text" maxlength="32" value="${esc(store.get('user', ''))}" autocomplete="off"></label><div class="tiny faint" id="userHint">Shown to everyone in a room. Letters, numbers, spaces, - and _.</div></div>
    </aside>
    <section class="rooms-main">
      <header class="rooms-bar" id="roomBar"></header>
      <div class="rooms-banner" id="roomBanner" hidden></div>
      <div class="msgs rmsgs" id="rmsgs" role="log" aria-label="Room transcript"><div class="inner" id="rinner"></div><button class="rjump" id="jumpNew" type="button" hidden></button></div>
      <div class="rtyping tiny faint" id="rtyping" aria-live="polite"></div>
      <div class="composer"><div class="inner">
        <div class="box"><textarea id="rinput" rows="1" maxlength="9000" placeholder="Select or create a room to start" aria-label="Message the room" disabled></textarea><button class="btn send" id="rsend" type="button" aria-label="Send" disabled>${icon.send}</button></div>
        <div class="controls"><div class="grp"><span>Path</span><div class="seg" id="rmode"><button data-m="auto">Auto</button><button data-m="fast">Fast</button><button data-m="slow">Deliberate</button></div></div>
          <span class="hint" id="rhint">One assistant answer at a time per room; questions queue.</span><span class="sp"></span><span class="tiny" id="rcount"></span></div>
        <div class="rcnote tiny" id="rcnote" role="alert"></div></div></div>
      <div class="rsr" id="rsr" aria-live="polite"></div>
    </section></div>`;
  const root = $('#rooms'), rinner = $('#rinner'), rmsgs = $('#rmsgs'), input = $('#rinput');
  // ---------------- state ----------------
  const S = { rooms: [], listOk: true, filter: '', room: null, me: null, users: [], assistant: { busy: false, working_for: null, queued: 0 }, limits: { message_chars: 8000 },
    conn: 'idle', attempt: 0, retryAt: 0, retryTimer: null, ws: null, gen: 0, hyd: 0, hydrating: false, buffer: [], hist: [], histStart: 0, turns: new Map(), slots: new Map(),
    lastDay: null, follow: true, unseen: 0, typers: new Map(), drafts: new Map(), lastSent: '', lastTyping: 0, mode: store.get('roomMode', 'auto'), dead: false };
  const WINDOW = Math.max(2, +store.get('roomWindow', 60) || 60); // stored messages shown at first; the rest sit behind "Show earlier"
  const seenMap = () => store.get('roomSeen', {}); const markSeen = (r) => { if (!r) return; const m = seenMap(); if (m[r.room_id] !== r.last_activity) { m[r.room_id] = r.last_activity; store.set('roomSeen', m); } };
  const say = (t) => { $('#rsr').textContent = t; };
  const note = (t, ms = 6000) => { const el = $('#rcnote'); el.textContent = t; clearTimeout(note._t); if (t) note._t = setTimeout(() => { el.textContent = ''; }, ms); };
  const online = () => S.conn === 'open';
  // ---------------- room list ----------------
  const unread = (r) => { const s = seenMap()[r.room_id]; return S.room?.room_id !== r.room_id && s !== undefined && r.last_activity > s; };
  function renderList() {
    const q = S.filter.trim().toLowerCase(); const shown = S.rooms.filter(r => !q || r.name.toLowerCase().includes(q)); const L = $('#roomList');
    $('#roomCount').textContent = S.rooms.length ? (q ? `${shown.length} of ${S.rooms.length}` : String(S.rooms.length)) : '';
    if (!S.listOk && !S.rooms.length) { L.innerHTML = `<div class="empty small">Could not load rooms.<br><button class="btn sm ghost" id="retryList" type="button" style="margin-top:8px">Try again</button></div>`; $('#retryList').onclick = () => refreshList(); return; }
    if (!S.rooms.length) { L.innerHTML = `<div class="empty small"><b>No rooms yet.</b><br>A room is a shared chat: everyone in it sees each question and the assistant's answer as it is written.<br><button class="btn sm" id="firstRoom" type="button" style="margin-top:10px">${icon.plus}Create the first room</button></div>`; $('#firstRoom').onclick = openNewForm; return; }
    if (!shown.length) { L.innerHTML = `<div class="empty small">No room matches “${esc(S.filter)}”.<br><button class="btn sm ghost" id="clearSearch" type="button" style="margin-top:8px">Clear search</button></div>`; $('#clearSearch').onclick = () => { $('#roomSearch').value = ''; S.filter = ''; renderList(); }; return; }
    L.innerHTML = shown.map((r) => `<button class="room${S.room?.room_id === r.room_id ? ' on' : ''}${unread(r) ? ' unread' : ''}" type="button" role="listitem" data-id="${r.room_id}"${S.room?.room_id === r.room_id ? ' aria-current="true"' : ''}>
      ${roomAvatar(r.name)}<span class="rbody"><span class="rname">${esc(r.name)}</span><span class="rmeta">${r.members ? `<i class="dot ok"></i>${r.members} online · ` : ''}${r.questions ? plural(r.questions, 'question') : 'no questions yet'} · ${ago(r.last_activity)}</span></span>
      <span class="rflags">${r.busy ? '<i class="dot warn pulse" title="The assistant is answering"></i>' : ''}${unread(r) ? '<i class="rnew-dot" title="New activity"></i>' : ''}</span></button>`).join('');
    $$('#roomList .room').forEach(el => el.onclick = () => { join(el.dataset.id); root.classList.remove('side-open'); });
  }
  let listBusy = false;
  async function refreshList() {
    if (listBusy || S.dead) return; listBusy = true;
    try { const rooms = await api.get('/v1/rooms'); if (S.dead || !alive()) return; S.rooms = rooms; S.listOk = true; const m = seenMap(); let ch = false;
      for (const r of rooms) if (!(r.room_id in m)) { m[r.room_id] = r.last_activity; ch = true; } if (ch) store.set('roomSeen', m);
      if (S.room) { const cur = rooms.find(r => r.room_id === S.room.room_id); if (cur) { S.room = { ...S.room, ...cur }; markSeen(cur); } } }
    catch { S.listOk = false; } finally { listBusy = false; } if (!S.dead) renderList();
  }
  const refreshSoon = (() => { let t; return () => { clearTimeout(t); t = setTimeout(refreshList, 600); }; })();
  $('#roomSearch').oninput = (e) => { S.filter = e.target.value; renderList(); };
  // ---------------- creating a room ----------------
  function openNewForm() { $('#newRoomForm').hidden = false; $('#newRoomBtn').hidden = true; $('#roomNameErr').textContent = ''; $('#roomName').focus(); }
  function closeNewForm() { $('#newRoomForm').hidden = true; $('#newRoomBtn').hidden = false; $('#roomName').value = ''; $('#roomNameErr').textContent = ''; }
  $('#newRoomBtn').onclick = openNewForm; $('#cancelNew').onclick = closeNewForm;
  $('#roomName').addEventListener('keydown', (e) => { if (e.key === 'Escape') closeNewForm(); });
  $('#newRoomForm').onsubmit = async (e) => { e.preventDefault(); const err = $('#roomNameErr'); const name = cleanRoomName($('#roomName').value);
    if (!name) { err.textContent = 'Give the room a name.'; return; }
    const clash = S.rooms.find(r => r.name.toLowerCase() === name.toLowerCase()); if (clash) { err.innerHTML = `A room named “${esc(clash.name)}” exists. <button class="linkbtn" type="button" id="openClash">Open it</button>`; $('#openClash').onclick = () => { closeNewForm(); join(clash.room_id); }; return; }
    $('#mkRoom').disabled = true;
    try { const r = await api.post('/v1/rooms', { name }); closeNewForm(); await refreshList(); await join(r.room_id); input.focus(); }
    catch (x) { err.textContent = x.status === 409 ? 'A room with that name already exists.' : x.message; } finally { $('#mkRoom').disabled = false; } };
  // ---------------- header ----------------
  function chips() {
    const room = S.room; if (!room) return '';
    const conn = { open: ['ok', 'Connected'], connecting: ['warn', 'Connecting…'], reconnecting: ['warn', `Reconnecting… retry in ${Math.max(0, Math.ceil((S.retryAt - Date.now()) / 1000))} s`], closed: ['bad', 'Disconnected'], idle: ['warn', 'Connecting…'] }[S.conn];
    const a = S.assistant; const asst = a.busy ? `<span class="rchip"><i class="dot warn pulse"></i>Answering ${esc(a.working_for === S.me ? 'your' : (a.working_for || 'a') + '’s')} question${a.queued ? ` · ${a.queued} waiting` : ''}</span>` : a.queued ? `<span class="rchip"><i class="dot warn pulse"></i>${plural(a.queued, 'question')} waiting</span>` : `<span class="rchip"><i class="dot ok"></i>Assistant ready</span>`;
    const users = S.users.slice(0, 6).map(u => roomAvatar(u, u === S.me)).join('') + (S.users.length > 6 ? `<span class="rav more" title="${esc(S.users.slice(6).join(', '))}">+${S.users.length - 6}</span>` : '');
    return `<span class="rpresence" title="${esc(S.users.join(', '))}">${users}<span class="small muted">${online() ? people(S.users.length) + ' here' : ''}</span></span>
      <span class="rchip" role="status"><i class="dot ${conn[0]}"></i>${conn[1]}${S.conn === 'closed' || S.conn === 'reconnecting' ? ` <button class="linkbtn" type="button" id="reconnectBtn">${S.conn === 'closed' ? 'Reconnect' : 'Retry now'}</button>` : ''}</span>${S.conn === 'open' ? asst : ''}`;
  }
  function renderBar() {
    const room = S.room; const bar = $('#roomBar');
    if (!room) { bar.innerHTML = `<button class="btn ghost sm rooms-toggle" id="roomsToggle" type="button">${icon.users}Rooms</button><b>Rooms</b>`; $('#roomsToggle').onclick = () => root.classList.toggle('side-open'); return; }
    if ($('#renameInput')) { const st = $('#rstate'); if (st) st.innerHTML = chips(); bindBar(); return; }
    bar.innerHTML = `<button class="btn ghost sm rooms-toggle" id="roomsToggle" type="button">${icon.users}Rooms</button>
      <div class="rtitle"><h2 id="rtitle" title="${esc(room.name)}">${esc(room.name)}</h2><button class="ibtn" id="renameBtn" type="button" title="Rename room" aria-label="Rename room"><svg viewBox="0 0 24 24"><path d="M4 20h4L19 9l-4-4L4 16z"/></svg></button></div>
      <div class="rstate" id="rstate">${chips()}</div>
      <div class="ractions"><button class="btn ghost sm" id="linkBtn" type="button" title="Copy a link to this room" aria-label="Copy a link to this room">${icon.copy}<span class="lbl">Link</span></button>
        <details class="rmenu" id="exportMenu"><summary class="btn ghost sm" aria-label="Export the transcript" title="Export the transcript">${icon.download}<span class="lbl">Export</span></summary><div class="rmenu-pop"><button type="button" data-x="md">Markdown transcript</button><button type="button" data-x="json">JSON (full records)</button></div></details>
        <button class="btn ghost sm rdel" id="delBtn" type="button" title="Delete this room for everyone" aria-label="Delete this room for everyone">${icon.x}<span class="lbl">Delete</span></button></div>`;
    bindBar();
  }
  function bindBar() {
    const t = $('#roomsToggle'); if (t) t.onclick = () => root.classList.toggle('side-open');
    const rc = $('#reconnectBtn'); if (rc) rc.onclick = reconnectNow;
    const rb = $('#renameBtn'); if (rb) rb.onclick = startRename;
    const lb = $('#linkBtn'); if (lb) lb.onclick = copyLink; const db = $('#delBtn'); if (db) db.onclick = deleteRoom;
    $$('#exportMenu [data-x]').forEach(b => b.onclick = () => { $('#exportMenu').open = false; exportRoom(b.dataset.x); });
  }
  function startRename() {
    const room = S.room; const wrap = $('.rtitle'); if (!room || !wrap) return;
    wrap.innerHTML = `<input id="renameInput" type="text" maxlength="80" value="${esc(room.name)}" aria-label="Room name"><span class="rerr" id="renameErr" role="alert"></span>`;
    const inp = $('#renameInput'); inp.focus(); inp.select(); let busy = false;
    const finish = (rebuild = true) => { if (rebuild) { const keep = S.room; S.room = keep; $('#roomBar').innerHTML = ''; renderBar(); } };
    inp.onkeydown = async (e) => { if (e.key === 'Escape') { finish(); return; } if (e.key !== 'Enter' || busy) return; e.preventDefault();
      const name = cleanRoomName(inp.value); if (!name || name === room.name) { finish(); return; } busy = true;
      try { const r = await api.patch(`/v1/rooms/${room.room_id}`, { name }); S.room = { ...S.room, ...r }; finish(); updateComposer(); refreshList(); toast('Room renamed'); }
      catch (x) { busy = false; $('#renameErr').textContent = x.status === 409 ? 'Another room already has that name.' : x.status === 404 ? 'This room no longer exists.' : x.message; } };
    inp.onblur = () => setTimeout(() => { if ($('#renameInput') && !busy) finish(); }, 150);
  }
  async function copyLink() { const url = `${location.origin}${location.pathname}#/rooms/${S.room.room_id}`; try { await navigator.clipboard.writeText(url); toast('Link copied'); } catch { modal(`<h3 style="margin-top:0">Link to this room</h3><input type="text" readonly style="width:100%" value="${esc(url)}" onfocus="this.select()">`); } }
  async function exportRoom(kind) {
    const room = S.room; try { const conv = await api.get('/v1/conversations/' + room.conversation_id); const slug = room.name.toLowerCase().replace(/[^\p{L}\p{N}]+/gu, '-').replace(/^-|-$/g, '').slice(0, 40) || 'room'; const day = new Date().toISOString().slice(0, 10);
      if (kind === 'json') downloadText(`room-${slug}-${day}.json`, JSON.stringify({ room: { room_id: room.room_id, name: room.name, created_at: room.created_at }, conversation: conv }, null, 2), 'application/json');
      else downloadText(`room-${slug}-${day}.md`, roomMarkdown(room, conv), 'text/markdown'); toast('Transcript exported'); } catch (e) { toast(`Export failed: ${e.message}`, 4000); }
  }
  async function deleteRoom() {
    const room = S.room; if (!room) return; const others = Math.max(0, S.users.length - 1);
    const ok = await confirmDialog({ title: `Delete “${room.name}”?`, ok: 'Delete room', body: `<p>This permanently deletes the room and everything said in it: ${plural(room.questions || 0, 'question')} and the assistant's answers, <b>for everyone</b>.${others ? ` ${others === 1 ? '1 other person is' : others + ' other people are'} in the room now and will be disconnected.` : ''}</p><p class="small muted">Its saved decisions, feedback and retrieval-memory entries are deleted too. This cannot be undone. To keep a copy, cancel and use Export first.</p>` });
    if (!ok) return;
    try { const rep = await api.del(`/v1/rooms/${room.room_id}`); toast(`Deleted “${rep.name}” (${plural(rep.removed.messages, 'message')})`); leaveRoom(); }
    catch (e) { if (e.status === 404) { toast('That room was already deleted.'); leaveRoom(); } else toast(`Could not delete the room: ${e.message}`, 4500); }
  }
  function leaveRoom() { closeSocket(); clearTranscript(); S.room = null; S.me = null; S.users = []; S.conn = 'idle'; store.set('lastRoom', null); history.replaceState(null, '', '#/rooms'); banner(null); renderBar(); renderMainEmpty(); updateComposer(); refreshList(); $('#roomSearch').focus(); }
  // ---------------- banner and empty states ----------------
  function banner(kind, extra) {
    const b = $('#roomBanner'); if (!kind) { b.hidden = true; b.innerHTML = ''; return; } b.hidden = false;
    if (kind === 'deleted') b.innerHTML = `<div class="notice bad">${icon.alert}<span><b>This room was deleted.</b> Its messages are gone for everyone.</span><span class="sp"></span><button class="btn sm ghost" id="bnBack" type="button">Back to rooms</button></div>`;
    if (kind === 'load') b.innerHTML = `<div class="notice">${icon.alert}<span>Earlier messages could not be loaded (${esc(extra)}).</span><span class="sp"></span><button class="btn sm ghost" id="bnRetry" type="button">Retry</button></div>`;
    if (kind === 'full') b.innerHTML = `<div class="notice bad">${icon.alert}<span>${esc(extra)}</span><span class="sp"></span><button class="btn sm ghost" id="bnBack" type="button">Back to rooms</button></div>`;
    const back = $('#bnBack'); if (back) back.onclick = leaveRoom; const rt = $('#bnRetry'); if (rt) rt.onclick = () => reconnectNow();
  }
  function renderMainEmpty() {
    rinner.innerHTML = `<div class="rwelcome"><h3>Think together</h3><p>Rooms are shared chats. Everyone connected sees every question and the assistant's answer as it is written, and the transcript is kept until the room is deleted.</p>
      <ul><li>Pick a room on the left, or create one.</li><li>Share its link (<b>Link</b> in the header) so others can join.</li><li>The assistant answers one question at a time; up to three more wait their turn.</li></ul>
      <button class="btn" id="welcomeNew" type="button">${icon.plus}New room</button></div>`; $('#welcomeNew').onclick = () => { root.classList.add('side-open'); openNewForm(); };
  }
  // ---------------- transcript ----------------
  function clearTranscript() { S.turns.forEach(t => t.dispose()); S.turns.clear(); S.slots.clear(); rinner.innerHTML = '<div id="rhist"></div><div id="rlive"></div>'; S.hist = []; S.histStart = 0; S.lastDay = null; S.unseen = 0; $('#jumpNew').hidden = true; }
  const nearBottom = () => rmsgs.scrollHeight - rmsgs.scrollTop - rmsgs.clientHeight < 90;
  const toBottom = () => { rmsgs.scrollTop = rmsgs.scrollHeight; S.follow = true; S.unseen = 0; $('#jumpNew').hidden = true; };
  rmsgs.addEventListener('scroll', () => { S.follow = nearBottom(); if (S.follow) { S.unseen = 0; $('#jumpNew').hidden = true; } });
  $('#jumpNew').onclick = toBottom;
  const resizer = new ResizeObserver(() => { if (S.follow) rmsgs.scrollTop = rmsgs.scrollHeight; }); resizer.observe(rinner);
  const retryAsk = (prompt, mode) => { if (!online()) return toast('Not connected; try again in a moment.'); sendRaw(prompt, mode === 'auto' ? null : mode); };
  function daySep(into, ts) { const k = new Date(ts).toDateString(); if (k === S.lastDay) return; S.lastDay = k; into.appendChild(h(`<div class="rsep"><span>${esc(dayLabel(ts))}</span></div>`)); }
  function bubble(into, who, text, ts) {
    const mine = who === S.me; const el = h(`<div class="msg user rmsg ${mine ? 'mine' : 'other'}"><div class="who">${mine ? 'You' : `<span style="color:${roomColor(who)}">${esc(who || 'Someone')}</span>`}<time title="${esc(new Date(ts).toLocaleString())}">${esc(clock(ts))}</time></div><div class="txt"></div></div>`);
    $('.txt', el).textContent = text; into.appendChild(el); return el;
  }
  function renderHistory() {
    const box = $('#rhist'); box.innerHTML = ''; S.turns.forEach((t, k) => { if (box.contains(t.el)) { t.dispose(); S.turns.delete(k); } }); S.lastDay = null;
    if (S.histStart > 0) { const more = h(`<button class="btn ghost sm rmore" type="button">Show ${Math.min(WINDOW, S.histStart)} earlier of ${S.histStart} hidden</button>`);
      more.onclick = () => { const before = rmsgs.scrollHeight; S.histStart = Math.max(0, S.histStart - WINDOW); renderHistory(); rmsgs.scrollTop += rmsgs.scrollHeight - before; }; box.appendChild(more); }
    let lastQ = '';
    for (const it of S.hist.slice(S.histStart)) { daySep(box, it.ts);
      if (it.kind === 'q') { lastQ = it.text; bubble(box, it.who, it.text, it.ts); }
      else Turn.fromStored(box, it.reply, { prompt: lastQ, onRetry: retryAsk }, it.feedback).dispose(); }
  }
  function addQuestion(id, who, text, ts) { const live = $('#rlive'); daySep(live, ts); bubble(live, who, text, ts); const slot = h('<div class="rslot"></div>'); live.appendChild(slot); if (id) S.slots.set(id, { slot, text }); }
  function liveTurn(id) {
    let t = S.turns.get(id); if (t) return t; let s = S.slots.get(id);
    if (!s) { const slot = h('<div class="rslot"></div>'); $('#rlive').appendChild(slot); s = { slot, text: '' }; S.slots.set(id, s); }
    t = new Turn(s.slot, { prompt: s.text, onRetry: retryAsk }); S.turns.set(id, t); return t;
  }
  function empty() { if (!S.hist.length && !$('#rlive').children.length && S.room && !S.hydrating) { if (!$('#rhist .rempty')) $('#rhist').innerHTML = `<div class="rempty empty"><b>Nobody has asked anything yet.</b><br>Ask a question below: everyone in “${esc(S.room.name)}” sees it, and the assistant's answer as it is written.</div>`; } }
  // ---------------- joining, the socket, reconnecting ----------------
  function closeSocket() { S.gen++; clearTimeout(S.retryTimer); const ws = S.ws; S.ws = null; if (ws) { ws.onclose = ws.onmessage = ws.onerror = ws.onopen = null; try { ws.close(); } catch {} } }
  async function join(id) {
    if (S.dead) return; if (S.room && S.room.room_id !== id) drafts_save();
    closeSocket(); clearTranscript(); banner(null); S.attempt = 0; S.users = []; S.me = null; S.assistant = { busy: false, working_for: null, queued: 0 }; S.typers.clear(); renderTyping();
    let r = S.rooms.find(x => x.room_id === id);
    try { r = await api.get(`/v1/rooms/${id}`); } catch (e) { if (e.status === 404) { toast('That room no longer exists.'); if (store.get('lastRoom') === id) store.set('lastRoom', null); S.room = null; history.replaceState(null, '', '#/rooms'); renderBar(); renderMainEmpty(); refreshList(); return; } if (!r) { toast(`Could not open the room: ${e.message}`, 4000); return; } }
    if (S.dead || !alive()) return;
    S.room = r; markSeen(r); store.set('lastRoom', id); history.replaceState(null, '', `#/rooms/${id}`); S.conn = 'connecting'; $('#roomBar').innerHTML = ''; renderBar(); renderList(); updateComposer();
    input.value = S.drafts.get(id) || ''; autosize(); connect(S.gen);
  }
  const drafts_save = () => { if (S.room) S.drafts.set(S.room.room_id, input.value); };
  function connect(gen) {
    if (gen !== S.gen || S.dead) return; S.conn = S.attempt ? 'reconnecting' : 'connecting'; renderBar(); updateComposer();
    const proto = location.protocol === 'https:' ? 'wss' : 'ws'; const tok = api.token(); let ws;
    try { ws = new WebSocket(`${proto}://${location.host}/ws/rooms/${S.room.room_id}?user=${encodeURIComponent(store.get('user', 'guest'))}${tok ? '&token=' + encodeURIComponent(tok) : ''}`); } catch { return scheduleReconnect(gen); }
    S.ws = ws; let lastRx = Date.now(); const watchdog = setInterval(() => { if (Date.now() - lastRx > 75000) { try { ws.close(); } catch {} } }, 10000);
    ws.onmessage = (e) => { if (gen !== S.gen) return; lastRx = Date.now(); let f; try { f = JSON.parse(e.data); } catch { return; } onFrame(f, gen); };
    ws.onclose = (ev) => { clearInterval(watchdog); if (gen !== S.gen) return; S.ws = null; if (ev.code === 4004) return handleDeleted(); scheduleReconnect(gen); };
    ws.onerror = () => {};
  }
  async function scheduleReconnect(gen) {
    if (gen !== S.gen || S.dead) return; S.hydrating = false; S.typers.clear(); renderTyping();
    if (S.attempt >= 10) { S.conn = 'closed'; renderBar(); updateComposer(); return; }
    const delay = Math.min(15000, 1000 * 2 ** Math.min(S.attempt, 4)) * (0.8 + Math.random() * 0.4); S.attempt++; S.conn = 'reconnecting'; S.retryAt = Date.now() + delay; renderBar(); updateComposer();
    clearTimeout(S.retryTimer);
    S.retryTimer = setTimeout(async () => { if (gen !== S.gen) return;
      try { const r = await api.get(`/v1/rooms/${S.room.room_id}`); if (gen !== S.gen) return; S.room = { ...S.room, ...r }; } catch (e) { if (e.status === 404) return handleDeleted(); }
      connect(gen); }, delay);
  }
  function reconnectNow() { if (!S.room) return; clearTimeout(S.retryTimer); const ws = S.ws; S.gen++; if (ws) { ws.onclose = ws.onmessage = null; try { ws.close(); } catch {} } S.attempt = 0; connect(S.gen); }
  function handleDeleted() {
    if (!S.room || S.conn === 'deleted') return; const id = S.room.room_id; closeSocket(); S.conn = 'deleted'; S.hydrating = false; if (store.get('lastRoom') === id) store.set('lastRoom', null);
    banner('deleted'); input.disabled = true; $('#rsend').disabled = true; input.placeholder = 'This room was deleted'; $('#roomBar').querySelector('#rstate')?.replaceChildren(); refreshList(); say('This room was deleted.');
  }
  // ---------------- frames ----------------
  function onFrame(f, gen) {
    switch (f.type) {
      case 'welcome': return onWelcome(f.payload, gen);
      case 'presence': S.users = f.payload.users || S.users;
        if (!S.hydrating && f.user && f.user !== S.me) { const live = $('#rlive'); live.appendChild(h(`<div class="rnote">${esc(f.user)} ${f.payload.joined ? 'joined' : 'left'}</div>`)); }
        renderBar(); refreshSoon(); return;
      case 'user_message': case 'assistant_event': if (S.hydrating) { S.buffer.push(f); return; } return onContent(f);
      case 'queue': S.assistant = f.payload; renderBar(); refreshSoon(); return;
      case 'typing': if (f.user !== S.me) { S.typers.set(f.user, Date.now() + 4500); renderTyping(); } return;
      case 'room_renamed': S.room = { ...S.room, name: f.payload.name }; $('#roomBar').innerHTML = ''; renderBar(); updateComposer(); refreshSoon(); $('#rlive').appendChild(h(`<div class="rnote">Room renamed to “${esc(f.payload.name)}”</div>`)); return;
      case 'room_deleted': return handleDeleted();
      case 'error': { const p = f.payload; note(p.message); toast(p.message, 4500); if (['E_TOO_LONG', 'E_RATE_LIMITED', 'E_QUEUE_FULL'].includes(p.code) && !input.value && S.lastSent) { input.value = S.lastSent; autosize(); updateComposer(); } if (p.code === 'E_ROOM_FULL') banner('full', p.message); return; }
      case 'resync': toast('Connection fell behind; reloading the transcript.', 3500); return reconnectNow();
      default: return; // heartbeat, pong
    }
  }
  async function onWelcome(w, gen) {
    S.me = w.you; S.users = w.users; S.assistant = w.assistant; S.limits = w.limits || S.limits; S.conn = 'open'; S.attempt = 0; S.hydrating = true; S.buffer = []; const my = ++S.hyd;
    clearTranscript(); banner(null); renderBar(); updateComposer(); const hint = $('#userHint'); const want = store.get('user', ''); hint.textContent = w.you !== want ? `You appear here as “${w.you}” because “${want}” is taken.` : 'Shown to everyone in a room. Letters, numbers, spaces, - and _.';
    try { const c = await api.get('/v1/conversations/' + S.room.conversation_id); if (my !== S.hyd || gen !== S.gen) return;
      S.hist = roomItems(c.messages.slice(0, w.cursor)); S.histStart = Math.max(0, S.hist.length - WINDOW); renderHistory(); }
    catch (e) { if (my !== S.hyd || gen !== S.gen) return; banner('load', e.message); }
    for (const p of w.pending || []) addQuestion(p.id, p.user, p.content, p.ts);
    S.hydrating = false; const buf = S.buffer; S.buffer = []; buf.forEach(onContent); toBottom(); empty(); markSeen(S.room); renderBar(); updateComposer();
  }
  function onContent(f) {
    const wasFollowing = S.follow; $('#rhist .rempty')?.remove();
    if (f.type === 'user_message') { const p = f.payload; addQuestion(p.id, f.user, p.content, p.ts); if (f.user === S.me) toBottom(); else { say(`${f.user}: ${p.content}`); if (!wasFollowing) { S.unseen++; const j = $('#jumpNew'); j.textContent = `${plural(S.unseen, 'new message')} ↓`; j.hidden = false; } } return; }
    const id = f.payload.reply_to, ev = f.payload.event; const t = liveTurn(id); t.on(ev.event, ev);
    if (ev.event === 'done') { say('The assistant answered.'); setTimeout(() => t.dispose(), 800); } if (ev.event === 'error') setTimeout(() => t.dispose(), 800);
  }
  // ---------------- composer ----------------
  const autosize = () => { input.style.height = 'auto'; input.style.height = Math.min(220, input.scrollHeight) + 'px'; };
  function updateComposer() {
    const len = [...input.value].length, max = S.limits.message_chars, over = len > max, ok = online();
    input.disabled = !S.room || S.conn === 'deleted'; $('#rsend').disabled = !ok || !input.value.trim() || over;
    input.placeholder = !S.room ? 'Select or create a room to start' : S.conn === 'deleted' ? 'This room was deleted' : ok ? `Message “${S.room.name}”…` : S.conn === 'closed' ? 'Disconnected. Your draft is kept; press Reconnect.' : 'Connecting… you can type while you wait';
    const c = $('#rcount'); c.textContent = len > max * 0.8 ? `${len.toLocaleString()} / ${max.toLocaleString()}` : ''; c.style.color = over ? 'var(--bad)' : 'var(--faint)';
  }
  function sendRaw(text, mode) { if (!online()) return false; S.lastSent = text; S.ws.send(JSON.stringify({ type: 'user_message', payload: { content: text, force_mode: mode, show_reasoning: true } })); S.follow = true; return true; }
  function sendRoom() {
    const t = input.value.trim(); if (!t) return;
    if (!online()) { toast('Not connected yet. Your message is kept.'); return; }
    if ([...t].length > S.limits.message_chars) { note(`Messages are limited to ${S.limits.message_chars.toLocaleString()} characters.`); return; }
    if (sendRaw(t, S.mode === 'auto' ? null : S.mode)) { input.value = ''; S.drafts.delete(S.room.room_id); autosize(); updateComposer(); note(''); }
  }
  $('#rsend').onclick = sendRoom;
  input.addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); sendRoom(); } });
  input.addEventListener('input', () => { autosize(); updateComposer(); if (online() && input.value.trim() && Date.now() - S.lastTyping > 2000) { S.lastTyping = Date.now(); S.ws.send('{"type":"typing"}'); } });
  $$('#rmode button').forEach(b => { b.classList.toggle('on', b.dataset.m === S.mode); b.onclick = () => { S.mode = b.dataset.m; store.set('roomMode', S.mode); $$('#rmode button').forEach(x => x.classList.toggle('on', x === b)); }; });
  function renderTyping() {
    const now = Date.now(); for (const [k, t] of S.typers) if (t < now) S.typers.delete(k); const n = [...S.typers.keys()];
    $('#rtyping').textContent = !n.length ? '' : n.length === 1 ? `${n[0]} is typing…` : n.length === 2 ? `${n[0]} and ${n[1]} are typing…` : `${n.length} people are typing…`;
  }
  // ---------------- your name ----------------
  $('#userName').onchange = (e) => { const name = cleanUserName(e.target.value); e.target.value = name; if (name === store.get('user', '')) return; store.set('user', name); if (S.room) { drafts_save(); join(S.room.room_id); } };
  // ---------------- lifecycle ----------------
  const tick = setInterval(() => { if (S.conn === 'reconnecting') { const st = $('#rstate'); if (st) st.innerHTML = chips(), bindBar(); } renderTyping(); }, 1000);
  const poll = setInterval(() => { if (!document.hidden) refreshList(); }, 8000);
  let lastVis = 0;
  const onVis = () => { if (document.hidden || Date.now() - lastVis < 3000) return; lastVis = Date.now(); refreshList(); if (S.room && (S.conn === 'closed' || S.conn === 'reconnecting')) reconnectNow(); };
  const onOnline = () => { if (S.room && S.conn !== 'open') reconnectNow(); };
  document.addEventListener('visibilitychange', onVis); window.addEventListener('online', onOnline);
  root.addEventListener('click', (e) => { if (e.target === root) root.classList.remove('side-open'); });
  clearTranscript(); renderBar(); renderMainEmpty(); updateComposer();
  await refreshList();
  if (!alive()) return () => { S.dead = true; clearInterval(tick); clearInterval(poll); resizer.disconnect(); };
  const want = (location.hash.match(/^#\/rooms\/([0-9a-f-]{36})/i) || [])[1] || store.get('lastRoom', null);
  if (want && (S.rooms.some(r => r.room_id === want) || location.hash.includes(want))) await join(want);
  else if (want) { store.set('lastRoom', null); }
  return () => { S.dead = true; drafts_save(); closeSocket(); S.turns.forEach(t => t.dispose()); clearInterval(tick); clearInterval(poll); clearTimeout(S.retryTimer); resizer.disconnect(); document.removeEventListener('visibilitychange', onVis); window.removeEventListener('online', onOnline); };
} };

// ============================== performance console ==============================
const REF_BENCH = [['Qwen3-1.7B Q4_K_M (System 1)', 131.0, 22.9], ['Qwen3.5-2B Q4_K_M (hybrid)', 112.8, 18.2], ['Qwen3-4B(-Thinking-2507) Q4_K_M (System 2)', 51.7, 10.3], ['Qwen3.5-4B Q4_K_M (hybrid)', 44.9, 8.2], ['vLLM Qwen3-1.7B FP32 (Lima VM, end-to-end)', null, 3.2]];
routes.bench = { title: 'Performance', async render(view, alive) {
  const sets = await api.get('/v1/bench/prompt-sets').catch(() => ({}));
  view.innerHTML = `<div class="page grid" style="gap:16px">
    <div class="card"><div class="card-h"><h2>Load test</h2><span class="hint">Closed-loop workers send requests through the full pipeline (gate, verifiers, both models). FR-052, ER-007.</span></div>
      <div class="card-b"><div class="row" style="align-items:flex-end;gap:14px">
        <label class="f">Prompt set<select id="bSet">${Object.entries(sets).map(([k, v]) => `<option value="${k}">${esc(v.label)}</option>`).join('')}</select></label>
        <label class="f">Path<select id="bMode"><option value="fast">Fast only</option><option value="auto">Auto (gate decides)</option><option value="slow">Deliberate only</option></select></label>
        <label class="f">Concurrency<select id="bConc"><option>1</option><option>2</option><option>4</option></select></label>
        <label class="f">Stop after<select id="bStop"><option value="r:6">6 requests</option><option value="r:12">12 requests</option><option value="r:24">24 requests</option><option value="d:60">60 seconds</option><option value="d:300">5 minutes</option></select></label>
        <label class="f">Max tokens<input id="bMax" type="number" value="128" min="16" max="2048" style="width:90px"></label>
        <span class="sp"></span><button class="btn" id="bStart">${icon.play}Start</button><button class="btn ghost" id="bCancel" hidden>Cancel</button></div>
        <div id="bProg" style="margin-top:14px" hidden><div class="progress"><i id="bBar" style="width:0"></i></div><div class="small muted" id="bProgTxt" style="margin-top:6px"></div></div></div></div>
    <div class="card"><div class="card-h"><h2>Load-test protocol (ER-007)</h2><span class="hint">The recorded protocol: the same prompt set at 1, 2 and 4 concurrent sessions, p50/p95/p99, and the NFR-007 verdict (p95 at 4 sessions at most twice the single-session value).</span><span class="sp"></span><button class="btn" id="pStart">${icon.play}Run protocol</button></div><div class="card-b" id="pOut"><div class="empty small">No protocol run yet.</div></div></div>
    <div class="card" id="bStages"></div>
    <div class="grid g4" id="bTiles"></div>
    <div class="grid g2"><div class="card"><div class="card-h"><h3>Latency per request</h3><div class="legend"><span><i style="background:var(--fast)"></i>fast path</span><span><i style="background:var(--slow)"></i>deliberate</span><span><i style="background:var(--bad)"></i>error</span></div></div><div class="card-b" id="bScatter"><div class="empty small">Start a run to see results.</div></div></div>
      <div class="card"><div class="card-h"><h3>Latency quantiles</h3></div><div class="card-b" id="bQuant"><div class="empty small">–</div></div></div></div>
    <div class="grid g2"><div class="card"><div class="card-h"><h3>Run history</h3></div><div class="card-b" style="padding:0" id="bHist"></div></div>
      <div class="card"><div class="card-h"><h3>Reference: raw model speed on this machine</h3><span class="hint">llama-bench pp512/tg128, 4 threads, idle i3-8100, 2026-10-18 (measured)</span></div><div class="card-b" style="padding:0"><table class="t"><tr><th>Model</th><th>Prefill tok/s</th><th>Decode tok/s</th></tr>${REF_BENCH.map(r => `<tr><td>${r[0]}</td><td>${r[1] ?? '–'}</td><td>${r[2]}</td></tr>`).join('')}</table></div></div></div></div>`;
  // P7-T01: where the time goes, from the stage timings of recent replies.
  const loadStages = async () => { const lab = store.get('stagesLab', false); const v = await api.get(`/v1/perf/stages?hours=24&lab=${lab}`).catch(() => null); if (!alive() || !v) return;
    const P = v.paths || [], E = v.engine || [];
    $s('#bStages').innerHTML = `<div class="card-h"><h2>Where the time goes</h2><span class="hint">Stage timings of the last 24 h of ${lab ? 'all' : 'live'} replies (${v.replies}). Model stages carry llama-server's own prefill and decode split and KV-cache reuse.</span><span class="sp"></span><label class="small"><input type="checkbox" id="stLab" ${lab ? 'checked' : ''}> include lab runs</label></div>
      <div class="card-b">${P.length ? P.map(p => { const parts = [...p.stages.filter(x => x.share != null).map(x => ({ k: x.stage, v: x.share, label: pct(x.share, 0) })), { k: 'overhead', v: p.overhead_share, label: pct(p.overhead_share, 0) }];
        return `<div style="margin-bottom:14px"><div class="row small"><b>${p.mode === 'slow' ? 'Deliberate path' : 'Fast path'}</b><span class="faint">${p.replies} replies · wall p50 ${fmtMs(p.wall_p50_ms)} · p95 ${fmtMs(p.wall_p95_ms)}</span></div>${stageBar(parts)}
          <div class="legend wrap">${parts.filter(x => x.v > 0.005).map(x => `<span><i style="background:${STAGE[x.k]?.[1] || 'var(--slate)'}"></i>${esc(STAGE[x.k]?.[0] || x.k)} ${x.label}</span>`).join('')}</div></div>`; }).join('')
        + (E.length ? `<table class="t"><tr><th>Model stage</th><th>Calls</th><th title="prompt tokens evaluated per second (llama-server timings)">Prefill tok/s</th><th title="generated tokens per second">Decode tok/s</th><th title="share of prompt tokens served from the KV cache">Cache reuse</th><th>Queue p95</th></tr>${E.map(e => `<tr><td>${esc(STAGE[e.stage]?.[0] || e.stage)}</td><td>${e.calls}</td><td>${num(e.prefill_tok_s, 0)}</td><td>${num(e.decode_tok_s, 1)}</td><td>${e.cache_reuse == null ? '–' : pct(e.cache_reuse, 0)}</td><td>${fmtMs(e.queue_p95_ms)}</td></tr>`).join('')}</table>` : '')
        : '<div class="empty small">No replies with stage timings yet. Send a message in Chat or run a load test.</div>'}</div>`;
    $s('#stLab').onchange = (e) => { store.set('stagesLab', e.target.checked); loadStages(); }; };
  loadStages();
  let poll = null, current = null;
  const tiles = (r) => { const x = r?.results || {}; const T = (k, v, s) => `<div class="card tile"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${s || ''}</div></div>`;
    $('#bTiles').innerHTML = T('Requests', `${x.ok ?? 0}<span class="small muted"> / ${x.requests ?? 0}</span>`, `error rate ${pct(x.error_rate ?? 0)}`) + T('Throughput', `${num(x.throughput_rps, 3)}<span class="small muted"> req/s</span>`, `${num(x.tokens_per_s, 1)} output tok/s`)
      + T('Time to first token', fmtMs(x.ttft_ms?.p50), `p95 ${fmtMs(x.ttft_ms?.p95)} · target ≤ 2 s (NFR-002)`) + T('Latency p50 / p95', `${fmtMs(x.latency_ms?.p50)}`, `p95 ${fmtMs(x.latency_ms?.p95)} · p99 ${fmtMs(x.latency_ms?.p99)} · escalated ${pct(x.escalation_rate)}`); };
  const show = (r) => { current = r; tiles(r); const S = r.samples || [];
    const pts = S.map(s => ({ x: s.t_start_ms / 1000, y: s.latency_ms, color: !s.ok ? 'var(--bad)' : s.mode === 'slow' ? 'var(--slow)' : 'var(--fast)', title: `${s.mode} · ${fmtMs(s.latency_ms)}${s.ttft_ms ? ' · TTFT ' + fmtMs(s.ttft_ms) : ''}` }));
    const p95 = r.results?.latency_ms?.p95; $('#bScatter').innerHTML = chart.scatter(pts, { xLabel: 'request start (s)', yLabel: 'latency', lines: p95 ? [{ y: p95, color: 'var(--amber)', label: 'p95' }] : [] });
    const q = r.results || {}; const rows = [['All', q.latency_ms], ['Fast path', q.latency_fast_ms], ['Deliberate', q.latency_slow_ms], ['TTFT', q.ttft_ms]];
    $('#bQuant').innerHTML = `<table class="t"><tr><th></th><th>n</th><th>p50</th><th>p95</th><th>p99</th><th>mean</th></tr>${rows.map(([k, v]) => `<tr><td>${k}</td><td>${v?.n ?? 0}</td><td>${fmtMs(v?.p50)}</td><td>${fmtMs(v?.p95)}</td><td>${fmtMs(v?.p99)}</td><td>${fmtMs(v?.mean)}</td></tr>`).join('')}</table>
      <p class="hint" style="margin:10px 0 0">Targets on the reference machine: fast-path TTFT ≤ 2 s and 150-token reply ≤ 12 s at p95; deliberate reply ≤ 240 s at p95 (NFR-002..004).</p>`;
    const req = r.request || {}; const total = req.requests || null; const done = r.results?.requests || 0;
    $('#bProg').hidden = false; $('#bBar').style.width = total ? `${Math.min(100, done / total * 100)}%` : `${Math.min(100, (r.elapsed_s || 0) / (req.duration_s || 1) * 100)}%`;
    $('#bProgTxt').textContent = `${r.status === 'running' ? 'Running' : 'Finished'} · ${done}${total ? ' / ' + total : ''} requests · ${num(r.elapsed_s, 0)} s elapsed`;
    $('#bStart').hidden = r.status === 'running'; $('#bCancel').hidden = r.status !== 'running'; };
  const history = async () => { const runs = await api.get('/v1/bench/runs').catch(() => []);
    $('#bHist').innerHTML = runs.length ? `<table class="t"><tr><th>When</th><th>Set</th><th>Path</th><th>Conc.</th><th>OK</th><th>p50</th><th>p95</th><th></th></tr>${runs.map(r => r.kind === 'protocol' ? `<tr class="click" data-proto="${esc(r.run_id)}"><td>${ago(r.created_at || new Date().toISOString())}</td><td>${esc(r.request?.prompt_set)}</td><td>${esc(r.request?.mode || 'fast')}</td><td>protocol ${esc((r.request?.levels || []).join('/'))}</td><td>${(r.table || []).reduce((a, x) => a + (x.ok || 0), 0) || '–'}</td><td colspan="2">${r.nfr007?.pass == null ? '–' : `NFR-007 ${r.nfr007.pass ? 'pass' : 'fail'} (${num(r.nfr007.ratio, 2)}×)`}</td><td>${r.status === 'running' ? '<span class="spin"></span>' : ''}</td></tr>`
      : `<tr class="click" data-id="${esc(r.run_id)}"><td>${ago(r.created_at || r.finished_at || new Date().toISOString())}</td><td>${esc(r.request?.prompt_set)}</td><td>${esc(r.request?.mode || 'auto')}</td><td>${r.request?.concurrency ?? '–'}</td><td>${r.results?.ok ?? '–'}</td><td>${fmtMs(r.results?.latency_ms?.p50)}</td><td>${fmtMs(r.results?.latency_ms?.p95)}</td><td>${r.status === 'running' ? '<span class="spin"></span>' : ''}</td></tr>`).join('')}</table>` : '<div class="empty small">No runs yet.</div>';
    $$('#bHist tr[data-id]').forEach(tr => tr.onclick = () => watch(tr.dataset.id)); $$('#bHist tr[data-proto]').forEach(tr => tr.onclick = () => { watchProto(tr.dataset.proto); $('#pOut').scrollIntoView({ behavior: 'smooth' }); }); };
  const watch = (id) => { clearInterval(poll); const tick = async () => { const r = await api.get('/v1/bench/runs/' + id); show(r); if (r.status !== 'running') { clearInterval(poll); history(); } }; tick(); poll = setInterval(tick, 1500); };
  $('#bStart').onclick = async () => { const [k, v] = $('#bStop').value.split(':'); const body = { prompt_set: $('#bSet').value, mode: $('#bMode').value, concurrency: +$('#bConc').value, max_tokens: +$('#bMax').value, [k === 'r' ? 'requests' : 'duration_s']: +v };
    try { const r = await api.post('/v1/bench/runs', body); toast('Load test started'); watch(r.run_id); history(); } catch (e) { toast(e.message); } };
  $('#bCancel').onclick = async () => { if (current) { await api.post(`/v1/bench/runs/${current.run_id}/cancel`).catch(() => {}); toast('Cancelling after in-flight requests finish…'); } };
  const showProto = (r) => { const T = r.table || []; const v = r.nfr007 || {};
    $('#pOut').innerHTML = `${r.status === 'running' ? `<p class="small"><span class="spin"></span> Running level ${(r.level_index ?? 0) + 1} of ${r.request?.levels?.length ?? 3}…</p>` : ''}${T.length ? `<table class="t"><tr><th>Sessions</th><th>Requests</th><th>OK</th><th>p50</th><th>p95</th><th>p99</th><th>TTFT p95</th><th>Throughput</th><th>Escalated</th><th>Cost / request</th></tr>
      ${T.map(x => `<tr><td><b>${x.concurrency}</b></td><td>${x.requests}</td><td>${x.ok}</td><td>${fmtMs(x.p50_ms)}</td><td>${fmtMs(x.p95_ms)}</td><td>${fmtMs(x.p99_ms)}</td><td>${fmtMs(x.ttft_p95_ms)}</td><td>${num(x.throughput_rps, 3)} req/s</td><td>${pct(x.escalation_rate, 0)}</td><td>${num(x.cost_per_request_s, 1)} s</td></tr>`).join('')}</table>
      <p class="small">${v.pass == null ? 'NFR-007: needs levels 1 and 4.' : `NFR-007 ${v.pass ? '<span class="badge b-verified">pass</span>' : '<span class="badge b-abstained">fail</span>'} p95 at 4 sessions is ${num(v.ratio, 2)}× the single-session value (target ≤ 2×)${(T[0]?.requests ?? 0) < 8 ? ` · <span class="badge" title="with fewer than 8 requests the p95 is just the maximum">indicative, n ${T[0]?.requests ?? 0}</span>` : ''}${v.pass ? '' : '. Roadmap fallback: document the single-session profile and treat concurrency as a GPU-profile feature.'} The gates dashboard judges the median over recent runs with at least 8 requests per session.`}</p>` : ''}`; };
  const watchProto = (id) => { const tick = async () => { const r = await api.get('/v1/bench/runs/' + id).catch(() => null); if (!r) return; showProto(r); if (r.status !== 'running') clearInterval(pPoll); }; clearInterval(pPoll); tick(); pPoll = setInterval(tick, 3000); };
  let pPoll = null;
  $('#pStart').onclick = async () => { try { const r = await api.post('/v1/bench/protocol', { prompt_set: $('#bSet').value, mode: 'fast', levels: [1, 2, 4], requests_per_session: 8, max_tokens: +$('#bMax').value }); toast('Protocol started: 1, 2, then 4 sessions'); watchProto(r.run_id); } catch (e) { toast(e.message); } };
  { const prev = (await api.get('/v1/bench/runs').catch(() => [])).find(r => r.kind === 'protocol'); if (prev) watchProto(prev.run_id); }
  tiles(null); history();
  const runs = await api.get('/v1/bench/runs').catch(() => []); const lastBench = runs.find(r => r.kind !== 'protocol'); if (lastBench) watch(lastBench.run_id);
  return () => { clearInterval(poll); clearInterval(pPoll); };
} };

// ============================== evaluation dashboard ==============================
const SYS_LABEL = { nafs_aql: 'Nafs-Aql (gated)', nafs_blind: 'Nafs-Aql (gate blind)', fast_only: 'Fast only (S1)', slow_only: 'Deliberate only (S2)', slow_conditioned: 'S2 sees draft (ablation)', slow_noexit: 'Deliberate, no early exit', slow_exit_consistency: 'Deliberate, exit on consistency', slow_exit_posterior: 'Deliberate, exit on posterior' };
const SYS_HELP = { nafs_aql: 'Full system; the exact verifier checks the draft before the gate decides.', nafs_blind: 'Full system with the verifier hidden at run time, so the gate must route on its estimate ê₁. Scored afterwards. This is the honest routing and calibration test.',
  fast_only: 'System 1 only.', slow_only: 'System 2 only, draft-blind.', slow_conditioned: 'Coupling ablation (ER-004): System 2 is shown the System 1 draft. Used to measure adoption of wrong drafts.' , slow_noexit: 'Deliberate only with the adaptive reasoning exit switched off: pairs with Deliberate only to measure the tokens and time the exit saves (P7-T08).', slow_exit_consistency: 'Deliberate only; on non-verifiable tasks the reasoning stops after two consecutive agreeing probes (the P7-T08 rule).', slow_exit_posterior: 'Deliberate only; on non-verifiable tasks the reasoning stops when the posterior that the plurality probe answer is the majority reaches a bound that collapses over the budget (Phase 8, Prop. N4).' };
const SYS_COLOR = { nafs_aql: 'var(--accent)', nafs_blind: 'var(--teal)', fast_only: 'var(--fast)', slow_only: 'var(--slow)', slow_conditioned: 'var(--amber)', slow_noexit: 'var(--navy)', slow_exit_consistency: 'var(--navy)', slow_exit_posterior: 'var(--teal)' };
const sysLabel = (k) => { const m = /^(nafs_aql|nafs_blind)_l(\d+)$/.exec(k); return esc(SYS_LABEL[k] || (m ? `${SYS_LABEL[m[1]]} λ=${(+m[2] / 100).toFixed(2)}` : /^slow_b\d+$/.test(k) ? `S2 budget ${k.slice(6)}` : k)); };
const shortLabel = (k) => { const m = /^(nafs_aql|nafs_blind)_l(\d+)$/.exec(k); if (m) return `${m[1] === 'nafs_blind' ? 'Blind gate' : 'Gate'} λ=${(+m[2] / 100).toFixed(2)}`;
  return { nafs_aql: 'Gate + verifier', nafs_blind: 'Blind gate', fast_only: 'Fast only', slow_only: 'Deliberate only', slow_conditioned: 'S2 sees draft' }[k] || (/^slow_b\d+$/.test(k) ? `Budget ${k.slice(6)}` : k); };
const sysColor = (k) => SYS_COLOR[k] || (/^nafs_blind_l/.test(k) ? 'var(--teal)' : /^nafs_aql_l/.test(k) ? 'var(--accent)' : /^slow_b/.test(k) ? 'var(--violet)' : 'var(--slate)');
const PRESETS = [
  { t: 'Routing study', q: 'When should \u02bfaql restrain the first impulse?', d: 'The gate routes on its own estimate with the verifier hidden; scored against the Theorem 2 oracle (H1).', ds: 'gsm8k', n: 20, sys: ['nafs_blind', 'fast_only', 'slow_only'] },
  { t: 'Frontier (λ sweep)', q: 'How much care is worth its cost?', d: 'The same gate at three prices of latency; the Pareto frontier shows the efficient policies (Prop. 2).', ds: 'gsm8k', n: 15, sys: ['nafs_blind_l05', 'nafs_blind_l20', 'nafs_blind_l60', 'fast_only', 'slow_only'] },
  { t: 'Coupling test', q: 'Does reflection repeat the impulse?', d: 'System 2 with and without sight of the draft; adoption of wrong drafts (ER-004, H5).', ds: 'gsm8k', n: 20, sys: ['fast_only', 'slow_only', 'slow_conditioned'] },
  { t: 'Budget curve', q: 'How long should reflection last?', d: 'Accuracy against thinking budget and the cost-optimal b* (H4, Prop. 9).', ds: 'gsm8k', n: 15, sys: ['fast_only', 'slow_b256', 'slow_b512', 'slow_only'] },
  { t: 'Full system (G2)', q: 'Is the cascade as good as always deliberating?', d: 'Gated system with and without the verifier against both baselines (ER-002).', ds: 'smoke_mixed', n: 12, sys: ['nafs_aql', 'nafs_blind', 'fast_only', 'slow_only'] },
  { t: 'Strong code tests', q: 'Do stronger tests change the verdict?', d: 'HumanEval+ (EvalPlus) tests catch answers the original tests let through.', ds: 'humaneval_plus', n: 10, sys: ['nafs_aql', 'fast_only', 'slow_only'] },
  { t: 'Early exit', q: 'When has reflection said enough?', d: 'Deliberation with and without the probe-and-verify exit: tokens and time saved against accuracy (FR-082, P7-T08).', ds: 'gsm8k', n: 12, sys: ['slow_only', 'slow_noexit'] },
  { t: 'Exit rule', q: 'Which stopping bound on non-verifiable tasks?', d: 'Two consecutive agreeing probes against the posterior rule with a collapsing bound (FR-084, Prop. N4): same answers, fewer or more probes?', ds: 'factual_probe', n: 13, sys: ['slow_exit_consistency', 'slow_exit_posterior'] },
];
const GLOSSARY = [['Accuracy (95% CI)', 'Share of scored items an exact verifier accepted, with an exact Clopper–Pearson interval. Small n gives wide intervals: read the interval, not the point.'],
  ['Escalation rate g', 'Share of requests the gate sent to System 2 (including degraded escalations).'],
  ['Routing accuracy', 'Agreement of the gate with the Theorem 2 oracle (escalate iff L·(fast wrong − slow wrong) > λ·Δc). Only blind-gate systems measure the gate itself.'],
  ['ECE of ê₁', 'Expected calibration error of the fast-path error estimate, 5 equal-mass bins. Promotion needs ≤ 0.05 on ≥ 2,900 held-out samples (NFR-020).'],
  ['Paired Δ and non-inferiority', 'Per-item difference against a baseline with a bootstrap 95% CI; non-inferior at 3 points if the lower bound is above −3 points (G2 needs ≥ 700 pairs).'],
  ['Cost ratio', 'Mean wall time of a system divided by the baseline\'s; the target is ≤ 0.60 of slow-only at matched accuracy (H2).'],
  ['Pareto frontier', 'Policies for which no other policy is at least as accurate and at most as costly. The frontier is where to choose λ.'],
  ['Adoption rate', 'Share of wrong drafts that System 2 repeats; draft-blind deliberation should keep its upper bound ≤ 5% (H5).'],
  ['r and φ', 'A verifier\'s true-accept rate on known-correct answers and false-accept rate on seeded wrong answers (mutants), with Clopper–Pearson bounds.']];
function glossary() { modal(`<h3 style="margin-top:0">${icon.book} Metrics glossary</h3><dl class="kv small">${GLOSSARY.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('')}</dl>`); }
routes.eval = { title: 'Lab', async render(view, alive) {
  const datasets = await api.get('/v1/eval/datasets').catch(() => []);
  if (!alive()) return;
  view.innerHTML = `<div class="page grid" style="gap:16px">
    <div class="card" id="gatesCard"><div class="card-b"><span class="spin"></span> Evaluating roadmap gates…</div></div>
    <div class="card" id="insCard"><div class="card-b"><span class="spin"></span> Pooling every run for insights…</div></div>
    <div class="card"><div class="card-h"><h2>Datasets</h2><span class="hint">Scored by exact verifiers only. Licences: GSM8K MIT, HumanEval MIT, MBPP CC-BY-4.0, task sets Apache-2.0.</span></div>
      <div class="card-b grid g4">${datasets.map(d => `<div class="ex" data-ds="${esc(d.name)}"><b>${esc(d.name)}</b><span>${d.tasks} tasks · ${d.verifiable} verifiable<br>${Object.entries(d.domains).map(([k, v]) => `${esc(k)} ${v}`).join(' · ')}</span></div>`).join('')}</div></div>
    <div class="card"><div class="card-h"><h2>Experiments</h2><span class="hint">Research questions about how System 1 (<i>nafs</i>) and System 2 (<i>\u02bfaql</i>) should interact. Pick one to fill the form.</span><span class="sp"></span><button class="btn sm ghost" id="eGloss">${icon.book}Metrics glossary</button></div>
      <div class="card-b grid g3">${PRESETS.map((p, i) => `<div class="ex preset" data-preset="${i}" tabindex="0"><b>${esc(p.t)}</b><span class="q">${esc(p.q)}</span><span>${esc(p.d)}</span><span class="tiny faint">${esc(p.ds)} · ${p.n} items · ${p.sys.length} systems</span></div>`).join('')}</div></div>
    <div class="card"><div class="card-h"><h2>New run</h2><span class="hint">Every selected system runs on identical tasks (paired design, FR-050/051).</span></div>
      <div class="card-b"><div class="row" style="align-items:flex-end;gap:14px">
        <label class="f">Dataset<select id="eDs">${datasets.map(d => `<option value="${esc(d.name)}" ${d.name === 'smoke_mixed' ? 'selected' : ''}>${esc(d.name)} (${d.tasks})</option>`).join('')}</select></label>
        <label class="f">Items<input id="eN" type="number" value="6" min="1" max="2000" style="width:80px"></label>
        <label class="f">Offset<input id="eOff" type="number" value="0" min="0" style="width:80px"></label>
        <label class="f">Thinking budget<select id="eB"><option value="">per domain</option><option>256</option><option>512</option><option>1024</option></select></label>
        <label class="f">λ<input id="eLam" type="number" value="0.2" step="0.05" min="0" max="1" style="width:70px"></label>
        <label class="toggle small"><input type="checkbox" id="eShuf"> shuffle</label>
        <span class="sp"></span><button class="btn" id="eStart">${icon.play}Start</button><button class="btn ghost" id="eCancel" hidden>Cancel</button></div>
        <div class="f" style="margin-top:12px"><span>Systems</span><div class="row" style="flex-wrap:wrap">${Object.keys(SYS_LABEL).map(k => `<label class="toggle small" title="${esc(SYS_HELP[k])}"><input type="checkbox" class="eSys" value="${k}" ${['nafs_aql', 'fast_only', 'slow_only'].includes(k) ? 'checked' : ''}>${sysLabel(k)}</label>`).join('')}
          <label class="toggle small" title="Budget curve (P0-T05, H4): deliberate-only at several thinking budgets"><input type="checkbox" id="eSweep">budget sweep</label><input id="eSweepB" type="text" value="256,512,1024" style="width:120px" title="Budgets for the sweep">
          <label class="toggle small" title="Frontier (Prop. 2): the blind gate at several prices of latency λ"><input type="checkbox" id="eLamSweep">λ sweep</label><input id="eLamB" type="text" value="0.05,0.2,0.6" style="width:110px" title="λ values for the sweep"></div></div>
        <p class="hint" style="margin:10px 0 0" id="eEst"></p>
        <div id="eProg" hidden style="margin-top:10px"><div class="progress"><i id="eBar"></i></div><div class="small muted" id="eProgTxt" style="margin-top:6px"></div></div></div></div>
    <div id="eResults"></div>
    <div class="card"><div class="card-h"><h3>Run history</h3><span class="hint">Select runs with the checkboxes to derive measured gate parameters from them (Phase 0).</span><span class="sp"></span><button class="btn sm ghost" id="eDerive" disabled>Derive parameters from selected</button></div><div class="card-b" style="padding:0" id="eHist"></div></div></div>`;
  api.get('/v1/gates').then(g => { if (!alive()) return; const C = { pass: 'b-verified', fail: 'b-abstained', pending: 'b-consensus' };
    $('#gatesCard').innerHTML = `<div class="card-h"><h2>Roadmap gates</h2><span class="hint">Go / no-go status computed from the stored runs and records (FR-053). Pending means the evidence is not yet enough to decide.</span></div>
      <div class="card-b gates">${g.gates.map(x => `<details class="gate ${x.status}"><summary><b>${esc(x.gate)}</b> <span>${esc(x.title)}</span><span class="sp"></span><span class="badge ${C[x.status]}">${esc(x.status)}</span></summary>
        <ul>${x.criteria.map(c => `<li><span class="badge ${C[c.status]}">${esc(c.status)}</span> <b>${esc(c.id)}</b> ${esc(c.criterion)}<div class="tiny muted">${esc(c.evidence)}</div></li>`).join('')}</ul></details>`).join('')}</div>`; }).catch(() => { if (alive()) $('#gatesCard').innerHTML = '<div class="card-b muted small">Gate status unavailable.</div>'; });
  const POL = { fast: ['Fast only', 'var(--fast)', 'nafs'], gated: ['Gate + verifier', 'var(--accent)', '\u02bfaql decides'], blind: ['Blind gate', 'var(--teal)', '\u02bfaql estimates'], slow: ['Deliberate only', 'var(--slow)', '\u02bfaql always'] };
  const loadInsights = async (lam) => { const ins = await api.get('/v1/lab/insights' + (lam != null ? `?lambda=${lam}` : '')).catch(() => null); if (!alive() || !ins) return;
    const D = ins.domains || [];
    $('#insCard').innerHTML = `<div class="card-h"><h2>Insights: when should System 2 be consulted?</h2><span class="hint">Pooled over ${ins.runs} stored runs. Each cell is the policy with the lowest expected loss L·(1−accuracy) + λ·cost at that price of latency (Prop. 2).</span></div>
      <div class="card-b">${D.length ? `<div style="overflow:auto"><table class="t policy-map"><tr><th>Domain</th>${ins.lambda_grid.map(l => `<th title="price of one minute of extra latency">λ ${+(+l).toFixed(2)}</th>`).join('')}<th title="Value of deliberation: accuracy of System 2 minus System 1 on the same items, with a 95% bootstrap interval">S2 − S1</th><th title="Share of Both trials in chat where people preferred the System 2 answer">Prefer S2</th></tr>
        ${D.map(d => `<tr><td class="dom"><b>${esc(d.domain)}</b><table class="mini">${d.policies.map(p => `<tr><td>${POL[p.policy]?.[0] || p.policy}</td><td>${pct(p.accuracy, 0)}</td><td>${num(p.mean_cost_s, 0)} s</td><td class="faint">n ${p.n}</td></tr>`).join('')}</table></td>
          ${ins.lambda_grid.map(l => { const b = (d.policy_map || []).find(x => Math.abs(x.lambda - l) < 1e-4); return b ? `<td><span class="pm" style="background:${POL[b.policy]?.[1]}" title="${esc(POL[b.policy]?.[2] || '')} · expected loss ${num(b.expected_loss, 3)} · n ${b.n} · ${b.confidence} confidence">${esc(POL[b.policy]?.[0] || b.policy)}</span>${b.confidence === 'low' ? '<span class="tiny faint"> ?</span>' : ''}</td>` : '<td class="faint">–</td>'; }).join('')}
          <td class="small">${d.value_of_deliberation ? `<b>${d.value_of_deliberation.delta_accuracy >= 0 ? '+' : ''}${(d.value_of_deliberation.delta_accuracy * 100).toFixed(0)} pts</b> ${d.value_of_deliberation.ci95 ? `[${(d.value_of_deliberation.ci95[0] * 100).toFixed(0)}, ${(d.value_of_deliberation.ci95[1] * 100).toFixed(0)}]` : ''} for +${num(d.value_of_deliberation.extra_cost_s, 0)} s <span class="faint">(${d.value_of_deliberation.pairs} pairs)</span>` : '<span class="faint">needs fast-only and deliberate-only on the same tasks</span>'}</td>
          <td class="small">${d.human_preference && (d.human_preference.slow + d.human_preference.fast) ? `${pct(d.human_preference.slow / (d.human_preference.slow + d.human_preference.fast), 0)} of ${d.human_preference.slow + d.human_preference.fast}` : '<span class="faint">no trials</span>'}</td></tr>`).join('')}</table></div>
        <p class="hint">${esc(ins.note)} "?" marks low confidence (fewer than 30 items). Run the presets below to fill the map.</p>` : '<div class="empty small">No runs yet. Start with the Frontier or Full system preset.</div>'}</div>`; };
  loadInsights();
  const systems = () => { const xs = $$('.eSys:checked').map(x => x.value); if ($('#eSweep').checked) xs.push(...$('#eSweepB').value.split(',').map(x => parseInt(x.trim(), 10)).filter(b => b >= 16 && b <= 8192).map(b => `slow_b${b}`));
    if ($('#eLamSweep').checked) xs.push(...$('#eLamB').value.split(',').map(x => Math.round(parseFloat(x) * 100)).filter(v => v >= 0 && v <= 100).map(v => `nafs_blind_l${String(v).padStart(2, '0')}`)); return [...new Set(xs)]; };
  $('#eGloss').onclick = glossary;
  $$('[data-preset]').forEach(el => { const go = () => { const p = PRESETS[+el.dataset.preset]; if (![...$('#eDs').options].some(o => o.value === p.ds)) return toast(`Dataset ${p.ds} is not built yet`);
    $('#eDs').value = p.ds; $('#eN').value = p.n; $('#eOff').value = 0; $('#eShuf').checked = true; $('#eSweep').checked = false; $('#eLamSweep').checked = false;
    $$('.eSys').forEach(c => { c.checked = p.sys.includes(c.value); });
    const extra = p.sys.filter(x => !Object.keys(SYS_LABEL).includes(x));
    if (extra.some(x => x.startsWith('slow_b'))) { $('#eSweep').checked = true; $('#eSweepB').value = extra.filter(x => x.startsWith('slow_b')).map(x => x.slice(6)).join(','); }
    if (extra.some(x => x.startsWith('nafs_blind_l'))) { $('#eLamSweep').checked = true; $('#eLamB').value = extra.filter(x => x.startsWith('nafs_blind_l')).map(x => (+x.slice(12) / 100).toString()).join(','); }
    est(); toast(`Preset loaded: ${p.t}. Press Start.`); $('#eStart').scrollIntoView({ behavior: 'smooth', block: 'center' }); };
    el.onclick = go; el.onkeydown = (e) => e.key === 'Enter' && go(); });
  const est = () => { const n = +$('#eN').value; const sy = systems(); const slow = sy.filter(x => x.startsWith('slow') || x === 'nafs_aql' || x === 'nafs_blind').length; $('#eEst').textContent = `${n * sy.length} model runs. On the reference CPU a deliberate run takes ~1–4 min and a fast run ~5–30 s: roughly ${Math.round(n * (slow * 1.5 + (sy.length - slow) * 0.3))}–${Math.round(n * (slow * 3 + (sy.length - slow) * 0.5))} minutes.`; };
  ['#eN', '.eSys', '#eSweep', '#eSweepB', '#eLamSweep', '#eLamB'].forEach(q => $$(q).forEach(el => el.addEventListener('input', est))); est();
  $$('[data-ds]').forEach(el => el.onclick = async () => { $('#eDs').value = el.dataset.ds; const items = await api.get(`/v1/eval/datasets/${encodeURIComponent(el.dataset.ds)}?limit=8`);
    modal(`<h3 style="margin-top:0">${esc(el.dataset.ds)}: first ${items.length} tasks</h3>${items.map(t => `<div style="margin-bottom:10px"><span class="badge b-domain">${esc(t.domain)}</span> <span class="mono tiny">${esc(t.id)}</span><div class="codebox" style="margin-top:4px">${esc(t.prompt.slice(0, 600))}</div><div class="tiny muted">verifier: ${esc(Object.keys(t.verification || {}).join(', ') || 'none')}</div></div>`).join('')}`); });
  let poll = null, current = null; const selected = new Set();
  const ci = (c) => c ? `[${(c[0] * 100).toFixed(1)}, ${(c[1] * 100).toFixed(1)}]` : '–';
  const render = (r) => { current = r; const sys = r.systems || {}; const keys = Object.keys(sys);
    const prog = r.progress || {}; $('#eProg').hidden = false; $('#eBar').style.width = prog.total ? `${prog.done / prog.total * 100}%` : '0';
    $('#eProgTxt').textContent = `${r.status === 'running' ? 'Running' : r.status === 'failed' ? 'Failed: ' + (r.error || '') : r.status === 'cancelled' ? 'Cancelled' : r.status === 'interrupted' ? 'Interrupted by a restart' : 'Finished'} · ${prog.done ?? 0} / ${prog.total ?? '?'} model runs · dataset ${r.request?.benchmark ?? ''}`;
    $('#eStart').hidden = r.status === 'running'; $('#eCancel').hidden = r.status !== 'running';
    const bars = keys.map(k => ({ label: SYS_LABEL[k] || k, v: sys[k].accuracy, color: sysColor(k) }));
    const lat = keys.map(k => ({ label: SYS_LABEL[k] || k, v: sys[k].mean_wall_ms, color: sysColor(k) }));
    const assureStack = (a, n) => { const C = { verified: 'var(--ok)', consensus: 'var(--warn)', unverified: 'var(--faint)', abstained: 'var(--bad)', error: 'var(--crimson)' };
      return `<div class="stack" title="${esc(Object.entries(a || {}).map(([k, v]) => `${k}: ${v}`).join(', '))}">${Object.entries(a || {}).map(([k, v]) => `<i style="width:${v / Math.max(1, n) * 100}%;background:${C[k] || 'var(--slate)'}"></i>`).join('')}</div>`; };
    const paired = (r.paired || []).map(p => { const L = (x) => 50 + x * 100; const c = p.ci95; return `<tr><td>${sysLabel(p.a)} − ${sysLabel(p.b)}</td><td>${p.n_pairs}</td><td><b>${p.delta_accuracy >= 0 ? '+' : ''}${(p.delta_accuracy * 100).toFixed(1)} pts</b></td><td>${ci(c)}</td>
        <td style="width:180px">${c ? `<div class="forest"><span class="ci" style="left:${Math.max(0, L(c[0]))}%;width:${Math.max(1, (c[1] - c[0]) * 100)}%"></span><span class="pt" style="left:${Math.min(100, Math.max(0, L(p.delta_accuracy)))}%"></span></div>` : '<span class="faint small">after the run</span>'}</td>
        <td title="Exact McNemar on discordant pairs (${p.discordant?.only_a_correct ?? 0} vs ${p.discordant?.only_b_correct ?? 0}); Bonferroni-adjusted across the run's comparisons">${p.mcnemar_p == null ? '–' : num(p.mcnemar_p, 3)} / ${p.p_bonferroni == null ? '–' : num(p.p_bonferroni, 3)}</td>
        <td>${p.cost_ratio == null ? '–' : num(p.cost_ratio, 2) + '×'}</td><td>${p.non_inferior_3pt == null ? '–' : p.non_inferior_3pt ? '<span class="badge b-verified">yes</span>' : '<span class="badge b-unverified">no</span>'}${p.n_pairs < 700 ? ' <span class="badge b-consensus" title="Release gate G2 needs ≥ 700 pairs (3-point margin)">indicative, n &lt; 700</span>' : ''}</td></tr>`; }).join('');
    const routing = Object.entries(r.routing || {}).map(([k, rt]) => { const conf = rt.confusion || {}; return `<div class="card"><div class="card-h"><h3>Routing vs the oracle · ${sysLabel(k)}</h3>${rt.measures_gate ? '<span class="badge b-verified">measures the gate</span>' : '<span class="badge b-unverified">verifier-driven</span>'}</div><div class="card-b">
        <div class="row" style="gap:24px"><div class="tile" style="padding:0"><div class="k">Routing accuracy</div><div class="v">${pct(rt.routing_accuracy, 0)}</div><div class="s">target ≥ 70% at g ≤ 30% (H1)</div></div><div class="tile" style="padding:0"><div class="k">Escalation rate g</div><div class="v">${pct(rt.escalation_rate, 0)}</div><div class="s">n = ${rt.n}</div></div></div>
        <table class="t" style="margin-top:12px"><tr><th></th><th>Oracle: escalate</th><th>Oracle: keep</th></tr><tr><td>Gate escalated</td><td><b>${conf.escalate_needed ?? 0}</b> needed</td><td>${conf.escalate_wasted ?? 0} wasted</td></tr><tr><td>Gate kept</td><td>${conf.keep_missed ?? 0} missed</td><td><b>${conf.keep_correct ?? 0}</b> correct</td></tr></table>
        <p class="hint">${esc(rt.note)}</p></div></div>`; }).join('');
    const cal = r.calibration;
    const calCard = cal ? `<div class="card"><div class="card-h"><h3>Calibration of ê₁</h3><button class="btn sm" id="eFit" title="Fit the gate calibration from this run's labelled drafts (5-fold held-out ECE; promoted only if NFR-020 holds)">Fit calibration</button></div><div class="card-b">
        <div class="row" style="gap:24px"><div class="tile" style="padding:0"><div class="k">ECE (5 bins)</div><div class="v">${num(cal.ece_e1, 3)}</div><div class="s">target ≤ 0.05 on ≥ 2,900 (NFR-020)</div></div><div class="tile" style="padding:0"><div class="k">Brier</div><div class="v">${num(cal.brier_e1, 3)}</div><div class="s">n = ${cal.n} labelled drafts</div></div><div class="tile" style="padding:0"><div class="k">Draft error rate</div><div class="v">${pct(cal.draft_error_rate, 0)}</div><div class="s">${esc(cal.calibration_in_use)}</div></div></div>
        <div style="max-width:340px;margin:8px auto 0">${chart.reliability(cal.reliability)}</div><p class="hint">${esc(cal.note)}</p></div></div>` : '';
    const cp = r.coupling; const cpRow = (k, l) => { const x = cp?.[k]; return x ? `<tr><td>${l}</td><td>${x.wrong_drafts}</td><td>${x.adopted}</td><td><b>${pct(x.adoption_rate)}</b></td><td>≤ ${pct(x.upper95)}</td><td>${x.h5_pass ? '<span class="badge b-verified">pass</span>' : '<span class="badge b-unverified">not shown</span>'}${x.er004_eligible ? '' : ' <span class="badge b-consensus">n &lt; 200</span>'}</td></tr>` : ''; };
    const cpCard = cp ? `<div class="card"><div class="card-h"><h3>Coupling test (ER-004, H5)</h3><span class="hint">How often System 2 repeats a wrong System 1 draft.</span></div><div class="card-b" style="padding:0"><table class="t"><tr><th>Mode</th><th>Wrong drafts</th><th>Adopted</th><th>Adoption rate</th><th>95% upper bound</th><th>H5 (≤ 5%)</th></tr>${cpRow('draft_blind', 'Draft-blind (production)')}${cpRow('draft_conditioned', 'Draft-conditioned (ablation)')}</table><p class="hint" style="padding:0 16px">${esc(cp.note)}</p></div></div>` : '';
    const bc = r.budget_curve; const pal = ['var(--accent)', 'var(--slow)', 'var(--fast)', 'var(--teal)', 'var(--amber)', 'var(--crimson)'];
    const fr = r.frontier; const frCard = fr ? `<div class="card" style="margin-top:16px"><div class="card-h"><h3>Accuracy–cost frontier (Prop. 2)</h3><span class="hint">Each point is a policy for combining System 1 and System 2. Filled points are Pareto-optimal: no other policy is both more accurate and cheaper.</span></div><div class="card-b">${chart.frontier(fr)}
      <table class="t small"><tr><th>Policy</th><th>n</th><th>Accuracy</th><th>Mean cost</th><th>vs deliberate-only</th><th></th></tr>${[...fr].sort((a, b) => a.mean_cost_s - b.mean_cost_s).map(p => `<tr><td><span class="dot" style="background:${sysColor(p.system)}"></span> ${sysLabel(p.system)}</td><td>${p.n}</td><td>${pct(p.accuracy, 0)}</td><td>${num(p.mean_cost_s, 0)} s</td><td>${p.cost_vs_slow == null ? '–' : num(p.cost_vs_slow, 2) + '×'}</td><td>${p.pareto ? '<span class="badge b-verified">efficient</span>' : ''}</td></tr>`).join('')}</table></div></div>` : '';
    const bcCard = bc ? `<div class="card"><div class="card-h"><h3>Accuracy vs thinking budget (H4, Prop. 9)</h3></div><div class="card-b">${chart.line(Object.entries(bc).map(([d, v], i) => ({ label: d, color: pal[i % pal.length], points: v.points.map(p => ({ x: p.budget, y: p.accuracy, title: `budget ${p.budget}: ${pct(p.accuracy, 0)} (n ${p.n}), ${num(p.mean_wall_s, 0)} s` })) })), { xLabel: 'thinking budget (tokens)', yLabel: 'accuracy', xLog: true })}
        <table class="t"><tr><th>Domain</th><th>Points</th><th>b* = argmin L(1−acc) + λ·c</th></tr>${Object.entries(bc).map(([d, v]) => `<tr><td>${esc(d)}</td><td class="small">${v.points.map(p => `${p.budget}: ${pct(p.accuracy, 0)} · ${num(p.mean_wall_s, 0)} s`).join(' · ')}</td><td><b>${v.b_star ?? '–'}</b></td></tr>`).join('')}</table></div></div>` : '';
    $('#eResults').innerHTML = `<div class="grid g2">
      <div class="card"><div class="card-h"><h3>Accuracy (exact verifiers)</h3></div><div class="card-b">${chart.bars(bars, { max: 1, fmt: v => v == null ? '–' : pct(v, 0) })}</div></div>
      <div class="card"><div class="card-h"><h3>Mean latency per task</h3></div><div class="card-b">${chart.bars(lat, { fmt: v => fmtMs(v) })}</div></div></div>
      <div class="card" style="margin-top:16px"><div class="card-h"><h3>Systems</h3></div><div class="card-b" style="padding:0;overflow:auto"><table class="t"><tr><th>System</th><th>n</th><th>Accuracy</th><th>95% CI</th><th>Selective acc.</th><th>Coverage</th><th>Escalated</th><th>Mean / p95 latency</th><th>TTFT</th><th>Model calls</th><th>Reasoning tok</th><th>Assurance</th></tr>
        ${keys.map(k => { const x = sys[k]; return `<tr><td><span class="dot" style="background:${sysColor(k)}"></span> ${sysLabel(k)}</td><td>${x.n}${x.errors ? ` <span class="badge b-abstained">${x.errors} err</span>` : ''}</td><td><b>${pct(x.accuracy)}</b></td><td class="small">${ci(x.accuracy_ci95)}</td><td>${pct(x.selective_accuracy)}</td><td>${pct(x.coverage)}</td><td>${pct(x.escalation_rate, 0)}</td><td>${fmtMs(x.mean_wall_ms)} / ${fmtMs(x.p95_wall_ms)}</td><td>${fmtMs(x.mean_ttft_ms)}</td><td>${num(x.mean_model_calls, 1)}</td><td>${num(x.mean_reasoning_tokens, 0)}</td><td>${assureStack(x.assurance, x.n)}</td></tr>`; }).join('')}</table></div></div>
      ${paired ? `<div class="card" style="margin-top:16px"><div class="card-h"><h3>Paired comparisons</h3><span class="hint">Bootstrap 95% CI. Gate G2 needs ≥ 700 pairs (3-point margin); G6 ≥ 6,200 pairs (1 point).</span></div><div class="card-b" style="padding:0;overflow:auto"><table class="t"><tr><th>Comparison</th><th>Pairs</th><th>Δ accuracy</th><th>95% CI</th><th>−50 … 0 … +50 pts</th><th title="McNemar p / Bonferroni-adjusted p">p / p adj.</th><th>Cost ratio</th><th>Non-inferior (3 pt)</th></tr>${paired}</table></div></div>` : ''}
      ${frCard}
      ${routing || calCard ? `<div class="grid g2" style="margin-top:16px">${routing}${calCard}</div>` : ''}
      ${cpCard || bcCard ? `<div class="grid g2" style="margin-top:16px">${cpCard}${bcCard}</div>` : ''}
      <div class="card" style="margin-top:16px"><div class="card-h"><h3>Items</h3><span class="sp"></span>${r.run_id ? `<a class="btn sm ghost" href="/v1/eval/runs/${esc(r.run_id)}/export?format=md" data-dl>${icon.download}Report (.md)</a><a class="btn sm ghost" href="/v1/eval/runs/${esc(r.run_id)}/export?format=csv" data-dl>${icon.download}Items (.csv)</a>` : ''}<div class="row"><select id="eFilt"><option value="all">all</option><option value="wrong">incorrect</option><option value="right">correct</option><option value="slow">escalated</option><option value="err">errors</option></select></div></div><div class="card-b" style="padding:0;max-height:480px;overflow:auto" id="eItems"></div></div>`;
    const items = () => { const f = $('#eFilt')?.value || 'all'; const all = r.items || []; const its = all.filter(i => f === 'all' || (f === 'wrong' && i.correct === false) || (f === 'right' && i.correct === true) || (f === 'slow' && i.escalated) || (f === 'err' && i.error));
      $('#eItems').innerHTML = `<table class="t"><tr><th>Task</th><th>System</th><th>Result</th><th>Path</th><th>Assurance</th><th>ê₁</th><th>Draft</th><th>Latency</th><th>Answer</th></tr>${its.slice(0, 600).map(i => `<tr class="click" data-ix="${all.indexOf(i)}"><td class="mono tiny">${esc(i.task_id)}</td><td>${sysLabel(i.system)}</td><td>${i.error ? '<span class="badge b-abstained">error</span>' : i.correct == null ? '<span class="badge b-unverified">n/a</span>' : i.correct ? `<span class="badge b-verified">${icon.check}correct</span>` : `<span class="badge b-abstained">${icon.x}wrong</span>`}</td><td>${i.mode === 'slow' ? modeBadge('slow') : i.mode === 'fast' ? modeBadge('fast') : ''}</td><td>${ASSURE[i.assurance] ? assureBadge(i.assurance) : esc(i.assurance)}</td><td>${num(i.e1_hat, 2)}</td><td>${i.draft_correct == null ? '–' : i.draft_correct ? '✓' : '✗'}</td><td>${fmtMs(i.wall_ms)}</td><td class="small" style="max-width:340px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${esc(i.error || i.answer_preview)}</td></tr>`).join('')}</table>${its.length > 600 ? `<p class="hint" style="padding:0 16px">Showing 600 of ${its.length}.</p>` : ''}`;
      $$('#eItems tr.click').forEach(tr => tr.onclick = () => { const i = all[+tr.dataset.ix]; modal(`<h3 style="margin-top:0">${esc(i.task_id)} · ${sysLabel(i.system)}</h3><div class="row">${i.mode !== 'error' ? modeBadge(i.mode) : ''}${ASSURE[i.assurance] ? assureBadge(i.assurance) : ''}<span class="badge b-domain">${esc(i.rule)}</span><span class="small muted">${fmtMs(i.wall_ms)} · ${i.model_calls} model calls · ${i.reasoning_tokens} reasoning tokens${i.attempts ? ` · ${i.attempts} S2 attempts${i.first_pass_attempt ? ', passed at ' + i.first_pass_attempt : ''}` : ''}</span></div>
          <h4>Answer</h4><div class="answer">${md(i.answer_preview)}</div>${i.evidence ? `<h4>Verifier evidence</h4><div class="codebox">${esc(i.evidence)}</div>` : ''}${i.error ? `<div class="notice bad">${esc(i.error)}</div>` : ''}
          <h4>Gate features</h4><div class="codebox">${esc(JSON.stringify(i.features, null, 1))}</div>${i.decision_id ? `<p><button class="linkbtn" data-decision="${esc(i.decision_id)}">decision record</button></p>` : ''}`); }); };
    items(); $('#eFilt') && ($('#eFilt').onchange = items);
    $('#eFit') && ($('#eFit').onclick = async () => { try { const c = await api.post(`/v1/eval/runs/${r.run_id}/calibrate`, { include_ledger: false }); showCalibration(c); } catch (e) { toast(e.message, 6000); } }); };
  const history = async () => { const runs = await api.get('/v1/eval/runs').catch(() => []); if (!alive()) return;
    const cols = ['nafs_aql', 'nafs_blind', 'fast_only', 'slow_only'];
    $('#eHist').innerHTML = runs.length ? `<table class="t"><tr><th></th><th>When</th><th>Dataset</th><th>Progress</th>${cols.map(k => `<th>${sysLabel(k)}</th>`).join('')}<th>Other</th><th></th></tr>${runs.map(r => `<tr class="click" data-id="${esc(r.run_id)}"><td><input type="checkbox" class="eSel" value="${esc(r.run_id)}" ${selected.has(r.run_id) ? 'checked' : ''}></td><td>${ago(r.started_at || r.created_at || new Date().toISOString())}</td><td>${esc(r.request?.benchmark)}</td><td>${r.progress?.done ?? 0}/${r.progress?.total ?? '?'}</td>${cols.map(k => `<td>${r.systems?.[k] ? pct(r.systems[k].accuracy, 0) : '–'}</td>`).join('')}<td class="small">${Object.keys(r.systems || {}).filter(k => !cols.includes(k)).map(sysLabel).join(', ')}</td><td>${r.status === 'running' ? '<span class="spin"></span>' : esc(r.status)}${r.status === 'interrupted' || r.status === 'cancelled' ? ` <button class="btn sm ghost" data-resume="${esc(r.run_id)}" title="Continue from the completed items">Resume</button>` : ''}</td></tr>`).join('')}</table>` : '<div class="empty small">No evaluation runs yet. Start with <b>smoke_mixed</b> and 4–6 items.</div>';
    $$('#eHist .eSel').forEach(cb => { cb.onclick = (e) => e.stopPropagation(); cb.onchange = () => { cb.checked ? selected.add(cb.value) : selected.delete(cb.value); $('#eDerive').disabled = !selected.size; }; });
    $$('#eHist [data-resume]').forEach(b => b.onclick = async (e) => { e.stopPropagation(); try { const r = await api.post(`/v1/eval/runs/${b.dataset.resume}/resume`); toast(`Resumed with ${r.kept_items} completed items kept`); watch(b.dataset.resume); history(); } catch (x) { toast(x.message); } });
    $$('#eHist tr.click').forEach(tr => tr.onclick = () => watch(tr.dataset.id)); };
  const watch = (id) => { clearInterval(poll); const tick = async () => { if (!alive()) return clearInterval(poll); const r = await api.get('/v1/eval/runs/' + id).catch(() => null); if (!r || !alive()) return; render(r); if (r.status !== 'running') { clearInterval(poll); history(); } }; tick(); poll = setInterval(tick, 3000); };
  $('#eStart').onclick = async () => { const body = { benchmark: $('#eDs').value, max_items: +$('#eN').value, offset: +$('#eOff').value, systems: systems(), lambda: +$('#eLam').value, shuffle: $('#eShuf').checked };
    if ($('#eB').value) body.thinking_budget = +$('#eB').value; if (!body.systems.length) return toast('Select at least one system');
    try { const r = await api.post('/v1/eval/runs', body); toast('Evaluation started'); watch(r.run_id); history(); } catch (e) { toast(e.message, 5000); } };
  $('#eCancel').onclick = async () => { if (current) { await api.post(`/v1/eval/runs/${current.run_id}/cancel`).catch(() => {}); toast('Cancelling the current item…'); } };
  $('#eDerive').onclick = () => deriveDialog([...selected]);
  await history(); if (!alive()) return; const runs = await api.get('/v1/eval/runs').catch(() => []); if (runs[0] && alive()) watch(runs[0].run_id);
  return () => clearInterval(poll);
} };

function showCalibration(c) {
  const k = c.calibration; const req = c.requirements || {};
  modal(`<h3 style="margin-top:0">Calibration ${c.promoted ? 'promoted' : 'not promoted'}</h3><p>${esc(c.note)}</p>${c.reason ? `<div class="notice">${icon.alert}<span>${esc(c.reason)}</span></div>` : ''}
    <dl class="kv"><dt>Version</dt><dd>${esc(k.version)}</dd><dt>Samples</dt><dd>${k.n} (${k.n_pos} wrong drafts, ${k.n_neg} right) · need ≥ ${req.min_n ?? '?'} with ≥ ${req.min_per_class ?? '?'} per class</dd>
    <dt>Held-out ECE</dt><dd><b>${num(k.ece, 3)}</b> (${k.folds}-fold) · target ≤ ${req.max_ece ?? 0.05}${k.ece_in_sample != null ? ` · in-sample ${num(k.ece_in_sample, 3)} (not used)` : ''}</dd><dt>Brier (held-out)</dt><dd>${num(k.brier, 3)}</dd>
    <dt>Weights</dt><dd class="mono tiny">${k.feature_names.map((n, j) => `${esc(n)}: ${num(k.weights[j], 3)}`).join('<br>')}</dd><dt>Active</dt><dd>${c.active ? 'yes' : 'no: the previous calibration or the fixed rule stays in force'}</dd></dl>
    ${k.reliability?.length ? `<div style="max-width:320px">${chart.reliability(k.reliability.map(([p, o, n]) => ({ predicted: p, observed: o, count: n })))}</div>` : ''}`);
}
async function deriveDialog(ids) {
  try {
    const r = await api.post('/v1/gate/derive', { run_ids: ids, apply: false, min_n: 10 }); const p = r.params; const det = p.details || {};
    const rows = Object.entries(det).filter(([d]) => d !== 'budget_curve').map(([d, x]) => `<tr><td>${esc(d)}</td><td>${x.n_slow ?? 0} / ${x.n_fast ?? 0}</td><td>${x.e2_hat ? `<b>${num(x.e2_hat.value, 3)}</b> <span class="tiny faint">(${x.e2_hat.wrong}/${x.e2_hat.n})</span>` : '<span class="faint">placeholder</span>'}</td><td>${x.delta_c_s ? `<b>${num(x.delta_c_s.value, 0)} s</b>` : '<span class="faint">placeholder</span>'}</td><td>${x.beta_fit ? `α ${num(x.beta_fit.alpha, 2)}, β ${num(x.beta_fit.beta, 2)} → M* ${x.beta_fit.M_star} (N 1, R ${x.beta_fit.R})` : '<span class="faint">–</span>'}</td><td>${p.budgets?.[d] ?? '–'}</td></tr>`).join('');
    modal(`<h3 style="margin-top:0">Measured gate parameters (preview)</h3><p class="small muted">From ${ids.length} run(s). Domains with fewer than 10 deliberate-only items keep their placeholders. Applying records the derivation with the configuration (FR-021).</p>
      <table class="t"><tr><th>Domain</th><th>n slow / fast</th><th>ê₂</th><th>Δc</th><th>Beta fit → caps (Prop. 3)</th><th>b*</th></tr>${rows}</table>
      <div class="row" style="margin-top:12px"><button class="btn sm" id="dApply" type="button">Apply these values</button></div>`);
    $('#dApply').onclick = async () => { try { await api.post('/v1/gate/derive', { run_ids: ids, apply: true, min_n: 10 }); $('#modal').close(); toast('Measured parameters applied'); } catch (e) { toast(e.message, 5000); } };
  } catch (e) { toast(e.message, 6000); }
}

// ============================== gate and ledger ==============================
routes.gate = { title: 'Muhasaba · gate and ledger', async render(view, alive) {
  view.innerHTML = `<div class="page grid" style="gap:16px">
    <div class="grid g2"><div class="card" id="gMode"><div class="card-b"><span class="spin"></span></div></div><div class="card" id="gCal"></div></div>
    <div class="card" id="gParams"></div>
    <div class="grid g2"><div class="card" id="gRates"></div><div class="card" id="gJudge"></div></div>
    <div class="card" id="gSelf"></div>
    <div class="grid g2"><div class="card" id="gPref"></div><div class="card" id="gDrift"></div></div>
    <div class="card" id="gAlerts"></div>
    <div class="card" id="gAudits"></div></div>`;
  let ratePoll = null;
  const load = async () => {
    const [g, sm, au, rates, rruns, pref, drift] = await Promise.all([api.get('/v1/gate'), api.get('/v1/self-model').catch(() => null), api.get('/v1/audits?limit=50').catch(() => null), api.get('/v1/verifiers').catch(() => null), api.get('/v1/verifiers/runs').catch(() => []), api.get('/v1/preferences').catch(() => null), api.get('/v1/drift').catch(() => null)]);
    if (!alive()) return; const cal = g.calibration, req = g.requirements;
    const eff = g.effective_rule;
    $('#gMode').innerHTML = `<div class="card-h"><h3>Aql Gate mode</h3><span class="badge ${eff === 'fixed' ? 'b-consensus' : 'b-verified'}">in force: ${esc(eff)}</span></div><div class="card-b">
      <div class="seg" id="gSeg">${['fixed', 'calibrated', 'budget'].map(m => `<button data-m="${m}" class="${g.mode === m ? 'on' : ''}" type="button">${m}</button>`).join('')}</div>
      <div class="grp slider" style="margin-top:12px" ${g.mode === 'budget' ? '' : 'hidden'} id="gBudgetRow" title="Target escalation rate B (FR-014)"><span>Escalation budget B</span><input type="range" id="gBudget" min="0.05" max="0.9" step="0.05" value="${g.escalation_budget}"><span class="mono" id="gBudgetV">${pct(g.escalation_budget, 0)}</span></div>
      <dl class="kv small" style="margin-top:12px"><dt>Rule</dt><dd>${eff === 'fixed' ? 'Phase-1 thresholds on log-probability and sample agreement' : eff === 'budget' ? `escalate iff L(ê₁−ê₂) &gt; λ*·Δc with λ* = ${num(g.lambda_star, 3)} chosen so that ${pct(g.escalation_budget, 0)} of requests escalate` : 'escalate iff L(ê₁−ê₂) &gt; λ·Δc (Theorem 2)'}</dd>
      <dt>Promotion</dt><dd>${g.promoted ? '<span class="badge b-verified">calibration passes NFR-020</span>' : `<span class="badge b-unverified">not promoted</span> <span class="small">${esc(g.promotion_reason || '')}</span>`}</dd>
      <dt>Budget mode</dt><dd>needs a promoted calibration with n ≥ ${req.dkw_n} for λ* within ±${num(req.epsilon, 3)} with ${pct(1 - req.delta, 0)} confidence (DKW, Prop. 6)</dd>
      <dt>Recalibration</dt><dd>${g.recalibrate_every_h ? `every ${g.recalibrate_every_h} h from the ledger` : 'manual'} · <button class="btn sm ghost" id="gRecal" type="button">Recalibrate now</button></dd></dl>
      <p class="hint">${g.mode !== eff ? `Mode <b>${esc(g.mode)}</b> is selected, but the fixed rule stays in force until a calibration is promoted.` : ''}</p></div>`;
    $$('#gSeg button').forEach(b => b.onclick = async () => { await api.post('/v1/gate', { mode: b.dataset.m }).catch(e => toast(e.message)); load(); });
    $('#gBudget') && ($('#gBudget').oninput = (e) => { $('#gBudgetV').textContent = pct(+e.target.value, 0); });
    $('#gBudget') && ($('#gBudget').onchange = async (e) => { await api.post('/v1/gate', { escalation_budget: +e.target.value }).catch(x => toast(x.message)); load(); });
    $('#gRecal').onclick = async () => { const b = $('#gRecal'); b.disabled = true; b.innerHTML = '<span class="spin"></span>'; try { showCalibration(await api.post('/v1/gate/recalibrate', {})); } catch (e) { toast(e.message, 6000); } finally { load(); } };
    $('#gCal').innerHTML = `<div class="card-h"><h3>Calibration of ê₁</h3><span class="badge b-domain">${esc(cal.version)}</span></div><div class="card-b">${cal.n ? `<div class="row" style="gap:20px"><div class="tile" style="padding:0"><div class="k">${cal.folds ? `Held-out ECE (${cal.folds}-fold)` : 'ECE (in-sample, older fit)'}</div><div class="v">${num(cal.ece, 3)}</div><div class="s">target ≤ ${req.max_ece}</div></div><div class="tile" style="padding:0"><div class="k">Samples</div><div class="v">${cal.n}</div><div class="s">need ≥ ${req.min_n}</div></div><div class="tile" style="padding:0"><div class="k">Wrong / right</div><div class="v">${cal.n_pos + cal.n_neg ? `${cal.n_pos} / ${cal.n_neg}` : '–'}</div><div class="s">≥ ${req.min_per_class} each</div></div></div>
      ${cal.reliability?.length ? `<div style="max-width:320px;margin:8px auto 0">${chart.reliability(cal.reliability.map(([p, o, n]) => ({ predicted: p, observed: o, count: n })))}</div>` : ''}` : '<p class="muted small">Uncalibrated prior in use. Labelled drafts come from verifiable requests and audits (the ledger) and from evaluation runs; fit one from the Lab page or press Recalibrate.</p>'}</div>`;
    const prov = g.params; const doms = Object.keys(prov);
    const src = (s) => s === 'measured' ? '<span class="badge b-verified">measured</span>' : '<span class="badge b-unverified">placeholder</span>';
    $('#gParams').innerHTML = `<div class="card-h"><h3>Gate parameters per domain</h3><span class="hint">Placeholders until derived from Phase-0 runs (FR-011, FR-021). ${g.derived?.derived_at ? `Derived ${ago(g.derived.derived_at)} from ${g.derived.source_runs.length} run(s).` : ''}</span><span class="sp"></span>${g.derived?.derived_at ? '<button class="btn sm ghost" id="gReset" type="button">Reset to placeholders</button>' : ''}</div><div class="card-b" style="padding:0;overflow:auto">
      <table class="t"><tr><th>Domain</th><th>ê₂</th><th></th><th>Δc</th><th></th><th>N</th><th>R</th><th>Budget</th><th></th></tr>${doms.map(d => { const x = prov[d]; return `<tr><td>${esc(d)}</td><td>${num(x.e2_hat, 3)}</td><td>${src(x.e2_source)}</td><td>${num(x.delta_c_s, 0)} s</td><td>${src(x.delta_c_source)}</td><td>${x.caps.N}</td><td>${x.caps.R}</td><td>${x.caps.budget}</td><td>${src(x.caps_source)} ${src(x.budget_source)}</td></tr>`; }).join('')}</table>
      <p class="hint" style="padding:0 16px">Select evaluation runs with fast_only and slow_only (and optionally a budget sweep) on the Lab page, then use "Derive parameters from selected".</p></div>`;
    $('#gReset') && ($('#gReset').onclick = async () => { if (await confirmDialog({ title: 'Discard measured parameters?', ok: 'Discard', danger: false, body: '<p>The gate goes back to the placeholder values in the configuration. Measured values can be derived again from the Lab.</p>' })) { await api.del('/v1/gate/params'); load(); } });
    const R = rates?.rates || {}; const running = (rruns || []).find(x => x.status === 'running');
    $('#gRates').innerHTML = `<div class="card-h"><h3 title="FR-032, ER-003">Verifier quality</h3><span class="sp"></span><select id="gRDs" title="Code test suites for the mutants"><option value="base">MBPP + HumanEval tests</option><option value="plus">EvalPlus tests (MBPP+, HumanEval+)</option><option value="judge">judge only</option></select><label class="toggle small"><input type="checkbox" id="gRJ"> include judge</label><button class="btn sm" id="gRRun" type="button" ${running ? 'disabled' : ''}>${running ? '<span class="spin"></span> measuring' : 'Measure'}</button></div><div class="card-b" style="padding:0">
      ${Object.keys(R).length ? `<table class="t"><tr><th>Verifier</th><th>r (true accept)</th><th>φ (false accept)</th><th>φ 95% upper</th><th>Mutants</th><th>Correct</th></tr>${Object.entries(R).map(([k, v]) => `<tr><td>${esc(k)}</td><td>${num(v.r, 3)}</td><td><b>${num(v.phi, 3)}</b></td><td>${num(v.phi_upper95, 3)}</td><td>${v.n_mutants}</td><td>${v.n_correct}</td></tr>`).join('')}</table>` : '<div class="empty small">Not measured yet.</div>'}
      ${running ? `<p class="hint" style="padding:0 16px">${esc(running.stage || '')} ${running.progress?.done ?? 0}/${running.progress?.total ?? '?'}</p>` : ''}<p class="hint" style="padding:0 16px">Known-correct answers give r; seeded wrong answers (mutants) give φ, with Clopper–Pearson bounds. Targets: φ upper bound ≤ 0.01 (code, math), 0 observed (schema).</p></div>`;
    $('#gRRun').onclick = async () => { const ds = $('#gRDs').value; const body = { include_judge: $('#gRJ').checked || ds === 'judge', judge_only: ds === 'judge', items_per_dataset: 210, code_datasets: ds === 'plus' ? ['mbpp_plus', 'humaneval_plus'] : ['mbpp', 'humaneval'] };
      try { await api.post('/v1/verifiers/runs', body); toast('Measuring verifier rates…'); load(); } catch (e) { toast(e.message); } };
    clearInterval(ratePoll); if (running) ratePoll = setInterval(() => alive() ? load() : clearInterval(ratePoll), 4000);
    if (!$('#gJudge').dataset.ready) { $('#gJudge').dataset.ready = '1';
      $('#gJudge').innerHTML = `<div class="card-h"><h3>Judge playground (FR-034)</h3><span class="hint">System 2 scores an answer; pass needs h, a, u ≥ 0.5 and V ≥ 0.7.</span></div><div class="card-b">
        <label class="f">Question<input id="jQ" type="text" value="Is it safe to look directly at the Sun during a partial eclipse?"></label>
        <label class="f" style="margin-top:8px">Answer<textarea id="jA" rows="3">Yes, sunglasses are enough during a partial eclipse.</textarea></label>
        <div class="row" style="margin-top:8px"><button class="btn sm" id="jRun" type="button">${icon.shield}Judge</button><span id="jOut"></span></div><div id="jDet"></div></div>`;
      $('#jRun').onclick = async () => { $('#jOut').innerHTML = '<span class="spin"></span> scoring (System 2)…'; $('#jDet').innerHTML = '';
        try { const v = await api.post('/v1/judge', { question: $('#jQ').value, answer: $('#jA').value }); $s('#jOut').innerHTML = `<span class="verif ${v.passed ? 'pass' : 'fail'}">${v.passed ? icon.check : icon.x} judge ${v.passed ? 'passed' : 'failed'} · V ${num(v.score, 2)} · ${fmtMs(v.duration_ms)}</span>`; $s('#jDet').innerHTML = judgeBars(v.details || {}); } catch (e) { $s('#jOut').textContent = e.message; } }; }
    const D = sm?.domains || [];
    $('#gSelf').innerHTML = `<div class="card-h"><h3>Muhasaba Ledger: capability per domain (FR-041, Prop. 5)</h3><span class="hint">Live traffic only${Object.keys(sm?.audits || {}).length ? ` · audits ${Object.entries(sm.audits).map(([k, v]) => `${esc(k)} ${esc(String(v))}`).join(', ')}` : ''}</span></div><div class="card-b" style="padding:0;overflow:auto">${D.length ? `<table class="t"><tr><th>Domain</th><th>Replies</th><th>Escalated g</th><th>p₂ | escalated</th><th>p₂ | audit</th><th>p̂₂ (unbiased)</th><th>± bound</th><th>p₁ on kept</th><th>Kept vs S2 agreement</th><th>Audits</th><th>Status</th></tr>
      ${D.map(d => `<tr><td>${esc(d.domain)}</td><td>${d.n}</td><td>${pct(d.escalation_rate, 0)}</td><td>${pct(d.p2_given_escalated)}<span class="tiny faint"> n ${d.n_escalated_labelled}</span></td><td>${pct(d.p2_given_audit)}<span class="tiny faint"> n ${d.n_audit_labelled}</span></td><td><b>${pct(d.p2_hat)}</b></td><td>${d.p2_se_bound == null ? '–' : '±' + pct(d.p2_se_bound)}</td><td>${pct(d.p1_kept_audited)}</td><td>${pct(d.kept_vs_slow_agreement, 0)}</td><td>${d.n_audit}</td><td>${d.status === 'measured' ? '<span class="badge b-verified">measured</span>' : '<span class="badge b-unverified">unmeasured</span>'}</td></tr>`).join('')}</table><p class="hint" style="padding:0 16px">${esc(sm.note)}</p>` : '<div class="empty small">No live replies recorded yet.</div>'}</div>`;
    const al = await api.get('/v1/alerts').catch(() => null); if (!alive()) return;
    if (al) { const ST = { firing: 'b-abstained', ok: 'b-verified', 'insufficient data': 'b-unverified' };
      $('#gAlerts').innerHTML = `<div class="card-h"><h3>Alerts (OR-004)</h3><span class="hint">Last ${al.window_min} min. ${al.drill ? '<b>Drill in progress: every rule reports firing.</b>' : ''}</span><span class="sp"></span><button class="btn sm ghost" id="gDrill" type="button" title="For two minutes every rule reports firing, to test notification paths">Run alert drill</button></div>
        <div class="card-b" style="padding:0"><table class="t"><tr><th>Rule</th><th>Value</th><th>Threshold</th><th>n</th><th>State</th></tr>${al.rules.map(r => `<tr><td>${esc(r.description)}</td><td>${r.value == null ? '–' : r.rule.includes('rate') || r.rule.includes('acceptance') ? pct(r.value, 1) : r.rule === 'fast_path_latency' ? fmtMs(r.value) : num(r.value, 0)}</td><td>${esc(r.threshold)}</td><td>${r.n}</td><td><span class="badge ${ST[r.state] || ''}">${esc(r.state)}</span></td></tr>`).join('')}</table>
        <p class="hint" style="padding:0 16px">${esc(al.note)}</p></div>`;
      $('#gDrill').onclick = async () => { try { const d = await api.post('/v1/alerts/drill', { seconds: 120 }); toast(`Drill: ${d.alerts.firing} rules firing for ${d.drill_seconds} s`); load(); } catch (e) { toast(e.message); } }; }
    const P = pref?.domains || [];
    $('#gPref').innerHTML = `<div class="card-h"><h3>Value of deliberation (lab trials)</h3><span class="hint">From <b>Both</b> trials in chat: how often people prefer System 2's answer.</span></div><div class="card-b" style="padding:0">${P.length ? `<table class="t"><tr><th>Domain</th><th>Trials</th><th>Prefer S2</th><th>95% CI</th><th>Prefer S1</th><th>Tie / both poor</th><th>S2 extra time</th></tr>${P.map(d => `<tr><td>${esc(d.domain)}</td><td>${d.n}</td><td><b>${pct(d.slow_preferred_rate, 0)}</b></td><td class="small">${d.ci95 ? `${pct(d.ci95[0], 0)}–${pct(d.ci95[1], 0)}` : '–'}</td><td>${d.fast}</td><td>${d.tie} / ${d.both_bad}</td><td>${d.mean_extra_ms == null ? '–' : '+' + fmtMs(d.mean_extra_ms)}</td></tr>`).join('')}</table><p class="hint" style="padding:0 16px">${esc(pref.note)}</p>` : '<div class="empty small">No trials yet. In Chat, choose the <b>Both</b> path and pick the better answer.</div>'}</div>`;
    const DD = drift?.domains || []; const spark = (a, b) => { const v = [...(a || []), ...(b || [])]; const m = Math.max(drift?.params?.h || 10, ...v); const W = 160, H = 32;
      const line = (xs, c) => xs?.length ? `<polyline fill="none" stroke="${c}" stroke-width="1.5" points="${xs.map((y, i) => `${(i / Math.max(1, xs.length - 1)) * W},${H - (y / m) * H}`).join(' ')}"/>` : '';
      return `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" class="spark"><line x1="0" x2="${W}" y1="${H - ((drift?.params?.h || 10) / m) * H}" y2="${H - ((drift?.params?.h || 10) / m) * H}" stroke="var(--bad)" stroke-dasharray="3 2"/>${line(a, 'var(--slow)')}${line(b, 'var(--fast)')}</svg>`; };
    const dcell = (c) => c.n == null || c.alarm == null ? `<span class="faint small">${esc(c.note || 'no data')}</span>` : `${spark(c.s_hi, c.s_lo)} ${c.alarm ? `<span class="badge b-abstained">alarm ${esc(c.direction || '')}</span>` : '<span class="badge b-verified">stable</span>'}${c.negative_control_alarm ? ' <span class="badge b-consensus" title="The shuffled-order control also alarmed: treat this alarm with caution">control alarmed</span>' : ''}`;
    $('#gDrift').innerHTML = `<div class="card-h"><h3>Drift monitor (FR-042)</h3><span class="hint">CUSUM, k = ${drift?.params?.k ?? '–'}σ, h = ${drift?.params?.h ?? '–'}σ, with a shuffled negative control</span></div><div class="card-b" style="padding:0">${DD.length ? `<table class="t small"><tr><th>Domain</th><th>n</th><th>Calibration error</th><th>Fast-path accuracy</th></tr>${DD.map(d => `<tr><td>${esc(d.domain)}</td><td>${d.calibration_error.n}</td><td>${dcell(d.calibration_error)}</td><td>${dcell(d.accuracy)}</td></tr>`).join('')}</table><p class="hint" style="padding:0 16px">${esc(drift.note)}</p>` : '<div class="empty small">No labelled observations yet.</div>'}</div>`;
    const A = au?.audits || [];
    $('#gAudits').innerHTML = `<div class="card-h"><h3>Audit stream (FR-040)</h3><span class="hint">${pct(au?.worker?.audit_fraction, 0)} of kept answers are re-answered by System 2 while the machine is idle.</span><span class="sp"></span><button class="btn sm ghost" id="gAud" type="button">Run next audit now</button></div><div class="card-b" style="padding:0;max-height:420px;overflow:auto">
      ${A.length ? `<table class="t"><tr><th>When</th><th>Domain</th><th>Status</th><th>ê₁</th><th>Kept correct</th><th>S2 correct</th><th>Agreement</th><th>Time</th><th>Kept answer</th></tr>${A.map(a => `<tr><td>${ago(a.created_at)}</td><td>${esc(a.domain)}</td><td>${esc(a.status)}${a.error ? ` <span class="tiny faint">${esc(a.error)}</span>` : ''}</td><td>${num(a.e1_hat, 2)}</td><td>${a.kept_correct == null ? '–' : a.kept_correct ? '✓' : '✗'}</td><td>${a.slow_correct == null ? '–' : a.slow_correct ? '✓' : '✗'}</td><td>${pct(a.agreement, 0)}</td><td>${fmtMs(a.wall_ms)}</td><td class="small" style="max-width:300px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${esc(a.kept_preview)}</td></tr>`).join('')}</table>` : '<div class="empty small">No audits yet. About one in twenty kept answers is queued.</div>'}</div>`;
    $('#gAud').onclick = async () => { const b = $('#gAud'); b.disabled = true; b.innerHTML = '<span class="spin"></span> running'; try { const r = await api.post('/v1/audits/run-next'); toast(r.completed ? 'Audit completed' : 'No pending audit'); } catch (e) { toast(e.message); } load(); };
  };
  await load(); const t = setInterval(() => alive() && load(), 20000);
  return () => { clearInterval(t); clearInterval(ratePoll); };
} };

// ============================== tazkiya (Phase 3: distillation) ==============================
routes.tazkiya = { title: 'Tazkiya', async render(view, alive) {
  view.innerHTML = `<div class="page grid" style="gap:16px">
    <div class="card"><div class="card-h"><h2>${icon.leaf} Tazkiya: verified reflection teaches the first impulse</h2></div><div class="card-b small">
      <p style="margin-top:0">Distillation turns System 2 answers that an exact verifier accepted into training data for System 1, so the fast path improves over time. Only verified traces enter the set, plus consensus answers a person has approved (TR-002, FR-033). Personal data is redacted and opted-out conversations are excluded (SR-004). Training runs on a GPU host, never on this machine (TR-005), and a new model is promoted only through the gate below (TR-007) with one-command rollback (OR-005).</p>
      <div class="row" id="tzRules"></div></div></div>
    <div class="grid g2"><div class="card" id="tzData"><div class="card-b"><span class="spin"></span></div></div><div class="card" id="tzReg"></div></div>
    <div class="card" id="tzReview"></div>
    <div class="card"><div class="card-h"><h3>Pipeline</h3><span class="hint">Run from the repository root. Each step writes a manifest with hashes.</span></div><div class="card-b">
      <ol class="steps small">
        <li><b>Build the dataset</b> from the ledger (verified only, deduplicated, ≤ 30% per domain, 5% abstention, 20% replay, redacted):<pre><code>python3 training/tazkiya.py build --out data/tazkiya/</code></pre></li>
        <li><b>Add synthetic examples</b> (optional, TR-003): System 2 on training-split prompts, kept only when an exact verifier passes:<pre><code>python3 training/tazkiya.py synthesize --n 200</code></pre></li>
        <li><b>Train on a GPU host</b> with Unsloth LoRA (rank 16, alpha 16, lr 2e-4; TR-004) and export GGUF q4_k_m and q8_0 (TR-006):<pre><code>python3 training/train_unsloth.py --data data/tazkiya/train.jsonl --base Qwen/Qwen3-1.7B</code></pre></li>
        <li><b>Gate the candidate</b> on the 500-item regression suite with paired confidence bounds and a human sign-off (TR-007, ER-008):<pre><code>python3 training/tazkiya.py promote --candidate-run RUN --baseline-run RUN --sign-off "name"</code></pre></li>
        <li><b>Roll back</b> in one command if needed (OR-005):<pre><code>python3 training/tazkiya.py rollback</code></pre></li></ol></div></div></div>`;
  const load = async () => {
    const [t, q] = await Promise.all([api.get('/v1/tazkiya'), api.get('/v1/tazkiya/review?limit=30').catch(() => [])]); if (!alive()) return;
    $('#tzRules').innerHTML = Object.entries(t.rules).map(([k, v]) => `<span class="chip" title="${esc(v)}"><b>${esc(k)}</b>&nbsp;${esc(v)}</span>`).join('');
    const D = t.ledger.domains; const tot = (k) => D.reduce((a, d) => a + (d[k] || 0), 0);
    $('#tzData').innerHTML = `<div class="card-h"><h3>Training data in the ledger</h3><span class="hint">Live conversations only; retention ${t.retention_days} days</span></div><div class="card-b" style="padding:0">${D.length ? `<table class="t"><tr><th>Domain</th><th title="Verified System 2 traces: the core of the set">Verified S2</th><th title="Verified fast drafts">Verified S1</th><th title="Consensus answers approved by a reviewer">Approved</th><th title="Consensus answers awaiting review">Pending</th><th title="Abstentions: source of abstention examples">Abstained</th><th>Replies</th></tr>
      ${D.map(d => `<tr><td>${esc(d.domain)}</td><td><b>${d.verified_slow}</b></td><td>${d.verified_fast}</td><td>${d.consensus_approved}</td><td>${d.consensus_pending}</td><td>${d.abstained}</td><td>${d.replies}</td></tr>`).join('')}
      <tr><td><b>Total</b></td><td><b>${tot('verified_slow')}</b></td><td>${tot('verified_fast')}</td><td>${tot('consensus_approved')}</td><td>${tot('consensus_pending')}</td><td>${tot('abstained')}</td><td>${tot('replies')}</td></tr></table>` : '<div class="empty small">No live conversations yet.</div>'}
      <p class="hint" style="padding:0 16px">Opted-out conversations: ${t.ledger.opted_out_conversations} · ratings ${t.ledger.feedback.up} up / ${t.ledger.feedback.down} down · reviews ${t.ledger.reviews.approved} approved / ${t.ledger.reviews.rejected} rejected. Phase-3 entry needs about 5,000 verified traces or synthetic examples.</p></div>`;
    const reg = t.registry;
    $('#tzReg').innerHTML = `<div class="card-h"><h3>Model registry (OR-005)</h3></div><div class="card-b"><dl class="kv small"><dt>System 1</dt><dd>${esc(t.active_models.system1)}</dd><dt>System 2</dt><dd>${esc(t.active_models.system2)}</dd><dt>Adapter</dt><dd>${esc(reg?.active?.system1?.adapter_hash || t.active_models.adapter_hash)}</dd></dl>
      ${reg?.history?.length ? `<table class="t small"><tr><th>When</th><th>Role</th><th>Artefact</th><th>Decision</th></tr>${reg.history.slice(-8).reverse().map(h => `<tr><td>${esc((h.at || '').slice(0, 16))}</td><td>${esc(h.role)}</td><td class="mono tiny">${esc(h.gguf || h.adapter_hash || '')}</td><td>${esc(h.action)}${h.signed_off_by ? ' · ' + esc(h.signed_off_by) : ''}</td></tr>`).join('')}</table>` : '<p class="muted small">No promotions yet: the base models are in use. Promotions and rollbacks are recorded here.</p>'}</div>`;
    $('#tzReview').innerHTML = `<div class="card-h"><h3>Review queue (FR-033)</h3><span class="hint">Consensus answers enter the training set only after a person approves them.</span></div><div class="card-b" style="padding:0">${q.length ? q.map(x => `<div class="review" data-id="${esc(x.reply_id)}"><div class="row small"><span class="badge b-domain">${esc(x.domain || '')}</span><span class="faint">${ago(x.created_at)}</span><span class="sp"></span><button class="btn sm" data-ok="1">${icon.check}Approve</button><button class="btn sm ghost" data-ok="0">${icon.x}Reject</button></div>
        <div class="q">${esc(x.question || '(question not stored)')}</div><div class="answer small">${md(x.answer)}</div></div>`).join('') : '<div class="empty small">Nothing to review. Consensus answers from live chats appear here.</div>'}</div>`;
    $$('#tzReview [data-ok]').forEach(b => b.onclick = async () => { const id = b.closest('.review').dataset.id; try { await api.post(`/v1/tazkiya/review/${id}`, { approved: b.dataset.ok === '1', reviewer: store.get('user', 'operator') }); toast(b.dataset.ok === '1' ? 'Approved for training' : 'Rejected'); load(); } catch (e) { toast(e.message); } });
  };
  await load(); const t = setInterval(() => alive() && load(), 30000); return () => clearInterval(t);
} };

// ============================== system ==============================
routes.system = { title: 'System', async render(view, alive) {
  view.innerHTML = `<div class="page grid" style="gap:16px"><div id="sysTop" class="grid g2"><div class="card"><div class="card-b"><span class="spin"></span> Loading…</div></div></div>
    <div class="grid g2"><div class="card" id="sysGate"></div><div class="card" id="sysSelf"></div></div>
    <div class="card"><div class="card-h"><h3>Verifier playground</h3><span class="hint">Run the exact verifiers on any answer (POST /v1/verify). Code runs in the network-denied sandbox.</span></div>
      <div class="card-b grid g2"><label class="f">Answer<textarea id="vpAns" rows="8" style="font-family:var(--mono);font-size:12px">\`\`\`python\ndef add(a, b):\n    return a + b\n\`\`\`</textarea></label>
      <div><div class="seg" id="vpTabs"><button class="on" data-t="tests" type="button">Python tests</button><button data-t="ref" type="button">Reference</button><button data-t="schema" type="button">JSON Schema</button></div>
        <textarea id="vpIn" rows="6" style="width:100%;margin-top:8px;font-family:var(--mono);font-size:12px">assert add(2, 3) == 5\nassert add(-1, 1) == 0</textarea>
        <div class="row" style="margin-top:8px"><button class="btn sm" id="vpRun">${icon.shield}Verify</button><span id="vpOut"></span></div><div id="vpEv"></div></div></div></div>
    <div class="card"><div class="card-h"><h3>API</h3></div><div class="card-b small"><a href="/openapi.json" target="_blank">/openapi.json</a> · <a href="/openapi.yaml" target="_blank">/openapi.yaml</a> · <a href="/metrics" target="_blank">/metrics</a> (Prometheus) · <a href="/healthz" target="_blank">/healthz</a> · <span class="faint">JSON endpoints under /v1 need the API token when auth is on.</span></div></div></div>`;
  let vpt = 'tests'; const presets = { tests: 'assert add(2, 3) == 5\nassert add(-1, 1) == 0', ref: '42', schema: '{"type":"object","required":["name"],"properties":{"name":{"type":"string"}}}' };
  $$('#vpTabs button').forEach(b => b.onclick = () => { vpt = b.dataset.t; $$('#vpTabs button').forEach(x => x.classList.toggle('on', x === b)); $('#vpIn').value = presets[vpt]; });
  $('#vpRun').onclick = async () => { const raw = $('#vpIn').value; const v = vpt === 'tests' ? { tests: raw.split('\n').filter(Boolean) } : vpt === 'ref' ? { reference_answer: raw.trim() } : (() => { try { return { json_schema: JSON.parse(raw) }; } catch { toast('Schema is not valid JSON'); return null; } })();
    if (!v) { $('#vpOut').innerHTML = '<span class="verif fail">Schema is not valid JSON</span>'; $('#vpEv').innerHTML = ''; return; } $('#vpOut').innerHTML = '<span class="spin"></span>';
    try { const r = await api.post('/v1/verify', { answer: $('#vpAns').value, verification: v }); $s('#vpOut').innerHTML = `<span class="verif ${r.passed ? 'pass' : 'fail'}">${r.passed ? icon.check : icon.x} ${esc(r.verifier)} ${r.passed ? 'passed' : 'failed'} · ${r.duration_ms} ms</span>`; $s('#vpEv').innerHTML = r.evidence ? `<div class="codebox" style="margin-top:8px">${esc(r.evidence)}</div>` : ''; }
    catch (e) { $s('#vpOut').textContent = e.message; } };
  const load = async () => {
    const s = await api.get('/v1/system'); const sm = await api.get('/v1/self-model').catch(() => null); if (!alive()) return;
    const hb = (x) => `<div class="row small"><span class="dot ${x.ok ? 'ok' : 'bad'}"></span><b>${x.role === 'system1' ? 'System 1' : 'System 2'}</b><span class="muted">${esc(x.hf_id)}</span><span class="sp"></span><span class="mono tiny">${x.ok ? fmtMs(x.latency_ms) : esc(x.error || 'down')}</span></div>
      <div class="tiny faint" style="margin:2px 0 8px 18px">${esc(x.url)} · in-flight ${x.in_flight}/${x.max_in_flight}${x.served_models?.length ? ' · serves ' + esc(x.served_models.join(', ')) : ''}</div>`;
    $('#sysTop').innerHTML = s.backends.map(b => `<div class="card"><div class="card-h"><h3>${esc(b.label)}</h3>${b.active ? '<span class="badge b-verified">active</span>' : b.system1.ok && b.system2.ok ? `<button class="btn sm ghost" data-use="${esc(b.name)}">Use this backend</button>` : '<button class="btn sm ghost" disabled title="Its servers are not running; start them first (command below)">Not running</button>'}</div><div class="card-b">${hb(b.system1)}${hb(b.system2)}
      <div class="tiny muted">${b.kind === 'vllm' ? 'vLLM 0.31 CPU backend in the <b>nafs-vllm</b> Lima VM (no Docker); one Qwen3-1.7B server in two modes. Start: <code>make up-vllm</code>' : 'llama-server, native on macOS. Start: <code>make up</code>'}</div></div></div>`).join('');
    $$('[data-use]').forEach(b => b.onclick = async () => { try { await api.post('/v1/system/backend', { name: b.dataset.use }); toast(`Active backend: ${b.dataset.use}`); } catch (e) { toast(e.message, 6000); } refreshStatus(); load(); });
    const g = s.gate, st = g.state, cal = st.calibration;
    $('#sysGate').innerHTML = `<div class="card-h"><h3>Aql Gate</h3><span class="badge ${st.effective_rule === 'fixed' ? 'b-consensus' : 'b-verified'}">${esc(st.effective_rule)} rule</span><span class="sp"></span><a class="btn sm ghost" href="#/gate">Gate &amp; Ledger →</a></div><div class="card-b"><dl class="kv">
      <dt>Rule</dt><dd>escalate iff L·(ê₁−ê₂) &gt; λ·Δc (Theorem 2)${st.effective_rule === 'fixed' ? `; until a calibration is promoted: mean log-prob &lt; ${r4(g.fixed.mean_logprob_threshold)} or agreement &lt; ${r4(g.fixed.agreement_threshold)} (shifted by λ)` : ''}</dd>
      <dt>Calibration</dt><dd>${esc(cal.version)}${cal.n ? ` · n = ${cal.n} · held-out ECE ${num(cal.ece, 3)}` : ''} · ${st.promoted ? 'promoted' : esc(st.promotion_reason || '')}</dd><dt>Cheap samples k</dt><dd>${g.k_samples}</dd><dt>λ default</dt><dd>${r4(g.lambda_default)} per ${r4(g.cost_unit_s)} s of extra latency</dd>
      <dt>ê₂ · Δc · caps</dt><dd class="small">${Object.entries(st.params).map(([k, v]) => `${esc(k)} ${r4(v.e2_hat)} · ${r4(v.delta_c_s)} s · ${v.caps.N},${v.caps.R},${v.caps.budget}`).join('<br>')}</dd>
      <dt>Judge</dt><dd>${s.judge.enabled ? `on: floors ${r4(s.judge.floor)}, V ≥ ${r4(s.judge.pass_v)}, ≤ ${s.judge.max_retries} retry` : 'off'}</dd><dt>Audits</dt><dd>ε = ${r4(g.audit_fraction)}${Object.keys(s.audit.counts || {}).length ? ` · ${Object.entries(s.audit.counts).map(([k, v]) => `${esc(k)} ${esc(String(v))}`).join(', ')}` : ' · none yet'}</dd><dt>Draft-blind</dt><dd>${s.deliberation.draft_blind ? 'yes (FR-023)' : 'no'}</dd>
      <dt>Security</dt><dd>${s.security ? `auth ${s.security.auth_enabled ? 'on' : 'off'} · ${s.security.loopback_only ? 'loopback only' : 'network bind'} · sandbox isolation ${s.security.sandbox_isolation ? 'on' : '<b>off</b>'} · retention ${s.security.retention_days} days · same-origin only · TLS at the proxy` : '–'}</dd>
      <dt>Config hash</dt><dd class="mono">${esc(s.config_hash)}</dd><dt>Host</dt><dd>${esc(s.host.os)}/${esc(s.host.arch)} · ${s.host.cpus} CPUs · uptime ${fmtMs(s.uptime_s * 1000)} · auth ${s.auth_enabled ? 'on' : 'off (local only)'} · ${s.inflight} in flight</dd></dl></div>`;
    $('#sysSelf').innerHTML = `<div class="card-h"><h3>Muhasaba Ledger (self-model)</h3><a class="btn sm ghost" href="#/gate">details →</a></div><div class="card-b" style="padding:0">${sm && sm.domains.length ? `<table class="t"><tr><th>Domain</th><th>Replies</th><th>Escalated</th><th>Verified</th><th>Abstained</th><th>p̂₂</th><th>Audits</th><th>Status</th></tr>${sm.domains.map(d => `<tr><td>${esc(d.domain)}</td><td>${d.n}</td><td>${pct(d.escalation_rate, 0)}</td><td>${d.verified}</td><td>${d.abstained}</td><td>${pct(d.p2_hat)}</td><td>${d.n_audit}</td><td>${d.status === 'measured' ? '<span class="badge b-verified">measured</span>' : '<span class="badge b-unverified">unmeasured</span>'}</td></tr>`).join('')}</table><p class="hint" style="padding:0 16px">${esc(sm.note)}</p>` : '<div class="empty small">No replies recorded yet.</div>'}</div>`;
  };
  await load(); const t = setInterval(() => alive() && load(), 15000); return () => clearInterval(t);
} };

route();
