#!/usr/bin/env node
// Scores answers to chart and diagram prompts with the real renderer (rust/crates/nafs-web/assets/render.js).
// Input (stdin): JSON array of { id, kind: 'chart' | 'diagram', expect: {...}, answer }. Output: JSON array of scores.
// Scored on what the chat page would show (markdown()): a drawn chart, or a drawn diagram, not only the canonical fence.
//   chart:   a drawn Nivo figure of the expected type with every expected number and label (`canonical`: the model also
//            used a ```chart fence; the page is lenient about a missing tag, a ```json block or no fence).
//   diagram: a drawn Mermaid figure of the expected kind, enough nodes, expected words present.
import { markdown } from '../rust/crates/nafs-web/assets/render.js';

const fences = (text, lang) => [...String(text).matchAll(new RegExp('```' + lang + '[^\\n]*\\n([\\s\\S]*?)```', 'g'))].map((m) => m[1]);
const nums = (o, out = []) => { if (typeof o === 'number') out.push(o); else if (Array.isArray(o)) o.forEach((x) => nums(x, out)); else if (o && typeof o === 'object') Object.values(o).forEach((x) => nums(x, out)); return out; };
const strs = (o, out = []) => { if (typeof o === 'string') out.push(o.toLowerCase()); else if (Array.isArray(o)) o.forEach((x) => strs(x, out)); else if (o && typeof o === 'object') Object.entries(o).forEach(([k, v]) => { out.push(String(k).toLowerCase()); strs(v, out); }); return out; };

const unesc = (t) => t.replace(/&quot;/g, '"').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&#39;/g, "'").replace(/&amp;/g, '&');

function scoreChart(a, e) {
  const r = { canonical: /```chart\n/.test(a), drawn: false, type: false, numbers: 0, labels: 0, ok: false, note: '' };
  const html = markdown(a); const m = /data-chart="([^"]*)"/.exec(html);
  if (!m) { r.note = /[█▇▆▅▄▃▂▁]{3,}/.test(a) ? 'ascii art instead of a chart' : /```(json)?\n?\s*\{/.test(a) || /^\s*\{/.test(a) ? 'json that is not a chart spec' : /```mermaid/.test(a) ? 'mermaid instead of a chart' : 'no chart'; return r; }
  r.drawn = true; const spec = JSON.parse(unesc(m[1])); r.type = spec.type === e.type;
  const pool = nums({ d: spec.data, s: spec.series, l: spec.links, root: spec.root }); const words = strs({ d: spec.data, s: spec.series, n: spec.nodes, l: spec.links, root: spec.root, k: spec.keys });
  r.numbers = (e.numbers || []).filter((n) => pool.some((p) => Math.abs(p - n) < 1e-9)).length / Math.max(1, (e.numbers || []).length);
  r.labels = (e.labels || []).filter((w) => words.some((x) => x.includes(w.toLowerCase()))).length / Math.max(1, (e.labels || []).length);
  r.ok = r.type && r.numbers >= 0.99 && r.labels >= 0.75; if (!r.ok) r.note = !r.type ? `type ${spec.type}` : r.numbers < 0.99 ? 'numbers differ' : 'labels missing';
  return r;
}

function scoreDiagram(a, e) {
  const r = { canonical: /```mermaid\n/.test(a), drawn: false, kindOk: false, size: false, words: 0, ok: false, note: '' };
  const html = markdown(a); const fig = html.includes('data-kind="diagram"') && /<svg class="mm-svg"/.test(html);
  if (!fig) { r.note = /```/.test(a) || /^\s*(flowchart|graph|sequenceDiagram|stateDiagram)/m.test(a) ? 'mermaid that does not draw' : 'no diagram'; return r; }
  r.drawn = true; const src = unesc((/<pre class="fig-source" hidden><code>([\s\S]*?)<\/code>/.exec(html) || [])[1] || ''); const first = src.trim().split('\n')[0].toLowerCase();
  r.kindOk = new RegExp('^(' + e.kind + ')', 'i').test(first);
  const count = e.kind === 'sequenceDiagram' ? (html.match(/mm-part/g) || []).length : e.kind === 'pie' ? (html.match(/mm-slice/g) || []).length : (html.match(/class="mm-node/g) || []).length;
  r.size = count >= e.min; r.words = (e.words || []).filter((w) => src.toLowerCase().includes(w)).length / Math.max(1, (e.words || []).length);
  r.ok = r.kindOk && r.size && r.words >= 0.6; if (!r.ok) r.note = !r.kindOk ? `kind ${first.slice(0, 20)}` : !r.size ? `only ${count} nodes` : 'words missing';
  return r;
}

let buf = ''; process.stdin.on('data', (d) => (buf += d)); process.stdin.on('end', () => {
  const items = JSON.parse(buf);
  console.log(JSON.stringify(items.map((it) => ({ id: it.id, kind: it.kind, ...(it.kind === 'chart' ? scoreChart(it.answer, it.expect) : scoreDiagram(it.answer, it.expect)) }))));
});
