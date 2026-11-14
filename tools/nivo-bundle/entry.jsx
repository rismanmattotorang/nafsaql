// Entry of nivo.js: Nivo (https://nivo.rocks) charts for the answers in the chat, bundled once with esbuild
// (see README.md here). `mount(el, spec, opts)` draws a canonical chart spec (render.js `chartSpec`) into `el`.
import React from 'react';
import { createRoot } from 'react-dom/client';
import { ResponsiveBar } from '@nivo/bar';
import { ResponsiveLine } from '@nivo/line';
import { ResponsivePie } from '@nivo/pie';
import { ResponsiveRadar } from '@nivo/radar';
import { ResponsiveHeatMap } from '@nivo/heatmap';
import { ResponsiveScatterPlot } from '@nivo/scatterplot';
import { ResponsiveTreeMap } from '@nivo/treemap';
import { ResponsiveFunnel } from '@nivo/funnel';
import { ResponsiveSankey } from '@nivo/sankey';
import { ResponsiveRadialBar } from '@nivo/radial-bar';

const h = React.createElement;
const css = (el, v, d) => getComputedStyle(el).getPropertyValue(v).trim() || d;

// The app's colours, read from its style sheet, become Nivo's theme, so a chart follows light and dark.
function themeOf(el) {
  const text = css(el, '--text', '#1b2733'), muted = css(el, '--muted', '#5b6b7b'), border = css(el, '--border', '#d9e0e7'), surface = css(el, '--surface', '#fff');
  const font = css(el, '--font', 'system-ui, sans-serif');
  return {
    background: 'transparent', text: { fontSize: 12, fill: text, fontFamily: font },
    axis: { domain: { line: { stroke: border, strokeWidth: 1 } }, ticks: { line: { stroke: border, strokeWidth: 1 }, text: { fill: muted, fontSize: 11 } }, legend: { text: { fill: muted, fontSize: 12, fontWeight: 600 } } },
    grid: { line: { stroke: border, strokeWidth: 1, strokeDasharray: '2 4' } },
    legends: { text: { fill: text, fontSize: 12 } }, labels: { text: { fill: text, fontSize: 11, fontWeight: 600 } },
    crosshair: { line: { stroke: muted, strokeWidth: 1, strokeOpacity: 0.6, strokeDasharray: '4 4' } },
    annotations: { text: { fill: text } },
    tooltip: { container: { background: surface, color: text, fontSize: 12.5, borderRadius: 8, boxShadow: '0 6px 24px rgba(0,0,0,.22)', border: `1px solid ${border}`, padding: '7px 11px' } },
  };
}

const fmt = (unit) => (v) => (typeof v === 'number' ? `${Number.isInteger(v) ? v.toLocaleString() : (Math.abs(v) >= 100 ? v.toFixed(0) : v.toPrecision(3).replace(/\.?0+$/, ''))}${unit ? ` ${unit}` : ''}` : String(v));
// Series legend: a column at the right on a wide chart, a row underneath on a narrow one.
const sideLegend = (tight, extra = {}) => (tight ? [{ anchor: 'bottom', direction: 'row', translateY: 88, itemWidth: 86, itemHeight: 18, symbolSize: 10, symbolShape: 'circle', itemsSpacing: 2, ...extra }] : [{ anchor: 'top-right', direction: 'column', translateX: 122, itemWidth: 108, itemHeight: 20, symbolSize: 11, symbolShape: 'circle', itemsSpacing: 3, ...extra }]);
// Black or white text, whichever reads better on a cell of this colour (a heat map runs from pale to dark).
const readable = (c) => { const m = /(\d+)[,\s]+(\d+)[,\s]+(\d+)/.exec(String(c)); if (!m) return '#111'; const [r, g, b] = [m[1], m[2], m[3]].map(Number); return (0.299 * r + 0.587 * g + 0.114 * b) / 255 > 0.55 ? '#111' : '#fff'; };
const longest = (xs) => Math.max(0, ...xs.map((x) => String(x).length));

function chartElement(spec, theme, animate, tight) {
  const colors = { scheme: spec.scheme || 'nivo' };
  const common = { theme, colors, animate, motionConfig: 'gentle' };
  const valueFormat = fmt(spec.unit);
  const axisB = (rot) => ({ tickSize: 0, tickPadding: 8, tickRotation: rot, legend: spec.xLabel, legendOffset: rot ? 68 : 36, legendPosition: 'middle' });
  const axisL = { tickSize: 0, tickPadding: 8, legend: spec.yLabel, legendOffset: -48, legendPosition: 'middle', format: (v) => (typeof v === 'number' ? fmt('')(v) : v) };
  switch (spec.type) {
    case 'bar': {
      const horiz = spec.horizontal; const labels = spec.data.map((r) => r[spec.indexBy]); const rot = !horiz && (labels.length > 6 || longest(labels) > 16 || (tight && labels.length > 2)) ? -35 : 0; const multi = spec.keys.length > 1;
      return h(ResponsiveBar, { ...common, data: spec.data, keys: spec.keys, indexBy: spec.indexBy, layout: horiz ? 'horizontal' : 'vertical', groupMode: spec.stacked ? 'stacked' : 'grouped', valueFormat,
        margin: { top: 12, right: multi && !tight ? 130 : 16, bottom: (horiz ? 40 : rot ? 92 : 56) + (spec.xLabel ? 8 : 0) + (multi && tight ? 52 : 0), left: horiz ? Math.min(tight ? 100 : 160, 24 + longest(labels) * 6.5) : tight ? 50 : 62 }, padding: 0.28, innerPadding: spec.stacked ? 0 : 2, borderRadius: 4,
        enableLabel: !tight && spec.data.length * (spec.stacked ? 1 : spec.keys.length) <= 14, labelSkipWidth: 22, labelSkipHeight: 16, labelTextColor: { from: 'color', modifiers: [['darker', 2.4]] },
        axisBottom: horiz ? { ...axisL, legend: spec.yLabel, legendOffset: 34 } : axisB(rot), axisLeft: horiz ? { tickSize: 0, tickPadding: 8, legend: spec.xLabel, legendOffset: -Math.min(150, 20 + longest(labels) * 6.5), legendPosition: 'middle' } : axisL,
        enableGridX: horiz, enableGridY: !horiz, legends: multi ? sideLegend(tight, { dataFrom: 'keys' }) : [], role: 'application' });
    }
    case 'line': case 'area': {
      const pts = spec.series.reduce((n, s) => n + s.data.length, 0); const numeric = spec.series.every((s) => s.data.every((p) => typeof p.x === 'number'));
      const xs = [...new Set(spec.series.flatMap((s) => s.data.map((p) => p.x)))]; const rot = !numeric && (xs.length > 8 || longest(xs) > 16 || (tight && xs.length > 3)) ? -35 : 0; const multi = spec.series.length > 1;
      return h(ResponsiveLine, { ...common, data: spec.series, xScale: numeric ? { type: 'linear', min: 'auto', max: 'auto' } : { type: 'point' }, yScale: { type: 'linear', min: 'auto', max: 'auto', stacked: spec.type === 'area' && spec.stacked }, curve: 'monotoneX', yFormat: (v) => valueFormat(v),
        margin: { top: 14, right: multi && !tight ? 130 : 22, bottom: (rot ? 92 : 56) + (spec.xLabel ? 8 : 0) + (multi && tight ? 52 : 0), left: tight ? 50 : 62 }, axisBottom: { ...axisB(rot), ...(numeric && xs.length <= 14 ? { tickValues: [...xs].sort((a, b) => a - b) } : {}) }, axisLeft: axisL, lineWidth: 3, pointSize: pts > 60 ? 0 : 8, pointColor: { theme: 'background' }, pointBorderWidth: 2, pointBorderColor: { from: 'serieColor' },
        enableArea: spec.type === 'area', areaOpacity: 0.16, useMesh: true, enableSlices: spec.series.length > 1 ? 'x' : false, enableGridX: false,
        legends: multi ? sideLegend(tight) : [] });
    }
    case 'pie': return h(ResponsivePie, { ...common, data: spec.data, valueFormat, margin: { top: 28, right: tight ? 28 : 120, bottom: 28, left: tight ? 28 : 28 }, innerRadius: spec.donut ? 0.58 : 0, padAngle: 1.2, cornerRadius: 5, activeOuterRadiusOffset: 8, sortByValue: false,
      arcLinkLabelsSkipAngle: 14, arcLinkLabelsTextColor: theme.text.fill, arcLinkLabelsThickness: 1.5, arcLinkLabelsColor: { from: 'color' }, arcLabelsSkipAngle: 18, arcLabelsTextColor: { from: 'color', modifiers: [['darker', 2.6]] }, enableArcLinkLabels: spec.data.length <= 8,
      legends: spec.data.length > 8 || !tight ? [{ anchor: 'right', direction: 'column', translateX: 112, itemWidth: 100, itemHeight: 20, symbolSize: 11, symbolShape: 'circle', itemsSpacing: 3 }] : [] });
    case 'radar': return h(ResponsiveRadar, { ...common, data: spec.data, keys: spec.keys, indexBy: spec.indexBy, valueFormat, margin: { top: 36, right: 90, bottom: 36, left: 90 }, gridShape: 'linear', gridLevels: 4, dotSize: 8, dotBorderWidth: 2, dotColor: { theme: 'background' }, dotBorderColor: { from: 'color' }, fillOpacity: 0.2, borderWidth: 2.5, gridLabelOffset: 16,
      legends: spec.keys.length > 1 ? [{ anchor: 'top-left', direction: 'column', translateX: -50, translateY: -20, itemWidth: 90, itemHeight: 20, symbolSize: 11, symbolShape: 'circle' }] : [] });
    case 'heatmap': return h(ResponsiveHeatMap, { ...common, colors: { type: 'sequential', scheme: ['blues', 'greens', 'oranges', 'purples', 'reds', 'viridis', 'cool', 'warm'].includes(spec.scheme) ? spec.scheme : 'blues' }, data: spec.series, valueFormat, margin: { top: 36, right: 70, bottom: 24, left: 20 + Math.min(120, longest(spec.series.map((s) => s.id)) * 6.5) }, axisTop: { tickSize: 0, tickPadding: 8, legend: spec.xLabel, legendOffset: -32, legendPosition: 'middle' }, axisLeft: { tickSize: 0, tickPadding: 8, legend: spec.yLabel, legendOffset: -60, legendPosition: 'middle' }, axisBottom: null,
      borderRadius: 4, borderWidth: 2, borderColor: { from: 'color', modifiers: [['darker', 0.1]] }, labelTextColor: (cell) => readable(cell.color), enableLabels: spec.series.length * (spec.series[0]?.data.length || 1) <= 80, hoverTarget: 'cell',
      legends: [{ anchor: 'right', translateX: 50, length: 150, thickness: 10, direction: 'column', tickPosition: 'after', tickSize: 3, tickSpacing: 3, titleAlign: 'start', titleOffset: 4 }] });
    case 'scatter': return h(ResponsiveScatterPlot, { ...common, data: spec.series, xScale: { type: 'linear', min: 'auto', max: 'auto' }, yScale: { type: 'linear', min: 'auto', max: 'auto' }, margin: { top: 14, right: spec.series.length > 1 && !tight ? 130 : 22, bottom: 58 + (spec.series.length > 1 && tight ? 44 : 0), left: tight ? 50 : 62 }, nodeSize: 11, blendMode: 'normal', useMesh: true, axisBottom: axisB(0), axisLeft: axisL, enableGridX: true, xFormat: (v) => valueFormat(v), yFormat: (v) => valueFormat(v),
      legends: spec.series.length > 1 ? sideLegend(tight) : [] });
    case 'treemap': return h(ResponsiveTreeMap, { ...common, data: spec.root, identity: 'name', value: 'value', valueFormat, margin: { top: 6, right: 6, bottom: 6, left: 6 }, label: (n) => `${n.id}`, labelSkipSize: 28, innerPadding: 3, outerPadding: 3, borderRadius: 4, borderWidth: 0, enableParentLabel: spec.root.children.some((c) => c.children), parentLabelPosition: 'top', parentLabelTextColor: theme.text.fill, labelTextColor: { from: 'color', modifiers: [['darker', 2.6]] }, nodeOpacity: 0.95 });
    case 'funnel': return h(ResponsiveFunnel, { ...common, data: spec.data, valueFormat, margin: { top: 14, right: 20, bottom: 14, left: 20 }, shapeBlending: 0.6, spacing: 4, borderWidth: 14, labelColor: { from: 'color', modifiers: [['darker', 3]] }, beforeSeparatorLength: 40, afterSeparatorLength: 40, currentPartSizeExtension: 8, enableBeforeSeparators: false, enableAfterSeparators: false });
    case 'sankey': return h(ResponsiveSankey, { ...common, data: { nodes: spec.nodes, links: spec.links }, valueFormat, margin: { top: 12, right: 120, bottom: 12, left: 20 }, nodeThickness: 14, nodeSpacing: 18, nodeBorderRadius: 3, linkOpacity: 0.35, linkHoverOthersOpacity: 0.08, enableLinkGradient: true, labelPadding: 10, labelTextColor: theme.text.fill, nodeOpacity: 1 });
    case 'radialbar': return h(ResponsiveRadialBar, { ...common, data: spec.series, valueFormat, margin: { top: 20, right: 110, bottom: 20, left: 20 }, padding: 0.35, cornerRadius: 6, radialAxisStart: { tickSize: 0, tickPadding: 6 }, circularAxisOuter: { tickSize: 0, tickPadding: 8 }, enableLabels: false, legends: [{ anchor: 'right', direction: 'column', translateX: 100, itemWidth: 90, itemHeight: 20, symbolSize: 11, symbolShape: 'circle' }] });
    default: return null;
  }
}

const roots = new WeakMap();
export function mount(el, spec, opts = {}) {
  let r = roots.get(el); if (!r) { r = createRoot(el); roots.set(el, r); }
  const theme = themeOf(el);
  r.render(h('div', { style: { width: '100%', height: '100%' } }, chartElement(spec, theme, opts.animate !== false, !!opts.tight)));
}
export function unmount(el) { const r = roots.get(el); if (r) { r.unmount(); roots.delete(el); } }
export const version = '0.99.0';
