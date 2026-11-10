// Build-time SVG chart helpers. Every chart on the site is assembled from these
// so the style stays identical: 1px gridlines, mono ticks, direct labels.
import { scaleLinear, scaleLog, scaleBand, scalePoint } from 'd3-scale';
import { line as d3line, area as d3area, curveMonotoneX, curveLinear } from 'd3-shape';

export { scaleLinear, scaleLog, scaleBand, scalePoint };

export type Margin = { top: number; right: number; bottom: number; left: number };

export interface Frame {
  width: number;
  height: number;
  margin: Margin;
  innerW: number;
  innerH: number;
}

export function frame(width: number, height: number, margin: Partial<Margin> = {}): Frame {
  const m = { top: 16, right: 16, bottom: 36, left: 44, ...margin };
  return { width, height, margin: m, innerW: width - m.left - m.right, innerH: height - m.top - m.bottom };
}

export function esc(s: string | number): string {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

export const r2 = (n: number) => Math.round(n * 100) / 100;

/** Horizontal gridlines with tick labels on the left. */
export function yGrid(
  f: Frame,
  scale: { (v: number): number },
  ticks: number[],
  fmt: (v: number) => string = String,
  opts: { labelDx?: number } = {},
): string {
  const dx = opts.labelDx ?? -8;
  return ticks
    .map((t) => {
      const y = r2(scale(t));
      return `<line class="grid-line" x1="0" x2="${f.innerW}" y1="${y}" y2="${y}"/>` +
        `<text class="tick" x="${dx}" y="${y}" dy="0.32em" text-anchor="end">${esc(fmt(t))}</text>`;
    })
    .join('');
}

/** Vertical gridlines with tick labels below. */
export function xGrid(
  f: Frame,
  scale: { (v: number): number },
  ticks: number[],
  fmt: (v: number) => string = String,
  opts: { grid?: boolean } = {},
): string {
  const grid = opts.grid ?? false;
  return ticks
    .map((t) => {
      const x = r2(scale(t));
      return (grid ? `<line class="grid-line" y1="0" y2="${f.innerH}" x1="${x}" x2="${x}"/>` : '') +
        `<text class="tick" x="${x}" y="${f.innerH + 20}" text-anchor="middle">${esc(fmt(t))}</text>`;
    })
    .join('');
}

export function xAxisLine(f: Frame): string {
  return `<line class="axis-line" x1="0" x2="${f.innerW}" y1="${f.innerH}" y2="${f.innerH}"/>`;
}

export function yAxisLine(f: Frame): string {
  return `<line class="axis-line" x1="0" x2="0" y1="0" y2="${f.innerH}"/>`;
}

export function axisTitleX(f: Frame, text: string, dy = 34): string {
  return `<text class="axis-title" x="${f.innerW / 2}" y="${f.innerH + dy}" text-anchor="middle">${esc(text)}</text>`;
}

export function axisTitleY(f: Frame, text: string, dx = -36): string {
  return `<text class="axis-title" transform="translate(${dx},${f.innerH / 2}) rotate(-90)" text-anchor="middle">${esc(text)}</text>`;
}

export function linePath(points: [number, number][], smooth = false): string {
  const gen = d3line<[number, number]>()
    .x((d) => r2(d[0]))
    .y((d) => r2(d[1]))
    .curve(smooth ? curveMonotoneX : curveLinear);
  return gen(points) ?? '';
}

export function bandPath(points: { x: number; y0: number; y1: number }[], smooth = false): string {
  const gen = d3area<{ x: number; y0: number; y1: number }>()
    .x((d) => r2(d.x))
    .y0((d) => r2(d.y0))
    .y1((d) => r2(d.y1))
    .curve(smooth ? curveMonotoneX : curveLinear);
  return gen(points) ?? '';
}

/** Nice log ticks (1, 2, 5 pattern) within [lo, hi]. */
export function logTicks(lo: number, hi: number, pattern: number[] = [1, 2, 5]): number[] {
  const out: number[] = [];
  const start = Math.floor(Math.log10(lo));
  const end = Math.ceil(Math.log10(hi));
  for (let e = start; e <= end; e++) {
    for (const p of pattern) {
      const v = p * 10 ** e;
      if (v >= lo * 0.999 && v <= hi * 1.001) out.push(v);
    }
  }
  return out;
}

export function fmtLog(v: number): string {
  if (v >= 1e6) return `${v / 1e6}M`;
  if (v >= 1e3) return `${v / 1e3}K`;
  if (v >= 1) return `${v}`;
  return `${v}`;
}

/** Data attributes used by the shared tooltip script. */
export function tip(label: string): string {
  return `data-tip="${esc(label)}"`;
}

export interface TableSpec {
  caption: string;
  columns: string[];
  rows: (string | number)[][];
  numeric?: boolean[];
}

/** HTML table that mirrors a chart's data (the "View data" toggle). */
export function dataTable(spec: TableSpec): string {
  const numeric = spec.numeric ?? spec.columns.map((_, i) => i > 0);
  const head = spec.columns
    .map((c, i) => `<th scope="col"${numeric[i] ? ' class="num"' : ''}>${esc(c)}</th>`)
    .join('');
  const body = spec.rows
    .map(
      (r) =>
        `<tr>${r
          .map((c, i) => (i === 0 ? `<th scope="row">${esc(c)}</th>` : `<td${numeric[i] ? ' class="num"' : ''}>${esc(c)}</td>`))
          .join('')}</tr>`,
    )
    .join('');
  return `<table><caption class="visually-hidden">${esc(spec.caption)}</caption><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}
