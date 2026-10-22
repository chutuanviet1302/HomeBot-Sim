// Hand-written SVG charts (no chart library).
import { useEffect, useRef, useState, type ReactNode } from "react";

/** Renders children(width) at the container's real pixel width, so chart text keeps its size on every screen. */
function Sized({ children, min = 280 }: { children: (w: number) => ReactNode; min?: number }) {
  const ref = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(0);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setW(el.clientWidth));
    ro.observe(el);
    setW(el.clientWidth);
    return () => ro.disconnect();
  }, []);
  return <div ref={ref} style={{ width: "100%", overflowX: w && w < min ? "auto" : undefined }}>{w > 0 && children(Math.max(min, w))}</div>;
}

export function useData<T>(load: () => Promise<T>, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let alive = true;
    setError(null);
    load()
      .then((d) => alive && setData(d))
      .catch((e) => alive && setError(String(e.message ?? e)));
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return { data, error };
}

export interface Bar {
  label: string;
  value: number;
  text?: string;
  color?: string;
}

/** Horizontal bars, one row per item, shared scale. */
function HBarsInner({ bars, max, width = 520 }: { bars: Bar[]; max?: number; width?: number }) {
  const row = 34;
  const labelW = width < 420 ? 112 : 150;
  const valW = 84;
  const m = max ?? Math.max(1e-9, ...bars.map((b) => b.value));
  const plotW = width - labelW - valW;
  return (
    <svg className="chart" width={width} height={bars.length * row} viewBox={`0 0 ${width} ${bars.length * row}`} role="img">
      {bars.map((b, i) => {
        const w = Math.max(b.value > 0 ? 2 : 0, (b.value / m) * plotW);
        const y = i * row;
        return (
          <g key={b.label}>
            <text x={0} y={y + row / 2 + 4} className="strong">
              {b.label}
            </text>
            <rect x={labelW} y={y + 7} width={plotW} height={row - 14} rx={4} fill="var(--surface-2)" />
            <rect x={labelW} y={y + 7} width={w} height={row - 14} rx={4} fill={b.color ?? "var(--accent)"} />
            <text x={labelW + plotW + 8} y={y + row / 2 + 4} className="strong">
              {b.text ?? b.value}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

/** Distribution of one value for several groups as jittered dots on a shared axis, with the median marked. */
function DotStripInner({
  groups,
  unit = "m",
  width = 560,
  clampMin,
  clampMax,
}: {
  groups: { label: string; values: number[]; color: string }[];
  unit?: string;
  width?: number;
  clampMin?: number;
  clampMax?: number;
}) {
  const all = groups.flatMap((g) => g.values);
  if (!all.length) return null;
  let lo = clampMin ?? Math.min(...all);
  let hi = clampMax ?? Math.max(...all);
  if (hi - lo < 1e-6) hi = lo + 1;
  const pad = (hi - lo) * 0.04;
  lo -= pad;
  hi += pad;
  const labelW = 96;
  const rowH = 54;
  const axisH = 28;
  const plotW = width - labelW - 10;
  const x = (v: number) => labelW + ((Math.min(hi, Math.max(lo, v)) - lo) / (hi - lo)) * plotW;
  const ticks = niceTicks(lo, hi, 5);
  const h = groups.length * rowH + axisH;
  return (
    <svg className="chart" width={width} height={h} viewBox={`0 0 ${width} ${h}`} role="img">
      {ticks.map((t) => (
        <g key={t}>
          <line className="grid-line" x1={x(t)} x2={x(t)} y1={0} y2={groups.length * rowH} />
          <text x={x(t)} y={h - 8} textAnchor="middle">
            {fmtTick(t)}
            {unit}
          </text>
        </g>
      ))}
      {lo < 0 && hi > 0 && (
        <line x1={x(0)} x2={x(0)} y1={0} y2={groups.length * rowH} stroke="var(--bad)" strokeDasharray="3 3" />
      )}
      {groups.map((g, gi) => {
        const cy = gi * rowH + rowH / 2;
        const sorted = [...g.values].sort((a, b) => a - b);
        const med = sorted.length ? sorted[Math.floor(sorted.length / 2)] : null;
        return (
          <g key={g.label}>
            <text x={0} y={cy + 4} className="strong">
              {g.label}
            </text>
            {g.values.map((v, i) => (
              <circle
                key={i}
                cx={x(v)}
                cy={cy + (((i * 7919) % 17) / 17 - 0.5) * (rowH - 22)}
                r={3}
                fill={g.color}
                fillOpacity={0.55}
              />
            ))}
            {med !== null && (
              <line x1={x(med)} x2={x(med)} y1={cy - rowH / 2 + 6} y2={cy + rowH / 2 - 6} stroke="var(--text)" strokeWidth={2} />
            )}
          </g>
        );
      })}
    </svg>
  );
}

/** Grouped vertical bars: categories on x, one bar per series. Values in [0, 1] (rates). */
function GroupedBarsInner({
  categories,
  series,
  width = 560,
}: {
  categories: string[];
  series: { name: string; color: string; values: (number | null)[]; labels?: string[] }[];
  width?: number;
}) {
  const h = 220;
  const top = 18;
  const bottom = 36;
  const left = 34;
  const plotH = h - top - bottom;
  const plotW = width - left;
  const gw = plotW / Math.max(1, categories.length);
  const bw = Math.min(46, (gw * 0.75) / Math.max(1, series.length));
  const y = (v: number) => top + plotH * (1 - v);
  return (
    <svg className="chart" width={width} height={h} viewBox={`0 0 ${width} ${h}`} role="img">
      {[0, 0.25, 0.5, 0.75, 1].map((t) => (
        <g key={t}>
          <line className="grid-line" x1={left} x2={width} y1={y(t)} y2={y(t)} />
          <text x={left - 6} y={y(t) + 4} textAnchor="end">
            {Math.round(t * 100)}%
          </text>
        </g>
      ))}
      {categories.map((c, ci) => {
        const gx = left + ci * gw + (gw - bw * series.length) / 2;
        return (
          <g key={c}>
            {series.map((s, si) => {
              const v = s.values[ci];
              if (v === null || v === undefined) return null;
              const bx = gx + si * bw;
              return (
                <g key={s.name}>
                  <rect x={bx + 2} y={y(v)} width={bw - 4} height={Math.max(1, plotH * v)} rx={3} fill={s.color} />
                  <text x={bx + bw / 2} y={y(v) - 5} textAnchor="middle" className="strong">
                    {s.labels?.[ci] ?? `${Math.round(v * 100)}%`}
                  </text>
                </g>
              );
            })}
            <text x={left + ci * gw + gw / 2} y={h - 14} textAnchor="middle">
              {c}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function niceTicks(lo: number, hi: number, n: number) {
  const step0 = (hi - lo) / n;
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map((s) => s * mag).find((s) => s >= step0) ?? step0;
  const out: number[] = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9; t += step) out.push(Math.round(t * 1e6) / 1e6);
  return out;
}

const fmtTick = (t: number) => (Math.abs(t) >= 10 ? t.toFixed(0) : t.toFixed(Math.abs(t) < 1 && t !== 0 ? 2 : 1));

export const HBars = (p: Omit<Parameters<typeof HBarsInner>[0], "width">) => (
  <Sized>{(w) => <HBarsInner {...p} width={w} />}</Sized>
);
export const DotStrip = (p: Omit<Parameters<typeof DotStripInner>[0], "width">) => (
  <Sized>{(w) => <DotStripInner {...p} width={w} />}</Sized>
);
export const GroupedBars = (p: Omit<Parameters<typeof GroupedBarsInner>[0], "width"> & { min?: number }) => (
  <Sized min={p.min}>{(w) => <GroupedBarsInner {...p} width={w} />}</Sized>
);
