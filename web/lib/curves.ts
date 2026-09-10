import type { LearningCurves, Method } from "./types";

export interface CurveSeries {
  method: Method;
  label: string;
  points: { step: number; mean: number; std: number }[];
}

export function buildCurveSeries(data: LearningCurves, metric: "vdr" | "coverage"): CurveSeries[] {
  const out: CurveSeries[] = [];
  for (const [method, entry] of Object.entries(data.methods)) {
    if (!entry || !entry.seeds.length) continue;
    const minLen = Math.min(...entry.seeds.map((s) => s.points.length));
    const points: CurveSeries["points"] = [];
    for (let i = 0; i < minLen; i++) {
      const step = entry.seeds[0].points[i].step;
      const vals = entry.seeds
        .map((s) => s.points[i][metric])
        .filter((v): v is number => v !== null && v !== undefined);
      if (!vals.length) continue;
      const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
      const variance = vals.reduce((a, b) => a + (b - mean) ** 2, 0) / vals.length;
      points.push({ step, mean, std: Math.sqrt(variance) });
    }
    if (points.length) out.push({ method: method as Method, label: entry.label, points });
  }
  return out;
}

/** Merge multiple series into one row-per-step array for Recharts (one line per method). */
export function mergeSeriesForChart(series: CurveSeries[]) {
  const stepSet = new Set<number>();
  series.forEach((s) => s.points.forEach((p) => stepSet.add(p.step)));
  const steps = Array.from(stepSet).sort((a, b) => a - b);
  return steps.map((step) => {
    const row: Record<string, number | null> = { step };
    series.forEach((s) => {
      const p = s.points.find((pt) => pt.step === step);
      row[s.method] = p ? p.mean : null;
      row[`${s.method}_upper`] = p ? p.mean + p.std : null;
      row[`${s.method}_lower`] = p ? Math.max(0, p.mean - p.std) : null;
    });
    return row;
  });
}
