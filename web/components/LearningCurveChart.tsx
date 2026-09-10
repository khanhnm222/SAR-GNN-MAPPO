"use client";

import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { CurveSeries } from "@/lib/curves";
import { mergeSeriesForChart } from "@/lib/curves";
import { METHOD_COLORS } from "@/lib/types";

export default function LearningCurveChart({
  series,
  yLabel,
  showBandFor,
}: {
  series: CurveSeries[];
  yLabel: string;
  /** method key to render a mean±std shaded band for (usually just one, e.g. gnn_mappo) */
  showBandFor?: string;
}) {
  const data = mergeSeriesForChart(series);
  if (!data.length) {
    return <div className="flex h-72 items-center justify-center text-sm text-slate-500">Chưa có dữ liệu</div>;
  }

  return (
    <ResponsiveContainer width="100%" height={340}>
      <ComposedChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
        <XAxis
          dataKey="step"
          stroke="#64748b"
          tick={{ fontSize: 11 }}
          tickFormatter={(v) => (v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v)}
          label={{ value: "Bước môi trường", position: "insideBottom", offset: -4, fontSize: 11, fill: "#64748b" }}
        />
        <YAxis
          stroke="#64748b"
          tick={{ fontSize: 11 }}
          label={{ value: yLabel, angle: -90, position: "insideLeft", fontSize: 11, fill: "#64748b" }}
        />
        <Tooltip
          contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", fontSize: 12 }}
          labelFormatter={(v) => `Bước: ${v}`}
        />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        {showBandFor && (
          <>
            <Area
              dataKey={`${showBandFor}_lower`}
              stroke="none"
              fill="transparent"
              legendType="none"
              tooltipType="none"
              isAnimationActive={false}
            />
            <Area
              dataKey={`${showBandFor}_upper`}
              stroke="none"
              fill={METHOD_COLORS[showBandFor as keyof typeof METHOD_COLORS]}
              fillOpacity={0.15}
              legendType="none"
              tooltipType="none"
              isAnimationActive={false}
            />
          </>
        )}
        {series.map((s) => (
          <Line
            key={s.method}
            dataKey={s.method}
            name={s.label}
            stroke={METHOD_COLORS[s.method]}
            strokeWidth={s.method === showBandFor ? 2.5 : 1.5}
            dot={false}
            connectNulls
            isAnimationActive={false}
          />
        ))}
      </ComposedChart>
    </ResponsiveContainer>
  );
}
