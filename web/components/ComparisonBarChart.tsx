"use client";

import { Bar, BarChart, CartesianGrid, Cell, ErrorBar, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { Comparison, Method, MetricKey } from "@/lib/types";
import { METHOD_COLORS, METHOD_LABELS, METHOD_ORDER } from "@/lib/types";

export default function ComparisonBarChart({ comparison, metric }: { comparison: Comparison; metric: MetricKey }) {
  const data = METHOD_ORDER.filter((m) => comparison.methods[m]?.metrics[metric]?.mean != null).map((m) => {
    const stat = comparison.methods[m]!.metrics[metric]!;
    return { method: m, label: METHOD_LABELS[m], mean: stat.mean ?? 0, std: stat.std ?? 0 };
  });

  if (!data.length) {
    return <div className="flex h-56 items-center justify-center text-xs text-slate-500">Chưa có dữ liệu</div>;
  }

  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={data} margin={{ top: 8, right: 8, bottom: 32, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
        <XAxis dataKey="label" stroke="#64748b" tick={{ fontSize: 10 }} angle={-30} textAnchor="end" interval={0} />
        <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
        <Tooltip
          contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", fontSize: 12 }}
          formatter={(v) => (typeof v === "number" ? v.toFixed(3) : String(v))}
        />
        <Bar dataKey="mean" radius={[4, 4, 0, 0]}>
          <ErrorBar dataKey="std" stroke="#cbd5e1" width={4} strokeWidth={1.5} />
          {data.map((d) => (
            <Cell key={d.method} fill={METHOD_COLORS[d.method as Method]} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
