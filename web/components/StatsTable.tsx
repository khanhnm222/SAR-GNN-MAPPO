"use client";

import clsx from "clsx";
import type { StatTestResult } from "@/lib/types";
import { METHOD_LABELS, type Method } from "@/lib/types";

export default function StatsTable({ comparisons }: { comparisons: Record<string, StatTestResult> }) {
  const rows = Object.entries(comparisons);
  if (!rows.length) {
    return <div className="text-xs text-slate-500">Chưa có dữ liệu kiểm định</div>;
  }
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-xs">
        <thead className="text-slate-400">
          <tr>
            <th className="py-1.5 pr-4">So với baseline</th>
            <th className="py-1.5 pr-4">Kiểm định</th>
            <th className="py-1.5 pr-4">Mean (GNN-MAPPO)</th>
            <th className="py-1.5 pr-4">Mean (baseline)</th>
            <th className="py-1.5 pr-4">Cohen&apos;s d</th>
            <th className="py-1.5 pr-4">p-value</th>
            <th className="py-1.5">Ý nghĩa (α hiệu chỉnh)</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([name, r]) => {
            const insufficient = r.test === "insufficient_data";
            return (
              <tr key={name} className="border-t border-[var(--panel-border)]">
                <td className="py-1.5 pr-4 font-medium text-slate-200">{METHOD_LABELS[name as Method] ?? name}</td>
                <td className="py-1.5 pr-4 text-slate-400">
                  {insufficient ? "—" : r.test === "welch_t" ? "Welch's t-test" : "Mann-Whitney U"}
                </td>
                <td className="py-1.5 pr-4">{r.mean_a != null ? r.mean_a.toFixed(3) : "-"}</td>
                <td className="py-1.5 pr-4">{r.mean_b != null ? r.mean_b.toFixed(3) : "-"}</td>
                <td className="py-1.5 pr-4">{r.cohens_d != null ? r.cohens_d.toFixed(2) : "-"}</td>
                <td className="py-1.5 pr-4 font-mono">
                  {r.p_value == null ? "-" : r.p_value < 0.001 ? "<0.001" : r.p_value.toFixed(3)}
                </td>
                <td className="py-1.5">
                  {insufficient ? (
                    <span className="rounded-full bg-slate-700/40 px-2 py-0.5 text-[11px] font-medium text-slate-400">
                      Không đủ seed
                    </span>
                  ) : (
                    <span
                      className={clsx(
                        "rounded-full px-2 py-0.5 text-[11px] font-medium",
                        r.significant_bonferroni
                          ? "bg-cyan-400/10 text-cyan-300"
                          : "bg-slate-700/40 text-slate-400"
                      )}
                    >
                      {r.significant_bonferroni ? "Có ý nghĩa" : "Chưa rõ"}
                    </span>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
