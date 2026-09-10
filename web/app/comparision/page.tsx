"use client";

import { useEffect, useState } from "react";
import Card from "@/components/Card";
import ComparisonBarChart from "@/components/ComparisonBarChart";
import AccordionSection from "@/components/shell/AccordionSection";
import PanelLayout from "@/components/shell/PanelLayout";
import SidebarOptionList from "@/components/shell/SidebarOptionList";
import StatsTable from "@/components/StatsTable";
import { fetchComparison, fetchManifest } from "@/lib/data";
import { useLog } from "@/lib/logContext";
import type { Comparison, Manifest, MetricKey, Scenario } from "@/lib/types";
import { METRIC_LABELS_VI, METRIC_ORDER } from "@/lib/types";
import { SCENARIO_LABELS_SHORT } from "@/lib/scenarios";

export default function ComparisonPage() {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [scenario, setScenario] = useState<Scenario>("easy");
  const [comparison, setComparison] = useState<Comparison | null>(null);
  const [loading, setLoading] = useState(true);
  const [metric, setMetric] = useState<MetricKey>("victim_detection_rate");
  const [prevScenario, setPrevScenario] = useState(scenario);
  const log = useLog();

  if (scenario !== prevScenario) {
    setPrevScenario(scenario);
    setLoading(true);
  }

  useEffect(() => {
    fetchManifest().then((m) => {
      setManifest(m);
      if (m && m.scenarios.length) setScenario(m.scenarios[0]);
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    let ignore = false;
    fetchComparison(scenario).then((c) => {
      if (ignore) return;
      setComparison(c);
      setLoading(false);
      log.push(`Đã tải so sánh baseline — kịch bản "${scenario}".`);
    });
    return () => {
      ignore = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scenario]);

  const sidebar = (
    <>
      <AccordionSection title="Kịch bản" badge="CORE" defaultOpen>
        <SidebarOptionList
          value={scenario}
          onChange={(v) => setScenario(v as Scenario)}
          options={(manifest?.scenarios ?? []).map((s) => ({ value: s, label: SCENARIO_LABELS_SHORT[s] }))}
        />
      </AccordionSection>

      <AccordionSection title="Chỉ số" badge="METRIC" defaultOpen>
        <SidebarOptionList
          value={metric}
          onChange={(v) => setMetric(v as MetricKey)}
          options={METRIC_ORDER.map((m) => ({ value: m, label: METRIC_LABELS_VI[m] }))}
        />
      </AccordionSection>
    </>
  );

  return (
    <PanelLayout sidebar={sidebar}>
      <div>
        <h1 className="text-xl font-bold text-slate-100">So sánh Baseline (Bảng 5)</h1>
        <p className="mt-1 text-sm text-slate-400">
          So sánh 7 phương pháp trên 5 chỉ số đánh giá (mục 1.5.5), mean ± std liên-seed, kèm kiểm định thống
          kê Welch&apos;s t-test / Mann-Whitney U (hiệu chỉnh Bonferroni) giữa GNN-MAPPO và các baseline.
        </p>
      </div>

      <Card title="Biểu đồ so sánh theo chỉ số" subtitle={METRIC_LABELS_VI[metric]}>
        {comparison ? (
          <ComparisonBarChart comparison={comparison} metric={metric} />
        ) : (
          <div className="flex h-56 items-center justify-center text-xs text-slate-500">
            {loading ? "Đang tải..." : "Chưa có dữ liệu"}
          </div>
        )}
      </Card>

      <Card title="Kiểm định thống kê: GNN-MAPPO so với từng baseline" subtitle={METRIC_LABELS_VI[metric]}>
        {comparison?.gnn_mappo_vs_baselines?.[metric] ? (
          <StatsTable comparisons={comparison.gnn_mappo_vs_baselines[metric]!} />
        ) : (
          <div className="text-xs text-slate-500">
            Chưa đủ dữ liệu (cần kết quả GNN-MAPPO và ít nhất một baseline cho kịch bản này).
          </div>
        )}
      </Card>

      <Card title="Bảng toàn bộ chỉ số">
        {comparison && Object.keys(comparison.methods).length ? (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="text-slate-400">
                <tr>
                  <th className="py-1.5 pr-4">Phương pháp</th>
                  {METRIC_ORDER.map((m) => (
                    <th key={m} className="py-1.5 pr-4">
                      {METRIC_LABELS_VI[m]}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.entries(comparison.methods).map(([method, entry]) => (
                  <tr key={method} className="border-t border-[var(--panel-border)]">
                    <td className="py-1.5 pr-4 font-medium text-slate-200">{entry!.label}</td>
                    {METRIC_ORDER.map((m) => {
                      const stat = entry!.metrics[m];
                      return (
                        <td key={m} className="py-1.5 pr-4 font-mono">
                          {stat?.mean != null ? `${stat.mean.toFixed(3)} ± ${stat.std?.toFixed(3)}` : "-"}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-xs text-slate-500">Chưa có dữ liệu</div>
        )}
      </Card>
    </PanelLayout>
  );
}
