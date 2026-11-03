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
import { METHOD_GROUP, METHOD_GROUP_LABELS, METHOD_GROUP_ORDER, METHOD_ORDER, METHOD_SHORT, METRIC_LABELS_VI, METRIC_ORDER, PROPOSED_METHOD } from "@/lib/types";
import type { Method } from "@/lib/types";
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
        <h1 className="text-xl font-bold text-slate-100">So sánh phương pháp (Bảng 5)</h1>
        <p className="mt-1 max-w-3xl text-sm text-slate-400">
          Bảy phương pháp thuộc ba nhóm: hai heuristic, MADDPG (MARL không đồ thị) và <strong>họ encoder của
          framework GNN-MAPPO</strong> — MAPPO-MLP, MAPPO-GCN, MAPPO-GAT, MAPPO-DGAT — dùng chung bộ huấn luyện MAPPO,
          chỉ khác bộ mã hoá. Mean ± std liên-seed trên cùng tập bản đồ test; kiểm định Welch&apos;s t-test /
          Mann-Whitney U với hiệu chỉnh Bonferroni, lấy phiên bản đề xuất <strong>{METHOD_SHORT[PROPOSED_METHOD]}</strong>{" "}
          làm mốc (được chọn bằng ablation có kiểm soát, xem trang Ablation).
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

      <Card title={`Kiểm định thống kê: ${METHOD_SHORT[PROPOSED_METHOD]} (đề xuất) so với từng phương pháp`} subtitle={METRIC_LABELS_VI[metric]}>
        {comparison?.proposed_vs_baselines?.[metric] ? (
          <StatsTable comparisons={comparison.proposed_vs_baselines[metric]!} targetLabel={METHOD_SHORT[PROPOSED_METHOD]} />
        ) : (
          <div className="text-xs text-slate-500">
            Chưa đủ dữ liệu (cần kết quả {METHOD_SHORT[PROPOSED_METHOD]} và ít nhất một phương pháp khác cho kịch bản này).
          </div>
        )}
      </Card>

      <Card
        title="RQ2 — bộ nhớ thời gian GRU: MAPPO-DGAT (có GRU) so với MAPPO-GAT (không GRU)"
        subtitle={`${METRIC_LABELS_VI[metric]} · hai kiến trúc chỉ khác đúng một khối GRUCell(64)`}
      >
        {comparison?.gru_ablation?.[metric] ? (
          <StatsTable comparisons={{ mappo_gat: comparison.gru_ablation[metric]! }} targetLabel={METHOD_SHORT.gnn_mappo} />
        ) : (
          <div className="text-xs text-slate-500">Chưa đủ dữ liệu cho cặp MAPPO-DGAT / MAPPO-GAT.</div>
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
                {METHOD_GROUP_ORDER.map((group) => {
                  const rows = METHOD_ORDER.filter((m) => METHOD_GROUP[m] === group && comparison.methods[m]);
                  if (!rows.length) return null;
                  return [
                    <tr key={`g-${group}`} className="border-t border-(--panel-border-strong)">
                      <td colSpan={METRIC_ORDER.length + 1} className="label-caps pt-3 pb-1 text-[10px] font-bold text-cyan-300">
                        {METHOD_GROUP_LABELS[group]}
                      </td>
                    </tr>,
                    ...rows.map((method: Method) => {
                      const entry = comparison.methods[method]!;
                      const proposed = method === PROPOSED_METHOD;
                      return (
                        <tr key={method} className="border-t border-(--panel-border)">
                          <td className={proposed ? "py-1.5 pr-4 font-semibold text-cyan-200" : "py-1.5 pr-4 font-medium text-slate-200"}>
                            {entry.label}
                          </td>
                          {METRIC_ORDER.map((m) => {
                            const stat = entry.metrics[m];
                            return (
                              <td key={m} className="py-1.5 pr-4 font-mono">
                                {stat?.mean != null ? `${stat.mean.toFixed(3)} ± ${stat.std?.toFixed(3)}` : "-"}
                              </td>
                            );
                          })}
                        </tr>
                      );
                    }),
                  ];
                })}
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
