"use client";

import { useEffect, useState } from "react";
import Card from "@/components/Card";
import AccordionSection from "@/components/shell/AccordionSection";
import PanelLayout from "@/components/shell/PanelLayout";
import SidebarOptionList from "@/components/shell/SidebarOptionList";
import StatsTable from "@/components/StatsTable";
import { fetchAblation, fetchManifest } from "@/lib/data";
import { useLog } from "@/lib/logContext";
import type { AblationReport, Manifest, MetricKey, Scenario } from "@/lib/types";
import { METRIC_LABELS_VI, METRIC_ORDER } from "@/lib/types";
import { SCENARIO_LABELS_SHORT } from "@/lib/scenarios";

export default function AblationPage() {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [scenario, setScenario] = useState<Scenario>("easy");
  const [report, setReport] = useState<AblationReport | null>(null);
  const [prevScenario, setPrevScenario] = useState(scenario);
  const log = useLog();

  if (scenario !== prevScenario) {
    setPrevScenario(scenario);
    setReport(null);
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
    fetchAblation(scenario).then((r) => {
      if (ignore) return;
      setReport(r);
      log.push(`Đã tải báo cáo ablation — kịch bản "${scenario}".`);
    });
    return () => {
      ignore = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scenario]);

  const sidebar = (
    <AccordionSection title="Kịch bản" badge="CORE" defaultOpen>
      <SidebarOptionList
        value={scenario}
        onChange={(v) => setScenario(v as Scenario)}
        options={(manifest?.scenarios ?? []).map((s) => ({ value: s, label: SCENARIO_LABELS_SHORT[s] }))}
      />
    </AccordionSection>
  );

  return (
    <PanelLayout sidebar={sidebar}>
      <div>
        <h1 className="text-xl font-bold text-slate-100">Ablation kiến trúc</h1>
        <p className="mt-1 max-w-3xl text-sm text-slate-400">
          Cô lập đóng góp của từng thành phần kiến trúc: MAPPO-MLP (không đồ thị) → MAPPO-GCN (đồ thị, trọng
          số cố định) → MAPPO-GAT (attention, không có GRU) → GNN-MAPPO (DGAT đầy đủ). Nếu GNN-MAPPO chỉ vượt
          trội MAPPO-MLP mà không vượt MAPPO-GCN/GAT, cải thiện nên được quy cho GNN nói chung thay vì riêng
          DGAT (mục 1.5.5).
        </p>
      </div>

      {METRIC_ORDER.map((metric: MetricKey) => {
        const entry = report?.metrics[metric];
        return (
          <Card key={metric} title={METRIC_LABELS_VI[metric]}>
            {entry?.comparisons ? (
              <StatsTable comparisons={entry.comparisons} />
            ) : (
              <div className="text-xs text-slate-500">Chưa có dữ liệu ablation cho chỉ số này.</div>
            )}
          </Card>
        );
      })}
    </PanelLayout>
  );
}
