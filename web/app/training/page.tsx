"use client";

import { useEffect, useState } from "react";
import Card from "@/components/Card";
import LearningCurveChart from "@/components/LearningCurveChart";
import AccordionSection from "@/components/shell/AccordionSection";
import PanelLayout from "@/components/shell/PanelLayout";
import SidebarOptionList from "@/components/shell/SidebarOptionList";
import { buildCurveSeries } from "@/lib/curves";
import { fetchLearningCurves, fetchManifest } from "@/lib/data";
import { useLog } from "@/lib/logContext";
import type { LearningCurves, Manifest, Scenario } from "@/lib/types";
import { METHOD_COLORS } from "@/lib/types";
import { SCENARIO_LABELS_SHORT } from "@/lib/scenarios";

export default function TrainingPage() {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [scenario, setScenario] = useState<Scenario>("easy");
  const [curves, setCurves] = useState<LearningCurves | null>(null);
  const [highlight, setHighlight] = useState<string>("gnn_mappo");
  const [prevScenario, setPrevScenario] = useState(scenario);
  const log = useLog();

  if (scenario !== prevScenario) {
    setPrevScenario(scenario);
    setCurves(null);
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
    fetchLearningCurves(scenario).then((c) => {
      if (ignore) return;
      setCurves(c);
      log.push(`Đã tải learning curve — kịch bản "${scenario}".`);
    });
    return () => {
      ignore = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scenario]);

  const vdrSeries = curves ? buildCurveSeries(curves, "vdr") : [];
  const covSeries = curves ? buildCurveSeries(curves, "coverage") : [];

  const sidebar = (
    <>
      <AccordionSection title="Kịch bản" badge="CORE" defaultOpen>
        <SidebarOptionList
          value={scenario}
          onChange={(v) => setScenario(v as Scenario)}
          options={(manifest?.scenarios ?? []).map((s) => ({ value: s, label: SCENARIO_LABELS_SHORT[s] }))}
        />
      </AccordionSection>

      <AccordionSection title="Phương pháp nổi bật" badge="FILTER" defaultOpen>
        <SidebarOptionList
          value={highlight}
          onChange={setHighlight}
          options={vdrSeries.map((s) => ({
            value: s.method,
            label: s.label,
            dotColor: METHOD_COLORS[s.method],
          }))}
        />
      </AccordionSection>
    </>
  );

  return (
    <PanelLayout sidebar={sidebar}>
      <div>
        <h1 className="text-xl font-bold text-slate-100">Quá trình huấn luyện</h1>
        <p className="mt-1 text-sm text-slate-400">
          Learning curve theo VDR/coverage đánh giá định kỳ trong lúc huấn luyện, trung bình ± độ lệch chuẩn
          liên-seed (dải mờ ứng với phương pháp được chọn ở sidebar).
        </p>
      </div>

      <Card title="Victim Detection Rate theo thời gian huấn luyện" subtitle="Đánh giá định kỳ trên tập validation seeds">
        <LearningCurveChart series={vdrSeries} yLabel="VDR" showBandFor={highlight} />
      </Card>

      <Card title="Coverage Rate theo thời gian huấn luyện">
        <LearningCurveChart series={covSeries} yLabel="Coverage" showBandFor={highlight} />
      </Card>
    </PanelLayout>
  );
}
