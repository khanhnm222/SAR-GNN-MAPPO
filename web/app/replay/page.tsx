"use client";

import { useEffect, useState } from "react";
import Card from "@/components/Card";
import ReplayCanvas from "@/components/ReplayCanvas";
import AccordionSection from "@/components/shell/AccordionSection";
import PanelLayout from "@/components/shell/PanelLayout";
import SidebarOptionList from "@/components/shell/SidebarOptionList";
import { fetchManifest, fetchTrajectory, fetchTrajectoryManifest } from "@/lib/data";
import { useLog } from "@/lib/logContext";
import type { Manifest, Method, Scenario, TrajectoryData } from "@/lib/types";
import { METHOD_COLORS, METHOD_LABELS, METHOD_ORDER } from "@/lib/types";
import { SCENARIO_DIMS, SCENARIO_LABELS_SHORT } from "@/lib/scenarios";

export default function ReplayPage() {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [scenario, setScenario] = useState<Scenario>("easy");
  const [available, setAvailable] = useState<Record<string, boolean>>({});
  const [method, setMethod] = useState<Method>("gnn_mappo");
  const [trajectory, setTrajectory] = useState<TrajectoryData | null>(null);
  const [prevKey, setPrevKey] = useState(`${scenario}|${method}`);
  const log = useLog();

  const key = `${scenario}|${method}`;
  if (key !== prevKey) {
    setPrevKey(key);
    setTrajectory(null);
  }

  useEffect(() => {
    fetchManifest().then((m) => {
      setManifest(m);
      if (m && m.scenarios.length) setScenario(m.scenarios[0]);
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    fetchTrajectoryManifest(scenario).then((m) => setAvailable(m ?? {}));
  }, [scenario]);

  useEffect(() => {
    if (!available[method]) return;
    let ignore = false;
    fetchTrajectory(scenario, method).then((t) => {
      if (ignore) return;
      setTrajectory(t);
      log.push(`Nạp quỹ đạo "${method}" — kịch bản "${scenario}" (${t?.frames.length ?? 0} khung hình).`);
    });
    return () => {
      ignore = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scenario, method, available]);

  const dims = SCENARIO_DIMS[scenario];

  const sidebar = (
    <>
      <AccordionSection title="Kịch bản" badge="CORE" defaultOpen>
        <SidebarOptionList
          value={scenario}
          onChange={(v) => setScenario(v as Scenario)}
          options={(manifest?.scenarios ?? []).map((s) => ({ value: s, label: SCENARIO_LABELS_SHORT[s] }))}
        />
      </AccordionSection>

      <AccordionSection title="Phương pháp" badge="REPLAY" defaultOpen>
        <SidebarOptionList
          value={method}
          onChange={(v) => setMethod(v as Method)}
          options={METHOD_ORDER.map((m) => ({
            value: m,
            label: METHOD_LABELS[m],
            dotColor: METHOD_COLORS[m],
            disabled: !available[m],
          }))}
        />
      </AccordionSection>
    </>
  );

  return (
    <PanelLayout sidebar={sidebar}>
      <div>
        <h1 className="text-xl font-bold text-slate-100">Phát lại mô phỏng tìm kiếm</h1>
        <p className="mt-1 max-w-3xl text-sm text-slate-400">
          Phát lại một episode thật của chính sách đã huấn luyện, đúng ký hiệu Hình 2 của đề cương: UAV hình
          thoi, nạn nhân hình tròn (màu theo mức ưu tiên, viền xanh khi đã tìm thấy), vật cản tĩnh ô vuông xám,
          cùng bán kính cảm biến và vùng đã quét tích lũy (xanh lá nhạt).
        </p>
      </div>

      <Card title={`${METHOD_LABELS[method]} — ${scenario}`} subtitle={`Bản đồ ${dims.width}×${dims.height}, ${dims.nUavs} UAV`}>
        {trajectory ? (
          <ReplayCanvas data={trajectory} width={dims.width} height={dims.height} />
        ) : (
          <div className="flex h-64 items-center justify-center text-xs text-slate-500">
            {available[method] ? "Đang tải quỹ đạo..." : "Chưa có dữ liệu quỹ đạo cho phương pháp này ở kịch bản này."}
          </div>
        )}
      </Card>
    </PanelLayout>
  );
}
