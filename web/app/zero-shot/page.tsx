"use client";

import { useEffect, useMemo, useState } from "react";
import {
  CartesianGrid, Legend, Line, LineChart, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import Card from "@/components/Card";
import AccordionSection from "@/components/shell/AccordionSection";
import PanelLayout from "@/components/shell/PanelLayout";
import SidebarOptionList from "@/components/shell/SidebarOptionList";
import { fetchManifest, fetchZeroShot } from "@/lib/data";
import { useLog } from "@/lib/logContext";
import type { Manifest, Method, Scenario, ZeroShotMode, ZeroShotReport } from "@/lib/types";
import { METHOD_COLORS, METHOD_LABELS, METHOD_ORDER, PROPOSED_METHOD, ZERO_SHOT_MODE_LABELS } from "@/lib/types";
import { SCENARIO_LABELS_SHORT } from "@/lib/scenarios";

type MetricKey = "victim_detection_rate" | "coverage_rate" | "collision_rate";

const METRICS: { key: MetricKey; label: string }[] = [
  { key: "victim_detection_rate", label: "Tỷ lệ phát hiện nạn nhân (VDR)" },
  { key: "coverage_rate", label: "Tỷ lệ bao phủ" },
  { key: "collision_rate", label: "Tỷ lệ va chạm" },
];

export default function ZeroShotPage() {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [scenario, setScenario] = useState<Scenario>("easy");
  const [metric, setMetric] = useState<MetricKey>("victim_detection_rate");
  const [mode, setMode] = useState<ZeroShotMode>("fixed_map");
  const [report, setReport] = useState<ZeroShotReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [prevKey, setPrevKey] = useState(`${scenario}/${mode}`);
  const log = useLog();

  // đặt lại trạng thái tải ngay trong lúc render khi đổi kịch bản/chế độ (không setState trong effect)
  if (`${scenario}/${mode}` !== prevKey) {
    setPrevKey(`${scenario}/${mode}`);
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
    fetchZeroShot(scenario, mode).then((r) => {
      if (ignore) return;
      setReport(r);
      setLoading(false);
      if (r) log.push(`Đã tải zero-shot ${mode === "density" ? "density-controlled" : "fixed-map"} "${scenario}" — ${Object.keys(r.methods).length} kiến trúc.`);
    });
    return () => {
      ignore = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scenario, mode]);

  const methods = useMemo(
    () => METHOD_ORDER.filter((m) => report?.methods[m]) as Method[],
    [report],
  );

  const chartData = useMemo(() => {
    if (!report) return [];
    return report.ns.map((n, i) => {
      const row: Record<string, number | null> = { n };
      for (const m of methods) row[m] = report.methods[m]![metric].mean[i];
      return row;
    });
  }, [report, methods, metric]);

  // hiệu năng giữ được so với N lúc huấn luyện — chỉ số thực sự trả lời RQ3
  const retention = useMemo(() => {
    if (!report) return [];
    const base = report.ns.indexOf(report.train_n);
    if (base < 0) return [];
    return methods.map((m) => {
      const s = report.methods[m]![metric].mean;
      const ref = s[base];
      return {
        method: m,
        values: report.ns.map((_, i) =>
          ref && s[i] != null ? (100 * (s[i] as number)) / ref : null,
        ),
      };
    });
  }, [report, methods, metric]);

  const sidebar = (
    <>
      <AccordionSection title="Kịch bản" badge="CORE" defaultOpen>
        <SidebarOptionList
          value={scenario}
          onChange={(v) => setScenario(v as Scenario)}
          options={(manifest?.scenarios ?? []).map((s) => ({ value: s, label: SCENARIO_LABELS_SHORT[s] }))}
        />
      </AccordionSection>
      <AccordionSection title="Chế độ" badge="RQ3" defaultOpen>
        <SidebarOptionList
          value={mode}
          onChange={(v) => setMode(v as ZeroShotMode)}
          options={(Object.keys(ZERO_SHOT_MODE_LABELS) as ZeroShotMode[]).map((k) => ({ value: k, label: ZERO_SHOT_MODE_LABELS[k] }))}
        />
      </AccordionSection>
      <AccordionSection title="Chỉ số" badge="METRIC" defaultOpen>
        <SidebarOptionList
          value={metric}
          onChange={(v) => setMetric(v as MetricKey)}
          options={METRICS.map((m) => ({ value: m.key, label: m.label }))}
        />
      </AccordionSection>
    </>
  );

  return (
    <PanelLayout sidebar={sidebar}>
      <div>
        <h1 className="text-xl font-bold text-slate-100">Zero-shot Scalability (RQ3)</h1>
        <p className="mt-1 max-w-3xl text-sm text-slate-400">
          Đánh giá chính sách đã huấn luyện trên số lượng UAV N khác với lúc huấn luyện, không huấn luyện lại.
          Hai chế độ theo đề cương: <strong>fixed-map</strong> giữ nguyên bản đồ (thêm UAV = đàn đông hơn);{" "}
          <strong>density-controlled</strong> scale diện tích bản đồ và số nạn nhân theo N để giữ mật độ không đổi
          (tách ảnh hưởng của số lượng UAV khỏi ảnh hưởng của mật độ). Vạch đứng nét đứt là N lúc huấn luyện. Mọi
          kiến trúc của họ encoder dùng chung giao thức lấy mẫu; MADDPG bị loại vì critic nối cố định N tác nhân.
        </p>
      </div>

      {loading && <div className="text-sm text-slate-500">Đang tải...</div>}

      {!loading && !report && (
        <Card title={`Chưa có dữ liệu zero-shot (${mode === "density" ? "density-controlled" : "fixed-map"})`}>
          <p className="text-xs text-slate-500">
            Chạy <code>python -m scripts.run_zero_shot_all --out-dir results_v4 --mode {mode} --seeds 0-9</code> rồi{" "}
            <code>python -m scripts.export_web_data --results-dir results_v4</code>.
          </p>
        </Card>
      )}

      {report && (
        <>
          <Card
            title={METRICS.find((m) => m.key === metric)!.label}
            subtitle={`Huấn luyện với N=${report.train_n} · trung bình ${
              report.methods[methods[0]]?.n_seeds ?? 0
            } seed · ${mode === "density" ? "density-controlled" : "fixed-map"}${
              report.env_by_n
                ? " · bản đồ: " +
                  report.ns
                    .map((n) => {
                      const e = report.env_by_n?.[String(n)];
                      return e ? `N=${n}→${e.width}×${e.height}` : null;
                    })
                    .filter(Boolean)
                    .join(", ")
                : ""
            }`}
          >
            <ResponsiveContainer width="100%" height={340}>
              <LineChart data={chartData} margin={{ top: 8, right: 150, bottom: 18, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis
                  dataKey="n"
                  stroke="#64748b"
                  tick={{ fontSize: 11 }}
                  label={{ value: "Số UAV lúc kiểm thử (N)", position: "insideBottom", offset: -8, fontSize: 11, fill: "#64748b" }}
                />
                <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", fontSize: 12 }}
                  formatter={(v, name) => [
                    typeof v === "number" ? v.toFixed(3) : "—",
                    METHOD_LABELS[name as Method] ?? String(name),
                  ]}
                  labelFormatter={(n) => `N = ${n}`}
                />
                <Legend
                  verticalAlign="top"
                  align="right"
                  layout="vertical"
                  iconSize={9}
                  wrapperStyle={{ fontSize: 11, paddingLeft: 12, lineHeight: "18px" }}
                  formatter={(v) => METHOD_LABELS[v as Method] ?? v}
                />
                <ReferenceLine
                  x={report.train_n}
                  stroke="#94a3b8"
                  strokeDasharray="4 4"
                  label={{ value: "N huấn luyện", fontSize: 10, fill: "#94a3b8", position: "top" }}
                />
                {methods.map((m) => (
                  <Line
                    key={m}
                    dataKey={m}
                    stroke={METHOD_COLORS[m]}
                    strokeWidth={m === PROPOSED_METHOD ? 2.75 : 1.75}
                    dot={{ r: 3 }}
                    connectNulls
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </Card>

          <Card
            title="Hiệu năng giữ được khi N thay đổi"
            subtitle="Phần trăm so với chính kiến trúc đó tại N lúc huấn luyện — càng phẳng/càng cao càng tổng quát hóa tốt"
          >
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-(--panel-border-strong) text-slate-400">
                    <th className="py-2 pr-4 text-left font-semibold">Phương pháp</th>
                    {report.ns.map((n) => (
                      <th key={n} className="px-3 py-2 text-right font-semibold tabular-nums">
                        N={n}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {retention.map(({ method, values }) => (
                    <tr key={method} className="border-b border-(--panel-border)">
                      <td className="py-2 pr-4">
                        <span className="flex items-center gap-2">
                          <span
                            className="inline-block h-2 w-2 rounded-full"
                            style={{ background: METHOD_COLORS[method] }}
                          />
                          <span className={method === PROPOSED_METHOD ? "font-semibold text-cyan-200" : "text-slate-300"}>
                            {METHOD_LABELS[method]}
                          </span>
                        </span>
                      </td>
                      {values.map((v, i) => (
                        <td
                          key={i}
                          className={`px-3 py-2 text-right tabular-nums ${
                            report.ns[i] === report.train_n ? "text-slate-500" : "text-slate-300"
                          }`}
                        >
                          {v == null ? "—" : `${v.toFixed(0)}%`}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      )}
    </PanelLayout>
  );
}
