"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import Card from "@/components/Card";
import AccordionSection from "@/components/shell/AccordionSection";
import PanelLayout from "@/components/shell/PanelLayout";
import { fetchComparison, fetchManifest } from "@/lib/data";
import { useLog } from "@/lib/logContext";
import type { Comparison, Manifest, Scenario } from "@/lib/types";
import { METHOD_LABELS, METHOD_ORDER } from "@/lib/types";
import { SCENARIO_DIMS } from "@/lib/scenarios";

const SCENARIO_LABELS: Record<Scenario, string> = {
  easy: "Easy",
  medium: "Medium",
  hard: "Hard",
};

const METHOD_NOTES: Record<string, string> = {
  random_walk: "Heuristic — di chuyển ngẫu nhiên",
  greedy: "Heuristic — chọn ô xác suất cao nhất",
  maddpg: "MARL — không đồ thị (Lowe et al. 2017)",
  mappo_mlp: "MARL — không đồ thị (Yu et al. 2022)",
  mappo_gcn: "Ablation kiến trúc — GCN, không attention",
  mappo_gat: "Ablation kiến trúc — GAT, không GRU",
  gnn_mappo: "Đề xuất — DGAT (GAT động + GRU)",
};

export default function DashboardPage() {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [comparisons, setComparisons] = useState<Partial<Record<Scenario, Comparison>>>({});
  const log = useLog();

  useEffect(() => {
    log.push("Kết nối tới public/data/manifest.json...");
    fetchManifest().then((m) => {
      setManifest(m);
      log.push(m ? `Tìm thấy ${m.scenarios.length} kịch bản có dữ liệu.` : "Không đọc được manifest.json.", m ? "info" : "error");
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!manifest) return;
    manifest.scenarios.forEach((s) => {
      fetchComparison(s).then((c) => {
        if (c) {
          setComparisons((prev) => ({ ...prev, [s]: c }));
          log.push(`Đã tải so sánh baseline cho kịch bản "${s}".`);
        }
      });
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [manifest]);

  const sidebar = (
    <>
      <AccordionSection title="Trạng thái dữ liệu" badge="LIVE" defaultOpen>
        <div className="space-y-1.5 text-xs">
          {(["easy", "medium", "hard"] as Scenario[]).map((s) => {
            const loaded = manifest?.scenarios.includes(s);
            return (
              <div key={s} className="flex items-center justify-between rounded-md bg-white/[0.03] px-2.5 py-1.5">
                <span className="text-slate-300">{SCENARIO_LABELS[s]}</span>
                <span
                  className={
                    loaded ? "label-caps text-[10px] font-bold text-cyan-300" : "label-caps text-[10px] text-slate-600"
                  }
                >
                  {loaded ? "Sẵn sàng" : "Chưa có"}
                </span>
              </div>
            );
          })}
        </div>
      </AccordionSection>

      <AccordionSection title="Phương pháp" badge={`${METHOD_ORDER.length}`}>
        <div className="space-y-2">
          {METHOD_ORDER.map((m) => (
            <div key={m} className="text-xs">
              <div className="font-semibold text-slate-200">{METHOD_LABELS[m]}</div>
              <div className="text-[11px] text-slate-500">{METHOD_NOTES[m]}</div>
            </div>
          ))}
        </div>
      </AccordionSection>

      <AccordionSection title="Tài liệu" badge="INFO" badgeTone="muted">
        <p className="text-xs text-slate-400">
          Đặc tả đầy đủ và phạm vi rút gọn (POC) nằm trong{" "}
          <code className="rounded bg-white/10 px-1 py-0.5 text-[11px]">PLAN.md ở gốc dự án.</code>
        </p>
      </AccordionSection>
    </>
  );

  return (
    <PanelLayout sidebar={sidebar}>
      <section>
        <h1 className="text-xl font-bold text-slate-100">
          Framework GNN-MAPPO cho phối hợp đàn UAV tìm kiếm cứu nạn
        </h1>
        <p className="mt-2 max-w-3xl text-sm text-slate-400">
          Trực quan hóa thực nghiệm cho đề tài <em>&quot;Xây dựng và đánh giá framework GNN-MAPPO cho bài toán
          phối hợp đàn UAV tìm kiếm và cứu nạn trong môi trường mô phỏng&quot;</em>. Dữ liệu trên trang này được
          sinh ra từ các lượt huấn luyện và đánh giá thật, chạy ở quy mô rút gọn (POC) so với quy mô đầy đủ của
          đề cương.
        </p>
      </section>

      {!manifest && <div className="text-sm text-slate-500">Đang tải dữ liệu...</div>}
      {manifest && manifest.scenarios.length === 0 && (
        <Card title="Chưa có dữ liệu">
          <p className="text-sm text-slate-400">
            Chưa tìm thấy kết quả nào trong <code>web/public/data</code>. Hãy chạy{" "}
            <code>python -m scripts.run_poc</code> rồi <code>python -m scripts.export_web_data</code>.
          </p>
        </Card>
      )}

      <div className="grid gap-5 md:grid-cols-3">
        {manifest?.scenarios.map((scenario) => {
          const dims = SCENARIO_DIMS[scenario];
          const comp = comparisons[scenario];
          const best = comp
            ? METHOD_ORDER.filter((m) => comp.methods[m]?.metrics.victim_detection_rate?.mean != null)
                .map((m) => ({ m, v: comp.methods[m]!.metrics.victim_detection_rate!.mean! }))
                .sort((a, b) => b.v - a.v)[0]
            : undefined;
          return (
            <Card key={scenario} title={`Kịch bản ${SCENARIO_LABELS[scenario]}`} subtitle={`${dims.nUavs} UAV · bản đồ ${dims.width}×${dims.height}`}>
              {best ? (
                <div className="space-y-1">
                  <div className="text-xs text-slate-500">Phương pháp tốt nhất (VDR trung bình)</div>
                  <div className="text-lg font-semibold text-cyan-300">{METHOD_LABELS[best.m]}</div>
                  <div className="text-sm text-slate-400">{(best.v * 100).toFixed(1)}% nạn nhân được phát hiện</div>
                </div>
              ) : (
                <div className="text-xs text-slate-500">Đang chờ dữ liệu đánh giá...</div>
              )}
              <div className="mt-4 flex flex-wrap gap-3">
                <Link href="/training" className="text-xs text-cyan-400 hover:underline">
                  Learning curve →
                </Link>
                <Link href="/comparison" className="text-xs text-cyan-400 hover:underline">
                  So sánh →
                </Link>
                <Link href="/replay" className="text-xs text-cyan-400 hover:underline">
                  Phát lại →
                </Link>
              </div>
            </Card>
          );
        })}
      </div>
      <div className="pb-2" />
    </PanelLayout>
  );
}
