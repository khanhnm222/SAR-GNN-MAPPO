"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { TrajectoryData } from "@/lib/types";
import { ACTION_LABELS } from "@/lib/types";

const UAV_COLORS = ["#22d3ee", "#a78bfa", "#f472b6", "#facc15", "#4ade80", "#fb923c", "#60a5fa", "#f87171"];
const R_SENSE = 3; // muc 1.5.1, Bang 2 (fixed across scenarios)
const CANVAS_SIZE = 560;
const DETECT_FLASH_FRAMES = 18; // how long a "victim found" pulse stays on screen

// mau theo do uu tien nan nhan, dung Hinh 2 cua de cuong (vang/cam/do = 1/2/3)
const PRIORITY_COLORS: Record<number, string> = { 1: "#eab308", 2: "#f97316", 3: "#ef4444" };
const priorityColor = (p: number) => PRIORITY_COLORS[p] ?? "#94a3b8";

function drawDiamond(ctx: CanvasRenderingContext2D, x: number, y: number, r: number) {
  ctx.beginPath();
  ctx.moveTo(x, y - r);
  ctx.lineTo(x + r, y);
  ctx.lineTo(x, y + r);
  ctx.lineTo(x - r, y);
  ctx.closePath();
}

type Layers = {
  coverage: boolean;
  comms: boolean;
  sense: boolean;
  trails: boolean;
  blocked: boolean;
};

export default function ReplayCanvas({
  data,
  width,
  height,
}: {
  data: TrajectoryData;
  width: number;
  height: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [frameIndex, setFrameIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(4);
  const [layers, setLayers] = useState<Layers>({
    coverage: true,
    comms: true,
    sense: true,
    trails: true,
    blocked: true,
  });
  const coverageGridRef = useRef<Uint8Array | null>(null);
  const terrainGridRef = useRef<Uint8Array | null>(null);
  const lastComputedRef = useRef(-1);

  const frames = data.frames;
  const scale = CANVAS_SIZE / Math.max(width, height);

  // uid -> stable colour, so a UAV keeps its colour even when the alive set changes
  const colorOf = (uid: number) => UAV_COLORS[uid % UAV_COLORS.length];

  // frame index at which each victim was found, for the detection pulse
  const detectionFrames = useMemo(() => {
    const m = new Map<number, { t: number; by: number | null; x: number; y: number; priority: number }>();
    const last = frames[frames.length - 1];
    if (!last) return m;
    last.victims.forEach((v, i) => {
      if (v.detected && v.detected_at_step != null) {
        m.set(i, { t: v.detected_at_step, by: v.detected_by ?? null, x: v.x, y: v.y, priority: v.priority });
      }
    });
    return m;
  }, [frames]);

  // reset accumulated grids whenever a new trajectory is loaded
  useEffect(() => {
    coverageGridRef.current = new Uint8Array(width * height);
    const terrain = new Uint8Array(width * height);
    for (const [ox, oy] of data.terrain_occupancy ?? []) {
      if (ox >= 0 && ox < width && oy >= 0 && oy < height) terrain[oy * width + ox] = 1;
    }
    terrainGridRef.current = terrain;
    lastComputedRef.current = -1;
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setFrameIndex(0);
  }, [data, width, height]);

  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => {
      setFrameIndex((i) => (i + 1 >= frames.length ? 0 : i + 1));
    }, Math.max(16, 200 / speed));
    return () => clearInterval(id);
  }, [playing, speed, frames.length]);

  // incrementally accumulate the coverage heatmap and the grown terrain up to frameIndex
  useEffect(() => {
    const grid = coverageGridRef.current;
    const terrain = terrainGridRef.current;
    if (!grid || !terrain) return;
    if (frameIndex < lastComputedRef.current) {
      grid.fill(0);
      terrain.fill(0);
      for (const [ox, oy] of data.terrain_occupancy ?? []) {
        if (ox >= 0 && ox < width && oy >= 0 && oy < height) terrain[oy * width + ox] = 1;
      }
      lastComputedRef.current = -1;
    }
    for (let f = lastComputedRef.current + 1; f <= frameIndex; f++) {
      const frame = frames[f];
      if (!frame) continue;
      // dynamic terrain (flood) growth — Hard scenario
      for (const [ox, oy] of frame.terrain_new ?? []) {
        if (ox >= 0 && ox < width && oy >= 0 && oy < height) terrain[oy * width + ox] = 1;
      }
      for (const uav of frame.uavs) {
        if (!uav.alive) continue;
        const cx = Math.round(uav.x);
        const cy = Math.round(uav.y);
        for (let dx = -R_SENSE; dx <= R_SENSE; dx++) {
          for (let dy = -R_SENSE; dy <= R_SENSE; dy++) {
            if (dx * dx + dy * dy > R_SENSE * R_SENSE) continue;
            const x = cx + dx;
            const y = cy + dy;
            if (x < 0 || x >= width || y < 0 || y >= height) continue;
            grid[y * width + x] = 1;
          }
        }
      }
    }
    lastComputedRef.current = frameIndex;
    draw();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [frameIndex, frames]);

  function draw() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const w = width * scale;
    const h = height * scale;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // coverage heatmap (cells already scanned by any UAV)
    const grid = coverageGridRef.current;
    if (grid && layers.coverage) {
      ctx.fillStyle = "rgba(16, 185, 129, 0.18)";
      const cell = scale;
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          if (grid[y * width + x]) ctx.fillRect(x * cell, y * cell, cell + 0.5, cell + 0.5);
        }
      }
    }

    // obstacles — Hinh 2: "vat can tinh" = o vuong xam; cells added by flood
    // growth appear as the replay advances (terrain_new per frame)
    const terrain = terrainGridRef.current;
    if (terrain) {
      ctx.fillStyle = "#475569";
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          if (terrain[y * width + x]) ctx.fillRect(x * scale, y * scale, scale + 0.5, scale + 0.5);
        }
      }
    }

    // grid border
    ctx.strokeStyle = "#1e293b";
    ctx.strokeRect(0, 0, w, h);

    const frame = frames[frameIndex];
    if (!frame) return;

    const posOf = new Map<number, { x: number; y: number; alive: boolean }>();
    for (const u of frame.uavs) posOf.set(u.uid, { x: u.x, y: u.y, alive: u.alive });

    // communication links — the live r_comm graph the DGAT passes messages over.
    // Opacity encodes signal strength q (graph_builder.signal_strength).
    if (layers.comms && frame.comm_edges) {
      for (const [a, b, q] of frame.comm_edges) {
        const pa = posOf.get(a);
        const pb = posOf.get(b);
        if (!pa || !pb) continue;
        ctx.beginPath();
        ctx.moveTo(pa.x * scale, pa.y * scale);
        ctx.lineTo(pb.x * scale, pb.y * scale);
        ctx.strokeStyle = "#38bdf8";
        ctx.globalAlpha = 0.12 + 0.55 * q;
        ctx.lineWidth = 0.6 + 1.8 * q;
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
      ctx.lineWidth = 1;
    }

    // victims — Hinh 2: hinh tron, mau theo do uu tien (vang/cam/do = 1/2/3)
    frame.victims.forEach((v, i) => {
      const x = v.x * scale;
      const y = v.y * scale;
      const r = 6 + v.priority;

      // detection pulse: expanding ring for a few frames after the find, plus a
      // line back to the UAV that found it
      const det = detectionFrames.get(i);
      if (det && frame.t >= det.t && frame.t - det.t < DETECT_FLASH_FRAMES) {
        const age = (frame.t - det.t) / DETECT_FLASH_FRAMES;
        ctx.beginPath();
        ctx.arc(x, y, r + age * 26, 0, Math.PI * 2);
        ctx.strokeStyle = "#4ade80";
        ctx.globalAlpha = 1 - age;
        ctx.lineWidth = 2;
        ctx.stroke();
        const finder = det.by != null ? posOf.get(det.by) : undefined;
        if (finder) {
          ctx.beginPath();
          ctx.moveTo(x, y);
          ctx.lineTo(finder.x * scale, finder.y * scale);
          ctx.strokeStyle = "#4ade80";
          ctx.globalAlpha = (1 - age) * 0.7;
          ctx.stroke();
        }
        ctx.globalAlpha = 1;
        ctx.lineWidth = 1;
      }

      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fillStyle = priorityColor(v.priority);
      ctx.globalAlpha = v.detected ? 0.35 : 1;
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.lineWidth = v.detected ? 2 : 1;
      ctx.strokeStyle = v.detected ? "#4ade80" : "#0f172a";
      ctx.stroke();
      if (r >= 7) {
        ctx.fillStyle = v.detected ? "#4ade80" : "#0f172a";
        ctx.font = "bold 9px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(String(v.priority), x, y + 0.5);
      }
    });

    // UAV trails (last 20 frames)
    if (layers.trails) {
      const trailStart = Math.max(0, frameIndex - 20);
      for (let i = 0; i < frame.uavs.length; i++) {
        const uid = frame.uavs[i].uid;
        ctx.beginPath();
        for (let f = trailStart; f <= frameIndex; f++) {
          const uav = frames[f]?.uavs[i];
          if (!uav) continue;
          const x = uav.x * scale;
          const y = uav.y * scale;
          if (f === trailStart) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.strokeStyle = colorOf(uid);
        ctx.globalAlpha = 0.5;
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
    }

    // UAV markers — Hinh 2: "UAV (hinh thoi)"
    for (const uav of frame.uavs) {
      const x = uav.x * scale;
      const y = uav.y * scale;

      // sense radius
      if (layers.sense) {
        ctx.beginPath();
        ctx.arc(x, y, R_SENSE * scale, 0, Math.PI * 2);
        ctx.strokeStyle = uav.alive ? colorOf(uav.uid) : "#475569";
        ctx.globalAlpha = 0.25;
        ctx.stroke();
        ctx.globalAlpha = 1;
      }

      // blocked move: the UAV asked to move but terrain refused it — this is
      // what obstacle avoidance looks like frame by frame
      if (layers.blocked && uav.blocked && uav.alive) {
        ctx.beginPath();
        ctx.arc(x, y, 10, 0, Math.PI * 2);
        ctx.strokeStyle = "#ef4444";
        ctx.globalAlpha = 0.85;
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.globalAlpha = 1;
        ctx.lineWidth = 1;
      }

      drawDiamond(ctx, x, y, 6);
      ctx.fillStyle = uav.alive ? colorOf(uav.uid) : "#475569";
      ctx.fill();
      ctx.strokeStyle = "#0f172a";
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  }

  useEffect(draw, [frameIndex, layers]); // eslint-disable-line react-hooks/exhaustive-deps

  const frame = frames[frameIndex];
  const nLinks = frame?.comm_edges?.length ?? 0;
  const nAlive = frame ? frame.uavs.filter((u) => u.alive).length : 0;
  const nBlocked = frame ? frame.uavs.filter((u) => u.blocked && u.alive).length : 0;
  const hasComms = frames.some((f) => f.comm_edges !== undefined);

  return (
    <div className="flex flex-col gap-3">
      <div className="viewport-grid flex justify-center rounded-lg border border-[var(--panel-border)] p-2">
        <canvas ref={canvasRef} width={width * scale} height={height * scale} className="rounded" />
      </div>

      <div className="flex flex-wrap items-center gap-2 text-[11px]">
        <span className="label-caps text-slate-500">Lớp hiển thị</span>
        {(
          [
            ["coverage", "Vùng đã quét"],
            ["comms", "Liên kết truyền tin"],
            ["sense", "Bán kính cảm biến"],
            ["trails", "Vệt bay"],
            ["blocked", "Va vật cản"],
          ] as [keyof Layers, string][]
        ).map(([key, label]) => (
          <button
            key={key}
            onClick={() => setLayers((l) => ({ ...l, [key]: !l[key] }))}
            className={`rounded-md border px-2 py-1 transition-colors ${
              layers[key]
                ? "border-cyan-400/40 bg-cyan-400/10 text-cyan-300"
                : "border-[var(--panel-border)] text-slate-500 hover:text-slate-300"
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 text-[11px] text-slate-400">
        <LegendItem shape="diamond" color="#22d3ee" label="UAV (hình thoi)" />
        <LegendItem shape="line" color="#38bdf8" label="Liên kết trong r_comm (đậm = tín hiệu mạnh)" />
        <LegendItem shape="circle" color="#eab308" label="Nạn nhân ưu tiên 1" />
        <LegendItem shape="circle" color="#f97316" label="Ưu tiên 2" />
        <LegendItem shape="circle" color="#ef4444" label="Ưu tiên 3" />
        <LegendItem shape="circle" color="#eab308" faded label="Đã tìm thấy (viền xanh)" />
        <LegendItem shape="ring" color="#ef4444" label="Bị vật cản chặn" />
        <LegendItem shape="square" color="#475569" label="Vật cản" />
      </div>

      <div className="grid grid-cols-2 gap-3 text-xs text-slate-400 sm:grid-cols-3 lg:grid-cols-6">
        <Stat label="Bước" value={frame ? `${frame.t}` : "-"} />
        <Stat label="Bao phủ" value={frame ? `${(frame.coverage_rate * 100).toFixed(1)}%` : "-"} />
        <Stat label="Nạn nhân đã tìm" value={frame ? `${frame.victims_detected}/${frame.n_victims}` : "-"} />
        <Stat label="UAV hoạt động" value={frame ? `${nAlive}/${frame.uavs.length}` : "-"} />
        <Stat label="Liên kết truyền tin" value={hasComms ? `${nLinks}` : "n/a"} />
        <Stat label="UAV bị chặn" value={frame ? `${nBlocked}` : "-"} />
      </div>

      {!hasComms && (
        <p className="text-[11px] text-amber-400/80">
          Quỹ đạo này được ghi trước khi bổ sung dữ liệu đồ thị giao tiếp. Chạy lại{" "}
          <code className="text-amber-300">python -m scripts.reevaluate</code> rồi{" "}
          <code className="text-amber-300">python -m scripts.export_web_data</code> để hiển thị liên kết
          truyền tin và trạng thái tránh vật cản.
        </p>
      )}

      {frame && (
        <div className="flex flex-wrap gap-1.5 text-[10px]">
          {frame.uavs.map((u) => (
            <span
              key={u.uid}
              className="flex items-center gap-1 rounded border border-[var(--panel-border)] px-1.5 py-0.5"
              style={{ opacity: u.alive ? 1 : 0.4 }}
            >
              <span
                className="inline-block h-2 w-2 rotate-45"
                style={{ background: u.alive ? colorOf(u.uid) : "#475569" }}
              />
              <span className="text-slate-400">#{u.uid}</span>
              <span className={u.blocked ? "text-red-400" : "text-slate-300"}>
                {u.action != null ? ACTION_LABELS[u.action] ?? u.action : "-"}
              </span>
              <span className="text-slate-500">{Math.round(u.energy_ratio * 100)}%</span>
            </span>
          ))}
        </div>
      )}

      <div className="flex items-center gap-3">
        <button
          onClick={() => setPlaying((p) => !p)}
          className="label-caps rounded-md bg-cyan-400/10 px-3 py-1.5 text-xs font-bold text-cyan-300 hover:bg-cyan-400/20"
        >
          {playing ? "Tạm dừng" : "Phát"}
        </button>
        <input
          type="range"
          min={0}
          max={frames.length - 1}
          value={frameIndex}
          onChange={(e) => setFrameIndex(Number(e.target.value))}
          className="flex-1 accent-cyan-400"
        />
        <span className="w-16 text-right text-xs text-slate-500">
          {frameIndex + 1}/{frames.length}
        </span>
        <select
          value={speed}
          onChange={(e) => setSpeed(Number(e.target.value))}
          className="rounded-md border border-[var(--panel-border-strong)] bg-[var(--panel)] px-2 py-1 text-xs"
        >
          {[1, 2, 4, 8, 16].map((s) => (
            <option key={s} value={s}>
              {s}×
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}

function LegendItem({
  shape,
  color,
  label,
  faded,
}: {
  shape: "diamond" | "circle" | "square" | "line" | "ring";
  color: string;
  label: string;
  faded?: boolean;
}) {
  return (
    <span className="flex items-center gap-1.5">
      <span
        className="inline-block shrink-0"
        style={{
          width: shape === "line" ? "14px" : "10px",
          height: shape === "line" ? "2px" : "10px",
          background: shape === "ring" ? "transparent" : color,
          border: shape === "ring" ? `1.5px solid ${color}` : undefined,
          opacity: faded ? 0.35 : 1,
          borderRadius: shape === "circle" || shape === "ring" ? "9999px" : shape === "diamond" ? "2px" : "1px",
          transform: shape === "diamond" ? "rotate(45deg)" : undefined,
          outline: faded ? "1.5px solid #4ade80" : undefined,
          outlineOffset: faded ? "1px" : undefined,
        }}
      />
      {label}
    </span>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-[var(--panel-border)] bg-[var(--panel)] px-3 py-2">
      <div className="text-[10px] uppercase tracking-wide text-slate-500">{label}</div>
      <div className="text-sm font-semibold text-slate-200">{value}</div>
    </div>
  );
}
