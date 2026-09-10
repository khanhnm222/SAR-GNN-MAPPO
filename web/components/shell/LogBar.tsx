"use client";

import clsx from "clsx";
import { useLog } from "@/lib/logContext";

export default function LogBar() {
  const { entries, clear } = useLog();
  const last = entries[entries.length - 1];

  const toneClass = {
    info: "text-slate-400",
    warn: "text-amber-400",
    error: "text-rose-400",
  } as const;

  return (
    <footer className="flex h-11 shrink-0 items-center gap-4 border-t border-[var(--panel-border)] bg-[var(--panel)] px-4 text-xs">
      <span className="label-caps font-bold text-slate-500">Log</span>
      <span className="text-slate-400">{entries.length === 0 ? "Ready" : "Đang chạy"}</span>
      <span
        className={clsx(
          "flex-1 truncate font-mono text-[11px]",
          last ? toneClass[last.level] : "text-slate-600"
        )}
        title={last?.message}
      >
        {last ? `(${last.time}) ${last.message}` : "Chưa có sự kiện nào."}
      </span>
      <button
        onClick={clear}
        className="label-caps rounded-md border border-[var(--panel-border-strong)] px-3 py-1 text-[10px] font-bold text-slate-300 hover:border-cyan-400/40 hover:text-cyan-300"
      >
        Clear
      </button>
    </footer>
  );
}
