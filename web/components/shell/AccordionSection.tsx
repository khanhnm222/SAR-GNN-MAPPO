"use client";

import { useState } from "react";
import clsx from "clsx";
import Badge from "./Badge";

export default function AccordionSection({
  title,
  badge,
  badgeTone = "accent",
  defaultOpen = false,
  children,
}: {
  title: string;
  badge?: string;
  badgeTone?: "accent" | "muted" | "warn";
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div
      className={clsx(
        "rounded-lg border bg-[var(--panel)] transition-colors",
        open ? "border-cyan-400/30" : "border-[var(--panel-border)]"
      )}
    >
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between gap-3 px-4 py-3 text-left"
      >
        <span className="flex items-center gap-2.5">
          <span className="label-caps text-xs font-bold text-slate-200">{title}</span>
          {badge && <Badge tone={badgeTone}>{badge}</Badge>}
        </span>
        <span
          className={clsx(
            "grid h-6 w-6 shrink-0 place-items-center rounded-full bg-white/5 text-[10px] text-cyan-300 transition-transform duration-200",
            open && "rotate-90"
          )}
        >
          &#9654;
        </span>
      </button>
      <div
        className="grid transition-[grid-template-rows] duration-200 ease-out"
        style={{ gridTemplateRows: open ? "1fr" : "0fr" }}
      >
        <div className="overflow-hidden">
          <div className="space-y-3 border-t border-[var(--panel-border)] px-4 py-3.5">{children}</div>
        </div>
      </div>
    </div>
  );
}
