"use client";

import clsx from "clsx";

export interface SidebarOption {
  value: string;
  label: string;
  hint?: string;
  disabled?: boolean;
  dotColor?: string;
}

export default function SidebarOptionList({
  options,
  value,
  onChange,
}: {
  options: SidebarOption[];
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <div className="space-y-1">
      {options.map((opt) => {
        const active = opt.value === value;
        return (
          <button
            key={opt.value}
            disabled={opt.disabled}
            onClick={() => onChange(opt.value)}
            className={clsx(
              "flex w-full items-center gap-2 rounded-md border px-3 py-2 text-left text-xs transition-colors",
              active
                ? "border-cyan-400/40 bg-cyan-400/10 text-cyan-200"
                : "border-transparent text-slate-400 hover:bg-white/5 hover:text-slate-200",
              opt.disabled && "cursor-not-allowed opacity-40 hover:bg-transparent"
            )}
          >
            {opt.dotColor && (
              <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: opt.dotColor }} />
            )}
            <span className="flex-1 font-medium">{opt.label}</span>
            {opt.hint && <span className="text-[10px] text-slate-500">{opt.hint}</span>}
          </button>
        );
      })}
    </div>
  );
}
