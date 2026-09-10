import type { ReactNode } from "react";

export default function Card({
  title,
  subtitle,
  children,
  className,
}: {
  title?: string;
  subtitle?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={`rounded-lg border border-[var(--panel-border)] bg-[var(--panel)]/90 p-5 backdrop-blur-sm ${className ?? ""}`}
    >
      {title && <h3 className="label-caps text-xs font-bold text-slate-200">{title}</h3>}
      {subtitle && <p className="mt-1 text-xs text-slate-500">{subtitle}</p>}
      <div className={title ? "mt-3" : ""}>{children}</div>
    </div>
  );
}
