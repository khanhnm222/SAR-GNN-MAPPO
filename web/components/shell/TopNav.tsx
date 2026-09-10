"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import clsx from "clsx";

const LINKS = [
  { href: "/", label: "Tổng quan" },
  { href: "/training", label: "Huấn luyện" },
  { href: "/comparison", label: "So sánh" },
  { href: "/ablation", label: "Ablation" },
  { href: "/zero-shot", label: "Zero-shot" },
  { href: "/replay", label: "Phát lại" },
];

export default function TopNav() {
  const pathname = usePathname();
  const [mobileOpen, setMobileOpen] = useState(false);
  const [prevPathname, setPrevPathname] = useState(pathname);

  if (pathname !== prevPathname) {
    setPrevPathname(pathname);
    setMobileOpen(false);
  }

  return (
    <header className="relative z-50 shrink-0 bg-[var(--panel)]">
      <div className="flex h-14 items-center gap-6 px-4">
        <Link href="/" className="flex flex-col leading-none">
          <span className="label-caps text-[10px] font-semibold text-slate-500">SAR Research Toolkit</span>
          <span className="label-caps text-sm font-extrabold text-slate-100">
            GNN-<span className="text-cyan-400">MAPPO</span>
          </span>
        </Link>

        <nav className="hidden flex-1 items-center gap-1 md:flex">
          {LINKS.map((l) => {
            const active = pathname === l.href;
            return (
              <Link
                key={l.href}
                href={l.href}
                className={clsx(
                  "label-caps rounded-full px-3.5 py-1.5 text-[11px] font-bold transition-colors",
                  active
                    ? "bg-cyan-400 text-slate-950"
                    : "text-slate-400 hover:bg-white/5 hover:text-slate-200"
                )}
              >
                {l.label}
              </Link>
            );
          })}
        </nav>
        <div className="flex-1 md:hidden" />

        <span className="label-caps hidden items-center rounded-full border border-[var(--panel-border-strong)] px-3 py-1 text-[10px] font-bold text-cyan-300 sm:inline-flex">
          Dữ liệu thật · POC
        </span>

        <button
          onClick={() => setMobileOpen((o) => !o)}
          className="grid h-9 w-9 shrink-0 place-items-center rounded-md border border-[var(--panel-border-strong)] text-slate-300 md:hidden"
          aria-label="Menu"
        >
          <div className="space-y-1">
            <span className="block h-0.5 w-4 bg-current" />
            <span className="block h-0.5 w-4 bg-current" />
            <span className="block h-0.5 w-4 bg-current" />
          </div>
        </button>
      </div>
      <div className="glow-divider" />

      {mobileOpen && (
        <nav className="flex flex-col gap-1 border-b border-[var(--panel-border)] bg-[var(--panel)] p-3 md:hidden">
          {LINKS.map((l) => (
            <Link
              key={l.href}
              href={l.href}
              className={clsx(
                "label-caps rounded-md px-3 py-2 text-xs font-bold",
                pathname === l.href ? "bg-cyan-400/10 text-cyan-300" : "text-slate-400"
              )}
            >
              {l.label}
            </Link>
          ))}
        </nav>
      )}
    </header>
  );
}
