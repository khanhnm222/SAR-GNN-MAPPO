import clsx from "clsx";

export default function Badge({
  children,
  tone = "accent",
}: {
  children: React.ReactNode;
  tone?: "accent" | "muted" | "warn";
}) {
  return (
    <span
      className={clsx(
        "label-caps inline-flex items-center rounded-full px-2 py-0.5 text-[10px] font-bold tracking-wider",
        tone === "accent" && "bg-cyan-400/10 text-cyan-300",
        tone === "muted" && "bg-slate-500/10 text-slate-400",
        tone === "warn" && "bg-amber-400/10 text-amber-300"
      )}
    >
      {children}
    </span>
  );
}
