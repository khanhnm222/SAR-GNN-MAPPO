export default function PanelLayout({
  sidebar,
  children,
}: {
  sidebar: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className="flex h-full flex-col md:min-h-0 md:flex-row">
      <aside className="shrink-0 space-y-3 border-b border-[var(--panel-border)] bg-[var(--panel-soft)] p-4 md:h-full md:w-[336px] md:overflow-y-auto md:border-b-0 md:border-r">
        {sidebar}
      </aside>
      <div className="viewport-grid flex-1 space-y-5 overflow-y-auto p-4 md:h-full md:p-6">{children}</div>
    </div>
  );
}
