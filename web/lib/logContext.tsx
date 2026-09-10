"use client";

import { createContext, useCallback, useContext, useRef, useState } from "react";

export type LogLevel = "info" | "warn" | "error";

export interface LogEntry {
  id: number;
  level: LogLevel;
  message: string;
  time: string;
}

interface LogContextValue {
  entries: LogEntry[];
  push: (message: string, level?: LogLevel) => void;
  clear: () => void;
}

const LogContext = createContext<LogContextValue | null>(null);

export function LogProvider({ children }: { children: React.ReactNode }) {
  const [entries, setEntries] = useState<LogEntry[]>([]);
  const idRef = useRef(0);

  const push = useCallback((message: string, level: LogLevel = "info") => {
    idRef.current += 1;
    const time = new Date().toLocaleTimeString("vi-VN", { hour12: false });
    setEntries((prev) => [...prev.slice(-49), { id: idRef.current, level, message, time }]);
  }, []);

  const clear = useCallback(() => setEntries([]), []);

  return <LogContext.Provider value={{ entries, push, clear }}>{children}</LogContext.Provider>;
}

export function useLog() {
  const ctx = useContext(LogContext);
  if (!ctx) throw new Error("useLog must be used within LogProvider");
  return ctx;
}
