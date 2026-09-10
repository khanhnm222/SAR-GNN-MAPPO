import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import TopNav from "@/components/shell/TopNav";
import LogBar from "@/components/shell/LogBar";
import { LogProvider } from "@/lib/logContext";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "GNN-MAPPO SAR Toolkit",
  description: "Trực quan hóa huấn luyện & thực nghiệm GNN-MAPPO cho phối hợp đàn UAV tìm kiếm cứu nạn",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="vi"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="flex h-full flex-col overflow-y-auto bg-[var(--background)] text-[var(--foreground)] md:h-screen md:overflow-hidden">
        <LogProvider>
          <TopNav />
          <main className="min-h-0 flex-1 md:overflow-hidden">{children}</main>
          <LogBar />
        </LogProvider>
      </body>
    </html>
  );
}
