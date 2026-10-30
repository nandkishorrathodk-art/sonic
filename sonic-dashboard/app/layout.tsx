import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: {
    default: "SONIC — Autonomous AI-Human Digital Being & Workstation",
    template: "%s · SONIC",
  },
  description:
    "Autonomous AI-Human Digital Being operating its dedicated personal computer workstation: dual-plane GUI & terminal agency, persistent cognitive memory, and self-evolving craft.",
  applicationName: "SONIC",
  authors: [{ name: "SONIC" }],
  keywords: ["autonomous agent", "AI-Human digital being", "computer workstation", "cognitive memory", "dual-plane agency", "AI colleague"],
};

export const viewport: Viewport = {
  themeColor: "#070910",
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="bg-ink-950 text-slate-200 min-h-screen antialiased font-sans selection:bg-primary-500 selection:text-white">
        {children}
      </body>
    </html>
  );
}
