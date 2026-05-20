import type { Metadata, Viewport } from "next";
import type { ReactNode } from "react";
import { PwaRegister } from "../components/PwaRegister";
import "./globals.css";
import { Plus_Jakarta_Sans, Space_Grotesk } from "next/font/google";

/* ── DISPLAY / HEADING font: Space Grotesk — geometric, punchy, ultra-modern ── */
const spaceGrotesk = Space_Grotesk({
  subsets: ["latin"],
  variable: "--font-heading",
  weight: ["300", "400", "500", "600", "700"],
  display: "swap",
});

/* ── BODY font: Plus Jakarta Sans — clean, humanist, premium ── */
const jakarta = Plus_Jakarta_Sans({
  subsets: ["latin"],
  variable: "--font-body",
  weight: ["300", "400", "500", "600", "700", "800"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "MindPulse",
  description: "Real-time stress detection powered by face and voice analysis",
  manifest: "/manifest.webmanifest",
  appleWebApp: { capable: true, title: "MindPulse", statusBarStyle: "black-translucent" },
  icons: { icon: "/icon.svg", apple: "/icon.svg" },
};

export const viewport: Viewport = { themeColor: "#090f0d" };

export default function RootLayout({ children }: Readonly<{ children: ReactNode }>) {
  return (
    <html lang="en" className={`${spaceGrotesk.variable} ${jakarta.variable}`}>
      <body>{children}<PwaRegister /></body>
    </html>
  );
}
