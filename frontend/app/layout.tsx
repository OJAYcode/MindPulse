import type { Metadata, Viewport } from "next";
import type { ReactNode } from "react";
import { PwaRegister } from "../components/PwaRegister";
import "./globals.css";

export const metadata: Metadata = {
  title: "MindPulse",
  description: "Wellbeing analytics workspace for stress and emotional-state tracking",
  manifest: "/manifest.webmanifest",
  appleWebApp: {
    capable: true,
    title: "MindPulse",
    statusBarStyle: "black-translucent"
  },
  icons: {
    icon: "/icon.svg",
    apple: "/icon.svg"
  }
};

export const viewport: Viewport = {
  themeColor: "#18211f"
};

export default function RootLayout({ children }: Readonly<{ children: ReactNode }>) {
  return (
    <html lang="en">
      <body>
        <PwaRegister />
        {children}
      </body>
    </html>
  );
}
