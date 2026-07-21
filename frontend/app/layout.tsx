import type { ReactNode } from "react";
import type { Metadata } from "next";
import { IBM_Plex_Mono, Instrument_Sans, Lora } from "next/font/google";
import "./globals.css";

const instrumentSans = Instrument_Sans({
  subsets: ["latin"],
  variable: "--font-instrument-sans",
  display: "swap",
});

const lora = Lora({
  subsets: ["latin"],
  style: ["normal", "italic"],
  variable: "--font-lora",
  display: "swap",
});

const plexMono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-plex-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "AI Tutor",
  description:
    "A modern chat workspace for the AI Tutor retrieval app, powered by FastAPI and LangGraph.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${instrumentSans.variable} ${lora.variable} ${plexMono.variable}`}
    >
      <body suppressHydrationWarning>{children}</body>
    </html>
  );
}
