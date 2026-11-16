import type { Metadata, Viewport } from "next";
import "./globals.css";

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: "#060913",
};

export const metadata: Metadata = {
  title: "News Detection — Multi-Model AI Neural Ensemble",
  description: "Enterprise-grade real-time news authenticity classification powered by RoBERTa, Distilled TinyBERT, and BERT-Detect with multilingual cross-lingual routing.",
  keywords: ["News Detection", "NLP", "Machine Learning", "Transformers", "Fact Checking", "AI Verification"],
  authors: [{ name: "News Detection AI Team" }],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
      </head>
      <body>
        <div className="ambient-glow" />
        <div className="grid-overlay" />
        <div style={{ position: "relative", zIndex: 1, minHeight: "100vh" }}>
          {children}
        </div>
      </body>
    </html>
  );
}
