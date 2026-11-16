"use client";

import React, { useState, useEffect, useCallback } from "react";
import dynamic from "next/dynamic";
import { Header } from "@/components/Header";
import { GaugeMeter } from "@/components/GaugeMeter";
import { ModelGrid } from "@/components/ModelGrid";
import { FactCheckEvidence } from "@/components/FactCheckEvidence";
import { ArticleMetaHeader } from "@/components/ArticleMetaHeader";
import { SAMPLE_PRESETS } from "@/data/presets";
import { AnalyzeResponse, AnalysisHistoryItem } from "@/types/news";
import {
  Sparkles,
  Trash2,
  AlertCircle,
  Loader2,
  Globe,
  Link2,
  FileText,
  DownloadCloud,
  Layers,
  Check,
  Copy,
  CheckCircle2,
} from "lucide-react";

// Phase 5 Performance Optimization: Code-Split Heavy Subsystems with next/dynamic
const BatchIngestionView = dynamic(
  () => import("@/components/BatchIngestionView").then((mod) => mod.BatchIngestionView),
  {
    ssr: false,
    loading: () => (
      <div
        style={{
          padding: "48px 24px",
          textAlign: "center",
          borderRadius: "var(--radius-xl)",
          background: "var(--card-bg)",
          border: "1px solid var(--card-border)",
          color: "var(--text-muted)",
        }}
      >
        <Loader2 className="spinner" size={28} style={{ margin: "0 auto 12px", display: "block" }} />
        <span style={{ fontSize: "0.9rem" }}>Loading High-Throughput Batch Ingestion Engine...</span>
      </div>
    ),
  }
);

const ExplainableAI = dynamic(
  () => import("@/components/ExplainableAI").then((mod) => mod.ExplainableAI),
  { ssr: false }
);

const HistoryDrawer = dynamic(
  () => import("@/components/HistoryDrawer").then((mod) => mod.HistoryDrawer),
  { ssr: false }
);

const SAMPLE_URL_PRESETS = [
  {
    title: "Wikipedia: Artificial Intelligence",
    url: "https://en.wikipedia.org/wiki/Artificial_intelligence",
    label: "Knowledge",
  },
  {
    title: "Wikipedia: James Webb Telescope",
    url: "https://en.wikipedia.org/wiki/James_Webb_Space_Telescope",
    label: "Science",
  },
  {
    title: "Wikipedia: Climate Change",
    url: "https://en.wikipedia.org/wiki/Climate_change",
    label: "Environment",
  },
];

function createHistoryId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `${Date.now()}`;
}

// Persistent LocalStorage Key & Capacity Constraint
const STORAGE_KEY_HISTORY = "news_detection_history_v1";
const MAX_HISTORY_ITEMS = 15;

function getInitialHistory(): AnalysisHistoryItem[] {
  if (typeof window === "undefined") return [];
  try {
    const saved = localStorage.getItem(STORAGE_KEY_HISTORY);
    if (saved) {
      const parsed: AnalysisHistoryItem[] = JSON.parse(saved);
      if (Array.isArray(parsed) && parsed.length > 0) {
        parsed.forEach((item) => {
          if (item?.response) {
            if (item.response.snippet) {
              clientPredictionCache.set(`text:${item.response.snippet.toLowerCase()}`, item.response);
            }
            if (item.response.article_meta?.url) {
              clientPredictionCache.set(`url:${item.response.article_meta.url.toLowerCase()}`, item.response);
            }
          }
        });
        return parsed.slice(0, MAX_HISTORY_ITEMS);
      }
    }
  } catch (err) {
    console.warn("Failed to load initial history:", err);
  }
  return [];
}

// Client-Side In-Memory Response Cache (Zero latency on repeated queries or history switching)
const clientPredictionCache = new Map<string, AnalyzeResponse>();

export default function Home() {
  const [inputText, setInputText] = useState<string>("");
  const [inputUrl, setInputUrl] = useState<string>("");
  const [inputMode, setInputMode] = useState<"text" | "url" | "batch">("text");
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isScraping, setIsScraping] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [history, setHistory] = useState<AnalysisHistoryItem[]>(getInitialHistory);

  // Synchronize history changes to localStorage
  useEffect(() => {
    try {
      if (typeof window !== "undefined") {
        if (history.length > 0) {
          localStorage.setItem(STORAGE_KEY_HISTORY, JSON.stringify(history.slice(0, MAX_HISTORY_ITEMS)));
        } else {
          localStorage.removeItem(STORAGE_KEY_HISTORY);
        }
      }
    } catch (err) {
      console.warn("Failed to save persistent analysis history:", err);
    }
  }, [history]);

  // Text metrics & Reading ergonomics
  const trimmed = inputText.trim();
  const wordCount = trimmed ? trimmed.split(/\s+/).filter(Boolean).length : 0;
  const charCount = inputText.length;
  const isInsufficientContext = wordCount > 0 && wordCount < 3;
  const isShortText = wordCount >= 3 && wordCount < 15;
  const isOptimalText = wordCount >= 15 && wordCount <= 500;
  const readingTimeSec = Math.max(5, Math.round(wordCount / 3.5));
  const readingTimeLabel = readingTimeSec < 60 ? `~${readingTimeSec}s read` : `~${Math.ceil(readingTimeSec / 60)}m read`;

  const [copiedToast, setCopiedToast] = useState<boolean>(false);

  const handleCopySummary = () => {
    if (!result) return;
    const summaryLines = [
      `==================================================`,
      `FACT-CHECK & AUTHENTICITY REPORT`,
      `==================================================`,
      `VERDICT         : ${result.verdict} (${result.soft_score.toFixed(1)}% Confidence)`,
      `CLAIM SHA-256   : ${result.claim_sha256 || "N/A"}`,
      `POLARIZED SPLIT : ${result.is_polarized ? `YES (Index: ${result.polarization_index ?? "N/A"})` : "NO"}`,
      `AGREEMENT       : ${result.agreement}`,
      `CATEGORY        : ${result.category?.category || "General News"} (${result.category?.confidence.toFixed(1) || "100.0"}%)`,
      `FACT-CHECK      : ${result.fact_check?.corroboration_status || "N/A"} (${result.fact_check?.corroboration_score?.toFixed(1) || "0.0"}% corroboration)`,
      `DETECTION MODEL : ${result.routing_engine || "4-Model AI Ensemble"}`,
      `PROCESSING TIME : ${result.execution_time_ms ? `${result.execution_time_ms.toFixed(1)}ms` : "Instant Cached (<0.1ms)"}`,
      `DATE GENERATED  : ${new Date().toUTCString()}`,
      `CLAIM SNIPPET   : "${result.snippet}"`,
      `==================================================`,
    ];
    navigator.clipboard.writeText(summaryLines.join("\n"));
    setCopiedToast(true);
    setTimeout(() => setCopiedToast(false), 2400);
  };

  const handleAnalyze = useCallback(async (textToAnalyze?: string) => {
    const text = (textToAnalyze !== undefined ? textToAnalyze : inputText).trim();
    if (!text) {
      setError("Please input or paste news text to analyze.");
      return;
    }

    const words = text.split(/\s+/).filter(Boolean);
    if (words.length < 3) {
      setError(
        `Input contains only ${words.length} word${words.length === 1 ? "" : "s"} ("${text}"). News authenticity verification requires a complete claim, headline, or article body (minimum 3 words).`
      );
      return;
    }

    // Phase 6.3: Lexical entropy & repetition spam guard
    const tokens = text
      .toLowerCase()
      .split(/\s+/)
      .map((t) => t.replace(/[^\w]/g, ""))
      .filter(Boolean);

    if (tokens.length >= 3 && new Set(tokens).size < 2) {
      setError(
        `Input consists of a single repeated word ("${tokens[0]}"). News authenticity verification requires a complete claim or headline with substantive context.`
      );
      return;
    }

    if (tokens.length >= 8) {
      const ttr = new Set(tokens).size / tokens.length;
      if (ttr < 0.35) {
        setError(
          `Input exhibits low lexical diversity (TTR: ${ttr.toFixed(2)} < 0.35) due to excessive word repetition. News authenticity verification requires genuine journalism or complete claims.`
        );
        return;
      }
    }

    // Phase 5: Client-Side In-Memory Cache Check (0ms Instant Return)
    const cacheKey = `text:${text.toLowerCase()}`;
    const cachedResponse = clientPredictionCache.get(cacheKey);
    if (cachedResponse) {
      const instantResponse: AnalyzeResponse = {
        ...cachedResponse,
        cached: true,
        cache_hit: true,
        cache_tier: "L1_RAM",
        execution_time_ms: 0.05,
      };
      setResult(instantResponse);
      setError(null);

      const now = new Date();
      const timeStr = now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
      const newItem: AnalysisHistoryItem = {
        id: createHistoryId(),
        timestamp: timeStr,
        textSnippet: instantResponse.article_meta
          ? `${instantResponse.article_meta.publisher}: ${instantResponse.article_meta.title}`
          : (text.length > 55 ? text.substring(0, 52) + "..." : text),
        response: instantResponse,
      };
      setHistory((prev) => (prev[0]?.textSnippet === newItem.textSnippet ? prev : [newItem, ...prev.slice(0, MAX_HISTORY_ITEMS - 1)]));
      return;
    }

    setIsLoading(true);
    setError(null);
    const startTime = performance.now();

    try {
      const res = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => null);
        if (res.status === 400 || res.status === 422) {
          throw new Error(errData?.detail || "Input contains insufficient context for analysis (minimum 3 words required).");
        }
        if (res.status === 429) {
          throw new Error("Rate limit exceeded: Max 30 requests per minute.");
        }
        if (res.status === 504) {
          throw new Error("Inference timeout: The neural ensemble exceeded the 30s deadline.");
        }
        throw new Error(errData?.detail || `Inference engine returned HTTP ${res.status}`);
      }

      const data: AnalyzeResponse = await res.json();
      data.execution_time_ms = performance.now() - startTime;
      clientPredictionCache.set(cacheKey, data);
      setResult(data);

      // Append to history
      const now = new Date();
      const timeStr = now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
      const newItem: AnalysisHistoryItem = {
        id: createHistoryId(),
        timestamp: timeStr,
        textSnippet: data.article_meta
          ? `${data.article_meta.publisher}: ${data.article_meta.title}`
          : (text.length > 55 ? text.substring(0, 52) + "..." : text),
        response: data,
      };
      setHistory((prev) => [newItem, ...prev.slice(0, MAX_HISTORY_ITEMS - 1)]);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Failed to analyze text. Please verify backend connection.";
      setError(message);
    } finally {
      setIsLoading(false);
    }
  }, [inputText]);

  const handleAnalyzeUrl = useCallback(async (targetUrl?: string) => {
    let url = (targetUrl !== undefined ? targetUrl : inputUrl).trim();
    if (!url) {
      setError("Please enter a valid news article or web page URL.");
      return;
    }

    if (!url.startsWith("http://") && !url.startsWith("https://")) {
      url = "https://" + url;
    }

    // Phase 6.2: Social media & streaming platform link guard
    const lowerUrl = url.toLowerCase();
    const blockedMedia = [
      "twitter.com",
      "x.com",
      "youtube.com",
      "youtu.be",
      "instagram.com",
      "tiktok.com",
      "facebook.com",
      "reddit.com",
    ];
    if (blockedMedia.some((d) => lowerUrl.includes(d))) {
      setError(
        "Social media and streaming platform links (e.g. YouTube, Twitter/X) cannot be scraped as news articles. Please copy and paste the text directly into the Claim / Text Input tab."
      );
      return;
    }

    // Phase 5: Client-Side In-Memory Cache Check (0ms Instant Return)
    const cacheKey = `url:${url.toLowerCase()}`;
    const cachedResponse = clientPredictionCache.get(cacheKey);
    if (cachedResponse) {
      const instantResponse: AnalyzeResponse = {
        ...cachedResponse,
        cached: true,
        cache_hit: true,
        cache_tier: "L1_RAM",
        execution_time_ms: 0.05,
      };
      setResult(instantResponse);
      setError(null);

      const now = new Date();
      const timeStr = now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
      const newItem: AnalysisHistoryItem = {
        id: createHistoryId(),
        timestamp: timeStr,
        textSnippet: instantResponse.article_meta
          ? `${instantResponse.article_meta.publisher}: ${instantResponse.article_meta.title}`
          : instantResponse.snippet,
        response: instantResponse,
      };
      setHistory((prev) => (prev[0]?.textSnippet === newItem.textSnippet ? prev : [newItem, ...prev.slice(0, MAX_HISTORY_ITEMS - 1)]));
      return;
    }

    setIsLoading(true);
    setError(null);
    const startTime = performance.now();

    try {
      const res = await fetch("/api/analyze-url", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => null);
        throw new Error(errData?.detail || `Server returned error (${res.status}) fetching URL.`);
      }

      const data: AnalyzeResponse = await res.json();
      data.execution_time_ms = performance.now() - startTime;
      clientPredictionCache.set(cacheKey, data);
      setResult(data);

      const now = new Date();
      const timeStr = now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
      const newItem: AnalysisHistoryItem = {
        id: createHistoryId(),
        timestamp: timeStr,
        textSnippet: data.article_meta ? `${data.article_meta.publisher}: ${data.article_meta.title}` : data.snippet,
        response: data,
      };
      setHistory((prev) => [newItem, ...prev.slice(0, MAX_HISTORY_ITEMS - 1)]);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Failed to scrape and analyze article URL.";
      setError(message);
    } finally {
      setIsLoading(false);
    }
  }, [inputUrl]);

  // Keyboard Command Shortcut: ⌘ + Enter or Ctrl + Enter
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
        if (inputMode === "text" && !isLoading && inputText.trim()) {
          e.preventDefault();
          handleAnalyze();
        } else if (inputMode === "url" && !isLoading && !isScraping && inputUrl.trim()) {
          e.preventDefault();
          handleAnalyzeUrl();
        }
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [inputMode, isLoading, isScraping, inputText, inputUrl, handleAnalyze, handleAnalyzeUrl]);

  const handleExtractTextOnly = async () => {
    let url = inputUrl.trim();
    if (!url) {
      setError("Please enter a valid article URL to extract.");
      return;
    }

    if (!url.startsWith("http://") && !url.startsWith("https://")) {
      url = "https://" + url;
    }

    const lowerUrl = url.toLowerCase();
    const blockedMedia = [
      "twitter.com",
      "x.com",
      "youtube.com",
      "youtu.be",
      "instagram.com",
      "tiktok.com",
      "facebook.com",
      "reddit.com",
    ];
    if (blockedMedia.some((d) => lowerUrl.includes(d))) {
      setError(
        "Social media and streaming platform links (e.g. YouTube, Twitter/X) cannot be scraped as news articles. Please copy and paste the text directly into the Claim / Text Input tab."
      );
      return;
    }

    setIsScraping(true);
    setError(null);

    try {
      const res = await fetch("/api/scrape", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => null);
        throw new Error(errData?.detail || `Extraction failed (${res.status}).`);
      }

      const data = await res.json();
      setInputText(data.text);
      setInputMode("text");
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Failed to extract article text.";
      setError(message);
    } finally {
      setIsScraping(false);
    }
  };

  const handleHistorySelect = (item: AnalysisHistoryItem) => {
    setResult(item.response);
    setError(null);
    if (item.response.article_meta?.url) {
      setInputUrl(item.response.article_meta.url);
      setInputMode("url");
    } else if (item.response.snippet) {
      setInputText(item.response.snippet);
      setInputMode("text");
    }
  };

  const handleClearHistory = () => {
    setHistory([]);
    try {
      if (typeof window !== "undefined") {
        localStorage.removeItem(STORAGE_KEY_HISTORY);
      }
    } catch (err) {
      console.warn("Failed to clear persistent history:", err);
    }
  };

  const handleClear = () => {
    setInputText("");
    setInputUrl("");
    setResult(null);
    setError(null);
  };

  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>
      <Header />

      <main
        style={{
          flex: 1,
          maxWidth: "1280px",
          width: "100%",
          margin: "0 auto",
          padding: "36px 24px 60px 24px",
        }}
      >
        {/* Hero Section */}
        <div style={{ textAlign: "center", marginBottom: "32px" }}>
          <div
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: "8px",
              padding: "5px 14px",
              borderRadius: "var(--radius-full)",
              background: "rgba(99, 102, 241, 0.08)",
              border: "1px solid rgba(99, 102, 241, 0.22)",
              marginBottom: "16px",
            }}
          >
            <Sparkles size={13} color="#60a5fa" />
            <span
              style={{
                fontSize: "0.76rem",
                fontWeight: 600,
                color: "#bfdbfe",
                letterSpacing: "0.04em",
                fontFamily: "var(--font-mono)",
              }}
            >
              MULTI-MODEL AI VERIFICATION
            </span>
          </div>

          <h1
            style={{
              fontSize: "clamp(2rem, 3.8vw, 3rem)",
              fontWeight: 800,
              lineHeight: 1.15,
              marginBottom: "14px",
              letterSpacing: "-0.03em",
              color: "#f8fafc",
            }}
          >
            Real-Time <span className="gradient-accent-text">News Detection</span> & Fact-Checking
          </h1>

          <p
            style={{
              maxWidth: "680px",
              margin: "0 auto",
              fontSize: "1rem",
              color: "var(--text-secondary)",
              lineHeight: 1.6,
            }}
          >
            Verify news claims and articles across 4 AI detection models
            with real-time web search and live fact-checking.
          </p>
        </div>

        {/* Input Panel Card */}
        <section
          className="glass-panel"
          style={{
            padding: "24px 28px",
            borderRadius: "var(--radius-xl)",
            background: "var(--card-bg)",
            border: "1px solid var(--card-border)",
            boxShadow: "var(--shadow-elevated)",
          }}
          aria-label="Article Input"
        >
          {/* Tab Switcher: Text Claim vs Article URL vs Batch Upload */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "8px",
              marginBottom: "20px",
              borderBottom: "1px solid rgba(255, 255, 255, 0.08)",
              paddingBottom: "12px",
            }}
          >
            <button
              onClick={() => { setInputMode("text"); setError(null); }}
              disabled={isLoading || isScraping}
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: "8px",
                padding: "8px 16px",
                borderRadius: "var(--radius-sm)",
                background: inputMode === "text" ? "rgba(37, 99, 235, 0.18)" : "transparent",
                border: inputMode === "text" ? "1px solid rgba(37, 99, 235, 0.45)" : "1px solid transparent",
                color: inputMode === "text" ? "#ffffff" : "var(--text-muted)",
                fontSize: "0.88rem",
                fontWeight: 600,
                cursor: "pointer",
                transition: "all 0.2s ease",
              }}
            >
              <FileText size={16} color={inputMode === "text" ? "#60a5fa" : "#64748b"} />
              <span>Text / Claim</span>
            </button>

            <button
              onClick={() => { setInputMode("url"); setError(null); }}
              disabled={isLoading || isScraping}
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: "8px",
                padding: "8px 16px",
                borderRadius: "var(--radius-sm)",
                background: inputMode === "url" ? "rgba(37, 99, 235, 0.18)" : "transparent",
                border: inputMode === "url" ? "1px solid rgba(37, 99, 235, 0.45)" : "1px solid transparent",
                color: inputMode === "url" ? "#ffffff" : "var(--text-muted)",
                fontSize: "0.88rem",
                fontWeight: 600,
                cursor: "pointer",
                transition: "all 0.2s ease",
              }}
            >
              <Globe size={16} color={inputMode === "url" ? "#60a5fa" : "#64748b"} />
              <span>Article Link</span>
            </button>

            <button
              onClick={() => { setInputMode("batch"); setError(null); }}
              disabled={isLoading || isScraping}
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: "8px",
                padding: "8px 16px",
                borderRadius: "var(--radius-sm)",
                background: inputMode === "batch" ? "rgba(37, 99, 235, 0.18)" : "transparent",
                border: inputMode === "batch" ? "1px solid rgba(37, 99, 235, 0.45)" : "1px solid transparent",
                color: inputMode === "batch" ? "#ffffff" : "var(--text-muted)",
                fontSize: "0.88rem",
                fontWeight: 600,
                cursor: "pointer",
                transition: "all 0.2s ease",
              }}
            >
              <Layers size={16} color={inputMode === "batch" ? "#60a5fa" : "#64748b"} />
              <span>Batch Upload</span>
            </button>
          </div>

          {inputMode === "text" ? (
            <>
              {/* Preset Sample Selector (Phase 5 Quick Test) */}
              <div style={{ marginBottom: "14px" }}>
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "8px",
                    marginBottom: "8px",
                    fontSize: "0.78rem",
                    fontWeight: 600,
                    color: "var(--text-muted)",
                  }}
                >
                  <Sparkles size={13} color="#60a5fa" />
                  <span>TRY SAMPLE EXAMPLES:</span>
                </div>
                <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
                  {SAMPLE_PRESETS.map((p) => (
                    <button
                      key={p.id}
                      onClick={() => {
                        setInputText(p.text);
                        handleAnalyze(p.text);
                      }}
                      disabled={isLoading}
                      className="btn-secondary"
                      style={{
                        fontSize: "0.78rem",
                        padding: "4px 10px",
                        borderLeft: p.category === "authentic" ? "3px solid #10b981" : "3px solid #f43f5e",
                      }}
                    >
                      <span
                        style={{
                          fontSize: "0.68rem",
                          fontWeight: 700,
                          padding: "1px 4px",
                          borderRadius: "3px",
                          background: "rgba(255,255,255,0.08)",
                        }}
                      >
                        {p.langTag}
                      </span>
                      <span>{p.title}</span>
                    </button>
                  ))}
                </div>
              </div>

              {/* Text Area with Scanner Beam */}
              <div className="neural-scanner-container" style={{ position: "relative" }}>
                {isLoading && <div className="neural-scanner-beam" />}
                <textarea
                  id="news-input-textarea"
                  aria-label="News text to verify"
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  placeholder="Paste any news article headline, body text, or claim in English, Hindi, Spanish, French, or 50+ languages..."
                  className="news-textarea"
                />
              </div>

              {/* Active Processing Indicator */}
              {isLoading && (
                <div
                  className="animate-pulse"
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "8px",
                    marginTop: "12px",
                    color: "#60a5fa",
                    fontSize: "0.85rem",
                    background: "rgba(37, 99, 235, 0.10)",
                    padding: "8px 14px",
                    borderRadius: "var(--radius-sm)",
                    border: "1px solid rgba(37, 99, 235, 0.25)",
                  }}
                >
                  <Loader2 size={15} className="spinner" />
                  <span>Analyzing claim with 4 AI models...</span>
                </div>
              )}

              {/* Metrics & Actions Bar */}
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  marginTop: "16px",
                  flexWrap: "wrap",
                  gap: "12px",
                }}
              >
                {/* Word & Char Counters */}
                <div style={{ display: "flex", alignItems: "center", gap: "14px", flexWrap: "wrap", fontSize: "0.82rem" }}>
                  <span style={{ color: "var(--text-muted)" }}>
                    Words: <strong style={{ color: "#ffffff" }}>{wordCount}</strong>
                  </span>
                  <span style={{ color: "var(--text-muted)" }}>
                    Chars: <strong style={{ color: "#ffffff" }}>{charCount}</strong>
                  </span>
                  {wordCount > 0 && (
                    <span style={{ color: "var(--text-muted)", fontFamily: "var(--font-mono)", fontSize: "0.76rem" }}>
                      {readingTimeLabel}
                    </span>
                  )}
                  {isInsufficientContext && (
                    <span
                      style={{
                        color: "#fbbf24",
                        fontSize: "0.78rem",
                        display: "flex",
                        alignItems: "center",
                        gap: "4px",
                        background: "rgba(217, 119, 6, 0.12)",
                        padding: "2px 8px",
                        borderRadius: "var(--radius-full)",
                        border: "1px solid rgba(217, 119, 6, 0.25)",
                      }}
                      title="News authenticity verification requires a complete claim or headline (minimum 3 words)"
                    >
                      <AlertCircle size={13} /> Minimum 3 Words Required
                    </span>
                  )}
                  {isShortText && (
                    <span
                      style={{
                        color: "#fbbf24",
                        fontSize: "0.78rem",
                        display: "flex",
                        alignItems: "center",
                        gap: "4px",
                        background: "rgba(217, 119, 6, 0.12)",
                        padding: "2px 8px",
                        borderRadius: "var(--radius-full)",
                        border: "1px solid rgba(217, 119, 6, 0.25)",
                      }}
                    >
                      <AlertCircle size={13} /> Short Claim (&lt;15 words)
                    </span>
                  )}
                  {isOptimalText && (
                    <span
                      style={{
                        color: "#34d399",
                        fontSize: "0.78rem",
                        display: "flex",
                        alignItems: "center",
                        gap: "4px",
                        background: "rgba(5, 150, 105, 0.10)",
                        padding: "2px 8px",
                        borderRadius: "var(--radius-full)",
                        border: "1px solid rgba(5, 150, 105, 0.25)",
                      }}
                    >
                      <CheckCircle2 size={13} /> Good Text Length
                    </span>
                  )}
                  {wordCount > 500 && (
                    <span
                      style={{
                        color: "#60a5fa",
                        fontSize: "0.78rem",
                        display: "flex",
                        alignItems: "center",
                        gap: "4px",
                        background: "rgba(37, 99, 235, 0.10)",
                        padding: "2px 8px",
                        borderRadius: "var(--radius-full)",
                        border: "1px solid rgba(37, 99, 235, 0.25)",
                      }}
                    >
                      <Layers size={13} /> Long Article (Trimmed for Analysis)
                    </span>
                  )}
                </div>

                {/* CTA Buttons */}
                <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                  {inputText && (
                    <button
                      onClick={handleClear}
                      disabled={isLoading}
                      className="btn-secondary"
                      title="Clear text"
                      id="clear-input-button"
                    >
                      <Trash2 size={16} />
                      <span>Clear</span>
                    </button>
                  )}

                  <button
                    onClick={() => handleAnalyze()}
                    disabled={isLoading || !inputText.trim() || isInsufficientContext}
                    className="btn-primary btn-shimmer"
                    id="analyze-button"
                    title={isInsufficientContext ? "Please enter at least 3 words to analyze news veracity" : undefined}
                  >
                    {isLoading ? (
                      <>
                        <Loader2 size={18} className="spinner" />
                        <span>Analyzing Text...</span>
                      </>
                    ) : (
                      <>
                        <Sparkles size={18} />
                        <span>Analyze News</span>
                        <span
                          style={{
                            marginLeft: "6px",
                            fontSize: "0.68rem",
                            opacity: 0.85,
                            padding: "2px 6px",
                            borderRadius: "4px",
                            background: "rgba(255, 255, 255, 0.18)",
                            fontFamily: "var(--font-mono)",
                            border: "1px solid rgba(255, 255, 255, 0.25)",
                            lineHeight: 1,
                          }}
                        >
                          ⌘↵
                        </span>
                      </>
                    )}
                  </button>
                </div>
              </div>
            </>
          ) : inputMode === "url" ? (
            <div>
              {/* URL Presets */}
              <div style={{ marginBottom: "18px" }}>
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "8px",
                    marginBottom: "10px",
                    fontSize: "0.8rem",
                    fontWeight: 600,
                    color: "var(--text-muted)",
                  }}
                >
                  <Globe size={14} color="#60a5fa" />
                  <span>TRY SAMPLE NEWS LINKS:</span>
                </div>
                <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
                  {SAMPLE_URL_PRESETS.map((p, idx) => (
                    <button
                      key={idx}
                      onClick={() => {
                        setInputUrl(p.url);
                        handleAnalyzeUrl(p.url);
                      }}
                      disabled={isLoading || isScraping}
                      className="btn-secondary"
                      style={{ borderLeft: "3px solid #60a5fa" }}
                    >
                      <span>{p.title}</span>
                    </button>
                  ))}
                </div>
              </div>

              {/* URL Input Bar */}
              <div
                className="neural-scanner-container"
                style={{ position: "relative", marginBottom: "16px", borderRadius: "var(--radius-md)" }}
              >
                {(isLoading || isScraping) && <div className="neural-scanner-beam" />}
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "10px",
                    background: "rgba(16, 24, 44, 0.75)",
                    border: "1px solid rgba(255, 255, 255, 0.12)",
                    borderRadius: "var(--radius-md)",
                    padding: "6px 14px",
                  }}
                >
                  <Link2 size={18} color="#60a5fa" />
                  <input
                    type="url"
                    id="article-url-input"
                    aria-label="Article URL to analyze"
                    value={inputUrl}
                    onChange={(e) => setInputUrl(e.target.value)}
                    placeholder="Paste any news article URL (e.g. https://www.bbc.com/news/..., https://reuters.com/...)"
                    style={{
                      flex: 1,
                      background: "transparent",
                      border: "none",
                      outline: "none",
                      color: "#f8fafc",
                      fontSize: "0.95rem",
                      fontFamily: "var(--font-body)",
                      padding: "8px 0",
                    }}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") {
                        handleAnalyzeUrl();
                      }
                    }}
                  />
                  {inputUrl && (
                    <button
                      onClick={() => setInputUrl("")}
                      style={{
                        background: "transparent",
                        border: "none",
                        color: "var(--text-muted)",
                        cursor: "pointer",
                        padding: "4px",
                      }}
                      title="Clear URL"
                    >
                      <Trash2 size={16} />
                    </button>
                  )}
                </div>
              </div>

              {/* Actions */}
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  flexWrap: "wrap",
                  gap: "12px",
                }}
              >
                <div style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
                  Secure link reader • Removes web ads & clutter • Extracts article title & author
                </div>

                <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                  <button
                    onClick={handleExtractTextOnly}
                    disabled={isLoading || isScraping || !inputUrl.trim()}
                    className="btn-secondary"
                    title="Extract article body text and load into editor"
                  >
                    {isScraping ? (
                      <>
                        <Loader2 size={16} className="spinner" />
                        <span>Extracting Article...</span>
                      </>
                    ) : (
                      <>
                        <DownloadCloud size={16} />
                        <span>Extract Text Only</span>
                      </>
                    )}
                  </button>

                  <button
                    onClick={() => handleAnalyzeUrl()}
                    disabled={isLoading || isScraping || !inputUrl.trim()}
                    className="btn-primary btn-shimmer"
                    id="analyze-url-button"
                  >
                    {isLoading ? (
                      <>
                        <Loader2 size={18} className="spinner" />
                        <span>Fetching & Analyzing...</span>
                      </>
                    ) : (
                      <>
                        <Sparkles size={18} />
                        <span>Fetch & Analyze Article</span>
                        <span
                          style={{
                            marginLeft: "6px",
                            fontSize: "0.68rem",
                            opacity: 0.85,
                            padding: "2px 6px",
                            borderRadius: "4px",
                            background: "rgba(255, 255, 255, 0.18)",
                            fontFamily: "var(--font-mono)",
                            border: "1px solid rgba(255, 255, 255, 0.25)",
                            lineHeight: 1,
                          }}
                        >
                          ⌘↵
                        </span>
                      </>
                    )}
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <BatchIngestionView />
          )}

          {/* Error Message */}
          {error && (
            <div
              className="animate-scale-in"
              style={{
                marginTop: "16px",
                padding: "12px 18px",
                borderRadius: "var(--radius-sm)",
                background: "rgba(244, 63, 94, 0.12)",
                border: "1px solid rgba(244, 63, 94, 0.3)",
                color: "#fb7185",
                fontSize: "0.88rem",
                display: "flex",
                alignItems: "center",
                gap: "10px",
              }}
            >
              <AlertCircle size={18} />
              <span>{error}</span>
            </div>
          )}
        </section>

        {/* Results Panel (for single-item analysis) */}
        {inputMode !== "batch" && result && (
          <section
            style={{ marginTop: "36px" }}
            aria-label="Ensemble Analysis Results"
            id="results-section"
          >
            {/* Quick Actions & Forensic Telemetry Toolbar */}
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                flexWrap: "wrap",
                gap: "12px",
                marginBottom: "20px",
                padding: "10px 18px",
                borderRadius: "var(--radius-md)",
                background: "rgba(16, 24, 44, 0.75)",
                border: "1px solid rgba(255, 255, 255, 0.08)",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: "10px", flexWrap: "wrap" }}>
                <span
                  style={{
                    fontFamily: "var(--font-mono)",
                    fontSize: "0.75rem",
                    fontWeight: 700,
                    letterSpacing: "0.05em",
                    color: "var(--text-muted)",
                  }}
                >
                  ANALYSIS STATS:
                </span>
                <span
                  style={{
                    fontFamily: "var(--font-mono)",
                    fontSize: "0.78rem",
                    color: "#60a5fa",
                    background: "rgba(37, 99, 235, 0.12)",
                    padding: "2px 8px",
                    borderRadius: "4px",
                    border: "1px solid rgba(37, 99, 235, 0.3)",
                  }}
                >
                  {result.execution_time_ms ? `${result.execution_time_ms.toFixed(1)}ms` : "Instant Cached (<0.1ms)"}
                </span>
                {result.cached && (
                  <span
                    style={{
                      fontFamily: "var(--font-mono)",
                      fontSize: "0.72rem",
                      color: "#34d399",
                      background: "rgba(5, 150, 105, 0.12)",
                      padding: "2px 8px",
                      borderRadius: "4px",
                      border: "1px solid rgba(5, 150, 105, 0.3)",
                    }}
                  >
                    INSTANT CACHE
                  </span>
                )}
                {result.claim_sha256 && (
                  <span
                    onClick={() => {
                      if (result.claim_sha256) {
                        navigator.clipboard.writeText(result.claim_sha256);
                        setCopiedToast(true);
                        setTimeout(() => setCopiedToast(false), 2000);
                      }
                    }}
                    style={{
                      fontFamily: "var(--font-mono)",
                      fontSize: "0.72rem",
                      color: "#94a3b8",
                      background: "rgba(255, 255, 255, 0.04)",
                      padding: "2px 8px",
                      borderRadius: "4px",
                      border: "1px solid rgba(255, 255, 255, 0.08)",
                      cursor: "pointer",
                      display: "inline-flex",
                      alignItems: "center",
                      gap: "4px",
                    }}
                    title={`Deterministic Claim SHA-256: ${result.claim_sha256} (Click to copy)`}
                  >
                    SHA256: <strong style={{ color: "#cbd5e1" }}>{result.claim_sha256.substring(0, 10)}...</strong>
                  </span>
                )}
                {result.is_polarized && (
                  <span
                    style={{
                      fontFamily: "var(--font-mono)",
                      fontSize: "0.72rem",
                      color: "#fbbf24",
                      background: "rgba(217, 119, 6, 0.12)",
                      padding: "2px 8px",
                      borderRadius: "4px",
                      border: "1px solid rgba(217, 119, 6, 0.3)",
                    }}
                    title="Consensus probability falls inside 45%–55% equivocal neutral zone"
                  >
                    SPLIT VERDICT (45%–55%)
                  </span>
                )}
              </div>

              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <button
                  onClick={handleCopySummary}
                  className="btn-secondary"
                  style={{
                    fontSize: "0.78rem",
                    padding: "6px 12px",
                    display: "inline-flex",
                    alignItems: "center",
                    gap: "6px",
                  }}
                  title="Copy analysis summary to clipboard"
                  id="copy-summary-button"
                >
                  {copiedToast ? (
                    <>
                      <Check size={14} color="#34d399" />
                      <span style={{ color: "#34d399", fontWeight: 600 }}>Report Copied!</span>
                    </>
                  ) : (
                    <>
                      <Copy size={14} color="#60a5fa" />
                      <span>Copy Summary Report</span>
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* Scraped Article Hero Banner */}
            <div className="animate-fade-in-up stagger-1">
              <ArticleMetaHeader meta={result.article_meta} />
            </div>

            {/* Top row: Gauge & High-level Verdict */}
            <div className="animate-fade-in-up stagger-2">
              <GaugeMeter data={result} />
            </div>

            {/* Model Breakdown */}
            <div className="animate-fade-in-up stagger-3">
              <ModelGrid data={result} />
            </div>

            {/* Real-Time Fact-Checking & Web Grounding */}
            <div className="animate-fade-in-up stagger-4">
              <FactCheckEvidence data={result.fact_check} />
            </div>

            {/* Decision Explanation */}
            <div className="animate-fade-in-up stagger-5">
              <ExplainableAI data={result} />
            </div>
          </section>
        )}

        {/* Session History (Lazily Loaded) */}
        <HistoryDrawer
          history={history}
          onSelect={handleHistorySelect}
          onClear={handleClearHistory}
        />
      </main>

      {/* Footer */}
      <footer
        style={{
          borderTop: "1px solid rgba(255, 255, 255, 0.08)",
          padding: "24px 32px",
          textAlign: "center",
          color: "var(--text-muted)",
          fontSize: "0.82rem",
          background: "rgba(6, 9, 19, 0.6)",
        }}
      >
        <p>
          News Detection Multi-Model Ensemble • RoBERTa (40%) • Distilled TinyBERT (25%) • BERT-Detect
          (35%)
        </p>
        <p style={{ marginTop: "6px", fontSize: "0.75rem" }}>
          High-Speed AI Verification System • Supports 50+ Global Languages
        </p>
      </footer>
    </div>
  );
}
