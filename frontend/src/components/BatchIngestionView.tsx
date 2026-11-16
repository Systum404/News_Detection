"use client";

import React, { useState, useEffect, useRef } from "react";
import {
  UploadCloud,
  CheckCircle2,
  AlertTriangle,
  Download,
  RefreshCw,
  Search,
  Layers,
  ChevronDown,
  ChevronRight,
  Sparkles,
  BarChart3,
  FileCode2,
} from "lucide-react";

interface BatchItemResult {
  id: string;
  text_snippet: string;
  verdict: "REAL" | "FAKE" | "UNCONFIRMED" | "ERROR" | string;
  confidence: number;
  claim_sha256?: string;
  is_polarized?: boolean;
  polarization_index?: number;
  category?: string;
  category_confidence?: number;
  is_fake: boolean;
  language: string;
  language_name: string;
  agreement: string;
  models: {
    roberta: { vote: string; score: string };
    tinybert: { vote: string; score: string };
    bert_detect: { vote: string; score: string };
    xlm_roberta: { vote: string; score: string };
  };
  metadata: Record<string, unknown>;
}

interface BatchJobSummary {
  total: number;
  real_count: number;
  fake_count: number;
  unconfirmed_count?: number;
  real_pct: number;
  fake_pct: number;
  unconfirmed_pct?: number;
  avg_confidence: number;
  languages: Record<string, number>;
  categories?: Record<string, number>;
  duration_seconds?: number;
}

interface BatchJobData {
  job_id: string;
  filename: string;
  status: "queued" | "processing" | "completed" | "failed";
  created_at: string;
  completed_at: string | null;
  total_items: number;
  processed_items: number;
  progress_percent: number;
  error?: string | null;
  metadata?: {
    detected_text_column?: string;
    total_extracted?: number;
  };
  summary: BatchJobSummary;
  results?: BatchItemResult[];
}

export function BatchIngestionView() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [customTextCol, setCustomTextCol] = useState<string>("");
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [activeJob, setActiveJob] = useState<BatchJobData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState<string>("");
  const [verdictFilter, setVerdictFilter] = useState<"ALL" | "REAL" | "FAKE" | "UNCONFIRMED">("ALL");
  const [expandedRowId, setExpandedRowId] = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState<boolean>(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Clean up polling interval on unmount
  useEffect(() => {
    return () => {
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
    };
  }, []);

  // Poll active job status while running
  useEffect(() => {
    const jobId = activeJob?.job_id;
    const currentStatus = activeJob?.status;

    if (!jobId || currentStatus === "completed" || currentStatus === "failed") {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
      }
      return;
    }

    if (!pollIntervalRef.current) {
      pollIntervalRef.current = setInterval(async () => {
        try {
          const res = await fetch(`/api/batch/jobs/${jobId}`);
          if (!res.ok) return;
          const data: BatchJobData = await res.json();
          setActiveJob(data);

          if (data.status === "completed" || data.status === "failed") {
            if (pollIntervalRef.current) {
              clearInterval(pollIntervalRef.current);
              pollIntervalRef.current = null;
            }
          }
        } catch {
          // Keep polling silently
        }
      }, 1000);
    }

    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
      }
    };
  }, [activeJob?.job_id, activeJob?.status]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
      setError(null);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setSelectedFile(e.dataTransfer.files[0]);
      setError(null);
    }
  };

  const handleUploadAndRun = async () => {
    if (!selectedFile) {
      setError("Please select a file to ingest (.csv, .tsv, .json, .jsonl, .txt, or .pdf).");
      return;
    }

    setIsUploading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", selectedFile);
    if (customTextCol.trim()) {
      formData.append("text_column", customTextCol.trim());
    }

    try {
      const res = await fetch("/api/batch/upload", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.detail || `Server returned HTTP ${res.status}`);
      }

      const initialData: BatchJobData = await res.json();
      setActiveJob(initialData);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Failed to upload and start batch processing.";
      setError(message);
    } finally {
      setIsUploading(false);
    }
  };

  const handleLoadSampleCSV = () => {
    const sampleCsvContent = `id,headline,source,author
101,World Health Organization deploys new global emergency disease surveillance protocol,Reuters,Health Desk
102,Secret microchips embedded in tap water to control human brainwaves,DarkWebLeaks,Anon99
103,NASA James Webb Space Telescope discovers atmospheric carbon dioxide on habitable exoplanet,AP Wire,Science Team
104,Drinking silver nitrate protects against 5G radio frequencies and cancels all viruses,MiracleHealer,Dr. Truth
105,El banco central anuncia medidas de liquidez para estabilizar el tipo de cambio oficial,EFE,Economía
106,Le ministre de la santé confirme le déploiement national du plan canicule renforcé,AFP,Actualités`;

    const blob = new Blob([sampleCsvContent], { type: "text/csv" });
    const file = new File([blob], "sample_enterprise_news_batch.csv", { type: "text/csv" });
    setSelectedFile(file);
    setError(null);
  };

  const handleExport = (format: "csv" | "json") => {
    if (!activeJob) return;
    window.open(`/api/batch/jobs/${activeJob.job_id}/export?format=${format}`, "_blank");
  };

  const handleReset = () => {
    setSelectedFile(null);
    setActiveJob(null);
    setError(null);
    setSearchTerm("");
    setVerdictFilter("ALL");
    setExpandedRowId(null);
  };

  // Filtered Results
  const results = activeJob?.results || [];
  const filteredResults = results.filter((item) => {
    const matchesVerdict =
      verdictFilter === "ALL" ? true : item.verdict === verdictFilter;
    const matchesSearch =
      !searchTerm.trim() ||
      item.text_snippet.toLowerCase().includes(searchTerm.toLowerCase()) ||
      item.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
      item.language_name.toLowerCase().includes(searchTerm.toLowerCase());
    return matchesVerdict && matchesSearch;
  });

  return (
    <div>
      {/* Header Info Banner */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "14px 18px",
          background: "rgba(99, 102, 241, 0.08)",
          border: "1px solid rgba(99, 102, 241, 0.22)",
          borderRadius: "var(--radius-md)",
          marginBottom: "20px",
          flexWrap: "wrap",
          gap: "10px",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <div
            style={{
              width: "32px",
              height: "32px",
              borderRadius: "8px",
              background: "rgba(37, 99, 235, 0.20)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <Layers size={18} color="#60a5fa" />
          </div>
          <div>
            <div style={{ fontSize: "0.92rem", fontWeight: 700, color: "#ffffff" }}>
              Enterprise Batch Document Ingestion
            </div>
            <div style={{ fontSize: "0.78rem", color: "var(--text-muted)" }}>
              Ingest CSV, TSV, JSON, JSONL, TXT, or PDF documents for parallel multi-model classification.
            </div>
          </div>
        </div>

        <button
          onClick={handleLoadSampleCSV}
          disabled={isUploading || (activeJob !== null && activeJob.status === "processing")}
          className="btn-secondary"
          style={{ fontSize: "0.8rem", padding: "6px 12px" }}
          title="Load 6 sample articles in English, Spanish, and French"
        >
          <Sparkles size={14} color="#60a5fa" />
          <span>Load Sample Batch CSV</span>
        </button>
      </div>

      {error && (
        <div
          style={{
            padding: "12px 16px",
            background: "rgba(225, 29, 72, 0.12)",
            border: "1px solid rgba(225, 29, 72, 0.3)",
            borderRadius: "var(--radius-sm)",
            color: "#fb7185",
            fontSize: "0.85rem",
            display: "flex",
            alignItems: "center",
            gap: "10px",
            marginBottom: "16px",
          }}
        >
          <AlertTriangle size={16} />
          <span>{error}</span>
        </div>
      )}

      {/* Upload Dropzone (shown when no completed or processing job) */}
      {!activeJob && (
        <div>
          <div
            onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
            onDragLeave={() => setIsDragOver(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            style={{
              border: `2px dashed ${isDragOver ? "#60a5fa" : "rgba(255, 255, 255, 0.15)"}`,
              borderRadius: "var(--radius-lg)",
              padding: "36px 20px",
              textAlign: "center",
              background: isDragOver ? "rgba(37, 99, 235, 0.12)" : "rgba(11, 15, 25, 0.5)",
              cursor: "pointer",
              transition: "all 0.2s ease",
              marginBottom: "16px",
            }}
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept=".csv,.tsv,.json,.jsonl,.txt,.pdf"
              style={{ display: "none" }}
            />

            <div
              style={{
                width: "48px",
                height: "48px",
                borderRadius: "50%",
                background: "rgba(37, 99, 235, 0.15)",
                display: "inline-flex",
                alignItems: "center",
                justifyContent: "center",
                marginBottom: "12px",
              }}
            >
              <UploadCloud size={24} color="#60a5fa" />
            </div>

            <div style={{ fontSize: "1rem", fontWeight: 600, color: "#ffffff", marginBottom: "6px" }}>
              {selectedFile ? selectedFile.name : "Drag & Drop document or click to browse"}
            </div>

            <div style={{ fontSize: "0.82rem", color: "var(--text-muted)" }}>
              Supports <strong>.csv, .tsv, .json, .jsonl, .txt, .pdf</strong> (Up to 1,000 articles per batch)
            </div>

            {selectedFile && (
              <div
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  gap: "8px",
                  marginTop: "12px",
                  padding: "4px 12px",
                  background: "rgba(16, 185, 129, 0.15)",
                  border: "1px solid rgba(16, 185, 129, 0.3)",
                  borderRadius: "20px",
                  color: "#34d399",
                  fontSize: "0.8rem",
                }}
              >
                <CheckCircle2 size={14} />
                <span>Selected: {selectedFile.name} ({(selectedFile.size / 1024).toFixed(1)} KB)</span>
              </div>
            )}
          </div>

          {/* Optional Text Column Input */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "12px",
              marginBottom: "20px",
              flexWrap: "wrap",
            }}
          >
            <div style={{ flex: 1, minWidth: "240px" }}>
              <label
                style={{
                  display: "block",
                  fontSize: "0.78rem",
                  color: "var(--text-muted)",
                  marginBottom: "6px",
                }}
              >
                Custom Text Column Header (Optional for CSV):
              </label>
              <input
                type="text"
                value={customTextCol}
                onChange={(e) => setCustomTextCol(e.target.value)}
                placeholder="e.g. headline, claim, text (Leave blank for auto-detection)"
                style={{
                  width: "100%",
                  background: "rgba(16, 24, 44, 0.75)",
                  border: "1px solid rgba(255, 255, 255, 0.12)",
                  borderRadius: "var(--radius-sm)",
                  padding: "8px 12px",
                  color: "#ffffff",
                  fontSize: "0.85rem",
                  outline: "none",
                }}
              />
            </div>

            <div style={{ paddingTop: "20px" }}>
              <button
                onClick={handleUploadAndRun}
                disabled={!selectedFile || isUploading}
                className="btn-primary"
                style={{
                  padding: "10px 22px",
                  display: "inline-flex",
                  alignItems: "center",
                  gap: "8px",
                  opacity: !selectedFile || isUploading ? 0.6 : 1,
                  cursor: !selectedFile || isUploading ? "not-allowed" : "pointer",
                }}
              >
                {isUploading ? (
                  <>
                    <RefreshCw size={16} className="animate-spin" />
                    <span>Uploading...</span>
                  </>
                ) : (
                  <>
                    <BarChart3 size={16} />
                    <span>Start Batch Analysis</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Progress & Live Status State */}
      {activeJob && activeJob.status !== "completed" && activeJob.status !== "failed" && (
        <div
          style={{
            padding: "24px",
            background: "rgba(16, 24, 44, 0.75)",
            border: "1px solid rgba(99, 102, 241, 0.3)",
            borderRadius: "var(--radius-md)",
            textAlign: "center",
            marginBottom: "20px",
          }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              gap: "10px",
              marginBottom: "12px",
            }}
          >
            <RefreshCw size={20} color="#60a5fa" className="animate-spin" />
            <span style={{ fontSize: "1.05rem", fontWeight: 700, color: "#ffffff" }}>
              Processing Batch: {activeJob.filename}
            </span>
          </div>

          <div style={{ fontSize: "0.85rem", color: "var(--text-muted)", marginBottom: "16px" }}>
            Analyzed <strong>{activeJob.processed_items}</strong> of <strong>{activeJob.total_items}</strong> documents ({activeJob.progress_percent}%)
          </div>

          {/* Progress Bar with Animated Sweep Stripes */}
          <div
            style={{
              width: "100%",
              height: "10px",
              background: "rgba(255, 255, 255, 0.08)",
              borderRadius: "5px",
              overflow: "hidden",
              marginBottom: "12px",
            }}
          >
            <div
              style={{
                width: `${activeJob.progress_percent}%`,
                height: "100%",
                background: "linear-gradient(90deg, #1d4ed8, #60a5fa)",
                backgroundImage:
                  "linear-gradient(45deg, rgba(255, 255, 255, 0.18) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.18) 50%, rgba(255, 255, 255, 0.18) 75%, transparent 75%, transparent), linear-gradient(90deg, #1d4ed8, #60a5fa)",
                backgroundSize: "24px 24px, 100% 100%",
                animation: "progressStripes 1.2s linear infinite",
                transition: "width 0.4s cubic-bezier(0.16, 1, 0.3, 1)",
                borderRadius: "5px",
              }}
            />
          </div>

          <div style={{ fontSize: "0.78rem", color: "#64748b" }}>
            Hardware Acceleration: Active (Fast Parallel Processing)
          </div>
        </div>
      )}

      {/* Completed Batch View */}
      {activeJob && activeJob.status === "completed" && (
        <div>
          {/* Executive Summary Cards */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))",
              gap: "12px",
              marginBottom: "20px",
            }}
          >
            <div
              className="interactive-card animate-fade-in-up stagger-1"
              style={{
                padding: "14px",
                background: "rgba(16, 24, 44, 0.75)",
                border: "1px solid rgba(255, 255, 255, 0.08)",
                borderRadius: "var(--radius-md)",
              }}
            >
              <div style={{ fontSize: "0.76rem", color: "var(--text-muted)", marginBottom: "4px" }}>
                TOTAL ANALYZED
              </div>
              <div style={{ fontSize: "1.4rem", fontWeight: 800, color: "#ffffff" }}>
                {activeJob.summary.total}
              </div>
              <div style={{ fontSize: "0.72rem", color: "#64748b" }}>
                {activeJob.summary.duration_seconds}s execution time
              </div>
            </div>

            <div
              className="interactive-card animate-fade-in-up stagger-2"
              style={{
                padding: "14px",
                background: "rgba(16, 185, 129, 0.08)",
                border: "1px solid rgba(16, 185, 129, 0.25)",
                borderRadius: "var(--radius-md)",
              }}
            >
              <div style={{ fontSize: "0.76rem", color: "#34d399", marginBottom: "4px" }}>
                AUTHENTIC NEWS
              </div>
              <div style={{ fontSize: "1.4rem", fontWeight: 800, color: "#10b981" }}>
                {activeJob.summary.real_count}{" "}
                <span style={{ fontSize: "0.85rem", fontWeight: 500 }}>
                  ({activeJob.summary.real_pct}%)
                </span>
              </div>
              <div style={{ fontSize: "0.72rem", color: "#34d399" }}>
                Verified genuine claims
              </div>
            </div>

            <div
              className="interactive-card animate-fade-in-up stagger-3"
              style={{
                padding: "14px",
                background: "rgba(225, 29, 72, 0.08)",
                border: "1px solid rgba(225, 29, 72, 0.25)",
                borderRadius: "var(--radius-md)",
              }}
            >
              <div style={{ fontSize: "0.76rem", color: "#fb7185", marginBottom: "4px" }}>
                FAKE NEWS DETECTED
              </div>
              <div style={{ fontSize: "1.4rem", fontWeight: 800, color: "#f43f5e" }}>
                {activeJob.summary.fake_count}{" "}
                <span style={{ fontSize: "0.85rem", fontWeight: 500 }}>
                  ({activeJob.summary.fake_pct}%)
                </span>
              </div>
              <div style={{ fontSize: "0.72rem", color: "#fb7185" }}>
                Disinformation flagged
              </div>
            </div>

            {Boolean(activeJob.summary.unconfirmed_count) && (
              <div
                className="interactive-card animate-fade-in-up stagger-3"
                style={{
                  padding: "14px",
                  background: "rgba(217, 119, 6, 0.08)",
                  border: "1px solid rgba(217, 119, 6, 0.25)",
                  borderRadius: "var(--radius-md)",
                }}
              >
                <div style={{ fontSize: "0.76rem", color: "#fbbf24", marginBottom: "4px" }}>
                  UNCONFIRMED / SPLIT
                </div>
                <div style={{ fontSize: "1.4rem", fontWeight: 800, color: "#fbbf24" }}>
                  {activeJob.summary.unconfirmed_count}{" "}
                  <span style={{ fontSize: "0.85rem", fontWeight: 500 }}>
                    ({activeJob.summary.unconfirmed_pct}%)
                  </span>
                </div>
                <div style={{ fontSize: "0.72rem", color: "#fbbf24" }}>
                  Borderline 45%–55% split
                </div>
              </div>
            )}

            <div
              style={{
                padding: "14px",
                background: "rgba(37, 99, 235, 0.08)",
                border: "1px solid rgba(37, 99, 235, 0.25)",
                borderRadius: "var(--radius-md)",
              }}
            >
              <div style={{ fontSize: "0.76rem", color: "#60a5fa", marginBottom: "4px" }}>
                AVG CONFIDENCE
              </div>
              <div style={{ fontSize: "1.4rem", fontWeight: 800, color: "#ffffff" }}>
                {activeJob.summary.avg_confidence}%
              </div>
              <div style={{ fontSize: "0.72rem", color: "#60a5fa" }}>
                Average model confidence
              </div>
            </div>
          </div>

          {/* Action & Filter Bar */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              marginBottom: "16px",
              flexWrap: "wrap",
              gap: "12px",
            }}
          >
            {/* Filter Pills */}
            <div style={{ display: "flex", gap: "6px", alignItems: "center" }}>
              <button
                onClick={() => setVerdictFilter("ALL")}
                className="btn-secondary"
                style={{
                  padding: "5px 12px",
                  fontSize: "0.8rem",
                  background: verdictFilter === "ALL" ? "rgba(37, 99, 235, 0.22)" : "transparent",
                  borderColor: verdictFilter === "ALL" ? "#60a5fa" : "rgba(255, 255, 255, 0.12)",
                }}
              >
                All ({activeJob.summary.total})
              </button>
              <button
                onClick={() => setVerdictFilter("REAL")}
                className="btn-secondary"
                style={{
                  padding: "5px 12px",
                  fontSize: "0.8rem",
                  background: verdictFilter === "REAL" ? "rgba(16, 185, 129, 0.25)" : "transparent",
                  borderColor: verdictFilter === "REAL" ? "#10b981" : "rgba(255, 255, 255, 0.12)",
                  color: "#34d399",
                }}
              >
                Authentic ({activeJob.summary.real_count})
              </button>
              <button
                onClick={() => setVerdictFilter("FAKE")}
                className="btn-secondary"
                style={{
                  padding: "5px 12px",
                  fontSize: "0.8rem",
                  background: verdictFilter === "FAKE" ? "rgba(220, 38, 38, 0.25)" : "transparent",
                  borderColor: verdictFilter === "FAKE" ? "#f43f5e" : "rgba(255, 255, 255, 0.12)",
                  color: "#fb7185",
                }}
              >
                Fake News ({activeJob.summary.fake_count})
              </button>
              {Boolean(activeJob.summary.unconfirmed_count) && (
                <button
                  onClick={() => setVerdictFilter("UNCONFIRMED")}
                  className="btn-secondary"
                  style={{
                    padding: "5px 12px",
                    fontSize: "0.8rem",
                    background: verdictFilter === "UNCONFIRMED" ? "rgba(217, 119, 6, 0.25)" : "transparent",
                    borderColor: verdictFilter === "UNCONFIRMED" ? "#fbbf24" : "rgba(255, 255, 255, 0.12)",
                    color: "#fbbf24",
                  }}
                >
                  Unconfirmed ({activeJob.summary.unconfirmed_count})
                </button>
              )}
            </div>

            {/* Export & Reset Actions */}
            <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
              <button
                onClick={() => handleExport("csv")}
                className="btn-secondary"
                style={{ fontSize: "0.8rem", padding: "6px 12px" }}
                title="Download enriched CSV with original metadata and neural votes"
              >
                <Download size={14} color="#60a5fa" />
                <span>Export CSV</span>
              </button>

              <button
                onClick={() => handleExport("json")}
                className="btn-secondary"
                style={{ fontSize: "0.8rem", padding: "6px 12px" }}
                title="Download full JSON audit report"
              >
                <FileCode2 size={14} color="#60a5fa" />
                <span>Export JSON</span>
              </button>

              <button
                onClick={handleReset}
                className="btn-secondary"
                style={{ fontSize: "0.8rem", padding: "6px 12px" }}
              >
                <span>New Batch</span>
              </button>
            </div>
          </div>

          {/* Search Box */}
          <div style={{ position: "relative", marginBottom: "14px" }}>
            <Search
              size={16}
              color="var(--text-muted)"
              style={{ position: "absolute", left: "12px", top: "50%", transform: "translateY(-50%)" }}
            />
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Search articles by text, ID, or language..."
              style={{
                width: "100%",
                background: "rgba(16, 24, 44, 0.65)",
                border: "1px solid rgba(255, 255, 255, 0.12)",
                borderRadius: "var(--radius-sm)",
                padding: "8px 12px 8px 36px",
                color: "#ffffff",
                fontSize: "0.85rem",
                outline: "none",
              }}
            />
          </div>

          {/* Table of Results */}
          <div
            style={{
              overflowX: "auto",
              border: "1px solid rgba(255, 255, 255, 0.08)",
              borderRadius: "var(--radius-md)",
              background: "rgba(11, 15, 25, 0.6)",
            }}
          >
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.82rem" }}>
              <thead>
                <tr
                  style={{
                    borderBottom: "1px solid rgba(255, 255, 255, 0.1)",
                    background: "rgba(16, 24, 44, 0.9)",
                    textAlign: "left",
                  }}
                >
                  <th style={{ padding: "10px 14px", color: "var(--text-muted)", width: "60px" }}>ID</th>
                  <th style={{ padding: "10px 14px", color: "var(--text-muted)" }}>Article Headline / Text Claim</th>
                  <th style={{ padding: "10px 14px", color: "var(--text-muted)", width: "100px" }}>Lang</th>
                  <th style={{ padding: "10px 14px", color: "var(--text-muted)", width: "120px" }}>Verdict</th>
                  <th style={{ padding: "10px 14px", color: "var(--text-muted)", width: "110px" }}>Confidence</th>
                  <th style={{ padding: "10px 14px", color: "var(--text-muted)", width: "80px" }}>Consensus</th>
                  <th style={{ padding: "10px 14px", color: "var(--text-muted)", width: "40px" }}></th>
                </tr>
              </thead>
              <tbody>
                {filteredResults.length === 0 ? (
                  <tr>
                    <td colSpan={7} style={{ padding: "30px", textAlign: "center", color: "var(--text-muted)" }}>
                      No articles match the active filter or search query.
                    </td>
                  </tr>
                ) : (
                  filteredResults.map((item) => {
                    const isExpanded = expandedRowId === item.id;
                    const isFake = item.verdict === "FAKE";
                    return (
                      <React.Fragment key={item.id}>
                        <tr
                          onClick={() => setExpandedRowId(isExpanded ? null : item.id)}
                          style={{
                            borderBottom: "1px solid rgba(255, 255, 255, 0.05)",
                            cursor: "pointer",
                            background: isExpanded ? "rgba(99, 102, 241, 0.08)" : "transparent",
                            transition: "background 0.15s ease",
                          }}
                        >
                          <td style={{ padding: "10px 14px", color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
                            {item.id}
                          </td>
                          <td style={{ padding: "10px 14px", color: "#f8fafc", fontWeight: 500 }}>
                            <div>{item.text_snippet}</div>
                            {item.category && (
                              <div style={{ marginTop: "4px" }}>
                                <span
                                  style={{
                                    fontSize: "0.68rem",
                                    padding: "2px 7px",
                                    borderRadius: "4px",
                                    background: "rgba(99, 102, 241, 0.12)",
                                    color: "#a5b4fc",
                                    border: "1px solid rgba(99, 102, 241, 0.25)",
                                    fontWeight: 600,
                                  }}
                                >
                                  🏷️ {item.category} ({item.category_confidence?.toFixed(1) || 0}%)
                                </span>
                              </div>
                            )}
                          </td>
                          <td style={{ padding: "10px 14px" }}>
                            <span
                              style={{
                                padding: "2px 6px",
                                borderRadius: "4px",
                                background: "rgba(255, 255, 255, 0.06)",
                                color: "var(--text-muted)",
                                fontSize: "0.75rem",
                              }}
                            >
                              {item.language_name} ({item.language.toUpperCase()})
                            </span>
                          </td>
                          <td style={{ padding: "10px 14px" }}>
                            {(() => {
                              const isUnconf = item.verdict === "UNCONFIRMED";
                              return (
                                <span
                                  style={{
                                    padding: "3px 8px",
                                    borderRadius: "4px",
                                    fontSize: "0.75rem",
                                    fontWeight: 700,
                                    background: isUnconf
                                      ? "rgba(217, 119, 6, 0.15)"
                                      : isFake
                                      ? "rgba(225, 29, 72, 0.15)"
                                      : "rgba(16, 185, 129, 0.15)",
                                    color: isUnconf ? "#fbbf24" : isFake ? "#fb7185" : "#34d399",
                                    border: `1px solid ${
                                      isUnconf
                                        ? "rgba(217, 119, 6, 0.3)"
                                        : isFake
                                        ? "rgba(225, 29, 72, 0.3)"
                                        : "rgba(16, 185, 129, 0.3)"
                                    }`,
                                  }}
                                >
                                  {isUnconf ? "UNCONFIRMED" : isFake ? "FAKE NEWS" : "AUTHENTIC"}
                                </span>
                              );
                            })()}
                          </td>
                          <td style={{ padding: "10px 14px", fontWeight: 600, color: "#ffffff" }}>
                            {item.confidence.toFixed(1)}%
                          </td>
                          <td style={{ padding: "10px 14px", color: "var(--text-muted)" }}>
                            {item.agreement}
                          </td>
                          <td style={{ padding: "10px 14px", textAlign: "right" }}>
                            {isExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                          </td>
                        </tr>

                        {/* Expandable Model Breakdown Drawer */}
                        {isExpanded && (
                          <tr style={{ background: "rgba(16, 24, 44, 0.95)" }}>
                            <td colSpan={7} style={{ padding: "14px 20px" }}>
                              <div style={{ fontSize: "0.78rem", fontWeight: 700, color: "var(--text-muted)", marginBottom: "8px" }}>
                                NEURAL MODEL ENSEMBLE BREAKDOWN:
                              </div>
                              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "10px" }}>
                                <div style={{ padding: "8px 12px", background: "rgba(255, 255, 255, 0.04)", borderRadius: "6px" }}>
                                  <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>RoBERTa-Base (125M)</div>
                                  <div style={{ fontWeight: 600, color: item.models.roberta.vote === "FAKE" ? "#fb7185" : "#34d399" }}>
                                    {item.models.roberta.vote || "N/A"} ({item.models.roberta.score || "N/A"})
                                  </div>
                                </div>
                                <div style={{ padding: "8px 12px", background: "rgba(255, 255, 255, 0.04)", borderRadius: "6px" }}>
                                  <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>Distilled TinyBERT (14M)</div>
                                  <div style={{ fontWeight: 600, color: item.models.tinybert.vote === "FAKE" ? "#fb7185" : "#34d399" }}>
                                    {item.models.tinybert.vote || "N/A"} ({item.models.tinybert.score || "N/A"})
                                  </div>
                                </div>
                                <div style={{ padding: "8px 12px", background: "rgba(255, 255, 255, 0.04)", borderRadius: "6px" }}>
                                  <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>BERT-Detect (110M)</div>
                                  <div style={{ fontWeight: 600, color: item.models.bert_detect.vote === "FAKE" ? "#fb7185" : "#34d399" }}>
                                    {item.models.bert_detect.vote || "N/A"} ({item.models.bert_detect.score || "N/A"})
                                  </div>
                                </div>
                                <div style={{ padding: "8px 12px", background: "rgba(255, 255, 255, 0.04)", borderRadius: "6px" }}>
                                  <div style={{ fontSize: "0.72rem", color: "var(--text-muted)" }}>XLM-RoBERTa (278M)</div>
                                  <div style={{ fontWeight: 600, color: item.models.xlm_roberta.vote === "FAKE" ? "#fb7185" : "#34d399" }}>
                                    {item.models.xlm_roberta.vote || "N/A"} ({item.models.xlm_roberta.score || "N/A"})
                                  </div>
                                </div>
                              </div>

                              {item.category && (
                                <div style={{ marginTop: "12px", display: "flex", alignItems: "center", gap: "8px", fontSize: "0.78rem" }}>
                                  <span style={{ color: "var(--text-muted)" }}>Topic Classification:</span>
                                  <span
                                    style={{
                                      padding: "3px 10px",
                                      borderRadius: "4px",
                                      fontWeight: 600,
                                      background: "rgba(99, 102, 241, 0.15)",
                                      color: "#c7d2fe",
                                      border: "1px solid rgba(99, 102, 241, 0.3)",
                                    }}
                                  >
                                    {item.category} • {item.category_confidence?.toFixed(1) || 0}% confidence
                                  </span>
                                </div>
                              )}

                              {item.metadata && Object.keys(item.metadata).length > 0 && (
                                <div style={{ marginTop: "10px", fontSize: "0.74rem", color: "var(--text-muted)" }}>
                                  <strong>Original File Metadata: </strong>
                                  {Object.entries(item.metadata).map(([k, v]) => `${k}: ${v}`).join(" | ")}
                                </div>
                              )}

                              {item.claim_sha256 && (
                                <div style={{ marginTop: "10px", fontSize: "0.72rem", fontFamily: "var(--font-mono)", color: "var(--text-muted)" }}>
                                  <strong>CLAIM SHA-256: </strong>
                                  <span style={{ color: "#cbd5e1" }}>{item.claim_sha256}</span>
                                  {item.is_polarized && (
                                    <span style={{ marginLeft: "10px", color: "#fbbf24" }}>
                                      ⚠️ Equivocal Polarization (Index: {item.polarization_index ?? "N/A"})
                                    </span>
                                  )}
                                </div>
                              )}
                            </td>
                          </tr>
                        )}
                      </React.Fragment>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
