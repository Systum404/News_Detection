"use client";

import React from "react";
import {
  Globe,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  ExternalLink,
  ShieldCheck,
  Search,
  BookOpen,
  Newspaper,
  Layers,
} from "lucide-react";
import { FactCheckData } from "@/types/news";

interface FactCheckEvidenceProps {
  data?: FactCheckData;
}

export const FactCheckEvidence: React.FC<FactCheckEvidenceProps> = ({ data }) => {
  if (!data) return null;

  const score = data.corroboration_score ?? 0;
  const status = data.corroboration_status;

  const getStatusConfig = () => {
    switch (status) {
      case "HIGHLY_CORROBORATED":
        return {
          label: "Highly Corroborated",
          color: "#34d399",
          bg: "rgba(16, 185, 129, 0.08)",
          border: "rgba(16, 185, 129, 0.22)",
          icon: <CheckCircle2 size={16} color="#34d399" />,
          barGradient: "linear-gradient(90deg, #059669, #10b981)",
        };
      case "HISTORICAL_FACT_GROUNDED":
        return {
          label: "Verified Historical Fact",
          color: "#38bdf8",
          bg: "rgba(2, 132, 199, 0.12)",
          border: "rgba(2, 132, 199, 0.35)",
          icon: <BookOpen size={16} color="#38bdf8" />,
          barGradient: "linear-gradient(90deg, #0284c7, #38bdf8)",
        };
      case "PARTIALLY_CORROBORATED":
        return {
          label: "Partially Corroborated",
          color: "#38bdf8",
          bg: "rgba(2, 132, 199, 0.08)",
          border: "rgba(2, 132, 199, 0.22)",
          icon: <CheckCircle2 size={16} color="#38bdf8" />,
          barGradient: "linear-gradient(90deg, #0284c7, #38bdf8)",
        };
      case "WEAK_CORROBORATION":
        return {
          label: "Weak Corroboration",
          color: "#fbbf24",
          bg: "rgba(245, 158, 11, 0.08)",
          border: "rgba(245, 158, 11, 0.22)",
          icon: <AlertTriangle size={16} color="#fbbf24" />,
          barGradient: "linear-gradient(90deg, #d97706, #f59e0b)",
        };
      case "TIMEOUT":
        return {
          label: "Search Timeout",
          color: "#fbbf24",
          bg: "rgba(245, 158, 11, 0.08)",
          border: "rgba(245, 158, 11, 0.22)",
          icon: <AlertTriangle size={16} color="#fbbf24" />,
          barGradient: "linear-gradient(90deg, #d97706, #f59e0b)",
        };
      case "UNVERIFIED_CLAIM":
      default:
        return {
          label: "Unverified Claim",
          color: "#f87171",
          bg: "rgba(220, 38, 38, 0.08)",
          border: "rgba(220, 38, 38, 0.22)",
          icon: <XCircle size={16} color="#f87171" />,
          barGradient: "linear-gradient(90deg, #b91c1c, #dc2626)",
        };
    }
  };

  const statusConfig = getStatusConfig();

  return (
    <div
      className="glass-panel"
      style={{
        marginTop: "24px",
        padding: "24px",
        borderRadius: "var(--radius-xl)",
        background: "var(--card-bg)",
        border: "1px solid var(--card-border)",
      }}
    >

      {/* Header with Title and Live Badge */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          flexWrap: "wrap",
          gap: "12px",
          marginBottom: "18px",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <div
            style={{
              width: "36px",
              height: "36px",
              borderRadius: "10px",
              background: "rgba(37, 99, 235, 0.18)",
              border: "1px solid rgba(37, 99, 235, 0.35)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <Globe size={20} color="#60a5fa" />
          </div>
          <div>
            <h3
              style={{
                fontSize: "1.15rem",
                fontWeight: 700,
                color: "#ffffff",
                letterSpacing: "-0.01em",
              }}
            >
              Real-Time Fact-Checking & Web Sources
            </h3>
            <p style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
              Live fact-checking against Google News and Wikipedia archives
            </p>
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "8px", flexWrap: "wrap" }}>
          {data.cached && (
            <div
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: "5px",
                padding: "5px 11px",
                borderRadius: "var(--radius-full)",
                background: "rgba(56, 189, 248, 0.12)",
                border: "1px solid rgba(56, 189, 248, 0.35)",
                color: "#38bdf8",
                fontSize: "0.75rem",
                fontWeight: 700,
                letterSpacing: "0.02em",
              }}
            >
              ⚡ Instant Cache
            </div>
          )}
          {data.is_historical && (
            <div
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: "5px",
                padding: "5px 11px",
                borderRadius: "var(--radius-full)",
                background: "rgba(168, 85, 247, 0.12)",
                border: "1px solid rgba(168, 85, 247, 0.35)",
                color: "#c084fc",
                fontSize: "0.75rem",
                fontWeight: 700,
                letterSpacing: "0.02em",
              }}
              title="Verified via Wikipedia historical archive"
            >
              🏛️ Historical Archive
            </div>
          )}
          {/* Corroboration Badge */}
          <div
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: "8px",
              padding: "6px 14px",
              borderRadius: "var(--radius-full)",
              background: statusConfig.bg,
              border: `1px solid ${statusConfig.border}`,
              color: statusConfig.color,
              fontSize: "0.85rem",
              fontWeight: 700,
            }}
          >
            {statusConfig.icon}
            <span>{statusConfig.label}</span>
          </div>
        </div>
      </div>

      {/* Corroboration Score Meter & Metrics Grid */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
          gap: "16px",
          marginBottom: "20px",
        }}
      >
        {/* Metric 1: Corroboration Score */}
        <div
          style={{
            background: "rgba(255, 255, 255, 0.03)",
            border: "1px solid rgba(255, 255, 255, 0.06)",
            borderRadius: "var(--radius-md)",
            padding: "16px",
          }}
        >
          <div
            style={{
              fontSize: "0.75rem",
              fontWeight: 600,
              color: "var(--text-muted)",
              marginBottom: "8px",
              textTransform: "uppercase",
              letterSpacing: "0.05em",
            }}
          >
            Corroboration Score
          </div>
          <div
            style={{
              fontSize: "1.8rem",
              fontWeight: 800,
              fontFamily: "var(--font-heading)",
              color: statusConfig.color,
              lineHeight: 1,
              marginBottom: "10px",
            }}
          >
            {score.toFixed(1)}%
          </div>
          {/* Progress Bar */}
          <div
            style={{
              width: "100%",
              height: "6px",
              borderRadius: "4px",
              background: "rgba(255, 255, 255, 0.08)",
              overflow: "hidden",
            }}
          >
            <div
              style={{
                width: `${score}%`,
                height: "100%",
                background: statusConfig.barGradient,
                transition: "width 0.8s ease-in-out",
              }}
            />
          </div>
        </div>

        {/* Metric 2: Authoritative Wires */}
        <div
          style={{
            background: "rgba(255, 255, 255, 0.03)",
            border: "1px solid rgba(255, 255, 255, 0.06)",
            borderRadius: "var(--radius-md)",
            padding: "16px",
          }}
        >
          <div
            style={{
              fontSize: "0.75rem",
              fontWeight: 600,
              color: "var(--text-muted)",
              marginBottom: "8px",
              textTransform: "uppercase",
              letterSpacing: "0.05em",
            }}
          >
            Verified News Sources
          </div>
          <div
            style={{
              fontSize: "1.8rem",
              fontWeight: 800,
              fontFamily: "var(--font-heading)",
              color: data.authoritative_count > 0 ? "#10b981" : "#94a3b8",
              lineHeight: 1,
              marginBottom: "6px",
              display: "flex",
              alignItems: "center",
              gap: "8px",
            }}
          >
            <span>{data.authoritative_count}</span>
            {data.authoritative_count > 0 && <ShieldCheck size={20} color="#10b981" />}
          </div>
          <div style={{ fontSize: "0.78rem", color: "var(--text-muted)" }}>
            Accredited news agencies (Reuters, BBC, AP, etc.)
          </div>
        </div>

        {/* Metric 3: Media Sources & Query */}
        <div
          style={{
            background: "rgba(255, 255, 255, 0.03)",
            border: "1px solid rgba(255, 255, 255, 0.06)",
            borderRadius: "var(--radius-md)",
            padding: "16px",
          }}
        >
          <div
            style={{
              fontSize: "0.75rem",
              fontWeight: 600,
              color: "var(--text-muted)",
              marginBottom: "8px",
              textTransform: "uppercase",
              letterSpacing: "0.05em",
            }}
          >
            Total Media Matches
          </div>
          <div
            style={{
              fontSize: "1.8rem",
              fontWeight: 800,
              fontFamily: "var(--font-heading)",
              color: "#f8fafc",
              lineHeight: 1,
              marginBottom: "6px",
            }}
          >
            {data.total_found}
          </div>
          <div style={{ fontSize: "0.78rem", color: "var(--text-muted)" }}>
            Articles found via live news search & archives
          </div>
        </div>
      </div>

      {/* Query Banner */}
      {data.search_query_used && (
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "10px",
            background: "rgba(0, 0, 0, 0.3)",
            padding: "10px 14px",
            borderRadius: "var(--radius-sm)",
            border: "1px solid rgba(255, 255, 255, 0.05)",
            marginBottom: "16px",
            fontSize: "0.82rem",
          }}
        >
          <Search size={14} color="#60a5fa" />
          <span style={{ color: "var(--text-muted)", fontWeight: 500 }}>Search Query Used:</span>
          <code
            style={{
              color: "#bfdbfe",
              fontFamily: "var(--font-mono)",
              background: "rgba(37, 99, 235, 0.15)",
              padding: "2px 8px",
              borderRadius: "4px",
              border: "1px solid rgba(37, 99, 235, 0.25)",
            }}
          >
            &quot;{data.search_query_used}&quot;
          </code>
        </div>
      )}

      {/* Historical Fact Disambiguation Banner */}
      {data.is_historical && data.temporal_metadata && (
        <div
          style={{
            display: "flex",
            alignItems: "flex-start",
            gap: "12px",
            background: "rgba(168, 85, 247, 0.08)",
            padding: "12px 16px",
            borderRadius: "var(--radius-sm)",
            border: "1px solid rgba(168, 85, 247, 0.25)",
            marginBottom: "16px",
            fontSize: "0.84rem",
            color: "#e9d5ff",
            lineHeight: 1.5,
          }}
        >
          <BookOpen size={18} color="#c084fc" style={{ flexShrink: 0, marginTop: "2px" }} />
          <div>
            <strong style={{ color: "#f3e8ff" }}>Verified Historical Event: </strong>
            Corroborated via {data.temporal_metadata.knowledge_source}
            {data.temporal_metadata.entity_title && (
              <> (<strong>{data.temporal_metadata.entity_title}</strong>)</>
            )}
            {data.temporal_metadata.verified_years && data.temporal_metadata.verified_years.length > 0 && (
              <> — Verified Year: <strong>{data.temporal_metadata.verified_years.join(", ")}</strong></>
            )}
            . Verified against historical knowledge archives.
          </div>
        </div>
      )}

      {/* Findings Narrative Summary */}
      <div
        style={{
          background: "rgba(16, 24, 44, 0.5)",
          borderLeft: `3px solid ${statusConfig.color}`,
          padding: "14px 18px",
          borderRadius: "0 var(--radius-sm) var(--radius-sm) 0",
          marginBottom: "20px",
          color: "var(--text-primary)",
          fontSize: "0.92rem",
          lineHeight: 1.5,
        }}
      >
        <div style={{ fontWeight: 600, color: statusConfig.color, marginBottom: "4px" }}>
          Fact-Check Summary:
        </div>
        <div>{data.summary}</div>
      </div>

      {/* Source Citation Cards */}
      <div>
        <div
          style={{
            fontSize: "0.85rem",
            fontWeight: 700,
            color: "#ffffff",
            marginBottom: "12px",
            display: "flex",
            alignItems: "center",
            gap: "8px",
          }}
        >
          <Layers size={16} color="#60a5fa" />
          <span>SOURCES & MATCHING ARTICLES ({data.sources?.length || 0})</span>
        </div>

        {data.sources && data.sources.length > 0 ? (
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))",
              gap: "12px",
            }}
          >
            {data.sources.map((src, idx) => (
              <a
                key={idx}
                href={src.url}
                target="_blank"
                rel="noopener noreferrer"
                style={{
                  display: "flex",
                  flexDirection: "column",
                  justifyContent: "space-between",
                  background: "rgba(255, 255, 255, 0.03)",
                  border: "1px solid rgba(255, 255, 255, 0.08)",
                  borderRadius: "var(--radius-md)",
                  padding: "14px",
                  textDecoration: "none",
                  transition: "all 0.2s ease",
                  cursor: "pointer",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.borderColor = "rgba(37, 99, 235, 0.45)";
                  e.currentTarget.style.background = "rgba(255, 255, 255, 0.06)";
                  e.currentTarget.style.transform = "translateY(-2px)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = "rgba(255, 255, 255, 0.08)";
                  e.currentTarget.style.background = "rgba(255, 255, 255, 0.03)";
                  e.currentTarget.style.transform = "translateY(0)";
                }}
              >
                <div>
                  {/* Publisher pill & type */}
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                      gap: "6px",
                      marginBottom: "8px",
                    }}
                  >
                    <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                      {src.type === "encyclopedia" ? (
                        <BookOpen size={14} color="#38bdf8" />
                      ) : (
                        <Newspaper size={14} color="#60a5fa" />
                      )}
                      <span
                        style={{
                          fontSize: "0.78rem",
                          fontWeight: 700,
                          color: src.is_authoritative ? "#34d399" : "#cbd5e1",
                        }}
                      >
                        {src.publisher}
                      </span>
                    </div>

                    {src.is_authoritative && (
                      <span
                        style={{
                          fontSize: "0.68rem",
                          fontWeight: 600,
                          padding: "2px 6px",
                          borderRadius: "4px",
                          background: "rgba(16, 185, 129, 0.15)",
                          color: "#10b981",
                          border: "1px solid rgba(16, 185, 129, 0.3)",
                          display: "inline-flex",
                          alignItems: "center",
                          gap: "3px",
                        }}
                      >
                        <ShieldCheck size={10} /> Verified
                      </span>
                    )}
                  </div>

                  {/* Title */}
                  <div
                    style={{
                      fontSize: "0.88rem",
                      fontWeight: 600,
                      color: "#f8fafc",
                      lineHeight: 1.4,
                      marginBottom: "6px",
                    }}
                  >
                    {src.title}
                  </div>

                  {/* Snippet if present */}
                  {src.snippet && (
                    <div
                      style={{
                        fontSize: "0.76rem",
                        color: "var(--text-muted)",
                        lineHeight: 1.35,
                        marginBottom: "8px",
                      }}
                    >
                      {src.snippet.slice(0, 140)}...
                    </div>
                  )}
                </div>

                {/* Footer with date & external link */}
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    paddingTop: "8px",
                    borderTop: "1px solid rgba(255, 255, 255, 0.05)",
                    fontSize: "0.72rem",
                    color: "var(--text-muted)",
                  }}
                >
                  <span>{src.published_at ? src.published_at.slice(0, 16) : "Live Wire"}</span>
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "4px",
                      color: "#60a5fa",
                      fontWeight: 600,
                    }}
                  >
                    <span>View Source</span>
                    <ExternalLink size={12} />
                  </div>
                </div>
              </a>
            ))}
          </div>
        ) : (
          <div
            style={{
              padding: "18px 24px",
              borderRadius: "var(--radius-md)",
              background: "rgba(244, 63, 94, 0.08)",
              border: "1px solid rgba(244, 63, 94, 0.2)",
              color: "#fb7185",
              fontSize: "0.88rem",
              display: "flex",
              alignItems: "center",
              gap: "12px",
            }}
          >
            <AlertTriangle size={20} color="#f43f5e" />
            <div>
              <strong>No Independent Corroborating Reporting Found:</strong>
              <div style={{ fontSize: "0.8rem", color: "rgba(251, 113, 133, 0.8)", marginTop: "2px" }}>
                Zero matching journalistic articles were indexed across global news wires (Reuters, AP, BBC)
                or regional press feeds for this claim. Fabricated stories typically have no external news record.
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
