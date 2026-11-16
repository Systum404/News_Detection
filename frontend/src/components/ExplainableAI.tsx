"use client";

import React from "react";
import { AnalyzeResponse } from "@/types/news";
import { Globe, FileText, CheckCircle2, AlertCircle, HelpCircle } from "lucide-react";

interface ExplainableAIProps {
  data: AnalyzeResponse;
}

export function ExplainableAI({ data }: ExplainableAIProps) {
  return (
    <div
      className="glass-panel"
      style={{
        marginTop: "24px",
        padding: "24px",
        borderRadius: "var(--radius-lg)",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "8px",
          marginBottom: "18px",
        }}
      >
        <HelpCircle size={18} color="#60a5fa" />
        <h3
          style={{
            fontSize: "1.05rem",
            fontWeight: 700,
            color: "#ffffff",
          }}
        >
          Analysis Explanation & Verification Details
        </h3>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
          gap: "16px",
        }}
      >
        {/* Language Routing Card */}
        <div
          style={{
            padding: "16px",
            borderRadius: "var(--radius-md)",
            background: "rgba(255, 255, 255, 0.03)",
            border: "1px solid rgba(255, 255, 255, 0.08)",
          }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "8px",
              marginBottom: "8px",
            }}
          >
            <Globe size={16} color="#38bdf8" />
            <span
              style={{
                fontSize: "0.85rem",
                fontWeight: 600,
                color: "#e2e8f0",
              }}
            >
              Detected Language & Routing
            </span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "8px", marginTop: "10px" }}>
            <span
              style={{
                fontSize: "0.75rem",
                fontWeight: 700,
                padding: "2px 8px",
                borderRadius: "4px",
                background: "rgba(56, 189, 248, 0.15)",
                color: "#38bdf8",
                border: "1px solid rgba(56, 189, 248, 0.3)",
                textTransform: "uppercase",
                fontFamily: "var(--font-mono)",
              }}
            >
              {data.language_code || "EN"}
            </span>
            <span style={{ fontSize: "0.9rem", fontWeight: 600, color: "#ffffff" }}>
              {data.language_name || "English"}
            </span>
          </div>
          <p
            style={{
              fontSize: "0.75rem",
              color: "var(--text-muted)",
              marginTop: "8px",
            }}
          >
            {data.routing_engine || "Standard Cross-Lingual Representation"}
          </p>
        </div>

        {/* Short Text Safeguard Status */}
        <div
          style={{
            padding: "16px",
            borderRadius: "var(--radius-md)",
            background: "rgba(255, 255, 255, 0.03)",
            border: "1px solid rgba(255, 255, 255, 0.08)",
          }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "8px",
              marginBottom: "8px",
            }}
          >
            <FileText size={16} color={data.is_short ? "#f59e0b" : "#34d399"} />
            <span
              style={{
                fontSize: "0.85rem",
                fontWeight: 600,
                color: "#e2e8f0",
              }}
            >
              Context Safeguard Status
            </span>
          </div>
          <div style={{ marginTop: "10px" }}>
            {data.is_short ? (
              <span
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  gap: "6px",
                  fontSize: "0.8rem",
                  color: "#fbbf24",
                  fontWeight: 600,
                  background: "rgba(245, 158, 11, 0.1)",
                  padding: "4px 10px",
                  borderRadius: "6px",
                  border: "1px solid rgba(245, 158, 11, 0.25)",
                }}
              >
                <AlertCircle size={14} /> Short Text (&lt; 15 words)
              </span>
            ) : (
              <span
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  gap: "6px",
                  fontSize: "0.8rem",
                  color: "#34d399",
                  fontWeight: 600,
                  background: "rgba(16, 185, 129, 0.1)",
                  padding: "4px 10px",
                  borderRadius: "6px",
                  border: "1px solid rgba(16, 185, 129, 0.25)",
                }}
              >
                <CheckCircle2 size={14} /> Standard Article Length
              </span>
            )}
          </div>
          <p
            style={{
              fontSize: "0.75rem",
              color: "var(--text-muted)",
              marginTop: "8px",
            }}
          >
            {data.is_short
              ? "Input is brief; confidence reflects single-claim classification without full editorial context."
              : "Sufficient editorial length for cross-referencing lexical and stylistic consensus."}
          </p>
        </div>

        {/* Soft Consensus Formula */}
        <div
          style={{
            padding: "16px",
            borderRadius: "var(--radius-md)",
            background: "rgba(255, 255, 255, 0.03)",
            border: "1px solid rgba(255, 255, 255, 0.08)",
          }}
        >
          <span
            style={{
              fontSize: "0.85rem",
              fontWeight: 600,
              color: "#e2e8f0",
              display: "block",
              marginBottom: "8px",
            }}
          >
            Model Weighting Formula
          </span>
          <div
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: "0.75rem",
              color: "#a5b4fc",
              background: "rgba(0, 0, 0, 0.4)",
              padding: "8px 10px",
              borderRadius: "6px",
              border: "1px solid rgba(255, 255, 255, 0.05)",
            }}
          >
            P(Fake) = 0.40·R + 0.25·T + 0.35·B
          </div>
          <p
            style={{
              fontSize: "0.75rem",
              color: "var(--text-muted)",
              marginTop: "8px",
            }}
          >
            Model weights: RoBERTa (40%), TinyBERT (25%), and BERT-Detect (35%).
          </p>
        </div>

        {/* Computing Hardware & Speed */}
        <div
          style={{
            padding: "16px",
            borderRadius: "var(--radius-md)",
            background: "rgba(255, 255, 255, 0.03)",
            border: "1px solid rgba(255, 255, 255, 0.08)",
          }}
        >
          <span
            style={{
              fontSize: "0.85rem",
              fontWeight: 600,
              color: "#e2e8f0",
              display: "block",
              marginBottom: "8px",
            }}
          >
            Computing Hardware & Speed
          </span>
          <div style={{ display: "flex", alignItems: "center", gap: "8px", marginTop: "10px" }}>
            <span
              style={{
                fontSize: "0.75rem",
                fontWeight: 700,
                padding: "2px 8px",
                borderRadius: "4px",
                background: "rgba(16, 185, 129, 0.15)",
                color: "#34d399",
                border: "1px solid rgba(16, 185, 129, 0.3)",
                textTransform: "uppercase",
                fontFamily: "var(--font-mono)",
              }}
            >
              {data.precision || "FP16"}
            </span>
            <span style={{ fontSize: "0.85rem", fontWeight: 600, color: "#ffffff" }}>
              {data.acceleration_device || "Hardware Accelerated"}
            </span>
          </div>
          <p
            style={{
              fontSize: "0.75rem",
              color: "var(--text-muted)",
              marginTop: "8px",
            }}
          >
            Hardware acceleration enabled for fast, low-latency prediction (~2.2x faster).
          </p>
        </div>
      </div>
    </div>
  );
}
