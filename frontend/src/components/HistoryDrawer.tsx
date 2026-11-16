"use client";

import React from "react";
import { AnalysisHistoryItem } from "@/types/news";
import { History, Clock, ShieldCheck, ShieldAlert, AlertTriangle } from "lucide-react";

interface HistoryDrawerProps {
  history: AnalysisHistoryItem[];
  onSelect: (item: AnalysisHistoryItem) => void;
  onClear: () => void;
}

export function HistoryDrawer({ history, onSelect, onClear }: HistoryDrawerProps) {
  if (history.length === 0) return null;

  return (
    <div
      className="glass-panel"
      style={{
        marginTop: "24px",
        padding: "20px 24px",
        borderRadius: "var(--radius-lg)",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          marginBottom: "14px",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <History size={18} color="#60a5fa" />
          <h3 style={{ fontSize: "0.95rem", fontWeight: 700, color: "#ffffff" }}>
            Analysis History ({history.length})
          </h3>
        </div>
        <button
          onClick={onClear}
          style={{
            background: "transparent",
            border: "none",
            color: "var(--text-muted)",
            fontSize: "0.75rem",
            cursor: "pointer",
            textDecoration: "underline",
          }}
        >
          Clear History
        </button>
      </div>

      <div
        style={{
          display: "flex",
          gap: "12px",
          overflowX: "auto",
          paddingBottom: "8px",
        }}
      >
        {history.map((item) => {
          const isPolarized = item.response.verdict === "UNCONFIRMED" || Boolean(item.response.is_polarized);
          const isFake = item.response.is_fake;
          const statusColor = isPolarized ? "#fbbf24" : isFake ? "#fb7185" : "#34d399";
          return (
            <button
              key={item.id}
              onClick={() => onSelect(item)}
              style={{
                flex: "0 0 240px",
                textAlign: "left",
                padding: "12px 14px",
                borderRadius: "var(--radius-md)",
                background: "rgba(255, 255, 255, 0.03)",
                border: "1px solid rgba(255, 255, 255, 0.08)",
                cursor: "pointer",
                transition: "all 0.2s ease",
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = isPolarized
                  ? "rgba(217, 119, 6, 0.4)"
                  : isFake
                  ? "rgba(244, 63, 94, 0.4)"
                  : "rgba(16, 185, 129, 0.4)";
                e.currentTarget.style.background = "rgba(255, 255, 255, 0.06)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = "rgba(255, 255, 255, 0.08)";
                e.currentTarget.style.background = "rgba(255, 255, 255, 0.03)";
              }}
            >
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  marginBottom: "6px",
                }}
              >
                <span
                  style={{
                    display: "inline-flex",
                    alignItems: "center",
                    gap: "4px",
                    fontSize: "0.75rem",
                    fontWeight: 700,
                    color: statusColor,
                  }}
                >
                  {isPolarized ? (
                    <AlertTriangle size={12} />
                  ) : isFake ? (
                    <ShieldAlert size={12} />
                  ) : (
                    <ShieldCheck size={12} />
                  )}
                  {item.response.verdict}
                </span>
                <span
                  style={{
                    fontSize: "0.7rem",
                    color: "var(--text-muted)",
                    display: "flex",
                    alignItems: "center",
                    gap: "4px",
                  }}
                >
                  <Clock size={10} />
                  {item.timestamp}
                </span>
              </div>
              <p
                style={{
                  fontSize: "0.75rem",
                  color: "var(--text-secondary)",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                }}
              >
                {item.textSnippet}
              </p>
            </button>
          );
        })}
      </div>
    </div>
  );
}
