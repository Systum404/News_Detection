"use client";

import React from "react";
import { AnalyzeResponse } from "@/types/news";
import { Cpu, CheckCircle, XCircle, Globe, Sparkles } from "lucide-react";

interface ModelGridProps {
  data: AnalyzeResponse;
}

export function ModelGrid({ data }: ModelGridProps) {
  const isMultiEnsemble = Boolean(data.has_xlmr && data.vote4 && data.vote4 !== "N/A");

  const models = [
    {
      id: "roberta",
      name: "RoBERTa Base",
      hfPath: "hamzab/roberta-fake-news-classification",
      params: "125M",
      weight: isMultiEnsemble ? "25%" : "40%",
      role: "Context & language structure analysis",
      vote: data.vote1,
      score: data.score1,
      isFake: data.is_fake1,
      badge: "Base Model",
    },
    {
      id: "tinybert",
      name: "Distilled TinyBERT",
      hfPath: "mrm8488/bert-tiny-finetuned-fake-news-detection",
      params: "14.3M",
      weight: isMultiEnsemble ? "15%" : "25%",
      role: "Fast lightweight fact pattern matching",
      vote: data.vote2,
      score: data.score2,
      isFake: data.is_fake2,
      badge: "Fast Model",
    },
    {
      id: "bert_detect",
      name: "BERT-Detect",
      hfPath: "jy46604790/Fake-News-Bert-Detect",
      params: "110M",
      weight: isMultiEnsemble ? "20%" : "35%",
      role: "Writing style and tone verification",
      vote: data.vote3,
      score: data.score3,
      isFake: data.is_fake3,
      badge: "Fine-Tuned",
    },
  ];

  if (isMultiEnsemble && data.vote4) {
    models.push({
      id: "xlm_roberta",
      name: "XLM-RoBERTa Multilingual",
      hfPath: "zodumair/xlmr-fake-news-multilingual",
      params: "278M",
      weight: "40%",
      role: "Global language & translation verification",
      vote: data.vote4,
      score: data.score4 || "N/A",
      isFake: Boolean(data.is_fake4),
      badge: "Multilingual",
    });
  }

  return (
    <div style={{ width: "100%", marginTop: "24px" }}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          marginBottom: "14px",
          flexWrap: "wrap",
          gap: "8px",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <Cpu size={16} color="#60a5fa" />
          <h3
            style={{
              fontSize: "1rem",
              fontWeight: 700,
              color: "#f8fafc",
              letterSpacing: "-0.01em",
            }}
          >
            AI Models Breakdown ({models.length} Models Consulted)
          </h3>
        </div>
        <span
          style={{
            fontSize: "0.78rem",
            color: isMultiEnsemble ? "#60a5fa" : "var(--text-muted)",
            display: "flex",
            alignItems: "center",
            gap: "6px",
            fontWeight: 500,
          }}
        >
          {isMultiEnsemble ? (
            <>
              <Globe size={13} /> 4-Model Consensus (Global Languages)
            </>
          ) : (
            "Combined Model Consensus"
          )}
        </span>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
          gap: "14px",
        }}
      >
        {models.map((m, idx) => {
          const isFake = m.isFake;
          const statusBorder = isFake
            ? "1px solid rgba(225, 29, 72, 0.22)"
            : "1px solid rgba(16, 185, 129, 0.22)";
          const statusBg = isFake
            ? "rgba(225, 29, 72, 0.08)"
            : "rgba(16, 185, 129, 0.08)";
          const statusText = isFake ? "#fb7185" : "#34d399";
          const isSpecialMultilingual = m.id === "xlm_roberta";
          const numericPercent = parseFloat(m.score.replace("%", "")) || 0;

          return (
            <div
              key={m.id}
              className="glass-panel interactive-card animate-fade-in-up"
              style={{
                padding: "18px",
                borderRadius: "var(--radius-md)",
                border: isSpecialMultilingual
                  ? "1px solid rgba(99, 102, 241, 0.3)"
                  : "1px solid var(--card-border)",
                background: isSpecialMultilingual
                  ? "linear-gradient(180deg, rgba(16, 24, 44, 0.78) 0%, rgba(13, 18, 30, 0.85) 100%)"
                  : "var(--card-bg)",
                display: "flex",
                flexDirection: "column",
                justifyContent: "space-between",
                animationDelay: `${idx * 80 + 100}ms`,
              }}
            >
              <div>
                {/* Header: Name & Weight */}
                <div
                  style={{
                    display: "flex",
                    alignItems: "flex-start",
                    justifyContent: "space-between",
                    marginBottom: "8px",
                  }}
                >
                  <div>
                    <h4
                      style={{
                        fontSize: "0.92rem",
                        fontWeight: 700,
                        color: "#ffffff",
                        display: "flex",
                        alignItems: "center",
                        gap: "6px",
                      }}
                    >
                      {m.name}
                      {isSpecialMultilingual && <Sparkles size={13} color="#60a5fa" />}
                    </h4>
                    <span
                      style={{
                        fontSize: "0.72rem",
                        fontFamily: "var(--font-mono)",
                        color: "var(--text-muted)",
                      }}
                    >
                      {m.params} params • {m.badge}
                    </span>
                  </div>
                  <span
                    style={{
                      fontSize: "0.7rem",
                      padding: "2px 7px",
                      borderRadius: "4px",
                      background: "rgba(255, 255, 255, 0.04)",
                      color: "#94a3b8",
                      fontWeight: 600,
                      fontFamily: "var(--font-mono)",
                      border: "1px solid rgba(255, 255, 255, 0.08)",
                    }}
                  >
                    Weight: {m.weight}
                  </span>
                </div>

                <p
                  style={{
                    fontSize: "0.76rem",
                    color: "var(--text-secondary)",
                    lineHeight: 1.4,
                    marginBottom: "14px",
                  }}
                >
                  {m.role}
                </p>
              </div>

              {/* Vote & Confidence Score Pill with Smooth Progress Bar */}
              <div>
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    padding: "8px 12px",
                    borderRadius: "var(--radius-sm)",
                    background: statusBg,
                    border: statusBorder,
                  }}
                >
                  <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                    {isFake ? (
                      <XCircle size={15} color="#fb7185" />
                    ) : (
                      <CheckCircle size={15} color="#34d399" />
                    )}
                    <span
                      style={{
                        fontFamily: "var(--font-heading)",
                        fontWeight: 700,
                        fontSize: "0.82rem",
                        color: statusText,
                      }}
                    >
                      {m.vote}
                    </span>
                  </div>
                  <span
                    style={{
                      fontFamily: "var(--font-mono)",
                      fontSize: "0.82rem",
                      fontWeight: 700,
                      color: "#ffffff",
                    }}
                  >
                    {m.score}
                  </span>
                </div>

                {/* Animated Confidence Bar */}
                <div
                  style={{
                    width: "100%",
                    height: "4px",
                    background: "rgba(255, 255, 255, 0.06)",
                    borderRadius: "2px",
                    marginTop: "8px",
                    overflow: "hidden",
                  }}
                >
                  <div
                    style={{
                      height: "100%",
                      width: `${numericPercent}%`,
                      background: isFake
                        ? "linear-gradient(90deg, #e11d48, #fb7185)"
                        : "linear-gradient(90deg, #059669, #34d399)",
                      borderRadius: "2px",
                      transition: "width 1s cubic-bezier(0.16, 1, 0.3, 1)",
                    }}
                  />
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
