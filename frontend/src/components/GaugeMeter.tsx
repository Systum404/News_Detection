"use client";

import React from "react";
import { AnalyzeResponse } from "@/types/news";
import { ShieldCheck, ShieldAlert, AlertTriangle } from "lucide-react";
import { useCountUp } from "@/hooks/useCountUp";

interface GaugeMeterProps {
  data: AnalyzeResponse;
}

const CATEGORY_CONFIG: Record<
  string,
  { label: string; icon: string; color: string; bg: string; border: string }
> = {
  Politics: {
    label: "Politics",
    icon: "🏛️",
    color: "#a5b4fc",
    bg: "rgba(99, 102, 241, 0.10)",
    border: "rgba(99, 102, 241, 0.25)",
  },
  Technology: {
    label: "Technology",
    icon: "💻",
    color: "#38bdf8",
    bg: "rgba(2, 132, 199, 0.10)",
    border: "rgba(2, 132, 199, 0.25)",
  },
  Business: {
    label: "Business",
    icon: "📈",
    color: "#34d399",
    bg: "rgba(5, 150, 105, 0.10)",
    border: "rgba(5, 150, 105, 0.25)",
  },
  Sports: {
    label: "Sports",
    icon: "🏆",
    color: "#fb923c",
    bg: "rgba(234, 88, 12, 0.10)",
    border: "rgba(234, 88, 12, 0.25)",
  },
  Science: {
    label: "Science",
    icon: "🔬",
    color: "#60a5fa",
    bg: "rgba(37, 99, 235, 0.10)",
    border: "rgba(37, 99, 235, 0.25)",
  },
  Entertainment: {
    label: "Entertainment",
    icon: "🎬",
    color: "#e879f9",
    bg: "rgba(162, 28, 175, 0.10)",
    border: "rgba(162, 28, 175, 0.25)",
  },
  "International News": {
    label: "International News",
    icon: "🌐",
    color: "#2dd4bf",
    bg: "rgba(13, 148, 136, 0.10)",
    border: "rgba(13, 148, 136, 0.25)",
  },
};

export function GaugeMeter({ data }: GaugeMeterProps) {
  const isPolarized = Boolean(data.is_polarized || data.verdict === "UNCONFIRMED");
  const isFake = data.is_fake;
  const score = data.soft_score; // 0 to 100
  const animatedScore = useCountUp(score, 1000);
  const radius = 70;
  const stroke = 11;
  const normalizedRadius = radius - stroke * 2;
  const circumference = normalizedRadius * 2 * Math.PI;
  const strokeDashoffset = circumference - (score / 100) * circumference;

  return (
    <div
      className="glass-panel animate-fade-in-up stagger-1 interactive-card"
      style={{
        padding: "30px 24px",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        position: "relative",
        overflow: "hidden",
        borderRadius: "var(--radius-xl)",
      }}
    >
      {/* Background Focused Radiance */}
      <div
        style={{
          position: "absolute",
          width: "240px",
          height: "240px",
          borderRadius: "50%",
          background: isPolarized
            ? "radial-gradient(circle, rgba(217, 119, 6, 0.10) 0%, transparent 70%)"
            : isFake
            ? "radial-gradient(circle, rgba(220, 38, 38, 0.08) 0%, transparent 70%)"
            : "radial-gradient(circle, rgba(5, 150, 105, 0.08) 0%, transparent 70%)",
          filter: "blur(24px)",
          pointerEvents: "none",
        }}
      />

      {/* Primary Verdict Banner with Concentric Radar Pulse */}
      <div
        style={{
          position: "relative",
          display: "inline-flex",
          alignItems: "center",
          gap: "8px",
          padding: "7px 18px",
          borderRadius: "var(--radius-full)",
          background: isPolarized
            ? "var(--color-warning-bg)"
            : isFake
            ? "var(--color-fake-bg)"
            : "var(--color-authentic-bg)",
          border: `1px solid ${
            isPolarized
              ? "var(--color-warning-border)"
              : isFake
              ? "var(--color-fake-border)"
              : "var(--color-authentic-border)"
          }`,
          marginBottom: "20px",
          boxShadow: isPolarized
            ? "0 2px 10px rgba(217, 119, 6, 0.15)"
            : isFake
            ? "0 2px 10px rgba(220, 38, 38, 0.15)"
            : "0 2px 10px rgba(5, 150, 105, 0.15)",
        }}
      >
        <div
          className={`radar-ring ${
            isPolarized ? "radar-polarized" : isFake ? "radar-fake" : "radar-authentic"
          }`}
        />
        {isPolarized ? (
          <AlertTriangle size={18} color="#fbbf24" />
        ) : isFake ? (
          <ShieldAlert size={18} color="#f87171" />
        ) : (
          <ShieldCheck size={18} color="#34d399" />
        )}
        <span
          style={{
            fontFamily: "var(--font-heading)",
            fontWeight: 700,
            fontSize: "0.98rem",
            letterSpacing: "0.03em",
            color: isPolarized ? "#fbbf24" : isFake ? "#f87171" : "#34d399",
            zIndex: 1,
          }}
        >
          {data.result}
        </span>
      </div>

      {/* Circular Animated SVG Gauge */}
      <div style={{ position: "relative", width: "170px", height: "170px" }}>
        <svg height="170" width="170" style={{ transform: "rotate(-90deg)" }}>
          <defs>
            <linearGradient id="gaugeGradientFake" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#f87171" />
              <stop offset="100%" stopColor="#dc2626" />
            </linearGradient>
            <linearGradient id="gaugeGradientAuthentic" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#34d399" />
              <stop offset="100%" stopColor="#059669" />
            </linearGradient>
            <linearGradient id="gaugeGradientPolarized" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#fbbf24" />
              <stop offset="100%" stopColor="#d97706" />
            </linearGradient>
          </defs>
          <circle
            stroke="rgba(255, 255, 255, 0.06)"
            fill="transparent"
            strokeWidth={stroke}
            r={normalizedRadius}
            cx="85"
            cy="85"
          />
          <circle
            stroke={
              isPolarized
                ? "url(#gaugeGradientPolarized)"
                : isFake
                ? "url(#gaugeGradientFake)"
                : "url(#gaugeGradientAuthentic)"
            }
            fill="transparent"
            strokeWidth={stroke}
            strokeDasharray={`${circumference} ${circumference}`}
            style={{ strokeDashoffset }}
            strokeLinecap="round"
            className="gauge-circle"
            r={normalizedRadius}
            cx="85"
            cy="85"
          />
        </svg>

        {/* Center Metric with Smooth Count-Up */}
        <div
          style={{
            position: "absolute",
            inset: 0,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <span
            style={{
              fontFamily: "var(--font-heading)",
              fontSize: "2.2rem",
              fontWeight: 800,
              letterSpacing: "-0.03em",
              color: "#ffffff",
              lineHeight: 1,
            }}
          >
            {animatedScore.toFixed(1)}%
          </span>
          <span
            style={{
              fontSize: "0.72rem",
              color: "var(--text-muted)",
              marginTop: "5px",
              fontWeight: 600,
              textTransform: "uppercase",
              letterSpacing: "0.06em",
            }}
          >
            Confidence
          </span>
        </div>
      </div>

      {/* Consensus & Agreement Badge */}
      <div
        style={{
          marginTop: "18px",
          display: "flex",
          alignItems: "center",
          gap: "8px",
          fontSize: "0.82rem",
          color: "var(--text-secondary)",
          background: "rgba(255, 255, 255, 0.03)",
          padding: "5px 14px",
          borderRadius: "var(--radius-sm)",
          border: "1px solid rgba(255, 255, 255, 0.075)",
        }}
      >
        <span style={{ fontWeight: 500, color: "var(--text-muted)" }}>
          Model Agreement:
        </span>
        <span
          style={{
            fontFamily: "var(--font-mono)",
            color: data.agreement.includes("3/3") || data.agreement.includes("4/4") ? "#34d399" : "#fbbf24",
            fontWeight: 700,
          }}
        >
          {data.agreement} {data.agreement.includes("3/3") || data.agreement.includes("4/4") ? "(Unanimous)" : "(Consensus)"}
        </span>
      </div>

      {/* News Category Classification Badge */}
      {data.category && (
        <div
          style={{
            marginTop: "12px",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: "6px",
            width: "100%",
          }}
        >
          {(() => {
            const cat = data.category.category;
            const config = CATEGORY_CONFIG[cat] || {
              label: cat,
              icon: "📰",
              color: "#94a3b8",
              bg: "rgba(148, 163, 184, 0.12)",
              border: "rgba(148, 163, 184, 0.28)",
            };

            return (
              <>
                <div
                  style={{
                    display: "inline-flex",
                    alignItems: "center",
                    gap: "7px",
                    padding: "4px 14px",
                    borderRadius: "var(--radius-full)",
                    background: config.bg,
                    border: `1px solid ${config.border}`,
                    boxShadow: `0 2px 10px ${config.bg}`,
                  }}
                  title={
                    data.category.runner_up
                      ? `Runner-up: ${data.category.runner_up.category} (${data.category.runner_up.confidence.toFixed(1)}%)`
                      : undefined
                  }
                >
                  <span style={{ fontSize: "0.95rem" }}>{config.icon}</span>
                  <span
                    style={{
                      fontFamily: "var(--font-heading)",
                      fontWeight: 700,
                      fontSize: "0.82rem",
                      color: config.color,
                      letterSpacing: "0.02em",
                    }}
                  >
                    {config.label}
                  </span>
                  <span
                    style={{
                      fontFamily: "var(--font-mono)",
                      fontSize: "0.72rem",
                      color: "rgba(255, 255, 255, 0.65)",
                      paddingLeft: "4px",
                      borderLeft: "1px solid rgba(255, 255, 255, 0.15)",
                    }}
                  >
                    {data.category.confidence.toFixed(1)}%
                  </span>
                </div>

                {data.category.matched_signals && data.category.matched_signals.length > 0 && (
                  <div
                    style={{
                      display: "flex",
                      gap: "4px",
                      flexWrap: "wrap",
                      justifyContent: "center",
                      maxWidth: "280px",
                    }}
                  >
                    {data.category.matched_signals.slice(0, 4).map((signal, idx) => (
                      <span
                        key={idx}
                        style={{
                          fontSize: "0.65rem",
                          fontFamily: "var(--font-mono)",
                          color: "var(--text-muted)",
                          background: "rgba(255, 255, 255, 0.03)",
                          border: "1px solid rgba(255, 255, 255, 0.06)",
                          padding: "1px 5px",
                          borderRadius: "3px",
                        }}
                      >
                        #{signal}
                      </span>
                    ))}
                  </div>
                )}
              </>
            );
          })()}
        </div>
      )}

      {/* Probability Calibration Telemetry Pill */}
      {data.calibration?.temperature_scaled && (
        <div
          style={{
            marginTop: "10px",
            display: "inline-flex",
            alignItems: "center",
            gap: "6px",
            fontSize: "0.72rem",
            fontWeight: 600,
            color: "#60a5fa",
            background: "rgba(37, 99, 235, 0.09)",
            padding: "4px 12px",
            borderRadius: "var(--radius-full)",
            border: "1px solid rgba(37, 99, 235, 0.22)",
          }}
        >
          <span style={{ width: "6px", height: "6px", borderRadius: "50%", background: "#60a5fa" }} />
          <span>Calibrated AI Confidence</span>
        </div>
      )}

      {/* Cache Telemetry Pill */}
      {data.cached && (
        <div
          style={{
            marginTop: "8px",
            display: "inline-flex",
            alignItems: "center",
            gap: "6px",
            fontSize: "0.72rem",
            fontWeight: 600,
            color: "#34d399",
            background: "rgba(5, 150, 105, 0.09)",
            padding: "3px 10px",
            borderRadius: "var(--radius-full)",
            border: "1px solid rgba(5, 150, 105, 0.25)",
          }}
          title={data.cache_key ? `Cache Key: ${data.cache_key}` : undefined}
        >
          <span style={{ width: "6px", height: "6px", borderRadius: "50%", background: "#34d399" }} />
          <span>
            {data.cache_tier === "L1_RAM" ? "⚡ Instant Cache (<0.1ms)" : "⚡ Saved Cache Result"}
          </span>
        </div>
      )}

      {/* Calibrated Dual Probability Bars */}
      <div style={{ width: "100%", marginTop: "22px" }}>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            fontSize: "0.75rem",
            marginBottom: "6px",
            color: "var(--text-muted)",
            fontWeight: 600,
            fontFamily: "var(--font-mono)",
          }}
        >
          <span style={{ color: "#34d399" }}>
            Authentic: {(data.soft_prob_auth * 100).toFixed(1)}%
          </span>
          <span style={{ color: "#f87171" }}>
            Fake: {(data.soft_prob_fake * 100).toFixed(1)}%
          </span>
        </div>
        <div
          style={{
            width: "100%",
            height: "6px",
            borderRadius: "3px",
            background: "rgba(255, 255, 255, 0.06)",
            overflow: "hidden",
            display: "flex",
          }}
        >
          <div
            style={{
              width: `${data.soft_prob_auth * 100}%`,
              background: "linear-gradient(90deg, #059669, #34d399)",
              transition: "width 0.8s ease",
            }}
          />
          <div
            style={{
              width: `${data.soft_prob_fake * 100}%`,
              background: "linear-gradient(90deg, #dc2626, #f87171)",
              transition: "width 0.8s ease",
            }}
          />
        </div>
      </div>
    </div>
  );
}
