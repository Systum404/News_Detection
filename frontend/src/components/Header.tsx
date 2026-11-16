"use client";

import React, { useEffect, useState } from "react";
import { BackendHealth } from "@/types/news";
import { ShieldCheck, Globe2 } from "lucide-react";

export function Header() {
  const [, setHealth] = useState<BackendHealth | null>(null);
  const [isLive, setIsLive] = useState<boolean>(false);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch("/api/health", { cache: "no-store" });
        if (res.ok) {
          const data: BackendHealth = await res.json();
          setHealth(data);
          setIsLive(true);
        } else {
          setIsLive(false);
        }
      } catch {
        setIsLive(false);
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header
      style={{
        width: "100%",
        padding: "16px 32px",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        borderBottom: "1px solid rgba(255, 255, 255, 0.075)",
        backdropFilter: "blur(20px)",
        WebkitBackdropFilter: "blur(20px)",
        background: "rgba(7, 9, 14, 0.82)",
        position: "sticky",
        top: 0,
        zIndex: 50,
      }}
    >
      {/* Brand & Title */}
      <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
        <div
          style={{
            width: "38px",
            height: "38px",
            borderRadius: "10px",
            background: "linear-gradient(180deg, #2563eb 0%, #1d4ed8 100%)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            boxShadow: "0 2px 10px rgba(37, 99, 235, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.2)",
            border: "1px solid rgba(255, 255, 255, 0.15)",
          }}
        >
          <ShieldCheck size={20} color="#ffffff" />
        </div>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <span
              style={{
                fontSize: "1.15rem",
                fontWeight: 700,
                letterSpacing: "-0.025em",
                fontFamily: "var(--font-heading)",
                color: "#f8fafc",
              }}
            >
              News Detection
            </span>
          </div>
          <p
            style={{
              fontSize: "0.75rem",
              color: "var(--text-muted)",
              marginTop: "1px",
            }}
          >
            Multi-Model AI Verification & Live Fact-Checking
          </p>
        </div>
      </div>

      {/* Real-time Status Badges */}
      <div style={{ display: "flex", alignItems: "center", gap: "10px", flexWrap: "wrap" }}>
        {/* Multilingual Chip */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "6px",
            padding: "5px 12px",
            borderRadius: "var(--radius-full)",
            background: "rgba(255, 255, 255, 0.035)",
            border: "1px solid rgba(255, 255, 255, 0.08)",
            fontSize: "0.76rem",
            color: "#bfdbfe",
            fontWeight: 500,
          }}
        >
          <Globe2 size={13} color="#60a5fa" />
          <span>50+ Languages</span>
        </div>

        {/* API Health Pill */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "7px",
            padding: "5px 12px",
            borderRadius: "var(--radius-full)",
            background: isLive ? "rgba(16, 185, 129, 0.08)" : "rgba(225, 29, 72, 0.08)",
            border: `1px solid ${isLive ? "rgba(16, 185, 129, 0.22)" : "rgba(225, 29, 72, 0.22)"}`,
            fontSize: "0.76rem",
            fontWeight: 600,
            color: isLive ? "#34d399" : "#fb7185",
          }}
        >
          <span
            className="pulse-indicator"
            style={{
              backgroundColor: isLive ? "#10b981" : "#f43f5e",
              boxShadow: isLive
                ? "0 0 8px rgba(16, 185, 129, 0.6)"
                : "0 0 8px rgba(244, 63, 94, 0.6)",
            }}
          />
          <span>{isLive ? "System Operational" : "Connecting..."}</span>
        </div>
      </div>
    </header>
  );
}
