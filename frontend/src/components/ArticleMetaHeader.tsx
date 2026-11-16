"use client";

import React from "react";
import { ExternalLink, Calendar, User, FileText, CheckCircle } from "lucide-react";
import { ScrapedArticleMeta } from "@/types/news";

interface ArticleMetaHeaderProps {
  meta?: ScrapedArticleMeta;
}

export const ArticleMetaHeader: React.FC<ArticleMetaHeaderProps> = ({ meta }) => {
  if (!meta) return null;

  return (
    <div
      className="glass-panel"
      style={{
        marginBottom: "24px",
        padding: "24px",
        borderRadius: "var(--radius-xl)",
        background: "linear-gradient(135deg, rgba(23, 33, 60, 0.7) 0%, rgba(13, 20, 36, 0.85) 100%)",
        border: "1px solid rgba(37, 99, 235, 0.25)",
        boxShadow: "0 10px 30px rgba(0, 0, 0, 0.4)",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "flex-start",
          gap: "20px",
          flexWrap: "wrap",
        }}
      >
        {/* Hero Image Thumbnail if available */}
        {meta.image_url && (
          <div
            style={{
              position: "relative",
              width: "140px",
              height: "100px",
              borderRadius: "var(--radius-md)",
              overflow: "hidden",
              flexShrink: 0,
              border: "1px solid rgba(255, 255, 255, 0.15)",
              background: "#0c1222",
            }}
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={meta.image_url}
              alt={meta.title}
              style={{
                width: "100%",
                height: "100%",
                objectFit: "cover",
              }}
              onError={(e) => {
                // Hide container if image fails to load (e.g. hotlink protection)
                e.currentTarget.parentElement!.style.display = "none";
              }}
            />
          </div>
        )}

        {/* Article Details */}
        <div style={{ flex: 1, minWidth: "260px" }}>
          {/* Top badges bar */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "8px",
              flexWrap: "wrap",
              marginBottom: "8px",
            }}
          >
            {/* Publisher Pill with Favicon */}
            <div
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: "6px",
                padding: "3px 10px",
                borderRadius: "var(--radius-full)",
                background: "rgba(37, 99, 235, 0.18)",
                border: "1px solid rgba(37, 99, 235, 0.35)",
                color: "#bfdbfe",
                fontSize: "0.78rem",
                fontWeight: 600,
              }}
            >
              {meta.favicon && (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={meta.favicon}
                  alt=""
                  style={{ width: "14px", height: "14px", borderRadius: "2px" }}
                  onError={(e) => {
                    e.currentTarget.style.display = "none";
                  }}
                />
              )}
              <span>{meta.publisher}</span>
            </div>

            {/* Domain Pill */}
            <span
              style={{
                fontSize: "0.74rem",
                color: "var(--text-muted)",
                fontFamily: "var(--font-mono)",
              }}
            >
              {meta.domain}
            </span>

            {/* Web Scraper Tag */}
            <span
              style={{
                fontSize: "0.7rem",
                fontWeight: 600,
                padding: "2px 8px",
                borderRadius: "4px",
                background: "rgba(16, 185, 129, 0.12)",
                color: "#10b981",
                border: "1px solid rgba(16, 185, 129, 0.25)",
                display: "inline-flex",
                alignItems: "center",
                gap: "4px",
              }}
            >
              <CheckCircle size={10} /> Scraped & Parsed
            </span>
          </div>

          {/* Headline Title */}
          <h2
            style={{
              fontSize: "1.25rem",
              fontWeight: 700,
              color: "#ffffff",
              lineHeight: 1.4,
              marginBottom: "10px",
              letterSpacing: "-0.01em",
            }}
          >
            {meta.title}
          </h2>

          {/* Byline / Metadata Row */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "16px",
              flexWrap: "wrap",
              fontSize: "0.78rem",
              color: "var(--text-muted)",
            }}
          >
            {meta.author && (
              <span style={{ display: "flex", alignItems: "center", gap: "4px" }}>
                <User size={12} color="#94a3b8" />
                <span style={{ color: "#cbd5e1" }}>{meta.author}</span>
              </span>
            )}

            {meta.published_date && (
              <span style={{ display: "flex", alignItems: "center", gap: "4px" }}>
                <Calendar size={12} color="#94a3b8" />
                <span>{meta.published_date.slice(0, 19).replace("T", " ")}</span>
              </span>
            )}

            <span style={{ display: "flex", alignItems: "center", gap: "4px" }}>
              <FileText size={12} color="#94a3b8" />
              <span>
                <strong style={{ color: "#f8fafc" }}>{meta.word_count.toLocaleString()}</strong> words (
                {(meta.char_count / 1024).toFixed(1)} KB)
              </span>
            </span>

            {/* Open Original Link */}
            <a
              href={meta.url}
              target="_blank"
              rel="noopener noreferrer"
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: "4px",
                color: "#60a5fa",
                textDecoration: "none",
                fontWeight: 600,
                marginLeft: "auto",
              }}
            >
              <span>View Original Article</span>
              <ExternalLink size={12} />
            </a>
          </div>
        </div>
      </div>
    </div>
  );
};
