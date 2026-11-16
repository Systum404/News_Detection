"""
News Category & Topic Classification Engine
============================================
Fast, high-precision contextual classifier categorizing news articles
into the 7 enterprise standard taxonomies:
  - Politics
  - Technology
  - Business
  - Sports
  - Science
  - Entertainment
  - International News

Features:
  - Multi-token semantic collocation matching (e.g. "artificial intelligence", "wall street").
  - Multilingual root awareness (English, Hindi, Spanish, French, German).
  - Temperature-scaled probabilistic softmax distributions.
  - Entity-level domain heuristics (geopolitical entities, corporate tickers, championships).
"""

import re
import math
import logging
import unicodedata
from typing import List, Dict, Any

logger = logging.getLogger("news_detection.categorizer")

CATEGORIES: List[str] = [
    "Politics",
    "Technology",
    "Business",
    "Sports",
    "Science",
    "Entertainment",
    "International News",
]

# Curated, weighted semantic ontologies with multi-word expressions and multilingual stems
DOMAIN_LEXICONS: Dict[str, Dict[str, float]] = {
    "Politics": {
        "politics": 2.5, "political": 2.5, "politician": 2.5,
        "parliament": 3.0, "congress": 3.0, "senate": 3.0, "senator": 3.0,
        "election": 3.0, "elections": 3.0, "electoral": 3.0, "vote": 2.5, "voting": 2.5,
        "legislation": 3.0, "legislative": 3.0, "legislator": 3.0, "lawmaker": 3.0,
        "prime minister": 3.5, "president": 2.0, "presidential": 2.5, "white house": 3.5,
        "cabinet": 2.0, "minister": 2.0, "ministry": 2.0, "governor": 2.5,
        "democrat": 3.0, "republican": 3.0, "democratic": 2.0, "gop": 3.0,
        "tory": 3.0, "labour party": 3.5, "parliamentary": 3.0,
        "impeachment": 3.5, "impeach": 3.0, "veto": 3.0, "constitution": 2.5,
        "supreme court": 2.5, "judiciary": 2.0, "referendum": 3.5, "ballot": 3.0,
        "campaign": 2.0, "bipartisan": 3.0, "filibuster": 3.5, "diplomacy": 2.0,
        "policy": 1.5, "sanction": 2.0, "sanctions": 2.0, "geopolitics": 2.5,
        # Multilingual hints
        "राजनाति": 3.0, "चुनाव": 3.0, "politica": 2.5, "politique": 2.5,
    },
    "Technology": {
        "technology": 2.5, "tech": 2.5, "technological": 2.5,
        "artificial intelligence": 3.5, "ai": 2.0, "machine learning": 3.5,
        "neural network": 3.5, "deep learning": 3.5, "algorithm": 2.5, "algorithms": 2.5,
        "software": 2.5, "hardware": 2.5, "semiconductor": 3.5, "chipmaker": 3.5,
        "microchip": 3.5, "gpu": 3.0, "cpu": 2.5, "nvidia": 3.0, "intel": 2.5,
        "cybersecurity": 3.5, "cyber": 2.5, "malware": 3.0, "ransomware": 3.5,
        "hacker": 2.5, "hacking": 2.5, "encryption": 3.0, "cloud computing": 3.5,
        "quantum computing": 3.5, "robotics": 3.0, "robot": 2.5, "autonomous": 2.5,
        "smartphone": 2.5, "iphone": 2.5, "android": 2.5, "apple inc": 2.5,
        "google": 2.0, "metaverse": 3.0, "virtual reality": 3.0, "openai": 3.5,
        "chatgpt": 3.5, "llm": 3.0, "database": 2.5, "developer": 2.0,
        "telecom": 2.0, "5g": 2.5, "blockchain": 3.0, "cryptocurrency": 3.0,
        # Multilingual hints
        "प्रौद्योगिकी": 3.0, "tecnologia": 2.5, "informatique": 2.5,
    },
    "Business": {
        "business": 2.5, "economy": 2.5, "economic": 2.5, "economist": 2.5,
        "wall street": 3.5, "stock market": 3.5, "stocks": 2.5, "shares": 2.0,
        "nasdaq": 3.5, "dow jones": 3.5, "s&p 500": 3.5, "nyse": 3.5,
        "inflation": 3.0, "interest rates": 3.0, "federal reserve": 3.5, "central bank": 3.0,
        "gdp": 3.0, "recession": 3.0, "revenue": 2.5, "quarterly earnings": 3.5,
        "profit": 2.0, "losses": 2.0, "investor": 2.5, "investors": 2.5,
        "investment": 2.5, "venture capital": 3.5, "private equity": 3.5,
        "merger": 3.0, "acquisition": 3.0, "takeover": 3.0, "ipo": 3.5,
        "valuation": 2.5, "dividend": 3.0, "corporate": 2.0, "corporation": 2.0,
        "commerce": 2.5, "retail": 2.0, "supply chain": 2.5, "treasury": 2.5,
        "real estate": 2.5, "mortgage": 2.5, "banking": 2.5, "bankruptcy": 3.0,
        # Multilingual hints
        "व्यापार": 3.0, "अर्थव्यवस्था": 3.0, "negocios": 2.5, "economie": 2.5,
    },
    "Sports": {
        "sports": 2.5, "sport": 2.5, "athlete": 3.0, "athletics": 3.0,
        "championship": 3.0, "champion": 2.5, "tournament": 3.0, "league": 2.5,
        "olympics": 3.5, "olympic": 3.5, "fifa": 3.5, "world cup": 3.5,
        "football": 2.5, "soccer": 3.0, "premier league": 3.5, "uefa": 3.5,
        "basketball": 3.0, "nba": 3.5, "nfl": 3.5, "baseball": 3.0, "mlb": 3.5,
        "cricket": 3.5, "ipl": 3.5, "tennis": 3.0, "grand slam": 3.5,
        "wimbledon": 3.5, "formula 1": 3.5, "f1": 3.0, "grand prix": 3.5,
        "stadium": 2.5, "referee": 3.0, "coach": 2.0, "playoffs": 3.0,
        "trophy": 2.5, "medal": 2.5, "quarterback": 3.5, "striker": 2.5,
        "touchdown": 3.5, "goal": 2.0, "scored": 2.0, "match": 1.5,
        # Multilingual hints
        "खेल": 3.0, "deportes": 2.5, "sportive": 2.5,
    },
    "Science": {
        "science": 2.5, "scientific": 2.5, "scientist": 2.5, "scientists": 2.5,
        "nasa": 3.5, "esa": 3.0, "isro": 3.5, "astronomy": 3.5, "astronomer": 3.5,
        "telescope": 3.0, "james webb": 3.5, "hubble": 3.5, "spacecraft": 3.0,
        "exoplanet": 3.5, "galaxy": 3.0, "universe": 2.5, "cosmic": 3.0,
        "physics": 3.0, "physicist": 3.0, "particle physics": 3.5, "cern": 3.5,
        "hadron collider": 3.5, "quantum mechanics": 3.5, "chemistry": 3.0,
        "biology": 3.0, "biologist": 3.0, "genetics": 3.5, "genome": 3.5,
        "dna": 3.0, "crispr": 3.5, "laboratory": 2.5, "clinical trial": 2.5,
        "climate change": 3.0, "global warming": 3.0, "ecosystem": 2.5,
        "paleontology": 3.5, "fossil": 3.0, "species": 2.5, "neuroscience": 3.5,
        # Multilingual hints
        "विज्ञान": 3.0, "ciencia": 2.5, "scientifique": 2.5,
    },
    "Entertainment": {
        "entertainment": 2.5, "entertainer": 2.5, "celebrity": 3.0, "celebrities": 3.0,
        "hollywood": 3.5, "bollywood": 3.5, "cinema": 3.0, "film": 2.5, "movie": 2.5,
        "actor": 2.5, "actress": 2.5, "director": 2.0, "filmmaker": 3.0,
        "oscar": 3.5, "oscars": 3.5, "academy awards": 3.5, "emmy": 3.5, "grammy": 3.5,
        "box office": 3.5, "blockbuster": 3.0, "trailer": 2.5, "premiere": 2.5,
        "netflix": 2.5, "disney": 2.5, "television": 2.0, "tv series": 3.0,
        "music": 2.0, "musician": 2.5, "album": 2.5, "concert": 3.0,
        "singer": 2.5, "pop star": 3.0, "soundtrack": 3.0, "broadway": 3.5,
        "cannes": 3.5, "film festival": 3.5, "red carpet": 3.0, "comedian": 2.5,
        # Multilingual hints
        "मनोरंजन": 3.0, "filme": 2.5, "musique": 2.5, "spectacle": 2.5,
    },
    "International News": {
        "international": 2.5, "global": 2.0, "worldwide": 2.0,
        "united nations": 3.5, "un security council": 4.0, "un general assembly": 4.0,
        "nato": 3.5, "treaty": 3.0, "accord": 2.5, "bilateral": 3.0, "multilateral": 3.0,
        "summit": 2.5, "foreign affairs": 3.5, "foreign minister": 3.5,
        "ambassador": 3.0, "embassy": 3.0, "consulate": 3.0, "diplomatic": 3.0,
        "peacekeeping": 3.5, "geneva": 2.5, "hague": 3.0, "international criminal court": 4.0,
        "cross-border": 3.0, "border dispute": 3.5, "ceasefire": 3.5, "truce": 3.5,
        "refugee": 3.0, "refugees": 3.0, "asylum": 2.5, "humanitarian": 2.5,
        "sovereignty": 3.0, "world health organization": 3.5, "unicef": 3.5,
        "world bank": 3.0, "imf": 3.0, "european union": 3.0, "g7": 3.5, "g20": 3.5,
        "foreign policy": 3.5, "geopolitical tension": 3.5,
        # Multilingual hints
        "अंतरराष्ट्रीय": 3.5, "internacional": 2.5, "etranger": 2.5,
    },
}


def normalize_text(text: str) -> str:
    """Canonical Unicode NFKC normalization and lowercasing."""
    if not text:
        return ""
    norm = unicodedata.normalize("NFKC", text.strip().lower())
    # Keep letters, numbers, combining marks (vowels/accents/matras), whitespace, and hyphens
    chars = []
    for ch in norm:
        cat = unicodedata.category(ch)
        if cat.startswith(("L", "N", "M")) or ch in " \t\n-":
            chars.append(ch)
        else:
            chars.append(" ")
    cleaned = "".join(chars)
    return re.sub(r"\s+", " ", cleaned).strip()


def predict_news_category(
    text: str,
    temperature: float = 1.15,
) -> Dict[str, Any]:
    """
    Evaluates news article content across the 7 target categories.
    Returns predicted category, confidence, full distribution, and matched signals.
    """
    cleaned = normalize_text(text)
    if not cleaned:
        # Default fallback for empty text
        return {
            "category": "International News",
            "confidence": 14.29,
            "distribution": {cat: 14.29 for cat in CATEGORIES},
            "matched_signals": [],
            "is_ambiguous": True,
        }

    raw_scores: Dict[str, float] = {cat: 0.2 for cat in CATEGORIES}  # Base prior
    matched_signals: Dict[str, List[str]] = {cat: [] for cat in CATEGORIES}
    padded_text = f" {cleaned} "

    # Word and Bigram/Trigram scan with universal boundary awareness
    for cat, lex_dict in DOMAIN_LEXICONS.items():
        for phrase, weight in lex_dict.items():
            pattern = r"(?<=\s)" + re.escape(phrase) + r"(?=\s)"
            matches = len(re.findall(pattern, padded_text))
            if matches > 0:
                # Diminishing returns for repeated occurrences: sqrt(matches)
                contribution = weight * (1.0 + math.log1p(matches))
                raw_scores[cat] += contribution
                matched_signals[cat].append(phrase)

    # Contextual Cross-Domain Disambiguation:
    # 1. If text has heavy multilateral/global terms, boost International News
    if raw_scores["International News"] > 2.0 and raw_scores["Politics"] > 2.0:
        # Check if focus is domestic or foreign
        if any(term in cleaned for term in ["united nations", "bilateral", "nato", "foreign minister", "cross-border", "treaty", "geneva"]):
            raw_scores["International News"] += 1.8

    # 2. If business + tech (e.g. Apple stock earnings), distinguish primary domain
    if raw_scores["Technology"] > 2.0 and raw_scores["Business"] > 2.0:
        tech_specific = sum(1 for t in ["software", "ai", "algorithm", "neural", "gpu", "chipmaker", "cyber"] if t in cleaned)
        biz_specific = sum(1 for t in ["shares", "quarterly earnings", "revenue", "wall street", "nasdaq"] if t in cleaned)
        if tech_specific > biz_specific:
            raw_scores["Technology"] += 1.5
        elif biz_specific > tech_specific:
            raw_scores["Business"] += 1.5

    # Compute Softmax Distribution with Temperature Scaling
    max_score = max(raw_scores.values())
    exp_scores = {
        cat: math.exp((raw_scores[cat] - max_score) / temperature)
        for cat in CATEGORIES
    }
    sum_exp = sum(exp_scores.values())

    distribution: Dict[str, float] = {
        cat: round((exp_scores[cat] / sum_exp) * 100.0, 2)
        for cat in CATEGORIES
    }

    # Identify primary prediction
    sorted_cats = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
    top_category, top_confidence = sorted_cats[0]
    runner_up_cat, runner_up_conf = sorted_cats[1]

    # Check if margin between top 2 is small (< 10%)
    is_ambiguous = (top_confidence - runner_up_conf) < 10.0 and top_confidence < 45.0

    all_signals = matched_signals.get(top_category, [])

    return {
        "category": top_category,
        "confidence": top_confidence,
        "distribution": distribution,
        "runner_up": {
            "category": runner_up_cat,
            "confidence": runner_up_conf,
        },
        "matched_signals": all_signals[:6],
        "is_ambiguous": is_ambiguous,
    }
