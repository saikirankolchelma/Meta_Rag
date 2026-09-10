"""Fetches plain-text Wikipedia article extracts into data/raw/ as .txt files.

Uses Wikipedia's public MediaWiki action API (no auth required), one title per
request (the API caps full-article extracts at exlimit=1, so batching titles
silently drops all but one article's text). Covers the same domain spread used
for the synthetic router training data, so the corpus and the router's
difficulty bands overlap.

Usage:
    python -m src.ingestion.fetch_wikipedia
"""

import re
import sys
import time

import requests

from src.utils.config import DATA_RAW_DIR

API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "MetaRAG-Ingestion-Pipeline/1.0 (educational RAG project; contact via GitHub)"

ARTICLE_TITLES = [
    # science and physics
    "Quantum mechanics", "Theory of relativity", "Thermodynamics", "Electromagnetism",
    "Nuclear fusion", "Particle physics",
    # history
    "World War II", "Roman Empire", "French Revolution", "Industrial Revolution",
    "Cold War", "Ancient Egypt",
    # technology and computing
    "Artificial intelligence", "Internet", "Blockchain", "Quantum computing",
    "Cloud computing", "Cybersecurity",
    # business and economics
    "Inflation", "Stock market", "Supply and demand", "Cryptocurrency",
    "Globalization", "Venture capital",
    # health and medicine
    "Vaccine", "Antibiotic resistance", "Cancer", "Mental health",
    "Nutrition", "Immune system",
    # everyday life and household
    "Recycling", "Personal finance", "Remote work", "Public transport",
    "Renewable energy", "Home automation",
    # law and politics
    "Democracy", "United Nations", "Constitutional law", "Human rights",
    "International trade law", "Election",
    # geography
    "Amazon rainforest", "Sahara", "Himalayas", "Great Barrier Reef",
    "Nile", "Pacific Ocean",
    # arts and culture
    "Renaissance", "Jazz", "Impressionism", "Classical music",
    "Film noir", "Modern art",
    # sports
    "Olympic Games", "Association football", "Marathon", "Cricket",
    "Tennis", "Basketball",
    # environment and climate
    "Climate change", "Deforestation", "Ocean acidification", "Biodiversity loss",
    "Carbon capture and storage", "Greenhouse gas",
    # psychology
    "Cognitive bias", "Behaviorism", "Memory", "Developmental psychology",
    "Social psychology", "Sleep",
    # mathematics
    "Prime number", "Calculus", "Probability theory", "Game theory",
    "Linear algebra", "Chaos theory",
    # food and cooking
    "Fermentation in food processing", "Sourdough", "Umami", "Food preservation",
    "Coffee", "Spice",
    # space and astronomy
    "Black hole", "Exoplanet", "International Space Station", "Big Bang",
    "Neutron star", "Solar System",
    # biology and nature
    "Photosynthesis", "Evolution", "DNA", "Ecosystem",
    "Symbiosis", "Coral reef",
]


def slugify(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")


def fetch_article(title: str, max_retries: int = 5) -> dict:
    # Wikipedia's API caps full (non-intro) extracts at exlimit=1, so batching
    # multiple titles per request silently drops all but one article's text.
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": 1,
        "titles": title,
        "redirects": 1,
    }
    for attempt in range(1, max_retries + 1):
        resp = requests.get(API_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=30)
        if resp.status_code == 429:
            wait_s = float(resp.headers.get("retry-after", 5 * attempt))
            print(f"Rate limited on '{title}' (attempt {attempt}/{max_retries}), waiting {wait_s:.0f}s", file=sys.stderr)
            time.sleep(wait_s)
            continue
        resp.raise_for_status()
        return resp.json()
    resp.raise_for_status()
    return resp.json()


def main():
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    saved = 0
    skipped = []

    for title in ARTICLE_TITLES:
        data = fetch_article(title)
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            page_title = page.get("title", title)
            extract = page.get("extract", "").strip()
            if "missing" in page or not extract:
                skipped.append(page_title)
                continue
            path = DATA_RAW_DIR / f"{slugify(page_title)}.txt"
            path.write_text(extract, encoding="utf-8")
            saved += 1
        time.sleep(1)

    print(f"Saved {saved} articles to {DATA_RAW_DIR}")
    if skipped:
        print(f"Skipped {len(skipped)} (missing or empty): {skipped}", file=sys.stderr)


if __name__ == "__main__":
    main()
