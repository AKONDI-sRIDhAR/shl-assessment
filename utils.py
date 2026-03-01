"""
utils.py -- shared recommendation logic
Used by both the Streamlit UI (app.py) and the FastAPI backend (api.py).
"""

import os
import re
import pickle
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# paths
DATA_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FAISS_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
META_PATH  = os.path.join(DATA_DIR, "metadata.pkl")
CSV_PATH   = os.path.join(DATA_DIR, "assessments_full.csv")

# singletons (loaded once, reused)
_model = None
_index = None
_metadata = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model


def get_index():
    global _index
    if _index is None and os.path.exists(FAISS_PATH):
        _index = faiss.read_index(FAISS_PATH)
    return _index


def get_metadata():
    global _metadata
    if _metadata is not None:
        return _metadata

    data = None
    if os.path.exists(META_PATH):
        with open(META_PATH, "rb") as f:
            data = pickle.load(f)
    if data is None and os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH, encoding="utf-8").fillna("")
        data = df.to_dict("records")
    if data is None:
        return None

    # normalise fields
    for row in data:
        for k in row:
            if not isinstance(row[k], str) and k != "score":
                row[k] = str(row[k]) if row[k] else ""
        if not row.get("rich_text", "").strip():
            row["rich_text"] = (
                f"Title: {row.get('name', '')}\n"
                f"Test Types: {row.get('test_types', '')}\n"
                f"Description: {row.get('description', '')}"
            )

    _metadata = data
    return _metadata


# -- keyword sets for K/P balance --

TECH_KW = {
    "technical", "coding", "programming", "software", "developer", "engineer",
    "data", "algorithm", "database", "sql", "python", "java", "javascript",
    "c++", "machine learning", "ai", "cloud", "devops", "testing", "qa",
    "network", "security", "accounting", "finance", "analytics", "mathematics",
    "science", "statistics", "excel", "sap", "erp", "it", "infrastructure",
    "architecture", "framework", ".net", "html", "css", "api", "backend",
    "frontend", "system", "admin", "skills", "knowledge", "aptitude",
    "numerical", "verbal", "reasoning", "mechanical",
}

BEHAV_KW = {
    "leadership", "team", "teamwork", "communication", "collaboration",
    "management", "interpersonal", "behavior", "behavioural", "personality",
    "motivation", "culture", "conflict", "emotional", "empathy", "coaching",
    "mentoring", "decision", "strategic", "vision", "influence", "negotiation",
    "customer", "service", "sales", "relationship", "adaptability", "resilience",
    "integrity", "values", "attitude", "work style", "cultural fit",
    "organizational", "supervisor", "manager", "director", "executive",
}

TYPE_LABELS = {
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "A": "Ability & Aptitude",
    "B": "Biodata & SJT",
    "S": "Simulations",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
}


def _intent(query):
    """Classify query as technical vs behavioural. Returns (tech_ratio, behav_ratio)."""
    low = query.lower()
    words = set(re.findall(r"\b\w+\b", low))
    t = len(words & TECH_KW)  + sum(1 for k in TECH_KW  if " " in k and k in low)
    b = len(words & BEHAV_KW) + sum(1 for k in BEHAV_KW if " " in k and k in low)
    s = t + b
    return (t / s, b / s) if s else (0.5, 0.5)


def _reason(row, query, score):
    """Short explanation for why this assessment was recommended."""
    qwords = set(re.findall(r"\b\w{3,}\b", query.lower()))
    blob = (row.get("rich_text", "") + " " + row.get("name", "")).lower()
    stop = {"the", "and", "for", "with", "that", "this", "are", "was", "has",
            "not", "some", "can", "also", "who", "what", "how", "suggest",
            "need", "want", "looking", "hiring", "available", "assessments",
            "assessment", "me"}
    hits = sorted([w for w in qwords if w in blob and w not in stop])[:5]

    types = row.get("test_types", "")
    covers = ", ".join(TYPE_LABELS.get(t, t) for t in types.split() if t in TYPE_LABELS)

    parts = []
    if hits:
        parts.append(f"Matches: {', '.join(hits)}")
    else:
        parts.append(f"Semantic similarity ({score:.2f})")
    if covers:
        parts.append(f"Type: {covers}")
    return ". ".join(parts) + "."


# -- core function --

def recommend(query: str, top_k: int = 8, balance: bool = True) -> list:
    """
    Returns up to top_k assessment dicts sorted by relevance.
    Each dict has: name, url, score, test_types, reason, description, etc.
    """
    model = get_model()
    index = get_index()
    meta = get_metadata()

    if index is None or meta is None:
        return []

    vec = model.encode([query], normalize_embeddings=True).astype("float32")

    # pull extra candidates when balancing so we can re-rank
    k = min(top_k * 3, index.ntotal) if balance else top_k
    scores, ids = index.search(vec, k)

    results = []
    for s, i in zip(scores[0], ids[0]):
        if i < 0 or i >= len(meta):
            continue
        item = meta[i].copy()
        item["score"] = float(s)
        item["reason"] = _reason(item, query, float(s))
        results.append(item)

    # K/P boost
    if balance and results:
        tr, br = _intent(query)
        for r in results:
            tt = r.get("test_types", "")
            boost = 0.0
            if "K" in tt and tr > 0.5: boost += 0.05 * tr
            if "A" in tt and tr > 0.5: boost += 0.03 * tr
            if "P" in tt and br > 0.5: boost += 0.05 * br
            if "B" in tt and br > 0.5: boost += 0.03 * br
            r["score"] = min(r["score"] + boost, 1.0)
        results.sort(key=lambda x: x["score"], reverse=True)

    for r in results:
        r["score"] = max(0.0, min(1.0, r["score"]))

    return results[:top_k]


def health_check():
    """Returns a dict describing system health."""
    status = {"status": "healthy", "components": {}}
    try:
        get_model()
        status["components"]["model"] = "ok"
    except Exception as e:
        status["components"]["model"] = str(e)
        status["status"] = "unhealthy"
    try:
        idx = get_index()
        status["components"]["faiss"] = "ok" if idx else "missing"
    except Exception as e:
        status["components"]["faiss"] = str(e)
        status["status"] = "unhealthy"
    try:
        m = get_metadata()
        status["components"]["metadata"] = f"ok ({len(m)} items)" if m else "missing"
    except Exception as e:
        status["components"]["metadata"] = str(e)
        status["status"] = "unhealthy"
    return status
