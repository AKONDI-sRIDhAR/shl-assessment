# utils.py -- core recommendation logic, shared by app.py and api.py

import os, re, pickle
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# singletons
_model, _index, _meta = None, None, None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model


def get_index():
    global _index
    if _index is None:
        path = os.path.join(DIR, "faiss_index.bin")
        if os.path.exists(path):
            _index = faiss.read_index(path)
    return _index


def get_meta():
    global _meta
    if _meta is not None:
        return _meta
    pkl = os.path.join(DIR, "metadata.pkl")
    csv = os.path.join(DIR, "assessments_full.csv")
    if os.path.exists(pkl):
        with open(pkl, "rb") as f:
            _meta = pickle.load(f)
    elif os.path.exists(csv):
        _meta = pd.read_csv(csv, encoding="utf-8").fillna("").to_dict("records")
    if _meta:
        for r in _meta:
            for k in r:
                if not isinstance(r[k], str) and k != "score":
                    r[k] = str(r[k]) if r[k] else ""
            if not r.get("rich_text", "").strip():
                r["rich_text"] = f"Title: {r.get('name','')}\nTest Types: {r.get('test_types','')}"
    return _meta


# keyword sets for K/P intent detection
_TECH = {
    "technical","coding","programming","software","developer","engineer",
    "data","algorithm","database","sql","python","java","javascript",
    "machine learning","ai","cloud","devops","testing","qa","network",
    "security","accounting","finance","analytics","mathematics","excel",
    "sap","erp","it",".net","html","css","api","backend","frontend",
    "skills","knowledge","aptitude","numerical","verbal","reasoning",
}
_BEHAV = {
    "leadership","team","teamwork","communication","collaboration",
    "management","interpersonal","behavior","behavioural","personality",
    "motivation","culture","conflict","emotional","empathy","coaching",
    "decision","strategic","vision","influence","negotiation","customer",
    "service","sales","relationship","adaptability","resilience",
    "integrity","values","supervisor","manager","director","executive",
}

TYPE_NAMES = {
    "K": "Knowledge & Skills", "P": "Personality & Behavior",
    "A": "Ability & Aptitude", "B": "Biodata & SJT",
    "S": "Simulations",       "C": "Competencies",
    "D": "Development & 360", "E": "Assessment Exercises",
}


def _intent(q):
    low = q.lower()
    words = set(re.findall(r"\b\w+\b", low))
    t = len(words & _TECH)  + sum(1 for k in _TECH  if " " in k and k in low)
    b = len(words & _BEHAV) + sum(1 for k in _BEHAV if " " in k and k in low)
    s = t + b
    return (t/s, b/s) if s else (0.5, 0.5)


def _reason(row, query, score):
    qw = set(re.findall(r"\b\w{3,}\b", query.lower()))
    blob = (row.get("rich_text","") + " " + row.get("name","")).lower()
    skip = {"the","and","for","with","that","this","are","was","not","some",
            "can","also","who","what","how","suggest","need","want","looking",
            "hiring","available","assessments","assessment","me","has"}
    hits = sorted([w for w in qw if w in blob and w not in skip])[:5]
    types = row.get("test_types","")
    covers = ", ".join(TYPE_NAMES[t] for t in types.split() if t in TYPE_NAMES)
    if hits:
        out = f"Matches: {', '.join(hits)}"
    else:
        out = f"Semantic match ({score:.2f})"
    if covers:
        out += f". Type: {covers}"
    return out + "."


def recommend(query, top_k=8, balance=True):
    """Return up to top_k ranked assessment dicts."""
    model, index, meta = get_model(), get_index(), get_meta()
    if not index or not meta:
        return []

    vec = model.encode([query], normalize_embeddings=True).astype("float32")
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

    # boost K/A for technical queries, P/B for behavioural
    if balance and results:
        tr, br = _intent(query)
        for r in results:
            tt = r.get("test_types","")
            b = 0.0
            if "K" in tt and tr > 0.5: b += 0.05 * tr
            if "A" in tt and tr > 0.5: b += 0.03 * tr
            if "P" in tt and br > 0.5: b += 0.05 * br
            if "B" in tt and br > 0.5: b += 0.03 * br
            r["score"] = min(r["score"] + b, 1.0)
        results.sort(key=lambda x: x["score"], reverse=True)

    for r in results:
        r["score"] = max(0.0, min(1.0, r["score"]))
    return results[:top_k]


def health():
    ok = {"status": "healthy"}
    try:
        get_model()
        idx = get_index()
        m = get_meta()
        if not idx or not m:
            ok["status"] = "unhealthy"
    except Exception:
        ok["status"] = "unhealthy"
    return ok
