"""
=============================================================================
SHL AI Assessment Recommendation Engine — Streamlit Application
=============================================================================
A modern, professional Streamlit web application that recommends the most
relevant SHL assessments based on a natural-language job description or query.

Architecture:
  1. Loads pre-built FAISS index + metadata (from data/ directory)
  2. Encodes user query with sentence-transformers/all-MiniLM-L6-v2
  3. Performs cosine similarity search via FAISS (Inner Product on L2-normed vectors)
  4. Optionally re-ranks results to balance Knowledge (K) vs Personality (P) tests
  5. Displays top-K results in a beautiful card layout

Run:  streamlit run app.py
=============================================================================
"""

import os
import re
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.pkl")
CSV_PATH = os.path.join(DATA_DIR, "assessments.csv")


# ---------------------------------------------------------------------------
# Streamlit page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SHL AI Assessment Recommendation Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Custom CSS for premium look & feel
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* ---------- Global ---------- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ---------- Header Banner ---------- */
    .hero-banner {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
    }
    .hero-banner h1 {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    .hero-banner p {
        color: #b8b8d1;
        font-size: 1.05rem;
        font-weight: 300;
    }

    /* ---------- Result Card ---------- */
    .result-card {
        background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    }
    .result-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(99, 102, 241, 0.2);
        border-color: rgba(99, 102, 241, 0.3);
    }
    .result-card h3 {
        color: #e0e0ff;
        font-size: 1.15rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }
    .result-card h3 a {
        color: #818cf8;
        text-decoration: none;
    }
    .result-card h3 a:hover {
        color: #a5b4fc;
        text-decoration: underline;
    }

    /* ---------- Badges ---------- */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-bottom: 0.3rem;
    }
    .badge-score {
        background: linear-gradient(135deg, #10b981, #059669);
        color: #fff;
    }
    .badge-type-K {
        background: rgba(99, 102, 241, 0.2);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    .badge-type-P {
        background: rgba(244, 114, 182, 0.2);
        color: #f472b6;
        border: 1px solid rgba(244, 114, 182, 0.3);
    }
    .badge-type-A {
        background: rgba(251, 191, 36, 0.2);
        color: #fbbf24;
        border: 1px solid rgba(251, 191, 36, 0.3);
    }
    .badge-type-B {
        background: rgba(34, 211, 238, 0.2);
        color: #22d3ee;
        border: 1px solid rgba(34, 211, 238, 0.3);
    }
    .badge-type-S {
        background: rgba(163, 230, 53, 0.2);
        color: #a3e635;
        border: 1px solid rgba(163, 230, 53, 0.3);
    }
    .badge-type-C {
        background: rgba(251, 146, 60, 0.2);
        color: #fb923c;
        border: 1px solid rgba(251, 146, 60, 0.3);
    }
    .badge-type-D {
        background: rgba(192, 132, 252, 0.2);
        color: #c084fc;
        border: 1px solid rgba(192, 132, 252, 0.3);
    }
    .badge-type-E {
        background: rgba(248, 113, 113, 0.2);
        color: #f87171;
        border: 1px solid rgba(248, 113, 113, 0.3);
    }

    .reason-text {
        color: #9ca3af;
        font-size: 0.9rem;
        margin-top: 0.6rem;
        line-height: 1.5;
    }

    /* ---------- Metrics Row ---------- */
    .metric-container {
        background: linear-gradient(145deg, #1e1b4b 0%, #312e81 100%);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid rgba(99, 102, 241, 0.2);
    }
    .metric-value {
        color: #818cf8;
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-label {
        color: #9ca3af;
        font-size: 0.85rem;
        margin-top: 0.2rem;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 100%);
    }

    /* ---------- Hide Streamlit branding ---------- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* ---------- Button ---------- */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Load resources (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    """Load the sentence-transformer model (cached across reruns)."""
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def load_faiss_index():
    """Load the pre-built FAISS index."""
    if not os.path.exists(FAISS_INDEX_PATH):
        return None
    return faiss.read_index(FAISS_INDEX_PATH)


@st.cache_resource
def load_metadata():
    """Load assessment metadata from pickle, with CSV fallback."""
    data = None
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "rb") as f:
            data = pickle.load(f)
    # Fallback: load from CSV
    if data is None and os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH, encoding="utf-8")
        df = df.fillna("")  # Replace NaN with empty strings
        data = df.to_dict("records")
    if data is None:
        return None
    # Ensure every item has rich_text (graceful fallback)
    for item in data:
        # Convert any remaining NaN-like values
        for key in item:
            if not isinstance(item[key], str) and key != "score":
                item[key] = str(item[key]) if item[key] else ""
        if not item.get("rich_text", "").strip():
            item["rich_text"] = (
                f"Title: {item.get('name', '')}\n"
                f"Test Types: {item.get('test_types', '')}\n"
                f"Description: {item.get('description', '')}"
            )
    return data


# ---------------------------------------------------------------------------
# Keyword-based balance / boost logic
# ---------------------------------------------------------------------------
# Keywords that signal technical / knowledge-based roles
TECHNICAL_KEYWORDS = {
    "technical", "coding", "programming", "software", "developer", "engineer",
    "data", "algorithm", "database", "sql", "python", "java", "javascript",
    "c++", "machine learning", "ai", "cloud", "devops", "testing", "qa",
    "network", "security", "accounting", "finance", "analytics", "mathematics",
    "science", "statistics", "excel", "sap", "erp", "it", "infrastructure",
    "architecture", "framework", ".net", "html", "css", "api", "backend",
    "frontend", "system", "admin", "administration", "skills", "knowledge",
    "aptitude", "numerical", "verbal", "reasoning", "mechanical",
}

# Keywords that signal behavioral / personality-based roles
BEHAVIORAL_KEYWORDS = {
    "leadership", "team", "teamwork", "communication", "collaboration",
    "management", "interpersonal", "behavior", "behavioural", "personality",
    "motivation", "culture", "conflict", "emotional", "empathy", "coaching",
    "mentoring", "decision", "strategic", "vision", "influence", "negotiation",
    "customer", "service", "sales", "relationship", "adaptability", "resilience",
    "integrity", "values", "attitude", "work style", "cultural fit",
    "organizational", "supervisor", "manager", "director", "executive",
}


def detect_query_intent(query: str):
    """
    Analyze query to determine if it leans toward
    technical/knowledge (K) or behavioral/personality (P) assessments.
    Returns (tech_score, behav_score) as floats 0-1.
    """
    query_lower = query.lower()
    words = set(re.findall(r'\b\w+\b', query_lower))

    tech_hits = len(words.intersection(TECHNICAL_KEYWORDS))
    behav_hits = len(words.intersection(BEHAVIORAL_KEYWORDS))

    # Also check multi-word phrases
    for kw in TECHNICAL_KEYWORDS:
        if " " in kw and kw in query_lower:
            tech_hits += 1
    for kw in BEHAVIORAL_KEYWORDS:
        if " " in kw and kw in query_lower:
            behav_hits += 1

    total = tech_hits + behav_hits
    if total == 0:
        return 0.5, 0.5

    return tech_hits / total, behav_hits / total


def generate_reason(assessment: dict, query: str, score: float):
    """
    Generate a simple keyword-based reason for why this assessment
    was recommended. Uses overlap between query terms and assessment text.
    """
    query_words = set(re.findall(r'\b\w{3,}\b', query.lower()))
    rich = assessment.get("rich_text", "").lower()
    name = assessment.get("name", "").lower()
    desc = assessment.get("description", "").lower()

    combined = rich + " " + name + " " + desc

    # Find overlapping meaningful words
    overlap = []
    for w in query_words:
        if w in combined and w not in {"the", "and", "for", "with", "that", "this"}:
            overlap.append(w)

    test_types = assessment.get("test_types", "")
    type_map = {
        "K": "Knowledge & Skills",
        "P": "Personality & Behavior",
        "A": "Ability & Aptitude",
        "B": "Biodata & Situational Judgement",
        "S": "Simulations",
        "C": "Competencies",
        "D": "Development & 360",
        "E": "Assessment Exercises",
    }
    type_desc = ", ".join(
        type_map.get(t, t) for t in test_types.split() if t in type_map
    )

    if overlap:
        kw_str = ", ".join(sorted(overlap)[:5])
        reason = f"Matches on: {kw_str}."
    else:
        reason = f"Semantically similar to your query (score: {score:.2f})."

    if type_desc:
        reason += f" Covers: {type_desc}."

    return reason


# ---------------------------------------------------------------------------
# Core recommendation function
# ---------------------------------------------------------------------------
def recommend(query: str, top_k: int = 8, balance: bool = True) -> list:
    """
    Core recommendation engine.

    Args:
        query:   Natural-language query or job description.
        top_k:   Number of results to return.
        balance: If True, re-rank to balance K vs P tests based on query intent.

    Returns:
        List of dicts: [{name, url, score, test_types, reason, ...}, ...]
    """
    model = load_model()
    index = load_faiss_index()
    metadata = load_metadata()

    if index is None or metadata is None:
        st.error("⚠️ Data not loaded. Please run `scraper.py` first to build the index.")
        return []

    # Encode query (normalized for cosine similarity via inner product)
    query_embedding = model.encode([query], normalize_embeddings=True)
    query_embedding = np.array(query_embedding, dtype="float32")

    # Search — fetch more than top_k if balance is on, to allow re-ranking
    search_k = min(top_k * 3, index.ntotal) if balance else top_k
    scores, indices = index.search(query_embedding, search_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        item = metadata[idx].copy()
        item["score"] = float(score)
        item["reason"] = generate_reason(item, query, float(score))
        results.append(item)

    # --- Balance re-ranking ---
    if balance and results:
        tech_ratio, behav_ratio = detect_query_intent(query)

        for r in results:
            types = r.get("test_types", "")
            boost = 0.0
            if "K" in types and tech_ratio > 0.5:
                boost += 0.05 * tech_ratio
            if "A" in types and tech_ratio > 0.5:
                boost += 0.03 * tech_ratio
            if "P" in types and behav_ratio > 0.5:
                boost += 0.05 * behav_ratio
            if "B" in types and behav_ratio > 0.5:
                boost += 0.03 * behav_ratio
            r["score"] = min(r["score"] + boost, 1.0)

        results.sort(key=lambda x: x["score"], reverse=True)

    # Clamp scores to [0, 1]
    for r in results:
        r["score"] = max(0.0, min(1.0, r["score"]))

    return results[:top_k]


# ---------------------------------------------------------------------------
# Test query auto-evaluation (for the 9 official test queries)
# ---------------------------------------------------------------------------
TEST_QUERIES = [
    "I am hiring for Java developers who can also handle customer calls. Suggest me some assessments.",
    "Looking for an assessment to evaluate leadership skills of mid-level managers.",
    "I need a personality test for entry-level sales representatives.",
    "Suggest assessments for hiring data analysts with SQL and Python skills.",
    "What assessments are available for evaluating teamwork and collaboration?",
    "I want to test numerical reasoning and problem-solving abilities.",
    "Recommend assessments for senior executive strategic decision-making.",
    "Looking for simulations or situational judgment tests for customer service roles.",
    "Suggest knowledge tests for .NET and cloud technologies.",
]


def generate_test_predictions():
    """Generate predictions for the 9 test queries and return as DataFrame."""
    rows = []
    for q in TEST_QUERIES:
        recs = recommend(q, top_k=10, balance=True)
        for r in recs:
            rows.append({
                "query": q,
                "assessment_name": r.get("name", ""),
                "assessment_url": r.get("url", ""),
                "relevance_score": round(r.get("score", 0), 4),
                "test_types": r.get("test_types", ""),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# UI Layout
# ---------------------------------------------------------------------------
def render_app():
    # Hero Banner
    st.markdown("""
    <div class="hero-banner">
        <h1>🧠 SHL AI Assessment Recommendation Engine</h1>
        <p>Powered by Semantic Search • sentence-transformers • FAISS</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        st.markdown("---")

        top_k = st.slider(
            "Number of results",
            min_value=5,
            max_value=10,
            value=8,
            step=1,
            help="How many assessment recommendations to show",
        )

        balance = st.toggle(
            "Balance K ↔ P tests",
            value=True,
            help="Boost Knowledge tests for technical queries, Personality tests for behavioral queries",
        )

        st.markdown("---")
        st.markdown("### 📊 Test Type Legend")
        type_legend = {
            "K": "Knowledge & Skills",
            "P": "Personality & Behavior",
            "A": "Ability & Aptitude",
            "B": "Biodata & SJT",
            "S": "Simulations",
            "C": "Competencies",
            "D": "Development & 360",
            "E": "Exercises",
        }
        for code, desc in type_legend.items():
            st.markdown(f"**`{code}`** — {desc}")

        st.markdown("---")
        st.markdown("### 🧪 Auto-Test Queries")
        if st.button("Generate test_predictions.csv", key="gen_test"):
            with st.spinner("Running 9 test queries..."):
                df = generate_test_predictions()
                csv_data = df.to_csv(index=False)
                st.download_button(
                    "📥 Download test_predictions.csv",
                    csv_data,
                    "test_predictions.csv",
                    "text/csv",
                    key="dl_test",
                )
                st.success(f"✅ Generated {len(df)} predictions across 9 queries")

        st.markdown("---")
        st.markdown(
            "<div style='text-align:center; color:#6b7280; font-size:0.8rem;'>"
            "Built for SHL AI Intern Assignment<br>© 2025"
            "</div>",
            unsafe_allow_html=True,
        )

    # Main content - Query Input
    st.markdown("### 🔍 Enter your query")
    col_input, col_btn = st.columns([4, 1])

    with col_input:
        query = st.text_area(
            "Describe the role or paste a job description",
            height=120,
            placeholder=(
                "Example: I am hiring for Java developers who can also handle "
                "customer calls. Suggest me some assessments."
            ),
            label_visibility="collapsed",
        )

    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        search_clicked = st.button("🚀 Get Recommendations", type="primary", use_container_width=True)

    # Quick example queries
    st.markdown("**Quick examples:**")
    example_cols = st.columns(3)
    example_queries = [
        "Java developers with customer call handling",
        "Leadership assessment for mid-level managers",
        "Personality test for entry-level sales",
    ]

    for i, eq in enumerate(example_queries):
        with example_cols[i]:
            if st.button(f"💡 {eq}", key=f"ex_{i}", use_container_width=True):
                query = eq
                search_clicked = True

    # Process & display results
    if search_clicked and query.strip():
        st.markdown("---")

        with st.spinner("🔄 Searching through assessments..."):
            results = recommend(query.strip(), top_k=top_k, balance=balance)

        if not results:
            st.warning("No results found. Please check that the scraper has been run.")
            return

        # Metrics row
        st.markdown("### 📋 Results")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(
                f'<div class="metric-container">'
                f'<div class="metric-value">{len(results)}</div>'
                f'<div class="metric-label">Assessments Found</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with m2:
            avg_score = np.mean([r["score"] for r in results])
            st.markdown(
                f'<div class="metric-container">'
                f'<div class="metric-value">{avg_score:.2f}</div>'
                f'<div class="metric-label">Avg Relevance</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with m3:
            types_found = set()
            for r in results:
                for t in r.get("test_types", "").split():
                    types_found.add(t)
            st.markdown(
                f'<div class="metric-container">'
                f'<div class="metric-value">{len(types_found)}</div>'
                f'<div class="metric-label">Test Type Categories</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Result cards - two columns
        for i in range(0, len(results), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                idx = i + j
                if idx >= len(results):
                    break
                r = results[idx]
                with col:
                    # Build badges HTML
                    score_badge = f'<span class="badge badge-score">Score: {r["score"]:.2f}</span>'
                    type_badges = ""
                    for t in r.get("test_types", "").split():
                        type_badges += f'<span class="badge badge-type-{t}">{t}</span>'

                    card_html = f"""
                    <div class="result-card">
                        <h3>#{idx + 1} <a href="{r.get('url', '#')}" target="_blank">{r.get('name', 'Unknown')}</a></h3>
                        <div>{score_badge} {type_badges}</div>
                        <p class="reason-text">{r.get('reason', '')}</p>
                    </div>
                    """
                    st.markdown(card_html, unsafe_allow_html=True)

                    # Expandable details
                    with st.expander("📄 View details"):
                        st.markdown(f"**URL:** [{r.get('url', '')}]({r.get('url', '')})")
                        if r.get("description"):
                            st.markdown(f"**Description:** {r['description'][:300]}...")
                        if r.get("duration"):
                            st.markdown(f"**Duration:** {r['duration']}")
                        if r.get("job_levels"):
                            st.markdown(f"**Job Levels:** {r['job_levels']}")
                        if r.get("test_types"):
                            st.markdown(f"**Test Types:** {r['test_types']}")

    elif search_clicked:
        st.warning("⚠️ Please enter a query to search.")


# ---------------------------------------------------------------------------
# Health check endpoint (for API/testing)
# ---------------------------------------------------------------------------
def health_check():
    """Simple health check — verifies all components are loadable."""
    status = {"status": "healthy", "components": {}}
    try:
        load_model()
        status["components"]["model"] = "ok"
    except Exception as e:
        status["components"]["model"] = f"error: {e}"
        status["status"] = "unhealthy"

    try:
        idx = load_faiss_index()
        status["components"]["faiss_index"] = "ok" if idx else "missing"
    except Exception as e:
        status["components"]["faiss_index"] = f"error: {e}"
        status["status"] = "unhealthy"

    try:
        meta = load_metadata()
        status["components"]["metadata"] = f"ok ({len(meta)} items)" if meta else "missing"
    except Exception as e:
        status["components"]["metadata"] = f"error: {e}"
        status["status"] = "unhealthy"

    return status


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    render_app()
