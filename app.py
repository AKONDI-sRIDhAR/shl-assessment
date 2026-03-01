"""
app.py -- Streamlit UI for the SHL Assessment Recommendation Engine
Run:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
from utils import recommend, TYPE_LABELS

# -- page config --
st.set_page_config(
    page_title="SHL Assessment Recommendation Engine",
    page_icon="https://www.shl.com/favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- flat dark CSS --
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.header {
    background: #1a1a1a;
    border-bottom: 1px solid #333;
    padding: 1.6rem 1.4rem;
    margin-bottom: 1.6rem;
}
.header h1 { color: #fff; font-size: 1.55rem; font-weight: 600; margin: 0 0 .2rem; }
.header p  { color: #888; font-size: .85rem; margin: 0; }

.card {
    background: #1a1a1a;
    border: 1px solid #333;
    padding: 1.2rem 1.3rem;
    margin-bottom: .75rem;
}
.card h3 { color: #fff; font-size: .98rem; font-weight: 600; margin: 0 0 .3rem; }
.card h3 a { color: #00b4d8; text-decoration: none; }
.card h3 a:hover { text-decoration: underline; }
.card .reason { color: #999; font-size: .82rem; margin-top: .4rem; line-height: 1.45; }

.pill {
    display: inline-block; padding: .13rem .5rem; font-size: .7rem;
    font-weight: 600; margin-right: .25rem; border: 1px solid #444; color: #ccc;
}
.pill-score { background: #00b4d8; color: #000; border-color: #00b4d8; }
.pill-K { background: #1b5e20; color: #a5d6a7; border-color: #2e7d32; }
.pill-P { background: #4a148c; color: #ce93d8; border-color: #6a1b9a; }
.pill-A { background: #e65100; color: #ffcc80; border-color: #ef6c00; }
.pill-B { background: #006064; color: #80deea; border-color: #00838f; }
.pill-S { background: #33691e; color: #c5e1a5; border-color: #558b2f; }
.pill-C { background: #bf360c; color: #ffab91; border-color: #d84315; }
.pill-D { background: #311b92; color: #b39ddb; border-color: #4527a0; }
.pill-E { background: #880e4f; color: #f48fb1; border-color: #ad1457; }

.stButton > button {
    background: #00b4d8; color: #000; border: none;
    padding: .5rem 1.2rem; font-weight: 600; font-size: .88rem; border-radius: 0;
}
.stButton > button:hover { background: #0096b7; }

#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# -- test queries for batch export --
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


def _batch_predictions():
    rows = []
    for q in TEST_QUERIES:
        for r in recommend(q, top_k=10, balance=True):
            rows.append({
                "query": q,
                "assessment_name": r.get("name", ""),
                "assessment_url": r.get("url", ""),
                "relevance_score": round(r.get("score", 0), 4),
                "test_types": r.get("test_types", ""),
            })
    return pd.DataFrame(rows)


# -- main UI --
def main():
    # header
    st.markdown("""
    <div class="header">
        <h1>SHL Assessment Recommendation Engine</h1>
        <p>Semantic search over 389 SHL assessments</p>
    </div>""", unsafe_allow_html=True)

    # sidebar
    with st.sidebar:
        st.markdown("**Settings**")
        top_k = st.slider("Number of results", 5, 10, 8)
        balance = st.toggle("Balance K / P tests", value=True,
                            help="Boost Knowledge tests for technical queries, "
                                 "Personality tests for behavioural queries.")
        st.markdown("---")
        st.markdown("**Test-Type Legend**")
        for code, label in TYPE_LABELS.items():
            st.markdown(f"`{code}` {label}")
        st.markdown("---")
        st.markdown("**Batch Predictions**")
        if st.button("Generate test_predictions.csv"):
            with st.spinner("Running 9 queries ..."):
                df = _batch_predictions()
                st.download_button("Download CSV", df.to_csv(index=False),
                                   "test_predictions.csv", "text/csv")
                st.success(f"{len(df)} predictions ready.")
        st.markdown("---")
        st.caption("SHL AI Intern Assignment | 2025-26")

    # query input
    query = st.text_area(
        "Describe the role or paste a job description",
        height=100,
        placeholder="e.g. I need assessments for Java developers who also handle customer calls.",
    )
    st.button("Get Recommendations", type="primary", key="go_btn")
    go = st.session_state.get("go_btn", False)

    # example buttons
    st.markdown("**Examples:**")
    ex_cols = st.columns(3)
    examples = [
        "Java developers with customer handling",
        "Leadership for mid-level managers",
        "Personality test for entry-level sales",
    ]
    for i, ex in enumerate(examples):
        with ex_cols[i]:
            if st.button(ex, key=f"ex{i}", use_container_width=True):
                query = ex
                go = True

    # results
    if go and query.strip():
        st.markdown("---")
        with st.spinner("Searching ..."):
            results = recommend(query.strip(), top_k=top_k, balance=balance)

        if not results:
            st.warning("No results. Run scraper.py first to build the index.")
            return

        st.markdown(f"**{len(results)} recommendations** for: _{query.strip()}_")

        for i in range(0, len(results), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                idx = i + j
                if idx >= len(results):
                    break
                r = results[idx]
                with col:
                    score_pill = f'<span class="pill pill-score">{r["score"]:.2f}</span>'
                    type_pills = "".join(
                        f'<span class="pill pill-{t}">{t}</span>'
                        for t in r.get("test_types", "").split()
                    )
                    st.markdown(f"""
                    <div class="card">
                        <h3><a href="{r.get('url','#')}" target="_blank">
                            {r.get('name','Unknown')}</a></h3>
                        <div>{score_pill} {type_pills}</div>
                        <p class="reason">{r.get('reason','')}</p>
                    </div>""", unsafe_allow_html=True)

                    with st.expander("Details"):
                        st.markdown(f"**URL:** [{r.get('url','')}]({r.get('url','')})")
                        if r.get("description"):
                            st.markdown(f"**Description:** {r['description'][:300]}")
                        if r.get("duration"):
                            st.markdown(f"**Duration:** {r['duration']}")
                        if r.get("job_levels"):
                            st.markdown(f"**Job Levels:** {r['job_levels']}")

    elif go:
        st.warning("Enter a query first.")


if __name__ == "__main__":
    main()
