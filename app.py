# app.py -- Streamlit UI for the SHL recommendation engine
# Run: streamlit run app.py

import streamlit as st
import pandas as pd
from utils import recommend, TYPE_NAMES

st.set_page_config(
    page_title="SHL Assessment Recommendation Engine",
    page_icon="https://www.shl.com/favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded",
)

# flat dark styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hdr { background:#1a1a1a; border-bottom:1px solid #333; padding:1.4rem; margin-bottom:1.4rem; }
.hdr h1 { color:#fff; font-size:1.5rem; font-weight:600; margin:0 0 .15rem; }
.hdr p  { color:#888; font-size:.83rem; margin:0; }

.card { background:#1a1a1a; border:1px solid #333; padding:1.1rem 1.2rem; margin-bottom:.7rem; }
.card h3 { color:#fff; font-size:.95rem; font-weight:600; margin:0 0 .25rem; }
.card h3 a { color:#00b4d8; text-decoration:none; }
.card h3 a:hover { text-decoration:underline; }
.card .rsn { color:#999; font-size:.8rem; margin-top:.35rem; line-height:1.4; }

.p { display:inline-block; padding:.12rem .45rem; font-size:.68rem; font-weight:600;
     margin-right:.2rem; border:1px solid #444; color:#ccc; }
.p-sc { background:#00b4d8; color:#000; border-color:#00b4d8; }
.p-K { background:#1b5e20; color:#a5d6a7; border-color:#2e7d32; }
.p-P { background:#4a148c; color:#ce93d8; border-color:#6a1b9a; }
.p-A { background:#e65100; color:#ffcc80; border-color:#ef6c00; }
.p-B { background:#006064; color:#80deea; border-color:#00838f; }
.p-S { background:#33691e; color:#c5e1a5; border-color:#558b2f; }
.p-C { background:#bf360c; color:#ffab91; border-color:#d84315; }
.p-D { background:#311b92; color:#b39ddb; border-color:#4527a0; }
.p-E { background:#880e4f; color:#f48fb1; border-color:#ad1457; }

.stButton > button { background:#00b4d8; color:#000; border:none;
    padding:.45rem 1rem; font-weight:600; font-size:.85rem; border-radius:0; }
.stButton > button:hover { background:#0096b7; }

#MainMenu, footer, header { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# header
st.markdown("""
<div class="hdr">
    <h1>SHL Assessment Recommendation Engine</h1>
    <p>Semantic search over 389 SHL individual test solutions</p>
</div>""", unsafe_allow_html=True)

# sidebar
with st.sidebar:
    st.markdown("**Settings**")
    top_k = st.slider("Results", 5, 10, 8)
    balance = st.toggle("Balance K / P", value=True,
                        help="Adjust ranking when query is technical vs behavioural")
    st.markdown("---")
    st.markdown("**Test types**")
    for c, l in TYPE_NAMES.items():
        st.markdown(f"`{c}` {l}")
    st.markdown("---")
    st.caption("SHL AI Intern Assignment 2025-26")

# query input
query = st.text_area("Job description or query", height=90,
                     placeholder="e.g. I need assessments for Java developers who handle customer calls")
go = st.button("Recommend", type="primary")

# quick examples
cols = st.columns(3)
examples = ["Java developers with customer handling",
            "Leadership for mid-level managers",
            "Personality test for entry-level sales"]
for i, ex in enumerate(examples):
    with cols[i]:
        if st.button(ex, key=f"e{i}", use_container_width=True):
            query, go = ex, True

# show results
if go and query.strip():
    st.markdown("---")
    with st.spinner("Searching..."):
        results = recommend(query.strip(), top_k=top_k, balance=balance)
    if not results:
        st.warning("No results. Run scraper.py first.")
    else:
        st.markdown(f"**{len(results)} results** for: _{query.strip()}_")
        for i in range(0, len(results), 2):
            c = st.columns(2)
            for j, col in enumerate(c):
                idx = i + j
                if idx >= len(results): break
                r = results[idx]
                with col:
                    sp = f'<span class="p p-sc">{r["score"]:.2f}</span>'
                    tp = "".join(f'<span class="p p-{t}">{t}</span>'
                                for t in r.get("test_types","").split())
                    st.markdown(f"""<div class="card">
                        <h3><a href="{r.get('url','#')}" target="_blank">{r.get('name','')}</a></h3>
                        <div>{sp} {tp}</div>
                        <p class="rsn">{r.get('reason','')}</p>
                    </div>""", unsafe_allow_html=True)
                    with st.expander("Details"):
                        st.markdown(f"[{r.get('url','')}]({r.get('url','')})")
                        if r.get("description"): st.text(r["description"][:250])
                        if r.get("duration"): st.markdown(f"Duration: {r['duration']}")
                        if r.get("job_levels"): st.markdown(f"Job levels: {r['job_levels']}")
elif go:
    st.warning("Enter a query.")
