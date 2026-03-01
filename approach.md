# Approach

## Problem

Given a natural-language query -- a job title, skill list, or full job
description -- recommend the most relevant assessments from SHL's catalog of
~390 individual test solutions.  Return a ranked list with scores and handle
everything from purely technical roles to behavioural/personality roles.

---

## Architecture

I split the system into three layers:

- **utils.py** -- core recommendation logic (model loading, FAISS search,
  K/P balance, reason generation).  Imported by both the UI and the API.
- **app.py** -- Streamlit frontend for interactive use.  HR professionals
  type a query, see ranked cards with clickable SHL links.
- **api.py** -- FastAPI backend with `GET /health` and `POST /recommend`
  endpoints.  This is the programmatic interface that automated evaluators
  or downstream systems can call.

Splitting UI from API means either can be deployed independently.  The
Streamlit app goes on Streamlit Cloud; the API can go on Render, Railway,
or any container host.

### Data Pipeline

I scraped the SHL Product Catalog filtered to "Individual Test Solutions"
(`?type=1`), 32 paginated pages, ~389 assessments.  For each one I visited
the detail page and extracted description, duration, job levels, languages,
and test-type badges (K, P, A, B, etc.).

All fields are combined into a `rich_text` string:
```
Title: {name}
Description: {description}
Test Types: {test_types}
Duration: {duration}
Job Levels: {job_levels}
```

This string gets embedded.

---

## Why Embeddings + FAISS

Keyword matching breaks when phrasing differs.  "Coding test" should match
"Java Programming Knowledge Assessment" even though they share no exact
words.  Sentence-transformer embeddings handle this because the model
understands synonyms and context.

I chose **all-MiniLM-L6-v2**: 384 dimensions, 80 MB, 82.8% STS benchmark.
Good balance of quality and speed -- fast enough for real-time queries on
CPU, small enough to deploy on free-tier hosting.

For the index I used FAISS `IndexFlatIP` on L2-normalised vectors.  Inner
product on normalised vectors equals cosine similarity.  With ~389 items
exact search is sub-millisecond; no need for approximate methods.

---

## K/P Balance

Pure cosine similarity can cluster results toward one test type.  Searching
"Python developer" might return only Knowledge tests, missing Personality
tests for cultural fit.

When the balance toggle is on, I classify the query into a technical ratio
and a behavioural ratio using two keyword sets (~35 words each).  Then I add
a small boost:

- Technical queries: K and A results get up to +0.05
- Behavioural queries: P and B results get up to +0.05

The boost is proportional to intent strength.  A mixed query like "Java
developer who handles customer calls" gets moderate nudges both ways.  The
base similarity score stays the primary signal.

---

## Why a Minimal UI

I went with a flat, dark, no-decoration interface.  The reasoning:

- This is a tool for HR teams, not a portfolio site.  Readability over
  aesthetics.
- Flat cards with 1px borders scan faster than cards with shadows and
  rounded corners.
- Single accent colour (#00b4d8) keeps the interface consistent without
  visual noise.
- No gradients, no animations, no glassmorphism.  Every pixel is
  functional.

The design matches SHL's own product pages -- clean, corporate, focused on
data.

---

## Estimated Performance

| Metric | Estimated |
|--------|-----------|
| Recall@3 | ~0.65 |
| Recall@5 | ~0.70 |
| Recall@10 | ~0.78 |
| MAP@10 | ~0.60 |

Specific queries ("knowledge tests for .NET") perform best.  Vague queries
("good test for graduates") are harder because many assessments are
potentially relevant.

---

## Trade-Offs

| Decision | Why |
|----------|-----|
| MiniLM-L6 over larger models | Deploys on free-tier hosting, marginal quality loss |
| Exact FAISS over approximate | 389 items, no need for approximation |
| Keyword balance over ML classifier | Transparent, no training data needed |
| UI + API split over monolith | Each deploys independently, cleaner separation |
| Flat design over flashy | Matches corporate expectations, faster to read |

---

## Future Improvements

1. **Cross-encoder re-ranker** -- second-stage model that jointly encodes
   query + assessment for better ranking precision.
2. **RAG with Gemini** -- generate natural-language explanations per result
   instead of keyword-overlap reasons.
3. **Hybrid search** -- combine embeddings with BM25 via Reciprocal Rank
   Fusion to catch exact-match keywords.
4. **Feedback loop** -- let users rate results, fine-tune embeddings on
   SHL-specific data.
5. **JD URL ingestion** -- accept a URL, scrape the job description, use it
   as the query automatically.

---

_Written for the SHL AI Intern / Research Engineer assignment, Feb 2026._
