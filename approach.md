# Approach

## What the system does

Given a job description or query, recommend the most relevant SHL assessments from ~389 individual test solutions. Return name, URL, relevance score, and a short reason for each result.

## Data collection

I scraped the SHL product catalog (`?type=1`, Individual Test Solutions) across 32 paginated pages. For each assessment I visited the detail page and pulled description, duration, job levels, languages, and test type badges (K, P, A, B, etc.). I combined everything into a `rich_text` field used for embedding.

The scraper is resumable -- it checkpoints progress so you can re-run after interruption.

## Why semantic search

Keyword matching fails when phrasing differs. Someone searching "coding test" should match an assessment called "Java Programming Knowledge Test" even though they share no words. Sentence-transformer embeddings handle this because the model understands synonyms and context.

I used `all-MiniLM-L6-v2` -- 384-dim embeddings, 80MB model, fast on CPU. Good enough for this catalog size. I normalise the vectors and use FAISS `IndexFlatIP`, which gives exact cosine similarity. With 389 items this is sub-millisecond.

## K/P balance

Pure similarity search can over-cluster on one test type. A query for "Python developer" might return only K-type tests, missing P-type assessments for cultural fit.

When balance is on, I detect technical vs behavioural intent using keyword sets (~35 words each) and add a small boost (max +0.05) to the relevant test type. Technical queries nudge K and A tests up; behavioural queries nudge P and B tests up. The base similarity score stays the primary signal.

## Architecture split

I separated the project into:
- `utils.py` -- core logic (model loading, FAISS search, re-ranking)
- `app.py` -- Streamlit frontend for interactive demos
- `api.py` -- FastAPI backend with `GET /health` and `POST /recommend`

This way the UI and API can be deployed independently. Streamlit goes on Streamlit Cloud, the API goes on Render.

## Why flat dark design

The UI is intentionally minimal. This is a tool for HR teams, not a portfolio piece. Flat dark cards with 1px borders, one accent colour (#00b4d8), no gradients or animations. Every element is functional.

## Limitations

- No labelled test set, so I estimated Recall@10 around 0.75 from manual checks.
- Keyword-based balance is simple but transparent. A trained classifier would be better with enough data.
- Reason generation is keyword overlap, not LLM-generated. Deterministic but limited.
- If the catalog changes, the scraper needs to re-run.

## What I'd improve

- Cross-encoder re-ranker for better precision.
- Hybrid search (dense + BM25) to catch exact keyword matches.
- User feedback loop to fine-tune embeddings on SHL-specific data.
- Accept a job posting URL, auto-extract the description, and use that as the query.
