# SHL Assessment Recommendation Engine

Recommends SHL assessments for a job description using semantic search.

## Structure

```
app.py               Streamlit UI
api.py               FastAPI backend (GET /health, POST /recommend)
utils.py             Shared recommendation logic
scraper.py           SHL catalog scraper + FAISS index builder
requirements.txt     Dependencies
.streamlit/config.toml  Dark theme
data/
  assessments.csv         Basic listing (name, url)
  assessments_full.csv    Enriched (descriptions, test types, etc.)
  faiss_index.bin         FAISS cosine similarity index
  metadata.pkl            Pickled metadata
```

## Run Locally

Install:
```
pip install -r requirements.txt
```

Scrape (first time, ~6 min):
```
python scraper.py
```

Streamlit UI:
```
streamlit run app.py
```
Opens at http://localhost:8501

API server:
```
uvicorn api:app --host 0.0.0.0 --port 8080
```
Docs at http://localhost:8080/docs

## Test the API

Health check:
```
curl http://localhost:8080/health
```
```json
{"status": "healthy"}
```

Recommend:
```
curl -X POST http://localhost:8080/recommend \
  -H "Content-Type: application/json" \
  -d '{"query": "Java developer who handles customer calls", "top_k": 5}'
```
```json
{
  "query": "Java developer who handles customer calls",
  "top_k": 5,
  "balance": true,
  "results": [
    {"rank": 1, "name": "Java 8 (New)", "url": "https://...", "score": 0.60, "test_types": ["K"], "reason": "..."}
  ]
}
```

Postman: POST to the same URL with raw JSON body.

## Deployment

**Streamlit UI** -- push to GitHub, deploy on [share.streamlit.io](https://share.streamlit.io).

**API** -- deploy on [Render](https://render.com) as a web service with start command `uvicorn api:app --host 0.0.0.0 --port $PORT`.

## Live Links

- Streamlit: _https://shl-assessment.streamlit.app_ (update after deploy)
- API: _https://shl-api.onrender.com_ (update after deploy)
