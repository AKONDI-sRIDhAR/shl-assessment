# SHL Assessment Recommendation Engine

Recommends relevant SHL assessments for a given job description or query.
Uses sentence-transformer embeddings and FAISS for semantic similarity search.

Two interfaces:
- **Streamlit UI** for interactive use
- **FastAPI backend** with REST endpoints for programmatic access

---

## Project Structure

```
shl-assessment/
  app.py               Streamlit web UI
  api.py               FastAPI REST backend
  utils.py             Shared recommendation logic
  scraper.py           Catalog scraper + embedding builder
  requirements.txt     Dependencies
  README.md
  approach.md          Design rationale
  .streamlit/
    config.toml        Dark theme config
  data/
    assessments.csv
    assessments_full.csv
    faiss_index.bin
    metadata.pkl
```

---

## How to Run Locally

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the scraper (first time only)

```bash
python scraper.py
```

Takes 5-8 minutes. Resumable -- safe to re-run if interrupted.

### 3a. Launch the Streamlit UI

```bash
streamlit run app.py
```

Opens at http://localhost:8501

### 3b. Launch the API server

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

API docs at http://localhost:8000/docs

---

## Live Demo

**Streamlit UI:** https://shl-assessment.streamlit.app _(update after deploying)_

**API:** https://shl-assessment-api.onrender.com _(update after deploying)_

---

## API Reference

### GET /health

Returns system health status.

```bash
curl https://shl-assessment-api.onrender.com/health
```

Response:
```json
{"status": "healthy"}
```

### POST /recommend

Accepts a query and returns ranked assessments.

```bash
curl -X POST https://shl-assessment-api.onrender.com/recommend \
  -H "Content-Type: application/json" \
  -d '{"query": "Java developers who handle customer calls", "top_k": 5, "balance": true}'
```

Response:
```json
{
  "query": "Java developers who handle customer calls",
  "top_k": 5,
  "balance": true,
  "results": [
    {
      "rank": 1,
      "name": "Java 8 (New)",
      "url": "https://www.shl.com/products/product-catalog/view/java-8-new/",
      "score": 0.6012,
      "test_types": ["K"],
      "reason": "Matches: customer, developers, java. Type: Knowledge & Skills."
    }
  ]
}
```

**Postman:** Import the URL, set method to POST, body to raw JSON.

---

## How SHL Reviewers Can Test

1. **Health check:**
   ```bash
   curl http://localhost:8000/health
   ```

2. **Recommendation:**
   ```bash
   curl -X POST http://localhost:8000/recommend \
     -H "Content-Type: application/json" \
     -d '{"query": "I need a personality test for entry-level sales", "top_k": 8}'
   ```

3. **Swagger docs:** Open http://localhost:8000/docs in your browser.

4. **Python:**
   ```python
   from utils import recommend
   results = recommend("data analyst with SQL skills", top_k=5)
   for r in results:
       print(r["name"], r["score"])
   ```

---

## Tech Stack

| Component | Tool |
|-----------|------|
| UI | Streamlit 1.42 |
| API | FastAPI + Uvicorn |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector search | FAISS (faiss-cpu) |
| Scraping | requests, BeautifulSoup4 |
| Data | pandas, numpy |

---

_Built for the SHL AI Intern / Research Engineer assignment, Feb 2026._
