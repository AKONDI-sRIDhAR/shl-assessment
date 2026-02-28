# SHL AI Assessment Recommendation Engine

> **Intelligent assessment recommendation system** powered by semantic search.  
> Built with `sentence-transformers`, `FAISS`, and `Streamlit`.

---

## Architecture Overview

```
                    +------------------+
                    |  SHL Product     |
                    |  Catalog Website |
                    +--------+---------+
                             | resumable_fast_scraper.py
                             | (6 parallel workers)
                    +--------v---------+
                    |  data/           |
                    |  - assessments.csv|
                    |  - faiss_index.bin|
                    |  - metadata.pkl  |
                    +--------+---------+
                             |
                    +--------v---------+
          Query --> |   app.py         | --> Top-K Recommendations
                    |  (Streamlit UI)  |
                    |  - Encode query  |
                    |  - FAISS search  |
                    |  - Re-rank/boost |
                    +------------------+
```

## Project Structure

```
SHL-Assessment-Recommendation-Engine/
|-- app.py                      # Main Streamlit web application
|-- resumable_fast_scraper.py   # Parallel, resumable scraper + index builder
|-- scraper.py                  # Original sequential scraper (backup)
|-- evaluate_train.py           # Evaluation on labeled training queries
|-- requirements.txt            # Python dependencies
|-- run_scraper.bat             # Windows: double-click to run scraper
|-- README.md                   # This file
|-- approach.md                 # Detailed architecture & approach document
+-- data/
    |-- assessments.csv         # Scraped assessment data (~389 rows)
    |-- faiss_index.bin         # FAISS vector index (384-dim)
    +-- metadata.pkl            # Assessment metadata (pickle)
```

---

## Quick Start

### Prerequisites

- **Python 3.9+** (tested on 3.10, 3.11)
- **pip** package manager
- **Windows 10/11** (also works on Linux/Mac)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run the Scraper (First Time Only)

**Windows (recommended):**
Double-click `run_scraper.bat` — it handles encoding automatically.

**Any OS (terminal):**
```bash
python -u resumable_fast_scraper.py
```

> **Takes ~3-4 minutes** with 6 parallel workers.  
> **Resumable** — safe to re-run if interrupted; it will skip already-scraped pages.  
> Output: `data/assessments.csv`, `data/faiss_index.bin`, `data/metadata.pkl`

### Step 3: Launch the Web Application

```bash
streamlit run app.py
```

Opens at `http://localhost:8501` in your browser.

### Step 4 (Optional): Run Evaluation

```bash
python evaluate_train.py
```

Runs 10 labeled training queries and computes Recall@K and MAP@K metrics.

---

## Deploy to Streamlit Community Cloud

1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "SHL Assessment Recommendation Engine"
   git remote add origin https://github.com/<user>/SHL-Assessment-Recommendation-Engine.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app" -> select your repo -> Main file: `app.py`
   - Click "Deploy"

3. **Important:** Commit the `data/` folder (with CSV, FAISS index, metadata.pkl) so the deployed app works without running the scraper.

---

## Screenshots

| Home Screen | Results View |
|:-----------:|:------------:|
| *Query input with modern dark UI* | *Top-K cards with scores and type badges* |

| Sidebar Settings | Expandable Details |
|:----------------:|:------------------:|
| *Adjustable K, balance K/P toggle* | *Description, duration, job levels* |

---

## How It Works

### Scraping Pipeline (`resumable_fast_scraper.py`)
1. Iterates 32 paginated catalog pages (`?type=1&start=0,12,...,372`)
2. Extracts ~389 assessment names and detail page URLs
3. Visits each detail page **in parallel** (6 workers, 0.4s rate limit each)
4. Extracts: description, duration, job levels, languages, test types
5. Builds `rich_text` combining all fields for embedding
6. Generates embeddings using `all-MiniLM-L6-v2` (384-dim, normalized)
7. Builds FAISS IndexFlatIP (Inner Product on normalized = cosine similarity)

### Recommendation Engine (`app.py`)
1. User enters a query or job description
2. Query encoded with same sentence-transformer model
3. FAISS returns top-K nearest neighbors by cosine similarity
4. Optional re-ranking boosts K-type (Knowledge) or P-type (Personality) based on query intent
5. Results shown with scores, type badges, and keyword-based reasons

### Balance Re-ranking
When "Balance K <-> P" toggle is on:
- **Technical queries** (coding, SQL, data...) -> boost K and A scores
- **Behavioral queries** (leadership, team...) -> boost P and B scores
- Boost magnitude is small (max +0.05) to nudge, not override, similarity

---

## Automated Testing & Health Check

### Health Check
```python
from app import health_check
status = health_check()
print(status)
# {'status': 'healthy', 'components': {'model': 'ok', 'faiss_index': 'ok', 'metadata': 'ok (389 items)'}}
```

### Recommendation API
```python
from app import recommend

results = recommend("Java developer with customer service", top_k=8, balance=True)
for r in results:
    print(f"{r['name']} -- Score: {r['score']:.2f} -- Types: {r['test_types']}")
```

### Evaluation
```bash
python evaluate_train.py
# Outputs: Recall@3, Recall@5, Recall@10, MAP@10 per query and overall
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Web Framework | Streamlit 1.42.0 |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Search | FAISS (faiss-cpu, IndexFlatIP) |
| Scraping | requests + BeautifulSoup4 + lxml |
| Data | pandas, numpy |
| Parallelism | concurrent.futures (ThreadPoolExecutor) |
| Language | Python 3.9+ |

---

## License

Built for the **SHL AI Intern / Research Engineer** assignment (2025-2026).
