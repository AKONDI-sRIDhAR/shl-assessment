# Approach Document — SHL AI Assessment Recommendation Engine

---

## 1. Problem Statement

Given a natural-language query (job description, role title, or skill requirement), recommend the most relevant SHL assessments from a catalog of ~380 individual test solutions. The system must return the top-K assessments with relevance scores and support balanced recommendations across Knowledge (K) and Personality (P) test types.

---

## 2. System Architecture

### High-Level Design

The system follows a **two-phase architecture**:

**Phase 1: Offline Data Pipeline (scraper.py)**
```
SHL Catalog → Web Scraping → Data Enrichment → Embedding Generation → FAISS Indexing
```

**Phase 2: Online Query Pipeline (app.py)**
```
User Query → Embedding → FAISS Nearest-Neighbor Search → Balance Re-ranking → Results
```

### Component Breakdown

| Component | Purpose | Technology |
|-----------|---------|-----------|
| **Scraper** | Extract assessment data from SHL catalog | `requests`, `BeautifulSoup4`, `lxml` |
| **Embedding Generator** | Create dense vector representations of assessment descriptions | `sentence-transformers/all-MiniLM-L6-v2` |
| **Vector Store** | Enable fast similarity search across 380+ assessments | `FAISS` (IndexFlatIP) |
| **Query Encoder** | Convert user query to same embedding space | Same `all-MiniLM-L6-v2` model |
| **Re-ranker** | Balance K vs P assessment types based on query intent | Keyword analysis + score boosting |
| **Web UI** | Interactive recommendation interface | `Streamlit` |

---

## 3. Why Embeddings + FAISS?

### Why Semantic Embeddings?

Traditional keyword-matching approaches (TF-IDF, BM25) fail when:
- The query uses different terminology than the assessment description (e.g., "coding test" vs. "programming knowledge assessment")
- The query is a full job description with many irrelevant details
- Assessments are described in domain-specific SHL terminology

**Sentence-transformer embeddings** (specifically `all-MiniLM-L6-v2`) solve this by:
- Mapping both queries and assessments into a shared 384-dimensional semantic space
- Capturing synonyms, paraphrases, and contextual similarity
- Being pre-trained on 1B+ sentence pairs from diverse domains
- Running fast enough for real-time inference (~50ms per query)

### Why FAISS?

FAISS (Facebook AI Similarity Search) provides:
- **Speed**: Sub-millisecond search over 380 vectors (exact search with `IndexFlatIP`)
- **Scalability**: Could handle millions of assessments with approximate indices (IVF, HNSW)
- **Memory efficiency**: Compact binary format for index persistence
- **Cosine similarity**: Using `IndexFlatIP` on L2-normalized vectors is mathematically equivalent to cosine similarity

For our catalog size (~380 items), exact search is fast enough, so we use `IndexFlatIP` rather than approximate methods. If the catalog grew to 100K+ items, we would switch to `IndexIVFFlat` or `IndexHNSWFlat`.

### Why all-MiniLM-L6-v2?

| Factor | Value |
|--------|-------|
| Embedding dimension | 384 |
| Model size | ~80MB |
| Inference speed | ~14K sentences/sec on CPU |
| Quality (STS Benchmark) | 82.8% Spearman correlation |
| Trained on | 1B+ diverse sentence pairs |

This model strikes the best balance between quality and speed for our use case. Larger models (e.g., `all-mpnet-base-v2`) offer marginal quality gains but are 3x slower.

---

## 4. Rich Text Construction

For each assessment, we construct a `rich_text` field that combines all available metadata into a single string for embedding:

```
Title: {name}
Description: {description}
Test Types: {test_type_codes}
Duration: {duration}
Job Levels: {job_levels}
```

This approach ensures:
- The embedding captures the **full semantic context** of each assessment
- Test type codes (K, P, A, B, S) are included, allowing the model to learn associations between assessment types and relevant queries
- Duration and job levels add contextual signals for matching

---

## 5. Balance Handling (K ↔ P Re-ranking)

### The Problem

Pure cosine similarity may return results that cluster around one test type. For example, a query about "software developers" might return only Knowledge (K) tests, missing relevant Personality (P) tests for cultural fit evaluation.

### The Solution

When the "Balance" toggle is enabled, the system applies a **two-step re-ranking**:

**Step 1: Query Intent Classification**
```python
tech_ratio, behav_ratio = detect_query_intent(query)
```
- Scans the query for technical keywords (coding, SQL, algorithms, etc.)
- Scans for behavioral keywords (leadership, teamwork, culture, etc.)
- Computes a ratio: e.g., `tech_ratio=0.7, behav_ratio=0.3`

**Step 2: Score Boosting**
```python
if "K" in test_types and tech_ratio > 0.5:
    score += 0.05 * tech_ratio  # Boost Knowledge tests for technical queries
if "P" in test_types and behav_ratio > 0.5:
    score += 0.05 * behav_ratio  # Boost Personality tests for behavioral queries
```

The boost is **proportional** to the intent strength, ensuring:
- Strong technical queries get a meaningful K boost
- Mixed queries get moderate, balanced boosting
- The base cosine similarity score remains the primary ranking signal

### Why This Works

- Boosts are small (max ±0.05) — they nudge, not override, semantic similarity
- Uses two keyword dictionaries (30+ words each) for robust intent detection
- Handles edge cases: queries with no clear intent get equal 0.5/0.5 ratios (no boost)
- Preserves diversity: a technical query won't lose ALL personality tests, just de-prioritize them slightly

---

## 6. Estimated Performance

### Recall@K Estimation

Based on the design and typical performance of similar semantic search systems:

| Metric | Estimated | Rationale |
|--------|:---------:|-----------|
| **Recall@3** | ~0.65 | Top-3 results contain relevant assessments ~65% of the time |
| **Recall@5** | ~0.70 | Broader coverage catches more relevant types |
| **Recall@10** | ~0.78 | Most relevant assessments appear within top-10 |
| **MAP@10** | ~0.60 | Relevant results tend to be ranked higher |

### Factors Affecting Performance

- **Data quality**: Rich descriptions improve embedding quality significantly
- **Query specificity**: Specific role queries (e.g., "Java developer") perform better than vague queries (e.g., "good test")
- **Catalog coverage**: SHL's catalog may not have assessments for every possible query
- **Balance re-ranking**: Can improve Recall for mixed-intent queries by +5-10%

---

## 7. Trade-offs & Design Decisions

| Decision | Trade-off |
|----------|-----------|
| **MiniLM-L6 over larger models** | Smaller model (80MB) is deployable on Streamlit Cloud free tier; marginal quality loss |
| **FAISS IndexFlatIP over IVF** | Exact search is fine for 380 items; no approximation error |
| **Keyword-based balance over ML classifier** | Simpler, transparent, no additional training data needed |
| **Pre-computed embeddings over real-time** | Faster queries; requires re-scraping if catalog changes |
| **Single rich_text embedding over multi-field** | Simpler architecture; may lose some field-specific signals |
| **Static scraping over API** | SHL doesn't expose a public API; scraping is the only option |

---

## 8. Future Improvements

### Short-term (v2.0)

1. **RAG with Gemini/GPT**: Use a LLM to generate more nuanced reasons for recommendations, incorporating the assessment description and query context
2. **Cross-encoder re-ranking**: Use a cross-encoder model (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) as a second-stage ranker for improved precision
3. **JD URL parsing**: Accept a URL, fetch the page content, and extract the job description automatically
4. **Multi-field embeddings**: Embed name, description, and test types separately and combine with learned weights

### Medium-term (v3.0)

5. **User feedback loop**: Allow users to rate recommendations; use feedback to fine-tune embeddings
6. **Hybrid search**: Combine semantic search with BM25 keyword search (Reciprocal Rank Fusion)
7. **Assessment compatibility matrix**: Model which assessments work well together as a battery
8. **Caching layer**: Cache frequent queries for sub-10ms responses

### Long-term (v4.0)

9. **Fine-tuned domain embeddings**: Train/fine-tune on SHL-specific assessment data
10. **Active learning**: Use evaluation results to continuously improve the recommendation quality
11. **Multi-language support**: Support queries in non-English languages
12. **API gateway**: Expose recommendations as a REST API for integration with ATS platforms

---

## 9. Conclusion

This recommendation engine demonstrates a practical, production-ready approach to assessment recommendation using modern NLP techniques. The combination of dense semantic embeddings with FAISS provides fast, accurate retrieval, while the balance re-ranking mechanism ensures diverse, role-appropriate recommendations. The architecture is extensible, allowing future enhancements with minimal changes to the core pipeline.

---

*Document prepared for the SHL AI Intern / Research Engineer assignment (2025-2026)*
