"""
=============================================================================
SHL Assessment Catalog — Resumable Fast Scraper
=============================================================================
Parallel, resumable scraper for SHL product catalog (Individual Test Solutions).

Key features:
  - Phase 1: Scrape all catalog listing pages -> basic CSV (name, url)
  - Phase 2: Parallel detail scraping with ThreadPoolExecutor (6 workers)
  - Checkpoint every 25 assessments -> auto-resume on next run
  - Phase 3: Generate embeddings + FAISS index for similarity search

Usage:
    python resumable_fast_scraper.py          # or double-click run_scraper.bat
    (Safe to re-run — automatically resumes from last checkpoint)

Output:
    data/assessments.csv        — final enriched dataset
    data/faiss_index.bin        — FAISS similarity search index
    data/metadata.pkl           — pickled metadata for the app
=============================================================================
"""

import sys
import io
import os
import re
import csv
import time
import pickle
import requests
import numpy as np
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Fix Windows console encoding
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer, encoding="utf-8", errors="replace"
        )

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_URL = "https://www.shl.com/products/product-catalog/"
CATALOG_URL = BASE_URL + "?type=1"
START_VALUES = list(range(0, 384, 12))  # 0, 12, 24, ..., 372  (32 pages)
MAX_WORKERS = 6                         # parallel detail scrapers
SLEEP_PER_REQUEST = 0.4                 # polite delay inside each worker
CHECKPOINT_EVERY = 25                   # save progress every N detail pages

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "assessments.csv")
FAISS_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.pkl")

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


# ============================= PHASE 1 =====================================
# Scrape all catalog pages to get (name, url) pairs
# ===========================================================================
def scrape_catalog_listing():
    """
    Iterate through all paginated listing pages and extract assessment
    names + detail page URLs.  Returns list of dicts: [{name, url}, ...]
    """
    assessments = []
    seen_urls = set()

    for start in START_VALUES:
        page_url = f"{CATALOG_URL}&start={start}" if start > 0 else CATALOG_URL
        print(f"[CATALOG] Page start={start}")

        try:
            resp = requests.get(page_url, headers=REQUEST_HEADERS, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"  [WARN] Failed: {e}")
            continue

        soup = BeautifulSoup(resp.text, "lxml")

        # Links to detail pages match /products/product-catalog/view/<slug>/
        for a_tag in soup.select("a[href*='/products/product-catalog/view/']"):
            href = a_tag.get("href", "").strip()
            name = a_tag.get_text(strip=True)

            if not name or not href:
                continue

            # Build absolute URL
            if href.startswith("/"):
                href = "https://www.shl.com" + href

            if href not in seen_urls:
                seen_urls.add(href)
                assessments.append({"name": name, "url": href})

        time.sleep(0.2)  # small pause between listing pages

    print(f"\n[OK] Found {len(assessments)} unique assessments from catalog.\n")
    return assessments


# ============================= PHASE 2 =====================================
# Scrape individual detail pages (parallel + resumable)
# ===========================================================================
def extract_test_types(soup):
    """Extract test-type badge letters (A, B, C, D, E, K, P, S) from a page."""
    valid = {"A", "B", "C", "D", "E", "K", "P", "S"}
    found = []

    # Method 1: specific badge spans
    for span in soup.select("span.product-catalogue__key"):
        letter = span.get_text(strip=True).upper()
        if letter in valid and letter not in found:
            found.append(letter)

    # Method 2: fallback badge containers
    if not found:
        for el in soup.select(".product-catalogue-badge, .catalogue-badge, .badge"):
            letter = el.get_text(strip=True).upper()
            if len(letter) == 1 and letter in valid and letter not in found:
                found.append(letter)

    # Method 3: regex in page text
    if not found:
        text = soup.get_text()
        match = re.search(r"Test Type[s]?:\s*([A-Z](?:\s*,?\s*[A-Z])*)", text)
        if match:
            for ch in match.group(1):
                if ch in valid and ch not in found:
                    found.append(ch)

    return " ".join(found)


def scrape_one_detail(name, url):
    """
    Scrape a single assessment detail page.
    Returns dict with: description, duration, job_levels, languages, test_types.
    """
    detail = {
        "description": "",
        "duration": "",
        "job_levels": "",
        "languages": "",
        "test_types": "",
    }

    try:
        time.sleep(SLEEP_PER_REQUEST)  # rate-limit inside each worker thread
        resp = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"    [FAIL] {name[:50]}: {e}")
        return detail

    soup = BeautifulSoup(resp.text, "lxml")
    page_text = soup.get_text(separator="\n")

    # --- Description ---
    desc_header = soup.find(
        lambda tag: tag.name in ["h4", "h3", "h2", "strong"]
        and "description" in tag.get_text(strip=True).lower()
    )
    if desc_header:
        parts = []
        for sib in desc_header.find_next_siblings():
            if sib.name in ["h4", "h3", "h2"]:
                break
            t = sib.get_text(strip=True)
            if t:
                parts.append(t)
        detail["description"] = " ".join(parts)
    else:
        og = soup.find("meta", attrs={"property": "og:description"})
        if og:
            detail["description"] = og.get("content", "")

    # --- Duration ---
    dur = re.search(
        r"(?:Approximate Completion Time|Assessment [Ll]ength)[^\d]*(\d+)\s*(?:minutes|mins?)?",
        page_text, re.IGNORECASE,
    )
    if dur:
        detail["duration"] = f"{dur.group(1)} minutes"
    else:
        dur2 = re.search(r"(\d+)\s*(?:minutes|mins)", page_text, re.IGNORECASE)
        if dur2:
            detail["duration"] = f"{dur2.group(1)} minutes"

    # --- Job Levels ---
    jl = soup.find(
        lambda tag: tag.name in ["h4", "h3", "h2", "strong"]
        and "job level" in tag.get_text(strip=True).lower()
    )
    if jl:
        parts = []
        for sib in jl.find_next_siblings():
            if sib.name in ["h4", "h3", "h2"]:
                break
            t = sib.get_text(strip=True)
            if t:
                parts.append(t)
        detail["job_levels"] = ", ".join(parts)
    else:
        jl_m = re.search(r"Job [Ll]evels?\s*[:\-]?\s*(.+?)(?:\n|$)", page_text)
        if jl_m:
            detail["job_levels"] = jl_m.group(1).strip().rstrip(",")

    # --- Languages ---
    lang = soup.find(
        lambda tag: tag.name in ["h4", "h3", "h2", "strong"]
        and "language" in tag.get_text(strip=True).lower()
    )
    if lang:
        parts = []
        for sib in lang.find_next_siblings():
            if sib.name in ["h4", "h3", "h2"]:
                break
            t = sib.get_text(strip=True)
            if t:
                parts.append(t)
        detail["languages"] = ", ".join(parts)
    else:
        l_m = re.search(r"Languages?\s*[:\-]?\s*(.+?)(?:\n|$)", page_text)
        if l_m:
            detail["languages"] = l_m.group(1).strip().rstrip(",")

    # --- Test Types ---
    detail["test_types"] = extract_test_types(soup)

    return detail


def build_rich_text(name, detail):
    """Build the rich_text string used for embedding generation."""
    return (
        f"Title: {name}\n"
        f"Description: {detail.get('description', '')}\n"
        f"Test Types: {detail.get('test_types', '')}\n"
        f"Duration: {detail.get('duration', '')}\n"
        f"Job Levels: {detail.get('job_levels', '')}"
    )


def save_checkpoint(assessments, path):
    """Save current assessment data to CSV (checkpoint)."""
    fieldnames = [
        "name", "url", "test_types", "description",
        "duration", "job_levels", "languages", "rich_text",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(assessments)


def load_checkpoint(path):
    """Load existing checkpoint CSV and return list of dicts."""
    if not os.path.exists(path):
        return []
    import pandas as pd
    df = pd.read_csv(path, encoding="utf-8")
    # Replace NaN with empty strings
    df = df.fillna("")
    return df.to_dict("records")


def scrape_all_details(assessments):
    """
    Parallel detail scraping with ThreadPoolExecutor.
    Skips assessments that already have a description (resume support).
    Saves checkpoints every CHECKPOINT_EVERY items.
    """
    # Identify which ones need scraping
    to_scrape = []
    already_done = 0
    for i, a in enumerate(assessments):
        if a.get("description", "").strip():
            already_done += 1
        else:
            to_scrape.append(i)

    if already_done > 0:
        print(f"[RESUME] {already_done} already scraped, {len(to_scrape)} remaining.\n")
    else:
        print(f"[INFO] Scraping details for {len(to_scrape)} assessments "
              f"with {MAX_WORKERS} workers...\n")

    if not to_scrape:
        print("[OK] All detail pages already scraped.\n")
        return

    completed = 0
    total = len(to_scrape)

    # Process in batches for checkpointing
    batch_start = 0
    while batch_start < total:
        batch_end = min(batch_start + CHECKPOINT_EVERY, total)
        batch_indices = to_scrape[batch_start:batch_end]

        # Submit batch to thread pool
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_idx = {}
            for idx in batch_indices:
                a = assessments[idx]
                future = executor.submit(scrape_one_detail, a["name"], a["url"])
                future_to_idx[future] = idx

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                completed += 1
                try:
                    detail = future.result(timeout=60)
                    assessments[idx].update(detail)
                    assessments[idx]["rich_text"] = build_rich_text(
                        assessments[idx]["name"], detail
                    )
                    print(f"  [{completed}/{total}] {assessments[idx]['name'][:55]}")
                except Exception as e:
                    print(f"  [{completed}/{total}] [FAIL] index={idx}: {e}")
                    # Build rich_text from what we have (at least the title)
                    assessments[idx]["rich_text"] = build_rich_text(
                        assessments[idx]["name"],
                        assessments[idx]
                    )

        # Save checkpoint after each batch
        save_checkpoint(assessments, CSV_PATH)
        print(f"  [CHECKPOINT] Saved {completed}/{total} to {CSV_PATH}\n")

        batch_start = batch_end

    # Ensure all have rich_text (even if detail scraping failed)
    for a in assessments:
        if not a.get("rich_text"):
            a["rich_text"] = build_rich_text(a["name"], a)


# ============================= PHASE 3 =====================================
# Generate embeddings + FAISS index
# ===========================================================================
def generate_embeddings_and_index(assessments):
    """
    Generate sentence-transformer embeddings for all assessments
    and build a FAISS index for cosine similarity search.
    """
    from sentence_transformers import SentenceTransformer
    import faiss

    print("[LOAD] Loading sentence-transformers model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    texts = [a.get("rich_text", a.get("name", "")) for a in assessments]
    print(f"[INFO] Generating embeddings for {len(texts)} assessments...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,
        batch_size=64,
    )
    embeddings = np.array(embeddings, dtype="float32")

    # Build FAISS index (Inner Product on L2-normalized = cosine similarity)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Save FAISS index
    faiss.write_index(index, FAISS_PATH)
    print(f"[OK] FAISS index saved: {FAISS_PATH}  (dim={dim}, n={index.ntotal})")

    # Save metadata as pickle (for fast loading by the app)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(assessments, f)
    print(f"[OK] Metadata saved: {METADATA_PATH}")

    return index


# ============================= MAIN ========================================
def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 66)
    print("  SHL Assessment Catalog — Resumable Fast Scraper")
    print("=" * 66)
    start_time = time.time()

    # --- Check for existing checkpoint ---
    existing = load_checkpoint(CSV_PATH)

    if existing and len(existing) > 10:
        print(f"\n[RESUME] Found existing checkpoint with {len(existing)} assessments.")
        assessments = existing
    else:
        # Phase 1: Scrape catalog listing
        print("\n--- PHASE 1: Catalog Listing ---")
        assessments = scrape_catalog_listing()

        if not assessments:
            print("[ERROR] No assessments found. Check your internet connection.")
            return

        # Initialize empty detail fields
        for a in assessments:
            a.setdefault("description", "")
            a.setdefault("duration", "")
            a.setdefault("job_levels", "")
            a.setdefault("languages", "")
            a.setdefault("test_types", "")
            a.setdefault("rich_text", "")

        # Save initial checkpoint (just names + URLs)
        save_checkpoint(assessments, CSV_PATH)
        print(f"[OK] Initial checkpoint saved: {len(assessments)} assessments.\n")

    # Phase 2: Parallel detail scraping
    print("--- PHASE 2: Detail Page Scraping (parallel) ---")
    scrape_all_details(assessments)

    # Final save
    save_checkpoint(assessments, CSV_PATH)
    elapsed_scrape = time.time() - start_time
    print(f"[OK] Scraping complete in {elapsed_scrape:.1f}s")
    print(f"[OK] {len(assessments)} assessments saved to {CSV_PATH}\n")

    # Phase 3: Generate embeddings + FAISS index
    print("--- PHASE 3: Embeddings + FAISS Index ---")
    generate_embeddings_and_index(assessments)

    elapsed_total = time.time() - start_time
    print("\n" + "=" * 66)
    print(f"  [DONE] All complete in {elapsed_total:.1f} seconds")
    print(f"  Data directory: {DATA_DIR}")
    print(f"  Next step: streamlit run app.py")
    print("=" * 66)


if __name__ == "__main__":
    main()
