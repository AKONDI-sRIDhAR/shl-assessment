"""
scraper.py -- SHL Product Catalog Scraper
==========================================
Scrapes the "Individual Test Solutions" section of SHL's Product Catalog.
URL pattern: https://www.shl.com/products/product-catalog/?type=1&start=N

Two phases:
  1. Listing phase  – crawl all paginated listing pages, collect assessment
     names and detail-page URLs into data/assessments.csv.
  2. Detail phase   – visit each detail page and pull description, duration,
     job levels, languages, and test-type badges. Results go into
     data/assessments_full.csv.

After scraping, the script generates sentence-transformer embeddings and
stores them in a FAISS index (data/faiss_index.bin + data/metadata.pkl).

The detail phase is **resumable**: if the script is interrupted and re-run,
it will skip any assessment that already has a description.

Usage:
    python scraper.py        # first run: scrapes listing + details + embeddings
    python scraper.py        # second run: resumes where it left off
"""

import os
import re
import sys
import csv
import time
import pickle
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

# -- make sure we don't crash on Windows with non-ASCII output --
if sys.platform == "win32":
    import io
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(SCRIPT_DIR, "data")
LISTING_CSV = os.path.join(DATA_DIR, "assessments.csv")      # phase-1 output
FULL_CSV    = os.path.join(DATA_DIR, "assessments_full.csv")  # phase-2 output
FAISS_PATH  = os.path.join(DATA_DIR, "faiss_index.bin")
META_PATH   = os.path.join(DATA_DIR, "metadata.pkl")

CATALOG_BASE = "https://www.shl.com/products/product-catalog/"
CATALOG_URL  = CATALOG_BASE + "?type=1"
PAGE_STARTS  = list(range(0, 384, 12))   # 32 pages, 12 items each

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
}


# ========================================================================== #
#  PHASE 1 -- listing pages                                                  #
# ========================================================================== #

def fetch_listing_pages():
    """
    Crawl all paginated catalog pages and return a list of dicts:
    [{"name": "...", "url": "https://..."}, ...]
    """
    assessments = []
    seen = set()

    for start in PAGE_STARTS:
        url = f"{CATALOG_URL}&start={start}" if start else CATALOG_URL
        print(f"  Listing page start={start}")

        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
        except requests.RequestException as exc:
            print(f"    WARNING: {exc}")
            continue

        soup = BeautifulSoup(r.text, "lxml")
        for a in soup.select("a[href*='/products/product-catalog/view/']"):
            href = a.get("href", "").strip()
            name = a.get_text(strip=True)
            if not name or not href:
                continue
            if href.startswith("/"):
                href = "https://www.shl.com" + href
            if href not in seen:
                seen.add(href)
                assessments.append({"name": name, "url": href})

        time.sleep(0.25)

    return assessments


def save_listing(assessments):
    """Write the basic listing CSV (name + url only)."""
    with open(LISTING_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["name", "url"], extrasaction="ignore")
        w.writeheader()
        w.writerows(assessments)


# ========================================================================== #
#  PHASE 2 -- detail pages (resumable)                                       #
# ========================================================================== #

def _extract_test_types(soup):
    """Pull single-letter test-type badges (K, P, A, B, ...) from the page."""
    valid = set("ABCDEKPS")
    found = []

    # branded badge spans
    for span in soup.select("span.product-catalogue__key"):
        ch = span.get_text(strip=True).upper()
        if ch in valid and ch not in found:
            found.append(ch)

    # fallback: regex on raw text
    if not found:
        m = re.search(r"Test Type[s]?:\s*([A-Z](?:\s*,?\s*[A-Z])*)", soup.get_text())
        if m:
            for ch in m.group(1):
                if ch in valid and ch not in found:
                    found.append(ch)

    return " ".join(found)


def _find_section_text(soup, keyword):
    """
    Look for an <h4>/<h3>/<strong> whose text contains `keyword`,
    then grab everything until the next heading.
    """
    header = soup.find(
        lambda t: t.name in ("h4", "h3", "h2", "strong")
        and keyword in t.get_text(strip=True).lower()
    )
    if not header:
        return ""
    parts = []
    for sib in header.find_next_siblings():
        if sib.name in ("h4", "h3", "h2"):
            break
        txt = sib.get_text(strip=True)
        if txt:
            parts.append(txt)
    return " ".join(parts)


def scrape_detail(url):
    """
    Fetch one detail page and return a dict with description,
    duration, job_levels, languages, and test_types.
    """
    blank = dict(description="", duration="", job_levels="", languages="", test_types="")

    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
    except requests.RequestException as exc:
        print(f"    FAILED: {exc}")
        return blank

    soup = BeautifulSoup(r.text, "lxml")
    text = soup.get_text(separator="\n")

    # description
    desc = _find_section_text(soup, "description")
    if not desc:
        og = soup.find("meta", attrs={"property": "og:description"})
        desc = og["content"] if og and og.get("content") else ""

    # duration
    dur_match = re.search(
        r"(?:Approximate Completion Time|Assessment [Ll]ength)[^\d]*(\d+)", text, re.I
    )
    duration = f"{dur_match.group(1)} minutes" if dur_match else ""
    if not duration:
        dur2 = re.search(r"(\d+)\s*(?:minutes|mins)", text, re.I)
        duration = f"{dur2.group(1)} minutes" if dur2 else ""

    # job levels
    job_levels = _find_section_text(soup, "job level")
    if not job_levels:
        jl = re.search(r"Job [Ll]evels?\s*[:\-]?\s*(.+?)(?:\n|$)", text)
        job_levels = jl.group(1).strip().rstrip(",") if jl else ""

    # languages
    languages = _find_section_text(soup, "language")
    if not languages:
        lm = re.search(r"Languages?\s*[:\-]?\s*(.+?)(?:\n|$)", text)
        languages = lm.group(1).strip().rstrip(",") if lm else ""

    # test types
    test_types = _extract_test_types(soup)

    return dict(
        description=desc, duration=duration,
        job_levels=job_levels, languages=languages,
        test_types=test_types,
    )


def build_rich_text(row):
    """Combine all fields into a single text blob for the embedding model."""
    return (
        f"Title: {row.get('name', '')}\n"
        f"Description: {row.get('description', '')}\n"
        f"Test Types: {row.get('test_types', '')}\n"
        f"Duration: {row.get('duration', '')}\n"
        f"Job Levels: {row.get('job_levels', '')}"
    )


def save_full_csv(rows):
    """Write the enriched CSV checkpoint."""
    cols = ["name", "url", "test_types", "description",
            "duration", "job_levels", "languages", "rich_text"]
    with open(FULL_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def run_detail_phase(assessments):
    """
    Scrape every detail page that doesn't already have a description.
    Save a checkpoint every 30 rows.  Keeps a 0.8-second delay between
    requests to stay polite.
    """
    total   = len(assessments)
    pending = [i for i, a in enumerate(assessments) if not a.get("description", "").strip()]
    done    = total - len(pending)

    if done:
        print(f"  Resuming: {done} already done, {len(pending)} remaining.")
    else:
        print(f"  Scraping detail pages for {total} assessments...")

    for count, idx in enumerate(pending, start=1):
        row  = assessments[idx]
        name = row["name"]
        print(f"  Scraping {done + count}/{total}: {name[:60]}")

        detail = scrape_detail(row["url"])
        row.update(detail)
        row["rich_text"] = build_rich_text(row)

        # checkpoint every 30
        if count % 30 == 0:
            save_full_csv(assessments)
            print(f"    [checkpoint saved at {done + count}/{total}]")

        time.sleep(0.8)

    # final save
    save_full_csv(assessments)
    print(f"  All {total} assessments saved to {FULL_CSV}")


# ========================================================================== #
#  PHASE 3 -- embeddings + FAISS                                             #
# ========================================================================== #

def build_index(assessments):
    """
    Encode every assessment's rich_text with sentence-transformers and
    drop the vectors into a FAISS inner-product index (which equals cosine
    similarity when the vectors are L2-normalised).
    """
    from sentence_transformers import SentenceTransformer
    import faiss

    print("  Loading embedding model (all-MiniLM-L6-v2) ...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    texts = [a.get("rich_text") or a.get("name", "") for a in assessments]
    print(f"  Encoding {len(texts)} assessments ...")
    vecs = model.encode(texts, show_progress_bar=True,
                        normalize_embeddings=True, batch_size=64)
    vecs = np.array(vecs, dtype="float32")

    dim   = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)

    faiss.write_index(index, FAISS_PATH)
    print(f"  FAISS index saved  -> {FAISS_PATH}  ({index.ntotal} vectors, dim={dim})")

    with open(META_PATH, "wb") as f:
        pickle.dump(assessments, f)
    print(f"  Metadata pickle    -> {META_PATH}")


# ========================================================================== #
#  main                                                                      #
# ========================================================================== #

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 60)
    print("  SHL Product Catalog Scraper")
    print("=" * 60)
    t0 = time.time()

    # --- try to resume from enriched CSV ---
    if os.path.exists(FULL_CSV):
        print("\nPhase 1: Loading existing enriched data ...")
        df = pd.read_csv(FULL_CSV, encoding="utf-8").fillna("")
        assessments = df.to_dict("records")
        print(f"  Loaded {len(assessments)} rows from {FULL_CSV}")

    elif os.path.exists(LISTING_CSV):
        print("\nPhase 1: Loading existing listing ...")
        df = pd.read_csv(LISTING_CSV, encoding="utf-8").fillna("")
        assessments = df.to_dict("records")
        print(f"  Loaded {len(assessments)} rows from {LISTING_CSV}")

    else:
        print("\nPhase 1: Fetching catalog listing pages ...")
        assessments = fetch_listing_pages()
        if not assessments:
            print("ERROR: no assessments found. Exiting.")
            return
        save_listing(assessments)
        print(f"  {len(assessments)} assessments -> {LISTING_CSV}")

    # initialise missing fields so the CSV columns are consistent
    for a in assessments:
        for col in ("description", "duration", "job_levels",
                     "languages", "test_types", "rich_text"):
            a.setdefault(col, "")

    # --- detail scraping ---
    needs_scraping = any(not a.get("description", "").strip() for a in assessments)
    if needs_scraping:
        print("\nPhase 2: Detail page scraping ...")
        run_detail_phase(assessments)
    else:
        print("\nPhase 2: All detail pages already scraped, skipping.")

    # --- embeddings + FAISS ---
    print("\nPhase 3: Building embeddings and FAISS index ...")
    build_index(assessments)

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"  Done in {elapsed:.0f}s.  Next step: streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
