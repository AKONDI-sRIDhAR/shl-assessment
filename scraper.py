"""
=============================================================================
SHL Assessment Catalog Scraper
=============================================================================
Scrapes "Individual Test Solutions" (type=1) from the SHL Product Catalog:
  https://www.shl.com/products/product-catalog/?type=1

Pagination: ?type=1&start=0,12,24,...,372  (~32 pages, ~380 assessments)

For each assessment it extracts:
  - name          : Assessment title
  - url           : Full absolute URL to the detail page
  - test_types    : Space-separated test type letters (K, P, A, B, etc.)
  - rich_text     : Combined text blob for embedding generation

Detail page fields:
  - description   : Full description paragraph
  - duration      : Assessment length / completion time
  - job_levels    : Target job levels
  - languages     : Available languages

Output  : data/assessments.csv
Schedule: Run once; data changes infrequently.
=============================================================================
"""

import os
import re
import csv
import time
import pickle
import requests
import numpy as np
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_URL = "https://www.shl.com/products/product-catalog/"
CATALOG_URL = BASE_URL + "?type=1"
START_VALUES = list(range(0, 384, 12))  # 0, 12, 24, ..., 372
SLEEP_BETWEEN_DETAIL = 0.6              # polite delay per detail request
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "assessments.csv")
METADATA_PKL = os.path.join(OUTPUT_DIR, "metadata.pkl")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


# ---------------------------------------------------------------------------
# Helper: Parse test-type badges from a page's HTML
# ---------------------------------------------------------------------------
def extract_test_types_from_soup(soup):
    """
    SHL pages show test-type badges as single-letter spans inside a
    styled container.  We look for the common patterns:
      - <span class="product-catalogue__key">K</span>
      - Or the letters A, B, C, D, E, K, P, S inside badge containers
    Returns a space-separated string like "K P".
    """
    valid_letters = {"A", "B", "C", "D", "E", "K", "P", "S"}
    found = []

    # Method 1: look for specific badge/key spans
    for span in soup.select("span.product-catalogue__key"):
        letter = span.get_text(strip=True).upper()
        if letter in valid_letters and letter not in found:
            found.append(letter)

    # Method 2: fallback — look inside catalogue-badge containers
    if not found:
        for el in soup.select(".product-catalogue-badge, .catalogue-badge, .badge"):
            letter = el.get_text(strip=True).upper()
            if len(letter) == 1 and letter in valid_letters and letter not in found:
                found.append(letter)

    # Method 3: broad fallback — look for "Test Type:" text in the page
    if not found:
        text = soup.get_text()
        match = re.search(r"Test Type[s]?:\s*([A-Z](?:\s*,?\s*[A-Z])*)", text)
        if match:
            for ch in match.group(1):
                if ch in valid_letters and ch not in found:
                    found.append(ch)

    return " ".join(found)


# ---------------------------------------------------------------------------
# Step 1: Scrape catalog listing pages to get assessment names + URLs
# ---------------------------------------------------------------------------
def scrape_catalog_listing():
    """
    Iterate over all pagination pages and collect (name, url) tuples.
    """
    assessments = []
    seen_urls = set()

    for start in START_VALUES:
        page_url = f"{CATALOG_URL}&start={start}" if start > 0 else CATALOG_URL
        print(f"[Catalog] Fetching page start={start} -> {page_url}")

        try:
            resp = requests.get(page_url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"  [WARN] Error fetching {page_url}: {e}")
            continue

        soup = BeautifulSoup(resp.text, "lxml")

        # Find assessment links — they point to /products/product-catalog/view/<slug>/
        for a_tag in soup.select("a[href*='/products/product-catalog/view/']"):
            href = a_tag.get("href", "").strip()
            name = a_tag.get_text(strip=True)

            # Skip empty / navigation links
            if not name or not href:
                continue

            # Build absolute URL
            if href.startswith("/"):
                href = "https://www.shl.com" + href

            if href not in seen_urls:
                seen_urls.add(href)
                assessments.append({"name": name, "url": href})

        time.sleep(0.3)  # small delay between listing pages

    print(f"\n[OK] Found {len(assessments)} unique assessments from catalog.\n")
    return assessments


# ---------------------------------------------------------------------------
# Step 2: Visit each detail page and extract rich metadata
# ---------------------------------------------------------------------------
def scrape_detail_page(url):
    """
    Visit a single assessment detail page and extract:
      - description, duration, job_levels, languages, test_types
    Returns a dict with these fields.
    """
    detail = {
        "description": "",
        "duration": "",
        "job_levels": "",
        "languages": "",
        "test_types": "",
    }

    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"    [WARN] Error fetching detail {url}: {e}")
        return detail

    soup = BeautifulSoup(resp.text, "lxml")
    page_text = soup.get_text(separator="\n")

    # --- Description ---
    # Usually under <h4>Description</h4> or similar
    desc_header = soup.find(
        lambda tag: tag.name in ["h4", "h3", "h2", "strong"]
        and "description" in tag.get_text(strip=True).lower()
    )
    if desc_header:
        # Grab next sibling paragraph(s)
        desc_parts = []
        for sib in desc_header.find_next_siblings():
            if sib.name in ["h4", "h3", "h2"]:
                break
            text = sib.get_text(strip=True)
            if text:
                desc_parts.append(text)
        detail["description"] = " ".join(desc_parts)
    else:
        # Fallback: grab from meta og:description
        og = soup.find("meta", attrs={"property": "og:description"})
        if og:
            detail["description"] = og.get("content", "")

    # --- Duration / Assessment Length ---
    duration_match = re.search(
        r"(?:Approximate Completion Time|Assessment [Ll]ength)[^\d]*(\d+)\s*(?:minutes|mins?)?",
        page_text,
        re.IGNORECASE,
    )
    if duration_match:
        detail["duration"] = f"{duration_match.group(1)} minutes"
    else:
        # Broader search
        dur2 = re.search(r"(\d+)\s*(?:minutes|mins)", page_text, re.IGNORECASE)
        if dur2:
            detail["duration"] = f"{dur2.group(1)} minutes"

    # --- Job Levels ---
    jl_header = soup.find(
        lambda tag: tag.name in ["h4", "h3", "h2", "strong"]
        and "job level" in tag.get_text(strip=True).lower()
    )
    if jl_header:
        parts = []
        for sib in jl_header.find_next_siblings():
            if sib.name in ["h4", "h3", "h2"]:
                break
            text = sib.get_text(strip=True)
            if text:
                parts.append(text)
        detail["job_levels"] = ", ".join(parts)
    else:
        jl_match = re.search(
            r"Job [Ll]evels?\s*[:\-]?\s*(.+?)(?:\n|$)", page_text
        )
        if jl_match:
            detail["job_levels"] = jl_match.group(1).strip().rstrip(",")

    # --- Languages ---
    lang_header = soup.find(
        lambda tag: tag.name in ["h4", "h3", "h2", "strong"]
        and "language" in tag.get_text(strip=True).lower()
    )
    if lang_header:
        parts = []
        for sib in lang_header.find_next_siblings():
            if sib.name in ["h4", "h3", "h2"]:
                break
            text = sib.get_text(strip=True)
            if text:
                parts.append(text)
        detail["languages"] = ", ".join(parts)
    else:
        lang_match = re.search(
            r"Languages?\s*[:\-]?\s*(.+?)(?:\n|$)", page_text
        )
        if lang_match:
            detail["languages"] = lang_match.group(1).strip().rstrip(",")

    # --- Test Types ---
    detail["test_types"] = extract_test_types_from_soup(soup)

    return detail


# ---------------------------------------------------------------------------
# Step 3: Build rich_text for embedding generation
# ---------------------------------------------------------------------------
def build_rich_text(name, detail):
    """
    Compose the rich_text string that will be embedded.
    """
    return (
        f"Title: {name}\n"
        f"Description: {detail['description']}\n"
        f"Test Types: {detail['test_types']}\n"
        f"Duration: {detail['duration']}\n"
        f"Job Levels: {detail['job_levels']}"
    )


# ---------------------------------------------------------------------------
# Step 4: Generate embeddings and FAISS index
# ---------------------------------------------------------------------------
def generate_embeddings_and_index(assessments):
    """
    Generate sentence-transformer embeddings for all assessments
    and build a FAISS index for fast similarity search.
    """
    from sentence_transformers import SentenceTransformer
    import faiss

    print("\n[LOAD] Loading sentence-transformers model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Extract rich_text for embedding
    texts = [a["rich_text"] for a in assessments]
    print(f"[INFO] Generating embeddings for {len(texts)} assessments...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    # Build FAISS index (Inner Product = cosine similarity when vectors are normalized)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner Product on normalized vectors = cosine sim
    index.add(embeddings)

    # Save FAISS index
    faiss_path = os.path.join(OUTPUT_DIR, "faiss_index.bin")
    faiss.write_index(index, faiss_path)
    print(f"[OK] FAISS index saved to {faiss_path}  (dim={dim}, n={index.ntotal})")

    # Save metadata (list of dicts with name, url, test_types, rich_text, etc.)
    with open(METADATA_PKL, "wb") as f:
        pickle.dump(assessments, f)
    print(f"[OK] Metadata saved to {METADATA_PKL}")

    return index, embeddings


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 70)
    print("  SHL Assessment Catalog Scraper")
    print("=" * 70)

    # Step 1: Scrape listing pages
    assessments = scrape_catalog_listing()

    if not assessments:
        print("[ERROR] No assessments found. Exiting.")
        return

    # Step 2: Scrape each detail page
    total = len(assessments)
    for idx, a in enumerate(assessments, 1):
        print(f"  [{idx}/{total}] Scraping detail: {a['name']}")
        detail = scrape_detail_page(a["url"])

        # Merge detail fields into the assessment dict
        a["test_types"] = detail["test_types"]
        a["description"] = detail["description"]
        a["duration"] = detail["duration"]
        a["job_levels"] = detail["job_levels"]
        a["languages"] = detail["languages"]

        # Build rich_text
        a["rich_text"] = build_rich_text(a["name"], detail)

        # Polite sleep between detail page requests
        time.sleep(SLEEP_BETWEEN_DETAIL)

    # Step 3: Save to CSV
    fieldnames = ["name", "url", "test_types", "rich_text"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(assessments)
    print(f"\n[OK] Saved {len(assessments)} assessments to {OUTPUT_CSV}")

    # Step 4: Generate embeddings & FAISS index
    generate_embeddings_and_index(assessments)

    print("\n" + "=" * 70)
    print("  [DONE] Scraping complete! All data is in the data/ directory.")
    print("=" * 70)


if __name__ == "__main__":
    main()
