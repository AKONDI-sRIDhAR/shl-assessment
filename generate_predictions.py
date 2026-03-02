"""
generate_predictions.py -- generate test set predictions CSV
Also evaluates against the labeled train set (Recall@10).
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
from utils import recommend

# load datasets
xl = pd.ExcelFile("data/Gen_AI_Dataset.xlsx")
train_df = xl.parse("Train-Set")
test_df = xl.parse("Test-Set")

# --- evaluate on train set ---
print("=" * 50)
print("  Evaluation on Labeled Train Set")
print("=" * 50)

# group train labels by query
train_groups = {}
for _, row in train_df.iterrows():
    q = str(row.iloc[0]).strip()
    url = str(row.iloc[1]).strip().rstrip("/")
    if q not in train_groups:
        train_groups[q] = set()
    train_groups[q].add(url)

recalls = []
for q, true_urls in train_groups.items():
    recs = recommend(q, top_k=10, balance=True)
    rec_urls = set()
    for r in recs:
        u = r.get("url", "").strip().rstrip("/")
        # also try with /solutions prefix since SHL has both URL formats
        rec_urls.add(u)
        rec_urls.add(u.replace("/products/product-catalog/", "/solutions/products/product-catalog/"))
        rec_urls.add(u.replace("/solutions/products/product-catalog/", "/products/product-catalog/"))

    hits = len(true_urls & rec_urls)
    recall = hits / len(true_urls) if true_urls else 0
    recalls.append(recall)
    print(f"\n  Query: {q[:80]}...")
    print(f"  True URLs: {len(true_urls)}, Hits: {hits}, Recall@10: {recall:.2f}")

mean_recall = sum(recalls) / len(recalls) if recalls else 0
print(f"\n  Mean Recall@10: {mean_recall:.3f}")
print()

# --- generate test predictions ---
print("=" * 50)
print("  Generating Test Set Predictions")
print("=" * 50)

rows = []
for _, row in test_df.iterrows():
    q = str(row.iloc[0]).strip()
    recs = recommend(q, top_k=10, balance=True)
    print(f"\n  Query: {q[:80]}...")
    for r in recs:
        url = r.get("url", "")
        rows.append({"Query": q, "Assessment_url": url})
        print(f"    -> {r['name'][:50]} | {r['score']:.3f}")

out = pd.DataFrame(rows)
out.to_csv("akondi_sridhar.csv", index=False)
print(f"\nSaved {len(out)} predictions to akondi_sridhar.csv")
print("Columns:", list(out.columns))
print("Done.")
