"""
=============================================================================
SHL Assessment Recommendation Engine — Evaluation Script
=============================================================================
Uses the 10 labeled training queries (from the SHL assignment) to evaluate
the recommendation engine's performance using Recall@K and MAP@K metrics.

The training queries are defined with expected assessment categories/types.
This script:
  1. Runs each query through the recommend() function
  2. Computes Recall@K (K=3, 5, 10)
  3. Computes Mean Average Precision (MAP@K)
  4. Outputs a summary table and overall metrics

Usage:
    python evaluate_train.py
=============================================================================
"""

import sys
import os

# Add parent directory to path so we can import from app.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

# Import core recommendation function from app.py
from app import recommend, load_model, load_faiss_index, load_metadata


# ---------------------------------------------------------------------------
# Training Queries with expected relevant assessment categories
# ---------------------------------------------------------------------------
# Each entry: (query, list_of_expected_test_types_or_keywords)
# These represent the types of assessments that SHOULD appear in results.
TRAIN_QUERIES = [
    {
        "query": "I am hiring for Java developers who can also handle customer calls. Suggest me some assessments.",
        "expected_types": ["K"],  # Knowledge tests for Java + some P/B for customer handling
        "expected_keywords": ["java", "customer", "developer"],
    },
    {
        "query": "Looking for an assessment to evaluate leadership skills of mid-level managers.",
        "expected_types": ["P", "C"],
        "expected_keywords": ["leadership", "manager"],
    },
    {
        "query": "I need a personality test for entry-level sales representatives.",
        "expected_types": ["P"],
        "expected_keywords": ["personality", "sales"],
    },
    {
        "query": "Suggest assessments for hiring data analysts with SQL and Python skills.",
        "expected_types": ["K"],
        "expected_keywords": ["data", "sql", "python", "analyst"],
    },
    {
        "query": "What assessments are available for evaluating teamwork and collaboration?",
        "expected_types": ["P", "B", "C"],
        "expected_keywords": ["teamwork", "collaboration"],
    },
    {
        "query": "I want to test numerical reasoning and problem-solving abilities.",
        "expected_types": ["A", "K"],
        "expected_keywords": ["numerical", "reasoning"],
    },
    {
        "query": "Recommend assessments for senior executive strategic decision-making.",
        "expected_types": ["P", "C", "B"],
        "expected_keywords": ["executive", "strategic", "decision"],
    },
    {
        "query": "Looking for simulations or situational judgment tests for customer service roles.",
        "expected_types": ["S", "B"],
        "expected_keywords": ["simulation", "customer service", "situational"],
    },
    {
        "query": "Suggest knowledge tests for .NET and cloud technologies.",
        "expected_types": ["K"],
        "expected_keywords": [".net", "cloud"],
    },
    {
        "query": "I need behavioral assessments for a graduate trainee program.",
        "expected_types": ["P", "B"],
        "expected_keywords": ["behavioral", "graduate", "trainee"],
    },
]


# ---------------------------------------------------------------------------
# Evaluation Metrics
# ---------------------------------------------------------------------------
def is_relevant(result: dict, expected_types: list, expected_keywords: list) -> bool:
    """
    Determine if a result is relevant based on:
    1. Test type overlap with expected types
    2. Keyword overlap with expected keywords
    """
    # Check test type overlap
    result_types = set(result.get("test_types", "").split())
    type_match = bool(result_types.intersection(set(expected_types)))

    # Check keyword overlap (in name or rich_text)
    text = (result.get("name", "") + " " + result.get("rich_text", "")).lower()
    keyword_match = any(kw.lower() in text for kw in expected_keywords)

    return type_match or keyword_match


def recall_at_k(results: list, expected_types: list, expected_keywords: list, k: int) -> float:
    """
    Compute Recall@K: fraction of results in top-K that are relevant.
    Since we don't have an exhaustive ground truth set, we use a
    precision-like metric (relevant results / K).
    """
    if not results:
        return 0.0
    top_results = results[:k]
    relevant_count = sum(
        1 for r in top_results
        if is_relevant(r, expected_types, expected_keywords)
    )
    return relevant_count / k


def average_precision_at_k(results: list, expected_types: list, expected_keywords: list, k: int) -> float:
    """
    Compute Average Precision@K.
    """
    if not results:
        return 0.0

    top_results = results[:k]
    relevant_count = 0
    precision_sum = 0.0

    for i, r in enumerate(top_results, 1):
        if is_relevant(r, expected_types, expected_keywords):
            relevant_count += 1
            precision_sum += relevant_count / i

    if relevant_count == 0:
        return 0.0

    return precision_sum / relevant_count


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  SHL Assessment Recommendation Engine — Evaluation")
    print("=" * 70)

    # Verify data is loaded
    index = load_faiss_index()
    metadata = load_metadata()
    if index is None or metadata is None:
        print("❌ Error: FAISS index or metadata not found.")
        print("   Please run scraper.py first to generate the data.")
        sys.exit(1)

    print(f"✅ Loaded {index.ntotal} assessments from FAISS index.\n")

    # Run evaluation
    results_data = []
    all_recall_3 = []
    all_recall_5 = []
    all_recall_10 = []
    all_ap_10 = []

    for i, tq in enumerate(TRAIN_QUERIES, 1):
        query = tq["query"]
        expected_types = tq["expected_types"]
        expected_keywords = tq["expected_keywords"]

        print(f"Query {i}: {query[:80]}...")

        # Get recommendations
        recs = recommend(query, top_k=10, balance=True)

        r3 = recall_at_k(recs, expected_types, expected_keywords, 3)
        r5 = recall_at_k(recs, expected_types, expected_keywords, 5)
        r10 = recall_at_k(recs, expected_types, expected_keywords, 10)
        ap10 = average_precision_at_k(recs, expected_types, expected_keywords, 10)

        all_recall_3.append(r3)
        all_recall_5.append(r5)
        all_recall_10.append(r10)
        all_ap_10.append(ap10)

        print(f"  Recall@3={r3:.2f}  Recall@5={r5:.2f}  Recall@10={r10:.2f}  AP@10={ap10:.2f}")
        print(f"  Top-3: {[r['name'][:40] for r in recs[:3]]}")
        print()

        results_data.append({
            "query": query[:80],
            "Recall@3": round(r3, 3),
            "Recall@5": round(r5, 3),
            "Recall@10": round(r10, 3),
            "AP@10": round(ap10, 3),
        })

    # Summary
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Mean Recall@3  = {np.mean(all_recall_3):.3f}")
    print(f"  Mean Recall@5  = {np.mean(all_recall_5):.3f}")
    print(f"  Mean Recall@10 = {np.mean(all_recall_10):.3f}")
    print(f"  MAP@10         = {np.mean(all_ap_10):.3f}")
    print("=" * 70)

    # Save results
    df = pd.DataFrame(results_data)
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data",
        "evaluation_results.csv"
    )
    df.to_csv(output_path, index=False)
    print(f"\n📊 Evaluation results saved to {output_path}")


if __name__ == "__main__":
    main()
