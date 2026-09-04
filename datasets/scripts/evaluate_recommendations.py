"""
evaluate_recommendations.py

Quantitatively evaluates the recommendation engine, producing the numbers
your project report needs to back up the claim "semantic search is better
than keyword matching" with evidence, not just an assertion.

TWO THINGS ARE MEASURED:

1. RECOMMENDATION QUALITY: Category-Match Precision@K
   arXiv's category codes (e.g. "cs.LG", "cs.IR") are assigned by the
   original authors/moderators, not by us -- they're a reasonable,
   independent proxy for topical relevance. For each paper, we take its
   top-K SBERT-recommended "similar papers" and measure what fraction
   share at least one category code with the source paper. We compute the
   SAME metric using a TF-IDF + cosine similarity baseline (the "before"
   in a before/after comparison) so the report can show a concrete number,
   not just "SBERT is better in theory."

   IMPORTANT CAVEAT (stated honestly, not glossed over): shared categories
   are a PROXY for relevance, not ground truth human relevance judgment.
   Two papers can be genuinely topically related while carrying different
   category codes (interdisciplinary work), and two papers can share a
   category while being only loosely related. Treat this metric as
   directional evidence, and mention this caveat in your project report --
   a real production evaluation would supplement it with human relevance
   ratings on a sample of query-result pairs.

2. PERFORMANCE: average query latency for both approaches, since Phase 1's
   non-functional requirement was sub-1-2-second response time.

Usage (run from the project root, after build_embeddings.py has been run):
    python datasets/scripts/evaluate_recommendations.py --top-k 5
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from app import create_app  # noqa: E402
from app.models.paper import Paper  # noqa: E402
from app.ml_engine.vector_store import VectorStore  # noqa: E402
from app.ml_engine.recommender import get_similar_papers  # noqa: E402


def category_set(keywords_str: str) -> set:
    """arXiv category strings look like 'cs.LG cs.AI' -- split on whitespace."""
    return set((keywords_str or "").split())


def evaluate_sbert(papers, vector_store: VectorStore, top_k: int):
    """Category-match precision@K and latency using the real SBERT engine."""
    categories_by_id = {p.id: category_set(p.keywords) for p in papers}
    eligible_ids = [pid for pid, cats in categories_by_id.items() if cats]

    if len(eligible_ids) < 2:
        return None  # not enough category-labeled papers to evaluate meaningfully

    precisions = []
    latencies = []

    for pid in eligible_ids:
        start = time.perf_counter()
        results = get_similar_papers(vector_store, pid, top_k=top_k)
        latencies.append(time.perf_counter() - start)

        if not results:
            continue

        source_cats = categories_by_id[pid]
        hits = sum(
            1 for (result_id, _score) in results
            if categories_by_id.get(result_id) and source_cats & categories_by_id[result_id]
        )
        precisions.append(hits / len(results))

    return {
        "mean_precision_at_k": float(np.mean(precisions)) if precisions else 0.0,
        "num_evaluated": len(precisions),
        "avg_latency_ms": float(np.mean(latencies)) * 1000 if latencies else 0.0,
    }


def evaluate_tfidf_baseline(papers, top_k: int):
    """
    Same category-match precision@K metric, but using TF-IDF + cosine
    similarity instead of SBERT -- this is the lexical baseline referenced
    throughout the project (Phase 2's architecture decision, Phase 6's
    search_service.py). Comparing against it is what turns "we used SBERT"
    into "we measured that SBERT outperforms keyword-based similarity."
    """
    texts = [f"{p.title}. {p.abstract}" for p in papers]
    ids = [p.id for p in papers]
    categories_by_id = {p.id: category_set(p.keywords) for p in papers}

    start = time.perf_counter()
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(texts)
    build_time = time.perf_counter() - start

    eligible_indices = [i for i, pid in enumerate(ids) if categories_by_id[pid]]
    if len(eligible_indices) < 2:
        return None

    precisions = []
    latencies = []

    sim_matrix = cosine_similarity(tfidf_matrix)  # precompute once, like an offline index

    for i in eligible_indices:
        start = time.perf_counter()
        scores = sim_matrix[i].copy()
        scores[i] = -1  # exclude self
        top_indices = np.argsort(-scores)[:top_k]
        latencies.append(time.perf_counter() - start)

        source_cats = categories_by_id[ids[i]]
        hits = sum(
            1 for j in top_indices
            if categories_by_id[ids[j]] and source_cats & categories_by_id[ids[j]]
        )
        precisions.append(hits / len(top_indices) if len(top_indices) else 0.0)

    return {
        "mean_precision_at_k": float(np.mean(precisions)) if precisions else 0.0,
        "num_evaluated": len(precisions),
        "avg_latency_ms": float(np.mean(latencies)) * 1000 if latencies else 0.0,
        "index_build_time_ms": build_time * 1000,
    }


def main(top_k: int):
    app = create_app("development")
    with app.app_context():
        papers = Paper.query.order_by(Paper.id.asc()).all()
        config = app.config

        embeddings_path = Path(config["EMBEDDINGS_DIR"]) / "embeddings.npy"
        ids_path = Path(config["EMBEDDINGS_DIR"]) / "paper_ids.npy"

        if not embeddings_path.exists():
            print(f"ERROR: {embeddings_path} not found. Run build_embeddings.py first.")
            sys.exit(1)

        vector_store = VectorStore.load(embeddings_path, ids_path)

        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("RECOMMENDATION ENGINE EVALUATION REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Total papers in database : {len(papers)}")
        report_lines.append(f"Total papers with embeddings: {vector_store.size}")
        report_lines.append(f"Top-K evaluated           : {top_k}")
        report_lines.append("")
        report_lines.append(
            "Metric: Category-Match Precision@K (fraction of top-K recommended\n"
            "papers that share at least one arXiv category with the source paper).\n"
            "This is a PROXY for relevance, not human judgment -- see script\n"
            "docstring for the full caveat."
        )
        report_lines.append("")

        sbert_results = evaluate_sbert(papers, vector_store, top_k)
        if sbert_results is None:
            report_lines.append("SBERT evaluation: SKIPPED (not enough category-labeled papers)")
        else:
            report_lines.append("-- SBERT (Sentence-Transformers + cosine similarity) --")
            report_lines.append(f"  Papers evaluated      : {sbert_results['num_evaluated']}")
            report_lines.append(f"  Mean Precision@{top_k}      : {sbert_results['mean_precision_at_k']:.3f}")
            report_lines.append(f"  Avg query latency     : {sbert_results['avg_latency_ms']:.3f} ms")

        report_lines.append("")

        tfidf_results = evaluate_tfidf_baseline(papers, top_k)
        if tfidf_results is None:
            report_lines.append("TF-IDF baseline evaluation: SKIPPED (not enough category-labeled papers)")
        else:
            report_lines.append("-- Baseline: TF-IDF + cosine similarity (lexical) --")
            report_lines.append(f"  Papers evaluated      : {tfidf_results['num_evaluated']}")
            report_lines.append(f"  Mean Precision@{top_k}      : {tfidf_results['mean_precision_at_k']:.3f}")
            report_lines.append(f"  Avg query latency     : {tfidf_results['avg_latency_ms']:.3f} ms")
            report_lines.append(f"  (Index build time     : {tfidf_results['index_build_time_ms']:.2f} ms, one-time)")

        report_lines.append("")
        if sbert_results and tfidf_results:
            delta = sbert_results["mean_precision_at_k"] - tfidf_results["mean_precision_at_k"]
            report_lines.append(
                f"SBERT vs TF-IDF baseline: {delta:+.3f} precision@{top_k} difference "
                f"({'SBERT ahead' if delta > 0 else 'TF-IDF ahead' if delta < 0 else 'tied'})."
            )

        report_lines.append("")
        report_lines.append(
            "NOTE: with a very small sample dataset, these numbers are illustrative\n"
            "of the pipeline working, not statistically meaningful. Re-run this\n"
            "script after loading the full arXiv subset for numbers worth citing\n"
            "in your project report."
        )

        report_text = "\n".join(report_lines)
        print(report_text)

        report_path = Path(config["EMBEDDINGS_DIR"]) / "evaluation_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the recommendation engine.")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    main(top_k=args.top_k)
