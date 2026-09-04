"""
evaluate_recommender.py

Computes a real, reportable accuracy metric for your recommender -- no
manual labeling required, using data you already have.

Since there's no user click/rating history to compute true precision
against, this uses a standard proxy for content-based recommenders:
Category/Concept Precision@k -- for a random sample of papers, look up
their top-k nearest neighbors by embedding similarity, and measure what
fraction of those neighbors share at least one keyword/category tag with
the query paper. High values mean the embedding space is grouping
topically-related papers together, which is what a content-based
recommender is supposed to do.

This is a WEAK proxy, not ground-truth relevance -- say so explicitly in
your paper (e.g. "as a proxy for relevance in the absence of user
interaction data, we measured category-overlap precision@k"). If you have
time before submission, a small human-judged sample (~20-30 query-result
pairs rated relevant/irrelevant by a few people) would give you a much
stronger, citable number -- this script's output is a reasonable stand-in
until then.

Run from datasets/scripts/:
    python evaluate_recommender.py --k 5 --sample-size 200
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app import create_app
from app.models.paper import Paper


def load_embeddings():
    """Loads the embeddings.npy array build_embeddings.py produced."""
    emb_path = Path(__file__).resolve().parent.parent / "embeddings" / "embeddings.npy"
    if not emb_path.exists():
        print(f"ERROR: {emb_path} not found. Run build_embeddings.py first.")
        sys.exit(1)
    return np.load(emb_path)


def cosine_sim_matrix_row(query_vec, all_vecs):
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    all_norms = all_vecs / (np.linalg.norm(all_vecs, axis=1, keepdims=True) + 1e-10)
    return all_norms @ query_norm


def tag_set(paper) -> set:
    """Splits a paper's keywords/categories field into a comparable token set."""
    if not paper.keywords:
        return set()
    return {t.strip().lower() for t in paper.keywords.replace(",", " ").split() if t.strip()}


def run_eval(k: int, sample_size: int, seed: int = 42):
    app = create_app("development")
    with app.app_context():
        papers = Paper.query.order_by(Paper.id).all()
        embeddings = load_embeddings()

        if len(papers) != embeddings.shape[0]:
            print(
                f"WARNING: {len(papers)} papers in DB but {embeddings.shape[0]} "
                f"embedding rows -- these should match 1:1 by insertion order. "
                f"Re-run build_embeddings.py if you've ingested new papers since "
                f"the last embedding build."
            )

        n = min(len(papers), embeddings.shape[0])
        rng = random.Random(seed)
        sample_size = min(sample_size, n)
        query_indices = rng.sample(range(n), sample_size)

        precisions = []
        similarities = []
        skipped_no_tags = 0

        for qi in query_indices:
            query_paper = papers[qi]
            query_tags = tag_set(query_paper)
            if not query_tags:
                skipped_no_tags += 1
                continue

            sims = cosine_sim_matrix_row(embeddings[qi], embeddings)
            sims[qi] = -1  # exclude the paper itself
            top_k_idx = np.argsort(-sims)[:k]

            hits = 0
            for ti in top_k_idx:
                neighbor_tags = tag_set(papers[ti])
                if query_tags & neighbor_tags:
                    hits += 1
                similarities.append(sims[ti])

            precisions.append(hits / k)

        if not precisions:
            print("No queries had usable category tags -- can't compute precision@k.")
            return

        mean_precision = float(np.mean(precisions))
        mean_similarity = float(np.mean(similarities))

        print("=" * 60)
        print("RECOMMENDER EVALUATION (proxy metric, not ground-truth)")
        print("=" * 60)
        print(f"Papers in corpus           : {n}")
        print(f"Queries sampled            : {len(precisions)} (seed={seed})")
        print(f"Skipped (no category tags) : {skipped_no_tags}")
        print(f"k                          : {k}")
        print(f"Mean Category Precision@{k} : {mean_precision:.3f}")
        print(f"Mean cosine similarity of top-{k} neighbors: {mean_similarity:.3f}")
        print()
        print(
            "Suggested phrasing for your paper:\n"
            f'  "Using category-overlap as a proxy for relevance in the absence\n'
            f'  of user interaction data, the system achieved a mean Category\n'
            f'  Precision@{k} of {mean_precision:.1%} over {len(precisions)} randomly\n'
            f'  sampled queries against a corpus of {n} papers, with a mean\n'
            f'  cosine similarity of {mean_similarity:.3f} among retrieved neighbors,\n'
            f'  indicating that the embedding space groups topically related\n'
            f'  papers together."'
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_eval(args.k, args.sample_size, args.seed)
