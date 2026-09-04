"""
recommender.py

High-level recommendation logic, built purely from paper_ids, vectors, and
text -- no Flask, no SQLAlchemy. app/services/recommendation_service.py is
responsible for translating to/from actual Paper database rows.

Three capabilities, matching the Phase 7 requirements:

1. get_similar_papers(paper_id, ...)      -> "More like this" on a paper's
                                              detail page
2. semantic_search(query_text, ...)        -> meaning-based search, as
                                              opposed to search_service.py's
                                              lexical/keyword search
3. personalized_recommendations(...)       -> dashboard "Recommended for
                                              you", built from the centroid
                                              of a user's bookmarked papers
"""

import numpy as np

from app.ml_engine.vector_store import VectorStore
from app.ml_engine.embedder import SBERTEmbedder


def get_similar_papers(vector_store: VectorStore, paper_id: int, top_k: int = 5):
    """
    Finds papers most similar to a given paper, by looking up that paper's
    own stored embedding and searching for its nearest neighbors.

    Returns:
        List of (paper_id, score) tuples. Empty list if paper_id has no
        stored embedding (e.g. embeddings haven't been (re)built since it
        was added).
    """
    query_vector = vector_store.get_vector(paper_id)
    if query_vector is None:
        return []

    # Exclude the paper itself -- it would otherwise always be its own
    # top result with a similarity score of 1.0, which is useless as a
    # recommendation.
    return vector_store.search(query_vector, top_k=top_k, exclude_ids={paper_id})


def semantic_search(embedder: SBERTEmbedder, vector_store: VectorStore, query_text: str, top_k: int = 10):
    """
    Encodes a free-text query and finds the papers whose abstracts are
    most semantically similar to it -- this is the "understands meaning,
    not just keywords" search required by the project's core problem
    statement.
    """
    query_vector = embedder.encode(query_text)
    return vector_store.search(query_vector, top_k=top_k)


def personalized_recommendations(
    vector_store: VectorStore,
    bookmarked_paper_ids,
    top_k: int = 5,
    exclude_ids=None,
):
    """
    Computes a user's "interest vector" as the centroid (mean) of their
    bookmarked papers' embeddings, then finds papers most similar to that
    centroid. This is what makes the dashboard "Recommended for you"
    personalized rather than generic -- two users with different bookmark
    histories get different recommendations from the same corpus.

    Returns:
        Empty list if the user has no bookmarks with stored embeddings yet
        (cold-start case -- see recommendation_service.py for how the
        dashboard route handles this gracefully).
    """
    vectors = [
        vector_store.get_vector(pid)
        for pid in bookmarked_paper_ids
        if vector_store.get_vector(pid) is not None
    ]
    if not vectors:
        return []

    centroid = np.mean(vectors, axis=0)
    # Re-normalize: the mean of several unit vectors is generally NOT itself
    # unit length, and our dot-product-as-cosine-similarity shortcut (see
    # vector_store.py) requires normalized query vectors to stay correct.
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm

    exclude_ids = set(exclude_ids or []) | set(bookmarked_paper_ids)
    return vector_store.search(centroid, top_k=top_k, exclude_ids=exclude_ids)
