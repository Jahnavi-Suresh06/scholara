"""
embedder.py

Thin wrapper around Sentence-Transformers. This is the ONLY module that
imports the `sentence_transformers` package -- if we ever swap models or
libraries (e.g. to a different embedding provider), this is the one file
that changes.

Model choice: 'all-MiniLM-L6-v2'
    - 384-dimensional output, ~90MB download, runs comfortably on CPU
    - Encodes roughly 14,000 sentences/sec on a modern CPU -- important
      since our deployment target (Render free/hobby tier) has no GPU
    - Trained with a contrastive objective specifically so that cosine
      similarity between output vectors reflects semantic similarity,
      which is exactly the property this project's recommendation engine
      depends on (see docs/RECOMMENDATION_ARCHITECTURE.md for the full
      trade-off discussion against Word2Vec/Doc2Vec/TF-IDF/BM25).

We L2-normalize every embedding at encode time. This means cosine
similarity between two vectors reduces to a plain dot product at search
time (cos(a,b) = a.b / (|a||b|) = a.b when |a|=|b|=1) -- this is a
meaningful performance optimization once you're scoring one query vector
against tens of thousands of paper vectors.
"""

from functools import lru_cache

import numpy as np

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class SBERTEmbedder:
    """
    Loads a Sentence-Transformers model once and reuses it for all
    subsequent encode() calls. Model loading (~1-2 seconds) is expensive
    enough that we never want to do it per-request.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        # Imported lazily inside __init__ (not at module top) so that code
        # merely importing this module (e.g. for type hints, or vector_store.py
        # importing shared constants) doesn't force sentence-transformers'
        # heavier dependency chain (torch, etc.) to load unless an embedder
        # is actually instantiated.
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts, batch_size: int = 32) -> np.ndarray:
        """
        Encodes a list of strings (or a single string) into L2-normalized
        float32 vectors.

        Args:
            texts: a single string or a list of strings
            batch_size: encoding batch size (only matters for large lists)

        Returns:
            np.ndarray of shape (len(texts), embedding_dim) if given a list,
            or shape (embedding_dim,) if given a single string.
        """
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2-normalize -- see module docstring
            show_progress_bar=False,
        ).astype(np.float32)

        return vectors[0] if single_input else vectors


@lru_cache(maxsize=1)
def get_embedder(model_name: str = DEFAULT_MODEL_NAME) -> SBERTEmbedder:
    """
    Process-wide singleton accessor. lru_cache(maxsize=1) guarantees the
    (expensive) model load happens at most once per process, no matter how
    many times get_embedder() is called -- same pattern as a typical
    dependency-injection singleton, without needing a DI framework.
    """
    return SBERTEmbedder(model_name)
