"""
vector_store.py

Stores and searches the paper embedding matrix. This is the module you'd
swap out for FAISS (or Pinecone, Weaviate, etc.) if the dataset grew past
where brute-force NumPy search is fast enough -- see the "Scaling beyond
NumPy" note at the bottom of this file. The public interface (search(),
get_vector(), size) is deliberately small so that swap would not require
touching recommender.py or the service layer above it.

STORAGE FORMAT
    Two parallel .npy files:
        embeddings.npy : float32 array, shape (N, D), L2-normalized rows
        paper_ids.npy  : int64 array, shape (N,)
    Row i of embeddings.npy is the embedding vector for the paper whose
    database id is paper_ids[i]. This indirection (rather than assuming
    row index == paper.id) means we don't have to keep a dense, gapless
    Paper.id range, and papers can be deleted/added without renumbering
    everything.

WHY .npy INSTEAD OF SQL
    - float32 .npy is a raw, contiguous binary dump -- no per-row overhead,
      no JSON/text encoding cost. For reference: 30,000 papers x 384 dims x
      4 bytes = ~44MB. Storing the same vectors as a SQL TEXT/JSON column
      would be several times larger and far slower to load into memory in
      bulk for search.
    - NumPy can memory-map or bulk-load this format near-instantly, whereas
      reading 30,000 rows out of SQLite one at a time is comparatively slow.

WHY L2-NORMALIZED VECTORS + DOT PRODUCT INSTEAD OF EXPLICIT COSINE SIMILARITY
    cosine_similarity(a, b) = (a . b) / (|a| * |b|). If every vector is
    pre-normalized to unit length at embedding time (see embedder.py), then
    |a| = |b| = 1 for all vectors, so cosine_similarity(a, b) reduces to
    just (a . b) -- a single dot product, no division, no repeated norm
    computation at query time. For scoring one query against N papers, this
    is the difference between N divisions per query (wasted work, since the
    result is identical) and zero.
"""

from pathlib import Path

import numpy as np


class VectorStore:
    def __init__(self, embeddings: np.ndarray, paper_ids: np.ndarray):
        """
        Args:
            embeddings: (N, D) float32 array, L2-normalized rows
            paper_ids: (N,) int array; paper_ids[i] is the DB id for embeddings[i]
        """
        if embeddings.shape[0] != paper_ids.shape[0]:
            raise ValueError(
                f"embeddings has {embeddings.shape[0]} rows but paper_ids has "
                f"{paper_ids.shape[0]} entries -- they must match 1:1."
            )

        self.embeddings = embeddings.astype(np.float32, copy=False)
        self.paper_ids = paper_ids.astype(np.int64, copy=False)

        # id -> row index, for O(1) lookup in get_vector()
        self._id_to_row = {pid: i for i, pid in enumerate(self.paper_ids)}

    @property
    def size(self) -> int:
        return self.embeddings.shape[0]

    @property
    def dim(self) -> int:
        return self.embeddings.shape[1] if self.embeddings.shape[0] else 0

    def get_vector(self, paper_id: int) -> np.ndarray:
        """Returns the stored embedding for a given paper id, or None if absent."""
        row = self._id_to_row.get(paper_id)
        return self.embeddings[row] if row is not None else None

    def search(self, query_vector: np.ndarray, top_k: int = 5, exclude_ids=None):
        """
        Finds the top_k papers most similar to query_vector.

        Args:
            query_vector: (D,) float32, should already be L2-normalized
                (embedder.py does this automatically)
            top_k: number of results to return
            exclude_ids: optional set/list of paper_ids to exclude from
                results (e.g. exclude the query paper itself, or papers the
                user has already bookmarked)

        Returns:
            List of (paper_id, similarity_score) tuples, sorted by score
            descending, similarity_score in [-1, 1] (in practice [0, 1] for
            semantically related text).

        PERFORMANCE NOTE: `self.embeddings @ query_vector` is a single
        vectorized matrix-vector multiply computed by NumPy's underlying
        BLAS library, not a Python loop. For N=30,000 and D=384 this runs
        in low single-digit milliseconds on typical hardware -- well within
        our <1-2 second response time requirement from Phase 1.
        """
        if self.size == 0 or top_k <= 0:
            return []

        scores = self.embeddings @ query_vector  # (N,) dot products = cosine similarities

        exclude_ids = set(exclude_ids or [])

        # argpartition is O(N) vs full sort's O(N log N) -- we only need the
        # top_k, not a fully sorted order over all N candidates, until the
        # very last step where we sort just those top_k.
        k = min(top_k + len(exclude_ids), self.size)  # over-fetch to survive exclusions
        candidate_rows = np.argpartition(-scores, k - 1)[:k] if k < self.size else np.arange(self.size)
        candidate_rows = candidate_rows[np.argsort(-scores[candidate_rows])]

        results = []
        for row in candidate_rows:
            pid = int(self.paper_ids[row])
            if pid in exclude_ids:
                continue
            results.append((pid, float(scores[row])))
            if len(results) >= top_k:
                break

        return results

    def save(self, embeddings_path: Path, ids_path: Path):
        embeddings_path = Path(embeddings_path)
        ids_path = Path(ids_path)
        embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(embeddings_path, self.embeddings)
        np.save(ids_path, self.paper_ids)

    @classmethod
    def load(cls, embeddings_path: Path, ids_path: Path) -> "VectorStore":
        embeddings = np.load(Path(embeddings_path))
        paper_ids = np.load(Path(ids_path))
        return cls(embeddings, paper_ids)


# -----------------------------------------------------------------------------
# SCALING BEYOND NUMPY (documented here, not implemented, per Phase 2's
# feasibility analysis -- our target scale of 20k-50k papers doesn't need it)
#
# If the corpus grows past roughly 200k-500k papers, brute-force search
# becomes the bottleneck. At that point, swap this class's internals for
# FAISS (pip install faiss-cpu) while keeping the exact same public
# interface, e.g.:
#
#     import faiss
#     index = faiss.IndexFlatIP(dim)   # IP = inner product = cosine sim
#                                       # for L2-normalized vectors
#     index.add(embeddings)
#     scores, row_indices = index.search(query_vector.reshape(1, -1), top_k)
#
# Because recommender.py and recommendation_service.py only ever call
# VectorStore.search()/get_vector()/size, this swap is contained entirely
# to this one file.
# -----------------------------------------------------------------------------
