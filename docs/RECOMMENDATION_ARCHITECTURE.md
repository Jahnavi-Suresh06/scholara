# Recommendation Engine Architecture

This document explains the complete recommendation pipeline built in Phase 7:
how it works, why it's built this way, and how to evaluate it. Read this
before touching `app/ml_engine/` or the recommendation-related services.

---

## 1. The Pipeline, End to End

```
                         OFFLINE (run manually, not per-request)
┌──────────────────────────────────────────────────────────────────────┐
│  datasets/scripts/build_embeddings.py                                │
│                                                                        │
│  Paper (title + abstract)  →  SBERT encode()  →  L2-normalized vector │
│           │                                              │            │
│           │                                              ▼            │
│           │                              datasets/embeddings/         │
│           │                                embeddings.npy (N x 384)   │
│           │                                paper_ids.npy  (N,)        │
│           ▼                                                           │
│  Paper.embedding_index updated in SQL (row index bookkeeping)         │
└──────────────────────────────────────────────────────────────────────┘

                         ONLINE (per web request)
┌──────────────────────────────────────────────────────────────────────┐
│  User query / paper view / dashboard load                            │
│           │                                                           │
│           ▼                                                           │
│  app/routes/*.py  (search_routes, paper_routes, dashboard_routes)     │
│           │                                                           │
│           ▼                                                           │
│  app/services/recommendation_service.py                              │
│    - loads VectorStore from disk once, caches by file mtime          │
│    - translates Paper DB rows <-> paper_id integers                  │
│           │                                                           │
│           ▼                                                           │
│  app/ml_engine/recommender.py  (pure functions, no Flask/DB)         │
│    - get_similar_papers(paper_id)                                    │
│    - semantic_search(query_text)                                     │
│    - personalized_recommendations(bookmarked_ids)                    │
│           │                                                           │
│           ▼                                                           │
│  app/ml_engine/vector_store.py                                       │
│    - query_vector = embedder.encode(text)   [only for semantic_search│
│      and personalized (centroid), not for get_similar_papers, which  │
│      reuses the already-stored vector for that paper]                │
│    - scores = embeddings @ query_vector   (cosine similarity, since   │
│      all vectors are L2-normalized -- dot product = cosine sim)      │
│    - top_k = argpartition + sort, excluding requested ids             │
│           │                                                           │
│           ▼                                                           │
│  app/ml_engine/similarity.py                                         │
│    - explain_similarity(query_text, result_text, score)              │
│    - TF-IDF shared-term extraction + qualitative score band           │
│           │                                                           │
│           ▼                                                           │
│  Rendered template (paper_detail.html / dashboard.html / search.html)│
└──────────────────────────────────────────────────────────────────────┘
```

---

## 2. Why This Architecture (Decisions Revisited From Phase 2)

### 2.1 Why Sentence-BERT over TF-IDF / Word2Vec / Doc2Vec

Re-affirming Phase 2's decision now that it's actually implemented:

| Approach | Captures word order/context? | Needs training on our corpus? | Verdict |
|---|---|---|---|
| TF-IDF | No (bag-of-words) | No | Lexical only -- exactly what the project brief says NOT to build |
| Word2Vec (averaged) | No (loses order when averaged) | Ideally yes, for domain fit | Weaker signal than SBERT for whole-abstract meaning |
| Doc2Vec | Partially | Yes | More setup for less gain than a pretrained transformer |
| **Sentence-BERT** | **Yes** (transformer attention) | **No** (pretrained) | **Chosen** -- best accuracy-to-effort ratio |

`all-MiniLM-L6-v2` was chosen specifically because it's trained with a
contrastive objective where cosine similarity between output vectors is
the trained signal, not an afterthought -- this is what lets us use plain
dot products for ranking (see 2.3).

### 2.2 Why Embeddings Are Precomputed Offline, Not Per-Request

`build_embeddings.py` runs once (and again after bulk data changes), not
on every search request. If we encoded all N papers' abstracts on every
query, latency would scale with corpus size per request -- unacceptable.
Instead, only the (short) user query is encoded at request time; it's
compared against an already-computed matrix. This is why
`recommender.get_similar_papers()` doesn't even call the embedder --
it reuses the paper's own precomputed vector as the query.

### 2.3 Why L2-Normalized Vectors + Dot Product Instead of Explicit Cosine Similarity

`cosine_similarity(a, b) = (a . b) / (|a| |b|)`. If every stored vector is
normalized to unit length at encode time (`normalize_embeddings=True` in
`embedder.py`), then `|a| = |b| = 1` for all vectors, so the formula
reduces to a plain dot product. For one query scored against N papers,
this avoids N redundant norm computations per query -- a real, measurable
saving once N is in the tens of thousands.

### 2.4 Why NumPy Brute-Force Instead of FAISS (For Now)

At our target scale (20k-50k papers), `embeddings @ query_vector` -- a
single BLAS-backed matrix-vector multiply -- runs in low single-digit
milliseconds. FAISS exists to solve a problem (searching hundreds of
thousands to billions of vectors) we don't have yet, and it would add a
new dependency and a less transparent code path for a final-year project
to explain in a viva. `vector_store.py`'s public interface
(`search()`, `get_vector()`, `size`) is intentionally the only surface
`recommender.py` touches, so swapping the internals for
`faiss.IndexFlatIP` later is a one-file change -- documented directly in
`vector_store.py`'s closing comment block.

### 2.5 Why the ML Engine Has Zero Flask/SQLAlchemy Imports

`app/ml_engine/*.py` takes and returns plain Python types (ints, strings,
NumPy arrays) -- never a `Paper` model instance or a Flask request. This
means:
- It's unit-testable with synthetic vectors, no app context or database
  required (see `tests/test_ml_engine.py`, which validates ranking,
  exclusion, and centroid logic entirely without a real SBERT model).
- The whole folder could be lifted into a separate inference microservice
  without modification, if this project ever needed to scale that way.
- `app/services/recommendation_service.py` is the single, clearly-defined
  bridge between the two worlds -- one file to look at if you need to
  understand "how does a Paper row become a vector and back."

### 2.6 Why Explanations Are Generated Separately From Ranking

The recommendation **ranking** is 100% SBERT cosine similarity -- nothing
else influences the order results appear in. The **explanation** shown to
the user (`"Strongly related (78% similarity). Shares key topics: ..."`)
is a separate, additive layer built with lightweight TF-IDF term overlap
over just the two documents being explained. This keeps the actual
recommendation logic honest (it really is semantic, not keyword-based)
while still giving users an interpretable reason for each suggestion --
this directly satisfies the Phase 7 requirement to "explain why recommended
papers are similar" without compromising the semantic-search claim the
whole project is built around.

---

## 3. Handling the Three Recommendation Surfaces

| Surface | Function | Query vector source | Cold-start handling |
|---|---|---|---|
| "Similar Papers" on paper detail page | `get_similar_papers(paper_id)` | The paper's own precomputed embedding | N/A -- paper either has an embedding or doesn't |
| Semantic search box | `semantic_search(query_text)` | Live SBERT encoding of the typed query | N/A -- always available if embeddings exist |
| Dashboard "Recommended for you" | `personalized_recommendations(bookmarked_ids)` | Mean (centroid) of the user's bookmarked papers' embeddings, re-normalized to unit length | Explicitly detected and surfaced as `cold_start: True` if the user has zero bookmarks (or none with embeddings yet) -- the template shows an actionable message instead of an empty, unexplained section |

---

## 4. Graceful Degradation

Every public function in `recommendation_service.py` returns
`{"available": False, ...}` rather than raising, in two distinct
situations:
1. `embeddings.npy` doesn't exist yet (embeddings haven't been built) --
   detected by checking file existence before attempting to load.
2. The embedder itself fails to load or run for any reason (package not
   installed, no internet on first model download, out-of-memory, etc.)
   -- caught with a broad exception handler specifically in
   `semantic_search()`, since that's the only function that invokes the
   embedder at request time. **This was an actual bug caught during
   testing** (see Section 6) -- initially, a missing `sentence-transformers`
   install crashed the whole request with a 500 error instead of degrading
   to a friendly message, which is now fixed and covered by a regression
   test (`test_semantic_search_degrades_gracefully_when_embedder_fails`).

Templates check `similar_available` / `recommendations_available` /
`semantic_available` and show a clear, actionable message
(`Run python datasets/scripts/build_embeddings.py`) instead of a broken
page or a silent empty state.

---

## 5. Evaluation Methodology

`datasets/scripts/evaluate_recommendations.py` measures two things:

1. **Quality: Category-Match Precision@K.** For each paper, we take its
   top-K SBERT-recommended neighbors and measure what fraction share at
   least one arXiv category code with the source paper (arXiv categories
   are assigned by paper authors/moderators -- an independent proxy signal,
   not something we control). The identical metric is computed using a
   TF-IDF + cosine similarity baseline, giving a direct, numeric
   before/after comparison rather than an unverified claim that "semantic
   search is better."

   **Honest caveat, stated directly in the script's docstring and repeated
   here:** shared categories are a proxy for relevance, not ground-truth
   human judgment. Interdisciplinary papers can be genuinely related while
   carrying different category codes, and category-sharing papers aren't
   always closely related. Treat Precision@K here as directional evidence
   for your project report, and consider supplementing it with a small
   human relevance-rating exercise on a sample of query-result pairs for a
   stronger report.

2. **Performance: average query latency**, for both SBERT and the TF-IDF
   baseline, directly validating Phase 1's non-functional requirement of
   sub-1-2-second responses.

Run it after `build_embeddings.py`:
```bash
python datasets/scripts/evaluate_recommendations.py --top-k 5
```

With only a handful of sample papers (the current `datasets/raw/` sample
file), these numbers are illustrative of the pipeline running correctly,
not statistically meaningful -- re-run after loading the full arXiv subset
for numbers worth citing in your report.

---

## 6. What Was Actually Verified (Not Just Written)

Being transparent about how this was tested, since this project emphasizes
that over blind trust in generated code:

- **Sandbox network constraint**: this sandbox cannot reach
  `huggingface.co` (confirmed: `x-deny-reason: host_not_allowed`), so the
  real `all-MiniLM-L6-v2` model could not be downloaded and run here.
  `embedder.py` is written to the real Sentence-Transformers API and will
  work normally with your machine's regular internet access.
- **What WAS verified in this sandbox**, using synthetic and TF-IDF-derived
  stand-in vectors (never shipped as part of the actual codebase):
  - `tests/test_ml_engine.py` (14 tests): ranking order, self-exclusion,
    bookmarked-paper exclusion, cold-start detection, save/load round-trip,
    mismatched-array validation, TF-IDF shared-term extraction, score-band
    labeling -- all against hand-constructed vectors with known, verifiable
    similarity relationships.
  - `tests/test_recommendation_service.py` (6 tests): the full Flask/DB
    bridge, using a synthetic `VectorStore` written to a temp directory and
    a monkeypatched embedder for the semantic-search path.
  - **A live, running Flask server** was hit with real HTTP requests
    (register → bookmark → view paper detail → dashboard → semantic search)
    using a TF-IDF+SVD-derived stand-in vector store built specifically for
    this manual verification pass, confirming the mtime-based cache
    invalidation, template rendering, and full request/response cycle all
    work correctly end-to-end -- not just in isolated unit tests.
  - This manual pass is exactly what surfaced the graceful-degradation bug
    described in Section 4.
- **What requires your machine to verify**: the actual *semantic quality*
  of `all-MiniLM-L6-v2`'s recommendations (e.g. does a "graph neural
  network fraud detection" query surface genuinely related papers it
  wouldn't have found via keyword search) can only be judged once you run
  `build_embeddings.py` with real internet access and real embeddings.
