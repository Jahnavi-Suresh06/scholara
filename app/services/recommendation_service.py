"""
recommendation_service.py

The ONLY place where app/ml_engine (pure paper_id/vector logic) and the
Flask/SQLAlchemy app (Paper model rows, current_app config) meet. Routes
call into this file; this file calls into app/ml_engine/*.py and
app/models/*.py.

Handles the "embeddings haven't been built yet" case gracefully everywhere
-- every public function returns {"available": False, ...} rather than
raising, so templates can show a friendly message instead of a 500 error
if someone runs the app before running build_embeddings.py.
"""

from pathlib import Path

from flask import current_app

from app import db
from app.models.paper import Paper
from app.ml_engine.vector_store import VectorStore
from app.ml_engine.embedder import get_embedder
from app.ml_engine.similarity import explain_similarity, explain_search_recommendation
from app.ml_engine import recommender as ml_recommender

# Process-wide cache: {"store": VectorStore, "mtime": float}. Keyed on the
# embeddings file's modification time so that re-running build_embeddings.py
# (e.g. after adding papers via the admin panel) is picked up on the next
# request without needing to restart the whole Flask app.
_vector_store_cache = {"store": None, "mtime": None}


def _get_vector_store():
    embeddings_path = Path(current_app.config["EMBEDDINGS_DIR"]) / "embeddings.npy"
    ids_path = Path(current_app.config["EMBEDDINGS_DIR"]) / "paper_ids.npy"

    if not embeddings_path.exists() or not ids_path.exists():
        return None

    mtime = embeddings_path.stat().st_mtime
    if _vector_store_cache["store"] is None or _vector_store_cache["mtime"] != mtime:
        _vector_store_cache["store"] = VectorStore.load(embeddings_path, ids_path)
        _vector_store_cache["mtime"] = mtime

    return _vector_store_cache["store"]


def _to_result_list(raw_results, query_text_for_explanation=None):
    """
    Converts [(paper_id, score), ...] into
    [{"paper": Paper, "score": ..., "explanation": {...}}, ...],
    fetching each Paper row and generating its explanation. Skips any
    paper_id that no longer exists in the DB (e.g. deleted after the
    embeddings were built but before they were rebuilt).
    """
    results = []
    for paper_id, score in raw_results:
        paper = db.session.get(Paper, paper_id)
        if paper is None:
            continue
        explanation = explain_similarity(
            query_text_for_explanation or "", paper.abstract, score
        )
        results.append({"paper": paper, "score": score, "explanation": explanation})
    return results


def _to_search_result_list(raw_results, query_text: str):
    """
    "Why This Paper?" variant of _to_result_list(), used ONLY by
    semantic_search() below. Converts [(paper_id, score), ...] into
    [{"paper": Paper, "score": ..., "explanation": {...}}, ...], using
    explain_search_recommendation() (query vs. paper, title+abstract+
    keywords, search-oriented wording) instead of explain_similarity()
    (paper vs. paper, abstract only).

    get_similar_papers() and get_personalized_recommendations() below still
    call the original _to_result_list() and are completely unaffected by
    this function's existence.
    """
    results = []
    for paper_id, score in raw_results:
        paper = db.session.get(Paper, paper_id)
        if paper is None:
            continue
        explanation = explain_search_recommendation(query_text, paper, score)
        results.append({"paper": paper, "score": score, "explanation": explanation})
    return results


def _keyword_fallback_results(query_text: str, top_k: int):
    """
    Converts search_service.keyword_fallback_search()'s output into the
    exact same shape semantic_search() normally returns --
    [{"paper": Paper, "score": ..., "explanation": {...}}, ...] -- so that
    nothing downstream (chatbot.py, search.html's semantic-mode template)
    needs to know or care whether a result came from real SBERT semantic
    search or this keyword fallback.

    Imported locally (not at module top) to match this file's existing
    pattern for service-to-service imports (see get_personalized_
    recommendations()'s local import of bookmark_service below), and to
    keep this fallback's dependency out of the module's always-loaded
    import list.
    """
    from app.services.search_service import keyword_fallback_search

    matches = keyword_fallback_search(query_text, top_k=top_k)

    results = []
    for paper, matched_keywords, score in matches:
        if matched_keywords:
            topics_str = ", ".join(matched_keywords[:5])
            summary = (
                f"This paper was found via keyword search (semantic search is "
                f"temporarily unavailable) because it mentions: {topics_str}."
            )
        else:
            summary = (
                "This paper was found via keyword search because semantic "
                "search is temporarily unavailable."
            )

        explanation = {
            "score": round(float(score), 4),
            "score_percent": round(min(1.0, score) * 100),
            "label": "Keyword match",
            "related_topics": matched_keywords,
            "summary": summary,
        }
        results.append({"paper": paper, "score": score, "explanation": explanation})

    return results


def get_similar_papers(paper_id: int, top_k: int = 5) -> dict:
    """For the paper detail page's 'Similar Papers' section."""
    store = _get_vector_store()
    if store is None:
        return {"available": False, "results": []}

    query_paper = db.session.get(Paper, paper_id)
    query_text = query_paper.abstract if query_paper else ""

    raw_results = ml_recommender.get_similar_papers(store, paper_id, top_k=top_k)
    return {"available": True, "results": _to_result_list(raw_results, query_text)}


def semantic_search(query_text: str, top_k: int = 10, fallback_on_error: bool = False) -> dict:
    """
    For the search page's semantic search mode, and for ResearchBot's
    paper retrieval.

    If embeddings haven't been built at all, this still returns
    {"available": False, "results": []}, exactly as before -- unchanged.

    If embeddings exist but the SBERT embedder itself fails to load or run
    (e.g. a native dependency -- torch, regex, etc. -- blocked by the host
    OS's Application Control / security policy, missing on first run with
    no internet, out of memory, or any other embedder failure), this now
    falls back to a dynamic keyword search over the same paper database
    (see search_service.keyword_fallback_search) instead of returning
    zero results. The vector/SBERT search path itself is completely
    unchanged and is tried first, every time -- the fallback only runs
    when that path actually raises.
    """
    store = _get_vector_store()
    if store is None:
        return {"available": False, "results": [], "fallback_used": False}

    try:
        embedder = get_embedder()
        raw_results = ml_recommender.semantic_search(embedder, store, query_text, top_k=top_k)
    except Exception:
        # Covers: sentence-transformers/torch/regex native DLL blocked by
        # OS policy, not installed, no internet on first model download,
        # out-of-memory, or any other embedder/search failure. Previously
        # this degraded all the way to "no results" -- now it degrades one
        # step further, to keyword search, before giving up entirely.
        current_app.logger.exception(
            "Semantic search failed to load/run the embedder"
        )
        if fallback_on_error:
            fallback_results = _keyword_fallback_results(query_text, top_k)
            return {
                "available": True,
                "results": fallback_results,
                "fallback_used": True,
            }
        return {"available": False, "results": [], "fallback_used": False}

    return {
        "available": True,
        "results": _to_search_result_list(raw_results, query_text),
        "fallback_used": False,
    }


def get_personalized_recommendations(user_id: int, top_k: int = 5) -> dict:
    """
    For the dashboard's 'Recommended for you' section. Cold-start case
    (user has no bookmarks yet, or none of their bookmarked papers have
    embeddings) is signaled via "cold_start": True so the template can show
    an appropriate message ("bookmark a few papers to get personalized
    recommendations") instead of an empty list with no explanation.
    """
    store = _get_vector_store()
    if store is None:
        return {"available": False, "cold_start": False, "results": []}

    from app.services.bookmark_service import get_user_bookmarks

    bookmarked_papers = get_user_bookmarks(user_id)
    bookmarked_ids = [p.id for p in bookmarked_papers]

    if not bookmarked_ids:
        return {"available": True, "cold_start": True, "results": []}

    raw_results = ml_recommender.personalized_recommendations(
        store, bookmarked_ids, top_k=top_k
    )

    if not raw_results:
        # User has bookmarks, but none of them have embeddings yet (e.g.
        # embeddings were built before these papers were bookmarked)
        return {"available": True, "cold_start": True, "results": []}

    # Use the titles of bookmarked papers as a rough "query text" for
    # explanation purposes -- there's no single source paper here since
    # this is a centroid across possibly several bookmarks.
    representative_text = " ".join(p.abstract for p in bookmarked_papers[:3])
    return {
        "available": True,
        "cold_start": False,
        "results": _to_result_list(raw_results, representative_text),
    }