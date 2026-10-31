"""
paper_retrieval_service.py

High-level retrieval, ranking, filtering, and serialization of research papers
for ResearchBot. Bridges semantic search (SBERT embeddings + cosine similarity)
and keyword fallback search with robust duplicate elimination, relevance score
thresholding, and metadata formatting.
"""

import re
from typing import List, Dict, Any, Optional

from flask import current_app
from app.services.recommendation_service import semantic_search
from app.services.search_service import keyword_fallback_search


# Minimum cosine similarity threshold for SBERT search to filter out irrelevant noise
MIN_SEMANTIC_SCORE = 0.30

# Minimum relevance ratio for keyword fallback search
MIN_KEYWORD_SCORE = 0.15


def _normalize_title(title: str) -> str:
    """Normalize a title for deduplication comparison."""
    if not title:
        return ""
    return re.sub(r"[^a-z0-9]", "", title.lower())


def _format_paper(paper, score: float, is_keyword_fallback: bool = False) -> Dict[str, Any]:
    """Serialize a Paper database row into a structured dictionary."""
    clean_score = round(float(score), 3)
    score_pct = int(round(min(1.0, max(0.0, float(score))) * 100))

    abstract_text = (paper.abstract or "").strip()
    snippet = abstract_text
    if len(snippet) > 260:
        snippet = snippet[:260].rsplit(" ", 1)[0] + "..."

    doi_url = f"https://doi.org/{paper.doi}" if paper.doi else None
    direct_url = doi_url or paper.pdf_link or f"/paper/{paper.id}"

    return {
        "id": paper.id,
        "title": paper.title or "Untitled Paper",
        "authors": paper.authors or "Unknown Authors",
        "publication_year": paper.publication_year,
        "journal_conference": paper.journal_conference or "Academic Publication",
        "abstract": abstract_text,
        "abstract_snippet": snippet,
        "score": clean_score,
        "score_percent": score_pct,
        "doi": paper.doi,
        "doi_url": doi_url,
        "pdf_link": paper.pdf_link,
        "url": direct_url,
        "detail_url": f"/paper/{paper.id}",
        "is_fallback": is_keyword_fallback,
    }


def retrieve_papers_for_query(
    topic: str,
    top_k: int = 5,
    min_score: float = MIN_SEMANTIC_SCORE
) -> Dict[str, Any]:
    """
    Retrieves and ranks the most semantically relevant papers for a given topic.

    1. Executes SBERT semantic search over 49k precomputed embeddings.
    2. Falls back to multi-keyword search if semantic search is unavailable or returns no candidates.
    3. Filters out low-similarity/irrelevant results.
    4. Eliminates duplicate papers by ID and normalized title.
    5. Returns formatted paper dicts with a human-readable intro message.
    """
    clean_topic = (topic or "").strip()
    if not clean_topic:
        return {
            "papers": [],
            "message": "Please specify a research topic to search for papers.",
            "available": True,
            "count": 0,
            "fallback_used": False,
        }

    raw_candidates = []
    fallback_used = False

    # 1. Try SBERT semantic search
    try:
        search_res = semantic_search(clean_topic, top_k=max(top_k * 3, 15))
        if search_res.get("available") and search_res.get("results"):
            for item in search_res["results"]:
                raw_candidates.append({
                    "paper": item["paper"],
                    "score": item["score"],
                    "is_fallback": False,
                })
    except Exception as e:
        current_app.logger.warning(f"Semantic search attempt failed: {e}")

    # 2. If no semantic candidates or semantic search unavailable, try keyword fallback
    if not raw_candidates:
        try:
            kw_matches = keyword_fallback_search(clean_topic, top_k=max(top_k * 3, 15))
            if kw_matches:
                fallback_used = True
                for paper, matched_kws, score in kw_matches:
                    raw_candidates.append({
                        "paper": paper,
                        "score": score,
                        "is_fallback": True,
                    })
        except Exception as e:
            current_app.logger.error(f"Keyword fallback search failed: {e}")

    # 3. Deduplicate and filter by score threshold
    seen_ids = set()
    seen_titles = set()
    filtered_papers = []

    for item in raw_candidates:
        paper = item["paper"]
        score = item["score"]
        is_fb = item["is_fallback"]

        if not paper or paper.id in seen_ids:
            continue

        norm_title = _normalize_title(paper.title)
        if norm_title in seen_titles:
            continue

        # Score filter
        cutoff = MIN_KEYWORD_SCORE if is_fb else min_score
        if score < cutoff:
            continue

        seen_ids.add(paper.id)
        seen_titles.add(norm_title)
        filtered_papers.append(_format_paper(paper, score, is_keyword_fallback=is_fb))

        if len(filtered_papers) >= top_k:
            break

    count = len(filtered_papers)

    # 4. Formulate contextual intro message
    if count == 0:
        message = (
            f"No relevant research papers were found matching **'{clean_topic}'** in the current database. "
            f"Try searching with broader or alternative scientific keywords."
        )
    elif count < top_k:
        message = (
            f"Found {count} relevant research paper{'s' if count > 1 else ''} matching **'{clean_topic}'** "
            f"(filtered for high relevance):"
        )
    else:
        message = (
            f"Here are {count} relevant research papers on **'{clean_topic}'** from our database, "
            f"ranked by semantic relevance:"
        )

    return {
        "papers": filtered_papers,
        "message": message,
        "available": True,
        "count": count,
        "fallback_used": fallback_used,
        "topic": clean_topic,
    }

