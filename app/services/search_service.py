"""
search_service.py

Traditional filter/keyword search (title, author, keyword, domain).

IMPORTANT: this is the LEXICAL baseline search required by Phase 6. It is
intentionally NOT the semantic recommendation engine -- that's Phase 8
(Sentence-BERT + cosine similarity). Keeping this service separate from the
future recommendation_service.py lets us honestly compare "before" (this)
and "after" (Phase 8) in the project report, which is exactly the kind of
evaluation a final-year AI project should be able to show.

Schema note: our `papers` table doesn't have a separate `categories` column
(see Phase 4 design) -- arXiv's category codes (e.g. "cs.LG cs.AI") are
stored in the `keywords` column. So "search by domain" and "search by
keyword" both query that same column; domain search is just a keyword
search scoped to a recognizable category code (e.g. "cs.LG").
"""

import re

from app.models.paper import Paper


def search_papers(
    title: str = None,
    author: str = None,
    keyword: str = None,
    domain: str = None,
    general_query: str = None,
    page: int = 1,
    per_page: int = 10,
):
    """
    Builds and executes a filtered, paginated query over the papers table.

    Args:
        title: substring match against Paper.title
        author: substring match against Paper.authors
        keyword: substring match against Paper.keywords
        domain: substring match against Paper.keywords (arXiv category codes)
        general_query: if provided (and no specific field given), matches
            against title OR authors OR keywords OR abstract -- this powers
            a single "search everything" box on the search page.
        page, per_page: standard Flask-SQLAlchemy pagination

    Returns:
        A Flask-SQLAlchemy Pagination object (has .items, .pages, .total, etc.)
    """
    query = Paper.query

    has_specific_filter = any([title, author, keyword, domain])

    if title:
        query = query.filter(Paper.title.ilike(f"%{title}%"))
    if author:
        query = query.filter(Paper.authors.ilike(f"%{author}%"))
    if keyword:
        query = query.filter(Paper.keywords.ilike(f"%{keyword}%"))
    if domain:
        query = query.filter(Paper.keywords.ilike(f"%{domain}%"))

    if general_query and not has_specific_filter:
        like_pattern = f"%{general_query}%"
        query = query.filter(
            db_or(
                Paper.title.ilike(like_pattern),
                Paper.authors.ilike(like_pattern),
                Paper.keywords.ilike(like_pattern),
                Paper.abstract.ilike(like_pattern),
            )
        )

    query = query.order_by(Paper.citation_count.desc(), Paper.id.desc())

    return query.paginate(page=page, per_page=per_page, error_out=False)


def db_or(*conditions):
    """Small wrapper so this module doesn't need a top-level SQLAlchemy import
    just for a single `or_` call -- keeps the import list minimal and explicit."""
    from sqlalchemy import or_
    return or_(*conditions)


def _matches(field_value: str, filter_value: str) -> bool:
    """Case-insensitive substring check, mirroring the DB's .ilike() logic
    above, with null-safety for missing paper fields."""
    if not field_value or not filter_value:
        return False
    return filter_value.lower() in field_value.lower()


def explain_lexical_match(
    paper,
    title: str = None,
    author: str = None,
    keyword: str = None,
    domain: str = None,
    general_query: str = None,
) -> dict:
    """
    "Why This Paper?" explanation for LEXICAL/filter search results.

    IMPORTANT: this does not affect search_papers() or the SQL query above
    in any way -- it runs AFTER search_papers() has already returned a page
    of matching Paper rows, and only re-checks (in Python, read-only) which
    of the filters the user actually typed are present in that paper's
    fields, purely to explain a result that's already been decided.
    """
    has_specific_filter = any([title, author, keyword, domain])
    reasons = []

    if title and _matches(paper.title, title):
        reasons.append(f'Its title contains "{title}"')
    if author and _matches(paper.authors, author):
        reasons.append(f'The author field contains "{author}"')
    if keyword and _matches(paper.keywords, keyword):
        reasons.append(f'Its keywords contain "{keyword}"')
    if domain and _matches(paper.keywords, domain):
        reasons.append(f'It belongs to the domain "{domain}"')

    if general_query and not has_specific_filter:
        matched_fields = []
        if _matches(paper.title, general_query):
            matched_fields.append("title")
        if _matches(paper.authors, general_query):
            matched_fields.append("authors")
        if _matches(paper.keywords, general_query):
            matched_fields.append("keywords")
        if _matches(paper.abstract, general_query):
            matched_fields.append("abstract")
        if matched_fields:
            fields_str = ", ".join(matched_fields)
            reasons.append(
                f'Your search "{general_query}" was found in its {fields_str}')

    if not reasons:
        reasons = ["This paper matched your current search filters"]

    if len(reasons) == 1:
        first_reason = reasons[0]
        summary = f"This paper is shown because {first_reason[0].lower() + first_reason[1:]}."
    else:
        summary = "This paper matches your search because:"

    return {"reasons": reasons, "summary": summary}


# -----------------------------------------------------------------------------
# KEYWORD FALLBACK FOR SEMANTIC SEARCH / RESEARCHBOT
#
# Used ONLY by recommendation_service.semantic_search() as an emergency
# substitute for SBERT-based semantic search, for the specific case where
# the embedder itself fails to load (e.g. a native dependency -- torch,
# regex, etc. -- blocked by the host OS's security policy). It is NOT used
# by the lexical search page above (search_papers() is untouched and still
# that page's implementation), and it is NOT used when embeddings simply
# haven't been built yet (that case still correctly reports
# "available": False, same as before this change).
#
# Unlike search_papers()'s single-phrase substring match, this extracts
# individual meaningful keywords from a free-text question and ranks
# papers by how many distinct keywords they match -- closer to what a
# user asking a natural-language question to a chatbot actually needs.
# -----------------------------------------------------------------------------

_STOP_WORDS = {
    "a", "an", "the", "and", "or", "for", "with", "about", "on", "in", "of",
    "to", "is", "are", "what", "which", "give", "me", "please", "show",
    "find", "tell", "list", "explain", "describe", "summarize", "paper",
    "papers", "research", "related", "regarding", "info", "information",
}


def _extract_keywords(question: str) -> list:
    """
    Lowercases and tokenizes a free-text question, discards a small
    stopword list and single-character tokens, and de-duplicates while
    preserving first-seen order. Works for any question dynamically --
    nothing here is specific to any one topic.
    """
    tokens = re.findall(r"[a-zA-Z0-9\-]+", question.lower())
    seen = set()
    keywords = []
    for token in tokens:
        if token in _STOP_WORDS or len(token) <= 1:
            continue
        if token not in seen:
            seen.add(token)
            keywords.append(token)
    return keywords


def keyword_fallback_search(question: str, top_k: int = 5):
    """
    Extracts meaningful keywords from a free-text question and searches
    title/abstract/keywords for each one, ranking papers by how many
    distinct keywords they match (ties broken by citation_count, then id,
    for stable, reproducible ordering).

    One query per extracted keyword (not per paper), so this stays fast
    even across a large table -- each query is a simple ILIKE scan, not an
    N+1 loop over paper rows. Each per-keyword fetch is capped at 200 rows
    as a safety bound against a single very common keyword (e.g. "model")
    pulling in a huge number of candidates before ranking narrows it down.

    Returns:
        A list of (Paper, matched_keywords, relevance_score) tuples for the
        top_k best-matching papers, relevance_score in (0, 1] = (number of
        distinct keywords matched) / (number of keywords extracted).
        Returns [] if no usable keywords were extracted, or if nothing in
        the database matched any of them.
    """
    keywords = _extract_keywords(question)
    if not keywords:
        return []

    paper_matches = {}  # paper.id -> {"paper": Paper, "matched": set()}

    for kw in keywords:
        like_pattern = f"%{kw}%"
        matches = (
            Paper.query.filter(
                db_or(
                    Paper.title.ilike(like_pattern),
                    Paper.abstract.ilike(like_pattern),
                    Paper.keywords.ilike(like_pattern),
                )
            )
            .limit(200)
            .all()
        )

        for paper in matches:
            entry = paper_matches.setdefault(
                paper.id, {"paper": paper, "matched": set()}
            )
            entry["matched"].add(kw)

    if not paper_matches:
        return []

    ranked = sorted(
        paper_matches.values(),
        key=lambda e: (
            -len(e["matched"]),
            -(e["paper"].citation_count or 0),
            e["paper"].id,
        ),
    )[:top_k]

    max_possible = len(keywords)
    results = []
    for entry in ranked:
        matched_sorted = sorted(entry["matched"])
        score = len(entry["matched"]) / max_possible
        results.append((entry["paper"], matched_sorted, score))

    return results
