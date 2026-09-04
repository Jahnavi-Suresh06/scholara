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

    Args:
        paper: a Paper model instance from an already-fetched results page
        title, author, keyword, domain, general_query: the same filter
            values passed to search_papers() for this request -- pass
            whichever ones the user actually supplied; the rest default to
            None and are simply skipped

    Returns:
        {
            "reasons": ['Its title contains "machine"', 'The author field
                         contains "John"'],
            "summary": "This paper matches your search because:"
        }
        For a single-filter match, "summary" is instead a complete sentence,
        e.g. "This paper is shown because its title contains \"machine\"."
        and "reasons" holds that one reason for the template to still use
        consistently if needed.
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

    # The "search everything" box (general_query) only applies when no
    # specific filter was given -- same rule search_papers() itself uses.
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
        # Fallback per spec: never leave the section blank/erroring. In
        # normal operation search_papers() only ever returns rows that
        # matched a filter, so this only guards edge cases (e.g. a filter
        # value not passed through to this function).
        reasons = ["This paper matched your current search filters"]

    if len(reasons) == 1:
        first_reason = reasons[0]
        summary = f"This paper is shown because {first_reason[0].lower() + first_reason[1:]}."
    else:
        summary = "This paper matches your search because:"

    return {"reasons": reasons, "summary": summary}
