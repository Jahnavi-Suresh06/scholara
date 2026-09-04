"""
search_routes.py

Handles:
- Lexical Search
- Semantic Search
- Recent Search History
"""

from flask import Blueprint, render_template, request
from flask_login import current_user, login_required

from app.services.search_service import (
    search_papers,
    explain_lexical_match
)

from app.services.history_service import (
    log_search,
    get_recent_history
)

from app.services.recommendation_service import semantic_search


search_bp = Blueprint("search", __name__)


# ==========================================
# SEARCH PAPERS
# ==========================================

@search_bp.route("/search")
def search():

    mode = request.args.get("mode", "lexical")

    general_query = request.args.get("q", "").strip()

    title = request.args.get("title", "").strip()

    author = request.args.get("author", "").strip()

    keyword = request.args.get("keyword", "").strip()

    domain = request.args.get("domain", "").strip()

    page = request.args.get("page", 1, type=int)

    # Default values
    results = None
    lexical_explanations = None
    semantic_results = None
    semantic_available = True

    # ==========================================
    # SEMANTIC SEARCH
    # ==========================================

    if mode == "semantic":

        if general_query:

            semantic_response = semantic_search(
                general_query,
                top_k=10
            )

            semantic_available = semantic_response["available"]

            semantic_results = semantic_response["results"]

            # Save search history
            if current_user.is_authenticated:

                log_search(
                    current_user.id,
                    general_query
                )

    # ==========================================
    # LEXICAL SEARCH
    # ==========================================

    else:

        has_any_input = any([
            general_query,
            title,
            author,
            keyword,
            domain
        ])

        if has_any_input:

            results = search_papers(

                title=title or None,

                author=author or None,

                keyword=keyword or None,

                domain=domain or None,

                general_query=general_query or None,

                page=page,

                per_page=10,
            )

            # ======================================
            # WHY THIS PAPER EXPLANATION
            # ======================================

            lexical_explanations = {

                paper.id: explain_lexical_match(

                    paper,

                    title=title or None,

                    author=author or None,

                    keyword=keyword or None,

                    domain=domain or None,

                    general_query=general_query or None,

                )

                for paper in results.items

            }

            # ======================================
            # SAVE SEARCH HISTORY
            # ======================================

            if current_user.is_authenticated:

                query_summary = (

                    general_query

                    or

                    ", ".join(

                        f"{label}: {value}"

                        for label, value in [

                            ("Title", title),
                            ("Author", author),
                            ("Keyword", keyword),
                            ("Domain", domain),

                        ]

                        if value

                    )

                )

                log_search(
                    current_user.id,
                    query_summary
                )

    # ==========================================
    # RENDER SEARCH PAGE
    # ==========================================

    return render_template(

        "search.html",

        mode=mode,

        results=results,

        lexical_explanations=lexical_explanations,

        semantic_results=semantic_results,

        semantic_available=semantic_available,

        general_query=general_query,

        title=title,

        author=author,

        keyword=keyword,

        domain=domain,

    )


# ==========================================
# RECENT SEARCH HISTORY
# ==========================================

@search_bp.route("/recent-searches")
@login_required
def recent_searches():

    searches = get_recent_history(

        user_id=current_user.id,

        limit=20

    )

    return render_template(

        "recent_searches.html",

        searches=searches

    )
