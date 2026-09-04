"""
paper_routes.py

Controller for viewing a single paper's detail page, including the
"Similar Papers" section powered by the SBERT recommendation engine
(app/services/recommendation_service.py).
"""

from flask import Blueprint, render_template, abort
from flask_login import current_user

from app import db
from app.models.paper import Paper
from app.services.bookmark_service import is_bookmarked
from app.services.recommendation_service import get_similar_papers

paper_bp = Blueprint("paper", __name__)


@paper_bp.route("/paper/<int:paper_id>")
def paper_detail(paper_id):
    paper = db.session.get(Paper, paper_id)
    if paper is None:
        abort(404)

    bookmarked = False
    if current_user.is_authenticated:
        bookmarked = is_bookmarked(current_user.id, paper_id)

    similar = get_similar_papers(paper_id, top_k=5)

    return render_template(
        "paper_detail.html",
        paper=paper,
        bookmarked=bookmarked,
        similar_available=similar["available"],
        similar_papers=similar["results"],
    )
