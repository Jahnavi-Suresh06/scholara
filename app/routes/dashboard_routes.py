"""
dashboard_routes.py

Controller for the personalized dashboard: bookmark/search-history
summaries, plus "Recommended for you" powered by the SBERT recommendation
engine's bookmark-centroid personalization.
"""

from flask import Blueprint, render_template
from flask_login import login_required, current_user
from zoneinfo import ZoneInfo

from app.services.bookmark_service import get_user_bookmarks
from app.services.history_service import get_recent_history
from app.services.recommendation_service import (
    get_personalized_recommendations
)


dashboard_bp = Blueprint("dashboard", __name__)


@dashboard_bp.route("/dashboard")
@login_required
def dashboard():

    # Get user bookmarks
    bookmarks = get_user_bookmarks(current_user.id)

    # Get recent searches
    recent_searches = get_recent_history(
        current_user.id,
        limit=5
    )

    # Get personalized recommendations
    recommendations = get_personalized_recommendations(
        current_user.id,
        top_k=5
    )

    # ==========================================
    # CONVERT SEARCH TIME FROM UTC TO IST
    # ==========================================

    ist = ZoneInfo("Asia/Kolkata")
    utc = ZoneInfo("UTC")

    for entry in recent_searches:

        if entry.searched_at:

            # SQLite may return the datetime without timezone information.
            # Treat stored timestamps as UTC before converting to IST.

            if entry.searched_at.tzinfo is None:

                entry.searched_at = entry.searched_at.replace(
                    tzinfo=utc
                )

            # Convert UTC → Indian Standard Time

            entry.searched_at = entry.searched_at.astimezone(
                ist
            )

    # Render dashboard

    return render_template(

        "dashboard.html",

        bookmarks=bookmarks,

        recent_searches=recent_searches,

        recommendations_available=recommendations["available"],

        cold_start=recommendations["cold_start"],

        recommended_papers=recommendations["results"],

    )
