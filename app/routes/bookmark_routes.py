"""
bookmark_routes.py

Controller for bookmarking papers and viewing your bookmark list.
Uses POST for state changes (add/remove) rather than GET, per standard
web security practice -- GET requests should never mutate data.
"""

from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_required, current_user

from app.services.bookmark_service import add_bookmark, remove_bookmark, get_user_bookmarks, BookmarkError

bookmark_bp = Blueprint("bookmark", __name__)


@bookmark_bp.route("/bookmarks")
@login_required
def list_bookmarks():
    papers = get_user_bookmarks(current_user.id)
    return render_template("bookmarks.html", papers=papers)


@bookmark_bp.route("/bookmark/<int:paper_id>/add", methods=["POST"])
@login_required
def add(paper_id):
    try:
        add_bookmark(current_user.id, paper_id)
        flash("Paper bookmarked.", "success")
    except BookmarkError as exc:
        flash(str(exc), "danger")

    return redirect(request.referrer or url_for("paper.paper_detail", paper_id=paper_id))


@bookmark_bp.route("/bookmark/<int:paper_id>/remove", methods=["POST"])
@login_required
def remove(paper_id):
    removed = remove_bookmark(current_user.id, paper_id)
    if removed:
        flash("Bookmark removed.", "info")
    else:
        flash("That paper wasn't bookmarked.", "warning")

    return redirect(request.referrer or url_for("bookmark.list_bookmarks"))
