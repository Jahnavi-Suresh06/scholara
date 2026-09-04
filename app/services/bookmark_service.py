"""
bookmark_service.py

Business logic for saving/removing/listing a user's bookmarked papers.
This is also the primary signal source for personalized recommendations
in Phase 8 (a user's "interest vector" = mean embedding of bookmarked papers).
"""

from app import db
from app.models.bookmark import Bookmark
from app.models.paper import Paper


class BookmarkError(Exception):
    """Raised for invalid bookmark operations (e.g. bookmarking a nonexistent paper)."""


def add_bookmark(user_id: int, paper_id: int) -> Bookmark:
    paper = db.session.get(Paper, paper_id)
    if paper is None:
        raise BookmarkError("That paper does not exist.")

    existing = Bookmark.query.filter_by(user_id=user_id, paper_id=paper_id).first()
    if existing:
        return existing  # idempotent: bookmarking twice is a no-op, not an error

    bookmark = Bookmark(user_id=user_id, paper_id=paper_id)
    db.session.add(bookmark)
    db.session.commit()
    return bookmark


def remove_bookmark(user_id: int, paper_id: int) -> bool:
    bookmark = Bookmark.query.filter_by(user_id=user_id, paper_id=paper_id).first()
    if bookmark is None:
        return False
    db.session.delete(bookmark)
    db.session.commit()
    return True


def get_user_bookmarks(user_id: int):
    """Returns the list of Paper objects a user has bookmarked, most recent first."""
    return (
        Paper.query.join(Bookmark, Bookmark.paper_id == Paper.id)
        .filter(Bookmark.user_id == user_id)
        .order_by(Bookmark.created_at.desc())
        .all()
    )


def is_bookmarked(user_id: int, paper_id: int) -> bool:
    return (
        Bookmark.query.filter_by(user_id=user_id, paper_id=paper_id).first()
        is not None
    )
