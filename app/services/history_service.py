"""
history_service.py

Logs and retrieves a user's search query history. Used by the dashboard
("recent searches") and, as an optional Phase 8 enhancement, could feed
into personalization alongside bookmarks.
"""

from app import db
from app.models.search_history import SearchHistory


def log_search(user_id: int, query_text: str) -> SearchHistory:
    query_text = (query_text or "").strip()
    if not query_text:
        return None  # don't log empty searches

    entry = SearchHistory(user_id=user_id, query_text=query_text)
    db.session.add(entry)
    db.session.commit()
    return entry


def get_recent_history(user_id: int, limit: int = 10):
    return (
        SearchHistory.query.filter_by(user_id=user_id)
        .order_by(SearchHistory.searched_at.desc())
        .limit(limit)
        .all()
    )
