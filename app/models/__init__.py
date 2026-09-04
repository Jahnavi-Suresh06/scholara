"""
app/models/__init__.py

Re-exports all model classes so other modules can do:
    from app.models import User, Paper, Bookmark, SearchHistory
instead of reaching into each individual file.
"""

from app.models.user import User
from app.models.paper import Paper
from app.models.bookmark import Bookmark
from app.models.search_history import SearchHistory

__all__ = ["User", "Paper", "Bookmark", "SearchHistory"]
