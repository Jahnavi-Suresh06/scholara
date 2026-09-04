"""
SearchHistory model.

Logs every query a user runs. Used for:
1. Displaying "recent searches" on the dashboard (Phase 7).
2. (Optional stretch feature) blending past query embeddings into
   personalization (Phase 8).
"""

from datetime import datetime, timezone

from app import db


class SearchHistory(db.Model):
    __tablename__ = "search_history"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    query_text = db.Column(db.Text, nullable=False)
    searched_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    def __repr__(self):
        return f"<SearchHistory user={self.user_id} query='{self.query_text[:30]}'>"
