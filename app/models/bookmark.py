"""
Bookmark model.

Join table linking a user to a paper they've saved. Also doubles as the
primary signal for personalized recommendations (Phase 8), since a user's
"interest vector" is computed as the mean embedding of their bookmarked
papers.
"""

from datetime import datetime, timezone

from app import db


class Bookmark(db.Model):
    __tablename__ = "bookmarks"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    paper_id = db.Column(db.Integer, db.ForeignKey("papers.id"), nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        db.UniqueConstraint("user_id", "paper_id", name="uq_user_paper_bookmark"),
    )

    def __repr__(self):
        return f"<Bookmark user={self.user_id} paper={self.paper_id}>"
