"""
test_models.py

Verifies the core models (User, Paper, Bookmark, SearchHistory) can be
created, related, and queried correctly against an in-memory SQLite
database. This is our safety net before building routes/services on
top of these models in later phases.
"""

import pytest

from app import create_app, db
from app.models import User, Paper, Bookmark, SearchHistory


@pytest.fixture
def app():
    app = create_app("testing")
    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()


def test_user_password_hashing(app):
    with app.app_context():
        user = User(username="jayshree", email="jayshree@example.com")
        user.set_password("supersecret")
        db.session.add(user)
        db.session.commit()

        fetched = User.query.filter_by(username="jayshree").first()
        assert fetched is not None
        assert fetched.password_hash != "supersecret"  # never store plaintext
        assert fetched.check_password("supersecret") is True
        assert fetched.check_password("wrongpassword") is False


def test_paper_to_dict(app):
    with app.app_context():
        paper = Paper(
            title="Semantic Search over Scientific Literature",
            authors="P. Iyer, S. Rao",
            abstract="We evaluate sentence embedding models for retrieval...",
            embedding_index=0,
        )
        db.session.add(paper)
        db.session.commit()

        data = paper.to_dict()
        assert data["title"] == "Semantic Search over Scientific Literature"
        assert "abstract" in data


def test_bookmark_unique_constraint(app):
    with app.app_context():
        user = User(username="testuser", email="test@example.com")
        user.set_password("pass1234")
        paper = Paper(
            title="Test Paper",
            authors="A. Author",
            abstract="An abstract long enough to be realistic for testing purposes.",
        )
        db.session.add_all([user, paper])
        db.session.commit()

        bookmark = Bookmark(user_id=user.id, paper_id=paper.id)
        db.session.add(bookmark)
        db.session.commit()

        assert len(user.bookmarks) == 1
        assert user.bookmarks[0].paper_id == paper.id


def test_search_history_logging(app):
    with app.app_context():
        user = User(username="searcher", email="searcher@example.com")
        user.set_password("pass1234")
        db.session.add(user)
        db.session.commit()

        history = SearchHistory(user_id=user.id, query_text="graph neural networks fraud detection")
        db.session.add(history)
        db.session.commit()

        assert len(user.search_history) == 1
        assert "fraud detection" in user.search_history[0].query_text
