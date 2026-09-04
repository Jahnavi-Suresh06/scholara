"""
test_bookmarks.py

Covers app/services/bookmark_service.py: add/remove/list/idempotency.
"""

import pytest

from app import create_app, db
from app.models.paper import Paper
from app.services.auth_service import register_user
from app.services.bookmark_service import (
    add_bookmark,
    remove_bookmark,
    get_user_bookmarks,
    is_bookmarked,
    BookmarkError,
)


@pytest.fixture
def app():
    app = create_app("testing")
    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()


def test_add_and_check_bookmark(app):
    with app.app_context():
        user = register_user("bookmarker", "bm@example.com", "password123")
        paper = Paper(title="Test Paper", authors="A. Author", abstract="A sufficiently long test abstract.")
        db.session.add(paper)
        db.session.commit()

        add_bookmark(user.id, paper.id)
        assert is_bookmarked(user.id, paper.id) is True


def test_bookmark_is_idempotent(app):
    with app.app_context():
        user = register_user("idempotent", "idem@example.com", "password123")
        paper = Paper(title="Idempotent Paper", authors="A. Author", abstract="A sufficiently long test abstract.")
        db.session.add(paper)
        db.session.commit()

        add_bookmark(user.id, paper.id)
        add_bookmark(user.id, paper.id)  # should not raise or create a duplicate row
        assert len(get_user_bookmarks(user.id)) == 1


def test_remove_bookmark(app):
    with app.app_context():
        user = register_user("remover", "remove@example.com", "password123")
        paper = Paper(title="Removable Paper", authors="A. Author", abstract="A sufficiently long test abstract.")
        db.session.add(paper)
        db.session.commit()

        add_bookmark(user.id, paper.id)
        removed = remove_bookmark(user.id, paper.id)
        assert removed is True
        assert is_bookmarked(user.id, paper.id) is False


def test_bookmark_nonexistent_paper_raises(app):
    with app.app_context():
        user = register_user("ghostbookmarker", "ghost@example.com", "password123")
        with pytest.raises(BookmarkError):
            add_bookmark(user.id, paper_id=9999)
