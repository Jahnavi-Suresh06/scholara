"""
test_app.py

Verifies the app factory produces a working Flask app, and that the
landing page + health-check routes respond correctly. Uses the
"testing" config (in-memory SQLite) so these tests never touch your
real development database.
"""

import pytest

from app import create_app, db


@pytest.fixture
def app():
    app = create_app("testing")
    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app):
    return app.test_client()


def test_app_creation(app):
    assert app is not None
    assert app.config["TESTING"] is True


def test_index_route_returns_200(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Scholara" in response.data


def test_health_route_reports_db_connected(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "ok"
    assert data["database"] == "connected"
