"""
test_auth.py

Covers registration, login, logout, and the validation rules in
app/services/auth_service.py.
"""

import pytest

from app import create_app, db
from app.services.auth_service import register_user, authenticate_user, AuthError


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


def test_register_user_success(app):
    with app.app_context():
        user = register_user("jayshree", "jayshree@example.com", "password123")
        assert user.id is not None
        assert user.check_password("password123")


def test_register_duplicate_username_fails(app):
    with app.app_context():
        register_user("jayshree", "a@example.com", "password123")
        with pytest.raises(AuthError):
            register_user("jayshree", "b@example.com", "password456")


def test_register_short_password_fails(app):
    with app.app_context():
        with pytest.raises(AuthError):
            register_user("shortpass", "short@example.com", "1234567")  # 7 chars


def test_authenticate_wrong_password_fails(app):
    with app.app_context():
        register_user("loginuser", "login@example.com", "correctpass")
        with pytest.raises(AuthError):
            authenticate_user("loginuser", "wrongpass")


def test_authenticate_by_email_succeeds(app):
    with app.app_context():
        register_user("emailuser", "emailuser@example.com", "password123")
        user = authenticate_user("emailuser@example.com", "password123")
        assert user.username == "emailuser"


def test_register_route_creates_user_and_logs_in(client):
    response = client.post(
        "/auth/register",
        data={
            "username": "webuser",
            "email": "webuser@example.com",
            "password": "password123",
        },
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"webuser" in response.data  # nav should show the username after login


def test_login_route_then_logout(client):
    client.post(
        "/auth/register",
        data={"username": "flowuser", "email": "flow@example.com", "password": "password123"},
    )
    client.get("/auth/logout", follow_redirects=True)

    response = client.post(
        "/auth/login",
        data={"identifier": "flowuser", "password": "password123"},
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"Dashboard" in response.data or b"Welcome back" in response.data


def test_dashboard_requires_login(client):
    response = client.get("/dashboard", follow_redirects=True)
    assert response.status_code == 200
    # Flask-Login redirects unauthenticated users to the login page
    assert b"Login" in response.data
