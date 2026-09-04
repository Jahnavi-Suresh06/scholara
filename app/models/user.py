"""
User model.

Handles user account data and authentication-related fields.
Password is NEVER stored in plaintext -- only a Werkzeug (PBKDF2) hash.
"""

from datetime import datetime, timezone

from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin

from app import db


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)

    # Grants access to the admin panel (Phase 6). Regular users default to False;
    # set via the `flask create-admin` CLI command (see app/cli.py).
    is_admin = db.Column(db.Boolean, default=False, nullable=False)

    # Used for cold-start personalization before the user has bookmark history
    domain_interest = db.Column(db.String(100), nullable=True)

    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships (deleting a user cleans up their bookmarks/history too)
    bookmarks = db.relationship(
        "Bookmark", backref="user", lazy=True, cascade="all, delete-orphan"
    )
    search_history = db.relationship(
        "SearchHistory", backref="user", lazy=True, cascade="all, delete-orphan"
    )

    def set_password(self, raw_password: str) -> None:
        """Hash and store the user's password. Never call with an already-hashed value."""
        self.password_hash = generate_password_hash(raw_password)

    def check_password(self, raw_password: str) -> bool:
        """Verify a plaintext password attempt against the stored hash."""
        return check_password_hash(self.password_hash, raw_password)

    def __repr__(self):
        return f"<User {self.username}>"
