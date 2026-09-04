"""
config.py

Central configuration for the Flask app. We use class-based config so the
app factory can select Development/Testing/Production config via a single
string, and so no environment-specific values are hardcoded in app code.

All secrets/environment-dependent values come from environment variables
(loaded from .env in development via python-dotenv). Never commit real
secrets — see .env.example for the variables you need to set locally.
"""

import os
from pathlib import Path

# Absolute path to the project root (one level above the app/ package)
BASE_DIR = Path(__file__).resolve().parent.parent


class BaseConfig:
    """Shared config across all environments."""

    # Flask's session/cookie signing key. MUST be overridden in production
    # via the SECRET_KEY environment variable.
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-me")

    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Paths used elsewhere in the app (e.g. by the ML engine in later phases)
    DATASET_PROCESSED_DIR = BASE_DIR / "datasets" / "processed"
    EMBEDDINGS_DIR = BASE_DIR / "datasets" / "embeddings"


class DevelopmentConfig(BaseConfig):
    """Local development: SQLite file, debug on."""

    DEBUG = True
    # NOTE: os.environ.get("DATABASE_URL", default) only falls back to
    # `default` if the key is entirely ABSENT. If .env sets DATABASE_URL=
    # (present but empty, as in our .env.example), get() returns "" instead
    # of the default -- so we explicitly treat an empty string as "unset"
    # with `or`.
    SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL") or (
        f"sqlite:///{BASE_DIR / 'instance' / 'app.db'}"
    )


class TestingConfig(BaseConfig):
    """Used by the automated test suite: isolated in-memory database."""

    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    WTF_CSRF_ENABLED = False


class ProductionConfig(BaseConfig):
    """
    Production (e.g. Render): DATABASE_URL must be set as an environment
    variable (Postgres connection string). DEBUG is always off.
    """

    DEBUG = False
    SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL") or None

    @classmethod
    def validate(cls):
        """
        Flask uses this class directly via app.config.from_object() and
        never instantiates it, so validation can't live in __init__. Call
        ProductionConfig.validate() explicitly at deploy time if you want to
        fail loudly when DATABASE_URL is missing, instead of silently
        crashing later on the first DB query.
        """
        if not cls.SQLALCHEMY_DATABASE_URI:
            raise RuntimeError(
                "DATABASE_URL environment variable must be set in production."
            )


# Maps the string used in create_app(config_name) to the actual class
config_by_name = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
}
