"""
app/__init__.py

Flask application factory.

Why the App Factory pattern (instead of a single global `app = Flask(__name__)`)?
- Lets us create multiple app instances with different configs (dev/test/prod)
  without one polluting another — critical for the automated test suite.
- Avoids circular imports: extensions (db, migrate) are instantiated here at
  module level WITHOUT an app, then bound to a specific app inside create_app()
  via `.init_app(app)`. Models/routes are imported INSIDE create_app(), after
  the app and extensions exist, which is what prevents circular import errors
  between app/__init__.py <-> app/models/*.py <-> app/routes/*.py.
"""

import os

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager

from app.config import config_by_name

# Extensions are created here, unbound to any specific app instance.
db = SQLAlchemy()
migrate = Migrate()
login_manager = LoginManager()


def create_app(config_name: str = None) -> Flask:
    """
    Application factory.

    Args:
        config_name: one of "development", "testing", "production".
                     Defaults to the FLASK_ENV environment variable, or
                     "development" if unset.
    """
    if config_name is None:
        config_name = os.environ.get("FLASK_ENV", "development")

    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(config_by_name[config_name])

    # Ensure the instance folder exists (holds the SQLite dev database)
    os.makedirs(app.instance_path, exist_ok=True)

    # Bind extensions to this specific app instance
    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)
    login_manager.login_view = "auth.login"
    login_manager.login_message_category = "warning"

    # Import models INSIDE create_app so Flask-Migrate can detect them via
    # SQLAlchemy's metadata, without triggering circular imports at module
    # load time (models import `db` from this file).
    with app.app_context():
        from app.models import user, paper, bookmark, search_history  # noqa: F401

    @login_manager.user_loader
    def load_user(user_id):
        # Imported here (not top-level) for the same circular-import reason
        # as above -- app/models/user.py imports `db` from this very file.
        from app.models.user import User
        return db.session.get(User, int(user_id))

    # Register blueprints (controllers). Each blueprint owns one feature area,
    # per our MVC layout -- routes call into app/services/*.py for business
    # logic and never touch the database directly.
    from app.routes.main_routes import main_bp
    from app.routes.auth_routes import auth_bp
    from app.routes.search_routes import search_bp
    from app.routes.paper_routes import paper_bp
    from app.routes.bookmark_routes import bookmark_bp
    from app.routes.dashboard_routes import dashboard_bp
    from app.routes.admin_routes import admin_bp
    from app.routes.chatbot import chatbot_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp, url_prefix="/auth")
    app.register_blueprint(search_bp)
    app.register_blueprint(paper_bp)
    app.register_blueprint(bookmark_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(admin_bp, url_prefix="/admin")
    app.register_blueprint(chatbot_bp)

    # Register custom CLI commands (e.g. `flask create-admin`)
    from app.cli import register_cli_commands
    register_cli_commands(app)

    return app
