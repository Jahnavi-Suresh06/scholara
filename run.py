"""
run.py

Entry point for running the Flask development server.

Usage:
    python run.py

For production, this file is NOT used directly -- Render/gunicorn imports
the `app` object from here instead (see render.yaml's startCommand).
"""

import os

from dotenv import load_dotenv

# Load environment variables from .env before creating the app, so
# app/config.py can read them via os.environ.get(...)
load_dotenv()

from app import create_app  # noqa: E402  (import after load_dotenv is intentional)

app = create_app(os.environ.get("FLASK_ENV", "development"))

if __name__ == "__main__":
    app.run(debug=app.config.get("DEBUG", False), host="0.0.0.0", port=5000)
