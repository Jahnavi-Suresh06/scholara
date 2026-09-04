"""
main_routes.py

Controller for the landing page and a basic health-check endpoint.

This is intentionally the ONLY blueprint in this phase. Its job right now
is to prove the full stack (Flask app factory + SQLAlchemy models +
Flask-Migrate + templates + static files) is wired together correctly and
runs end-to-end, and to surface dataset preprocessing stats so you can see
Phase 5's output reflected in the running app.

Business logic (reading the dataset stats) is kept minimal and inline here
since it's just a status display, not a real feature yet -- from Phase 6
onward, real business logic moves into app/services/*.py per our MVC layout,
and routes only orchestrate calls to those services.
"""

from pathlib import Path

from flask import Blueprint, render_template, jsonify, current_app

from app import db
from app.models import Paper

main_bp = Blueprint("main", __name__)


def _get_dataset_stats() -> dict:
    """
    Reads datasets/processed/preprocessing_report.txt if present, and
    returns a small summary for display on the landing page. Returns
    a "not yet generated" message if the report doesn't exist -- this
    keeps the page from crashing if someone runs the app before running
    the Phase 5 preprocessing script.
    """
    report_path = Path(current_app.config["DATASET_PROCESSED_DIR"]) / "preprocessing_report.txt"
    csv_path = Path(current_app.config["DATASET_PROCESSED_DIR"]) / "papers_cleaned.csv"

    if not report_path.exists() or not csv_path.exists():
        return {
            "available": False,
            "message": "Dataset not yet processed. Run datasets/scripts/preprocess_arxiv.py first.",
        }

    # Just count rows cheaply without pulling pandas into the request path
    with open(csv_path, "r", encoding="utf-8") as f:
        row_count = sum(1 for _ in f) - 1  # minus header row

    return {
        "available": True,
        "row_count": max(row_count, 0),
        "csv_path": str(csv_path),
    }


@main_bp.route("/")
def index():
    """Landing page: project status + dataset stats."""
    dataset_stats = _get_dataset_stats()
    paper_count_in_db = Paper.query.count()
    return render_template(
        "index.html",
        dataset_stats=dataset_stats,
        paper_count_in_db=paper_count_in_db,
    )


@main_bp.route("/health")
def health():
    """
    Basic health-check endpoint. Confirms:
    - the Flask app is running
    - the database connection is alive
    Useful for Render's health checks after deployment (Phase 10).
    """
    try:
        db.session.execute(db.text("SELECT 1"))
        db_status = "connected"
    except Exception as exc:  # noqa: BLE001 - we want to report ANY db failure here
        db_status = f"error: {exc}"

    return jsonify(
        {
            "status": "ok",
            "database": db_status,
        }
    )
