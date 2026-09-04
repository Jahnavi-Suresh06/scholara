"""
admin_routes.py

Controller for the admin panel (paper management CRUD). Every route here
requires BOTH @login_required (must be logged in) AND @admin_required
(must have is_admin=True) -- see app/utils/decorators.py.
"""

from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_required

from app.utils.decorators import admin_required
from app.services.admin_service import (
    list_papers_paginated,
    create_paper,
    update_paper,
    delete_paper,
    AdminServiceError,
)
from app.models.paper import Paper

admin_bp = Blueprint("admin", __name__)


@admin_bp.route("/papers")
@login_required
@admin_required
def manage_papers():
    page = request.args.get("page", 1, type=int)
    papers_page = list_papers_paginated(page=page, per_page=20)
    return render_template("admin/manage_papers.html", papers_page=papers_page)


@admin_bp.route("/papers/new", methods=["GET", "POST"])
@login_required
@admin_required
def new_paper():
    if request.method == "POST":
        try:
            paper = create_paper(request.form)
            flash(f"Paper '{paper.title}' created.", "success")
            return redirect(url_for("admin.manage_papers"))
        except AdminServiceError as exc:
            flash(str(exc), "danger")
            return render_template("admin/paper_form.html", paper=None, form_data=request.form)

    return render_template("admin/paper_form.html", paper=None, form_data=None)


@admin_bp.route("/papers/<int:paper_id>/edit", methods=["GET", "POST"])
@login_required
@admin_required
def edit_paper(paper_id):
    paper = Paper.query.get_or_404(paper_id)

    if request.method == "POST":
        try:
            paper = update_paper(paper_id, request.form)
            flash(f"Paper '{paper.title}' updated.", "success")
            return redirect(url_for("admin.manage_papers"))
        except AdminServiceError as exc:
            flash(str(exc), "danger")

    return render_template("admin/paper_form.html", paper=paper, form_data=None)


@admin_bp.route("/papers/<int:paper_id>/delete", methods=["POST"])
@login_required
@admin_required
def delete_paper_route(paper_id):
    if delete_paper(paper_id):
        flash("Paper deleted.", "info")
    else:
        flash("Paper not found.", "warning")
    return redirect(url_for("admin.manage_papers"))
