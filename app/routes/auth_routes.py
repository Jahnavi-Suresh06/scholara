"""
auth_routes.py

Handles:
- Registration
- Login
- Logout
- Profile
- Edit Profile
- Change Password
"""

from flask import Blueprint, render_template, request, redirect, url_for, flash

from flask_login import (
    login_user,
    logout_user,
    login_required,
    current_user
)

from app.services.auth_service import (
    register_user,
    authenticate_user,
    update_user_password,
    update_user_profile,
    AuthError
)


auth_bp = Blueprint("auth", __name__)


# ==========================================
# REGISTER
# ==========================================

@auth_bp.route("/register", methods=["GET", "POST"])
def register():

    # Redirect logged-in users
    if current_user.is_authenticated:
        return redirect(
            url_for("dashboard.dashboard")
        )

    if request.method == "POST":

        try:

            user = register_user(
                username=request.form.get("username"),
                email=request.form.get("email"),
                password=request.form.get("password"),
                domain_interest=request.form.get(
                    "domain_interest"
                ),
            )

        except AuthError as exc:

            flash(
                str(exc),
                "danger"
            )

            return render_template(
                "auth/register.html"
            )

        # Log user in
        login_user(user)

        flash(
            f"Welcome, {user.username}! Your account has been created.",
            "success"
        )

        return redirect(
            url_for("dashboard.dashboard")
        )

    return render_template(
        "auth/register.html"
    )


# ==========================================
# LOGIN
# ==========================================

@auth_bp.route("/login", methods=["GET", "POST"])
def login():

    # Redirect logged-in users
    if current_user.is_authenticated:
        return redirect(
            url_for("dashboard.dashboard")
        )

    if request.method == "POST":

        try:

            user = authenticate_user(
                username_or_email=request.form.get(
                    "identifier"
                ),
                password=request.form.get(
                    "password"
                ),
            )

        except AuthError as exc:

            flash(
                str(exc),
                "danger"
            )

            return render_template(
                "auth/login.html"
            )

        # Login user
        login_user(user)

        flash(
            "Logged in successfully.",
            "success"
        )

        next_page = request.args.get("next")

        return redirect(
            next_page
            or url_for("dashboard.dashboard")
        )

    return render_template(
        "auth/login.html"
    )


# ==========================================
# LOGOUT
# ==========================================

@auth_bp.route("/logout")
@login_required
def logout():

    logout_user()

    flash(
        "You have been logged out.",
        "info"
    )

    return redirect(
        url_for("main.index")
    )


# ==========================================
# PROFILE
# ==========================================

@auth_bp.route("/profile")
@login_required
def profile():

    return render_template(
        "auth/profile.html",
        user=current_user
    )


# ==========================================
# EDIT PROFILE
# ==========================================

@auth_bp.route("/edit-profile", methods=["GET", "POST"])
@login_required
def edit_profile():

    # ------------------------------------------
    # SHOW EDIT PROFILE PAGE
    # ------------------------------------------

    if request.method == "GET":

        return render_template(
            "auth/edit_profile.html",
            user=current_user
        )

    # ------------------------------------------
    # UPDATE PROFILE
    # ------------------------------------------

    try:

        update_user_profile(

            user=current_user,

            username=request.form.get(
                "username"
            ),

            email=request.form.get(
                "email"
            ),

            domain_interest=request.form.get(
                "domain_interest"
            )
        )

    except AuthError as exc:

        flash(
            str(exc),
            "danger"
        )

        return render_template(
            "auth/edit_profile.html",
            user=current_user
        )

    # ------------------------------------------
    # SUCCESS MESSAGE
    # ------------------------------------------

    flash(
        "Your profile has been updated successfully.",
        "success"
    )

    return redirect(
        url_for("auth.profile")
    )


# ==========================================
# CHANGE PASSWORD
# ==========================================

@auth_bp.route("/change-password", methods=["GET", "POST"])
@login_required
def update_password():

    # ------------------------------------------
    # SHOW CHANGE PASSWORD PAGE
    # ------------------------------------------

    if request.method == "GET":

        return render_template(
            "auth/change_password.html"
        )

    # ------------------------------------------
    # UPDATE PASSWORD
    # ------------------------------------------

    try:

        update_user_password(

            user=current_user,

            current_password=request.form.get(
                "current_password"
            ),

            new_password=request.form.get(
                "new_password"
            ),

            confirm_password=request.form.get(
                "confirm_password"
            )
        )

    except AuthError as exc:

        flash(
            str(exc),
            "danger"
        )

        return render_template(
            "auth/change_password.html"
        )

    # ------------------------------------------
    # SUCCESS MESSAGE
    # ------------------------------------------

    flash(
        "Your password has been changed successfully.",
        "success"
    )

    return redirect(
        url_for("auth.profile")
    )
