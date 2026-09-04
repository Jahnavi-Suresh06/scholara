"""
decorators.py

Custom route decorators. Flask-Login already provides @login_required;
this module adds @admin_required on top of it for admin-panel routes.
"""

from functools import wraps

from flask import abort
from flask_login import current_user


def admin_required(view_func):
    """
    Restricts a route to logged-in users with is_admin=True.
    Returns 403 Forbidden for logged-in non-admins, and relies on
    Flask-Login's @login_required (applied alongside this) to handle
    the not-logged-in case by redirecting to the login page.
    """

    @wraps(view_func)
    def wrapped_view(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            abort(403)
        return view_func(*args, **kwargs)

    return wrapped_view
