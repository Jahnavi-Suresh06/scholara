"""
auth_service.py

Business logic for:
- User registration
- User authentication
- Password updates
- Profile updates
"""

from app import db
from app.models.user import User


# ==========================================
# AUTH ERROR
# ==========================================

class AuthError(Exception):
    """Raised when authentication or validation fails."""
    pass


# ==========================================
# REGISTER USER
# ==========================================

def register_user(username, email, password, domain_interest=None):

    # Clean input
    username = (username or "").strip()
    email = (email or "").strip().lower()
    password = password or ""

    # Validate required fields
    if not username or not email or not password:
        raise AuthError(
            "Username, email, and password are required."
        )

    # Password validation
    if len(password) < 8:
        raise AuthError(
            "Password must be at least 8 characters long."
        )

    # Check username
    existing_username = User.query.filter_by(
        username=username
    ).first()

    if existing_username:
        raise AuthError(
            "That username is already taken."
        )

    # Check email
    existing_email = User.query.filter_by(
        email=email
    ).first()

    if existing_email:
        raise AuthError(
            "An account with that email already exists."
        )

    # Create user
    user = User(
        username=username,
        email=email,
        domain_interest=domain_interest or None
    )

    # Hash password
    user.set_password(password)

    # Save user
    db.session.add(user)
    db.session.commit()

    return user


# ==========================================
# AUTHENTICATE USER
# ==========================================

def authenticate_user(username_or_email, password):

    username_or_email = (
        username_or_email or ""
    ).strip()

    password = password or ""

    # Search using username or email
    user = User.query.filter(
        (User.username == username_or_email)
        |
        (User.email == username_or_email.lower())
    ).first()

    # Check user
    if user is None:
        raise AuthError(
            "Invalid username/email or password."
        )

    # Check password
    if not user.check_password(password):
        raise AuthError(
            "Invalid username/email or password."
        )

    return user


# ==========================================
# UPDATE PASSWORD
# ==========================================

def update_user_password(
    user,
    current_password,
    new_password,
    confirm_password
):

    # Ensure fields exist
    current_password = current_password or ""
    new_password = new_password or ""
    confirm_password = confirm_password or ""

    # Check current password
    if not user.check_password(current_password):
        raise AuthError(
            "Your current password is incorrect."
        )

    # Check password length
    if len(new_password) < 8:
        raise AuthError(
            "New password must be at least 8 characters long."
        )

    # Check passwords match
    if new_password != confirm_password:
        raise AuthError(
            "New password and confirmation password do not match."
        )

    # Prevent same password
    if user.check_password(new_password):
        raise AuthError(
            "Your new password cannot be the same as your current password."
        )

    # Update password
    user.set_password(new_password)

    # Save changes
    db.session.commit()

    return user


# ==========================================
# UPDATE USER PROFILE
# ==========================================

def update_user_profile(
    user,
    username,
    email,
    domain_interest=None
):

    # Clean inputs
    username = (username or "").strip()
    email = (email or "").strip().lower()
    domain_interest = (domain_interest or "").strip()

    # Validate username
    if not username:
        raise AuthError(
            "Username is required."
        )

    # Validate email
    if not email:
        raise AuthError(
            "Email is required."
        )

    # Check whether another user has this username
    existing_username = User.query.filter(
        User.username == username,
        User.id != user.id
    ).first()

    if existing_username:
        raise AuthError(
            "That username is already taken."
        )

    # Check whether another user has this email
    existing_email = User.query.filter(
        User.email == email,
        User.id != user.id
    ).first()

    if existing_email:
        raise AuthError(
            "An account with that email already exists."
        )

    # Update user details
    user.username = username
    user.email = email
    user.domain_interest = domain_interest or None

    # Save changes
    db.session.commit()

    return user
