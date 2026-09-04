"""
app/cli.py

Custom Flask CLI commands, registered onto the app in create_app().

Usage (after `flask db upgrade` has created the tables):
    flask create-admin --username admin --email admin@example.com --password changeme
"""

import click
from flask.cli import with_appcontext

from app import db


def register_cli_commands(app):
    app.cli.add_command(create_admin_command)


@click.command("create-admin")
@click.option("--username", prompt=True)
@click.option("--email", prompt=True)
@click.option("--password", prompt=True, hide_input=True, confirmation_prompt=True)
@with_appcontext
def create_admin_command(username, email, password):
    """Create an admin user who can access /admin routes."""
    from app.models.user import User

    if User.query.filter((User.username == username) | (User.email == email)).first():
        click.echo("A user with that username or email already exists.")
        return

    user = User(username=username, email=email, is_admin=True)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    click.echo(f"Admin user '{username}' created successfully.")
