"""
admin_service.py

CRUD operations on the papers table, used by the admin panel routes.
Kept separate from search_service.py because these operations mutate data
and are only reachable by admins, whereas search_service.py is read-only
and open to all logged-in users.
"""

from app import db
from app.models.paper import Paper


class AdminServiceError(Exception):
    pass


def list_papers_paginated(page: int = 1, per_page: int = 20):
    return Paper.query.order_by(Paper.id.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )


def create_paper(form_data: dict) -> Paper:
    title = (form_data.get("title") or "").strip()
    authors = (form_data.get("authors") or "").strip()
    abstract = (form_data.get("abstract") or "").strip()

    if not title or not authors or not abstract:
        raise AdminServiceError("Title, authors, and abstract are required.")

    paper = Paper(
        title=title,
        authors=authors,
        abstract=abstract,
        keywords=(form_data.get("keywords") or "").strip() or None,
        publication_year=_parse_year(form_data.get("publication_year")),
        journal_conference=(form_data.get("journal_conference") or "").strip() or None,
        citation_count=_parse_int(form_data.get("citation_count"), default=0),
        doi=(form_data.get("doi") or "").strip() or None,
        pdf_link=(form_data.get("pdf_link") or "").strip() or None,
    )
    db.session.add(paper)
    db.session.commit()
    return paper


def update_paper(paper_id: int, form_data: dict) -> Paper:
    paper = db.session.get(Paper, paper_id)
    if paper is None:
        raise AdminServiceError("Paper not found.")

    paper.title = (form_data.get("title") or paper.title).strip()
    paper.authors = (form_data.get("authors") or paper.authors).strip()
    paper.abstract = (form_data.get("abstract") or paper.abstract).strip()
    paper.keywords = (form_data.get("keywords") or "").strip() or None
    paper.publication_year = _parse_year(form_data.get("publication_year"))
    paper.journal_conference = (form_data.get("journal_conference") or "").strip() or None
    paper.citation_count = _parse_int(form_data.get("citation_count"), default=paper.citation_count)
    paper.doi = (form_data.get("doi") or "").strip() or None
    paper.pdf_link = (form_data.get("pdf_link") or "").strip() or None

    db.session.commit()
    return paper


def delete_paper(paper_id: int) -> bool:
    paper = db.session.get(Paper, paper_id)
    if paper is None:
        return False
    db.session.delete(paper)
    db.session.commit()
    return True


def _parse_year(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
