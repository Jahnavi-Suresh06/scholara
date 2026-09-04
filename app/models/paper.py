"""
Paper model.

Stores research paper metadata. The semantic embedding vector for this
paper's abstract is NOT stored here -- it lives in a separate NumPy array
(datasets/embeddings/embeddings.npy, built in Phase 8). `embedding_index`
is the bridge between this SQL row and that vector's position in the array.

Why keep vectors out of SQL?
- SQLite/Postgres aren't optimized for dense vector storage/search at scale.
- Keeps the ML layer swappable (NumPy today, FAISS later) without touching
  this table.
"""

from app import db


class Paper(db.Model):
    __tablename__ = "papers"

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.Text, nullable=False)
    authors = db.Column(db.Text, nullable=False)  # comma-separated for now
    abstract = db.Column(db.Text, nullable=False)  # source text for embeddings
    keywords = db.Column(db.Text, nullable=True)
    publication_year = db.Column(db.Integer, nullable=True, index=True)
    journal_conference = db.Column(db.String(255), nullable=True)
    citation_count = db.Column(db.Integer, default=0)
    doi = db.Column(db.String(255), nullable=True, unique=True)
    pdf_link = db.Column(db.String(500), nullable=True)

    # Bridge to this paper's row index in datasets/embeddings/embeddings.npy
    embedding_index = db.Column(db.Integer, nullable=True, unique=True, index=True)

    bookmarks = db.relationship(
        "Bookmark", backref="paper", lazy=True, cascade="all, delete-orphan"
    )

    def to_dict(self) -> dict:
        """Serialize for JSON API responses / template rendering."""
        return {
            "id": self.id,
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "keywords": self.keywords,
            "publication_year": self.publication_year,
            "journal_conference": self.journal_conference,
            "citation_count": self.citation_count,
            "doi": self.doi,
            "pdf_link": self.pdf_link,
        }

    def __repr__(self):
        return f"<Paper {self.id}: {self.title[:50]}>"
