"""
ingest_to_db.py

Loads datasets/processed/papers_cleaned.csv (produced by preprocess_arxiv.py,
Phase 5) into the `papers` SQL table.

Usage (run from the project root, NOT from datasets/scripts/):
    python datasets/scripts/ingest_to_db.py --csv datasets/processed/papers_cleaned.csv

Design notes:
- This script does NOT set Paper.embedding_index. That column is populated
  by Phase 8's build_embeddings.py, which computes SBERT embeddings for
  whatever papers exist in the DB at that time and records where each
  paper's vector lives in the embeddings array. Setting a placeholder value
  here would just create a value that Phase 8 immediately has to overwrite,
  so we leave it NULL until it has real meaning.
- Dedupes by DOI first when the CSV provides one (OpenAlex records almost
  always have one, and Paper.doi is a unique DB column so we must catch
  this ourselves before it becomes an IntegrityError). Falls back to
  (title, authors) for rows without a DOI, since arXiv preprints frequently
  lack one.
- published_date arrives in different formats depending on the source
  pipeline: OpenAlex's fetch_openalex.py writes plain ISO 'YYYY-MM-DD',
  while the legacy arXiv preprocess_arxiv.py writes an RFC-2822-style
  string (e.g. "Mon, 4 Jan 2021 10:00:00 GMT"). parse_year() below handles
  both so either CSV ingests correctly.
- citation_count/pdf_link are read straight from the CSV when the source
  pipeline provides them (OpenAlex does); otherwise falls back to the old
  defaults (0, and an arxiv.org abstract-page URL built from paper_id) so
  legacy arXiv-derived CSVs keep working unchanged.
"""

import argparse
import csv
import sys
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path

# Make the project root importable when this script is run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app import create_app, db  # noqa: E402
from app.models.paper import Paper  # noqa: E402


def parse_year(date_str: str):
    """Extract a 4-digit year from either an ISO date ('2021-01-04', as
    written by fetch_openalex.py) or arXiv's RFC-2822-style date string
    ('Mon, 4 Jan 2021 10:00:00 GMT', as written by preprocess_arxiv.py)."""
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(date_str[:10]).year
    except (TypeError, ValueError):
        pass
    try:
        return parsedate_to_datetime(date_str).year
    except (TypeError, ValueError):
        return None


def parse_citation_count(value: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def build_pdf_link(row: dict) -> str:
    """Use the source pipeline's own pdf_link column when present (OpenAlex
    gives a real open-access/landing URL). Falls back to constructing an
    arXiv abstract-page URL from paper_id, for legacy arXiv CSVs that don't
    have a pdf_link column at all."""
    pdf_link = (row.get("pdf_link") or "").strip()
    if pdf_link:
        return pdf_link
    paper_id = row.get("paper_id")
    return f"https://arxiv.org/abs/{paper_id}" if paper_id else None


def ingest(csv_path: Path, limit: int = None):
    app = create_app("development")

    with app.app_context():
        existing_pairs = {
            (p.title, p.authors) for p in Paper.query.with_entities(Paper.title, Paper.authors)
        }
        existing_dois = {
            p.doi for p in Paper.query.with_entities(Paper.doi) if p.doi
        }

        inserted = 0
        skipped_duplicate = 0
        skipped_invalid = 0

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if limit and i >= limit:
                    break

                title = (row.get("title") or "").strip()
                authors = (row.get("authors") or "").strip()
                abstract = (row.get("abstract") or "").strip()
                doi = (row.get("doi") or "").strip() or None

                if not title or not authors or not abstract:
                    skipped_invalid += 1
                    continue

                # DOI is the more reliable key when present (Paper.doi is
                # also a unique DB column, so this also prevents an
                # IntegrityError on insert). Fall back to (title, authors)
                # for rows without one, e.g. arXiv preprints.
                if doi and doi in existing_dois:
                    skipped_duplicate += 1
                    continue
                if (title, authors) in existing_pairs:
                    skipped_duplicate += 1
                    continue

                paper = Paper(
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    keywords=(row.get("categories") or "").strip() or None,
                    publication_year=parse_year(row.get("published_date")),
                    journal_conference=(row.get("journal_ref") or "").strip() or None,
                    citation_count=parse_citation_count(row.get("citation_count")),
                    doi=doi,
                    pdf_link=build_pdf_link(row),
                )
                db.session.add(paper)
                existing_pairs.add((title, authors))
                if doi:
                    existing_dois.add(doi)
                inserted += 1

        db.session.commit()

        print("=" * 50)
        print("INGESTION SUMMARY")
        print("=" * 50)
        print(f"Inserted          : {inserted}")
        print(f"Skipped (duplicate): {skipped_duplicate}")
        print(f"Skipped (invalid)  : {skipped_invalid}")
        print(f"Total papers in DB now: {Paper.query.count()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest the cleaned arXiv CSV into the papers table.")
    parser.add_argument("--csv", type=str, default="datasets/processed/papers_cleaned.csv")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: CSV not found at {csv_path}")
        sys.exit(1)

    ingest(csv_path, limit=args.limit)
