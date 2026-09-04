"""
convert_arxivdata_to_kaggle_format.py

Adapter script: converts the "arxivData.json" dataset format (a single JSON
array, with `author`/`tag` stored as stringified Python lists of dicts, and
`year`/`month`/`day` as separate int fields) into the JSON-lines format that
preprocess_arxiv.py expects (one JSON object per line, matching the Kaggle
Cornell arXiv snapshot's field names: authors, categories, versions, id,
title, abstract, journal-ref, doi).

This lets you use arxivData.json as a drop-in alternative data source
without touching preprocess_arxiv.py at all.

Usage:
    python convert_arxivdata_to_kaggle_format.py \
        --input ../raw/arxivData.json \
        --output ../raw/arxiv-metadata-oai-snapshot.json

After running this, follow the normal README pipeline:
    cd datasets/scripts
    python preprocess_arxiv.py --input ../raw/arxiv-metadata-oai-snapshot.json \
                                --output ../processed/papers_cleaned.csv \
                                --limit 30000
    python ingest_to_db.py --csv ../processed/papers_cleaned.csv
    python build_embeddings.py
"""

import argparse
import ast
import json
import sys
from pathlib import Path


def parse_stringified_list(value):
    """
    Safely parse fields like "[{'name': 'Ahmed Osman'}, ...]" back into a
    real Python list of dicts. Falls back to an empty list if parsing fails
    (some records in the wild have malformed strings here).
    """
    if not value:
        return []
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return parsed
    except (ValueError, SyntaxError):
        pass
    return []


def convert_record(raw: dict) -> dict:
    """Map one arxivData.json record to the Kaggle snapshot schema."""
    authors_list = parse_stringified_list(raw.get("author", ""))
    author_names = [a.get("name", "") for a in authors_list if isinstance(a, dict)]
    authors_str = ", ".join(name for name in author_names if name)

    tags_list = parse_stringified_list(raw.get("tag", ""))
    category_terms = [t.get("term", "") for t in tags_list if isinstance(t, dict)]
    categories_str = " ".join(term for term in category_terms if term)

    year = raw.get("year", "")
    month = raw.get("month", 1) or 1
    day = raw.get("day", 1) or 1
    # Kaggle's `versions[0].created` is a free-text date string; a simple
    # ISO-ish YYYY-MM-DD is enough since preprocess_arxiv.py just stores it
    # as-is in `published_date` without parsing it.
    created_date = f"{year}-{int(month):02d}-{int(day):02d}" if year else ""

    return {
        # Strip the "v1"/"v2" version suffix arxivData.json includes in id
        "id": raw.get("id", "").split("v")[0].strip(),
        "title": raw.get("title", ""),
        "authors": authors_str,
        "abstract": raw.get("summary", ""),
        "categories": categories_str,
        "versions": [{"created": created_date}] if created_date else [],
        "journal-ref": "",
        "doi": "",
    }


def run_conversion(input_path: Path, output_path: Path):
    with open(input_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for raw in records:
            try:
                converted = convert_record(raw)
            except Exception:
                skipped += 1
                continue
            out.write(json.dumps(converted) + "\n")
            written += 1

    print(f"Read {len(records)} records from {input_path}")
    print(f"Wrote {written} JSON-lines records to {output_path}")
    if skipped:
        print(f"Skipped {skipped} malformed records")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert arxivData.json into Kaggle-style JSON-lines format."
    )
    parser.add_argument("--input", type=str, required=True, help="Path to arxivData.json")
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to write JSON-lines output (e.g. ../raw/arxiv-metadata-oai-snapshot.json)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"ERROR: input file not found at {input_path}")
        sys.exit(1)

    run_conversion(input_path, output_path)
