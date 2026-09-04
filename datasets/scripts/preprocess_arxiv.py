"""
preprocess_arxiv.py

Reusable preprocessing script for the Cornell/Kaggle arXiv Metadata Dataset.

INPUT : datasets/raw/arxiv-metadata-oai-snapshot.json
        (JSON-lines format: one JSON object per line, as distributed on Kaggle)
OUTPUT: datasets/processed/papers_cleaned.csv
        datasets/processed/preprocessing_report.txt

Usage:
    python preprocess_arxiv.py --input ../raw/arxiv-metadata-oai-snapshot.json \
                                --output ../processed/papers_cleaned.csv \
                                --limit 50000

--limit is optional. The real snapshot has ~2-3 million records; for a final-year
project a curated subset (e.g. 20k-50k papers, optionally filtered to CS/AI/DS
categories) is both faster to embed and easier to demo. Omit --limit to process
everything.

Design notes:
- Streamed line-by-line (not json.load on the whole file) because the raw file
  is 1GB+; loading it all into memory at once is a common student mistake that
  crashes on typical laptops.
- Every cleaning decision is logged into counters so the preprocessing report
  is generated from real numbers, not guessed after the fact.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd


# Columns required by our database schema (Phase 4)
REQUIRED_COLUMNS = [
    "paper_id",
    "title",
    "authors",
    "abstract",
    "categories",
    "published_date",
    "journal_ref",
    "doi",
]

# A record is unusable for our recommendation engine without these —
# everything else is allowed to be missing/blank.
MANDATORY_FIELDS = ["paper_id", "title", "abstract"]

# Minimum abstract length (in words). Very short abstracts are usually
# placeholder text ("see comments", "withdrawn", etc.) and would produce
# meaningless embeddings — this threshold is a known best practice from
# published arXiv-preprocessing work (see: filtering criteria used in
# academic subsetting of this same dataset).
MIN_ABSTRACT_WORDS = 20


def clean_text(text: str) -> str:
    """Normalize whitespace/newlines that are common in raw arXiv abstracts."""
    if not isinstance(text, str):
        return ""
    text = text.replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def extract_year(versions_or_date) -> str:
    """
    The raw arXiv record doesn't have a single clean 'published_date' field —
    it has an 'update_date' field and a 'versions' list (first version = original
    submission). We use the first version's creation date as published_date.
    """
    if isinstance(versions_or_date, list) and len(versions_or_date) > 0:
        raw_date = versions_or_date[0].get("created", "")
        return raw_date
    return ""


def parse_record(raw: dict) -> dict:
    """Map one raw arXiv JSON record to our required schema."""
    return {
        "paper_id": raw.get("id", "").strip(),
        "title": clean_text(raw.get("title", "")),
        "authors": clean_text(raw.get("authors", "")),
        "abstract": clean_text(raw.get("abstract", "")),
        "categories": clean_text(raw.get("categories", "")),
        "published_date": extract_year(raw.get("versions", [])),
        "journal_ref": clean_text(raw.get("journal-ref", "") or ""),
        "doi": (raw.get("doi") or "").strip(),
    }


def run_preprocessing(input_path: Path, output_path: Path, report_path: Path, limit: int = None):
    counters = {
        "total_read": 0,
        "dropped_missing_mandatory": 0,
        "dropped_short_abstract": 0,
        "dropped_duplicate": 0,
        "final_kept": 0,
    }

    seen_ids = set()
    records = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            counters["total_read"] += 1
            if limit and counters["total_read"] > limit:
                counters["total_read"] -= 1
                break

            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue

            record = parse_record(raw)

            # Rule 1: mandatory fields must be present and non-empty
            if any(not record[field] for field in MANDATORY_FIELDS):
                counters["dropped_missing_mandatory"] += 1
                continue

            # Rule 2: abstract must be substantial enough to embed meaningfully
            if len(record["abstract"].split()) < MIN_ABSTRACT_WORDS:
                counters["dropped_short_abstract"] += 1
                continue

            # Rule 3: drop duplicate paper_ids (arXiv snapshot can contain
            # revision duplicates depending on export)
            if record["paper_id"] in seen_ids:
                counters["dropped_duplicate"] += 1
                continue
            seen_ids.add(record["paper_id"])

            records.append(record)
            counters["final_kept"] += 1

    df = pd.DataFrame(records, columns=REQUIRED_COLUMNS)

    # Missing-value stats computed on the FINAL cleaned dataframe
    # (mandatory fields will show 0 missing by construction; optional fields
    # like journal_ref/doi are legitimately often blank for preprints)
    missing_stats = df.isna().sum() + (df == "").sum()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # ---- Build the preprocessing report ----
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("ARXIV DATASET PREPROCESSING REPORT")
    report_lines.append("=" * 60)
    report_lines.append(f"Source file        : {input_path}")
    report_lines.append(f"Original rows read  : {counters['total_read']}")
    report_lines.append("")
    report_lines.append("Rows removed (by reason):")
    report_lines.append(f"  - Missing mandatory field(s)      : {counters['dropped_missing_mandatory']}")
    report_lines.append(f"  - Abstract too short (<{MIN_ABSTRACT_WORDS} words)  : {counters['dropped_short_abstract']}")
    report_lines.append(f"  - Duplicate paper_id               : {counters['dropped_duplicate']}")
    total_removed = (
        counters["dropped_missing_mandatory"]
        + counters["dropped_short_abstract"]
        + counters["dropped_duplicate"]
    )
    report_lines.append(f"  - TOTAL REMOVED                     : {total_removed}")
    report_lines.append("")
    report_lines.append(f"Final cleaned dataset size: {counters['final_kept']} rows")
    report_lines.append("")
    report_lines.append("Missing value counts (in final cleaned dataset, per column):")
    for col, count in missing_stats.items():
        report_lines.append(f"  - {col}: {count}")
    report_lines.append("")
    report_lines.append("Sample records (first 3 rows):")
    report_lines.append("-" * 60)
    for i, row in df.head(3).iterrows():
        report_lines.append(f"[{i}] paper_id: {row['paper_id']}")
        report_lines.append(f"    title: {row['title'][:100]}")
        report_lines.append(f"    authors: {row['authors'][:80]}")
        report_lines.append(f"    categories: {row['categories']}")
        report_lines.append(f"    abstract (first 150 chars): {row['abstract'][:150]}...")
        report_lines.append("-" * 60)

    report_text = "\n".join(report_lines)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
    return df, counters


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess the raw arXiv metadata dataset.")
    parser.add_argument("--input", type=str, required=True, help="Path to raw arxiv-metadata-oai-snapshot.json")
    parser.add_argument("--output", type=str, required=True, help="Path to write cleaned CSV")
    parser.add_argument("--report", type=str, default=None, help="Path to write the preprocessing report")
    parser.add_argument("--limit", type=int, default=None, help="Optional: only process first N records")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report) if args.report else output_path.parent / "preprocessing_report.txt"

    if not input_path.exists():
        print(f"ERROR: input file not found at {input_path}")
        sys.exit(1)

    run_preprocessing(input_path, output_path, report_path, limit=args.limit)
