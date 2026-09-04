"""
fetch_openalex.py

Replaces arxivData.json / convert_arxivdata_to_kaggle_format.py / preprocess_arxiv.py
as the dataset source. Pulls works directly from the OpenAlex REST API
(https://api.openalex.org/works) and writes datasets/processed/papers_cleaned.csv
in the SAME schema preprocess_arxiv.py used to produce, so ingest_to_db.py,
build_embeddings.py, and everything downstream needs no changes to run.

Why OpenAlex instead of the arXiv snapshot:
- Covers IEEE, ACM, Springer, Elsevier, etc. -- not just arXiv preprints.
- Real DOIs on the large majority of records (arXiv preprints frequently
  lack one).
- Real citation counts (`cited_by_count`) -- the arXiv pipeline could only
  ever default this to 0, since the Kaggle snapshot doesn't have it.
- A real venue/journal name per paper (`primary_location.source.display_name`),
  not just a blank journal_ref like most arXiv preprints have.

Two extra columns beyond the original REQUIRED_COLUMNS are written --
citation_count and pdf_link. ingest_to_db.py has been updated to read them
when present and fall back to its old defaults when they're not (so old
arXiv-derived CSVs still ingest fine).

REQUIRES a free OpenAlex API key as of Feb 13, 2026 (OpenAlex retired the
old "polite pool" mailto param and now requires api_key on every request --
see https://help.openalex.org/hc/en-us/articles/38868153578263-Changelog).
Get a free key in ~30 seconds at https://openalex.org/settings/api, then
pass it with --api-key, or set it once as an environment variable:
    setx OPENALEX_API_KEY "your-key-here"      (Windows, new terminals)
    $env:OPENALEX_API_KEY = "your-key-here"     (Windows, current session)
    export OPENALEX_API_KEY="your-key-here"     (macOS/Linux)
Single-record lookups by DOI/ID stay free regardless; it's filter/search/
list queries (everything this script does) that now require the key.

Usage examples:
    # 20,000 CS papers, newest first
    python fetch_openalex.py --search "machine learning" \
        --concept "computer science" --limit 20000 --api-key YOUR_KEY

    # Everything published in IEEE venues matching a concept
    # ('ieee' is resolved to the real OpenAlex publisher ID automatically)
    python fetch_openalex.py --concept "artificial intelligence" \
        --publisher ieee --limit 20000 --api-key YOUR_KEY

    # Full control via a raw OpenAlex filter string
    python fetch_openalex.py \
        --raw-filter "concepts.id:C41008148,from_publication_date:2019-01-01" \
        --limit 20000 --api-key YOUR_KEY
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

API_URL = "https://api.openalex.org/works"
CONCEPTS_URL = "https://api.openalex.org/concepts"
PUBLISHERS_URL = "https://api.openalex.org/publishers"

# Same contract preprocess_arxiv.py used -- ingest_to_db.py depends on these
# column names existing. citation_count/pdf_link are additive (see module
# docstring), not a breaking change to that contract.
REQUIRED_COLUMNS = [
    "paper_id",
    "title",
    "authors",
    "abstract",
    "categories",
    "published_date",
    "journal_ref",
    "doi",
    "citation_count",
    "pdf_link",
]

MANDATORY_FIELDS = ["paper_id", "title", "abstract"]
MIN_ABSTRACT_WORDS = 20
PER_PAGE = 200  # OpenAlex max page size


def resolve_concept_id(name: str, session: requests.Session, api_key: str = None) -> str:
    """Look up an OpenAlex concept ID (e.g. 'C41008148') from a plain-English
    name like 'computer science' via the /concepts search endpoint."""
    params = {"search": name, "per-page": 1}
    if api_key:
        params["api_key"] = api_key
    resp = session.get(CONCEPTS_URL, params=params, timeout=30)
    resp.raise_for_status()
    results = resp.json().get("results", [])
    if not results:
        print(f"ERROR: no OpenAlex concept found matching '{name}'")
        sys.exit(1)
    concept = results[0]
    print(f"Resolved concept '{name}' -> {concept['id']} ({concept['display_name']})")
    # concept['id'] is a full URL like https://openalex.org/C41008148
    return concept["id"].rsplit("/", 1)[-1]


def resolve_publisher_id(name: str, session: requests.Session, api_key: str = None) -> str:
    """Look up an OpenAlex publisher ID (e.g. 'P4310320595') from a plain
    name like 'ieee' via the /publishers search endpoint. OpenAlex's
    host_organization_lineage filter requires this real ID -- passing the
    literal string 'ieee' into the filter would silently match nothing."""
    params = {"search": name, "per-page": 1}
    if api_key:
        params["api_key"] = api_key
    resp = session.get(PUBLISHERS_URL, params=params, timeout=30)
    resp.raise_for_status()
    results = resp.json().get("results", [])
    if not results:
        print(f"ERROR: no OpenAlex publisher found matching '{name}'")
        sys.exit(1)
    publisher = results[0]
    print(f"Resolved publisher '{name}' -> {publisher['id']} ({publisher['display_name']})")
    # publisher['id'] is a full URL like https://openalex.org/P4310320595
    return publisher["id"].rsplit("/", 1)[-1]


def reconstruct_abstract(inverted_index: dict) -> str:
    """OpenAlex stores abstracts as {word: [positions]} to respect
    publisher copyright on abstract text reproduction. Rebuild the plain
    text by placing each word back at its recorded position(s)."""
    if not inverted_index:
        return ""
    max_pos = max(pos for positions in inverted_index.values() for pos in positions)
    words = [""] * (max_pos + 1)
    for word, positions in inverted_index.items():
        for pos in positions:
            words[pos] = word
    return " ".join(w for w in words if w)


def extract_authors(authorships: list) -> str:
    names = [
        a.get("author", {}).get("display_name", "")
        for a in (authorships or [])
        if a.get("author")
    ]
    return ", ".join(n for n in names if n)


def extract_categories(concepts: list, top_n: int = 5) -> str:
    """Space-joined top concepts by relevance score -- fills the same role
    arXiv's space-separated category codes (e.g. 'cs.LG cs.AI') played."""
    sorted_concepts = sorted(concepts or [], key=lambda c: c.get("score", 0), reverse=True)
    names = [c.get("display_name", "") for c in sorted_concepts[:top_n]]
    return " ".join(n.replace(" ", "_") for n in names if n)


def extract_pdf_link(work: dict) -> str:
    """Prefer a real open-access PDF/landing URL; fall back to the DOI
    resolver link; fall back to the OpenAlex work page itself."""
    oa = work.get("open_access") or {}
    if oa.get("oa_url"):
        return oa["oa_url"]
    primary = work.get("primary_location") or {}
    if primary.get("landing_page_url"):
        return primary["landing_page_url"]
    if work.get("doi"):
        return work["doi"]  # OpenAlex already returns this as a full https://doi.org/... URL
    return work.get("id")  # https://openalex.org/W... as a last resort


def parse_record(work: dict) -> dict:
    """Map one OpenAlex `work` object to our CSV schema."""
    doi = (work.get("doi") or "").replace("https://doi.org/", "").strip()
    primary_location = work.get("primary_location") or {}
    source = primary_location.get("source") or {}

    return {
        "paper_id": (work.get("id") or "").rsplit("/", 1)[-1],  # e.g. "W2741809807"
        "title": (work.get("title") or work.get("display_name") or "").strip(),
        "authors": extract_authors(work.get("authorships")),
        "abstract": reconstruct_abstract(work.get("abstract_inverted_index")),
        "categories": extract_categories(work.get("concepts")),
        "published_date": work.get("publication_date") or "",
        "journal_ref": (source.get("display_name") or "").strip(),
        "doi": doi,
        "citation_count": work.get("cited_by_count", 0) or 0,
        "pdf_link": extract_pdf_link(work),
    }


def build_filter(args) -> str:
    """Assemble the OpenAlex `filter` query param from the CLI flags."""
    clauses = []

    if args.raw_filter:
        clauses.append(args.raw_filter)

    if args.concept_id:
        clauses.append(f"concepts.id:{args.concept_id}")

    if args.publisher_id:
        # publisher_lineage covers publisher imprints/subsidiaries too,
        # e.g. IEEE conference proceedings published under an IEEE imprint
        clauses.append(f"primary_location.source.publisher_lineage:{args.publisher_id}")

    if args.from_year:
        clauses.append(f"from_publication_date:{args.from_year}-01-01")

    if args.has_abstract:
        clauses.append("has_abstract:true")

    return ",".join(clauses) if clauses else None


def run_fetch(args):
    session = requests.Session()

    if not args.api_key:
        print(
            "WARNING: no API key provided. OpenAlex has required an api_key on\n"
            "every filter/search/list request since Feb 13, 2026 -- without one\n"
            "this will likely return zero results or fail outright. Get a free\n"
            "key at https://openalex.org/settings/api and pass it with --api-key\n"
            "(or set the OPENALEX_API_KEY environment variable).\n"
        )

    concept_id = None
    if args.concept:
        concept_id = resolve_concept_id(args.concept, session, api_key=args.api_key)
    args.concept_id = concept_id

    publisher_id = None
    if args.publisher:
        publisher_id = resolve_publisher_id(args.publisher, session, api_key=args.api_key)
    args.publisher_id = publisher_id

    filter_str = build_filter(args)

    params = {
        "per-page": PER_PAGE,
        "cursor": "*",
        "sort": "cited_by_count:desc",
    }
    if args.search:
        params["search"] = args.search
    if filter_str:
        params["filter"] = filter_str
    if args.api_key:
        params["api_key"] = args.api_key

    print("OpenAlex query:")
    print(f"  search : {args.search or '(none)'}")
    print(f"  filter : {filter_str or '(none)'}")
    print(f"  target : {args.limit or 'unlimited'} records")
    print()

    counters = {
        "total_fetched": 0,
        "dropped_missing_mandatory": 0,
        "dropped_short_abstract": 0,
        "dropped_duplicate": 0,
        "final_kept": 0,
    }
    seen_ids = set()
    records = []

    page = 0
    while True:
        resp = session.get(API_URL, params=params, timeout=30)
        if resp.status_code == 429:
            print("Rate limited -- backing off 5s...")
            time.sleep(5)
            continue
        if resp.status_code == 409:
            print(
                "ERROR: OpenAlex returned 409 (usually means your free daily\n"
                "API-key credit allowance is used up, or no key was supplied).\n"
                "Get/check a key at https://openalex.org/settings/api and pass\n"
                "it with --api-key."
            )
            sys.exit(1)
        resp.raise_for_status()
        payload = resp.json()

        results = payload.get("results", [])
        if not results:
            break

        page += 1
        for work in results:
            counters["total_fetched"] += 1
            record = parse_record(work)

            if any(not record[field] for field in MANDATORY_FIELDS):
                counters["dropped_missing_mandatory"] += 1
                continue
            if len(record["abstract"].split()) < MIN_ABSTRACT_WORDS:
                counters["dropped_short_abstract"] += 1
                continue
            if record["paper_id"] in seen_ids:
                counters["dropped_duplicate"] += 1
                continue
            seen_ids.add(record["paper_id"])

            records.append(record)
            counters["final_kept"] += 1

            if args.limit and counters["final_kept"] >= args.limit:
                break

        print(
            f"  page {page}: fetched {counters['total_fetched']} so far, "
            f"kept {counters['final_kept']}"
        )

        if args.limit and counters["final_kept"] >= args.limit:
            break

        next_cursor = payload.get("meta", {}).get("next_cursor")
        if not next_cursor:
            break
        params["cursor"] = next_cursor

        # Be a good API citizen even in the polite pool.
        time.sleep(0.1)

    df = pd.DataFrame(records, columns=REQUIRED_COLUMNS)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    missing_stats = df.isna().sum() + (df == "").sum()

    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("OPENALEX DATASET FETCH REPORT")
    report_lines.append("=" * 60)
    report_lines.append(f"Search query        : {args.search or '(none)'}")
    report_lines.append(f"Filter              : {filter_str or '(none)'}")
    report_lines.append(f"Records fetched from API : {counters['total_fetched']}")
    report_lines.append("")
    report_lines.append("Rows removed (by reason):")
    report_lines.append(f"  - Missing mandatory field(s)      : {counters['dropped_missing_mandatory']}")
    report_lines.append(f"  - Abstract too short (<{MIN_ABSTRACT_WORDS} words)  : {counters['dropped_short_abstract']}")
    report_lines.append(f"  - Duplicate paper_id               : {counters['dropped_duplicate']}")
    report_lines.append(f"Final cleaned dataset size: {counters['final_kept']} rows")
    report_lines.append("")
    report_lines.append("Missing value counts (in final cleaned dataset, per column):")
    for col, count in missing_stats.items():
        report_lines.append(f"  - {col}: {count}")
    report_lines.append("")
    report_lines.append("Sample records (first 3 rows):")
    report_lines.append("-" * 60)
    for i, row in df.head(3).iterrows():
        report_lines.append(f"[{i}] paper_id: {row['paper_id']}  doi: {row['doi']}")
        report_lines.append(f"    title: {row['title'][:100]}")
        report_lines.append(f"    venue: {row['journal_ref']}  citations: {row['citation_count']}")
        report_lines.append(f"    abstract (first 150 chars): {row['abstract'][:150]}...")
        report_lines.append("-" * 60)

    report_text = "\n".join(report_lines)
    print()
    print(report_text)

    report_path = args.output.parent / "preprocessing_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    return df, counters


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch papers from the OpenAlex API and write papers_cleaned.csv "
                     "(drop-in replacement for the arXiv preprocessing pipeline)."
    )
    parser.add_argument("--search", type=str, default=None,
                         help="Free-text search, e.g. 'machine learning'")
    parser.add_argument("--concept", type=str, default=None,
                         help="Plain-English OpenAlex concept name to filter by, "
                              "e.g. 'artificial intelligence' (resolved via the API)")
    parser.add_argument("--publisher", type=str, default=None,
                         help="Publisher/host-org OpenAlex ID or name fragment, "
                              "e.g. 'ieee' to restrict to IEEE-published venues")
    parser.add_argument("--from-year", type=str, default=None,
                         help="Only include works published from this year onward, e.g. 2019")
    parser.add_argument("--has-abstract", action=argparse.BooleanOptionalAction, default=True,
                         help="Only include works with an abstract (default: on). "
                              "Use --no-has-abstract to disable this filter entirely.")
    parser.add_argument("--raw-filter", type=str, default=None,
                         help="Raw OpenAlex filter string, combined with the flags above, "
                              "e.g. 'institutions.country_code:us'")
    parser.add_argument("--limit", type=int, default=20000,
                         help="Max number of records to keep (default: 20000)")
    parser.add_argument("--output", type=str, default="../processed/papers_cleaned.csv")
    parser.add_argument("--api-key", type=str, default=os.environ.get("OPENALEX_API_KEY"),
                         help="Your free OpenAlex API key (required since Feb 13, 2026 -- "
                              "get one at https://openalex.org/settings/api). Falls back to "
                              "the OPENALEX_API_KEY environment variable if not passed here.")
    parser.add_argument("--mailto", type=str, default=None,
                         help="DEPRECATED -- OpenAlex retired the mailto/polite-pool "
                              "parameter on Feb 13, 2026. Use --api-key instead. Kept here "
                              "only so old commands don't hard-fail with an unknown flag.")
    args = parser.parse_args()
    args.output = Path(args.output)

    if args.mailto and not args.api_key:
        print(
            "NOTE: --mailto is deprecated and no longer has any effect -- OpenAlex "
            "retired it on Feb 13, 2026. Use --api-key instead (get a free one at "
            "https://openalex.org/settings/api).\n"
        )

    if not args.search and not args.concept and not args.raw_filter:
        print("ERROR: provide at least one of --search, --concept, or --raw-filter "
              "so the query doesn't try to pull all ~250M OpenAlex works.")
        sys.exit(1)

    run_fetch(args)
