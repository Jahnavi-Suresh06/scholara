"""
Read-only diagnostic for the ResearchBot "zero papers" issue.
Does not modify the database, embeddings, or any files.
"""

from app import create_app, db
from app.models.paper import Paper

app = create_app()

with app.app_context():
    print("=" * 60)
    print("1. DATABASE PAPER COUNT")
    print("=" * 60)
    total = Paper.query.count()
    print(f"Total papers in database: {total}")

    if total > 0:
        sample = Paper.query.first()
        print(f"Sample paper id={sample.id}: {sample.title!r}")
        print(f"  authors={sample.authors!r}")
        print(f"  keywords={sample.keywords!r}")
        abstract_preview = (sample.abstract or "")[:100]
        print(f"  abstract (first 100 chars)={abstract_preview!r}")

    print()
    print("=" * 60)
    print("2. KEYWORD/LEXICAL CHECK FOR 'heart failure'")
    print("=" * 60)
    from sqlalchemy import or_
    lexical_hits = Paper.query.filter(
        or_(
            Paper.title.ilike("%heart%"),
            Paper.abstract.ilike("%heart%"),
            Paper.keywords.ilike("%heart%"),
        )
    ).count()
    print(
        f"Papers containing 'heart' (title/abstract/keywords): {lexical_hits}")

    print()
    print("=" * 60)
    print("3. EMBEDDINGS / VECTOR STORE FILES")
    print("=" * 60)
    import os
    for candidate in [
        "datasets/embeddings.npy",
        "datasets/paper_ids.npy",
        "instance/embeddings.npy",
        "instance/paper_ids.npy",
        "app/ml_engine/embeddings.npy",
        "app/ml_engine/paper_ids.npy",
    ]:
        exists = os.path.exists(candidate)
        print(f"  {candidate}: {'FOUND' if exists else 'missing'}")

    print()
    print("=" * 60)
    print("4. SEMANTIC SEARCH SERVICE — DIRECT TEST")
    print("=" * 60)
    from app.services.recommendation_service import semantic_search

    for query in ["heart failure", "machine learning", "deep learning", "artificial intelligence"]:
        try:
            result = semantic_search(query, top_k=5)
            available = result.get("available")
            n_results = len(result.get("results", []))
            print(
                f"  Query: {query!r:30} -> available={available}, results={n_results}")
            if n_results > 0:
                top = result["results"][0]
                print(
                    f"    top match: {top['paper'].title!r} (score={top['score']:.3f})")
        except Exception as e:
            print(f"  Query: {query!r:30} -> ERROR: {e}")

    print()
    print("=" * 60)
    print("5. DATASET FOLDER CONTENTS")
    print("=" * 60)
    for d in ["datasets", "datasets/scripts", "instance"]:
        if os.path.isdir(d):
            print(f"  {d}/:")
            for f in sorted(os.listdir(d))[:20]:
                print(f"    - {f}")
        else:
            print(f"  {d}/: does not exist")
