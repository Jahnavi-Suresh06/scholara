"""
build_embeddings.py

Offline script that computes a Sentence-BERT embedding for every paper
currently in the database and saves them to datasets/embeddings/.

This is the ONE place in the whole project where Flask/SQLAlchemy code and
ml_engine code meet directly -- intentionally, since this is a one-off
build step, not a request-time code path. Everything downstream (the
Flask app, at request time) only ever loads the resulting .npy files
through VectorStore.load() -- it never re-runs this script per-request.

Usage (run from the project root):
    python datasets/scripts/build_embeddings.py

Re-run this any time papers are added/edited/removed (e.g. after running
ingest_to_db.py again, or after using the admin panel), since the
embeddings and the `papers` table can otherwise drift out of sync.

FIRST RUN NOTE: this downloads the 'all-MiniLM-L6-v2' model (~90MB) from
Hugging Face the first time it runs, then caches it locally
(~/.cache/torch/sentence_transformers/) for every run after that. This
requires normal internet access -- see README.md if you're running this
inside a network-restricted environment.
"""

import sys
import time
from pathlib import Path

import numpy as np

# Make the project root importable when this script is run directly
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from app import create_app, db  # noqa: E402
from app.models.paper import Paper  # noqa: E402
from app.ml_engine.embedder import get_embedder  # noqa: E402
from app.ml_engine.vector_store import VectorStore  # noqa: E402


def build_embeddings(batch_size: int = 32):
    app = create_app("development")

    with app.app_context():
        papers = Paper.query.order_by(Paper.id.asc()).all()

        if not papers:
            print("No papers found in the database. Run ingest_to_db.py first.")
            return

        print(f"Found {len(papers)} papers. Loading SBERT model...")
        embedder = get_embedder()
        print(f"Model loaded: {embedder.model_name} (dim={embedder.embedding_dim})")

        # Combine title + abstract as the embedding input. The title alone
        # is too short to carry much semantic signal, and the abstract
        # alone sometimes omits terms that are only in the title -- using
        # both gives SBERT the fullest available picture of the paper's
        # actual content.
        texts = [f"{p.title}. {p.abstract}" for p in papers]
        paper_ids = [p.id for p in papers]

        start = time.time()
        embeddings = embedder.encode(texts, batch_size=batch_size)
        elapsed = time.time() - start
        print(f"Encoded {len(texts)} papers in {elapsed:.2f}s "
              f"({len(texts) / elapsed:.1f} papers/sec)")

        store = VectorStore(embeddings, np.array(paper_ids))

        config = app.config
        embeddings_path = Path(config["EMBEDDINGS_DIR"]) / "embeddings.npy"
        ids_path = Path(config["EMBEDDINGS_DIR"]) / "paper_ids.npy"
        store.save(embeddings_path, ids_path)

        # Record each paper's row index in the saved arrays back onto the
        # Paper row itself -- this keeps the `embedding_index` column
        # (designed in Phase 4 as the SQL<->vector bridge) meaningful and
        # inspectable directly from the database, even though VectorStore
        # itself looks vectors up by paper_id (not row index) at runtime.
        for row_index, paper in enumerate(papers):
            paper.embedding_index = row_index
        db.session.commit()

        print(f"Saved embeddings to {embeddings_path}")
        print(f"Saved paper_id mapping to {ids_path}")
        print(f"Vector store size: {store.size} vectors, dimension {store.dim}")


if __name__ == "__main__":
    build_embeddings()
