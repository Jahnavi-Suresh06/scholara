"""
test_recommendation_service.py

Tests the Flask/DB bridge layer. Rather than requiring the real SBERT
model (which needs internet access to Hugging Face and isn't available in
every environment this test suite might run in), we:

1. Build a small synthetic VectorStore and save it to a temp directory,
   then point the test app's EMBEDDINGS_DIR config at that temp directory.
   This fully exercises get_similar_papers() and
   get_personalized_recommendations(), which only need a VectorStore --
   they never call the embedder directly.
2. For semantic_search(), which DOES need to encode free-text into a
   vector, we monkeypatch app.ml_engine.embedder.get_embedder to return a
   fake embedder with a deterministic encode() function, so the test
   verifies the service's plumbing (calling the embedder, calling the
   store, building explanations) without needing real model weights.
"""

import numpy as np
import pytest

from app import create_app, db
from app.models.paper import Paper
from app.services.auth_service import register_user
from app.services.bookmark_service import add_bookmark
from app.ml_engine.vector_store import VectorStore
import app.services.recommendation_service as recommendation_service


def make_unit_vector(seed: int, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def app(tmp_path):
    app = create_app("testing")
    app.config["EMBEDDINGS_DIR"] = tmp_path

    with app.app_context():
        db.create_all()

        papers = [
            Paper(title=f"Paper {i}", authors="Author", abstract=f"Abstract number {i} about topic {i}.")
            for i in range(1, 6)
        ]
        db.session.add_all(papers)
        db.session.commit()

        # Build a synthetic embedding for each paper, in the same order as
        # their (auto-assigned, sequential) ids, and save to the temp
        # EMBEDDINGS_DIR so recommendation_service picks it up.
        paper_ids = np.array([p.id for p in papers])
        embeddings = np.stack([make_unit_vector(pid) for pid in paper_ids])
        store = VectorStore(embeddings, paper_ids)
        store.save(tmp_path / "embeddings.npy", tmp_path / "paper_ids.npy")

        # Reset the module-level cache between tests -- otherwise the
        # mtime-based cache could serve a VectorStore built by a previous
        # test's temp directory.
        recommendation_service._vector_store_cache["store"] = None
        recommendation_service._vector_store_cache["mtime"] = None

        yield app

        db.session.remove()
        db.drop_all()


def test_get_similar_papers_returns_results(app):
    with app.app_context():
        result = recommendation_service.get_similar_papers(1, top_k=3)
        assert result["available"] is True
        assert len(result["results"]) <= 3
        # the query paper itself should never appear in its own recommendations
        assert all(item["paper"].id != 1 for item in result["results"])
        # each result should carry a fully-formed explanation
        for item in result["results"]:
            assert "summary" in item["explanation"]
            assert 0 <= item["explanation"]["score_percent"] <= 100


def test_get_similar_papers_unavailable_when_no_embeddings(app, tmp_path):
    with app.app_context():
        # Point at an empty directory -- simulates "build_embeddings.py
        # hasn't been run yet"
        app.config["EMBEDDINGS_DIR"] = tmp_path / "does_not_exist"
        recommendation_service._vector_store_cache["store"] = None
        recommendation_service._vector_store_cache["mtime"] = None

        result = recommendation_service.get_similar_papers(1, top_k=3)
        assert result["available"] is False
        assert result["results"] == []


def test_personalized_recommendations_cold_start_no_bookmarks(app):
    with app.app_context():
        user = register_user("colduser", "cold@example.com", "password123")
        result = recommendation_service.get_personalized_recommendations(user.id, top_k=3)
        assert result["available"] is True
        assert result["cold_start"] is True
        assert result["results"] == []


def test_personalized_recommendations_with_bookmarks(app):
    with app.app_context():
        user = register_user("bookmarkeduser", "bm2@example.com", "password123")
        add_bookmark(user.id, paper_id=1)
        add_bookmark(user.id, paper_id=2)

        result = recommendation_service.get_personalized_recommendations(user.id, top_k=3)
        assert result["available"] is True
        assert result["cold_start"] is False
        recommended_ids = [item["paper"].id for item in result["results"]]
        # bookmarked papers should never recommend themselves back
        assert 1 not in recommended_ids
        assert 2 not in recommended_ids


def test_semantic_search_with_monkeypatched_embedder(app, monkeypatch):
    class FakeEmbedder:
        def encode(self, text, batch_size=32):
            # deterministic vector derived from text length, just to prove
            # the plumbing (encode -> search -> explain) works end to end
            return make_unit_vector(seed=len(text))

    monkeypatch.setattr(
        "app.services.recommendation_service.get_embedder",
        lambda: FakeEmbedder(),
    )

    with app.app_context():
        result = recommendation_service.semantic_search("some query text", top_k=3)
        assert result["available"] is True
        assert len(result["results"]) <= 3
        for item in result["results"]:
            assert "score" in item
            assert "summary" in item["explanation"]


def test_semantic_search_degrades_gracefully_when_embedder_fails(app, monkeypatch):
    """
    Regression test: if the embedder fails to load for ANY reason (model
    not installed, no internet on first download, OOM, etc.), the request
    should degrade to {"available": False}, not raise a 500. This exact bug
    was caught during manual testing when sentence-transformers wasn't
    installed in the review environment.
    """
    def raise_error():
        raise RuntimeError("Simulated embedder load failure")

    monkeypatch.setattr(
        "app.services.recommendation_service.get_embedder",
        raise_error,
    )

    with app.app_context():
        result = recommendation_service.semantic_search("some query text", top_k=3)
        assert result["available"] is False
        assert result["results"] == []
