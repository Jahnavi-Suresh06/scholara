"""
test_search.py

Covers app/services/search_service.py: filtering by title, author,
keyword, domain, and the combined general query.
"""

import pytest

from app import create_app, db
from app.models.paper import Paper
from app.services.search_service import search_papers


@pytest.fixture
def app():
    app = create_app("testing")
    with app.app_context():
        db.create_all()

        papers = [
            Paper(
                title="Graph Neural Networks for Fraud Detection",
                authors="A. Sharma, R. Gupta",
                abstract="A GNN-based approach to detecting fraudulent transactions.",
                keywords="cs.LG cs.AI",
                citation_count=50,
            ),
            Paper(
                title="Semantic Search over Scientific Literature",
                authors="P. Iyer, S. Rao",
                abstract="Sentence embeddings for scientific document retrieval.",
                keywords="cs.IR cs.CL",
                citation_count=30,
            ),
            Paper(
                title="Reinforcement Learning for Robotics",
                authors="A. Sharma, K. Nair",
                abstract="Applying RL algorithms to robotic control tasks.",
                keywords="cs.RO cs.LG",
                citation_count=10,
            ),
        ]
        db.session.add_all(papers)
        db.session.commit()

        yield app
        db.session.remove()
        db.drop_all()


def test_search_by_title(app):
    with app.app_context():
        results = search_papers(title="Graph Neural")
        assert results.total == 1
        assert results.items[0].title == "Graph Neural Networks for Fraud Detection"


def test_search_by_author_matches_multiple(app):
    with app.app_context():
        results = search_papers(author="A. Sharma")
        assert results.total == 2


def test_search_by_domain(app):
    with app.app_context():
        results = search_papers(domain="cs.LG")
        assert results.total == 2  # GNN paper and RL paper both tagged cs.LG


def test_search_by_keyword_no_match(app):
    with app.app_context():
        results = search_papers(keyword="quantum computing")
        assert results.total == 0


def test_general_query_searches_across_fields(app):
    with app.app_context():
        results = search_papers(general_query="embeddings")
        assert results.total == 1
        assert "Semantic Search" in results.items[0].title


def test_results_ordered_by_citation_count_desc(app):
    with app.app_context():
        results = search_papers(general_query="")
        # empty general_query with no other filters -> has_specific_filter is False,
        # and general_query is falsy, so no filter is applied; all 3 papers returned
        assert results.total == 3
        citation_counts = [p.citation_count for p in results.items]
        assert citation_counts == sorted(citation_counts, reverse=True)
