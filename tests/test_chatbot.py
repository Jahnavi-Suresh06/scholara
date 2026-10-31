"""
tests/test_chatbot.py

Comprehensive test suite for ResearchBot query classification,
general knowledge generation, paper retrieval, mixed queries,
paper summarization, and edge cases.
"""

import json
import pytest
from app import create_app, db
from app.models.paper import Paper
from app.services.query_classifier import classify_query, IntentType


@pytest.fixture
def app():
    app = create_app("testing")
    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app):
    return app.test_client()


# -----------------------------------------------------------------------------
# 1. INTENT CLASSIFICATION TESTS
# -----------------------------------------------------------------------------

def test_intent_classification_general_knowledge():
    queries = [
        "What is Machine Learning?",
        "Explain Deep Learning",
        "What is NLP?",
        "What are the applications of AI?",
        "What is the difference between AI and ML?",
        "What is Machine Learning and how is it used in research?",
        "Explain supervised and unsupervised learning.",
        "How does a neural network work?",
        "What is the difference between CNN and RNN?",
    ]
    for q in queries:
        analysis = classify_query(q)
        assert analysis.intent == IntentType.GENERAL_KNOWLEDGE, f"Failed for query: {q}"


def test_intent_classification_paper_search():
    queries = [
        ("Give me five research papers on Machine Learning", 5),
        ("Find research papers related to CNN", 5),
        ("Show research papers about sentiment analysis", 5),
        ("Recommend five research papers about sentiment analysis", 5),
        ("Find research papers published on Transformer architectures", 5),
    ]
    for q, expected_count in queries:
        analysis = classify_query(q)
        assert analysis.intent == IntentType.RESEARCH_PAPER_SEARCH, f"Failed for query: {q}"
        assert analysis.paper_count == expected_count


def test_intent_classification_mixed_query():
    queries = [
        "Explain Machine Learning and give me five related papers",
        "What is Deep Learning? Also give me five research papers",
        "Explain NLP and show relevant research publications",
    ]
    for q in queries:
        analysis = classify_query(q)
        assert analysis.intent == IntentType.MIXED_QUERY, f"Failed for query: {q}"


def test_intent_classification_paper_summarization():
    queries = [
        "Summarize this research paper",
        "Explain the abstract of this paper",
        "What is the methodology used in this research?",
        "What are the main findings?",
    ]
    for q in queries:
        analysis = classify_query(q)
        assert analysis.intent == IntentType.PAPER_SUMMARIZATION, f"Failed for query: {q}"


# -----------------------------------------------------------------------------
# 2. ENDPOINT API TESTS (/api/chatbot)
# -----------------------------------------------------------------------------

def test_api_empty_input(client):
    response = client.post("/api/chatbot", json={"message": "   "})
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False
    assert "error" in data


def test_api_general_knowledge_ml(client):
    response = client.post("/api/chatbot", json={"message": "What is Machine Learning?"})
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["intent"] == IntentType.GENERAL_KNOWLEDGE
    assert "Machine Learning" in data["answer"]
    # Ordinary knowledge questions MUST NOT return research papers
    assert data["papers"] == []


def test_api_general_knowledge_deep_learning(client):
    response = client.post("/api/chatbot", json={"message": "Explain Deep Learning"})
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["intent"] == IntentType.GENERAL_KNOWLEDGE
    assert len(data["answer"]) > 50
    assert data["papers"] == []


def test_api_general_knowledge_nlp(client):
    response = client.post("/api/chatbot", json={"message": "What is NLP?"})
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["intent"] == IntentType.GENERAL_KNOWLEDGE
    assert data["papers"] == []


def test_api_general_knowledge_ai_applications(client):
    response = client.post("/api/chatbot", json={"message": "What are the applications of AI?"})
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["intent"] == IntentType.GENERAL_KNOWLEDGE
    assert "Application" in data["answer"] or "application" in data["answer"].lower()
    assert data["papers"] == []


def test_api_general_knowledge_ai_vs_ml(client):
    response = client.post("/api/chatbot", json={"message": "What is the difference between AI and ML?"})
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["intent"] == IntentType.GENERAL_KNOWLEDGE
    assert "Comparison" in data["answer"] or "difference" in data["answer"].lower()
    assert data["papers"] == []


def test_api_research_paper_search(client, app):
    # Seed a paper if db is empty (tests run with in-memory SQLite)
    with app.app_context():
        if Paper.query.count() == 0:
            p = Paper(
                title="Supervised Machine Learning: A Review of Classification Techniques",
                authors="Kotsiantis, S. B.",
                abstract="Supervised machine learning is the search for algorithms that reason from externally supplied instances.",
                publication_year=2007,
                journal_conference="Informatica",
                keywords="machine learning, classification",
                doi="10.1234/sml",
                embedding_index=0
            )
            db.session.add(p)
            db.session.commit()

    response = client.post(
        "/api/chatbot",
        json={"message": "Give me five research papers on Machine Learning"}
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["intent"] == IntentType.RESEARCH_PAPER_SEARCH
    assert isinstance(data["papers"], list)


def test_api_mixed_query(client, app):
    response = client.post(
        "/api/chatbot",
        json={"message": "Explain Machine Learning and give me five related papers"}
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["intent"] == IntentType.MIXED_QUERY
    # Must have both explanation section AND papers section
    assert "Machine Learning" in data["answer"]
    assert "Relevant Research Publications" in data["answer"] or "Relevant" in data["answer"]


def test_api_paper_summarization_with_history(client, app):
    with app.app_context():
        p = Paper.query.first()
        if not p:
            p = Paper(
                title="Deep Residual Learning for Image Recognition",
                authors="Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun",
                abstract="Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously.",
                publication_year=2016,
                journal_conference="CVPR",
                keywords="deep learning, CNN, resnet",
                doi="10.1109/CVPR.2016.90",
                embedding_index=1
            )
            db.session.add(p)
            db.session.commit()
        paper_id = p.id
        paper_title = p.title

    history = [
        {
            "role": "bot",
            "text": "Here are relevant papers:",
            "papers": [{"id": paper_id, "title": paper_title}]
        }
    ]

    response = client.post(
        "/api/chatbot",
        json={
            "message": "Summarize this research paper",
            "history": history
        }
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["intent"] == IntentType.PAPER_SUMMARIZATION
    assert "Research Objective" in data["answer"] or "Summary" in data["answer"]
