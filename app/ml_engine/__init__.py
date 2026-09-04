"""
app/ml_engine/

The recommendation engine, deliberately kept 100% independent of Flask and
SQLAlchemy. Every function here takes and returns plain Python types
(strings, ints, numpy arrays) -- never a Flask request, a Paper model
instance, or a database session.

Why this separation matters:
- Testability: these modules can be unit-tested with plain numpy arrays,
  no app context or database needed.
- Portability: this whole folder could be lifted into a separate
  microservice (e.g. a FastAPI inference server) without modification.
- Clear responsibility boundary: app/services/recommendation_service.py is
  the ONLY place that bridges between this package and the Flask app --
  it converts Paper model rows to plain text/ids before calling in here,
  and converts paper_ids back to Paper rows after calling out.

Modules:
    embedder.py    - wraps Sentence-Transformers, text -> vector
    vector_store.py - stores/loads/searches the embedding matrix
    similarity.py   - cosine similarity scoring + human-readable explanations
    recommender.py  - orchestrates the above into similar-papers / semantic
                      search / personalized recommendation logic
"""
