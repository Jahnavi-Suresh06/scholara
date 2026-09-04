"""
similarity.py

Generates a human-readable explanation for WHY the recommendation engine
considers two papers similar. A cosine similarity score alone ("0.87") means
nothing to an end user -- this module turns that score, plus the actual
text of both papers, into something a researcher can act on, e.g.:

    "Strongly related (87% similarity). Shares key topics: graph neural
    network, fraud detection, transaction."

Design choice: shared-term extraction uses TF-IDF over just the two
documents being compared (not the whole corpus) via scikit-learn's
TfidfVectorizer. This is intentionally lightweight -- it reuses a library
already in requirements.txt for other purposes, needs no pretrained model,
and runs in microseconds. It is explicitly a keyword-overlap explanation
layer sitting on top of the SBERT similarity score, not a replacement for
it -- the ranking itself is 100% SBERT + cosine similarity; this module
only explains a ranking that's already been computed.
"""

from sklearn.feature_extraction.text import TfidfVectorizer

# Similarity score bands, calibrated for L2-normalized SBERT cosine
# similarity output, where scores for genuinely related abstracts typically
# fall in the 0.4-0.9 range (unrelated text rarely scores below ~0.1 due to
# how sentence embeddings distribute in vector space).
SIMILARITY_BANDS = [
    (0.75, "Very strongly related"),
    (0.55, "Strongly related"),
    (0.35, "Moderately related"),
    (0.0, "Loosely related"),
]


def describe_score(score: float) -> str:
    """Maps a raw cosine similarity score to a qualitative label."""
    for threshold, label in SIMILARITY_BANDS:
        if score >= threshold:
            return label
    return "Loosely related"


def extract_shared_terms(text_a: str, text_b: str, top_n: int = 5):
    """
    Returns up to top_n meaningful terms/phrases that appear prominently in
    BOTH text_a and text_b, ranked by combined TF-IDF weight. Falls back to
    an empty list if the texts are too short/degenerate for TF-IDF to
    produce anything (e.g. both are just a couple of words).
    """
    if not text_a or not text_b:
        return []

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),  # capture short phrases like "graph neural"
            max_features=200,
        )
        tfidf_matrix = vectorizer.fit_transform([text_a, text_b])
    except ValueError:
        # Happens if, after stopword removal, there's no vocabulary left
        return []

    terms = vectorizer.get_feature_names_out()
    scores_a = tfidf_matrix[0].toarray().flatten()
    scores_b = tfidf_matrix[1].toarray().flatten()

    # A term only counts as "shared" if it has non-zero weight in both docs
    combined_scores = []
    for i, term in enumerate(terms):
        if scores_a[i] > 0 and scores_b[i] > 0:
            combined_scores.append((term, scores_a[i] + scores_b[i]))

    combined_scores.sort(key=lambda pair: pair[1], reverse=True)
    return [term for term, _ in combined_scores[:top_n]]


def build_paper_text(paper) -> str:
    """
    Combines a paper's title, abstract, and keywords into a single text blob
    for explanation purposes (NOT for ranking -- ranking always uses the
    precomputed SBERT embedding, see recommender.py / vector_store.py).

    Handles any of title/abstract/keywords being None or empty safely --
    this is what lets the "Why This Paper?" feature degrade gracefully
    instead of raising when a paper record is incomplete.
    """
    parts = [paper.title, paper.abstract, paper.keywords]
    return " ".join(part for part in parts if part)


def explain_search_recommendation(query_text: str, paper, score: float) -> dict:
    """
    "Why This Paper?" explanation for the SEMANTIC SEARCH feature.

    This is deliberately a SEPARATE function from explain_similarity()
    below, even though the mechanics (describe_score + extract_shared_terms)
    are the same. The two functions compare fundamentally different pairs
    of things:

        explain_similarity()          -> paper  <-> paper
                                          (Similar Papers, personalized recs)
        explain_search_recommendation() -> search query <-> paper
                                          (semantic search results)

    Keeping them separate means the wording here can say "your search"
    without changing a single word of the "Similar Papers" explanations on
    the paper detail page or the dashboard's personalized recommendations --
    both of which must keep working exactly as before.

    The candidate text is title + abstract + keywords combined (via
    build_paper_text), per the feature spec -- richer than abstract alone,
    and still just an explanation input, never a ranking input.

    Args:
        query_text: the user's raw search query
        paper: a Paper model instance (needs .title, .abstract, .keywords)
        score: the SBERT cosine similarity score already computed by the
               ranking step -- this function only formats it, never
               recalculates or overrides it

    Returns:
        {
            "score": 0.87,
            "score_percent": 87,
            "label": "Very strongly related",
            "related_topics": ["machine learning", "healthcare", ...],
            "summary": "This paper was recommended because it closely
                         matches the meaning of your search, sharing key
                         topics: machine learning, healthcare."
        }
    """
    candidate_text = build_paper_text(paper)
    label = describe_score(score)
    related_topics = extract_shared_terms(query_text, candidate_text)
    score_percent = round(max(0.0, min(1.0, score)) * 100)

    if related_topics:
        topics_str = ", ".join(related_topics[:3])
        summary = (
            f"This paper was recommended because it closely matches the "
            f"meaning of your search, sharing key topics: {topics_str}."
        )
    else:
        # Fallback per spec: never show an error/empty state, always give
        # the user a reason -- this covers missing abstract/keywords, a
        # very short query, or TF-IDF simply finding no overlapping terms
        # even though SBERT still scored the pair as semantically related.
        summary = (
            "This paper was recommended because its overall meaning is "
            "closely related to your search query."
        )

    return {
        "score": round(float(score), 4),
        "score_percent": score_percent,
        "label": label,
        "related_topics": related_topics,
        "summary": summary,
    }


def explain_similarity(query_text: str, candidate_text: str, score: float) -> dict:
    """
    Builds the full explanation payload for one recommended paper.

    Returns:
        {
            "score": 0.87,
            "score_percent": 87,
            "label": "Very strongly related",
            "shared_terms": ["graph neural network", "fraud detection", ...],
            "summary": "Very strongly related (87% similarity). Shares key
                         topics: graph neural network, fraud detection."
        }
    """
    label = describe_score(score)
    shared_terms = extract_shared_terms(query_text, candidate_text)
    score_percent = round(max(0.0, min(1.0, score)) * 100)

    if shared_terms:
        topics_str = ", ".join(shared_terms[:3])
        summary = f"{label} ({score_percent}% similarity). Shares key topics: {topics_str}."
    else:
        summary = (
            f"{label} ({score_percent}% similarity) based on the overall "
            f"meaning of the abstracts, though no single shared keyword stood out."
        )

    return {
        "score": round(float(score), 4),
        "score_percent": score_percent,
        "label": label,
        "shared_terms": shared_terms,
        "summary": summary,
    }