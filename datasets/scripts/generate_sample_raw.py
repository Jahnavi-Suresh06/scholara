"""
generate_sample_raw.py

Creates a small sample file in EXACTLY the same JSON-lines schema as the real
Cornell/Kaggle arXiv snapshot, so we can test preprocess_arxiv.py right now.
This is a stand-in for development only -- it is NOT used in the final app.

Intentionally includes:
- clean, valid records
- a record missing 'title'
- a record with a too-short abstract
- a duplicate paper_id
- a record with missing journal_ref/doi (realistic — most preprints have none)
"""

import json
from pathlib import Path

sample_records = [
    {
        "id": "2101.00001",
        "title": "Graph Neural Networks for Fraud Detection in Financial Transactions",
        "authors": "A. Sharma, R. Gupta",
        "abstract": (
            "Financial fraud detection is a critical problem in the banking sector. "
            "In this work we propose a graph neural network architecture that models "
            "transactions as a heterogeneous graph and learns node embeddings to "
            "distinguish fraudulent behavior from legitimate transactions. Our approach "
            "outperforms traditional rule-based systems on a large-scale real-world dataset."
        ),
        "categories": "cs.LG cs.AI",
        "versions": [{"created": "Mon, 4 Jan 2021 10:00:00 GMT"}],
        "journal-ref": "",
        "doi": "",
    },
    {
        "id": "2101.00002",
        "title": "Semantic Search over Scientific Literature using Sentence Transformers",
        "authors": "P. Iyer, S. Rao, N. Verma",
        "abstract": (
            "Traditional keyword-based search engines fail to capture the semantic "
            "meaning of research queries. We evaluate sentence embedding models for "
            "building a semantic retrieval system over a large corpus of scientific "
            "abstracts and show significant improvements in relevance over TF-IDF baselines."
        ),
        "categories": "cs.IR cs.CL",
        "versions": [{"created": "Tue, 5 Jan 2021 09:15:00 GMT"}],
        "journal-ref": "Proc. ACL 2021",
        "doi": "10.1000/example.doi.1",
    },
    {
        # Missing title -> should be dropped (mandatory field)
        "id": "2101.00003",
        "title": "",
        "authors": "Unknown Author",
        "abstract": "This paper has no title and should be dropped by the pipeline entirely.",
        "categories": "cs.LG",
        "versions": [{"created": "Wed, 6 Jan 2021 11:00:00 GMT"}],
        "journal-ref": "",
        "doi": "",
    },
    {
        # Abstract too short -> should be dropped
        "id": "2101.00004",
        "title": "Withdrawn Paper Notice",
        "authors": "J. Doe",
        "abstract": "See comments. Withdrawn.",
        "categories": "cs.LG",
        "versions": [{"created": "Thu, 7 Jan 2021 12:00:00 GMT"}],
        "journal-ref": "",
        "doi": "",
    },
    {
        "id": "2101.00005",
        "title": "A Hybrid Retrieval Approach Combining BM25 and Dense Embeddings",
        "authors": "K. Nair, V. Menon",
        "abstract": (
            "We present a hybrid retrieval system that combines the lexical precision "
            "of BM25 with the semantic understanding of dense sentence embeddings. "
            "By blending both similarity scores, our system achieves higher recall on "
            "queries with vocabulary mismatch while retaining strong performance on "
            "exact-term queries, evaluated on a benchmark of research paper abstracts."
        ),
        "categories": "cs.IR",
        "versions": [{"created": "Fri, 8 Jan 2021 14:30:00 GMT"}],
        "journal-ref": "",
        "doi": "",
    },
    {
        # Duplicate of 2101.00001 -> second occurrence should be dropped
        "id": "2101.00001",
        "title": "Graph Neural Networks for Fraud Detection in Financial Transactions",
        "authors": "A. Sharma, R. Gupta",
        "abstract": (
            "Financial fraud detection is a critical problem in the banking sector. "
            "In this work we propose a graph neural network architecture that models "
            "transactions as a heterogeneous graph and learns node embeddings to "
            "distinguish fraudulent behavior from legitimate transactions."
        ),
        "categories": "cs.LG cs.AI",
        "versions": [{"created": "Mon, 4 Jan 2021 10:00:00 GMT"}],
        "journal-ref": "",
        "doi": "",
    },
    {
        "id": "2101.00006",
        "title": "Personalized Recommendation Systems using User Interest Embeddings",
        "authors": "T. Reddy, M. Krishnan",
        "abstract": (
            "Recommender systems often rely on collaborative filtering, which suffers "
            "from cold-start problems for new users. We propose an approach that "
            "represents user interest as the centroid of embeddings of items they have "
            "previously engaged with, enabling personalized recommendations even with "
            "limited interaction history, applied here to a research paper recommendation "
            "use case."
        ),
        "categories": "cs.IR cs.LG",
        "versions": [{"created": "Sat, 9 Jan 2021 08:45:00 GMT"}],
        "journal-ref": "",
        "doi": "",
    },
]

output_path = Path(__file__).parent.parent / "raw" / "arxiv-metadata-oai-snapshot.json"
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    for record in sample_records:
        f.write(json.dumps(record) + "\n")

print(f"Sample raw file written to: {output_path}")
print(f"Total records written (including intentionally dirty ones): {len(sample_records)}")
