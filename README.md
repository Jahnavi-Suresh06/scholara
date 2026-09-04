# Scholara

An AI-powered system that recommends research papers using **semantic
understanding** (Sentence Transformers + cosine similarity) rather than
plain keyword matching.

> **Current phase:** AI Core complete (Phases 1-7) — the SBERT-based
> semantic recommendation engine is live: similar papers, semantic search,
> and personalized dashboard recommendations all work end-to-end. See
> [`docs/RECOMMENDATION_ARCHITECTURE.md`](docs/RECOMMENDATION_ARCHITECTURE.md)
> for the full pipeline explanation. Frontend polish (Phase 7 visual pass)
> and formal testing/deployment (Phases 9-10) are next.

---

## Tech Stack

- **Backend:** Python, Flask (App Factory pattern, Blueprints)
- **Database:** SQLAlchemy ORM + Flask-Migrate (SQLite for dev, PostgreSQL for prod)
- **Frontend:** HTML, Bootstrap 5
- **ML (upcoming phases):** Sentence-Transformers, scikit-learn, NumPy
- **Deployment:** Render (see `render.yaml`)

---

## Project Structure

```
research-paper-recommender/
├── app/
│   ├── __init__.py          # App factory (Flask, SQLAlchemy, Flask-Migrate, Flask-Login)
│   ├── config.py             # Dev/Test/Prod config classes
│   ├── cli.py                 # `flask create-admin` CLI command
│   ├── models/                # User, Paper, Bookmark, SearchHistory
│   ├── routes/                # Blueprints (controllers): main, auth, search, paper, bookmark, dashboard, admin
│   ├── services/               # Business logic: auth, search, bookmark, history, admin, recommendation
│   ├── ml_engine/               # SBERT embedder, vector store, similarity/explanations, recommender (Flask-independent)
│   ├── utils/                  # admin_required decorator
│   ├── static/                 # CSS/JS
│   └── templates/              # Jinja2 templates (auth/, admin/, + top-level pages)
├── datasets/
│   ├── raw/                    # Original arXiv snapshot (gitignored, download separately)
│   ├── processed/               # Cleaned CSV + preprocessing report
│   ├── embeddings/               # embeddings.npy, paper_ids.npy, evaluation_report.txt (generated)
│   └── scripts/                  # preprocess_arxiv.py, ingest_to_db.py, build_embeddings.py, evaluate_recommendations.py
├── docs/
│   └── RECOMMENDATION_ARCHITECTURE.md  # Full pipeline design + evaluation methodology
├── migrations/                    # Flask-Migrate (Alembic) migration files
├── instance/                       # SQLite dev database (gitignored)
├── tests/                           # Pytest test suite (44 tests)
├── run.py                            # App entry point
├── requirements.txt
├── render.yaml
├── .env.example
└── .gitignore
```

---

## Setup Instructions (VS Code / local machine)

### 1. Clone/extract and open in VS Code

Extract the ZIP, then open the `research-paper-recommender/` folder in VS Code
(`File > Open Folder...`).

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

```bash
# Windows
copy .env.example .env

# macOS/Linux
cp .env.example .env
```

Open `.env` and set a real `SECRET_KEY` (any long random string is fine for
local dev). Leave `DATABASE_URL` empty for local SQLite.

### 5. Initialize the database

```bash
flask db init        # only needed the very first time (creates migrations/)
flask db migrate -m "Initial migration: users, papers, bookmarks, search_history"
flask db upgrade
```

This creates `instance/app.db` with all four tables.

> **Note:** This ZIP already includes a generated `migrations/` folder with
> the initial migration applied, so `flask db init` will say it's already
> initialized — that's expected. Just run `flask db upgrade` to create your
> local `instance/app.db`.

### 6. Run the app

```bash
python run.py
```

Visit **http://127.0.0.1:5000** for the landing page.

### 7. Create an admin account (to access /admin)

```bash
flask create-admin
```
You'll be prompted for a username, email, and password interactively.

### 8. Load the dataset into the database

See "Dataset Preprocessing" below — you need to both preprocess the CSV
(Phase 5) and ingest it into the database (Phase 6) before search/browsing
will show real papers.

### 9. Run the test suite

```bash
pytest
```

All tests use an in-memory database and never touch your real `instance/app.db`.

---

## Dataset Preprocessing (Phase 5)

The `datasets/raw/` folder currently contains a small **sample** file so the
pipeline can be demonstrated without a 1GB+ download. To use the real dataset:

1. Download `arxiv-metadata-oai-snapshot.json` from
   [kaggle.com/datasets/Cornell-University/arxiv](https://www.kaggle.com/datasets/Cornell-University/arxiv)
2. Replace `datasets/raw/arxiv-metadata-oai-snapshot.json` with the downloaded file
3. Run:
   ```bash
   cd datasets/scripts
   python preprocess_arxiv.py \
     --input ../raw/arxiv-metadata-oai-snapshot.json \
     --output ../processed/papers_cleaned.csv \
     --limit 30000
   ```
4. Restart the Flask app — the landing page will now show the real row count
   from `papers_cleaned.csv`.
5. Load the cleaned data into the actual database:
   ```bash
   python datasets/scripts/ingest_to_db.py --csv datasets/processed/papers_cleaned.csv
   ```
   This populates the `papers` SQL table so search, browsing, and bookmarking
   work against real data. `embedding_index` is left NULL for every paper —
   the next step fills it in.
6. Generate the semantic embeddings (this is the AI core of the project):
   ```bash
   python datasets/scripts/build_embeddings.py
   ```
   **First run downloads the `all-MiniLM-L6-v2` model (~90MB) from Hugging
   Face** — this needs normal internet access and takes a minute or two;
   every run after that uses the local cache and is fast. This populates
   `datasets/embeddings/embeddings.npy` and `paper_ids.npy`, and updates
   each paper's `embedding_index` in the database.

   Once this has run, restart `python run.py` (or just refresh the page if
   already running) — "Similar Papers" on any paper's detail page, the
   "Semantic" search tab, and the dashboard's "Recommended for you" section
   will all switch from a "not available yet" message to real results.
7. (Optional but recommended) Evaluate the recommendation engine:
   ```bash
   python datasets/scripts/evaluate_recommendations.py --top-k 5
   ```
   Produces `datasets/embeddings/evaluation_report.txt` comparing SBERT
   against a TF-IDF baseline on category-match precision@K and latency —
   useful numbers for your project report. See
   [`docs/RECOMMENDATION_ARCHITECTURE.md`](docs/RECOMMENDATION_ARCHITECTURE.md#5-evaluation-methodology)
   for the methodology and its honest caveats.

`--limit 30000` (used in Phase 5's preprocessing step) is a recommended
starting size: enough to demonstrate a convincing recommendation system
without the multi-million-row full corpus slowing down embedding generation
in Phase 8.

---

## Feature Overview (Phase 6)

| Feature | Route(s) | Notes |
|---|---|---|
| Register / Login / Logout | `/auth/register`, `/auth/login`, `/auth/logout` | Flask-Login session management, Werkzeug password hashing |
| Profile | `/auth/profile` | View account details |
| Lexical search (title/author/keyword/domain) | `/search?mode=lexical` | Exact substring filtering |
| **Semantic search** | `/search?mode=semantic&q=...` | SBERT cosine similarity — ranks by meaning, not keywords |
| Paper detail + **Similar Papers** | `/paper/<id>` | Similar papers ranked by SBERT cosine similarity, with plain-language explanations |
| Bookmarks | `/bookmarks`, `/bookmark/<id>/add`, `/bookmark/<id>/remove` | POST-only mutations, idempotent add |
| Search history | (logged automatically on search) | Visible on the dashboard |
| Dashboard + **Recommended for you** | `/dashboard` | Personalized recommendations from the centroid of your bookmarked papers' embeddings |
| Admin panel | `/admin/papers`, `/admin/papers/new`, `/admin/papers/<id>/edit`, `/admin/papers/<id>/delete` | Requires `is_admin=True`, protected by `@admin_required` |

---

## Roadmap

| Phase | Status |
|---|---|
| 1-2: Requirement Analysis & System Design | ✅ Done |
| 3-4: Folder Structure & Database Design | ✅ Done |
| 5: Dataset Collection & Preprocessing | ✅ Done |
| 6: Backend Development (Auth, Search, Bookmarks, History, Dashboard, Admin) | ✅ Done |
| 7: AI Core — SBERT Embeddings & Recommendation Engine | ✅ Done |
| Frontend visual polish | ⬜ Next (optional) |
| 9: Testing | ⬜ |
| 10: Deployment | ⬜ |

---

## Design Notes

- **App Factory pattern** (`create_app()`) avoids circular imports and lets
  the test suite spin up isolated app instances.
- **`embedding_index`** column on `Paper` is the bridge between SQL metadata
  and the NumPy embedding array built in Phase 8 — vectors are deliberately
  kept out of the relational database.
- **Blueprints only for what currently exists** — you'll see additional
  blueprint files (auth, search, dashboard, bookmark, admin) appear in the
  Phase 6/7 delivery, not as empty stubs now.
