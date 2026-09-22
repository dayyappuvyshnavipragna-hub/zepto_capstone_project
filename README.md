# Zepto Data & AI Platform Capstone

This repository implements the three required modules in one project:

- `data_pipeline/` — scrape books.toscrape.com, clean/convert data, load SQLite, run SQL and pandas checks.
- `analytics/` — load Titanic once, perform EDA, build/evaluate classifiers, imbalance comparison, tuning, regression, and save a complete pipeline.
- `support_assistant/` — local embeddings + ChromaDB + LangGraph + FastAPI with the required deterministic `MOCK_LLM` baseline.

## Project setup

Create a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Module 1

```bash
cd data_pipeline
python pipeline.py
```

This creates `books.db` and `query_outputs.txt`.

## Run Module 2

From the project root:

```bash
cd analytics
python 01_eda.py
python 02_modeling.py
```

The first script downloads Seaborn's Titanic dataset once and saves `titanic.csv`.
The second script reads only `titanic.csv`.

## Run Module 3

From the project root:

```bash
cd support_assistant
python build_index.py
uvicorn main:app --reload
```

Then POST:

```bash
curl -X POST http://127.0.0.1:8000/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"What is the delivery fee?\"}"
```

And:

```bash
curl -X POST http://127.0.0.1:8000/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"Hello, how are you?\"}"
```

For PowerShell use `curl.exe` if needed.

## Docker

```bash
cd support_assistant
docker build -t zepto-support .
docker run -p 7860:7860 zepto-support
```

Endpoint:

```text
POST http://127.0.0.1:7860/ask
```

## Design decisions

### Data pipeline
The scraper uses `requests` and `BeautifulSoup`. The required project conversion is fixed at **1 GBP = 105.50 INR**. A normalized SQLite database uses `categories` and `books`, connected through a foreign key. Parsing failures are handled without crashing the whole pipeline.

### Analytics
The raw Titanic dataset is loaded from Seaborn exactly once by `01_eda.py`, then committed as `analytics/titanic.csv`. Modeling uses that offline CSV and fits preprocessing only on training data through a scikit-learn `ColumnTransformer`/`Pipeline`.

### Support assistant
The support corpus is chunked one document per chunk, embedded locally using `all-MiniLM-L6-v2`, and stored in ChromaDB. LangGraph routes policy questions to retrieval and unrelated questions to a direct answer. The default `MOCK_LLM` path makes no LLM network calls.

## Git workflow required by the assignment

After creating the repository:

```bash
git init
git add .
git commit -m "Initial capstone implementation"

git checkout -b feature/capstone-improvements
git add .
git commit -m "Add pipeline improvements"

git add .
git commit -m "Add documentation and validation"

git checkout main
git merge --no-ff feature/capstone-improvements -m "Merge capstone feature branch"

git log --graph --oneline --all
```

Make sure the merge is visible in Git history.
