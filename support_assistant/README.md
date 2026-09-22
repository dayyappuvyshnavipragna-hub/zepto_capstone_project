# Module 3 — Support Assistant

## 1. Build the vector index

From this directory:

```bash
python build_index.py
```

The script loads all eight required policy documents, embeds them with `all-MiniLM-L6-v2`, and stores them in the persistent ChromaDB collection `zepto_policies`.

## 2. Start the FastAPI service

The graded baseline is mock mode, so leave `MOCK_LLM` unset.

```bash
uvicorn main:app --reload
```

## 3. Example calls

Policy question:

```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d "{\"query\":\"What is the delivery fee?\"}"
```

General question:

```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d "{\"query\":\"What is your favorite color?\"}"
```

Expected response shape:

```json
{
  "answer": "Based on the retrieved context: ...",
  "sources": ["doc_01"],
  "confidence": 1.0
}
```

For the general query:

```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

The exact retrieved snippet can vary with the embedding library version, but the response schema and mock behavior remain deterministic in structure.

## Prompt skeleton

The actual prompt is in `main.py` and contains:

- ROLE
- CONTEXT
- TASK
- FORMAT
- LENGTH
- a negative constraint not to use information outside the retrieved context
- a few-shot example

The prompt is used as the extension point for `MOCK_LLM=0`.

## Architecture

```text
8 policy .txt files
       |
       v
  ingestion/chunking
       |
       v
SentenceTransformer all-MiniLM-L6-v2
       |
       v
ChromaDB: zepto_policies
       |
       v
POST /ask -> LangGraph
       |
       v
classify_intent
   |               |
policy_question  general_question
   |               |
   v               v
retrieve_and_answer  direct_answer
   |
   v
answer + sources + confidence
```

### Stage details

**Ingestion:** `build_index.py` reads `docs/*.txt`; each document is one chunk.

**Embedding:** `SentenceTransformer("all-MiniLM-L6-v2")` creates local embeddings.

**Retrieval:** `retrieve_and_answer` embeds the incoming query and retrieves the top three chunks from the `zepto_policies` ChromaDB collection using cosine similarity.

**Generation:** In the required default mode, the retrieved top chunk is placed into the deterministic mock answer template. No LLM network call is made.

**MOCK_LLM:** The default is mock mode. `classify_intent`, `retrieve_and_answer`, and `direct_answer` contain explicit mock branches. Setting `MOCK_LLM=0` is an optional extension point for real LLM generation.

## Docker

```bash
docker build -t zepto-support .
docker run -p 7860:7860 zepto-support
```

The app then serves:

```text
POST http://127.0.0.1:7860/ask
```
