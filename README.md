# Document Chat

A local RAG document assistant with a Streamlit frontend, FastAPI backend, Ollama generation, Ollama embeddings, hybrid retrieval, RRF fusion, and optional cross-encoder reranking.

## Features

- Upload one or more PDF or TXT documents.
- Parse, chunk, embed, and index documents in the backend.
- Keep indexed chunks isolated per browser session.
- Ask document-grounded questions in a chat interface.
- Retrieve with dense semantic search and lexical search on every query.
- Fuse retrieval rankings with Reciprocal Rank Fusion.
- Rerank fused candidates with a local cross-encoder when available.
- Return answers grounded in retrieved chunks with source snippets and scores.
- Run locally with Python or as a two-service Docker Compose application.

## Architecture

```text
Streamlit frontend
  -> uploads files with session_id
  -> sends chat questions with the same session_id

FastAPI backend
  -> parses PDF/TXT
  -> chunks text
  -> embeds chunks with Ollama nomic-embed-text
  -> stores in-memory chunks by session_id
  -> retrieves dense + lexical candidates
  -> fuses with RRF
  -> reranks with a cross-encoder when enabled
  -> sends grounded prompt to Ollama llama3.2:3b
  -> returns answer and source metadata
```

## Why These Choices

- Streamlit keeps the frontend small and demo-friendly while still providing upload and chat primitives.
- FastAPI gives a clear backend boundary for document processing, retrieval, and model orchestration.
- Ollama keeps both generation and embeddings local, avoiding API keys for the case-study demo.
- In-memory indexing keeps scope deliberate for a local demo and avoids adding a database or vector store.
- Hybrid retrieval improves recall over semantic-only or lexical-only search without adding infrastructure.
- RRF is simple, deterministic, and easy to explain during review.
- The cross-encoder reranker is behind a small adapter so the model can be swapped later.
- Docker Compose runs only the application services and uses the host Ollama service on macOS.

## Requirements

- Python 3.11 recommended for local development.
- Docker Desktop for containerized runs.
- Ollama running on the host machine.
- Ollama models:
  - `llama3.2:3b`
  - `nomic-embed-text`

## Environment Variables

| Variable | Default | Used by |
| --- | --- | --- |
| `DOCUMENT_CHAT_API_URL` | `http://localhost:8000` locally, `http://backend:8000` in Docker Compose | frontend |
| `OLLAMA_BASE_URL` | `http://localhost:11434` locally, `http://host.docker.internal:11434` in Docker Compose | backend |
| `OLLAMA_MODEL` | `llama3.2:3b` | backend |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | backend |
| `RAG_TOP_K` | `4` | backend |
| `RAG_DENSE_TOP_K` | `20` | backend |
| `RAG_LEXICAL_TOP_K` | `20` | backend |
| `RAG_CANDIDATE_TOP_K` | `20` | backend |
| `RAG_RERANKER_MODEL` | `cross-encoder/ms-marco-TinyBERT-L-2-v2` | backend |
| `RAG_RERANKER_ENABLED` | `auto` | backend |

No API keys are required.

## Run With Docker

Start Ollama and pull the required models:

```bash
ollama serve
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

If Docker containers cannot reach Ollama on macOS, start Ollama with a host binding:

```bash
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

Start the app:

```bash
docker compose up --build
```

Open:

```text
http://localhost:8501
```

The backend is exposed at:

```text
http://localhost:8000
```

## Run Locally Without Docker

Install dependencies:

```bash
python3 -m pip install -r backend/requirements.txt
python3 -m pip install -r frontend/requirements.txt
```

Start the backend:

```bash
python3 -m uvicorn backend.app.main:app --reload
```

Start the frontend in another terminal:

```bash
python3 -m streamlit run frontend/app.py
```

Open:

```text
http://localhost:8501
```

## Test

```bash
python3 -m pytest
```

## Retrieval Evaluation

```bash
python3 backend/scripts/evaluate_retrieval.py
```

The script runs a small local Recall@k check with deterministic test embeddings.

## Project Structure

```text
backend/
  app/
    api/routes/
    schemas/
    document_index.py
    main.py
  scripts/
  tests/
  Dockerfile
frontend/
  app.py
  tests/
  Dockerfile
docker-compose.yml
```

## What I Would Improve Next

- Persist uploaded document indexes outside process memory.
- Add document deletion and reprocessing controls.
- Cache the reranker model in a Docker volume for faster first-query startup.
- Add citation highlighting against the original uploaded document.
- Expand retrieval evaluation with real documents and more query sets.
