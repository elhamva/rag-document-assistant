# Document Chat

Document Chat is a small RAG document assistant prototype with a Streamlit frontend and a FastAPI backend.

## What is included

- Dark Streamlit interface
- Documents tab with PDF/TXT upload
- FastAPI health endpoint
- Frontend backend-health check before document processing
- Mock document processing while ingestion is not implemented yet
- Chat tab with disabled input until documents are ready
- Mock assistant response after the user asks a question
- Mock source snippets under assistant answers

## Run locally

Install dependencies from the project root:

```bash
python3 -m pip install -r frontend/requirements.txt
python3 -m pip install -r backend/requirements.txt
```

Start the backend:

```bash
python3 -m uvicorn backend.app.main:app --reload
```

In another terminal, start the frontend:

```bash
python3 -m streamlit run frontend/app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`. The frontend expects the backend at `http://localhost:8000` by default. Override that with:

```bash
DOCUMENT_CHAT_API_URL=http://localhost:8000 python3 -m streamlit run frontend/app.py
```

## Test

```bash
python3 -m pytest backend/tests frontend/tests -q
```

## Project structure

```text
backend/
  app/
    api/routes/health.py
    main.py
    schemas/health.py
  tests/
frontend/
  app.py
  tests/
  requirements.txt
  .streamlit/config.toml
docs/
  architecture.drawio
```
