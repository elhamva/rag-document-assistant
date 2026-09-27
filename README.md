# Document Chat

Document Chat is a Streamlit frontend for uploading documents and reviewing a simple chat flow.

## What is included

- Dark Streamlit interface
- Documents tab with PDF/TXT upload
- Mock document processing
- Chat tab with disabled input until documents are ready
- Mock assistant response after the user asks a question
- Mock source snippets under assistant answers

## Run locally

From the project root:

```bash
python3 -m pip install -r frontend/requirements.txt
python3 -m streamlit run frontend/app.py
```

Streamlit will print a local URL, usually:

```text
http://localhost:8501
```

## Project structure

```text
frontend/
  app.py
  requirements.txt
  .streamlit/config.toml
docs/
  architecture.drawio
```

## Backend

Backend details will be added later.
