# Document Chat

A chat app for your own documents. You upload PDF, TXT or Markdown files, ask questions, and get answers based only on those files, with the passages they came from.

- Frontend: Streamlit
- Backend: FastAPI
- Models: Ollama, running locally (`llama3.2:3b` for answers, `nomic-embed-text` for embeddings)

There is an architecture diagram in [`docs/architecture.drawio`](docs/architecture.drawio) (open it with [draw.io](https://app.diagrams.net)).

## Running it

You only need Docker. No API keys.

```bash
docker compose up --build
```

Open http://localhost:8501 and upload a document. `sample_docs/brightline_service_handbook.md` is a good one to start with.

The first start takes a few minutes. The backend image installs CPU-only PyTorch and the reranker model, and the `ollama-pull` service downloads the two Ollama models (about 2.3 GB) into a Docker volume. The backend waits for that download. After that, starts take a few seconds.

Give Docker about 6 GB of memory. Ollama runs on the CPU inside Docker, so the first answer can take up to a minute while the model loads. After that, answers take a few seconds.

### Using Ollama on the host instead

If Ollama is installed on the machine itself, it can use the GPU and answers come back in about a second. Pull the models and start with the override file, which turns off the bundled Ollama:

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
docker compose -f docker-compose.yml -f docker-compose.host-ollama.yml up --build
```

On Linux, Ollama listens only on localhost by default, so start it with `OLLAMA_HOST=0.0.0.0 ollama serve`. On macOS and Windows the desktop app works as it is.

### Comparing two models

Under "Answer options" in the chat you can send the same question to a second model. Inside Docker only downloaded models work, so name the second model when you start and it gets downloaded too:

```bash
DOCUMENT_CHAT_COMPARE_MODEL=qwen2.5:3b docker compose up --build
```

## Environment variables

Nothing is required; everything has a default.

| Variable | Default | What it does |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://ollama:11434` in Docker, `http://localhost:11434` locally | Where the backend finds Ollama |
| `OLLAMA_MODEL` | `llama3.2:3b` | Model that writes the answers |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `RAG_TOP_K` | `4` | Max number of chunks given to the model |
| `RAG_RERANKER_ENABLED` | `true` | Turns the reranker on or off |
| `RAG_RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker model (also a Docker build arg) |
| `DOCUMENT_CHAT_API_URL` | `http://localhost:8000` | Backend URL for the frontend (set by Docker Compose) |
| `DOCUMENT_CHAT_MODEL` | `llama3.2:3b` | Model name shown in the frontend |
| `DOCUMENT_CHAT_COMPARE_MODEL` | empty | Optional second model for the comparison |

## Running without Docker

You need Python 3.9+ and Ollama with both models pulled.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn backend.app.main:app --reload     # terminal 1, repository root
cd frontend && streamlit run app.py       # terminal 2, from frontend/ so the theme loads
```

Without `sentence-transformers` installed, the backend still runs, just without the reranker.

## How it works

Uploading (`POST /documents/process`):

1. `ingestion.py` reads the text of each PDF page with pypdf, or decodes text files.
2. The text is cut into chunks of 180 words with 40 words of overlap. A chunk never crosses a page, so every source points to one page.
3. `retrieval.py` embeds the chunks with Ollama and keeps them in memory, per browser session.

The frontend always sends the full list of files, and the backend replaces the session's index with it. Files are recognised by a hash of their content, so a file that didn't change isn't embedded again. Sending an empty list clears the session.

Asking (`POST /chat`):

1. If there is a conversation, `query_rewriter.py` turns the question into a standalone search query. For example, "And for the Vega line?" after a warranty question becomes "How long is the warranty for Vega line products?". The rewrite is only used for search; the answer prompt still gets the original question and the conversation.
2. Dense search and BM25 each return up to 20 chunks, and Reciprocal Rank Fusion merges the two lists.
3. A cross-encoder rescores the candidates. Chunks that score far below the best one are dropped, and at most 4 are kept.
4. The model answers from those chunks as JSON: `{"answerable": ..., "answer": ...}`. If nothing relevant was found or the model says it can't answer, the user gets a fixed "not enough information" reply and no sources.

The query rewrite comes from a small model, so I don't trust it blindly. If it's empty, multi-line or too long, or if it drops an identifier from the question (like `HB-150`), the original question is used instead.

In the UI, the source panels highlight words from the answer inside the retrieved passages, and each answer can be downloaded as a Markdown file with its sources.

## Decisions

**Streamlit instead of React.** Upload, chat and expandable panels come built in, so I could spend the time on retrieval. React would look nicer, but it means a second language and build setup, and that didn't pay off for this scope.

**Local models with Ollama instead of OpenAI or Anthropic.** Nobody needs an API key, it works offline after the first start, and documents never leave the machine. Hosted models would give better answers. The Ollama client is one small class, so switching later is easy.

**Ollama inside Docker by default.** The app has to start on your machine with only Docker. The cost is speed: on a CPU an answer takes 2 to 7 seconds. Running Ollama on the host by default would be faster, but the app would fail unless you install Ollama and pull the models first. That's why it's an optional override.

**In-memory index instead of a vector database.** A few documents per session fit easily in memory. Even with 500 pages (1,073 chunks), dense search takes 19 ms and BM25 42 ms. Qdrant or Chroma would add persistence I don't need yet, and one more container.

**Hybrid search (dense + BM25) with RRF.** Dense search finds paraphrases, and BM25 finds exact product codes and numbers. RRF only looks at ranks, so the two score scales don't need calibrating. Weighted blending of scores would need tuning.

**Cross-encoder reranker.** It reads the question and the chunk together, so it ranks better than embeddings alone and can drop chunks that are clearly irrelevant. Reranking with the LLM would be slower.

**JSON reply with an `answerable` flag.** At first I checked whether the answer started with a refusal sentence, which caught only 7 of 15 unanswerable questions, because the model phrases refusals in many ways. Now Ollama forces the reply into a JSON schema, and the refusal is a boolean. Putting `answerable` before `answer` matters: the model decides first and then writes. With the answer first, it tends to believe what it just wrote (12 of 15 instead of 15 of 15).

**No LangChain or LlamaIndex.** The pipeline is a handful of short functions I can explain and test. A framework would have been quicker to start with, but it would hide the parts I wanted to control.

**Simple word-based chunks.** They're predictable and give page-accurate sources. Chunking by headings would be better for structured documents; it's on the list below.

## Evaluation

The scripts in `backend/scripts/` need Ollama and the reranker, so the easiest place to run them is the backend container:

```bash
docker compose run --rm -v ./sample_docs:/app/sample_docs backend python backend/scripts/evaluate_retrieval.py --large
docker compose run --rm -v ./sample_docs:/app/sample_docs backend python backend/scripts/evaluate_answers.py
docker compose run --rm -v ./sample_docs:/app/sample_docs backend python backend/scripts/tune_parameters.py
```

Unit and API tests don't need Ollama:

```bash
pytest
```

### Retrieval

I used two question sets. The handbook set has 22 questions about `sample_docs/brightline_service_handbook.md`, mostly worded differently from the text ("Do I have to pay for delivery?" vs. "Shipping is free of charge"). The catalogue set has 40 questions about a generated product catalogue of about 500 pages, where the entries are nearly identical and differ only in product codes and numbers. The catalogue is also mixed into the index as noise for the handbook questions. A question counts as found if one of the retrieved chunks contains the expected phrase.

| Strategy | Handbook alone R@4 / MRR | Handbook in 500 pages R@4 / MRR | Catalogue codes R@4 / MRR | Search time |
| --- | --- | --- | --- | --- |
| Dense only | 1.00 / 0.79 | 0.95 / 0.68 | 0.78 / 0.69 | 19 ms |
| BM25 only | 0.95 / 0.83 | 0.95 / 0.81 | 1.00 / 0.93 | 42 ms |
| Hybrid (RRF) | 1.00 / 0.92 | 0.95 / 0.86 | 0.97 / 0.87 | 61 ms |
| Hybrid + reranker (used) | 1.00 / 0.95 | 1.00 / 0.92 | 0.97 / 0.88 | 492 ms |

Dense search struggles with the catalogue because embeddings don't capture codes and numbers well. BM25 is best for exact codes but misses paraphrases. Only the full pipeline finds every handbook answer inside the 500 pages.

The one catalogue miss ("How bright is the LN-3486?") is a chunking problem. That entry is split across two chunks, and the lumen value sits in the chunk that is mostly about the previous product. Chunking by heading would fix it.

Most of the search time is the reranker (about 430 ms on CPU). Writing the answer still takes longer than the whole search.

### Refusals

`evaluate_answers.py` runs the full pipeline on the 22 handbook questions and on 15 questions the handbook can't answer, like "Who is the CEO of Brightline?".

| How refusals are detected | Answerable questions answered | Unanswerable questions refused |
| --- | --- | --- |
| Answer starts with the refusal sentence (first version) | 22 / 22 | 7 / 15 |
| JSON, `answer` first | 22 / 22 | 12 / 15 |
| JSON, `answerable` first (used) | 22 / 22 | 15 / 15 |

I also tried letting the model pick which passages it used and showing only those. It cut the sources per answer from 2.5 to 1.8, but in 3 of 22 answers it hid the passage that actually had the answer, so I show all retrieved passages.

### How I picked the parameters

These come from `tune_parameters.py` on the 500-page corpus:

- **Chunk size 180 words, overlap 40.** Recall@4 was 1.00 for 100/20, 0.98 for 180/40, 0.92 for 180 without overlap, 0.90 for 300/60 and 0.84 for 500/100. 100 words won by one question here because catalogue entries are about 90 words long. I kept 180 because in normal prose an answer often spans a whole paragraph.
- **Dense similarity floor 0.5.** The correct chunk always scored at least 0.59, so 0.5 loses nothing and filters out junk for unrelated questions. It can't be the refusal check, though: unanswerable questions still got best matches between 0.52 and 0.74. It depends on the embedding model and needs measuring again if the model changes.
- **Reranker score gap 8.0.** The correct chunk was never more than 4.0 below the best one, so 8 leaves a safe margin. It depends on the reranker model.

These numbers compare options with each other. The catalogue is synthetic and I wrote the questions myself, so they don't say how well it works on real documents.

## Limitations

- The index is kept in memory. Restarting the backend clears it, and old sessions are never removed.
- Scanned PDFs don't work, because there is no OCR.
- `llama3.2:3b` is small. It sometimes refuses when the answer needs two facts combined. A bigger model (for example `qwen2.5:14b` on the host) answers better.
- If a chat request fails, the next follow-up rewrite uses the last successful exchange.

## What I'd do next

- Store the index (SQLite or Qdrant) and clean up idle sessions.
- Stream answers token by token.
- Keep the uploaded files and highlight citations directly in the PDF.
- Build an evaluation set from real documents and check whether the answers are correct, not only retrieval and refusals.
- Chunk by headings and tables, which fixes the remaining miss above.

## Project structure

```text
backend/app/
  api/routes/        HTTP endpoints (documents, chat, health)
  schemas/           Request and response models
  ingestion.py       Parsing and chunking
  retrieval.py       In-memory index, dense search, BM25, RRF
  reranker.py        Cross-encoder reranker
  query_rewriter.py  Standalone search queries for follow-ups
  rag.py             Prompt and answer
  ollama.py          Ollama client
  config.py          Settings from environment variables
backend/scripts/     Evaluation scripts
frontend/app.py      Streamlit UI
sample_docs/         Sample handbook for the demo and evaluation
docs/                Architecture diagram
```
