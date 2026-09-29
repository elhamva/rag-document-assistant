# Document Chat

Upload PDF, TXT or Markdown files and ask questions about them. Answers are grounded in the uploaded documents and show the passages they came from.

> **Prerequisite: Ollama.** The models run in [Ollama](https://ollama.com), not in the app containers. Either install Ollama on the host (option A below, faster) or let Docker run it for you (option B, nothing to install besides Docker).

- **Frontend:** Streamlit (upload, chat, source panels)
- **Backend:** FastAPI (parsing, chunking, hybrid retrieval, reranking, answer generation)
- **Models:** Ollama running locally, `llama3.2:3b` for answers and `nomic-embed-text` for embeddings. No API keys.

The architecture diagram is in [`docs/architecture.drawio`](docs/architecture.drawio). Open it with [draw.io](https://app.diagrams.net).

## Run with Docker

### Option A: Ollama on the host (recommended)

Uses the host GPU (Apple Silicon or NVIDIA), so answers come back in seconds.

1. Install [Ollama](https://ollama.com) and pull the two models:

   ```bash
   ollama pull llama3.2:3b
   ollama pull nomic-embed-text
   ```

2. Make sure Ollama is running. On **Linux**, Ollama only listens on localhost by default, so containers cannot reach it; start it with `OLLAMA_HOST=0.0.0.0 ollama serve`. On macOS and Windows the desktop app works as is.

3. Start the app:

   ```bash
   docker compose up --build
   ```

4. Open http://localhost:8501. A sample document is in `sample_docs/` if you want something to try.

### Option B: everything in Docker

No host install. Ollama runs as a container and pulls both models (about 2.3 GB) into a named volume on the first start, so later starts are quick.

```bash
docker compose -f docker-compose.yml -f docker-compose.ollama.yml up --build
```

The backend waits until the models are downloaded, then the app is available at http://localhost:8501. Docker on macOS cannot use the GPU, so answers are noticeably slower than with option A.

The first build takes a few minutes because it installs CPU-only PyTorch and downloads the reranker model into the image.

## Environment variables

All have defaults, so nothing is required.

| Variable | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434` (Docker), `http://ollama:11434` (option B), `http://localhost:11434` (local) | Where the backend finds Ollama |
| `OLLAMA_MODEL` | `llama3.2:3b` | Model that writes the answers |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `RAG_TOP_K` | `4` | Maximum number of chunks passed to the model |
| `RAG_RERANKER_ENABLED` | `true` | Turn the cross-encoder reranker on or off |
| `RAG_RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker model (Docker build arg as well) |
| `DOCUMENT_CHAT_API_URL` | `http://localhost:8000` | Backend URL used by the frontend (set automatically in Docker) |

## Run locally without Docker

Needs Python 3.9+ and Ollama with the models above.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn backend.app.main:app --reload            # terminal 1, from the repository root
cd frontend && streamlit run app.py              # terminal 2, from frontend/ so the theme loads
```
 If `sentence-transformers` is not installed, the backend logs a warning and runs without the reranker.

## Tests and evaluation

```bash
pytest                                           # unit and API tests, no Ollama needed
python backend/scripts/evaluate_retrieval.py     # needs Ollama running
```

The evaluation asks 22 questions about `sample_docs/brightline_service_handbook.md`. A question counts as found when a retrieved chunk contains the expected phrase. Most questions are paraphrased on purpose ("Do I have to pay for delivery?" vs. "Shipping is free of charge…"), and a few use product codes such as `HB-150`.

| Strategy | Recall@4 | MRR |
| --- | --- | --- |
| Dense only | 1.00 | 0.79 |
| BM25 only | 0.95 | 0.83 |
| Hybrid (RRF) | 1.00 | 0.92 |
| Hybrid + reranker | 1.00 | 0.95 |

The document is small (10 chunks), so Recall@4 is easy. MRR is the more useful number here: hybrid search and reranking move the right chunk to the top. The next step would be a larger set of real documents.

## How it works

**Indexing** (`POST /documents/process`)
1. `ingestion.py` extracts text per PDF page (pypdf) or decodes text files.
2. The text is split into 180-word chunks with 40 words of overlap. Chunks never cross a page, so each one cites a single page.
3. `retrieval.py` embeds the chunks through Ollama and stores them in memory under the browser's `session_id`.

The frontend always sends its complete file list, and the backend replaces the session's index with it. Files are identified by a content hash, so an unchanged file is not parsed or embedded again. If embedding fails, the old index stays and the API returns 503 with Ollama's error message.

**Answering** (`POST /chat`)
1. `query_rewriter.py` turns a follow-up question into a standalone search query (see below). The answer prompt still receives the original question and conversation.
2. Dense search (cosine ≥ 0.5) and BM25 each return up to 20 chunks. Reciprocal Rank Fusion merges the two lists by rank.
3. The cross-encoder rescores the candidates. Chunks scoring far below the best one are dropped, and at most 4 are kept.
4. The model answers from those chunks only. If nothing relevant is retrieved, or the model says the answer is not in the documents, the API returns a fixed refusal with no sources.

### Conversation-aware retrieval

Follow-up questions are rewritten as standalone search queries before retrieval. For example, after discussing Marco Stein, "Where did he study?" becomes "Where did Marco Stein study?". A question that starts a new topic should remain unchanged.

The rewrite is produced by the local chat model, so it is treated as untrusted output. Empty, multi-line or overly long rewrites are rejected. The application also checks that identifiers such as `llama3.2:3b` or `HB-150` have not been changed or removed. If validation fails or the model times out, retrieval falls back to the original question.

This adds one short model call for questions with conversation history. The fallback keeps retrieval usable when the small local model does not follow the rewriting instructions reliably.

## Decisions and alternatives I rejected

| Decision | Why | Rejected |
| --- | --- | --- |
| **Streamlit** frontend | Upload, chat and expanders are built in, so time went into retrieval quality instead of UI plumbing. | React/Next.js: nicer UI, but a second language and build toolchain for no gain in this scope. |
| **Ollama, local models** | No API keys to hand over. Reviewers can run it offline, and documents never leave the machine. | OpenAI/Anthropic APIs: better answers, but they need a key and send documents to a third party. The client is one small class, so swapping is easy. |
| **Ollama on the host by default, in Docker optionally** | Host Ollama uses the GPU: about 1 s per answer on an Apple Silicon Mac. The optional compose file needs nothing but Docker. | Ollama always in Docker: a one-command start, but on macOS it runs on the CPU, at 5–7 s per answer (15 s for the first). |
| **In-memory index** | A few documents per session fit in memory. Brute-force cosine over a few hundred vectors takes milliseconds. | Qdrant/Chroma: persistence and scale this demo does not need, plus another container to run. |
| **Hybrid dense + BM25 with RRF** | Dense search handles paraphrases; BM25 catches exact codes and numbers. RRF needs no score calibration. The table above shows the gain. | Dense only: misses exact identifiers. Weighted score blending: needs tuning because the scores live on different scales. |
| **Cross-encoder reranker** | It reads question and chunk together, so it ranks better. It also lets the app drop clearly irrelevant chunks. | LLM-based reranking: slower and costlier per query. |
| **Hand-written pipeline** | Every step is a short function I can explain and test. | LangChain/LlamaIndex: faster to start, but more abstraction than this pipeline needs. |
| **Word-based chunks, page-aware** | Simple, predictable, and gives page-accurate citations. | Semantic or heading-based chunking: better for long structured documents, planned as a next step. |

## Limitations

- The index lives in process memory: restarting the backend clears it, and sessions are never evicted.
- Scanned PDFs need OCR, which is not included.
- `llama3.2:3b` is small. It sometimes refuses when the answer requires combining two facts. `OLLAMA_MODEL=qwen2.5:14b` answers better if the machine can run it.
- If a chat request fails, the next query rewrite may use the last successful exchange rather than the failed turn.

## What I would do next

- Persist the index (SQLite or Qdrant) and evict idle sessions.
- Stream answers token by token.
- Highlight the cited passage inside the original PDF.
- Build an evaluation set from real customer documents and add answer-quality checks, not only retrieval.
- Use structure-aware chunking that follows headings and tables.

## Project structure

```text
backend/app/
  api/routes/        HTTP endpoints (documents, chat, health)
  schemas/           Request and response models
  ingestion.py       Parsing and chunking
  retrieval.py       In-memory index, dense search, BM25, RRF
  reranker.py        Cross-encoder reranker
  query_rewriter.py  Standalone search queries for follow-ups
  rag.py             Retrieval call, prompt, answer
  ollama.py          Ollama client (embeddings + generation)
  config.py          Environment settings
backend/scripts/     Retrieval evaluation
frontend/app.py      Streamlit UI
sample_docs/         Fictional handbook for demos and evaluation
docs/                Architecture diagram
docker-compose.ollama.yml  Optional override that runs Ollama in Docker
```
