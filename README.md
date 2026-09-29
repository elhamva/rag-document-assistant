# Document Chat

Upload PDF, TXT or Markdown files and ask questions about them. Answers are grounded in the uploaded documents and show the passages they came from.

> **Only Docker is needed.** `docker compose up --build` starts the frontend, the backend and Ollama, and downloads the models on the first start. No API keys.

- **Frontend:** Streamlit (upload, chat, highlighted source panels, Markdown answer artifacts, optional model comparison)
- **Backend:** FastAPI (parsing, chunking, hybrid retrieval, reranking, answer generation)
- **Models:** Ollama running locally, `llama3.2:3b` for answers and `nomic-embed-text` for embeddings. No API keys.

The architecture diagram is in [`docs/architecture.drawio`](docs/architecture.drawio). Open it with [draw.io](https://app.diagrams.net).

## Run with Docker

```bash
docker compose up --build
```

Then open http://localhost:8501 and upload a document; `sample_docs/brightline_service_handbook.md` is a good one to try.

The first start takes a few minutes: the backend image installs CPU-only PyTorch and the reranker model, and the `ollama-pull` service downloads `llama3.2:3b` and `nomic-embed-text` (about 2.3 GB) into a Docker volume. The backend waits until the download has finished. Later starts reuse the images and the volume and are ready in seconds.

Docker needs about 6 GB of memory. In the bundled container, Ollama runs on the CPU, so the first answer takes up to a minute while the model loads, and later answers take a few seconds.

### Faster: Ollama on the host (optional)

Ollama installed directly on the machine can use the GPU (Apple Silicon or NVIDIA), and answers take about a second. Install [Ollama](https://ollama.com), pull the models, and start the app with the override file, which disables the bundled Ollama:

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
docker compose -f docker-compose.yml -f docker-compose.host-ollama.yml up --build
```

On **Linux**, Ollama only listens on localhost by default, so containers cannot reach it; start it with `OLLAMA_HOST=0.0.0.0 ollama serve`. On macOS and Windows the desktop app works as is.

### Comparing two models

The chat's **Answer options** can send the same question to a second model. In the bundled setup, only downloaded models can be used, so name the second model at start-up and `ollama-pull` downloads it too. A small model keeps it usable on a CPU:

```bash
DOCUMENT_CHAT_COMPARE_MODEL=qwen2.5:3b docker compose up --build
```

A model that has not been downloaded returns a clear error in the chat ("model … not found").

## Environment variables

All have defaults, so nothing is required.

| Variable | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://ollama:11434` (Docker), `http://host.docker.internal:11434` (host override), `http://localhost:11434` (local) | Where the backend finds Ollama |
| `OLLAMA_MODEL` | `llama3.2:3b` | Model that writes the answers |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `RAG_TOP_K` | `4` | Maximum number of chunks passed to the model |
| `RAG_RERANKER_ENABLED` | `true` | Turn the cross-encoder reranker on or off |
| `RAG_RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker model (Docker build arg as well) |
| `DOCUMENT_CHAT_API_URL` | `http://localhost:8000` | Backend URL used by the frontend (set automatically in Docker) |
| `DOCUMENT_CHAT_MODEL` | `llama3.2:3b` | Primary model name shown in the frontend |
| `DOCUMENT_CHAT_COMPARE_MODEL` | empty | Optional second model: downloaded at start-up and prefilled in the comparison UI |

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
```

The evaluation scripts need Ollama and the reranker. The easiest way to run them is inside the backend image, which has both; `docker compose run` also starts Ollama. With the bundled CPU Ollama they run several times slower than the numbers below suggest, and `tune_parameters.py` takes the longest because it embeds the 500-page corpus once per chunk size:

```bash
docker compose run --rm -v ./sample_docs:/app/sample_docs backend python backend/scripts/evaluate_retrieval.py --large
docker compose run --rm -v ./sample_docs:/app/sample_docs backend python backend/scripts/evaluate_answers.py
docker compose run --rm -v ./sample_docs:/app/sample_docs backend python backend/scripts/tune_parameters.py
```

All numbers below come from these scripts with `llama3.2:3b` and `nomic-embed-text`.

### Retrieval quality

A question counts as found when a retrieved chunk contains the expected phrase. There are two question sets:

- **Handbook** (22 questions): `sample_docs/brightline_service_handbook.md`, mostly paraphrased on purpose ("Do I have to pay for delivery?" vs. "Shipping is free of charge…").
- **Catalogue** (40 questions): a generated product catalogue of about 500 pages (1,700 entries, 150,000 words) that is added to the index as a distractor. The entries are near-identical and differ only in product codes and numbers, like a real luminaire catalogue. Questions look up one product by its code ("How bright is the LN-3486?").

The handbook alone has only 10 chunks, so every strategy finds almost everything there. The large corpus is the meaningful test:

| Strategy | Handbook only (10 chunks) R@4 / MRR | Handbook in 500 pages (1,073 chunks) R@4 / MRR | Catalogue codes R@4 / MRR | Search time |
| --- | --- | --- | --- | --- |
| Dense only | 1.00 / 0.79 | 0.95 / 0.68 | 0.78 / 0.69 | 19 ms |
| BM25 only | 0.95 / 0.83 | 0.95 / 0.81 | 1.00 / 0.93 | 42 ms |
| Hybrid (RRF) | 1.00 / 0.92 | 0.95 / 0.86 | 0.97 / 0.87 | 61 ms |
| Hybrid + reranker (used) | 1.00 / 0.95 | **1.00 / 0.92** | 0.97 / 0.88 | 492 ms |

What this shows:

- Dense search alone breaks down on near-identical entries (0.78): the entries differ only in codes and numbers, which embeddings capture poorly. BM25 alone misses paraphrases. The full pipeline is the only one that finds every handbook answer inside 500 pages of distractors.
- BM25 is the best strategy for pure code lookups. The one miss of the full pipeline ("How bright is the LN-3486?") is a chunking problem: the LN-3486 entry is split across two chunks. The reranker correctly ranks the chunk with the second half of the entry first, but the lumen figure is in the other chunk, which is mostly about the previous product and drops to rank 8. Chunking by heading, one entry per chunk, would fix this.
- At 500 pages, search itself is fast. Indexing took 24 s of embedding. Per question, the reranker (about 430 ms on CPU) and the query embedding (32 ms) dominate retrieval; generating the answer takes longer than both.

### Refusals

`evaluate_answers.py` runs the full pipeline, including the chat model, on the 22 handbook questions and on 15 questions the handbook cannot answer (for example "Who is the CEO of Brightline?" or "What is the IP rating of the Aurora AP-600?").

| Refusal detection | Answerable: answered | Unanswerable: refused |
| --- | --- | --- |
| Free-text answer, check if it starts with the refusal sentence (previous version) | 22 / 22 | 7 / 15 |
| JSON with `answer` first, then `answerable` | 22 / 22 | 12 / 15 |
| JSON with `answerable` first, then `answer` (used) | 22 / 22 | **15 / 15** |

In the previous version, 7 of the 8 misses were real refusals worded differently ("The context does not contain information about…"). The app did not recognise them and showed sources under them. The model now returns a JSON object whose schema Ollama enforces while generating, so the refusal is a boolean rather than a sentence to match. Putting `answerable` first makes the model decide before it writes; with the answer first, it tends to trust what it just wrote.

I also tried letting the model name the passages it used and showing only those. That cut the sources per answer from 2.5 to 1.8, but in 3 of 22 answers it hid the passage that actually contained the answer, so all retrieved passages are shown.

### Why these numbers

| Setting | Value | Measured (500-page corpus, `tune_parameters.py`) | Would it change for other data? |
| --- | --- | --- | --- |
| Chunk size / overlap | 180 / 40 words | 100/20: R@4 1.00, 180/0: 0.92, **180/40: 0.98**, 300/60: 0.90, 500/100: 0.84. | Yes. 100 words scored one question better here because catalogue entries are about 90 words long. I kept 180 because answers in prose documents often span a paragraph, and the difference is one question out of 62. |
| Overlap | 40 words | Without overlap, recall drops from 0.98 to 0.92: facts are cut at chunk borders. With 40 words of overlap, every sentence of up to 40 words is complete in at least one chunk. | Rarely. It depends on sentence length, not on the domain. |
| `MIN_DENSE_SIMILARITY` | 0.5 | The correct chunk always scored ≥ 0.59. The closest chunk for unanswerable questions scored 0.52–0.74, so no cut-off separates the two. 0.5 costs no recall and removes dense matches for unrelated questions (20 → 15.7 per query). At 0.65, dense recall falls from 0.92 to 0.69. | Yes, it depends on the embedding model and needs to be measured again when the model changes. It is a safety floor, not the refusal mechanism. |
| `MAX_SCORE_GAP` | 8.0 | The correct chunk was never more than 4.0 below the best chunk (median 0.0). Any gap ≥ 4 keeps the same recall; 2 loses 3 points. 8 leaves a margin of two times the worst case observed. | Yes, it depends on the reranker model. Like the dense cut-off, it only drops clear noise; refusals come from the model's `answerable` flag. |

**Limits of this evaluation:** the catalogue is synthetic and all questions were written for this project, so these numbers show relative differences between options, not absolute quality on real documents.

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
4. The model answers from those chunks only, as a JSON object with an `answerable` flag (see [Refusals](#refusals)). If nothing relevant is retrieved, or the model marks the question as not answerable, the API returns a fixed refusal with no sources.

**Frontend rendering**
- Source panels highlight words from the answer inside the retrieved passages, so the cited evidence is easier to scan.
- Each answer also gets a Markdown artifact with the answer, model name and source snippets, plus a download button.
- The chat options can run the same question against two Ollama models side by side. Retrieval stays identical, so only the answer changes. The one exception is a follow-up question, where the selected model also rewrites the search query.

### Conversation-aware retrieval

Follow-up questions are rewritten as standalone search queries before retrieval. For example, after discussing Marco Stein, "Where did he study?" becomes "Where did Marco Stein study?". A question that starts a new topic should remain unchanged.

The rewrite is produced by the local chat model, so it is treated as untrusted output. Empty, multi-line or overly long rewrites are rejected. The application also checks that identifiers such as `llama3.2:3b` or `HB-150` have not been changed or removed. If validation fails or the model times out, retrieval falls back to the original question.

This adds one short model call for questions with conversation history. The fallback keeps retrieval usable when the small local model does not follow the rewriting instructions reliably.

## Decisions and alternatives I rejected

| Decision | Why | Rejected |
| --- | --- | --- |
| **Streamlit** frontend | Upload, chat and expanders are built in, so time went into retrieval quality instead of UI plumbing. | React/Next.js: nicer UI, but a second language and build toolchain for no gain in this scope. |
| **Ollama, local models** | No API keys to hand over. After the first start it runs offline, and documents never leave the machine. | OpenAI/Anthropic APIs: better answers, but they need a key and send documents to a third party. The client is one small class, so swapping is easy. |
| **Ollama in Docker by default, on the host optionally** | The brief asks for a start via Docker on the reviewer's machine, so the default needs nothing but Docker. The trade-off is CPU speed: 2–7 s per answer on a Mac, and up to a minute for the first one. The host override brings back the GPU (about 1 s per answer). | Ollama on the host by default: faster, but the app fails unless the reviewer installs Ollama and pulls the models first. |
| **In-memory index** | A few documents per session fit in memory. Even at 500 pages (1,073 chunks), brute-force dense search takes 19 ms and BM25 42 ms. | Qdrant/Chroma: persistence and scale this demo does not need, plus another container to run. |
| **Hybrid dense + BM25 with RRF** | Dense search handles paraphrases; BM25 catches exact codes and numbers. RRF needs no score calibration. The table above shows the gain. | Dense only: misses exact identifiers. Weighted score blending: needs tuning because the scores live on different scales. |
| **Cross-encoder reranker** | It reads question and chunk together, so it ranks better. It also lets the app drop clearly irrelevant chunks. | LLM-based reranking: slower and costlier per query. |
| **Structured JSON reply for refusals** | Ollama enforces the schema, so a refusal is a boolean, whatever the wording. Refusals went from 7 to 15 of 15 with no wrongly refused questions. | Matching refusal phrases: already failed 7 of 15 times. A second model call to verify the answer: doubles the latency. |
| **Optional model comparison in the UI** | It keeps one retrieval result and varies only the answer model, which is enough to compare answer quality during the demo. | Full model routing: useful in production, but unnecessary for this local case study. |
| **Hand-written pipeline** | Every step is a short function I can explain and test. | LangChain/LlamaIndex: faster to start, but more abstraction than this pipeline needs. |
| **Word-based chunks, page-aware** | Simple, predictable, and gives page-accurate citations. | Semantic or heading-based chunking: better for long structured documents, planned as a next step. |

## Limitations

- The index lives in process memory: restarting the backend clears it, and sessions are never evicted.
- Scanned PDFs need OCR, which is not included.
- `llama3.2:3b` is small. It sometimes refuses when the answer requires combining two facts, or when it is conditional. With the host override and enough memory, `OLLAMA_MODEL=qwen2.5:14b` answers better.
- If a chat request fails, the next query rewrite may use the last successful exchange rather than the failed turn.

## What I would do next

- Persist the index (SQLite or Qdrant) and evict idle sessions.
- Stream answers token by token.
- Persist uploaded source files and render PDF-native citation overlays, instead of highlighting only the retrieved source snippets.
- Build an evaluation set from real customer documents and check answer correctness, not only retrieval and refusals.
- Use structure-aware chunking that follows headings and tables. This would fix the one remaining miss in the large evaluation.

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
backend/scripts/     Evaluation: retrieval, refusals, parameter sweeps
frontend/app.py      Streamlit UI
sample_docs/         Fictional handbook for demos and evaluation
docs/                Architecture diagram
docker-compose.yml              Frontend, backend and Ollama
docker-compose.host-ollama.yml  Optional override that uses Ollama on the host
```
