FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /app/backend/requirements.txt

ARG RAG_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
ENV RAG_RERANKER_MODEL=${RAG_RERANKER_MODEL}
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('${RAG_RERANKER_MODEL}')"
# The reranker is baked into the image, so runtime never needs the Hugging Face Hub.
ENV HF_HUB_OFFLINE=1

COPY backend /app/backend

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
