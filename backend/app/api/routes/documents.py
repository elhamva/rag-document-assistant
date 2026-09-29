from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from backend.app.dependencies import get_index
from backend.app.ingestion import Chunk, DocumentError, content_hash, parse_document
from backend.app.ollama import OllamaError
from backend.app.retrieval import DocumentIndex
from backend.app.schemas.documents import ProcessDocumentsResponse, ProcessedDocument


router = APIRouter(prefix="/documents")


@router.post("/process", response_model=ProcessDocumentsResponse)
def process_documents(
    files: list[UploadFile] = File(default=[]),  # an empty list clears the session
    session_id: str = Form("default"),
    index: DocumentIndex = Depends(get_index),
) -> ProcessDocumentsResponse:
    previous = index.documents(session_id)
    documents: dict[str, list[Chunk]] = {}
    results = []

    for upload in files:
        filename = upload.filename or "document"
        content = upload.file.read()
        document_id = content_hash(content)

        if document_id in previous:
            chunks = previous[document_id]
        else:
            try:
                chunks = parse_document(filename, content)
            except DocumentError as exc:
                results.append(
                    ProcessedDocument(filename=filename, status="failed", chunk_count=0, error=str(exc))
                )
                continue

        documents[document_id] = chunks
        results.append(ProcessedDocument(filename=filename, status="ready", chunk_count=len(chunks)))

    try:
        index.replace(session_id, documents)
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return ProcessDocumentsResponse(documents=results)
