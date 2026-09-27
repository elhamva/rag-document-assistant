from __future__ import annotations

from fastapi import APIRouter, File, Form, UploadFile

from backend.app.document_index import DocumentProcessingError, document_index, process_document
from backend.app.schemas.documents import ProcessedDocument, ProcessDocumentsResponse


router = APIRouter(prefix="/documents")


@router.post("/process", response_model=ProcessDocumentsResponse)
async def process_documents(
    files: list[UploadFile] = File(...),
    session_id: str = Form("default"),
) -> ProcessDocumentsResponse:
    indexed_chunks = []
    documents: list[ProcessedDocument] = []

    for uploaded_file in files:
        filename = uploaded_file.filename or "uploaded-file"
        content = await uploaded_file.read()
        document_id = build_document_id(
            filename=filename,
            size=len(content),
            content_type=uploaded_file.content_type,
        )

        try:
            chunks = process_document(filename, content)
        except DocumentProcessingError as exc:
            documents.append(
                ProcessedDocument(
                    id=document_id,
                    filename=filename,
                    status="Failed",
                    chunk_count=0,
                    error=str(exc),
                )
            )
            continue

        indexed_chunks.extend(chunks)
        documents.append(
            ProcessedDocument(
                id=document_id,
                filename=filename,
                status="Ready",
                chunk_count=len(chunks),
            )
        )

    document_index.replace_chunks(indexed_chunks, session_id=session_id)
    return ProcessDocumentsResponse(documents=documents)


def build_document_id(filename: str, size: int, content_type: str | None) -> str:
    return f"{filename}:{size}:{content_type or ''}"
