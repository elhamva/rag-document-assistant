import pytest

from backend.app.ingestion import CHUNK_OVERLAP_WORDS, CHUNK_WORDS, DocumentError, parse_document, split_text


def test_split_text_keeps_short_text_in_one_chunk() -> None:
    assert split_text("one two three") == ["one two three"]
    assert split_text("   ") == []


def test_split_text_overlaps_neighbouring_chunks() -> None:
    words = [f"w{i}" for i in range(CHUNK_WORDS + 50)]

    chunks = split_text(" ".join(words))

    assert len(chunks) == 2
    assert chunks[0].split() == words[:CHUNK_WORDS]
    assert chunks[1].split()[:CHUNK_OVERLAP_WORDS] == words[CHUNK_WORDS - CHUNK_OVERLAP_WORDS : CHUNK_WORDS]
    assert chunks[1].split()[-1] == words[-1]


def test_parse_txt_document() -> None:
    chunks = parse_document("notes.txt", b"The warranty lasts five years.")

    assert len(chunks) == 1
    assert chunks[0].filename == "notes.txt"
    assert chunks[0].page is None
    assert chunks[0].text == "The warranty lasts five years."


def test_parse_pdf_keeps_page_numbers() -> None:
    chunks = parse_document("guide.pdf", build_test_pdf("RAG PDF citation text"))

    assert [(chunk.page, chunk.text) for chunk in chunks] == [(1, "RAG PDF citation text")]


def test_chunk_ids_are_unique_per_document() -> None:
    text = " ".join(f"word{i}" for i in range(500)).encode()

    chunks = parse_document("long.txt", text)

    assert len({chunk.id for chunk in chunks}) == len(chunks) > 1


@pytest.mark.parametrize(
    "filename, content, message",
    [
        ("image.png", b"not a document", "Unsupported file type"),
        ("broken.pdf", b"not a pdf", "Could not read the PDF"),
        ("empty.txt", b"  \n ", "No readable text found"),
    ],
)
def test_parse_document_rejects_unusable_files(filename: str, content: bytes, message: str) -> None:
    with pytest.raises(DocumentError, match=message):
        parse_document(filename, content)


def build_test_pdf(text: str) -> bytes:
    stream = f"BT /F1 24 Tf 72 720 Td ({text}) Tj ET".encode("utf-8")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length "
        + str(len(stream)).encode("utf-8")
        + b" >>\nstream\n"
        + stream
        + b"\nendstream",
    ]

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]

    for index, value in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{index} 0 obj\n".encode("utf-8"))
        pdf.extend(value)
        pdf.extend(b"\nendobj\n")

    startxref = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("utf-8"))
    pdf.extend(b"0000000000 65535 f \n")

    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("utf-8"))

    pdf.extend(
        (
            f"trailer\n<< /Root 1 0 R /Size {len(objects) + 1} >>\n"
            f"startxref\n{startxref}\n%%EOF\n"
        ).encode("utf-8")
    )

    return bytes(pdf)
