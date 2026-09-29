"""Shared corpus, questions and metrics for the evaluation scripts."""

from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.app.ingestion import CHUNK_OVERLAP_WORDS, CHUNK_WORDS, Chunk, parse_document  # noqa: E402
from backend.app.ollama import OllamaClient  # noqa: E402
from backend.app.retrieval import (  # noqa: E402
    DENSE_TOP_K,
    KEYWORD_TOP_K,
    MIN_DENSE_SIMILARITY,
    RERANK_CANDIDATES,
    SearchResult,
    dense_search,
    keyword_search,
    normalize,
    reciprocal_rank_fusion,
)

SCRIPTS = Path(__file__).resolve().parent
HANDBOOK = ROOT / "sample_docs" / "brightline_service_handbook.md"
CATALOG_PRODUCTS = 1700
CATALOG_QUESTIONS = 40
WORDS_PER_PAGE = 300


@dataclass
class Question:
    text: str
    expected: str  # a phrase that only the correct chunk contains
    group: str


def handbook_questions() -> list[Question]:
    items = json.loads((SCRIPTS / "eval_questions.json").read_text())
    return [Question(item["question"], item["expected"], "handbook") for item in items]


def unanswerable_questions() -> list[str]:
    return json.loads((SCRIPTS / "unanswerable_questions.json").read_text())


FAMILIES = [
    ("Aurora", "AP", "recessed office panel", "offices and meeting rooms"),
    ("Halden", "HB", "high-bay luminaire", "warehouses and production halls"),
    ("Vega", "SL", "street light", "streets and car parks"),
    ("Stadion", "FL", "floodlight", "sports fields"),
    ("Linea", "LN", "linear pendant", "open-plan offices"),
    ("Orbit", "DL", "downlight", "corridors and lobbies"),
]

ENTRY = (
    "## {family} {code}\n\n"
    "The {family} {code} is a {kind} for {application}. The {code} delivers {lumens:,} lumens at "
    "{watts} watts. Its colour temperature is {cct:,} kelvin with a colour rendering index of {cri}. "
    "The {code} housing is made of {material} and rated {ip}. The {code} weighs {weight} kilograms. "
    "Dimming options for the {code}: {dimming}. The recommended mounting height for the {code} is "
    "{low} to {high} metres. The {code} has a list price of {price} euros and ships within {days} "
    "working days. Warranty: {years} years.\n"
)

CATALOG_TEMPLATES = [
    ("How bright is the {code}?", "The {code} delivers {lumens:,} lumens"),
    ("How heavy is the {code}?", "The {code} weighs {weight} kilograms"),
    ("What does the {code} cost?", "The {code} has a list price of {price} euros"),
    ("Is the {code} protected against dust and water?", "The {code} housing is made of {material} and rated {ip}"),
    ("Which dimming protocols does the {code} support?", "Dimming options for the {code}: {dimming}"),
]


def build_catalog(products: int = CATALOG_PRODUCTS, seed: int = 7) -> tuple[str, list[Question]]:
    """A synthetic product catalogue: hundreds of near-identical entries that differ only in codes and numbers."""
    rng = random.Random(seed)
    entries, facts = [], []
    for number in range(products):
        family, prefix, kind, application = FAMILIES[number % len(FAMILIES)]
        fact = {
            "family": family,
            "kind": kind,
            "application": application,
            "code": f"{prefix}-{2000 + number}",
            "lumens": rng.randrange(1000, 60000, 100),
            "watts": rng.randrange(10, 500),
            "cct": rng.choice([2700, 3000, 4000, 5000]),
            "cri": rng.choice([70, 80, 90]),
            "material": rng.choice(["die-cast aluminium", "extruded aluminium", "polycarbonate", "steel"]),
            "ip": rng.choice(["IP20", "IP40", "IP54", "IP65", "IP66"]),
            "weight": round(rng.uniform(0.5, 30), 1),
            "dimming": rng.choice(["DALI-2", "1-10 volt", "DALI-2 and 1-10 volt", "none"]),
            "low": rng.randrange(2, 8),
            "high": rng.randrange(8, 20),
            "price": rng.randrange(40, 2500),
            "days": rng.randrange(2, 30),
            "years": rng.choice([5, 7]),
        }
        entries.append(ENTRY.format(**fact))
        facts.append(fact)

    questions = []
    for fact in rng.sample(facts, CATALOG_QUESTIONS):
        question, expected = rng.choice(CATALOG_TEMPLATES)
        questions.append(Question(question.format(**fact), expected.format(**fact), "catalogue"))
    return "# Brightline Product Catalogue\n\n" + "\n".join(entries), questions


@dataclass
class Corpus:
    chunks: list[Chunk]
    questions: list[Question]
    words: int
    embed_seconds: float


def build_corpus(
    ollama: OllamaClient,
    large: bool,
    chunk_words: int = CHUNK_WORDS,
    overlap_words: int = CHUNK_OVERLAP_WORDS,
) -> Corpus:
    """The handbook alone, or (large) the handbook plus a ~500-page catalogue of distractors."""
    documents = [(HANDBOOK.name, HANDBOOK.read_text())]
    questions = handbook_questions()
    if large:
        catalog, catalog_questions = build_catalog()
        documents.append(("catalogue.md", catalog))
        questions += catalog_questions

    chunks = [
        chunk
        for name, text in documents
        for chunk in parse_document(name, text.encode(), chunk_words, overlap_words)
    ]
    started = time.perf_counter()
    for chunk, vector in zip(chunks, ollama.embed_documents([chunk.text for chunk in chunks])):
        chunk.embedding = normalize(vector)
    words = sum(len(text.split()) for _name, text in documents)
    return Corpus(chunks, questions, words, time.perf_counter() - started)


def hybrid(
    chunks: list[Chunk],
    query: str,
    query_vector: list[float],
    min_similarity: float = MIN_DENSE_SIMILARITY,
) -> list[SearchResult]:
    return reciprocal_rank_fusion(
        [
            dense_search(chunks, query_vector, DENSE_TOP_K, min_similarity),
            keyword_search(chunks, query, KEYWORD_TOP_K),
        ]
    )[:RERANK_CANDIDATES]


def first_hit(results: list[SearchResult], expected: str) -> Optional[int]:
    for rank, result in enumerate(results, start=1):
        if expected.lower() in result.chunk.text.lower():
            return rank
    return None


def recall_and_mrr(ranks: list[Optional[int]], k: int) -> tuple[float, float]:
    hits = [rank for rank in ranks if rank is not None and rank <= k]
    return len(hits) / len(ranks), sum(1 / rank for rank in hits) / len(ranks)
