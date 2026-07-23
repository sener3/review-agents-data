from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, List

from dotenv import load_dotenv

from schemas import CorpusContext, CorpusDocument, ParsedManuscript

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

TOKEN_RE = re.compile(r"[A-Za-z0-9]{3,}")
STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "this",
    "that",
    "from",
    "into",
    "their",
    "have",
    "been",
    "were",
    "which",
    "using",
    "used",
    "between",
    "also",
    "than",
    "such",
    "paper",
    "study",
    "results",
    "method",
    "methods",
    "approach",
    "based",
    "data",
}


def _tokenize(text: str) -> list[str]:
    return [
        t.lower() for t in TOKEN_RE.findall(text or "") if t.lower() not in STOPWORDS
    ]


def _first_non_empty(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _coerce_authors(payload: dict[str, Any]) -> list[str]:
    authors = payload.get("authors") or payload.get("author") or []
    if isinstance(authors, str):
        authors = re.split(r"\s*[,;]\s*", authors)
    if isinstance(authors, list):
        return [str(x).strip() for x in authors if str(x).strip()]

    metadata = payload.get("pdf_metadata") or {}
    author_meta = metadata.get("/Author") or metadata.get("author")
    if isinstance(author_meta, str) and author_meta.strip():
        return [x.strip() for x in re.split(r"\s*[,;]\s*", author_meta) if x.strip()]
    return []


def _coerce_year(payload: dict[str, Any]) -> int | None:
    candidates = [payload.get("year")]
    metadata = payload.get("pdf_metadata") or {}
    candidates.extend([metadata.get("year"), metadata.get("/CreationDate")])
    for value in candidates:
        if value is None:
            continue
        match = re.search(r"(19|20)\d{2}", str(value))
        if match:
            try:
                return int(match.group(0))
            except ValueError:
                continue
    return None


def _coerce_venue(payload: dict[str, Any]) -> str | None:
    metadata = payload.get("pdf_metadata") or {}
    venue = _first_non_empty(
        payload.get("venue"),
        payload.get("journal"),
        metadata.get("journal"),
        metadata.get("/Subject"),
        metadata.get("/Producer"),
    )
    return venue or None


def _extract_hit_items(raw: Any) -> list[Any]:
    if raw is None:
        return []
    for attr in ("points", "result"):
        value = getattr(raw, attr, None)
        if isinstance(value, list):
            return value
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ("points", "result"):
            value = raw.get(key)
            if isinstance(value, list):
                return value
    return []


class QdrantCollectionRetriever:
    def __init__(
        self, *, collection_name: str, source_prefix: str, source_label: str
    ) -> None:
        try:
            from qdrant_client import QdrantClient
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "qdrant_client is required for the active curated runtime path. Install it before running reviews."
            ) from exc

        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers is required for the active curated runtime path. Install it before running reviews."
            ) from exc

        qdrant_url = os.getenv("QDRANT_ENDPOINT") or os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        if not qdrant_url or not qdrant_api_key:
            raise RuntimeError(
                "QDRANT_ENDPOINT/QDRANT_URL and QDRANT_API_KEY must be set. The runtime now requires external corpora."
            )

        self.collection_name = collection_name
        self.source_prefix = source_prefix
        self.source_label = source_label
        self.embedding_model_name = os.getenv(
            "RAG_EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2"
        )
        self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.embedder = SentenceTransformer(self.embedding_model_name, device="cpu")

        collection_exists = getattr(self.client, "collection_exists", None)
        if callable(collection_exists) and not collection_exists(self.collection_name):
            raise RuntimeError(
                f"Required Qdrant collection '{self.collection_name}' does not exist. Populate it before running the review pipeline."
            )

    def _search_hits(self, query: str, top_k: int) -> list[Any]:
        query_vector = self.embedder.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).tolist()

        if hasattr(self.client, "query_points"):
            raw = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                with_payload=True,
            )
            hits = _extract_hit_items(raw)
            if hits:
                return hits

        if hasattr(self.client, "search"):
            return self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True,
            )

        raise RuntimeError(
            "The installed qdrant_client version does not expose query_points or search."
        )

    def search(self, query: str, top_k: int = 6) -> CorpusContext:
        hits = self._search_hits(query=query, top_k=top_k)
        docs: List[CorpusDocument] = []

        for idx, hit in enumerate(hits):
            payload = getattr(hit, "payload", None)
            if payload is None and isinstance(hit, dict):
                payload = hit.get("payload", {})
            payload = payload or {}

            score = getattr(hit, "score", None)
            if score is None and isinstance(hit, dict):
                score = hit.get("score", 0.0)
            similarity_score = round(float(score or 0.0), 4)

            title = _first_non_empty(
                payload.get("title"), payload.get("file_name"), f"Document {idx + 1}"
            )
            chunk_text = _first_non_empty(
                payload.get("chunk_text"),
                payload.get("abstract"),
                payload.get("snippet"),
                payload.get("text"),
            )
            file_path = _first_non_empty(
                payload.get("file_path"),
                payload.get("file_name"),
                payload.get("doc_id"),
            )
            chunk_index = payload.get("chunk_index")
            section_heading = payload.get("section_heading")
            source_trace = file_path
            if chunk_index is not None:
                source_trace = f"{source_trace}#chunk-{chunk_index}"
            if section_heading:
                source_trace = f"{source_trace}::{section_heading}"

            docs.append(
                CorpusDocument(
                    document_id=str(
                        payload.get("doc_id") or payload.get("file_name") or idx
                    ),
                    title=title,
                    authors=_coerce_authors(payload),
                    venue=_coerce_venue(payload),
                    year=_coerce_year(payload),
                    abstract_or_snippet=chunk_text,
                    similarity_score=similarity_score,
                    source=f"{self.source_prefix}:{self.collection_name}",
                    source_trace=source_trace,
                )
            )

        limitations = [
            f"Retrieved from the {self.source_label} in Qdrant collection '{self.collection_name}'.",
            "Returned passages are chunk-level excerpts and may omit relevant context outside the retrieved chunk.",
        ]
        if self.source_prefix.startswith("external_q1q2"):
            limitations.append(
                "Absence of a close match in the literature corpus does not imply invalidity; it can reflect conceptual divergence, sparse coverage, or indexing gaps."
            )
        else:
            limitations.append(
                "Guideline retrieval reflects the indexed reviewer guidance corpus and should complement, not replace, manuscript-specific evidentiary analysis."
            )
        return CorpusContext(query=query, top_documents=docs, limitations=limitations)


@lru_cache(maxsize=1)
def _get_literature_retriever() -> QdrantCollectionRetriever:
    return QdrantCollectionRetriever(
        collection_name=os.getenv(
            "QDRANT_Q1Q2_COLLECTION", "q1q2_openalex_pdf_articles"
        ),
        source_prefix="external_q1q2_qdrant",
        source_label="external curated Q1/Q2 literature corpus",
    )


@lru_cache(maxsize=1)
def _get_guideline_retriever() -> QdrantCollectionRetriever:
    return QdrantCollectionRetriever(
        collection_name=os.getenv("QDRANT_GUIDELINES_COLLECTION", "guideline_chunks"),
        source_prefix="review_guidelines_qdrant",
        source_label="external reviewer-guidelines corpus",
    )


def build_literature_corpus_context(
    manuscript: ParsedManuscript,
    query: str,
    top_k: int = 6,
) -> CorpusContext:
    manuscript_title = (
        manuscript.metadata.title or manuscript.name or "untitled manuscript"
    )
    query_with_anchor = f"{manuscript_title}. {query}".strip()
    return _get_literature_retriever().search(query=query_with_anchor, top_k=top_k)


def build_guideline_corpus_context(
    manuscript: ParsedManuscript,
    query: str,
    top_k: int = 4,
) -> CorpusContext:
    manuscript_title = (
        manuscript.metadata.title or manuscript.name or "untitled manuscript"
    )
    guideline_query = (
        f"peer review guidelines reviewer best practices manuscript evaluation {query}. {manuscript_title}"
    ).strip()
    return _get_guideline_retriever().search(query=guideline_query, top_k=top_k)


def merge_corpus_contexts(
    query: str, *contexts: CorpusContext, max_documents: int = 10
) -> CorpusContext:
    docs: List[CorpusDocument] = []
    limitations: List[str] = []
    seen_traces: set[str] = set()

    for context in contexts:
        limitations.extend(context.limitations)
        for doc in context.top_documents:
            trace_key = doc.source_trace or f"{doc.source}:{doc.document_id}"
            if trace_key in seen_traces:
                continue
            docs.append(doc)
            seen_traces.add(trace_key)

    docs.sort(key=lambda item: item.similarity_score, reverse=True)
    merged_limitations = [
        "This reviewer received dual retrieval context: external Q1/Q2 literature plus reviewer-guideline evidence.",
        *dict.fromkeys(limitations),
    ]
    return CorpusContext(
        query=query, top_documents=docs[:max_documents], limitations=merged_limitations
    )


# Backward-compatible function name retained so existing imports keep working.
def build_reference_corpus_context(
    manuscript: ParsedManuscript,
    query: str,
    top_k: int = 6,
) -> CorpusContext:
    literature_context = build_literature_corpus_context(
        manuscript, query=query, top_k=top_k
    )
    guideline_top_k = int(os.getenv("QDRANT_GUIDELINES_TOP_K", "4"))
    guideline_context = build_guideline_corpus_context(
        manuscript, query=query, top_k=guideline_top_k
    )
    return merge_corpus_contexts(
        query,
        literature_context,
        guideline_context,
        max_documents=max(top_k + guideline_top_k, top_k),
    )
