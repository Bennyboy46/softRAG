import logging
import math
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from backend.cache import cache, make_cache_key
from backend.config import settings
from backend.models import ChunkMetadata, VectorSearchResult
from backend.vector_store import VectorStore

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    normalized = text.lower()
    normalized = normalized.replace("/", " ")
    normalized = re.sub(r"[_-]+", " ", normalized)
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    tokens = normalized.split()
    return [token for token in tokens if token]


def _extract_identifier_terms(text: str) -> List[str]:
    terms: List[str] = []
    if not text:
        return terms
    cleaned = text.strip()
    if not cleaned:
        return terms
    matches = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", cleaned)
    terms.extend(matches)
    terms.extend([token for token in cleaned.lower().split() if token])
    return [term.lower() for term in terms if term]


def _has_exact_identifier(query_terms: Sequence[str], candidate_text: str) -> bool:
    if not query_terms:
        return False
    normalized = candidate_text.lower()
    for term in query_terms:
        if term.lower() in normalized:
            return True
    return False


def _minmax_normalize(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    minimum = min(values)
    maximum = max(values)
    if maximum <= minimum:
        return [1.0 if value > 0 else 0.0 for value in values]
    return [(value - minimum) / (maximum - minimum) for value in values]


class KeywordRetriever:
    """Simple BM25-like keyword retriever tuned for repository code chunks."""

    def __init__(self, vector_store: Optional[VectorStore] = None):
        self.vector_store = vector_store or VectorStore()

    def _fetch_repo_documents(self, repo_id: str) -> List[Tuple[str, Dict[str, Any]]]:
        collection = self.vector_store.client.get_collection(name=self.vector_store.get_collection_name(repo_id))
        raw = collection.get(include=["documents", "metadatas"], limit=collection.count())
        docs = raw.get("documents") or []
        metadatas = raw.get("metadatas") or []
        return [
            (doc or "", meta or {})
            for doc, meta in zip(docs, metadatas)
        ]

    def _compute_keyword_scores(self, query: str, docs: Sequence[Tuple[str, Dict[str, Any]]]) -> List[Tuple[str, Dict[str, Any], float]]:
        query_terms = [term for term in _extract_identifier_terms(query) if term]
        query_terms = list(dict.fromkeys(query_terms))
        if not query_terms:
            return []

        doc_tokens: List[List[str]] = []
        for content, meta in docs:
            combined = " ".join(
                [
                    content or "",
                    meta.get("file", "") or "",
                    meta.get("name", "") or "",
                    meta.get("type", "") or "",
                ]
            )
            doc_tokens.append(_tokenize(combined))

        total_docs = len(doc_tokens)
        if total_docs == 0:
            return []

        avgdl = sum(len(tokens) for tokens in doc_tokens) / total_docs
        doc_freq: Dict[str, int] = {}
        for tokens in doc_tokens:
            seen = set()
            for token in tokens:
                if token not in seen:
                    doc_freq[token] = doc_freq.get(token, 0) + 1
                    seen.add(token)

        scored: List[Tuple[str, Dict[str, Any], float]] = []
        for (content, meta), tokens in zip(docs, doc_tokens):
            counts = Counter(tokens)
            score = 0.0
            for term in query_terms:
                freq = counts.get(term, 0)
                if freq == 0:
                    continue
                df = doc_freq.get(term, 0)
                idf = math.log(((total_docs - df + 0.5) / (df + 0.5)) + 1.0)
                k1 = 1.5
                b = 0.75
                score += ((freq * (k1 + 1)) / (freq + k1 * (1 - b + b * (len(tokens) / max(avgdl, 1.0))))) * idf

                identifier_context = " ".join([
                    meta.get("file", "") or "",
                    meta.get("name", "") or "",
                    meta.get("type", "") or "",
                    content or "",
                ])
                if term.lower() in (meta.get("name") or "").lower():
                    score += 3.0
                if term.lower() in (meta.get("file") or "").lower():
                    score += 2.0
                if term.lower() in identifier_context.lower():
                    score += 1.0

            if score > 0:
                scored.append((content, meta, score))

        if not scored:
            return []

        raw_scores = [score for _, _, score in scored]
        normalized_scores = _minmax_normalize(raw_scores)
        results: List[Tuple[str, Dict[str, Any], float]] = []
        for (content, meta, raw_score), normalized in zip(scored, normalized_scores):
            results.append((content, meta, normalized))
        return results

    def keyword_search(self, repo_id: str, query: str, top_k: Optional[int] = None) -> List[VectorSearchResult]:
        if not repo_id or not repo_id.strip():
            raise ValueError("Repository ID cannot be empty.")
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        limit = top_k if top_k and top_k > 0 else settings.keyword_top_k
        cache_key = make_cache_key("keyword", repo_id, query.strip(), limit)
        cached = cache.get(cache_key)
        if cached is not None:
            logger.info(f"Returning cached keyword retrieval for repo '{repo_id}' query='{query}'")
            return cached

        try:
            collection = self.vector_store.client.get_collection(
                name=self.vector_store.get_collection_name(repo_id)
            )
            if collection.count() == 0:
                logger.warning(f"Repository '{repo_id}' has no keyword index entries.")
                return []
        except Exception:
            logger.warning(f"Repository '{repo_id}' is not indexed for keyword retrieval.")
            return []

        docs = self._fetch_repo_documents(repo_id)
        scored = self._compute_keyword_scores(query, docs)
        if not scored:
            logger.info(f"No keyword matches for query '{query}' in repository '{repo_id}'.")
            return []

        results: List[VectorSearchResult] = []
        for content, meta, score in scored:
            parsed_meta = ChunkMetadata.from_chroma_metadata(meta)
            results.append(
                VectorSearchResult(
                    chunk_id=f"{parsed_meta.repository}:{parsed_meta.file}:{parsed_meta.start_line}_{parsed_meta.end_line}",
                    content=content,
                    metadata=parsed_meta,
                    distance=round(max(0.0, 1.0 - score), 4),
                    similarity=round(score, 4),
                )
            )

        results.sort(key=lambda item: item.similarity, reverse=True)
        results = results[:limit]
        logger.info(
            f"Keyword retrieval for repo '{repo_id}' found {len(results)} results for query '{query}'."
        )
        cache.set(cache_key, results)
        return results


def keyword_search(repo_id: str, query: str, top_k: Optional[int] = None) -> List[VectorSearchResult]:
    """Convenience wrapper for keyword retrieval."""
    return KeywordRetriever().keyword_search(repo_id=repo_id, query=query, top_k=top_k)
