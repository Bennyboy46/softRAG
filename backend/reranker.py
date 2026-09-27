import logging
from typing import List, Optional, Sequence, Tuple

from backend.models import VectorSearchResult

logger = logging.getLogger(__name__)


class Reranker:
    """Lightweight lexical reranker for code chunks. This keeps the Stage 4 design modular and explainable."""

    def __init__(self, top_k: Optional[int] = None):
        self.top_k = top_k

    @staticmethod
    def _normalize(value: float, minimum: float, maximum: float) -> float:
        if maximum <= minimum:
            return 1.0 if value > 0 else 0.0
        return (value - minimum) / (maximum - minimum)

    def rerank(self, query: str, candidates: Sequence[VectorSearchResult], top_k: Optional[int] = None) -> List[VectorSearchResult]:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")
        if not candidates:
            return []

        query_tokens = [token.lower() for token in query.replace("/", " ").replace("_", " ").split() if token]
        if not query_tokens:
            return list(candidates[: top_k or len(candidates)])

        scored: List[Tuple[VectorSearchResult, float]] = []
        for candidate in candidates:
            meta = candidate.metadata
            haystack = " ".join([
                candidate.content or "",
                meta.file or "",
                meta.name or "",
                meta.type or "",
                meta.language or "",
            ]).lower()

            lexical_score = 0.0
            for token in query_tokens:
                if token in haystack:
                    lexical_score += 1.0
                if token in (meta.name or "").lower():
                    lexical_score += 2.0
                if token in (meta.file or "").lower():
                    lexical_score += 2.0

            combined_score = candidate.similarity + (lexical_score * 0.25)
            scored.append((candidate, combined_score))

        if not scored:
            return list(candidates[: top_k or len(candidates)])

        minimum = min(score for _, score in scored)
        maximum = max(score for _, score in scored)
        reranked: List[VectorSearchResult] = []
        for candidate, original_score in scored:
            normalized = self._normalize(original_score, minimum, maximum)
            candidate_dict = candidate.model_copy(deep=True)
            candidate_dict.similarity = round(normalized, 6)
            reranked.append(candidate_dict)

        reranked.sort(key=lambda item: item.similarity, reverse=True)
        limit = top_k if top_k and top_k > 0 else self.top_k
        if limit is not None and limit > 0:
            reranked = reranked[:limit]
        logger.info(f"Reranked {len(candidates)} candidates to {len(reranked)} for query '{query}'.")
        return reranked


def rerank(query: str, candidates: Sequence[VectorSearchResult], top_k: Optional[int] = None) -> List[VectorSearchResult]:
    """Convenience wrapper for reranking."""
    return Reranker(top_k=top_k).rerank(query=query, candidates=candidates, top_k=top_k)
