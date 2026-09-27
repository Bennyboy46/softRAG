import logging
from typing import Dict, List, Optional, Sequence

from backend.config import settings
from backend.keyword_retriever import KeywordRetriever
from backend.models import VectorSearchResult
from backend.retriever import Retriever
from backend.vector_store import VectorStore

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Combine semantic and keyword retrieval with score normalization and deduplication."""

    def __init__(
        self,
        semantic_retriever: Optional[Retriever] = None,
        keyword_retriever: Optional[KeywordRetriever] = None,
        vector_store: Optional[VectorStore] = None,
    ):
        self.semantic_retriever = semantic_retriever or Retriever(vector_store=vector_store or VectorStore())
        self.keyword_retriever = keyword_retriever or KeywordRetriever(vector_store=vector_store or VectorStore())
        self.vector_store = vector_store or VectorStore()

    def _normalize_scores(self, values: Sequence[float]) -> List[float]:
        if not values:
            return []
        minimum = min(values)
        maximum = max(values)
        if maximum <= minimum:
            return [1.0 for _ in values]
        return [(value - minimum) / (maximum - minimum) for value in values]

    def _merge_candidates(self, semantic_results: Sequence[VectorSearchResult], keyword_results: Sequence[VectorSearchResult]) -> List[VectorSearchResult]:
        semantic_by_id: Dict[str, VectorSearchResult] = {}
        for item in semantic_results:
            semantic_by_id[item.chunk_id] = item

        keyword_by_id: Dict[str, VectorSearchResult] = {}
        for item in keyword_results:
            keyword_by_id[item.chunk_id] = item

        all_ids = sorted(set(semantic_by_id) | set(keyword_by_id), key=lambda user_id: user_id)
        if not all_ids:
            return []

        semantic_values = [semantic_by_id.get(item_id).similarity for item_id in all_ids if item_id in semantic_by_id]
        keyword_values = [keyword_by_id.get(item_id).similarity for item_id in all_ids if item_id in keyword_by_id]
        semantic_norm = self._normalize_scores(semantic_values)
        keyword_norm = self._normalize_scores(keyword_values)

        combined: List[VectorSearchResult] = []
        semantic_lookup = {item.chunk_id: idx for idx, item in enumerate(semantic_results)}
        keyword_lookup = {item.chunk_id: idx for idx, item in enumerate(keyword_results)}

        for idx, item_id in enumerate(all_ids):
            semantic_item = semantic_by_id.get(item_id)
            keyword_item = keyword_by_id.get(item_id)

            if semantic_item is None and keyword_item is None:
                continue

            base = semantic_item or keyword_item
            if base is None:
                continue

            final = base.model_copy(deep=True)
            semantic_score = 0.0
            keyword_score = 0.0

            if semantic_item is not None:
                semantic_idx = semantic_lookup.get(item_id)
                if semantic_idx is not None and semantic_idx < len(semantic_norm):
                    semantic_score = semantic_norm[semantic_idx]
            if keyword_item is not None:
                keyword_idx = keyword_lookup.get(item_id)
                if keyword_idx is not None and keyword_idx < len(keyword_norm):
                    keyword_score = keyword_norm[keyword_idx]

            combined_score = (settings.semantic_weight * semantic_score) + (settings.keyword_weight * keyword_score)
            if semantic_item is not None and keyword_item is None:
                combined_score = settings.semantic_weight * semantic_score
            elif keyword_item is not None and semantic_item is None:
                combined_score = settings.keyword_weight * keyword_score

            final.similarity = round(combined_score, 6)
            final.distance = round(max(0.0, 1.0 - combined_score), 6)
            combined.append(final)

        combined.sort(key=lambda item: item.similarity, reverse=True)
        return combined

    def retrieve(
        self,
        repo_id: str,
        question: str,
        top_k: Optional[int] = None,
        semantic_top_k: Optional[int] = None,
        keyword_top_k: Optional[int] = None,
    ) -> List[VectorSearchResult]:
        if not repo_id or not repo_id.strip():
            raise ValueError("Repository ID cannot be empty.")
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        semantic_k = semantic_top_k or settings.semantic_top_k
        keyword_k = keyword_top_k or settings.keyword_top_k
        final_k = top_k or settings.hybrid_top_k

        logger.info(
            f"Hybrid retrieval request received for repo '{repo_id}' question='{question}' "
            f"semantic_top_k={semantic_k}, keyword_top_k={keyword_k}, final_k={final_k}"
        )

        semantic_results = self.semantic_retriever.retrieve(
            repo_id=repo_id,
            question=question,
            top_k=semantic_k,
        )
        keyword_results = self.keyword_retriever.keyword_search(
            repo_id=repo_id,
            query=question,
            top_k=keyword_k,
        )

        merged = self._merge_candidates(semantic_results, keyword_results)
        if not merged:
            logger.info(f"Hybrid retrieval returned no merged candidates for repo '{repo_id}'.")
            return []

        final_results = merged[:final_k]
        logger.info(
            f"Hybrid retrieval for repo '{repo_id}' produced {len(final_results)} final candidates."
        )
        return final_results


def hybrid_retrieve(
    repo_id: str,
    question: str,
    top_k: Optional[int] = None,
    semantic_top_k: Optional[int] = None,
    keyword_top_k: Optional[int] = None,
) -> List[VectorSearchResult]:
    """Convenience wrapper for hybrid retrieval."""
    return HybridRetriever().retrieve(
        repo_id=repo_id,
        question=question,
        top_k=top_k,
        semantic_top_k=semantic_top_k,
        keyword_top_k=keyword_top_k,
    )
