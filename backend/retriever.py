import logging
from typing import List, Optional

from backend.cache import cache, make_cache_key
from backend.config import settings
from backend.embeddings import EmbeddingService
from backend.models import VectorSearchResult
from backend.vector_store import VectorStore

DEFAULT_TOP_K = settings.top_k

logger = logging.getLogger(__name__)


class Retriever:
    """
    Retrieval service responsible for embedding natural-language questions
    using the exact same SentenceTransformer model used during ingestion,
    and performing similarity search in the repository's isolated ChromaDB collection.
    """

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[VectorStore] = None,
        default_top_k: Optional[int] = None,
    ):
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store = vector_store or VectorStore()
        self.default_top_k = default_top_k or settings.top_k

    def retrieve(
        self,
        repo_id: str,
        question: str,
        top_k: Optional[int] = None,
    ) -> List[VectorSearchResult]:
        """
        Execute semantic retrieval for a user question against an isolated repository.
        """
        if not repo_id or not repo_id.strip():
            raise ValueError("Repository ID cannot be empty.")

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        k = top_k if (top_k is not None and top_k > 0) else self.default_top_k
        cache_key = make_cache_key("semantic", repo_id, question.strip(), k)
        cached = cache.get(cache_key)
        if cached is not None:
            logger.info(f"Returning cached semantic retrieval for repo '{repo_id}' question='{question}'")
            return cached

        logger.info(f"Query received: '{question}' for repository '{repo_id}' (top_k={k})")

        if not self.vector_store.is_repository_indexed(repo_id):
            logger.warning(f"Repository '{repo_id}' is not indexed or is empty in vector store.")
            return []

        query_embedding = self.embedding_service.embed_text(question)
        results = self.vector_store.search(
            repo_id=repo_id,
            query_embedding=query_embedding,
            top_k=k,
        )

        retrieved_files = [r.metadata.file for r in results]
        logger.info(
            f"Repository '{repo_id}' queried: retrieved {len(results)} chunks. "
            f"Files: {list(dict.fromkeys(retrieved_files))}"
        )

        cache.set(cache_key, results)
        return results
