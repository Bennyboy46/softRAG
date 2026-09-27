import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.api.models.Collection import Collection

from backend.config import settings
from backend.models import ChunkMetadata, CodeChunk, VectorSearchResult

logger = logging.getLogger(__name__)


def sanitize_collection_name(repo_id: str) -> str:
    """
    Sanitize repository ID into a valid ChromaDB collection name:
    - Length between 3 and 63 characters
    - Must start and end with an alphanumeric character
    - Can contain only alphanumeric characters, underscores, or hyphens
    """
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", repo_id).strip("_-")
    name = f"repo_{cleaned}"
    if len(name) > 63:
        name = name[:63].rstrip("_-")
    if len(name) < 3:
        name = f"{name}_db"
    return name.lower()


class VectorStore:
    """
    ChromaDB-backed vector database service providing strict repository isolation
    through dedicated per-repository collections.
    """

    def __init__(self, chroma_path: Optional[Path] = None):
        self.chroma_path = Path(chroma_path or settings.chroma_path)
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        try:
            self.client = chromadb.PersistentClient(path=str(self.chroma_path))
            logger.info(f"Initialized ChromaDB PersistentClient at: {self.chroma_path}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB PersistentClient: {e}")
            raise RuntimeError(f"ChromaDB initialization failure: {e}") from e

    def get_collection_name(self, repo_id: str) -> str:
        """Derive the isolated collection name for a given repository ID."""
        return sanitize_collection_name(repo_id)

    def get_or_create_collection(self, repo_id: str) -> Collection:
        """Get or create a dedicated ChromaDB collection with cosine similarity metric."""
        col_name = self.get_collection_name(repo_id)
        try:
            collection = self.client.get_or_create_collection(
                name=col_name,
                metadata={"hnsw:space": "cosine"},
            )
            return collection
        except Exception as e:
            logger.error(f"Failed to get_or_create_collection '{col_name}': {e}")
            raise RuntimeError(f"Collection error for repository '{repo_id}': {e}") from e

    def is_repository_indexed(self, repo_id: str) -> bool:
        """Check whether a repository has already been indexed with stored chunks."""
        col_name = self.get_collection_name(repo_id)
        try:
            collection = self.client.get_collection(name=col_name)
            return collection.count() > 0
        except Exception:
            return False

    def add_chunks(
        self,
        repo_id: str,
        chunks: List[CodeChunk],
        embeddings: List[List[float]],
        batch_size: int = 200,
    ) -> int:
        """
        Store code chunks, embeddings, and metadata into the repository's collection.
        Handles deduplication of chunk IDs and batch insertion.
        """
        if not repo_id or not repo_id.strip():
            raise ValueError("Repository ID cannot be empty.")

        if not chunks:
            logger.warning(f"No chunks provided for repository '{repo_id}'. Skipping store.")
            return 0

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: received {len(chunks)} chunks but {len(embeddings)} embeddings."
            )

        collection = self.get_or_create_collection(repo_id)
        logger.info(f"Indexing repository '{repo_id}' into collection '{collection.name}'...")

        # Prepare payloads with deduplication
        seen_ids = set()
        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        clean_embeddings: List[List[float]] = []

        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            cid = chunk.chunk_id
            if cid in seen_ids:
                cid = f"{cid}_{idx}"
            seen_ids.add(cid)

            ids.append(cid)
            documents.append(chunk.content)
            metadatas.append(chunk.metadata.to_chroma_metadata())
            clean_embeddings.append(emb)

        # Upsert in batches
        total_chunks = len(ids)
        for i in range(0, total_chunks, batch_size):
            end_idx = min(i + batch_size, total_chunks)
            collection.upsert(
                ids=ids[i:end_idx],
                documents=documents[i:end_idx],
                embeddings=clean_embeddings[i:end_idx],
                metadatas=metadatas[i:end_idx],
            )

        logger.info(
            f"Stored {total_chunks} documents and embeddings for repository '{repo_id}'."
        )
        return total_chunks

    def search(
        self,
        repo_id: str,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[VectorSearchResult]:
        """
        Search for the most semantically relevant code chunks in a specific repository.
        Returns top_k results sorted by cosine similarity.
        """
        if not repo_id or not repo_id.strip():
            raise ValueError("Repository ID cannot be empty for vector search.")

        if not query_embedding:
            raise ValueError("Query embedding cannot be empty.")

        col_name = self.get_collection_name(repo_id)
        try:
            collection = self.client.get_collection(name=col_name)
        except Exception:
            logger.warning(f"Collection '{col_name}' for repository '{repo_id}' does not exist.")
            return []

        doc_count = collection.count()
        if doc_count == 0:
            logger.warning(f"Collection '{col_name}' is empty.")
            return []

        effective_top_k = min(top_k, doc_count)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=effective_top_k,
            include=["documents", "metadatas", "distances"],
        )

        search_results: List[VectorSearchResult] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return []

        retrieved_ids = results["ids"][0]
        retrieved_docs = results["documents"][0] if results["documents"] else [""] * len(retrieved_ids)
        retrieved_metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(retrieved_ids)
        retrieved_distances = results["distances"][0] if results["distances"] else [0.0] * len(retrieved_ids)

        for cid, doc, meta, dist in zip(
            retrieved_ids, retrieved_docs, retrieved_metas, retrieved_distances
        ):
            parsed_meta = ChunkMetadata.from_chroma_metadata(meta)
            # Cosine similarity: 1.0 - distance
            similarity = round(max(0.0, 1.0 - dist), 4)

            search_results.append(
                VectorSearchResult(
                    chunk_id=cid,
                    content=doc,
                    metadata=parsed_meta,
                    distance=dist,
                    similarity=similarity,
                )
            )

        logger.info(
            f"Retrieved {len(search_results)} search results for repository '{repo_id}' (top_k={effective_top_k})."
        )
        return search_results

    def delete_repository(self, repo_id: str) -> bool:
        """
        Atomically delete all indexed data and vectors for a repository.
        """
        col_name = self.get_collection_name(repo_id)
        try:
            self.client.delete_collection(name=col_name)
            logger.info(f"Successfully deleted collection '{col_name}' for repository '{repo_id}'.")
            return True
        except Exception as e:
            logger.warning(f"Could not delete collection '{col_name}': {e}")
            return False

    def get_repository_stats(self, repo_id: str) -> Dict[str, Any]:
        """Retrieve chunk count and details for a repository."""
        col_name = self.get_collection_name(repo_id)
        try:
            collection = self.client.get_collection(name=col_name)
            return {
                "collection_name": col_name,
                "document_count": collection.count(),
                "exists": True,
            }
        except Exception:
            return {
                "collection_name": col_name,
                "document_count": 0,
                "exists": False,
            }
