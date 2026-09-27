import logging
from typing import List, Optional
import torch

from backend.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Embedding service utilizing Sentence Transformers.
    Converts code chunks and natural-language queries into normalized dense vectors.
    Completely decoupled from LLM inference.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.embedding_model
        self._model = None
        self._device = None

    def _get_device(self) -> str:
        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._device

    def load_model(self):
        """Lazily load the SentenceTransformer model onto the target device."""
        if self._model is None:
            device = self._get_device()
            logger.info(f"Loading SentenceTransformer model '{self.model_name}' on device: {device}...")
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device=device)
                logger.info(f"Successfully loaded embedding model '{self.model_name}'.")
            except Exception as e:
                logger.error(f"Failed to load embedding model '{self.model_name}': {e}")
                raise RuntimeError(f"Embedding model loading failed for '{self.model_name}': {e}") from e
        return self._model

    def embed_text(self, text: str) -> List[float]:
        """
        Generate a normalized embedding vector for a single query text.
        Normalizing embeddings allows cosine distance computation via inner product.
        """
        if not text or not text.strip():
            raise ValueError("Query text cannot be empty for embedding generation.")

        model = self.load_model()
        try:
            embedding = model.encode(
                text.strip(),
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embedding for query: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}") from e

    def embed_chunks(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate normalized embedding vectors for multiple code chunks in batches.
        Returns a list of float vectors directly consumable by ChromaDB.
        """
        if not texts:
            logger.warning("embed_chunks received empty list; returning empty embeddings.")
            return []

        logger.info(
            f"Generating embeddings for {len(texts)} chunks using '{self.model_name}' (batch_size={batch_size})..."
        )
        model = self.load_model()
        try:
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            logger.info(f"Successfully generated {len(embeddings)} embeddings.")
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Batch embedding generation failed: {e}")
            raise RuntimeError(f"Batch embedding generation failed: {e}") from e
