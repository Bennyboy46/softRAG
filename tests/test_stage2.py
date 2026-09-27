from pathlib import Path
import pytest

from backend.models import ChunkMetadata, CodeChunk
from backend.vector_store import sanitize_collection_name, VectorStore
from backend.embeddings import EmbeddingService


def test_sanitize_collection_name():
    # Length test and valid character test
    name1 = sanitize_collection_name("user/repo-name.git")
    assert 3 <= len(name1) <= 63
    assert name1.startswith("repo_")
    assert "/" not in name1
    assert "." not in name1

    # Overlength test
    long_id = "a" * 100
    name2 = sanitize_collection_name(long_id)
    assert len(name2) <= 63


def test_metadata_serialization():
    meta = ChunkMetadata(
        repository="test_repo",
        file="auth.py",
        language="python",
        type="function",
        name=None,  # None must be converted to "" for ChromaDB
        start_line=10,
        end_line=25,
    )
    chroma_dict = meta.to_chroma_metadata()
    assert chroma_dict["name"] == ""
    assert isinstance(chroma_dict["start_line"], int)

    # Reconstruct from Chroma
    reconstructed = ChunkMetadata.from_chroma_metadata(chroma_dict)
    assert reconstructed.name is None
    assert reconstructed.file == "auth.py"
    assert reconstructed.start_line == 10
    assert reconstructed.end_line == 25


def test_embedding_service_validation():
    service = EmbeddingService()
    # Empty query must raise ValueError
    with pytest.raises(ValueError):
        service.embed_text("")
    with pytest.raises(ValueError):
        service.embed_text("   ")

    # Empty chunk list must return empty list without crashing
    assert service.embed_chunks([]) == []
