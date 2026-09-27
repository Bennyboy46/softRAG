from fastapi.testclient import TestClient

from backend.ingest import ingest_repository
from backend.hybrid_retriever import HybridRetriever
from backend.main import app
from backend.reranker import Reranker
from backend.models import ChunkMetadata, VectorSearchResult


def test_hybrid_retriever_returns_repo_scoped_results():
    _, stats = ingest_repository(local_path="tests/sample_repo")
    retriever = HybridRetriever()
    results = retriever.retrieve(repo_id=stats.repo_id, question="Where is authentication implemented?", top_k=5)

    assert len(results) > 0
    assert all(item.metadata.repository == stats.repo_id for item in results)
    assert any("auth.py" in item.metadata.file for item in results)


def test_keyword_search_handles_identifier_queries():
    _, stats = ingest_repository(local_path="tests/sample_repo")
    retriever = HybridRetriever()
    results = retriever.keyword_retriever.keyword_search(
        repo_id=stats.repo_id,
        query="AuthService",
        top_k=5,
    )

    assert len(results) > 0
    assert any("auth.py" in item.metadata.file for item in results)


def test_reranker_preserves_metadata_and_keeps_top_results():
    candidates = [
        VectorSearchResult(
            chunk_id="a",
            content="def login(): ...",
            metadata=ChunkMetadata(repository="repo", file="auth.py", language="python", type="function", name="login", start_line=10, end_line=20),
            distance=0.1,
            similarity=0.8,
        ),
        VectorSearchResult(
            chunk_id="b",
            content="def create_connection(): ...",
            metadata=ChunkMetadata(repository="repo", file="database.py", language="python", type="function", name="create_connection", start_line=4, end_line=12),
            distance=0.2,
            similarity=0.6,
        ),
    ]

    reranked = Reranker(top_k=2).rerank("Where is login implemented?", candidates, top_k=2)

    assert len(reranked) == 2
    assert reranked[0].metadata.file == "auth.py"
    assert reranked[0].metadata.name == "login"


def test_fastapi_query_debug_payload_contains_retrieval_details():
    _, stats = ingest_repository(local_path="tests/sample_repo")
    client = TestClient(app)
    response = client.post(
        "/query",
        json={"repo_id": stats.repo_id, "question": "Where is authentication implemented?", "debug": True},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["debug"] is not None
    assert payload["debug"]["retrieved_chunks_count"] >= 1
    assert payload["debug"]["repo_id"] == stats.repo_id
