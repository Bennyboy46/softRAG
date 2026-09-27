from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.llm import build_citations, build_rag_prompt, GroqService
from backend.main import app
from backend.models import ChunkMetadata, VectorSearchResult
from backend.retriever import Retriever


@pytest.fixture
def mock_search_results():
    return [
        VectorSearchResult(
            chunk_id="test:auth.py:function:login:10_25",
            content="def login(): return True",
            metadata=ChunkMetadata(
                repository="test",
                file="auth.py",
                language="python",
                type="function",
                name="login",
                start_line=10,
                end_line=25,
            ),
            distance=0.2,
            similarity=0.8,
        ),
        VectorSearchResult(
            chunk_id="test:auth.py:function:login:10_25",  # Duplicate key test
            content="def login(): return True",
            metadata=ChunkMetadata(
                repository="test",
                file="auth.py",
                language="python",
                type="function",
                name="login",
                start_line=10,
                end_line=25,
            ),
            distance=0.2,
            similarity=0.8,
        ),
    ]


def test_build_citations_deduplication(mock_search_results):
    citations = build_citations(mock_search_results)
    # Must deduplicate the 2 identical chunk entries
    assert len(citations) == 1
    assert citations[0].file == "auth.py"
    assert citations[0].start_line == 10
    assert citations[0].end_line == 25
    assert citations[0].name == "login"


def test_build_rag_prompt(mock_search_results):
    system_prompt, user_prompt = build_rag_prompt(
        question="Where is login?",
        chunks=mock_search_results,
    )
    assert "Strict Grounding" in system_prompt
    assert "Anti-Hallucination" in system_prompt
    assert "auth.py" in user_prompt
    assert "Lines: 10-25" in user_prompt
    assert "USER QUESTION: Where is login?" in user_prompt


def test_groq_empty_chunks_guardrail():
    groq = GroqService(api_key="mock_key")
    ans = groq.generate_answer(question="Any question?", chunks=[])
    assert "I could not find relevant code in this repository" in ans


def test_fastapi_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "all-MiniLM-L6-v2" in data["embedding_model"]


def test_fastapi_query_missing_repo():
    client = TestClient(app)
    response = client.post(
        "/query",
        json={"repo_id": "non_existent_repo_12345", "question": "Where is auth?"},
    )
    assert response.status_code == 404
