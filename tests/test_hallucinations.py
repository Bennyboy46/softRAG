from fastapi.testclient import TestClient

from backend.ingest import ingest_repository
from backend.main import app


def test_hallucination_guardrail_for_missing_dependencies():
    _, stats = ingest_repository(local_path="tests/sample_repo")
    client = TestClient(app)

    for question in [
        "What payment gateway does this repository use?",
        "Where is the Redis cache configured?",
        "Which AWS Lambda function handles payments?",
        "What Stripe API key is used?",
    ]:
        response = client.post(
            "/query",
            json={"repo_id": stats.repo_id, "question": question},
        )

        assert response.status_code == 200, response.text
        answer = response.json()["answer"].lower()
        assert "sufficient evidence" in answer or "could not find relevant code" in answer or "not provide enough evidence" in answer
        assert "stripe" not in answer.lower()
        assert "redis" not in answer.lower()
        assert "lambda" not in answer.lower()
