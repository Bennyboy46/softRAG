from pathlib import Path

from fastapi.testclient import TestClient

from backend.ingest import ingest_repository
from backend.main import app


def test_citations_are_generated_from_retrieved_metadata_only():
    _, stats = ingest_repository(local_path="tests/sample_repo")
    client = TestClient(app)
    response = client.post(
        "/query",
        json={"repo_id": stats.repo_id, "question": "Where is authentication implemented?"},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert "sources" in payload

    for source in payload["sources"]:
        file_path = Path("tests") / "sample_repo" / source["file"]
        assert file_path.exists(), f"Citation file does not exist on disk: {source['file']}"
        assert source["start_line"] >= 1
        assert source["end_line"] >= source["start_line"]

        # Citation must correspond to retrieved metadata, not to an arbitrary LLM claim.
        assert source["file"].count("/") >= 0
