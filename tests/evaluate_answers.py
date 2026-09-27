import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.ingest import ingest_repository
from backend.main import app
from fastapi.testclient import TestClient

DATASET_PATH = PROJECT_ROOT / "evaluation" / "questions.json"


def _safe_lower(text):
    return (text or "").lower()


def evaluate_answers():
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    _, stats = ingest_repository(local_path="tests/sample_repo")
    client = TestClient(app)

    rows = []
    for item in dataset:
        response = client.post(
            "/query",
            json={"repo_id": stats.repo_id, "question": item["question"]},
        )
        payload = response.json()
        answer = payload.get("answer", "")
        sources = payload.get("sources", [])

        expected = item["expected_sources"]
        answer_correct = any(term.lower() in _safe_lower(answer) for term in item.get("expected_terms", [])) or bool(expected and any(source["file"].endswith(path) for source in sources for path in expected))
        grounded = "could not find sufficient evidence" not in _safe_lower(answer) or item["category"] == "unanswerable"
        cited = all(
            source.get("file") and source.get("start_line") is not None and source.get("end_line") is not None
            for source in sources
        )
        rows.append({
            "question": item["question"],
            "category": item["category"],
            "correct": answer_correct,
            "grounded": grounded,
            "citation_correct": cited,
        })

    correctness = sum(1 for row in rows if row["correct"]) / max(len(rows), 1)
    groundedness = sum(1 for row in rows if row["grounded"]) / max(len(rows), 1)
    citations = sum(1 for row in rows if row["citation_correct"]) / max(len(rows), 1)

    print("Answer evaluation")
    print("=" * 60)
    print(f"Correctness: {correctness:.4f}")
    print(f"Groundedness: {groundedness:.4f}")
    print(f"Citation correctness: {citations:.4f}")
    print("=" * 60)

    return {
        "correctness": round(correctness, 4),
        "groundedness": round(groundedness, 4),
        "citation_correctness": round(citations, 4),
        "rows": rows,
    }


if __name__ == "__main__":
    evaluate_answers()
