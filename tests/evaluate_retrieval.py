import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.ingest import ingest_repository
from backend.keyword_retriever import KeywordRetriever
from backend.hybrid_retriever import HybridRetriever
from backend.reranker import Reranker
from backend.retriever import Retriever

DATASET_PATH = PROJECT_ROOT / "evaluation" / "questions.json"


def _recall_at_k(results, expected_files, k):
    if not expected_files:
        return 1.0 if not results[:k] else 0.0
    expected = set(expected_files)
    hits = expected.intersection(set(results[:k]))
    return len(hits) / max(len(expected), 1)


def _mrr(results, expected_files):
    if not expected_files:
        return 1.0 if not results else 0.0
    expected = set(expected_files)
    for idx, file_name in enumerate(results, start=1):
        if file_name in expected:
            return 1.0 / idx
    return 0.0


def evaluate_method(method_name, runner, dataset, repo_id):
    metrics = {"Recall@1": [], "Recall@3": [], "Recall@5": [], "Recall@10": [], "MRR": []}
    for item in dataset:
        question = item["question"]
        expected = item["expected_sources"]
        result = runner(repo_id=repo_id, question=question, top_k=10)
        files = [chunk.metadata.file for chunk in result]
        metrics["Recall@1"].append(_recall_at_k(files, expected, 1))
        metrics["Recall@3"].append(_recall_at_k(files, expected, 3))
        metrics["Recall@5"].append(_recall_at_k(files, expected, 5))
        metrics["Recall@10"].append(_recall_at_k(files, expected, 10))
        metrics["MRR"].append(_mrr(files, expected))

    return {
        "Method": method_name,
        "Recall@1": round(sum(metrics["Recall@1"]) / len(metrics["Recall@1"]), 4),
        "Recall@3": round(sum(metrics["Recall@3"]) / len(metrics["Recall@3"]), 4),
        "Recall@5": round(sum(metrics["Recall@5"]) / len(metrics["Recall@5"]), 4),
        "Recall@10": round(sum(metrics["Recall@10"]) / len(metrics["Recall@10"]), 4),
        "MRR": round(sum(metrics["MRR"]) / len(metrics["MRR"]), 4),
    }


def main():
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    _, stats = ingest_repository(local_path="tests/sample_repo")
    repo_id = stats.repo_id

    semantic = Retriever()
    keyword = KeywordRetriever()
    hybrid = HybridRetriever()
    reranker = Reranker(top_k=5)

    def semantic_runner(repo_id, question, top_k):
        return semantic.retrieve(repo_id=repo_id, question=question, top_k=top_k)

    def keyword_runner(repo_id, question, top_k):
        return keyword.keyword_search(repo_id=repo_id, query=question, top_k=top_k)

    def hybrid_runner(repo_id, question, top_k):
        return hybrid.retrieve(repo_id=repo_id, question=question, top_k=top_k)

    def reranked_runner(repo_id, question, top_k):
        hybrid_results = hybrid.retrieve(repo_id=repo_id, question=question, top_k=10)
        return reranker.rerank(question, hybrid_results, top_k=top_k)

    rows = [
        evaluate_method("Semantic only", semantic_runner, dataset, repo_id),
        evaluate_method("Keyword only", keyword_runner, dataset, repo_id),
        evaluate_method("Hybrid", hybrid_runner, dataset, repo_id),
        evaluate_method("Hybrid + reranking", reranked_runner, dataset, repo_id),
    ]

    print("Retrieval evaluation on sample repository")
    print("=" * 80)
    print(f"Dataset size: {len(dataset)}")
    print(f"Repo ID: {repo_id}")
    print("\nMethod                  Recall@1   Recall@3   Recall@5   Recall@10   MRR")
    for row in rows:
        print(
            f"{row['Method']:<22} "
            f"{row['Recall@1']:<9} "
            f"{row['Recall@3']:<9} "
            f"{row['Recall@5']:<9} "
            f"{row['Recall@10']:<10} "
            f"{row['MRR']:<6}"
        )

    return rows


if __name__ == "__main__":
    main()
