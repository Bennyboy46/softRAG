import logging
import sys
from pathlib import Path

# Configure UTF-8 stdout
if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.ingest import ingest_repository
from backend.llm import GroqService, build_citations
from backend.retriever import Retriever
from backend.vector_store import VectorStore

logging.basicConfig(
    level=logging.WARNING,  # Suppress verbose info logs during presentation
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def run_rag_tests():
    sample_repo_dir = PROJECT_ROOT / "tests" / "sample_repo"
    vector_store = VectorStore()
    retriever = Retriever(vector_store=vector_store)
    groq_service = GroqService()

    # 1. Ensure sample repo is indexed
    print("=" * 80)
    print("STEP 1: Verifying Repository Ingestion...")
    print("=" * 80)
    chunks, stats = ingest_repository(local_path=str(sample_repo_dir))
    repo_id = stats.repo_id
    print(f"Repository ID:     {repo_id}")
    print(f"Chunks Indexed:    {stats.chunks_created}")
    print(f"Embeddings Stored: {stats.embeddings_created}")
    print(f"Groq LLM Model:    {groq_service.model}")
    print("=" * 80)

    # 2. Test questions (answerable and unanswerable)
    test_cases = [
        {
            "category": "Answerable Query (Authentication)",
            "question": "Where is authentication implemented and how does it work?",
        },
        {
            "category": "Answerable Query (Database)",
            "question": "Where is the database connection created?",
        },
        {
            "category": "Answerable Query (User Profiles / TS)",
            "question": "How are user profiles fetched or deactivated?",
        },
        {
            "category": "Answerable Query (Framework Architecture)",
            "question": "What framework does this repository use?",
        },
        {
            "category": "Unanswerable / Anti-Hallucination Test (Negative Control)",
            "question": "What payment gateway does this repository use?",
        },
    ]

    print("\n" + "=" * 80)
    print("STEP 2: Executing Grounded RAG Pipeline Evaluation")
    print("=" * 80)

    for i, test in enumerate(test_cases, start=1):
        q = test["question"]
        cat = test["category"]

        print(f"\n[{i}] TEST CASE: {cat}")
        print("-" * 80)
        print(f"QUESTION:\n  {q}\n")

        # 1. Retrieve relevant chunks
        chunks = retriever.retrieve(repo_id=repo_id, question=q, top_k=5)

        print("RETRIEVED SOURCES:")
        if not chunks:
            print("  (None retrieved)")
        else:
            for rank, c in enumerate(chunks, start=1):
                m = c.metadata
                name_str = f" | Symbol: {m.name}" if m.name else ""
                print(
                    f"  [{rank}] {m.file}:{m.start_line}-{m.end_line} "
                    f"({m.type}{name_str}) — Similarity: {c.similarity:.4f}"
                )

        # 2. Generate answer
        answer = groq_service.generate_answer(question=q, chunks=chunks)
        print(f"\nGENERATED ANSWER:\n{answer}\n")

        # 3. Backend citations
        citations = build_citations(chunks)
        print("CITATIONS (Backend Generated from Metadata):")
        for cit in citations:
            symbol_info = f" ({cit.name})" if cit.name else ""
            print(f"  - {cit.file}:{cit.start_line}-{cit.end_line}{symbol_info}")

        print("=" * 80)


if __name__ == "__main__":
    run_rag_tests()
