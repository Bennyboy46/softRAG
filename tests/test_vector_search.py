import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.embeddings import EmbeddingService
from backend.ingest import ingest_repository
from backend.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_vector_search")


def test_semantic_search():
    sample_repo_dir = PROJECT_ROOT / "tests" / "sample_repo"
    print("\n" + "=" * 70)
    print("STEP 1: Ingesting repository & generating embeddings for ChromaDB...")
    print("=" * 70)

    # 1. Ingest repository, generate embeddings, and store in ChromaDB
    chunks, stats = ingest_repository(local_path=str(sample_repo_dir))
    repo_id = stats.repo_id

    print(f"\nRepository ID:        {repo_id}")
    print(f"Chunks indexed:       {stats.chunks_created}")
    print(f"Embeddings stored:    {stats.embeddings_created}")

    # 2. Initialize embedding service and vector store
    embedding_service = EmbeddingService()
    vector_store = VectorStore()

    # 3. Test queries
    queries = [
        "Where is authentication implemented?",
        "Where is the database connection created?",
        "How are user profiles fetched or deactivated?",
    ]

    print("\n" + "=" * 70)
    print("STEP 2: Executing Natural Language Queries via Semantic Search")
    print("=" * 70)

    for query in queries:
        print(f"\nQuery:\n\"{query}\"\n")
        print("Results:\n")

        # Encode query into dense vector
        query_vector = embedding_service.embed_text(query)

        # Retrieve top 5 semantic chunks
        results = vector_store.search(
            repo_id=repo_id,
            query_embedding=query_vector,
            top_k=5,
        )

        if not results:
            print("  (No relevant results found)")
            continue

        for rank, res in enumerate(results, start=1):
            m = res.metadata
            symbol_display = m.name if m.name else "(anonymous block)"
            print(f"{rank}. {m.file}")
            print(f"   {m.type.capitalize()}: {symbol_display}")
            print(f"   Lines: {m.start_line}-{m.end_line}")
            print(f"   Similarity: {res.similarity:.4f} (Distance: {res.distance:.4f})")
            # First line preview
            first_line = res.content.strip().splitlines()[0] if res.content else ""
            print(f"   Preview: {first_line[:75]}")
            print()

        print("-" * 70)


if __name__ == "__main__":
    test_semantic_search()
