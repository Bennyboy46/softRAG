import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

from backend.chunker import CodeAwareChunker
from backend.code_graph import build_repository_graph
from backend.config import settings
from backend.embeddings import EmbeddingService
from backend.graph_store import GraphStore
from backend.models import CodeChunk, IngestionStats
from backend.repository import clone_repository, filter_repository_files, generate_repo_id, get_repo_version_details
from backend.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ingest")


def ingest_repository(
    repo_url: Optional[str] = None,
    local_path: Optional[str] = None,
    skip_embeddings: bool = False,
) -> Tuple[List[CodeChunk], IngestionStats]:
    """
    Complete end-to-end repository ingestion pipeline:
    1. Clone repository from GitHub (or locate local path)
    2. Filter eligible source files (ignoring caches, node_modules, binaries)
    3. Parse AST structures and generate code-aware semantic chunks
    4. Generate dense vector embeddings via Sentence Transformers
    5. Store chunks, embeddings, and metadata into isolated ChromaDB collection
    6. Return chunks and comprehensive ingestion statistics
    """
    start_time = time.time()

    if not repo_url and not local_path:
        raise ValueError("Must provide either repo_url or local_path for ingestion.")

    # 1. Resolve repository directory and unique identifier
    if repo_url:
        repo_id, repo_dir = clone_repository(repo_url)
    else:
        repo_dir = Path(local_path).resolve()
        if not repo_dir.exists():
            raise FileNotFoundError(f"Local repository path does not exist: {repo_dir}")
        repo_id = generate_repo_id(repo_dir.name)

    repository_url, commit_hash, branch = get_repo_version_details(repo_dir, repo_url=repo_url)
    logger.info(f"Starting ingestion for repository '{repo_id}' at {repo_dir}")

    # 2. Filter eligible source files
    all_repo_items = list(repo_dir.rglob("*"))
    total_files_scanned = sum(1 for p in all_repo_items if p.is_file())
    eligible_files = filter_repository_files(repo_dir)

    # 3. Parse AST structures and create semantic chunks
    chunker = CodeAwareChunker()
    all_chunks: List[CodeChunk] = []
    language_counts: Counter = Counter()
    symbol_counts: Counter = Counter()

    for file_path in eligible_files:
        relative_path = str(file_path.relative_to(repo_dir)).replace("\\", "/")
        ext = file_path.suffix.lower()
        language = settings.extension_to_language.get(ext, "unknown")

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                code_content = f.read()
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            continue

        file_chunks = chunker.chunk_file(
            repo_id=repo_id,
            relative_path=relative_path,
            code=code_content,
            language=language,
        )

        all_chunks.extend(file_chunks)
        language_counts[language] += 1
        for c in file_chunks:
            symbol_counts[c.metadata.type] += 1

    embeddings_count = 0

    # 4. Generate embeddings and store in ChromaDB
    if not skip_embeddings and all_chunks:
        embedding_service = EmbeddingService()
        chunk_texts = [c.content for c in all_chunks]
        embeddings = embedding_service.embed_chunks(chunk_texts)
        embeddings_count = len(embeddings)

        # 5. Persist to ChromaDB vector store
        vector_store = VectorStore()
        vector_store.add_chunks(
            repo_id=repo_id,
            chunks=all_chunks,
            embeddings=embeddings,
        )

    # 6. Build and persist repository code graph for graph-aware retrieval
    graph_store = GraphStore()
    graph = build_repository_graph(repo_id=repo_id, repo_dir=repo_dir, files=eligible_files)
    graph_store.add_graph(repo_id=repo_id, graph=graph)

    duration = time.time() - start_time

    # 6. Generate statistics
    stats = IngestionStats(
        repository=repo_id,
        repo_id=repo_id,
        repository_path=str(repo_dir),
        repository_url=repository_url,
        commit_hash=commit_hash,
        branch=branch,
        indexed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        files_processed=len(eligible_files),
        total_files_scanned=total_files_scanned,
        skipped_files_count=total_files_scanned - len(eligible_files),
        chunks_created=len(all_chunks),
        embeddings_created=embeddings_count,
        languages=dict(language_counts),
        symbols_breakdown=dict(symbol_counts),
        duration_seconds=round(duration, 3),
        status="success",
    )

    logger.info(
        f"Ingestion complete: {len(eligible_files)} files indexed, "
        f"{len(all_chunks)} chunks created, {embeddings_count} embeddings stored in {stats.duration_seconds}s."
    )

    return all_chunks, stats


def print_cli_summary(chunks: List[CodeChunk], stats: IngestionStats, sample_limit: int = 5):
    """Format and print an executive summary and sample chunks for CLI inspection."""
    print("\n" + "=" * 70)
    print("           SOFTWARE REPOSITORY RAG - INGESTION SUMMARY (STAGE 2)")
    print("=" * 70)
    print(f"Repository ID:         {stats.repo_id}")
    print(f"Repository Path:       {stats.repository_path}")
    print(f"Files Processed:       {stats.files_processed}")
    print(f"Total Files Scanned:   {stats.total_files_scanned}")
    print(f"Files Skipped:         {stats.skipped_files_count}")
    print(f"Total Chunks Created:  {stats.chunks_created}")
    print(f"Embeddings Generated:  {stats.embeddings_created} (Model: {settings.embedding_model})")
    print(f"ChromaDB Path:         {settings.chroma_path}")
    print(f"Status:                {stats.status}")
    print(f"Ingestion Duration:    {stats.duration_seconds:.3f} seconds")
    print("-" * 70)
    print("Indexed Languages Breakdown:")
    for lang, count in sorted(stats.languages.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {lang:15s}: {count} files")
    print("-" * 70)
    print("Semantic Chunk Types Breakdown:")
    for sym_type, count in sorted(stats.symbols_breakdown.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {sym_type:15s}: {count} chunks")
    print("=" * 70)

    # Print sample chunks
    display_chunks = chunks[:sample_limit]
    print(f"\nDisplaying first {len(display_chunks)} generated chunks with metadata:\n")

    for idx, chunk in enumerate(display_chunks, start=1):
        m = chunk.metadata
        print(f"[{idx}] CHUNK ID: {chunk.chunk_id}")
        print(f"    File:       {m.file}")
        print(f"    Language:   {m.language}")
        print(f"    Type:       {m.type}")
        print(f"    Name:       {m.name or '(None)'}")
        print(f"    Line Range: {m.start_line} - {m.end_line}")
        print("    --- Code Content Preview ---")
        preview_lines = chunk.content.splitlines()[:8]
        for line in preview_lines:
            print(f"    | {line}")
        if len(chunk.content.splitlines()) > 8:
            print(f"    | ... ({len(chunk.content.splitlines()) - 8} more lines)")
        print("-" * 70)


def main():
    parser = argparse.ArgumentParser(description="Software Repository RAG - Ingestion CLI")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--repo-url", type=str, help="GitHub repository URL (e.g. https://github.com/user/repo)")
    group.add_argument("--local-path", type=str, help="Path to a local repository folder")
    parser.add_argument("--sample-limit", type=int, default=5, help="Number of sample chunks to print (default: 5)")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip embedding generation and vector storage")
    parser.add_argument("--json", action="store_true", help="Output full ingestion statistics in JSON format")

    args = parser.parse_args()

    try:
        chunks, stats = ingest_repository(
            repo_url=args.repo_url,
            local_path=args.local_path,
            skip_embeddings=args.skip_embeddings,
        )
        if args.json:
            print(json.dumps(stats.model_dump(), indent=2))
        else:
            print_cli_summary(chunks, stats, sample_limit=args.sample_limit)
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
