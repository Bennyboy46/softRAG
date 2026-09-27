import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.graph_retriever import GraphRetriever
from backend.graph_store import GraphStore
from backend.hybrid_retriever import HybridRetriever
from backend.ingest import ingest_repository
from backend.llm import GroqService, build_citations, build_rag_prompt
from backend.models import (
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from backend.query_router import route_query, should_use_graph
from backend.retriever import Retriever
from backend.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")

REPOSITORY_PATHS: Dict[str, Any] = {}
REPOSITORY_METADATA: Dict[str, Any] = {}

app = FastAPI(
    title="Software Repository RAG API",
    description="Grounded AI code understanding and retrieval over GitHub repositories.",
    version="0.3.0",
)

# Enable CORS for frontend integration with environment-based origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins or [settings.frontend_url],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Global services
vector_store = VectorStore()
retriever = Retriever(vector_store=vector_store)
hybrid_retriever = HybridRetriever(
    semantic_retriever=retriever,
    vector_store=vector_store,
)
groq_service = GroqService()
graph_store = GraphStore()
graph_retriever = GraphRetriever(graph_store=graph_store)


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Health check endpoint providing status of vector database and embedding model."""
    return {
        "status": "healthy",
        "embedding_model": settings.embedding_model,
        "llm_model": settings.groq_model,
        "chroma_path": str(settings.chroma_path),
    }


@app.get("/ready", status_code=status.HTTP_200_OK)
def readiness_check():
    """Simple readiness check for required infrastructure components."""
    ready = True
    issues = []
    try:
        vector_store.get_repository_stats("__health__")
    except Exception:
        ready = False
        issues.append("vector_store")
    if not settings.groq_api_key:
        issues.append("groq_api_key")
    return {"status": "ready" if ready else "not_ready", "checks": issues}


@app.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_200_OK)
def ingest_endpoint(request: IngestRequest):
    """
    Ingest a GitHub repository or local folder:
    clones, filters files, parses AST, chunks code, generates embeddings,
    and stores vectors in an isolated ChromaDB collection.
    """
    if not request.repo_url and not request.local_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must provide either 'repo_url' or 'local_path'.",
        )

    try:
        chunks, stats = ingest_repository(
            repo_url=request.repo_url,
            local_path=request.local_path,
        )
        REPOSITORY_PATHS[stats.repo_id] = Path(stats.repository_path)
        REPOSITORY_METADATA[stats.repo_id] = stats
        return IngestResponse(
            status="success",
            message=f"Repository '{stats.repo_id}' successfully indexed.",
            stats=stats,
        )
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}",
        ) from e


@app.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
def query_endpoint(request: QueryRequest):
    """
    Query an indexed software repository using grounded RAG:
    User Question -> Semantic Search -> ChromaDB -> Grounded Prompt -> Groq LLM -> Answer + Citations.
    """
    if not request.repo_id or not request.repo_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Parameter 'repo_id' is required.",
        )

    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Parameter 'question' cannot be empty.",
        )

    # 1. Verify repository exists in ChromaDB
    if not vector_store.is_repository_indexed(request.repo_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository '{request.repo_id}' is not indexed. Please index it first via POST /ingest.",
        )

    # 2. Decide whether the question is graph-structured and gather graph context if useful
    graph_context = ""
    routing_enabled = should_use_graph(request.question)
    if routing_enabled:
        graph_context = graph_retriever.build_context_for_query(
            repo_id=request.repo_id,
            question=request.question,
            chunks=[],
        )

    # 3. Retrieve top-k hybrid code chunks while preserving the existing API contract
    try:
        chunks = hybrid_retriever.retrieve(
            repo_id=request.repo_id,
            question=request.question,
            top_k=request.top_k,
        )
    except Exception as e:
        logger.error(f"Retrieval error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval failed: {str(e)}",
        ) from e

    if routing_enabled:
        graph_context = graph_retriever.build_context_for_query(
            repo_id=request.repo_id,
            question=request.question,
            chunks=chunks,
        )

    # 4. Handle zero relevant chunks (anti-hallucination guardrail)
    if not chunks and not graph_context.strip():
        return QueryResponse(
            answer="I could not find relevant code in this repository to answer that question.",
            sources=[],
            debug={"message": "No chunks found in vector store"} if request.debug else None,
        )

    # 5. Generate answer via Groq LLM
    try:
        answer = groq_service.generate_answer(
            question=request.question,
            chunks=chunks,
            graph_context=graph_context or None,
        )
    except Exception as e:
        logger.error(f"LLM generation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LLM generation failed: {str(e)}",
        ) from e

    # 6. Build citations directly from metadata (prevents LLM fabricated citations)
    sources = build_citations(chunks)

    # 7. Assemble optional debug trace
    debug_payload: Optional[Dict[str, Any]] = None
    if request.debug:
        system_prompt, user_prompt = build_rag_prompt(request.question, chunks, graph_context=graph_context or None)
        debug_payload = {
            "query": request.question,
            "repo_id": request.repo_id,
            "model": groq_service.model,
            "routing": route_query(request.question)[1],
            "retrieved_chunks_count": len(chunks),
            "graph_context": graph_context or None,
            "retrieved_chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "similarity": c.similarity,
                    "distance": c.distance,
                    "file": c.metadata.file,
                    "line_range": f"{c.metadata.start_line}-{c.metadata.end_line}",
                    "name": c.metadata.name,
                    "type": c.metadata.type,
                }
                for c in chunks
            ],
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        }

    return QueryResponse(
        answer=answer,
        sources=sources,
        debug=debug_payload,
    )


@app.get("/repository/{repo_id}/overview", status_code=status.HTTP_200_OK)
def repository_overview_endpoint(repo_id: str):
    """Return summary information for a repo that was indexed in this backend process."""
    graph = graph_store.get_repository_graph(repo_id)
    metadata = REPOSITORY_METADATA.get(repo_id)
    if graph is None and metadata is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository '{repo_id}' was not found in the local index.",
        )

    important_files = sorted({node.file for node in graph.nodes.values()})[:20] if graph else []
    routes = [node.name for node in graph.nodes.values() if node.type == "route"] if graph else []
    languages = metadata.languages if metadata else {}
    return {
        "repo_id": repo_id,
        "repository": getattr(metadata, "repository", repo_id),
        "repository_url": getattr(metadata, "repository_url", None),
        "commit_hash": getattr(metadata, "commit_hash", None),
        "branch": getattr(metadata, "branch", None),
        "indexed_at": getattr(metadata, "indexed_at", None),
        "files_processed": getattr(metadata, "files_processed", len(important_files)),
        "chunks_created": getattr(metadata, "chunks_created", 0),
        "embeddings_created": getattr(metadata, "embeddings_created", 0),
        "languages": languages,
        "important_files": important_files,
        "api_routes": routes,
        "symbol_count": len(graph.nodes) if graph else 0,
    }


@app.get("/repository/{repo_id}/source", status_code=status.HTTP_200_OK)
def repository_source_endpoint(repo_id: str, file: str):
    """Safely return the content for a source file belonging to the repository."""
    if not file or not file.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Parameter 'file' is required.")

    repo_dir = REPOSITORY_PATHS.get(repo_id)
    if repo_dir is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Repository '{repo_id}' is not known to this backend.")

    safe_file = Path(file)
    if safe_file.is_absolute() or ".." in safe_file.parts:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Path traversal is not allowed.")

    resolved = (Path(repo_dir) / safe_file).resolve()
    repo_root = Path(repo_dir).resolve()
    if repo_root not in resolved.parents and resolved != repo_root:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Requested file is outside the repository root.")
    if not resolved.exists() or not resolved.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Source file '{file}' was not found in repository '{repo_id}'.")

    text = resolved.read_text(encoding="utf-8", errors="replace")
    suffix = resolved.suffix.lower().lstrip(".")
    language = settings.extension_to_language.get(resolved.suffix.lower(), suffix or "text")
    return {
        "repo_id": repo_id,
        "file": str(safe_file).replace("\\", "/"),
        "language": language,
        "content": text,
        "path": str(resolved).replace("\\", "/"),
    }


@app.get("/repository/{repo_id}/graph", status_code=status.HTTP_200_OK)
def repository_graph_endpoint(repo_id: str):
    """Return the repository code graph for a given repo ID."""
    graph = graph_store.get_repository_graph(repo_id)
    if graph is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository graph for '{repo_id}' was not found.",
        )
    return graph.to_dict()


@app.get("/repository/{repo_id}/symbol/{symbol_name}", status_code=status.HTTP_200_OK)
def repository_symbol_endpoint(repo_id: str, symbol_name: str):
    """Return a symbol node and its linked graph context from the repository graph."""
    graph = graph_store.get_repository_graph(repo_id)
    if graph is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository graph for '{repo_id}' was not found.",
        )
    node = graph.find_node(symbol_name)
    if node is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Symbol '{symbol_name}' was not found in repository '{repo_id}'.",
        )
    related = graph_retriever.get_related_entities(repo_id, symbol_name)
    return {
        "symbol": {
            "id": node.id,
            "repository": node.repository,
            "file": node.file,
            "type": node.type,
            "name": node.name,
            "start_line": node.start_line,
            "end_line": node.end_line,
            "metadata": node.metadata,
        },
        "related": related,
    }


@app.delete("/repository/{repo_id}", status_code=status.HTTP_200_OK)
def delete_repository_endpoint(repo_id: str):
    """Delete a repository's indexed data and vector collection from ChromaDB."""
    deleted = vector_store.delete_repository(repo_id)
    graph_store.clear_repository(repo_id)
    REPOSITORY_PATHS.pop(repo_id, None)
    REPOSITORY_METADATA.pop(repo_id, None)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository collection for '{repo_id}' not found.",
        )
    return {"status": "success", "message": f"Repository '{repo_id}' deleted successfully."}
