from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ParsedSymbol(BaseModel):
    """Intermediate AST symbol representation extracted by the code parser."""
    name: Optional[str] = None
    type: str  # e.g., 'function', 'class', 'method', 'import', 'route', 'query', 'section'
    start_line: int  # 1-indexed
    end_line: int  # 1-indexed
    start_byte: int
    end_byte: int
    parent_name: Optional[str] = None
    children: List["ParsedSymbol"] = Field(default_factory=list)


class ChunkMetadata(BaseModel):
    """Standardized metadata attached to every indexed code chunk."""
    repository: str
    file: str  # Relative file path within repository
    language: str  # Canonical language, e.g., 'python', 'typescript'
    type: str  # 'function', 'class', 'method', 'import', 'section', 'block'
    name: Optional[str] = None  # Identifier name (e.g., function or class name)
    start_line: int  # 1-indexed
    end_line: int  # 1-indexed

    def to_chroma_metadata(self) -> Dict[str, Any]:
        """Convert metadata to ChromaDB-safe types (null values converted to empty strings)."""
        data = self.model_dump()
        if data.get("name") is None:
            data["name"] = ""
        return data

    @classmethod
    def from_chroma_metadata(cls, meta: Dict[str, Any]) -> "ChunkMetadata":
        """Reconstruct ChunkMetadata from ChromaDB metadata dictionary."""
        raw_name = meta.get("name")
        name = None if raw_name == "" else raw_name
        return cls(
            repository=meta.get("repository", ""),
            file=meta.get("file", ""),
            language=meta.get("language", ""),
            type=meta.get("type", "block"),
            name=name,
            start_line=int(meta.get("start_line", 1)),
            end_line=int(meta.get("end_line", 1)),
        )


class CodeChunk(BaseModel):
    """A semantic chunk of code with complete provenance metadata."""
    chunk_id: str
    content: str
    metadata: ChunkMetadata

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk into a dictionary format compatible with vector databases."""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "metadata": self.metadata.to_chroma_metadata(),
        }


class VectorSearchResult(BaseModel):
    """Result retrieved from vector database similarity search."""
    chunk_id: str
    content: str
    metadata: ChunkMetadata
    distance: float  # Cosine distance (0 = identical, 2 = opposite)
    similarity: float  # Cosine similarity (1.0 - distance)


class Citation(BaseModel):
    """Accurate source citation constructed directly from retrieved chunk metadata."""
    file: str
    start_line: int
    end_line: int
    name: Optional[str] = None
    type: Optional[str] = None


class GraphNode(BaseModel):
    """Node in the repository dependency graph."""
    id: str
    repository: str
    file: str
    type: str
    name: str
    start_line: int = 0
    end_line: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    """Relationship between repository entities."""
    source: str
    target: str
    relationship: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestionStats(BaseModel):
    """Metrics and statistics generated during repository ingestion."""
    repository: str
    repo_id: str
    repository_path: str
    repository_url: Optional[str] = None
    commit_hash: Optional[str] = None
    branch: Optional[str] = None
    indexed_at: Optional[str] = None
    files_processed: int
    total_files_scanned: int
    skipped_files_count: int
    chunks_created: int
    embeddings_created: int
    languages: Dict[str, int]
    symbols_breakdown: Dict[str, int]
    duration_seconds: float
    status: str = "success"


class IngestRequest(BaseModel):
    """Payload for POST /ingest."""
    repo_url: Optional[str] = None
    local_path: Optional[str] = None


class IngestResponse(BaseModel):
    """Response returned upon successful ingestion."""
    status: str
    message: str
    stats: IngestionStats


class QueryRequest(BaseModel):
    """Payload for POST /query."""
    repo_id: str
    question: str
    top_k: Optional[int] = None
    debug: bool = False


class QueryResponse(BaseModel):
    """Response returned upon querying repository."""
    answer: str
    sources: List[Citation]
    debug: Optional[Dict[str, Any]] = None
