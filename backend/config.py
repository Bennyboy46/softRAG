from pathlib import Path
from typing import Dict, Set
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for Software Repository RAG."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Storage Paths
    base_dir: Path = Path(__file__).resolve().parent.parent
    repositories_path: Path = base_dir / "data" / "repositories"
    chroma_path: Path = base_dir / "data" / "chroma"

    # LLM Settings (Stage 3)
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"

    # Embeddings & Vector Store (Stage 2)
    embedding_model: str = "all-MiniLM-L6-v2"
    top_k: int = 5

    # Retrieval tuning (Stage 4)
    semantic_top_k: int = 10
    keyword_top_k: int = 10
    hybrid_top_k: int = 15
    rerank_top_k: int = 5
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3
    reranker_model: str = ""
    keyword_retrieval_enabled: bool = True
    graph_max_depth: int = 3

    # Stage 6: evaluation, context budgets, retry and observability
    max_context_tokens: int = 4000
    max_chunks: int = 8
    max_graph_nodes: int = 20
    max_retries: int = 3
    initial_retry_delay: float = 0.5

    # Stage 7: frontend and production configuration
    frontend_url: str = "http://localhost:5173"
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    debug_mode: bool = False

    # File Filtering Restrictions
    max_file_size_bytes: int = 1_048_576  # 1 MB

    # Directories to ignore unconditionally
    ignored_directories: Set[str] = {
        ".git",
        ".github",
        ".vscode",
        ".idea",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "node_modules",
        ".next",
        ".nuxt",
        "dist",
        "build",
        "out",
        "target",
        "bin",
        "obj",
        "venv",
        ".venv",
        "env",
        ".env",
        ".tox",
        ".gradle",
    }

    # Files to ignore (e.g. lockfiles, minified files)
    ignored_file_names: Set[str] = {
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "poetry.lock",
        "cargo.lock",
        "pipfile.lock",
        "composer.lock",
        "go.sum",
    }

    # Ignored extensions (binaries, documents, bundles, minified)
    ignored_extensions: Set[str] = {
        ".exe", ".dll", ".so", ".dylib", ".bin",
        ".zip", ".tar", ".gz", ".7z", ".rar",
        ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp",
        ".mp3", ".mp4", ".wav", ".avi", ".mov",
        ".pdf", ".docx", ".xlsx", ".pptx",
        ".pyc", ".pyo", ".pyd", ".class", ".jar",
        ".min.js", ".min.css", ".map",
    }

    # Supported file extensions mapped to canonical language names
    extension_to_language: Dict[str, str] = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".java": "java",
        ".c": "c",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".h": "cpp",
        ".hpp": "cpp",
        ".sql": "sql",
        ".md": "markdown",
        ".markdown": "markdown",
    }


# Singleton settings instance
settings = Settings()
