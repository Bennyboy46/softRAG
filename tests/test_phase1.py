from pathlib import Path
import pytest

from backend.chunker import CodeAwareChunker
from backend.models import CodeChunk
from backend.parser import CodeParser
from backend.repository import filter_repository_files, generate_repo_id


@pytest.fixture
def sample_repo_path():
    return Path(__file__).parent / "sample_repo"


def test_repo_id_generation():
    url = "https://github.com/fastapi/fastapi.git"
    repo_id = generate_repo_id(url)
    assert "fastapi_fastapi" in repo_id
    assert len(repo_id.split("_")[-1]) == 8


def test_file_filtering(sample_repo_path):
    files = filter_repository_files(sample_repo_path)
    file_names = {f.name for f in files}

    # Verified eligible source files
    assert "auth.py" in file_names
    assert "database.py" in file_names
    assert "user_service.ts" in file_names
    assert "schema.sql" in file_names
    assert "README.md" in file_names

    # Verified excluded files
    assert "fake.js" not in file_names  # Inside node_modules/
    assert "package-lock.json" not in file_names  # Ignored lockfile


def test_tree_sitter_python_ast():
    parser = CodeParser()
    py_code = """
import os

@app.post("/login")
def login_handler(user: str):
    return user

class AuthManager:
    def verify(self):
        return True
"""
    symbols = parser.parse(py_code, "python")
    symbol_names = [s.name for s in symbols]
    symbol_types = [s.type for s in symbols]

    assert "login_handler" in symbol_names
    assert "AuthManager" in symbol_names
    assert "verify" in symbol_names
    assert "function" in symbol_types
    assert "class" in symbol_types
    assert "method" in symbol_types

    # Validate 1-indexed lines
    login_sym = next(s for s in symbols if s.name == "login_handler")
    assert login_sym.start_line == 4  # Starts at decorator
    assert login_sym.end_line == 6


def test_tree_sitter_typescript_ast():
    parser = CodeParser()
    ts_code = """
export class UserService {
  getUser(id: string) {
    return id;
  }
}

export const fetchAll = () => [];
"""
    symbols = parser.parse(ts_code, "typescript")
    symbol_names = [s.name for s in symbols]

    assert "UserService" in symbol_names
    assert "getUser" in symbol_names
    assert "fetchAll" in symbol_names


def test_markdown_section_parsing():
    parser = CodeParser()
    md_content = """# Title
Some intro text.

## Features
- Feature 1
- Feature 2

### Sub feature
Details here.
"""
    symbols = parser.parse(md_content, "markdown")
    assert len(symbols) == 3
    assert symbols[0].name == "Title"
    assert symbols[0].start_line == 1
    assert symbols[1].name == "Features"
    assert symbols[1].start_line == 4
    assert symbols[2].name == "Sub feature"
    assert symbols[2].start_line == 8


def test_code_aware_chunker(sample_repo_path):
    chunker = CodeAwareChunker()
    auth_file = sample_repo_path / "auth.py"
    with open(auth_file, "r", encoding="utf-8") as f:
        code = f.read()

    chunks = chunker.chunk_file(
        repo_id="test_repo",
        relative_path="auth.py",
        code=code,
        language="python",
    )

    assert len(chunks) > 0

    # Ensure module header was extracted
    headers = [c for c in chunks if c.metadata.name == "module_header"]
    assert len(headers) == 1
    assert headers[0].metadata.start_line == 1

    # Ensure login function chunk exists with decorator
    login_chunks = [c for c in chunks if c.metadata.name == "login"]
    assert len(login_chunks) == 1
    assert "@app.post" in login_chunks[0].content
    assert login_chunks[0].metadata.start_line == 12
    assert login_chunks[0].metadata.end_line == 22

    # Ensure all chunks have 1-indexed valid start/end lines
    for chunk in chunks:
        assert chunk.metadata.start_line >= 1
        assert chunk.metadata.end_line >= chunk.metadata.start_line
        assert chunk.metadata.repository == "test_repo"
        assert chunk.metadata.file == "auth.py"
