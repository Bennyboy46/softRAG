# softRAG

softRAG is a repository-aware code Q&A system that ingests a Git repository, builds searchable vector and graph context, and answers code questions grounded in the repository itself.

It combines:

- a FastAPI backend for ingestion and retrieval
- ChromaDB-backed vector search
- keyword + semantic hybrid retrieval
- repository dependency graph context
- Groq-based grounded answer generation
- a React + Vite frontend for repository indexing and Q&A

The project is designed to help developers ask questions like:

- “What does this function do?”
- “Which files define the API routes?”
- “Where is this dependency used?”
- “Explain how this module works in context.”

---

## Quick Start

### 1. Install backend dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a local `.env` file from [.env.example](.env.example) and add your Groq key if needed.

```env
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHROMA_PATH=./data/chroma
```

### 3. Start the backend

```bash
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Start the frontend

```bash
cd frontend
npm install
npm run dev -- --host 0.0.0.0
```

Then open the frontend in the browser and ingest a repository to start asking code questions.

---



## Architecture

```text
Repository / Local Folder
        ↓
File filtering + metadata capture
        ↓
Parse code with tree-sitter
        ↓
Chunk code into searchable segments
        ↓
Generate embeddings with Sentence Transformers
        ↓
Store vectors in per-repository ChromaDB collections
        ↓
Hybrid retrieval: semantic + keyword
        ↓
Optional code graph context
        ↓
Prompt + grounded answer generation with Groq
        ↓
Verified citations back to source files/lines
```

### Backend

The backend lives in [backend](backend) and exposes the API used by the frontend.

Core modules:

- [backend/main.py](backend/main.py): FastAPI app and routes
- [backend/ingest.py](backend/ingest.py): repository ingestion pipeline
- [backend/retriever.py](backend/retriever.py): semantic search
- [backend/hybrid_retriever.py](backend/hybrid_retriever.py): merged retrieval logic
- [backend/graph_store.py](backend/graph_store.py): graph persistence
- [backend/graph_retriever.py](backend/graph_retriever.py): graph-aware retrieval context
- [backend/llm.py](backend/llm.py): model prompting and citation logic
- [backend/config.py](backend/config.py): settings and environment configuration

### Frontend

The frontend is a React + Vite app under [frontend](frontend).

It includes:

- repository ingest form
- repository ID / question UI
- answer display with markdown rendering
- source file viewer
- repository summary and graph overview

---

## Features

- Repository indexing from GitHub URL or local path
- Repo-isolated vector storage
- Hybrid retrieval for better code search quality
- Graph-aware context for dependency and symbol questions
- Grounded answers with source citations
- Frontend for repository intake and chat-style interaction
- Hallucination protections and citation validation logic
- Evaluation suite for retrieval and answer quality

---

## Project layout

```text
softRAG/
├── backend/
│   ├── __init__.py
│   ├── config.py
│   ├── graph_retriever.py
│   ├── graph_store.py
│   ├── hybrid_retriever.py
│   ├── ingest.py
│   ├── llm.py
│   ├── main.py
│   ├── models.py
│   ├── parser.py
│   ├── repository.py
│   ├── retriever.py
│   └── vector_store.py
├── data/
│   ├── chroma/
│   └── repositories/
├── evaluation/
│   └── questions.json
├── frontend/
│   ├── src/
│   ├── package.json
│   ├── vite.config.js
│   └── ...
├── tests/
│   ├── test_phase1.py
│   ├── test_rag.py
│   ├── test_stage2.py
│   ├── test_stage3.py
│   └── test_vector_search.py
├── .env.example
├── pytest.ini
├── requirements.txt
├── README.md
└── ...
```

---

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Copy [.env.example](.env.example) to a local `.env` and fill in the required values.

Example:

```env
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHROMA_PATH=./data/chroma
```

---

## Run the app

### Start the backend

From the project root:

```bash
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### Start the frontend

From [frontend](frontend):

```bash
npm install
npm run dev -- --host 0.0.0.0
```

The frontend will connect to the backend at the configured API base URL, defaulting to:

```text
http://localhost:8000
```

---

## API overview

The backend exposes endpoints for ingestion and querying, including:

- `POST /ingest` to index a repository
- `POST /query` to ask a question about an indexed repo
- `GET /health` to check service health
- `GET /ready` for readiness checks
- `GET /repository/{repo_id}/overview` for repo summary data
- `GET /repository/{repo_id}/source` for file source lookup
- `GET /repository/{repo_id}/graph` for graph metadata

---

## Testing

Run the backend regression suite:

```bash
pytest -q
```

Run the frontend test suite:

```bash
cd frontend
npm test
```

---

## Evaluation and reliability

The project includes evaluation scripts and repository-grounded validation logic for:

- retrieval quality
- answer correctness
- groundedness
- citation reliability
- hallucination resistance

Relevant files include:

- [tests](tests)
- [evaluation/questions.json](evaluation/questions.json)

---

## Notes

- Repository files are treated as untrusted input.
- The system does not execute repository code.
- Retrieval and answer generation remain grounded in indexed repository evidence.
- This project is intended as a local repository intelligence tool and product prototype, not a full production multi-user SaaS deployment.

---

## License

This project is currently provided as a local engineering prototype for repository-aware code Q&A. Update the license file if you plan to distribute it publicly.
