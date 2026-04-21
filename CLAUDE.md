## Codebase Overview

### What's Built (notebook/document.ipynb)

**Data Pipeline:**
- Loads `.txt` files via `TextLoader` / `DirectoryLoader`
- Loads `.pdf` files via `PyMuPDFLoader`
- Splits docs with `RecursiveCharacterTextSplitter` (chunk_size=1000, overlap=200)

**Core Classes (Sections 1–3, cells 0–16):**

| Class | Cell | Purpose |
|-------|------|---------|
| `EmbeddingManager` | 8 | Wraps `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| `VectorStore` | 11 | ChromaDB wrapper, persists to `../data/vector_store/`, collection `pdf_documents` |
| `RAGRetriever` | 14 | Queries ChromaDB by embedding similarity, returns top-k chunks |

**LLM Integration (Section 4, cells 17–18):**

| Cell | Purpose |
|------|---------|
| 17 | `build_rag_prompt`, `GemmaOllama` class, `rag_query_ollama` — full RAG via Ollama |
| 18 | Demo query: retrieves 3 chunks, generates answer, prints sources |

Gemma 2B runs locally via **Ollama**: `ollama pull gemma:2b` + `ollama serve`

**Known bugs fixed:**
- `RAGRetriever.retrieve()` had `return` inside `for` loop — fixed in cell 14 (return now outside loop)

**Data directories:**
- `data/text_files/` — raw `.txt` docs
- `data/pdf_files/` — raw `.pdf` docs
- `data/vector_store/` — ChromaDB persistence (286 docs indexed)

**NOT YET BUILT:**
- `src/` module extraction (all classes still live in notebook)
- `main.py` is a stub only

**Dependencies:** `langchain`, `sentence-transformers`, `chromadb`, `faiss-cpu`, `pypdf`, `pymupdf`, `python-dotenv`

<!-- code-review-graph MCP tools -->
## MCP Tools: code-review-graph

**IMPORTANT: This project has a knowledge graph. ALWAYS use the
code-review-graph MCP tools BEFORE using Grep/Glob/Read to explore
the codebase.** The graph is faster, cheaper (fewer tokens), and gives
you structural context (callers, dependents, test coverage) that file
scanning cannot.

### When to use graph tools FIRST

- **Exploring code**: `semantic_search_nodes` or `query_graph` instead of Grep
- **Understanding impact**: `get_impact_radius` instead of manually tracing imports
- **Code review**: `detect_changes` + `get_review_context` instead of reading entire files
- **Finding relationships**: `query_graph` with callers_of/callees_of/imports_of/tests_for
- **Architecture questions**: `get_architecture_overview` + `list_communities`

Fall back to Grep/Glob/Read **only** when the graph doesn't cover what you need.

### Key Tools

| Tool | Use when |
|------|----------|
| `detect_changes` | Reviewing code changes — gives risk-scored analysis |
| `get_review_context` | Need source snippets for review — token-efficient |
| `get_impact_radius` | Understanding blast radius of a change |
| `get_affected_flows` | Finding which execution paths are impacted |
| `query_graph` | Tracing callers, callees, imports, tests, dependencies |
| `semantic_search_nodes` | Finding functions/classes by name or keyword |
| `get_architecture_overview` | Understanding high-level codebase structure |
| `refactor_tool` | Planning renames, finding dead code |

### Workflow

1. The graph auto-updates on file changes (via hooks).
2. Use `detect_changes` for code review.
3. Use `get_affected_flows` to understand impact.
4. Use `query_graph` pattern="tests_for" to check coverage.
