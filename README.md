# ytrag — Local RAG with Gemma 2B

A fully local Retrieval-Augmented Generation (RAG) pipeline. Ingests PDF and text documents, indexes them in a vector database, and answers questions using Gemma 2B running on-device via Ollama. No API keys, no cloud calls.

---

## Architecture

```
Documents (PDF / TXT)
        │
        ▼
 ┌─────────────────┐
 │  Document Loader│  LangChain TextLoader / PyMuPDFLoader
 └────────┬────────┘
          │
          ▼
 ┌─────────────────┐
 │  Text Splitter  │  RecursiveCharacterTextSplitter
 │  chunk=1000     │  chunk_size=1000, overlap=200
 │  overlap=200    │
 └────────┬────────┘
          │
          ▼
 ┌─────────────────┐
 │ EmbeddingManager│  sentence-transformers/all-MiniLM-L6-v2
 │  384-dim vectors│  (local, no API)
 └────────┬────────┘
          │
          ▼
 ┌─────────────────┐
 │   VectorStore   │  ChromaDB — persisted to data/vector_store/
 │   (ChromaDB)    │  collection: pdf_documents
 └────────┬────────┘
          │
    query │
          ▼
 ┌─────────────────┐
 │  RAGRetriever   │  embeds query → cosine similarity search → top-k chunks
 └────────┬────────┘
          │
          ▼
 ┌─────────────────┐
 │  build_rag_     │  injects retrieved chunks as [Source N] context blocks
 │  prompt()       │
 └────────┬────────┘
          │
          ▼
 ┌─────────────────┐
 │  GemmaOllama    │  Gemma 2B-IT via Ollama HTTP API (localhost:11434)
 │  (LLM)         │  temperature=0, deterministic output
 └────────┬────────┘
          │
          ▼
      Answer + Sources
```

---

## Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| **Embedding model** | `all-MiniLM-L6-v2` (384-dim) | Small, fast, runs fully local, good retrieval quality for English docs |
| **Vector DB** | ChromaDB (persistent) | Zero-config local setup, persists to disk, no server needed |
| **Chunk size** | 1000 chars / 200 overlap | Balances context per chunk vs. retrieval precision; overlap prevents boundary splits losing context |
| **LLM** | Gemma 2B-IT via Ollama | Fully local, no API key, instruction-tuned for Q&A, fits in consumer RAM |
| **LLM serving** | Ollama (HTTP API) | Simple pull-and-run, handles model loading/quantization automatically, no Python ML deps for inference |
| **Prompt strategy** | Context-grounded only | Instructs model to answer from retrieved sources only — reduces hallucination |
| **Similarity metric** | Cosine (ChromaDB default) | Standard for sentence-transformer embeddings; distance converted to score as `1 - distance` |
| **top-k retrieval** | 3 chunks default | Enough context for most questions without exceeding Gemma 2B's context window |

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Ollama with Gemma 2B

```bash
ollama pull gemma:2b
ollama serve          # auto-starts on Mac
```

### 3. Run the notebook

```bash
jupyter notebook notebook/document.ipynb
```

Run all cells in order. The notebook:
1. Loads documents from `data/text_files/` and `data/pdf_files/`
2. Splits, embeds, and indexes them into ChromaDB
3. Connects to Gemma 2B via Ollama
4. Answers questions grounded in the indexed documents

---

## Project Structure

```
ytrag/
├── notebook/
│   └── document.ipynb    # full pipeline — data ingestion + RAG + LLM
├── data/
│   ├── text_files/       # raw .txt documents
│   ├── pdf_files/        # raw .pdf documents
│   └── vector_store/     # ChromaDB persistence (286 docs indexed)
├── requirements.txt
├── main.py               # stub (not yet wired)
└── LICENSE
```

---

## Requirements

- Python 3.12+
- [Ollama](https://ollama.com/download) with `gemma:2b` pulled
- `pip install -r requirements.txt`

---

## License

MIT — see [LICENSE](LICENSE)
