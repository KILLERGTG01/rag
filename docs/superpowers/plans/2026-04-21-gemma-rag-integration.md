# Gemma 2B RAG Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the existing ChromaDB retrieval pipeline with Gemma 2B (instruction-tuned) to produce grounded, context-aware answers from retrieved document chunks.

**Architecture:** The existing `EmbeddingManager` → `VectorStore` → `RAGRetriever` pipeline stays unchanged. New layers are: `PromptBuilder` (formats retrieved context + query into Gemma chat template), `LLMGenerator` (loads Gemma 2B-IT via HuggingFace Transformers and runs inference), and `RAGPipeline` (orchestrator that chains retrieval → prompt → generation). A demo notebook shows end-to-end usage.

**Tech Stack:** `transformers`, `torch`, `accelerate` (new); `chromadb`, `sentence-transformers`, `langchain` (existing); Python 3.12; `python-dotenv` for HF token.

---

## File Map

| Action   | Path                          | Responsibility                                      |
|----------|-------------------------------|-----------------------------------------------------|
| Create   | `src/__init__.py`             | Package marker                                      |
| Create   | `src/embedding_manager.py`    | `EmbeddingManager` class (extracted from notebook)  |
| Create   | `src/vector_store.py`         | `VectorStore` class (extracted from notebook)       |
| Create   | `src/rag_retriever.py`        | `RAGRetriever` class (extracted from notebook)      |
| Create   | `src/prompt_builder.py`       | Format context + query into Gemma chat template     |
| Create   | `src/llm_generator.py`        | Load Gemma 2B-IT, run inference                     |
| Create   | `src/rag_pipeline.py`         | Orchestrate retrieval → prompt → generation         |
| Modify   | `requirements.txt`            | Add `torch`, `transformers`, `accelerate`, `python-dotenv` |
| Modify   | `.env`                        | Add `HF_TOKEN=your_token_here`                      |
| Create   | `notebook/rag_with_llm.ipynb` | End-to-end demo notebook                            |

---

## ⚠️ Prerequisites (Do Before Task 1)

Gemma 2B is a **gated model**. Before running any code:

1. Go to `https://huggingface.co/google/gemma-2b-it` and accept the license agreement.
2. Get your HuggingFace token from `https://huggingface.co/settings/tokens`.
3. Add to `.env`: `HF_TOKEN=hf_your_token_here`

Hardware requirements:
- **CPU-only Mac**: Works but slow (~30s/response). Uses `torch.float32` automatically.
- **GPU (CUDA/MPS)**: Faster. `device_map="auto"` handles placement.

---

## Task 1: Update Dependencies

**Files:**
- Modify: `requirements.txt`
- Modify: `.env`

- [ ] **Step 1: Update requirements.txt**

Replace contents of `requirements.txt` with:

```
langchain
langchain-core
langchain-community
pypdf
pymupdf
sentence-transformers
faiss-cpu
chromadb
torch
transformers
accelerate
python-dotenv
scikit-learn
```

- [ ] **Step 2: Install new dependencies**

```bash
cd /Users/anurag/development/rag_projects/ytrag
source .venv/bin/activate
pip install torch transformers accelerate python-dotenv scikit-learn
```

Expected: All packages install without error. `transformers` version should be >= 4.38.0 (Gemma support).

Verify:
```bash
python -c "import transformers; print(transformers.__version__)"
```
Expected: prints version >= `4.38.0`

- [ ] **Step 3: Add HF_TOKEN to .env**

Open `.env` and add (replace with your real token):
```
HF_TOKEN=hf_your_token_here
```

- [ ] **Step 4: Commit**

```bash
git add requirements.txt .env
git commit -m "chore: add transformers, torch, accelerate for Gemma 2B integration"
```

---

## Task 2: Extract EmbeddingManager to src/

**Files:**
- Create: `src/__init__.py`
- Create: `src/embedding_manager.py`

- [ ] **Step 1: Create src package**

```bash
mkdir -p /Users/anurag/development/rag_projects/ytrag/src
touch /Users/anurag/development/rag_projects/ytrag/src/__init__.py
```

- [ ] **Step 2: Write failing test**

Create `tests/__init__.py` and `tests/test_embedding_manager.py`:

```bash
mkdir -p /Users/anurag/development/rag_projects/ytrag/tests
touch /Users/anurag/development/rag_projects/ytrag/tests/__init__.py
```

`tests/test_embedding_manager.py`:
```python
import numpy as np
import pytest
from src.embedding_manager import EmbeddingManager


def test_generates_embeddings_with_correct_shape():
    manager = EmbeddingManager()
    embeddings = manager.generate_embeddings(["hello world", "test text"])
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (2, 384)


def test_single_text_embedding():
    manager = EmbeddingManager()
    embeddings = manager.generate_embeddings(["single sentence"])
    assert embeddings.shape[0] == 1
```

- [ ] **Step 3: Run test to verify it fails**

```bash
cd /Users/anurag/development/rag_projects/ytrag
source .venv/bin/activate
python -m pytest tests/test_embedding_manager.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.embedding_manager'`

- [ ] **Step 4: Create src/embedding_manager.py**

```python
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List


class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=False)
```

- [ ] **Step 5: Run test to verify it passes**

```bash
python -m pytest tests/test_embedding_manager.py -v
```

Expected: `2 passed`

- [ ] **Step 6: Commit**

```bash
git add src/__init__.py src/embedding_manager.py tests/__init__.py tests/test_embedding_manager.py
git commit -m "feat: extract EmbeddingManager into src module with tests"
```

---

## Task 3: Extract VectorStore and RAGRetriever to src/

**Files:**
- Create: `src/vector_store.py`
- Create: `src/rag_retriever.py`
- Create: `tests/test_rag_retriever.py`

- [ ] **Step 1: Write failing tests**

`tests/test_rag_retriever.py`:
```python
import pytest
import tempfile
from src.embedding_manager import EmbeddingManager
from src.vector_store import VectorStore
from src.rag_retriever import RAGRetriever
from langchain_core.documents import Document


@pytest.fixture
def rag_setup(tmp_path):
    persist_dir = str(tmp_path / "vector_store")
    embedding_manager = EmbeddingManager()
    vector_store = VectorStore(
        collection_name="test_collection",
        persist_directory=persist_dir
    )
    docs = [
        Document(page_content="Python is a programming language.", metadata={"source": "test.txt"}),
        Document(page_content="ChromaDB is a vector database.", metadata={"source": "test.txt"}),
    ]
    import numpy as np
    texts = [d.page_content for d in docs]
    embeddings = embedding_manager.generate_embeddings(texts)
    vector_store.add_documents(docs, embeddings)
    retriever = RAGRetriever(vector_store, embedding_manager)
    return retriever


def test_retrieve_returns_list(rag_setup):
    results = rag_setup.retrieve("What is Python?", top_k=1)
    assert isinstance(results, list)
    assert len(results) == 1


def test_retrieve_result_has_required_keys(rag_setup):
    results = rag_setup.retrieve("What is Python?", top_k=1)
    assert "content" in results[0]
    assert "similarity_score" in results[0]
    assert "metadata" in results[0]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_rag_retriever.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.vector_store'`

- [ ] **Step 3: Create src/vector_store.py**

```python
import os
import uuid
import numpy as np
import chromadb
from typing import List, Any


class VectorStore:
    def __init__(self, collection_name: str = "pdf_documents", persist_directory: str = "./data/vector_store"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Document Embeddings for RAG"}
        )

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents and embeddings must match.")

        ids, metadatas, texts, embeddings_list = [], [], [], []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            ids.append(f"doc_{uuid.uuid4().hex[:8]}_{i}")
            metadata = dict(doc.metadata)
            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)
            metadatas.append(metadata)
            texts.append(doc.page_content)
            embeddings_list.append(embedding.tolist())

        self.collection.add(ids=ids, metadatas=metadatas, documents=texts, embeddings=embeddings_list)

    def count(self) -> int:
        return self.collection.count()
```

- [ ] **Step 4: Create src/rag_retriever.py**

```python
from typing import List, Dict, Any
from src.vector_store import VectorStore
from src.embedding_manager import EmbeddingManager


class RAGRetriever:
    def __init__(self, vectorstore: VectorStore, embedding_manager: EmbeddingManager):
        self.vectorstore = vectorstore
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        results = self.vectorstore.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )

        retrieved_docs = []
        if not results["documents"] or not results["documents"][0]:
            return retrieved_docs

        for doc_id, document, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            similarity_score = 1 - distance
            if similarity_score >= score_threshold:
                retrieved_docs.append({
                    "id": doc_id,
                    "content": document,
                    "metadata": metadata,
                    "similarity_score": similarity_score,
                })

        return retrieved_docs
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/test_rag_retriever.py -v
```

Expected: `2 passed`

- [ ] **Step 6: Commit**

```bash
git add src/vector_store.py src/rag_retriever.py tests/test_rag_retriever.py
git commit -m "feat: extract VectorStore and RAGRetriever into src modules with tests"
```

---

## Task 4: Build PromptBuilder

**Files:**
- Create: `src/prompt_builder.py`
- Create: `tests/test_prompt_builder.py`

Gemma 2B-IT uses this chat template:
```
<start_of_turn>user
{message}<end_of_turn>
<start_of_turn>model
```

- [ ] **Step 1: Write failing test**

`tests/test_prompt_builder.py`:
```python
from src.prompt_builder import PromptBuilder


def test_prompt_contains_query():
    builder = PromptBuilder()
    context_docs = [{"content": "Python was created in 1991.", "similarity_score": 0.9}]
    prompt = builder.build(query="When was Python created?", context_docs=context_docs)
    assert "When was Python created?" in prompt


def test_prompt_contains_context():
    builder = PromptBuilder()
    context_docs = [{"content": "Python was created in 1991.", "similarity_score": 0.9}]
    prompt = builder.build(query="When was Python created?", context_docs=context_docs)
    assert "Python was created in 1991." in prompt


def test_prompt_uses_gemma_template():
    builder = PromptBuilder()
    context_docs = [{"content": "Some content.", "similarity_score": 0.9}]
    prompt = builder.build(query="A question?", context_docs=context_docs)
    assert "<start_of_turn>user" in prompt
    assert "<start_of_turn>model" in prompt


def test_empty_context_still_builds_prompt():
    builder = PromptBuilder()
    prompt = builder.build(query="What is Python?", context_docs=[])
    assert "What is Python?" in prompt
    assert "<start_of_turn>user" in prompt
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_prompt_builder.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.prompt_builder'`

- [ ] **Step 3: Create src/prompt_builder.py**

```python
from typing import List, Dict, Any


class PromptBuilder:
    def build(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        if context_docs:
            context_text = "\n\n".join(
                f"[Source {i+1}] {doc['content']}"
                for i, doc in enumerate(context_docs)
            )
            user_message = (
                f"Use the following context to answer the question.\n\n"
                f"Context:\n{context_text}\n\n"
                f"Question: {query}\n\n"
                f"Answer based only on the provided context. If the context does not contain enough information, say so."
            )
        else:
            user_message = f"Question: {query}\n\nAnswer:"

        return (
            f"<start_of_turn>user\n"
            f"{user_message}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_prompt_builder.py -v
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/prompt_builder.py tests/test_prompt_builder.py
git commit -m "feat: add PromptBuilder with Gemma 2B-IT chat template"
```

---

## Task 5: Build LLMGenerator (Gemma 2B-IT)

**Files:**
- Create: `src/llm_generator.py`
- Create: `tests/test_llm_generator.py`

> Note: Tests mock the model to avoid downloading 5GB weights during CI.

- [ ] **Step 1: Write failing test**

`tests/test_llm_generator.py`:
```python
from unittest.mock import MagicMock, patch
import pytest
from src.llm_generator import LLMGenerator


@patch("src.llm_generator.AutoModelForCausalLM.from_pretrained")
@patch("src.llm_generator.AutoTokenizer.from_pretrained")
def test_generate_returns_string(mock_tokenizer_cls, mock_model_cls):
    mock_tokenizer = MagicMock()
    mock_tokenizer.decode.return_value = "<start_of_turn>model\nPython is a language.<end_of_turn>"
    mock_tokenizer.return_value = {"input_ids": MagicMock()}
    mock_tokenizer.__call__ = MagicMock(return_value={"input_ids": MagicMock(to=MagicMock(return_value=MagicMock()))})
    mock_tokenizer_cls.return_value = mock_tokenizer

    mock_model = MagicMock()
    mock_model.generate.return_value = MagicMock()
    mock_model.device = "cpu"
    mock_model_cls.return_value = mock_model

    generator = LLMGenerator.__new__(LLMGenerator)
    generator.tokenizer = mock_tokenizer
    generator.model = mock_model
    generator.device = "cpu"
    generator.max_new_tokens = 256

    result = generator.generate("<start_of_turn>user\nWhat is Python?<end_of_turn>\n<start_of_turn>model\n")
    assert isinstance(result, str)


@patch("src.llm_generator.AutoModelForCausalLM.from_pretrained")
@patch("src.llm_generator.AutoTokenizer.from_pretrained")
def test_generator_strips_prompt_from_output(mock_tokenizer_cls, mock_model_cls):
    mock_tokenizer = MagicMock()
    full_output = "<start_of_turn>user\nWhat is Python?<end_of_turn>\n<start_of_turn>model\nPython is great.<end_of_turn>"
    mock_tokenizer.decode.return_value = full_output
    mock_tokenizer_cls.return_value = mock_tokenizer

    mock_model = MagicMock()
    mock_model.generate.return_value = MagicMock()
    mock_model.device = "cpu"
    mock_model_cls.return_value = mock_model

    generator = LLMGenerator.__new__(LLMGenerator)
    generator.tokenizer = mock_tokenizer
    generator.model = mock_model
    generator.device = "cpu"
    generator.max_new_tokens = 256

    result = generator.generate("<start_of_turn>user\nWhat is Python?<end_of_turn>\n<start_of_turn>model\n")
    assert "What is Python?" not in result
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_llm_generator.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.llm_generator'`

- [ ] **Step 3: Create src/llm_generator.py**

```python
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

load_dotenv()

MODEL_ID = "google/gemma-2b-it"


class LLMGenerator:
    def __init__(self, model_id: str = MODEL_ID, max_new_tokens: int = 256):
        self.max_new_tokens = max_new_tokens
        hf_token = os.getenv("HF_TOKEN")

        if torch.cuda.is_available():
            self.device = "cuda"
            dtype = torch.float16
        elif torch.backends.mps.is_available():
            self.device = "mps"
            dtype = torch.float16
        else:
            self.device = "cpu"
            dtype = torch.float32

        print(f"Loading {model_id} on {self.device} ({dtype})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
            token=hf_token,
        )
        print("Model loaded.")

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        full_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Strip the prompt — return only the model's reply
        marker = "<start_of_turn>model"
        if marker in full_text:
            reply = full_text.split(marker)[-1].strip()
        else:
            reply = full_text[len(prompt):].strip()
        return reply
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_llm_generator.py -v
```

Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/llm_generator.py tests/test_llm_generator.py
git commit -m "feat: add LLMGenerator wrapper for Gemma 2B-IT"
```

---

## Task 6: Build RAGPipeline Orchestrator

**Files:**
- Create: `src/rag_pipeline.py`
- Create: `tests/test_rag_pipeline.py`

- [ ] **Step 1: Write failing test**

`tests/test_rag_pipeline.py`:
```python
from unittest.mock import MagicMock
from src.rag_pipeline import RAGPipeline


def test_pipeline_returns_answer_and_sources():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        {"content": "Python was created in 1991.", "similarity_score": 0.9, "metadata": {"source": "test.txt"}, "id": "doc_1"}
    ]
    mock_generator = MagicMock()
    mock_generator.generate.return_value = "Python was created in 1991 by Guido van Rossum."

    mock_prompt_builder = MagicMock()
    mock_prompt_builder.build.return_value = "<start_of_turn>user\nWhen was Python created?<end_of_turn>\n<start_of_turn>model\n"

    pipeline = RAGPipeline(
        retriever=mock_retriever,
        generator=mock_generator,
        prompt_builder=mock_prompt_builder,
    )
    result = pipeline.query("When was Python created?")

    assert "answer" in result
    assert "sources" in result
    assert isinstance(result["answer"], str)
    assert isinstance(result["sources"], list)
    assert len(result["sources"]) == 1


def test_pipeline_passes_top_k_to_retriever():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []
    mock_generator = MagicMock()
    mock_generator.generate.return_value = "I don't know."
    mock_prompt_builder = MagicMock()
    mock_prompt_builder.build.return_value = "prompt"

    pipeline = RAGPipeline(
        retriever=mock_retriever,
        generator=mock_generator,
        prompt_builder=mock_prompt_builder,
    )
    pipeline.query("Some question?", top_k=3)

    mock_retriever.retrieve.assert_called_once_with("Some question?", top_k=3)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_rag_pipeline.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.rag_pipeline'`

- [ ] **Step 3: Create src/rag_pipeline.py**

```python
from typing import Any, Dict
from src.rag_retriever import RAGRetriever
from src.llm_generator import LLMGenerator
from src.prompt_builder import PromptBuilder


class RAGPipeline:
    def __init__(
        self,
        retriever: RAGRetriever,
        generator: LLMGenerator,
        prompt_builder: PromptBuilder = None,
    ):
        self.retriever = retriever
        self.generator = generator
        self.prompt_builder = prompt_builder or PromptBuilder()

    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        context_docs = self.retriever.retrieve(question, top_k=top_k)
        prompt = self.prompt_builder.build(query=question, context_docs=context_docs)
        answer = self.generator.generate(prompt)
        return {
            "question": question,
            "answer": answer,
            "sources": context_docs,
        }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_rag_pipeline.py -v
```

Expected: `2 passed`

- [ ] **Step 5: Run full test suite**

```bash
python -m pytest tests/ -v
```

Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/rag_pipeline.py tests/test_rag_pipeline.py
git commit -m "feat: add RAGPipeline orchestrator connecting retrieval and Gemma generation"
```

---

## Task 7: End-to-End Demo Notebook

**Files:**
- Create: `notebook/rag_with_llm.ipynb`

> This step downloads Gemma 2B weights (~5GB). Ensure `HF_TOKEN` is set in `.env` and you have accepted the Gemma license on HuggingFace.

- [ ] **Step 1: Create notebook/rag_with_llm.ipynb**

Create a new Jupyter notebook at `notebook/rag_with_llm.ipynb` with these cells:

**Cell 1 — Imports & setup:**
```python
import sys
sys.path.insert(0, "..")  # so src/ is importable from notebook/

from dotenv import load_dotenv
load_dotenv("../.env")

from src.embedding_manager import EmbeddingManager
from src.vector_store import VectorStore
from src.rag_retriever import RAGRetriever
from src.prompt_builder import PromptBuilder
from src.llm_generator import LLMGenerator
from src.rag_pipeline import RAGPipeline
```

**Cell 2 — Build retrieval layer (reuse existing ChromaDB):**
```python
embedding_manager = EmbeddingManager()
vector_store = VectorStore(
    collection_name="pdf_documents",
    persist_directory="../data/vector_store"
)
print(f"Loaded collection with {vector_store.count()} documents")

retriever = RAGRetriever(vector_store, embedding_manager)
```

**Cell 3 — Load Gemma 2B-IT (downloads ~5GB first time):**
```python
# This cell takes 1-3 minutes on first run (model download)
# Subsequent runs load from HuggingFace cache (~seconds)
generator = LLMGenerator(max_new_tokens=256)
```

**Cell 4 — Build pipeline:**
```python
pipeline = RAGPipeline(
    retriever=retriever,
    generator=generator,
)
```

**Cell 5 — Ask a question:**
```python
result = pipeline.query("What is TurboQuant and how does it work?", top_k=3)

print("=" * 60)
print("QUESTION:", result["question"])
print("=" * 60)
print("ANSWER:")
print(result["answer"])
print("=" * 60)
print(f"SOURCES ({len(result['sources'])} retrieved):")
for i, src in enumerate(result["sources"]):
    print(f"\n[{i+1}] Score: {src['similarity_score']:.4f}")
    print(f"     File: {src['metadata'].get('source', 'unknown')}, Page: {src['metadata'].get('page', 'N/A')}")
    print(f"     Content: {src['content'][:200]}...")
```

**Cell 6 — Try another question:**
```python
result2 = pipeline.query("What problem does inner product TurboQuant solve?", top_k=5)
print("ANSWER:", result2["answer"])
```

- [ ] **Step 2: Run notebook end-to-end**

Open Jupyter and run all cells:
```bash
cd /Users/anurag/development/rag_projects/ytrag
source .venv/bin/activate
jupyter notebook notebook/rag_with_llm.ipynb
```

Expected: Cell 3 prints "Model loaded." Cell 5 prints a coherent answer about TurboQuant grounded in retrieved context.

- [ ] **Step 3: Commit**

```bash
git add notebook/rag_with_llm.ipynb
git commit -m "feat: add end-to-end RAG+Gemma 2B demo notebook"
```

---

## Task 8: Wire up main.py CLI

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Update main.py**

```python
import sys
from dotenv import load_dotenv

load_dotenv()

from src.embedding_manager import EmbeddingManager
from src.vector_store import VectorStore
from src.rag_retriever import RAGRetriever
from src.llm_generator import LLMGenerator
from src.rag_pipeline import RAGPipeline


def main():
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is TurboQuant?"

    embedding_manager = EmbeddingManager()
    vector_store = VectorStore(
        collection_name="pdf_documents",
        persist_directory="./data/vector_store",
    )
    retriever = RAGRetriever(vector_store, embedding_manager)
    generator = LLMGenerator(max_new_tokens=256)
    pipeline = RAGPipeline(retriever=retriever, generator=generator)

    result = pipeline.query(question, top_k=3)
    print(f"\nQ: {result['question']}")
    print(f"\nA: {result['answer']}")
    print(f"\nSources: {len(result['sources'])} chunks retrieved")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run a quick smoke test**

```bash
cd /Users/anurag/development/rag_projects/ytrag
source .venv/bin/activate
python main.py "What is TurboQuant?"
```

Expected: prints question, answer from Gemma, and number of retrieved sources.

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "feat: wire RAGPipeline into main.py CLI entry point"
```

---

## Self-Review

**Spec coverage:** ✅ vectordb context pipeline → LLM output covered end-to-end. ✅ Gemma 2B used. ✅ existing ChromaDB/EmbeddingManager/RAGRetriever preserved and refactored into modules. ✅ prompt format uses official Gemma IT chat template. ✅ notebook demo shows full pipeline. ✅ CLI entry point.

**Placeholder scan:** No TBD/TODO in any code block. All steps show exact code. Commands include expected outputs.

**Type consistency:** `RAGRetriever.retrieve()` returns `List[Dict[str, Any]]` — used identically in `RAGPipeline.query()` and `PromptBuilder.build()`. `LLMGenerator.generate()` takes `str`, returns `str` — matches `RAGPipeline` usage throughout.
