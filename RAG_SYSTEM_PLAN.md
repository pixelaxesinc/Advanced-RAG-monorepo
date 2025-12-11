# Enterprise RAG System Design Plan

Based on the "Enterprise AI Systems Design Guide" and your specified technology stack, this document outlines a comprehensive plan to build a production-ready, scalable, and cost-effective RAG system.

## 1. High-Level Architecture

The system follows a **Hybrid Retrieval-Augmented Generation** architecture with a **Multi-Agent Orchestration** layer. It leverages local compute for cost/privacy (vLLM) and cloud providers for peak intelligence (OpenRouter), managed via a robust Ops stack.

### Technology Stack

| Component         | Technology                     | Role                                                                           |
| :---------------- | :----------------------------- | :----------------------------------------------------------------------------- |
| **Frontend**      | **Open Web UI**                | User interface for chat, history, and settings.                                |
| **LLM Serving**   | **vLLM**                       | High-throughput serving for local open-source models (Llama 3, Mistral, Qwen). |
| **External LLMs** | **OpenRouter**                 | Fallback/Router destination for SOTA models (Claude 3.5, GPT-4o).              |
| **Vector DB**     | **Qdrant**                     | Hybrid storage (Dense Vectors + Sparse/BM25) and Metadata filtering.           |
| **Ingestion**     | **Docling**                    | Advanced document parsing (PDFs, Tables, Layouts).                             |
| **OCR**           | **DeepSeek-OCR**               | Handling scanned documents and handwriting.                                    |
| **Orchestration** | **LlamaIndex** / **LangGraph** | RAG logic, query transformations, and agentic workflows.                       |
| **Observability** | **Langfuse**                   | Tracing, cost tracking, and prompt management.                                 |
| **Evaluation**    | **DeepEval**                   | CI/CD testing for RAG metrics (Faithfulness, Recall, etc.).                    |
| **Packaging**     | **Poetry**                     | Python dependency management.                                                  |
| **Deployment**    | **Docker**                     | Containerization of all services.                                              |

---

## 2. Detailed Component Design

### 2.1. Document Processing Pipeline (ETL)

**Goal:** Handle multi-format ingestion with high fidelity, specifically addressing the "Garbage In, Garbage Out" problem.

1.  **Ingestion Router**:
    - **Digital PDFs/Office Docs**: Route to **Docling**. Docling is excellent at preserving document structure (headers, tables) which is critical for hierarchical chunking.
    - **Scanned/Image-based Docs**: Route to **DeepSeek-OCR**. Implement a voting mechanism if possible (e.g., compare with Tesseract) or rely on DeepSeek's high accuracy for handwriting/complex layouts.
2.  **Metadata Enrichment**:
    - Before chunking, use a small local LLM (via vLLM) to extract metadata: `Document Type`, `Date`, `Department`, `Sensitivity Level`.
    - Attach this metadata to _every_ chunk to enable pre-filtering in Qdrant.
3.  **Chunking Strategy**:
    - **Hierarchical Chunking (Parent-Child)**:
      - Parent: Large chunks (e.g., 1024 tokens) for context.
      - Child: Small chunks (e.g., 256 tokens) for precise retrieval.
      - _Retrieval matches child chunks but returns the parent text to the LLM._
    - **Semantic Chunking**: For narrative text, use embedding distances to split at topic changes rather than fixed token counts.

### 2.2. Retrieval Layer (The "R" in RAG)

**Goal:** Maximize Recall and Precision using Hybrid Search.

1.  **Storage (Qdrant)**:
    - Create a collection with support for **Dense Vectors** (e.g., `text-embedding-3-small` or `bge-m3`) and **Sparse Vectors** (BM25/SPLADE) for keyword matching.
    - _Why?_ Dense vectors miss exact terminology (part numbers, specific acronyms); Sparse vectors catch them.
2.  **Query Processing**:
    - **Query Rewriting**: Convert chatty user input into a search-optimized query.
    - **HyDE (Hypothetical Document Embeddings)**: Generate a fake answer using a fast local model, embed that, and search. Good for ambiguous queries.
3.  **Re-ranking**:
    - Retrieve top 50 results (25 dense + 25 sparse).
    - Use a **Cross-Encoder** (e.g., `bge-reranker-v2-m3` hosted on vLLM or locally) to re-rank and select the top 5-10 chunks.

### 2.3. Generation & Orchestration Layer

**Goal:** Cost-effective intelligence using Model Routing.

1.  **Semantic Caching (Cost Optimization)**:
    - Check **Qdrant** (or Redis) for semantically similar past queries (threshold > 0.95).
    - If hit -> Return cached response immediately (Zero cost).
2.  **Model Router**:
    - Analyze query complexity.
    - **Tier 1 (Simple/Greeting)**: Use **Llama-3-8B-Instruct** (Local vLLM).
    - **Tier 2 (RAG/Summary)**: Use **Mistral-Large** or **Qwen-2.5-72B** (Local vLLM).
    - **Tier 3 (Complex Reasoning/Coding)**: Route to **OpenRouter** (Claude 3.5 Sonnet).
3.  **Multi-Agent System (LangGraph/LlamaIndex)**:
    - **Orchestrator Agent**: Breaks down complex user requests.
    - **Retriever Tool**: Accesses Qdrant.
    - **Math/Code Tool**: Executes Python code if needed.
    - **State Management**: Maintain conversation history and intermediate steps.

### 2.4. Frontend (Open Web UI)

- Deploy **Open Web UI** via Docker.
- Connect it to your backend via an OpenAI-compatible API endpoint (which your orchestration layer will expose).
- Enable features like "Web Search" (via OpenRouter or local tools) and "Document Upload" (triggering the ETL pipeline).

---

## 3. Evaluation & Observability

### 3.1. Observability (Langfuse)

- **Tracing**: Log every step (Retrieval, Re-rank, LLM Call) to Langfuse.
- **Cost Tracking**: Tag calls with model names to visualize Local vs. Cloud costs.
- **User Feedback**: Enable Thumbs Up/Down in Open Web UI and sync it to Langfuse datasets.

### 3.2. Evaluation (DeepEval)

- Create a "Golden Dataset" of Q&A pairs.
- **CI/CD Pipeline**: Run `deepeval` tests on every prompt/pipeline change.
  - **Faithfulness**: Does the answer come from the context?
  - **Answer Relevance**: Did we answer the user's question?
  - **Context Recall**: Did we find the right document?

---

## 4. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)

- [ ] Set up **Docker Compose** for Qdrant, vLLM, Langfuse, and Open Web UI.
- [ ] Initialize **Poetry** project.
- [ ] Build basic **ETL Pipeline** using Docling (PDF -> Text).
- [ ] Implement basic **Dense Search** with Qdrant.

### Phase 2: Advanced RAG (Weeks 3-4)

- [ ] Integrate **DeepSeek-OCR** for complex docs.
- [ ] Implement **Hybrid Search** (Dense + Sparse) in Qdrant.
- [ ] Add **Re-ranking** step (Cross-encoder).
- [ ] Set up **DeepEval** and run baseline metrics.

### Phase 3: Optimization & Agents (Weeks 5-6)

- [ ] Implement **Model Routing** (Local vs. OpenRouter).
- [ ] Add **Semantic Caching**.
- [ ] Develop **Multi-Agent** workflow for complex queries.
- [ ] Deploy to production environment.

---

## 5. Folder Structure

```
my-rag-system/
├── docker-compose.yml       # Services: Qdrant, vLLM, Langfuse, OpenWebUI
├── pyproject.toml           # Poetry dependencies
├── src/
│   ├── ingestion/
│   │   ├── docling_parser.py
│   │   ├── deepseek_ocr.py
│   │   └── chunking.py      # Hierarchical/Semantic chunking
│   ├── retrieval/
│   │   ├── qdrant_client.py
│   │   └── reranker.py
│   ├── generation/
│   │   ├── router.py        # Logic to choose vLLM vs OpenRouter
│   │   └── agents.py        # Multi-agent definitions
│   └── main.py              # API Entrypoint
├── tests/
│   └── evaluation/          # DeepEval test cases
└── .env                     # API Keys (OpenRouter, Langfuse, etc.)
```

## 6. Key Configuration Snippets

### Docker Compose (Simplified)

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports: ["6333:6333"]

  vllm:
    image: vllm/vllm-openai:latest
    environment:
      - MODEL=meta-llama/Meta-Llama-3-8B-Instruct
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    ports: ["3000:8080"]
    environment:
      - OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1 # Point to your API
```

### Poetry Dependencies

```bash
poetry add llama-index qdrant-client docling langfuse deepeval vllm openai
```
