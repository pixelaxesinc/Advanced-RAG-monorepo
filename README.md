# 🚀 Advanced RAG System

A production-ready **Retrieval-Augmented Generation (RAG)** platform featuring local LLM inference, hybrid retrieval, multi-agent orchestration, semantic caching, and full observability.

![Architecture](https://img.shields.io/badge/Architecture-Microservices-blue)
![Python](https://img.shields.io/badge/Python-3.11+-green)
![Docker](https://img.shields.io/badge/Docker-Compose-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Features

- **🤖 Local LLM Inference** - Run models locally with vLLM (no API costs for development)
- **🌐 Cloud LLM Fallback** - Route complex queries to OpenRouter (Claude, GPT-4, etc.)
- **🔍 Hybrid Retrieval** - Dense + Sparse vector search with Qdrant
- **📊 Full Observability** - Langfuse tracing with session & user tracking
- **💾 Semantic Caching** - Instant responses for similar queries
- **📄 Multi-Format Ingestion** - PDF, DOCX, HTML, Markdown (+ OCR for images)
- **🎯 OpenAI-Compatible API** - Drop-in replacement for the OpenAI API

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                              Open WebUI                                │
│                           (localhost:3000)                             │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                         RAG Backend (FastAPI)                          │
│                           (localhost:5001)                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │  Semantic   │  │   Query     │  │  Re-Ranker  │  │   Model     │    │
│  │   Cache     │  │  Rewriting  │  │  (Cross-Enc)│  │   Router    │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
└────────────────────────────────────────────────────────────────────────┘
           │                   │                              │
           ▼                   ▼                              ▼
┌─────────────────┐  ┌─────────────────┐           ┌─────────────────────┐
│     Qdrant      │  │      vLLM       │           │    OpenRouter API   │
│  (Vector DB)    │  │  (Local LLM)    │           │   (Cloud Fallback)  │
│  localhost:6333 │  │  localhost:9999 │           │                     │
└─────────────────┘  └─────────────────┘           └─────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│                         Observability Stack                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │  Langfuse   │  │ ClickHouse  │  │    MinIO    │  │    Redis    │    │
│  │  (UI:3001)  │  │   (OLAP)    │  │    (S3)     │  │   (Queue)   │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🐳 Docker Services

| Container             | Image                           | Port      | Purpose                    |
| --------------------- | ------------------------------- | --------- | -------------------------- |
| `rag-open-webui`      | `ghcr.io/open-webui/open-webui` | 3000      | Chat UI (like ChatGPT)     |
| `rag-backend`         | Custom (Dockerfile)             | 5001      | FastAPI RAG orchestrator   |
| `rag-vllm`            | `vllm/vllm-openai`              | 9999      | Local LLM inference        |
| `rag-qdrant`          | `qdrant/qdrant`                 | 6333      | Vector database            |
| `rag-langfuse`        | `langfuse/langfuse:3`           | 3001      | Observability UI           |
| `rag-langfuse-worker` | `langfuse/langfuse-worker:3`    | 3030      | Trace processing           |
| `rag-clickhouse`      | `clickhouse/clickhouse-server`  | 18123     | Trace storage (OLAP)       |
| `rag-minio`           | `minio/minio`                   | 9000/9001 | S3-compatible blob storage |
| `rag-redis`           | `redis:7.2`                     | 6379      | Queue & cache              |
| `rag-langfuse-db`     | `postgres:16`                   | -         | Langfuse metadata DB       |

---

## 🚀 Quick Start

### Prerequisites

- **Docker Desktop** (with GPU support for vLLM)
- **NVIDIA GPU** (Recommended, 8GB+ VRAM)
- **Git**

### 1. Clone & Configure

```bash
git clone https://github.com/yourusername/Advanced-RAG.git
cd Advanced-RAG

# Copy environment template
cp .env.example .env
# Edit .env with your API keys (OpenRouter, etc.)
```

### 2. Start All Services

```bash
docker compose up -d
```

### 3. Access the UI

- **Chat UI**: http://localhost:3000 (Open WebUI)
- **Langfuse Dashboard**: http://localhost:3001
- **API Docs**: http://localhost:5001/docs

### Default Langfuse Credentials

- Email: `admin@rag.local`
- Password: `ragadmin123`

---

## ⚙️ Environment Variables

| Variable             | Description                                | Default                      |
| -------------------- | ------------------------------------------ | ---------------------------- |
| `OPENROUTER_API_KEY` | API key for cloud LLM fallback             | Required for cloud models    |
| `LOCAL_MODEL_NAME`   | Model to run with vLLM                     | `Qwen/Qwen2.5-0.5B-Instruct` |
| `ENABLE_OCR`         | Enable OCR for image files (GPU intensive) | `false`                      |
| `LANGFUSE_DEBUG`     | Enable Langfuse debug logging              | `false`                      |
| `WEBUI_SECRET_KEY`   | Secret for Open WebUI sessions             | Set in compose               |

See `.env.example` for the full list.

---

## 📁 Project Structure

```
Advanced-RAG/
├── src/
│   ├── main.py                 # FastAPI app & endpoints
│   ├── config.py               # Model & provider configuration
│   ├── ingestion/              # Document processing pipeline
│   │   ├── router.py           # Ingestion orchestrator
│   │   ├── docling_parser.py   # PDF/DOCX parser
│   │   ├── deepseek_ocr.py     # OCR for images (optional)
│   │   ├── metadata.py         # LLM-based metadata extraction
│   │   └── chunking.py         # Hierarchical chunking
│   ├── retrieval/              # Search & retrieval
│   │   ├── engine.py           # Query rewriting, HyDE
│   │   ├── qdrant_client.py    # Vector DB operations
│   │   └── reranker.py         # Cross-encoder reranking
│   ├── generation/             # Response generation
│   │   ├── agents.py           # Multi-agent orchestration
│   │   ├── router.py           # Model routing (local/cloud)
│   │   └── semantic_cache.py   # Query caching
│   └── observability/          # Monitoring
│       └── config.py           # Langfuse setup
├── docker-compose.yml          # All services
├── Dockerfile                  # RAG backend image
├── pyproject.toml              # Python dependencies
└── requirements.txt            # Pip dependencies
```

---

## 🔄 How It Works

### Ingestion Pipeline (Upload a Document)

1. **File Detection** → Route to Docling (PDF/DOCX) or OCR (images)
2. **Text Extraction** → Preserve structure (tables, headers)
3. **Metadata Enrichment** → LLM extracts department, date, summary
4. **Hierarchical Chunking** → Parent (1024 tok) + Child (256 tok) chunks
5. **Vector Upsert** → Dense + Sparse embeddings to Qdrant

### Query Pipeline (Ask a Question)

1. **Semantic Cache Check** → Return cached answer if similarity > 0.95
2. **Query Rewriting** → Expand ambiguous queries
3. **Hybrid Search** → Dense (semantic) + Sparse (keyword) in Qdrant
4. **Re-ranking** → Cross-encoder scores top 50 → keep top 5
5. **Model Routing** → Simple → Local vLLM, Complex → OpenRouter
6. **Response Generation** → Stream answer with context
7. **Cache Update** → Store Q&A for future queries

---

## 📊 Observability (Langfuse)

Access the Langfuse dashboard at http://localhost:3001

### Features

- **Traces** - Full execution path for each request
- **Sessions** - Group traces by conversation (chat thread)
- **Users** - Track usage per user
- **Costs** - Token usage and cost breakdown
- **Scores** - User feedback (thumbs up/down)

### Session Tracking

Open WebUI automatically sends session headers when `ENABLE_OPENWEBUI_USER_HEADERS=true`:

- `X-OpenWebUI-Chat-Id` → Groups all messages in a conversation
- `X-OpenWebUI-User-Id` → Links traces to users

---

## 🛠️ Development

### Running Locally (without Docker)

```bash
# Install dependencies
pip install poetry
poetry install

# Start backend
poetry run uvicorn src.main:app --reload --port 8000
```

### Adding New Models

Edit `src/config.py` to add new models:

```python
ModelConfig(
    id="your-model-id",
    name="Display Name",
    provider=Provider.OPENROUTER,  # or Provider.VLLM
    context_window=8192,
)
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [LlamaIndex](https://www.llamaindex.ai/) - RAG framework
- [vLLM](https://github.com/vllm-project/vllm) - Fast LLM inference
- [Qdrant](https://qdrant.tech/) - Vector database
- [Langfuse](https://langfuse.com/) - LLM observability
- [Open WebUI](https://openwebui.com/) - Chat interface
- [Docling](https://github.com/DS4SD/docling) - Document parsing
