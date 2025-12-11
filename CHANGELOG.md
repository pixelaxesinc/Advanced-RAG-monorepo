# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial release of Advanced RAG monorepo
- Multi-container Docker Compose setup
- Local LLM inference with vLLM
- Cloud LLM fallback via OpenRouter
- Hybrid retrieval (Dense + Sparse) with Qdrant
- Semantic caching for instant responses
- Full observability with Langfuse v3
- Session and user tracking for conversation grouping
- Open WebUI integration for chat interface
- Document ingestion pipeline with Docling
- Optional OCR support (configurable via `ENABLE_OCR`)
- Hierarchical chunking strategy
- Cross-encoder reranking
- Query rewriting and HyDE

### Infrastructure

- Langfuse v3 with ClickHouse backend
- MinIO for S3-compatible blob storage
- Redis for queue and cache operations
- PostgreSQL for Langfuse metadata
- Qdrant for vector storage

### Documentation

- Comprehensive README with architecture diagram
- Environment variable documentation
- Contributing guidelines
- MIT License

## [0.1.0] - 2024-12-11

### Added

- Initial project setup
- Core RAG pipeline implementation
- Docker Compose orchestration
