# RAG Chatbot Demo (Local, Privacy-First)

## Overview

This repository contains a **simple, end-to-end Retrieval-Augmented Generation (RAG) chatbot** built as a learning and demonstration project.

The goal of this project was to **understand RAG fundamentals by implementing them from scratch**, rather than relying entirely on high-level frameworks. The system ingests documents, prepares them for semantic search, retrieves relevant context at query time, and generates grounded answers using a locally hosted language model.

The implementation is intentionally modular and minimal, focusing on clarity over complexity.

---

## What This Project Demonstrates

- Document ingestion and preprocessing  
- Chunking strategy with overlap  
- Semantic embeddings using sentence transformers  
- Vector similarity search using FAISS  
- Context-grounded answer generation using a local LLM (Ollama)  
- Clear separation between preprocessing, retrieval, and generation  

This project is meant as a **learning artifact and technical foundation**, not a production-ready system.

---

## High-Level Architecture

```
PDF Documents
      ↓
Text Extraction
      ↓
Chunking + Metadata
      ↓
Embeddings
      ↓
FAISS Vector Store
      ↓
User Query
      ↓
Similarity Search
      ↓
Retrieved Context
      ↓
Local LLM (Ollama)
      ↓
Grounded Answer
```

---

## Project Structure

```
src/
 ├── ingest.py    # Loads PDFs and extracts raw text
 ├── rag.py       # Chunking and embedding + FAISS index creation
 └── chat.py      # Query, retrieval, and answer generation

README.md
```

> Generated data (PDFs, chunks, vector indexes) are intentionally excluded from the repository.

---

## Key Design Decisions

### Chunking
- Fixed-size chunks with overlap are used to preserve semantic continuity.
- Chunking is deterministic and metadata-rich.
- Similarity is not computed during chunking; it is introduced later via embeddings.

### Embeddings
- Sentence-Transformer embeddings are used to convert text into vectors.
- The same model is used for both document chunks and user queries.

### Vector Store
- FAISS is used for local vector similarity search.
- Metadata is stored separately to preserve traceability.

### Generation
- A locally hosted LLM (via Ollama) generates answers.
- The prompt explicitly instructs the model to answer **only from retrieved context** to reduce hallucination.

---

## Why Local-First?

This project uses local inference and storage to:
- Avoid sending data to external APIs
- Improve privacy and transparency
- Better understand system behavior and failure modes

This approach aligns well with privacy-sensitive or enterprise contexts.

---

## What This Project Is Not

- Not a production deployment
- Not optimized for large-scale datasets
- Not using advanced agent orchestration yet

These extensions are intentionally left out to keep the learning surface manageable.

---

## Possible Extensions

- Replace FAISS with Pinecone or another managed vector database
- Add document source routing and intent detection
- Add conversation memory for multi-turn dialogue
- Integrate higher-level frameworks such as LlamaIndex
- Add a web UI for interactive use

---

## Motivation

This project was built to:
- Learn RAG concepts deeply rather than abstractly
- Be able to explain system behavior confidently in interviews
- Serve as a foundation for more advanced applied AI projects

---

## Status

✔️ End-to-end RAG pipeline working  
✔️ Local inference validated  
✔️ Grounded responses confirmed  

---

## Disclaimer

This repository reflects a **learning-in-progress** project.  
Design choices favor clarity and understanding over completeness.
