# Knowledge Graph Enhanced Retrieval-Augmented Generation (KG-RAG)

A Knowledge Graph Enhanced Retrieval-Augmented Generation system designed for multi-hop question answering.

The project enhances a traditional Retrieval-Augmented Generation (RAG) pipeline by combining dense document retrieval, knowledge graph-based context expansion, neural reranking, context filtering, supporting-fact extraction, and LLM-based answer generation.

---

## Overview

Traditional RAG systems primarily retrieve documents based on semantic similarity between the user query and the available documents.

However, multi-hop questions often require information to be connected across multiple documents. Retrieving individually relevant documents may not be sufficient to identify the relationships required to answer such questions correctly.

This project addresses this challenge by integrating a **Knowledge Graph (KG)** into the retrieval pipeline.

The KG-RAG system first performs broad semantic retrieval, then uses knowledge graph relationships to expand the retrieved context. The resulting candidates are reranked using a neural reranker, filtered and deduplicated, and finally passed to an LLM for answer generation.

The system was evaluated on the **HotpotQA** benchmark and demonstrated improvements in answer quality and supporting-fact precision compared with the baseline retrieval approach.

---

## Key Features

- Dense vector-based document retrieval
- Knowledge graph-enhanced context expansion
- Neural reranking using BGE Reranker
- Context filtering and deduplication
- Supporting-fact extraction
- LLM-based answer generation using Ollama
- Multi-hop question answering support
- Evaluation using HotpotQA
- Retrieval and generation pipeline implemented using LlamaIndex

---

## System Architecture

```text
                         User Query
                              |
                              v
                    +-------------------+
                    | Dense Retrieval   |
                    +-------------------+
                              |
                              v
                    +-------------------+
                    | Knowledge Graph   |
                    | Context Expansion |
                    +-------------------+
                              |
                              v
                    +-------------------+
                    | Neural Reranking  |
                    |   BGE Reranker    |
                    +-------------------+
                              |
                              v
                    +-------------------+
                    | Context Filtering |
                    | & Deduplication   |
                    +-------------------+
                              |
                              v
                    +-------------------+
                    | LLM Response      |
                    | Generation        |
                    +-------------------+
                              |
                              v
                   Answer + Supporting Facts
---

## Retrieval Pipeline

The KG-RAG pipeline consists of the following stages:

### 1. Dense Retrieval

The system initially retrieves a broad set of potentially relevant documents using vector-based semantic retrieval.

### 2. Knowledge Graph Expansion

Knowledge graph relationships are used to identify additional contextual information associated with the retrieved content.

This helps connect information that may be distributed across multiple documents.

### 3. Neural Reranking

Retrieved candidates are reranked using a BGE-based neural reranker to improve the relevance ordering of the retrieved context.

### 4. Context Filtering

The retrieved results are filtered and deduplicated to reduce redundant information and retain the most useful context.

### 5. Response Generation

The refined context is passed to an LLM through Ollama to generate the final answer.

### 6. Supporting-Fact Extraction

The system extracts and deduplicates supporting facts associated with the generated response.
---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core implementation |
| LlamaIndex | Retrieval and query pipeline |
| Ollama | Local LLM and embedding integration |
| BGE Reranker | Neural reranking |
| Knowledge Graphs | Context expansion |
| Vector Retrieval | Semantic document retrieval |
| HotpotQA | Multi-hop QA evaluation |
