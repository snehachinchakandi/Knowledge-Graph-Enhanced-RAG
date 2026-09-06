# Knowledge Graph Enhanced Retrieval-Augmented Generation (KG-RAG)

A Knowledge Graph Enhanced Retrieval-Augmented Generation system for multi-hop question answering. The project combines dense retrieval, knowledge graph-based context expansion, neural reranking, and LLM-based answer generation to improve retrieval quality and supporting-fact selection.

## Overview

Traditional Retrieval-Augmented Generation (RAG) systems retrieve relevant documents based primarily on semantic similarity. However, multi-hop questions often require connecting information across multiple documents.

This project enhances the retrieval pipeline by incorporating a knowledge graph to expand and refine the retrieved context before generating the final answer.

The KG-RAG pipeline consists of:

1. Dense document retrieval
2. Knowledge graph-based context expansion
3. Neural reranking
4. Deduplication and context filtering
5. LLM-based response generation
6. Supporting-fact extraction

## Key Features

- Dense vector-based document retrieval
- Knowledge graph-enhanced retrieval
- Neural reranking using BGE Reranker
- Context filtering and deduplication
- Supporting-fact extraction
- LLM-based answer generation using Ollama
- Designed for multi-hop question answering
- Evaluation on the HotpotQA benchmark

## System Pipeline

```text
User Query
    ↓
Dense Retrieval
    ↓
Knowledge Graph Expansion
    ↓
Neural Reranking
    ↓
Context Filtering & Deduplication
    ↓
LLM Response Generation
    ↓
Answer + Supporting Facts
