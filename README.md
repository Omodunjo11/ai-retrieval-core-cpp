# AI Retrieval Core (C++)

> Understanding retrieval latency at the systems level, before recommending it at the product level.

**Portfolio:** [lapoodunjo.com/projects/ai-retrieval-core](https://lapoodunjo.com/projects/ai-retrieval-core)

## The problem

Most AI PMs spec RAG systems without understanding latency, cost, and accuracy tradeoffs underneath retrieval. "Fast vector search" is a black box until you benchmark it yourself.

## What this is

A CPU-based vector search engine in C++ simulating production-scale embedding retrieval workloads:

- **100,000 vectors** × **768 dimensions** (BERT-scale embeddings)
- Brute-force dot-product similarity with multithreaded scanning
- Per-thread Top-K heap aggregation
- High-resolution latency benchmarking

## Architecture

- Brute-force dot product similarity
- Compiler auto-vectorization (ARM NEON on Apple Silicon)
- Multithreaded scanning across CPU cores
- Per-thread Top-K heap aggregation

## Benchmark results (Apple Silicon, 10 cores)

| Configuration | Latency |
|---------------|---------|
| Single-thread scalar | ~5.1 ms |
| Manual NEON SIMD | ~9.6 ms (compiler already auto-vectorized) |
| Multithreaded brute-force | ~3.2 ms |
| Multithreaded + per-thread heaps | ~3.4 ms |

## Key findings

- Compiler auto-vectorization on ARM was highly effective; manual SIMD did not improve performance.
- Multithreading produced sublinear scaling (~1.6× on 10 cores) due to **memory bandwidth saturation**.
- The workload is memory-bandwidth bound, not compute-bound.

## Why this matters

Embedding retrieval is central to RAG, semantic search, recommendations, and fraud pattern matching. Understanding memory ceilings and scaling limits is critical for production ML infrastructure decisions.

## Stack

C++ · Vector search · Benchmarking
