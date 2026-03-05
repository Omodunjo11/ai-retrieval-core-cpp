AI Retrieval Core (C++)
Overview
This project implements a CPU-based vector search engine in C++ simulating production-scale embedding retrieval workloads.
It performs brute-force similarity search over:
100,000 vectors
768 dimensions (BERT-scale embeddings)
Top-K nearest neighbor retrieval
The system was built to explore performance bottlenecks in AI retrieval pipelines and understand the impact of memory bandwidth, threading, and compiler optimizations.
Architecture
Brute-force dot product similarity
Compiler auto-vectorization (ARM NEON on Apple Silicon)
Multithreaded scanning across CPU cores
Per-thread Top-K heap aggregation
High-resolution latency benchmarking
Benchmark Results (Apple Silicon, 10 cores)
Configuration	Latency
Single-thread scalar	~5.1 ms
Manual NEON SIMD	~9.6 ms (compiler already auto-vectorized)
Multithreaded brute-force	~3.2 ms
Multithreaded + per-thread heaps	~3.4 ms
Key Findings
Compiler auto-vectorization on ARM was highly effective; manual SIMD did not improve performance.
Multithreading produced sublinear scaling (~1.6× on 10 cores) due to memory bandwidth saturation.
Sorting was not the dominant cost; dot product compute and memory movement dominated latency.
The workload is memory-bandwidth bound rather than compute-bound.
Why This Matters
Embedding retrieval is central to:
RAG systems
Semantic search
Recommendation engines
AI tutoring systems
Fraud pattern matching
Understanding memory ceilings and scaling limits is critical for production ML infrastructure.
