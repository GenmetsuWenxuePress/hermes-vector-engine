# Hermes Vector Engine ⚡

<div align="center">

**Industrial-grade local semantic vector search & knowledge hub for Multi-Agent systems and heterogeneous compute topologies**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://python.org)
[![FAISS FlatIP](https://img.shields.io/badge/FAISS-FlatIP%20(<150μs)-orange.svg)](https://github.com/facebookresearch/faiss)
[![BGE-M3](https://img.shields.io/badge/Embedding-BGE--M3%201024d-blueviolet.svg)](https://huggingface.co/BAAI/bge-m3)
[![Multi-Platform](https://img.shields.io/badge/Compute-CUDA%20%7C%20Metal%20%7C%20Vulkan%20%7C%20CPU-red.svg)](https://vulkan.org)

[中文说明文档 (README.md)](README.md) · [Architecture Whitepaper (ARCHITECTURE.md)](ARCHITECTURE.md) · [BGE-M3 Vulkan Setup](references/bge-m3-vulkan-setup.md) · [Hybrid Search Tuning](references/hybrid-search-tuning.md)

</div>

---

## 🌟 1. Vision & Core Philosophy

In autonomous AI agent systems and large language model workflows, **long-term memory consolidation and Knowledge Retrieval (RAG)** serve as the critical "hippocampus" that determines whether an agent can evolve, recall context, and solve complex multi-turn tasks.

**Hermes Vector Engine** is an **industrial-grade, high-reliability, 100% offline local vector retrieval engine** engineered for multi-agent ecosystems and cross-platform compute topologies (Linux servers, macOS Apple Silicon, Windows native, and WSL2 hybrid setups).

### Architectural Principles
* 🔒 **100% Offline & Zero Cost**: Driven by local `BGE-M3` (1024-dimensional dense embeddings). **Zero external API charges, zero token consumption, zero privacy leaks**.
* ⚡ **Ultra-Fast Dual Storage**: Embedded `SQLite (WAL) + FAISS (FlatIP)` architecture. No heavy vector database daemons; sub-millisecond cold start.
* 🧩 **AST-Aware Syntax Chunking**: Block-level Markdown parser protecting code blocks (```` ``` ````) and tables (`|...|`) from mid-statement truncation.
* 🌐 **Multi-Profile & Multi-Tenant Native**: Built-in multi-profile namespace routing with complete isolation and traceable source provenance.
* 🛡️ **Confidence-Gated Anti-Hallucination**: Vector search enforces a strict threshold (`min_score >= 0.70`), filtering weak associations; paired with an agent-level **Tiered Fallback Hybrid Search** pattern for optimal compute efficiency.

---

## 💥 2. The 3 Systemic Pain Points of Agentic RAG

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        Traditional Vector Search / RAG Pain Points                     │
├─────────────────────────┬──────────────────────────────┬───────────────────────────────┤
│ 💥 1. Context Amnesia   │ 💥 2. Broken Code & Tables   │ 💥 3. Multi-Agent Cross-Pollution│
│ Fixed retention cutoffs │ Naive character-length split │ Multiple agents share a flat  │
│ destroy critical context│ cuts functions or markdown   │ vector space; memories        │
│ and history, leading    │ tables in half, destroying   │ contaminate across tasks      │
│ to hallucination loops. │ syntax and table headers.    │ without source isolation.     │
└─────────────────────────┴──────────────────────────────┴───────────────────────────────┘
```

---

## 🏛️ 3. Comprehensive System Topology

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. Multi-Profile Ingestion Layer (WSL2 / Linux / macOS)                                │
│    ├── Primary Profile (~/.hermes/) ──► Active sessions / Archived history / Skills     │
│    └── Named Profiles (~/.hermes/profiles/*) ──► Named profiles / Plans / Memories     │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │ (1. Traversal + AST-Aware Chunking + Tagging)
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 2. Indexing Control & Defense Layer (Python: scripts/index_all.py)                     │
│    ├── Dynamic Privacy Sentinel (Interception barrier for incognito / private tasks)   │
│    └── 4 SQLite Incremental Trackers (mtime/size/count fingerprinting: 0s skip)        │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │ (2. Streaming HTTP batch requests, Batch=32)
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 3. Heterogeneous Compute Acceleration Layer                                            │
│    ├── Windows / WSL2: Native Vulkan GPU offload (-ngl 99 full offload on port :8081)  │
│    ├── Linux: NVIDIA CUDA / AMD ROCm direct hardware acceleration                      │
│    └── macOS / CPU: Apple Metal (MPS) or AVX-512 SIMD execution                       │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │ (3. Sub-millisecond 1024-dim Float32 Blobs)
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 4. Dual-Storage & Spatial Index Layer                                                  │
│    ├── SQLite Database: vector_store.db (Text chunks, 4096-byte Blobs, Metadata)       │
│    └── FAISS Index: vector_index.faiss (FlatIP exact cosine similarity space, <150μs)  │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │ (4. Dense vector cosine search + min-score 0.70)
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 5. Runtime Consumers & Integration Patterns                                            │
│    ├── CLI Search Tool: index_all.py search (--kind filtering & --min-score gating)   │
│    ├── Agent Tiered Hybrid: session_search (FTS5 exact first + Vector confidence fall) │
│    └── Standalone Agent SDK: examples/standalone_rag_demo.py                           │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🧩 4. 7 Core Architectural Pillars

### Pillar 1: Markdown & Code AST-Aware Chunking

```text
❌ Naive Fixed-Character Chunking:
┌──────────────────────────────────────┐  ┌──────────────────────────────────────┐
│ def query_database(sql: str):        │  │     rows = cursor.fetchall()         │
│     conn = sqlite3.connect(db_path)  │  │     return rows                      │  <-- Syntax truncated,
│     cursor = conn.cursor()           │  │                                      │      function body lost
└──────────────────┬───────────────────┘  └──────────────────────────────────────┘
                   └─ Chunk 1 ─── Broken ─► Chunk 2 ─┘

✅ Hermes AST-Aware Chunking:
┌────────────────────────────────────────────────────────────────────────┐
│ ### Database Access Layer                                              │  <-- Header context kept
│ ```python                                                              │
│ def query_database(sql: str):                                          │  <-- Code block atomic
│     conn = sqlite3.connect(db_path)                                   │
│     cursor = conn.cursor()                                             │
│     rows = cursor.fetchall()                                           │
│     return rows                                                        │
│ ```                                                                    │
└────────────────────────────────────────────────────────────────────────┘
```

* **Atomic Blocks**: Code fences (```` ``` ````) and tables (`|...|`) are treated as indivisible atomic units.
* **Context Inheritance**: Chunks automatically inherit the nearest Markdown heading hierarchy.

---

### Pillar 2: Native Multi-Profile Isolation

| Asset Kind | Default Profile Tag | Named Profile Tag (`ops`, etc.) | Search `--kind` |
| :--- | :--- | :--- | :---: |
| **Active Sessions** | `active_session:{sid}:b{bi}_c{ci}` | `profile:ops:active_session:{sid}:b{bi}_c{ci}` | `session` |
| **Dumped JSONL** | `session:{name}:b{bi}_c{ci}` | `profile:ops:session:{name}:b{bi}_c{ci}` | `session` |
| **Skill Overview** | `skill:{name}:body` | `profile:ops:skill:{name}:body` | `skill` |
| **Skill References**| `skill:{name}:ref:{ref_name}` | `profile:ops:skill:{name}:ref:{ref_name}` | `skill` |
| **Persistent Memory**| `memory:{file}:entry_{idx}` | `profile:ops:memory:{file}:entry_{idx}` | `memory` |
| **Architecture Plans**| `plan:{filename}` | `profile:ops:plan:{filename}` | `plan` |

---

### Pillar 3: 5-Dimensional Heterogeneous Knowledge Coverage

1. **Full-history Sessions (`sessions`)**: Connects to the agent session database, retaining full multi-turn context.
2. **Hierarchical Skills & Deep References (`skills`)**: Recursively indexes `SKILL.md` along with nested technical manuals (`references/*.md`).
3. **Persistent Memory (`memory`)**: Core system operational facts, agreements, and user profiles.
4. **Architecture Plans (`plans`)**: Execution blueprints, dependency graphs, and staged deliverables.
5. **Dynamic Privacy Sentinel (`privacy sentinel`)**: Lightweight runtime barrier skipping sensitive/incognito sessions before embedding.

---

### Pillar 4: SQLite Binary Blob + FAISS FlatIP Dual Storage

* **SQLite Table (`vectors`)**: Stores raw text, MD5 fingerprints, source tags, and 4096-byte binary blobs.
* **FAISS Memory Index (`vector_index.faiss`)**: Evaluates `IndexFlatIP` inner product on L2-normalized vectors (<150μs latency).
* **Zero Overhead**: Zero external database daemons required. Complete backups require copying just 2 files.

---

### Pillar 5: Tiered Fallback Hybrid Search Pattern

When integrating with agents, the engine recommends a **Tiered Fallback Architecture** to maximize precision while saving GPU compute:

```
                  [Agent Receives User Query / Troubleshooting Request]
                                           │
                                           ▼
                         ┌───────────────────────────────────┐
                         │ Step 1: FTS5 Exact Keyword Search │ ──► Sufficient Hits? ──► [Return Direct (0 GPU)]
                         └─────────────────┬─────────────────┘           (Yes)
                                           │ (No: Sparse hits / Conceptual)
                                           ▼
                         ┌───────────────────────────────────┐
                         │ Step 2: Trigger Vector Supplement │
                         │ • index_all.py search --kind ...  │
                         │ • Filter min_score < 0.70 noise   │
                         └─────────────────┬─────────────────┘
                                           │
                                           ▼
                         ┌───────────────────────────────────┐
                         │ Step 3: Inject High-Confidence RAG│
                         └───────────────────────────────────┘
```

---

### Pillar 6: Confidence Threshold Gating (`min_score = 0.70`)

```text
Score Range            Relevance Level            Engine Strategy
┌────────────────────────────────────────────────────────────────────────┐
│ 0.85 ~ 1.00  │ 🌟 Exact / High Match  │ Injected directly as authoritative fact│
│ 0.75 ~ 0.85  │ 🔥 Highly Relevant     │ Included as key context                │
│ 0.70 ~ 0.75  │ 💡 Moderately Relevant │ Supplementary background               │
├────────────────────────────────────────────────────────────────────────┤
│ < 0.70       │ ❄️ Low Confidence Noise│ 🛡️ Automatically suppressed             │
└────────────────────────────────────────────────────────────────────────┘
```

---

### Pillar 7: Streaming Batch Commits & Lifecycle GC

* **Streaming Batches**: Commits in batches of 30 files / 32 chunks to avoid long lock contention.
* **Storage GC**: Built-in `vacuum` command and automated WAL checkpoint management.

---

## 🖥️ 5. Cross-Platform Compute Topologies

```
┌────────────────────────────────────────────────────────────────────────┐
│                   Compute Topologies & Acceleration Matrix             │
├──────────────────┬──────────────────┬──────────────────────────────────┤
│ Platform         │ Recommended GPU  │ Engine Mode & Advantage          │
├──────────────────┼──────────────────┼──────────────────────────────────┤
│ Linux / Server   │ NVIDIA CUDA      │ llama-server --ngl 99 (Max perf) │
│ macOS (Apple Sil)│ Apple Metal/MPS  │ llama-server --ngl 99 (Unified)  │
│ Windows / WSL2   │ Windows Vulkan   │ Vulkan GPU Host Bridge (:8081)   │
│ Universal CPU    │ CPU (AVX2/512)   │ llama-server -ngl 0 (Zero GPU)   │
└──────────────────┴──────────────────┴──────────────────────────────────┘
```

---

## 🚀 6. Quick Start & Integration Patterns

### Pattern A: 5-Minute Standalone Demo

```bash
git clone https://github.com/GenmetsuWenxuePress/hermes-vector-engine.git
cd hermes-vector-engine
pip install -r requirements.txt
python3 examples/standalone_rag_demo.py
```

---

### Pattern B: CLI Usage with Local GPU Server

```bash
# 1. Run incremental indexing across all profiles
python3 scripts/index_all.py

# 2. View statistics and distribution
python3 scripts/index_all.py stats

# 3. Search with kind filtering and threshold gating
python3 scripts/index_all.py search --kind skill --min-score 0.75 "WARP Chain Proxy" 3

# 4. Perform SQLite vacuum and reclaim space
python3 scripts/index_all.py vacuum
```

---

## ⚖️ 7. Architecture Design Trade-offs & FAQ

### Q1: Why FAISS FlatIP instead of HNSW?
* For personal knowledge bases and agent memory (< 100k vectors), `FlatIP` completes exact cosine search in **< 0.15ms** with 100% recall and zero index-building overhead.

### Q2: Why set overlap to 0 (`OVERLAP_CHARS = 0`)?
* With AST-aware chunking, each block is semantically complete. Adding artificial overlap introduces redundant fragments that dilute Top-1 precision.

### Q3: Why embedded SQLite Binary Blob instead of Milvus/Qdrant?
* Agents require instant startup, zero background daemon overhead, and portable single-file backups.

---

## 📊 8. Production Scale Benchmarks

The following benchmarks are measured on a production multi-profile setup (encompassing 850+ deep references, full history sessions, totaling 35,189 1024-dim vectors):

| Metric | Measured Physical Result (AMD Ryzen 7 8845H / Radeon 780M) |
| :--- | :--- |
| **FAISS FlatIP Latency** | **< 150 microseconds (0.15ms)** across 35k+ vectors |
| **Embedding Throughput** | **~35 Chunks / sec** (Vulkan GPU Acceleration) |
| **Incremental Fingerprinting** | **< 0.05 seconds** across 300+ files |
| **VACUUM Reclaim** | Reclaimed **8.40MB** fragmented space |
| **RAM Footprint** | **~1.36 GB** stable resident memory |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
Contributions, stars, and pull requests are welcome!
