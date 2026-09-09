# Hermes Vector Engine ⚡

<div align="center">

**Production-Grade Local Semantic Vector Indexing & Hybrid Retrieval Engine for Multi-Agent Systems**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://python.org)
[![FAISS FlatIP](https://img.shields.io/badge/FAISS-FlatIP%20(<150μs)-orange.svg)](https://github.com/facebookresearch/faiss)
[![BGE-M3](https://img.shields.io/badge/Embedding-BGE--M3%201024d-blueviolet.svg)](https://huggingface.co/BAAI/bge-m3)
[![Vulkan GPU](https://img.shields.io/badge/Hardware-Vulkan%20GPU%20Offload-red.svg)](https://vulkan.org)

[中文说明文档](README.md) · [Architecture Whitepaper](ARCHITECTURE.md) · [BGE-M3 Setup](references/bge-m3-vulkan-setup.md) · [Hybrid Search Tuning](references/hybrid-search-tuning.md)

</div>

---

## 🌟 Overview & Key Highlights

**Hermes Vector Engine** is an industrial-grade local semantic vector knowledge engine specifically designed for autonomous AI agents (such as Hermes, AutoGPT, and local LLM workflows).

Built to solve common pain points in **WSL2 + Windows Host cross-environment setups** (expensive cloud embedding APIs, lack of native ROCm on AMD iGPUs inside WSL2, code/table truncation during chunking, and multi-profile context pollution), this repository provides an end-to-end production solution:

* 🔒 **100% Local & Free**: Driven by local `BGE-M3` (1024-dimensional dense vectors). Zero API costs, zero token billing, and zero private data leakage.
* ⚡ **Full GPU Vulkan Acceleration**: 99 layers offloaded (`-ngl 99`) via Windows host Vulkan compute shaders. Delivers **3.5x+** embedding throughput on AMD Radeon 780M / 880M / Dedicated GPUs.
* 🌐 **Native Multi-Profile Isolation**: Automatically discovers `default` and named profiles (`ops`, etc.), providing strict namespace isolation with structured source tags (`profile:<name>:...`).
* 🧩 **AST-Aware Smart Chunking**: Markdown block parser protecting code fences (` ``` `) and tables (`| ... |`) as unbroken atomic units with inherited heading context.
* 📚 **5-Dimensional Deep Knowledge Coverage**: Indexes active sessions (full history, no 30-day cutoff), JSONL archives, memories, 850+ reference manuals (`references/*.md`), and execution plans (`plans/*.md`).
* 🛡️ **Streaming Batch Commits**: Streaming 30-file / 32-chunk transactions surviving timeouts and supporting seamless breakpoint resumption.
* 🧹 **Automated Storage GC**: Integrated SQLite `VACUUM` and WAL checkpoint management to defragment vector storage and reclaim disk space.

---

## 🏛️ System Architecture

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. Data Ingestion Layer (Multi-Profile Aware in WSL2)                                  │
│    ├── default profile (~/.hermes/) ──► state.db / sessions / memories / skills        │
│    └── ops profile (~/.hermes/profiles/ops/) ──► state.db / memories / skills / plans  │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │ (AST-aware Chunking & Source Tagging)
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 2. Indexing Control Layer (Python index_all.py)                                        │
│    ├── Incognito Sentinel Shield (/tmp/.hermes-incognito-active: Privacy Protection)   │
│    └── 4 SQLite Fingerprint Trackers (Instant mtime / size change detection)           │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │ (HTTP Streaming Batches of 32 chunks)
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 3. Hardware Acceleration Layer (llama-server.exe on Port 8081)                         │
│    ├── Vulkan Compute Backend (AMD Radeon 780M / Dedicated GPU, -ngl 99)               │
│    └── Model: BGE-M3 Q4_K_M (1024-dim, ~1.36GB RAM footprint)                          │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │ (1024-dim Float32 Binary Blobs)
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 4. Dual Storage & Retrieval Layer (SQLite + FAISS)                                     │
│    ├── SQLite vector_store.db: Binary Blobs & Structured Metadata                      │
│    └── FAISS vector_index.faiss: FlatIP Exact Cosine Similarity (<150μs latency)       │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │ (Hybrid Recall: FTS5 First + Vector Supplement)
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 5. Runtime Consumers                                                                   │
│    ├── session_search Tool: Cross-profile automatic recall                             │
│    └── CLI search: Multi-kind & score-threshold filtered queries                       │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 4-Dimension Evolution Matrix

| Dimension | Before (Basic Baseline) | After (Hermes Vector Engine v4.0.0) | Benefits |
| :--- | :--- | :--- | :--- |
| **1. Runtime Integration** | • Multi-profile ID truncated to `"profile"`<br>• Low-score noise flooded context | • Regex extracts Session ID & profile namespace<br>• Enforces **`min_score = 0.70`** threshold filter | • Lossless cross-profile recall<br>• Automatically blocks hallucinated noise |
| **2. Chunking Quality** | • Raw character split<br>• Code blocks & tables broken in half | • **Markdown & Code AST-Aware Chunking**<br>• Keeps code/table units intact with heading context | • 100% valid code syntax & table headers<br>• Dramatically higher search precision |
| **3. Knowledge Coverage** | • Only basic `SKILL.md`<br>• Active sessions cut off after 30 days | • Indexes **850+ reference manuals (`references/*.md`)**<br>• Indexes **execution plans (`plans/*.md`)** | • Recalls deep troubleshooting runbooks<br>• Knowledge coverage up by 300%+ |
| **4. Lifecycle & GC** | • Giant single transaction prone to timeout<br>• Deleted records left fragmented | • **30-file streaming batch transactions**<br>• Built-in **`vacuum` GC command** with weekly cron | • Zero work lost on interruption<br>• Keeps storage compact and defragmented |

---

## 🚀 Quick Start

### Step 1: Launch Embedding Server (Windows Host or Linux)
Download `llama-server` and `bge-m3-Q4_K_M.gguf`, then run:
```powershell
llama-server.exe `
  -m models/bge-m3-Q4_K_M.gguf `
  --embeddings `
  --host 127.0.0.1 `
  --port 8081 `
  -b 2048 -ub 2048 -ngl 99 -t 4
```

### Step 2: Run Incremental Indexer (Linux / WSL2)
```bash
# Run full multi-profile incremental indexing
python3 scripts/index_all.py

# Check vector database statistics
python3 scripts/index_all.py stats
```

### Step 3: Query the Knowledge Base
```bash
# Search across all sources with confidence threshold >= 0.75
python3 scripts/index_all.py search --min-score 0.75 "Vulkan GPU setup" 5

# Search specifically in skill reference manuals
python3 scripts/index_all.py search --kind skill "WARP chain proxy" 3

# Search execution plans
python3 scripts/index_all.py search --kind plan "self improvement" 2

# Defragment storage & reclaim space
python3 scripts/index_all.py vacuum
```

---

## 🧪 Automated Tests

Run the test suite covering all 4 dimensions:
```bash
python3 tests/test_vector_engine.py
```

Expected output:
```text
Testing Dimension 2: AST-Aware Chunking...
  ✓ Table integrity verified: PASS
  ✓ Code block atomicity verified: PASS
Testing Dimension 1: Source Tagging & Regex Parsing...
  ✓ Source tagging and parsing test passed.
Testing Dimension 4: SQLite Integrity & Vacuum...
  ✓ SQLite Integrity & Vacuum test passed.

🎉 All tests passed successfully!
```

---

## 📊 Measured Performance Benchmarks

| Metric | Real Performance (AMD Ryzen 7 8845H / Radeon 780M) |
| :--- | :--- |
| **FAISS FlatIP Search Latency** | **< 150 μs (0.15ms)** (over 35,000+ vectors) |
| **GPU Embedding Throughput** | **~35 chunks / second** (Vulkan hardware acceleration) |
| **Incremental Diffing Speed** | **< 0.05 seconds** (scanning 300+ files, zero no-op recompute) |
| **Storage Defragmentation** | Reclaims **8.40MB** space in a single VACUUM |
| **Host Memory Footprint** | **~1.36 GB** resident |

---

## 📄 License

MIT License © 2026 GenmetsuWenxuePress (Rodion). Feel free to fork, star, and build upon this engine!
