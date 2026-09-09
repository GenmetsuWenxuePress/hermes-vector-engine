# Hermes Vector Engine — Architecture & Design Whitepaper

Production-grade semantic vector indexing & hybrid retrieval architecture for AI Agent systems.

---

## 1. System Topology

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. Data Ingestion Layer (Multi-Profile Aware)                                          │
│    ├── Primary Profile (~/.hermes/) ──► Session DB / JSONL dumps / Memories / Skills   │
│    └── Named Profiles (~/.hermes/profiles/*) ──► Isolated Profiles / Plans / Memories │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │ (AST-aware Chunking & Profile Source Tagging)
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 2. Indexing Control & Sentinel Layer (Python: scripts/index_all.py)                    │
│    ├── Privacy Sentinel Barrier (Dynamic check for incognito / private tasks)          │
│    └── Incremental Fingerprint Trackers (SQLite-backed mtime/size/message_count)       │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │ (HTTP Streaming Batches of 32 chunks)
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 3. Hardware Acceleration Layer                                                         │
│    ├── Windows / WSL2: Native Vulkan GPU offload (-ngl 99 on Port 8081)                │
│    ├── Linux: NVIDIA CUDA / AMD ROCm direct acceleration                               │
│    └── macOS / CPU: Apple Metal (MPS) or AVX-512 SIMD execution                       │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │ (1024 * float32 binary blobs)
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 4. Dual Storage & Index Layer                                                          │
│    ├── SQLite vector_store.db: Persistent binary blobs, metadata, source provenance    │
│    └── FAISS vector_index.faiss: FlatIP in-memory exact cosine similarity (<150μs)     │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │ (Tiered Fallback: FTS5 Exact + Vector Supplement)
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 5. Runtime Consumer Layer                                                              │
│    ├── session_search Tool: Tiered Fallback (FTS5 keyword first + Vector supplement)   │
│    ├── CLI search: Multi-kind & score-threshold filtered queries                       │
│    └── Standalone SDK: Custom AI Agent integration                                     │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. The 4-Dimension Evolution

### Dimension 1: Runtime Multi-Profile Namespaces & Confidence Gating
- **Problem**: In multi-agent or multi-profile systems, session identifiers carry profile prefixes (`profile:ops:active_session:...`). Naive string splitting truncates the session ID to `"profile"`.
- **Solution**: Robust regex extraction `(?:active_session:|session:)([\w\d_-]+)` extracts the true session ID while preserving profile namespace binding.
- **Noise Control**: Automatically gates vector results with similarity threshold `>= 0.70`.

### Dimension 2: Markdown & Code AST-Aware Chunking
- **Problem**: Fixed-character or naive paragraph splitting breaks code fences (` ``` `) and markdown tables (`| ... |`) mid-statement, destroying syntactic validity and header context.
- **Solution**: Block-level AST parser detects atomic components and ensures tables and functions remain unbroken units, while inheriting the active Markdown section heading.

### Dimension 3: Tiered Fallback & Direct CLI Precision
- **Tiered Fallback Pattern**:
  1. Exact keyword search (FTS5) for specific symbols/error codes (0 GPU overhead).
  2. Dense vector supplement triggered only when keyword hits are sparse or conceptual.
- **CLI Precision**: Native `--kind [session|skill|memory|plan]` and `--min-score <float>` flags.

### Dimension 4: Storage Lifecycle & Automated GC
- **Incremental Vacuuming**: Built-in `vacuum` command and weekly WAL checkpoint integration reclaiming deleted vector disk space.
