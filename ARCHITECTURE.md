# Hermes Vector Engine — Architecture & Design Whitepaper

Production-grade semantic vector indexing & hybrid retrieval architecture for AI Agent systems.

---

## 1. System Topology

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. Data Ingestion Layer (Multi-Profile Aware)                                          │
│    ├── Primary Profile (~/.hermes/) ──► state.db / sessions/*.jsonl / memories / skills│
│    └── Named Profiles (~/.hermes/profiles/*) ──► state.db / memories / skills / plans  │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │ (AST-aware Chunking & Profile Source Tagging)
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 2. Indexing Control Layer (Python index_all.py)                                        │
│    ├── Incognito Sentinel Barrier (/tmp/.hermes-incognito-active)                      │
│    └── Incremental Fingerprint Trackers (SQLite-backed mtime/size/message_count)       │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │ (HTTP Streaming Batches of 32 chunks)
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 3. Hardware Acceleration Layer (Windows llama-server on Port 8081)                     │
│    ├── Vulkan Compute Backend: AMD Radeon / Intel / NVIDIA GPU offload (-ngl 99)       │
│    └── Embedding Model: BGE-M3 (1024-dimensional dense vectors, ~1.36GB RAM)           │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │ (1024 * float32 binary blobs)
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 4. Dual Storage & Index Layer (WSL2 / Linux Native)                                    │
│    ├── SQLite vector_store.db: Persistent binary blobs, metadata, source provenance    │
│    └── FAISS vector_index.faiss: FlatIP in-memory exact cosine similarity (<150μs)     │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │ (FTS5 + Vector Reciprocal Rank Fusion)
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 5. Runtime Consumer Layer                                                              │
│    ├── session_search Tool: Hybrid FTS5 + Vector Fallback                              │
│    └── CLI search: Multi-kind & score-threshold filtered queries                       │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. The 4-Dimension Evolution

### Dimension 1: Runtime Multi-Profile Parsing & Confidence Gating
- **Problem**: In multi-agent or multi-profile systems, session identifiers carry profile prefixes (`profile:ops:active_session:...`). Naive splitting truncates the session ID to `"profile"`.
- **Solution**: Robust regex extraction `(?:active_session:|session:)([\w\d_-]+)` extracts the true session ID while binding the profile namespace.
- **Noise Control**: Automatically gates vector results with similarity threshold `>= 0.70`.

### Dimension 2: Markdown & Code AST-Aware Chunking
- **Problem**: Fixed-character or naive paragraph splitting breaks code fences (` ``` `) and markdown tables (`| ... |`) mid-statement, destroying syntactic validity and header context.
- **Solution**: Block-level AST parser detects atomic components and ensures tables and functions remain unbroken units.

### Dimension 3: Reciprocal Rank Fusion & CLI Filtering
- **Hybrid Fusion Formula**:
  $$RRF(d) = \sum_{m \in M} \frac{w_m}{60 + \text{rank}_m(d)}$$
- **CLI Precision**: Native `--kind [session|skill|memory|plan]` and `--min-score <float>` flags.

### Dimension 4: Storage Lifecycle & Automated GC
- **Incremental Vacuuming**: Built-in `vacuum` command and weekly cron integration reclaiming deleted vector disk space.
