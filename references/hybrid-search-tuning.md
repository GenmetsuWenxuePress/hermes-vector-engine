# Hybrid Search & RRF Ranking Tuning Guide

Understanding how FTS5 keyword search and FAISS dense vector search fuse to create high-precision recall.

---

## 1. Why Hybrid Search Beats Vector-Only or BM25-Only

| Query Pattern | BM25 / FTS5 Alone | Dense Vector Alone | Hybrid Fusion (RRF) |
| :--- | :---: | :---: | :---: |
| Exact symbol / Error code (`WinError 10106`) | 🌟 Perfect (1.00) | ⚠️ May miss exact token | 🌟 1.00 (Exact hit) |
| Conceptual / Synonyms (`核显` vs `780M / GPU`) | ❌ 0.00 (Miss) | 🌟 0.82 (High match) | 🌟 0.85 (High match) |
| Long questions / Natural language | ⚠️ Partial keyword match | 🌟 High semantic match | 🌟 Top-ranked |

---

## 2. Reciprocal Rank Fusion (RRF) Formula

$$RRF(d) = \sum_{m \in M} \frac{w_m}{60 + \text{rank}_m(d)}$$

Where:
- $w_{\text{fts5}} = 1.0$ (Keyword weight)
- $w_{\text{vec}} = 1.0$ (Semantic weight)
- Constant $k = 60$ smooths rank differences

---

## 3. Threshold Guidelines

- **`>= 0.85`**: Direct hit / High-confidence evidence
- **`0.75 - 0.85`**: Strongly related background
- **`0.70 - 0.75`**: Relevant context
- **`< 0.70`**: Low-confidence noise (automatically filtered)
