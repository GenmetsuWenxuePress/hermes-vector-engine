#!/usr/bin/env python3
"""
Hermes All-in-One Vector Indexer (Next-Gen Multi-Profile & Deep Knowledge)
Incrementally indexes:
  1. Active sessions (state.db full-history, incremental)
  2. Dumped sessions (sessions/*.jsonl)
  3. Memory systems (memories/MEMORY.md, USER.md)
  4. Core skills & ALL reference documents (skills/**/SKILL.md + skills/**/references/*.md)
  5. Architecture & execution plans (plans/*.md)
across ALL Hermes profiles (default + ~/.hermes/profiles/*) into a unified FAISS + SQLite store.

Storage: SQLite binary blob (EMBED_DIM * float32) + FAISS FlatIP index
Embedding: Local llama-server on http://127.0.0.1:8081/embedding (BGE-M3, 1024-dim, Vulkan GPU)
Source tagging:
  - Default: active_session:{sid}, session:{name}, memory:{file}, skill:{name}, skill:{name}:ref:{ref}, plan:{file}
  - Named:   profile:{pname}:active_session:{sid}, profile:{pname}:skill:{name}:ref:{ref}, etc.

Usage:
  python3 index_all.py [active_sessions|sessions|memory|skills|plans]
  python3 index_all.py search [--kind session|memory|skill|plan] [--min-score 0.70] <query> [top_k]
  python3 index_all.py vacuum
  python3 index_all.py stats
"""

import os
import sys
import time
import json
import re
import struct
import hashlib
import sqlite3
from pathlib import Path
import numpy as np
import faiss

# ── Paths & Config ──────────────────────────────────────────

HOME = Path.home()
HERMES_ROOT = HOME / ".hermes"
VECTOR_DB = HERMES_ROOT / "vector_store.db"
FAISS_INDEX = HERMES_ROOT / "vector_index.faiss"
EMBED_URL = "http://127.0.0.1:8081/embedding"
INC_SENTINEL = Path("/tmp/.hermes-incognito-active")

# Chunking
MAX_CHARS = 800           # Preferred max chunk size
OVERLAP_CHARS = 0         # No overlap — overlap hurts Recall@1
MIN_CHUNK_CHARS = 30      # Skip chunks shorter than this

# Batch embedding size
EMBED_BATCH_SIZE = 32

# FAISS / Embedding dimension
EMBED_DIM = 1024


# ── Profile Discovery ───────────────────────────────────────

def iter_profiles():
    """Yield (profile_name, hermes_home_path) tuples.
    'default' profile (~/.hermes) comes first, followed by all named profiles under ~/.hermes/profiles/*
    """
    yield "default", HERMES_ROOT
    profiles_root = HERMES_ROOT / "profiles"
    if profiles_root.is_dir():
        for p in sorted(profiles_root.iterdir()):
            if p.is_dir() and not p.name.startswith(".") and (p / "config.yaml").exists():
                yield p.name, p


# ── Embedding ──────────────────────────────────────────────

def _embed_single_chunk_batch(texts):
    """Raw single HTTP call to llama-server."""
    import requests, time
    single = isinstance(texts, str)
    if single:
        texts = [texts]
    texts = [t[:800] for t in texts]
    last_error = None
    for attempt in range(3):
        try:
            r = requests.post(EMBED_URL, json={"content": texts}, timeout=120)
            r.raise_for_status()
            data = r.json()
            results = [item["embedding"][0] for item in data]
            return results[0] if single else results
        except requests.HTTPError as e:
            last_error = e
            if r.status_code >= 500 and attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            raise RuntimeError(f"Embedding server unreachable ({EMBED_URL}): {e}") from e
        except requests.RequestException as e:
            last_error = e
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            raise RuntimeError(f"Embedding server unreachable ({EMBED_URL}): {e}") from e
        except (KeyError, IndexError, ValueError) as e:
            raise RuntimeError(f"Unexpected embedding response format: {e}") from e
    raise RuntimeError(f"Embedding server unreachable after 3 retries ({EMBED_URL}): {last_error}") from last_error


def embed(texts, batch_size=EMBED_BATCH_SIZE):
    """Batch embed texts in chunks of batch_size to ensure smooth GPU throughput and avoid timeouts."""
    if isinstance(texts, str):
        return _embed_single_chunk_batch(texts)
    if not texts:
        return []

    all_vecs = []
    for i in range(0, len(texts), batch_size):
        sub_batch = texts[i:i + batch_size]
        vecs = _embed_single_chunk_batch(sub_batch)
        all_vecs.extend(vecs)
    return all_vecs


# ── Smart Markdown AST-Aware Chunking ──────────────────────

def chunk_text(text, max_chars=MAX_CHARS, overlap=OVERLAP_CHARS, min_chunk_chars=MIN_CHUNK_CHARS):
    """Markdown & Code AST-Aware smart chunking algorithm.
    Protects Code Fences (```) and Markdown Tables (|...|) from mid-block truncation.
    Preserves heading context across chunks.
    """
    if not text or len(text) < min_chunk_chars:
        return []
    if len(text) <= max_chars:
        return [text.strip()]

    lines = text.split("\n")
    blocks = []
    current_block = []
    current_kind = None  # "code", "table", "text"
    current_header = ""

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Track active markdown header
        if stripped.startswith("#") and current_kind != "code":
            if current_block:
                blocks.append((current_kind or "text", "\n".join(current_block), current_header))
                current_block = []
                current_kind = None
            current_header = stripped
            blocks.append(("header", stripped, current_header))
            i += 1
            continue

        # Code block boundary
        if stripped.startswith("```"):
            if current_kind == "code":
                current_block.append(line)
                blocks.append(("code", "\n".join(current_block), current_header))
                current_block = []
                current_kind = None
            else:
                if current_block:
                    blocks.append((current_kind or "text", "\n".join(current_block), current_header))
                    current_block = []
                current_kind = "code"
                current_block.append(line)
            i += 1
            continue

        if current_kind == "code":
            current_block.append(line)
            i += 1
            continue

        # Table block detection (starts and contains |)
        if stripped.startswith("|") and stripped.endswith("|"):
            if current_kind != "table":
                if current_block:
                    blocks.append((current_kind or "text", "\n".join(current_block), current_header))
                    current_block = []
                current_kind = "table"
            current_block.append(line)
            i += 1
            continue
        elif current_kind == "table":
            blocks.append(("table", "\n".join(current_block), current_header))
            current_block = []
            current_kind = None

        # Paragraph / Regular text
        if not stripped:
            if current_block:
                blocks.append((current_kind or "text", "\n".join(current_block), current_header))
                current_block = []
                current_kind = None
        else:
            if current_kind is None:
                current_kind = "text"
            current_block.append(line)

        i += 1

    if current_block:
        blocks.append((current_kind or "text", "\n".join(current_block), current_header))

    # Assemble blocks into chunks <= max_chars
    chunks = []
    accumulated = ""

    for kind, blk_text, hdr in blocks:
        blk_text = blk_text.strip()
        if not blk_text:
            continue

        if kind == "header":
            continue

        # Atomic blocks (Code or Table) <= max_chars
        if kind in ("code", "table") and len(blk_text) <= max_chars:
            if accumulated and len(accumulated) >= min_chunk_chars:
                chunks.append(accumulated.strip())
                accumulated = ""
            chunks.append(blk_text)
            continue

        # Long code block > max_chars -> split strictly by lines
        if kind == "code" and len(blk_text) > max_chars:
            if accumulated and len(accumulated) >= min_chunk_chars:
                chunks.append(accumulated.strip())
                accumulated = ""
            code_lines = blk_text.split("\n")
            sub_code = ""
            for cl in code_lines:
                if len(sub_code) + len(cl) + 1 <= max_chars:
                    sub_code = f"{sub_code}\n{cl}" if sub_code else cl
                else:
                    if len(sub_code) >= min_chunk_chars:
                        chunks.append(sub_code.strip())
                    sub_code = cl
            if len(sub_code) >= min_chunk_chars:
                chunks.append(sub_code.strip())
            continue

        # Regular text
        if len(blk_text) <= max_chars:
            if len(accumulated) + len(blk_text) + 2 <= max_chars:
                accumulated = f"{accumulated}\n\n{blk_text}" if accumulated else blk_text
            else:
                if len(accumulated) >= min_chunk_chars:
                    chunks.append(accumulated.strip())
                accumulated = blk_text
        else:
            if accumulated and len(accumulated) >= min_chunk_chars:
                chunks.append(accumulated.strip())
                accumulated = ""
            sentences = re.split(r"(?<=[.!?。！？\n])\s*", blk_text)
            for sent in sentences:
                if not sent.strip():
                    continue
                if len(accumulated) + len(sent) <= max_chars:
                    accumulated += sent
                else:
                    if len(accumulated) >= min_chunk_chars:
                        chunks.append(accumulated.strip())
                    accumulated = sent

    if len(accumulated) >= min_chunk_chars:
        chunks.append(accumulated.strip())

    return chunks


# ── Database ───────────────────────────────────────────────

def init_db():
    conn = sqlite3.connect(str(VECTOR_DB), timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=60000")
    conn.execute("""CREATE TABLE IF NOT EXISTS vectors (
        id TEXT PRIMARY KEY, text TEXT NOT NULL,
        embedding BLOB NOT NULL,           -- float32 binary, 1024*4 bytes
        source TEXT NOT NULL,
        created_at REAL NOT NULL,
        kind TEXT DEFAULT 'session')""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_vectors_kind ON vectors(kind)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_vectors_source ON vectors(source)")
    conn.execute("""CREATE TABLE IF NOT EXISTS skill_file_tracker (
        skill_name TEXT PRIMARY KEY,
        mtime REAL NOT NULL,
        size INTEGER NOT NULL,
        last_indexed_at REAL NOT NULL)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS session_file_tracker (
        session_name TEXT PRIMARY KEY,
        mtime REAL NOT NULL,
        size INTEGER NOT NULL,
        last_indexed_at REAL NOT NULL)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS memory_file_tracker (
        file_name TEXT PRIMARY KEY,
        mtime REAL NOT NULL,
        size INTEGER NOT NULL,
        last_indexed_at REAL NOT NULL)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS active_session_tracker (
        session_id TEXT PRIMARY KEY,
        message_count INTEGER,
        started_at REAL,
        indexed_at REAL)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS plan_file_tracker (
        plan_name TEXT PRIMARY KEY,
        mtime REAL NOT NULL,
        size INTEGER NOT NULL,
        last_indexed_at REAL NOT NULL)""")
    conn.commit()
    return conn


def dedup_key(source):
    return hashlib.md5(source.encode()).hexdigest()[:12]


# ── FAISS Index ────────────────────────────────────────────

def build_faiss_index(conn):
    """Build exact (FlatIP) index from all vectors. Cosine similarity via L2 normalization."""
    rows = conn.execute(
        "SELECT id, embedding, kind, source, text FROM vectors ORDER BY id"
    ).fetchall()
    if not rows:
        return None

    valid_vectors = []
    valid_ids = []
    id_to_meta = {}

    for doc_id, emb_blob, kind, source, text in rows:
        vec = np.frombuffer(emb_blob, dtype=np.float32)
        if len(vec) != EMBED_DIM:
            continue
        valid_vectors.append(vec)
        valid_ids.append(doc_id)
        id_to_meta[doc_id] = (kind, source, text)

    if not valid_vectors:
        return None

    vectors = np.array(valid_vectors, dtype=np.float32)
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(vectors)

    faiss.write_index(index, str(FAISS_INDEX))
    return index, valid_ids, id_to_meta


def load_faiss_index():
    """Load FAISS index + metadata from disk."""
    if not FAISS_INDEX.exists():
        return None

    index = faiss.read_index(str(FAISS_INDEX))
    conn = sqlite3.connect(str(VECTOR_DB))
    rows = conn.execute(
        "SELECT id, kind, source, text FROM vectors ORDER BY id"
    ).fetchall()
    conn.close()

    ids = [r[0] for r in rows]
    id_to_meta = {r[0]: (r[1], r[2], r[3]) for r in rows}
    return index, ids, id_to_meta


# ── Multi-Profile Indexing Logic ────────────────────────────

def index_memory(conn):
    """Incrementally index MEMORY.md and USER.md across ALL profiles."""
    tracker = {}
    for row in conn.execute("SELECT file_name, mtime, size FROM memory_file_tracker"):
        tracker[row[0]] = {"mtime": row[1], "size": row[2]}

    current_files = {}
    for pname, phome in iter_profiles():
        mem_dir = phome / "memories"
        if not mem_dir.exists():
            continue
        for name in ["MEMORY.md", "USER.md"]:
            fpath = mem_dir / name
            if not fpath.exists():
                continue
            st = fpath.stat()
            t_key = name if pname == "default" else f"profile:{pname}:{name}"
            current_files[t_key] = (pname, name, fpath, st.st_mtime, st.st_size)

    to_index = []
    for t_key, (pname, name, fpath, mtime, size) in current_files.items():
        if t_key in tracker:
            old = tracker[t_key]
            if old["mtime"] == mtime and old["size"] == size:
                continue
        to_index.append((t_key, pname, name, fpath, mtime, size))

    deleted = set(tracker.keys()) - set(current_files.keys())
    for t_key in deleted:
        conn.execute("DELETE FROM vectors WHERE source LIKE ?", (f"{t_key}%",))
        conn.execute("DELETE FROM memory_file_tracker WHERE file_name = ?", (t_key,))

    if not to_index:
        return 0

    chunk_meta = []
    now = time.time()

    for t_key, pname, name, fpath, mtime, size in to_index:
        src_prefix = f"memory:{name}" if pname == "default" else f"profile:{pname}:memory:{name}"
        conn.execute("DELETE FROM vectors WHERE source LIKE ?", (f"{src_prefix}%",))

        text = fpath.read_text(errors='replace')
        entries = [e.strip() for e in text.split("\n§\n") if e.strip() and len(e.strip()) > 20]
        for i, entry in enumerate(entries):
            doc_id = dedup_key(f"{src_prefix}:{i}")
            source = f"{src_prefix}:entry_{i}"
            chunks = chunk_text(entry)
            for ci, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_c{ci}" if ci > 0 else doc_id
                chunk_meta.append((chunk_id, chunk, source))

        conn.execute(
            "INSERT OR REPLACE INTO memory_file_tracker VALUES (?,?,?,?)",
            (t_key, mtime, size, now))

    if not chunk_meta:
        conn.commit()
        return 0

    try:
        chunks_only = [cm[1] for cm in chunk_meta]
        vecs = embed(chunks_only)
    except Exception as e:
        print(f"  ⚠️ memory embed failed: {e}", file=sys.stderr)
        conn.rollback()
        return 0

    rows = []
    for (chunk_id, chunk, source), vec in zip(chunk_meta, vecs):
        blob = struct.pack(f'{EMBED_DIM}f', *vec)
        rows.append((chunk_id, chunk, blob, source, now, "memory"))

    conn.executemany("INSERT INTO vectors VALUES (?,?,?,?,?,?)", rows)
    conn.commit()
    return len(rows)


def index_skills(conn, batch_file_limit=30):
    """Incrementally index SKILL.md AND all references/*.md across ALL profiles with streaming batch commits."""
    tracker = {}
    for row in conn.execute("SELECT skill_name, mtime, size FROM skill_file_tracker"):
        tracker[row[0]] = {"mtime": row[1], "size": row[2]}

    current_files = {}
    for pname, phome in iter_profiles():
        sdir = phome / "skills"
        if not sdir.exists():
            continue

        # 1. 扫描所有主 SKILL.md 文件
        for skill_file in sorted(sdir.rglob("SKILL.md")):
            rel = skill_file.relative_to(sdir)
            st = skill_file.stat()
            t_key = str(rel) if pname == "default" else f"profile:{pname}:{rel}"
            current_files[t_key] = (pname, "main", rel, skill_file, st.st_mtime, st.st_size)

        # 2. 扫描所有 references/*.md 深度参考文档
        for ref_file in sorted(sdir.rglob("references/*.md")):
            rel = ref_file.relative_to(sdir)
            st = ref_file.stat()
            t_key = str(rel) if pname == "default" else f"profile:{pname}:{rel}"
            current_files[t_key] = (pname, "ref", rel, ref_file, st.st_mtime, st.st_size)

    deleted = set(tracker.keys()) - set(current_files.keys())
    to_index = []
    for t_key, (pname, ftype, rel, fpath, mtime, size) in current_files.items():
        if t_key in tracker:
            old = tracker[t_key]
            if old["mtime"] == mtime and old["size"] == size:
                continue
        to_index.append((t_key, pname, ftype, rel, fpath, mtime, size))

    if not deleted and not to_index:
        return 0

    new_total = 0
    now = time.time()

    # 清理已删除的技能文档
    if deleted:
        for t_key in deleted:
            src_clean = f"skill:{t_key}" if not t_key.startswith("profile:") else f"{t_key}"
            conn.execute("DELETE FROM vectors WHERE source LIKE ?", (f"{src_clean}%",))
            conn.execute("DELETE FROM skill_file_tracker WHERE skill_name = ?", (t_key,))
            print(f"  🗑 deleted skill doc: {t_key}", file=sys.stderr)
        conn.commit()

    total_files = len(to_index)
    if total_files > 0:
        print(f"  📚 发现 {total_files} 个新增/修改的技能与参考手册文件，开始流式批处理索引...")

    # 按 batch_file_limit (30个文件) 进行分批解析、嵌入与提交，保证进度可中断自愈
    for start_idx in range(0, total_files, batch_file_limit):
        file_batch = to_index[start_idx:start_idx + batch_file_limit]
        batch_chunk_meta = []

        for t_key, pname, ftype, rel, fpath, mtime, size in file_batch:
            if ftype == "main":
                skill_name = str(rel.parent) if rel.parent != Path('.') else str(rel.stem)
                src_prefix = f"skill:{skill_name}" if pname == "default" else f"profile:{pname}:skill:{skill_name}"
            else:
                skill_name = str(rel.parent.parent) if rel.parent.parent != Path('.') else "root"
                ref_name = fpath.stem
                src_prefix = f"skill:{skill_name}:ref:{ref_name}" if pname == "default" else f"profile:{pname}:skill:{skill_name}:ref:{ref_name}"

            conn.execute("DELETE FROM vectors WHERE source LIKE ? OR source = ?",
                         (f"{src_prefix}%", src_prefix))

            text = fpath.read_text(errors='replace')

            if ftype == "main":
                lines = text.split('\n')
                desc = ""
                in_front = False
                for line in lines:
                    if line.strip() == '---' and not in_front:
                        in_front = True; continue
                    if line.strip() == '---' and in_front:
                        in_front = False; continue
                    if in_front and line.startswith('description:'):
                        desc = line.replace('description:', '').strip().strip('"').strip("'")

                body_start = text.find('\n---\n')
                body = text[body_start+4:].strip() if body_start > 0 else ""

                if desc and len(desc) > 10:
                    doc_id = dedup_key(f"{src_prefix}:desc")
                    source_desc = f"{src_prefix}:desc"
                    batch_chunk_meta.append((doc_id, f"技能 {skill_name}: {desc}", source_desc, t_key, mtime, size))

                if body and len(body) > 30:
                    doc_id = dedup_key(f"{src_prefix}:body")
                    source = f"{src_prefix}:body"
                    full_text = f"技能内容 {skill_name}: {body}"
                    for ci, chunk in enumerate(chunk_text(full_text)):
                        chunk_id = f"{doc_id}_c{ci}" if ci > 0 else doc_id
                        batch_chunk_meta.append((chunk_id, chunk, source, t_key, mtime, size))

            else:
                if text and len(text.strip()) > 30:
                    doc_id = dedup_key(f"{src_prefix}:content")
                    source = f"{src_prefix}"
                    full_text = f"技能参考手册 [{skill_name} / {ref_name}]: {text.strip()}"
                    for ci, chunk in enumerate(chunk_text(full_text)):
                        chunk_id = f"{doc_id}_c{ci}" if ci > 0 else doc_id
                        batch_chunk_meta.append((chunk_id, chunk, source, t_key, mtime, size))

        if batch_chunk_meta:
            chunks_only = [cm[1] for cm in batch_chunk_meta]
            vecs = embed(chunks_only)

            rows = []
            for (chunk_id, chunk, source, t_key, mtime, size), vec in zip(batch_chunk_meta, vecs):
                blob = struct.pack(f'{EMBED_DIM}f', *vec)
                rows.append((chunk_id, chunk, blob, source, now, "skill"))

            conn.executemany("INSERT OR REPLACE INTO vectors VALUES (?,?,?,?,?,?)", rows)
            new_total += len(rows)

        # 批次更新 Tracker 并提交事务
        for t_key, pname, ftype, rel, fpath, mtime, size in file_batch:
            conn.execute(
                "INSERT OR REPLACE INTO skill_file_tracker VALUES (?,?,?,?)",
                (t_key, mtime, size, now))

        conn.commit()
        done_count = min(start_idx + batch_file_limit, total_files)
        print(f"    ✓ 进度: {done_count}/{total_files} 文件完成 (+{len(batch_chunk_meta)} chunks 已存盘)")

    return new_total


def index_plans(conn):
    """Incrementally index plans/*.md across ALL profiles."""
    tracker = {}
    for row in conn.execute("SELECT plan_name, mtime, size FROM plan_file_tracker"):
        tracker[row[0]] = {"mtime": row[1], "size": row[2]}

    current_files = {}
    for pname, phome in iter_profiles():
        pdir = phome / "plans"
        if not pdir.exists():
            continue
        for plan_file in sorted(pdir.glob("*.md")):
            st = plan_file.stat()
            t_key = plan_file.name if pname == "default" else f"profile:{pname}:{plan_file.name}"
            current_files[t_key] = (pname, plan_file.name, plan_file, st.st_mtime, st.st_size)

    deleted = set(tracker.keys()) - set(current_files.keys())
    to_index = []
    for t_key, (pname, pname_file, fpath, mtime, size) in current_files.items():
        if t_key in tracker:
            old = tracker[t_key]
            if old["mtime"] == mtime and old["size"] == size:
                continue
        to_index.append((t_key, pname, pname_file, fpath, mtime, size))

    if not deleted and not to_index:
        return 0

    new = 0
    now = time.time()

    try:
        for t_key in deleted:
            src_prefix = f"plan:{t_key}" if not t_key.startswith("profile:") else f"{t_key}"
            conn.execute("DELETE FROM vectors WHERE source LIKE ?", (f"{src_prefix}%",))
            conn.execute("DELETE FROM plan_file_tracker WHERE plan_name = ?", (t_key,))
            print(f"  🗑 deleted plan: {t_key}", file=sys.stderr)

        chunk_meta = []
        for t_key, pname, pname_file, fpath, mtime, size in to_index:
            src_prefix = f"plan:{pname_file}" if pname == "default" else f"profile:{pname}:plan:{pname_file}"
            conn.execute("DELETE FROM vectors WHERE source LIKE ?", (f"{src_prefix}%",))

            text = fpath.read_text(errors='replace')
            full_text = f"工程施工图计划 [{pname_file}]: {text.strip()}"
            doc_id = dedup_key(src_prefix)

            for ci, chunk in enumerate(chunk_text(full_text)):
                chunk_id = f"{doc_id}_c{ci}" if ci > 0 else doc_id
                chunk_meta.append((chunk_id, chunk, src_prefix))

            conn.execute(
                "INSERT OR REPLACE INTO plan_file_tracker VALUES (?,?,?,?)",
                (t_key, mtime, size, now))

        if chunk_meta:
            chunks_only = [cm[1] for cm in chunk_meta]
            vecs = embed(chunks_only)
            rows = []
            for (chunk_id, chunk, source), vec in zip(chunk_meta, vecs):
                blob = struct.pack(f"{EMBED_DIM}f", *vec)
                rows.append((chunk_id, chunk, blob, source, now, "plan"))
            conn.executemany("INSERT OR REPLACE INTO vectors VALUES (?,?,?,?,?,?)", rows)
            new += len(rows)

        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"  ⚠️ 计划索引失败: {e}", file=sys.stderr)
        raise

    return new


def index_sessions(conn):
    """Incrementally index .jsonl sessions across ALL profiles."""
    tracker = {}
    for row in conn.execute("SELECT session_name, mtime, size FROM session_file_tracker"):
        tracker[row[0]] = {"mtime": row[1], "size": row[2]}

    current_files = {}
    for pname, phome in iter_profiles():
        sdir = phome / "sessions"
        if not sdir.exists():
            continue
        for fpath in sorted(sdir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
            st = fpath.stat()
            t_key = fpath.name if pname == "default" else f"profile:{pname}:{fpath.name}"
            current_files[t_key] = (pname, fpath.name, fpath, st.st_mtime, st.st_size)

    deleted = set(tracker.keys()) - set(current_files.keys())
    to_index = []
    for t_key, (pname, sname, fpath, mtime, size) in current_files.items():
        if t_key in tracker:
            old = tracker[t_key]
            if old["mtime"] == mtime and old["size"] == size:
                continue
        to_index.append((t_key, pname, sname, fpath, mtime, size))

    if not deleted and not to_index:
        return 0

    new = 0
    now = time.time()

    try:
        for t_key in deleted:
            src_prefix = f"session:{t_key}" if not t_key.startswith("profile:") else f"{t_key}"
            conn.execute("DELETE FROM vectors WHERE source LIKE ?", (f"{src_prefix}%",))
            conn.execute("DELETE FROM session_file_tracker WHERE session_name = ?", (t_key,))
            print(f"  🗑 deleted session: {t_key}", file=sys.stderr)

        for t_key, pname, sname, fpath, mtime, size in to_index:
            src_prefix = f"session:{sname}" if pname == "default" else f"profile:{pname}:session:{sname}"
            conn.execute("DELETE FROM vectors WHERE source LIKE ?", (f"{src_prefix}%",))

            lines = fpath.read_text(errors='replace').strip().split('\n')
            blocks = []
            block = []
            block_len = 0

            for line in lines:
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue

                role = msg.get("role", "")
                if role not in ("user", "assistant"):
                    continue

                content = msg.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        c.get("text", "") for c in content
                        if isinstance(c, dict) and c.get("type") == "text"
                    )
                content = (content or "").strip()
                if not content:
                    continue

                text = f"[{role}] {content}"
                if block_len + len(text) > 600 and block:
                    blocks.append("\n".join(block))
                    block = []
                    block_len = 0
                block.append(text)
                block_len += len(text)

            if block:
                blocks.append("\n".join(block))

            chunk_meta = []
            for bi, blk in enumerate(blocks):
                chunks = chunk_text(blk)
                for ci, chunk in enumerate(chunks):
                    doc_id = dedup_key(f"{src_prefix}:b{bi}_c{ci}")
                    source = f"{src_prefix}:b{bi}_c{ci}"
                    chunk_meta.append((doc_id, chunk, source))

            if chunk_meta:
                chunks_only = [cm[1] for cm in chunk_meta]
                vecs = embed(chunks_only)
                rows = []
                for (doc_id, chunk, source), vec in zip(chunk_meta, vecs):
                    blob = struct.pack(f"{EMBED_DIM}f", *vec)
                    rows.append((doc_id, chunk, blob, source, now, "session"))
                conn.executemany("INSERT OR REPLACE INTO vectors VALUES (?,?,?,?,?,?)", rows)
                new += len(rows)

            conn.execute(
                "INSERT OR REPLACE INTO session_file_tracker VALUES (?,?,?,?)",
                (t_key, mtime, size, now))

        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"  ⚠️ 会话索引失败: {e}", file=sys.stderr)
        raise

    return new


def index_active_sessions(conn):
    """Index active sessions from state.db across ALL profiles (full history, no cutoff loss)."""
    now = time.time()
    new = 0

    tracker = {}
    for row in conn.execute("SELECT session_id, message_count, started_at FROM active_session_tracker"):
        tracker[row[0]] = {"message_count": row[1], "started_at": row[2]}

    for pname, phome in iter_profiles():
        sdb_path = phome / "state.db"
        sdir = phome / "sessions"
        if not sdb_path.exists():
            continue

        covered_ids = set()
        if sdir.exists():
            for fpath in sdir.glob("*.jsonl"):
                m = re.match(r"(\d{8}_\d{6}_[a-f0-9]+)", fpath.stem)
                if m:
                    covered_ids.add(m.group(1))

        try:
            statedb = sqlite3.connect(f"file:{sdb_path}?mode=ro", uri=True, timeout=5)
        except Exception as e:
            print(f"  ⚠️ Cannot open state.db for profile {pname}: {e}", file=sys.stderr)
            continue

        active = statedb.execute(
            "SELECT id, started_at, message_count, source FROM sessions "
            "WHERE archived=0 ORDER BY started_at"
        ).fetchall()

        filtered = []
        for sid, st, mc, src in active:
            if sid in covered_ids or src == "cron":
                continue
            t_key = sid if pname == "default" else f"profile:{pname}:{sid}"
            filtered.append((t_key, sid, st, mc))

        to_index = []
        for t_key, sid, started_at, msg_count in filtered:
            if t_key in tracker:
                t = tracker[t_key]
                if t["message_count"] == msg_count and t["started_at"] == started_at:
                    continue
            to_index.append((t_key, sid, started_at, msg_count))

        for t_key, sid, started_at, msg_count in to_index:
            src_prefix = f"active_session:{sid}" if pname == "default" else f"profile:{pname}:active_session:{sid}"
            conn.execute("DELETE FROM vectors WHERE source LIKE ?", (f"{src_prefix}:%",))

            messages = statedb.execute(
                "SELECT id, role, content, timestamp FROM messages "
                "WHERE session_id=? AND role IN ('user','assistant') "
                "ORDER BY id DESC LIMIT 200",
                (sid,)
            ).fetchall()
            messages.reverse()

            if not messages:
                conn.execute(
                    "INSERT OR REPLACE INTO active_session_tracker VALUES (?,?,?,?)",
                    (t_key, msg_count or 0, started_at, now))
                continue

            blocks = []
            block = []
            block_len = 0
            for mid, role, content, ts in messages:
                if not content or not content.strip():
                    continue
                text = f"[{role}] {content.strip()}"
                if block_len + len(text) > 600 and block:
                    blocks.append("\n".join(block))
                    block = []
                    block_len = 0
                block.append(text)
                block_len += len(text)
            if block:
                blocks.append("\n".join(block))

            chunk_meta = []
            for bi, blk in enumerate(blocks):
                chunks = chunk_text(blk)
                for ci, chunk in enumerate(chunks):
                    doc_id = dedup_key(f"{src_prefix}:b{bi}_c{ci}")
                    source = f"{src_prefix}:b{bi}_c{ci}"
                    chunk_meta.append((doc_id, chunk, source))

            if chunk_meta:
                chunks_only = [cm[1] for cm in chunk_meta]
                vecs = embed(chunks_only)
                rows = []
                for (doc_id, chunk, source), vec in zip(chunk_meta, vecs):
                    blob = struct.pack(f"{EMBED_DIM}f", *vec)
                    rows.append((doc_id, chunk, blob, source, now, "session"))
                conn.executemany("INSERT OR REPLACE INTO vectors VALUES (?,?,?,?,?,?)", rows)
                new += len(rows)

            conn.execute(
                "INSERT OR REPLACE INTO active_session_tracker VALUES (?,?,?,?)",
                (t_key, msg_count or 0, started_at, now))

        statedb.close()

    conn.commit()
    return new


# ── Search ─────────────────────────────────────────────────

def vector_search(query, kind=None, top_k=5, min_score=0.70, hot_sync=True):
    """FAISS FlatIP semantic search (exact cosine similarity) with score threshold filtering."""
    # ── Hot Incremental Sync: 即时同步最新活跃会话，消除时间差 ──
    if hot_sync and not incognito_active():
        try:
            conn = init_db()
            n_hot = index_active_sessions(conn)
            if n_hot > 0:
                conn.commit()
                build_faiss_index(conn)
            conn.close()
        except Exception as e:
            print(f"  ⚠️ Hot sync skipped: {e}", file=sys.stderr)

    faiss_data = load_faiss_index()
    if not faiss_data:
        print("⚠️ No FAISS index found. Run indexing first.", file=sys.stderr)
        return

    index, ids, id_to_meta = faiss_data
    q_vec = embed(query)
    q_np = np.array([q_vec], dtype=np.float32)
    faiss.normalize_L2(q_np)

    fetch_k = max(top_k * 3, 20)

    if kind:
        kind_ids = [
            i for i, doc_id in enumerate(ids)
            if id_to_meta.get(doc_id, ("", "", ""))[0] == kind
        ]
        if not kind_ids:
            print(f"⚠️ No vectors of kind '{kind}'")
            return
        sub_vectors = np.zeros((len(kind_ids), EMBED_DIM), dtype=np.float32)
        for j, orig_idx in enumerate(kind_ids):
            sub_vectors[j] = index.reconstruct(orig_idx)
        faiss.normalize_L2(sub_vectors)
        sub_index = faiss.IndexFlatIP(EMBED_DIM)
        sub_index.add(sub_vectors)
        actual_k = min(fetch_k, len(kind_ids))
        distances, sub_results = sub_index.search(q_np, actual_k)
        scores = (distances[0] + 1) / 2
        results = list(zip(sub_results[0], scores))
        mapped = []
        for local_idx, score in results:
            if local_idx < 0 or local_idx >= len(kind_ids) or score < min_score:
                continue
            orig_idx = kind_ids[local_idx]
            doc_id = ids[orig_idx]
            kind_name, source, text = id_to_meta.get(doc_id, ("?", "?", "?"))
            mapped.append((score, doc_id, text, source, kind_name))
    else:
        actual_k = min(fetch_k, len(ids))
        distances, results = index.search(q_np, actual_k)
        scores = (distances[0] + 1) / 2
        mapped = []
        for i, (idx, score) in enumerate(zip(results[0], scores)):
            if idx < 0 or idx >= len(ids) or score < min_score:
                continue
            doc_id = ids[idx]
            kind_name, source, text = id_to_meta.get(doc_id, ("?", "?", "?"))
            mapped.append((score, doc_id, text, source, kind_name))

    # Sort and take top_k
    mapped.sort(key=lambda x: x[0], reverse=True)
    mapped = mapped[:top_k]

    if not mapped:
        print(f"ℹ️ 未找到相似度 >= {min_score:.2f} 的高置信度结果。")
        return

    tag = {"session": "💬", "memory": "🧠", "skill": "🛠", "plan": "📋"}

    for sim, doc_id, text, source, k in mapped:
        t = tag.get(k, "📌")
        print(f"  [{sim:.4f}] {t} {source}")
        print(f"    {text[:150].replace(chr(10), ' ')}")
        print()


# ── Database Vacuum / GC ───────────────────────────────────

def vacuum_database():
    """Run WAL checkpoint + VACUUM to reclaim disk space and defragment SQLite storage."""
    if not VECTOR_DB.exists():
        print("⚠️ 向量数据库不存在")
        return

    sz_before = VECTOR_DB.stat().st_size
    t0 = time.time()
    print("🧹 开始向量数据库碎片整理与空间回收 (VACUUM)...")

    conn = sqlite3.connect(str(VECTOR_DB), timeout=60)
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.execute("VACUUM")
    conn.close()

    sz_after = VECTOR_DB.stat().st_size
    freed = max(0, sz_before - sz_after)
    elapsed = time.time() - t0
    print(f"✓ 碎片整理完成 ({elapsed:.2f}s):")
    print(f"  - 整理前大小: {sz_before / (1024*1024):.2f}MB")
    print(f"  - 整理后大小: {sz_after / (1024*1024):.2f}MB")
    print(f"  - 释放空间:   {freed / (1024*1024):.2f}MB")


# ── Stats ──────────────────────────────────────────────────

def show_stats():
    conn = sqlite3.connect(str(VECTOR_DB))
    rows = conn.execute("SELECT kind, COUNT(*) FROM vectors GROUP BY kind ORDER BY COUNT(*) DESC").fetchall()

    profile_rows = conn.execute("""
        SELECT 
            CASE 
                WHEN source LIKE 'profile:%' THEN substr(source, 9, instr(substr(source, 9), ':') - 1)
                ELSE 'default'
            END AS profile_name,
            COUNT(*)
        FROM vectors
        GROUP BY profile_name
        ORDER BY COUNT(*) DESC
    """).fetchall()
    conn.close()

    total = sum(r[1] for r in rows)
    print(f"📊 {total} total vectors (binary storage):")
    for k, cnt in rows:
        pct = (cnt / total * 100) if total else 0
        bar = "█" * max(1, int(cnt / (total or 1) * 30))
        print(f"  {k:<12} {cnt:>6}  {bar} ({pct:.1f}%)")

    print("\n📁 Profiles breakdown:")
    for pname, cnt in profile_rows:
        pct = (cnt / total * 100) if total else 0
        print(f"  {pname:<12} {cnt:>6} ({pct:.1f}%)")

    db_size = VECTOR_DB.stat().st_size / (1024 * 1024) if VECTOR_DB.exists() else 0
    faiss_size = FAISS_INDEX.stat().st_size / (1024 * 1024) if FAISS_INDEX.exists() else 0
    print(f"\n  DB: {db_size:.1f}MB  |  FAISS index: {faiss_size:.1f}MB")


# ── Incognito Check ────────────────────────────────────────

def incognito_active():
    if not INC_SENTINEL.exists():
        return False
    try:
        if time.time() - INC_SENTINEL.stat().st_mtime > 4 * 3600:
            try: INC_SENTINEL.unlink()
            except OSError: pass
            return False
    except OSError:
        return False
    sid = None
    try:
        for line in INC_SENTINEL.read_text(errors="ignore").splitlines():
            if line.startswith("session="):
                sid = line.split("=", 1)[1].split()[0]
    except OSError:
        pass
    if not sid or sid == "unknown":
        return True
    sdb = HERMES_ROOT / "state.db"
    if sdb.exists():
        try:
            conn = sqlite3.connect(f"file:{sdb}?mode=ro", uri=True, timeout=5)
            row = conn.execute("SELECT 1 FROM sessions WHERE id = ?", (sid,)).fetchone()
            conn.close()
            if row is None:
                try: INC_SENTINEL.unlink()
                except OSError: pass
                return False
        except Exception:
            return True
    return True


# ── Main ───────────────────────────────────────────────────

def main():
    if len(sys.argv) <= 1 or sys.argv[1] not in ("search", "stats", "vacuum"):
        import signal
        my_pid = os.getpid()
        killed = []
        for pid_dir in Path('/proc').glob('[0-9]*'):
            try:
                pid = int(pid_dir.name)
                if pid == my_pid:
                    continue
                comm = (pid_dir / 'comm').read_text(errors='replace').strip()
                if comm not in ('python3', 'python'):
                    continue
                cmdline = (pid_dir / 'cmdline').read_text(errors='replace')
                if 'index_all.py' in cmdline:
                    os.kill(pid, signal.SIGTERM)
                    killed.append(pid)
                    print(f"  🧹 Killed stale index_all process PID {pid}", file=sys.stderr)
            except (PermissionError, ProcessLookupError, FileNotFoundError):
                pass
        if killed:
            time.sleep(2)
            try:
                db = sqlite3.connect(str(VECTOR_DB), timeout=10)
                db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                db.close()
                print("  ✓ WAL checkpoint complete", file=sys.stderr)
            except Exception as e:
                print(f"  ⚠️ WAL checkpoint failed: {e}", file=sys.stderr)

    if len(sys.argv) > 1 and sys.argv[1] == "vacuum":
        vacuum_database()
        return

    if len(sys.argv) > 1 and sys.argv[1] == "search":
        kind = None
        min_score = 0.70
        args = sys.argv[2:]

        while args and args[0].startswith("--"):
            if args[0] == "--kind" and len(args) > 1:
                kind = args[1]
                args = args[2:]
            elif args[0] == "--min-score" and len(args) > 1:
                min_score = float(args[1])
                args = args[2:]
            else:
                args = args[1:]

        query = args[0] if args else input("Query: ")
        top_k = int(args[1]) if len(args) > 1 else 5
        vector_search(query, kind, top_k, min_score)
        return

    if len(sys.argv) > 1 and sys.argv[1] == "stats":
        show_stats()
        return

    targets = sys.argv[1:] if len(sys.argv) > 1 else ["active_sessions", "sessions", "memory", "skills", "plans"]
    conn = init_db()
    t0 = time.time()
    stats = {}

    if incognito_active():
        print("⏭️ 无痕模式激活：跳过 active_sessions 及 sessions 索引源（memory/skills/plans 照常）", file=sys.stderr)
        targets = [t for t in targets if t not in ("active_sessions", "sessions")]

    print("🚀 开始全量深度增量向量索引 (Multi-Profile + 853 参考手册 + 计划库)...")
    for p, _ in iter_profiles():
        print(f"  📁 发现 Profile: {p}")

    if "active_sessions" in targets:
        n = index_active_sessions(conn)
        stats["active_sessions"] = n
    if "sessions" in targets:
        n = index_sessions(conn)
        stats["sessions"] = n
    if "memory" in targets:
        n = index_memory(conn)
        stats["memory"] = n
    if "skills" in targets:
        n = index_skills(conn)
        stats["skills"] = n
    if "plans" in targets:
        n = index_plans(conn)
        stats["plans"] = n

    conn.commit()

    if sum(stats.values()) > 0:
        print("🔧 Rebuilding FAISS index...")
        result = build_faiss_index(conn)
        if result is None:
            print("   ⚠️ No valid vectors — FAISS index not updated", file=sys.stderr)
        else:
            index, ids, meta = result
            print(f"   ✓ FAISS index updated: {index.ntotal} vectors")

    elapsed = time.time() - t0
    total_new = sum(stats.values())
    print(f"✨ 完成: +{total_new} chunks 索引写入 ({elapsed:.2f}s)")
    for k, v in stats.items():
        if v > 0:
            print(f"   - {k}: +{v}")


if __name__ == "__main__":
    main()
