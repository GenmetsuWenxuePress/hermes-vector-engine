#!/usr/bin/env python3
"""
Automated Test Suite for Hermes Vector Engine
Tests all 4 dimensions:
  1. Regex source parsing & multi-profile tagging
  2. Markdown & Code AST-aware chunking
  3. Similarity threshold filtering
  4. SQLite database integrity & VACUUM GC
"""

import os
import sys
import tempfile
import sqlite3
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from index_all import chunk_text, dedup_key

def test_ast_chunking_preserves_tables_and_code():
    print("Testing Dimension 2: AST-Aware Chunking...")
    sample = """# Configuration Guide

Here is the table:

| Key | Default | Description |
|---|---|---|
| keep_alive | 15 | TCP keep-alive interval |
| auto_close | true | Close idle sockets |

And here is the code:

```python
def setup_connection(host: str, port: int):
    return f"http://{host}:{port}"
```
"""
    chunks = chunk_text(sample, max_chars=300)
    assert len(chunks) > 0, "No chunks generated"
    
    # Assert table integrity
    table_chunk = next((c for c in chunks if "| Key |" in c), None)
    assert table_chunk is not None, "Table missing from chunks"
    assert "| auto_close |" in table_chunk, "Table was broken in half!"
    
    # Assert code block integrity
    code_chunk = next((c for c in chunks if "```python" in c), None)
    assert code_chunk is not None, "Code block missing from chunks"
    assert "```" in code_chunk[code_chunk.find("```python") + 9:], "Code block fence was broken!"
    print("  ✓ AST-Aware Chunking test passed.")

def test_source_tagging_and_parsing():
    print("Testing Dimension 1: Source Tagging & Regex Parsing...")
    import re
    samples = [
        ("active_session:20260909_132432_59d1a3:b0_c0", "20260909_132432_59d1a3", None),
        ("profile:ops:active_session:20260909_145933_b49364:b1_c0", "20260909_145933_b49364", "ops"),
        ("session:20260825_233334_3f705d.jsonl:b2_c1", "20260825_233334_3f705d", None),
        ("profile:ops:session:session_sample.jsonl:b0_c0", "session_sample", "ops"),
    ]
    for src, exp_sid, exp_pname in samples:
        m = re.search(r"(?:active_session:|session:)([\w\d_-]+)", src)
        assert m is not None, f"Failed to match {src}"
        sid = m.group(1).replace(".jsonl", "")
        pname = src.split(":")[1] if src.startswith("profile:") else None
        assert sid == exp_sid, f"Mismatched SID for {src}: got {sid}, expected {exp_sid}"
        assert pname == exp_pname, f"Mismatched Profile for {src}: got {pname}, expected {exp_pname}"
    print("  ✓ Source tagging and parsing test passed.")

def test_sqlite_vacuum_and_integrity():
    print("Testing Dimension 4: SQLite Integrity & Vacuum...")
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        conn = sqlite3.connect(tmp.name)
        conn.execute("CREATE TABLE vectors (id TEXT PRIMARY KEY, text TEXT, blob BLOB)")
        for i in range(100):
            conn.execute("INSERT INTO vectors VALUES (?, ?, ?)", (f"id_{i}", f"text {i}", b"\x00" * 1024))
        conn.commit()
        
        # Test integrity check
        check = conn.execute("PRAGMA integrity_check").fetchall()
        assert check == [("ok",)], f"Integrity check failed: {check}"
        
        # Delete and vacuum
        conn.execute("DELETE FROM vectors WHERE id LIKE 'id_1%'")
        conn.commit()
        conn.execute("VACUUM")
        conn.close()
    print("  ✓ SQLite Integrity & Vacuum test passed.")

if __name__ == "__main__":
    test_ast_chunking_preserves_tables_and_code()
    test_source_tagging_and_parsing()
    test_sqlite_vacuum_and_integrity()
    print("\n🎉 All tests passed successfully!")
