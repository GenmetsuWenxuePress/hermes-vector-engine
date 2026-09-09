#!/usr/bin/env python3
"""Knowledge Search Tool - Semantic Vector Search across skills, references, plans, and memories.

Enables the AI Agent to perform Agentic RAG over 850+ reference manuals,
engineering plans, and persistent system memories using the local GPU BGE-M3 engine.
"""

import os
import sys
import json
import logging
import subprocess
import re
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

KNOWLEDGE_SEARCH_SCHEMA = {
    "name": "knowledge_search",
    "description": (
        "跨系统本地知识中枢语义检索工具。基于 BGE-M3 GPU 向量引擎与 FAISS，"
        "秒级检索 850+ 篇深度技术参考手册 (references/*.md)、工程施工图计划 (plans/*.md)、"
        "持久系统记忆与技能库。遇到疑难报错、底层架构规范、部署SOP或历史设计方案时优先调用。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词或自然语言描述（例如：'WARP 502 报错排查'、'Vulkan 显卡卸载参数'、'施工图计划'）",
            },
            "kind": {
                "type": "string",
                "enum": ["all", "skill", "plan", "memory"],
                "default": "all",
                "description": "知识类别过滤：skill (技能与参考手册), plan (工程施工图计划), memory (系统记忆与用户画像), all (全域知识)",
            },
            "limit": {
                "type": "integer",
                "default": 3,
                "description": "返回的最优候选条数 (1-10，默认 3)",
            },
            "min_score": {
                "type": "number",
                "default": 0.70,
                "description": "最低相似度置信度门限 (默认 0.70，过滤无关噪音)",
            },
        },
        "required": ["query"],
    },
}


def _resolve_source_to_path(source: str) -> Optional[str]:
    """Convert a vector source tag back to its physical file path on disk."""
    home = Path.home()
    hermes_root = home / ".hermes"

    # Profile prefix: profile:<name>:...
    pname = None
    rest = source
    if source.startswith("profile:"):
        parts = source.split(":", 2)
        pname = parts[1]
        rest = parts[2] if len(parts) > 2 else ""

    base_dir = hermes_root if not pname or pname == "default" else hermes_root / "profiles" / pname

    if rest.startswith("skill:"):
        # skill:<category>/<name>:ref:<ref_name> or skill:<category>/<name>
        skill_rest = rest[len("skill:"):]
        if ":ref:" in skill_rest:
            skill_name, ref_name = skill_rest.split(":ref:", 1)
            ref_path = base_dir / "skills" / skill_name / "references" / f"{ref_name}.md"
            if ref_path.exists():
                return str(ref_path)
        else:
            skill_name = skill_rest.split(":")[0]
            skill_path = base_dir / "skills" / skill_name / "SKILL.md"
            if skill_path.exists():
                return str(skill_path)

    elif rest.startswith("plan:"):
        plan_name = rest[len("plan:"):].split(":")[0]
        plan_path = base_dir / "plans" / plan_name
        if plan_path.exists():
            return str(plan_path)

    elif rest.startswith("memory:"):
        mem_file = rest[len("memory:"):].split(":")[0]
        mem_path = base_dir / "memories" / mem_file
        if mem_path.exists():
            return str(mem_path)

    return None


def knowledge_search(
    query: str,
    kind: str = "all",
    limit: int = 3,
    min_score: float = 0.70,
) -> str:
    """Execute semantic search across local knowledge sources via index_all.py."""
    if not query or not query.strip():
        return json.dumps({"success": False, "error": "Query string cannot be empty.", "results": []}, ensure_ascii=False)

    script_path = Path.home() / ".hermes" / "scripts" / "index_all.py"
    if not script_path.exists():
        return json.dumps({"success": False, "error": f"Index script not found at {script_path}", "results": []}, ensure_ascii=False)

    cmd = [
        sys.executable or "/usr/bin/python3",
        str(script_path),
        "search",
    ]

    if kind and kind != "all":
        cmd.extend(["--kind", kind])

    cmd.extend(["--min-score", str(min_score), query.strip(), str(limit)])

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    except subprocess.TimeoutExpired:
        return json.dumps({"success": False, "error": "Knowledge search subprocess timed out.", "results": []}, ensure_ascii=False)
    except Exception as e:
        logger.error("knowledge_search failed: %s", e, exc_info=True)
        return json.dumps({"success": False, "error": f"Search execution failed: {e}", "results": []}, ensure_ascii=False)

    if res.returncode != 0:
        return json.dumps({"success": False, "error": res.stderr.strip() or "Vector search returned error.", "results": []}, ensure_ascii=False)

    lines = res.stdout.splitlines()
    results = []
    i = 0

    while i < len(lines):
        line = lines[i]
        # Format: "  [0.8228] 🛠 profile:ops:skill:devops/clash-network-ops:ref:warp-chain-proxy"
        if line.startswith("  [") and "]" in line:
            try:
                score_str = line.split("]")[0].replace("  [", "").replace("[", "").strip()
                score = float(score_str)

                # Icon and source part
                after_score = line.split("]", 1)[1].strip()
                parts = after_score.split(maxsplit=1)
                icon = parts[0] if len(parts) > 0 else "📌"
                source = parts[1] if len(parts) > 1 else ""

                snippet = ""
                if i + 1 < len(lines) and not lines[i + 1].startswith("  ["):
                    snippet = lines[i + 1].strip()

                # Derive kind
                detected_kind = "unknown"
                if "skill" in source or icon == "🛠":
                    detected_kind = "skill"
                elif "plan" in source or icon == "📋":
                    detected_kind = "plan"
                elif "memory" in source or icon == "🧠":
                    detected_kind = "memory"
                elif "session" in source or icon == "💬":
                    detected_kind = "session"

                file_path = _resolve_source_to_path(source)

                results.append({
                    "score": score,
                    "kind": detected_kind,
                    "source": source,
                    "path": file_path,
                    "snippet": snippet,
                })
            except Exception as e:
                logger.debug("knowledge_search parse error on line %r: %s", line, e)
        i += 1

    return json.dumps({
        "success": True,
        "query": query,
        "count": len(results),
        "min_score": min_score,
        "results": results,
    }, ensure_ascii=False, indent=2)


def check_knowledge_search_requirements() -> bool:
    """True when index_all.py exists."""
    script_path = Path.home() / ".hermes" / "scripts" / "index_all.py"
    return script_path.exists()


# ── Registry Registration ───────────────────────────────────
from tools.registry import registry, tool_error  # noqa: E402

registry.register(
    name="knowledge_search",
    toolset="skills",
    schema=KNOWLEDGE_SEARCH_SCHEMA,
    handler=lambda args, **kw: knowledge_search(
        query=args.get("query") or "",
        kind=args.get("kind", "all"),
        limit=args.get("limit", 3),
        min_score=args.get("min_score", 0.70),
    ),
    check_fn=check_knowledge_search_requirements,
    emoji="🧠",
    max_result_size_chars=50_000,
)
