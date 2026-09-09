#!/usr/bin/env python3
"""
Idempotent patch applicator for Hermes session_search score-aware calibrated RRF fusion,
smart gap-gated hydration, and knowledge_search core toolset registration.
Used by systemd ExecStartPre before gateway startup and by hermes-maintenance updates.
"""

import os
import re
import sys
import py_compile
from pathlib import Path

HERMES_ROOT = Path.home() / ".hermes" / "hermes-agent"
SESSION_TOOL_FILE = HERMES_ROOT / "tools" / "session_search_tool.py"
TOOLSETS_FILE = HERMES_ROOT / "toolsets.py"

RRF_HYBRID_DISCOVER_BLOCK = """def _discover(db, query: str, role_filter: Optional[List[str]], limit: int, sort: Optional[str],
              detail: str, current_session_id: str = None, link_profile: str = None) -> str:
    '''Discovery shape: Score-Aware Calibrated Dual-Track RRF (FTS5 + Dense Vector) Fusion & Smart Hydration.'''
    current_lineage_root = _resolve_lineage(db, current_session_id) if current_session_id else None
    title_result = _title_match_result(db, query, current_lineage_root)

    # ── Track 1: FTS5 BM25 Search ──
    raw_results, err = _loud(lambda: db.search_messages(
        query=query, role_filter=role_filter or ["user", "assistant"],
        exclude_sources=list(_HIDDEN_SESSION_SOURCES), limit=_DISCOVER_SCAN_LIMIT, offset=0, sort=sort,
        fields=_DISCOVER_SEARCH_FIELDS), "FTS5 search failed: %s", "Search failed")

    raw_results = sorted(raw_results or [], key=lambda r: (r.get("source") or "") in _DEMOTED_SESSION_SOURCES)
    
    fts_ranked = {}
    fts_rank = 1
    for r in raw_results:
        raw_sid, resolved_sid = r["session_id"], _resolve_lineage(db, r["session_id"])
        is_compacted_hit = _is_compacted_message(db, r.get("id"))
        if current_lineage_root and resolved_sid == current_lineage_root and not (
                _session_left_live_context(db, raw_sid) or is_compacted_hit):
            continue
        if current_session_id and raw_sid == current_session_id and not is_compacted_hit:
            continue
        lineage = resolved_sid or raw_sid
        if lineage not in fts_ranked:
            fts_ranked[lineage] = (fts_rank, {**r, "_lineage_root": lineage})
            fts_rank += 1

    # ── Track 2: Concurrent Dense Vector Search ──
    vec_ranked = {}
    try:
        vec_ranked = _vector_search_candidates(
            db, query, limit=max(limit * 3, 20), min_score=0.70,
            current_lineage_root=current_lineage_root, current_session_id=current_session_id
        )
    except Exception as e:
        logging.debug('concurrent vector search failed: %s', e)

    # ── Score-Aware Calibrated RRF Fusion ──
    all_lineages = set(fts_ranked.keys()) | set(vec_ranked.keys())
    if not all_lineages and not title_result:
        return _discover_payload(db, query, detail, [], message=(
            "No matching sessions found in FTS5 or Vector knowledge space."))

    k_rrf = 60
    w_vec = 1.25         # High semantic intent weight
    w_fts = 1.00         # Keyword lexical baseline
    synergy_boost = 0.005 # Synergy boost when hit in both FTS5 & Vector

    fused_scores = {}
    for lin in all_lineages:
        r_f = fts_ranked[lin][0] if lin in fts_ranked else None
        r_v = vec_ranked[lin][0] if lin in vec_ranked else None

        score_f = (w_fts / (k_rrf + r_f)) if r_f else 0.0
        
        # Vector score-aware calibration: (vector_score / 0.80)^2 non-linear boost
        score_v = 0.0
        v_score = None
        if r_v:
            v_score = vec_ranked[lin][1].get("vector_score") or 0.70
            conf_multiplier = (v_score / 0.80) ** 2
            score_v = (w_vec * conf_multiplier) / (k_rrf + r_v)

        total_rrf = score_f + score_v
        if lin in fts_ranked and lin in vec_ranked:
            total_rrf += synergy_boost

        if lin in fts_ranked and lin in vec_ranked:
            match_info = dict(fts_ranked[lin][1])
            match_info["match_source"] = "hybrid"
            match_info["vector_score"] = v_score
            match_info["rrf_score"] = total_rrf
            if "profile" in vec_ranked[lin][1]:
                match_info["profile"] = vec_ranked[lin][1]["profile"]
        elif lin in vec_ranked:
            match_info = dict(vec_ranked[lin][1])
            match_info["match_source"] = "vector"
            match_info["vector_score"] = v_score
            match_info["rrf_score"] = total_rrf
        else:
            match_info = dict(fts_ranked[lin][1])
            match_info["match_source"] = "fts5"
            match_info["rrf_score"] = total_rrf

        fused_scores[lin] = (total_rrf, match_info)

    # Sort strictly by calibrated RRF score descending
    sorted_fused = sorted(fused_scores.items(), key=lambda x: x[1][0], reverse=True)
    top_fused = sorted_fused[:limit]

    # ── Dynamic Gap-Gated Smart Hydration ──
    results = [title_result] if title_result else []
    top_score = top_fused[0][1][0] if top_fused else 0.0

    for idx, (lin, (rrf_sc, match_info)) in enumerate(top_fused):
        if match_info.get("_title_only"):
            continue

        # Decide hydration level
        if detail == "full":
            hydrate_mode = "full"
        elif detail == "compact":
            hydrate_mode = "compact"
        else:
            # detail == "adaptive"
            # Top 1 always full; Top 2~3 full if score >= 85% of Top 1 or vector_score >= 0.82
            v_sc = match_info.get("vector_score") or 0.0
            if idx == 0 or (idx < 3 and (rrf_sc >= top_score * 0.85 or v_sc >= 0.82)):
                hydrate_mode = "full"
            else:
                hydrate_mode = "compact"

        entry = _hydrate_hit(db, lin, match_info, hydrate_mode)
        if entry is None and match_info.get("match_source") == "vector":
            entry = dict(match_info)

        if entry is not None:
            entry["match_source"] = match_info.get("match_source", "fts5")
            if "rrf_score" in match_info:
                entry["rrf_score"] = round(match_info["rrf_score"], 5)
            if "vector_score" in match_info:
                entry["vector_score"] = match_info["vector_score"]
            if "profile" in match_info:
                entry["profile"] = match_info["profile"]
            results.append(entry)

    for entry in results:
        entry["link"] = _session_link(entry["session_id"], link_profile)

    return _discover_payload(db, query, detail, results, sessions_searched=len(fused_scores), link_hint=(
        "When referring the user to a session, write its `link` value "
        "verbatim inline mid-sentence (it renders as a titled link) — never "
        "as markdown, in backticks, on its own line, or next to the "
        "title/id/date. To read more around a compact result, scroll: "
        "session_search(session_id=..., around_message_id=match_message_id)."))
"""


def patch_session_search_tool():
    if not SESSION_TOOL_FILE.exists():
        print(f'Error: target file {SESSION_TOOL_FILE} does not exist', file=sys.stderr)
        return False

    content = SESSION_TOOL_FILE.read_text(encoding='utf-8')
    original_backup = content

    pattern = r'def _discover\(db, query: str[\s\S]*?(?=def _resolve_profile_db)'
    if re.search(pattern, content):
        new_content = re.sub(pattern, lambda m: RRF_HYBRID_DISCOVER_BLOCK + '\n\n\n', content, count=1)
    else:
        print('Warning: Could not locate _discover anchor in session_search_tool.py', file=sys.stderr)
        return False

    try:
        SESSION_TOOL_FILE.write_text(new_content, encoding='utf-8')
        py_compile.compile(str(SESSION_TOOL_FILE), doraise=True)
        print('✓ session_search_tool calibrated RRF & smart hydration patch verified')
        return True
    except Exception as e:
        print(f'Syntax validation failed after patching session_search_tool: {e}. Rolling back.', file=sys.stderr)
        SESSION_TOOL_FILE.write_text(original_backup, encoding='utf-8')
        return False


def patch_toolsets():
    if not TOOLSETS_FILE.exists():
        print(f'Error: target file {TOOLSETS_FILE} does not exist', file=sys.stderr)
        return False

    content = TOOLSETS_FILE.read_text(encoding='utf-8')
    original_backup = content
    modified = False

    # 1. 确保 _HERMES_CORE_TOOLS 包含 "knowledge_search"
    if '"knowledge_search"' not in content:
        if '"skill_manage",' in content:
            content = content.replace('"skill_manage",', '"skill_manage", "knowledge_search",', 1)
            modified = True

    # 2. 确保 "skills" toolset 包含 "knowledge_search"
    if '["skills_list", "skill_view", "skill_manage"]' in content:
        content = content.replace(
            '["skills_list", "skill_view", "skill_manage"]',
            '["skills_list", "skill_view", "skill_manage", "knowledge_search"]',
            1
        )
        modified = True

    if modified:
        try:
            TOOLSETS_FILE.write_text(content, encoding='utf-8')
            py_compile.compile(str(TOOLSETS_FILE), doraise=True)
            print('✓ toolsets.py knowledge_search registration verified')
        except Exception as e:
            print(f'Syntax validation failed after patching toolsets: {e}. Rolling back.', file=sys.stderr)
            TOOLSETS_FILE.write_text(original_backup, encoding='utf-8')
            return False
    else:
        print('✓ toolsets.py already includes knowledge_search')
    return True


def apply_patch():
    print("Applying Hermes calibrated RRF, smart hydration & knowledge_search auto-restore patch...")
    ok1 = patch_session_search_tool()
    ok2 = patch_toolsets()
    if ok1 and ok2:
        print("✓ All vector hybrid patches applied successfully")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(apply_patch())
