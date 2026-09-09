#!/usr/bin/env python3
"""
Idempotent patch applicator for Hermes session_search true concurrent dual-track RRF fusion.
Used by systemd ExecStartPre before gateway startup.
"""

import sys
import py_compile
import re
from pathlib import Path

TARGET_FILE = Path.home() / '.hermes' / 'hermes-agent' / 'tools' / 'session_search_tool.py'
MARKER = 'VECTOR-CONCURRENT-RRF-HYBRID-PATCH'

RRF_HYBRID_DISCOVER_BLOCK = r"""# VECTOR-CONCURRENT-RRF-HYBRID-PATCH: True Concurrent Dual-Track RRF Fusion Engine
def _vector_search_candidates(
    db,
    query: str,
    limit: int = 20,
    min_score: float = 0.70,
    current_lineage_root: str = None,
    current_session_id: str = None,
) -> dict:
    '''Query vector engine via index_all.py and return {lineage_root: (rank, match_info)}.'''
    import subprocess
    import re as _re
    from pathlib import Path

    script_path = Path.home() / '.hermes' / 'scripts' / 'index_all.py'
    if not script_path.exists():
        return {}

    cmd = [
        '/usr/bin/python3',
        str(script_path),
        'search',
        '--kind',
        'session',
        '--min-score',
        str(min_score),
        query,
        str(limit),
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except Exception as e:
        logging.debug('vector candidate search subprocess failed: %s', e)
        return {}

    if res.returncode != 0 or not res.stdout:
        return {}

    lines = res.stdout.splitlines()
    vec_ranked = {}
    rank = 1

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('  [') and ']' in line and '💬' in line:
            try:
                score_part = line.split(']')[0].replace('  [', '').replace('[', '').strip()
                score = float(score_part)

                if score < min_score:
                    i += 1
                    continue

                source_part = line.split('💬')[1].strip()
                m = _re.search(r'(?:active_session:|session:)([\w\d_-]+)', source_part)
                if not m:
                    i += 1
                    continue
                sid = m.group(1).replace('.jsonl', '')

                pname = None
                if source_part.startswith('profile:'):
                    pname = source_part.split(':')[1]

                snippet = ''
                if i + 1 < len(lines) and not lines[i + 1].startswith('  ['):
                    snippet = lines[i + 1].strip()

                if sid:
                    resolved_sid, _ = _resolve_to_parent(db, sid)
                    lineage = resolved_sid or sid

                    if current_lineage_root and (lineage == current_lineage_root or sid == current_lineage_root):
                        i += 1
                        continue
                    if current_session_id and (sid == current_session_id or lineage == current_session_id):
                        i += 1
                        continue

                    if lineage not in vec_ranked:
                        meta = (
                            db.get_session(resolved_sid)
                            or db.get_session(sid)
                            or {}
                        )
                        if meta.get('source') not in _HIDDEN_SESSION_SOURCES:
                            entry = {
                                'session_id': sid,
                                'when': _format_timestamp(meta.get('started_at')),
                                'source': meta.get('source', 'unknown'),
                                'title': meta.get('title') or None,
                                'match_source': 'vector',
                                'vector_score': score,
                                'snippet': snippet[:200] if snippet else '',
                            }
                            if pname:
                                entry['profile'] = pname
                            if resolved_sid and resolved_sid != sid:
                                entry['parent_session_id'] = resolved_sid
                            vec_ranked[lineage] = (rank, entry)
                            rank += 1
            except Exception as e:
                logging.debug('vector candidate parse error: %s', e)
        i += 1

    return vec_ranked


def _discover(db, query: str, role_filter: Optional[List[str]], limit: int, sort: Optional[str],
              detail: str, current_session_id: str = None, link_profile: str = None) -> str:
    '''Discovery shape: Concurrent Dual-Track RRF (FTS5 + Dense Vector) Fusion.'''
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

    # ── RRF Fusion (Reciprocal Rank Fusion) ──
    all_lineages = set(fts_ranked.keys()) | set(vec_ranked.keys())
    if not all_lineages and not title_result:
        return _discover_payload(db, query, detail, [], message=(
            "No matching sessions found in FTS5 or Vector knowledge space."))

    k_rrf = 60
    fused_scores = {}
    for lin in all_lineages:
        r_f = fts_ranked[lin][0] if lin in fts_ranked else None
        r_v = vec_ranked[lin][0] if lin in vec_ranked else None

        score_f = (1.0 / (k_rrf + r_f)) if r_f else 0.0
        score_v = (1.0 / (k_rrf + r_v)) if r_v else 0.0
        total_rrf = score_f + score_v

        if lin in fts_ranked and lin in vec_ranked:
            match_info = dict(fts_ranked[lin][1])
            match_info["match_source"] = "hybrid"
            match_info["vector_score"] = vec_ranked[lin][1].get("vector_score")
            match_info["rrf_score"] = total_rrf
            if "profile" in vec_ranked[lin][1]:
                match_info["profile"] = vec_ranked[lin][1]["profile"]
        elif lin in vec_ranked:
            match_info = dict(vec_ranked[lin][1])
            match_info["match_source"] = "vector"
            match_info["rrf_score"] = total_rrf
        else:
            match_info = dict(fts_ranked[lin][1])
            match_info["match_source"] = "fts5"
            match_info["rrf_score"] = total_rrf

        fused_scores[lin] = (total_rrf, match_info)

    # Sort strictly by RRF score descending
    sorted_fused = sorted(fused_scores.items(), key=lambda x: x[1][0], reverse=True)
    top_fused = sorted_fused[:limit]

    # ── Assembly & Hydration ──
    results = [title_result] if title_result else []
    for lin, (rrf_sc, match_info) in top_fused:
        if match_info.get("match_source") == "vector":
            results.append(match_info)
            continue
        if match_info.get("_title_only"):
            continue
        entry = _hydrate_hit(db, lin, match_info, "full" if detail == "full" or not results else "compact")
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


def apply_patch():
    if not TARGET_FILE.exists():
        print(f'Error: target file {TARGET_FILE} does not exist', file=sys.stderr)
        sys.exit(2)

    content = TARGET_FILE.read_text(encoding='utf-8')
    original_backup = content

    # 1. 查找替换精确的 _discover 函数块
    pattern = r'def _discover\(db, query: str[\s\S]*?(?=def _resolve_profile_db)'
    if re.search(pattern, content):
        new_content = re.sub(pattern, lambda m: RRF_HYBRID_DISCOVER_BLOCK + '\n\n\n', content, count=1)
    else:
        print('Error: Could not locate _discover anchor in target file', file=sys.stderr)
        sys.exit(2)

    try:
        TARGET_FILE.write_text(new_content, encoding='utf-8')
        py_compile.compile(str(TARGET_FILE), doraise=True)
    except Exception as e:
        print(f'Syntax validation failed after patching: {e}. Rolling back.', file=sys.stderr)
        TARGET_FILE.write_text(original_backup, encoding='utf-8')
        sys.exit(3)

    print('vector-concurrent-rrf patch applied successfully')
    sys.exit(0)


if __name__ == '__main__':
    apply_patch()
