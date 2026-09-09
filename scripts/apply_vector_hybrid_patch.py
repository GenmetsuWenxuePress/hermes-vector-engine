#!/usr/bin/env python3
"""
Idempotent patch applicator for Hermes session_search vector hybrid fallback.
Used by systemd ExecStartPre before gateway startup.
"""

import sys
import py_compile
import re
from pathlib import Path

TARGET_FILE = Path.home() / '.hermes' / 'hermes-agent' / 'tools' / 'session_search_tool.py'
MARKER = 'VECTOR-HYBRID-PATCH'

VECTOR_SUPPLEMENT_FUNC = r"""def _vector_supplement(
    db,
    query: str,
    seen_sessions: dict,
    limit: int,
    current_lineage_root: str = None,
    role_list: list = None,
) -> None:
    '''Supplement sparse FTS5 results with semantic vector search via index_all.py (Multi-Profile & Score Threshold Aware).'''
    import subprocess
    import re as _re
    from pathlib import Path

    if len(seen_sessions) >= limit:
        return

    script_path = Path.home() / '.hermes' / 'scripts' / 'index_all.py'
    if not script_path.exists():
        return

    cmd = [
        '/usr/bin/python3',
        str(script_path),
        'search',
        '--kind',
        'session',
        query,
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except Exception as e:
        logging.debug('vector supplement subprocess failed: %s', e)
        return

    if res.returncode != 0 or not res.stdout:
        return

    lines = res.stdout.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('  [') and ']' in line and '💬' in line:
            try:
                score_part = line.split(']')[0].replace('  [', '').replace('[', '').strip()
                score = float(score_part)

                # 置信度截断：过滤相似度低于 0.70 的弱相关噪音
                if score < 0.70:
                    i += 1
                    continue

                source_part = line.split('💬')[1].strip()
                
                # 兼容 default 与多 Profile (如 profile:ops:active_session:xxx 或 active_session:xxx)
                m = _re.search(r'(?:active_session:|session:)([\w\d_-]+)', source_part)
                if not m:
                    i += 1
                    continue
                sid = m.group(1).replace('.jsonl', '')

                # 提取 profile 标识
                pname = None
                if source_part.startswith('profile:'):
                    pname = source_part.split(':')[1]

                snippet = ''
                if i + 1 < len(lines) and not lines[i + 1].startswith('  ['):
                    snippet = lines[i + 1].strip()

                if sid:
                    resolved_sid, _ = _resolve_to_parent(db, sid)
                    if (
                        sid not in seen_sessions
                        and resolved_sid not in seen_sessions
                    ):
                        if not (
                            current_lineage_root
                            and (
                                resolved_sid == current_lineage_root
                                or sid == current_lineage_root
                            )
                        ):
                            meta = (
                                db.get_session(resolved_sid)
                                or db.get_session(sid)
                                or {}
                            )
                            if (
                                meta.get('source')
                                not in _HIDDEN_SESSION_SOURCES
                            ):
                                entry = {
                                    'session_id': sid,
                                    'when': _format_timestamp(
                                        meta.get('started_at')
                                    ),
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
                                seen_sessions[resolved_sid] = entry
                                if len(seen_sessions) >= limit:
                                    break
            except Exception as e:
                logging.debug('vector supplement parse error: %s', e)
        i += 1


"""

CALL_BLOCK = """    # VECTOR-HYBRID-PATCH: semantic fallback when FTS5 results are sparse
    if len(seen_sessions) < limit:
        try:
            _vector_supplement(db, query, seen_sessions, limit, current_lineage_root, role_list)
        except Exception as e:
            logging.debug('vector supplement skipped: %s', e)

    for lineage_root, match_info in seen_sessions.items():
        if match_info.get('match_source') == 'vector':
            results.append(match_info)
            continue"""

EARLY_EXIT_BLOCK = """    if not raw_results and not title_result:
        _empty_payload = {
            "success": True,
            "mode": "discover",
            "query": query,
            "results": [],
            "count": 0,
            "message": "No matching sessions found.",
        }
        _annotate_rebuild_status(db, _empty_payload)
        return json.dumps(_empty_payload, ensure_ascii=False)

"""

ANCHOR_CALL = '    for lineage_root, match_info in seen_sessions.items():'
ANCHOR_IMPORT = 'import json'
ANCHOR_DOCSTRING = 'No LLM calls — every shape returns actual DB messages.\n"""'
NEW_DOCSTRING = 'No LLM calls — every shape returns actual DB messages. Performs semantic vector fallback via index_all when FTS5 results are sparse.\n"""'


def apply_patch():
    if not TARGET_FILE.exists():
        print(f'Error: target file {TARGET_FILE} does not exist', file=sys.stderr)
        sys.exit(2)

    content = TARGET_FILE.read_text(encoding='utf-8')
    original_backup = content

    if MARKER in content:
        pattern = r'def _vector_supplement\([\s\S]*?\n\n\n'
        if re.search(pattern, content):
            new_content = re.sub(pattern, lambda m: VECTOR_SUPPLEMENT_FUNC, content, count=1)
            TARGET_FILE.write_text(new_content, encoding='utf-8')
            py_compile.compile(str(TARGET_FILE), doraise=True)
            print('vector-hybrid patch updated with multi-profile and threshold support')
            sys.exit(0)

    if ANCHOR_CALL not in content or ANCHOR_IMPORT not in content or ANCHOR_DOCSTRING not in content:
        print('Warning: patch anchors not found in target file (upstream changed). Skipping patch.', file=sys.stderr)
        sys.exit(2)

    new_content = content.replace(ANCHOR_DOCSTRING, NEW_DOCSTRING, 1)
    new_content = new_content.replace(ANCHOR_IMPORT, VECTOR_SUPPLEMENT_FUNC + ANCHOR_IMPORT, 1)
    new_content = new_content.replace(ANCHOR_CALL, CALL_BLOCK, 1)

    if EARLY_EXIT_BLOCK in new_content:
        new_content = new_content.replace(EARLY_EXIT_BLOCK, '', 1)

    try:
        TARGET_FILE.write_text(new_content, encoding='utf-8')
        py_compile.compile(str(TARGET_FILE), doraise=True)
    except Exception as e:
        print(f'Syntax validation failed after patching: {e}. Rolling back.', file=sys.stderr)
        TARGET_FILE.write_text(original_backup, encoding='utf-8')
        sys.exit(3)

    print('vector-hybrid patch applied')
    sys.exit(0)


if __name__ == '__main__':
    apply_patch()
