#!/bin/bash
# Vector index cron wrapper — runs indexer + formats message
# Used by cron job: 向量索引（会话）

set -e

INDEXER="/usr/bin/python3 $HOME/.hermes/scripts/index_all.py"
LOG=$(mktemp)

# ── On-demand heal: embedding server runs on Windows (port 8081, Vulkan).
#    If down, relaunch it before indexing.
HEALED=0
if ! curl -sf --noproxy '*' -o /dev/null --max-time 5 http://127.0.0.1:8081/health; then
    powershell.exe -NoProfile -WindowStyle Hidden -Command "Start-Process 'llama-server.exe' -ArgumentList '-m','models/bge-m3-Q4_K_M.gguf','--embeddings','--host','127.0.0.1','--port','8081','-b','2048','-ub','2048','-ngl','99','-t','4' -WindowStyle Hidden" 2>/dev/null || true
    for i in $(seq 1 20); do
        curl -sf --noproxy '*' -o /dev/null --max-time 2 http://127.0.0.1:8081/health && HEALED=1 && break
        sleep 1
    done
fi

# Run indexer
$INDEXER > "$LOG" 2>&1 || true

cat "$LOG"
rm -f "$LOG"
