#!/usr/bin/env bash
#
# Start EdgeSense locally without Docker.
#
#   ./scripts/dev-stack.sh          start ClickHouse + API + portal
#   ./scripts/dev-stack.sh stop     stop them
#   ./scripts/dev-stack.sh status   show what is running
#
# Exists because Docker is a heavy dependency for something whose interesting
# parts (redaction, the agent loop, the eval) need no infrastructure at all.
# docker-compose.yml remains the deployment story; this is the laptop story.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_DIR="${EDGESENSE_RUN_DIR:-$ROOT/.run}"
# Binary and data directory must not be the same path.
CH_DATA="${EDGESENSE_CH_DATA:-$RUN_DIR/ch-data}"
CH_BIN="${EDGESENSE_CH_BIN:-$RUN_DIR/bin/clickhouse}"
PY="$ROOT/.venv/bin/python"
API_PORT="${API_PORT:-8099}"
PORTAL_PORT="${PORTAL_PORT:-5173}"

mkdir -p "$RUN_DIR" "$CH_DATA" "$(dirname "$CH_BIN")"

log()  { printf '\033[36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[33m!!\033[0m  %s\n' "$*"; }

pidfile() { echo "$RUN_DIR/$1.pid"; }

alive() {
  local f; f="$(pidfile "$1")"
  [[ -f "$f" ]] && kill -0 "$(cat "$f")" 2>/dev/null
}

start_one() {
  local name="$1"; shift
  if alive "$name"; then log "$name already running (pid $(cat "$(pidfile "$name")"))"; return; fi
  "$@" > "$RUN_DIR/$name.log" 2>&1 &
  echo $! > "$(pidfile "$name")"
  log "$name started (pid $!) -> $RUN_DIR/$name.log"
}

stop_all() {
  for name in portal api clickhouse; do
    local f; f="$(pidfile "$name")"
    if [[ -f "$f" ]]; then
      # Kill the process group: uvicorn and vite both spawn children that
      # would otherwise survive and keep the port bound.
      pkill -P "$(cat "$f")" 2>/dev/null || true
      kill "$(cat "$f")" 2>/dev/null || true
      rm -f "$f"
      log "$name stopped"
    fi
  done
}

ensure_clickhouse_binary() {
  [[ -x "$CH_BIN" ]] && return
  log "fetching the ClickHouse single-file binary (~150 MB, once)"
  local arch; arch="$(uname -m)"
  local plat="macos-aarch64"
  case "$(uname -s)-$arch" in
    Darwin-arm64|Darwin-aarch64) plat="macos-aarch64" ;;
    Darwin-x86_64)               plat="macos" ;;
    Linux-x86_64)                plat="amd64" ;;
    Linux-aarch64|Linux-arm64)   plat="aarch64" ;;
  esac
  curl -sSL -o "$CH_BIN" "https://builds.clickhouse.com/master/$plat/clickhouse"
  chmod +x "$CH_BIN"
  xattr -d com.apple.quarantine "$CH_BIN" 2>/dev/null || true
}

wait_for() {
  local url="$1" name="$2" tries="${3:-60}"
  for _ in $(seq 1 "$tries"); do
    curl -sf -m 2 "$url" >/dev/null 2>&1 && { log "$name ready"; return 0; }
    sleep 1
  done
  warn "$name did not become ready; see $RUN_DIR/$name.log"
  return 1
}

case "${1:-start}" in
  stop)   stop_all; exit 0 ;;
  status)
    for name in clickhouse api portal; do
      alive "$name" && echo "  $name  running (pid $(cat "$(pidfile "$name")"))" \
                    || echo "  $name  stopped"
    done
    exit 0 ;;
esac

[[ -x "$PY" ]] || { warn "no venv found - run 'make install' first"; exit 1; }

ensure_clickhouse_binary
start_one clickhouse env -C "$CH_DATA" "$CH_BIN" server
wait_for "http://localhost:8123/ping" clickhouse 60

log "applying schema"
"$PY" - <<'PY'
import pathlib, clickhouse_connect
sql = pathlib.Path("deploy/clickhouse/schema.sql").read_text()
client = clickhouse_connect.get_client(host="localhost", port=8123)
stmt, current = [], []
for line in sql.splitlines():
    s = line.strip()
    if s.startswith("--") or not s:
        continue
    current.append(line)
    if s.endswith(";"):
        stmt.append("\n".join(current).strip().rstrip(";")); current = []
for s in stmt:
    client.command(s)
rows = client.query("SELECT count() FROM edgesense.call_summaries").result_rows[0][0]
print(f"    schema applied; {rows} calls already stored")
PY

start_one api env \
  PYTHONPATH="$ROOT/services/sink:$ROOT/services/worker:$ROOT" \
  POLICY_CATALOG="$ROOT/tools/corpus/policies.yaml" \
  "$PY" -m uvicorn sink.api:app --host 0.0.0.0 --port "$API_PORT" --log-level warning
wait_for "http://localhost:$API_PORT/api/health" api 40

if [[ -d "$ROOT/portal/node_modules" ]]; then
  start_one portal env -C "$ROOT/portal" npm run dev -- --port "$PORTAL_PORT"
  wait_for "http://localhost:$PORTAL_PORT/" portal 40
else
  warn "portal deps missing - run 'make portal-install' then re-run this script"
fi

CALLS=$("$PY" -c "
import clickhouse_connect
c=clickhouse_connect.get_client(host='localhost',port=8123,database='edgesense')
print(c.query('SELECT count() FROM call_summaries').result_rows[0][0])" 2>/dev/null || echo 0)

echo
log "EdgeSense is up"
echo "    portal   http://localhost:$PORTAL_PORT"
echo "    api      http://localhost:$API_PORT/api/health"
echo
if [[ "$CALLS" == "0" ]]; then
  warn "no analytics data yet - populate it with:"
  echo "      make seed        # ~5 min, runs the real pipeline over the corpus"
  echo "      make loadtest    # fills the latency panel with real measurements"
fi
echo "    stop with: ./scripts/dev-stack.sh stop"
