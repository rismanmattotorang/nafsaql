#!/usr/bin/env bash
# SurrealDB for the Knowledge Base (Phase 20): one server process, like the engines. Usage:
#   scripts/serve_surreal.sh install|start|stop|status|logs|spike
# Listens on 127.0.0.1:8300 (SURREAL_PORT), stores in data/runtime/surreal/kb.db (surrealkv, pure Rust, no RocksDB).
# The root password is generated once into data/runtime/surreal/root.secret (mode 600); the gateway reads the same file
# ([knowledge] password_file). SURREAL_VERSION pins the release; `install` fetches it from GitHub and checks its SHA-256.
set -euo pipefail
cd "$(dirname "$0")/.."
VERSION=${SURREAL_VERSION:-3.3.5}
SHA256_DARWIN_AMD64=a274451b30cf639df3b180b0e82ab8f7c52bc5244fa66aa65155a085f3eb6d31
BIN=${SURREAL_BIN:-tools/surrealdb/surreal}
PORT=${SURREAL_PORT:-8300}
DIR=data/runtime/surreal
SECRET=$DIR/root.secret
RUN=run; mkdir -p "$RUN" logs "$DIR"

install() {
  if [ -x "$BIN" ] && "$BIN" version 2>/dev/null | grep -q "^$VERSION "; then echo "surreal $VERSION already installed"; return; fi
  local os arch asset
  case "$(uname -s)" in Darwin) os=darwin ;; Linux) os=linux ;; *) echo "unsupported OS"; exit 1 ;; esac
  case "$(uname -m)" in x86_64) arch=amd64 ;; arm64|aarch64) arch=arm64 ;; *) echo "unsupported CPU"; exit 1 ;; esac
  asset="surreal-v$VERSION.$os-$arch.tgz"
  mkdir -p tools/surrealdb
  curl -fL --retry 3 -o "tools/surrealdb/$asset" "https://github.com/surrealdb/surrealdb/releases/download/v$VERSION/$asset"
  if [ "$os-$arch" = darwin-amd64 ] && [ "$VERSION" = 3.3.5 ]; then
    echo "$SHA256_DARWIN_AMD64  tools/surrealdb/$asset" | shasum -a 256 -c -
  fi
  tar -xzf "tools/surrealdb/$asset" -C tools/surrealdb && rm -f "tools/surrealdb/$asset"
  "$BIN" version
}

secret() {
  if [ ! -s "$SECRET" ]; then (umask 077; LC_ALL=C tr -dc 'A-Za-z0-9' < /dev/urandom | head -c 32 > "$SECRET"); fi
  cat "$SECRET"
}

sql() { # run SurrealQL from stdin; the script selects its namespace and database with USE (headers naming a missing
  # namespace fail the whole request)
  curl -sf -u "root:$(secret)" -H 'Accept: application/json' --data-binary @- "http://127.0.0.1:$PORT/sql"
}

case "${1:-status}" in
  install) install ;;
  start)
    if [ -f "$RUN/surreal.pid" ] && kill -0 "$(cat "$RUN/surreal.pid")" 2>/dev/null; then echo "surreal already running (pid $(cat $RUN/surreal.pid))"; exit 0; fi
    [ -x "$BIN" ] || install
    pass=$(secret)
    # The password goes through the environment, not argv, so it does not show in `ps`.
    SURREAL_USER=root SURREAL_PASS="$pass" nohup "$BIN" start --bind "127.0.0.1:$PORT" --log "${SURREAL_LOG:-warn}" \
      --no-banner "surrealkv://$DIR/kb.db" > logs/surreal.log 2>&1 &
    echo $! > "$RUN/surreal.pid"
    for _ in $(seq 1 60); do curl -sf "127.0.0.1:$PORT/health" >/dev/null && { echo "surreal $VERSION healthy on :$PORT (pid $(cat $RUN/surreal.pid))"; exit 0; }; sleep 0.5; done
    echo "surreal did not become healthy; see logs/surreal.log"; exit 1 ;;
  stop) [ -f "$RUN/surreal.pid" ] && kill "$(cat $RUN/surreal.pid)" 2>/dev/null && echo "stopped surreal"; rm -f "$RUN/surreal.pid" ;;
  status) printf ":%s " "$PORT"; if curl -sf "127.0.0.1:$PORT/health" >/dev/null; then curl -s "127.0.0.1:$PORT/version"; echo; else echo down; fi ;;
  logs) tail -n 40 logs/surreal.log ;;
  spike)
    # Proves vector (HNSW), full-text (BM25) and graph (RELATE, traversal) queries in a throwaway database.
    out=$(sql <<'EOF'
DEFINE NAMESPACE IF NOT EXISTS nafs; USE NS nafs; DEFINE DATABASE IF NOT EXISTS spike; USE DB spike;
REMOVE TABLE IF EXISTS s_chunk; REMOVE TABLE IF EXISTS s_entity; REMOVE TABLE IF EXISTS s_mentions; REMOVE TABLE IF EXISTS s_related;
DEFINE ANALYZER IF NOT EXISTS s_text TOKENIZERS class FILTERS lowercase, ascii, snowball(english);
DEFINE TABLE s_chunk SCHEMAFULL; DEFINE FIELD text ON s_chunk TYPE string; DEFINE FIELD embedding ON s_chunk TYPE array<float>;
DEFINE INDEX s_chunk_text ON s_chunk FIELDS text FULLTEXT ANALYZER s_text BM25;
DEFINE INDEX s_chunk_vec ON s_chunk FIELDS embedding HNSW DIMENSION 3 DIST COSINE;
DEFINE TABLE s_entity SCHEMALESS;
DEFINE TABLE s_mentions TYPE RELATION FROM s_chunk TO s_entity; DEFINE TABLE s_related TYPE RELATION FROM s_entity TO s_entity;
CREATE s_chunk:a SET text = "The memory module compresses documents into sixteen tokens", embedding = [1.0, 0.0, 0.0];
CREATE s_chunk:b SET text = "SurrealDB stores vectors and graphs", embedding = [0.0, 1.0, 0.0];
CREATE s_chunk:c SET text = "Graph retrieval walks entity relations", embedding = [0.0, 0.7, 0.7];
CREATE s_entity:surrealdb SET name = "SurrealDB"; CREATE s_entity:graph SET name = "graph";
RELATE s_chunk:b->s_mentions->s_entity:surrealdb; RELATE s_chunk:c->s_mentions->s_entity:graph;
RELATE s_entity:surrealdb->s_related->s_entity:graph SET predicate = "stores";
SELECT VALUE id FROM s_chunk WHERE embedding <|1,40|> [0.0, 0.9, 0.3];
SELECT id, search::score(1) AS score FROM s_chunk WHERE text @1,OR@ "graphs vectors" ORDER BY score DESC LIMIT 1;
SELECT VALUE ->s_mentions->s_entity->s_related->s_entity.name FROM ONLY s_chunk:b;
REMOVE DATABASE spike;
EOF
)
    python3 - "$out" <<'PY'
import json, sys
res = json.loads(sys.argv[1])
bad = [r for r in res if r.get("status") != "OK"]
knn, hop = res[-4]["result"], res[-2]["result"]
fts = [r["id"] for r in res[-3]["result"]]
ok = not bad and knn == ["s_chunk:b"] and fts == ["s_chunk:b"] and hop == ["graph"]
print(f"vector knn -> {knn}; full-text -> {fts}; two-hop graph -> {hop}; {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
PY
    ;;
  *) echo "usage: $0 install|start|stop|status|logs|spike"; exit 2 ;;
esac
