#!/usr/bin/env bash
# Interleaved head-to-head: for each round, vLLM Semantic Router's Go benchmarks (offline, from third_party/semantic-router) and then
# Nafs-Aql's counterparts (rust/crates/nafs-bench/examples/h2h.rs), on the same machine, so load drift hits both. ROUNDS=3 by default.
# Usage: benchmarks/headtohead/run.sh   (writes benchmarks/headtohead/round_N_{go.txt,rust.json}, then summarise.py)
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=benchmarks/headtohead
ROUNDS=${ROUNDS:-3}
(cd rust && cargo build --release -p nafs-bench --example h2h 2>&1 | tail -n 1)
for r in $(seq 1 $ROUNDS); do
  echo "round $r: load $(uptime | sed 's/.*load averages*: //')" | tee $OUT/round_${r}_load.txt
  (cd third_party/semantic-router/src/semantic-router && {
     GOPROXY=off GOFLAGS=-mod=mod go test ./pkg/classification/lexical -run xxx -bench . -benchtime 300ms 2>&1 | grep -E "^Benchmark"
     GOPROXY=off GOFLAGS=-mod=mod go test ./pkg/promptcompression -run xxx -bench . -benchtime 300ms 2>&1 | grep -E "^Benchmark"
     GOPROXY=off GOFLAGS=-mod=mod go test ./pkg/embedding/vecmath -run xxx -bench 'Dot' -benchtime 300ms 2>&1 | grep -E "^Benchmark"
  }) > $OUT/round_${r}_go.txt
  rust/target/release/examples/h2h --json $OUT/round_${r}_rust.json > $OUT/round_${r}_rust.txt
done
python3 $OUT/summarise.py
