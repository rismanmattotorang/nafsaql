#!/usr/bin/env bash
# Build a release archive (OR-002): gateway and eval binaries, configuration schema and samples, API contracts,
# the model manifest with SHA-256 hashes, deployment recipes, training tools and documentation, plus SHA256SUMS.
# Usage: scripts/package_release.sh            -> dist/nafs-aql-<version>-<os>-<arch>.tar.gz
set -euo pipefail
cd "$(dirname "$0")/.."
VER=$(grep -m1 '^version' rust/Cargo.toml | cut -d'"' -f2)
OS=$(uname -s | tr '[:upper:]' '[:lower:]'); ARCH=$(uname -m)
NAME="nafs-aql-${VER}-${OS}-${ARCH}"; OUT="dist/$NAME"
(cd rust && cargo build --release -p nafs-gateway -p nafs-eval >/dev/null)
rm -rf "$OUT"; mkdir -p "$OUT"/{bin,config,api,deploy,training,docs,data}
cp rust/target/release/nafs-gateway rust/target/release/nafs-eval "$OUT/bin/"
cp config/nafs-aql.sample.toml config/nafs-aql.docker.toml config/models.manifest.json config/vllm_launch.yaml "$OUT/config/"
cp -R api/openapi.yaml api/openapi.json api/schemas "$OUT/api/"
cp -R deploy/docker deploy/systemd deploy/launchd deploy/tls deploy/monitoring deploy/lima "$OUT/deploy/"
rm -f "$OUT/deploy/docker/secrets/nafs_tokens"
cp training/tazkiya.py training/train_unsloth.py training/RUNBOOK.md "$OUT/training/"
cp -R data/bench "$OUT/data/"
cp README.md ASSUMPTIONS.md docs/roadmap/STATUS.md "$OUT/docs/"
for f in docs/operations/*.md docs/user_guide.md docs/api/api_reference.md docs/security/security_review.json; do [ -f "$f" ] && cp "$f" "$OUT/docs/"; done
scripts_keep="serve_llama.sh serve_vllm_docker.sh gateway.sh fetch_models.sh verify_deploy.sh security_tests.py mvp_acceptance.py contract_test.py e2e_sweep.py vllm_check.py eagle3_ab.py"
mkdir -p "$OUT/scripts"; for s in $scripts_keep; do cp "scripts/$s" "$OUT/scripts/"; done
"$OUT/bin/nafs-gateway" --version > "$OUT/VERSION"
(cd "$OUT" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 shasum -a 256 > SHA256SUMS)
tar -C dist -czf "dist/$NAME.tar.gz" "$NAME"
shasum -a 256 "dist/$NAME.tar.gz" | tee "dist/$NAME.tar.gz.sha256"
echo "release: dist/$NAME.tar.gz ($(du -h "dist/$NAME.tar.gz" | cut -f1)), $(wc -l < "$OUT/SHA256SUMS") files hashed"
