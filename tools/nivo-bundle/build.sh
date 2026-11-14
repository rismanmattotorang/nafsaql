#!/usr/bin/env bash
# Rebuild rust/crates/nafs-web/assets/nivo.js from entry.jsx with the versions pinned in package-lock.json.
set -euo pipefail
cd "$(dirname "$0")"
npm ci --ignore-scripts --no-audit --no-fund
npx esbuild entry.jsx --bundle --minify --format=esm --target=es2020 \
  --define:process.env.NODE_ENV='"production"' --legal-comments=none \
  --outfile=../../rust/crates/nafs-web/assets/nivo.js
ls -l ../../rust/crates/nafs-web/assets/nivo.js
