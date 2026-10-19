#!/usr/bin/env bash
# Validate every machine-readable artefact. Needs: python3 (jsonschema, pyyaml), uv (for openapi-spec-validator and tomllib), cargo.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "== OpenAPI 3.1";   uvx --python 3.12 --from openapi-spec-validator openapi-spec-validator api/openapi.yaml
echo "== JSON Schemas (draft 2020-12)"
python3 -I - <<'PY'
import glob, json
from jsonschema import Draft202012Validator
for f in sorted(glob.glob("api/schemas/*.json")):
    Draft202012Validator.check_schema(json.load(open(f))); print("ok", f)
PY
echo "== YAML / TOML / shell"
python3 -I -c "import yaml; yaml.safe_load(open('config/vllm_launch.yaml')); print('ok config/vllm_launch.yaml')"
uvx --python 3.12 python -c "import tomllib; tomllib.load(open('config/nafs-aql.sample.toml','rb')); print('ok config/nafs-aql.sample.toml')"
bash -n config/llama_server_profile_a.sh && echo "ok config/llama_server_profile_a.sh"
echo "== Cargo workspace"; (cd rust && cargo metadata --format-version 1 >/dev/null && echo "ok rust/ workspace resolves")
echo "== Requirement and task consistency"; python3 scripts/gen_tables.py
