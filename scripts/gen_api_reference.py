"""Generate docs/api/api_reference.md from api/openapi.yaml (P6-T02: API reference generated from OpenAPI)."""
import os, yaml
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
d = yaml.safe_load(open(os.path.join(ROOT, "api/openapi.yaml")))
out = [f"# Nafs-Aql API reference (v{d['info']['version']})", "", "Generated from `api/openapi.yaml` by `scripts/gen_api_reference.py`; do not edit by hand.", "",
       d["info"]["description"].strip(), "",
       "Authentication: send `Authorization: Bearer <token>` when the gateway runs with `NAFS_AUTH_TOKENS`. Browser requests must be same-origin (cross-origin requests get 403).", ""]
by_tag = {}
for path, ops in d["paths"].items():
    for m, op in ops.items():
        if not isinstance(op, dict):
            continue
        by_tag.setdefault((op.get("tags") or ["other"])[0], []).append((path, m.upper(), op))
def schema_line(sch):
    if not sch: return ""
    if "$ref" in sch: return f"`{sch['$ref'].split('/')[-1]}`"
    if sch.get("type") == "object" and sch.get("properties"):
        req = set(sch.get("required", []))
        return ", ".join(f"`{k}`{'*' if k in req else ''}" for k in sch["properties"])
    return f"`{sch.get('type', 'object')}`"
for tag in sorted(by_tag):
    out += [f"## {tag}", ""]
    for path, m, op in sorted(by_tag[tag]):
        out += [f"### `{m} {path}`", "", op.get("summary", ""), ""]
        params = op.get("parameters") or []
        if params:
            out.append("Parameters: " + ", ".join(f"`{p['name']}` ({p['in']}{', required' if p.get('required') else ''})" for p in params) + ".")
            out.append("")
        rb = (op.get("requestBody") or {}).get("content", {}).get("application/json", {}).get("schema")
        if rb:
            out += [f"Request body: {schema_line(rb)} (* = required).", ""]
        codes = ", ".join(f"`{c}`" for c in op.get("responses", {}))
        out += [f"Responses: {codes}.", ""]
out += ["## Schemas", ""] + [f"- `{k}`" + (f": {v.get('description')}" if v.get("description") else "") for k, v in d["components"]["schemas"].items()]
open(os.path.join(ROOT, "docs/api/api_reference.md"), "w").write("\n".join(out) + "\n")
print(f"docs/api/api_reference.md: {len(d['paths'])} paths")
