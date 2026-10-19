"""Generate api/schemas/Config.schema.json (OR-002 configuration schema) from config/nafs-aql.sample.toml and
validate the shipped configurations against it. Run with Python >= 3.11 (tomllib):
    uvx --python 3.12 --with jsonschema python scripts/gen_config_schema.py"""
import json, os, sys
try:
    import tomllib
except ImportError:  # Python < 3.11
    import tomli as tomllib
from jsonschema import Draft202012Validator
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REQUIRED_SECTIONS = ["schema_version", "server", "assistant", "backend", "backends", "gate", "deliberation", "storage"]
def schema_of(v, path=""):
    if isinstance(v, bool): return {"type": "boolean"}
    if isinstance(v, int): return {"type": "integer"} if not path.endswith(("_ms", "_s", "_tokens")) else {"type": "integer", "minimum": 0}
    if isinstance(v, float): return {"type": "number"}
    if isinstance(v, str): return {"type": "string"}
    if isinstance(v, list): return {"type": "array", "items": schema_of(v[0], path) if v else {}}
    if isinstance(v, dict):
        if path in (".backends", ".gate.e2_hat", ".gate.delta_c_s", ".deliberation.caps"):  # maps keyed by name or domain
            inner = schema_of(next(iter(v.values())), path + ".*")
            return {"type": "object", "additionalProperties": inner}
        return {"type": "object", "properties": {k: schema_of(x, f"{path}.{k}") for k, x in v.items()}, "additionalProperties": False}
    return {}
sample = tomllib.load(open(os.path.join(ROOT, "config/nafs-aql.sample.toml"), "rb"))
s = schema_of(sample)
s["required"] = REQUIRED_SECTIONS
s = {"$schema": "https://json-schema.org/draft/2020-12/schema", "$id": "https://nafs-aql.example/schemas/Config.schema.json",
     "title": "Nafs-Aql gateway configuration (config/nafs-aql.toml)", **s}
json.dump(s, open(os.path.join(ROOT, "api/schemas/Config.schema.json"), "w"), indent=2)
v = Draft202012Validator(s); bad = 0
for f in ("config/nafs-aql.toml", "config/nafs-aql.sample.toml", "config/nafs-aql.docker.toml"):
    errs = list(v.iter_errors(tomllib.load(open(os.path.join(ROOT, f), "rb"))))
    print(("ok   " if not errs else "FAIL ") + f, "; ".join(e.message for e in errs[:3])); bad += bool(errs)
sys.exit(1 if bad else 0)
