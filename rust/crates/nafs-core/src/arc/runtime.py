"""Nafs-Aql reasoning cache runtime (Phase 38, after CPU-ARC, arXiv 2611.09891).

A cache entry is a set of *semantic functions*: plain Python functions that name their inputs, their outputs and
the functions they call by natural-language descriptions instead of identifiers. This file is the whole runtime
they need. It runs inside the verifier sandbox (no network, no files outside the run directory), reads one JSON
request on stdin and writes one JSON result on stdout:

    stdin : {"root": "<function id>", "params": {"<description>": value, ...}, "links": {"<call>": "<id>"}}
    stdout: {"ok": true, "outputs": {...}, "answer": ..., "narration": [...]}  or  {"ok": false, "error": "..."}

Primitives (the CPU-ARC set without the model calls; extraction happens before dispatch, see arc/mod.rs):
    load(ctx, description, d_type=None)    a value from the context, found by its description
    do(ctx, description, payloads)         call another semantic function, found by its description
    narrate(ctx, message)                  one human-readable step of the explanation
    convert(value, d_type)                 programmatic type coercion ("12 cm" -> 12.0, "[1, 2]" -> [1, 2])

Descriptions are matched without a model: exact text first, then the static link table the gateway resolved with
the embedder when the entry was registered, then word overlap. Only the standard library is used.
"""

import ast
import json
import math
import re
import sys
from fractions import Fraction

REGISTRY = {}  # function id -> (description, callable, is_root)
LINKS = {}  # call description -> function id (resolved at registration with the embedder)
MIN_MATCH = 0.5  # word-overlap floor for a description that is neither exact nor linked


class CacheInvocationError(Exception):
    """The entry cannot answer this request; the gateway falls through to the models."""


def semantic(description, root=False, fid=None):
    def wrap(fn):
        REGISTRY[fid or fn.__name__] = (description, fn, root)
        fn.__semantic__ = description
        return fn

    return wrap


_STOP = set(
    "a an the of to for in on at by with from and or is are be as its it this that which what into per each "
    "value given number".split()
)


def _words(s):
    s = str(s).lower().replace("_", " ")
    out = []
    for w in re.findall(r"[a-z0-9]+", s):
        if w in _STOP:
            continue
        # A light stem, so "integers" meets "integer" and "divisors" meets "divisor".
        for suf in ("ies", "es", "s"):
            if len(w) > 4 and w.endswith(suf):
                w = w[: -len(suf)] + ("y" if suf == "ies" else "")
                break
        out.append(w)
    return out


def similarity(a, b):
    """Word-overlap similarity in [0, 1]: Dice over content words, plus a small bonus for a shared head word."""
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return 1.0 if str(a).strip().lower() == str(b).strip().lower() else 0.0
    sa, sb = set(wa), set(wb)
    dice = 2 * len(sa & sb) / (len(sa) + len(sb))
    return min(1.0, dice + (0.05 if wa[-1] == wb[-1] else 0.0))


def best_match(query, keys, floor=MIN_MATCH):
    if query in keys:
        return query, 1.0
    q = str(query).strip().lower()
    for k in keys:
        if str(k).strip().lower() == q:
            return k, 1.0
    scored = sorted(((similarity(query, k), k) for k in keys), key=lambda t: -t[0])
    if scored and scored[0][0] >= floor:
        # Two keys equally close is an ambiguity, not a match.
        if len(scored) > 1 and abs(scored[0][0] - scored[1][0]) < 1e-9:
            raise CacheInvocationError(f"ambiguous description {query!r}: {scored[0][1]!r} or {scored[1][1]!r}")
        return scored[0][1], scored[0][0]
    return None, scored[0][0] if scored else 0.0


class Context:
    """A stack of memory layers. `load` searches the most local layer first."""

    def __init__(self, params=None, narration=None, depth=0):
        self.layers = [dict(params or {})]
        self.narration = narration if narration is not None else []
        self.depth = depth

    def push(self, payloads):
        self.layers.append(dict(payloads or {}))
        self.depth += 1

    def pop(self):
        self.layers.pop()
        self.depth -= 1


class Result(dict):
    """A function's output dictionary; `load(result, ...)` reads it by description."""


_NUM = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?|[-+]?\.\d+")


def _number(s):
    s = str(s).strip()
    frac = re.fullmatch(r"\s*([-+]?\d+)\s*/\s*(\d+)\s*", s)
    if frac:
        return Fraction(int(frac.group(1)), int(frac.group(2)))
    m = _NUM.search(s.replace("−", "-"))
    if not m:
        raise ValueError(f"no number in {s!r}")
    t = m.group(0).replace(",", "")
    return float(t) if any(c in t for c in ".eE") else int(t)


def convert(value, d_type=None):
    """Programmatic coercion. The CPU-ARC `smart_convert` model tier is not used: a value that cannot be converted
    here is an invocation error, and the gateway answers with the models instead."""
    if d_type is None or value is None:
        return value
    try:
        if d_type is int:
            if isinstance(value, bool):
                return int(value)
            if isinstance(value, (int,)):
                return value
            if isinstance(value, (float, Fraction)):
                if float(value) != int(value):
                    raise ValueError(f"{value!r} is not a whole number")
                return int(value)
            n = _number(value)
            if float(n) != int(n):
                raise ValueError(f"{value!r} is not a whole number")
            return int(n)
        if d_type is float:
            if isinstance(value, (int, float, Fraction)) and not isinstance(value, bool):
                return float(value)
            return float(_number(value))
        if d_type is Fraction:
            if isinstance(value, Fraction):
                return value
            if isinstance(value, (int,)):
                return Fraction(value)
            return Fraction(str(value).strip())
        if d_type is bool:
            if isinstance(value, bool):
                return value
            s = str(value).strip().lower()
            if s in ("true", "yes", "1", "y"):
                return True
            if s in ("false", "no", "0", "n"):
                return False
            raise ValueError(f"{value!r} is not a yes or no")
        if d_type is str:
            return str(value)
        if d_type in (list, tuple, set, dict):
            if isinstance(value, d_type):
                return value
            if isinstance(value, (list, tuple, set)) and d_type is not dict:
                return d_type(value)
            v = ast.literal_eval(str(value).strip())
            return d_type(v) if d_type is not dict else dict(v)
        return d_type(value)
    except CacheInvocationError:
        raise
    except Exception as e:  # noqa: BLE001 - every failure means "cannot answer from the cache"
        raise CacheInvocationError(f"cannot convert {value!r} to {getattr(d_type, '__name__', d_type)}: {e}")


def _exact(description, keys):
    q = " ".join(str(description).lower().split())
    for k in keys:
        if " ".join(str(k).lower().split()) == q:
            return k
    return None


def load(ctx, description, d_type=None, **_):
    layers = ctx.layers if isinstance(ctx, Context) else [ctx]
    # The same description in any scope beats a near one in a closer scope: a callee given some inputs reads the rest
    # from its caller, and a near miss among the given ones would bind the wrong value.
    for layer in reversed(layers):
        key = _exact(description, layer.keys())
        if key is not None:
            return convert(layer[key], d_type)
    for layer in reversed(layers):
        key, _score = best_match(description, list(layer.keys()))
        if key is not None:
            return convert(layer[key], d_type)
    have = sorted({k for layer in layers for k in layer})
    raise CacheInvocationError(f"no value for {description!r} (have {have})")


def narrate(ctx, message):
    if isinstance(ctx, Context):
        ctx.narration.append(("  " * max(0, ctx.depth)) + str(message))


def resolve(description):
    if description in REGISTRY:
        return description
    fid = LINKS.get(description)
    if fid in REGISTRY:
        return fid
    keys = {d: k for k, (d, _f, _r) in REGISTRY.items()}
    hit, _score = best_match(description, list(keys))
    if hit is None:
        raise CacheInvocationError(f"no semantic function for {description!r}")
    return keys[hit]


def do(ctx, description, payloads=None, **kw):
    payloads = dict(payloads or {}, **kw.get("params", {}))
    fid = resolve(description)
    _d, fn, _root = REGISTRY[fid]
    if ctx.depth > 24:
        raise CacheInvocationError("semantic call depth exceeded")
    ctx.push(payloads)
    try:
        out = fn(ctx)
    finally:
        ctx.pop()
    if not isinstance(out, dict):
        out = {"result": out}
    return Result(out)


def _plain(v):
    """JSON-safe form of a value: fractions as "p/q" strings plus a float, sets as sorted lists."""
    if isinstance(v, bool) or v is None or isinstance(v, (int, str)):
        return v
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            raise CacheInvocationError("the computation produced a non-finite number")
        return int(v) if v.is_integer() and abs(v) < 1e15 else v
    if isinstance(v, Fraction):
        return str(v) if v.denominator != 1 else v.numerator
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, set):
        return sorted((_plain(x) for x in v), key=lambda x: (str(type(x)), x))
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    try:
        return float(v)
    except Exception:  # noqa: BLE001
        return str(v)


def load_entries(code):
    """Load each entry's code into its own namespace; its semantic functions register as "<entry>:<name>"."""
    import math as _m, fractions as _f, itertools as _it, functools as _ft
    for eid, src in code:
        bad = problems_of(ast.parse(src)) if "problems_of" in globals() else []
        if bad:
            raise CacheInvocationError(f"entry {eid}: {'; '.join(bad)}")

        def scoped(description, root=False, fid=None, _eid=eid):
            def wrap(fn):
                REGISTRY[fid or f"{_eid}:{fn.__name__}"] = (description, fn, root)
                fn.__semantic__ = description
                return fn

            return wrap

        ns = {"__name__": f"arc_{eid}", "__builtins__": __builtins__, "semantic": scoped, "load": load, "do": do,
              "narrate": narrate, "convert": convert, "CacheInvocationError": CacheInvocationError,
              "math": _m, "Fraction": _f.Fraction, "fractions": _f, "itertools": _it, "functools": _ft}
        exec(compile(src, f"entry_{eid}.py", "exec"), ns)


def invoke(root, params, links=None):
    LINKS.update(links or {})
    if root not in REGISTRY:
        roots = [k for k, (_d, _f, r) in REGISTRY.items() if r]
        if len(roots) != 1:
            raise CacheInvocationError(f"no root function {root!r}")
        root = roots[0]
    _d, fn, _r = REGISTRY[root]
    ctx = Context(params)
    out = fn(ctx)
    if not isinstance(out, dict):
        out = {"answer": out}
    outputs = _plain(out)
    answer = outputs.get("answer")
    if answer is None:
        k, _s = best_match("answer final result", list(outputs.keys()), floor=0.0)
        answer = outputs.get(k)
    return {"ok": True, "outputs": outputs, "answer": answer, "narration": ctx.narration[:200]}


def main():
    try:
        req = json.loads(sys.stdin.read() or "{}")
        mode = req.get("mode", "invoke")
        if mode in ("describe", "compile"):
            main_tools_req(req)
            return
        load_entries(req.get("code") or [])
        res = invoke(req.get("root", ""), req.get("params") or {}, req.get("links") or {})
    except CacheInvocationError as e:
        res = {"ok": False, "error": str(e), "kind": "invocation"}
    except RecursionError:
        res = {"ok": False, "error": "recursion too deep", "kind": "exception"}
    except Exception as e:  # noqa: BLE001
        res = {"ok": False, "error": f"{type(e).__name__}: {e}", "kind": "exception"}
    sys.stdout.write(json.dumps(res, default=str))
