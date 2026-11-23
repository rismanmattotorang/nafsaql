"""Nafs-Aql reasoning cache tools (Phase 38): describe an entry's code, and compile a whole cache into one module.

Runs inside the verifier sandbox with the runtime prepended (so `similarity`, `best_match` and `convert` exist).
Nothing here executes the entry code except `describe`, which only imports it to read the registry.

    {"mode": "describe", "entry": "<id>", "code": "..."}
        -> {"ok": true, "functions": [{"name", "description", "root", "loads", "calls", "returns", "narrates"}],
            "problems": [...]}
    {"mode": "compile", "entries": [{"id", "code", "template", "params": [{"name","type","description"}],
                                      "links": {...}}]}
        -> {"ok": true, "module": "<python source>"}

Compilation follows CPU-ARC section 2.2.3: every `load(ctx, d)` becomes a named parameter, every `do(ctx, d, p)`
a direct call to the function the description resolves to, and every `load(result, d)` a key of that function's
returned dictionary when it is a literal. The module needs no embedder and no vector store; it also carries each
entry's template, so it can answer a question that matches a template without any model.
"""

import ast
import builtins
import json
import keyword
import re
import sys

ALLOWED_IMPORTS = {"math", "fractions", "itertools", "functools", "statistics", "decimal", "collections",
                   "operator", "re", "string", "datetime", "cmath", "numbers", "heapq", "bisect"}
FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "__import__", "input", "globals", "locals", "vars",
                   "getattr", "setattr", "delattr", "breakpoint", "exit", "quit", "memoryview"}
PRIMITIVES = {"load", "do", "narrate", "convert", "semantic"}


def _const_str(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return None
    return None


def _decorator(fn):
    """(description, root) of a function decorated with @semantic, else None."""
    for d in fn.decorator_list:
        if isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == "semantic":
            desc = _const_str(d.args[0]) if d.args else None
            root = False
            if len(d.args) > 1 and isinstance(d.args[1], ast.Constant):
                root = bool(d.args[1].value)
            for kw in d.keywords:
                if kw.arg == "root" and isinstance(kw.value, ast.Constant):
                    root = bool(kw.value.value)
                if kw.arg == "description":
                    desc = _const_str(kw.value)
            return desc, root
    return None


def _calls(fn, name):
    return [n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name]


def _returns(fn):
    keys = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Return) and isinstance(n.value, ast.Dict):
            for k in n.value.keys:
                s = _const_str(k) if k is not None else None
                if s is not None:
                    keys.add(s)
    return sorted(keys)


def problems_of(tree):
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    out.append(f"import of {a.name} is not allowed")
        elif isinstance(n, ast.ImportFrom):
            if (n.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                out.append(f"import from {n.module} is not allowed")
        elif isinstance(n, ast.Name) and n.id in FORBIDDEN_NAMES:
            out.append(f"{n.id} is not allowed")
        elif isinstance(n, ast.Attribute) and n.attr.startswith("__") and n.attr.endswith("__"):
            out.append(f"attribute {n.attr} is not allowed")
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "print":
            out.append("print() is not allowed; use narrate()")
    return sorted(set(out))


def describe(code):
    tree = ast.parse(code)
    fns = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            d = _decorator(node)
            if d is None:
                continue
            desc, root = d
            cx = node.args.args[0].arg if node.args.args else "ctx"
            loads = [s for c in _calls(node, "load") if len(c.args) > 1 and isinstance(c.args[0], ast.Name)
                     and c.args[0].id == cx for s in [_const_str(c.args[1])] if s]
            calls = [s for c in _calls(node, "do") if len(c.args) > 1 for s in [_const_str(c.args[1])] if s]
            fns.append({"name": node.name, "description": desc or "", "root": root, "loads": loads,
                        "calls": calls, "returns": _returns(node), "narrates": len(_calls(node, "narrate")),
                        "branches": sum(isinstance(n, (ast.If, ast.For, ast.While)) for n in ast.walk(node))})
    problems = problems_of(tree)
    roots = [f for f in fns if f["root"]]
    if not fns:
        problems.append("no function is decorated with @semantic")
    elif len(roots) != 1:
        problems.append(f"exactly one function must have root=True (found {len(roots)})")
    for f in fns:
        if not f["description"]:
            problems.append(f"{f['name']}: the @semantic description must be a plain string")
        if f["narrates"] == 0:
            problems.append(f"{f['name']}: no narrate() call")
    return {"ok": True, "functions": fns, "problems": problems}


# ---------------------------------------------------------------- compile


def _ident(text, taken):
    words = re.findall(r"[a-z0-9]+", str(text).lower())
    words = [w for w in words if w not in ("the", "a", "an", "of", "to", "for", "in", "on", "and", "with")]
    base = "_".join(words[:5]) or "value"
    if base[0].isdigit():
        base = "v_" + base
    if keyword.iskeyword(base) or hasattr(builtins, base):
        base += "_"
    name, i = base, 2
    while name in taken:
        name, i = f"{base}_{i}", i + 1
    taken.add(name)
    return name


class _Fn:
    def __init__(self, entry, node, desc, root):
        self.entry, self.node, self.desc, self.root = entry, node, desc, root
        self.ctx = node.args.args[0].arg if node.args.args else "ctx"
        self.loads = []  # (description, param name)
        self.returns = _returns(node)
        self.pyname = None
        self.orig = node.name


class _Rewrite(ast.NodeTransformer):
    def __init__(self, fn, resolve_fn):
        self.fn, self.resolve_fn = fn, resolve_fn
        self.taken = set()
        self.params = {}  # description -> name
        self.results = {}  # variable name -> callee _Fn

    def param(self, desc):
        if desc not in self.params:
            self.params[desc] = _ident(desc, self.taken)
        return self.params[desc]

    def visit_Assign(self, node):
        self.generic_visit(node)
        v = node.value
        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and isinstance(v, ast.Call)
                and isinstance(v.func, ast.Name) and getattr(v, "_arc_callee", None) is not None):
            self.results[node.targets[0].id] = v._arc_callee
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        if not isinstance(node.func, ast.Name):
            return node
        name = node.func.id
        d_type = node.args[2] if len(node.args) > 2 else next((k.value for k in node.keywords if k.arg == "d_type"), None)
        if name == "load" and len(node.args) >= 2:
            src, desc = node.args[0], _const_str(node.args[1])
            if desc is None:
                return node
            if isinstance(src, ast.Name) and src.id == self.fn.ctx:
                value = ast.Name(id=self.param(desc), ctx=ast.Load())
            elif isinstance(src, ast.Name) and src.id in self.results and self.results[src.id].returns:
                key, _s = best_match(desc, self.results[src.id].returns, floor=0.0)
                value = ast.Subscript(value=ast.Name(id=src.id, ctx=ast.Load()), slice=ast.Constant(key), ctx=ast.Load())
            else:
                value = ast.Call(func=ast.Name(id="pick", ctx=ast.Load()), args=[src, ast.Constant(desc)], keywords=[])
            args = [value] + ([d_type] if d_type is not None else [])
            return ast.copy_location(ast.Call(func=ast.Name(id="convert", ctx=ast.Load()), args=args, keywords=[]), node)
        if name == "do" and len(node.args) >= 2:
            desc = _const_str(node.args[1])
            callee = self.resolve_fn(desc) if desc else None
            if callee is None:
                raise ValueError(f"do() description {desc!r} matches no function")
            payload = node.args[2] if len(node.args) > 2 else next((k.value for k in node.keywords if k.arg in ("payloads", "params")), None)
            kws, bound = [], set()
            names = {d: n for d, n in callee.loads}
            if isinstance(payload, ast.Dict):
                for k, v in zip(payload.keys, payload.values):
                    ks = _const_str(k) if k is not None else None
                    if ks is None:
                        raise ValueError("a do() payload key is not a plain string")
                    if not names:
                        continue
                    target = _exact(ks, list(names)) or best_match(ks, list(names))[0]
                    if target is None or target in bound:
                        continue
                    bound.add(target)
                    kws.append(ast.keyword(arg=names[target], value=v))
            elif payload is not None:
                raise ValueError("a do() payload is not a dictionary literal")
            # At run time the callee reads what it was not given from the caller's scope (the context stack).
            for d, n in callee.loads:
                if d in bound:
                    continue
                mine = _exact(d, list(self.params))
                if mine is not None:
                    kws.append(ast.keyword(arg=n, value=ast.Name(id=self.params[mine], ctx=ast.Load())))
            call = ast.Call(func=ast.Name(id=callee.pyname, ctx=ast.Load()),
                            args=[ast.Name(id=self.fn.ctx, ctx=ast.Load())], keywords=kws)
            call._arc_callee = callee
            return ast.copy_location(call, node)
        return node


class _Rename(ast.NodeTransformer):
    """Renames an entry's top-level names (helpers, constants, semantic functions) everywhere in its code."""

    def __init__(self, names):
        self.names = names

    def visit_Name(self, node):
        if node.id in self.names:
            node.id = self.names[node.id]
        return node

    def visit_arg(self, node):
        if node.arg in self.names:
            node.arg = self.names[node.arg]
        return node


def _entry_parts(e, taken, header):
    """(functions, top-level statements) of one entry, every top-level name made unique to it."""
    tree = ast.parse(e["code"])
    bad = problems_of(tree)
    if bad:
        raise ValueError("; ".join(bad))
    names, mine, stmts = {}, [], []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            d = _decorator(node)
            if d is None:
                names[node.name] = _ident(f"{node.name} {e['id']}", taken)
                f = _Fn(e, node, None, False)
            else:
                f = _Fn(e, node, d[0], d[1])
                f.pyname = _ident(d[0], taken)
                names[node.name] = f.pyname
            f.orig = node.name
            mine.append(f)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name):
                        names[n.id] = _ident(f"{n.id} {e['id']}", taken)
            stmts.append(node)
    # Semantic functions are reached only through do(), so only their definitions are renamed; a local variable that
    # happens to share a function's name (seen: `total_work = do(ctx, "...total work...")`) must keep its own name.
    semantic = {f.orig for f in mine if f.desc}
    for f in mine:
        # A semantic function called directly (not through do()) shares its caller's context at run time; compiled, it
        # would lose its inputs. Such an entry is not compiled.
        for c in ast.walk(f.node):
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id in semantic:
                raise ValueError(f"{f.orig} calls a semantic function directly instead of through do()")
    rn = _Rename({k: v for k, v in names.items() if k not in semantic})
    for f in mine:
        rn.visit(f.node)
        f.node.name = names[f.orig]
    for st in stmts:
        rn.visit(st)
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            header.append(ast.unparse(node))
    return mine, stmts


def compile_cache(entries, runtime_src=""):
    import copy
    _originals = copy.deepcopy(entries)
    fns, by_entry, consts, header, taken, skipped = [], {}, {}, [], set(), {}
    kept = []
    for e in entries:
        try:
            mine, stmts = _entry_parts(e, taken, header)
        except (ValueError, SyntaxError) as err:
            skipped[e["id"]] = str(err)
            continue
        if not any(f.root for f in mine):
            skipped[e["id"]] = "no root function"
            continue
        kept.append(e)
        by_entry[e["id"]] = mine
        consts[e["id"]] = stmts
        fns.extend(mine)
    entries = kept
    if not entries:
        raise ValueError(f"no entry could be compiled: {skipped}")
    semantic_fns = [f for f in fns if f.desc]
    # Pass 1: the descriptions each function loads, so callers can bind payloads to parameter names.
    for f in semantic_fns:
        rw = _Rewrite(f, lambda d: None)
        for c in _calls(f.node, "load"):
            if len(c.args) > 1 and isinstance(c.args[0], ast.Name) and c.args[0].id == f.ctx:
                s = _const_str(c.args[1])
                if s:
                    rw.param(s)
        f.loads = list(rw.params.items())

    def resolver(entry):
        links = entry.get("links") or {}

        def resolve_fn(desc):
            target = links.get(desc)
            pool = semantic_fns
            if target:
                for f in pool:
                    if f"{f.entry['id']}:{f.orig}" == target or f.orig == target:
                        return f
            own = [f for f in pool if f.entry is entry]
            for group in (own, pool):
                hit, _s = best_match(desc, [f.desc for f in group])
                if hit is not None:
                    return next(f for f in group if f.desc == hit)
            return None

        return resolve_fn

    def calls_of(f, res):
        out = []
        for c in _calls(f.node, "do"):
            desc = _const_str(c.args[1]) if len(c.args) > 1 else None
            callee = res(desc) if desc else None
            if callee is None:
                continue
            payload = c.args[2] if len(c.args) > 2 else next((k.value for k in c.keywords if k.arg in ("payloads", "params")), None)
            keys = [_const_str(k) for k in payload.keys] if isinstance(payload, ast.Dict) else []
            out.append((callee, [k for k in keys if k]))
        return out

    edges = {}
    for e in entries:
        res = resolver(e)
        for f in by_entry[e["id"]]:
            if f.desc:
                edges[id(f)] = calls_of(f, res)
    changed = True
    while changed:
        changed = False
        for f in semantic_fns:
            have = [d for d, _n in f.loads]
            for callee, keys in edges.get(id(f), []):
                for d, _n in list(callee.loads):
                    if keys and best_match(d, keys)[0] is not None:
                        continue
                    if _exact(d, have) is not None:
                        continue
                    taken = {n for _d, n in f.loads}
                    f.loads.append((d, _ident(d, taken)))
                    have.append(d)
                    changed = True

    out = [
        '"""Compiled Nafs-Aql reasoning cache (standard library only; no model, embedder or vector store).',
        "",
        "    python3 arc_cache.py \"<question>\"            answer a question that matches a stored template",
        "    python3 arc_cache.py --entry <id> '<json>'      run one entry on parameters given by description",
        '"""',
        "import functools, itertools, json, math, re, sys",
        "from fractions import Fraction",
    ]
    out.extend(sorted(set(header)))
    out.append("")
    out.append("import ast")
    out.append(runtime_parts(runtime_src))
    out.append(RUNTIME_HELPERS)
    entries_table = []
    for e in entries:
        res = resolver(e)
        part = []
        try:
            part.extend(_emit(e, res, consts, by_entry))
        except ValueError as err:
            skipped[e["id"]] = str(err)
            continue
        out.extend(part)
        root = next(f for f in by_entry[e["id"]] if f.root)
        entries_table.append((e, root))
    if not entries_table:
        raise ValueError(f"no entry could be compiled: {skipped}")
    out.append("ENTRIES = {")
    for e, root in entries_table:
        binding = {}
        for p in e.get("params", []):
            d = p.get("description", "")
            target, _s = best_match(d, [x for x, _n in root.loads], floor=0.0) if root.loads else (None, 0)
            if target is not None:
                binding[d] = dict(root.loads)[target]
        out.append(f"    {e['id']!r}: {{'root': {root.pyname}, 'template': {e.get('template', '')!r}, "
                   f"'params': {json.dumps(e.get('params', []))}, 'binding': {binding!r}}},")
    out.append("}")
    out.append(f"SKIPPED = {skipped!r}")
    out.append(MODULE_TAIL)
    module = "\n".join(out) + "\n"
    ast.parse(module)
    # Each compiled entry must give its expected answer on the values of the question it was built from; one that
    # does not is left out and the module is built again without it.
    ns = {"__name__": "arc_compiled_check"}
    exec(compile(module, "arc_cache.py", "exec"), ns)
    wrong = {}
    for e, _root in entries_table:
        if e.get("expected") is None:
            continue
        try:
            got = ns["solve"](e["id"], {p["description"]: p["value"] for p in e.get("params", [])})["answer"]
            ok = _same(got, e["expected"])
        except Exception as err:  # noqa: BLE001
            got, ok = f"{type(err).__name__}: {err}", False
        if not ok:
            wrong[e["id"]] = f"the compiled form gives {got!r} on the original question, not {e['expected']!r}"
    if wrong and len(wrong) < len(entries_table):
        again = [e for e, _r in entries_table if e["id"] not in wrong]
        module, more = compile_cache([dict(x) for x in _originals if x["id"] in {e["id"] for e in again}], runtime_src)
        return module, dict(skipped, **wrong, **more)
    if wrong:
        raise ValueError(f"no entry could be compiled: {dict(skipped, **wrong)}")
    return module, skipped


def _same(got, want):
    try:
        a, b = float(Fraction(str(got).strip())), float(Fraction(str(want).strip()))
    except (ValueError, ZeroDivisionError):
        try:
            a, b = float(got), float(str(want).replace(",", ""))
        except (TypeError, ValueError):
            return str(got).strip().lower() == str(want).strip().lower()
    want_s = str(want)
    dec = len(want_s.split(".")[1]) if "." in want_s else 0
    return abs(a - b) <= max(1e-6 * max(1.0, abs(b)), 0.5 * 10 ** (-dec) if dec else 1e-9)


def _emit(e, res, consts, by_entry):
    """The source lines of one entry; ValueError when a part of it cannot be resolved statically."""
    out = []
    for st in consts[e["id"]]:
        out.append(ast.unparse(st))
    for f in by_entry[e["id"]]:
        node = f.node
        node.decorator_list = []
        if f.desc:
            rw = _Rewrite(f, res)
            rw.params = dict(f.loads)
            rw.taken = set(rw.params.values())
            node = rw.visit(node)
            ast.fix_missing_locations(node)
            node.name = f.pyname
            a = node.args
            a.args = a.args[:1] + [ast.arg(arg=n) for _d, n in f.loads]
            a.defaults = [ast.Constant(None) for _ in f.loads]
            node.body.insert(0, ast.Expr(ast.Constant(f.desc)))
        out.append(ast.unparse(node))
        out.append("")
    return out


RUNTIME_HELPERS = r'''
class _Ctx:
    def __init__(self):
        self.narration = []


def narrate(ctx, message):
    ctx.narration.append(str(message))


def pick(result, description):
    key, _s = best_match(description, list(result.keys()), floor=0.0)
    return result[key]
'''

RUNTIME_PARTS = ("CacheInvocationError", "MIN_MATCH", "_exact", "_STOP", "_words", "similarity", "best_match", "_NUM", "_number", "convert")


def runtime_parts(runtime_src):
    """The runtime's matching and conversion code, copied into the compiled module."""
    out = []
    for node in ast.parse(runtime_src).body:
        name = getattr(node, "name", None)
        if name is None and isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        if name in RUNTIME_PARTS:
            out.append(ast.unparse(node))
    return "\n\n".join(out)

MODULE_TAIL = r'''

def _template_regex(template, params):
    types = {p["name"]: p.get("type", "str") for p in params}
    parts, out, pos = re.split(r"\{(\w+)\}", template), "", 0
    for i, part in enumerate(parts):
        if i % 2 == 0:
            out += r"\s+".join(re.escape(w) for w in part.split()) if part.strip() else ""
            if part[:1].isspace() and out and not out.endswith(r"\s*"):
                pass
        else:
            num = types.get(part) in ("int", "float", "Fraction")
            out += r"\s*(" + (r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?" if num else r".+?") + r")\s*"
    return re.compile(r"^\s*" + out + r"\s*[.?!]?\s*$", re.I | re.S)


def match(question):
    q = " ".join(question.split())
    for eid, e in ENTRIES.items():
        m = _template_regex(e["template"], e["params"]).match(q)
        if m:
            names = re.findall(r"\{(\w+)\}", e["template"])
            values = dict(zip(names, m.groups()))
            return eid, {p["description"]: values.get(p["name"]) for p in e["params"]}
    return None, None


def solve(entry_id, params):
    e = ENTRIES[entry_id]
    ctx = _Ctx()
    kwargs = {e["binding"][d]: v for d, v in params.items() if d in e["binding"]}
    out = e["root"](ctx, **kwargs)
    if not isinstance(out, dict):
        out = {"answer": out}
    return {"answer": out.get("answer"), "outputs": out, "narration": ctx.narration}


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--entry":
        print(json.dumps(solve(sys.argv[2], json.loads(sys.argv[3])), default=str))
    elif len(sys.argv) >= 2:
        eid, params = match(" ".join(sys.argv[1:]))
        if eid is None:
            print(json.dumps({"hit": False}))
            sys.exit(1)
        print(json.dumps(dict(solve(eid, params), hit=True, entry=eid), default=str))
'''


def main_tools_req(req):
    try:
        if req.get("mode") == "describe":
            res = describe(req.get("code", ""))
        elif req.get("mode") == "compile":
            module, skipped = compile_cache(req.get("entries", []), req.get("runtime", ""))
            res = {"ok": True, "module": module, "skipped": skipped}
        else:
            res = {"ok": False, "error": "unknown mode"}
    except SyntaxError as e:
        res = {"ok": False, "error": f"SyntaxError: {e.msg} (line {e.lineno})"}
    except Exception as e:  # noqa: BLE001
        res = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    sys.stdout.write(json.dumps(res))
