#!/usr/bin/env bash
# Verify the deployment recipes (OR-003/OR-004, P5-T02) and record the result in deploy/verified.json
# (read by gate G5). Checks run with the real validators where they exist on this host; checks whose tool
# is missing are reported as "skipped", never as passed. Usage: scripts/verify_deploy.sh [--live]
set -uo pipefail
cd "$(dirname "$0")/.."
python3 -I - "$@" <<'PY'
import json, os, re, shutil, subprocess, sys, time, urllib.request
checks = []
def add(name, status, detail):
    checks.append({"check": name, "status": status, "detail": detail[:300]}); print(f"{status.upper():8s} {name}: {detail[:120]}", flush=True)
def run(cmd, **kw):
    p = subprocess.run(cmd, capture_output=True, text=True, **kw); return p.returncode, (p.stdout + p.stderr).strip()
# Prometheus rules and config
if shutil.which("promtool"):
    rc, out = run(["promtool", "check", "rules", "deploy/monitoring/alerts.yml"]); add("prometheus alert rules", "pass" if rc == 0 else "fail", out.splitlines()[-1] if out else "")
    rc, out = run(["promtool", "check", "config", "prometheus.yml"], cwd="deploy/monitoring"); add("prometheus config", "pass" if rc == 0 else "fail", out.splitlines()[-1] if out else "")
else:
    add("prometheus rules and config", "skipped", "promtool not installed")
# Grafana dashboard
try:
    d = json.load(open("deploy/monitoring/grafana-dashboard.json")); add("grafana dashboard", "pass" if len(d.get("panels", [])) >= 5 else "fail", f"{len(d['panels'])} panels")
except Exception as e:
    add("grafana dashboard", "fail", str(e))
# docker compose (needs a token file to resolve secrets)
if shutil.which("docker"):
    tok = "deploy/docker/secrets/nafs_tokens"; created = not os.path.exists(tok)
    if created: open(tok, "w").write("verification-placeholder-token\n")
    rc, out = run(["docker", "compose", "-f", "deploy/docker/docker-compose.yml", "config", "-q"])
    if created: os.remove(tok)
    add("docker compose file", "pass" if rc == 0 else "fail", out or "valid")
else:
    add("docker compose file", "skipped", "docker not installed")
# launchd plists (reference machine)
if shutil.which("plutil"):
    rc, out = run(["plutil", "-lint", *sorted(__import__("glob").glob("deploy/launchd/*.plist"))]); add("launchd service files", "pass" if rc == 0 else "fail", out.replace("\n", "; "))
else:
    add("launchd service files", "skipped", "plutil not available")
# systemd units: structural check (systemd-analyze verify where available)
units = sorted(__import__("glob").glob("deploy/systemd/*.service"))
if shutil.which("systemd-analyze"):
    rc, out = run(["systemd-analyze", "verify", *units]); add("systemd units", "pass" if rc == 0 else "fail", out or "valid")
else:
    bad = [u for u in units if not all(re.search(rf"^\[{s}\]", open(u).read(), re.M) for s in ("Unit", "Service", "Install")) or "ExecStart=" not in open(u).read()]
    add("systemd units (structure; systemd-analyze unavailable on this host)", "pass" if not bad else "fail", f"{len(units)} units" + (f", bad: {bad}" if bad else ""))
# TLS proxy configs
cf = open("deploy/tls/Caddyfile").read(); ng = open("deploy/tls/nginx.conf").read()
if shutil.which("caddy"):
    rc, out = run(["caddy", "validate", "--config", "deploy/tls/Caddyfile", "--adapter", "caddyfile"]); add("Caddyfile", "pass" if rc == 0 else "fail", out)
else:
    add("Caddyfile (structure; caddy not installed)", "pass" if "reverse_proxy gateway:8080" in cf and "flush_interval -1" in cf else "fail", "reverse proxy with streaming flush")
add("nginx TLS config (structure; nginx not installed)", "pass" if "ssl_certificate" in ng and "proxy_buffering off" in ng and "Upgrade" in ng else "fail", "TLS, SSE and WebSocket pass-through")
# gateway configuration check (secrets, bind policy, sandbox isolation)
for cfgf in ("config/nafs-aql.toml", "config/nafs-aql.docker.toml"):
    env = {**os.environ}
    if "docker" in cfgf: env.update({"NAFS_BIND": "0.0.0.0:8080", "NAFS_AUTH_TOKENS": "verification-placeholder-token"})
    rc, out = run(["rust/target/release/nafs-gateway", "--config", cfgf, "--check"], env=env)
    add(f"gateway --check {cfgf}", "pass" if rc == 0 else "fail", " ".join(out.split())[:200])
# one-command start on the reference machine (live)
if "--live" in sys.argv:
    try:
        h = json.load(urllib.request.urlopen("http://127.0.0.1:8080/healthz", timeout=10))
        add("reference machine stack (make up) healthy", "pass" if h.get("status") == "ok" else "fail", json.dumps(h))
    except Exception as e:
        add("reference machine stack (make up) healthy", "fail", str(e))
    add("Linux vLLM-CPU compose stack started on a clean host", "skipped", "needs a Linux AVX-512 host; this machine is macOS (A-038)")
fails = [c for c in checks if c["status"] == "fail"]
out = {"verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "checks": checks, "all_pass": not fails,
       "summary": f"{sum(c['status'] == 'pass' for c in checks)} passed, {len(fails)} failed, {sum(c['status'] == 'skipped' for c in checks)} skipped (tool or host not available)"}
json.dump(out, open("deploy/verified.json", "w"), indent=2); print(out["summary"]); sys.exit(1 if fails else 0)
PY
