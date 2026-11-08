#!/usr/bin/env bash
# Runs the knowledge security sweep (Phase 8) on its own throw-away gateway, then stops that gateway.
# Needs the main stack running (engines and SurrealDB). Extra arguments go to kb_security_sweep.py, e.g. --chat.
set -uo pipefail
cd "$(dirname "$0")/.."
scripts/kb_security_gateway.sh start || exit 1
python3 -I scripts/kb_security_sweep.py "$@"
rc=$?
scripts/kb_security_gateway.sh stop
exit $rc
