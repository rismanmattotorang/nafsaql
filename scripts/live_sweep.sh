#!/usr/bin/env bash
# Every live suite against the running stack, one after the other (about 45 minutes with the 4B model on the reference
# machine): contract, security, Rooms, end to end, harness sweep, MVP acceptance. Logs go to logs/sweep_<name>.log and a
# summary line per suite is printed. Start the stack first (`make up`). Usage: scripts/live_sweep.sh [name ...]
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
declare -a SUITES=(
  "contract:python3 -I scripts/contract_test.py --live http://127.0.0.1:8080"
  "security:python3 -I scripts/security_tests.py"
  "rooms:python3 -I scripts/rooms_test.py"
  "e2e:python3 -I scripts/e2e_sweep.py"
  "harness:python3 -I scripts/harness_sweep.py"
  "acceptance:python3 -I scripts/mvp_acceptance.py"
)
failed=0
for entry in "${SUITES[@]}"; do
  name=${entry%%:*}; cmd=${entry#*:}
  if [ $# -gt 0 ] && ! printf '%s\n' "$@" | grep -qx "$name"; then continue; fi
  echo "=== $name start $(date +%T)"
  $cmd > "logs/sweep_$name.log" 2>&1; rc=$?
  [ $rc -ne 0 ] && failed=1
  echo "=== $name exit $rc $(date +%T)"; tail -n 2 "logs/sweep_$name.log"
done
echo ALLDONE
exit $failed
