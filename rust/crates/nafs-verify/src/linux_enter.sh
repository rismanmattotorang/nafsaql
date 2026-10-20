# Linux verifier sandbox, stage 3 (SR-002). Runs as PID 1 of a fresh PID namespace, still mapped root with all
# capabilities, inside the mount namespace stage 2 prepared: `/bin/sh -c <this file> nafs-sbx CMD...`.
#   - /proc is replaced. The host /proc would expose the gateway's environment and, through /proc/PID/root, every file
#     the gateway can see. A fresh procfs shows only this namespace; where that is refused (container runtimes mask
#     parts of /proc) an empty read-only tmpfs takes its place.
#   - setpriv empties the bounding and inheritable capability sets and sets no_new_privs, so CMD (which keeps the
#     mapped uid but gets no capabilities on exec) cannot unmount, remount or re-mount anything.
set -eu
fail() {
    echo "sandbox setup failed: $*" >&2
    exit 125
}
mount() { command mount -n "$@"; }
if mount -t proc -o nosuid,nodev,noexec proc /proc 2>/dev/null; then
    mount -o remount,bind,ro,nosuid,nodev,noexec /proc || fail "cannot make /proc read-only"
else
    mount -t tmpfs -o ro,nosuid,nodev,noexec,mode=0555,size=4k tmpfs /proc || fail "cannot replace /proc"
fi
exec /usr/bin/setpriv --no-new-privs --inh-caps=-all --bounding-set=-all "$@"
