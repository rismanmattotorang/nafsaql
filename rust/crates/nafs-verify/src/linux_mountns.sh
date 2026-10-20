# Linux verifier sandbox, stage 2 (SR-002). Runs as mapped root inside fresh user, network and mount namespaces, started by
# `unshare -r -n -m -- /bin/sh -c <this file> nafs-sbx STAGE3 SCRATCH [KEEP...] -- CMD...`. It runs in the host PID
# namespace on purpose: mount(8) reads /proc/self/mountinfo, which is only valid there.
#   1. user data directories (home, tmp, mnt, media, srv, secrets ...) are replaced by empty tmpfs mounts, with the
#      scratch directory and any KEEP directory (the interpreter prefix, when it lives under one of them) bound back in;
#   2. every mount in the namespace is remounted read-only, except the scratch directory;
#   3. cwd is re-entered through the new mounts (the inherited cwd would still reach the hidden tree through `..`);
#   4. it enters a new PID namespace and hands over to STAGE3 (linux_enter.sh), which replaces /proc and drops the
#      capabilities before running CMD.
# Any failure aborts before CMD runs (fail closed).
set -eu
set -f
IFS='
'
fail() {
    echo "sandbox setup failed: $*" >&2
    exit 125
}
# -n: never touch /run/mount/utab or /etc/mtab (owned by the real root, unwritable here: mount would report failure
# for a mount that worked).
mount() { command mount -n "$@"; }
umount() { command umount -n "$@"; }

stage3=$1
scratch=$2
shift 2
keeps=
while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do
    keeps="$keeps$1
"
    shift
done
[ "$#" -gt 0 ] || fail "missing command"
shift

# Directories replaced by an empty tmpfs. /tmp is listed because the scratch directory normally lives there.
hide='/home
/root
/mnt
/media
/srv
/tmp
/var/tmp
/var/mail
/run/user
/run/secrets'

under() { # under DIR PATH: PATH is DIR or inside it
    case $2 in "$1" | "$1"/*) return 0 ;; esac
    return 1
}
hidden() { # hidden PATH: PATH is inside a hidden directory (or is one)
    for h in $hide; do
        under "$h" "$1" && return 0
    done
    return 1
}

# 1a. stage what must survive the hiding in a directory that is itself about to be hidden
hold=
for h in /mnt /media /srv; do
    [ -d "$h" ] || continue
    inside=0
    for p in "$scratch" $keeps; do
        under "$h" "$p" && inside=1
    done
    if [ "$inside" = 0 ]; then
        hold=$h
        break
    fi
done
[ -n "$hold" ] || fail "no staging directory"
mount -t tmpfs -o nosuid,nodev,mode=0755,size=64k tmpfs "$hold" || fail "cannot stage"
n=0
for p in "$scratch" $keeps; do
    if hidden "$p"; then
        mkdir "$hold/$n" && mount --bind "$p" "$hold/$n" || fail "cannot stage $p"
    fi
    n=$((n + 1))
done

# 1b. hide user data
for h in $hide; do
    [ -d "$h" ] || continue
    [ "$h" = "$hold" ] && continue
    mode=0755
    case $h in /tmp | /var/tmp) mode=1777 ;; esac
    mount -t tmpfs -o nosuid,nodev,noexec,mode=$mode,size=1m tmpfs "$h" || fail "cannot hide $h"
done

# 1c. bind the staged directories back at their original paths, then drop the staging mounts
n=0
for p in "$scratch" $keeps; do
    if hidden "$p"; then
        mkdir -p "$p" && mount --bind "$hold/$n" "$p" || fail "cannot restore $p"
        umount "$hold/$n" || fail "cannot release staging mount"
    fi
    n=$((n + 1))
done
umount "$hold" || fail "cannot release staging directory"
mount -t tmpfs -o nosuid,nodev,noexec,mode=0755,size=1m tmpfs "$hold" || fail "cannot hide $hold"
# The scratch directory must be a mount of its own so that read-only on its parent does not reach it.
hidden "$scratch" || mount --bind "$scratch" "$scratch" || fail "cannot isolate scratch"

# 2. read-only everywhere except the scratch directory. Mounts keep their own nosuid/nodev/noexec/atime flags:
# the kernel refuses a remount that changes flags inherited from a more privileged namespace.
for h in $hide; do
    [ -d "$h" ] || continue
    mount -o remount,bind,ro,nosuid,nodev,noexec "$h" || fail "cannot make $h read-only"
done
covered() { # hidden directories, and /proc, which stage 3 replaces
    for h in $hide /proc; do
        under "$h" "$1" && return 0
    done
    return 1
}
remount_ro() { # remount_ro MOUNTPOINT OPTIONS(as printed in mountinfo)
    flags=
    for o in $(printf '%s' "$2" | tr ',' '\n'); do
        case $o in rw | ro | idmapped) ;; *) flags="$flags,$o" ;; esac
    done
    mount -o "remount,bind,ro$flags" "$1" || fail "cannot make $1 read-only"
}
mountinfo=$(cat /proc/self/mountinfo) || fail "cannot read mountinfo"
for line in $mountinfo; do
    IFS=' ' read -r _id _parent _dev _root mp opts _rest <<EOF
$line
EOF
    IFS='
'
    mp=$(printf '%b' "$mp")
    under "$scratch" "$mp" && continue
    if covered "$mp"; then
        # Restored interpreter directories sit inside a hidden one; everything else there is already handled.
        for p in $keeps; do
            [ "$mp" = "$p" ] && remount_ro "$mp" "$opts"
        done
        continue
    fi
    remount_ro "$mp" "$opts"
done

# 3. re-enter cwd through the new mounts; 4. new PID namespace, then stage 3
cd / && cd "$scratch" || fail "cannot enter scratch"
exec /usr/bin/unshare -p -f --kill-child -- /bin/sh -c "$stage3" nafs-sbx "$@"
