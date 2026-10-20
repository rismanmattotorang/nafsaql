# Linux verifier sandbox, stage 3 (SR-002). Run as `python -I -B -c <this file> python -I -B entry` once the mount
# namespace is set up and the capabilities are dropped: it installs a seccomp-bpf filter, then replaces itself with
# the real interpreter (filters survive execve). The filter
#   - makes fork, vfork and clone without CLONE_THREAD fail with EPERM (threads stay allowed), and clone3 fail with
#     ENOSYS so libc falls back to clone, which is filtered on its flags;
#   - makes mount, umount2, pivot_root, chroot, unshare, setns and the new mount API fail with EPERM, so even a
#     nested user namespace cannot rearrange the filesystem view built by the shell stage;
#   - kills the process on any foreign syscall ABI (32-bit, x32).
# Any failure exits non-zero before the code under test starts (fail closed).
import ctypes
import os
import platform
import struct
import sys

machine = platform.machine()
if sys.byteorder != "little":
    sys.exit("seccomp: unsupported byte order")
if machine == "x86_64":
    ARCH = 0xC000003E
    DENY = [165, 166, 155, 161, 272, 308, 57, 58]  # mount umount2 pivot_root chroot unshare setns fork vfork
    CLONE = 56
elif machine in ("aarch64", "arm64"):
    ARCH = 0xC00000B7
    DENY = [40, 39, 41, 51, 97, 268]  # mount umount2 pivot_root chroot unshare setns (no fork/vfork on arm64)
    CLONE = 220
else:
    sys.exit("seccomp: unsupported architecture " + machine)
DENY += [428, 429, 430, 431, 432, 433, 442]  # open_tree move_mount fsopen fsconfig fsmount fspick mount_setattr
CLONE3 = 435
CLONE_THREAD = 0x10000
ALLOW, EPERM, ENOSYS, KILL = 0x7FFF0000, 0x00050001, 0x00050026, 0x80000000

# seccomp_data: nr at 0, arch at 4, args at 16 + 8 * i (little endian: the low word comes first).
program = [("ld", 4), ("jeq", ARCH, "native"), ("ret", KILL), ("lbl", "native"), ("ld", 0)]
if machine == "x86_64":
    program.append(("jset", 0x40000000, "kill"))  # x32 ABI
program += [("jeq", n, "eperm") for n in DENY]
program += [("jeq", CLONE, "clone"), ("jeq", CLONE3, "enosys"), ("ret", ALLOW)]
program += [("lbl", "clone"), ("ld", 16), ("jset", CLONE_THREAD, "allow"), ("ret", EPERM)]
program += [("lbl", "allow"), ("ret", ALLOW), ("lbl", "eperm"), ("ret", EPERM)]
program += [("lbl", "enosys"), ("ret", ENOSYS), ("lbl", "kill"), ("ret", KILL)]


def assemble(program):
    at, i = {}, 0
    for op in program:
        if op[0] == "lbl":
            at[op[1]] = i
        else:
            i += 1
    out, i = [], 0
    for op in program:
        if op[0] == "lbl":
            continue
        i += 1
        if op[0] == "ld":
            out.append((0x20, 0, op[1]))  # BPF_LD | BPF_W | BPF_ABS
        elif op[0] == "ret":
            out.append((0x06, 0, op[1]))  # BPF_RET | BPF_K
        else:
            jump = at[op[2]] - i
            if not 0 <= jump <= 255:
                sys.exit("seccomp: jump out of range")
            out.append((0x15 if op[0] == "jeq" else 0x45, jump, op[1]))  # BPF_JMP | JEQ/JSET | BPF_K
    return b"".join(struct.pack("<HBBI", code, jt, 0, k) for code, jt, k in out)


if __name__ == "__main__":
    blob = assemble(program)
    buf = ctypes.create_string_buffer(blob, len(blob))

    class SockFprog(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.c_void_p)]

    prog = SockFprog(len(blob) // 8, ctypes.cast(buf, ctypes.c_void_p))
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
        sys.exit("seccomp: no_new_privs failed")
    if libc.prctl(22, 2, ctypes.byref(prog), 0, 0) != 0:  # PR_SET_SECCOMP, SECCOMP_MODE_FILTER
        sys.exit("seccomp: filter rejected, errno %d" % ctypes.get_errno())
    os.execv(sys.argv[1], sys.argv[1:])
