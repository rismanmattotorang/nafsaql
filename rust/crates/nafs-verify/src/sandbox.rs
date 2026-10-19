//! Process sandbox for model-written code (FR-035, SR-002).
//! - macOS: `sandbox-exec` profile denying network, forking, writes outside the sandbox directory and
//!   reads of user data (/Users, /Volumes, other temp dirs); Linux: `unshare -rn` (no network).
//! - Fails closed: if the isolation tool is missing and network denial is required, nothing runs.
//! - Own process group (setsid), killed as a whole on exit, timeout, memory or output overrun.
//! - CPU and file-size rlimits; memory enforced by RSS polling (macOS ignores RLIMIT_AS).
//! - Output is streamed into bounded tail buffers; never more than a few KB is kept in memory.
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[derive(Debug, Clone)]
pub struct SandboxLimits {
    pub cpu_time_s: u64,
    pub wall_clock_s: u64,
    pub max_output_bytes: usize,
    /// Total bytes a run may write to stdout+stderr before it is killed.
    pub hard_output_bytes: usize,
    pub memory_mb: u64,
    pub python: String,
    /// Interpreter with SymPy, for the symbolic math verifier (None disables it).
    pub sympy_python: Option<String>,
    pub deny_network: bool,
}
impl Default for SandboxLimits {
    fn default() -> Self {
        Self {
            cpu_time_s: 10,
            wall_clock_s: 30,
            max_output_bytes: 2048,
            hard_output_bytes: 1 << 20,
            memory_mb: 512,
            python: "python3".into(),
            sympy_python: None,
            deny_network: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SandboxOutput {
    pub exit_ok: bool,
    pub stdout: String,
    pub stderr: String,
    /// Why the run was stopped by the sandbox, if it was.
    pub killed: Option<String>,
}
impl SandboxOutput {
    /// Combined, human-readable tail for verifier evidence.
    pub fn tail(&self, max: usize) -> String {
        let mut t = self.stdout.clone();
        if !self.stderr.trim().is_empty() {
            if !t.is_empty() {
                t.push('\n');
            }
            t.push_str(&self.stderr);
        }
        if let Some(k) = &self.killed {
            t.push_str(&format!("\n[sandbox] {k}"));
        }
        tail_chars(t.trim(), max)
    }
}

/// Last `max` bytes of `s`, cut on a char boundary (never panics on multi-byte text).
pub fn tail_chars(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut i = s.len() - max;
    while !s.is_char_boundary(i) {
        i += 1;
    }
    format!("…{}", &s[i..])
}

struct Tail {
    buf: Vec<u8>,
    total: usize,
    keep: usize,
}
impl Tail {
    fn push(&mut self, data: &[u8]) {
        self.total += data.len();
        self.buf.extend_from_slice(data);
        if self.buf.len() > self.keep * 2 {
            let cut = self.buf.len() - self.keep;
            self.buf.drain(..cut);
        }
    }
    fn text(&self) -> String {
        let start = self.buf.len().saturating_sub(self.keep);
        String::from_utf8_lossy(&self.buf[start..]).to_string()
    }
}

/// Resolve an interpreter name to an absolute path, preferring real installs over the macOS
/// `/usr/bin/python3` developer-tools shim (which writes caches outside the sandbox).
pub fn resolve_python(name: &str) -> String {
    if name.contains('/') {
        return name.to_string();
    }
    let path = std::env::var("PATH").unwrap_or_default();
    let mut found: Vec<String> = path
        .split(':')
        .map(|d| format!("{d}/{name}"))
        .filter(|p| Path::new(p).is_file())
        .collect();
    for extra in ["/usr/local/bin", "/opt/homebrew/bin"] {
        let p = format!("{extra}/{name}");
        if Path::new(&p).is_file() && !found.contains(&p) {
            found.push(p);
        }
    }
    found.sort_by_key(|p| p.starts_with("/usr/bin/") as u8);
    found.into_iter().next().unwrap_or_else(|| name.to_string())
}

/// Whether untrusted code can be isolated on this host. On Linux this is a functional probe, not a file
/// check: the sandbox needs unprivileged user, network and PID namespaces, which stock Ubuntu 24.04 blocks
/// through AppArmor (`kernel.apparmor_restrict_unprivileged_userns=1`). A host with the `unshare` binary but
/// without that capability must report "not available", or the readiness check would claim isolation that fails
/// at the first request. The probe runs once per process.
pub fn isolation_available() -> bool {
    if cfg!(target_os = "macos") {
        Path::new("/usr/bin/sandbox-exec").exists()
    } else if cfg!(target_os = "linux") {
        static PROBE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *PROBE.get_or_init(|| {
            Path::new("/usr/bin/unshare").exists()
                && std::process::Command::new("/usr/bin/unshare")
                    .args(["-r", "-n", "-p", "-f", "--kill-child", "true"])
                    .stdin(std::process::Stdio::null())
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .status()
                    .map(|s| s.success())
                    .unwrap_or(false)
        })
    } else {
        false
    }
}

fn command_for(
    limits: &SandboxLimits,
    python: &str,
    dir: &Path,
    entry: &Path,
) -> Result<tokio::process::Command, String> {
    if limits.deny_network && !isolation_available() {
        return Err("sandbox isolation is not available on this host; refusing to run untrusted code (fail closed). On Linux it needs unprivileged user namespaces: on Ubuntu 24.04 run `sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0`".into());
    }
    #[cfg(target_os = "macos")]
    if limits.deny_network {
        let d = dir.display();
        let profile = format!(
            "(version 1)(allow default)(deny network*)(deny process-fork)(deny signal)(allow signal (target self))\
             (deny file-write* (require-not (subpath \"{d}\")))\
             (deny file-read* (subpath \"/Users\") (subpath \"/Volumes\") (subpath \"/private/var/folders\") (subpath \"/private/tmp\"))\
             (allow file-read* (subpath \"{d}\"))"
        );
        let mut c = tokio::process::Command::new("/usr/bin/sandbox-exec");
        c.arg("-p")
            .arg(profile)
            .arg(python)
            .arg("-I")
            .arg("-B")
            .arg(entry);
        return Ok(c);
    }
    #[cfg(target_os = "linux")]
    if limits.deny_network {
        let mut c = tokio::process::Command::new("/usr/bin/unshare");
        // New user, network and PID namespaces: no network, and no other process is addressable by
        // signal (security review ST-19); --kill-child reaps the whole tree with the namespace.
        c.arg("-r")
            .arg("-n")
            .arg("-p")
            .arg("-f")
            .arg("--kill-child")
            .arg(python)
            .arg("-I")
            .arg("-B")
            .arg(entry);
        return Ok(c);
    }
    let _ = dir;
    let mut c = tokio::process::Command::new(python);
    c.arg("-I").arg("-B").arg(entry);
    Ok(c)
}

#[cfg(target_os = "macos")]
fn rss_bytes(pid: i32) -> Option<u64> {
    let mut info: libc::proc_taskinfo = unsafe { std::mem::zeroed() };
    let size = std::mem::size_of::<libc::proc_taskinfo>() as i32;
    let n = unsafe {
        libc::proc_pidinfo(
            pid,
            libc::PROC_PIDTASKINFO,
            0,
            &mut info as *mut _ as *mut libc::c_void,
            size,
        )
    };
    if n == size {
        Some(info.pti_resident_size)
    } else {
        None
    }
}
#[cfg(target_os = "linux")]
fn rss_bytes(pid: i32) -> Option<u64> {
    let s = std::fs::read_to_string(format!("/proc/{pid}/statm")).ok()?;
    let pages: u64 = s.split_whitespace().nth(1)?.parse().ok()?;
    Some(pages * 4096)
}
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn rss_bytes(_pid: i32) -> Option<u64> {
    None
}

fn kill_group(pid: i32) {
    unsafe {
        libc::kill(-pid, libc::SIGKILL);
        libc::kill(pid, libc::SIGKILL);
    }
}

/// Run `entry` (one of `files`) with `python` inside the sandbox. `stdin` is written then closed.
pub async fn run(
    files: &[(&str, &str)],
    entry: &str,
    stdin: Option<&str>,
    python: &str,
    limits: &SandboxLimits,
) -> Result<SandboxOutput, String> {
    let raw = std::env::temp_dir().join(format!("nafs-sbx-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&raw).map_err(|e| format!("sandbox setup failed: {e}"))?;
    // Seatbelt matches resolved paths: /var/folders is a symlink to /private/var/folders.
    let dir: PathBuf = std::fs::canonicalize(&raw).unwrap_or(raw.clone());
    let result = run_in(&dir, files, entry, stdin, python, limits).await;
    let _ = std::fs::remove_dir_all(&dir);
    result
}

async fn run_in(
    dir: &Path,
    files: &[(&str, &str)],
    entry: &str,
    stdin: Option<&str>,
    python: &str,
    limits: &SandboxLimits,
) -> Result<SandboxOutput, String> {
    for (name, content) in files {
        if name.contains('/') || name.contains("..") {
            return Err("invalid sandbox file name".into());
        }
        std::fs::write(dir.join(name), content)
            .map_err(|e| format!("sandbox setup failed: {e}"))?;
    }
    let mut cmd = command_for(limits, python, dir, &dir.join(entry))?;
    cmd.current_dir(dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    cmd.env_clear()
        .env("PATH", "/usr/local/bin:/usr/bin:/bin")
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .env("HOME", dir)
        .env("TMPDIR", dir)
        .env("LANG", "C.UTF-8");
    let (cpu, mem) = (limits.cpu_time_s, limits.memory_mb);
    unsafe {
        cmd.pre_exec(move || {
            libc::setsid(); // own process group so the whole tree can be killed
            let lim = libc::rlimit {
                rlim_cur: cpu as libc::rlim_t,
                rlim_max: cpu as libc::rlim_t,
            };
            libc::setrlimit(libc::RLIMIT_CPU, &lim);
            let fs = libc::rlimit {
                rlim_cur: 16 << 20,
                rlim_max: 16 << 20,
            };
            libc::setrlimit(libc::RLIMIT_FSIZE, &fs);
            #[cfg(target_os = "linux")]
            {
                let m = (mem << 20) as libc::rlim_t;
                let r = libc::rlimit {
                    rlim_cur: m,
                    rlim_max: m,
                };
                libc::setrlimit(libc::RLIMIT_AS, &r);
            }
            let _ = mem;
            Ok(())
        });
    }
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("could not start python: {e}"))?;
    let pid = child.id().map(|p| p as i32).unwrap_or(0);
    if let Some(mut si) = child.stdin.take() {
        if let Some(text) = stdin {
            let _ = si.write_all(text.as_bytes()).await;
        }
        drop(si);
    }
    let killed: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let over = Arc::new(AtomicBool::new(false));
    let keep = limits.max_output_bytes.max(256);
    let out_tail = Arc::new(Mutex::new(Tail {
        buf: vec![],
        total: 0,
        keep,
    }));
    let err_tail = Arc::new(Mutex::new(Tail {
        buf: vec![],
        total: 0,
        keep,
    }));
    let hard = limits.hard_output_bytes;
    let reader = |mut r: Box<dyn tokio::io::AsyncRead + Unpin + Send>,
                  tail: Arc<Mutex<Tail>>,
                  other: Arc<Mutex<Tail>>,
                  over: Arc<AtomicBool>| async move {
        let mut buf = vec![0u8; 8192];
        loop {
            match r.read(&mut buf).await {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    let total = {
                        let mut t = tail.lock().unwrap();
                        t.push(&buf[..n]);
                        t.total
                    } + other.lock().unwrap().total;
                    if total > hard {
                        over.store(true, Ordering::SeqCst);
                    }
                }
            }
        }
    };
    let so = child.stdout.take().map(|s| {
        tokio::spawn(reader(
            Box::new(s),
            out_tail.clone(),
            err_tail.clone(),
            over.clone(),
        ))
    });
    let se = child.stderr.take().map(|s| {
        tokio::spawn(reader(
            Box::new(s),
            err_tail.clone(),
            out_tail.clone(),
            over.clone(),
        ))
    });
    let deadline = tokio::time::Instant::now() + Duration::from_secs(limits.wall_clock_s);
    let mem_limit = limits.memory_mb << 20;
    let status = loop {
        tokio::select! {
            st = child.wait() => break st.ok(),
            _ = tokio::time::sleep(Duration::from_millis(100)) => {
                let reason = if tokio::time::Instant::now() >= deadline { Some(format!("wall-clock limit of {}s exceeded", limits.wall_clock_s)) }
                    else if over.load(Ordering::SeqCst) { Some(format!("output limit of {} bytes exceeded", hard)) }
                    else if pid > 0 && rss_bytes(pid).map(|b| b > mem_limit).unwrap_or(false) { Some(format!("memory limit of {} MB exceeded", limits.memory_mb)) }
                    else { None };
                if let Some(r) = reason { *killed.lock().unwrap() = Some(r); if pid > 0 { kill_group(pid); } let _ = child.kill().await; break None; }
            }
        }
    };
    if pid > 0 {
        kill_group(pid);
    } // reap any grandchildren left in the group
    for h in [so, se].into_iter().flatten() {
        let _ = tokio::time::timeout(Duration::from_secs(2), h).await;
    }
    let killed = killed.lock().unwrap().clone();
    let stdout = out_tail.lock().unwrap().text();
    let stderr = err_tail.lock().unwrap().text();
    Ok(SandboxOutput {
        exit_ok: status.map(|s| s.success()).unwrap_or(false) && killed.is_none(),
        stdout,
        stderr,
        killed,
    })
}
