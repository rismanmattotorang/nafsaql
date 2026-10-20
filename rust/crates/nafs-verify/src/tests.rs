use super::*;
use crate::extract::numbers;

fn hub() -> VerifierHub {
    VerifierHub::new(
        SandboxLimits {
            sympy_python: Some("python3.10".into()),
            ..Default::default()
        },
        1e-6,
    )
}
fn tests(t: &[&str]) -> VerificationInputs {
    VerificationInputs {
        tests: Some(t.iter().map(|s| s.to_string()).collect()),
        ..Default::default()
    }
}

#[test]
fn number_parsing() {
    assert_eq!(parse_number("$1,234.50"), Some(1234.5));
    assert_eq!(parse_number("3/4"), Some(0.75));
    assert_eq!(parse_number("\\frac{1}{2}"), Some(0.5));
    assert_eq!(parse_number("\u{2212}5"), Some(-5.0));
}
#[test]
fn hyphens_are_not_minus_signs_in_ranges_and_dates() {
    assert_eq!(numbers("pages 10-12"), vec!["10", "12"]);
    assert_eq!(numbers("on 2024-05-01"), vec!["2024", "05", "01"]);
    assert_eq!(numbers("it is -3 degrees"), vec!["-3"]);
}
#[test]
fn final_answer_extraction() {
    assert_eq!(
        extract_final_answer("so \\boxed{42}").as_deref(),
        Some("42")
    );
    assert_eq!(
        extract_final_answer("The answer is 1,200 dollars.").as_deref(),
        Some("1,200")
    );
    assert_eq!(
        extract_final_answer("Answer: 12 × 3 = 36").as_deref(),
        Some("36")
    );
    assert_eq!(
        extract_final_answer("\\boxed{\\frac{1}{2}}").as_deref(),
        Some("(1)/(2)")
    );
    assert_eq!(extract_final_answer("a 3 b 7").as_deref(), Some("7"));
}
#[test]
fn math_verifier() {
    let h = hub();
    assert!(
        h.verify_math("So the answer is \\boxed{145}.", "#### 145")
            .passed
    );
    assert!(h.verify_math("The trip takes 145 minutes.", "145").passed);
    assert!(h.verify_math("Answer: 12 × 3 = 36", "36").passed);
    assert!(h.verify_math("\\boxed{\\frac{1}{2}}", "0.5").passed);
    assert!(!h.verify_math("The trip takes 150 minutes.", "145").passed);
    assert!(
        !h.verify_math("Answer: 145\nWait, correcting:\nAnswer: 147", "145")
            .passed
    );
}
#[tokio::test]
async fn symbolic_math_when_sympy_available() {
    let h = hub();
    if h.limits
        .sympy_python
        .as_deref()
        .map(|p| !std::path::Path::new(p).exists())
        .unwrap_or(true)
    {
        return;
    }
    let r = h
        .verify_math_full("Answer: \\frac{3\\sqrt{2}}{2}", "3*sqrt(2)/2")
        .await;
    assert!(r.passed, "{r:?}");
    assert_eq!(r.verifier, "math_symbolic");
    assert!(
        !h.verify_math_full("Answer: 2\\sqrt{2}", "3*sqrt(2)/2")
            .await
            .passed
    );
}
#[test]
fn json_verifier() {
    let schema = serde_json::json!({"type":"object","required":["name","age"],"properties":{"name":{"type":"string"},"age":{"type":"integer","minimum":0}}});
    assert!(verify_json_schema("```json\n{\"name\":\"A\",\"age\":3}\n```", &schema).passed);
    assert!(verify_json_schema("See [1]: {\"name\":\"A\",\"age\":3}", &schema).passed);
    assert!(!verify_json_schema("{\"name\":\"A\",\"age\":-1}", &schema).passed);
    assert!(!verify_json_schema("no json here", &schema).passed);
    assert!(verify_json_schema("42", &serde_json::json!({"type":"integer"})).passed);
}
#[test]
fn code_extraction_prefers_the_target_function() {
    let ans = "```python\ndef add(a, b):\n    return a + b\n```\nTests:\n```python\ndef test_add():\n    assert add(1, 2) == 3\n```";
    assert!(extract_code(ans, Some("add"))
        .unwrap()
        .contains("return a + b"));
    assert!(extract_code(ans, None).unwrap().contains("return a + b"));
}
#[test]
fn test_code_needs_entry_point() {
    let v = VerificationInputs {
        test_code: Some("def check(c): assert c(1) == 2".into()),
        ..Default::default()
    };
    assert!(!v.is_verifiable());
    assert!(v.validation_error().is_some());
}
#[tokio::test]
async fn code_verifier_pass_fail_and_limits() {
    let h = hub();
    assert!(
        h.verify_code(
            "```python\ndef add(a,b):\n    return a+b\n```",
            &tests(&["assert add(2,3)==5"])
        )
        .await
        .passed
    );
    let bad = h
        .verify_code(
            "```python\ndef add(a,b):\n    return a-b\n```",
            &tests(&["assert add(2,3)==5"]),
        )
        .await;
    assert!(!bad.passed && bad.evidence.unwrap().contains("AssertionError"));
    let looping = h
        .verify_code(
            "```python\ndef add(a,b):\n    while True: pass\n```",
            &tests(&["assert add(1,2)==3"]),
        )
        .await;
    assert!(!looping.passed);
}
#[tokio::test]
async fn sandbox_denies_network_home_reads_and_forks() {
    let h = hub();
    let attempts = [
        "import socket\ns=socket.create_connection(('1.1.1.1',80),timeout=3)",
        "import os\nopen(os.path.expanduser('/Users')).read()",
        "import os\nos.fork()",
    ];
    for t in attempts {
        assert!(
            !h.verify_code("```python\nx=1\n```", &tests(&[t]))
                .await
                .passed,
            "{t}"
        );
    }
    // writes inside the sandbox directory are allowed (M3)
    assert!(h.verify_code("```python\ndef w():\n    open('scratch.txt','w').write('x')\n    return open('scratch.txt').read()\n```", &tests(&["assert w() == 'x'"])).await.passed);
}
#[tokio::test]
async fn output_flood_is_bounded_and_killed() {
    let h = hub();
    let r = h
        .verify_code(
            "```python\nwhile True:\n    print('x' * 100000)\n```",
            &tests(&["pass"]),
        )
        .await;
    assert!(!r.passed);
    let ev = r.evidence.unwrap();
    assert!(
        ev.len() < 5000,
        "evidence must stay bounded, got {} bytes",
        ev.len()
    );
    assert!(ev.contains("output limit"));
}
#[tokio::test]
async fn multibyte_output_truncation_does_not_panic() {
    let h = hub();
    let r = h
        .verify_code(
            "```python\nprint('é' * 3001)\nraise SystemExit(1)\n```",
            &tests(&["pass"]),
        )
        .await;
    assert!(!r.passed);
}
#[tokio::test]
async fn printed_sentinel_cannot_spoof_a_pass() {
    let h = hub();
    let r = h
        .verify_code(
            "```python\nimport os\nprint('NAFS_ALL_TESTS_PASSED')\nos._exit(0)\n```",
            &tests(&["assert False"]),
        )
        .await;
    assert!(!r.passed);
}
#[tokio::test]
async fn humaneval_prefix_supplies_imports() {
    let h = hub();
    let v = VerificationInputs { test_code: Some("def check(candidate):\n    assert candidate([1.0, 2.0]) == 3.0\n".into()), entry_point: Some("total".into()),
        code_prefix: Some("from typing import List\n\ndef total(xs: List[float]) -> float:\n    \"\"\"Sum.\"\"\"\n".into()), ..Default::default() };
    assert!(
        h.verify_code(
            "```python\ndef total(xs: List[float]) -> float:\n    return sum(xs)\n```",
            &v
        )
        .await
        .passed
    );
}
#[test]
fn clopper_pearson_bounds() {
    let (lo, hi) = mutation::clopper_pearson(0, 59);
    assert_eq!(lo, 0.0);
    assert!((hi - 0.0606).abs() < 0.002, "{hi}");
    let (lo, hi) = mutation::clopper_pearson(5, 10);
    assert!(
        (lo - 0.187).abs() < 0.003 && (hi - 0.813).abs() < 0.003,
        "{lo} {hi}"
    );
}
#[tokio::test]
async fn mutation_harness_measures_rates() {
    let h = hub();
    let cases = mutation::math_cases(&[("t1".into(), 145.0), ("t2".into(), 1200.0)]);
    let rep = mutation::run(&h, cases, 4, |_, _| {}).await;
    assert_eq!(rep["verifiers"]["math"]["accepted_wrong"], 0);
    assert_eq!(rep["verifiers"]["math"]["r"], 1.0);
}

/// Regression: EvalPlus tests put >100k-character lines with multi-byte text in the test code; Python
/// 3.10's file tokenizer can split a character across its buffer and reject valid code.
#[tokio::test]
async fn long_unicode_test_lines_are_accepted() {
    let h = hub();
    let line = format!(
        "_inputs = [{}]\n",
        (0..4000)
            .map(|i| format!("'こんにちは ສະບາຍດີ ă 😀 {i}'"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let v = VerificationInputs {
        test_code: Some(format!("{line}def check(candidate):\n    for s in _inputs:\n        assert candidate(s) == len(s)\n")),
        entry_point: Some("count".into()),
        ..Default::default()
    };
    let r = h
        .verify_code("```python\ndef count(s):\n    return len(s)\n```", &v)
        .await;
    assert!(r.passed, "{:?}", r.evidence);
}

/// Sandbox escape tests (SR-002, P5-T01): each attempt must fail closed, and the host must be unchanged. The
/// same attempts run on macOS (sandbox-exec profile) and Linux (mount namespace plus seccomp).
#[tokio::test]
async fn sandbox_escape_attempts_fail_closed() {
    let h = hub();
    // A user file in a directory the sandbox must hide: the temp dir (/tmp on Linux, /private/var/folders on macOS).
    let canary_dir = std::env::temp_dir().join(format!("nafs-canary-{}", std::process::id()));
    std::fs::create_dir_all(&canary_dir).unwrap();
    let canary = canary_dir.join("secret.txt");
    std::fs::write(&canary, "nafs-canary-contents").unwrap();
    let outside = std::env::temp_dir().join(format!("nafs-escape-{}", std::process::id()));
    let in_cwd = std::env::current_dir()
        .unwrap()
        .join(format!("nafs-escape-{}", std::process::id()));
    let _ = std::fs::remove_file(&outside);
    let _ = std::fs::remove_file(&in_cwd);
    // A process outside the sandbox that sandboxed code must not be able to signal (security review ST-19).
    // macOS denies the signal; on Linux the PID namespace hides the process, so the call raises.
    let mut sentinel = std::process::Command::new("sleep")
        .arg("120")
        .spawn()
        .expect("start the sentinel process");
    let sentinel_pid = sentinel.id();
    let mut attempts = vec![
        (
            "signal a process outside the sandbox (security review ST-19)",
            format!("import os, signal\nos.kill({sentinel_pid}, signal.SIGKILL)"),
        ),
        (
            "read a user file",
            format!("print(open({:?}).read()[:50])", canary.display().to_string()),
        ),
        (
            "write outside the sandbox",
            format!("open({:?}, 'w').write('x')", outside.display().to_string()),
        ),
        (
            "write into the gateway's working directory",
            format!("open({:?}, 'w').write('x')", in_cwd.display().to_string()),
        ),
        (
            "write to the filesystem root",
            "open('/nafs-escape', 'w').write('x')".to_string(),
        ),
        (
            "symlink to the user folder",
            format!(
                "import os\nos.symlink({:?}, 'l')\nprint(open('l').read()[:50])",
                canary.display().to_string()
            ),
        ),
        (
            "reach the hidden tree through the working directory",
            format!(
                "import os\nprint(open(os.path.join('..', {:?}, 'secret.txt')).read())",
                canary_dir.file_name().unwrap().to_string_lossy()
            ),
        ),
        (
            "read another process's environment through /proc",
            "import os\nme = os.getpid()\nassert any(open('/proc/%s/environ' % p, 'rb').read() for p in os.listdir('/proc') if p.isdigit() and int(p) != me)".to_string(),
        ),
        (
            "unmount the hiding mounts",
            "import ctypes\nl = ctypes.CDLL(None)\nassert l.umount2(b'/tmp', 0) == 0".to_string(),
        ),
        (
            "unshare a new user namespace",
            "import ctypes\nl = ctypes.CDLL(None)\nassert l.unshare(0x10000000) == 0".to_string(),
        ),
        (
            "spawn a process",
            "import subprocess\nsubprocess.run(['/bin/echo', 'hi'], check=True)".to_string(),
        ),
        (
            "fork bomb",
            "import os\nwhile True:\n    os.fork()".to_string(),
        ),
        (
            "memory bomb",
            "x = bytearray(3 * 1024 ** 3)\nx[-1] = 1".to_string(),
        ),
        ("CPU spin", "while True:\n    pass".to_string()),
        (
            "network",
            "import urllib.request\nurllib.request.urlopen('http://example.com', timeout=3).read()"
                .to_string(),
        ),
    ];
    // On Linux the interpreter is PID 1 of its namespace and getppid() is 0, so "the parent" is not addressable
    // (kill(0) only reaches the sandbox's own group); the sentinel above is the Linux check.
    if cfg!(target_os = "macos") {
        attempts.push((
            "signal the parent (security review ST-19)",
            "import os, signal\nos.kill(os.getppid(), signal.SIGCONT)".to_string(),
        ));
    }
    for (name, code) in &attempts {
        let r = h
            .verify_code("```python\nx = 1\n```", &tests(&[code.as_str()]))
            .await;
        assert!(!r.passed, "escape attempt succeeded: {name}");
    }
    // What a failing run prints is returned to the user and the model: the file's contents must not be in it.
    let leak = h
        .verify_code(
            "```python\nx = 1\n```",
            &tests(&[&format!(
                "import sys\nsys.stdout.write(open({:?}).read())\nsys.exit(1)",
                canary.display().to_string()
            )]),
        )
        .await;
    assert!(
        !format!("{:?}", leak.evidence).contains("nafs-canary-contents"),
        "file contents leaked into the verifier evidence"
    );
    let _ = std::fs::remove_dir_all(&canary_dir);
    assert!(!outside.exists(), "a file was written outside the sandbox");
    assert!(
        !in_cwd.exists(),
        "a file was written into the working directory"
    );
    let alive = sentinel.try_wait().expect("poll the sentinel").is_none();
    let _ = sentinel.kill();
    let _ = sentinel.wait();
    assert!(alive, "sandboxed code killed a process outside the sandbox");
}
#[tokio::test]
async fn sandbox_environment_carries_no_secrets() {
    std::env::set_var("NAFS_AUTH_TOKENS", "a-secret-token-that-must-not-leak");
    let h = hub();
    let r = h.verify_code("```python\nx = 1\n```", &tests(&["import os\nassert 'NAFS_AUTH_TOKENS' not in os.environ and not any('secret' in v for v in os.environ.values())"])).await;
    assert!(r.passed, "{:?}", r.evidence);
}
