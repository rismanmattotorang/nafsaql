//! The Lean verifier (FR-090): Open Math Harness's strict protocol (`nafs_harness::lean`) with Lean run in
//! this crate's OS sandbox. Lean is trusted; its input is not, so the textual rules run first and the
//! process gets no network, no fork, no writes outside its scratch directory and no reads of user data
//! beyond the toolchain directory.
use crate::sandbox::{self, SandboxLimits};
use nafs_harness::lean::{self as protocol, RunError};
use nafs_types::{LeanTask, VerifierResult};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct LeanVerifier {
    /// The `lean` executable (an absolute path; the sandbox clears PATH).
    pub exe: PathBuf,
    /// Directories Lean must read: its toolchain, and a Mathlib build when one is used.
    pub read_roots: Vec<PathBuf>,
    /// `LEAN_PATH` for a Mathlib build (empty: core Lean only).
    pub lean_path: String,
    pub limits: SandboxLimits,
    /// Pristine elaborated type per frozen statement, computed once (OMH `pristine_type`).
    pristine: Arc<RwLock<HashMap<String, String>>>,
}

impl LeanVerifier {
    /// A verifier for `exe`, or None when it is not an executable file. The toolchain root (two levels above
    /// `bin/lean`, symlinks resolved) is readable inside the sandbox.
    pub fn new(
        exe: &str,
        lean_path: &str,
        timeout_s: u64,
        memory_mb: u64,
        deny_network: bool,
    ) -> Option<Self> {
        let exe = std::fs::canonicalize(exe).ok()?;
        if !exe.is_file() {
            return None;
        }
        let mut read_roots: Vec<PathBuf> = exe
            .parent()
            .and_then(|b| b.parent())
            .map(|p| vec![p.to_path_buf()])
            .unwrap_or_default();
        for p in lean_path.split(':').filter(|p| !p.is_empty()) {
            read_roots.push(PathBuf::from(p));
        }
        Some(LeanVerifier {
            exe,
            read_roots,
            lean_path: lean_path.to_string(),
            limits: SandboxLimits {
                cpu_time_s: timeout_s,
                wall_clock_s: timeout_s,
                max_output_bytes: 64 * 1024,
                hard_output_bytes: 4 << 20,
                memory_mb,
                deny_network,
                ..SandboxLimits::default()
            },
            pristine: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Run Lean on one probe file inside the sandbox (blocking; called from `spawn_blocking`).
    fn run_blocking(&self, probe: &str) -> Result<String, RunError> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| RunError::NoVerdict(format!("runtime: {e}")))?;
        let env: Vec<(String, String)> = if self.lean_path.is_empty() {
            vec![]
        } else {
            vec![("LEAN_PATH".into(), self.lean_path.clone())]
        };
        let out = rt
            .block_on(sandbox::run_program(
                &[("Probe.lean", probe)],
                &self.exe,
                &["{dir}/Probe.lean".to_string()],
                &self.read_roots,
                &env,
                None,
                &self.limits,
            ))
            .map_err(RunError::NoVerdict)?;
        let combined = format!("{}{}", out.stdout, out.stderr);
        match (&out.killed, out.exit_ok) {
            (Some(k), _) => Err(RunError::NoVerdict(format!("{k}\n{combined}"))),
            (None, true) => Ok(combined),
            (None, false) => Err(RunError::Refused(combined)),
        }
    }

    /// Grade an answer. The candidate is the answer's last ```lean block (the whole answer when there is
    /// none). Never a pass on a run without a clean report.
    pub async fn verify(&self, answer: &str, task: &LeanTask) -> VerifierResult {
        let t = Instant::now();
        let this = self.clone();
        let task = task.clone();
        let answer = answer.to_string();
        let graded = tokio::task::spawn_blocking(move || {
            if let Some(why) = protocol::frozen_objection(&task.statement) {
                return Err(why);
            }
            let thm = task
                .theorem
                .clone()
                .or_else(|| protocol::theorem_name(&task.statement))
                .ok_or("the theorem has no name")?;
            let cached = this.pristine.read().unwrap().get(&task.statement).cloned();
            let pristine = match cached {
                Some(p) => p,
                None => {
                    let p = protocol::pristine_type(&task.statement, &thm, &|probe| {
                        this.run_blocking(probe)
                    })
                    .map_err(|e| {
                        format!(
                            "the Lean statement does not elaborate: {}",
                            e.text().chars().take(600).collect::<String>()
                        )
                    })?;
                    this.pristine
                        .write()
                        .unwrap()
                        .insert(task.statement.clone(), p.clone());
                    p
                }
            };
            let candidate =
                protocol::last_fenced_block(&answer).unwrap_or_else(|| answer.trim().to_string());
            Ok(protocol::grade(
                &task.statement,
                &pristine,
                &candidate,
                &thm,
                &|probe| this.run_blocking(probe),
            ))
        })
        .await
        .unwrap_or_else(|e| Err(format!("Lean verifier failed: {e}")));
        match graded {
            Ok(g) => VerifierResult {
                verifier: "lean_proof".into(),
                passed: g.verified,
                score: Some(g.score as f32),
                evidence: (!g.verified).then(|| g.feedback.clone()),
                rates: None,
                duration_ms: t.elapsed().as_millis() as u64,
                exact: true,
                details: Some(serde_json::json!({
                    "verdict": format!("{:?}", g.verdict).to_lowercase(),
                    "kind": g.kind,
                    "error_classes": g.error_classes,
                    "score": g.score,
                })),
            },
            Err(why) => VerifierResult {
                verifier: "lean_proof".into(),
                passed: false,
                score: Some(0.0),
                evidence: Some(why),
                rates: None,
                duration_ms: t.elapsed().as_millis() as u64,
                exact: true,
                details: None,
            },
        }
    }
}
