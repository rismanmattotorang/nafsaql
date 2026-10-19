//! Transport and secrets (SR-006, P5-T01): tokens from the environment or a secret file, and a
//! secure-by-default bind policy. TLS is terminated by a reverse proxy in front of the gateway
//! (deploy/tls/); the gateway itself refuses to listen on a non-loopback address without tokens.

/// Bearer tokens from `<ENV>` (comma-separated) or from the file named by `<ENV>_FILE` (one per line;
/// the docker/systemd secret-store convention). The file wins when both are set.
pub fn load_tokens(env: &str) -> Result<Vec<String>, String> {
    let raw = match std::env::var(format!("{env}_FILE")) {
        Ok(path) if !path.trim().is_empty() => {
            std::fs::read_to_string(path.trim()).map_err(|e| format!("{env}_FILE: {e}"))?
        }
        _ => std::env::var(env).unwrap_or_default(),
    };
    let toks: Vec<String> = raw
        .split([',', '\n'])
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty() && !s.starts_with('#'))
        .collect();
    if let Some(t) = toks.iter().find(|t| t.len() < 16) {
        return Err(format!(
            "auth token of length {} is too short (minimum 16 characters)",
            t.len()
        ));
    }
    Ok(toks)
}

pub fn is_loopback(bind: &str) -> bool {
    let host = bind
        .rsplit_once(':')
        .map(|(h, _)| h)
        .unwrap_or(bind)
        .trim_matches(['[', ']']);
    host == "localhost" || host == "::1" || host.starts_with("127.")
}

/// Refuse an unauthenticated gateway reachable from the network, unless explicitly overridden.
pub fn check_bind(bind: &str, tokens: &[String], allow_insecure: bool) -> Result<(), String> {
    if tokens.is_empty() && !is_loopback(bind) && !allow_insecure {
        return Err(format!(
            "refusing to listen on {bind} without authentication: set NAFS_AUTH_TOKENS (or NAFS_AUTH_TOKENS_FILE), \
             bind to 127.0.0.1 behind a TLS proxy, or set NAFS_ALLOW_INSECURE=1 for an isolated lab network (SR-006)"
        ));
    }
    Ok(())
}

/// Redact secrets from text before it is logged (secret-leak scan, ST-*).
pub fn redact_secrets(text: &str, tokens: &[String]) -> String {
    let mut out = text.to_string();
    for t in tokens {
        if !t.is_empty() {
            out = out.replace(t.as_str(), "[REDACTED]");
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bind_policy() {
        assert!(check_bind("127.0.0.1:8080", &[], false).is_ok());
        assert!(check_bind("[::1]:8080", &[], false).is_ok());
        assert!(check_bind("0.0.0.0:8080", &[], false).is_err());
        assert!(check_bind("0.0.0.0:8080", &["a-long-enough-token".into()], false).is_ok());
        assert!(check_bind("0.0.0.0:8080", &[], true).is_ok());
    }
    #[test]
    fn tokens_from_file_and_minimum_length() {
        let d = std::env::temp_dir().join(format!("nafs-tok-{}", std::process::id()));
        std::fs::write(&d, "# comment\nabcdefghijklmnopqrst\n").unwrap();
        std::env::set_var("NAFS_TEST_TOK_FILE", &d);
        assert_eq!(
            load_tokens("NAFS_TEST_TOK").unwrap(),
            vec!["abcdefghijklmnopqrst".to_string()]
        );
        std::fs::write(&d, "short\n").unwrap();
        assert!(load_tokens("NAFS_TEST_TOK").is_err());
        std::env::remove_var("NAFS_TEST_TOK_FILE");
        let _ = std::fs::remove_file(d);
    }
    #[test]
    fn secrets_never_reach_logs() {
        assert_eq!(
            redact_secrets(
                "Bearer abcdefghijklmnopqrst failed",
                &["abcdefghijklmnopqrst".into()]
            ),
            "Bearer [REDACTED] failed"
        );
    }
}
