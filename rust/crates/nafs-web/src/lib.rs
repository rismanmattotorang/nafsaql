//! nafs-web: the Web UI (chat, rooms, performance console, evaluation dashboard, system view).
//! Assets are embedded in the gateway binary; set NAFS_WEB_DIR to serve them from disk while editing.
pub const INDEX_HTML: &str = include_str!("../assets/index.html");
pub const APP_JS: &str = include_str!("../assets/app.js");
pub const APP_CSS: &str = include_str!("../assets/app.css");
pub const FAVICON_SVG: &str = include_str!("../assets/favicon.svg");

/// Return (content, mime) for a UI asset, preferring files under `dir` when given.
pub fn asset(name: &str, dir: Option<&str>) -> Option<(String, &'static str)> {
    let mime = match name.rsplit('.').next() {
        Some("js") => "text/javascript; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("svg") => "image/svg+xml",
        _ => "text/html; charset=utf-8",
    };
    if let Some(d) = dir {
        if let Ok(s) = std::fs::read_to_string(format!("{d}/{name}")) {
            return Some((s, mime));
        }
    }
    let s = match name {
        "index.html" => INDEX_HTML,
        "app.js" => APP_JS,
        "app.css" => APP_CSS,
        "favicon.svg" => FAVICON_SVG,
        _ => return None,
    };
    Some((s.to_string(), mime))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Internal vocabulary the UI must not show: requirement and task ids, theorem references, the model
    /// symbols of the research report, and the "System 1/2" names (the UI says quick and thorough).
    fn internal_term(code: &str) -> Option<String> {
        let b = code.as_bytes();
        let word_start = |i: usize| i == 0 || !(b[i - 1] as char).is_ascii_alphanumeric();
        for (i, _) in code.char_indices() {
            let rest = &code[i..];
            for id in ["FR-", "NFR-", "ER-", "TR-", "OR-", "SR-"] {
                if word_start(i)
                    && rest.starts_with(id)
                    && rest[id.len()..].starts_with(|c: char| c.is_ascii_digit())
                {
                    return Some(rest.chars().take(10).collect());
                }
            }
            for lead in ["Prop. ", "Theorem ", "Thm ", "System "] {
                if word_start(i)
                    && rest.starts_with(lead)
                    && rest[lead.len()..].starts_with(|c: char| c.is_ascii_digit())
                {
                    return Some(rest.chars().take(lead.len() + 2).collect());
                }
            }
            for sym in ["ê₁", "ê₂", "λ", "Δc"] {
                if rest.starts_with(sym) {
                    return Some(sym.into());
                }
            }
        }
        None
    }

    #[test]
    fn ui_copy_has_no_internal_references() {
        let mut found = vec![];
        for (file, text) in [("app.js", APP_JS), ("index.html", INDEX_HTML)] {
            for (n, line) in text.lines().enumerate() {
                // Comments may cite requirements; a line marked `copy-lint: allow` holds a pattern, not copy.
                if line.trim_start().starts_with("//") || line.contains("copy-lint: allow") {
                    continue;
                }
                let code = line.find(" // ").map_or(line, |i| &line[..i]);
                if let Some(t) = internal_term(code) {
                    found.push(format!("{file}:{}: {t}", n + 1));
                }
            }
        }
        assert!(
            found.is_empty(),
            "internal terms in UI copy:\n{}",
            found.join("\n")
        );
    }

    #[test]
    fn the_lint_catches_what_it_should() {
        assert_eq!(
            internal_term("needs 700 pairs (FR-050)").as_deref(),
            Some("FR-050)")
        );
        assert!(
            internal_term("System 2 thinks").is_some() && internal_term("by Theorem 2").is_some()
        );
        assert!(internal_term("ê₁ 0.3").is_some() && internal_term("speed λ").is_some());
        assert!(internal_term("PERFORMER-1 and the System and SR-x").is_none());
    }
}
