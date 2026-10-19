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
