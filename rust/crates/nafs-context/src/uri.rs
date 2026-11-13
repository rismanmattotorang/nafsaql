//! `nafs://` URIs (spec §1): one namespace over the knowledge bases, the memory files, the skills and the committed
//! sessions. A URI names an item in the store that already holds it; nothing is copied into a second store.

use std::fmt;

/// The top-level spaces of the namespace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Space {
    Kb,
    Memory,
    Skills,
    Sessions,
}

impl Space {
    pub const ALL: [Space; 4] = [Space::Kb, Space::Memory, Space::Skills, Space::Sessions];

    pub fn as_str(self) -> &'static str {
        match self {
            Space::Kb => "kb",
            Space::Memory => "memory",
            Space::Skills => "skills",
            Space::Sessions => "sessions",
        }
    }

    pub fn parse(s: &str) -> Option<Space> {
        Space::ALL.into_iter().find(|x| x.as_str() == s)
    }

    /// What the space holds, in plain words (the root's listing shows it).
    pub fn describe(self) -> &'static str {
        match self {
            Space::Kb => "Knowledge bases: documents and their passages",
            Space::Memory => "What the assistant remembers about you: profile, preferences, people and things, events, notes",
            Space::Skills => "How to do tasks: instructions the assistant can follow",
            Space::Sessions => "Conversations saved as files, with what was learned from them",
        }
    }
}

/// The memory kinds (spec §5), in the order they are listed.
pub const MEMORY_KINDS: [&str; 5] = ["profile", "preferences", "entities", "events", "notes"];

/// A parsed `nafs://` URI. `space = None` is the root.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Uri {
    pub space: Option<Space>,
    pub path: Vec<String>,
    /// A passage of a document: `#p{ord}`.
    pub passage: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UriError(pub String);

impl fmt::Display for UriError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for UriError {}

const SCHEME: &str = "nafs://";
const MAX_LEN: usize = 1024;
const MAX_DEPTH: usize = 8;

fn unreserved(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '~' | '-')
}

/// Percent-encode a path segment.
pub fn encode_segment(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if unreserved(c) {
            out.push(c);
        } else {
            let mut buf = [0u8; 4];
            for b in c.encode_utf8(&mut buf).bytes() {
                out.push_str(&format!("%{b:02X}"));
            }
        }
    }
    out
}

fn decode_segment(s: &str) -> Result<String, UriError> {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let hex = s
                .get(i + 1..i + 3)
                .and_then(|h| u8::from_str_radix(h, 16).ok())
                .ok_or_else(|| UriError(format!("bad escape in \"{s}\"")))?;
            out.push(hex);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).map_err(|_| UriError(format!("\"{s}\" is not UTF-8")))
}

impl Uri {
    pub fn root() -> Uri {
        Uri {
            space: None,
            path: Vec::new(),
            passage: None,
        }
    }

    pub fn of(space: Space, path: &[&str]) -> Uri {
        Uri {
            space: Some(space),
            path: path.iter().map(|s| s.to_string()).collect(),
            passage: None,
        }
    }

    pub fn kb(base: &str) -> Uri {
        Uri::of(Space::Kb, &[base])
    }

    pub fn doc(base: &str, doc: &str) -> Uri {
        Uri::of(Space::Kb, &[base, doc])
    }

    pub fn passage(base: &str, doc: &str, ord: u64) -> Uri {
        Uri {
            passage: Some(ord),
            ..Uri::doc(base, doc)
        }
    }

    pub fn memory(kind: &str, key: &str) -> Uri {
        Uri::of(Space::Memory, &[kind, key])
    }

    pub fn skill(name: &str) -> Uri {
        Uri::of(Space::Skills, &[name])
    }

    pub fn session(id: &str) -> Uri {
        Uri::of(Space::Sessions, &[id])
    }

    /// Parse and normalise: the scheme is required, the space lower-case, segments non-empty, no `.` or `..`, a
    /// passage fragment only on a document.
    pub fn parse(s: &str) -> Result<Uri, UriError> {
        if s.len() > MAX_LEN {
            return Err(UriError("the URI is too long".into()));
        }
        let rest = s
            .strip_prefix(SCHEME)
            .ok_or_else(|| UriError(format!("\"{s}\" does not start with {SCHEME}")))?;
        let (rest, fragment) = match rest.split_once('#') {
            Some((a, b)) => (a, Some(b)),
            None => (rest, None),
        };
        let rest = rest.trim_end_matches('/');
        if rest.is_empty() {
            if fragment.is_some() {
                return Err(UriError("the root has no passages".into()));
            }
            return Ok(Uri::root());
        }
        let mut parts = rest.split('/');
        let space_s = parts.next().unwrap_or_default().to_ascii_lowercase();
        let space = Space::parse(&space_s).ok_or_else(|| {
            UriError(format!(
                "unknown space \"{space_s}\" (kb, memory, skills or sessions)"
            ))
        })?;
        let mut path = Vec::new();
        for p in parts {
            if p.is_empty() || p == "." || p == ".." {
                return Err(UriError(format!("bad path segment in \"{s}\"")));
            }
            let d = decode_segment(p)?;
            if d.contains('/') || d.chars().any(char::is_control) {
                return Err(UriError(format!("bad path segment in \"{s}\"")));
            }
            path.push(d);
        }
        if path.len() > MAX_DEPTH {
            return Err(UriError("the path is too deep".into()));
        }
        let passage = match fragment {
            None => None,
            Some(f) => {
                if space != Space::Kb || path.len() != 2 {
                    return Err(UriError("only a document has passages (#p<number>)".into()));
                }
                let n = f
                    .strip_prefix('p')
                    .and_then(|n| n.parse::<u64>().ok())
                    .ok_or_else(|| UriError(format!("bad passage \"#{f}\" (use #p<number>)")))?;
                Some(n)
            }
        };
        Ok(Uri {
            space: Some(space),
            path,
            passage,
        })
    }

    /// `true` when `other` is this URI or lies under it (a scope test). A passage lies under its document.
    pub fn contains(&self, other: &Uri) -> bool {
        match self.space {
            None => true,
            Some(s) => {
                other.space == Some(s)
                    && other.path.len() >= self.path.len()
                    && other.path[..self.path.len()] == self.path[..]
                    && (self.passage.is_none() || self.passage == other.passage)
            }
        }
    }

    pub fn parent(&self) -> Option<Uri> {
        if self.passage.is_some() {
            return Some(Uri {
                passage: None,
                ..self.clone()
            });
        }
        let space = self.space?;
        if self.path.is_empty() {
            return Some(Uri::root());
        }
        let mut p = self.path.clone();
        p.pop();
        Some(Uri {
            space: Some(space),
            path: p,
            passage: None,
        })
    }

    pub fn child(&self, name: &str) -> Uri {
        match self.space {
            None => Uri {
                space: Space::parse(name),
                path: Vec::new(),
                passage: None,
            },
            Some(_) => {
                let mut p = self.path.clone();
                p.push(name.to_string());
                Uri {
                    path: p,
                    passage: None,
                    ..self.clone()
                }
            }
        }
    }

    /// The last segment, for display.
    pub fn name(&self) -> String {
        if let Some(n) = self.passage {
            return format!("p{n}");
        }
        self.path
            .last()
            .cloned()
            .or_else(|| self.space.map(|s| s.as_str().to_string()))
            .unwrap_or_default()
    }
}

impl fmt::Display for Uri {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(SCHEME)?;
        if let Some(s) = self.space {
            f.write_str(s.as_str())?;
            for p in &self.path {
                write!(f, "/{}", encode_segment(p))?;
            }
        }
        if let Some(n) = self.passage {
            write!(f, "#p{n}")?;
        }
        Ok(())
    }
}

/// A glob over URIs: `*` matches within one segment, `**` across segments.
pub fn glob_match(pattern: &str, uri: &str) -> bool {
    fn rec(p: &[&str], u: &[&str]) -> bool {
        match (p.first(), u.first()) {
            (None, None) => true,
            (Some(&"**"), _) => rec(&p[1..], u) || (!u.is_empty() && rec(p, &u[1..])),
            (Some(ps), Some(us)) => seg(ps, us) && rec(&p[1..], &u[1..]),
            _ => false,
        }
    }
    fn seg(p: &str, s: &str) -> bool {
        // `*` within a segment: split on it and match the pieces in order.
        let pieces: Vec<&str> = p.split('*').collect();
        if pieces.len() == 1 {
            return p == s;
        }
        let mut rest = s;
        for (i, piece) in pieces.iter().enumerate() {
            if i == 0 {
                match rest.strip_prefix(piece) {
                    Some(r) => rest = r,
                    None => return false,
                }
            } else if i == pieces.len() - 1 {
                return rest.ends_with(piece);
            } else {
                match rest.find(piece) {
                    Some(at) => rest = &rest[at + piece.len()..],
                    None => return false,
                }
            }
        }
        true
    }
    let strip = |s: &'_ str| -> Vec<String> {
        s.trim_start_matches(SCHEME)
            .split(['/', '#'])
            .filter(|x| !x.is_empty())
            .map(str::to_string)
            .collect()
    };
    let (p, u) = (strip(pattern), strip(uri));
    let p: Vec<&str> = p.iter().map(String::as_str).collect();
    let u: Vec<&str> = u.iter().map(String::as_str).collect();
    rec(&p, &u)
}

/// A file-name-safe key from free text: lower-case ASCII letters and digits joined by `-`, at most 64 characters.
pub fn slug(text: &str) -> String {
    let mut out = String::new();
    let mut dash = false;
    for c in text.chars().flat_map(char::to_lowercase) {
        let c = match c {
            'à'..='å' => 'a',
            'ç' => 'c',
            'è'..='ë' => 'e',
            'ì'..='ï' => 'i',
            'ñ' => 'n',
            'ò'..='ö' => 'o',
            'ù'..='ü' => 'u',
            c => c,
        };
        if c.is_ascii_alphanumeric() {
            if dash && !out.is_empty() {
                out.push('-');
            }
            dash = false;
            out.push(c);
        } else {
            dash = true;
        }
        if out.len() >= 64 {
            break;
        }
    }
    let out = out.trim_end_matches('-').to_string();
    if out.is_empty() {
        "item".into()
    } else {
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_format_round_trip_and_normalise() {
        for s in [
            "nafs://",
            "nafs://kb",
            "nafs://kb/abc",
            "nafs://kb/abc/doc1#p12",
            "nafs://memory/preferences/answer-language",
            "nafs://skills/write-release-notes",
            "nafs://sessions/2f1c",
            "nafs://kb/a%20b",
        ] {
            assert_eq!(Uri::parse(s).unwrap().to_string(), s, "{s}");
        }
        assert_eq!(
            Uri::parse("nafs://KB/abc/").unwrap().to_string(),
            "nafs://kb/abc"
        );
        assert_eq!(Uri::parse("nafs://kb/a%20b").unwrap().path[0], "a b");
    }

    #[test]
    fn bad_uris_are_refused() {
        for s in [
            "http://kb",
            "nafs://files/x",
            "nafs://kb/../etc",
            "nafs://kb//x",
            "nafs://kb/a#p1",
            "nafs://memory/x/y#p1",
            "nafs://kb/a/b#q1",
            "nafs://kb/a%2Fb",
            "nafs://kb/%zz",
            "nafs://#p1",
        ] {
            assert!(Uri::parse(s).is_err(), "{s}");
        }
        assert!(Uri::parse(&format!("nafs://kb/{}", "a/".repeat(10))).is_err());
    }

    #[test]
    fn scopes_parents_and_children() {
        let base = Uri::kb("b");
        let doc = Uri::doc("b", "d");
        let p = Uri::passage("b", "d", 3);
        assert!(Uri::root().contains(&p));
        assert!(base.contains(&doc) && base.contains(&p) && doc.contains(&p));
        assert!(!doc.contains(&base));
        assert!(!Uri::kb("bb").contains(&doc));
        assert!(!Uri::of(Space::Memory, &[]).contains(&doc));
        assert_eq!(p.parent().unwrap(), doc);
        assert_eq!(doc.parent().unwrap(), base);
        assert_eq!(base.parent().unwrap(), Uri::of(Space::Kb, &[]));
        assert_eq!(Uri::of(Space::Kb, &[]).parent().unwrap(), Uri::root());
        assert_eq!(Uri::root().child("memory"), Uri::of(Space::Memory, &[]));
        assert_eq!(p.name(), "p3");
    }

    #[test]
    fn globs() {
        assert!(glob_match(
            "nafs://memory/*/*",
            "nafs://memory/preferences/x"
        ));
        assert!(!glob_match(
            "nafs://memory/*",
            "nafs://memory/preferences/x"
        ));
        assert!(glob_match(
            "nafs://memory/**",
            "nafs://memory/preferences/x"
        ));
        assert!(glob_match(
            "nafs://**/answer-*",
            "nafs://memory/preferences/answer-language"
        ));
        assert!(glob_match("nafs://kb/*/*", "nafs://kb/b/d"));
        assert!(!glob_match("nafs://skills/*", "nafs://kb/b"));
    }

    #[test]
    fn slugs() {
        assert_eq!(slug("Answer language: Bahasa!"), "answer-language-bahasa");
        assert_eq!(slug("Café Ünïon"), "cafe-union");
        assert_eq!(slug("!!!"), "item");
        assert!(slug(&"word ".repeat(40)).len() <= 64);
    }
}
