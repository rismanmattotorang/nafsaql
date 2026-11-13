//! Skills (spec §7): `SKILL.md` files that say how to do a task, found by their description and read in full only
//! when used.

use serde::{Deserialize, Serialize};

/// Largest skill file, in bytes.
pub const MAX_SKILL: usize = 256 * 1024;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Skill {
    pub name: String,
    pub description: String,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub tools: Vec<String>,
    pub body: String,
}

fn list(v: &str) -> Vec<String> {
    v.trim()
        .trim_matches(['[', ']'])
        .split(',')
        .map(|s| s.trim().trim_matches(['"', '\'']).to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// A name a skill can have: `[a-z0-9-]{1,64}`, not starting or ending with `-`.
pub fn valid_name(n: &str) -> bool {
    (1..=64).contains(&n.len())
        && n.bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
        && !n.starts_with('-')
        && !n.ends_with('-')
}

impl Skill {
    /// Parse a `SKILL.md`. `name` from the URI must match the front matter's name when it has one.
    pub fn parse(name: &str, text: &str) -> Result<Skill, String> {
        if text.len() > MAX_SKILL {
            return Err(format!("a skill holds at most {} KB", MAX_SKILL / 1024));
        }
        if !valid_name(name) {
            return Err("a skill name is lower-case letters, digits and dashes, at most 64".into());
        }
        let t = text.trim_start_matches('\u{feff}');
        let (head, body) = match t
            .strip_prefix("---\n")
            .and_then(|r| r.find("\n---").map(|e| (r, e)))
        {
            Some((r, e)) => (&r[..e], r[e + 4..].trim_start_matches('\n')),
            None => ("", t),
        };
        let mut s = Skill {
            name: name.to_string(),
            description: String::new(),
            tags: Vec::new(),
            tools: Vec::new(),
            body: body.trim().to_string(),
        };
        for line in head.lines() {
            let Some((k, v)) = line.split_once(':') else {
                continue;
            };
            match k.trim() {
                "name" if v.trim() != name => {
                    return Err(format!(
                        "the file names the skill \"{}\" but it is saved as \"{name}\"",
                        v.trim()
                    ))
                }
                "description" => s.description = v.trim().trim_matches(['"', '\'']).to_string(),
                "tags" => s.tags = list(v),
                "tools" | "allowed-tools" => s.tools = list(v),
                _ => {}
            }
        }
        if s.description.is_empty() {
            // The first paragraph says what the skill is for.
            s.description = s
                .body
                .split("\n\n")
                .map(|p| p.trim().trim_start_matches('#').trim())
                .find(|p| !p.is_empty())
                .unwrap_or("")
                .chars()
                .take(300)
                .collect();
        }
        if s.description.chars().count() > 300 {
            return Err("a description is at most 300 characters".into());
        }
        if s.body.is_empty() {
            return Err("a skill needs instructions after its front matter".into());
        }
        Ok(s)
    }

    pub fn to_markdown(&self) -> String {
        let mut out = format!(
            "---\nname: {}\ndescription: {}\n",
            self.name, self.description
        );
        if !self.tags.is_empty() {
            out.push_str(&format!("tags: [{}]\n", self.tags.join(", ")));
        }
        if !self.tools.is_empty() {
            out.push_str(&format!("tools: [{}]\n", self.tools.join(", ")));
        }
        out.push_str("---\n");
        out.push_str(&self.body);
        out.push('\n');
        out
    }

    /// L0: name, description and tags, the text a skill is found by.
    pub fn abstract_line(&self) -> String {
        let mut s = format!("{}: {}", self.name, self.description);
        if !self.tags.is_empty() {
            s.push_str(&format!(" ({})", self.tags.join(", ")));
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_and_render() {
        let text = "---\nname: release-notes\ndescription: Write release notes from merged PRs\ntags: [writing, git]\ntools: [shell]\n---\n# Steps\n1. List the PRs.\n";
        let s = Skill::parse("release-notes", text).unwrap();
        assert_eq!(s.tags, ["writing", "git"]);
        assert_eq!(s.tools, ["shell"]);
        assert_eq!(Skill::parse("release-notes", &s.to_markdown()).unwrap(), s);
        assert!(s
            .abstract_line()
            .starts_with("release-notes: Write release notes"));
    }

    #[test]
    fn plain_markdown_and_bad_input() {
        let s = Skill::parse(
            "summarise",
            "# Summarise a paper\n\nRead the abstract first.",
        )
        .unwrap();
        assert_eq!(s.description, "Summarise a paper");
        assert!(Skill::parse("Bad Name", "x").is_err());
        assert!(Skill::parse("a", "---\nname: b\n---\nbody").is_err());
        assert!(Skill::parse("a", "---\nname: a\n---\n").is_err());
        assert!(!valid_name("-a") && valid_name("a-1"));
    }
}
