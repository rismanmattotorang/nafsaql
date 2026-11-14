//! Compile (spec §8): a scope rendered into Markdown files, with no model call. The inputs are what the stores
//! already hold: layers, sections, distinctive terms and the graphs.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::uri::slug;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DocInfo {
    pub uri: String,
    pub title: String,
    pub l0: String,
    pub l1: String,
    /// (heading path, URI of the first passage under it)
    pub sections: Vec<(String, String)>,
    pub terms: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Entity {
    pub name: String,
    pub kind: String,
    /// URIs of the passages that name it.
    pub mentions: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Relation {
    pub from: String,
    pub relation: String,
    pub to: String,
    pub evidence: Vec<String>,
    #[serde(default)]
    pub quote: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutFile {
    pub path: String,
    pub content: String,
}

/// Unique file names: a name already taken gets `-2`, `-3`, ... (two documents with one title, or titles that all
/// slug to `item`, would otherwise overwrite each other's page).
fn unique_names<'a>(names: impl Iterator<Item = &'a str>) -> Vec<String> {
    let mut seen: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    names
        .map(|n| {
            let base = slug(n);
            let k = seen.entry(base.clone()).or_insert(0);
            *k += 1;
            if *k == 1 {
                base
            } else {
                format!("{base}-{k}")
            }
        })
        .collect()
}

/// An overview without its own first heading (the page supplies it).
fn body_of(overview: &str) -> &str {
    let t = overview.trim();
    match t.strip_prefix("# ") {
        Some(rest) => rest.split_once('\n').map_or("", |(_, b)| b).trim(),
        None => t,
    }
}

/// A wiki: `index.md`, one page per document and one per term shared by two or more documents.
pub fn wiki(name: &str, overview: &str, docs: &[DocInfo]) -> Vec<OutFile> {
    let mut out = Vec::new();
    let names = unique_names(docs.iter().map(|d| d.title.as_str()));
    let by_uri: std::collections::HashMap<&str, &str> = docs
        .iter()
        .zip(&names)
        .map(|(d, n)| (d.uri.as_str(), n.as_str()))
        .collect();
    let page = |d: &DocInfo| {
        format!(
            "docs/{}.md",
            by_uri.get(d.uri.as_str()).copied().unwrap_or("item")
        )
    };
    let mut index = format!("# {name}\n\n{}\n\n## Documents\n\n", body_of(overview));
    for d in docs {
        index.push_str(&format!("- [{}]({}) — {}\n", d.title, page(d), d.l0.trim()));
    }
    // Terms named by two or more documents link them.
    let mut by_term: BTreeMap<String, Vec<&DocInfo>> = BTreeMap::new();
    for d in docs {
        for t in &d.terms {
            by_term.entry(t.clone()).or_default().push(d);
        }
    }
    by_term.retain(|_, ds| ds.len() >= 2);
    let term_names: std::collections::HashMap<String, String> = by_term
        .keys()
        .cloned()
        .zip(unique_names(by_term.keys().map(String::as_str)))
        .collect();
    let topic = |t: &str| term_names.get(t).cloned().unwrap_or_else(|| slug(t));
    if !by_term.is_empty() {
        index.push_str("\n## Shared topics\n\n");
        for (t, ds) in &by_term {
            index.push_str(&format!(
                "- [{t}](topics/{}.md) ({} documents)\n",
                topic(t),
                ds.len()
            ));
        }
    }
    out.push(OutFile {
        path: "index.md".into(),
        content: index,
    });
    for d in docs {
        let mut p = format!(
            "# {}\n\nSource: `{}`\n\n{}\n",
            d.title,
            d.uri,
            body_of(&d.l1)
        );
        if !d.sections.is_empty() {
            p.push_str("\n## Sections\n\n");
            for (h, uri) in &d.sections {
                p.push_str(&format!("- {h} — `{uri}`\n"));
            }
        }
        let shared: Vec<&String> = d
            .terms
            .iter()
            .filter(|t| by_term.contains_key(*t))
            .collect();
        if !shared.is_empty() {
            p.push_str("\n## See also\n\n");
            for t in shared {
                p.push_str(&format!("- [{t}](../topics/{}.md)\n", topic(t)));
            }
        }
        out.push(OutFile {
            path: page(d),
            content: p,
        });
    }
    for (t, ds) in &by_term {
        let mut p = format!("# {t}\n\nNamed in:\n\n");
        for d in ds {
            p.push_str(&format!(
                "- [{}](../{}) — {}\n",
                d.title,
                page(d),
                d.l0.trim()
            ));
        }
        out.push(OutFile {
            path: format!("topics/{}.md", topic(t)),
            content: p,
        });
    }
    out
}

/// A knowledge graph: one page per entity and `relations.jsonl` with evidence URIs.
pub fn graph(entities: &[Entity], relations: &[Relation]) -> Vec<OutFile> {
    let mut out = Vec::new();
    let names = unique_names(entities.iter().map(|e| e.name.as_str()));
    for (e, file) in entities.iter().zip(&names) {
        let mut p = format!(
            "---\nname: {}\ntype: {}\nmentions: {}\n---\n# {}\n\n",
            e.name,
            e.kind,
            e.mentions.len(),
            e.name
        );
        let rels: Vec<&Relation> = relations
            .iter()
            .filter(|r| r.from == e.name || r.to == e.name)
            .collect();
        if !rels.is_empty() {
            p.push_str("## Relations\n\n");
            for r in rels {
                p.push_str(&format!("- {} **{}** {}", r.from, r.relation, r.to));
                if !r.quote.is_empty() {
                    p.push_str(&format!(" — \"{}\"", r.quote));
                }
                p.push('\n');
            }
            p.push('\n');
        }
        p.push_str("## Mentioned in\n\n");
        for m in e.mentions.iter().take(50) {
            p.push_str(&format!("- `{m}`\n"));
        }
        out.push(OutFile {
            path: format!("entities/{file}.md"),
            content: p,
        });
    }
    let lines: Vec<String> = relations
        .iter()
        .filter_map(|r| serde_json::to_string(r).ok())
        .collect();
    out.push(OutFile {
        path: "relations.jsonl".into(),
        content: lines.join("\n") + if lines.is_empty() { "" } else { "\n" },
    });
    out
}

/// A report: the scope's overview and each document's.
pub fn report(name: &str, overview: &str, docs: &[DocInfo]) -> Vec<OutFile> {
    let mut p = format!("# {name}\n\n{}\n", body_of(overview));
    for d in docs {
        // The document's own sections become third-level headings under its name.
        let body = body_of(&d.l1).replace("\n## ", "\n### ");
        p.push_str(&format!(
            "\n## {}\n\nSource: `{}`\n\n{}\n",
            d.title, d.uri, body
        ));
    }
    vec![OutFile {
        path: format!("{}.md", slug(name)),
        content: p,
    }]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(title: &str, terms: &[&str]) -> DocInfo {
        DocInfo {
            uri: format!("nafs://kb/b/{}", slug(title)),
            title: title.into(),
            l0: format!("{title} in one line"),
            l1: format!("{title} overview"),
            sections: vec![("Intro".into(), format!("nafs://kb/b/{}#p0", slug(title)))],
            terms: terms.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn wiki_links_documents_through_shared_terms() {
        let files = wiki(
            "Papers",
            "Two papers",
            &[doc("Gate", &["BM25", "verifier"]), doc("Index", &["BM25"])],
        );
        let paths: Vec<&str> = files.iter().map(|f| f.path.as_str()).collect();
        assert_eq!(
            paths,
            [
                "index.md",
                "docs/gate.md",
                "docs/index.md",
                "topics/bm25.md"
            ]
        );
        assert!(files[0]
            .content
            .contains("[BM25](topics/bm25.md) (2 documents)"));
        assert!(
            files[1].content.contains("../topics/bm25.md")
                && !files[1].content.contains("verifier.md")
        );
    }

    #[test]
    fn pages_of_documents_with_one_title_do_not_collide() {
        let mut a = doc("Notes", &[]);
        let mut b = doc("Notes", &[]);
        a.uri = "nafs://kb/b/1".into();
        b.uri = "nafs://kb/b/2".into();
        let (mut c, mut d) = (doc("Обзор", &[]), doc("Сводка", &[]));
        c.uri = "nafs://kb/b/3".into();
        d.uri = "nafs://kb/b/4".into();
        let files = wiki("W", "o", &[a, b, c, d]);
        let mut paths: Vec<&str> = files.iter().map(|f| f.path.as_str()).collect();
        let n = paths.len();
        paths.sort();
        paths.dedup();
        assert_eq!(paths.len(), n, "{paths:?}");
    }

    #[test]
    fn graph_pages_and_relations() {
        let e = vec![Entity {
            name: "SurrealDB".into(),
            kind: "system".into(),
            mentions: vec!["nafs://kb/b/d#p1".into()],
        }];
        let r = vec![Relation {
            from: "Nafs-Aql".into(),
            relation: "stores data in".into(),
            to: "SurrealDB".into(),
            evidence: vec!["nafs://kb/b/d#p1".into()],
            quote: "SurrealDB is the source of truth".into(),
        }];
        let files = graph(&e, &r);
        assert_eq!(files[0].path, "entities/surrealdb.md");
        assert!(files[0].content.contains("**stores data in**"));
        assert!(files[1]
            .content
            .contains("\"evidence\":[\"nafs://kb/b/d#p1\"]"));
        assert_eq!(report("R", "o", &[doc("Gate", &[])])[0].path, "r.md");
    }
}
