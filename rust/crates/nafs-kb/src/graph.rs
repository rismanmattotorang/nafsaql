//! The knowledge graph (Phase 3, FR-K30..K34): entities and relations extracted from each passage by the quick model,
//! checked against a JSON Schema, merged by name (and by meaning, in `Kb`), stored as SurrealDB edges, and grouped into
//! communities (Louvain). Graph search starts from the entities closest to a question and walks their relations.

use std::collections::HashMap;
use std::sync::OnceLock;
use std::time::Duration;

use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::KbError;

/// The default ontology (FR-K54): the entity types a knowledge base starts from, each with the one-line definition
/// that reaches the extraction prompt. A base may replace it; the types are data, not a branch in the code.
///
/// Measured on the quick model (Qwen3-1.7B) in Phase 3: objects with one worked example on an unrelated topic give
/// valid, faithful JSON; a description per extracted item costs 2.4x the output tokens, and arrays or line formats
/// made the model copy the template.
pub const DEFAULT_TYPES: [(&str, &str); 8] = [
    ("system", "a program, service or piece of software"),
    ("model", "a trained model or model family"),
    ("method", "a technique, algorithm or procedure"),
    ("dataset", "a named collection of data"),
    ("metric", "a measure or score"),
    ("person", "a named individual"),
    ("organization", "a company, institution or group"),
    ("concept", "anything else the passage names specifically"),
];

/// The entity types kept; anything else the model writes is mapped onto them (`algorithm` -> `method`).
pub const TYPES: [&str; 8] = [
    "system",
    "model",
    "method",
    "dataset",
    "metric",
    "person",
    "organization",
    "concept",
];

/// One entity type of an ontology: its name as the model writes it, and the definition the prompt carries.
#[derive(Debug, Clone, PartialEq, Serialize, serde::Deserialize)]
pub struct OntoType {
    pub name: String,
    pub definition: String,
}

/// FR-K54: the ontology of one knowledge base — the types a build runs under, as data.
#[derive(Debug, Clone, PartialEq, Serialize, serde::Deserialize)]
pub struct Ontology {
    /// In the order they reach the prompt; the last one is the catch-all a stray type is mapped onto.
    pub types: Vec<OntoType>,
}

impl Default for Ontology {
    fn default() -> Self {
        Ontology {
            types: DEFAULT_TYPES
                .iter()
                .map(|(t, d)| OntoType {
                    name: t.to_string(),
                    definition: d.to_string(),
                })
                .collect(),
        }
    }
}

impl Ontology {
    /// The sha256 of the ontology as the prompt sees it, recorded with every extraction so a graph says what it was
    /// built under.
    pub fn hash(&self) -> String {
        let joined = self
            .types
            .iter()
            .map(|t| format!("{}={}", t.name, t.definition))
            .collect::<Vec<_>>()
            .join(";");
        hex(&Sha256::digest(joined.as_bytes()))[..16].to_string()
    }

    /// Whether a type the model wrote is one of ours.
    pub fn has(&self, t: &str) -> bool {
        self.types.iter().any(|x| x.name == t)
    }

    /// What an ontology a person submits must satisfy: 1 to 24 types, each a short lower-case name (letters, digits,
    /// `_`, `-`) that is unique, with a definition of at most 200 characters.
    pub fn validate(&self) -> Result<(), String> {
        if self.types.is_empty() || self.types.len() > 24 {
            return Err("an ontology has between 1 and 24 types".into());
        }
        let mut seen = std::collections::HashSet::new();
        for t in &self.types {
            let ok_name = !t.name.is_empty()
                && t.name.len() <= 32
                && t.name
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-')
                && t.name
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_ascii_lowercase());
            if !ok_name {
                return Err(format!(
                    "type name \"{}\" must be lower-case letters, digits, _ or -, starting with a letter",
                    t.name
                ));
            }
            if !seen.insert(t.name.as_str()) {
                return Err(format!("type \"{}\" is listed twice", t.name));
            }
            if t.definition.trim().is_empty() || t.definition.chars().count() > 200 {
                return Err(format!(
                    "type \"{}\" needs a definition of 1 to 200 characters",
                    t.name
                ));
            }
        }
        Ok(())
    }

    /// The extraction prompt built from this ontology (FR-K53, FR-K54, FR-K55): the types and their definitions, a
    /// worked example on an unrelated topic, a context line, and a quote per relation.
    pub fn extraction_prompt(&self) -> String {
        let types = self
            .types
            .iter()
            .map(|t| format!("{} ({})", t.name, t.definition.trim()))
            .collect::<Vec<_>>()
            .join(", ");
        let names = self
            .types
            .iter()
            .map(|t| t.name.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        // The worked example is typed with this ontology's own first type: an example naming a type the ontology
        // does not declare teaches the model to write one we would have to map away.
        let any = self
            .types
            .first()
            .map(|t| t.name.as_str())
            .unwrap_or("concept");
        format!(
            r#"You extract a knowledge graph from one passage and reply with one JSON object only.

Example passage: "The Rust compiler, maintained by the Rust Foundation, uses LLVM to generate machine code."
Example reply: {{"context":"How the Rust compiler is maintained and what it generates code with.","entities":[{{"name":"Rust compiler","type":"{any}"}},{{"name":"Rust Foundation","type":"{any}"}},{{"name":"LLVM","type":"{any}"}}],"relations":[{{"source":"Rust Foundation","predicate":"maintains","target":"Rust compiler","quote":"The Rust compiler, maintained by the Rust Foundation, uses LLVM to generate machine code."}}]}}

Rules: "context" is one sentence saying what this passage is about, in the document's own terms. Entities are the specific things the passage names; "type" is one of: {names}. The types mean: {types}. Each relation joins two listed entities with a short lower-case verb phrase, and "quote" is the sentence of the passage that states it, copied word for word. Only facts stated in the passage. At most 10 entities and 10 relations."#
        )
    }
}

/// The JSON Schema every extraction must satisfy before it is stored (FR-K30).
pub fn extraction_schema() -> &'static Value {
    static S: OnceLock<Value> = OnceLock::new();
    S.get_or_init(|| {
        json!({
            "type": "object",
            "required": ["entities"],
            "properties": {
                "entities": { "type": "array", "maxItems": 40, "items": {
                    "type": "object", "required": ["name"],
                    "properties": { "name": { "type": "string", "minLength": 1 }, "type": { "type": "string" } } } },
                "relations": { "type": "array", "maxItems": 40, "items": {
                    "type": "object", "required": ["source", "predicate", "target"],
                    "properties": { "source": { "type": "string", "minLength": 1 }, "predicate": { "type": "string", "minLength": 1 },
                                    "target": { "type": "string", "minLength": 1 },
                                    "quote": { "type": "string" } } } },
                "context": { "type": "string" }
            }
        })
    })
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ExtractedEntity {
    pub name: String,
    pub kind: String,
    pub key: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ExtractedRelation {
    pub source: String,
    pub predicate: String,
    pub target: String,
    /// FR-K55: the sentence of the passage that states it, checked to occur verbatim before it is kept. Empty only
    /// for a relation written by a model that did not supply one, which the quote gate drops.
    pub quote: String,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Extraction {
    pub entities: Vec<ExtractedEntity>,
    pub relations: Vec<ExtractedRelation>,
    /// FR-K53: one sentence saying what the passage is about, in the document's own terms. Indexed with the passage
    /// so a passage that says "it" is still found by what it is about; never shown to the person.
    pub context: String,
    /// FR-K55: relations dropped because their quote is not in the passage. Recorded, not hidden.
    pub quote_drops: usize,
}

/// The matching key of an entity name: lower case, letters and digits only, words joined by one space, a leading
/// article dropped. "The KV-cache" and "kv cache" share a key.
pub fn entity_key(name: &str) -> String {
    let lower = name.to_lowercase();
    let words: Vec<String> = lower
        .split(|c: char| !c.is_alphanumeric() && c != '.')
        .map(|w| w.trim_matches('.').to_string())
        .filter(|w| !w.is_empty())
        .collect();
    let start = usize::from(words.len() > 1 && matches!(words[0].as_str(), "the" | "a" | "an"));
    words[start..].join(" ")
}

fn kind_of(t: &str) -> String {
    let t = t.trim().to_lowercase();
    let mapped = match t.as_str() {
        x if TYPES.contains(&x) => x,
        "algorithm" | "technique" | "approach" | "architecture" | "module" => "method",
        "software" | "tool" | "framework" | "library" | "engine" | "service" => "system",
        "benchmark" | "corpus" | "data" => "dataset",
        "measure" | "score" | "quantity" => "metric",
        "company" | "institution" | "university" | "lab" | "team" => "organization",
        "author" | "researcher" => "person",
        _ => "concept",
    };
    mapped.to_string()
}

fn clean_name(s: &str) -> String {
    let s: String = s.chars().filter(|c| !c.is_control()).collect();
    let s = s.split_whitespace().collect::<Vec<_>>().join(" ");
    s.trim_matches(|c: char| c == '"' || c == '\'' || c == '`')
        .chars()
        .take(80)
        .collect()
}

/// Words too generic to be an entity on their own (the type names, and what documents call their own parts).
const GENERIC: [&str; 18] = [
    "system",
    "model",
    "method",
    "dataset",
    "metric",
    "person",
    "organization",
    "concept",
    "architecture",
    "components",
    "component",
    "data",
    "paper",
    "passage",
    "section",
    "figure",
    "table",
    "document",
];

/// The passage as the extractor reads it: Markdown heading lines dropped (a heading names a part of the document, not
/// a thing in it; measured: with headings in the input the model extracted "Architecture and Design" as the most
/// mentioned entity).
pub fn extraction_input(text: &str) -> String {
    text.lines()
        .filter(|l| !l.trim_start().starts_with('#'))
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

/// Parse the model's reply: the first JSON object in it, checked against the schema, then cleaned. Entities are
/// deduplicated by key and **grounded**: every word of the name must occur in the passage (measured: the 1.7B model
/// sometimes copies the prompt's example, "LLVM", into unrelated passages). Generic words are dropped. A relation
/// whose endpoint was not listed adds that endpoint as a concept (if grounded); self-relations are dropped. At most 12
/// of each.
pub fn parse_extraction(reply: &str, passage: &str) -> Result<Extraction, String> {
    parse_extraction_under(reply, passage, &Ontology::default())
}

/// The same, under one knowledge base's ontology (FR-K54): a type the model wrote that the ontology does not declare
/// is mapped onto one that it does, and the last declared type is the catch-all.
pub fn parse_extraction_under(
    reply: &str,
    passage: &str,
    ontology: &Ontology,
) -> Result<Extraction, String> {
    let words: std::collections::HashSet<String> =
        entity_key(passage).split(' ').map(str::to_string).collect();
    let start = reply.find('{').ok_or("no JSON object in the reply")?;
    let mut de = serde_json::Deserializer::from_str(&reply[start..]).into_iter::<Value>();
    let v = match de.next() {
        Some(Ok(v)) => v,
        Some(Err(e)) => return Err(format!("the reply is not valid JSON: {e}")),
        None => return Err("no JSON object in the reply".into()),
    };
    if !jsonschema::is_valid(extraction_schema(), &v) {
        return Err("the reply does not match the extraction schema".into());
    }
    let mut out = Extraction::default();
    let mut by_key: HashMap<String, usize> = HashMap::new();
    let mut add = |name: &str, kind: &str, out: &mut Extraction| -> Option<String> {
        let name = clean_name(name);
        let key = entity_key(&name);
        if key.is_empty() || key.len() < 2 && !key.chars().all(|c| c.is_ascii_uppercase()) {
            return None;
        }
        if GENERIC.contains(&key.as_str()) || !key.split(' ').all(|w| words.contains(w)) {
            return None;
        }
        if !by_key.contains_key(&key) && out.entities.len() < 12 {
            by_key.insert(key.clone(), out.entities.len());
            let mapped = kind_of(kind);
            let fallback = || {
                ontology
                    .types
                    .last()
                    .map(|t| t.name.clone())
                    .unwrap_or_else(|| "concept".to_string())
            };
            out.entities.push(ExtractedEntity {
                name,
                kind: if ontology.has(&mapped) {
                    mapped
                } else {
                    fallback()
                },
                key: key.clone(),
            });
        }
        by_key.contains_key(&key).then_some(key)
    };
    for e in v["entities"].as_array().into_iter().flatten() {
        add(
            e["name"].as_str().unwrap_or(""),
            e["type"].as_str().unwrap_or("concept"),
            &mut out,
        );
    }
    let mut seen = std::collections::HashSet::new();
    for r in v["relations"].as_array().into_iter().flatten() {
        let (Some(s), Some(t)) = (
            add(r["source"].as_str().unwrap_or(""), "concept", &mut out),
            add(r["target"].as_str().unwrap_or(""), "concept", &mut out),
        ) else {
            continue;
        };
        let p = r["predicate"]
            .as_str()
            .unwrap_or("")
            .replace('_', " ")
            .to_lowercase();
        let p = p
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .chars()
            .take(60)
            .collect::<String>();
        if s == t
            || p.is_empty()
            || out.relations.len() >= 12
            || !seen.insert((s.clone(), p.clone(), t.clone()))
        {
            continue;
        }
        // FR-K55, SR-K10: a relation is kept only with a quote that occurs in the passage verbatim. A model that
        // writes a plausible sentence the passage does not contain loses the relation, and the drop is counted.
        let quote = match verified_quote(r["quote"].as_str().unwrap_or(""), passage) {
            Some(q) => q,
            None => {
                out.quote_drops += 1;
                continue;
            }
        };
        out.relations.push(ExtractedRelation {
            source: s,
            predicate: p,
            target: t,
            quote,
        });
    }
    out.context = context_line(v["context"].as_str().unwrap_or(""), passage);
    Ok(out)
}

/// FR-K55: the quote as the passage states it, or `None`. The claimed sentence must occur in the passage after
/// whitespace is normalised on both sides; the stored form is the passage's own spacing, cut to 400 characters.
pub fn verified_quote(claimed: &str, passage: &str) -> Option<String> {
    let norm = |t: &str| t.split_whitespace().collect::<Vec<_>>().join(" ");
    let (q, p) = (norm(claimed), norm(passage));
    if q.len() < 12 || !p.contains(&q) {
        return None;
    }
    Some(q.chars().take(400).collect())
}

/// FR-K53: the extractor's context line, kept only when it says something of its own. One sentence, cut to 200
/// characters; a line that merely repeats the passage's first words adds nothing to the index and is dropped.
pub fn context_line(claimed: &str, passage: &str) -> String {
    let line: String = claimed
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .chars()
        .take(200)
        .collect();
    if line.chars().count() < 12 {
        return String::new();
    }
    // A line that opens the way the passage opens is the passage, not a description of it. Comparing the line's
    // first words against the passage's own opening catches it whether the line is shorter or longer.
    let opening: String = line
        .split_whitespace()
        .take(6)
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase();
    let passage_open = passage
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase();
    if !opening.is_empty() && passage_open.starts_with(&opening) {
        return String::new();
    }
    line
}

/// A stable record key for an entity of a knowledge base (so the same entity is upserted, never duplicated).
pub fn entity_id(kb: &str, key: &str) -> String {
    format!(
        "e{}",
        &hex(&Sha256::digest(format!("{kb}\0{key}").as_bytes()))[..24]
    )
}

/// A stable record key for a relation (source key, predicate, target key): repeats raise its weight.
pub fn relation_id(kb: &str, source: &str, predicate: &str, target: &str) -> String {
    format!(
        "r{}",
        &hex(&Sha256::digest(
            format!("{kb}\0{source}\0{predicate}\0{target}").as_bytes()
        ))[..24]
    )
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

#[derive(Debug, Clone)]
pub struct GraphConfig {
    /// OpenAI-compatible base URL of the extraction model (System 1's server), e.g. `http://127.0.0.1:8201/v1`.
    pub llm_url: String,
    pub model: String,
    pub max_tokens: usize,
    /// Two entities of the same type whose name vectors are at least this similar are one entity.
    pub merge_similarity: f64,
}

#[derive(Clone)]
pub struct Extractor {
    http: reqwest::Client,
    pub cfg: GraphConfig,
    /// FR-K54: the types this extractor runs under, and the prompt built from them.
    ontology: Ontology,
    prompt: String,
}

/// One extraction call's result and cost.
pub struct ExtractCall {
    pub result: Result<Extraction, String>,
    pub completion_tokens: u64,
    pub ms: u64,
}

impl Extractor {
    pub fn new(cfg: GraphConfig) -> Extractor {
        Extractor::with_ontology(cfg, Ontology::default())
    }

    /// FR-K54: the extractor built for one knowledge base's ontology; its prompt is built from the types.
    pub fn with_ontology(cfg: GraphConfig, ontology: Ontology) -> Extractor {
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(2))
            .timeout(Duration::from_secs(900))
            .build()
            .expect("HTTP client");
        let prompt = ontology.extraction_prompt();
        Extractor {
            http,
            cfg,
            ontology,
            prompt,
        }
    }

    pub fn ontology(&self) -> &Ontology {
        &self.ontology
    }

    /// Extract one passage (greedy, thinking off) under the ontology this extractor was built with. Network errors
    /// are `Err`; a reply that fails the schema is an `Ok` call whose `result` is the reason (the passage is then
    /// skipped, not retried).
    pub async fn extract(&self, passage: &str) -> Result<ExtractCall, KbError> {
        self.extract_with(passage, &self.ontology, &self.prompt)
            .await
    }

    /// FR-K54: extract under a knowledge base's own ontology. The prompt is built by the caller once per document
    /// (`Ontology::extraction_prompt`), not once per passage.
    pub async fn extract_with(
        &self,
        passage: &str,
        ontology: &Ontology,
        prompt: &str,
    ) -> Result<ExtractCall, KbError> {
        let t0 = std::time::Instant::now();
        let body = json!({
            "model": self.cfg.model,
            "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": format!("Passage:\n{}", extraction_input(passage))}],
            "temperature": 0, "max_tokens": self.cfg.max_tokens,
            "chat_template_kwargs": {"enable_thinking": false},
        });
        let resp = self
            .http
            .post(format!(
                "{}/chat/completions",
                self.cfg.llm_url.trim_end_matches('/')
            ))
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                KbError::Unavailable(format!("the extraction model is not reachable: {e}"))
            })?;
        let status = resp.status();
        let v: Value = resp.json().await.map_err(|e| {
            KbError::Store(format!(
                "bad reply from the extraction model ({status}): {e}"
            ))
        })?;
        if !status.is_success() {
            return Err(KbError::Store(format!(
                "the extraction model refused the request ({status})"
            )));
        }
        let text = v
            .pointer("/choices/0/message/content")
            .and_then(Value::as_str)
            .unwrap_or("");
        Ok(ExtractCall {
            result: parse_extraction_under(text, passage, ontology),
            completion_tokens: v
                .pointer("/usage/completion_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0),
            ms: t0.elapsed().as_millis() as u64,
        })
    }
}

/// Louvain community detection (Blondel et al. 2008) on an undirected weighted graph of `n` nodes. Returns a community
/// index per node, numbered from 0 by size (largest first). Deterministic: nodes are visited in index order.
pub fn louvain(n: usize, edges: &[(usize, usize, f64)]) -> Vec<usize> {
    // `node_of[i]`: the current super-node of original node i.
    let mut node_of: Vec<usize> = (0..n).collect();
    let mut level_edges: Vec<(usize, usize, f64)> = edges
        .iter()
        .filter(|e| e.0 < n && e.1 < n)
        .copied()
        .collect();
    let mut m = n;
    loop {
        let comm = one_level(m, &level_edges);
        let distinct = {
            let mut c = comm.clone();
            c.sort_unstable();
            c.dedup();
            c.len()
        };
        for x in node_of.iter_mut() {
            *x = comm[*x];
        }
        if distinct == m {
            break;
        }
        // Aggregate: communities become nodes, edge weights summed (self-loops keep internal weight).
        let mut agg: HashMap<(usize, usize), f64> = HashMap::new();
        for &(a, b, w) in &level_edges {
            let (x, y) = (comm[a].min(comm[b]), comm[a].max(comm[b]));
            *agg.entry((x, y)).or_default() += w;
        }
        level_edges = agg.into_iter().map(|((a, b), w)| (a, b, w)).collect();
        level_edges.sort_by_key(|e| (e.0, e.1));
        m = distinct;
    }
    // Renumber by size, largest first.
    let mut size: HashMap<usize, usize> = HashMap::new();
    for &c in &node_of {
        *size.entry(c).or_default() += 1;
    }
    let mut order: Vec<(usize, usize)> = size.into_iter().collect();
    order.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let rank: HashMap<usize, usize> = order
        .iter()
        .enumerate()
        .map(|(i, (c, _))| (*c, i))
        .collect();
    node_of.into_iter().map(|c| rank[&c]).collect()
}

/// One Louvain pass: local moves until no node improves modularity; communities numbered 0..k densely.
fn one_level(n: usize, edges: &[(usize, usize, f64)]) -> Vec<usize> {
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    let mut k = vec![0.0f64; n];
    let mut m2 = 0.0;
    for &(a, b, w) in edges {
        if a == b {
            k[a] += 2.0 * w;
            m2 += 2.0 * w;
            adj[a].push((a, 2.0 * w));
            continue;
        }
        adj[a].push((b, w));
        adj[b].push((a, w));
        k[a] += w;
        k[b] += w;
        m2 += 2.0 * w;
    }
    let mut comm: Vec<usize> = (0..n).collect();
    if m2 == 0.0 {
        return comm;
    }
    let mut tot = k.clone();
    let mut improved = true;
    let mut rounds = 0;
    while improved && rounds < 50 {
        improved = false;
        rounds += 1;
        for i in 0..n {
            let ci = comm[i];
            let mut w_to: HashMap<usize, f64> = HashMap::new();
            for &(j, w) in &adj[i] {
                if j != i {
                    *w_to.entry(comm[j]).or_default() += w;
                }
            }
            tot[ci] -= k[i];
            let gain = |c: usize, wic: f64| wic - tot[c] * k[i] / m2;
            let mut best = (ci, gain(ci, *w_to.get(&ci).unwrap_or(&0.0)));
            let mut cands: Vec<(usize, f64)> = w_to.into_iter().collect();
            cands.sort_by_key(|x| x.0);
            for (c, wic) in cands {
                let g = gain(c, wic);
                if g > best.1 + 1e-12 {
                    best = (c, g);
                }
            }
            tot[best.0] += k[i];
            if best.0 != ci {
                comm[i] = best.0;
                improved = true;
            }
        }
    }
    let mut ids: HashMap<usize, usize> = HashMap::new();
    comm.iter()
        .map(|c| {
            let l = ids.len();
            *ids.entry(*c).or_insert(l)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_merge_spelling_variants() {
        assert_eq!(entity_key("The KV-cache"), "kv cache");
        assert_eq!(entity_key("kv cache"), "kv cache");
        assert_eq!(entity_key("Qwen3-8B"), "qwen3 8b");
        assert_eq!(entity_key("vLLM"), "vllm");
        assert_eq!(entity_key("Qwen3.5"), "qwen3.5");
        assert_eq!(entity_key("the"), "the");
        assert_ne!(entity_id("kb1", "vllm"), entity_id("kb2", "vllm"));
        assert!(entity_id("kb1", "vllm").starts_with('e'));
    }

    #[test]
    fn extraction_from_the_measured_reply() {
        let q1 =
            "To address this, PagedAttention is inspired by virtual memory and paging techniques.";
        let q2 = "vLLM uses it and is compared to Orca.";
        let reply = format!(
            r#"Here: {{"context":"How PagedAttention came about and what uses it.","entities":[{{"name":"PagedAttention","type":"algorithm"}},{{"name":"vLLM","type":"system"}},{{"name":"vllm","type":"system"}},{{"name":"Orca","type":"model"}}],"relations":[{{"source":"PagedAttention","predicate":"inspired_by","target":"virtual memory and paging techniques","quote":"{q1}"}},{{"source":"vLLM","predicate":"uses","target":"PagedAttention","quote":"{q2}"}},{{"source":"vLLM","predicate":"compared_to","target":"Orca","quote":"{q2}"}},{{"source":"vLLM","predicate":"uses","target":"PagedAttention","quote":"{q2}"}},{{"source":"vLLM","predicate":"is","target":"vLLM","quote":"{q2}"}}]}} trailing"#
        );
        let reply = reply.as_str();
        let passage = "To address this, PagedAttention is inspired by virtual memory and paging techniques. vLLM uses it \
                       and is compared to Orca. The system is fast.";
        let x = parse_extraction(reply, passage).unwrap();
        let names: Vec<&str> = x.entities.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "PagedAttention",
                "vLLM",
                "Orca",
                "virtual memory and paging techniques"
            ]
        );
        assert_eq!(x.entities[0].kind, "method");
        assert_eq!(x.entities[3].kind, "concept");
        assert_eq!(x.relations.len(), 3, "{:?}", x.relations);
        assert_eq!(x.relations[0].predicate, "inspired by");
        assert_eq!(
            x.relations[0].target,
            "virtual memory and paging techniques"
        );
        // FR-K53, FR-K55: the context line and a quote the passage really contains.
        assert_eq!(x.context, "How PagedAttention came about and what uses it.");
        assert_eq!(x.relations[0].quote, q1);
        assert_eq!(x.quote_drops, 0);
    }

    /// FR-K55, SR-K10: a relation is kept only with a sentence the passage states, and a drop is counted.
    #[test]
    fn a_relation_without_a_real_quote_is_dropped_and_counted() {
        let passage = "The gateway reaches SurrealDB over HTTP on port 8300.";
        let reply = r#"{"entities":[{"name":"gateway","type":"system"},{"name":"SurrealDB","type":"system"}],
            "relations":[{"source":"gateway","predicate":"reaches","target":"SurrealDB","quote":"The gateway reaches SurrealDB over HTTP on port 8300."},
                         {"source":"SurrealDB","predicate":"stores","target":"gateway","quote":"SurrealDB stores every passage of the knowledge base."},
                         {"source":"gateway","predicate":"uses","target":"SurrealDB"}]}"#;
        let x = parse_extraction(reply, passage).unwrap();
        assert_eq!(x.relations.len(), 1, "{:?}", x.relations);
        assert_eq!(x.relations[0].predicate, "reaches");
        assert_eq!(x.quote_drops, 2, "an invented quote and a missing one");

        // Whitespace differs, the words do not: that is the same sentence.
        assert_eq!(
            verified_quote(
                "The  gateway\n reaches SurrealDB over HTTP on port 8300.",
                passage
            ),
            Some("The gateway reaches SurrealDB over HTTP on port 8300.".to_string())
        );
        assert_eq!(verified_quote("short", passage), None);
        assert_eq!(verified_quote("", passage), None);
    }

    /// FR-K53: a context line that only repeats the passage's opening adds nothing to the index.
    #[test]
    fn a_context_line_must_say_something_of_its_own() {
        let passage = "The gateway reaches SurrealDB over HTTP on port 8300.";
        assert_eq!(
            context_line("What the gateway talks to.", passage),
            "What the gateway talks to."
        );
        assert_eq!(
            context_line("The gateway reaches SurrealDB over HTTP", passage),
            ""
        );
        assert_eq!(context_line("too short", passage), "");
    }

    /// FR-K54: the types are data; the prompt is built from them and a type outside them is mapped onto one inside.
    #[test]
    fn the_ontology_builds_the_prompt_and_bounds_the_types() {
        let o = Ontology::default();
        assert!(o
            .extraction_prompt()
            .contains("system (a program, service or piece of software)"));
        assert!(o.extraction_prompt().contains("\"quote\""));
        assert_eq!(o.hash().len(), 16);
        let mine = Ontology {
            types: vec![
                OntoType {
                    name: "ship".into(),
                    definition: "a named vessel".into(),
                },
                OntoType {
                    name: "port".into(),
                    definition: "a place a ship calls at".into(),
                },
            ],
        };
        assert!(mine.validate().is_ok());
        assert!(Ontology { types: vec![] }.validate().is_err());
        assert!(Ontology {
            types: vec![OntoType {
                name: "Bad Name".into(),
                definition: "x".into()
            }]
        }
        .validate()
        .is_err());
        assert!(Ontology {
            types: vec![
                OntoType {
                    name: "a".into(),
                    definition: "x".into()
                },
                OntoType {
                    name: "a".into(),
                    definition: "y".into()
                }
            ]
        }
        .validate()
        .is_err());
        assert_ne!(mine.hash(), o.hash());
        assert!(mine.extraction_prompt().contains("ship (a named vessel)"));
        assert!(!mine.extraction_prompt().contains("organization"));
        // "system" is not in this ontology, so it falls to the last declared type.
        let x = parse_extraction_under(
            r#"{"entities":[{"name":"Endeavour","type":"system"}],"relations":[]}"#,
            "Endeavour sailed in 1768.",
            &mine,
        )
        .unwrap();
        assert_eq!(x.entities[0].kind, "port");
    }

    #[test]
    fn bad_replies_are_refused() {
        assert!(parse_extraction("no json", "").is_err());
        assert!(parse_extraction(r#"{"entities": "x"}"#, "")
            .unwrap_err()
            .contains("schema"));
        assert!(parse_extraction(r#"{"relations": []}"#, "")
            .unwrap_err()
            .contains("schema"));
        assert!(parse_extraction(r#"{"entities": [{"name": ""}]}"#, "").is_err());
        assert!(
            parse_extraction(r#"{"entities": [], "relations": [{"source": "a"}]}"#, "").is_err()
        );
    }

    #[test]
    fn entities_must_be_in_the_passage_and_specific() {
        let reply = r#"{"entities":[{"name":"LLVM","type":"system"},{"name":"Rust compiler","type":"system"},{"name":"SurrealDB","type":"system"},{"name":"system","type":"concept"},{"name":"the gateway","type":"system"}],
            "relations":[{"source":"gateway","predicate":"reaches","target":"SurrealDB","quote":"The gateway reaches SurrealDB over HTTP."},{"source":"Rust compiler","predicate":"uses","target":"LLVM","quote":"The gateway reaches SurrealDB over HTTP."}]}"#;
        let x = parse_extraction(
            reply,
            "The gateway reaches SurrealDB over HTTP. The system is local.",
        )
        .unwrap();
        let names: Vec<&str> = x.entities.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(names, ["SurrealDB", "the gateway"]);
        assert_eq!(x.relations.len(), 1);
        assert_eq!(
            extraction_input("# Title\n\nBody line\n## Sub\nMore"),
            "Body line\nMore"
        );
    }

    #[test]
    fn louvain_finds_two_cliques_joined_by_one_edge() {
        let mut e = Vec::new();
        for (a, b) in [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)] {
            e.push((a, b, 1.0));
            e.push((a + 4, b + 4, 1.0));
        }
        e.push((3, 4, 1.0));
        let c = louvain(9, &e); // node 8 is isolated
        assert!(c[0] == c[1] && c[1] == c[2] && c[2] == c[3]);
        assert!(c[4] == c[5] && c[5] == c[6] && c[6] == c[7]);
        assert_ne!(c[0], c[4]);
        assert_ne!(c[8], c[0]);
        assert_eq!(c[8], 2, "the singleton is the smallest community");
        assert_eq!(louvain(3, &[]), vec![0, 1, 2]);
    }
}
