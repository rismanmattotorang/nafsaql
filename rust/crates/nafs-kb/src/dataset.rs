//! Training data for a Memorilla module from one knowledge base (Phase 6, FR-K41).
//!
//! Every row reads the **whole** knowledge base, as chat's `memory` mode does when it serves. Two kinds need no model:
//! - reconstruction (Memorilla's Stage 1 task on our own passages): "Restate the passage under H that begins W",
//!   answered by the passage, cut at a sentence near `answer_chars` so a CPU step stays affordable;
//! - graph claims (FactKG-style, exact answers): a relation of the knowledge graph as a true claim, and the same
//!   relation with its object swapped for another entity of the same type as a false one.
//!
//! Questions are embedded without an instruction, as Memorilla reads them. Rows whose question hashes to 0 mod 10 are
//! the validation split. Written as `dataset.json` (rows and passage ids), `documents.safetensors` and
//! `questions.safetensors` (f32 `embeddings`); `training/memorilla/export_dataset.py` converts it to Memorilla's parquet
//! layout for the PyTorch trainer.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::{Kb, KbError};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Row {
    pub kind: String,
    pub question: String,
    pub answer: String,
    pub split: String,
    /// The passage a reconstruction row restates, or the passages that state a claim's relation.
    #[serde(default)]
    pub evidence: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetInfo {
    pub kb: String,
    pub kb_name: String,
    pub encoder: String,
    pub dims: usize,
    pub documents: Vec<String>,
    pub rows: Vec<Row>,
    pub created_at: String,
}

#[derive(Debug, Clone, Copy)]
pub struct DatasetOpts {
    pub max_reconstruction: usize,
    pub max_claims: usize,
    pub answer_chars: usize,
}

impl Default for DatasetOpts {
    fn default() -> Self {
        DatasetOpts {
            max_reconstruction: 400,
            max_claims: 400,
            answer_chars: 300,
        }
    }
}

/// FNV-1a, for a split that does not change between runs.
fn hash(s: &str) -> u64 {
    s.bytes().fold(0xcbf29ce484222325u64, |h, b| {
        (h ^ b as u64).wrapping_mul(0x100000001b3)
    })
}

fn split_of(q: &str) -> &'static str {
    if hash(q).is_multiple_of(10) {
        "validation"
    } else {
        "train"
    }
}

/// The passage without heading lines, cut at the last sentence end before `max` characters (or a word boundary).
pub fn answer_text(text: &str, max: usize) -> String {
    let body: String = text
        .lines()
        .filter(|l| !l.trim_start().starts_with('#'))
        .collect::<Vec<_>>()
        .join(" ");
    let body = body.split_whitespace().collect::<Vec<_>>().join(" ");
    if body.chars().count() <= max {
        return body;
    }
    let cut: String = body.chars().take(max).collect();
    // The last sentence end, as a character position: `rfind` gives bytes, and comparing those with a character
    // count would let a sentence end a sixth of the way into a non-ASCII passage pass the "not too short" guard.
    let end = cut
        .char_indices()
        .enumerate()
        .filter(|(_, (_, c))| matches!(c, '.' | '!' | '?'))
        .map(|(n, (b, c))| (n, b + c.len_utf8()))
        .last();
    match end {
        Some((n, b)) if n > max / 3 => cut[..b].to_string(),
        _ => cut
            .rsplit_once(' ')
            .map(|(a, _)| a.to_string())
            .unwrap_or(cut),
    }
}

/// The reconstruction question for a passage.
pub fn reconstruction_question(doc_title: &str, heading: &str, text: &str) -> String {
    let words: Vec<&str> = text
        .lines()
        .filter(|l| !l.trim_start().starts_with('#'))
        .flat_map(str::split_whitespace)
        .take(8)
        .collect();
    let place = heading
        .rsplit(" > ")
        .next()
        .filter(|h| !h.trim().is_empty())
        .unwrap_or(doc_title);
    format!(
        "Restate the passage under \"{place}\" that begins \"{}\".",
        words.join(" ")
    )
}

/// The claim question for a relation.
pub fn claim_question(s: &str, p: &str, o: &str) -> String {
    format!(
        "Claim: {s} {p} {o}. Is the claim supported by the knowledge base? Answer True or False."
    )
}

/// Build the rows of a knowledge base (reconstruction from its passages, claims from its graph).
pub async fn build(kb: &Kb, base: &str, opts: DatasetOpts) -> Result<DatasetInfo, KbError> {
    kb.ready().await?;
    let store = kb.store();
    let summary = store.get_kb(base).await?;
    let dims = kb
        .embed_dims()
        .ok_or_else(|| KbError::Unavailable("searching by meaning is turned off".into()))?;
    let r = store
        .query(
            "SELECT record::id(id) AS id, doc.title AS title, heading, text, ord FROM chunk \
             WHERE kb = type::record('kb', $kb) AND embedding != NONE ORDER BY doc, ord;",
            json!({ "kb": base }),
        )
        .await?;
    let chunks: Vec<Value> = r
        .into_iter()
        .next()
        .and_then(|v| v.as_array().cloned())
        .unwrap_or_default();
    if chunks.is_empty() {
        return Err(KbError::BadInput(
            "the knowledge base has no indexed passages".into(),
        ));
    }
    let s = |v: &Value, k: &str| v[k].as_str().unwrap_or("").to_string();
    let documents: Vec<String> = chunks.iter().map(|c| s(c, "id")).collect();
    let mut rows = Vec::new();
    // Reconstruction, spread over the whole base when it is larger than the cap.
    let step = (chunks.len() as f64 / opts.max_reconstruction.max(1) as f64).max(1.0);
    let mut i = 0.0;
    while (i as usize) < chunks.len() && rows.len() < opts.max_reconstruction {
        let c = &chunks[i as usize];
        let answer = answer_text(&s(c, "text"), opts.answer_chars);
        if answer.split_whitespace().count() >= 6 {
            let q = reconstruction_question(&s(c, "title"), &s(c, "heading"), &s(c, "text"));
            rows.push(Row {
                kind: "reconstruction".into(),
                split: split_of(&q).into(),
                question: q,
                answer,
                evidence: vec![s(c, "id")],
            });
        }
        i += step;
    }
    // Graph claims: each relation true, and once with the object swapped for an entity of the same type.
    let r = store
        .query(
            "LET $k = type::record('kb', $kb); \
             SELECT in.name AS s, predicate AS p, out.name AS o, out.type AS t, record::id(out) AS oid, \
               evidence.map(|$c| record::id($c)) AS ev FROM related WHERE kb = $k ORDER BY weight DESC; \
             SELECT record::id(id) AS id, name, type FROM entity WHERE kb = $k;",
            json!({ "kb": base }),
        )
        .await?;
    let mut it = r.into_iter().skip(1);
    let rels: Vec<Value> = it
        .next()
        .and_then(|v| v.as_array().cloned())
        .unwrap_or_default();
    let ents: Vec<Value> = it
        .next()
        .and_then(|v| v.as_array().cloned())
        .unwrap_or_default();
    let mut by_type: HashMap<String, Vec<(String, String)>> = HashMap::new();
    for e in &ents {
        by_type
            .entry(s(e, "type"))
            .or_default()
            .push((s(e, "id"), s(e, "name")));
    }
    let known: HashSet<(String, String, String)> = rels
        .iter()
        .map(|r| (s(r, "s"), s(r, "p"), s(r, "o")))
        .collect();
    let mut claims = 0;
    for r in &rels {
        if claims + 2 > opts.max_claims {
            break;
        }
        let (subj, pred, obj) = (s(r, "s"), s(r, "p"), s(r, "o"));
        let ev: Vec<String> = r["ev"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|x| x.as_str().map(str::to_string))
            .collect();
        let q = claim_question(&subj, &pred, &obj);
        rows.push(Row {
            kind: "claim".into(),
            split: split_of(&q).into(),
            question: q,
            answer: "True".into(),
            evidence: ev,
        });
        claims += 1;
        // A false claim: the first entity of the same type (by a hash of the claim) that makes an unknown relation.
        // The same type when it has another entity to offer; otherwise the largest type, chosen by name on a tie so
        // the dataset (and with it the train/validation split) is the same from one run to the next.
        let pool = by_type.get(&s(r, "t")).filter(|p| p.len() > 1).or_else(|| {
            by_type
                .iter()
                .max_by_key(|(t, p)| (p.len(), std::cmp::Reverse(t.as_str())))
                .map(|(_, p)| p)
        });
        if let Some(pool) = pool {
            let start = (hash(&format!("{subj}|{pred}|{obj}")) as usize) % pool.len();
            let pick = (0..pool.len())
                .map(|k| &pool[(start + k) % pool.len()])
                .find(|(id, name)| {
                    *id != s(r, "oid")
                        && *name != subj
                        && !known.contains(&(subj.clone(), pred.clone(), name.clone()))
                });
            if let Some((_, name)) = pick {
                let q = claim_question(&subj, &pred, name);
                rows.push(Row {
                    kind: "claim".into(),
                    split: split_of(&q).into(),
                    question: q,
                    answer: "False".into(),
                    evidence: vec![],
                });
                claims += 1;
            }
        }
    }
    Ok(DatasetInfo {
        kb: base.into(),
        kb_name: summary.name,
        encoder: kb
            .config()
            .embed
            .as_ref()
            .map(|e| e.model.clone())
            .unwrap_or_default(),
        dims,
        documents,
        rows,
        created_at: chrono::Utc::now().to_rfc3339(),
    })
}

/// Write the dataset: `dataset.json`, `documents.safetensors` (passage vectors in `documents` order) and
/// `questions.safetensors` (question vectors in row order).
pub async fn write(kb: &Kb, info: &DatasetInfo, dir: &Path) -> Result<(), KbError> {
    let io = |e: std::io::Error| KbError::Io(e.to_string());
    std::fs::create_dir_all(dir).map_err(io)?;
    let (docs, n) = kb.store().vectors_of(&info.kb, info.dims).await?;
    if n != info.documents.len() {
        return Err(KbError::Store(format!(
            "{n} passage vectors for {} passages; the base changed while the dataset was built",
            info.documents.len()
        )));
    }
    let mut questions = Vec::with_capacity(info.rows.len() * info.dims);
    for batch in info.rows.chunks(16) {
        let texts: Vec<String> = batch.iter().map(|r| r.question.clone()).collect();
        for v in kb.embed_texts_plain(&texts).await? {
            questions.extend(v);
        }
    }
    write_safetensors(&dir.join("documents.safetensors"), &docs, n, info.dims).map_err(io)?;
    write_safetensors(
        &dir.join("questions.safetensors"),
        &questions,
        info.rows.len(),
        info.dims,
    )
    .map_err(io)?;
    std::fs::write(
        dir.join("dataset.json"),
        serde_json::to_vec_pretty(info).unwrap_or_default(),
    )
    .map_err(io)
}

/// A minimal safetensors writer: one f32 tensor named `embeddings` of shape [rows, cols].
pub fn write_safetensors(
    path: &Path,
    data: &[f32],
    rows: usize,
    cols: usize,
) -> std::io::Result<()> {
    let bytes = data.len() * 4;
    let header = json!({ "embeddings": { "dtype": "F32", "shape": [rows, cols], "data_offsets": [0, bytes] } }).to_string();
    let mut header = header.into_bytes();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut out = Vec::with_capacity(8 + header.len() + bytes);
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(&header);
    for x in data {
        out.extend_from_slice(&x.to_le_bytes());
    }
    std::fs::write(path, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn answers_cut_at_sentences_and_questions_name_the_place() {
        let t = "# Title\n\nFirst sentence is here. Second sentence follows it closely. Third one runs on and on and on.";
        assert_eq!(
            answer_text(t, 60),
            "First sentence is here. Second sentence follows it closely."
        );
        assert_eq!(answer_text("short text", 60), "short text");
        assert_eq!(
            reconstruction_question("Paper", "Paper > 2 Method", t),
            "Restate the passage under \"2 Method\" that begins \"First sentence is here. Second sentence follows it\"."
        );
        assert!(reconstruction_question("Paper", "", "x y").contains("under \"Paper\""));
        assert_eq!(split_of("a"), split_of("a"));
        let splits: usize = (0..1000)
            .filter(|i| split_of(&format!("q{i}")) == "validation")
            .count();
        assert!((60..140).contains(&splits), "{splits}");
    }

    #[test]
    fn safetensors_round_trip_layout() {
        let p =
            std::env::temp_dir().join(format!("st-{}.safetensors", uuid::Uuid::new_v4().simple()));
        write_safetensors(&p, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3).unwrap();
        let b = std::fs::read(&p).unwrap();
        let n = u64::from_le_bytes(b[..8].try_into().unwrap()) as usize;
        assert_eq!(n % 8, 0);
        let h: Value = serde_json::from_slice(&b[8..8 + n]).unwrap();
        assert_eq!(h["embeddings"]["shape"], json!([2, 3]));
        assert_eq!(
            f32::from_le_bytes(b[8 + n + 20..8 + n + 24].try_into().unwrap()),
            6.0
        );
        let _ = std::fs::remove_file(p);
    }
}
