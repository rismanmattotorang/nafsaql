//! The embedding client (Phase 2): `nafs-smiley-server --embedding` serving Qwen3-Embedding-0.6B (`POST
//! /v1/embeddings`). Passages are embedded without an instruction; a search query gets the retrieval instruction, the
//! model's query format.

use std::time::Duration;

use serde_json::{json, Value};

use crate::KbError;

/// The instruction a retrieval query is embedded with (Qwen3-Embedding's query format).
pub const QUERY_INSTRUCTION: &str =
    "Given a question, retrieve passages of the knowledge base that answer it";

#[derive(Debug, Clone)]
pub struct EmbedConfig {
    pub url: String,
    /// Model id recorded with the vectors (`kb.embedder`); vectors of different models are never mixed.
    pub model: String,
    pub dims: usize,
    /// Passages per request.
    pub batch: usize,
}

#[derive(Clone)]
pub struct Embedder {
    http: reqwest::Client,
    pub cfg: EmbedConfig,
}

impl Embedder {
    pub fn new(cfg: EmbedConfig) -> Embedder {
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(2))
            // A batch of long passages on a busy CPU can take a while; the server answers in order.
            .timeout(Duration::from_secs(600))
            .build()
            .expect("HTTP client");
        Embedder { http, cfg }
    }

    /// Embed texts (passages: `instruction = None`). Returns one unit vector per text, in order.
    pub async fn embed(
        &self,
        texts: &[String],
        instruction: Option<&str>,
    ) -> Result<Vec<Vec<f32>>, KbError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let mut body = json!({ "input": texts });
        if let Some(i) = instruction {
            body["instruction"] = json!(i);
        }
        let resp = self
            .http
            .post(format!(
                "{}/v1/embeddings",
                self.cfg.url.trim_end_matches('/')
            ))
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                KbError::Unavailable(format!("the embedding server is not reachable: {e}"))
            })?;
        let status = resp.status();
        let v: Value = resp.json().await.map_err(|e| {
            KbError::Store(format!(
                "bad reply from the embedding server ({status}): {e}"
            ))
        })?;
        if !status.is_success() {
            let msg = v
                .pointer("/error/message")
                .and_then(Value::as_str)
                .unwrap_or("unknown error");
            return Err(KbError::Store(format!(
                "the embedding server refused the request: {msg}"
            )));
        }
        let data = v
            .get("data")
            .and_then(Value::as_array)
            .ok_or_else(|| KbError::Store("no data in the embedding reply".into()))?;
        let mut out = vec![Vec::new(); texts.len()];
        for d in data {
            let i = d.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
            let e: Vec<f32> = d
                .get("embedding")
                .and_then(Value::as_array)
                .map(|a| {
                    a.iter()
                        .filter_map(Value::as_f64)
                        .map(|x| x as f32)
                        .collect()
                })
                .unwrap_or_default();
            if e.len() != self.cfg.dims {
                return Err(KbError::Store(format!(
                    "the embedding server returned {} dimensions; [knowledge] embed_dims is {}",
                    e.len(),
                    self.cfg.dims
                )));
            }
            if let Some(slot) = out.get_mut(i) {
                *slot = e;
            }
        }
        if out.iter().any(Vec::is_empty) {
            return Err(KbError::Store(
                "the embedding reply is missing some texts".into(),
            ));
        }
        Ok(out)
    }

    /// Whether the server answers.
    pub async fn health(&self) -> bool {
        self.http
            .get(format!("{}/health", self.cfg.url.trim_end_matches('/')))
            .timeout(Duration::from_secs(2))
            .send()
            .await
            .is_ok_and(|r| r.status().is_success())
    }
}

/// The text a passage is embedded and searched as (FR-K51, FR-K53): its document title and heading path, then the
/// context line the extractor wrote for it, then the passage. A passage that says "it" still carries what it is
/// about, and the same text backs both the vectors and the words, so neither arm is blind to the context the other
/// can see.
///
/// The context line is empty until the graph pass has read the passage (it comes free from that call); a base
/// indexed before it, or with the graph off, keeps the title-and-heading header alone.
pub fn passage_text(doc_title: &str, heading: &str, context: &str, text: &str) -> String {
    let mut head = doc_title.trim().to_string();
    if !heading.trim().is_empty() && heading.trim() != doc_title.trim() {
        head.push_str(" > ");
        head.push_str(heading.trim());
    }
    if !context.trim().is_empty() {
        if !head.is_empty() {
            head.push('\n');
        }
        head.push_str(context.trim());
    }
    if head.is_empty() {
        text.to_string()
    } else {
        format!("{head}\n\n{text}")
    }
}

/// Reciprocal-rank fusion (Cormack et al. 2009, k = 60): each list contributes 1 / (k + rank) for every item.
pub fn rrf<'a>(lists: &[Vec<&'a str>], k: f64) -> Vec<(&'a str, f64)> {
    let mut score: Vec<(&str, f64)> = Vec::new();
    for list in lists {
        for (rank, id) in list.iter().enumerate() {
            let add = 1.0 / (k + rank as f64 + 1.0);
            match score.iter_mut().find(|(x, _)| x == id) {
                Some((_, s)) => *s += add,
                None => score.push((id, add)),
            }
        }
    }
    score.sort_by(|a, b| b.1.total_cmp(&a.1));
    score
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fusion_rewards_agreement() {
        let a = vec!["x", "y", "z"];
        let b = vec!["y", "w"];
        let f = rrf(&[a, b], 60.0);
        assert_eq!(f[0].0, "y");
        assert_eq!(f.len(), 4);
        assert!((f[0].1 - (1.0 / 62.0 + 1.0 / 61.0)).abs() < 1e-12);
    }

    #[test]
    fn passage_headers() {
        assert_eq!(
            passage_text("Paper", "Intro", "", "Body"),
            "Paper > Intro\n\nBody"
        );
        assert_eq!(passage_text("Paper", "Paper", "", "Body"), "Paper\n\nBody");
        assert_eq!(passage_text("", "", "", "Body"), "Body");
        // FR-K53: the context line sits under the heading path, above the body.
        assert_eq!(
            passage_text("Paper", "Intro", "What the method is.", "Body"),
            "Paper > Intro\nWhat the method is.\n\nBody"
        );
        assert_eq!(passage_text("", "", " ctx ", "Body"), "ctx\n\nBody");
    }
}
