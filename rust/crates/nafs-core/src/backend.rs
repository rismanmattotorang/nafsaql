//! OpenAI-compatible backend client used for both llama-server and vLLM (IF-006..IF-008).
use crate::config::{EndpointCfg, RetryCfg};
use eventsource_stream::Eventsource;
use futures::StreamExt;
use nafs_types::{Message, NafsError, Result};
use serde::Serialize;
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;

/// Cooperative cancellation: returns true once the caller no longer wants the result (review M8).
#[derive(Clone)]
pub struct CancelFlag(pub Arc<dyn Fn() -> bool + Send + Sync>);
impl std::fmt::Debug for CancelFlag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CancelFlag")
    }
}
impl CancelFlag {
    pub fn is_set(&self) -> bool {
        (self.0)()
    }
}

#[derive(Debug, Clone, Default)]
pub struct GenRequest {
    pub messages: Vec<Message>,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub logprobs: bool,
    pub thinking: Option<bool>,
    pub thinking_budget: Option<u32>,
    pub seed: Option<u64>,
    pub cancel: Option<CancelFlag>,
    /// OpenAI-style `response_format` (e.g. a JSON schema): grammar-constrained answer text. Both
    /// llama-server and vLLM apply it after the reasoning span.
    pub response_format: Option<Value>,
    /// P7-T03 / Phase 8: stop the stream early (finish_reason "early_escalation") when a confidence
    /// signal crosses its threshold.
    pub early_stop: Option<EarlyStop>,
}

/// Early-escalation thresholds on the draft's token log-probs. `below_mean` is the P7-T03 rule (running
/// mean below the threshold after `min_n` tokens; predictive, error not controlled). `below_window` is the
/// Phase-8 rule: the lowest `window`-token mean falls below the threshold, which is exactly the decision
/// the full draft would receive (the signal is a running minimum).
#[derive(Debug, Clone, Default)]
pub struct EarlyStop {
    pub min_n: usize,
    pub below_mean: Option<f32>,
    pub window: usize,
    pub below_window: Option<f32>,
}

/// A raw continuation request (P7-T08): the rendered chat template plus any assistant prefix.
#[derive(Debug, Clone, Default)]
pub struct RawRequest {
    pub prompt: String,
    pub n_predict: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub seed: Option<u64>,
    /// Stop strings; the server removes the matched string from the output and reports it.
    pub stop: Vec<String>,
    pub cancel: Option<CancelFlag>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct RawOutput {
    pub text: String,
    pub tokens: u32,
    /// "eos", "word" (a stop string matched) or "limit" (n_predict reached).
    pub stop_type: String,
    pub stopping_word: String,
    pub ttft_ms: Option<u64>,
    pub wall_ms: u64,
    pub queue_ms: u64,
    pub prefill_ms: Option<f64>,
    pub decode_ms: Option<f64>,
    pub prompt_n: Option<u32>,
    pub cache_n: Option<u32>,
}
impl RawOutput {
    pub fn stage(&self, name: &str) -> nafs_types::StageTiming {
        nafs_types::StageTiming {
            stage: name.into(),
            ms: self.wall_ms + self.queue_ms,
            queue_ms: Some(self.queue_ms),
            prefill_ms: self.prefill_ms,
            decode_ms: self.decode_ms,
            prompt_n: self.prompt_n,
            cache_n: self.cache_n,
            tokens: Some(self.tokens),
            note: None,
        }
    }
}

/// Run `fut`, returning Cancelled as soon as the flag is set (checked every 250 ms).
async fn with_cancel<T, F: std::future::Future<Output = Result<T>>>(
    cancel: Option<&CancelFlag>,
    fut: F,
) -> Result<T> {
    tokio::pin!(fut);
    loop {
        tokio::select! {
            r = &mut fut => return r,
            _ = tokio::time::sleep(Duration::from_millis(250)) => {
                if cancel.map(|c| c.is_set()).unwrap_or(false) {
                    return Err(NafsError::Cancelled);
                }
            }
        }
    }
}

/// Decrements the in-flight gauge even when the future is dropped (review L6).
struct InflightGuard(String);
impl InflightGuard {
    fn new(role: &str) -> Self {
        metrics::gauge!("nafs_backend_inflight", "backend" => role.to_string()).increment(1.0);
        Self(role.to_string())
    }
}
impl Drop for InflightGuard {
    fn drop(&mut self) {
        metrics::gauge!("nafs_backend_inflight", "backend" => self.0.clone()).decrement(1.0);
    }
}

/// Run `fut` but give up as soon as the request's cancel flag is set (polled every 200 ms).
async fn cancellable<T>(
    req: &GenRequest,
    fut: impl std::future::Future<Output = Result<T>>,
) -> Result<T> {
    let Some(c) = req.cancel.clone() else {
        return fut.await;
    };
    tokio::pin!(fut);
    loop {
        tokio::select! {
            r = &mut fut => return r,
            _ = tokio::time::sleep(Duration::from_millis(200)) => {
                if c.is_set() { return Err(NafsError::Cancelled); }
            }
        }
    }
}

/// Splits inline `<think>...</think>` spans out of streamed content when the server has no reasoning
/// parser, so chain-of-thought is never streamed as answer text (review L7). Tags split across chunks
/// are held back until they can be classified.
#[derive(Default)]
struct ThinkSplitter {
    in_think: bool,
    pending: String,
}
impl ThinkSplitter {
    /// Returns (text, is_reasoning) pieces in order.
    fn push(&mut self, chunk: &str) -> Vec<(String, bool)> {
        let mut buf = std::mem::take(&mut self.pending);
        buf.push_str(chunk);
        let mut out = vec![];
        loop {
            let tag = if self.in_think { "</think>" } else { "<think>" };
            if let Some(i) = buf.find(tag) {
                if i > 0 {
                    out.push((buf[..i].to_string(), self.in_think));
                }
                buf = buf[i + tag.len()..].to_string();
                self.in_think = !self.in_think;
                continue;
            }
            // Hold back a suffix that could be the start of the tag.
            let mut keep = 0;
            for k in (1..tag.len()).rev() {
                if buf.len() >= k
                    && buf.is_char_boundary(buf.len() - k)
                    && tag.starts_with(&buf[buf.len() - k..])
                {
                    keep = k;
                    break;
                }
            }
            let cut = buf.len() - keep;
            if cut > 0 {
                out.push((buf[..cut].to_string(), self.in_think));
            }
            self.pending = buf[cut..].to_string();
            return out;
        }
    }
    fn flush(&mut self) -> Option<(String, bool)> {
        let p = std::mem::take(&mut self.pending);
        (!p.is_empty()).then_some((p, self.in_think))
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct GenOutput {
    pub content: String,
    pub reasoning: String,
    pub logprobs: Vec<f32>,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub reasoning_tokens: u32,
    pub finish_reason: String,
    pub ttft_ms: Option<u64>,
    pub wall_ms: u64,
    /// Time spent waiting for a slot permit in the gateway before the request was sent.
    pub queue_ms: u64,
    /// llama-server's own timings (absent on servers that do not report them).
    pub server_prefill_ms: Option<f64>,
    pub server_decode_ms: Option<f64>,
    pub server_prompt_n: Option<u32>,
    pub server_cache_n: Option<u32>,
    /// Free text for the stage record (P7-T08 exit reason).
    pub note: Option<String>,
}
impl GenOutput {
    /// This call as a latency stage (P7-T01).
    pub fn stage(&self, name: &str) -> nafs_types::StageTiming {
        nafs_types::StageTiming {
            stage: name.into(),
            ms: self.wall_ms + self.queue_ms,
            queue_ms: Some(self.queue_ms),
            prefill_ms: self.server_prefill_ms,
            decode_ms: self.server_decode_ms,
            prompt_n: self.server_prompt_n,
            cache_n: self.server_cache_n,
            tokens: Some(self.completion_tokens),
            note: self.note.clone(),
        }
    }
    fn read_timings(&mut self, t: &Value) {
        if t.is_object() {
            self.server_prefill_ms = t["prompt_ms"].as_f64();
            self.server_decode_ms = t["predicted_ms"].as_f64();
            self.server_prompt_n = t["prompt_n"].as_u64().map(|x| x as u32);
            self.server_cache_n = t["cache_n"].as_u64().map(|x| x as u32);
        }
    }
    pub fn mean_logprob(&self) -> Option<f32> {
        if self.logprobs.is_empty() {
            None
        } else {
            Some(self.logprobs.iter().sum::<f32>() / self.logprobs.len() as f32)
        }
    }
    pub fn min_logprob(&self) -> Option<f32> {
        self.logprobs.iter().cloned().reduce(f32::min)
    }
}

/// Streamed piece of output: (text, is_reasoning).
pub type DeltaSink<'a> = &'a mut (dyn FnMut(&str, bool) + Send);

#[derive(Debug, Clone, Serialize)]
pub struct BackendHealth {
    pub role: String,
    pub url: String,
    pub model: String,
    pub hf_id: String,
    pub ok: bool,
    pub latency_ms: Option<u64>,
    pub served_models: Vec<String>,
    pub error: Option<String>,
    pub in_flight: usize,
    pub max_in_flight: usize,
}

pub struct Backend {
    pub role: String,
    pub kind: String,
    pub cfg: EndpointCfg,
    retry: RetryCfg,
    client: reqwest::Client,
    sem: Arc<Semaphore>,
}

impl Backend {
    pub fn new(role: &str, kind: &str, cfg: EndpointCfg, retry: RetryCfg) -> Self {
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_millis(retry.connect_timeout_ms))
            .pool_idle_timeout(Duration::from_secs(30))
            .build()
            .expect("http client");
        let sem = Arc::new(Semaphore::new(cfg.max_in_flight.max(1)));
        Self {
            role: role.into(),
            kind: kind.into(),
            cfg,
            retry,
            client,
            sem,
        }
    }

    fn body(&self, req: &GenRequest, stream: bool) -> Value {
        let mut b = json!({
            "model": self.cfg.model, "messages": req.messages, "max_tokens": req.max_tokens,
            "temperature": req.temperature, "top_p": req.top_p, "stream": stream,
        });
        if stream {
            b["stream_options"] = json!({"include_usage": true});
        }
        if req.logprobs {
            b["logprobs"] = json!(true);
            b["top_logprobs"] = json!(1);
        }
        if let Some(t) = req.thinking {
            b["chat_template_kwargs"] = json!({"enable_thinking": t});
        }
        if let Some(s) = req.seed {
            b["seed"] = json!(s);
        }
        if let Some(rf) = &req.response_format {
            b["response_format"] = rf.clone();
        }
        if let Some(budget) = req.thinking_budget {
            // Server-side budget forcing (Prop. 9): llama-server closes the reasoning span at the budget;
            // vLLM forces reasoning_end_str once thinking_token_budget is reached.
            if self.kind == "vllm" {
                b["thinking_token_budget"] = json!(budget);
            } else {
                b["thinking_budget_tokens"] = json!(budget);
            }
        }
        b
    }

    async fn acquire(&self) -> Result<tokio::sync::OwnedSemaphorePermit> {
        tokio::time::timeout(
            Duration::from_millis(self.retry.queue_timeout_ms),
            self.sem.clone().acquire_owned(),
        )
        .await
        .map_err(|_| NafsError::QueueTimeout(self.role.clone()))?
        .map_err(|e| NafsError::Internal(e.to_string()))
    }

    /// POST with retry on connection errors and HTTP 5xx only (IF-007); never on 4xx.
    async fn post(&self, body: &Value) -> Result<reqwest::Response> {
        let url = format!("{}/chat/completions", self.cfg.url.trim_end_matches('/'));
        let mut attempt = 0u32;
        loop {
            attempt += 1;
            let res = self.client.post(&url).json(body).send().await;
            match res {
                Ok(r) if r.status().is_success() => return Ok(r),
                Ok(r) if r.status().is_client_error() => {
                    let s = r.status();
                    let t = r.text().await.unwrap_or_default();
                    // A prompt longer than the model's context window is the caller's error, not a protocol
                    // fault: the character-based estimate of the prompt cap can undercount code and CJK text.
                    if t.contains("exceeds the available context size")
                        || t.contains("context_length_exceeded")
                    {
                        return Err(NafsError::BadRequest(
                            "the prompt is longer than the model's context window; shorten it or start a new conversation".into(),
                        ));
                    }
                    return Err(NafsError::BackendProtocol(format!(
                        "{} returned {}: {}",
                        self.role,
                        s,
                        t.chars().take(400).collect::<String>()
                    )));
                }
                Ok(r) => {
                    if attempt >= self.retry.max_attempts {
                        return Err(NafsError::BackendUnavailable(format!(
                            "{} returned {}",
                            self.role,
                            r.status()
                        )));
                    }
                }
                Err(e) => {
                    if attempt >= self.retry.max_attempts {
                        return Err(NafsError::BackendUnavailable(format!(
                            "{}: {}",
                            self.role, e
                        )));
                    }
                }
            }
            let cap = (self.retry.base_backoff_ms << (attempt - 1)).min(self.retry.max_backoff_ms);
            tokio::time::sleep(Duration::from_millis(fastrand::u64(0..=cap))).await;
        }
    }

    /// Streaming completion; `sink` receives (text, is_reasoning) deltas as they arrive. Cancellation
    /// is honoured while queued, while connecting and between stream events; dropping the response
    /// closes the connection, which stops generation on llama-server and vLLM.
    pub async fn stream(&self, req: &GenRequest, sink: DeltaSink<'_>) -> Result<GenOutput> {
        let tq = Instant::now();
        let _permit = cancellable(req, self.acquire()).await?;
        let queue_ms = tq.elapsed().as_millis() as u64;
        let _g = InflightGuard::new(&self.role);
        let t0 = Instant::now();
        let mut out = self.stream_inner(req, sink, t0).await?;
        out.queue_ms = queue_ms;
        Ok(out)
    }

    async fn stream_inner(
        &self,
        req: &GenRequest,
        sink: DeltaSink<'_>,
        t0: Instant,
    ) -> Result<GenOutput> {
        let resp = cancellable(req, self.post(&self.body(req, true))).await?;
        let mut events = resp.bytes_stream().eventsource();
        let mut out = GenOutput::default();
        let idle = Duration::from_millis(self.retry.stream_idle_timeout_ms);
        let mut split = ThinkSplitter::default();
        let mut window = req
            .early_stop
            .as_ref()
            .map(|e| crate::sequential::WindowTracker::new(e.window.max(1)));
        let mut last = Instant::now();
        let emit = |out: &mut GenOutput,
                    text: &str,
                    reasoning: bool,
                    sink: &mut (dyn FnMut(&str, bool) + Send)| {
            if text.is_empty() {
                return;
            }
            if out.ttft_ms.is_none() {
                out.ttft_ms = Some(t0.elapsed().as_millis() as u64);
            }
            if reasoning {
                out.reasoning.push_str(text);
            } else {
                out.content.push_str(text);
            }
            sink(text, reasoning);
        };
        loop {
            let next = tokio::select! {
                ev = events.next() => ev,
                _ = tokio::time::sleep(Duration::from_millis(250)) => {
                    if req.cancel.as_ref().map(|c| c.is_set()).unwrap_or(false) {
                        return Err(NafsError::Cancelled);
                    }
                    if last.elapsed() > idle {
                        return Err(NafsError::BackendTimeout(format!("{}: no data for {:?}", self.role, idle)));
                    }
                    continue;
                }
            };
            last = Instant::now();
            if req.cancel.as_ref().map(|c| c.is_set()).unwrap_or(false) {
                return Err(NafsError::Cancelled);
            }
            let Some(ev) = next else { break };
            let ev = ev
                .map_err(|e| NafsError::BackendProtocol(format!("{}: SSE error {e}", self.role)))?;
            if ev.data.trim() == "[DONE]" {
                break;
            }
            let v: Value = match serde_json::from_str(&ev.data) {
                Ok(v) => v,
                Err(_) => continue,
            };
            if let Some(t) = v.get("timings") {
                out.read_timings(t);
            }
            if let Some(u) = v.get("usage").filter(|u| !u.is_null()) {
                out.prompt_tokens = u["prompt_tokens"].as_u64().unwrap_or(0) as u32;
                out.completion_tokens = u["completion_tokens"].as_u64().unwrap_or(0) as u32;
            }
            let Some(choice) = v["choices"].get(0) else {
                continue;
            };
            let delta = &choice["delta"];
            let reasoning = delta
                .get("reasoning_content")
                .and_then(|x| x.as_str())
                .or_else(|| delta.get("reasoning").and_then(|x| x.as_str()))
                .unwrap_or("");
            if !reasoning.is_empty() {
                out.reasoning_tokens += 1;
                emit(&mut out, reasoning, true, sink);
            }
            if let Some(c) = delta.get("content").and_then(|x| x.as_str()) {
                let pieces = split.push(c);
                if split.in_think || pieces.iter().any(|(_, r)| *r) {
                    out.reasoning_tokens += 1;
                }
                for (t, r) in pieces {
                    emit(&mut out, &t, r, sink);
                }
            }
            let mut low_window: Option<f32> = None;
            if let Some(lp) = choice["logprobs"]["content"].as_array() {
                for t in lp {
                    if let Some(x) = t["logprob"].as_f64() {
                        if x.is_finite() {
                            out.logprobs.push(x as f32);
                            if let Some(w) = window.as_mut() {
                                low_window = w.push(x as f32);
                            }
                        }
                    }
                }
            }
            if let Some(f) = choice["finish_reason"].as_str() {
                out.finish_reason = f.to_string();
            }
            if let Some(e) = req.early_stop.as_ref() {
                let mean_hit = e.below_mean.is_some_and(|below| {
                    out.logprobs.len() >= e.min_n.max(1)
                        && out.mean_logprob().map(|m| m < below).unwrap_or(false)
                });
                let window_hit = match (e.below_window, low_window) {
                    (Some(thr), Some(v)) => v < thr,
                    _ => false,
                };
                if mean_hit || window_hit {
                    // Dropping the stream closes the connection, which stops generation on the server.
                    out.finish_reason = "early_escalation".into();
                    out.completion_tokens = out.logprobs.len() as u32;
                    out.note = Some(if window_hit {
                        format!(
                            "lowest {}-token window log-prob {:.2} < {:.2} (exact)",
                            e.window,
                            low_window.unwrap_or(f32::NAN),
                            e.below_window.unwrap_or(f32::NAN)
                        )
                    } else {
                        format!(
                            "running mean log-prob {:.2} < {:.2} after {} tokens",
                            out.mean_logprob().unwrap_or(f32::NAN),
                            e.below_mean.unwrap_or(f32::NAN),
                            out.logprobs.len()
                        )
                    });
                    break;
                }
            }
        }
        if let Some((t, r)) = split.flush() {
            emit(&mut out, &t, r, sink);
        }
        finish(&mut out, t0);
        Ok(out)
    }

    /// Non-streaming completion (used for the gate's cheap samples).
    pub async fn complete(&self, req: &GenRequest) -> Result<GenOutput> {
        let tq = Instant::now();
        let _permit = cancellable(req, self.acquire()).await?;
        let queue_ms = tq.elapsed().as_millis() as u64;
        let _g = InflightGuard::new(&self.role);
        let t0 = Instant::now();
        // Total timeout (review L5): a server that accepts but never answers must not wedge the caller.
        let total = Duration::from_millis(self.retry.stream_idle_timeout_ms);
        let call = async {
            let resp = self.post(&self.body(req, false)).await?;
            resp.json::<Value>()
                .await
                .map_err(|e| NafsError::BackendProtocol(format!("{}: {e}", self.role)))
        };
        let v: Value = cancellable(req, async {
            tokio::time::timeout(total, call).await.map_err(|_| {
                NafsError::BackendTimeout(format!("{}: no reply within {:?}", self.role, total))
            })?
        })
        .await?;
        let choice = &v["choices"][0];
        let msg = &choice["message"];
        let mut out = GenOutput {
            content: msg["content"].as_str().unwrap_or("").to_string(),
            reasoning: msg
                .get("reasoning_content")
                .and_then(|x| x.as_str())
                .or_else(|| msg.get("reasoning").and_then(|x| x.as_str()))
                .unwrap_or("")
                .to_string(),
            finish_reason: choice["finish_reason"].as_str().unwrap_or("").to_string(),
            prompt_tokens: v["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as u32,
            completion_tokens: v["usage"]["completion_tokens"].as_u64().unwrap_or(0) as u32,
            ..Default::default()
        };
        out.queue_ms = queue_ms;
        out.read_timings(&v["timings"]);
        if let Some(lp) = choice["logprobs"]["content"].as_array() {
            for t in lp {
                if let Some(x) = t["logprob"].as_f64() {
                    out.logprobs.push(x as f32);
                }
            }
        }
        if !out.reasoning.is_empty() {
            // Non-streaming replies carry no per-token count for reasoning: estimate ~4 chars/token.
            out.reasoning_tokens = (out.reasoning.len() as f64 / 4.0).ceil() as u32;
        }
        finish(&mut out, t0);
        Ok(out)
    }

    fn root_url(&self) -> String {
        self.cfg
            .url
            .trim_end_matches('/')
            .trim_end_matches("/v1")
            .to_string()
    }

    /// Raw continuation with a cached prefix is a llama-server feature (`/completion`, `/apply-template`).
    pub fn supports_raw(&self) -> bool {
        self.kind == "llama"
    }

    /// Render the chat template for `messages` (the server's own template, thinking enabled).
    pub async fn render_prompt(&self, messages: &[Message]) -> Result<String> {
        let url = format!("{}/apply-template", self.root_url());
        let r = self
            .client
            .post(&url)
            .json(&json!({"messages": messages, "chat_template_kwargs": {"enable_thinking": true}}))
            .send()
            .await
            .map_err(|e| NafsError::BackendUnavailable(format!("{}: {e}", self.role)))?;
        if !r.status().is_success() {
            return Err(NafsError::BackendProtocol(format!(
                "{}: apply-template returned {}",
                self.role,
                r.status()
            )));
        }
        let v: Value = r
            .json()
            .await
            .map_err(|e| NafsError::BackendProtocol(format!("{}: {e}", self.role)))?;
        v["prompt"].as_str().map(|s| s.to_string()).ok_or_else(|| {
            NafsError::BackendProtocol(format!("{}: apply-template gave no prompt", self.role))
        })
    }

    /// Continue generation from a raw prompt (rendered template plus assistant prefix) on the server's
    /// completion endpoint. The server reuses the cached prefix, so a continuation costs only the new tokens.
    pub async fn complete_raw(
        &self,
        req: &RawRequest,
        sink: &mut (dyn FnMut(&str) + Send),
    ) -> Result<RawOutput> {
        let tq = Instant::now();
        let _permit = with_cancel(req.cancel.as_ref(), self.acquire()).await?;
        let queue_ms = tq.elapsed().as_millis() as u64;
        let _g = InflightGuard::new(&self.role);
        let t0 = Instant::now();
        let body = json!({"prompt": req.prompt, "n_predict": req.n_predict, "temperature": req.temperature, "top_p": req.top_p,
            "seed": req.seed, "stop": req.stop, "stream": true, "cache_prompt": true});
        let url = format!("{}/completion", self.root_url());
        let resp = with_cancel(req.cancel.as_ref(), async {
            let r = self
                .client
                .post(&url)
                .json(&body)
                .send()
                .await
                .map_err(|e| NafsError::BackendUnavailable(format!("{}: {e}", self.role)))?;
            if !r.status().is_success() {
                return Err(NafsError::BackendProtocol(format!(
                    "{}: completion returned {}",
                    self.role,
                    r.status()
                )));
            }
            Ok(r)
        })
        .await?;
        let mut events = resp.bytes_stream().eventsource();
        let mut out = RawOutput {
            queue_ms,
            ..Default::default()
        };
        let idle = Duration::from_millis(self.retry.stream_idle_timeout_ms);
        let mut last = Instant::now();
        let cancelled = || req.cancel.as_ref().map(|c| c.is_set()).unwrap_or(false);
        loop {
            let next = tokio::select! {
                ev = events.next() => ev,
                _ = tokio::time::sleep(Duration::from_millis(250)) => {
                    if cancelled() {
                        return Err(NafsError::Cancelled);
                    }
                    if last.elapsed() > idle {
                        return Err(NafsError::BackendTimeout(format!("{}: no data for {:?}", self.role, idle)));
                    }
                    continue;
                }
            };
            last = Instant::now();
            if cancelled() {
                return Err(NafsError::Cancelled);
            }
            let Some(ev) = next else { break };
            let ev = ev
                .map_err(|e| NafsError::BackendProtocol(format!("{}: SSE error {e}", self.role)))?;
            if ev.data.trim() == "[DONE]" {
                break;
            }
            let v: Value = match serde_json::from_str(&ev.data) {
                Ok(v) => v,
                Err(_) => continue,
            };
            if let Some(c) = v["content"].as_str() {
                if !c.is_empty() {
                    if out.ttft_ms.is_none() {
                        out.ttft_ms = Some(t0.elapsed().as_millis() as u64);
                    }
                    out.text.push_str(c);
                    sink(c);
                }
            }
            if v["stop"].as_bool() == Some(true) {
                out.stop_type = v["stop_type"].as_str().unwrap_or("").to_string();
                out.stopping_word = v["stopping_word"].as_str().unwrap_or("").to_string();
                out.tokens = v["tokens_predicted"].as_u64().unwrap_or(0) as u32;
                let t = &v["timings"];
                out.prefill_ms = t["prompt_ms"].as_f64();
                out.decode_ms = t["predicted_ms"].as_f64();
                out.prompt_n = t["prompt_n"].as_u64().map(|x| x as u32);
                out.cache_n = t["cache_n"].as_u64().map(|x| x as u32);
            }
        }
        out.wall_ms = t0.elapsed().as_millis() as u64;
        Ok(out)
    }

    pub async fn health(&self) -> BackendHealth {
        let url = format!("{}/models", self.cfg.url.trim_end_matches('/'));
        let t0 = Instant::now();
        let (ok, served, error) = match self
            .client
            .get(&url)
            .timeout(Duration::from_secs(3))
            .send()
            .await
        {
            Ok(r) if r.status().is_success() => {
                let v: Value = r.json().await.unwrap_or(Value::Null);
                let ids = v["data"]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|m| m["id"].as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default();
                (true, ids, None)
            }
            Ok(r) => (false, vec![], Some(format!("HTTP {}", r.status()))),
            Err(e) => (
                false,
                vec![],
                Some(if e.is_connect() {
                    "not reachable".to_string()
                } else {
                    e.to_string()
                }),
            ),
        };
        let max = self.cfg.max_in_flight.max(1);
        BackendHealth {
            role: self.role.clone(),
            url: self.cfg.url.clone(),
            model: self.cfg.model.clone(),
            hf_id: self.cfg.hf_id.clone(),
            ok,
            latency_ms: if ok {
                Some(t0.elapsed().as_millis() as u64)
            } else {
                None
            },
            served_models: served,
            error,
            in_flight: max - self.sem.available_permits(),
            max_in_flight: max,
        }
    }
}

fn finish(out: &mut GenOutput, t0: Instant) {
    out.wall_ms = t0.elapsed().as_millis() as u64;
    // Some servers inline <think>..</think> in content when no parser is configured: split it out.
    if out.reasoning.is_empty() {
        if let Some(end) = out.content.find("</think>") {
            let r = out.content[..end].replace("<think>", "").trim().to_string();
            out.content = out.content[end + 8..].trim_start().to_string();
            out.reasoning_tokens = (r.len() as f64 / 4.0).ceil() as u32;
            out.reasoning = r;
        }
    }
    if out.completion_tokens > 0 && out.reasoning_tokens > out.completion_tokens {
        out.reasoning_tokens = out.completion_tokens;
    }
    out.content = out.content.trim().to_string();
}

#[cfg(test)]
mod tests {
    use super::ThinkSplitter;
    fn run(chunks: &[&str]) -> (String, String) {
        let mut s = ThinkSplitter::default();
        let (mut a, mut r) = (String::new(), String::new());
        for c in chunks {
            for (t, is_r) in s.push(c) {
                if is_r {
                    r.push_str(&t)
                } else {
                    a.push_str(&t)
                }
            }
        }
        if let Some((t, is_r)) = s.flush() {
            if is_r {
                r.push_str(&t)
            } else {
                a.push_str(&t)
            }
        }
        (a, r)
    }
    #[test]
    fn inline_think_is_routed_as_reasoning_even_when_tags_are_split() {
        let (a, r) = run(&["<thi", "nk>let me ", "think</th", "ink>Answer: 4"]);
        assert_eq!(r, "let me think");
        assert_eq!(a, "Answer: 4");
        let (a, r) = run(&["plain ", "answer <b>"]);
        assert_eq!((a.as_str(), r.as_str()), ("plain answer <b>", ""));
    }
}
