//! Rooms (IF-004): shared chats over WebSocket, with live presence, a short queue of assistant turns, and a
//! REST surface to create, list, rename and delete them.
//!
//! Frames the server sends (`{type, user?, payload}`), all JSON text:
//! * `welcome`  (to the new member only, always first): the name assigned to you, who is here, the
//!   assistant's state, the limits of the room, and the snapshot to build the transcript from: `cursor`
//!   (how many stored messages precede the turn that is running now) and `pending` (the questions of the
//!   running and queued turns). Render the first `cursor` stored messages, then `pending`, then live frames:
//!   every turn appears exactly once, however late you joined or reconnected.
//! * `presence` (everyone): someone joined or left; `members` is the count, `users` the names.
//! * `user_message` (everyone, the sender included): `{content, id, ts}`.
//! * `assistant_event` (everyone): one event of the assistant's answer to `reply_to`, as on the chat stream.
//! * `queue`    (everyone): `{busy, working_for, queued}` whenever the assistant starts, finishes or queues.
//! * `typing`   (everyone): someone is typing (at most one frame per two seconds per member).
//! * `room_renamed` (everyone): `{name}`.
//! * `room_deleted` (everyone, then the socket is closed with code 4004).
//! * `heartbeat` (each member, every 25 s): liveness for clients and proxies; `pong` answers a `ping`.
//! * `error` / `resync` (to one member): rate limits, a full queue, an over-long message, or missed frames.
//!
//! Frames a client may send: `user_message` (`content`, optional `force_mode`, `lambda`, `show_reasoning`),
//! `typing`, `ping`.
use crate::api::{ApiResult, AppState};
use axum::extract::ws::{CloseFrame, Message as WsMessage, WebSocket, WebSocketUpgrade};
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::Response;
use axum::Json;
use futures::{SinkExt, StreamExt};
use nafs_core::orchestrator::Emitter;
use nafs_types::{ChatRequest, Message, NafsError, ReasoningMode, StreamEvent};
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, mpsc};
use uuid::Uuid;

/// Assistant turns that may wait behind the current one in a room (review M7).
pub const ROOM_QUEUE_CAP: usize = 3;
/// Members one room accepts at a time.
pub const ROOM_MEMBER_CAP: usize = 64;
/// Longest message, in characters; longer ones are refused, not truncated.
pub const MESSAGE_MAX_CHARS: usize = 8000;
/// Longest display name, in characters.
pub const USER_NAME_MAX: usize = 32;
/// Per-connection message bucket: burst and refill interval (review M7, NFR-011).
const WS_BURST: f64 = 4.0;
const WS_REFILL_S: f64 = 6.0;
const HEARTBEAT_S: u64 = 25;
const TYPING_GAP: Duration = Duration::from_millis(2000);
/// How long a deletion waits for the assistant's running turn to stop before it purges anyway.
const DELETE_WAIT: Duration = Duration::from_secs(10);
/// How long a deleted room's hub is kept as a tombstone.
const TOMBSTONE_TTL: Duration = Duration::from_secs(60);
/// Names a member may not take, so nobody can pose as the assistant.
const RESERVED_NAMES: [&str; 4] = ["assistant", "nafs-aql", "nafs aql", "system"];
/// Sent to every member when the room is deleted; the forwarding task closes the socket after it.
const ROOM_DELETED_FRAME: &str = r#"{"type":"room_deleted","payload":{}}"#;
const CLOSE_ROOM_DELETED: u16 = 4004;

/// A question that has been asked but whose turn has not finished.
#[derive(Clone)]
struct PendingTurn {
    id: String,
    user: String,
    content: String,
    ts: String,
}

/// The part of a room's state a joining member needs a consistent snapshot of.
#[derive(Default)]
struct LiveState {
    /// Questions asked but not yet answered, oldest first (the running one first).
    pending: Vec<PendingTurn>,
    /// Stored messages before the running turn started; `None` when no turn is running.
    base: Option<i64>,
}

pub struct RoomHub {
    tx: broadcast::Sender<String>,
    live: Mutex<LiveState>,
    users: Mutex<Vec<(u64, String)>>,
    next_conn: AtomicU64,
    queued: AtomicUsize,
    busy: Arc<tokio::sync::Mutex<()>>,
    working_for: Mutex<Option<String>>,
    /// Set when the room is deleted. The hub stays in the map as a tombstone, so a connection or turn that
    /// races the deletion finds the flag instead of a fresh hub (and cannot recreate the conversation).
    deleted: AtomicBool,
    /// Shared by the room's running turn; set on deletion so the model stops at once.
    cancel: Arc<AtomicBool>,
}

impl RoomHub {
    fn new() -> Self {
        Self {
            tx: broadcast::channel(4096).0,
            live: Mutex::new(LiveState::default()),
            users: Mutex::new(Vec::new()),
            next_conn: AtomicU64::new(1),
            queued: AtomicUsize::new(0),
            busy: Arc::new(tokio::sync::Mutex::new(())),
            working_for: Mutex::new(None),
            deleted: AtomicBool::new(false),
            cancel: Arc::new(AtomicBool::new(false)),
        }
    }
    fn names(&self) -> Vec<String> {
        self.users
            .lock()
            .unwrap()
            .iter()
            .map(|(_, n)| n.clone())
            .collect()
    }
    /// Register a member under a name that is unique in the room; `None` if the room is full.
    fn join(&self, wanted: &str) -> Option<(u64, String, Vec<String>)> {
        let mut u = self.users.lock().unwrap();
        if u.len() >= ROOM_MEMBER_CAP {
            return None;
        }
        let mut taken: Vec<String> = u.iter().map(|(_, n)| n.clone()).collect();
        taken.extend(RESERVED_NAMES.iter().map(|n| n.to_string()));
        let name = unique_name(&taken, wanted);
        let id = self.next_conn.fetch_add(1, Ordering::SeqCst);
        u.push((id, name.clone()));
        let all = u.iter().map(|(_, n)| n.clone()).collect();
        Some((id, name, all))
    }
    fn leave(&self, conn: u64) -> Vec<String> {
        let mut u = self.users.lock().unwrap();
        u.retain(|(id, _)| *id != conn);
        u.iter().map(|(_, n)| n.clone()).collect()
    }
    fn assistant(&self) -> Value {
        let w = self.working_for.lock().unwrap().clone();
        json!({"busy": w.is_some(), "working_for": w, "queued": self.queued.load(Ordering::SeqCst)})
    }
    fn send(&self, v: Value) {
        // No receivers just means nobody is connected.
        let _ = self.tx.send(v.to_string());
    }
    fn send_queue(&self) {
        self.send(json!({"type": "queue", "payload": self.assistant()}));
    }
    /// Record a question and tell the room about it, as one step: a member who joins concurrently either
    /// receives this frame or finds the question in its snapshot, never both and never neither.
    fn enqueue(&self, p: PendingTurn) {
        let mut l = self.live.lock().unwrap();
        let frame = json!({"type": "user_message", "user": p.user,
            "payload": {"content": p.content, "id": p.id, "ts": p.ts}});
        l.pending.push(p);
        self.send(frame);
    }
    /// The running turn begins; `base` is the number of messages stored before it.
    fn start_turn(&self, base: i64) {
        self.live.lock().unwrap().base = Some(base);
    }
    /// A turn is over (answered, failed or skipped): drop its question from the snapshot and, in the same
    /// step, send the closing frame, so the snapshot and the stream hand over cleanly.
    fn finish_turn(&self, id: &str, closing: Option<String>) {
        let mut l = self.live.lock().unwrap();
        l.pending.retain(|p| p.id != id);
        l.base = None;
        if let Some(f) = closing {
            let _ = self.tx.send(f);
        }
    }
    /// Subscribe and snapshot together. `count` reads the stored message count; it runs under the lock, so the
    /// cursor and the pending list agree about which turns are stored and which are still live.
    fn subscribe(
        &self,
        count: impl FnOnce() -> i64,
    ) -> (broadcast::Receiver<String>, Vec<PendingTurn>, i64) {
        let l = self.live.lock().unwrap();
        let rx = self.tx.subscribe();
        let cursor = l.base.unwrap_or_else(count);
        (rx, l.pending.clone(), cursor)
    }
    fn is_deleted(&self) -> bool {
        self.deleted.load(Ordering::SeqCst)
    }
}

/// A display name: letters, digits, spaces, `-` and `_`; runs of spaces collapsed; at most `USER_NAME_MAX`
/// characters; `guest` when nothing usable is left.
pub fn sanitize_user(raw: &str) -> String {
    let kept: String = raw
        .chars()
        .filter(|c| c.is_alphanumeric() || matches!(c, '_' | '-' | ' '))
        .collect();
    let joined = kept.split_whitespace().collect::<Vec<_>>().join(" ");
    let cut: String = joined.chars().take(USER_NAME_MAX).collect();
    let cut = cut.trim().to_string();
    if cut.is_empty() {
        "guest".into()
    } else {
        cut
    }
}

/// `wanted` if no one in `taken` has it (ignoring case), else `wanted 2`, `wanted 3`, ... within the length limit.
pub fn unique_name(taken: &[String], wanted: &str) -> String {
    let lower: Vec<String> = taken.iter().map(|n| n.to_lowercase()).collect();
    if !lower.contains(&wanted.to_lowercase()) {
        return wanted.to_string();
    }
    (2u32..)
        .map(|k| {
            let suffix = format!(" {k}");
            let base: String = wanted
                .chars()
                .take(USER_NAME_MAX - suffix.chars().count())
                .collect();
            format!("{}{suffix}", base.trim_end())
        })
        .find(|c| !lower.contains(&c.to_lowercase()))
        .expect("an unused suffix exists")
}

pub(crate) fn hub(s: &AppState, id: Uuid) -> Arc<RoomHub> {
    s.rooms
        .lock()
        .unwrap()
        .entry(id)
        .or_insert_with(|| Arc::new(RoomHub::new()))
        .clone()
}

/// A room as the API shows it: the stored fields plus who is here now and what the assistant is doing.
fn room_json(s: &AppState, room: &nafs_memory::Room) -> Value {
    let h = s.rooms.lock().unwrap().get(&room.room_id).cloned();
    let users = h.as_ref().map(|h| h.names()).unwrap_or_default();
    let a = h
        .as_ref()
        .map(|h| h.assistant())
        .unwrap_or_else(|| json!({"busy": false, "working_for": null, "queued": 0}));
    let mut v = json!(room);
    v["members"] = json!(users.len());
    v["users"] = json!(users);
    v["busy"] = a["busy"].clone();
    v["queued"] = a["queued"].clone();
    v
}

// ---------------- REST ----------------
pub async fn list_rooms(State(s): State<AppState>) -> ApiResult<Json<Value>> {
    let rooms = s.store.list_rooms()?;
    Ok(Json(json!(rooms
        .iter()
        .map(|r| room_json(&s, r))
        .collect::<Vec<_>>())))
}

pub async fn get_room(State(s): State<AppState>, Path(id): Path<Uuid>) -> ApiResult<Json<Value>> {
    Ok(Json(room_json(&s, &s.store.get_room(id)?)))
}

#[derive(Deserialize)]
pub struct NewRoom {
    name: String,
    #[serde(default)]
    profile: Option<String>,
}

pub async fn create_room(
    State(s): State<AppState>,
    Json(b): Json<NewRoom>,
) -> ApiResult<(StatusCode, Json<Value>)> {
    let profile: String = b
        .profile
        .unwrap_or_else(|| s.cfg.assistant.profile.clone())
        .chars()
        .filter(|c| c.is_alphanumeric())
        .take(8)
        .collect();
    let room = s.store.create_room(&b.name, &profile)?;
    Ok((StatusCode::CREATED, Json(room_json(&s, &room))))
}

#[derive(Deserialize)]
pub struct RenameRoom {
    name: String,
}

pub async fn rename_room(
    State(s): State<AppState>,
    Path(id): Path<Uuid>,
    Json(b): Json<RenameRoom>,
) -> ApiResult<Json<Value>> {
    let room = s.store.rename_room(id, &b.name)?;
    hub(&s, id).send(json!({"type": "room_renamed", "payload": {"name": room.name}}));
    Ok(Json(room_json(&s, &room)))
}

/// Delete a room and everything stored from it. Members are told first and disconnected; a running answer is
/// cancelled and given a few seconds to stop, so it cannot write into the conversation as it disappears.
pub async fn delete_room(
    State(s): State<AppState>,
    Path(id): Path<Uuid>,
) -> ApiResult<Json<Value>> {
    let room = s.store.get_room(id)?;
    let h = hub(&s, id);
    h.deleted.store(true, Ordering::SeqCst);
    h.cancel.store(true, Ordering::SeqCst);
    let members = h.names().len();
    let _ = h.tx.send(ROOM_DELETED_FRAME.to_string());
    // Holding the turn lock keeps a queued turn from starting while the rows go.
    let guard = tokio::time::timeout(DELETE_WAIT, h.busy.clone().lock_owned())
        .await
        .ok();
    let report = s.store.delete_room(id);
    drop(guard);
    let report = report?;
    tracing::info!(room = %id, name = %room.name, members, messages = report.messages, replies = report.replies, "room deleted");
    // The tombstone only has to outlive a connection or turn that raced the deletion. After that the database
    // answers 404 before a hub could be recreated, so the hub (and its broadcast buffer) can go.
    let sweep = s.clone();
    tokio::spawn(async move {
        tokio::time::sleep(TOMBSTONE_TTL).await;
        sweep.rooms.lock().unwrap().remove(&id);
    });
    Ok(Json(json!({
        "deleted": true,
        "room_id": id,
        "name": room.name,
        "disconnected": members,
        "removed": {"messages": report.messages, "replies": report.replies, "memory_entries": report.memory_entries},
    })))
}

// ---------------- WebSocket ----------------
#[derive(Deserialize)]
pub struct WsQuery {
    #[serde(default)]
    user: Option<String>,
}

pub async fn room_socket(
    State(s): State<AppState>,
    Path(id): Path<Uuid>,
    Query(q): Query<WsQuery>,
    ws: WebSocketUpgrade,
) -> ApiResult<Response> {
    let room = s.store.get_room(id)?;
    let h = hub(&s, id);
    if h.is_deleted() {
        return Err(NafsError::NotFound(format!("room {id}")).into());
    }
    if h.names().len() >= ROOM_MEMBER_CAP {
        return Err(NafsError::BadRequest(format!(
            "this room is full ({ROOM_MEMBER_CAP} members)"
        ))
        .into());
    }
    let user = sanitize_user(q.user.as_deref().unwrap_or(""));
    Ok(ws
        .max_message_size(64 * 1024)
        .on_upgrade(move |socket| room_session(s, room, user, socket)))
}

fn text(v: Value) -> WsMessage {
    WsMessage::Text(v.to_string().into())
}

fn private_err(code: &str, message: &str) -> String {
    json!({"type": "error", "payload": {"code": code, "message": message}}).to_string()
}

async fn room_session(s: AppState, room: nafs_memory::Room, wanted: String, socket: WebSocket) {
    let h = hub(&s, room.room_id);
    let (mut sink, mut stream) = socket.split();
    // Subscribe and snapshot before announcing, so this member also receives concurrent frames and its own
    // presence, and so its transcript is built from exactly the turns it will not receive live.
    let conv = room.conversation_id;
    let (mut rx, pending, cursor) = h.subscribe(|| s.store.message_count(conv).unwrap_or(0));
    let Some((conn, user, users)) = h.join(&wanted) else {
        let _ = sink
            .send(WsMessage::Text(
                private_err("E_ROOM_FULL", "This room is full.").into(),
            ))
            .await;
        return;
    };
    let welcome = json!({"type": "welcome", "payload": {
        "you": user,
        "room": {"room_id": room.room_id, "name": room.name},
        "users": users,
        "assistant": h.assistant(),
        "limits": {"message_chars": MESSAGE_MAX_CHARS, "burst": WS_BURST, "queue": ROOM_QUEUE_CAP},
        "cursor": cursor,
        "pending": pending.iter().map(|p| json!({"id": p.id, "user": p.user, "content": p.content, "ts": p.ts})).collect::<Vec<_>>(),
    }});
    // Sent straight on the socket before anything else can be queued behind it.
    if sink.send(text(welcome)).await.is_err() {
        h.leave(conn);
        return;
    }
    // Frames meant only for this connection (rate-limit and queue errors, pong).
    let (ptx, mut prx) = mpsc::unbounded_channel::<String>();
    let mut fwd = tokio::spawn(async move {
        let mut beat = tokio::time::interval_at(
            tokio::time::Instant::now() + Duration::from_secs(HEARTBEAT_S),
            Duration::from_secs(HEARTBEAT_S),
        );
        loop {
            let msg = tokio::select! {
                r = rx.recv() => match r {
                    Ok(m) => m,
                    // A slow member skips frames instead of silently stopping (review M6).
                    Err(broadcast::error::RecvError::Lagged(k)) => json!({"type": "resync", "payload": {"missed": k}}).to_string(),
                    Err(broadcast::error::RecvError::Closed) => break,
                },
                p = prx.recv() => match p { Some(m) => m, None => break },
                _ = beat.tick() => json!({"type": "heartbeat"}).to_string(),
            };
            let last = msg == ROOM_DELETED_FRAME;
            if sink.send(WsMessage::Text(msg.into())).await.is_err() {
                break;
            }
            if last {
                let _ = sink
                    .send(WsMessage::Close(Some(CloseFrame {
                        code: CLOSE_ROOM_DELETED,
                        reason: "room deleted".into(),
                    })))
                    .await;
                break;
            }
        }
    });
    h.send(json!({"type": "presence", "user": user, "payload": {"joined": true, "members": users.len(), "users": users}}));
    let alive = Arc::new(AtomicBool::new(true));
    let (mut tokens, mut last) = (WS_BURST, Instant::now());
    let mut last_typing: Option<Instant> = None;
    loop {
        let next = tokio::select! { m = stream.next() => m, _ = &mut fwd => break };
        let Some(Ok(msg)) = next else { break };
        let WsMessage::Text(raw) = msg else { continue };
        let Ok(frame) = serde_json::from_str::<Value>(raw.as_str()) else {
            continue;
        };
        match frame["type"].as_str() {
            Some("ping") => {
                let _ = ptx.send(json!({"type": "pong"}).to_string());
                continue;
            }
            Some("typing") => {
                if last_typing.is_none_or(|t| t.elapsed() >= TYPING_GAP) {
                    last_typing = Some(Instant::now());
                    h.send(json!({"type": "typing", "user": user, "payload": {}}));
                }
                continue;
            }
            Some("user_message") => {}
            _ => continue,
        }
        let content = frame["payload"]["content"]
            .as_str()
            .unwrap_or("")
            .trim()
            .to_string();
        if content.is_empty() {
            continue;
        }
        let chars = content.chars().count();
        if chars > MESSAGE_MAX_CHARS {
            let _ = ptx.send(private_err(
                "E_TOO_LONG",
                &format!(
                    "Messages are limited to {MESSAGE_MAX_CHARS} characters; yours has {chars}."
                ),
            ));
            continue;
        }
        if h.is_deleted() {
            break;
        }
        // Per-connection token bucket (review M7).
        tokens = (tokens + last.elapsed().as_secs_f64() / WS_REFILL_S).min(WS_BURST);
        last = Instant::now();
        if tokens < 1.0 {
            let _ = ptx.send(private_err(
                "E_RATE_LIMITED",
                "You are sending messages too quickly; wait a few seconds.",
            ));
            continue;
        }
        if h.queued.load(Ordering::SeqCst) >= ROOM_QUEUE_CAP {
            let _ = ptx.send(private_err("E_QUEUE_FULL", "The assistant already has several questions queued in this room; try again shortly."));
            continue;
        }
        tokens -= 1.0;
        let msg_id = Uuid::new_v4().to_string();
        h.queued.fetch_add(1, Ordering::SeqCst);
        h.enqueue(PendingTurn {
            id: msg_id.clone(),
            user: user.clone(),
            content: content.clone(),
            ts: chrono::Utc::now().to_rfc3339(),
        });
        h.send_queue();
        let opts = frame["payload"].clone();
        tokio::spawn(run_turn(
            s.clone(),
            h.clone(),
            room.conversation_id,
            user.clone(),
            content,
            msg_id,
            opts,
            alive.clone(),
        ));
    }
    alive.store(false, Ordering::SeqCst);
    fwd.abort();
    let left = h.leave(conn);
    if !h.is_deleted() {
        h.send(json!({"type": "presence", "user": user, "payload": {"joined": false, "members": left.len(), "users": left}}));
    }
}

/// One assistant turn: wait for the room's turn, answer, and broadcast every event of the answer.
#[allow(clippy::too_many_arguments)]
async fn run_turn(
    s: AppState,
    h: Arc<RoomHub>,
    conv: Uuid,
    user: String,
    content: String,
    msg_id: String,
    opts: Value,
    alive: Arc<AtomicBool>,
) {
    let guard = h.busy.lock().await; // one assistant turn at a time per room
    h.queued.fetch_sub(1, Ordering::SeqCst);
    if h.is_deleted() {
        h.finish_turn(&msg_id, None);
        h.send_queue();
        return;
    }
    if !alive.load(Ordering::SeqCst) {
        // The sender left before their turn started: drop it.
        let ev = json!({"type": "assistant_event", "payload": {"reply_to": msg_id, "event": {"event": "error", "data": {"code": "E_CANCELLED", "message": "sender disconnected before the turn started"}}}});
        h.finish_turn(&msg_id, Some(ev.to_string()));
        h.send_queue();
        return;
    }
    *h.working_for.lock().unwrap() = Some(user.clone());
    h.start_turn(s.store.message_count(conv).unwrap_or(0));
    h.send_queue();
    let (tx, mut rx) = mpsc::unbounded_channel::<StreamEvent>();
    let (h3, reply_to) = (h.clone(), msg_id.clone());
    let fwd = tokio::spawn(async move {
        while let Some(ev) = rx.recv().await {
            let done = matches!(ev, StreamEvent::Done(_));
            let frame =
                json!({"type": "assistant_event", "payload": {"reply_to": reply_to, "event": ev}});
            if done {
                // The reply is stored before this event is emitted, so from here the transcript has the turn.
                h3.finish_turn(&reply_to, Some(frame.to_string()));
            } else {
                h3.send(frame);
            }
        }
    });
    let req = ChatRequest {
        conversation_id: Some(conv),
        messages: vec![Message {
            role: "user".into(),
            content: format!("[{user}] {content}"),
        }],
        lambda: opts["lambda"]
            .as_f64()
            .map(|x| x as f32)
            .filter(|l| (0.0..=1.0).contains(l)),
        force_mode: opts["force_mode"].as_str().and_then(|m| match m {
            "fast" => Some(ReasoningMode::Fast),
            "slow" => Some(ReasoningMode::Slow),
            _ => None,
        }),
        show_reasoning: Some(opts["show_reasoning"].as_bool().unwrap_or(true)),
        ..Default::default()
    };
    let inner = {
        let (s3, tx3, cancel) = (s.clone(), tx.clone(), h.cancel.clone());
        tokio::spawn(async move {
            crate::api::run_with_cap(&s3, req, Emitter(Some(tx3)), Some(cancel)).await
        })
    };
    match inner.await {
        Ok(Ok(_)) => {}
        Ok(Err(NafsError::Cancelled)) if h.is_deleted() => {}
        Ok(Err(e)) => {
            let _ = tx.send(StreamEvent::Error {
                code: e.code().into(),
                message: e.to_string(),
            });
        }
        Err(_) => {
            let _ = tx.send(StreamEvent::Error {
                code: "E_INTERNAL".into(),
                message: "internal error: the request task stopped unexpectedly".into(),
            });
        }
    }
    drop(tx);
    let _ = fwd.await;
    // Idempotent: a turn that ended without a done event (error, cancel) leaves the snapshot here.
    h.finish_turn(&msg_id, None);
    *h.working_for.lock().unwrap() = None;
    if h.is_deleted() {
        // The room went away while this turn ran: whatever it saved for the room goes too.
        let _ = s.store.purge_conversation_data(conv, true);
    }
    drop(guard);
    h.send_queue();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn display_names_are_cleaned() {
        assert_eq!(sanitize_user("  Aisha   B "), "Aisha B");
        // Markup characters are dropped, never escaped into the name.
        assert_eq!(sanitize_user("<b>Bob</b>"), "bBobb");
        assert_eq!(sanitize_user(""), "guest");
        assert_eq!(sanitize_user(" !!! "), "guest");
        assert_eq!(sanitize_user("Zoë 李_x-1"), "Zoë 李_x-1");
        assert_eq!(
            sanitize_user(&"n".repeat(100)).chars().count(),
            USER_NAME_MAX
        );
    }

    #[test]
    fn duplicate_names_get_a_number_without_case() {
        assert_eq!(unique_name(&names(&["Ann"]), "Bob"), "Bob");
        assert_eq!(unique_name(&names(&["Sam"]), "sam"), "sam 2");
        assert_eq!(unique_name(&names(&["Sam", "Sam 2"]), "Sam"), "Sam 3");
        // A full-length name still fits after the suffix.
        let long = "x".repeat(USER_NAME_MAX);
        let n = unique_name(std::slice::from_ref(&long), &long);
        assert!(
            n.chars().count() <= USER_NAME_MAX && n.ends_with(" 2"),
            "{n}"
        );
    }

    #[test]
    fn members_join_leave_and_hit_the_cap() {
        let h = RoomHub::new();
        let (a, name_a, all) = h.join("Sam").unwrap();
        assert_eq!((name_a.as_str(), all), ("Sam", names(&["Sam"])));
        let (b, name_b, _) = h.join("sam").unwrap();
        assert_eq!(name_b, "sam 2");
        assert_ne!(a, b);
        // Nobody may pose as the assistant.
        assert_eq!(h.join("Assistant").unwrap().1, "Assistant 2");
        assert_eq!(h.join("system").unwrap().1, "system 2");
        assert_eq!(h.leave(a), names(&["sam 2", "Assistant 2", "system 2"]));
        // The freed name is available again.
        assert_eq!(h.join("Sam").unwrap().1, "Sam");
        while h.names().len() < ROOM_MEMBER_CAP {
            assert!(h.join("guest").is_some());
        }
        assert!(h.join("late").is_none(), "room is full");
        assert_eq!(h.names().len(), ROOM_MEMBER_CAP);
    }

    #[test]
    fn the_assistant_state_follows_the_queue() {
        let h = RoomHub::new();
        assert_eq!(
            h.assistant(),
            json!({"busy": false, "working_for": null, "queued": 0})
        );
        h.queued.fetch_add(2, Ordering::SeqCst);
        *h.working_for.lock().unwrap() = Some("Ann".into());
        assert_eq!(
            h.assistant(),
            json!({"busy": true, "working_for": "Ann", "queued": 2})
        );
        let mut rx = h.tx.subscribe();
        h.send_queue();
        let f: Value = serde_json::from_str(&rx.try_recv().unwrap()).unwrap();
        assert_eq!(f["type"], "queue");
        assert_eq!(f["payload"]["queued"], 2);
    }

    fn turn(i: usize) -> PendingTurn {
        PendingTurn {
            id: format!("t{i}"),
            user: "Ann".into(),
            content: format!("q{i}"),
            ts: "t".into(),
        }
    }

    #[test]
    fn a_joiner_snapshot_hands_over_to_the_stream_in_order() {
        let h = RoomHub::new();
        // Nothing asked yet: the cursor is the stored count and nothing is pending.
        let (_rx0, p, c) = h.subscribe(|| 10);
        assert_eq!((p.len(), c), (0, 10));
        // A question is queued but not running: it is pending, the cursor is still the stored count.
        h.enqueue(turn(0));
        let (_rx1, p, c) = h.subscribe(|| 10);
        assert_eq!(
            (p.iter().map(|x| x.id.as_str()).collect::<Vec<_>>(), c),
            (vec!["t0"], 10)
        );
        // Running: the cursor stays at the count before the turn, even after the reply has been stored.
        h.start_turn(10);
        let (mut rx2, p, c) = h.subscribe(|| 12);
        assert_eq!(
            (p.len(), c),
            (1, 10),
            "stored but not finished: still the turn's base"
        );
        // Finishing hands over: the done frame goes to those already subscribed, and a new joiner finds
        // the turn in the stored messages and not in pending.
        h.finish_turn("t0", Some("DONE".into()));
        assert_eq!(rx2.try_recv().unwrap(), "DONE");
        let (_rx3, p, c) = h.subscribe(|| 12);
        assert_eq!((p.len(), c), (0, 12));
        // Finishing a turn that never ran (sender left, room deleted) only clears it.
        h.enqueue(turn(1));
        h.finish_turn("t1", None);
        assert_eq!(h.subscribe(|| 12).1.len(), 0);
    }

    #[test]
    fn every_turn_reaches_a_member_exactly_once_however_it_joins() {
        use std::collections::HashSet;
        use std::sync::atomic::{AtomicI64, AtomicUsize};
        const TURNS: usize = 300;
        // Joins the run must see, spread over it: turn i does not finish before (i + 1) * MIN_JOINS / TURNS
        // joins, so a fast producer cannot outrun the joiners.
        const MIN_JOINS: usize = 20;
        let h = Arc::new(RoomHub::new());
        // Two messages per turn, as the store keeps them.
        let stored = Arc::new(AtomicI64::new(0));
        // Members that join while the producer runs: each keeps its snapshot and its receiver.
        let joined = Arc::new(Mutex::new(Vec::new()));
        let joins = Arc::new(AtomicUsize::new(0));
        let producer = {
            let (h, stored, joins) = (h.clone(), stored.clone(), joins.clone());
            std::thread::spawn(move || {
                for i in 0..TURNS {
                    h.enqueue(turn(i));
                    std::thread::yield_now();
                    h.start_turn(stored.load(Ordering::SeqCst));
                    // Wait while the turn runs, before it is stored: the joiners keep going until the last
                    // turn is stored, so they are still joining while the producer waits here.
                    while joins.load(Ordering::SeqCst) < (i + 1) * MIN_JOINS / TURNS {
                        std::thread::yield_now();
                    }
                    stored.fetch_add(2, Ordering::SeqCst); // the orchestrator stores before it emits done
                    std::thread::yield_now();
                    h.finish_turn(&format!("t{i}"), Some(format!("DONE t{i}")));
                }
            })
        };
        let joiners: Vec<_> = (0..4)
            .map(|_| {
                let (h, stored, joined, joins) =
                    (h.clone(), stored.clone(), joined.clone(), joins.clone());
                std::thread::spawn(move || {
                    while stored.load(Ordering::SeqCst) < 2 * TURNS as i64 {
                        let snap = h.subscribe(|| stored.load(Ordering::SeqCst));
                        joined.lock().unwrap().push(snap);
                        joins.fetch_add(1, Ordering::SeqCst);
                        std::thread::sleep(std::time::Duration::from_micros(30));
                    }
                })
            })
            .collect();
        producer.join().unwrap();
        for j in joiners {
            j.join().unwrap();
        }
        let members = std::mem::take(&mut *joined.lock().unwrap());
        assert!(members.len() >= MIN_JOINS, "joiners ran: {}", members.len());
        for (mut rx, pending, cursor) in members {
            // The turns the member gets from the stored messages, from `pending`, and from live frames.
            let from_store: HashSet<usize> = (0..(cursor as usize / 2)).collect();
            let from_pending: HashSet<usize> =
                pending.iter().map(|p| p.id[1..].parse().unwrap()).collect();
            let mut live_questions = HashSet::new();
            let mut live_done = HashSet::new();
            while let Ok(f) = rx.try_recv() {
                if let Some(id) = f.strip_prefix("DONE t") {
                    live_done.insert(id.parse::<usize>().unwrap());
                } else {
                    let v: Value = serde_json::from_str(&f).unwrap();
                    if v["type"] == "user_message" {
                        live_questions.insert(
                            v["payload"]["id"].as_str().unwrap()[1..]
                                .parse::<usize>()
                                .unwrap(),
                        );
                    }
                }
            }
            // Every turn is delivered by exactly one route (stored, pending, or a live question) ...
            for i in 0..TURNS {
                let routes = from_store.contains(&i) as u8
                    + from_pending.contains(&i) as u8
                    + live_questions.contains(&i) as u8;
                assert_eq!(
                    routes,
                    1,
                    "turn {i}: stored={} pending={} live={}",
                    from_store.contains(&i),
                    from_pending.contains(&i),
                    live_questions.contains(&i)
                );
            }
            // ... and a pending turn's answer closes with a live done frame, which no stored turn ever repeats.
            assert!(from_pending.iter().all(|i| live_done.contains(i)));
            assert!(from_store.iter().all(|i| !live_done.contains(i)));
        }
    }

    #[test]
    fn the_deleted_frame_is_valid_json_the_session_recognises() {
        let v: Value = serde_json::from_str(ROOM_DELETED_FRAME).unwrap();
        assert_eq!(v["type"], "room_deleted");
        let h = RoomHub::new();
        let mut rx = h.tx.subscribe();
        let _ = h.tx.send(ROOM_DELETED_FRAME.to_string());
        assert_eq!(rx.try_recv().unwrap(), ROOM_DELETED_FRAME);
        assert!(!h.is_deleted());
        h.deleted.store(true, Ordering::SeqCst);
        assert!(h.is_deleted());
    }
}
