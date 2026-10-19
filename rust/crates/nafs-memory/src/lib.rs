//! nafs-memory: the Context Manager (FR-003, FR-060, IF-010).
use chrono::Utc;
use nafs_types::{ChatReply, DecisionRecord, Message, NafsError, Result, TraceRecord};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::io::Write;
use std::path::Path;
use std::sync::Mutex;
use uuid::Uuid;

fn st(e: impl std::fmt::Display) -> NafsError {
    NafsError::Storage(e.to_string())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationSummary {
    pub conversation_id: Uuid,
    pub title: String,
    pub room_id: Option<Uuid>,
    pub updated_at: String,
    pub messages: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredMessage {
    pub role: String,
    pub content: String,
    pub created_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reply: Option<Value>,
    /// Latest human rating of this reply (+1 / -1), if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub feedback: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    pub conversation_id: Uuid,
    pub title: String,
    pub room_id: Option<Uuid>,
    pub messages: Vec<StoredMessage>,
    #[serde(default)]
    pub training_opt_out: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Room {
    pub room_id: Uuid,
    pub name: String,
    pub profile: String,
    pub conversation_id: Uuid,
    pub created_at: String,
}

/// (mode, wall_ms, escalated, verifier passed, verifier exact) for the alert window.
pub type RecentRow = (String, i64, bool, Option<bool>, bool);
/// (reply_id, conversation_id, created_at, domain, question, answer, verifier)
type MemoryDoc = (
    String,
    String,
    String,
    Option<String>,
    String,
    String,
    Option<String>,
);

pub struct Store {
    conn: Mutex<Connection>,
    trace: Mutex<Option<std::fs::File>>,
}

impl Store {
    pub fn open(db_path: &Path, trace_path: &Path) -> Result<Self> {
        if let Some(p) = db_path.parent() {
            std::fs::create_dir_all(p).map_err(st)?;
        }
        if let Some(p) = trace_path.parent() {
            std::fs::create_dir_all(p).map_err(st)?;
        }
        let conn = Connection::open(db_path).map_err(st)?;
        conn.execute_batch(r#"
            PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;
            CREATE TABLE IF NOT EXISTS conversations (id TEXT PRIMARY KEY, title TEXT NOT NULL, room_id TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                role TEXT NOT NULL, content TEXT NOT NULL, reply_id TEXT, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS replies (id TEXT PRIMARY KEY, conversation_id TEXT, created_at TEXT NOT NULL, domain TEXT, mode TEXT, assurance TEXT,
                escalated INTEGER, wall_ms INTEGER, ttft_ms INTEGER, rule TEXT, backend TEXT, audit INTEGER, ephemeral INTEGER DEFAULT 0, json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS decisions (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS rooms (id TEXT PRIMARY KEY, name TEXT NOT NULL, profile TEXT NOT NULL, conversation_id TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, kind TEXT NOT NULL, created_at TEXT NOT NULL, status TEXT NOT NULL, json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS audits (id TEXT PRIMARY KEY, decision_id TEXT NOT NULL, created_at TEXT NOT NULL, status TEXT NOT NULL,
                domain TEXT NOT NULL, request_json TEXT NOT NULL, kept_answer TEXT NOT NULL, kept_assurance TEXT, e1_hat REAL, features_json TEXT,
                slow_answer TEXT, kept_correct INTEGER, slow_correct INTEGER, agreement REAL, judge_v REAL, wall_ms INTEGER, completed_at TEXT, error TEXT);
            CREATE INDEX IF NOT EXISTS idx_audits_status ON audits(status, created_at);
            CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id);
            CREATE INDEX IF NOT EXISTS idx_replies_created ON replies(created_at);
            CREATE TABLE IF NOT EXISTS feedback (id INTEGER PRIMARY KEY AUTOINCREMENT, reply_id TEXT NOT NULL, created_at TEXT NOT NULL, rating INTEGER NOT NULL, note TEXT);
            CREATE INDEX IF NOT EXISTS idx_feedback_reply ON feedback(reply_id);
            CREATE TABLE IF NOT EXISTS reviews (reply_id TEXT PRIMARY KEY, decision TEXT NOT NULL, reviewer TEXT NOT NULL, note TEXT, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS memory_index (reply_id TEXT PRIMARY KEY, conversation_id TEXT, created_at TEXT NOT NULL, domain TEXT, question TEXT NOT NULL, answer TEXT NOT NULL, verifier TEXT);
            CREATE TABLE IF NOT EXISTS preferences (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, domain TEXT, prompt TEXT NOT NULL, fast_reply_id TEXT NOT NULL, slow_reply_id TEXT NOT NULL,
                preferred TEXT NOT NULL, fast_wall_ms INTEGER, slow_wall_ms INTEGER, note TEXT);
        "#).map_err(st)?;
        // Migration: per-conversation training opt-out (SR-004).
        let has_opt: bool = conn
            .prepare(
                "SELECT 1 FROM pragma_table_info('conversations') WHERE name='training_opt_out'",
            )
            .and_then(|mut q| q.exists([]))
            .map_err(st)?;
        if !has_opt {
            conn.execute_batch(
                "ALTER TABLE conversations ADD COLUMN training_opt_out INTEGER NOT NULL DEFAULT 0;",
            )
            .map_err(st)?;
        }
        let trace = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(trace_path)
            .map_err(st)?;
        Ok(Self {
            conn: Mutex::new(conn),
            trace: Mutex::new(Some(trace)),
        })
    }

    pub fn ensure_conversation(
        &self,
        id: Option<Uuid>,
        title_hint: &str,
        room_id: Option<Uuid>,
    ) -> Result<Uuid> {
        let now = Utc::now().to_rfc3339();
        let c = self.conn.lock().unwrap();
        if let Some(id) = id {
            let exists: Option<String> = c
                .query_row(
                    "SELECT id FROM conversations WHERE id=?1",
                    params![id.to_string()],
                    |r| r.get(0),
                )
                .optional()
                .map_err(st)?;
            if exists.is_some() {
                return Ok(id);
            }
        }
        let id = id.unwrap_or_else(Uuid::new_v4);
        let title: String = title_hint
            .chars()
            .take(60)
            .collect::<String>()
            .trim()
            .to_string();
        c.execute("INSERT INTO conversations (id,title,room_id,created_at,updated_at) VALUES (?1,?2,?3,?4,?4)",
            params![id.to_string(), if title.is_empty() { "New conversation".to_string() } else { title }, room_id.map(|r| r.to_string()), now]).map_err(st)?;
        Ok(id)
    }

    pub fn append_message(
        &self,
        conv: Uuid,
        role: &str,
        content: &str,
        reply_id: Option<Uuid>,
    ) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        let c = self.conn.lock().unwrap();
        c.execute("INSERT INTO messages (conversation_id,role,content,reply_id,created_at) VALUES (?1,?2,?3,?4,?5)",
            params![conv.to_string(), role, content, reply_id.map(|r| r.to_string()), now]).map_err(st)?;
        c.execute(
            "UPDATE conversations SET updated_at=?2 WHERE id=?1",
            params![conv.to_string(), now],
        )
        .map_err(st)?;
        Ok(())
    }

    pub fn history(&self, conv: Uuid) -> Result<Vec<Message>> {
        let c = self.conn.lock().unwrap();
        let mut s = c
            .prepare("SELECT role, content FROM messages WHERE conversation_id=?1 ORDER BY id")
            .map_err(st)?;
        let rows = s
            .query_map(params![conv.to_string()], |r| {
                Ok(Message {
                    role: r.get(0)?,
                    content: r.get(1)?,
                })
            })
            .map_err(st)?;
        rows.collect::<std::result::Result<Vec<_>, _>>().map_err(st)
    }

    pub fn list_conversations(&self, limit: i64) -> Result<Vec<ConversationSummary>> {
        let c = self.conn.lock().unwrap();
        let mut s = c.prepare("SELECT c.id, c.title, c.room_id, c.updated_at, (SELECT COUNT(*) FROM messages m WHERE m.conversation_id=c.id) FROM conversations c WHERE c.room_id IS NULL ORDER BY c.updated_at DESC LIMIT ?1").map_err(st)?;
        let rows = s
            .query_map(params![limit], |r| {
                let id: String = r.get(0)?;
                let room: Option<String> = r.get(2)?;
                Ok(ConversationSummary {
                    conversation_id: Uuid::parse_str(&id).unwrap_or_default(),
                    title: r.get(1)?,
                    room_id: room.and_then(|x| Uuid::parse_str(&x).ok()),
                    updated_at: r.get(3)?,
                    messages: r.get(4)?,
                })
            })
            .map_err(st)?;
        rows.collect::<std::result::Result<Vec<_>, _>>().map_err(st)
    }

    pub fn get_conversation(&self, id: Uuid) -> Result<Conversation> {
        let c = self.conn.lock().unwrap();
        let (title, room, opt): (String, Option<String>, i64) = c
            .query_row(
                "SELECT title, room_id, training_opt_out FROM conversations WHERE id=?1",
                params![id.to_string()],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .optional()
            .map_err(st)?
            .ok_or_else(|| NafsError::NotFound(format!("conversation {id}")))?;
        let mut s = c.prepare("SELECT m.role, m.content, m.created_at, r.json, (SELECT f.rating FROM feedback f WHERE f.reply_id = m.reply_id ORDER BY f.id DESC LIMIT 1) FROM messages m LEFT JOIN replies r ON r.id = m.reply_id WHERE m.conversation_id=?1 ORDER BY m.id").map_err(st)?;
        let rows = s
            .query_map(params![id.to_string()], |r| {
                let j: Option<String> = r.get(3)?;
                Ok(StoredMessage {
                    role: r.get(0)?,
                    content: r.get(1)?,
                    created_at: r.get(2)?,
                    reply: j.and_then(|x| serde_json::from_str(&x).ok()),
                    feedback: r.get(4)?,
                })
            })
            .map_err(st)?;
        let messages = rows
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(st)?;
        Ok(Conversation {
            conversation_id: id,
            title,
            room_id: room.and_then(|x| Uuid::parse_str(&x).ok()),
            messages,
            training_opt_out: opt != 0,
        })
    }

    pub fn rename_conversation(&self, id: Uuid, title: &str) -> Result<()> {
        self.conn
            .lock()
            .unwrap()
            .execute(
                "UPDATE conversations SET title=?2 WHERE id=?1",
                params![id.to_string(), title],
            )
            .map_err(st)?;
        Ok(())
    }
    pub fn delete_conversation(&self, id: Uuid) -> Result<()> {
        self.conn
            .lock()
            .unwrap()
            .execute(
                "DELETE FROM conversations WHERE id=?1",
                params![id.to_string()],
            )
            .map_err(st)?;
        Ok(())
    }

    /// `ephemeral` column: 0 persisted, 1 ephemeral (API, Both trials), 2 lab traffic (evaluation, load tests).
    pub fn save_reply(&self, reply: &ChatReply, ephemeral: bool, lab: bool) -> Result<()> {
        let tag: i64 = if lab { 2 } else { ephemeral as i64 };
        let json = serde_json::to_string(reply).map_err(st)?;
        let c = self.conn.lock().unwrap();
        c.execute("INSERT OR REPLACE INTO replies (id,conversation_id,created_at,domain,mode,assurance,escalated,wall_ms,ttft_ms,rule,backend,audit,ephemeral,json) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14)",
            params![reply.reply_id.to_string(), reply.conversation_id.map(|x| x.to_string()), Utc::now().to_rfc3339(), reply.domain.as_str(),
                serde_json::to_value(reply.reasoning_mode).unwrap().as_str().unwrap_or("").to_string(), serde_json::to_value(reply.assurance).unwrap().as_str().unwrap_or("").to_string(),
                reply.decision.escalate as i64, reply.usage.wall_ms as i64, reply.usage.ttft_ms.map(|x| x as i64), serde_json::to_value(reply.decision.rule).unwrap().as_str().unwrap_or("").to_string(),
                reply.decision.backend, reply.decision.audit as i64, tag, json]).map_err(st)?;
        c.execute(
            "INSERT OR REPLACE INTO decisions (id,created_at,json) VALUES (?1,?2,?3)",
            params![
                reply.decision.decision_id.to_string(),
                Utc::now().to_rfc3339(),
                serde_json::to_string(&reply.decision).map_err(st)?
            ],
        )
        .map_err(st)?;
        Ok(())
    }

    pub fn get_decision(&self, id: Uuid) -> Result<DecisionRecord> {
        let c = self.conn.lock().unwrap();
        let j: String = c
            .query_row(
                "SELECT json FROM decisions WHERE id=?1",
                params![id.to_string()],
                |r| r.get(0),
            )
            .optional()
            .map_err(st)?
            .ok_or_else(|| NafsError::NotFound(format!("decision {id}")))?;
        serde_json::from_str(&j).map_err(st)
    }

    /// Per-domain aggregates for the self-model and the dashboard (FR-041). Unaudited domains are "unmeasured".
    pub fn reply_stats(&self) -> Result<Vec<Value>> {
        let c = self.conn.lock().unwrap();
        let mut s = c.prepare("SELECT domain, COUNT(*), SUM(escalated), SUM(assurance='verified'), SUM(assurance='consensus'), SUM(assurance='abstained'), AVG(wall_ms), AVG(ttft_ms), SUM(audit), SUM(ephemeral > 0) FROM replies GROUP BY domain ORDER BY COUNT(*) DESC").map_err(st)?;
        let rows = s.query_map([], |r| {
            let n: i64 = r.get(1)?; let slow: i64 = r.get::<_, Option<i64>>(2)?.unwrap_or(0);
            Ok(serde_json::json!({"domain": r.get::<_, String>(0)?, "n": n, "escalation_rate": if n>0 { slow as f64 / n as f64 } else { 0.0 },
                "verified": r.get::<_, Option<i64>>(3)?.unwrap_or(0), "consensus": r.get::<_, Option<i64>>(4)?.unwrap_or(0), "abstained": r.get::<_, Option<i64>>(5)?.unwrap_or(0),
                "mean_wall_ms": r.get::<_, Option<f64>>(6)?, "mean_ttft_ms": r.get::<_, Option<f64>>(7)?, "n_audit": r.get::<_, Option<i64>>(8)?.unwrap_or(0), "n_eval": r.get::<_, Option<i64>>(9)?.unwrap_or(0)}))
        }).map_err(st)?;
        rows.collect::<std::result::Result<Vec<_>, _>>().map_err(st)
    }

    pub fn recent_replies(&self, limit: i64) -> Result<Vec<Value>> {
        let c = self.conn.lock().unwrap();
        let mut s = c.prepare("SELECT created_at, domain, mode, assurance, wall_ms, ttft_ms, rule, backend FROM replies ORDER BY created_at DESC LIMIT ?1").map_err(st)?;
        let rows = s.query_map(params![limit], |r| Ok(serde_json::json!({"created_at": r.get::<_, String>(0)?, "domain": r.get::<_, Option<String>>(1)?, "mode": r.get::<_, Option<String>>(2)?,
            "assurance": r.get::<_, Option<String>>(3)?, "wall_ms": r.get::<_, Option<i64>>(4)?, "ttft_ms": r.get::<_, Option<i64>>(5)?, "rule": r.get::<_, Option<String>>(6)?, "backend": r.get::<_, Option<String>>(7)?}))).map_err(st)?;
        rows.collect::<std::result::Result<Vec<_>, _>>().map_err(st)
    }

    pub fn create_room(&self, name: &str, profile: &str) -> Result<Room> {
        let id = Uuid::new_v4();
        let conv = self.ensure_conversation(None, &format!("Room: {name}"), Some(id))?;
        let now = Utc::now().to_rfc3339();
        self.conn.lock().unwrap().execute("INSERT INTO rooms (id,name,profile,conversation_id,created_at) VALUES (?1,?2,?3,?4,?5)",
            params![id.to_string(), name, profile, conv.to_string(), now]).map_err(st)?;
        Ok(Room {
            room_id: id,
            name: name.into(),
            profile: profile.into(),
            conversation_id: conv,
            created_at: now,
        })
    }
    pub fn list_rooms(&self) -> Result<Vec<Room>> {
        let c = self.conn.lock().unwrap();
        let mut s = c
            .prepare(
                "SELECT id,name,profile,conversation_id,created_at FROM rooms ORDER BY created_at",
            )
            .map_err(st)?;
        let rows = s
            .query_map([], |r| {
                Ok(Room {
                    room_id: Uuid::parse_str(&r.get::<_, String>(0)?).unwrap_or_default(),
                    name: r.get(1)?,
                    profile: r.get(2)?,
                    conversation_id: Uuid::parse_str(&r.get::<_, String>(3)?).unwrap_or_default(),
                    created_at: r.get(4)?,
                })
            })
            .map_err(st)?;
        rows.collect::<std::result::Result<Vec<_>, _>>().map_err(st)
    }
    pub fn get_room(&self, id: Uuid) -> Result<Room> {
        self.list_rooms()?
            .into_iter()
            .find(|r| r.room_id == id)
            .ok_or_else(|| NafsError::NotFound(format!("room {id}")))
    }

    pub fn save_run(&self, kind: &str, id: Uuid, status: &str, json: &Value) -> Result<()> {
        self.conn.lock().unwrap().execute("INSERT INTO runs (id,kind,created_at,status,json) VALUES (?1,?2,?3,?4,?5) ON CONFLICT(id) DO UPDATE SET status=?4, json=?5",
            params![id.to_string(), kind, Utc::now().to_rfc3339(), status, json.to_string()]).map_err(st)?;
        Ok(())
    }
    pub fn get_run(&self, id: Uuid) -> Result<Value> {
        let c = self.conn.lock().unwrap();
        let j: String = c
            .query_row(
                "SELECT json FROM runs WHERE id=?1",
                params![id.to_string()],
                |r| r.get(0),
            )
            .optional()
            .map_err(st)?
            .ok_or_else(|| NafsError::NotFound(format!("run {id}")))?;
        serde_json::from_str(&j).map_err(st)
    }
    pub fn list_runs(&self, kind: &str, limit: i64) -> Result<Vec<Value>> {
        let c = self.conn.lock().unwrap();
        let mut s = c
            .prepare("SELECT json FROM runs WHERE kind=?1 ORDER BY created_at DESC LIMIT ?2")
            .map_err(st)?;
        let rows = s
            .query_map(params![kind, limit], |r| r.get::<_, String>(0))
            .map_err(st)?;
        Ok(rows
            .filter_map(|r| r.ok())
            .filter_map(|j| serde_json::from_str(&j).ok())
            .collect())
    }

    // ---------------- Audit stream (FR-040) and self-model (Prop. 5, FR-041) ----------------

    /// Queue an audit of a kept (fast-path) reply: System 2 will re-answer it offline, draft-blind.
    #[allow(clippy::too_many_arguments)]
    pub fn insert_audit(
        &self,
        decision_id: Uuid,
        domain: &str,
        request: &Value,
        kept_answer: &str,
        kept_assurance: &str,
        e1_hat: f32,
        features: &Value,
    ) -> Result<Uuid> {
        let id = Uuid::new_v4();
        self.conn.lock().unwrap().execute(
            "INSERT INTO audits (id,decision_id,created_at,status,domain,request_json,kept_answer,kept_assurance,e1_hat,features_json) VALUES (?1,?2,?3,'pending',?4,?5,?6,?7,?8,?9)",
            params![id.to_string(), decision_id.to_string(), Utc::now().to_rfc3339(), domain, request.to_string(), kept_answer, kept_assurance, e1_hat as f64, features.to_string()],
        ).map_err(st)?;
        Ok(id)
    }
    /// Oldest pending audit: (id, request JSON, kept answer).
    pub fn next_pending_audit(&self) -> Result<Option<(Uuid, Value, String)>> {
        let c = self.conn.lock().unwrap();
        let row: Option<(String, String, String)> = c
            .query_row("SELECT id, request_json, kept_answer FROM audits WHERE status='pending' ORDER BY created_at LIMIT 1", [], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
            .optional()
            .map_err(st)?;
        Ok(row.map(|(id, rq, k)| {
            (
                Uuid::parse_str(&id).unwrap_or_default(),
                serde_json::from_str(&rq).unwrap_or(Value::Null),
                k,
            )
        }))
    }
    #[allow(clippy::too_many_arguments)]
    pub fn complete_audit(
        &self,
        id: Uuid,
        status: &str,
        slow_answer: Option<&str>,
        kept_correct: Option<bool>,
        slow_correct: Option<bool>,
        agreement: Option<f32>,
        judge_v: Option<f32>,
        wall_ms: u64,
        error: Option<&str>,
    ) -> Result<()> {
        self.conn.lock().unwrap().execute(
            "UPDATE audits SET status=?2, slow_answer=?3, kept_correct=?4, slow_correct=?5, agreement=?6, judge_v=?7, wall_ms=?8, completed_at=?9, error=?10 WHERE id=?1",
            params![id.to_string(), status, slow_answer, kept_correct, slow_correct, agreement.map(|x| x as f64), judge_v.map(|x| x as f64), wall_ms as i64, Utc::now().to_rfc3339(), error],
        ).map_err(st)?;
        Ok(())
    }
    pub fn list_audits(&self, limit: i64) -> Result<Vec<Value>> {
        let c = self.conn.lock().unwrap();
        let mut s = c.prepare("SELECT id, decision_id, created_at, status, domain, kept_assurance, e1_hat, kept_correct, slow_correct, agreement, judge_v, wall_ms, completed_at, error, substr(kept_answer,1,300), substr(coalesce(slow_answer,''),1,300) FROM audits ORDER BY created_at DESC LIMIT ?1").map_err(st)?;
        let rows = s.query_map(params![limit], |r| Ok(serde_json::json!({
            "audit_id": r.get::<_, String>(0)?, "decision_id": r.get::<_, String>(1)?, "created_at": r.get::<_, String>(2)?, "status": r.get::<_, String>(3)?,
            "domain": r.get::<_, String>(4)?, "kept_assurance": r.get::<_, Option<String>>(5)?, "e1_hat": r.get::<_, Option<f64>>(6)?,
            "kept_correct": r.get::<_, Option<bool>>(7)?, "slow_correct": r.get::<_, Option<bool>>(8)?, "agreement": r.get::<_, Option<f64>>(9)?,
            "judge_v": r.get::<_, Option<f64>>(10)?, "wall_ms": r.get::<_, Option<i64>>(11)?, "completed_at": r.get::<_, Option<String>>(12)?,
            "error": r.get::<_, Option<String>>(13)?, "kept_preview": r.get::<_, String>(14)?, "slow_preview": r.get::<_, String>(15)?}))).map_err(st)?;
        rows.collect::<std::result::Result<Vec<_>, _>>().map_err(st)
    }
    pub fn audit_counts(&self) -> Result<Value> {
        let c = self.conn.lock().unwrap();
        let mut s = c
            .prepare("SELECT status, COUNT(*) FROM audits GROUP BY status")
            .map_err(st)?;
        let mut m = serde_json::Map::new();
        for r in s
            .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))
            .map_err(st)?
            .flatten()
        {
            m.insert(r.0, r.1.into());
        }
        Ok(Value::Object(m))
    }

    /// Per-domain capability table (FR-041) with the audit-weighted estimator of Prop. 5:
    /// p2 = g * p2|esc + (1-g) * p2|audit, SE <= 0.5 sqrt((g + (1-g)/eps) / n). Unknown values are null, never 0.
    pub fn self_model(&self, eps: f64, min_audits: i64, backend: &str) -> Result<Vec<Value>> {
        let c = self.conn.lock().unwrap();
        // Live (non-evaluation) traffic of one model pair, keyed by backend (the adapter hash is "none"
        // until Phase 3); evaluation runs have their own labelled reports.
        let mut s = c
            .prepare("SELECT domain, escalated, assurance, json FROM replies WHERE ephemeral=0 AND backend=?1")
            .map_err(st)?;
        #[derive(Default)]
        struct D {
            n: i64,
            esc: i64,
            esc_ver_n: i64,
            esc_ver_ok: i64,
            fast_ver_n: i64,
            fast_ver_ok: i64,
            verified: i64,
            consensus: i64,
            abstained: i64,
            audits: i64,
            a_slow_n: i64,
            a_slow_ok: i64,
            a_kept_n: i64,
            a_kept_ok: i64,
            a_agree_sum: f64,
            a_agree_n: i64,
        }
        let mut by: std::collections::BTreeMap<String, D> = Default::default();
        let rows = s
            .query_map(params![backend], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, i64>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, String>(3)?,
                ))
            })
            .map_err(st)?;
        for (dom, esc, assurance, json) in rows.flatten() {
            let d = by.entry(dom).or_default();
            d.n += 1;
            d.esc += esc;
            match assurance.as_str() {
                "verified" => d.verified += 1,
                "consensus" => d.consensus += 1,
                "abstained" => d.abstained += 1,
                _ => {}
            }
            let v: Value = serde_json::from_str(&json).unwrap_or(Value::Null);
            if v["decision"]["features"]["verifiable"].as_bool() == Some(true) {
                let ok = assurance == "verified";
                if esc == 1 {
                    d.esc_ver_n += 1;
                    d.esc_ver_ok += ok as i64;
                } else {
                    d.fast_ver_n += 1;
                    d.fast_ver_ok += ok as i64;
                }
            }
        }
        let mut s = c
            .prepare("SELECT a.domain, a.kept_correct, a.slow_correct, a.agreement FROM audits a JOIN decisions d ON d.id = a.decision_id WHERE a.status='done' AND json_extract(d.json, '$.backend') = ?1")
            .map_err(st)?;
        let rows = s
            .query_map(params![backend], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, Option<bool>>(1)?,
                    r.get::<_, Option<bool>>(2)?,
                    r.get::<_, Option<f64>>(3)?,
                ))
            })
            .map_err(st)?;
        for (dom, kept, slow, agree) in rows.flatten() {
            let d = by.entry(dom).or_default();
            d.audits += 1;
            if let Some(x) = slow {
                d.a_slow_n += 1;
                d.a_slow_ok += x as i64;
            }
            if let Some(x) = kept {
                d.a_kept_n += 1;
                d.a_kept_ok += x as i64;
            }
            if let Some(x) = agree {
                d.a_agree_n += 1;
                d.a_agree_sum += x;
            }
        }
        let rate = |k: i64, n: i64| {
            if n > 0 {
                Some(k as f64 / n as f64)
            } else {
                None
            }
        };
        Ok(by.into_iter().map(|(dom, d)| {
            let g = rate(d.esc, d.n);
            let p_esc = rate(d.esc_ver_ok, d.esc_ver_n);
            let p_aud = rate(d.a_slow_ok, d.a_slow_n);
            let p2 = match (g, p_esc, p_aud) {
                (Some(g), Some(a), Some(b)) => Some(g * a + (1.0 - g) * b),
                (Some(0.0), None, Some(b)) => Some(b),
                (Some(1.0), Some(a), None) => Some(a),
                _ => None,
            };
            let se = g.map(|g| 0.5 * ((g + (1.0 - g) / eps.max(1e-6)) / d.n.max(1) as f64).sqrt());
            serde_json::json!({
                "domain": dom, "n": d.n, "escalation_rate": g, "verified": d.verified, "consensus": d.consensus, "abstained": d.abstained,
                "n_audit": d.audits, "n_audit_labelled": d.a_slow_n,
                "p2_given_escalated": p_esc, "n_escalated_labelled": d.esc_ver_n,
                "p2_given_audit": p_aud, "p2_hat": p2, "p2_se_bound": if p2.is_some() { se } else { None },
                "p1_kept_audited": rate(d.a_kept_ok, d.a_kept_n), "kept_vs_slow_agreement": if d.a_agree_n > 0 { Some(d.a_agree_sum / d.a_agree_n as f64) } else { None },
                "naive_p2_escalated_only": p_esc,
                "status": if d.audits >= min_audits { "measured" } else { "unmeasured" },
            })
        }).collect())
    }

    /// Labelled gate samples from the ledger (FR-015): decisions whose draft an exact verifier checked,
    /// plus completed audits with an exact label for the kept draft. Returns (features JSON, draft_wrong).
    /// Returns (decision id, features JSON, draft_wrong).
    pub fn labelled_gate_samples(&self, limit: i64) -> Result<Vec<(String, Value, bool)>> {
        let c = self.conn.lock().unwrap();
        let mut out = vec![];
        let mut s = c
            .prepare("SELECT id, json FROM decisions ORDER BY created_at DESC LIMIT ?1")
            .map_err(st)?;
        for (id, j) in s
            .query_map(params![limit], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
            })
            .map_err(st)?
            .flatten()
        {
            let v: Value = serde_json::from_str(&j).unwrap_or(Value::Null);
            let f = &v["features"];
            if let (Some(ver), true) = (
                f["draft_verified"].as_bool(),
                f["k"].as_u64().unwrap_or(0) > 0,
            ) {
                out.push((id, f.clone(), !ver));
            }
        }
        let mut s = c.prepare("SELECT decision_id, features_json, kept_correct FROM audits WHERE status='done' AND kept_correct IS NOT NULL").map_err(st)?;
        for (id, f, k) in s
            .query_map([], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, Option<String>>(1)?,
                    r.get::<_, bool>(2)?,
                ))
            })
            .map_err(st)?
            .flatten()
        {
            let fv: Value = f
                .and_then(|x| serde_json::from_str(&x).ok())
                .unwrap_or(Value::Null);
            if fv["draft_verified"].is_null() && fv["k"].as_u64().unwrap_or(0) > 0 {
                out.push((id, fv, !k));
            }
        }
        Ok(out)
    }
    /// Requests currently stored as running (used to mark interrupted runs on startup).
    pub fn mark_interrupted_runs(&self) -> Result<Vec<(String, Uuid)>> {
        let c = self.conn.lock().unwrap();
        let mut s = c
            .prepare("SELECT kind, id FROM runs WHERE status='running'")
            .map_err(st)?;
        let ids: Vec<(String, Uuid)> = s
            .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))
            .map_err(st)?
            .flatten()
            .filter_map(|(k, id)| Uuid::parse_str(&id).ok().map(|u| (k, u)))
            .collect();
        drop(s);
        c.execute("UPDATE runs SET status='interrupted', json=json_set(json, '$.status', 'interrupted') WHERE status='running'", []).map_err(st)?;
        Ok(ids)
    }

    /// Stage timings of replies from the last `hours` (P7-T01): (mode, wall_ms, stages). Lab traffic is
    /// included only when asked; replies stored before stage timing existed are skipped.
    pub fn stage_rows(
        &self,
        hours: i64,
        include_lab: bool,
    ) -> Result<Vec<(String, u64, Vec<nafs_types::StageTiming>)>> {
        let cutoff = (Utc::now() - chrono::Duration::hours(hours)).to_rfc3339();
        let c = self.conn.lock().unwrap();
        let mut s = c
            .prepare("SELECT mode, wall_ms, json FROM replies WHERE created_at >= ?1 AND (?2 OR ephemeral < 2) ORDER BY created_at DESC LIMIT 5000")
            .map_err(st)?;
        let rows = s
            .query_map(params![cutoff, include_lab], |r| {
                Ok((
                    r.get::<_, Option<String>>(0)?,
                    r.get::<_, Option<i64>>(1)?,
                    r.get::<_, String>(2)?,
                ))
            })
            .map_err(st)?;
        Ok(rows
            .flatten()
            .filter_map(|(m, w, j)| {
                let v: Value = serde_json::from_str(&j).ok()?;
                let stages: Vec<nafs_types::StageTiming> =
                    serde_json::from_value(v["usage"]["stages"].clone()).ok()?;
                (!stages.is_empty())
                    .then(|| (m.unwrap_or_default(), w.unwrap_or(0).max(0) as u64, stages))
            })
            .collect())
    }

    /// Live replies (lab traffic excluded) created in the last `minutes`: (mode, wall_ms, escalated, verifier passed, verifier exact).
    pub fn recent_window(&self, minutes: i64) -> Result<Vec<RecentRow>> {
        let cutoff = (Utc::now() - chrono::Duration::minutes(minutes)).to_rfc3339();
        let c = self.conn.lock().unwrap();
        let mut s = c
            .prepare("SELECT mode, wall_ms, escalated, json FROM replies WHERE created_at >= ?1 AND ephemeral < 2")
            .map_err(st)?;
        let rows = s
            .query_map(params![cutoff], |r| {
                Ok((
                    r.get::<_, Option<String>>(0)?,
                    r.get::<_, Option<i64>>(1)?,
                    r.get::<_, Option<i64>>(2)?,
                    r.get::<_, String>(3)?,
                ))
            })
            .map_err(st)?;
        Ok(rows
            .flatten()
            .map(|(m, w, e, j)| {
                let v: Value = serde_json::from_str(&j).unwrap_or(Value::Null);
                (
                    m.unwrap_or_default(),
                    w.unwrap_or(0),
                    e.unwrap_or(0) != 0,
                    v["verifier"]["passed"].as_bool(),
                    v["verifier"]["exact"].as_bool().unwrap_or(false),
                )
            })
            .collect())
    }

    // ---------------- Retrieval memory (FR-061) ----------------

    /// Index a verified answer for later retrieval.
    pub fn index_verified(
        &self,
        reply_id: Uuid,
        conv: Uuid,
        domain: &str,
        question: &str,
        answer: &str,
        verifier: &str,
    ) -> Result<()> {
        self.conn.lock().unwrap().execute(
            "INSERT OR REPLACE INTO memory_index (reply_id, conversation_id, created_at, domain, question, answer, verifier) VALUES (?1,?2,?3,?4,?5,?6,?7)",
            params![reply_id.to_string(), conv.to_string(), Utc::now().to_rfc3339(), domain, question, answer, verifier],
        ).map_err(st)?;
        Ok(())
    }
    /// BM25 retrieval over indexed verified answers of conversations that allow research use. Scores are
    /// normalised by the best score the query could reach (every term saturated), so they lie in [0, 1].
    pub fn retrieve(
        &self,
        query: &str,
        k: usize,
        exclude_conv: Option<Uuid>,
    ) -> Result<Vec<Value>> {
        let c = self.conn.lock().unwrap();
        let mut s = c.prepare("SELECT m.reply_id, m.conversation_id, m.created_at, m.domain, m.question, m.answer, m.verifier FROM memory_index m LEFT JOIN conversations cv ON cv.id = m.conversation_id WHERE COALESCE(cv.training_opt_out, 0) = 0").map_err(st)?;
        let docs: Vec<MemoryDoc> = s
            .query_map([], |r| {
                Ok((
                    r.get(0)?,
                    r.get::<_, Option<String>>(1)?.unwrap_or_default(),
                    r.get(2)?,
                    r.get(3)?,
                    r.get(4)?,
                    r.get(5)?,
                    r.get(6)?,
                ))
            })
            .map_err(st)?
            .flatten()
            .filter(|d| exclude_conv.map(|x| x.to_string() != d.1).unwrap_or(true))
            .collect();
        drop(s);
        drop(c);
        if docs.is_empty() {
            return Ok(vec![]);
        }
        const STOP: [&str; 24] = [
            "the", "and", "for", "with", "what", "how", "many", "much", "from", "that", "this",
            "are", "was", "his", "her", "its", "you", "your", "into", "then", "than", "who", "has",
            "have",
        ];
        let toks = |t: &str| -> Vec<String> {
            t.to_lowercase()
                .split(|ch: char| !ch.is_alphanumeric())
                .filter(|w| w.len() > 1 && !STOP.contains(w))
                .map(String::from)
                .collect()
        };
        let dt: Vec<Vec<String>> = docs
            .iter()
            .map(|d| toks(&format!("{} {}", d.4, d.4)))
            .collect(); // questions weigh most
        let n = dt.len() as f64;
        let avg = dt.iter().map(|d| d.len()).sum::<usize>() as f64 / n;
        let mut q: Vec<String> = toks(query);
        q.sort();
        q.dedup();
        let idf = |w: &str| {
            let df = dt.iter().filter(|d| d.iter().any(|x| x == w)).count() as f64;
            ((n - df + 0.5) / (df + 0.5) + 1.0).ln()
        };
        let (k1, b) = (1.2, 0.75);
        let idfs: Vec<(String, f64)> = q.iter().map(|w| (w.clone(), idf(w))).collect();
        // Normalise by the best score reachable with the query terms that occur in the memory at all.
        let max_score: f64 = idfs
            .iter()
            .filter(|(w, _)| dt.iter().any(|d| d.iter().any(|x| x == w)))
            .map(|(_, i)| i * (k1 + 1.0))
            .sum::<f64>()
            .max(1e-9);
        let mut scored: Vec<(f64, usize)> = dt
            .iter()
            .enumerate()
            .map(|(i, d)| {
                let len = d.len() as f64;
                let sc: f64 = idfs
                    .iter()
                    .map(|(w, idf)| {
                        let tf = d.iter().filter(|x| *x == w).count() as f64;
                        idf * tf * (k1 + 1.0) / (tf + k1 * (1.0 - b + b * len / avg.max(1.0)))
                    })
                    .sum();
                (sc / max_score, i)
            })
            .filter(|(sc, _)| *sc > 0.0)
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scored
            .into_iter()
            .take(k)
            .map(|(sc, i)| {
                let d = &docs[i];
                serde_json::json!({"reply_id": d.0, "created_at": d.2, "domain": d.3, "question": d.4, "answer": d.5, "verifier": d.6, "score": sc})
            })
            .collect())
    }

    // ---------------- Lab: paired human preferences between the two paths ----------------

    /// Store one compare-both-paths trial: the same prompt answered by System 1 and System 2 and the
    /// person's choice (fast | slow | tie | both_bad). An empirical estimate of the value of deliberation
    /// where no exact verifier exists.
    #[allow(clippy::too_many_arguments)]
    pub fn add_preference(
        &self,
        domain: &str,
        prompt: &str,
        fast_reply: Uuid,
        slow_reply: Uuid,
        preferred: &str,
        note: Option<&str>,
    ) -> Result<Uuid> {
        if !["fast", "slow", "tie", "both_bad"].contains(&preferred) {
            return Err(NafsError::BadRequest(
                "preferred must be fast, slow, tie or both_bad".into(),
            ));
        }
        let c = self.conn.lock().unwrap();
        let wall = |id: Uuid| -> Result<Option<i64>> {
            c.query_row(
                "SELECT wall_ms FROM replies WHERE id=?1",
                params![id.to_string()],
                |r| r.get(0),
            )
            .optional()
            .map_err(st)
        };
        let (fw, sw) = (wall(fast_reply)?, wall(slow_reply)?);
        if fw.is_none() || sw.is_none() {
            return Err(NafsError::NotFound("both replies must exist".into()));
        }
        let id = Uuid::new_v4();
        c.execute(
            "INSERT INTO preferences (id,created_at,domain,prompt,fast_reply_id,slow_reply_id,preferred,fast_wall_ms,slow_wall_ms,note) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)",
            params![id.to_string(), Utc::now().to_rfc3339(), domain, prompt, fast_reply.to_string(), slow_reply.to_string(), preferred, fw, sw, note],
        ).map_err(st)?;
        Ok(id)
    }
    /// Per-domain preference counts and the mean extra latency the slow path cost.
    pub fn preference_stats(&self) -> Result<Vec<Value>> {
        let c = self.conn.lock().unwrap();
        let mut s = c.prepare("SELECT COALESCE(domain,'other'), COUNT(*), SUM(preferred='slow'), SUM(preferred='fast'), SUM(preferred='tie'), SUM(preferred='both_bad'), AVG(slow_wall_ms - fast_wall_ms) FROM preferences GROUP BY 1 ORDER BY 2 DESC").map_err(st)?;
        let rows = s.query_map([], |r| Ok(serde_json::json!({"domain": r.get::<_, String>(0)?, "n": r.get::<_, i64>(1)?, "slow": r.get::<_, i64>(2)?, "fast": r.get::<_, i64>(3)?,
            "tie": r.get::<_, i64>(4)?, "both_bad": r.get::<_, i64>(5)?, "mean_extra_ms": r.get::<_, Option<f64>>(6)?}))).map_err(st)?;
        rows.collect::<std::result::Result<Vec<_>, _>>().map_err(st)
    }

    /// Labelled gate observations in time order for drift monitoring (FR-042): (timestamp, domain,
    /// e1_hat, draft_wrong) from decisions whose draft an exact verifier checked, plus exact-labelled audits.
    pub fn labelled_series(&self) -> Result<Vec<(String, String, f64, bool)>> {
        let c = self.conn.lock().unwrap();
        let mut out = vec![];
        let mut s = c
            .prepare("SELECT created_at, json FROM decisions ORDER BY created_at")
            .map_err(st)?;
        for (t, j) in s
            .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))
            .map_err(st)?
            .flatten()
        {
            let v: Value = serde_json::from_str(&j).unwrap_or(Value::Null);
            if let (Some(ver), Some(e1)) = (
                v["features"]["draft_verified"].as_bool(),
                v["e1_hat"].as_f64(),
            ) {
                out.push((
                    t,
                    v["features"]["domain"]
                        .as_str()
                        .unwrap_or("other")
                        .to_string(),
                    e1,
                    !ver,
                ));
            }
        }
        let mut s = c.prepare("SELECT completed_at, domain, e1_hat, kept_correct FROM audits WHERE status='done' AND kept_correct IS NOT NULL").map_err(st)?;
        for row in s
            .query_map([], |r| {
                Ok((
                    r.get::<_, Option<String>>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, Option<f64>>(2)?,
                    r.get::<_, bool>(3)?,
                ))
            })
            .map_err(st)?
            .flatten()
        {
            if let (Some(t), Some(e1)) = (row.0, row.2) {
                out.push((t, row.1, e1, !row.3));
            }
        }
        out.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(out)
    }

    // ---------------- Human feedback, review and privacy (FR-033, SR-004, Phase 3) ----------------

    pub fn add_feedback(&self, reply_id: Uuid, rating: i64, note: Option<&str>) -> Result<()> {
        let c = self.conn.lock().unwrap();
        let exists: Option<String> = c
            .query_row(
                "SELECT id FROM replies WHERE id=?1",
                params![reply_id.to_string()],
                |r| r.get(0),
            )
            .optional()
            .map_err(st)?;
        if exists.is_none() {
            return Err(NafsError::NotFound(format!("reply {reply_id}")));
        }
        c.execute(
            "INSERT INTO feedback (reply_id, created_at, rating, note) VALUES (?1, ?2, ?3, ?4)",
            params![
                reply_id.to_string(),
                Utc::now().to_rfc3339(),
                rating.signum(),
                note
            ],
        )
        .map_err(st)?;
        Ok(())
    }
    /// FR-033: a reviewer approves or rejects a consensus-assured reply for the distillation set.
    pub fn set_review(
        &self,
        reply_id: Uuid,
        approved: bool,
        reviewer: &str,
        note: Option<&str>,
    ) -> Result<()> {
        self.conn.lock().unwrap().execute(
            "INSERT INTO reviews (reply_id, decision, reviewer, note, created_at) VALUES (?1, ?2, ?3, ?4, ?5) ON CONFLICT(reply_id) DO UPDATE SET decision=?2, reviewer=?3, note=?4, created_at=?5",
            params![reply_id.to_string(), if approved { "approved" } else { "rejected" }, reviewer, note, Utc::now().to_rfc3339()],
        ).map_err(st)?;
        Ok(())
    }
    pub fn set_training_opt_out(&self, conv: Uuid, opt_out: bool) -> Result<()> {
        let n = self
            .conn
            .lock()
            .unwrap()
            .execute(
                "UPDATE conversations SET training_opt_out=?2 WHERE id=?1",
                params![conv.to_string(), opt_out as i64],
            )
            .map_err(st)?;
        if n == 0 {
            return Err(NafsError::NotFound(format!("conversation {conv}")));
        }
        Ok(())
    }
    /// Consensus replies awaiting human review, with the question that produced them.
    pub fn review_queue(&self, limit: i64) -> Result<Vec<Value>> {
        let c = self.conn.lock().unwrap();
        let mut s = c.prepare(
            "SELECT r.id, r.created_at, r.domain, r.mode, (SELECT u.content FROM messages u WHERE u.conversation_id = m.conversation_id AND u.role='user' AND u.id < m.id ORDER BY u.id DESC LIMIT 1), m.content
             FROM replies r JOIN messages m ON m.reply_id = r.id JOIN conversations cv ON cv.id = m.conversation_id
             WHERE r.ephemeral=0 AND r.assurance='consensus' AND cv.training_opt_out=0 AND NOT EXISTS (SELECT 1 FROM reviews v WHERE v.reply_id = r.id)
             ORDER BY r.created_at DESC LIMIT ?1").map_err(st)?;
        let rows = s
            .query_map(params![limit], |r| {
                Ok(serde_json::json!({"reply_id": r.get::<_, String>(0)?, "created_at": r.get::<_, String>(1)?, "domain": r.get::<_, Option<String>>(2)?,
                    "mode": r.get::<_, Option<String>>(3)?, "question": r.get::<_, Option<String>>(4)?, "answer": r.get::<_, String>(5)?}))
            })
            .map_err(st)?;
        rows.collect::<std::result::Result<Vec<_>, _>>().map_err(st)
    }
    /// What the distillation dataset builder would see (TR-002, FR-033, SR-004), per domain.
    pub fn tazkiya_summary(&self) -> Result<Value> {
        let c = self.conn.lock().unwrap();
        let mut s = c.prepare(
            "SELECT r.domain,
                    SUM(r.assurance='verified' AND r.mode='slow'),
                    SUM(r.assurance='verified' AND r.mode='fast'),
                    SUM(r.assurance='consensus' AND v.decision='approved'),
                    SUM(r.assurance='consensus' AND v.decision IS NULL),
                    SUM(r.assurance='abstained'),
                    COUNT(*)
             FROM replies r JOIN messages m ON m.reply_id = r.id JOIN conversations cv ON cv.id = m.conversation_id
             LEFT JOIN reviews v ON v.reply_id = r.id
             WHERE r.ephemeral=0 AND cv.training_opt_out=0 GROUP BY r.domain").map_err(st)?;
        let rows: Vec<Value> = s
            .query_map([], |r| {
                Ok(serde_json::json!({"domain": r.get::<_, Option<String>>(0)?, "verified_slow": r.get::<_, i64>(1)?, "verified_fast": r.get::<_, i64>(2)?,
                    "consensus_approved": r.get::<_, i64>(3)?, "consensus_pending": r.get::<_, i64>(4)?, "abstained": r.get::<_, i64>(5)?, "replies": r.get::<_, i64>(6)?}))
            })
            .map_err(st)?
            .flatten()
            .collect();
        let opted: i64 = c
            .query_row(
                "SELECT COUNT(*) FROM conversations WHERE training_opt_out=1",
                [],
                |r| r.get(0),
            )
            .map_err(st)?;
        let fb: (i64, i64) = c
            .query_row(
                "SELECT COALESCE(SUM(rating>0),0), COALESCE(SUM(rating<0),0) FROM feedback",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .map_err(st)?;
        let rv: (i64, i64) = c
            .query_row("SELECT COALESCE(SUM(decision='approved'),0), COALESCE(SUM(decision='rejected'),0) FROM reviews", [], |r| Ok((r.get(0)?, r.get(1)?)))
            .map_err(st)?;
        Ok(
            serde_json::json!({"domains": rows, "opted_out_conversations": opted, "feedback": {"up": fb.0, "down": fb.1}, "reviews": {"approved": rv.0, "rejected": rv.1}}),
        )
    }
    /// Per-domain human ratings on live traffic (a soft signal, never a calibration label).
    pub fn feedback_by_domain(&self) -> Result<std::collections::BTreeMap<String, (i64, i64)>> {
        let c = self.conn.lock().unwrap();
        let mut s = c
            .prepare("SELECT r.domain, SUM(f.rating>0), SUM(f.rating<0) FROM feedback f JOIN replies r ON r.id=f.reply_id GROUP BY r.domain")
            .map_err(st)?;
        let rows = s
            .query_map([], |r| {
                Ok((
                    r.get::<_, Option<String>>(0)?.unwrap_or_default(),
                    (r.get::<_, i64>(1)?, r.get::<_, i64>(2)?),
                ))
            })
            .map_err(st)?;
        Ok(rows.flatten().collect())
    }
    /// SR-004: bounded retention. Deletes conversations (and their messages) not updated for `days`,
    /// plus evaluation replies older than that. Returns the number of conversations removed.
    pub fn purge_older_than(&self, days: u32) -> Result<usize> {
        let cutoff = (Utc::now() - chrono::Duration::days(days as i64)).to_rfc3339();
        let c = self.conn.lock().unwrap();
        let n = c
            .execute(
                "DELETE FROM conversations WHERE updated_at < ?1 AND room_id IS NULL",
                params![cutoff],
            )
            .map_err(st)?;
        c.execute("DELETE FROM replies WHERE created_at < ?1", params![cutoff])
            .map_err(st)?;
        for t in [
            "feedback",
            "decisions",
            "audits",
            "reviews",
            "preferences",
            "memory_index",
        ] {
            c.execute(
                &format!("DELETE FROM {t} WHERE created_at < ?1"),
                params![cutoff],
            )
            .map_err(st)?;
        }
        Ok(n)
    }

    /// Append-only, fsync'd JSONL trace (IF-010).
    pub fn write_trace(&self, rec: &TraceRecord) -> Result<()> {
        let mut g = self.trace.lock().unwrap();
        if let Some(f) = g.as_mut() {
            let line = serde_json::to_string(rec).map_err(st)?;
            f.write_all(line.as_bytes()).map_err(st)?;
            f.write_all(b"\n").map_err(st)?;
            f.sync_data().map_err(st)?;
        }
        Ok(())
    }
}

/// Prompt assembly within a token cap (FR-003, NFR-006): keep the system prompt and the newest whole turns,
/// dropping the oldest turns entirely so the remaining prefix is stable for the backend prompt cache.
pub fn assemble_prompt(
    system: &str,
    history: &[Message],
    new_turn: &[Message],
    cap_tokens: usize,
) -> Vec<Message> {
    let est = |m: &Message| (m.content.chars().count() as f64 / 3.5).ceil() as usize + 4;
    let mut budget = cap_tokens.saturating_sub(
        (system.len() as f64 / 3.5) as usize + new_turn.iter().map(est).sum::<usize>(),
    );
    let mut kept: Vec<Message> = Vec::new();
    // walk turns (user+assistant pairs) from newest to oldest
    let mut i = history.len();
    while i > 0 {
        let start = if i >= 2 && history[i - 2].role == "user" {
            i - 2
        } else {
            i - 1
        };
        let cost: usize = history[start..i].iter().map(est).sum();
        if cost > budget {
            break;
        }
        budget -= cost;
        let mut turn = history[start..i].to_vec();
        turn.extend(kept);
        kept = turn;
        i = start;
    }
    let mut out = vec![Message {
        role: "system".into(),
        content: system.into(),
    }];
    out.extend(kept);
    out.extend(new_turn.iter().cloned());
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    fn m(r: &str, c: &str) -> Message {
        Message {
            role: r.into(),
            content: c.into(),
        }
    }
    fn tmp_store() -> (Store, std::path::PathBuf) {
        let d = std::env::temp_dir().join(format!("nafs-mem-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&d).unwrap();
        (
            Store::open(&d.join("db.sqlite"), &d.join("t.jsonl")).unwrap(),
            d,
        )
    }
    fn fake_reply(st: &Store, conv: Uuid, assurance: &str, mode: &str) -> Uuid {
        let id = Uuid::new_v4();
        st.append_message(conv, "user", "What is patience?", None)
            .unwrap();
        st.conn.lock().unwrap().execute(
            "INSERT INTO replies (id,conversation_id,created_at,domain,mode,assurance,escalated,wall_ms,rule,backend,audit,ephemeral,json) VALUES (?1,?2,?3,'open',?4,?5,0,1,'fixed','llama',0,0,'{}')",
            params![id.to_string(), conv.to_string(), Utc::now().to_rfc3339(), mode, assurance]).unwrap();
        st.append_message(conv, "assistant", "Patience is steadfastness.", Some(id))
            .unwrap();
        id
    }
    #[test]
    fn feedback_review_and_opt_out_shape_the_training_view() {
        let (st, d) = tmp_store();
        let conv = st.ensure_conversation(None, "t", None).unwrap();
        let cons = fake_reply(&st, conv, "consensus", "slow");
        let ver = fake_reply(&st, conv, "verified", "slow");
        st.add_feedback(ver, 1, Some("good")).unwrap();
        assert!(st.add_feedback(Uuid::new_v4(), 1, None).is_err());
        let q = st.review_queue(10).unwrap();
        assert_eq!(q.len(), 1);
        assert_eq!(q[0]["question"], "What is patience?");
        st.set_review(cons, true, "tester", None).unwrap();
        assert!(st.review_queue(10).unwrap().is_empty());
        let sum = st.tazkiya_summary().unwrap();
        assert_eq!(sum["domains"][0]["verified_slow"], 1);
        assert_eq!(sum["domains"][0]["consensus_approved"], 1);
        assert_eq!(sum["feedback"]["up"], 1);
        let c = st.get_conversation(conv).unwrap();
        assert_eq!(
            c.messages
                .iter()
                .filter_map(|m| m.feedback)
                .collect::<Vec<_>>(),
            vec![1]
        );
        // Opt-out removes the conversation from the training view (SR-004).
        st.set_training_opt_out(conv, true).unwrap();
        assert!(st.tazkiya_summary().unwrap()["domains"]
            .as_array()
            .unwrap()
            .is_empty());
        assert!(st.get_conversation(conv).unwrap().training_opt_out);
        // Bounded retention: nothing is older than 1 day yet.
        assert_eq!(st.purge_older_than(1).unwrap(), 0);
        let _ = std::fs::remove_dir_all(d);
    }
    #[test]
    fn bm25_retrieves_the_relevant_verified_answer() {
        let (st, d) = tmp_store();
        let conv = st.ensure_conversation(None, "t", None).unwrap();
        st.index_verified(
            Uuid::new_v4(),
            conv,
            "math",
            "How many minutes from 3:40 pm to 6:05 pm?",
            "Answer: 145",
            "math_numeric",
        )
        .unwrap();
        st.index_verified(
            Uuid::new_v4(),
            conv,
            "code",
            "Write a function to reverse a string",
            "def rev(s): return s[::-1]",
            "code_tests",
        )
        .unwrap();
        let r = st
            .retrieve("minutes between 3:40 pm and 6:05 pm", 2, None)
            .unwrap();
        assert_eq!(r[0]["answer"], "Answer: 145");
        assert!(r[0]["score"].as_f64().unwrap() > 0.3);
        assert!(
            st.retrieve("minutes between 3:40 pm and 6:05 pm", 2, Some(conv))
                .unwrap()
                .is_empty(),
            "own conversation excluded"
        );
        st.set_training_opt_out(conv, true).unwrap();
        assert!(
            st.retrieve("reverse a string", 2, None).unwrap().is_empty(),
            "opted-out conversations are not reused"
        );
        let _ = std::fs::remove_dir_all(d);
    }
    #[test]
    fn drops_whole_old_turns() {
        let hist = vec![
            m("user", &"a".repeat(400)),
            m("assistant", &"b".repeat(400)),
            m("user", "hi"),
            m("assistant", "hello"),
        ];
        let p = assemble_prompt("sys", &hist, &[m("user", "next")], 60);
        assert_eq!(p.len(), 4);
        assert_eq!(p[1].content, "hi");
    }
}
