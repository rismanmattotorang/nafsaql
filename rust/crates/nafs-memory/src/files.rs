//! Phase 29: the files of the context filesystem that live in this store: memory files (`nafs://memory/{kind}/{key}`),
//! the log of every memory write (the diff, which can be undone), skills (`nafs://skills/{name}`) and committed
//! sessions (`nafs://sessions/{id}`). The formats and merge rules are `nafs_context`'s; this module stores them.

use crate::{st, Store};
use chrono::{DateTime, Utc};
use nafs_context::memory::{self, Candidate, MemoryFile, Op};
use nafs_context::skill::Skill;
use nafs_types::{NafsError, Result};
use rusqlite::{params, OptionalExtension};
use serde::Serialize;
use serde_json::Value;
use uuid::Uuid;

pub(crate) const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS mem_file (kind TEXT NOT NULL, key TEXT NOT NULL, title TEXT NOT NULL, body TEXT NOT NULL,
    updated_at TEXT NOT NULL, sources TEXT NOT NULL DEFAULT '[]', PRIMARY KEY (kind, key));
CREATE TABLE IF NOT EXISTS mem_diff (id TEXT PRIMARY KEY, at TEXT NOT NULL, session TEXT, op TEXT NOT NULL,
    kind TEXT NOT NULL, key TEXT NOT NULL, title TEXT NOT NULL DEFAULT '', before TEXT, after TEXT, reason TEXT NOT NULL,
    undone INTEGER NOT NULL DEFAULT 0);
CREATE INDEX IF NOT EXISTS mem_diff_at ON mem_diff(at);
CREATE TABLE IF NOT EXISTS skill (name TEXT PRIMARY KEY, description TEXT NOT NULL, tags TEXT NOT NULL DEFAULT '[]',
    tools TEXT NOT NULL DEFAULT '[]', body TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS ctx_session (conversation_id TEXT PRIMARY KEY, title TEXT NOT NULL, committed_at TEXT NOT NULL,
    last_message_id INTEGER NOT NULL, messages INTEGER NOT NULL, transcript TEXT NOT NULL, extracted TEXT NOT NULL DEFAULT '[]');
"#;

/// Largest number of memory files kept; past it an add is refused with a reason (the person tidies the store).
pub const MAX_MEMORY_FILES: usize = 5_000;

/// A row of the memory diff.
#[derive(Debug, Clone, Serialize)]
pub struct DiffRow {
    pub id: String,
    pub at: String,
    pub session: Option<String>,
    pub op: String,
    pub uri: String,
    pub title: String,
    pub before: Option<String>,
    pub after: Option<String>,
    pub reason: String,
    pub undone: bool,
}

/// A committed session.
#[derive(Debug, Clone, Serialize)]
pub struct SessionFile {
    pub id: String,
    pub title: String,
    pub committed_at: String,
    pub messages: i64,
    pub transcript: String,
    pub extracted: Value,
}

/// What a commit did.
#[derive(Debug, Clone, Serialize)]
pub struct CommitReport {
    pub session: String,
    pub messages_read: usize,
    pub diff: Vec<DiffRow>,
}

fn file_from_row(r: &rusqlite::Row) -> rusqlite::Result<MemoryFile> {
    let updated: String = r.get(4)?;
    let sources: String = r.get(5)?;
    Ok(MemoryFile {
        kind: r.get(0)?,
        key: r.get(1)?,
        title: r.get(2)?,
        body: r.get(3)?,
        updated: DateTime::parse_from_rfc3339(&updated)
            .map(|d| d.with_timezone(&Utc))
            .unwrap_or_else(|_| Utc::now()),
        sources: serde_json::from_str(&sources).unwrap_or_default(),
    })
}

fn diff_from_row(r: &rusqlite::Row) -> rusqlite::Result<DiffRow> {
    let kind: String = r.get(4)?;
    let key: String = r.get(5)?;
    Ok(DiffRow {
        id: r.get(0)?,
        at: r.get(1)?,
        session: r.get(2)?,
        op: r.get(3)?,
        uri: nafs_context::Uri::memory(&kind, &key).to_string(),
        title: r.get(6)?,
        before: r.get(7)?,
        after: r.get(8)?,
        reason: r.get(9)?,
        undone: r.get::<_, i64>(10)? != 0,
    })
}

fn op_name(op: Op) -> &'static str {
    match op {
        Op::Add => "add",
        Op::Update => "update",
        Op::Delete => "delete",
        Op::Skip => "skip",
    }
}

const DIFF_COLS: &str = "id, at, session, op, kind, key, title, before, after, reason, undone";

impl Store {
    // ------------------------------------------------------------------------------------- memory files

    pub fn memory_files(&self, kind: Option<&str>) -> Result<Vec<MemoryFile>> {
        let c = self.db();
        let mut s = c
            .prepare("SELECT kind, key, title, body, updated_at, sources FROM mem_file WHERE ?1 IS NULL OR kind = ?1 ORDER BY kind, updated_at DESC")
            .map_err(st)?;
        let rows = s.query_map(params![kind], file_from_row).map_err(st)?;
        rows.collect::<std::result::Result<Vec<_>, _>>().map_err(st)
    }

    pub fn memory_file(&self, kind: &str, key: &str) -> Result<Option<MemoryFile>> {
        self.db()
            .query_row(
                "SELECT kind, key, title, body, updated_at, sources FROM mem_file WHERE kind = ?1 AND key = ?2",
                params![kind, key],
                file_from_row,
            )
            .optional()
            .map_err(st)
    }

    fn record_diff(
        c: &rusqlite::Connection,
        session: Option<&str>,
        op: Op,
        f: &MemoryFile,
        before: Option<&str>,
        after: Option<&str>,
        reason: &str,
    ) -> Result<DiffRow> {
        let id = Uuid::new_v4().to_string();
        let at = Utc::now().to_rfc3339();
        c.execute(
            "INSERT INTO mem_diff (id, at, session, op, kind, key, title, before, after, reason) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)",
            params![id, at, session, op_name(op), f.kind, f.key, f.title, before, after, reason],
        )
        .map_err(st)?;
        Ok(DiffRow {
            id,
            at,
            session: session.map(str::to_string),
            op: op_name(op).into(),
            uri: f.uri().to_string(),
            title: f.title.clone(),
            before: before.map(str::to_string),
            after: after.map(str::to_string),
            reason: reason.into(),
            undone: false,
        })
    }

    fn put_file(c: &rusqlite::Connection, f: &MemoryFile) -> Result<()> {
        c.execute(
            "INSERT INTO mem_file (kind, key, title, body, updated_at, sources) VALUES (?1,?2,?3,?4,?5,?6) \
             ON CONFLICT(kind, key) DO UPDATE SET title=?3, body=?4, updated_at=?5, sources=?6",
            params![f.kind, f.key, f.title, f.body, f.updated.to_rfc3339(), serde_json::to_string(&f.sources).unwrap_or_default()],
        )
        .map_err(st)?;
        Ok(())
    }

    /// Write a memory file as the person typed it (the editor): the whole body replaces the old one.
    pub fn write_memory_file(&self, f: &MemoryFile) -> Result<DiffRow> {
        let old = self.memory_file(&f.kind, &f.key)?;
        let c = self.db();
        if old.is_none() {
            let n: i64 = c
                .query_row("SELECT COUNT(*) FROM mem_file", [], |r| r.get(0))
                .map_err(st)?;
            if n as usize >= MAX_MEMORY_FILES {
                return Err(NafsError::BadRequest(format!(
                    "the memory holds {MAX_MEMORY_FILES} files, the most it keeps; remove some first"
                )));
            }
        }
        let tx = c.unchecked_transaction().map_err(st)?;
        Self::put_file(&tx, f)?;
        let op = if old.is_some() { Op::Update } else { Op::Add };
        let row = Self::record_diff(
            &tx,
            None,
            op,
            f,
            old.as_ref().map(|o| o.body.as_str()),
            Some(&f.body),
            "written by hand",
        )?;
        tx.commit().map_err(st)?;
        Ok(row)
    }

    pub fn delete_memory_file(&self, kind: &str, key: &str) -> Result<DiffRow> {
        let old = self
            .memory_file(kind, key)?
            .ok_or_else(|| NafsError::NotFound(format!("nafs://memory/{kind}/{key}")))?;
        let c = self.db();
        let tx = c.unchecked_transaction().map_err(st)?;
        tx.execute(
            "DELETE FROM mem_file WHERE kind = ?1 AND key = ?2",
            params![kind, key],
        )
        .map_err(st)?;
        let row = Self::record_diff(
            &tx,
            None,
            Op::Delete,
            &old,
            Some(&old.body),
            None,
            "removed by hand",
        )?;
        tx.commit().map_err(st)?;
        Ok(row)
    }

    /// Merge a candidate into the memory (spec §5): the file its key names is created, updated or left alone, and
    /// the write is recorded. An event whose key is taken by a different event gets the next free suffix.
    pub fn remember_candidate(&self, c0: &Candidate, session: Option<&str>) -> Result<DiffRow> {
        let mut key = c0.key();
        let mut existing = self.memory_file(&c0.kind, &key)?;
        if c0.kind == "events" {
            let mut n = 2;
            while let Some(e) = &existing {
                if memory::merge(Some(e), c0).0 == Op::Skip {
                    break;
                }
                key = format!("{}-{n}", c0.key());
                existing = self.memory_file(&c0.kind, &key)?;
                n += 1;
            }
        }
        let (op, body, reason) = memory::merge(existing.as_ref(), c0);
        let mut sources = existing
            .as_ref()
            .map(|e| e.sources.clone())
            .unwrap_or_default();
        if let Some(s) = session {
            let uri = nafs_context::Uri::session(s).to_string();
            if !sources.contains(&uri) {
                sources.push(uri);
            }
        }
        let f = MemoryFile {
            kind: c0.kind.clone(),
            key,
            title: existing
                .as_ref()
                .map(|e| e.title.clone())
                .unwrap_or_else(|| c0.title.clone()),
            body,
            updated: Utc::now(),
            sources,
        };
        let c = self.db();
        let tx = c.unchecked_transaction().map_err(st)?;
        let before = existing.as_ref().map(|e| e.body.clone());
        let row = if op == Op::Skip {
            Self::record_diff(&tx, session, op, &f, before.as_deref(), None, reason)?
        } else {
            let n: i64 = tx
                .query_row("SELECT COUNT(*) FROM mem_file", [], |r| r.get(0))
                .map_err(st)?;
            if existing.is_none() && n as usize >= MAX_MEMORY_FILES {
                Self::record_diff(&tx, session, Op::Skip, &f, None, None, "the memory is full")?
            } else {
                Self::put_file(&tx, &f)?;
                Self::record_diff(
                    &tx,
                    session,
                    op,
                    &f,
                    before.as_deref(),
                    Some(&f.body),
                    reason,
                )?
            }
        };
        tx.commit().map_err(st)?;
        Ok(row)
    }

    pub fn memory_diff(&self, limit: usize) -> Result<Vec<DiffRow>> {
        let c = self.db();
        let mut s = c
            .prepare(&format!(
                "SELECT {DIFF_COLS} FROM mem_diff ORDER BY at DESC LIMIT ?1"
            ))
            .map_err(st)?;
        let rows = s
            .query_map(params![limit as i64], diff_from_row)
            .map_err(st)?;
        rows.collect::<std::result::Result<Vec<_>, _>>().map_err(st)
    }

    /// Undo one write: the file returns to its body before the write (an add is removed). A file changed again since
    /// is not overwritten.
    pub fn undo_memory_diff(&self, id: &str) -> Result<DiffRow> {
        let c = self.db();
        let d = c
            .query_row(
                &format!("SELECT {DIFF_COLS} FROM mem_diff WHERE id = ?1"),
                params![id],
                diff_from_row,
            )
            .optional()
            .map_err(st)?
            .ok_or_else(|| NafsError::NotFound(format!("memory change {id}")))?;
        if d.undone || d.op == "skip" {
            return Err(NafsError::BadRequest(
                "this change has nothing to undo".into(),
            ));
        }
        let uri =
            nafs_context::Uri::parse(&d.uri).map_err(|e| NafsError::Internal(e.to_string()))?;
        let (kind, key) = (uri.path[0].clone(), uri.path[1].clone());
        let now: Option<String> = c
            .query_row(
                "SELECT body FROM mem_file WHERE kind = ?1 AND key = ?2",
                params![kind, key],
                |r| r.get(0),
            )
            .optional()
            .map_err(st)?;
        if now != d.after {
            return Err(NafsError::BadRequest(
                "the file was changed again after this; undo the later change first".into(),
            ));
        }
        let tx = c.unchecked_transaction().map_err(st)?;
        match &d.before {
            None => {
                tx.execute(
                    "DELETE FROM mem_file WHERE kind = ?1 AND key = ?2",
                    params![kind, key],
                )
                .map_err(st)?;
            }
            Some(b) => {
                tx.execute(
                    "INSERT INTO mem_file (kind, key, title, body, updated_at) VALUES (?1,?2,?3,?4,?5) \
                     ON CONFLICT(kind, key) DO UPDATE SET body=?4, updated_at=?5",
                    params![kind, key, d.title, b, Utc::now().to_rfc3339()],
                )
                .map_err(st)?;
            }
        }
        tx.execute("UPDATE mem_diff SET undone = 1 WHERE id = ?1", params![id])
            .map_err(st)?;
        tx.commit().map_err(st)?;
        Ok(DiffRow { undone: true, ..d })
    }

    // ------------------------------------------------------------------------------------- skills

    pub fn skills(&self) -> Result<Vec<Skill>> {
        let c = self.db();
        let mut s = c
            .prepare("SELECT name, description, tags, tools, body FROM skill ORDER BY name")
            .map_err(st)?;
        let rows = s
            .query_map([], |r| {
                Ok(Skill {
                    name: r.get(0)?,
                    description: r.get(1)?,
                    tags: serde_json::from_str(&r.get::<_, String>(2)?).unwrap_or_default(),
                    tools: serde_json::from_str(&r.get::<_, String>(3)?).unwrap_or_default(),
                    body: r.get(4)?,
                })
            })
            .map_err(st)?;
        rows.collect::<std::result::Result<Vec<_>, _>>().map_err(st)
    }

    pub fn skill(&self, name: &str) -> Result<Option<Skill>> {
        Ok(self.skills()?.into_iter().find(|s| s.name == name))
    }

    pub fn put_skill(&self, s: &Skill) -> Result<()> {
        self.db()
            .execute(
                "INSERT INTO skill (name, description, tags, tools, body, updated_at) VALUES (?1,?2,?3,?4,?5,?6) \
                 ON CONFLICT(name) DO UPDATE SET description=?2, tags=?3, tools=?4, body=?5, updated_at=?6",
                params![
                    s.name,
                    s.description,
                    serde_json::to_string(&s.tags).unwrap_or_default(),
                    serde_json::to_string(&s.tools).unwrap_or_default(),
                    s.body,
                    Utc::now().to_rfc3339()
                ],
            )
            .map_err(st)?;
        Ok(())
    }

    pub fn delete_skill(&self, name: &str) -> Result<()> {
        let n = self
            .db()
            .execute("DELETE FROM skill WHERE name = ?1", params![name])
            .map_err(st)?;
        if n == 0 {
            return Err(NafsError::NotFound(format!("nafs://skills/{name}")));
        }
        Ok(())
    }

    // ------------------------------------------------------------------------------------- sessions

    pub fn sessions(&self) -> Result<Vec<SessionFile>> {
        let c = self.db();
        let mut s = c
            .prepare("SELECT conversation_id, title, committed_at, messages, '' , extracted FROM ctx_session ORDER BY committed_at DESC")
            .map_err(st)?;
        let rows = s
            .query_map([], |r| {
                Ok(SessionFile {
                    id: r.get(0)?,
                    title: r.get(1)?,
                    committed_at: r.get(2)?,
                    messages: r.get(3)?,
                    transcript: r.get(4)?,
                    extracted: serde_json::from_str(&r.get::<_, String>(5)?).unwrap_or(Value::Null),
                })
            })
            .map_err(st)?;
        rows.collect::<std::result::Result<Vec<_>, _>>().map_err(st)
    }

    pub fn session_file(&self, id: &str) -> Result<Option<SessionFile>> {
        self.db()
            .query_row(
                "SELECT conversation_id, title, committed_at, messages, transcript, extracted FROM ctx_session WHERE conversation_id = ?1",
                params![id],
                |r| {
                    Ok(SessionFile {
                        id: r.get(0)?,
                        title: r.get(1)?,
                        committed_at: r.get(2)?,
                        messages: r.get(3)?,
                        transcript: r.get(4)?,
                        extracted: serde_json::from_str(&r.get::<_, String>(5)?).unwrap_or(Value::Null),
                    })
                },
            )
            .optional()
            .map_err(st)
    }

    pub fn delete_session_file(&self, id: &str) -> Result<()> {
        let n = self
            .db()
            .execute(
                "DELETE FROM ctx_session WHERE conversation_id = ?1",
                params![id],
            )
            .map_err(st)?;
        if n == 0 {
            return Err(NafsError::NotFound(format!("nafs://sessions/{id}")));
        }
        Ok(())
    }

    /// Commit a conversation (spec §6): the transcript is written as Markdown (all messages so far), the person's
    /// messages since the last commit are read for memories by rule, `extra` (a model pass, when on) is merged by the
    /// same rules, and every write is in the returned diff.
    pub fn commit_session(&self, conv: Uuid, extra: &[Candidate]) -> Result<CommitReport> {
        let id = conv.to_string();
        let (title, last_done): (String, i64) = {
            let c = self.db();
            let title: String = c
                .query_row(
                    "SELECT title FROM conversations WHERE id = ?1",
                    params![id],
                    |r| r.get(0),
                )
                .optional()
                .map_err(st)?
                .ok_or_else(|| NafsError::NotFound(format!("conversation {id}")))?;
            let last: i64 = c
                .query_row(
                    "SELECT last_message_id FROM ctx_session WHERE conversation_id = ?1",
                    params![id],
                    |r| r.get(0),
                )
                .optional()
                .map_err(st)?
                .unwrap_or(0);
            (title, last)
        };
        let msgs: Vec<(i64, String, String, String)> = {
            let c = self.db();
            let mut s = c
                .prepare("SELECT id, role, content, created_at FROM messages WHERE conversation_id = ?1 ORDER BY id")
                .map_err(st)?;
            let rows = s
                .query_map(params![id], |r| {
                    Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?))
                })
                .map_err(st)?;
            rows.collect::<std::result::Result<Vec<_>, _>>()
                .map_err(st)?
        };
        let mut transcript = format!(
            "# {title}\n\nSession `{}`, {} messages.\n",
            nafs_context::Uri::session(&id),
            msgs.len()
        );
        for (_, role, content, at) in &msgs {
            let who = if role == "user" { "You" } else { "Assistant" };
            transcript.push_str(&format!("\n## {who} — {at}\n\n{}\n", content.trim()));
        }
        let new_user: Vec<&str> = msgs
            .iter()
            .filter(|m| m.0 > last_done && m.1 == "user")
            .map(|m| m.2.as_str())
            .collect();
        let mut diff = Vec::new();
        let mut cands: Vec<Candidate> = new_user.iter().flat_map(|m| memory::extract(m)).collect();
        cands.extend(extra.iter().cloned());
        for c in &cands {
            diff.push(self.remember_candidate(c, Some(&id))?);
        }
        let last = msgs.last().map_or(last_done, |m| m.0);
        let extracted = serde_json::to_string(
            &diff
                .iter()
                .map(|d| serde_json::json!({"op": d.op, "uri": d.uri, "reason": d.reason}))
                .collect::<Vec<_>>(),
        )
        .unwrap_or_else(|_| "[]".into());
        self.db()
            .execute(
                "INSERT INTO ctx_session (conversation_id, title, committed_at, last_message_id, messages, transcript, extracted) \
                 VALUES (?1,?2,?3,?4,?5,?6,?7) ON CONFLICT(conversation_id) DO UPDATE SET title=?2, committed_at=?3, \
                 last_message_id=?4, messages=?5, transcript=?6, extracted=?7",
                params![id, title, Utc::now().to_rfc3339(), last, msgs.len() as i64, transcript, extracted],
            )
            .map_err(st)?;
        Ok(CommitReport {
            session: id,
            messages_read: new_user.len(),
            diff,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> (Store, tempdir::Dir) {
        let d = tempdir::Dir::new();
        let s = Store::open(&d.0.join("m.db"), &d.0.join("t.jsonl")).unwrap();
        (s, d)
    }

    mod tempdir {
        pub struct Dir(pub std::path::PathBuf);
        impl Dir {
            pub fn new() -> Dir {
                let p = std::env::temp_dir().join(format!("nafs-files-{}", uuid::Uuid::new_v4()));
                std::fs::create_dir_all(&p).unwrap();
                Dir(p)
            }
        }
        impl Drop for Dir {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
    }

    #[test]
    fn commit_extracts_merges_records_and_undoes() {
        let (s, _d) = store();
        let conv = s.ensure_conversation(None, "Plans", None).unwrap();
        s.append_message(conv, "user", "Call me Ris. I prefer short answers.", None)
            .unwrap();
        s.append_message(conv, "assistant", "Noted.", None).unwrap();
        let r = s.commit_session(conv, &[]).unwrap();
        assert_eq!(
            r.diff.iter().map(|d| d.op.as_str()).collect::<Vec<_>>(),
            ["add", "add"]
        );
        assert!(s
            .session_file(&conv.to_string())
            .unwrap()
            .unwrap()
            .transcript
            .contains("## You"));
        // A second commit reads only new messages; a changed preference updates its file.
        s.append_message(
            conv,
            "user",
            "I prefer short answers. Call me Risman.",
            None,
        )
        .unwrap();
        let r2 = s.commit_session(conv, &[]).unwrap();
        let ops: Vec<&str> = r2.diff.iter().map(|d| d.op.as_str()).collect();
        assert_eq!(ops, ["skip", "update"], "{:?}", r2.diff);
        assert_eq!(
            s.memory_file("profile", "profile").unwrap().unwrap().body,
            "name: Risman"
        );
        // Undo the update: the old name returns.
        s.undo_memory_diff(&r2.diff[1].id).unwrap();
        assert_eq!(
            s.memory_file("profile", "profile").unwrap().unwrap().body,
            "name: Ris"
        );
        assert!(s.undo_memory_diff(&r2.diff[1].id).is_err());
        assert_eq!(s.memory_diff(10).unwrap().len(), 4);
    }

    #[test]
    fn hand_written_files_skills_and_events() {
        let (s, _d) = store();
        let f = MemoryFile::from_markdown("notes", "wifi", "# Wifi\nOn the fridge.", Utc::now())
            .unwrap();
        assert_eq!(s.write_memory_file(&f).unwrap().op, "add");
        assert_eq!(s.memory_files(Some("notes")).unwrap().len(), 1);
        s.delete_memory_file("notes", "wifi").unwrap();
        assert!(s.memory_file("notes", "wifi").unwrap().is_none());
        let e = |body: &str| Candidate {
            kind: "events".into(),
            title: "launch".into(),
            body: body.into(),
            date: chrono::NaiveDate::from_ymd_opt(2026, 11, 3),
        };
        s.remember_candidate(&e("We launched the beta."), None)
            .unwrap();
        let second = s
            .remember_candidate(&e("We launched the app."), None)
            .unwrap();
        assert!(
            second.uri.ends_with("2026-11-03-launch-2"),
            "{}",
            second.uri
        );
        assert_eq!(
            s.remember_candidate(&e("We launched the beta."), None)
                .unwrap()
                .op,
            "skip"
        );
        let sk = Skill::parse(
            "triage",
            "---\nname: triage\ndescription: Sort bug reports\n---\nSteps.",
        )
        .unwrap();
        s.put_skill(&sk).unwrap();
        assert_eq!(s.skill("triage").unwrap().unwrap(), sk);
        s.delete_skill("triage").unwrap();
        assert!(s.delete_skill("triage").is_err());
    }
}
