//! SQLite persistence layer for training session history.
//!
//! Stores completed/failed training sessions so they survive server restarts.
//! The database is stored at `./helix-data/training.db` (created automatically).

use std::path::Path;
use std::sync::Mutex;

use rusqlite::{params, Connection};

use crate::dashboard::TrainingSessionState;

/// Thread-safe SQLite handle for persisting training sessions.
pub struct TrainingDb {
    conn: Mutex<Connection>,
}

impl TrainingDb {
    /// Open (or create) the database at `path`, set WAL mode, and ensure tables exist.
    pub fn open(path: &Path) -> Result<Self, rusqlite::Error> {
        // Create parent directories if they don't exist
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }

        let conn = Connection::open(path)?;

        // WAL mode for better concurrent read performance
        conn.pragma_update(None, "journal_mode", "WAL")?;

        let db = Self {
            conn: Mutex::new(conn),
        };
        db.create_tables()?;
        Ok(db)
    }

    /// Create the training_sessions table if it doesn't exist.
    fn create_tables(&self) -> Result<(), rusqlite::Error> {
        let conn = self.conn.lock().expect("DB mutex poisoned");
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS training_sessions (
                session_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                model_name TEXT,
                model_slug TEXT,
                current_step INTEGER DEFAULT 0,
                total_steps INTEGER DEFAULT 0,
                current_loss REAL DEFAULT 0.0,
                losses TEXT DEFAULT '[]',
                accuracy REAL,
                checkpoints_submitted INTEGER DEFAULT 0,
                mac_checks_passed INTEGER DEFAULT 0,
                zk_proofs_generated INTEGER DEFAULT 0,
                cheater_detected TEXT,
                phase INTEGER DEFAULT 0,
                phase_description TEXT DEFAULT '',
                coordinator_address TEXT DEFAULT '',
                job_id INTEGER DEFAULT 0,
                elapsed_secs REAL DEFAULT 0.0,
                started_at REAL DEFAULT 0.0,
                completed_at REAL,
                error_message TEXT
            );",
        )?;
        Ok(())
    }

    /// Upsert a training session into the database.
    pub fn save_session(&self, session: &TrainingSessionState) {
        let conn = self.conn.lock().expect("DB mutex poisoned");

        let losses_json = serde_json::to_string(&session.losses).unwrap_or_else(|_| "[]".to_string());
        let cheater_json = session
            .cheater_detected
            .as_ref()
            .map(|v| serde_json::to_string(v).unwrap_or_else(|_| "null".to_string()));

        // Compute completed_at for terminal states
        let completed_at: Option<f64> = if session.status == "complete" || session.status == "failed" {
            Some(session.started_at + session.elapsed_secs)
        } else {
            None
        };

        // Extract error message from phase_description if failed
        let error_message: Option<&str> = if session.status == "failed" {
            Some(&session.phase_description)
        } else {
            None
        };

        let result = conn.execute(
            "INSERT INTO training_sessions (
                session_id, status, model_name, model_slug,
                current_step, total_steps, current_loss, losses,
                accuracy, checkpoints_submitted, mac_checks_passed,
                zk_proofs_generated, cheater_detected,
                phase, phase_description, coordinator_address,
                job_id, elapsed_secs, started_at,
                completed_at, error_message
            ) VALUES (
                ?1, ?2, ?3, ?4,
                ?5, ?6, ?7, ?8,
                ?9, ?10, ?11,
                ?12, ?13,
                ?14, ?15, ?16,
                ?17, ?18, ?19,
                ?20, ?21
            )
            ON CONFLICT(session_id) DO UPDATE SET
                status = excluded.status,
                model_name = excluded.model_name,
                model_slug = excluded.model_slug,
                current_step = excluded.current_step,
                total_steps = excluded.total_steps,
                current_loss = excluded.current_loss,
                losses = excluded.losses,
                accuracy = excluded.accuracy,
                checkpoints_submitted = excluded.checkpoints_submitted,
                mac_checks_passed = excluded.mac_checks_passed,
                zk_proofs_generated = excluded.zk_proofs_generated,
                cheater_detected = excluded.cheater_detected,
                phase = excluded.phase,
                phase_description = excluded.phase_description,
                coordinator_address = excluded.coordinator_address,
                job_id = excluded.job_id,
                elapsed_secs = excluded.elapsed_secs,
                started_at = excluded.started_at,
                completed_at = excluded.completed_at,
                error_message = excluded.error_message",
            params![
                session.session_id,
                session.status,
                session.model_name,
                session.model_slug,
                session.current_step as i64,
                session.total_steps as i64,
                session.current_loss,
                losses_json,
                session.accuracy,
                session.checkpoints_submitted as i64,
                session.mac_checks_passed as i64,
                session.zk_proofs_generated as i64,
                cheater_json,
                session.phase as i64,
                session.phase_description,
                session.coordinator_address,
                session.job_id as i64,
                session.elapsed_secs,
                session.started_at,
                completed_at,
                error_message,
            ],
        );

        if let Err(e) = result {
            tracing::warn!(error = %e, session_id = %session.session_id, "Failed to save session to DB");
        }
    }

    /// Load all training sessions from the database, ordered by started_at DESC.
    pub fn load_all_sessions(&self) -> Vec<TrainingSessionState> {
        let conn = self.conn.lock().expect("DB mutex poisoned");

        let mut stmt = match conn.prepare(
            "SELECT
                session_id, status, model_name, model_slug,
                current_step, total_steps, current_loss, losses,
                accuracy, checkpoints_submitted, mac_checks_passed,
                zk_proofs_generated, cheater_detected,
                phase, phase_description, coordinator_address,
                job_id, elapsed_secs, started_at
            FROM training_sessions
            ORDER BY started_at DESC",
        ) {
            Ok(stmt) => stmt,
            Err(e) => {
                tracing::warn!(error = %e, "Failed to prepare load_all_sessions query");
                return Vec::new();
            }
        };

        let rows = stmt.query_map([], |row| {
            let session_id: String = row.get(0)?;
            let status: String = row.get(1)?;
            let model_name: Option<String> = row.get(2)?;
            let model_slug: Option<String> = row.get(3)?;
            let current_step: i64 = row.get(4)?;
            let total_steps: i64 = row.get(5)?;
            let current_loss: f64 = row.get(6)?;
            let losses_str: String = row.get(7)?;
            let accuracy: Option<f64> = row.get(8)?;
            let checkpoints_submitted: i64 = row.get(9)?;
            let mac_checks_passed: i64 = row.get(10)?;
            let zk_proofs_generated: i64 = row.get(11)?;
            let cheater_str: Option<String> = row.get(12)?;
            let phase: i64 = row.get(13)?;
            let phase_description: String = row.get(14)?;
            let coordinator_address: String = row.get(15)?;
            let job_id: i64 = row.get(16)?;
            let elapsed_secs: f64 = row.get(17)?;
            let started_at: f64 = row.get(18)?;

            // Deserialize losses from JSON
            let losses: Vec<f64> = serde_json::from_str(&losses_str).unwrap_or_default();

            // Deserialize cheater_detected from JSON
            let cheater_detected: Option<serde_json::Value> = cheater_str
                .and_then(|s| serde_json::from_str(&s).ok());

            Ok(TrainingSessionState {
                session_id,
                status,
                current_step: current_step as usize,
                total_steps: total_steps as usize,
                current_loss,
                losses,
                accuracy,
                checkpoints_submitted: checkpoints_submitted as usize,
                mac_checks_passed: mac_checks_passed as usize,
                cheater_detected,
                zk_proofs_generated: zk_proofs_generated as usize,
                zk_activated_by_risk: false, // Not stored in DB
                phase: phase as u32,
                phase_description,
                coordinator_address,
                job_id: job_id as u64,
                elapsed_secs,
                started_at,
                final_weights: None,       // Not stored in DB (too large)
                model_name,
                model_slug,
                model_token_id: None,       // Not stored in DB
                model_version_index: None,  // Not stored in DB
            })
        });

        match rows {
            Ok(mapped) => mapped.filter_map(|r| r.ok()).collect(),
            Err(e) => {
                tracing::warn!(error = %e, "Failed to load sessions from DB");
                Vec::new()
            }
        }
    }
}
