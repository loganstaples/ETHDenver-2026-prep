//! Node-wide metrics collection with atomic counters.
//!
//! Provides zero-cost (when not read) metrics tracking for all node subsystems:
//! proof generation/verification, training, network, MPC, and on-chain operations.
//!
//! Metrics can be exported as:
//! - [`MetricsSnapshot`] (serializable JSON via `snapshot()`)
//! - Prometheus text exposition format (via `to_prometheus()`)

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::SystemTime;

use serde::Serialize;

/// Node-wide metrics collection.
///
/// All counters use `AtomicU64` with `Ordering::Relaxed` — metrics do not need
/// sequential consistency, only eventual visibility.
#[derive(Debug)]
pub struct NodeMetrics {
    // Proof metrics
    pub proofs_generated: AtomicU64,
    pub proofs_verified: AtomicU64,
    pub proofs_failed: AtomicU64,
    /// Cumulative proof generation time in milliseconds.
    pub proof_generation_time_ms: AtomicU64,

    // Training metrics
    pub training_steps_completed: AtomicU64,
    pub training_rounds_completed: AtomicU64,
    /// Current loss stored as f64 bits (use `f64::from_bits` to read).
    pub current_loss: AtomicU64,

    // Network metrics
    pub rpc_requests_served: AtomicU64,
    pub rpc_errors: AtomicU64,
    pub active_peer_connections: AtomicU64,
    pub messages_sent: AtomicU64,
    pub messages_received: AtomicU64,

    // MPC metrics
    pub mpc_sessions_active: AtomicU64,
    pub mpc_sessions_completed: AtomicU64,
    pub mpc_sessions_failed: AtomicU64,

    // On-chain metrics
    pub onchain_submissions: AtomicU64,
    pub onchain_submissions_failed: AtomicU64,
    /// Cumulative gas used across all on-chain submissions.
    pub onchain_gas_used: AtomicU64,

    // System
    pub uptime_seconds: AtomicU64,
    pub start_time: SystemTime,
}

impl Default for NodeMetrics {
    fn default() -> Self {
        Self {
            proofs_generated: AtomicU64::new(0),
            proofs_verified: AtomicU64::new(0),
            proofs_failed: AtomicU64::new(0),
            proof_generation_time_ms: AtomicU64::new(0),
            training_steps_completed: AtomicU64::new(0),
            training_rounds_completed: AtomicU64::new(0),
            current_loss: AtomicU64::new(0),
            rpc_requests_served: AtomicU64::new(0),
            rpc_errors: AtomicU64::new(0),
            active_peer_connections: AtomicU64::new(0),
            messages_sent: AtomicU64::new(0),
            messages_received: AtomicU64::new(0),
            mpc_sessions_active: AtomicU64::new(0),
            mpc_sessions_completed: AtomicU64::new(0),
            mpc_sessions_failed: AtomicU64::new(0),
            onchain_submissions: AtomicU64::new(0),
            onchain_submissions_failed: AtomicU64::new(0),
            onchain_gas_used: AtomicU64::new(0),
            uptime_seconds: AtomicU64::new(0),
            start_time: SystemTime::now(),
        }
    }
}

impl NodeMetrics {
    /// Creates a new metrics instance wrapped in `Arc` for shared ownership.
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    // -- Proof metrics -------------------------------------------------------

    /// Records a successful proof generation with its duration.
    pub fn record_proof_generated(&self, duration_ms: u64) {
        self.proofs_generated.fetch_add(1, Ordering::Relaxed);
        self.proof_generation_time_ms
            .fetch_add(duration_ms, Ordering::Relaxed);
    }

    /// Records a successful proof verification.
    pub fn record_proof_verified(&self) {
        self.proofs_verified.fetch_add(1, Ordering::Relaxed);
    }

    /// Records a failed proof verification.
    pub fn record_proof_failed(&self) {
        self.proofs_failed.fetch_add(1, Ordering::Relaxed);
    }

    // -- Training metrics ----------------------------------------------------

    /// Records a completed training step with its loss value.
    pub fn record_training_step(&self, loss: f64) {
        self.training_steps_completed
            .fetch_add(1, Ordering::Relaxed);
        self.current_loss
            .store(loss.to_bits(), Ordering::Relaxed);
    }

    /// Records a completed training round.
    pub fn record_training_round(&self) {
        self.training_rounds_completed
            .fetch_add(1, Ordering::Relaxed);
    }

    // -- Network metrics -----------------------------------------------------

    /// Records a served RPC request.
    pub fn record_rpc_request(&self) {
        self.rpc_requests_served.fetch_add(1, Ordering::Relaxed);
    }

    /// Records an RPC error.
    pub fn record_rpc_error(&self) {
        self.rpc_errors.fetch_add(1, Ordering::Relaxed);
    }

    /// Records a new peer connection.
    pub fn record_peer_connected(&self) {
        self.active_peer_connections
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Records a peer disconnection.
    pub fn record_peer_disconnected(&self) {
        // Saturating subtract: if already 0, stay at 0.
        let _ = self
            .active_peer_connections
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                Some(v.saturating_sub(1))
            });
    }

    /// Records a sent network message.
    pub fn record_message_sent(&self) {
        self.messages_sent.fetch_add(1, Ordering::Relaxed);
    }

    /// Records a received network message.
    pub fn record_message_received(&self) {
        self.messages_received.fetch_add(1, Ordering::Relaxed);
    }

    // -- MPC metrics ---------------------------------------------------------

    /// Records that an MPC session has started.
    pub fn record_mpc_session_started(&self) {
        self.mpc_sessions_active.fetch_add(1, Ordering::Relaxed);
    }

    /// Records that an MPC session completed successfully.
    pub fn record_mpc_session_completed(&self) {
        self.mpc_sessions_completed
            .fetch_add(1, Ordering::Relaxed);
        let _ = self
            .mpc_sessions_active
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                Some(v.saturating_sub(1))
            });
    }

    /// Records that an MPC session failed.
    pub fn record_mpc_session_failed(&self) {
        self.mpc_sessions_failed.fetch_add(1, Ordering::Relaxed);
        let _ = self
            .mpc_sessions_active
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                Some(v.saturating_sub(1))
            });
    }

    // -- On-chain metrics ----------------------------------------------------

    /// Records a successful on-chain proof submission.
    pub fn record_onchain_submission(&self, gas: u64) {
        self.onchain_submissions.fetch_add(1, Ordering::Relaxed);
        self.onchain_gas_used.fetch_add(gas, Ordering::Relaxed);
    }

    /// Records a failed on-chain submission.
    pub fn record_onchain_failure(&self) {
        self.onchain_submissions_failed
            .fetch_add(1, Ordering::Relaxed);
    }

    // -- Snapshots -----------------------------------------------------------

    /// Creates a serializable snapshot of all current metrics.
    pub fn snapshot(&self) -> NodeMetricsSnapshot {
        let proofs_generated = self.proofs_generated.load(Ordering::Relaxed);
        let proof_generation_time_ms = self.proof_generation_time_ms.load(Ordering::Relaxed);

        let avg_proof_time_ms = if proofs_generated > 0 {
            proof_generation_time_ms as f64 / proofs_generated as f64
        } else {
            0.0
        };

        let uptime_seconds = self
            .start_time
            .elapsed()
            .map(|d| d.as_secs())
            .unwrap_or(0);

        NodeMetricsSnapshot {
            // Proof
            proofs_generated,
            proofs_verified: self.proofs_verified.load(Ordering::Relaxed),
            proofs_failed: self.proofs_failed.load(Ordering::Relaxed),
            proof_generation_time_ms,
            avg_proof_time_ms,

            // Training
            training_steps_completed: self.training_steps_completed.load(Ordering::Relaxed),
            training_rounds_completed: self.training_rounds_completed.load(Ordering::Relaxed),
            current_loss: f64::from_bits(self.current_loss.load(Ordering::Relaxed)),

            // Network
            rpc_requests_served: self.rpc_requests_served.load(Ordering::Relaxed),
            rpc_errors: self.rpc_errors.load(Ordering::Relaxed),
            active_peer_connections: self.active_peer_connections.load(Ordering::Relaxed),
            messages_sent: self.messages_sent.load(Ordering::Relaxed),
            messages_received: self.messages_received.load(Ordering::Relaxed),

            // MPC
            mpc_sessions_active: self.mpc_sessions_active.load(Ordering::Relaxed),
            mpc_sessions_completed: self.mpc_sessions_completed.load(Ordering::Relaxed),
            mpc_sessions_failed: self.mpc_sessions_failed.load(Ordering::Relaxed),

            // On-chain
            onchain_submissions: self.onchain_submissions.load(Ordering::Relaxed),
            onchain_submissions_failed: self.onchain_submissions_failed.load(Ordering::Relaxed),
            onchain_gas_used: self.onchain_gas_used.load(Ordering::Relaxed),

            // System
            uptime_seconds,
        }
    }

    /// Renders all metrics in Prometheus text exposition format.
    ///
    /// Output follows <https://prometheus.io/docs/instrumenting/exposition_formats/>.
    pub fn to_prometheus(&self) -> String {
        let snap = self.snapshot();
        let mut out = String::with_capacity(2048);

        // Proof metrics
        push_counter(
            &mut out,
            "helix_proofs_generated_total",
            "Total number of proofs generated",
            snap.proofs_generated,
        );
        push_counter(
            &mut out,
            "helix_proofs_verified_total",
            "Total number of proofs verified",
            snap.proofs_verified,
        );
        push_counter(
            &mut out,
            "helix_proofs_failed_total",
            "Total number of proofs that failed verification",
            snap.proofs_failed,
        );
        push_counter(
            &mut out,
            "helix_proof_generation_time_ms_total",
            "Cumulative proof generation time in milliseconds",
            snap.proof_generation_time_ms,
        );
        push_gauge_f64(
            &mut out,
            "helix_proof_generation_time_ms_avg",
            "Average proof generation time in milliseconds",
            snap.avg_proof_time_ms,
        );

        // Training metrics
        push_counter(
            &mut out,
            "helix_training_steps_completed_total",
            "Total training steps completed",
            snap.training_steps_completed,
        );
        push_counter(
            &mut out,
            "helix_training_rounds_completed_total",
            "Total training rounds completed",
            snap.training_rounds_completed,
        );
        push_gauge_f64(
            &mut out,
            "helix_current_loss",
            "Current training loss value",
            snap.current_loss,
        );

        // Network metrics
        push_counter(
            &mut out,
            "helix_rpc_requests_served_total",
            "Total RPC requests served",
            snap.rpc_requests_served,
        );
        push_counter(
            &mut out,
            "helix_rpc_errors_total",
            "Total RPC errors",
            snap.rpc_errors,
        );
        push_gauge(
            &mut out,
            "helix_active_peer_connections",
            "Number of currently active peer connections",
            snap.active_peer_connections,
        );
        push_counter(
            &mut out,
            "helix_messages_sent_total",
            "Total network messages sent",
            snap.messages_sent,
        );
        push_counter(
            &mut out,
            "helix_messages_received_total",
            "Total network messages received",
            snap.messages_received,
        );

        // MPC metrics
        push_gauge(
            &mut out,
            "helix_mpc_sessions_active",
            "Number of currently active MPC sessions",
            snap.mpc_sessions_active,
        );
        push_counter(
            &mut out,
            "helix_mpc_sessions_completed_total",
            "Total MPC sessions completed successfully",
            snap.mpc_sessions_completed,
        );
        push_counter(
            &mut out,
            "helix_mpc_sessions_failed_total",
            "Total MPC sessions that failed",
            snap.mpc_sessions_failed,
        );

        // On-chain metrics
        push_counter(
            &mut out,
            "helix_onchain_submissions_total",
            "Total on-chain proof submissions",
            snap.onchain_submissions,
        );
        push_counter(
            &mut out,
            "helix_onchain_submissions_failed_total",
            "Total failed on-chain submissions",
            snap.onchain_submissions_failed,
        );
        push_counter(
            &mut out,
            "helix_onchain_gas_used_total",
            "Cumulative gas used for on-chain submissions",
            snap.onchain_gas_used,
        );

        // System metrics
        push_gauge(
            &mut out,
            "helix_uptime_seconds",
            "Node uptime in seconds",
            snap.uptime_seconds,
        );

        out
    }
}

// ---------------------------------------------------------------------------
// Serializable snapshot
// ---------------------------------------------------------------------------

/// A point-in-time snapshot of all node metrics, suitable for JSON serialization.
#[derive(Debug, Clone, Serialize)]
pub struct NodeMetricsSnapshot {
    // Proof
    pub proofs_generated: u64,
    pub proofs_verified: u64,
    pub proofs_failed: u64,
    pub proof_generation_time_ms: u64,
    pub avg_proof_time_ms: f64,

    // Training
    pub training_steps_completed: u64,
    pub training_rounds_completed: u64,
    pub current_loss: f64,

    // Network
    pub rpc_requests_served: u64,
    pub rpc_errors: u64,
    pub active_peer_connections: u64,
    pub messages_sent: u64,
    pub messages_received: u64,

    // MPC
    pub mpc_sessions_active: u64,
    pub mpc_sessions_completed: u64,
    pub mpc_sessions_failed: u64,

    // On-chain
    pub onchain_submissions: u64,
    pub onchain_submissions_failed: u64,
    pub onchain_gas_used: u64,

    // System
    pub uptime_seconds: u64,
}

// ---------------------------------------------------------------------------
// Prometheus text format helpers
// ---------------------------------------------------------------------------

fn push_counter(out: &mut String, name: &str, help: &str, value: u64) {
    out.push_str("# HELP ");
    out.push_str(name);
    out.push(' ');
    out.push_str(help);
    out.push('\n');
    out.push_str("# TYPE ");
    out.push_str(name);
    out.push_str(" counter\n");
    out.push_str(name);
    out.push(' ');
    out.push_str(&value.to_string());
    out.push('\n');
}

fn push_gauge(out: &mut String, name: &str, help: &str, value: u64) {
    out.push_str("# HELP ");
    out.push_str(name);
    out.push(' ');
    out.push_str(help);
    out.push('\n');
    out.push_str("# TYPE ");
    out.push_str(name);
    out.push_str(" gauge\n");
    out.push_str(name);
    out.push(' ');
    out.push_str(&value.to_string());
    out.push('\n');
}

fn push_gauge_f64(out: &mut String, name: &str, help: &str, value: f64) {
    out.push_str("# HELP ");
    out.push_str(name);
    out.push(' ');
    out.push_str(help);
    out.push('\n');
    out.push_str("# TYPE ");
    out.push_str(name);
    out.push_str(" gauge\n");
    out.push_str(name);
    out.push(' ');
    // Use enough precision for meaningful values without excessive noise.
    out.push_str(&format!("{:.6}", value));
    out.push('\n');
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_increment() {
        let metrics = NodeMetrics::new();

        metrics.record_proof_generated(100);
        metrics.record_proof_generated(200);
        metrics.record_proof_verified();
        metrics.record_proof_verified();
        metrics.record_proof_verified();
        metrics.record_proof_failed();

        assert_eq!(metrics.proofs_generated.load(Ordering::Relaxed), 2);
        assert_eq!(
            metrics.proof_generation_time_ms.load(Ordering::Relaxed),
            300
        );
        assert_eq!(metrics.proofs_verified.load(Ordering::Relaxed), 3);
        assert_eq!(metrics.proofs_failed.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_training_step_loss() {
        let metrics = NodeMetrics::new();

        metrics.record_training_step(0.5);
        assert_eq!(
            metrics.training_steps_completed.load(Ordering::Relaxed),
            1
        );
        let stored_loss = f64::from_bits(metrics.current_loss.load(Ordering::Relaxed));
        assert!((stored_loss - 0.5).abs() < f64::EPSILON);

        // Second step overwrites the loss
        metrics.record_training_step(0.25);
        assert_eq!(
            metrics.training_steps_completed.load(Ordering::Relaxed),
            2
        );
        let stored_loss = f64::from_bits(metrics.current_loss.load(Ordering::Relaxed));
        assert!((stored_loss - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn test_peer_connect_disconnect() {
        let metrics = NodeMetrics::new();

        metrics.record_peer_connected();
        metrics.record_peer_connected();
        metrics.record_peer_connected();
        assert_eq!(
            metrics.active_peer_connections.load(Ordering::Relaxed),
            3
        );

        metrics.record_peer_disconnected();
        assert_eq!(
            metrics.active_peer_connections.load(Ordering::Relaxed),
            2
        );

        // Disconnect all and one more (should saturate at 0)
        metrics.record_peer_disconnected();
        metrics.record_peer_disconnected();
        metrics.record_peer_disconnected(); // extra
        assert_eq!(
            metrics.active_peer_connections.load(Ordering::Relaxed),
            0
        );
    }

    #[test]
    fn test_mpc_session_lifecycle() {
        let metrics = NodeMetrics::new();

        metrics.record_mpc_session_started();
        metrics.record_mpc_session_started();
        assert_eq!(metrics.mpc_sessions_active.load(Ordering::Relaxed), 2);

        metrics.record_mpc_session_completed();
        assert_eq!(metrics.mpc_sessions_active.load(Ordering::Relaxed), 1);
        assert_eq!(
            metrics.mpc_sessions_completed.load(Ordering::Relaxed),
            1
        );

        metrics.record_mpc_session_failed();
        assert_eq!(metrics.mpc_sessions_active.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.mpc_sessions_failed.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_onchain_metrics() {
        let metrics = NodeMetrics::new();

        metrics.record_onchain_submission(7_500_000);
        metrics.record_onchain_submission(8_000_000);
        metrics.record_onchain_failure();

        assert_eq!(metrics.onchain_submissions.load(Ordering::Relaxed), 2);
        assert_eq!(
            metrics.onchain_gas_used.load(Ordering::Relaxed),
            15_500_000
        );
        assert_eq!(
            metrics.onchain_submissions_failed.load(Ordering::Relaxed),
            1
        );
    }

    #[test]
    fn test_metrics_snapshot() {
        let metrics = NodeMetrics::new();

        metrics.record_proof_generated(100);
        metrics.record_proof_generated(300);
        metrics.record_proof_verified();
        metrics.record_proof_failed();
        metrics.record_training_step(0.42);
        metrics.record_training_round();
        metrics.record_rpc_request();
        metrics.record_rpc_request();
        metrics.record_rpc_error();
        metrics.record_peer_connected();
        metrics.record_message_sent();
        metrics.record_message_received();
        metrics.record_message_received();
        metrics.record_mpc_session_started();
        metrics.record_mpc_session_completed();
        metrics.record_onchain_submission(5_000_000);

        let snap = metrics.snapshot();

        assert_eq!(snap.proofs_generated, 2);
        assert_eq!(snap.proofs_verified, 1);
        assert_eq!(snap.proofs_failed, 1);
        assert_eq!(snap.proof_generation_time_ms, 400);
        assert!((snap.avg_proof_time_ms - 200.0).abs() < f64::EPSILON);
        assert_eq!(snap.training_steps_completed, 1);
        assert_eq!(snap.training_rounds_completed, 1);
        assert!((snap.current_loss - 0.42).abs() < f64::EPSILON);
        assert_eq!(snap.rpc_requests_served, 2);
        assert_eq!(snap.rpc_errors, 1);
        assert_eq!(snap.active_peer_connections, 1);
        assert_eq!(snap.messages_sent, 1);
        assert_eq!(snap.messages_received, 2);
        assert_eq!(snap.mpc_sessions_active, 0);
        assert_eq!(snap.mpc_sessions_completed, 1);
        assert_eq!(snap.mpc_sessions_failed, 0);
        assert_eq!(snap.onchain_submissions, 1);
        assert_eq!(snap.onchain_gas_used, 5_000_000);
        // Uptime should be very small (< 2 seconds since we just created it)
        assert!(snap.uptime_seconds < 2);
    }

    #[test]
    fn test_prometheus_format() {
        let metrics = NodeMetrics::new();

        metrics.record_proof_generated(500);
        metrics.record_proof_verified();
        metrics.record_proof_verified();
        metrics.record_training_step(0.123);
        metrics.record_rpc_request();
        metrics.record_peer_connected();
        metrics.record_mpc_session_started();
        metrics.record_onchain_submission(7_500_000);

        let output = metrics.to_prometheus();

        // Verify counter format
        assert!(output.contains("# HELP helix_proofs_generated_total Total number of proofs generated"));
        assert!(output.contains("# TYPE helix_proofs_generated_total counter"));
        assert!(output.contains("helix_proofs_generated_total 1"));

        assert!(output.contains("# TYPE helix_proofs_verified_total counter"));
        assert!(output.contains("helix_proofs_verified_total 2"));

        assert!(output.contains("# TYPE helix_proofs_failed_total counter"));
        assert!(output.contains("helix_proofs_failed_total 0"));

        // Verify gauge format
        assert!(output.contains("# TYPE helix_active_peer_connections gauge"));
        assert!(output.contains("helix_active_peer_connections 1"));

        assert!(output.contains("# TYPE helix_mpc_sessions_active gauge"));
        assert!(output.contains("helix_mpc_sessions_active 1"));

        // Verify f64 gauge
        assert!(output.contains("# TYPE helix_current_loss gauge"));
        assert!(output.contains("helix_current_loss 0.123000"));

        // Verify cumulative time
        assert!(output.contains("helix_proof_generation_time_ms_total 500"));

        // Verify on-chain
        assert!(output.contains("helix_onchain_submissions_total 1"));
        assert!(output.contains("helix_onchain_gas_used_total 7500000"));

        // Verify uptime gauge
        assert!(output.contains("# TYPE helix_uptime_seconds gauge"));

        // Every metric line should be properly formatted (no empty lines within a metric block)
        for line in output.lines() {
            assert!(
                line.starts_with('#') || line.starts_with("helix_") || line.is_empty(),
                "Unexpected line format: '{}'",
                line,
            );
        }
    }

    #[test]
    fn test_prometheus_avg_proof_time_zero_proofs() {
        let metrics = NodeMetrics::new();
        let output = metrics.to_prometheus();

        // With zero proofs, average should be 0.0
        assert!(output.contains("helix_proof_generation_time_ms_avg 0.000000"));
    }

    #[test]
    fn test_concurrent_metrics() {
        use std::sync::Arc;
        use std::thread;

        let metrics = NodeMetrics::new();
        let num_threads = 8;
        let ops_per_thread = 1000;

        let mut handles = Vec::new();
        for _ in 0..num_threads {
            let m = Arc::clone(&metrics);
            handles.push(thread::spawn(move || {
                for _ in 0..ops_per_thread {
                    m.record_proof_generated(10);
                    m.record_proof_verified();
                    m.record_rpc_request();
                    m.record_message_sent();
                    m.record_message_received();
                }
            }));
        }

        for h in handles {
            h.join().expect("thread panicked");
        }

        let expected = (num_threads * ops_per_thread) as u64;
        assert_eq!(
            metrics.proofs_generated.load(Ordering::Relaxed),
            expected
        );
        assert_eq!(
            metrics.proofs_verified.load(Ordering::Relaxed),
            expected
        );
        assert_eq!(
            metrics.rpc_requests_served.load(Ordering::Relaxed),
            expected
        );
        assert_eq!(
            metrics.messages_sent.load(Ordering::Relaxed),
            expected
        );
        assert_eq!(
            metrics.messages_received.load(Ordering::Relaxed),
            expected
        );
        assert_eq!(
            metrics
                .proof_generation_time_ms
                .load(Ordering::Relaxed),
            expected * 10
        );
    }

    #[test]
    fn test_snapshot_serializes_to_json() {
        let metrics = NodeMetrics::new();
        metrics.record_proof_generated(42);
        metrics.record_training_step(1.5);

        let snap = metrics.snapshot();
        let json = serde_json::to_string(&snap).expect("snapshot should serialize to JSON");

        assert!(json.contains("\"proofs_generated\":1"));
        assert!(json.contains("\"current_loss\":1.5"));
        assert!(json.contains("\"avg_proof_time_ms\":42.0"));
    }
}
