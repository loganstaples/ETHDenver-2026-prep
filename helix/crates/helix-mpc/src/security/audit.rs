//! Audit trail for MPC operations.
//!
//! Records all significant MPC events (share distribution, openings,
//! multiplications, re-sharings) for post-hoc verification and
//! dispute resolution.

use std::collections::VecDeque;
use serde::{Deserialize, Serialize};
use sha2::{Sha256, Digest};

use crate::types::{MPCPhase, PartyId};

/// A single audit event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    /// Monotonically increasing event ID.
    pub event_id: u64,
    /// When this event occurred (training step).
    pub step: u64,
    /// MPC phase at time of event.
    pub phase: MPCPhase,
    /// Type of event.
    pub event_type: AuditEventType,
    /// Parties involved.
    pub parties: Vec<PartyId>,
    /// Hash of the data involved (for verification without storing data).
    pub data_hash: [u8; 32],
    /// Human-readable description.
    pub description: String,
}

/// Types of auditable MPC events.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditEventType {
    /// Shares were distributed to parties.
    ShareDistribution {
        secret_id: String,
        num_shares: usize,
    },
    /// A value was opened (reconstructed from shares).
    ValueOpening {
        description: String,
    },
    /// A Beaver triple was consumed for multiplication.
    BeaverConsumption {
        operation: String,
    },
    /// Shares were refreshed via re-sharing.
    Resharing {
        secret_id: String,
    },
    /// A commitment was created.
    CommitmentCreated {
        party: PartyId,
        description: String,
    },
    /// A commitment was verified.
    CommitmentVerified {
        party: PartyId,
        success: bool,
    },
    /// Malicious behavior was detected.
    MaliciousDetected {
        party: PartyId,
        description: String,
    },
    /// A training step was completed.
    TrainingStepComplete {
        step: u64,
        loss: Option<f64>,
    },
    /// Session phase transition.
    PhaseTransition {
        from: MPCPhase,
        to: MPCPhase,
    },
    /// Gradient aggregation completed.
    GradientAggregation {
        num_contributors: usize,
        num_excluded: usize,
    },
}

/// Append-only audit log for MPC operations.
#[derive(Debug)]
pub struct AuditLog {
    /// All events in chronological order.
    events: VecDeque<AuditEvent>,
    /// Next event ID.
    next_id: u64,
    /// Current training step.
    current_step: u64,
    /// Current MPC phase.
    current_phase: MPCPhase,
    /// Maximum number of events to keep (0 = unlimited).
    max_events: usize,
    /// Running hash chain for tamper detection.
    chain_hash: [u8; 32],
}

impl AuditLog {
    /// Creates a new audit log.
    pub fn new() -> Self {
        Self {
            events: VecDeque::new(),
            next_id: 0,
            current_step: 0,
            current_phase: MPCPhase::Preprocessing,
            max_events: 0,
            chain_hash: [0u8; 32],
        }
    }

    /// Creates a log with a maximum event retention count.
    pub fn with_capacity(max_events: usize) -> Self {
        Self {
            max_events,
            ..Self::new()
        }
    }

    /// Records an event.
    pub fn record(
        &mut self,
        event_type: AuditEventType,
        parties: Vec<PartyId>,
        data: &[u8],
        description: impl Into<String>,
    ) {
        let data_hash = {
            let mut hasher = Sha256::new();
            hasher.update(data);
            hasher.finalize().into()
        };

        // Update chain hash: H(prev_hash || event_id || data_hash).
        let chain_hash = {
            let mut hasher = Sha256::new();
            hasher.update(&self.chain_hash);
            hasher.update(self.next_id.to_le_bytes());
            hasher.update(&data_hash);
            hasher.finalize().into()
        };
        self.chain_hash = chain_hash;

        let event = AuditEvent {
            event_id: self.next_id,
            step: self.current_step,
            phase: self.current_phase,
            event_type,
            parties,
            data_hash,
            description: description.into(),
        };

        self.events.push_back(event);
        self.next_id += 1;

        // Trim if over capacity.
        if self.max_events > 0 && self.events.len() > self.max_events {
            self.events.pop_front();
        }
    }

    /// Records a share distribution event.
    pub fn record_distribution(
        &mut self,
        secret_id: &str,
        num_shares: usize,
        parties: &[PartyId],
    ) {
        self.record(
            AuditEventType::ShareDistribution {
                secret_id: secret_id.to_string(),
                num_shares,
            },
            parties.to_vec(),
            secret_id.as_bytes(),
            format!("Distributed {} shares of '{}'", num_shares, secret_id),
        );
    }

    /// Records a value opening event.
    pub fn record_opening(
        &mut self,
        description: &str,
        parties: &[PartyId],
    ) {
        self.record(
            AuditEventType::ValueOpening {
                description: description.to_string(),
            },
            parties.to_vec(),
            description.as_bytes(),
            format!("Opened value: {}", description),
        );
    }

    /// Records a Beaver triple consumption.
    pub fn record_beaver_consumption(&mut self, operation: &str) {
        self.record(
            AuditEventType::BeaverConsumption {
                operation: operation.to_string(),
            },
            vec![],
            operation.as_bytes(),
            format!("Consumed Beaver triple for: {}", operation),
        );
    }

    /// Records a re-sharing event.
    pub fn record_resharing(&mut self, secret_id: &str, parties: &[PartyId]) {
        self.record(
            AuditEventType::Resharing {
                secret_id: secret_id.to_string(),
            },
            parties.to_vec(),
            secret_id.as_bytes(),
            format!("Re-shared '{}'", secret_id),
        );
    }

    /// Records malicious behavior detection.
    pub fn record_malicious(
        &mut self,
        party: &PartyId,
        description: &str,
    ) {
        self.record(
            AuditEventType::MaliciousDetected {
                party: party.clone(),
                description: description.to_string(),
            },
            vec![party.clone()],
            description.as_bytes(),
            format!("MALICIOUS: party {} - {}", party, description),
        );
    }

    /// Records a training step completion.
    pub fn record_training_step(&mut self, step: u64, loss: Option<f64>) {
        self.current_step = step;
        self.record(
            AuditEventType::TrainingStepComplete { step, loss },
            vec![],
            &step.to_le_bytes(),
            format!(
                "Training step {} complete{}",
                step,
                loss.map_or(String::new(), |l| format!(", loss={:.4}", l)),
            ),
        );
    }

    /// Records a phase transition.
    pub fn record_phase_transition(&mut self, from: MPCPhase, to: MPCPhase) {
        self.current_phase = to;
        self.record(
            AuditEventType::PhaseTransition { from, to },
            vec![],
            &[],
            format!("Phase: {} → {}", from, to),
        );
    }

    /// Sets the current step (for event timestamps).
    pub fn set_step(&mut self, step: u64) {
        self.current_step = step;
    }

    /// Returns all events.
    pub fn events(&self) -> &VecDeque<AuditEvent> {
        &self.events
    }

    /// Returns events of a specific type.
    pub fn events_by_type(&self, filter: &str) -> Vec<&AuditEvent> {
        self.events
            .iter()
            .filter(|e| {
                let type_name = match &e.event_type {
                    AuditEventType::ShareDistribution { .. } => "distribution",
                    AuditEventType::ValueOpening { .. } => "opening",
                    AuditEventType::BeaverConsumption { .. } => "beaver",
                    AuditEventType::Resharing { .. } => "resharing",
                    AuditEventType::CommitmentCreated { .. } => "commitment_created",
                    AuditEventType::CommitmentVerified { .. } => "commitment_verified",
                    AuditEventType::MaliciousDetected { .. } => "malicious",
                    AuditEventType::TrainingStepComplete { .. } => "training_step",
                    AuditEventType::PhaseTransition { .. } => "phase_transition",
                    AuditEventType::GradientAggregation { .. } => "aggregation",
                };
                type_name == filter
            })
            .collect()
    }

    /// Returns events involving a specific party.
    pub fn events_for_party(&self, party: &PartyId) -> Vec<&AuditEvent> {
        self.events
            .iter()
            .filter(|e| e.parties.contains(party))
            .collect()
    }

    /// Returns the number of events.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Returns true if empty.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Returns the chain hash (for tamper detection).
    pub fn chain_hash(&self) -> [u8; 32] {
        self.chain_hash
    }

    /// Generates a summary of the audit log.
    pub fn summary(&self) -> AuditSummary {
        let mut distributions = 0;
        let mut openings = 0;
        let mut beaver_ops = 0;
        let mut resharings = 0;
        let mut malicious_events = 0;
        let mut training_steps = 0;

        for event in &self.events {
            match &event.event_type {
                AuditEventType::ShareDistribution { .. } => distributions += 1,
                AuditEventType::ValueOpening { .. } => openings += 1,
                AuditEventType::BeaverConsumption { .. } => beaver_ops += 1,
                AuditEventType::Resharing { .. } => resharings += 1,
                AuditEventType::MaliciousDetected { .. } => malicious_events += 1,
                AuditEventType::TrainingStepComplete { .. } => training_steps += 1,
                _ => {}
            }
        }

        AuditSummary {
            total_events: self.events.len(),
            distributions,
            openings,
            beaver_ops,
            resharings,
            malicious_events,
            training_steps,
            chain_hash: self.chain_hash,
        }
    }
}

impl Default for AuditLog {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of audit log contents.
#[derive(Debug, Clone)]
pub struct AuditSummary {
    pub total_events: usize,
    pub distributions: usize,
    pub openings: usize,
    pub beaver_ops: usize,
    pub resharings: usize,
    pub malicious_events: usize,
    pub training_steps: usize,
    pub chain_hash: [u8; 32],
}

impl std::fmt::Display for AuditSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "AuditLog: {} events ({} distributions, {} openings, {} beaver ops, {} resharings, {} malicious, {} steps)",
            self.total_events,
            self.distributions,
            self.openings,
            self.beaver_ops,
            self.resharings,
            self.malicious_events,
            self.training_steps,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audit_log_basic() {
        let mut log = AuditLog::new();
        let parties: Vec<PartyId> = (0..3).map(PartyId::from_index).collect();

        log.record_distribution("weights", 3, &parties);
        log.record_opening("activation", &parties);
        log.record_beaver_consumption("matmul");
        log.record_training_step(1, Some(2.5));

        assert_eq!(log.len(), 4);
        assert!(!log.is_empty());

        let summary = log.summary();
        assert_eq!(summary.distributions, 1);
        assert_eq!(summary.openings, 1);
        assert_eq!(summary.beaver_ops, 1);
        assert_eq!(summary.training_steps, 1);
    }

    #[test]
    fn test_audit_chain_hash() {
        let mut log = AuditLog::new();
        let initial = log.chain_hash();

        log.record_beaver_consumption("mul1");
        let after_first = log.chain_hash();
        assert_ne!(initial, after_first);

        log.record_beaver_consumption("mul2");
        let after_second = log.chain_hash();
        assert_ne!(after_first, after_second);
    }

    #[test]
    fn test_audit_capacity() {
        let mut log = AuditLog::with_capacity(3);

        for i in 0..5 {
            log.record_training_step(i, None);
        }

        assert_eq!(log.len(), 3); // Oldest 2 were trimmed.
    }

    #[test]
    fn test_events_for_party() {
        let mut log = AuditLog::new();
        let p0 = PartyId::from_index(0);
        let p1 = PartyId::from_index(1);

        log.record_distribution("w1", 2, &[p0.clone(), p1.clone()]);
        log.record_malicious(&p1, "bad gradient");

        let p0_events = log.events_for_party(&p0);
        assert_eq!(p0_events.len(), 1);

        let p1_events = log.events_for_party(&p1);
        assert_eq!(p1_events.len(), 2);
    }

    #[test]
    fn test_events_by_type() {
        let mut log = AuditLog::new();
        log.record_beaver_consumption("mul1");
        log.record_beaver_consumption("mul2");
        log.record_training_step(1, None);

        let beaver = log.events_by_type("beaver");
        assert_eq!(beaver.len(), 2);

        let steps = log.events_by_type("training_step");
        assert_eq!(steps.len(), 1);
    }
}
