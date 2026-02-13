//! Participation Tracking.
//!
//! Records per-round participation history, earnings, and reputation metrics.
//! Used for RPC reporting and auto-join profitability decisions.

use serde::{Deserialize, Serialize};
use std::time::SystemTime;

/// Record of participation in a single training round.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundRecord {
    /// Round ID.
    pub round_id: u64,
    /// Model ID.
    pub model_id: u64,
    /// Unix timestamp when the worker joined the round.
    pub joined_at: u64,
    /// Unix timestamp when the round completed (0 if still active or failed).
    pub completed_at: u64,
    /// Number of training steps completed.
    pub steps_completed: u32,
    /// Number of proofs generated and submitted.
    pub proofs_submitted: u32,
    /// Whether the round completed successfully.
    pub success: bool,
    /// Error bound achieved.
    pub error_bound: f64,
    /// Loss at the end of training.
    pub final_loss: f64,
    /// Estimated earnings from this round (in wei).
    pub earnings_wei: u64,
    /// SHA-256 hash of the proof submitted (first 16 bytes, hex).
    pub proof_hash: String,
}

impl RoundRecord {
    /// Creates a new round record for a round being joined now.
    pub fn new_joining(round_id: u64, model_id: u64) -> Self {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            round_id,
            model_id,
            joined_at: now,
            completed_at: 0,
            steps_completed: 0,
            proofs_submitted: 0,
            success: false,
            error_bound: 0.0,
            final_loss: 0.0,
            earnings_wei: 0,
            proof_hash: String::new(),
        }
    }

    /// Marks this record as completed.
    pub fn complete(
        &mut self,
        steps: u32,
        proofs: u32,
        error_bound: f64,
        loss: f64,
        proof_hash: String,
    ) {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.completed_at = now;
        self.steps_completed = steps;
        self.proofs_submitted = proofs;
        self.success = true;
        self.error_bound = error_bound;
        self.final_loss = loss;
        self.proof_hash = proof_hash;
    }

    /// Marks this record as failed.
    pub fn fail(&mut self) {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.completed_at = now;
        self.success = false;
    }

    /// Duration of participation in seconds (0 if not completed).
    pub fn duration_secs(&self) -> u64 {
        if self.completed_at > 0 {
            self.completed_at.saturating_sub(self.joined_at)
        } else {
            0
        }
    }
}

/// Summary of a worker's earnings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarningsSummary {
    /// Total earnings across all rounds (in wei).
    pub total_earnings_wei: u64,
    /// Number of rounds participated in.
    pub rounds_participated: u64,
    /// Number of rounds completed successfully.
    pub rounds_succeeded: u64,
    /// Number of rounds that failed.
    pub rounds_failed: u64,
    /// Total proofs submitted.
    pub total_proofs_submitted: u64,
    /// Total training steps completed.
    pub total_steps_completed: u64,
    /// Average earnings per successful round (in wei).
    pub avg_earnings_per_round_wei: u64,
    /// Success rate (0.0 to 1.0).
    pub success_rate: f64,
    /// Reputation score (0.0 to 1.0, based on success rate and consistency).
    pub reputation_score: f64,
}

/// Tracks participation history, earnings, and reputation.
#[derive(Debug)]
pub struct ParticipationTracker {
    /// All round records, ordered by join time.
    records: Vec<RoundRecord>,
    /// Maximum records to keep (oldest evicted first).
    max_records: usize,
    /// Base reward per proof (in wei), used for earnings estimation.
    base_reward_per_proof_wei: u64,
}

impl ParticipationTracker {
    /// Creates a new tracker with the given capacity.
    pub fn new(max_records: usize) -> Self {
        Self {
            records: Vec::new(),
            max_records,
            base_reward_per_proof_wei: 1_000_000_000_000_000, // 0.001 ETH default
        }
    }

    /// Sets the base reward per proof for earnings estimation.
    pub fn set_base_reward(&mut self, reward_wei: u64) {
        self.base_reward_per_proof_wei = reward_wei;
    }

    /// Records joining a new round. Returns the index of the new record.
    pub fn record_join(&mut self, round_id: u64, model_id: u64) -> usize {
        // Evict oldest if at capacity
        if self.records.len() >= self.max_records {
            self.records.remove(0);
        }

        let record = RoundRecord::new_joining(round_id, model_id);
        self.records.push(record);
        self.records.len() - 1
    }

    /// Records successful completion of a round.
    pub fn record_completion(
        &mut self,
        round_id: u64,
        steps: u32,
        proofs: u32,
        error_bound: f64,
        loss: f64,
        proof_hash: String,
    ) {
        if let Some(record) = self.records.iter_mut().rev().find(|r| r.round_id == round_id) {
            record.complete(steps, proofs, error_bound, loss, proof_hash);
            // Estimate earnings
            record.earnings_wei = proofs as u64 * self.base_reward_per_proof_wei;
        }
    }

    /// Records failure of a round.
    pub fn record_failure(&mut self, round_id: u64) {
        if let Some(record) = self.records.iter_mut().rev().find(|r| r.round_id == round_id) {
            record.fail();
        }
    }

    /// Returns the earnings summary.
    pub fn earnings_summary(&self) -> EarningsSummary {
        let rounds_participated = self.records.len() as u64;
        let rounds_succeeded = self.records.iter().filter(|r| r.success).count() as u64;
        let rounds_failed = self.records.iter().filter(|r| r.completed_at > 0 && !r.success).count() as u64;

        let total_earnings_wei: u64 = self.records.iter().map(|r| r.earnings_wei).sum();
        let total_proofs: u64 = self.records.iter().map(|r| r.proofs_submitted as u64).sum();
        let total_steps: u64 = self.records.iter().map(|r| r.steps_completed as u64).sum();

        let avg_earnings = if rounds_succeeded > 0 {
            total_earnings_wei / rounds_succeeded
        } else {
            0
        };

        let success_rate = if rounds_participated > 0 {
            rounds_succeeded as f64 / rounds_participated as f64
        } else {
            0.0
        };

        // Reputation: weighted average of success rate and consistency
        // - Base: success_rate
        // - Bonus for more rounds participated (up to +0.1 for 100+ rounds)
        // - Penalty for recent failures
        let consistency_bonus = (rounds_participated as f64 / 100.0).min(0.1);
        let recent_failure_penalty = self.recent_failure_rate(10) * 0.2;
        let reputation = (success_rate + consistency_bonus - recent_failure_penalty)
            .clamp(0.0, 1.0);

        EarningsSummary {
            total_earnings_wei,
            rounds_participated,
            rounds_succeeded,
            rounds_failed,
            total_proofs_submitted: total_proofs,
            total_steps_completed: total_steps,
            avg_earnings_per_round_wei: avg_earnings,
            success_rate,
            reputation_score: reputation,
        }
    }

    /// Returns the records for a specific round.
    pub fn get_round_record(&self, round_id: u64) -> Option<&RoundRecord> {
        self.records.iter().rev().find(|r| r.round_id == round_id)
    }

    /// Returns all records.
    pub fn all_records(&self) -> &[RoundRecord] {
        &self.records
    }

    /// Returns the number of tracked rounds.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Returns whether the tracker has no records.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Failure rate in the last `n` completed rounds.
    fn recent_failure_rate(&self, n: usize) -> f64 {
        let completed: Vec<&RoundRecord> = self.records.iter()
            .rev()
            .filter(|r| r.completed_at > 0)
            .take(n)
            .collect();

        if completed.is_empty() {
            return 0.0;
        }

        let failures = completed.iter().filter(|r| !r.success).count();
        failures as f64 / completed.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_record_lifecycle() {
        let mut record = RoundRecord::new_joining(1, 100);
        assert_eq!(record.round_id, 1);
        assert_eq!(record.model_id, 100);
        assert!(record.joined_at > 0);
        assert_eq!(record.completed_at, 0);
        assert!(!record.success);

        record.complete(10, 5, 0.01, 0.5, "0xabc".to_string());
        assert!(record.success);
        assert!(record.completed_at > 0);
        assert_eq!(record.steps_completed, 10);
        assert_eq!(record.proofs_submitted, 5);
        assert_eq!(record.proof_hash, "0xabc");
    }

    #[test]
    fn test_round_record_fail() {
        let mut record = RoundRecord::new_joining(2, 100);
        record.fail();
        assert!(!record.success);
        assert!(record.completed_at > 0);
    }

    #[test]
    fn test_tracker_record_join() {
        let mut tracker = ParticipationTracker::new(100);
        let idx = tracker.record_join(1, 100);
        assert_eq!(idx, 0);
        assert_eq!(tracker.len(), 1);

        let record = tracker.get_round_record(1).unwrap();
        assert_eq!(record.round_id, 1);
        assert_eq!(record.model_id, 100);
    }

    #[test]
    fn test_tracker_record_completion() {
        let mut tracker = ParticipationTracker::new(100);
        tracker.record_join(1, 100);
        tracker.record_completion(1, 10, 5, 0.01, 0.5, "0xdef".to_string());

        let record = tracker.get_round_record(1).unwrap();
        assert!(record.success);
        assert_eq!(record.steps_completed, 10);
        assert_eq!(record.proofs_submitted, 5);
        assert!(record.earnings_wei > 0);
    }

    #[test]
    fn test_tracker_record_failure() {
        let mut tracker = ParticipationTracker::new(100);
        tracker.record_join(1, 100);
        tracker.record_failure(1);

        let record = tracker.get_round_record(1).unwrap();
        assert!(!record.success);
    }

    #[test]
    fn test_tracker_earnings_summary_empty() {
        let tracker = ParticipationTracker::new(100);
        let summary = tracker.earnings_summary();
        assert_eq!(summary.rounds_participated, 0);
        assert_eq!(summary.total_earnings_wei, 0);
        assert_eq!(summary.success_rate, 0.0);
    }

    #[test]
    fn test_tracker_earnings_summary_with_data() {
        let mut tracker = ParticipationTracker::new(100);
        tracker.set_base_reward(1_000_000);

        // Round 1: success
        tracker.record_join(1, 100);
        tracker.record_completion(1, 10, 3, 0.01, 0.5, "0xa".to_string());

        // Round 2: failure
        tracker.record_join(2, 100);
        tracker.record_failure(2);

        // Round 3: success
        tracker.record_join(3, 100);
        tracker.record_completion(3, 20, 5, 0.02, 0.3, "0xb".to_string());

        let summary = tracker.earnings_summary();
        assert_eq!(summary.rounds_participated, 3);
        assert_eq!(summary.rounds_succeeded, 2);
        assert_eq!(summary.rounds_failed, 1);
        assert_eq!(summary.total_proofs_submitted, 8);
        assert_eq!(summary.total_steps_completed, 30);
        assert_eq!(summary.total_earnings_wei, 8_000_000); // 8 proofs × 1M wei
        assert!((summary.success_rate - 2.0/3.0).abs() < 0.01);
        assert!(summary.reputation_score > 0.0);
    }

    #[test]
    fn test_tracker_evicts_oldest() {
        let mut tracker = ParticipationTracker::new(3);

        tracker.record_join(1, 100);
        tracker.record_join(2, 100);
        tracker.record_join(3, 100);
        assert_eq!(tracker.len(), 3);

        // Adding a 4th should evict the oldest
        tracker.record_join(4, 100);
        assert_eq!(tracker.len(), 3);
        assert!(tracker.get_round_record(1).is_none());
        assert!(tracker.get_round_record(4).is_some());
    }

    #[test]
    fn test_earnings_summary_serializable() {
        let summary = EarningsSummary {
            total_earnings_wei: 1000,
            rounds_participated: 5,
            rounds_succeeded: 4,
            rounds_failed: 1,
            total_proofs_submitted: 20,
            total_steps_completed: 100,
            avg_earnings_per_round_wei: 250,
            success_rate: 0.8,
            reputation_score: 0.85,
        };
        let json = serde_json::to_string(&summary).unwrap();
        let deserialized: EarningsSummary = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.total_earnings_wei, 1000);
        assert_eq!(deserialized.rounds_participated, 5);
    }

    #[test]
    fn test_round_record_duration() {
        let mut record = RoundRecord::new_joining(1, 100);
        assert_eq!(record.duration_secs(), 0);

        // Manually set times to test
        record.joined_at = 1000;
        record.completed_at = 1060;
        assert_eq!(record.duration_secs(), 60);
    }

    #[test]
    fn test_reputation_score_bounds() {
        let mut tracker = ParticipationTracker::new(100);
        // All failures: reputation should be 0.0
        for i in 0..20 {
            tracker.record_join(i, 100);
            tracker.record_failure(i);
        }
        let summary = tracker.earnings_summary();
        assert!(summary.reputation_score >= 0.0);
        assert!(summary.reputation_score <= 1.0);
    }
}
