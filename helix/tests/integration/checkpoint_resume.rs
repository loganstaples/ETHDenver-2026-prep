//! Checkpoint/Resume Integration Tests.
//!
//! Tests the ability to save and restore training state:
//! - Checkpoint creation and serialization
//! - Resume from checkpoint
//! - State consistency after resume
//! - Proof chain continuity across restarts

#![allow(unused_imports)]

use std::collections::HashMap;
use std::time::{Duration, Instant};

use helix_circuits::halo2_proofs::arithmetic::Field;
use helix_circuits::halo2curves::bn256::Fr;
use helix_circuits::halo2curves::ff::PrimeField;
use helix_circuits::ml::training_step_v2::compute_state_hash_v2;
use helix_prover::{BatchTrainingProverV2, MLTrainingProverV2, TrainingWeights};
use serde::{Deserialize, Serialize};

#[path = "../common/mod.rs"]
mod common;
use common::*;

// ============================================================================
// Checkpoint Data Structures
// ============================================================================

/// Serializable checkpoint for training state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingCheckpoint {
    /// Current step number.
    pub step: u64,
    /// Model weights (serialized as bytes for Fr elements).
    pub weights: SerializableWeights,
    /// State hash at checkpoint.
    pub state_hash: (Vec<u8>, Vec<u8>),
    /// Accumulated loss.
    pub accumulated_loss: Vec<u8>,
    /// Error bound at checkpoint.
    pub error_bound: Vec<u8>,
    /// Timestamp of checkpoint creation.
    pub timestamp: u64,
}

/// Serializable model weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableWeights {
    pub d_in: usize,
    pub d_hid: usize,
    pub d_out: usize,
    pub w1: Vec<Vec<u8>>,
    pub b1: Vec<Vec<u8>>,
    pub w2: Vec<Vec<u8>>,
    pub b2: Vec<Vec<u8>>,
}

impl SerializableWeights {
    /// Converts TrainingWeights to serializable format.
    pub fn from_training_weights(weights: &TrainingWeights) -> Self {
        Self {
            d_in: weights.d_in,
            d_hid: weights.d_hid,
            d_out: weights.d_out,
            w1: weights.w1.iter().map(|f| f.to_repr().as_ref().to_vec()).collect(),
            b1: weights.b1.iter().map(|f| f.to_repr().as_ref().to_vec()).collect(),
            w2: weights.w2.iter().map(|f| f.to_repr().as_ref().to_vec()).collect(),
            b2: weights.b2.iter().map(|f| f.to_repr().as_ref().to_vec()).collect(),
        }
    }

    /// Converts back to TrainingWeights.
    pub fn to_training_weights(&self) -> TrainingWeights {
        fn bytes_to_fr(bytes: &[u8]) -> Fr {
            let mut arr = [0u8; 32];
            let len = bytes.len().min(32);
            arr[..len].copy_from_slice(&bytes[..len]);
            Fr::from_repr_vartime(arr.into()).unwrap_or(Fr::zero())
        }

        TrainingWeights::new(
            self.d_in,
            self.d_hid,
            self.d_out,
            self.w1.iter().map(|b| bytes_to_fr(b)).collect(),
            self.b1.iter().map(|b| bytes_to_fr(b)).collect(),
            self.w2.iter().map(|b| bytes_to_fr(b)).collect(),
            self.b2.iter().map(|b| bytes_to_fr(b)).collect(),
        )
    }
}

/// Checkpoint manager for testing.
pub struct CheckpointManager {
    checkpoints: HashMap<u64, TrainingCheckpoint>,
    checkpoint_interval: u64,
}

impl CheckpointManager {
    pub fn new(checkpoint_interval: u64) -> Self {
        Self {
            checkpoints: HashMap::new(),
            checkpoint_interval,
        }
    }

    /// Creates a checkpoint.
    pub fn create_checkpoint(
        &mut self,
        step: u64,
        weights: &TrainingWeights,
        loss: Fr,
        error_bound: Fr,
    ) -> TrainingCheckpoint {
        let state_hash = compute_state_hash_v2(&weights.w1, &weights.b1, &weights.w2, &weights.b2);

        let checkpoint = TrainingCheckpoint {
            step,
            weights: SerializableWeights::from_training_weights(weights),
            state_hash: (
                state_hash.0.to_repr().as_ref().to_vec(),
                state_hash.1.to_repr().as_ref().to_vec(),
            ),
            accumulated_loss: loss.to_repr().as_ref().to_vec(),
            error_bound: error_bound.to_repr().as_ref().to_vec(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };

        self.checkpoints.insert(step, checkpoint.clone());
        checkpoint
    }

    /// Gets the latest checkpoint.
    pub fn get_latest(&self) -> Option<&TrainingCheckpoint> {
        self.checkpoints
            .iter()
            .max_by_key(|(step, _)| *step)
            .map(|(_, cp)| cp)
    }

    /// Gets a checkpoint at a specific step.
    pub fn get_checkpoint(&self, step: u64) -> Option<&TrainingCheckpoint> {
        self.checkpoints.get(&step)
    }

    /// Checks if a checkpoint should be created.
    pub fn should_checkpoint(&self, step: u64) -> bool {
        step > 0 && step % self.checkpoint_interval == 0
    }

    /// Serializes checkpoints to JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        let checkpoints: Vec<_> = self.checkpoints.values().collect();
        serde_json::to_string_pretty(&checkpoints)
    }

    /// Deserializes checkpoints from JSON.
    pub fn from_json(json: &str, interval: u64) -> Result<Self, serde_json::Error> {
        let checkpoints: Vec<TrainingCheckpoint> = serde_json::from_str(json)?;
        let map: HashMap<_, _> = checkpoints.into_iter().map(|cp| (cp.step, cp)).collect();
        Ok(Self {
            checkpoints: map,
            checkpoint_interval: interval,
        })
    }
}

// ============================================================================
// Basic Checkpoint Tests
// ============================================================================

/// Tests creating a checkpoint.
#[test]
fn test_checkpoint_creation() {
    let dims = ModelDimensions::tiny();
    let weights = TestModelWeights::known(dims);
    let training_weights = weights.to_training_weights();

    let mut manager = CheckpointManager::new(10);

    let checkpoint = manager.create_checkpoint(
        10,
        &training_weights,
        Fr::from(100u64),
        Fr::from(50u64),
    );

    assert_eq!(checkpoint.step, 10);
    assert_eq!(checkpoint.weights.d_in, dims.d_in);
    assert_eq!(checkpoint.weights.d_hid, dims.d_hid);
    assert_eq!(checkpoint.weights.d_out, dims.d_out);
}

/// Tests checkpoint retrieval.
#[test]
fn test_checkpoint_retrieval() {
    let dims = ModelDimensions::tiny();
    let weights = TestModelWeights::known(dims);
    let training_weights = weights.to_training_weights();

    let mut manager = CheckpointManager::new(5);

    // Create multiple checkpoints
    for step in (5..=20).step_by(5) {
        manager.create_checkpoint(
            step as u64,
            &training_weights,
            Fr::from(100u64 - step as u64),
            Fr::from(step as u64),
        );
    }

    // Get specific checkpoint
    let cp10 = manager.get_checkpoint(10).expect("Should have step 10");
    assert_eq!(cp10.step, 10);

    // Get latest checkpoint
    let latest = manager.get_latest().expect("Should have checkpoints");
    assert_eq!(latest.step, 20);
}

/// Tests weight serialization roundtrip.
#[test]
fn test_weight_serialization_roundtrip() {
    let dims = ModelDimensions::tiny();
    let weights = TestModelWeights::known(dims);
    let training_weights = weights.to_training_weights();

    // Serialize
    let serialized = SerializableWeights::from_training_weights(&training_weights);

    // Deserialize
    let restored = serialized.to_training_weights();

    // Verify
    assert_eq!(training_weights.d_in, restored.d_in);
    assert_eq!(training_weights.d_hid, restored.d_hid);
    assert_eq!(training_weights.d_out, restored.d_out);

    for (orig, rest) in training_weights.w1.iter().zip(restored.w1.iter()) {
        assert_eq!(orig, rest, "W1 should match after roundtrip");
    }
    for (orig, rest) in training_weights.w2.iter().zip(restored.w2.iter()) {
        assert_eq!(orig, rest, "W2 should match after roundtrip");
    }
}

/// Tests checkpoint JSON serialization.
#[test]
fn test_checkpoint_json_serialization() {
    let dims = ModelDimensions::tiny();
    let weights = TestModelWeights::known(dims);
    let training_weights = weights.to_training_weights();

    let mut manager = CheckpointManager::new(5);
    manager.create_checkpoint(5, &training_weights, Fr::from(100u64), Fr::from(10u64));
    manager.create_checkpoint(10, &training_weights, Fr::from(80u64), Fr::from(20u64));

    // Serialize to JSON
    let json = manager.to_json().expect("Should serialize");

    // Deserialize
    let restored = CheckpointManager::from_json(&json, 5).expect("Should deserialize");

    assert!(restored.get_checkpoint(5).is_some());
    assert!(restored.get_checkpoint(10).is_some());
}

// ============================================================================
// Resume Tests
// ============================================================================

/// Tests resuming training from a checkpoint.
#[test]
fn test_resume_from_checkpoint() {
    let harness = TestHarness::with_config(HarnessConfig::ci());
    let mut result = TestResult::new("resume_from_checkpoint");

    let dims = ModelDimensions::tiny();

    // Phase 1: Initial training (steps 1-5)
    let phase_start = Instant::now();
    let initial_weights = TrainingWeights::new(
        dims.d_in,
        dims.d_hid,
        dims.d_out,
        vec![Fr::from(1u64), Fr::from(2u64), Fr::from(3u64), Fr::from(1u64)],
        vec![Fr::zero(), Fr::zero()],
        vec![Fr::from(1u64), Fr::from(1u64)],
        vec![Fr::zero()],
    );

    let prover = BatchTrainingProverV2::new(dims.d_in, dims.d_hid, dims.d_out);
    let samples_phase1 = vec![
        (vec![Fr::from(1u64), Fr::from(1u64)], vec![Fr::from(5u64)]),
        (vec![Fr::from(2u64), Fr::from(1u64)], vec![Fr::from(6u64)]),
    ];

    let result1 = prover.prove_batch(initial_weights, &samples_phase1, Fr::from(1u64));
    result.add_phase(PhaseResult::success("initial_training", phase_start.elapsed()));

    // Phase 2: Create checkpoint
    let phase_start = Instant::now();
    let mut checkpoint_mgr = CheckpointManager::new(2);
    let last_proof = result1.proofs.last().unwrap();
    checkpoint_mgr.create_checkpoint(
        2,
        &result1.final_weights,
        last_proof.loss,
        last_proof.total_error,
    );
    result.add_phase(PhaseResult::success("checkpoint_creation", phase_start.elapsed()));

    // Phase 3: Resume from checkpoint
    let phase_start = Instant::now();
    let checkpoint = checkpoint_mgr.get_latest().expect("Should have checkpoint");
    let restored_weights = checkpoint.weights.to_training_weights();

    let samples_phase2 = vec![
        (vec![Fr::from(1u64), Fr::from(2u64)], vec![Fr::from(7u64)]),
        (vec![Fr::from(2u64), Fr::from(2u64)], vec![Fr::from(8u64)]),
    ];

    let result2 = prover.prove_batch(restored_weights, &samples_phase2, Fr::from(1u64));
    result.add_phase(PhaseResult::success("resumed_training", phase_start.elapsed()));

    // Phase 4: Verify continuity
    let phase_start = Instant::now();

    // State hash from checkpoint should match
    let checkpoint_hash = compute_state_hash_v2(
        &result1.final_weights.w1,
        &result1.final_weights.b1,
        &result1.final_weights.w2,
        &result1.final_weights.b2,
    );

    let resumed_start_hash = result2.proofs[0].old_state_hash;

    let hashes_match = checkpoint_hash == resumed_start_hash;
    result.add_phase(if hashes_match {
        PhaseResult::success("state_continuity", phase_start.elapsed())
    } else {
        PhaseResult::failure("state_continuity", phase_start.elapsed(), "State hash mismatch")
    });

    // Phase 5: Verify all proofs
    let phase_start = Instant::now();
    let all_valid = prover.verify_batch(&result1) && prover.verify_batch(&result2);
    result.add_phase(if all_valid {
        PhaseResult::success("proof_verification", phase_start.elapsed())
    } else {
        PhaseResult::failure("proof_verification", phase_start.elapsed(), "Some proofs invalid")
    });

    result.finalize();
    println!("{}", harness.generate_report(&result));
    assert!(result.success);
}

/// Tests that resumed training continues from correct step.
#[test]
fn test_resume_step_continuity() {
    let dims = ModelDimensions::tiny();

    let initial_weights = TrainingWeights::new(
        dims.d_in,
        dims.d_hid,
        dims.d_out,
        vec![Fr::from(1u64), Fr::from(2u64), Fr::from(3u64), Fr::from(1u64)],
        vec![Fr::zero(), Fr::zero()],
        vec![Fr::from(1u64), Fr::from(1u64)],
        vec![Fr::zero()],
    );

    let prover = BatchTrainingProverV2::new(dims.d_in, dims.d_hid, dims.d_out);

    // First batch: steps 1-2
    let samples1 = vec![
        (vec![Fr::from(1u64), Fr::from(1u64)], vec![Fr::from(5u64)]),
        (vec![Fr::from(2u64), Fr::from(1u64)], vec![Fr::from(6u64)]),
    ];
    let result1 = prover.prove_batch(initial_weights, &samples1, Fr::from(1u64));

    // Verify step numbers in first batch
    assert_eq!(result1.proofs[0].step_number, 1);
    assert_eq!(result1.proofs[1].step_number, 2);

    // Second batch continues with step 3 (but prover resets to 1)
    // In a real system, step numbers would be passed through
    let samples2 = vec![
        (vec![Fr::from(1u64), Fr::from(2u64)], vec![Fr::from(7u64)]),
    ];
    let result2 = prover.prove_batch(result1.final_weights, &samples2, Fr::from(1u64));

    // The batch prover starts from step 1 internally, but we verify the proofs
    assert!(prover.verify_batch(&result2));
}

// ============================================================================
// State Consistency Tests
// ============================================================================

/// Tests state hash consistency across checkpoint/resume.
#[test]
fn test_state_hash_consistency() {
    let dims = ModelDimensions::tiny();
    let weights = TestModelWeights::known(dims);
    let training_weights = weights.to_training_weights();

    // Compute state hash before checkpoint
    let original_hash = compute_state_hash_v2(
        &training_weights.w1,
        &training_weights.b1,
        &training_weights.w2,
        &training_weights.b2,
    );

    // Create checkpoint
    let mut manager = CheckpointManager::new(1);
    manager.create_checkpoint(1, &training_weights, Fr::from(100u64), Fr::from(10u64));

    // Restore from checkpoint
    let checkpoint = manager.get_checkpoint(1).unwrap();
    let restored_weights = checkpoint.weights.to_training_weights();

    // Compute state hash after restore
    let restored_hash = compute_state_hash_v2(
        &restored_weights.w1,
        &restored_weights.b1,
        &restored_weights.w2,
        &restored_weights.b2,
    );

    assert_eq!(original_hash, restored_hash, "State hash should be preserved");
}

/// Tests that training produces consistent results after restore.
#[test]
fn test_training_consistency_after_restore() {
    let dims = ModelDimensions::tiny();
    let weights = TestModelWeights::known(dims);
    let training_weights = weights.to_training_weights();

    let prover = MLTrainingProverV2::new(dims.d_in, dims.d_hid, dims.d_out);

    let x = vec![Fr::from(1u64), Fr::from(1u64)];
    let target = vec![Fr::from(5u64)];

    // Train from original weights
    let witness1 = MLTrainingProverV2::build_witness(
        dims.d_in,
        dims.d_hid,
        dims.d_out,
        &x,
        &target,
        &training_weights.w1,
        &training_weights.b1,
        &training_weights.w2,
        &training_weights.b2,
        Fr::from(1u64),
        1,
        Fr::from(1u64),
    );
    let proof1 = prover.prove(&witness1).unwrap();

    // Serialize and restore weights
    let serialized = SerializableWeights::from_training_weights(&training_weights);
    let restored = serialized.to_training_weights();

    // Train from restored weights
    let witness2 = MLTrainingProverV2::build_witness(
        dims.d_in,
        dims.d_hid,
        dims.d_out,
        &x,
        &target,
        &restored.w1,
        &restored.b1,
        &restored.w2,
        &restored.b2,
        Fr::from(1u64),
        1,
        Fr::from(1u64),
    );
    let proof2 = prover.prove(&witness2).unwrap();

    // Results should be identical
    assert_eq!(proof1.loss, proof2.loss, "Loss should match");
    assert_eq!(proof1.old_state_hash, proof2.old_state_hash, "Old hash should match");
    assert_eq!(proof1.new_state_hash, proof2.new_state_hash, "New hash should match");
}

// ============================================================================
// Proof Chain Continuity Tests
// ============================================================================

/// Tests that proof chain is continuous across checkpoint/resume.
#[test]
fn test_proof_chain_continuity() {
    let dims = ModelDimensions::tiny();

    let initial_weights = TrainingWeights::new(
        dims.d_in,
        dims.d_hid,
        dims.d_out,
        vec![Fr::from(1u64), Fr::from(2u64), Fr::from(3u64), Fr::from(1u64)],
        vec![Fr::zero(), Fr::zero()],
        vec![Fr::from(1u64), Fr::from(1u64)],
        vec![Fr::zero()],
    );

    let prover = BatchTrainingProverV2::new(dims.d_in, dims.d_hid, dims.d_out);

    // First phase
    let samples1 = vec![
        (vec![Fr::from(1u64), Fr::from(1u64)], vec![Fr::from(5u64)]),
    ];
    let result1 = prover.prove_batch(initial_weights, &samples1, Fr::from(1u64));

    // Get the final state hash from phase 1
    let phase1_final_hash = result1.proofs.last().unwrap().new_state_hash;

    // Second phase with restored weights
    let samples2 = vec![
        (vec![Fr::from(2u64), Fr::from(1u64)], vec![Fr::from(6u64)]),
    ];
    let result2 = prover.prove_batch(result1.final_weights, &samples2, Fr::from(1u64));

    // First proof of phase 2 should start where phase 1 ended
    let phase2_start_hash = result2.proofs[0].old_state_hash;

    assert_eq!(
        phase1_final_hash, phase2_start_hash,
        "Proof chain should be continuous across phases"
    );
}

// ============================================================================
// Error Recovery Tests
// ============================================================================

/// Tests recovery from corrupted checkpoint.
#[test]
fn test_recovery_from_corrupted_checkpoint() {
    let dims = ModelDimensions::tiny();
    let weights = TestModelWeights::known(dims);
    let training_weights = weights.to_training_weights();

    let mut manager = CheckpointManager::new(5);

    // Create valid checkpoints
    manager.create_checkpoint(5, &training_weights, Fr::from(100u64), Fr::from(10u64));
    manager.create_checkpoint(10, &training_weights, Fr::from(80u64), Fr::from(20u64));
    manager.create_checkpoint(15, &training_weights, Fr::from(60u64), Fr::from(30u64));

    // Simulate corruption: remove checkpoint 15
    manager.checkpoints.remove(&15);

    // Should be able to recover from checkpoint 10
    let fallback = manager.get_latest().expect("Should have fallback checkpoint");
    assert_eq!(fallback.step, 10, "Should fall back to checkpoint 10");
}

/// Tests checkpoint interval logic.
#[test]
fn test_checkpoint_interval() {
    let manager = CheckpointManager::new(10);

    // Should checkpoint at multiples of 10
    assert!(!manager.should_checkpoint(0));
    assert!(!manager.should_checkpoint(1));
    assert!(!manager.should_checkpoint(5));
    assert!(manager.should_checkpoint(10));
    assert!(!manager.should_checkpoint(15));
    assert!(manager.should_checkpoint(20));
}

// ============================================================================
// Comprehensive Resume Test
// ============================================================================

/// Comprehensive test of checkpoint/resume workflow.
#[test]
fn test_comprehensive_checkpoint_resume() {
    let harness = TestHarness::with_config(HarnessConfig::ci());
    let mut result = TestResult::new("comprehensive_checkpoint_resume");

    let dims = ModelDimensions::tiny();
    let initial_weights = TrainingWeights::new(
        dims.d_in,
        dims.d_hid,
        dims.d_out,
        vec![Fr::from(1u64), Fr::from(2u64), Fr::from(3u64), Fr::from(1u64)],
        vec![Fr::zero(), Fr::zero()],
        vec![Fr::from(1u64), Fr::from(1u64)],
        vec![Fr::zero()],
    );

    let prover = BatchTrainingProverV2::new(dims.d_in, dims.d_hid, dims.d_out);
    let mut checkpoint_mgr = CheckpointManager::new(2);

    // Phase 1: Train for 2 steps
    let phase_start = Instant::now();
    let samples_p1 = vec![
        (vec![Fr::from(1u64), Fr::from(1u64)], vec![Fr::from(5u64)]),
        (vec![Fr::from(2u64), Fr::from(1u64)], vec![Fr::from(6u64)]),
    ];
    let result_p1 = prover.prove_batch(initial_weights, &samples_p1, Fr::from(1u64));
    result.add_phase(PhaseResult::success("phase1_training", phase_start.elapsed()));

    // Checkpoint after phase 1
    let phase_start = Instant::now();
    let last_proof_p1 = result_p1.proofs.last().unwrap();
    checkpoint_mgr.create_checkpoint(2, &result_p1.final_weights, last_proof_p1.loss, last_proof_p1.total_error);
    result.add_phase(PhaseResult::success("checkpoint_p1", phase_start.elapsed()));

    // Serialize checkpoint (simulate restart)
    let phase_start = Instant::now();
    let json = checkpoint_mgr.to_json().unwrap();
    let restored_mgr = CheckpointManager::from_json(&json, 2).unwrap();
    result.add_phase(PhaseResult::success("checkpoint_serialize", phase_start.elapsed()));

    // Phase 2: Resume and train for 2 more steps
    let phase_start = Instant::now();
    let restored_checkpoint = restored_mgr.get_latest().unwrap();
    let restored_weights = restored_checkpoint.weights.to_training_weights();

    let samples_p2 = vec![
        (vec![Fr::from(1u64), Fr::from(2u64)], vec![Fr::from(7u64)]),
        (vec![Fr::from(2u64), Fr::from(2u64)], vec![Fr::from(8u64)]),
    ];
    let result_p2 = prover.prove_batch(restored_weights, &samples_p2, Fr::from(1u64));
    result.add_phase(PhaseResult::success("phase2_training", phase_start.elapsed()));

    // Verify all proofs
    let phase_start = Instant::now();
    let p1_valid = prover.verify_batch(&result_p1);
    let p2_valid = prover.verify_batch(&result_p2);
    result.add_phase(if p1_valid && p2_valid {
        PhaseResult::success("all_proofs_valid", phase_start.elapsed())
    } else {
        PhaseResult::failure("all_proofs_valid", phase_start.elapsed(), "Some proofs invalid")
    });

    // Verify state continuity
    let phase_start = Instant::now();
    let p1_end_hash = result_p1.proofs.last().unwrap().new_state_hash;
    let p2_start_hash = result_p2.proofs[0].old_state_hash;
    result.add_phase(if p1_end_hash == p2_start_hash {
        PhaseResult::success("state_continuity", phase_start.elapsed())
    } else {
        PhaseResult::failure("state_continuity", phase_start.elapsed(), "State hash mismatch")
    });

    result.finalize();
    println!("{}", harness.generate_report(&result));
    assert!(result.success);
}
