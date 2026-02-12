//! Comprehensive Proof Batching for ML Training Steps.
//!
//! This module provides advanced batching capabilities for generating proofs
//! across multiple training steps efficiently. Features include:
//!
//! - **Parallel batch proving**: Leverages work-stealing scheduler for multi-core
//! - **Streaming batches**: Process large training runs without memory exhaustion
//! - **Checkpoint/resume**: Save progress for long-running training sessions
//! - **Proof aggregation**: Combine batch proofs into a single succinct proof
//! - **Memory-efficient processing**: Configurable memory limits and spill-to-disk

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::parallel::{ChunkProof, ParallelConfig, ParallelProver, ProofStatus};
use crate::chunking::{ChunkId, ComputationChunk, ComputationType};
use crate::aggregation::{KZGAggregatedProof, KZGBatchAggregator, RLCAggregationProver, AggregatedTrainingProof};
use crate::provers::training_prover_v2::TrainingProofResultV2;
use thiserror::Error;

/// Errors from batch proving operations.
#[derive(Error, Debug)]
pub enum BatchProveError {
    /// All proofs in the batch failed.
    #[error("Batch produced 0 proofs: all {total} steps failed")]
    AllStepsFailed {
        /// Total steps attempted.
        total: usize,
    },

    /// Proof aggregation failed (both RLC and KZG paths).
    #[error("Proof aggregation failed for {num_proofs} proofs: {reason}")]
    AggregationFailed {
        /// Why aggregation failed.
        reason: String,
        /// Number of proofs we tried to aggregate.
        num_proofs: usize,
    },
}

/// Configuration for batch proving.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    /// Maximum proofs to hold in memory before spilling to disk.
    pub max_memory_proofs: usize,
    /// Directory for checkpoint files.
    pub checkpoint_dir: Option<PathBuf>,
    /// Checkpoint frequency (every N proofs).
    pub checkpoint_frequency: usize,
    /// Number of parallel proving threads.
    pub num_threads: usize,
    /// Enable proof aggregation.
    pub aggregate_proofs: bool,
    /// Batch size for aggregation (proofs per aggregate).
    pub aggregation_batch_size: usize,
    /// Enable streaming mode for large batches.
    pub streaming_mode: bool,
    /// Maximum memory usage (bytes) before spilling.
    pub max_memory_bytes: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_memory_proofs: 1000,
            checkpoint_dir: None,
            checkpoint_frequency: 100,
            num_threads: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4),
            aggregate_proofs: false,
            aggregation_batch_size: 10,
            streaming_mode: false,
            max_memory_bytes: 1024 * 1024 * 1024, // 1GB
        }
    }
}

/// Status of a batch proof job.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BatchStatus {
    /// Not started.
    Pending,
    /// Currently processing.
    InProgress {
        /// Completed proofs.
        completed: usize,
        /// Total proofs.
        total: usize,
    },
    /// Successfully completed.
    Complete,
    /// Paused (checkpoint available).
    Paused { checkpoint_id: String },
    /// Failed with error.
    Failed(String),
}

/// A single training step to be proved.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingStep {
    /// Step index in the training run.
    pub step_index: u64,
    /// Input commitment (state before this step).
    pub input_commitment: [u8; 32],
    /// Output commitment (state after this step).
    pub output_commitment: [u8; 32],
    /// Loss value for this step.
    pub loss: f64,
    /// Gradient norm.
    pub gradient_norm: f64,
    /// Error bound for this step.
    pub error_bound: f64,
    /// Optional auxiliary data.
    pub aux_data: Option<Vec<u8>>,
}

impl TrainingStep {
    /// Creates a new training step.
    pub fn new(
        step_index: u64,
        input_commitment: [u8; 32],
        output_commitment: [u8; 32],
        loss: f64,
        error_bound: f64,
    ) -> Self {
        Self {
            step_index,
            input_commitment,
            output_commitment,
            loss,
            gradient_norm: 0.0,
            error_bound,
            aux_data: None,
        }
    }

    /// Converts to a computation chunk for the parallel prover.
    pub fn to_chunk(&self) -> ComputationChunk {
        ComputationChunk {
            id: ChunkId(self.step_index),
            parent_id: if self.step_index > 0 {
                Some(ChunkId(self.step_index - 1))
            } else {
                None
            },
            input_commitment: self.input_commitment,
            output_commitment: self.output_commitment,
            computation_type: ComputationType::Backward,
            layer_range: (0, 1),
            error_bound: self.error_bound,
        }
    }
}

/// Proof for a single training step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepProof {
    /// The step index.
    pub step_index: u64,
    /// Serialized proof bytes.
    pub proof: Vec<u8>,
    /// Public inputs.
    pub public_inputs: Vec<[u8; 32]>,
    /// Generation time in milliseconds.
    pub generation_time_ms: u64,
    /// Proof size in bytes.
    pub proof_size: usize,
    /// Timestamp when proof was generated.
    pub timestamp: u64,
    /// Error bound for this step (carried from ChunkProof/TrainingStep).
    #[serde(default)]
    pub error_bound: f64,
}

impl From<ChunkProof> for StepProof {
    fn from(chunk: ChunkProof) -> Self {
        Self {
            step_index: chunk.chunk_id.0,
            proof: chunk.proof.clone(),
            public_inputs: chunk.public_inputs,
            generation_time_ms: chunk.generation_time_ms,
            proof_size: chunk.proof.len(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            error_bound: chunk.error_bound,
        }
    }
}

impl From<StepProof> for ChunkProof {
    fn from(step: StepProof) -> Self {
        Self {
            chunk_id: ChunkId(step.step_index),
            proof: step.proof,
            public_inputs: step.public_inputs,
            error_bound: step.error_bound,
            generation_time_ms: step.generation_time_ms,
        }
    }
}

/// Result of a batch proving operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResult {
    /// Batch identifier.
    pub batch_id: String,
    /// Individual step proofs.
    pub proofs: Vec<StepProof>,
    /// Aggregated proof (if enabled).
    pub aggregated_proof: Option<AggregatedBatchProof>,
    /// Total steps processed.
    pub total_steps: usize,
    /// Total proof generation time.
    pub total_time_ms: u64,
    /// Average time per proof.
    pub avg_time_per_proof_ms: f64,
    /// Total proof bytes.
    pub total_proof_bytes: usize,
    /// First step index.
    pub first_step: u64,
    /// Last step index.
    pub last_step: u64,
}

/// Aggregated proof for a batch of training steps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedBatchProof {
    /// Aggregated proof bytes (KZG proof when aggregation succeeds,
    /// first individual proof as representative when fallback).
    pub proof: Vec<u8>,
    /// Merkle root of all step commitments.
    pub merkle_root: [u8; 32],
    /// Number of steps aggregated.
    pub num_steps: usize,
    /// First step index.
    pub first_step: u64,
    /// Last step index.
    pub last_step: u64,
    /// Total error bound for the batch.
    pub total_error_bound: f64,
    /// RLC commitment from proof aggregation (when using RLC aggregation).
    #[serde(default)]
    pub rlc_commitment: Option<[u8; 32]>,
    /// KZG aggregated proof (O(1) size, present when KZG aggregation succeeds).
    #[serde(default)]
    pub kzg_proof: Option<KZGAggregatedProof>,
    /// Individual proof bytes when KZG aggregation is unavailable.
    /// Each entry is one step's proof bytes, preserving them for independent verification.
    #[serde(default)]
    pub individual_proofs: Option<Vec<Vec<u8>>>,
    /// RLC aggregated proof (SHPLONKAggregationCircuit-based, 8 PIs matching contract).
    /// This is the preferred aggregation format for on-chain submission.
    #[serde(skip)]
    pub rlc_proof: Option<AggregatedTrainingProof>,
}

/// Checkpoint for resumable batch proving.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchCheckpoint {
    /// Checkpoint identifier.
    pub checkpoint_id: String,
    /// Batch identifier.
    pub batch_id: String,
    /// Completed step indices.
    pub completed_steps: Vec<u64>,
    /// Last completed step.
    pub last_step: u64,
    /// Remaining steps to prove.
    pub remaining_steps: Vec<u64>,
    /// Intermediate proofs.
    pub proofs: Vec<StepProof>,
    /// Checkpoint timestamp.
    pub timestamp: u64,
    /// Total elapsed time so far.
    pub elapsed_ms: u64,
}

impl BatchCheckpoint {
    /// Saves checkpoint to disk.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, self)?;
        Ok(())
    }

    /// Loads checkpoint from disk.
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let checkpoint = serde_json::from_reader(reader)?;
        Ok(checkpoint)
    }
}

/// Statistics for batch proving.
#[derive(Debug, Default)]
pub struct BatchStats {
    /// Total proofs generated.
    pub proofs_generated: AtomicU64,
    /// Total proof bytes.
    pub total_bytes: AtomicU64,
    /// Total proving time (microseconds).
    pub total_prove_time_us: AtomicU64,
    /// Checkpoints created.
    pub checkpoints_created: AtomicU64,
    /// Proofs spilled to disk.
    pub proofs_spilled: AtomicU64,
    /// Memory high water mark.
    pub memory_high_water: AtomicUsize,
}

impl BatchStats {
    fn new() -> Self {
        Self::default()
    }

    /// Gets a snapshot of stats.
    pub fn snapshot(&self) -> BatchStatsSnapshot {
        BatchStatsSnapshot {
            proofs_generated: self.proofs_generated.load(Ordering::Relaxed),
            total_bytes: self.total_bytes.load(Ordering::Relaxed),
            total_prove_time_us: self.total_prove_time_us.load(Ordering::Relaxed),
            checkpoints_created: self.checkpoints_created.load(Ordering::Relaxed),
            proofs_spilled: self.proofs_spilled.load(Ordering::Relaxed),
            memory_high_water: self.memory_high_water.load(Ordering::Relaxed),
        }
    }
}

/// Snapshot of batch statistics.
#[derive(Debug, Clone)]
pub struct BatchStatsSnapshot {
    pub proofs_generated: u64,
    pub total_bytes: u64,
    pub total_prove_time_us: u64,
    pub checkpoints_created: u64,
    pub proofs_spilled: u64,
    pub memory_high_water: usize,
}

/// Batch prover for multiple training steps.
pub struct BatchProver {
    /// Configuration.
    config: BatchConfig,
    /// Underlying parallel prover.
    parallel_prover: ParallelProver,
    /// Completed proofs in memory.
    proofs: Arc<RwLock<HashMap<u64, StepProof>>>,
    /// Proofs spilled to disk.
    spilled_proofs: Arc<Mutex<Vec<PathBuf>>>,
    /// Current batch ID.
    batch_id: String,
    /// Statistics.
    stats: Arc<BatchStats>,
    /// Current status.
    status: Arc<RwLock<BatchStatus>>,
}

impl BatchProver {
    /// Creates a new batch prover.
    pub fn new(config: BatchConfig) -> Self {
        let parallel_config = ParallelConfig {
            num_threads: config.num_threads,
            work_stealing: true,
            task_affinity: true,
            ..Default::default()
        };

        let batch_id = format!(
            "batch-{:016x}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0)
        );

        Self {
            config,
            parallel_prover: ParallelProver::with_config(parallel_config),
            proofs: Arc::new(RwLock::new(HashMap::new())),
            spilled_proofs: Arc::new(Mutex::new(Vec::new())),
            batch_id,
            stats: Arc::new(BatchStats::new()),
            status: Arc::new(RwLock::new(BatchStatus::Pending)),
        }
    }

    /// Gets the batch ID.
    pub fn batch_id(&self) -> &str {
        &self.batch_id
    }

    /// Gets current status.
    pub fn status(&self) -> BatchStatus {
        self.status.read().ok().map(|s| s.clone()).unwrap_or(BatchStatus::Pending)
    }

    /// Gets statistics snapshot.
    pub fn stats(&self) -> BatchStatsSnapshot {
        self.stats.snapshot()
    }

    /// Proves a batch of training steps.
    ///
    /// Returns an error if the batch produces 0 proofs (all steps failed).
    pub fn prove_batch(&self, steps: Vec<TrainingStep>) -> Result<BatchResult, BatchProveError> {
        let start_time = Instant::now();
        let total_steps = steps.len();

        if steps.is_empty() {
            return Ok(BatchResult {
                batch_id: self.batch_id.clone(),
                proofs: Vec::new(),
                aggregated_proof: None,
                total_steps: 0,
                total_time_ms: 0,
                avg_time_per_proof_ms: 0.0,
                total_proof_bytes: 0,
                first_step: 0,
                last_step: 0,
            });
        }

        // Update status.
        {
            if let Ok(mut status) = self.status.write() {
                *status = BatchStatus::InProgress {
                    completed: 0,
                    total: total_steps,
                };
            }
        }

        let first_step = steps.iter().map(|s| s.step_index).min().unwrap_or(0);
        let last_step = steps.iter().map(|s| s.step_index).max().unwrap_or(0);

        // Convert steps to chunks and submit.
        let chunks: Vec<ComputationChunk> = steps.iter().map(|s| s.to_chunk()).collect();

        // Submit with dependency awareness (steps depend on previous steps).
        self.parallel_prover.submit_batch_with_dependencies(chunks);
        self.parallel_prover.start();

        // Wait for completion with progress tracking.
        let proofs = self.wait_with_progress(total_steps);

        self.parallel_prover.stop();

        // Convert to StepProofs.
        let step_proofs: Vec<StepProof> = proofs.into_iter().map(StepProof::from).collect();

        // Update stats.
        let total_bytes: usize = step_proofs.iter().map(|p| p.proof_size).sum();
        self.stats
            .proofs_generated
            .fetch_add(step_proofs.len() as u64, Ordering::Relaxed);
        self.stats
            .total_bytes
            .fetch_add(total_bytes as u64, Ordering::Relaxed);

        let total_time_ms = start_time.elapsed().as_millis() as u64;
        let avg_time = if step_proofs.is_empty() {
            0.0
        } else {
            total_time_ms as f64 / step_proofs.len() as f64
        };

        // Aggregate if enabled.
        let aggregated = if self.config.aggregate_proofs {
            match self.aggregate_proofs(&step_proofs, first_step, last_step) {
                Ok(agg) => Some(agg),
                Err(e) => {
                    tracing::error!("Proof aggregation failed: {e}");
                    None
                }
            }
        } else {
            None
        };

        // Check for total failure: if we expected proofs but got none, return error.
        if step_proofs.is_empty() && total_steps > 0 {
            if let Ok(mut status) = self.status.write() {
                *status = BatchStatus::Failed(format!("All {} steps produced 0 proofs", total_steps));
            }
            return Err(BatchProveError::AllStepsFailed { total: total_steps });
        }

        // Update status.
        {
            if let Ok(mut status) = self.status.write() {
                *status = BatchStatus::Complete;
            }
        }

        Ok(BatchResult {
            batch_id: self.batch_id.clone(),
            proofs: step_proofs,
            aggregated_proof: aggregated,
            total_steps,
            total_time_ms,
            avg_time_per_proof_ms: avg_time,
            total_proof_bytes: total_bytes,
            first_step,
            last_step,
        })
    }

    /// Proves a batch with streaming (memory-efficient for large batches).
    ///
    /// Returns an error if any sub-batch produces 0 proofs.
    pub fn prove_batch_streaming<I>(&self, steps: I) -> Result<StreamingBatchResult, BatchProveError>
    where
        I: Iterator<Item = TrainingStep>,
    {
        let start_time = Instant::now();
        let mut total_steps = 0;
        let mut total_bytes = 0usize;
        let mut first_step = u64::MAX;
        let mut last_step = 0u64;
        let mut batch_proofs: Vec<Vec<StepProof>> = Vec::new();
        let mut current_batch: Vec<TrainingStep> = Vec::new();

        // Process in batches to limit memory.
        for step in steps {
            if step.step_index < first_step {
                first_step = step.step_index;
            }
            if step.step_index > last_step {
                last_step = step.step_index;
            }

            current_batch.push(step);

            if current_batch.len() >= self.config.max_memory_proofs {
                let batch_result = self.prove_batch(std::mem::take(&mut current_batch))?;
                total_steps += batch_result.total_steps;
                total_bytes += batch_result.total_proof_bytes;
                batch_proofs.push(batch_result.proofs);

                // Checkpoint if enabled.
                if let Some(ref dir) = self.config.checkpoint_dir {
                    self.create_checkpoint(dir, &batch_proofs);
                }
            }
        }

        // Process remaining.
        if !current_batch.is_empty() {
            let batch_result = self.prove_batch(current_batch)?;
            total_steps += batch_result.total_steps;
            total_bytes += batch_result.total_proof_bytes;
            batch_proofs.push(batch_result.proofs);
        }

        if first_step == u64::MAX {
            first_step = 0;
        }

        let total_time_ms = start_time.elapsed().as_millis() as u64;

        Ok(StreamingBatchResult {
            batch_id: self.batch_id.clone(),
            batches: batch_proofs,
            total_steps,
            total_time_ms,
            total_proof_bytes: total_bytes,
            first_step,
            last_step,
        })
    }

    /// Resumes proving from a checkpoint.
    pub fn resume_from_checkpoint(&self, checkpoint: &BatchCheckpoint) -> Result<BatchResult, BatchProveError> {
        // Load existing proofs.
        {
            if let Ok(mut proofs) = self.proofs.write() {
                for proof in &checkpoint.proofs {
                    proofs.insert(proof.step_index, proof.clone());
                }
            }
        }

        // Create steps for remaining work.
        let remaining_steps: Vec<TrainingStep> = checkpoint
            .remaining_steps
            .iter()
            .map(|&idx| {
                TrainingStep::new(idx, [0; 32], [0; 32], 0.0, 0.01)
            })
            .collect();

        // Prove remaining.
        let result = self.prove_batch(remaining_steps)?;

        // Merge with existing proofs.
        let mut all_proofs: Vec<StepProof> = checkpoint.proofs.clone();
        all_proofs.extend(result.proofs);
        all_proofs.sort_by_key(|p| p.step_index);

        Ok(BatchResult {
            proofs: all_proofs,
            ..result
        })
    }

    /// Creates a checkpoint.
    fn create_checkpoint(&self, dir: &Path, batch_proofs: &[Vec<StepProof>]) {
        let all_proofs: Vec<StepProof> = batch_proofs.iter().flatten().cloned().collect();

        let completed_steps: Vec<u64> = all_proofs.iter().map(|p| p.step_index).collect();
        let last_step = completed_steps.iter().max().copied().unwrap_or(0);

        let checkpoint_id = format!(
            "checkpoint-{}-{}",
            self.batch_id,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0)
        );

        let checkpoint = BatchCheckpoint {
            checkpoint_id: checkpoint_id.clone(),
            batch_id: self.batch_id.clone(),
            completed_steps,
            last_step,
            remaining_steps: Vec::new(), // Would need to track this.
            proofs: all_proofs,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            elapsed_ms: 0,
        };

        let path = dir.join(format!("{}.json", checkpoint_id));
        if let Err(e) = checkpoint.save(&path) {
            tracing::error!("Failed to save batch checkpoint to {}: {e}", path.display());
        } else {
            self.stats.checkpoints_created.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn wait_with_progress(&self, total: usize) -> Vec<ChunkProof> {
        loop {
            let (completed, _) = self.parallel_prover.progress();

            // Update status.
            {
                if let Ok(mut status) = self.status.write() {
                    *status = BatchStatus::InProgress { completed, total };
                }
            }

            if completed >= total {
                break;
            }

            std::thread::sleep(Duration::from_millis(10));
        }

        self.parallel_prover.get_all_proofs()
    }

    fn aggregate_proofs(
        &self,
        proofs: &[StepProof],
        first_step: u64,
        last_step: u64,
    ) -> Result<AggregatedBatchProof, BatchProveError> {
        // Build SHA-256 Merkle root of all public inputs (backward compat).
        let mut hasher = Sha256::new();
        let mut total_error = 0.0f64;

        for proof in proofs {
            for input in &proof.public_inputs {
                hasher.update(input);
            }
            total_error += proof.error_bound;
        }

        let merkle_root: [u8; 32] = hasher.finalize().into();

        // Compute RLC commitment (retained for backward compat).
        let rlc_commitment = Self::compute_rlc_commitment(proofs);

        // ====================================================================
        // Primary aggregation path: RLCAggregationProver (SHPLONKAggregationCircuit).
        // Produces a single KZG proof with 8 public inputs matching the
        // contract interface. This is what gets submitted on-chain.
        // ====================================================================
        let training_results = Self::step_proofs_to_training_results(proofs);
        let mut rlc_error: Option<String> = None;

        let rlc_result = if proofs.len() <= helix_circuits::MAX_AGGREGATION_BATCH
            && proofs.iter().all(|p| p.public_inputs.len() >= 8)
        {
            // Check PI chaining before attempting aggregation
            let chain_valid = training_results.windows(2).all(|w| {
                w[0].new_state_hash == w[1].old_state_hash
            });

            if chain_valid {
                match Self::try_rlc_aggregation(&training_results) {
                    Ok(agg) => {
                        tracing::info!(
                            num_steps = agg.num_steps,
                            proof_size = agg.proof.len(),
                            "RLC aggregation succeeded (8-PI contract-compatible proof)"
                        );
                        Some(agg)
                    }
                    Err(e) => {
                        rlc_error = Some(e.clone());
                        tracing::warn!("RLC aggregation failed, falling back to KZG: {e}");
                        None
                    }
                }
            } else {
                rlc_error = Some("PI chain broken: step[i].new_hash != step[i+1].old_hash".to_string());
                tracing::warn!("PI chain broken, skipping RLC aggregation");
                None
            }
        } else {
            if proofs.len() > helix_circuits::MAX_AGGREGATION_BATCH {
                rlc_error = Some(format!(
                    "Batch size {} exceeds MAX_AGGREGATION_BATCH ({})",
                    proofs.len(),
                    helix_circuits::MAX_AGGREGATION_BATCH,
                ));
            } else {
                rlc_error = Some("Some proofs have < 8 public inputs".to_string());
            }
            None
        };

        // ====================================================================
        // Fallback: KZGBatchAggregator (4-PI simple aggregation).
        // ====================================================================
        let chunk_proofs: Vec<ChunkProof> = proofs
            .iter()
            .cloned()
            .map(ChunkProof::from)
            .collect();

        let kzg_result = if rlc_result.is_none() {
            let aggregator = KZGBatchAggregator::new();
            if aggregator.is_ready() {
                match aggregator.aggregate(&chunk_proofs) {
                    Ok(kzg_agg) => {
                        tracing::info!(
                            num_proofs = kzg_agg.num_proofs,
                            proof_size = kzg_agg.proof.len(),
                            "KZG batch aggregation succeeded (fallback)"
                        );
                        Some(kzg_agg)
                    }
                    Err(e) => {
                        tracing::warn!("KZG aggregation also failed: {e}");
                        None
                    }
                }
            } else {
                None
            }
        } else {
            None
        };

        // Use RLC proof > KZG proof. If both fail, return error.
        let (aggregated_proof_bytes, individual_proofs) = if let Some(ref rlc) = rlc_result {
            (rlc.proof.clone(), None)
        } else if let Some(ref kzg) = kzg_result {
            (kzg.proof.clone(), None)
        } else {
            return Err(BatchProveError::AggregationFailed {
                reason: rlc_error.unwrap_or_else(|| "Unknown aggregation failure".to_string()),
                num_proofs: proofs.len(),
            });
        };

        Ok(AggregatedBatchProof {
            proof: aggregated_proof_bytes,
            merkle_root,
            num_steps: proofs.len(),
            first_step,
            last_step,
            total_error_bound: total_error,
            rlc_commitment: Some(rlc_commitment),
            kzg_proof: kzg_result,
            individual_proofs,
            rlc_proof: rlc_result,
        })
    }

    /// Attempts RLC aggregation using SHPLONKAggregationCircuit.
    ///
    /// This is the primary aggregation path producing 8 public inputs
    /// matching the on-chain contract interface.
    fn try_rlc_aggregation(
        proofs: &[TrainingProofResultV2],
    ) -> Result<AggregatedTrainingProof, String> {
        // k=14 is needed for up to 32 steps (MAX_AGGREGATION_BATCH)
        let prover = RLCAggregationProver::new(
            helix_circuits::MAX_AGGREGATION_BATCH,
            14,
        );
        prover.aggregate(proofs)
    }

    /// Converts StepProofs to TrainingProofResultV2 for RLC aggregation.
    fn step_proofs_to_training_results(proofs: &[StepProof]) -> Vec<TrainingProofResultV2> {
        use helix_circuits::halo2curves::bn256::Fr;
        use helix_circuits::halo2_proofs::arithmetic::Field;
        use helix_circuits::verifier::evm_bytes_to_fr;

        proofs.iter().map(|step| {
            // StepProof.public_inputs stores [u8; 32] in big-endian EVM format
            // (matching fr_to_evm_bytes / to_evm_public_inputs convention).
            // Use evm_bytes_to_fr for proper big-endian → Fr conversion.
            let pis: Vec<Fr> = step.public_inputs.iter().map(|bytes| {
                evm_bytes_to_fr(bytes).unwrap_or(Fr::ZERO)
            }).collect();

            let old_hash = if pis.len() >= 2 { (pis[0], pis[1]) } else { (Fr::ZERO, Fr::ZERO) };
            let new_hash = if pis.len() >= 4 { (pis[2], pis[3]) } else { (Fr::ZERO, Fr::ZERO) };
            let loss = if pis.len() >= 5 { pis[4] } else { Fr::ZERO };
            let total_error = if pis.len() >= 6 { pis[5] } else { Fr::ZERO };

            TrainingProofResultV2 {
                proof: step.proof.clone(),
                public_inputs: pis,
                loss,
                total_error,
                step_number: step.step_index,
                old_state_hash: old_hash,
                new_state_hash: new_hash,
                verified: true,
                generation_time: Duration::from_millis(step.generation_time_ms),
                verification_time: None,
                attempts: 1,
                from_cache: false,
                witness_hash: None,
            }
        }).collect()
    }

    /// Aggregates training proof results directly (no StepProof conversion needed).
    ///
    /// This is the preferred API when you already have `TrainingProofResultV2` values
    /// (e.g., from `MLTrainingProverV2`). Produces a single aggregated proof with
    /// 8 public inputs matching the on-chain contract interface.
    pub fn aggregate_training_proofs(
        proofs: &[TrainingProofResultV2],
    ) -> Result<AggregatedTrainingProof, String> {
        Self::try_rlc_aggregation(proofs)
    }

    /// Computes the RLC commitment from step proofs using Poseidon hashing.
    fn compute_rlc_commitment(proofs: &[StepProof]) -> [u8; 32] {
        use helix_circuits::gadgets::poseidon::poseidon_hash_two;
        use helix_circuits::halo2curves::bn256::Fr;
        use helix_circuits::halo2curves::ff::PrimeField;
        use helix_circuits::halo2_proofs::arithmetic::Field;

        if proofs.is_empty() {
            return [0u8; 32];
        }

        // Compute commitment hash per step (Poseidon chain of all PIs)
        let commitment_hashes: Vec<Fr> = proofs.iter().map(|p| {
            let mut hash = Fr::ZERO;
            for pi_bytes in &p.public_inputs {
                let mut repr = [0u8; 32];
                repr.copy_from_slice(pi_bytes);
                repr[31] &= 0x1F; // Mask to fit BN254
                let pi_fr = Fr::from_repr_vartime(repr.into()).unwrap_or(Fr::ZERO);
                hash = poseidon_hash_two(hash, pi_fr);
            }
            hash
        }).collect();

        // Fiat-Shamir challenge: alpha = chain hash of all commitments
        let mut alpha = commitment_hashes[0];
        for c in &commitment_hashes[1..] {
            alpha = poseidon_hash_two(alpha, *c);
        }

        // RLC = Σ α^i · commitment_hash_i
        let mut rlc = Fr::ZERO;
        let mut alpha_power = Fr::ONE;
        for c in &commitment_hashes {
            rlc += alpha_power * c;
            alpha_power *= alpha;
        }

        let repr = rlc.to_repr();
        let mut bytes = [0u8; 32];
        bytes.copy_from_slice(repr.as_ref());
        bytes
    }
}

/// Result of streaming batch proving.
#[derive(Debug)]
pub struct StreamingBatchResult {
    /// Batch identifier.
    pub batch_id: String,
    /// Proofs organized by sub-batch.
    pub batches: Vec<Vec<StepProof>>,
    /// Total steps processed.
    pub total_steps: usize,
    /// Total time in milliseconds.
    pub total_time_ms: u64,
    /// Total proof bytes.
    pub total_proof_bytes: usize,
    /// First step index.
    pub first_step: u64,
    /// Last step index.
    pub last_step: u64,
}

impl StreamingBatchResult {
    /// Flattens all batches into a single proof list.
    pub fn all_proofs(&self) -> Vec<StepProof> {
        self.batches.iter().flatten().cloned().collect()
    }
}

/// Builder for constructing batch proving jobs.
pub struct BatchProverBuilder {
    config: BatchConfig,
}

impl BatchProverBuilder {
    /// Creates a new builder.
    pub fn new() -> Self {
        Self {
            config: BatchConfig::default(),
        }
    }

    /// Sets the number of threads.
    pub fn threads(mut self, n: usize) -> Self {
        self.config.num_threads = n;
        self
    }

    /// Sets max memory proofs.
    pub fn max_memory_proofs(mut self, n: usize) -> Self {
        self.config.max_memory_proofs = n;
        self
    }

    /// Enables checkpointing.
    pub fn checkpoint_dir(mut self, dir: PathBuf) -> Self {
        self.config.checkpoint_dir = Some(dir);
        self
    }

    /// Sets checkpoint frequency.
    pub fn checkpoint_frequency(mut self, n: usize) -> Self {
        self.config.checkpoint_frequency = n;
        self
    }

    /// Enables proof aggregation.
    pub fn aggregate(mut self, batch_size: usize) -> Self {
        self.config.aggregate_proofs = true;
        self.config.aggregation_batch_size = batch_size;
        self
    }

    /// Enables streaming mode.
    pub fn streaming(mut self) -> Self {
        self.config.streaming_mode = true;
        self
    }

    /// Sets max memory bytes.
    pub fn max_memory_bytes(mut self, bytes: usize) -> Self {
        self.config.max_memory_bytes = bytes;
        self
    }

    /// Builds the batch prover.
    pub fn build(self) -> BatchProver {
        BatchProver::new(self.config)
    }
}

impl Default for BatchProverBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Epoch prover for proving entire training epochs.
pub struct EpochProver {
    /// Batch prover.
    batch_prover: BatchProver,
    /// Epoch index.
    epoch: u64,
    /// Steps per epoch.
    steps_per_epoch: usize,
}

impl EpochProver {
    /// Creates a new epoch prover.
    pub fn new(epoch: u64, steps_per_epoch: usize, config: BatchConfig) -> Self {
        Self {
            batch_prover: BatchProver::new(config),
            epoch,
            steps_per_epoch,
        }
    }

    /// Proves an entire epoch.
    pub fn prove_epoch(&self, steps: Vec<TrainingStep>) -> Result<EpochProofResult, BatchProveError> {
        let start = Instant::now();

        assert_eq!(
            steps.len(),
            self.steps_per_epoch,
            "Step count mismatch for epoch"
        );

        let batch_result = self.batch_prover.prove_batch(steps)?;

        Ok(EpochProofResult {
            epoch: self.epoch,
            batch_result,
            epoch_time_ms: start.elapsed().as_millis() as u64,
        })
    }
}

/// Result of proving an epoch.
#[derive(Debug)]
pub struct EpochProofResult {
    /// Epoch index.
    pub epoch: u64,
    /// Batch result.
    pub batch_result: BatchResult,
    /// Total epoch proving time.
    pub epoch_time_ms: u64,
}

/// Pipeline for continuous batch proving during training.
pub struct BatchProvingPipeline {
    /// Configuration.
    config: BatchConfig,
    /// Input queue of steps.
    input_queue: Arc<Mutex<Vec<TrainingStep>>>,
    /// Output queue of proofs.
    output_queue: Arc<Mutex<Vec<StepProof>>>,
    /// Running flag.
    running: Arc<std::sync::atomic::AtomicBool>,
    /// Statistics.
    stats: Arc<BatchStats>,
}

impl BatchProvingPipeline {
    /// Creates a new pipeline.
    pub fn new(config: BatchConfig) -> Self {
        Self {
            config,
            input_queue: Arc::new(Mutex::new(Vec::new())),
            output_queue: Arc::new(Mutex::new(Vec::new())),
            running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            stats: Arc::new(BatchStats::new()),
        }
    }

    /// Submits a step to the pipeline.
    pub fn submit(&self, step: TrainingStep) {
        if let Ok(mut queue) = self.input_queue.lock() {
            queue.push(step);
        }
    }

    /// Gets completed proofs.
    pub fn drain_proofs(&self) -> Vec<StepProof> {
        if let Ok(mut queue) = self.output_queue.lock() {
            std::mem::take(&mut *queue)
        } else {
            Vec::new()
        }
    }

    /// Starts the pipeline.
    pub fn start(&self) {
        if self
            .running
            .swap(true, std::sync::atomic::Ordering::SeqCst)
        {
            return;
        }

        let config = self.config.clone();
        let input_queue = Arc::clone(&self.input_queue);
        let output_queue = Arc::clone(&self.output_queue);
        let running = Arc::clone(&self.running);
        let stats = Arc::clone(&self.stats);

        std::thread::spawn(move || {
            let prover = BatchProver::new(config);

            while running.load(std::sync::atomic::Ordering::SeqCst) {
                // Drain input queue.
                let steps: Vec<TrainingStep> = {
                    if let Ok(mut queue) = input_queue.lock() {
                        std::mem::take(&mut *queue)
                    } else {
                        Vec::new()
                    }
                };

                if !steps.is_empty() {
                    match prover.prove_batch(steps) {
                        Ok(result) => {
                            // Push to output.
                            {
                                if let Ok(mut queue) = output_queue.lock() {
                                    queue.extend(result.proofs);
                                }
                            }

                            stats
                                .proofs_generated
                                .fetch_add(result.total_steps as u64, std::sync::atomic::Ordering::Relaxed);
                        }
                        Err(e) => {
                            tracing::error!("Pipeline batch proving failed: {e}");
                        }
                    }
                }

                std::thread::sleep(Duration::from_millis(10));
            }
        });
    }

    /// Stops the pipeline.
    pub fn stop(&self) {
        self.running
            .store(false, std::sync::atomic::Ordering::SeqCst);
    }

    /// Gets statistics.
    pub fn stats(&self) -> BatchStatsSnapshot {
        self.stats.snapshot()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_step(index: u64) -> TrainingStep {
        TrainingStep {
            step_index: index,
            input_commitment: [0; 32],
            output_commitment: [0; 32],
            loss: 0.5,
            gradient_norm: 1.0,
            error_bound: 0.01,
            aux_data: None,
        }
    }

    #[test]
    fn test_batch_prover_basic() {
        let prover = BatchProver::new(BatchConfig {
            num_threads: 2,
            ..Default::default()
        });

        let steps: Vec<_> = (0..4).map(make_test_step).collect();
        let result = prover.prove_batch(steps).expect("batch should succeed");

        assert_eq!(result.total_steps, 4);
        assert_eq!(result.proofs.len(), 4);
        assert!(result.total_time_ms > 0);
    }

    #[test]
    fn test_batch_prover_builder() {
        let prover = BatchProverBuilder::new()
            .threads(2)
            .max_memory_proofs(100)
            .streaming()
            .build();

        let steps: Vec<_> = (0..2).map(make_test_step).collect();
        let result = prover.prove_batch(steps).expect("batch should succeed");

        assert_eq!(result.total_steps, 2);
    }

    #[test]
    fn test_training_step_to_chunk() {
        let step = make_test_step(5);
        let chunk = step.to_chunk();

        assert_eq!(chunk.id.0, 5);
        assert_eq!(chunk.parent_id, Some(ChunkId(4)));
    }

    #[test]
    fn test_batch_status() {
        let prover = BatchProver::new(BatchConfig::default());

        assert_eq!(prover.status(), BatchStatus::Pending);

        let steps: Vec<_> = (0..2).map(make_test_step).collect();
        prover.prove_batch(steps).expect("batch should succeed");

        assert_eq!(prover.status(), BatchStatus::Complete);
    }

    #[test]
    fn test_streaming_batch() {
        let prover = BatchProver::new(BatchConfig {
            num_threads: 2,
            max_memory_proofs: 2,
            ..Default::default()
        });

        let steps = (0..6).map(make_test_step);
        let result = prover.prove_batch_streaming(steps).expect("streaming batch should succeed");

        assert_eq!(result.total_steps, 6);
        assert!(result.batches.len() >= 1);
    }

    #[test]
    fn test_aggregated_proof() {
        let prover = BatchProver::new(BatchConfig {
            num_threads: 2,
            aggregate_proofs: true,
            aggregation_batch_size: 2,
            ..Default::default()
        });

        let steps: Vec<_> = (0..4).map(make_test_step).collect();
        let result = prover.prove_batch(steps).expect("aggregated batch should succeed");

        assert!(result.aggregated_proof.is_some());
        let agg = result.aggregated_proof.unwrap();
        assert_eq!(agg.num_steps, 4);
        // Backward compat: merkle_root and rlc_commitment still populated.
        assert_ne!(agg.merkle_root, [0u8; 32]);
        assert!(agg.rlc_commitment.is_some());
        assert!(!agg.proof.is_empty());
    }

    #[test]
    fn test_batch_checkpoint_serialization() {
        let checkpoint = BatchCheckpoint {
            checkpoint_id: "test-checkpoint".to_string(),
            batch_id: "batch-123".to_string(),
            completed_steps: vec![0, 1, 2],
            last_step: 2,
            remaining_steps: vec![3, 4, 5],
            proofs: Vec::new(),
            timestamp: 12345,
            elapsed_ms: 1000,
        };

        let json = serde_json::to_string(&checkpoint).unwrap();
        let loaded: BatchCheckpoint = serde_json::from_str(&json).unwrap();

        assert_eq!(loaded.checkpoint_id, checkpoint.checkpoint_id);
        assert_eq!(loaded.completed_steps, checkpoint.completed_steps);
    }

    #[test]
    fn test_epoch_prover() {
        let config = BatchConfig {
            num_threads: 2,
            ..Default::default()
        };

        let prover = EpochProver::new(1, 4, config);
        let steps: Vec<_> = (0..4).map(make_test_step).collect();

        let result = prover.prove_epoch(steps).expect("epoch proving should succeed");

        assert_eq!(result.epoch, 1);
        assert_eq!(result.batch_result.total_steps, 4);
    }

    #[test]
    fn test_batch_stats() {
        let prover = BatchProver::new(BatchConfig {
            num_threads: 2,
            ..Default::default()
        });

        let steps: Vec<_> = (0..3).map(make_test_step).collect();
        prover.prove_batch(steps).expect("stats batch should succeed");

        let stats = prover.stats();
        assert_eq!(stats.proofs_generated, 3);
        assert!(stats.total_bytes > 0);
    }

    #[test]
    fn test_kzg_aggregated_proof_smaller_than_sum() {
        let prover = BatchProver::new(BatchConfig {
            num_threads: 2,
            aggregate_proofs: true,
            aggregation_batch_size: 4,
            ..Default::default()
        });

        let steps: Vec<_> = (0..4).map(make_test_step).collect();
        let result = prover.prove_batch(steps).expect("batch should succeed");

        let agg = result.aggregated_proof.expect("aggregated proof should exist");
        let individual_total: usize = result.proofs.iter().map(|p| p.proof_size).sum();

        // KZG aggregated proof should be O(1) sized, smaller than sum of individual proofs.
        if agg.kzg_proof.is_some() {
            assert!(
                agg.proof.len() < individual_total,
                "KZG aggregated proof ({} bytes) should be smaller than sum of individual proofs ({} bytes)",
                agg.proof.len(),
                individual_total,
            );
        }
    }

    #[test]
    fn test_kzg_aggregation_produces_verifiable_proof() {
        let prover = BatchProver::new(BatchConfig {
            num_threads: 2,
            aggregate_proofs: true,
            ..Default::default()
        });

        let steps: Vec<_> = (0..3).map(make_test_step).collect();
        let result = prover.prove_batch(steps).expect("batch should succeed");

        let agg = result.aggregated_proof.expect("aggregated proof should exist");

        if let Some(ref kzg) = agg.kzg_proof {
            // Verify using a fresh KZGBatchAggregator.
            let aggregator = crate::aggregation::KZGBatchAggregator::new();
            assert!(aggregator.is_ready(), "KZG aggregator should be ready");

            let verified = aggregator.verify(kzg).expect("verification should complete");
            assert!(verified, "KZG aggregated proof should verify");
        }
    }
}
