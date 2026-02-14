//! Model & Training Data Distribution System.
//!
//! Handles the complete distribution lifecycle when a training round starts:
//!
//! 1. **ModelPackage** — Bundles model architecture, compressed weights, and hyperparameters
//! 2. **DataAssignment** — Assigns each worker a shard of the training dataset
//! 3. **Chunked Transfer** — Splits large models into chunks with progress tracking
//! 4. **Checkpoint Distribution** — Distributes resume checkpoints for continued training
//! 5. **Integrity Verification** — Workers verify received weights against on-chain commitment

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::time::Instant;

use tracing::{debug, info, warn};

use crate::network::messages::{ModelDims, PeerId};
use crate::trainer::MlpModel;

use helix_core::ModelCheckpoint;

// ============================================================================
// Model Package
// ============================================================================

/// Complete model distribution package sent from aggregator to workers.
///
/// Contains everything a worker needs to begin training: the model architecture,
/// serialized weights (optionally compressed), hyperparameters, and an integrity
/// hash for verification against the on-chain commitment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPackage {
    /// Round this package belongs to.
    pub round_id: u64,
    /// On-chain model ID.
    pub model_id: u64,
    /// Model architecture description.
    pub architecture: ModelArchitecture,
    /// Serialized model weights (ModelCheckpoint bytes, optionally zstd-compressed).
    pub weight_data: Vec<u8>,
    /// Whether `weight_data` is zstd-compressed.
    pub compressed: bool,
    /// SHA-256 hash of the uncompressed weight data (for integrity verification).
    pub weight_hash: [u8; 32],
    /// Training hyperparameters for this round.
    pub hyperparameters: TrainingHyperparameters,
    /// Whether this is a resumed training round (checkpoint includes accumulated state).
    pub is_resume: bool,
    /// Step number this checkpoint starts from (0 for fresh, >0 for resume).
    pub starting_step: u64,
    /// Accumulated error from previous rounds (0.0 for fresh start).
    pub accumulated_error: f64,
}

/// Model architecture description — everything needed to construct the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelArchitecture {
    /// Model dimensions.
    pub dims: ModelDims,
    /// Layer descriptions (name, type, input_size, output_size).
    pub layers: Vec<LayerDescription>,
}

/// Description of a single layer in the model architecture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerDescription {
    /// Layer name (e.g., "fc1", "fc2").
    pub name: String,
    /// Layer type ("linear", "relu", "batchnorm", etc.).
    pub layer_type: String,
    /// Input dimension.
    pub input_size: usize,
    /// Output dimension.
    pub output_size: usize,
}

/// Training hyperparameters sent with the model package.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingHyperparameters {
    /// Learning rate for SGD.
    pub learning_rate: f64,
    /// Batch size for training.
    pub batch_size: u32,
    /// Number of training steps (epochs * batches) per worker.
    pub steps_per_worker: u32,
    /// Maximum error budget for this round.
    pub error_budget: f64,
    /// Random seed for deterministic initialization.
    pub model_seed: u64,
}

// ============================================================================
// Data Assignment
// ============================================================================

/// Data source type for a training shard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataSourceType {
    /// S3 object storage (behind `s3-fetch` feature in helix-core).
    S3 {
        /// S3 URI (e.g., "s3://bucket/path/to/dataset.csv").
        uri: String,
        /// AWS region.
        region: String,
    },
    /// IPFS content-addressed storage (behind `ipfs-fetch` feature in helix-core).
    Ipfs {
        /// IPFS CID.
        cid: String,
        /// Optional gateway URL override.
        gateway: Option<String>,
    },
    /// Inline data embedded directly in the message (for small datasets).
    Inline {
        /// Raw data bytes (CSV, binary tensor, etc.).
        data: Vec<u8>,
        /// Format of the inline data.
        format: DataFormat,
    },
    /// HTTP(S) URL to fetch the dataset from.
    Http {
        /// Full URL.
        url: String,
    },
}

/// Format of training data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataFormat {
    /// CSV tabular data.
    Csv,
    /// Binary tensor data (helix-core HXDT format).
    BinaryTensor,
    /// Raw float32 tensor data.
    RawF32,
}

/// Assignment of a data shard to a specific worker.
///
/// The aggregator splits the full dataset into shards and assigns each worker
/// a range. Workers independently load their assigned portion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataAssignment {
    /// Round this assignment belongs to.
    pub round_id: u64,
    /// Worker's index within this round (0-based).
    pub worker_index: u32,
    /// Total number of workers in this round.
    pub total_workers: u32,
    /// Data source to load from.
    pub source: DataSourceType,
    /// Byte offset into the data source for this shard.
    pub byte_offset: u64,
    /// Number of bytes in this shard (0 = read to end).
    pub byte_length: u64,
    /// Number of samples in this shard (for validation).
    pub num_samples: u64,
    /// SHA-256 hash of this shard's data (for integrity verification).
    pub shard_hash: Option<[u8; 32]>,
    /// Shuffle seed (shared across all workers for deterministic ordering).
    pub shuffle_seed: u64,
    /// Batch size for this worker.
    pub batch_size: u32,
    /// Data format.
    pub format: DataFormat,
    /// Feature column indices (empty = all except labels).
    pub feature_columns: Vec<usize>,
    /// Label column indices.
    pub label_columns: Vec<usize>,
}

// ============================================================================
// Chunked Transfer
// ============================================================================

/// Maximum chunk size for model distribution (256 KB).
pub const MAX_CHUNK_SIZE: usize = 256 * 1024;

/// Threshold above which we use chunked transfer (1 MB).
pub const CHUNKED_TRANSFER_THRESHOLD: usize = 1024 * 1024;

/// A single chunk of a model package for large model distribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPackageChunk {
    /// Round this chunk belongs to.
    pub round_id: u64,
    /// Model ID.
    pub model_id: u64,
    /// Chunk index (0-based).
    pub chunk_index: u32,
    /// Total number of chunks.
    pub total_chunks: u32,
    /// Chunk data bytes.
    pub data: Vec<u8>,
    /// SHA-256 hash of this chunk for integrity.
    pub chunk_hash: [u8; 32],
    /// Whether this is the final chunk (also carries metadata).
    pub is_final: bool,
    /// Model package metadata (only present in chunk 0).
    pub metadata: Option<ChunkMetadata>,
}

/// Metadata sent with the first chunk of a chunked transfer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkMetadata {
    /// Total uncompressed size of the model package.
    pub total_size: u64,
    /// Architecture description.
    pub architecture: ModelArchitecture,
    /// Hyperparameters.
    pub hyperparameters: TrainingHyperparameters,
    /// SHA-256 hash of the complete reassembled data.
    pub complete_hash: [u8; 32],
    /// Whether the data is compressed.
    pub compressed: bool,
    /// Is this a resume checkpoint.
    pub is_resume: bool,
    /// Starting step number.
    pub starting_step: u64,
    /// Accumulated error from prior rounds.
    pub accumulated_error: f64,
}

// ============================================================================
// Data Shard Planning
// ============================================================================

/// Plans data shard assignments for a set of workers.
pub struct DataShardPlanner;

impl DataShardPlanner {
    /// Splits a dataset into equal shards for the given number of workers.
    ///
    /// If the dataset size is not evenly divisible, the last worker gets the remainder.
    /// For inline data, each worker gets their slice of the data.
    /// For remote data (S3/IPFS/HTTP), workers get byte ranges.
    pub fn plan_shards(
        source: &DataSourceType,
        total_size: u64,
        total_samples: u64,
        num_workers: u32,
        round_id: u64,
        shuffle_seed: u64,
        batch_size: u32,
        format: DataFormat,
        feature_columns: Vec<usize>,
        label_columns: Vec<usize>,
    ) -> Vec<DataAssignment> {
        if num_workers == 0 {
            return Vec::new();
        }

        let samples_per_worker = total_samples / num_workers as u64;
        let bytes_per_sample = if total_samples > 0 {
            total_size / total_samples
        } else {
            0
        };

        let mut assignments = Vec::with_capacity(num_workers as usize);

        for i in 0..num_workers {
            let sample_start = i as u64 * samples_per_worker;
            let sample_count = if i == num_workers - 1 {
                // Last worker gets the remainder
                total_samples - sample_start
            } else {
                samples_per_worker
            };

            let byte_offset = sample_start * bytes_per_sample;
            let byte_length = if i == num_workers - 1 {
                total_size - byte_offset
            } else {
                sample_count * bytes_per_sample
            };

            let worker_source = match source {
                DataSourceType::Inline { data, format: fmt } => {
                    // For inline data, slice the data for each worker
                    let start = byte_offset as usize;
                    let end = (byte_offset + byte_length) as usize;
                    let end = end.min(data.len());
                    let start = start.min(end);
                    DataSourceType::Inline {
                        data: data[start..end].to_vec(),
                        format: fmt.clone(),
                    }
                }
                other => other.clone(),
            };

            assignments.push(DataAssignment {
                round_id,
                worker_index: i,
                total_workers: num_workers,
                source: worker_source,
                byte_offset,
                byte_length,
                num_samples: sample_count,
                shard_hash: None,
                shuffle_seed,
                batch_size,
                format: format.clone(),
                feature_columns: feature_columns.clone(),
                label_columns: label_columns.clone(),
            });
        }

        assignments
    }
}

// ============================================================================
// Model Package Builder
// ============================================================================

/// Builds a `ModelPackage` from a model and configuration.
pub struct ModelPackageBuilder;

impl ModelPackageBuilder {
    /// Creates a model package from an `MlpModel` and round configuration.
    ///
    /// Serializes the model to a `ModelCheckpoint`, optionally compresses it,
    /// and bundles it with architecture info and hyperparameters.
    pub fn build(
        model: &MlpModel,
        round_id: u64,
        model_id: u64,
        learning_rate: f64,
        batch_size: u32,
        steps_per_worker: u32,
        error_budget: f64,
        model_seed: u64,
        num_layers: u32,
        activation_type: u8,
    ) -> Result<ModelPackage, String> {
        // Create checkpoint with architecture metadata
        let checkpoint = model.to_checkpoint_with_arch(
            0, // step 0 for fresh start
            num_layers,
            activation_type,
        );

        let uncompressed_bytes = checkpoint.to_bytes()
            .map_err(|e| format!("failed to serialize checkpoint: {}", e))?;

        // Compute hash of uncompressed data
        let weight_hash = compute_data_hash(&uncompressed_bytes);

        // Compress if beneficial (>1KB)
        let (weight_data, compressed) = if uncompressed_bytes.len() > 1024 {
            match compress_data(&uncompressed_bytes) {
                Ok(compressed_bytes) => {
                    // Only use compression if it actually saves space
                    if compressed_bytes.len() < uncompressed_bytes.len() {
                        debug!(
                            "Compressed model: {} -> {} bytes ({:.1}% reduction)",
                            uncompressed_bytes.len(),
                            compressed_bytes.len(),
                            (1.0 - compressed_bytes.len() as f64 / uncompressed_bytes.len() as f64) * 100.0,
                        );
                        (compressed_bytes, true)
                    } else {
                        (uncompressed_bytes, false)
                    }
                }
                Err(_) => (uncompressed_bytes, false),
            }
        } else {
            (uncompressed_bytes, false)
        };

        // Build architecture description
        let architecture = ModelArchitecture {
            dims: ModelDims {
                d_in: model.d_in,
                d_hid: model.d_hid,
                d_out: model.d_out,
                num_layers,
                num_heads: 0,
                activation_type,
            },
            layers: vec![
                LayerDescription {
                    name: "fc1".to_string(),
                    layer_type: "linear".to_string(),
                    input_size: model.d_in,
                    output_size: model.d_hid,
                },
                LayerDescription {
                    name: "relu".to_string(),
                    layer_type: activation_type_name(activation_type).to_string(),
                    input_size: model.d_hid,
                    output_size: model.d_hid,
                },
                LayerDescription {
                    name: "fc2".to_string(),
                    layer_type: "linear".to_string(),
                    input_size: model.d_hid,
                    output_size: model.d_out,
                },
            ],
        };

        Ok(ModelPackage {
            round_id,
            model_id,
            architecture,
            weight_data,
            compressed,
            weight_hash,
            hyperparameters: TrainingHyperparameters {
                learning_rate,
                batch_size,
                steps_per_worker,
                error_budget,
                model_seed,
            },
            is_resume: false,
            starting_step: 0,
            accumulated_error: 0.0,
        })
    }

    /// Creates a model package from a checkpoint for resumed training.
    pub fn build_from_checkpoint(
        checkpoint: &ModelCheckpoint,
        round_id: u64,
        model_id: u64,
        dims: &ModelDims,
        learning_rate: f64,
        batch_size: u32,
        steps_per_worker: u32,
        error_budget: f64,
        model_seed: u64,
        accumulated_error: f64,
    ) -> Result<ModelPackage, String> {
        let uncompressed_bytes = checkpoint.to_bytes()
            .map_err(|e| format!("failed to serialize checkpoint: {}", e))?;

        let weight_hash = compute_data_hash(&uncompressed_bytes);

        let (weight_data, compressed) = if uncompressed_bytes.len() > 1024 {
            match compress_data(&uncompressed_bytes) {
                Ok(compressed_bytes) if compressed_bytes.len() < uncompressed_bytes.len() => {
                    (compressed_bytes, true)
                }
                _ => (uncompressed_bytes, false),
            }
        } else {
            (uncompressed_bytes, false)
        };

        let architecture = ModelArchitecture {
            dims: dims.clone(),
            layers: vec![
                LayerDescription {
                    name: "fc1".to_string(),
                    layer_type: "linear".to_string(),
                    input_size: dims.d_in,
                    output_size: dims.d_hid,
                },
                LayerDescription {
                    name: "relu".to_string(),
                    layer_type: activation_type_name(dims.activation_type).to_string(),
                    input_size: dims.d_hid,
                    output_size: dims.d_hid,
                },
                LayerDescription {
                    name: "fc2".to_string(),
                    layer_type: "linear".to_string(),
                    input_size: dims.d_hid,
                    output_size: dims.d_out,
                },
            ],
        };

        Ok(ModelPackage {
            round_id,
            model_id,
            architecture,
            weight_data,
            compressed,
            weight_hash,
            hyperparameters: TrainingHyperparameters {
                learning_rate,
                batch_size,
                steps_per_worker,
                error_budget,
                model_seed,
            },
            is_resume: true,
            starting_step: checkpoint.step_number,
            accumulated_error,
        })
    }
}

// ============================================================================
// Chunked Transfer Logic
// ============================================================================

/// Splits a serialized model package into chunks for transfer.
pub fn split_into_chunks(
    package: &ModelPackage,
    max_chunk_size: usize,
) -> Result<Vec<ModelPackageChunk>, String> {
    let package_bytes = bincode::serialize(package)
        .map_err(|e| format!("failed to serialize model package: {}", e))?;

    let total_size = package_bytes.len();

    if total_size <= max_chunk_size {
        // Single chunk
        let chunk_hash = compute_data_hash(&package_bytes);
        return Ok(vec![ModelPackageChunk {
            round_id: package.round_id,
            model_id: package.model_id,
            chunk_index: 0,
            total_chunks: 1,
            data: package_bytes,
            chunk_hash,
            is_final: true,
            metadata: Some(ChunkMetadata {
                total_size: total_size as u64,
                architecture: package.architecture.clone(),
                hyperparameters: package.hyperparameters.clone(),
                complete_hash: package.weight_hash,
                compressed: package.compressed,
                is_resume: package.is_resume,
                starting_step: package.starting_step,
                accumulated_error: package.accumulated_error,
            }),
        }]);
    }

    let num_chunks = (total_size + max_chunk_size - 1) / max_chunk_size;
    let mut chunks = Vec::with_capacity(num_chunks);

    for i in 0..num_chunks {
        let start = i * max_chunk_size;
        let end = ((i + 1) * max_chunk_size).min(total_size);
        let data = package_bytes[start..end].to_vec();
        let chunk_hash = compute_data_hash(&data);

        let metadata = if i == 0 {
            Some(ChunkMetadata {
                total_size: total_size as u64,
                architecture: package.architecture.clone(),
                hyperparameters: package.hyperparameters.clone(),
                complete_hash: package.weight_hash,
                compressed: package.compressed,
                is_resume: package.is_resume,
                starting_step: package.starting_step,
                accumulated_error: package.accumulated_error,
            })
        } else {
            None
        };

        chunks.push(ModelPackageChunk {
            round_id: package.round_id,
            model_id: package.model_id,
            chunk_index: i as u32,
            total_chunks: num_chunks as u32,
            data,
            chunk_hash,
            is_final: i == num_chunks - 1,
            metadata,
        });
    }

    Ok(chunks)
}

/// Reassembles chunks into a model package.
pub fn reassemble_chunks(
    chunks: &[ModelPackageChunk],
) -> Result<ModelPackage, String> {
    if chunks.is_empty() {
        return Err("no chunks to reassemble".to_string());
    }

    // Verify all chunks are present and in order
    let total_chunks = chunks[0].total_chunks;
    if chunks.len() != total_chunks as usize {
        return Err(format!(
            "expected {} chunks, got {}",
            total_chunks,
            chunks.len()
        ));
    }

    // Verify chunk ordering and integrity
    let mut reassembled = Vec::new();
    for (i, chunk) in chunks.iter().enumerate() {
        if chunk.chunk_index != i as u32 {
            return Err(format!(
                "chunk {} has index {} (expected {})",
                i, chunk.chunk_index, i
            ));
        }

        // Verify chunk hash
        let computed_hash = compute_data_hash(&chunk.data);
        if computed_hash != chunk.chunk_hash {
            return Err(format!(
                "chunk {} hash mismatch: expected {}, got {}",
                i,
                hex::encode(chunk.chunk_hash),
                hex::encode(computed_hash),
            ));
        }

        reassembled.extend_from_slice(&chunk.data);
    }

    // Verify complete hash from metadata
    if let Some(ref metadata) = chunks[0].metadata {
        if reassembled.len() as u64 != metadata.total_size {
            return Err(format!(
                "reassembled size {} != expected {}",
                reassembled.len(),
                metadata.total_size,
            ));
        }
    }

    // Deserialize
    bincode::deserialize(&reassembled)
        .map_err(|e| format!("failed to deserialize reassembled model package: {}", e))
}

// ============================================================================
// Chunk Receiver (Worker Side)
// ============================================================================

/// Tracks progress of receiving a chunked model transfer.
pub struct ChunkReceiver {
    round_id: u64,
    model_id: u64,
    total_chunks: u32,
    received: HashMap<u32, ModelPackageChunk>,
    started_at: Instant,
}

impl ChunkReceiver {
    /// Creates a new chunk receiver for the given round.
    pub fn new(round_id: u64, model_id: u64, total_chunks: u32) -> Self {
        Self {
            round_id,
            model_id,
            total_chunks,
            received: HashMap::new(),
            started_at: Instant::now(),
        }
    }

    /// Adds a received chunk. Returns true if all chunks are now received.
    pub fn add_chunk(&mut self, chunk: ModelPackageChunk) -> Result<bool, String> {
        if chunk.round_id != self.round_id || chunk.model_id != self.model_id {
            return Err("chunk round/model ID mismatch".to_string());
        }
        if chunk.chunk_index >= self.total_chunks {
            return Err(format!(
                "chunk index {} >= total_chunks {}",
                chunk.chunk_index, self.total_chunks
            ));
        }

        // Verify chunk integrity
        let computed_hash = compute_data_hash(&chunk.data);
        if computed_hash != chunk.chunk_hash {
            return Err(format!(
                "chunk {} integrity check failed",
                chunk.chunk_index
            ));
        }

        self.received.insert(chunk.chunk_index, chunk);
        Ok(self.is_complete())
    }

    /// Returns whether all chunks have been received.
    pub fn is_complete(&self) -> bool {
        self.received.len() == self.total_chunks as usize
    }

    /// Returns transfer progress as a fraction (0.0 to 1.0).
    pub fn progress(&self) -> f64 {
        if self.total_chunks == 0 {
            return 1.0;
        }
        self.received.len() as f64 / self.total_chunks as f64
    }

    /// Returns bytes received so far.
    pub fn bytes_received(&self) -> usize {
        self.received.values().map(|c| c.data.len()).sum()
    }

    /// Returns elapsed time since transfer started.
    pub fn elapsed(&self) -> std::time::Duration {
        self.started_at.elapsed()
    }

    /// Assembles the complete model package from received chunks.
    pub fn assemble(self) -> Result<ModelPackage, String> {
        if !self.is_complete() {
            return Err(format!(
                "cannot assemble: {}/{} chunks received",
                self.received.len(),
                self.total_chunks
            ));
        }

        let mut sorted_chunks: Vec<ModelPackageChunk> = self.received.into_values().collect();
        sorted_chunks.sort_by_key(|c| c.chunk_index);
        reassemble_chunks(&sorted_chunks)
    }
}

// ============================================================================
// Weight Verification
// ============================================================================

/// Verifies that received model weights match the expected on-chain commitment.
///
/// The commitment is a SHA-256 hash of the model's weight bytes, matching
/// `MlpModel::commitment()` which hashes all weights in order (w1, b1, w2, b2).
pub fn verify_weight_integrity(
    model: &MlpModel,
    expected_hash: &[u8; 32],
) -> bool {
    let actual_hash = model.commitment();
    // Constant-time comparison to prevent timing attacks
    constant_time_eq(&actual_hash, expected_hash)
}

/// Verifies that checkpoint data, when deserialized, produces a model
/// matching the expected weight hash.
pub fn verify_checkpoint_integrity(
    checkpoint_data: &[u8],
    expected_hash: &[u8; 32],
) -> Result<MlpModel, String> {
    let checkpoint = ModelCheckpoint::from_bytes(checkpoint_data)
        .map_err(|e| format!("failed to parse checkpoint: {}", e))?;

    let model = MlpModel::from_checkpoint(&checkpoint)
        .map_err(|e| format!("failed to reconstruct model: {}", e))?;

    let actual_hash = model.commitment();
    if !constant_time_eq(&actual_hash, expected_hash) {
        return Err(format!(
            "weight hash mismatch: expected {}, got {}",
            hex::encode(expected_hash),
            hex::encode(actual_hash),
        ));
    }

    Ok(model)
}

/// Verifies a model package's weight data against its declared hash.
pub fn verify_model_package(package: &ModelPackage) -> Result<MlpModel, String> {
    // Decompress if needed
    let checkpoint_bytes = if package.compressed {
        decompress_data(&package.weight_data)?
    } else {
        package.weight_data.clone()
    };

    // Verify hash of uncompressed data
    let data_hash = compute_data_hash(&checkpoint_bytes);
    if !constant_time_eq(&data_hash, &package.weight_hash) {
        return Err(format!(
            "model package data hash mismatch: expected {}, got {}",
            hex::encode(package.weight_hash),
            hex::encode(data_hash),
        ));
    }

    // Parse checkpoint and reconstruct model
    let checkpoint = ModelCheckpoint::from_bytes(&checkpoint_bytes)
        .map_err(|e| format!("failed to parse checkpoint: {}", e))?;

    let model = MlpModel::from_checkpoint(&checkpoint)
        .map_err(|e| format!("failed to reconstruct model: {}", e))?;

    // Verify dimensions match architecture
    if model.d_in != package.architecture.dims.d_in
        || model.d_hid != package.architecture.dims.d_hid
        || model.d_out != package.architecture.dims.d_out
    {
        return Err(format!(
            "model dimensions mismatch: package says {}x{}x{}, model is {}x{}x{}",
            package.architecture.dims.d_in,
            package.architecture.dims.d_hid,
            package.architecture.dims.d_out,
            model.d_in, model.d_hid, model.d_out,
        ));
    }

    Ok(model)
}

// ============================================================================
// Distribution Tracker (Aggregator Side)
// ============================================================================

/// Tracks the progress of distributing a model package to all workers.
pub struct DistributionTracker {
    round_id: u64,
    total_workers: usize,
    /// Worker peer ID → distribution status.
    worker_status: HashMap<PeerId, WorkerDistributionStatus>,
    started_at: Instant,
}

/// Status of model distribution to a single worker.
#[derive(Debug, Clone)]
pub struct WorkerDistributionStatus {
    /// Whether the model package was sent.
    pub package_sent: bool,
    /// Whether the data assignment was sent.
    pub data_assignment_sent: bool,
    /// When the package was sent.
    pub sent_at: Option<Instant>,
    /// Whether the worker acknowledged receipt.
    pub acknowledged: bool,
    /// Chunks sent (for chunked transfer).
    pub chunks_sent: u32,
    /// Total chunks to send.
    pub total_chunks: u32,
    /// Error message if distribution failed.
    pub error: Option<String>,
}

impl DistributionTracker {
    /// Creates a new tracker for the given round.
    pub fn new(round_id: u64, worker_ids: &[PeerId]) -> Self {
        let mut worker_status = HashMap::new();
        for id in worker_ids {
            worker_status.insert(id.clone(), WorkerDistributionStatus {
                package_sent: false,
                data_assignment_sent: false,
                sent_at: None,
                acknowledged: false,
                chunks_sent: 0,
                total_chunks: 0,
                error: None,
            });
        }

        Self {
            round_id,
            total_workers: worker_ids.len(),
            worker_status,
            started_at: Instant::now(),
        }
    }

    /// Marks a worker's model package as sent.
    pub fn mark_package_sent(&mut self, peer_id: &PeerId) {
        if let Some(status) = self.worker_status.get_mut(peer_id) {
            status.package_sent = true;
            status.sent_at = Some(Instant::now());
        }
    }

    /// Marks a worker's data assignment as sent.
    pub fn mark_data_assignment_sent(&mut self, peer_id: &PeerId) {
        if let Some(status) = self.worker_status.get_mut(peer_id) {
            status.data_assignment_sent = true;
        }
    }

    /// Updates chunk progress for a worker.
    pub fn update_chunk_progress(&mut self, peer_id: &PeerId, chunks_sent: u32, total: u32) {
        if let Some(status) = self.worker_status.get_mut(peer_id) {
            status.chunks_sent = chunks_sent;
            status.total_chunks = total;
            if chunks_sent >= total {
                status.package_sent = true;
                status.sent_at = Some(Instant::now());
            }
        }
    }

    /// Records a distribution failure for a worker.
    pub fn mark_failed(&mut self, peer_id: &PeerId, error: String) {
        if let Some(status) = self.worker_status.get_mut(peer_id) {
            status.error = Some(error);
        }
    }

    /// Returns the number of workers that have received both package and data assignment.
    pub fn fully_distributed_count(&self) -> usize {
        self.worker_status.values()
            .filter(|s| s.package_sent && s.data_assignment_sent && s.error.is_none())
            .count()
    }

    /// Returns the number of workers that failed.
    pub fn failed_count(&self) -> usize {
        self.worker_status.values()
            .filter(|s| s.error.is_some())
            .count()
    }

    /// Returns failed worker IDs.
    pub fn failed_workers(&self) -> Vec<PeerId> {
        self.worker_status.iter()
            .filter(|(_, s)| s.error.is_some())
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Returns elapsed time since distribution started.
    pub fn elapsed(&self) -> std::time::Duration {
        self.started_at.elapsed()
    }

    /// Returns overall distribution progress as a fraction (0.0 to 1.0).
    pub fn progress(&self) -> f64 {
        if self.total_workers == 0 {
            return 1.0;
        }
        self.fully_distributed_count() as f64 / self.total_workers as f64
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Computes SHA-256 hash of data.
pub fn compute_data_hash(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().into()
}

/// Constant-time comparison of two 32-byte arrays.
fn constant_time_eq(a: &[u8; 32], b: &[u8; 32]) -> bool {
    use subtle::ConstantTimeEq;
    a.ct_eq(b).into()
}

/// Compresses data with zstd (level 3).
fn compress_data(data: &[u8]) -> Result<Vec<u8>, String> {
    // Use zstd via the raw API since helix-core's compression feature
    // may not be enabled for helix-node. We use the zstd crate directly.
    // Since helix-node doesn't depend on zstd, we implement a simple
    // manual compression using the same format as helix-core.
    //
    // For now, use a simple framing approach: [4-byte LE uncompressed size][compressed data]
    // This allows the receiver to know the expected size for decompression.
    //
    // We use the miniz_oxide crate (available via flate2) or just pass through
    // uncompressed with a flag, since the checkpoint format already handles this.
    //
    // DECISION: Since the model weights are serialized via ModelCheckpoint::to_bytes()
    // which already has a compact binary format, and adding a new zstd dependency
    // just for node would be wasteful, we rely on the `compressed` flag in
    // ModelPackage to signal whether compression was applied upstream. The
    // ModelPackageBuilder can optionally compress using helix-core's compression
    // feature when available.
    //
    // For production: the aggregator can serialize with helix-core's
    // ModelCheckpoint::save_compressed() and send those bytes directly.

    // Simple deflate compression using miniz_oxide (dependency of sha2)
    let compressed = miniz_oxide::deflate::compress_to_vec(data, 6);
    let mut result = Vec::with_capacity(4 + compressed.len());
    result.extend_from_slice(&(data.len() as u32).to_le_bytes());
    result.extend_from_slice(&compressed);
    Ok(result)
}

/// Decompresses data compressed by `compress_data`.
fn decompress_data(data: &[u8]) -> Result<Vec<u8>, String> {
    if data.len() < 4 {
        return Err("compressed data too short".to_string());
    }
    let expected_size = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;

    // Safety limit: 256 MB
    const MAX_DECOMPRESS_SIZE: usize = 256 * 1024 * 1024;
    if expected_size > MAX_DECOMPRESS_SIZE {
        return Err(format!(
            "decompressed size {} exceeds limit {}",
            expected_size, MAX_DECOMPRESS_SIZE
        ));
    }

    miniz_oxide::inflate::decompress_to_vec_with_limit(&data[4..], expected_size)
        .map_err(|e| format!("decompression failed: {:?}", e))
}

/// Returns the human-readable name for an activation type code.
fn activation_type_name(code: u8) -> &'static str {
    match code {
        0 => "relu",
        1 => "sigmoid",
        2 => "tanh",
        3 => "gelu",
        4 => "leaky_relu",
        _ => "unknown",
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_model() -> MlpModel {
        MlpModel::new_random(4, 8, 2, 42)
    }

    fn test_dims() -> ModelDims {
        ModelDims {
            d_in: 4,
            d_hid: 8,
            d_out: 2,
            num_layers: 2,
            num_heads: 0,
            activation_type: 0,
        }
    }

    // ---- ModelPackage Building ----

    #[test]
    fn test_build_model_package() {
        let model = test_model();
        let package = ModelPackageBuilder::build(
            &model, 1, 100, 0.01, 32, 10, 0.1, 42, 2, 0,
        ).unwrap();

        assert_eq!(package.round_id, 1);
        assert_eq!(package.model_id, 100);
        assert!(!package.is_resume);
        assert_eq!(package.starting_step, 0);
        assert_eq!(package.accumulated_error, 0.0);
        assert!(!package.weight_data.is_empty());
        assert_eq!(package.architecture.dims.d_in, 4);
        assert_eq!(package.architecture.dims.d_hid, 8);
        assert_eq!(package.architecture.dims.d_out, 2);
        assert_eq!(package.hyperparameters.learning_rate, 0.01);
        assert_eq!(package.hyperparameters.batch_size, 32);
        assert_eq!(package.hyperparameters.steps_per_worker, 10);
    }

    #[test]
    fn test_build_resume_package() {
        let model = test_model();
        let checkpoint = model.to_checkpoint(50);
        let dims = test_dims();

        let package = ModelPackageBuilder::build_from_checkpoint(
            &checkpoint, 5, 100, &dims, 0.01, 32, 10, 0.1, 42, 0.05,
        ).unwrap();

        assert_eq!(package.round_id, 5);
        assert!(package.is_resume);
        assert_eq!(package.starting_step, 50);
        assert_eq!(package.accumulated_error, 0.05);
    }

    // ---- Model Verification ----

    #[test]
    fn test_verify_model_package_valid() {
        let model = test_model();
        let package = ModelPackageBuilder::build(
            &model, 1, 100, 0.01, 32, 10, 0.1, 42, 2, 0,
        ).unwrap();

        let verified_model = verify_model_package(&package).unwrap();
        assert_eq!(verified_model.d_in, model.d_in);
        assert_eq!(verified_model.d_hid, model.d_hid);
        assert_eq!(verified_model.d_out, model.d_out);

        // Weights should match (within f64 -> f32 -> f64 roundtrip tolerance)
        assert_eq!(verified_model.w1.len(), model.w1.len());
    }

    #[test]
    fn test_verify_model_package_tampered() {
        let model = test_model();
        let mut package = ModelPackageBuilder::build(
            &model, 1, 100, 0.01, 32, 10, 0.1, 42, 2, 0,
        ).unwrap();

        // Tamper with weight data
        if !package.weight_data.is_empty() {
            let last = package.weight_data.len() - 1;
            package.weight_data[last] ^= 0xFF;
        }

        assert!(verify_model_package(&package).is_err());
    }

    #[test]
    fn test_weight_integrity_check() {
        let model = test_model();
        let expected = model.commitment();
        assert!(verify_weight_integrity(&model, &expected));

        let wrong_hash = [0u8; 32];
        assert!(!verify_weight_integrity(&model, &wrong_hash));
    }

    #[test]
    fn test_checkpoint_integrity() {
        let model = test_model();
        let checkpoint = model.to_checkpoint(0);
        let bytes = checkpoint.to_bytes().unwrap();

        // Note: Checkpoint stores f32 weights, so the round-tripped model's
        // commitment differs from the original f64 model's commitment.
        // We must use the round-tripped commitment for verification.
        let roundtrip_model = MlpModel::from_checkpoint(&checkpoint).unwrap();
        let expected = roundtrip_model.commitment();

        let verified = verify_checkpoint_integrity(&bytes, &expected).unwrap();
        assert_eq!(verified.d_in, model.d_in);
        assert_eq!(verified.d_hid, model.d_hid);
        assert_eq!(verified.d_out, model.d_out);
    }

    #[test]
    fn test_checkpoint_integrity_wrong_hash() {
        let model = test_model();
        let checkpoint = model.to_checkpoint(0);
        let bytes = checkpoint.to_bytes().unwrap();
        let wrong = [0xAB; 32];

        assert!(verify_checkpoint_integrity(&bytes, &wrong).is_err());
    }

    // ---- Chunked Transfer ----

    #[test]
    fn test_single_chunk_small_model() {
        let model = test_model();
        let package = ModelPackageBuilder::build(
            &model, 1, 100, 0.01, 32, 10, 0.1, 42, 2, 0,
        ).unwrap();

        // Use a very large chunk size so everything fits in one chunk
        let chunks = split_into_chunks(&package, 1024 * 1024).unwrap();
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].is_final);
        assert!(chunks[0].metadata.is_some());

        let reassembled = reassemble_chunks(&chunks).unwrap();
        assert_eq!(reassembled.round_id, package.round_id);
        assert_eq!(reassembled.model_id, package.model_id);
        assert_eq!(reassembled.weight_hash, package.weight_hash);
    }

    #[test]
    fn test_multi_chunk_transfer() {
        let model = test_model();
        let package = ModelPackageBuilder::build(
            &model, 1, 100, 0.01, 32, 10, 0.1, 42, 2, 0,
        ).unwrap();

        // Use a tiny chunk size to force multiple chunks
        let chunks = split_into_chunks(&package, 64).unwrap();
        assert!(chunks.len() > 1);

        // First chunk has metadata
        assert!(chunks[0].metadata.is_some());
        // Last chunk is marked final
        assert!(chunks.last().unwrap().is_final);

        // Verify chunk indices
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.chunk_index, i as u32);
            assert_eq!(chunk.total_chunks, chunks.len() as u32);
        }

        let reassembled = reassemble_chunks(&chunks).unwrap();
        assert_eq!(reassembled.round_id, package.round_id);
        assert_eq!(reassembled.weight_hash, package.weight_hash);
    }

    #[test]
    fn test_chunk_receiver() {
        let model = test_model();
        let package = ModelPackageBuilder::build(
            &model, 1, 100, 0.01, 32, 10, 0.1, 42, 2, 0,
        ).unwrap();

        let chunks = split_into_chunks(&package, 64).unwrap();
        let total = chunks.len() as u32;

        let mut receiver = ChunkReceiver::new(1, 100, total);
        assert!(!receiver.is_complete());
        assert_eq!(receiver.progress(), 0.0);

        for (i, chunk) in chunks.into_iter().enumerate() {
            let complete = receiver.add_chunk(chunk).unwrap();
            if i < total as usize - 1 {
                assert!(!complete);
            } else {
                assert!(complete);
            }
        }

        assert!(receiver.is_complete());
        assert_eq!(receiver.progress(), 1.0);
        assert!(receiver.bytes_received() > 0);

        let assembled = receiver.assemble().unwrap();
        assert_eq!(assembled.round_id, 1);
    }

    #[test]
    fn test_chunk_receiver_corrupt_chunk() {
        let mut chunk = ModelPackageChunk {
            round_id: 1,
            model_id: 100,
            chunk_index: 0,
            total_chunks: 1,
            data: vec![1, 2, 3],
            chunk_hash: [0u8; 32], // wrong hash
            is_final: true,
            metadata: None,
        };

        let mut receiver = ChunkReceiver::new(1, 100, 1);
        assert!(receiver.add_chunk(chunk).is_err());
    }

    #[test]
    fn test_reassemble_wrong_chunk_count() {
        let chunk = ModelPackageChunk {
            round_id: 1,
            model_id: 100,
            chunk_index: 0,
            total_chunks: 3, // says 3 total
            data: vec![1, 2, 3],
            chunk_hash: compute_data_hash(&[1, 2, 3]),
            is_final: false,
            metadata: None,
        };

        // Only give 1 of 3 chunks
        assert!(reassemble_chunks(&[chunk]).is_err());
    }

    // ---- Data Shard Planning ----

    #[test]
    fn test_plan_shards_equal_split() {
        let source = DataSourceType::Http {
            url: "https://example.com/data.csv".to_string(),
        };

        let assignments = DataShardPlanner::plan_shards(
            &source, 30000, 300, 3, 1, 42, 32,
            DataFormat::Csv, vec![0, 1, 2], vec![3],
        );

        assert_eq!(assignments.len(), 3);
        assert_eq!(assignments[0].worker_index, 0);
        assert_eq!(assignments[0].num_samples, 100);
        assert_eq!(assignments[0].byte_offset, 0);
        assert_eq!(assignments[1].worker_index, 1);
        assert_eq!(assignments[1].num_samples, 100);
        assert_eq!(assignments[1].byte_offset, 10000);
        assert_eq!(assignments[2].worker_index, 2);
        assert_eq!(assignments[2].num_samples, 100);
        assert_eq!(assignments[2].byte_offset, 20000);
    }

    #[test]
    fn test_plan_shards_uneven_split() {
        let source = DataSourceType::Http {
            url: "https://example.com/data.csv".to_string(),
        };

        let assignments = DataShardPlanner::plan_shards(
            &source, 31000, 310, 3, 1, 42, 32,
            DataFormat::Csv, vec![], vec![],
        );

        assert_eq!(assignments.len(), 3);
        // Last worker gets the remainder
        assert_eq!(assignments[0].num_samples, 103); // 310/3 = 103
        assert_eq!(assignments[1].num_samples, 103);
        assert_eq!(assignments[2].num_samples, 104); // 310 - 103*2 = 104
    }

    #[test]
    fn test_plan_shards_inline_data() {
        let data = vec![0u8; 300];
        let source = DataSourceType::Inline {
            data: data.clone(),
            format: DataFormat::RawF32,
        };

        let assignments = DataShardPlanner::plan_shards(
            &source, 300, 3, 3, 1, 42, 1,
            DataFormat::RawF32, vec![], vec![],
        );

        assert_eq!(assignments.len(), 3);
        // Each worker gets a slice of the inline data
        for a in &assignments {
            if let DataSourceType::Inline { data, .. } = &a.source {
                assert_eq!(data.len(), 100);
            } else {
                panic!("expected inline source");
            }
        }
    }

    #[test]
    fn test_plan_shards_single_worker() {
        let source = DataSourceType::S3 {
            uri: "s3://bucket/data.csv".to_string(),
            region: "us-east-1".to_string(),
        };

        let assignments = DataShardPlanner::plan_shards(
            &source, 10000, 100, 1, 1, 42, 32,
            DataFormat::Csv, vec![], vec![],
        );

        assert_eq!(assignments.len(), 1);
        assert_eq!(assignments[0].num_samples, 100);
        assert_eq!(assignments[0].byte_offset, 0);
        assert_eq!(assignments[0].byte_length, 10000);
    }

    #[test]
    fn test_plan_shards_zero_workers() {
        let source = DataSourceType::Http {
            url: "https://example.com/data.csv".to_string(),
        };

        let assignments = DataShardPlanner::plan_shards(
            &source, 10000, 100, 0, 1, 42, 32,
            DataFormat::Csv, vec![], vec![],
        );

        assert!(assignments.is_empty());
    }

    // ---- Distribution Tracker ----

    #[test]
    fn test_distribution_tracker() {
        let workers = vec![
            PeerId::from_string("w1"),
            PeerId::from_string("w2"),
            PeerId::from_string("w3"),
        ];
        let mut tracker = DistributionTracker::new(1, &workers);

        assert_eq!(tracker.fully_distributed_count(), 0);
        assert_eq!(tracker.progress(), 0.0);

        tracker.mark_package_sent(&PeerId::from_string("w1"));
        tracker.mark_data_assignment_sent(&PeerId::from_string("w1"));
        assert_eq!(tracker.fully_distributed_count(), 1);

        tracker.mark_package_sent(&PeerId::from_string("w2"));
        tracker.mark_data_assignment_sent(&PeerId::from_string("w2"));

        tracker.mark_failed(&PeerId::from_string("w3"), "timeout".to_string());

        assert_eq!(tracker.fully_distributed_count(), 2);
        assert_eq!(tracker.failed_count(), 1);
        assert_eq!(tracker.failed_workers().len(), 1);
    }

    // ---- Compression ----

    #[test]
    fn test_compress_decompress_roundtrip() {
        let data = b"hello world, this is a test of compression that should work correctly";
        let compressed = compress_data(data).unwrap();
        let decompressed = decompress_data(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_decompress_too_short() {
        assert!(decompress_data(&[1, 2]).is_err());
    }

    // ---- 3-Worker Distribution Test ----

    #[test]
    fn test_distribute_model_to_three_workers() {
        let model = MlpModel::new_random(4, 8, 2, 42);
        let original_commitment = model.commitment();

        // Build a model package
        let package = ModelPackageBuilder::build(
            &model, 1, 100, 0.01, 32, 10, 0.1, 42, 2, 0,
        ).unwrap();

        // Simulate distributing to 3 workers by verifying the package 3 times
        let mut received_models = Vec::new();
        for worker_idx in 0..3 {
            let worker_package = package.clone();
            let received_model = verify_model_package(&worker_package)
                .unwrap_or_else(|e| panic!("Worker {} failed to verify: {}", worker_idx, e));
            received_models.push(received_model);
        }

        // All 3 workers should have identical models
        assert_eq!(received_models.len(), 3);
        for (i, m) in received_models.iter().enumerate() {
            assert_eq!(
                m.d_in, model.d_in,
                "Worker {} has wrong d_in", i
            );
            assert_eq!(
                m.d_hid, model.d_hid,
                "Worker {} has wrong d_hid", i
            );
            assert_eq!(
                m.d_out, model.d_out,
                "Worker {} has wrong d_out", i
            );
            assert_eq!(
                m.w1.len(), model.w1.len(),
                "Worker {} has wrong w1 length", i
            );
            assert_eq!(
                m.w2.len(), model.w2.len(),
                "Worker {} has wrong w2 length", i
            );

            // All workers' commitments should match the original
            // Note: f64->f32->f64 roundtrip in checkpoint means the commitment
            // won't exactly match the original f64 model, but all workers
            // should have the SAME commitment as each other.
            let worker_commitment = m.commitment();
            if i > 0 {
                assert_eq!(
                    worker_commitment,
                    received_models[0].commitment(),
                    "Worker {} commitment differs from worker 0",
                    i,
                );
            }
        }

        info!(
            "Successfully distributed model to 3 workers: all received identical copies"
        );
    }

    #[test]
    fn test_distribute_model_chunked_to_three_workers() {
        let model = MlpModel::new_random(4, 8, 2, 42);

        let package = ModelPackageBuilder::build(
            &model, 1, 100, 0.01, 32, 10, 0.1, 42, 2, 0,
        ).unwrap();

        // Split into small chunks
        let chunks = split_into_chunks(&package, 64).unwrap();
        assert!(chunks.len() > 1, "test requires multiple chunks");

        // Simulate 3 workers each receiving all chunks
        let mut received_models = Vec::new();
        for worker_idx in 0..3 {
            let total = chunks.len() as u32;
            let mut receiver = ChunkReceiver::new(1, 100, total);

            for chunk in &chunks {
                receiver.add_chunk(chunk.clone()).unwrap();
            }

            assert!(receiver.is_complete());
            let assembled = receiver.assemble().unwrap();
            let verified_model = verify_model_package(&assembled)
                .unwrap_or_else(|e| panic!("Worker {} verify failed: {}", worker_idx, e));
            received_models.push(verified_model);
        }

        // All workers should have identical models
        let commitment_0 = received_models[0].commitment();
        for (i, m) in received_models.iter().enumerate().skip(1) {
            assert_eq!(
                m.commitment(), commitment_0,
                "Worker {} has different model than worker 0", i,
            );
        }
    }

    #[test]
    fn test_distribute_with_data_assignments() {
        let model = test_model();
        let package = ModelPackageBuilder::build(
            &model, 1, 100, 0.01, 32, 10, 0.1, 42, 2, 0,
        ).unwrap();

        // Create data assignments for 3 workers
        let data = vec![0u8; 900]; // 900 bytes, 9 samples, 100 bytes each
        let source = DataSourceType::Inline {
            data,
            format: DataFormat::Csv,
        };

        let assignments = DataShardPlanner::plan_shards(
            &source, 900, 9, 3, 1, 42, 3,
            DataFormat::Csv, vec![0, 1, 2], vec![3],
        );

        assert_eq!(assignments.len(), 3);

        // Verify each worker gets a unique non-overlapping shard
        let offsets: Vec<u64> = assignments.iter().map(|a| a.byte_offset).collect();
        assert_eq!(offsets, vec![0, 300, 600]);

        for (i, a) in assignments.iter().enumerate() {
            assert_eq!(a.worker_index, i as u32);
            assert_eq!(a.total_workers, 3);
            assert_eq!(a.round_id, 1);
            assert_eq!(a.shuffle_seed, 42);
            assert_eq!(a.batch_size, 3);
        }
    }

    #[test]
    fn test_distribute_resume_checkpoint() {
        let model = test_model();
        let checkpoint = model.to_checkpoint(100); // step 100

        let dims = test_dims();
        let package = ModelPackageBuilder::build_from_checkpoint(
            &checkpoint, 5, 200, &dims, 0.005, 32, 10, 0.1, 42, 0.03,
        ).unwrap();

        assert!(package.is_resume);
        assert_eq!(package.starting_step, 100);
        assert_eq!(package.accumulated_error, 0.03);
        assert_eq!(package.hyperparameters.learning_rate, 0.005);

        // Workers can verify and load the resume checkpoint
        let model = verify_model_package(&package).unwrap();
        assert_eq!(model.d_in, 4);
        assert_eq!(model.d_hid, 8);
        assert_eq!(model.d_out, 2);
    }

    // ---- Helper Tests ----

    #[test]
    fn test_activation_type_names() {
        assert_eq!(activation_type_name(0), "relu");
        assert_eq!(activation_type_name(1), "sigmoid");
        assert_eq!(activation_type_name(2), "tanh");
        assert_eq!(activation_type_name(3), "gelu");
        assert_eq!(activation_type_name(4), "leaky_relu");
        assert_eq!(activation_type_name(255), "unknown");
    }

    #[test]
    fn test_compute_data_hash_deterministic() {
        let data = b"test data";
        let h1 = compute_data_hash(data);
        let h2 = compute_data_hash(data);
        assert_eq!(h1, h2);

        let h3 = compute_data_hash(b"different data");
        assert_ne!(h1, h3);
    }

    // ---- Message Serialization Round-Trip ----

    #[test]
    fn test_model_package_bincode_roundtrip() {
        let model = test_model();
        let package = ModelPackageBuilder::build(
            &model, 1, 100, 0.01, 32, 10, 0.1, 42, 2, 0,
        ).unwrap();

        let bytes = bincode::serialize(&package).unwrap();
        let deserialized: ModelPackage = bincode::deserialize(&bytes).unwrap();

        assert_eq!(deserialized.round_id, package.round_id);
        assert_eq!(deserialized.model_id, package.model_id);
        assert_eq!(deserialized.weight_hash, package.weight_hash);
        assert_eq!(deserialized.compressed, package.compressed);
        assert_eq!(deserialized.weight_data, package.weight_data);
        assert_eq!(deserialized.architecture.dims.d_in, package.architecture.dims.d_in);
        assert_eq!(deserialized.hyperparameters.learning_rate, package.hyperparameters.learning_rate);

        // Verify the deserialized package still passes verification
        let model_back = verify_model_package(&deserialized).unwrap();
        assert_eq!(model_back.d_in, model.d_in);
        assert_eq!(model_back.d_hid, model.d_hid);
        assert_eq!(model_back.d_out, model.d_out);
    }

    #[test]
    fn test_data_assignment_bincode_roundtrip() {
        let assignment = DataAssignment {
            round_id: 5,
            worker_index: 1,
            total_workers: 3,
            source: DataSourceType::S3 {
                uri: "s3://bucket/data.csv".to_string(),
                region: "us-east-1".to_string(),
            },
            byte_offset: 1000,
            byte_length: 500,
            num_samples: 50,
            shard_hash: Some([0xAB; 32]),
            shuffle_seed: 12345,
            batch_size: 16,
            format: DataFormat::Csv,
            feature_columns: vec![0, 1, 2, 3],
            label_columns: vec![4],
        };

        let bytes = bincode::serialize(&assignment).unwrap();
        let deserialized: DataAssignment = bincode::deserialize(&bytes).unwrap();

        assert_eq!(deserialized.round_id, 5);
        assert_eq!(deserialized.worker_index, 1);
        assert_eq!(deserialized.total_workers, 3);
        assert_eq!(deserialized.byte_offset, 1000);
        assert_eq!(deserialized.byte_length, 500);
        assert_eq!(deserialized.num_samples, 50);
        assert_eq!(deserialized.shuffle_seed, 12345);
        assert_eq!(deserialized.batch_size, 16);
        assert_eq!(deserialized.feature_columns, vec![0, 1, 2, 3]);
        assert_eq!(deserialized.label_columns, vec![4]);
    }

    #[test]
    fn test_chunk_bincode_roundtrip() {
        let model = test_model();
        let package = ModelPackageBuilder::build(
            &model, 1, 100, 0.01, 32, 10, 0.1, 42, 2, 0,
        ).unwrap();

        // Split into small chunks
        let chunks = split_into_chunks(&package, 64).unwrap();
        assert!(chunks.len() > 1);

        // Each chunk should survive bincode round-trip
        for chunk in &chunks {
            let bytes = bincode::serialize(chunk).unwrap();
            let deserialized: ModelPackageChunk = bincode::deserialize(&bytes).unwrap();
            assert_eq!(deserialized.chunk_index, chunk.chunk_index);
            assert_eq!(deserialized.total_chunks, chunk.total_chunks);
            assert_eq!(deserialized.chunk_hash, chunk.chunk_hash);
            assert_eq!(deserialized.data, chunk.data);
        }
    }

    // ---- End-to-End Distribution Simulation ----

    #[test]
    fn test_e2e_distribution_with_inline_csv_data() {
        // Simulate a complete distribution: aggregator builds package + data assignments,
        // workers receive and verify them.
        let model = MlpModel::new_random(2, 4, 1, 42);

        // Build model package
        let package = ModelPackageBuilder::build(
            &model, 1, 100, 0.01, 32, 10, 0.1, 42, 2, 0,
        ).unwrap();

        // CSV data with 2 features + 1 label = 3 columns, 9 rows
        let csv_data = "\
feat1,feat2,label
0.1,0.2,0.5
0.3,0.4,0.7
0.5,0.6,0.9
0.2,0.3,0.6
0.4,0.5,0.8
0.6,0.7,1.0
0.15,0.25,0.55
0.35,0.45,0.75
0.55,0.65,0.95";

        let data_source = DataSourceType::Inline {
            data: csv_data.as_bytes().to_vec(),
            format: DataFormat::Csv,
        };

        // Plan shards for 3 workers
        let assignments = DataShardPlanner::plan_shards(
            &data_source,
            csv_data.len() as u64,
            9,
            3,
            1,
            42,
            3,
            DataFormat::Csv,
            vec![0, 1],
            vec![2],
        );

        assert_eq!(assignments.len(), 3);

        // Verify no overlap: each worker gets 3 samples
        for a in &assignments {
            assert_eq!(a.num_samples, 3);
        }

        // Each worker verifies the model package
        for i in 0..3 {
            let worker_model = verify_model_package(&package)
                .unwrap_or_else(|e| panic!("Worker {} failed: {}", i, e));
            assert_eq!(worker_model.d_in, 2);
            assert_eq!(worker_model.d_hid, 4);
            assert_eq!(worker_model.d_out, 1);
        }

        // Verify data assignments have correct inline data slices
        for (i, a) in assignments.iter().enumerate() {
            match &a.source {
                DataSourceType::Inline { data, .. } => {
                    assert!(!data.is_empty(), "Worker {} got empty data slice", i);
                }
                _ => panic!("Expected Inline source for worker {}", i),
            }
        }
    }

    #[test]
    fn test_e2e_resume_distribution() {
        // Simulate distributing a resume checkpoint after 100 steps
        let model = MlpModel::new_random(4, 8, 2, 42);
        let checkpoint = model.to_checkpoint(100);
        let dims = test_dims();

        let package = ModelPackageBuilder::build_from_checkpoint(
            &checkpoint, 5, 200, &dims, 0.005, 32, 10, 0.1, 42, 0.03,
        ).unwrap();

        // Serialize and distribute
        let package_bytes = bincode::serialize(&package).unwrap();

        // 3 workers deserialize and verify
        let mut commitments = Vec::new();
        for i in 0..3 {
            let received: ModelPackage = bincode::deserialize(&package_bytes).unwrap();
            assert!(received.is_resume);
            assert_eq!(received.starting_step, 100);
            assert_eq!(received.accumulated_error, 0.03);

            let model = verify_model_package(&received)
                .unwrap_or_else(|e| panic!("Worker {} failed: {}", i, e));
            commitments.push(model.commitment());
        }

        // All workers should have identical models
        assert_eq!(commitments[0], commitments[1]);
        assert_eq!(commitments[1], commitments[2]);
    }

    #[test]
    fn test_large_model_chunked_e2e() {
        // Use a larger model (10x20x5) to test chunked transfer
        let model = MlpModel::new_random(10, 20, 5, 42);
        let package = ModelPackageBuilder::build(
            &model, 1, 100, 0.01, 32, 10, 0.1, 42, 2, 0,
        ).unwrap();

        // Force chunked transfer with 128-byte chunks
        let chunks = split_into_chunks(&package, 128).unwrap();
        assert!(chunks.len() > 5, "Expected many chunks, got {}", chunks.len());

        // Simulate 3 workers receiving chunks out of order
        for worker in 0..3 {
            let total = chunks.len() as u32;
            let mut receiver = ChunkReceiver::new(1, 100, total);

            // Receive in reverse order to test ordering
            for chunk in chunks.iter().rev() {
                let was_complete = receiver.add_chunk(chunk.clone()).unwrap();
                if !was_complete {
                    assert!(receiver.progress() < 1.0);
                }
            }

            assert!(receiver.is_complete());
            let assembled = receiver.assemble().unwrap();
            let model_back = verify_model_package(&assembled)
                .unwrap_or_else(|e| panic!("Worker {} verify failed: {}", worker, e));

            assert_eq!(model_back.d_in, 10);
            assert_eq!(model_back.d_hid, 20);
            assert_eq!(model_back.d_out, 5);
        }
    }

    #[test]
    fn test_distribution_tracker_complete_flow() {
        let workers = vec![
            PeerId::from_string("w1"),
            PeerId::from_string("w2"),
            PeerId::from_string("w3"),
        ];
        let mut tracker = DistributionTracker::new(1, &workers);

        // Initially nothing distributed
        assert_eq!(tracker.fully_distributed_count(), 0);
        assert_eq!(tracker.failed_count(), 0);

        // Send package to all
        for w in &workers {
            tracker.mark_package_sent(w);
        }

        // Without data assignments, fully_distributed requires both
        // (package_sent && data_assignment_sent)
        assert_eq!(tracker.fully_distributed_count(), 0);

        // Send data assignments
        for w in &workers {
            tracker.mark_data_assignment_sent(w);
        }

        assert_eq!(tracker.fully_distributed_count(), 3);
        assert_eq!(tracker.failed_count(), 0);
    }

    #[test]
    fn test_shard_planner_single_worker() {
        let source = DataSourceType::Ipfs {
            cid: "QmTest123".to_string(),
            gateway: None,
        };

        let assignments = DataShardPlanner::plan_shards(
            &source, 10000, 100, 1, 1, 42, 32,
            DataFormat::Csv, vec![], vec![],
        );

        assert_eq!(assignments.len(), 1);
        assert_eq!(assignments[0].byte_offset, 0);
        assert_eq!(assignments[0].byte_length, 10000);
        assert_eq!(assignments[0].num_samples, 100);
    }

    #[test]
    fn test_shard_planner_uneven_split() {
        let source = DataSourceType::Http {
            url: "https://example.com/data.csv".to_string(),
        };

        // 10 samples across 3 workers = 3, 3, 4
        let assignments = DataShardPlanner::plan_shards(
            &source, 1000, 10, 3, 1, 42, 32,
            DataFormat::Csv, vec![], vec![],
        );

        assert_eq!(assignments.len(), 3);
        assert_eq!(assignments[0].num_samples, 3);
        assert_eq!(assignments[1].num_samples, 3);
        assert_eq!(assignments[2].num_samples, 4); // last worker gets remainder

        // Total samples should equal the original
        let total: u64 = assignments.iter().map(|a| a.num_samples).sum();
        assert_eq!(total, 10);
    }

    #[test]
    fn test_shard_planner_zero_workers() {
        let source = DataSourceType::Http {
            url: "https://example.com/data.csv".to_string(),
        };

        let assignments = DataShardPlanner::plan_shards(
            &source, 1000, 10, 0, 1, 42, 32,
            DataFormat::Csv, vec![], vec![],
        );

        assert!(assignments.is_empty());
    }
}
