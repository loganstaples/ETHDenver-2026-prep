//! Model checkpoint serialization.
//!
//! Provides save/load of model weights alongside error tracking state,
//! enabling training resumption from any point. Checkpoints are versioned
//! and integrity-protected with SHA-256.
//!
//! # Binary format
//!
//! ```text
//! [magic: 4 bytes "HXCK"]
//! [version: u32 LE]
//! [flags: u32 LE]  // bit 0: has_error_state
//! [model_id: 32 bytes]
//! [step_number: u64 LE]
//! [timestamp: u64 LE]
//! [weight_hash: 32 bytes]  // SHA-256 of raw weights
//! [num_layers: u32 LE]
//! [layer entries...]
//! [error_state: optional, present if flags bit 0]
//! [json_metadata_len: u32 LE]
//! [json_metadata: UTF-8 bytes]
//! [checksum: SHA-256 of all preceding bytes]
//! ```

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

use crate::error::{HelixError, HelixResult, SerializationError};

/// Magic bytes for checkpoint files.
const CHECKPOINT_MAGIC: &[u8; 4] = b"HXCK";
/// Current checkpoint format version.
const CHECKPOINT_VERSION: u32 = 1;
/// Flag: checkpoint includes error tracking state.
const FLAG_HAS_ERROR_STATE: u32 = 1;

/// A complete model checkpoint that can be saved to disk and restored.
///
/// Includes model weights, training metadata, and optionally the full
/// error tracking state for seamless training resumption.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCheckpoint {
    /// Model identifier (32 bytes).
    pub model_id: [u8; 32],
    /// Training step number at checkpoint time.
    pub step_number: u64,
    /// Timestamp when checkpoint was created (unix seconds).
    pub timestamp: u64,
    /// SHA-256 hash of the concatenated raw weight bytes.
    pub weight_hash: [u8; 32],
    /// Layer weights: (layer_name, tensor_name) -> flat f32 values.
    pub layers: Vec<CheckpointLayer>,
    /// Error tracking state (if training with error budgets).
    pub error_state: Option<CheckpointErrorState>,
    /// Arbitrary metadata (JSON-serializable).
    pub metadata: HashMap<String, String>,
}

/// A single layer in a checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointLayer {
    /// Layer name (e.g., "fc1").
    pub name: String,
    /// Tensors in this layer.
    pub tensors: Vec<CheckpointTensor>,
}

/// A single tensor within a checkpoint layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointTensor {
    /// Tensor name (e.g., "weight", "bias").
    pub name: String,
    /// Tensor shape.
    pub shape: Vec<usize>,
    /// Raw f32 values.
    pub data: Vec<f32>,
}

impl CheckpointTensor {
    /// Returns the number of elements.
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Computes SHA-256 hash of this tensor's data.
    pub fn hash(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        for v in &self.data {
            hasher.update(v.to_le_bytes());
        }
        let result = hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        hash
    }
}

/// Error tracking state saved alongside model weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointErrorState {
    /// Total accumulated error at checkpoint time.
    pub accumulated_error: f64,
    /// Error budget limit.
    pub budget_limit: f64,
    /// Budget utilization (0.0 to 1.0+).
    pub utilization: f64,
    /// Per-step error history (if tracking enabled).
    pub error_history: Vec<f64>,
    /// Error checksum (compact u64, matches on-chain format).
    pub error_checksum: u64,
}

impl ModelCheckpoint {
    /// Creates a new checkpoint from model weights.
    pub fn new(
        model_id: [u8; 32],
        step_number: u64,
        timestamp: u64,
        layers: Vec<CheckpointLayer>,
    ) -> Self {
        let weight_hash = Self::compute_weight_hash(&layers);
        Self {
            model_id,
            step_number,
            timestamp,
            weight_hash,
            layers,
            error_state: None,
            metadata: HashMap::new(),
        }
    }

    /// Sets the error tracking state.
    pub fn with_error_state(mut self, error_state: CheckpointErrorState) -> Self {
        self.error_state = Some(error_state);
        self
    }

    /// Adds a metadata entry.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Computes the SHA-256 hash over all weight data.
    pub fn compute_weight_hash(layers: &[CheckpointLayer]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        for layer in layers {
            hasher.update(layer.name.as_bytes());
            for tensor in &layer.tensors {
                hasher.update(tensor.name.as_bytes());
                for &dim in &tensor.shape {
                    hasher.update((dim as u64).to_le_bytes());
                }
                for v in &tensor.data {
                    hasher.update(v.to_le_bytes());
                }
            }
        }
        let result = hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        hash
    }

    /// Verifies the weight hash matches the stored weights.
    pub fn verify_integrity(&self) -> bool {
        self.weight_hash == Self::compute_weight_hash(&self.layers)
    }

    /// Returns the total number of parameters.
    pub fn total_params(&self) -> usize {
        self.layers.iter()
            .flat_map(|l| l.tensors.iter())
            .map(|t| t.num_elements())
            .sum()
    }

    /// Serializes the checkpoint to bytes.
    pub fn to_bytes(&self) -> HelixResult<Vec<u8>> {
        let mut buf = Vec::new();

        // Header
        buf.extend_from_slice(CHECKPOINT_MAGIC);
        buf.extend_from_slice(&CHECKPOINT_VERSION.to_le_bytes());

        let mut flags: u32 = 0;
        if self.error_state.is_some() {
            flags |= FLAG_HAS_ERROR_STATE;
        }
        buf.extend_from_slice(&flags.to_le_bytes());
        buf.extend_from_slice(&self.model_id);
        buf.extend_from_slice(&self.step_number.to_le_bytes());
        buf.extend_from_slice(&self.timestamp.to_le_bytes());
        buf.extend_from_slice(&self.weight_hash);

        // Layers
        buf.extend_from_slice(&(self.layers.len() as u32).to_le_bytes());
        for layer in &self.layers {
            write_string(&mut buf, &layer.name);
            buf.extend_from_slice(&(layer.tensors.len() as u32).to_le_bytes());
            for tensor in &layer.tensors {
                write_string(&mut buf, &tensor.name);
                buf.extend_from_slice(&(tensor.shape.len() as u32).to_le_bytes());
                for &dim in &tensor.shape {
                    buf.extend_from_slice(&(dim as u64).to_le_bytes());
                }
                buf.extend_from_slice(&(tensor.data.len() as u32).to_le_bytes());
                for &v in &tensor.data {
                    buf.extend_from_slice(&v.to_le_bytes());
                }
            }
        }

        // Error state
        if let Some(es) = &self.error_state {
            buf.extend_from_slice(&es.accumulated_error.to_le_bytes());
            buf.extend_from_slice(&es.budget_limit.to_le_bytes());
            buf.extend_from_slice(&es.utilization.to_le_bytes());
            buf.extend_from_slice(&(es.error_history.len() as u32).to_le_bytes());
            for &e in &es.error_history {
                buf.extend_from_slice(&e.to_le_bytes());
            }
            buf.extend_from_slice(&es.error_checksum.to_le_bytes());
        }

        // JSON metadata
        let meta_json = serde_json::to_vec(&self.metadata)
            .map_err(|e| HelixError::Serialization(SerializationError::JsonError(e.to_string())))?;
        buf.extend_from_slice(&(meta_json.len() as u32).to_le_bytes());
        buf.extend_from_slice(&meta_json);

        // Checksum
        let mut hasher = Sha256::new();
        hasher.update(&buf);
        let checksum: [u8; 32] = hasher.finalize().into();
        buf.extend_from_slice(&checksum);

        Ok(buf)
    }

    /// Deserializes a checkpoint from bytes.
    pub fn from_bytes(data: &[u8]) -> HelixResult<Self> {
        if data.len() < 36 {
            return Err(HelixError::Serialization(SerializationError::BinaryError(
                "checkpoint data too short".to_string()
            )));
        }

        // Verify checksum first
        let payload = &data[..data.len() - 32];
        let stored_checksum = &data[data.len() - 32..];
        let mut hasher = Sha256::new();
        hasher.update(payload);
        let computed: [u8; 32] = hasher.finalize().into();
        if computed != stored_checksum[..] {
            return Err(HelixError::Serialization(SerializationError::Corruption(
                "checkpoint checksum mismatch".to_string()
            )));
        }

        let mut offset = 0;

        // Verify magic
        if &data[offset..offset + 4] != CHECKPOINT_MAGIC {
            return Err(HelixError::Serialization(SerializationError::BinaryError(
                "invalid checkpoint magic".to_string()
            )));
        }
        offset += 4;

        // Version
        let version = read_u32(data, &mut offset)?;
        if version != CHECKPOINT_VERSION {
            return Err(HelixError::Serialization(SerializationError::VersionMismatch {
                expected: CHECKPOINT_VERSION,
                actual: version,
            }));
        }

        // Flags
        let flags = read_u32(data, &mut offset)?;
        let has_error_state = (flags & FLAG_HAS_ERROR_STATE) != 0;

        // Model ID
        let mut model_id = [0u8; 32];
        model_id.copy_from_slice(&data[offset..offset + 32]);
        offset += 32;

        let step_number = read_u64(data, &mut offset)?;
        let timestamp = read_u64(data, &mut offset)?;

        let mut weight_hash = [0u8; 32];
        weight_hash.copy_from_slice(&data[offset..offset + 32]);
        offset += 32;

        // Layers
        let num_layers = read_u32(data, &mut offset)? as usize;
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            let layer_name = read_string(data, &mut offset)?;
            let num_tensors = read_u32(data, &mut offset)? as usize;
            let mut tensors = Vec::with_capacity(num_tensors);
            for _ in 0..num_tensors {
                let tensor_name = read_string(data, &mut offset)?;
                let num_dims = read_u32(data, &mut offset)? as usize;
                let mut shape = Vec::with_capacity(num_dims);
                for _ in 0..num_dims {
                    shape.push(read_u64(data, &mut offset)? as usize);
                }
                let num_values = read_u32(data, &mut offset)? as usize;
                let mut tensor_data = Vec::with_capacity(num_values);
                for _ in 0..num_values {
                    tensor_data.push(read_f32(data, &mut offset)?);
                }
                tensors.push(CheckpointTensor {
                    name: tensor_name,
                    shape,
                    data: tensor_data,
                });
            }
            layers.push(CheckpointLayer {
                name: layer_name,
                tensors,
            });
        }

        // Error state
        let error_state = if has_error_state {
            let accumulated_error = read_f64(data, &mut offset)?;
            let budget_limit = read_f64(data, &mut offset)?;
            let utilization = read_f64(data, &mut offset)?;
            let history_len = read_u32(data, &mut offset)? as usize;
            let mut error_history = Vec::with_capacity(history_len);
            for _ in 0..history_len {
                error_history.push(read_f64(data, &mut offset)?);
            }
            let error_checksum = read_u64(data, &mut offset)?;
            Some(CheckpointErrorState {
                accumulated_error,
                budget_limit,
                utilization,
                error_history,
                error_checksum,
            })
        } else {
            None
        };

        // Metadata
        let meta_len = read_u32(data, &mut offset)? as usize;
        let meta_bytes = &data[offset..offset + meta_len];
        let metadata: HashMap<String, String> = serde_json::from_slice(meta_bytes)
            .map_err(|e| HelixError::Serialization(SerializationError::JsonError(e.to_string())))?;

        Ok(Self {
            model_id,
            step_number,
            timestamp,
            weight_hash,
            layers,
            error_state,
            metadata,
        })
    }

    /// Saves the checkpoint to a file.
    pub fn save(&self, path: &str) -> HelixResult<()> {
        let bytes = self.to_bytes()?;
        std::fs::write(path, &bytes).map_err(HelixError::Io)
    }

    /// Loads a checkpoint from a file.
    pub fn load(path: &str) -> HelixResult<Self> {
        let bytes = std::fs::read(path).map_err(HelixError::Io)?;
        Self::from_bytes(&bytes)
    }

    /// Saves the checkpoint to a file with zstd compression.
    ///
    /// Requires the `compression` feature. Typically achieves 2-5x compression
    /// on model weight data.
    #[cfg(feature = "compression")]
    pub fn save_compressed(&self, path: &str) -> HelixResult<()> {
        let bytes = self.to_bytes()?;
        let compressed = zstd::bulk::compress(&bytes, 3)
            .map_err(|e| HelixError::Serialization(SerializationError::BinaryError(
                format!("zstd compression failed: {}", e)
            )))?;
        std::fs::write(path, &compressed).map_err(HelixError::Io)
    }

    /// Loads a checkpoint from a zstd-compressed file.
    ///
    /// Requires the `compression` feature.
    #[cfg(feature = "compression")]
    pub fn load_compressed(path: &str) -> HelixResult<Self> {
        let compressed = std::fs::read(path).map_err(HelixError::Io)?;
        let bytes = zstd::bulk::decompress(&compressed, 256 * 1024 * 1024) // 256MB max
            .map_err(|e| HelixError::Serialization(SerializationError::BinaryError(
                format!("zstd decompression failed: {}", e)
            )))?;
        Self::from_bytes(&bytes)
    }
}

// === Binary read helpers ===

fn read_u32(data: &[u8], offset: &mut usize) -> HelixResult<u32> {
    if *offset + 4 > data.len() {
        return Err(HelixError::Serialization(SerializationError::BinaryError(
            "unexpected end of data reading u32".to_string()
        )));
    }
    let v = u32::from_le_bytes(data[*offset..*offset + 4].try_into()
        .map_err(|_| HelixError::Serialization(SerializationError::BinaryError(
            "invalid u32 bytes in checkpoint".to_string()
        )))?);
    *offset += 4;
    Ok(v)
}

fn read_u64(data: &[u8], offset: &mut usize) -> HelixResult<u64> {
    if *offset + 8 > data.len() {
        return Err(HelixError::Serialization(SerializationError::BinaryError(
            "unexpected end of data reading u64".to_string()
        )));
    }
    let v = u64::from_le_bytes(data[*offset..*offset + 8].try_into()
        .map_err(|_| HelixError::Serialization(SerializationError::BinaryError(
            "invalid u64 bytes in checkpoint".to_string()
        )))?);
    *offset += 8;
    Ok(v)
}

fn read_f32(data: &[u8], offset: &mut usize) -> HelixResult<f32> {
    if *offset + 4 > data.len() {
        return Err(HelixError::Serialization(SerializationError::BinaryError(
            "unexpected end of data reading f32".to_string()
        )));
    }
    let v = f32::from_le_bytes(data[*offset..*offset + 4].try_into()
        .map_err(|_| HelixError::Serialization(SerializationError::BinaryError(
            "invalid f32 bytes in checkpoint".to_string()
        )))?);
    *offset += 4;
    Ok(v)
}

fn read_f64(data: &[u8], offset: &mut usize) -> HelixResult<f64> {
    if *offset + 8 > data.len() {
        return Err(HelixError::Serialization(SerializationError::BinaryError(
            "unexpected end of data reading f64".to_string()
        )));
    }
    let v = f64::from_le_bytes(data[*offset..*offset + 8].try_into()
        .map_err(|_| HelixError::Serialization(SerializationError::BinaryError(
            "invalid f64 bytes in checkpoint".to_string()
        )))?);
    *offset += 8;
    Ok(v)
}

fn read_string(data: &[u8], offset: &mut usize) -> HelixResult<String> {
    let len = read_u32(data, offset)? as usize;
    if *offset + len > data.len() {
        return Err(HelixError::Serialization(SerializationError::BinaryError(
            "unexpected end of data reading string".to_string()
        )));
    }
    let s = std::str::from_utf8(&data[*offset..*offset + len])
        .map_err(|e| HelixError::Serialization(SerializationError::BinaryError(
            format!("invalid UTF-8: {}", e)
        )))?
        .to_string();
    *offset += len;
    Ok(s)
}

fn write_string(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u32).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_checkpoint() -> ModelCheckpoint {
        let layers = vec![
            CheckpointLayer {
                name: "fc1".to_string(),
                tensors: vec![
                    CheckpointTensor {
                        name: "weight".to_string(),
                        shape: vec![3, 2],
                        data: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                    },
                    CheckpointTensor {
                        name: "bias".to_string(),
                        shape: vec![3],
                        data: vec![0.01, 0.02, 0.03],
                    },
                ],
            },
            CheckpointLayer {
                name: "fc2".to_string(),
                tensors: vec![CheckpointTensor {
                    name: "weight".to_string(),
                    shape: vec![1, 3],
                    data: vec![0.7, 0.8, 0.9],
                }],
            },
        ];

        ModelCheckpoint::new([1u8; 32], 42, 1700000000, layers)
            .with_error_state(CheckpointErrorState {
                accumulated_error: 0.003,
                budget_limit: 0.01,
                utilization: 0.3,
                error_history: vec![0.001, 0.001, 0.001],
                error_checksum: 12345,
            })
            .with_metadata("training_config", "lr=0.001,bs=32")
    }

    #[test]
    fn test_checkpoint_creation() {
        let ckpt = make_test_checkpoint();
        assert_eq!(ckpt.step_number, 42);
        assert_eq!(ckpt.total_params(), 12); // 6 + 3 + 3
        assert!(ckpt.error_state.is_some());
    }

    #[test]
    fn test_checkpoint_weight_hash() {
        let ckpt = make_test_checkpoint();
        assert!(ckpt.verify_integrity());

        // Different weights -> different hash
        let mut ckpt2 = ckpt.clone();
        ckpt2.layers[0].tensors[0].data[0] = 99.0;
        assert!(!ckpt2.verify_integrity()); // hash was computed from original weights
    }

    #[test]
    fn test_checkpoint_binary_roundtrip() {
        let original = make_test_checkpoint();
        let bytes = original.to_bytes().unwrap();
        let restored = ModelCheckpoint::from_bytes(&bytes).unwrap();

        assert_eq!(original.model_id, restored.model_id);
        assert_eq!(original.step_number, restored.step_number);
        assert_eq!(original.timestamp, restored.timestamp);
        assert_eq!(original.weight_hash, restored.weight_hash);
        assert_eq!(original.layers.len(), restored.layers.len());

        // Verify layer data
        for (orig_layer, rest_layer) in original.layers.iter().zip(restored.layers.iter()) {
            assert_eq!(orig_layer.name, rest_layer.name);
            assert_eq!(orig_layer.tensors.len(), rest_layer.tensors.len());
            for (orig_t, rest_t) in orig_layer.tensors.iter().zip(rest_layer.tensors.iter()) {
                assert_eq!(orig_t.name, rest_t.name);
                assert_eq!(orig_t.shape, rest_t.shape);
                assert_eq!(orig_t.data, rest_t.data);
            }
        }

        // Verify error state
        let orig_es = original.error_state.unwrap();
        let rest_es = restored.error_state.unwrap();
        assert!((orig_es.accumulated_error - rest_es.accumulated_error).abs() < 1e-15);
        assert!((orig_es.budget_limit - rest_es.budget_limit).abs() < 1e-15);
        assert_eq!(orig_es.error_history, rest_es.error_history);
        assert_eq!(orig_es.error_checksum, rest_es.error_checksum);

        // Verify metadata
        assert_eq!(original.metadata, restored.metadata);
    }

    #[test]
    fn test_checkpoint_without_error_state() {
        let layers = vec![CheckpointLayer {
            name: "fc1".to_string(),
            tensors: vec![CheckpointTensor {
                name: "weight".to_string(),
                shape: vec![2],
                data: vec![1.0, 2.0],
            }],
        }];
        let ckpt = ModelCheckpoint::new([0u8; 32], 0, 0, layers);

        let bytes = ckpt.to_bytes().unwrap();
        let restored = ModelCheckpoint::from_bytes(&bytes).unwrap();
        assert!(restored.error_state.is_none());
    }

    #[test]
    fn test_checkpoint_file_roundtrip() {
        let original = make_test_checkpoint();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.hxck");
        let path_str = path.to_str().unwrap();

        original.save(path_str).unwrap();
        let restored = ModelCheckpoint::load(path_str).unwrap();

        assert_eq!(original.model_id, restored.model_id);
        assert_eq!(original.step_number, restored.step_number);
        assert!(restored.verify_integrity());
    }

    #[test]
    fn test_checkpoint_corruption_detection() {
        let ckpt = make_test_checkpoint();
        let mut bytes = ckpt.to_bytes().unwrap();

        // Corrupt a byte in the middle
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0xFF;

        let result = ModelCheckpoint::from_bytes(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_checkpoint_invalid_magic() {
        let result = ModelCheckpoint::from_bytes(&[0u8; 100]);
        assert!(result.is_err());
    }

    #[test]
    fn test_checkpoint_version_mismatch() {
        let ckpt = make_test_checkpoint();
        let mut bytes = ckpt.to_bytes().unwrap();

        // Overwrite version to 99
        bytes[4..8].copy_from_slice(&99u32.to_le_bytes());

        // Recompute checksum
        let payload_len = bytes.len() - 32;
        let mut hasher = Sha256::new();
        hasher.update(&bytes[..payload_len]);
        let checksum: [u8; 32] = hasher.finalize().into();
        bytes[payload_len..].copy_from_slice(&checksum);

        let result = ModelCheckpoint::from_bytes(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_tensor_hash() {
        let t1 = CheckpointTensor {
            name: "w".to_string(),
            shape: vec![2, 2],
            data: vec![1.0, 2.0, 3.0, 4.0],
        };
        let t2 = CheckpointTensor {
            name: "w".to_string(),
            shape: vec![2, 2],
            data: vec![1.0, 2.0, 3.0, 4.0],
        };
        assert_eq!(t1.hash(), t2.hash());

        let t3 = CheckpointTensor {
            name: "w".to_string(),
            shape: vec![2, 2],
            data: vec![1.0, 2.0, 3.0, 5.0],
        };
        assert_ne!(t1.hash(), t3.hash());
    }

    #[test]
    fn test_empty_checkpoint() {
        let ckpt = ModelCheckpoint::new([0u8; 32], 0, 0, Vec::new());
        assert_eq!(ckpt.total_params(), 0);
        assert!(ckpt.verify_integrity());

        let bytes = ckpt.to_bytes().unwrap();
        let restored = ModelCheckpoint::from_bytes(&bytes).unwrap();
        assert_eq!(restored.total_params(), 0);
    }

    #[cfg(feature = "compression")]
    #[test]
    fn test_checkpoint_compressed_file_roundtrip() {
        let original = make_test_checkpoint();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.hxck.zst");
        let path_str = path.to_str().unwrap();

        original.save_compressed(path_str).unwrap();

        // Compressed file should be smaller than uncompressed
        let uncompressed_bytes = original.to_bytes().unwrap();
        let compressed_size = std::fs::metadata(path_str).unwrap().len();
        // For small data, compression overhead may make it larger, but it should still roundtrip
        assert!(compressed_size > 0);

        let restored = ModelCheckpoint::load_compressed(path_str).unwrap();
        assert_eq!(original.model_id, restored.model_id);
        assert_eq!(original.step_number, restored.step_number);
        assert_eq!(original.weight_hash, restored.weight_hash);
        assert!(restored.verify_integrity());

        // For a larger checkpoint, compression should save space
        let mut big_layers = Vec::new();
        for i in 0..10 {
            big_layers.push(CheckpointLayer {
                name: format!("layer_{}", i),
                tensors: vec![CheckpointTensor {
                    name: "weight".to_string(),
                    shape: vec![100, 100],
                    data: vec![0.1f32; 10000],
                }],
            });
        }
        let big_ckpt = ModelCheckpoint::new([2u8; 32], 100, 1700000000, big_layers);
        let big_uncompressed = big_ckpt.to_bytes().unwrap();
        let big_path = dir.path().join("big_model.hxck.zst");
        let big_path_str = big_path.to_str().unwrap();
        big_ckpt.save_compressed(big_path_str).unwrap();
        let big_compressed_size = std::fs::metadata(big_path_str).unwrap().len();
        assert!(
            big_compressed_size < big_uncompressed.len() as u64,
            "compressed {} should be less than uncompressed {}",
            big_compressed_size,
            big_uncompressed.len()
        );
    }
}
