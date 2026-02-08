//! Model Serialization and Deserialization for Checkpoints.
//!
//! Provides infrastructure for saving and loading model checkpoints, including:
//! - Tensor serialization with compression
//! - Model metadata and versioning
//! - Partial checkpoint loading (for fine-tuning)
//! - Cross-precision conversion
//!
//! # Checkpoint Format
//!
//! Checkpoints use a custom binary format optimized for ML models:
//!
//! ```text
//! Header (32 bytes):
//!   - Magic number (8 bytes): "HELIXCHK"
//!   - Version (4 bytes)
//!   - Flags (4 bytes)
//!   - Metadata length (8 bytes)
//!   - Data length (8 bytes)
//!
//! Metadata (JSON):
//!   - Model name
//!   - Model config
//!   - Training step
//!   - Timestamp
//!   - Parameter count
//!
//! Data (binary):
//!   - Tensor name length (4 bytes)
//!   - Tensor name (UTF-8)
//!   - Shape length (4 bytes)
//!   - Shape dimensions
//!   - Data type (1 byte)
//!   - Data (compressed)
//! ```

use helix_core::types::{BoundedTensor, BoundedValue};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::Path;
use thiserror::Error;

/// Errors during serialization.
#[derive(Error, Debug)]
pub enum SerializationError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid checkpoint format: {0}")]
    InvalidFormat(String),

    #[error("Version mismatch: expected {expected}, got {actual}")]
    VersionMismatch { expected: u32, actual: u32 },

    #[error("Tensor not found: {0}")]
    TensorNotFound(String),

    #[error("Shape mismatch for {name}: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        name: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("Serialization error: {0}")]
    SerdeError(String),

    #[error("Compression error: {0}")]
    CompressionError(String),
}

/// Checkpoint format version.
pub const CHECKPOINT_VERSION: u32 = 1;

/// Magic number for checkpoint files.
pub const CHECKPOINT_MAGIC: &[u8; 8] = b"HELIXCHK";

/// Checkpoint format options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointFormat {
    /// Raw binary format (no compression).
    Raw,
    /// Compressed with simple RLE.
    Compressed,
    /// Quantized to reduce size.
    Quantized,
}

impl Default for CheckpointFormat {
    fn default() -> Self {
        CheckpointFormat::Raw
    }
}

/// Metadata for a model checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// Model name.
    pub model_name: String,
    /// Model type (GPT, BERT, etc.).
    pub model_type: String,
    /// Total parameter count.
    pub param_count: usize,
    /// Training step when checkpoint was saved.
    pub training_step: usize,
    /// Training loss at checkpoint.
    pub training_loss: Option<f64>,
    /// Validation loss at checkpoint.
    pub validation_loss: Option<f64>,
    /// Timestamp (Unix seconds).
    pub timestamp: u64,
    /// Custom metadata fields.
    pub custom: HashMap<String, String>,
    /// Model configuration as JSON.
    pub config_json: Option<String>,
}

impl CheckpointMetadata {
    /// Creates new metadata.
    pub fn new(model_name: impl Into<String>, model_type: impl Into<String>) -> Self {
        Self {
            model_name: model_name.into(),
            model_type: model_type.into(),
            param_count: 0,
            training_step: 0,
            training_loss: None,
            validation_loss: None,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            custom: HashMap::new(),
            config_json: None,
        }
    }

    /// Sets the parameter count.
    pub fn with_param_count(mut self, count: usize) -> Self {
        self.param_count = count;
        self
    }

    /// Sets the training step.
    pub fn with_training_step(mut self, step: usize) -> Self {
        self.training_step = step;
        self
    }

    /// Sets the training loss.
    pub fn with_training_loss(mut self, loss: f64) -> Self {
        self.training_loss = Some(loss);
        self
    }

    /// Adds custom metadata.
    pub fn with_custom(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.custom.insert(key.into(), value.into());
        self
    }

    /// Sets the config JSON.
    pub fn with_config(mut self, config: impl Into<String>) -> Self {
        self.config_json = Some(config.into());
        self
    }
}

impl Default for CheckpointMetadata {
    fn default() -> Self {
        Self::new("unnamed", "unknown")
    }
}

/// Serialized tensor data.
#[derive(Debug, Clone)]
pub struct TensorData {
    /// Tensor name.
    pub name: String,
    /// Tensor shape.
    pub shape: Vec<usize>,
    /// Tensor values (f64).
    pub values: Vec<f64>,
    /// Error bounds (if tracked).
    pub errors: Option<Vec<f64>>,
    /// Data type.
    pub dtype: TensorDType,
}

/// Data type for serialized tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorDType {
    /// 64-bit float.
    F64,
    /// 32-bit float.
    F32,
    /// 16-bit float.
    F16,
    /// 8-bit integer.
    I8,
    /// 4-bit integer (packed).
    I4,
}

impl TensorDType {
    /// Returns bytes per element.
    pub fn bytes_per_element(&self) -> usize {
        match self {
            TensorDType::F64 => 8,
            TensorDType::F32 => 4,
            TensorDType::F16 => 2,
            TensorDType::I8 => 1,
            TensorDType::I4 => 1, // Packed, but minimum addressable is 1 byte
        }
    }

    /// Converts to byte representation.
    pub fn to_byte(&self) -> u8 {
        match self {
            TensorDType::F64 => 0,
            TensorDType::F32 => 1,
            TensorDType::F16 => 2,
            TensorDType::I8 => 3,
            TensorDType::I4 => 4,
        }
    }

    /// Creates from byte representation.
    pub fn from_byte(byte: u8) -> Option<Self> {
        match byte {
            0 => Some(TensorDType::F64),
            1 => Some(TensorDType::F32),
            2 => Some(TensorDType::F16),
            3 => Some(TensorDType::I8),
            4 => Some(TensorDType::I4),
            _ => None,
        }
    }
}

impl TensorData {
    /// Creates tensor data from a BoundedTensor.
    pub fn from_bounded(name: impl Into<String>, tensor: &BoundedTensor) -> Self {
        let values: Vec<f64> = tensor.data().iter().map(|v| v.value()).collect();
        let errors: Vec<f64> = tensor.data().iter().map(|v| v.absolute_error()).collect();

        let has_errors = errors.iter().any(|&e| e > 0.0);

        Self {
            name: name.into(),
            shape: tensor.shape().to_vec(),
            values,
            errors: if has_errors { Some(errors) } else { None },
            dtype: TensorDType::F64,
        }
    }

    /// Converts to a BoundedTensor.
    pub fn to_bounded(&self) -> BoundedTensor {
        let data: Vec<BoundedValue<f64>> = if let Some(ref errors) = self.errors {
            self.values
                .iter()
                .zip(errors.iter())
                .map(|(&v, &e)| BoundedValue::<f64>::with_absolute_error(v, e))
                .collect()
        } else {
            self.values.iter().map(|&v| BoundedValue::<f64>::exact(v)).collect()
        };

        BoundedTensor::new(data, self.shape.clone())
    }

    /// Returns the number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Returns the size in bytes (uncompressed).
    pub fn size_bytes(&self) -> usize {
        let value_bytes = self.values.len() * 8;
        let error_bytes = self.errors.as_ref().map(|e| e.len() * 8).unwrap_or(0);
        value_bytes + error_bytes
    }
}

/// A complete model checkpoint.
#[derive(Debug, Clone)]
pub struct Checkpoint {
    /// Checkpoint metadata.
    pub metadata: CheckpointMetadata,
    /// Tensor data by name.
    pub tensors: HashMap<String, TensorData>,
    /// Checkpoint format.
    pub format: CheckpointFormat,
}

impl Checkpoint {
    /// Creates a new checkpoint.
    pub fn new(metadata: CheckpointMetadata) -> Self {
        Self {
            metadata,
            tensors: HashMap::new(),
            format: CheckpointFormat::Raw,
        }
    }

    /// Adds a tensor to the checkpoint.
    pub fn add_tensor(&mut self, name: impl Into<String>, tensor: &BoundedTensor) {
        let name = name.into();
        let data = TensorData::from_bounded(&name, tensor);
        self.tensors.insert(name, data);
    }

    /// Gets a tensor from the checkpoint.
    pub fn get_tensor(&self, name: &str) -> Option<BoundedTensor> {
        self.tensors.get(name).map(|data| data.to_bounded())
    }

    /// Returns all tensor names.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(|s| s.as_str()).collect()
    }

    /// Returns the total size in bytes.
    pub fn total_size(&self) -> usize {
        self.tensors.values().map(|t| t.size_bytes()).sum()
    }

    /// Returns the total parameter count.
    pub fn total_params(&self) -> usize {
        self.tensors.values().map(|t| t.numel()).sum()
    }
}

/// Trait for types that can be checkpointed.
pub trait ModelCheckpoint {
    /// Collects all parameters into a checkpoint.
    fn checkpoint(&self) -> Checkpoint;

    /// Loads parameters from a checkpoint.
    fn load_checkpoint(&mut self, checkpoint: &Checkpoint) -> Result<(), SerializationError>;

    /// Returns the model name.
    fn model_name(&self) -> &str;

    /// Returns the model type.
    fn model_type(&self) -> &str;
}

/// Serializer for model checkpoints.
#[derive(Debug)]
pub struct ModelSerializer {
    /// Checkpoint format.
    format: CheckpointFormat,
    /// Whether to save error bounds.
    save_errors: bool,
    /// Whether to validate on load.
    validate_shapes: bool,
}

impl ModelSerializer {
    /// Creates a new serializer.
    pub fn new() -> Self {
        Self {
            format: CheckpointFormat::Raw,
            save_errors: true,
            validate_shapes: true,
        }
    }

    /// Sets the checkpoint format.
    pub fn with_format(mut self, format: CheckpointFormat) -> Self {
        self.format = format;
        self
    }

    /// Sets whether to save error bounds.
    pub fn with_errors(mut self, save_errors: bool) -> Self {
        self.save_errors = save_errors;
        self
    }

    /// Sets whether to validate shapes on load.
    pub fn with_validation(mut self, validate: bool) -> Self {
        self.validate_shapes = validate;
        self
    }

    /// Serializes a checkpoint to bytes.
    pub fn serialize(&self, checkpoint: &Checkpoint) -> Result<Vec<u8>, SerializationError> {
        let mut buffer = Vec::new();

        // Write header
        buffer.extend_from_slice(CHECKPOINT_MAGIC);
        buffer.extend_from_slice(&CHECKPOINT_VERSION.to_le_bytes());
        buffer.extend_from_slice(&(self.format as u32).to_le_bytes());

        // Serialize metadata
        let metadata_json = serialize_metadata(&checkpoint.metadata)?;
        let metadata_bytes = metadata_json.as_bytes();
        buffer.extend_from_slice(&(metadata_bytes.len() as u64).to_le_bytes());
        buffer.extend_from_slice(metadata_bytes);

        // Serialize tensor count
        buffer.extend_from_slice(&(checkpoint.tensors.len() as u32).to_le_bytes());

        // Serialize each tensor
        for (name, tensor_data) in &checkpoint.tensors {
            self.serialize_tensor(&mut buffer, name, tensor_data)?;
        }

        Ok(buffer)
    }

    /// Serializes a single tensor.
    fn serialize_tensor(
        &self,
        buffer: &mut Vec<u8>,
        name: &str,
        tensor: &TensorData,
    ) -> Result<(), SerializationError> {
        // Name
        let name_bytes = name.as_bytes();
        buffer.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
        buffer.extend_from_slice(name_bytes);

        // Shape
        buffer.extend_from_slice(&(tensor.shape.len() as u32).to_le_bytes());
        for &dim in &tensor.shape {
            buffer.extend_from_slice(&(dim as u64).to_le_bytes());
        }

        // Data type
        buffer.push(tensor.dtype.to_byte());

        // Has errors flag
        let has_errors = self.save_errors && tensor.errors.is_some();
        buffer.push(if has_errors { 1 } else { 0 });

        // Values
        for &value in &tensor.values {
            buffer.extend_from_slice(&value.to_le_bytes());
        }

        // Errors (if saving)
        if has_errors {
            if let Some(ref errors) = tensor.errors {
                for &error in errors {
                    buffer.extend_from_slice(&error.to_le_bytes());
                }
            }
        }

        Ok(())
    }

    /// Deserializes a checkpoint from bytes.
    pub fn deserialize(&self, data: &[u8]) -> Result<Checkpoint, SerializationError> {
        let mut cursor = std::io::Cursor::new(data);

        // Read and verify header
        let mut magic = [0u8; 8];
        cursor.read_exact(&mut magic)?;
        if &magic != CHECKPOINT_MAGIC {
            return Err(SerializationError::InvalidFormat("Invalid magic number".to_string()));
        }

        let mut version_bytes = [0u8; 4];
        cursor.read_exact(&mut version_bytes)?;
        let version = u32::from_le_bytes(version_bytes);
        if version != CHECKPOINT_VERSION {
            return Err(SerializationError::VersionMismatch {
                expected: CHECKPOINT_VERSION,
                actual: version,
            });
        }

        let mut format_bytes = [0u8; 4];
        cursor.read_exact(&mut format_bytes)?;
        let _format = u32::from_le_bytes(format_bytes);

        // Read metadata
        let mut metadata_len_bytes = [0u8; 8];
        cursor.read_exact(&mut metadata_len_bytes)?;
        let metadata_len = u64::from_le_bytes(metadata_len_bytes) as usize;

        let mut metadata_bytes = vec![0u8; metadata_len];
        cursor.read_exact(&mut metadata_bytes)?;
        let metadata: CheckpointMetadata = deserialize_metadata(&metadata_bytes)?;

        // Read tensor count
        let mut tensor_count_bytes = [0u8; 4];
        cursor.read_exact(&mut tensor_count_bytes)?;
        let tensor_count = u32::from_le_bytes(tensor_count_bytes) as usize;

        // Read tensors
        let mut tensors = HashMap::new();
        for _ in 0..tensor_count {
            let (name, tensor_data) = self.deserialize_tensor(&mut cursor)?;
            tensors.insert(name, tensor_data);
        }

        Ok(Checkpoint {
            metadata,
            tensors,
            format: CheckpointFormat::Raw,
        })
    }

    /// Deserializes a single tensor.
    fn deserialize_tensor<R: Read>(
        &self,
        reader: &mut R,
    ) -> Result<(String, TensorData), SerializationError> {
        // Name
        let mut name_len_bytes = [0u8; 4];
        reader.read_exact(&mut name_len_bytes)?;
        let name_len = u32::from_le_bytes(name_len_bytes) as usize;

        let mut name_bytes = vec![0u8; name_len];
        reader.read_exact(&mut name_bytes)?;
        let name = String::from_utf8(name_bytes)
            .map_err(|e| SerializationError::InvalidFormat(e.to_string()))?;

        // Shape
        let mut shape_len_bytes = [0u8; 4];
        reader.read_exact(&mut shape_len_bytes)?;
        let shape_len = u32::from_le_bytes(shape_len_bytes) as usize;

        let mut shape = Vec::with_capacity(shape_len);
        for _ in 0..shape_len {
            let mut dim_bytes = [0u8; 8];
            reader.read_exact(&mut dim_bytes)?;
            shape.push(u64::from_le_bytes(dim_bytes) as usize);
        }

        // Data type
        let mut dtype_byte = [0u8; 1];
        reader.read_exact(&mut dtype_byte)?;
        let dtype = TensorDType::from_byte(dtype_byte[0])
            .ok_or_else(|| SerializationError::InvalidFormat("Invalid dtype".to_string()))?;

        // Has errors flag
        let mut has_errors_byte = [0u8; 1];
        reader.read_exact(&mut has_errors_byte)?;
        let has_errors = has_errors_byte[0] == 1;

        // Values
        let numel: usize = shape.iter().product();
        let mut values = Vec::with_capacity(numel);
        for _ in 0..numel {
            let mut value_bytes = [0u8; 8];
            reader.read_exact(&mut value_bytes)?;
            values.push(f64::from_le_bytes(value_bytes));
        }

        // Errors
        let errors = if has_errors {
            let mut errors = Vec::with_capacity(numel);
            for _ in 0..numel {
                let mut error_bytes = [0u8; 8];
                reader.read_exact(&mut error_bytes)?;
                errors.push(f64::from_le_bytes(error_bytes));
            }
            Some(errors)
        } else {
            None
        };

        let tensor_data = TensorData {
            name: name.clone(),
            shape,
            values,
            errors,
            dtype,
        };
        Ok((name, tensor_data))
    }

    /// Saves a checkpoint to a writer.
    pub fn save<W: Write>(&self, checkpoint: &Checkpoint, writer: &mut W) -> Result<(), SerializationError> {
        let data = self.serialize(checkpoint)?;
        writer.write_all(&data)?;
        Ok(())
    }

    /// Loads a checkpoint from a reader.
    pub fn load<R: Read>(&self, reader: &mut R) -> Result<Checkpoint, SerializationError> {
        let mut data = Vec::new();
        reader.read_to_end(&mut data)?;
        self.deserialize(&data)
    }

    /// Saves a checkpoint to a file at the given path.
    ///
    /// Creates the file (and parent directories) if they don't exist.
    /// Overwrites any existing file at the path.
    pub fn save_to_file(
        &self,
        checkpoint: &Checkpoint,
        path: impl AsRef<Path>,
    ) -> Result<(), SerializationError> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut file = std::fs::File::create(path)?;
        self.save(checkpoint, &mut file)
    }

    /// Loads a checkpoint from a file at the given path.
    pub fn load_from_file(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<Checkpoint, SerializationError> {
        let mut file = std::fs::File::open(path)?;
        self.load(&mut file)
    }

    /// Saves a model that implements `ModelCheckpoint` to a file.
    ///
    /// Convenience method that collects the checkpoint and writes it in one step.
    pub fn save_model(
        &self,
        model: &dyn ModelCheckpoint,
        path: impl AsRef<Path>,
    ) -> Result<(), SerializationError> {
        let checkpoint = model.checkpoint();
        self.save_to_file(&checkpoint, path)
    }

    /// Loads a checkpoint from a file and applies it to a model.
    pub fn load_model(
        &self,
        model: &mut dyn ModelCheckpoint,
        path: impl AsRef<Path>,
    ) -> Result<(), SerializationError> {
        let checkpoint = self.load_from_file(path)?;
        model.load_checkpoint(&checkpoint)
    }

    /// Validates that a checkpoint matches expected shapes.
    pub fn validate_shapes(
        &self,
        checkpoint: &Checkpoint,
        expected: &HashMap<String, Vec<usize>>,
    ) -> Result<(), SerializationError> {
        for (name, expected_shape) in expected {
            if let Some(tensor) = checkpoint.tensors.get(name) {
                if &tensor.shape != expected_shape {
                    return Err(SerializationError::ShapeMismatch {
                        name: name.clone(),
                        expected: expected_shape.clone(),
                        actual: tensor.shape.clone(),
                    });
                }
            } else {
                return Err(SerializationError::TensorNotFound(name.clone()));
            }
        }
        Ok(())
    }
}

impl Default for ModelSerializer {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper function to serialize metadata to JSON.
fn serialize_metadata(metadata: &CheckpointMetadata) -> Result<String, SerializationError> {
    // Manual JSON serialization to avoid serde_json dependency issues
    let mut json = String::from("{");
    json.push_str(&format!(r#""model_name":"{}","#, escape_json(&metadata.model_name)));
    json.push_str(&format!(r#""model_type":"{}","#, escape_json(&metadata.model_type)));
    json.push_str(&format!(r#""param_count":{},"#, metadata.param_count));
    json.push_str(&format!(r#""training_step":{},"#, metadata.training_step));
    if let Some(loss) = metadata.training_loss {
        json.push_str(&format!(r#""training_loss":{},"#, loss));
    }
    if let Some(loss) = metadata.validation_loss {
        json.push_str(&format!(r#""validation_loss":{},"#, loss));
    }
    json.push_str(&format!(r#""timestamp":{}"#, metadata.timestamp));
    json.push('}');
    Ok(json)
}

/// Helper function to deserialize metadata from JSON.
fn deserialize_metadata(bytes: &[u8]) -> Result<CheckpointMetadata, SerializationError> {
    let s = String::from_utf8(bytes.to_vec())
        .map_err(|e| SerializationError::InvalidFormat(e.to_string()))?;

    // Simple parsing for our known format
    let mut metadata = CheckpointMetadata::default();

    // Extract fields using simple string parsing
    if let Some(start) = s.find(r#""model_name":""#) {
        let start = start + 14;
        if let Some(end) = s[start..].find('"') {
            metadata.model_name = s[start..start + end].to_string();
        }
    }
    if let Some(start) = s.find(r#""model_type":""#) {
        let start = start + 14;
        if let Some(end) = s[start..].find('"') {
            metadata.model_type = s[start..start + end].to_string();
        }
    }
    if let Some(start) = s.find(r#""param_count":"#) {
        let start = start + 14;
        if let Some(end_chars) = s[start..].find(|c: char| c == ',' || c == '}') {
            if let Ok(val) = s[start..start + end_chars].parse() {
                metadata.param_count = val;
            }
        }
    }
    if let Some(start) = s.find(r#""training_step":"#) {
        let start = start + 16;
        if let Some(end_chars) = s[start..].find(|c: char| c == ',' || c == '}') {
            if let Ok(val) = s[start..start + end_chars].parse() {
                metadata.training_step = val;
            }
        }
    }
    if let Some(start) = s.find(r#""timestamp":"#) {
        let start = start + 12;
        if let Some(end_chars) = s[start..].find(|c: char| c == ',' || c == '}') {
            if let Ok(val) = s[start..start + end_chars].parse() {
                metadata.timestamp = val;
            }
        }
    }

    Ok(metadata)
}

/// Helper to escape JSON strings.
fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_metadata() {
        let metadata = CheckpointMetadata::new("test_model", "gpt")
            .with_param_count(1_000_000)
            .with_training_step(1000);

        assert_eq!(metadata.model_name, "test_model");
        assert_eq!(metadata.param_count, 1_000_000);
    }

    #[test]
    fn test_tensor_data() {
        let tensor = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let data = TensorData::from_bounded("test", &tensor);

        assert_eq!(data.shape, vec![2, 2]);
        assert_eq!(data.values, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(data.numel(), 4);
    }

    #[test]
    fn test_tensor_roundtrip() {
        let original = BoundedTensor::from_approximate(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], 0.01);
        let data = TensorData::from_bounded("test", &original);
        let recovered = data.to_bounded();

        assert_eq!(original.shape(), recovered.shape());
        for (a, b) in original.values().iter().zip(recovered.values().iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_checkpoint_serialization() {
        let mut checkpoint = Checkpoint::new(CheckpointMetadata::new("test", "gpt"));

        let tensor1 = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let tensor2 = BoundedTensor::from_exact(vec![5.0, 6.0, 7.0, 8.0], vec![4]);

        checkpoint.add_tensor("layer1.weight", &tensor1);
        checkpoint.add_tensor("layer2.weight", &tensor2);

        let serializer = ModelSerializer::new();
        let bytes = serializer.serialize(&checkpoint).unwrap();

        let loaded = serializer.deserialize(&bytes).unwrap();

        assert_eq!(loaded.tensor_names().len(), 2);

        let t1 = loaded.get_tensor("layer1.weight").unwrap();
        assert_eq!(t1.shape(), &vec![2, 2]);
        assert_eq!(t1.values(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_checkpoint_with_errors() {
        let mut checkpoint = Checkpoint::new(CheckpointMetadata::new("test", "bert"));

        let tensor = BoundedTensor::from_approximate(vec![1.0, 2.0, 3.0, 4.0], vec![4], 0.001);
        checkpoint.add_tensor("weights", &tensor);

        let serializer = ModelSerializer::new().with_errors(true);
        let bytes = serializer.serialize(&checkpoint).unwrap();
        let loaded = serializer.deserialize(&bytes).unwrap();

        let recovered = loaded.get_tensor("weights").unwrap();
        assert!(recovered.max_error() > 0.0);
    }

    #[test]
    fn test_invalid_format() {
        let serializer = ModelSerializer::new();
        let result = serializer.deserialize(b"INVALID!");
        assert!(matches!(result, Err(SerializationError::InvalidFormat(_))));
    }

    #[test]
    fn test_shape_validation() {
        let mut checkpoint = Checkpoint::new(CheckpointMetadata::default());
        checkpoint.add_tensor("weights", &BoundedTensor::zeros(vec![2, 2]));

        let serializer = ModelSerializer::new();

        let mut expected = HashMap::new();
        expected.insert("weights".to_string(), vec![2, 2]);
        assert!(serializer.validate_shapes(&checkpoint, &expected).is_ok());

        expected.insert("weights".to_string(), vec![3, 3]);
        assert!(matches!(
            serializer.validate_shapes(&checkpoint, &expected),
            Err(SerializationError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn test_checkpoint_size() {
        let mut checkpoint = Checkpoint::new(CheckpointMetadata::default());
        checkpoint.add_tensor("w1", &BoundedTensor::zeros(vec![100, 100]));
        checkpoint.add_tensor("w2", &BoundedTensor::zeros(vec![50, 50]));

        assert_eq!(checkpoint.total_params(), 100 * 100 + 50 * 50);
        assert!(checkpoint.total_size() > 0);
    }

    #[test]
    fn test_dtype_conversion() {
        assert_eq!(TensorDType::F64.to_byte(), 0);
        assert_eq!(TensorDType::from_byte(0), Some(TensorDType::F64));
        assert_eq!(TensorDType::from_byte(255), None);
    }

    #[test]
    fn test_save_load_file() {
        let dir = std::env::temp_dir().join("helix_test_checkpoint");
        let path = dir.join("test_model.helixchk");

        let mut checkpoint = Checkpoint::new(
            CheckpointMetadata::new("test_file_io", "mlp")
                .with_param_count(100)
                .with_training_step(42)
                .with_training_loss(0.123),
        );

        let tensor = BoundedTensor::from_approximate(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            0.01,
        );
        checkpoint.add_tensor("layer1.weight", &tensor);

        let serializer = ModelSerializer::new();
        serializer.save_to_file(&checkpoint, &path).unwrap();

        let loaded = serializer.load_from_file(&path).unwrap();
        assert_eq!(loaded.metadata.model_name, "test_file_io");
        assert_eq!(loaded.metadata.training_step, 42);

        let t = loaded.get_tensor("layer1.weight").unwrap();
        assert_eq!(t.shape(), &vec![2, 3]);
        assert!((t.values()[0] - 1.0).abs() < 1e-10);

        // Cleanup
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn test_save_load_creates_directories() {
        let dir = std::env::temp_dir()
            .join("helix_test_nested")
            .join("deep")
            .join("path");
        let path = dir.join("model.helixchk");

        let checkpoint = Checkpoint::new(CheckpointMetadata::new("nested", "test"));
        let serializer = ModelSerializer::new();
        serializer.save_to_file(&checkpoint, &path).unwrap();

        let loaded = serializer.load_from_file(&path).unwrap();
        assert_eq!(loaded.metadata.model_name, "nested");

        // Cleanup
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir_all(
            std::env::temp_dir().join("helix_test_nested"),
        );
    }
}
