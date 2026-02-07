//! Model Serialization Module.
//!
//! Provides functionality for serializing and deserializing ML models
//! with support for different formats and compression.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Model format for serialization.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ModelFormat {
    /// Binary format (native).
    Binary,
    /// JSON format.
    Json,
    /// MessagePack format.
    MsgPack,
    /// ONNX format.
    Onnx,
    /// SafeTensors format.
    SafeTensors,
}

impl Default for ModelFormat {
    fn default() -> Self {
        Self::Binary
    }
}

/// Compression for model files.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ModelCompression {
    /// No compression.
    None,
    /// Gzip compression.
    Gzip,
    /// LZ4 compression.
    Lz4,
    /// Zstd compression.
    Zstd,
}

impl Default for ModelCompression {
    fn default() -> Self {
        Self::None
    }
}

/// Layer type in a model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayerType {
    Linear { in_features: usize, out_features: usize },
    Conv2d { in_channels: usize, out_channels: usize, kernel_size: usize },
    Embedding { num_embeddings: usize, embedding_dim: usize },
    LayerNorm { normalized_shape: Vec<usize> },
    Attention { num_heads: usize, embed_dim: usize },
    Activation(String),
    Dropout { p: f64 },
    Custom(String),
}

/// A single layer's weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerWeights {
    /// Layer name.
    pub name: String,
    /// Layer type.
    pub layer_type: LayerType,
    /// Weight tensors (name -> data).
    pub tensors: HashMap<String, TensorData>,
}

/// Tensor data for serialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorData {
    /// Shape of the tensor.
    pub shape: Vec<usize>,
    /// Data type.
    pub dtype: String,
    /// Raw bytes.
    pub data: Vec<u8>,
}

impl TensorData {
    /// Creates tensor data from f32 values.
    pub fn from_f32(shape: Vec<usize>, values: &[f32]) -> Self {
        let data: Vec<u8> = values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        
        Self {
            shape,
            dtype: "float32".to_string(),
            data,
        }
    }

    /// Converts to f32 values.
    pub fn to_f32(&self) -> Vec<f32> {
        self.data
            .chunks(4)
            .map(|chunk| {
                let arr: [u8; 4] = chunk.try_into().unwrap_or([0; 4]);
                f32::from_le_bytes(arr)
            })
            .collect()
    }

    /// Returns the number of elements.
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Complete serialized model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedModel {
    /// Model name.
    pub name: String,
    /// Model version.
    pub version: String,
    /// Architecture description.
    pub architecture: String,
    /// Layer weights.
    pub layers: Vec<LayerWeights>,
    /// Model metadata.
    pub metadata: ModelMetadata,
    /// Hash of the model.
    pub hash: [u8; 32],
}

/// Model metadata.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Training round (for federated learning).
    pub training_round: u64,
    /// Total parameters.
    pub total_params: usize,
    /// Creation timestamp.
    pub created_at: u64,
    /// Error bound.
    pub error_bound: f64,
    /// Additional fields.
    pub extra: HashMap<String, String>,
}

impl SerializedModel {
    /// Creates a new empty model.
    pub fn new(name: String, version: String, architecture: String) -> Self {
        Self {
            name,
            version,
            architecture,
            layers: Vec::new(),
            metadata: ModelMetadata::default(),
            hash: [0; 32],
        }
    }

    /// Adds a layer.
    pub fn add_layer(&mut self, layer: LayerWeights) {
        self.layers.push(layer);
    }

    /// Computes the model hash.
    pub fn compute_hash(&mut self) {
        // Simple hash based on layer data
        let mut hash = [0u8; 32];
        for (i, layer) in self.layers.iter().enumerate() {
            for (j, (_, tensor)) in layer.tensors.iter().enumerate() {
                if let Some(byte) = tensor.data.first() {
                    hash[(i + j) % 32] ^= byte;
                }
            }
        }
        self.hash = hash;
    }

    /// Returns total parameter count.
    pub fn total_params(&self) -> usize {
        self.layers
            .iter()
            .flat_map(|l| l.tensors.values())
            .map(|t| t.num_elements())
            .sum()
    }
}

/// Model serializer.
pub struct ModelSerializer {
    /// Format to use.
    format: ModelFormat,
    /// Compression to use.
    #[allow(dead_code)]
    compression: ModelCompression,
}

impl ModelSerializer {
    /// Creates a new serializer.
    pub fn new(format: ModelFormat, compression: ModelCompression) -> Self {
        Self { format, compression }
    }

    /// Serializes a model to bytes.
    pub fn serialize(&self, model: &SerializedModel) -> Result<Vec<u8>, String> {
        let data = match self.format {
            ModelFormat::Binary | ModelFormat::MsgPack | ModelFormat::Onnx | ModelFormat::SafeTensors => {
                // Use JSON as fallback for all formats in this implementation
                serde_json::to_vec(model)
                    .map_err(|e| format!("Serialization error: {}", e))?
            }
            ModelFormat::Json => {
                serde_json::to_vec_pretty(model)
                    .map_err(|e| format!("JSON serialization error: {}", e))?
            }
        };

        // Apply compression (simplified - no actual compression in this impl)
        Ok(data)
    }

    /// Deserializes a model from bytes.
    pub fn deserialize(&self, data: &[u8]) -> Result<SerializedModel, String> {
        // Decompress if needed (simplified)
        
        match self.format {
            ModelFormat::Binary | ModelFormat::MsgPack | ModelFormat::Onnx | ModelFormat::SafeTensors | ModelFormat::Json => {
                serde_json::from_slice(data)
                    .map_err(|e| format!("Deserialization error: {}", e))
            }
        }
    }

    /// Saves a model to a file.
    pub fn save_to_file(&self, model: &SerializedModel, path: &str) -> Result<(), String> {
        let data = self.serialize(model)?;
        std::fs::write(path, data)
            .map_err(|e| format!("Failed to write file: {}", e))
    }

    /// Loads a model from a file.
    pub fn load_from_file(&self, path: &str) -> Result<SerializedModel, String> {
        let data = std::fs::read(path)
            .map_err(|e| format!("Failed to read file: {}", e))?;
        self.deserialize(&data)
    }
}

/// Model diff for incremental updates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDiff {
    /// Base model hash.
    pub base_hash: [u8; 32],
    /// New model hash.
    pub new_hash: [u8; 32],
    /// Changed layer indices.
    pub changed_layers: Vec<usize>,
    /// Layer diffs.
    pub diffs: Vec<LayerDiff>,
}

/// Diff for a single layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerDiff {
    /// Layer index.
    pub layer_idx: usize,
    /// Tensor name.
    pub tensor_name: String,
    /// Delta values (sparse representation).
    pub deltas: Vec<(usize, f32)>,
}

impl ModelDiff {
    /// Creates a diff between two models.
    pub fn create(old: &SerializedModel, new: &SerializedModel) -> Self {
        let mut changed_layers = Vec::new();
        let mut diffs = Vec::new();

        for (i, (old_layer, new_layer)) in old.layers.iter().zip(new.layers.iter()).enumerate() {
            let mut layer_changed = false;
            
            for (name, old_tensor) in &old_layer.tensors {
                if let Some(new_tensor) = new_layer.tensors.get(name) {
                    let old_vals = old_tensor.to_f32();
                    let new_vals = new_tensor.to_f32();
                    
                    let deltas: Vec<(usize, f32)> = old_vals
                        .iter()
                        .zip(new_vals.iter())
                        .enumerate()
                        .filter(|(_, (o, n))| (*o - *n).abs() > 1e-7)
                        .map(|(idx, (_, n))| (idx, *n))
                        .collect();

                    if !deltas.is_empty() {
                        layer_changed = true;
                        diffs.push(LayerDiff {
                            layer_idx: i,
                            tensor_name: name.clone(),
                            deltas,
                        });
                    }
                }
            }

            if layer_changed {
                changed_layers.push(i);
            }
        }

        Self {
            base_hash: old.hash,
            new_hash: new.hash,
            changed_layers,
            diffs,
        }
    }

    /// Applies the diff to a base model.
    pub fn apply(&self, base: &mut SerializedModel) -> Result<(), String> {
        if base.hash != self.base_hash {
            return Err("Base hash mismatch".to_string());
        }

        for diff in &self.diffs {
            if let Some(layer) = base.layers.get_mut(diff.layer_idx) {
                if let Some(tensor) = layer.tensors.get_mut(&diff.tensor_name) {
                    let mut values = tensor.to_f32();
                    for &(idx, val) in &diff.deltas {
                        if idx < values.len() {
                            values[idx] = val;
                        }
                    }
                    *tensor = TensorData::from_f32(tensor.shape.clone(), &values);
                }
            }
        }

        base.hash = self.new_hash;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_data_roundtrip() {
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let tensor = TensorData::from_f32(vec![2, 2], &values);
        
        let recovered = tensor.to_f32();
        assert_eq!(values, recovered);
    }

    #[test]
    fn test_model_serialization() {
        let mut model = SerializedModel::new(
            "test_model".to_string(),
            "1.0".to_string(),
            "mlp".to_string(),
        );

        let layer = LayerWeights {
            name: "fc1".to_string(),
            layer_type: LayerType::Linear { in_features: 10, out_features: 5 },
            tensors: {
                let mut t = HashMap::new();
                t.insert("weight".to_string(), TensorData::from_f32(vec![5, 10], &vec![0.1; 50]));
                t.insert("bias".to_string(), TensorData::from_f32(vec![5], &vec![0.0; 5]));
                t
            },
        };

        model.add_layer(layer);
        model.compute_hash();

        let serializer = ModelSerializer::new(ModelFormat::Json, ModelCompression::None);
        let data = serializer.serialize(&model).unwrap();
        let recovered = serializer.deserialize(&data).unwrap();

        assert_eq!(model.name, recovered.name);
        assert_eq!(model.layers.len(), recovered.layers.len());
    }

    #[test]
    fn test_model_diff() {
        let mut model1 = SerializedModel::new(
            "test".to_string(),
            "1.0".to_string(),
            "mlp".to_string(),
        );
        
        model1.add_layer(LayerWeights {
            name: "fc1".to_string(),
            layer_type: LayerType::Linear { in_features: 2, out_features: 2 },
            tensors: {
                let mut t = HashMap::new();
                t.insert("weight".to_string(), TensorData::from_f32(vec![4], &vec![1.0, 2.0, 3.0, 4.0]));
                t
            },
        });
        model1.compute_hash();

        let mut model2 = model1.clone();
        if let Some(tensor) = model2.layers[0].tensors.get_mut("weight") {
            *tensor = TensorData::from_f32(vec![4], &vec![1.0, 2.5, 3.0, 4.5]);
        }
        model2.compute_hash();

        let diff = ModelDiff::create(&model1, &model2);
        assert_eq!(diff.changed_layers.len(), 1);
        assert_eq!(diff.diffs[0].deltas.len(), 2);
    }
}
