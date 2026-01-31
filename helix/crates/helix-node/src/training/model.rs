//! Model Weight Management.
//!
//! Handles loading, saving, and updating model weights for distributed training.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};
use helix_core::types::{BoundedTensor, BoundedValue};

/// Model weight identifier.
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct WeightId(pub String);

impl WeightId {
    /// Creates a new weight ID.
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }
}

impl std::fmt::Display for WeightId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Model version tracking.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct ModelVersion(pub u64);

impl ModelVersion {
    /// Initial version.
    pub fn initial() -> Self {
        Self(0)
    }

    /// Increments the version.
    pub fn next(&self) -> Self {
        Self(self.0 + 1)
    }
}

/// Metadata about the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model name/identifier.
    pub name: String,
    /// Model version.
    pub version: ModelVersion,
    /// Model architecture type.
    pub architecture: String,
    /// Number of parameters.
    pub num_parameters: usize,
    /// Hidden dimension.
    pub hidden_dim: usize,
    /// Number of layers.
    pub num_layers: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// Training iteration count.
    pub training_iterations: u64,
    /// Commitment hash for on-chain verification.
    pub commitment: [u8; 32],
}

impl Default for ModelMetadata {
    fn default() -> Self {
        Self {
            name: "unnamed".to_string(),
            version: ModelVersion::initial(),
            architecture: "transformer".to_string(),
            num_parameters: 0,
            hidden_dim: 256,
            num_layers: 4,
            num_heads: 4,
            vocab_size: 32000,
            max_seq_len: 512,
            training_iterations: 0,
            commitment: [0u8; 32],
        }
    }
}

/// A single layer's weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerWeights {
    /// Layer index.
    pub layer_idx: usize,
    /// Weight tensors in this layer.
    pub weights: HashMap<String, WeightData>,
}

/// Serialized weight data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightData {
    /// Shape of the tensor.
    pub shape: Vec<usize>,
    /// Flattened data (f32 for storage efficiency).
    pub data: Vec<f32>,
    /// Error bound for this weight.
    pub error_bound: f64,
}

impl WeightData {
    /// Creates weight data from a BoundedTensor.
    pub fn from_tensor(tensor: &BoundedTensor) -> Self {
        let data: Vec<f32> = tensor.data()
            .iter()
            .map(|bv| bv.value() as f32)
            .collect();
        
        let error_bound = tensor.max_error();

        Self {
            shape: tensor.shape().clone(),
            data,
            error_bound,
        }
    }

    /// Converts to a BoundedTensor.
    pub fn to_tensor(&self) -> BoundedTensor {
        let data: Vec<f64> = self.data.iter().map(|&v| v as f64).collect();
        BoundedTensor::from_approximate(data, self.shape.clone(), self.error_bound)
    }

    /// Number of elements.
    pub fn numel(&self) -> usize {
        self.data.len()
    }
}

/// Complete model weights.
#[derive(Debug, Clone)]
pub struct ModelWeights {
    /// Model metadata.
    pub metadata: ModelMetadata,
    /// Embedding weights.
    pub embeddings: Option<WeightData>,
    /// Layer weights.
    pub layers: Vec<LayerWeights>,
    /// Output projection / LM head.
    pub lm_head: Option<WeightData>,
    /// Additional named weights.
    pub extra_weights: HashMap<WeightId, WeightData>,
}

impl ModelWeights {
    /// Creates empty model weights.
    pub fn empty(metadata: ModelMetadata) -> Self {
        Self {
            metadata,
            embeddings: None,
            layers: Vec::new(),
            lm_head: None,
            extra_weights: HashMap::new(),
        }
    }

    /// Creates model weights with random initialization.
    pub fn random(metadata: ModelMetadata, seed: u64) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        seed.hash(&mut hasher);
        let mut rng_state = hasher.finish();

        let next_rand = |state: &mut u64| -> f32 {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let bits = (*state >> 33) as u32;
            (bits as f32 / u32::MAX as f32 - 0.5) * 0.02
        };

        // Create embeddings [vocab_size, hidden_dim]
        let embed_size = metadata.vocab_size * metadata.hidden_dim;
        let embed_data: Vec<f32> = (0..embed_size).map(|_| next_rand(&mut rng_state)).collect();
        let embeddings = WeightData {
            shape: vec![metadata.vocab_size, metadata.hidden_dim],
            data: embed_data,
            error_bound: 0.0,
        };

        // Create layer weights
        let mut layers = Vec::with_capacity(metadata.num_layers);
        for layer_idx in 0..metadata.num_layers {
            let mut weights = HashMap::new();

            // Attention weights
            let head_dim = metadata.hidden_dim / metadata.num_heads;
            let attn_size = metadata.hidden_dim * metadata.hidden_dim;
            
            for name in &["q_proj", "k_proj", "v_proj", "o_proj"] {
                let data: Vec<f32> = (0..attn_size).map(|_| next_rand(&mut rng_state)).collect();
                weights.insert(name.to_string(), WeightData {
                    shape: vec![metadata.hidden_dim, metadata.hidden_dim],
                    data,
                    error_bound: 0.0,
                });
            }

            // MLP weights (typically 4x hidden_dim)
            let mlp_hidden = metadata.hidden_dim * 4;
            
            let up_data: Vec<f32> = (0..metadata.hidden_dim * mlp_hidden)
                .map(|_| next_rand(&mut rng_state)).collect();
            weights.insert("mlp_up".to_string(), WeightData {
                shape: vec![metadata.hidden_dim, mlp_hidden],
                data: up_data,
                error_bound: 0.0,
            });

            let down_data: Vec<f32> = (0..mlp_hidden * metadata.hidden_dim)
                .map(|_| next_rand(&mut rng_state)).collect();
            weights.insert("mlp_down".to_string(), WeightData {
                shape: vec![mlp_hidden, metadata.hidden_dim],
                data: down_data,
                error_bound: 0.0,
            });

            // Layer norms
            let norm_data: Vec<f32> = vec![1.0; metadata.hidden_dim];
            weights.insert("ln1_weight".to_string(), WeightData {
                shape: vec![metadata.hidden_dim],
                data: norm_data.clone(),
                error_bound: 0.0,
            });
            weights.insert("ln2_weight".to_string(), WeightData {
                shape: vec![metadata.hidden_dim],
                data: norm_data,
                error_bound: 0.0,
            });

            layers.push(LayerWeights { layer_idx, weights });
        }

        // LM head (tied to embeddings in many models, but separate here)
        let lm_head_size = metadata.hidden_dim * metadata.vocab_size;
        let lm_head_data: Vec<f32> = (0..lm_head_size).map(|_| next_rand(&mut rng_state)).collect();
        let lm_head = WeightData {
            shape: vec![metadata.hidden_dim, metadata.vocab_size],
            data: lm_head_data,
            error_bound: 0.0,
        };

        let mut model = Self {
            metadata,
            embeddings: Some(embeddings),
            layers,
            lm_head: Some(lm_head),
            extra_weights: HashMap::new(),
        };

        model.metadata.num_parameters = model.count_parameters();
        model
    }

    /// Counts total parameters.
    pub fn count_parameters(&self) -> usize {
        let mut count = 0;
        
        if let Some(embed) = &self.embeddings {
            count += embed.numel();
        }
        
        for layer in &self.layers {
            for weight in layer.weights.values() {
                count += weight.numel();
            }
        }
        
        if let Some(lm_head) = &self.lm_head {
            count += lm_head.numel();
        }
        
        for weight in self.extra_weights.values() {
            count += weight.numel();
        }
        
        count
    }

    /// Gets a specific weight by name.
    pub fn get_weight(&self, name: &str) -> Option<&WeightData> {
        // Check embeddings
        if name == "embeddings" {
            return self.embeddings.as_ref();
        }
        
        // Check lm_head
        if name == "lm_head" {
            return self.lm_head.as_ref();
        }
        
        // Check layer weights (format: "layer.{idx}.{name}")
        if name.starts_with("layer.") {
            let parts: Vec<&str> = name.split('.').collect();
            if parts.len() >= 3 {
                if let Ok(layer_idx) = parts[1].parse::<usize>() {
                    if layer_idx < self.layers.len() {
                        let weight_name = parts[2..].join(".");
                        return self.layers[layer_idx].weights.get(&weight_name);
                    }
                }
            }
        }
        
        // Check extra weights
        self.extra_weights.get(&WeightId::new(name))
    }

    /// Applies a gradient update to weights.
    pub fn apply_gradient(
        &mut self,
        gradient: &ModelGradient,
        learning_rate: f64,
    ) -> Result<(), ModelError> {
        // Apply to embeddings
        if let (Some(embed), Some(grad)) = (&mut self.embeddings, &gradient.embeddings) {
            apply_gradient_to_weight(embed, grad, learning_rate)?;
        }

        // Apply to layers
        for (layer, layer_grad) in self.layers.iter_mut().zip(gradient.layers.iter()) {
            for (name, weight) in layer.weights.iter_mut() {
                if let Some(grad) = layer_grad.gradients.get(name) {
                    apply_gradient_to_weight(weight, grad, learning_rate)?;
                }
            }
        }

        // Apply to lm_head
        if let (Some(lm_head), Some(grad)) = (&mut self.lm_head, &gradient.lm_head) {
            apply_gradient_to_weight(lm_head, grad, learning_rate)?;
        }

        // Update metadata
        self.metadata.training_iterations += 1;
        self.metadata.version = self.metadata.version.next();
        self.update_commitment();

        Ok(())
    }

    /// Computes and updates the model commitment hash.
    pub fn update_commitment(&mut self) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        
        // Hash all weights
        if let Some(embed) = &self.embeddings {
            for v in &embed.data {
                v.to_bits().hash(&mut hasher);
            }
        }
        
        for layer in &self.layers {
            for weight in layer.weights.values() {
                for v in &weight.data {
                    v.to_bits().hash(&mut hasher);
                }
            }
        }
        
        if let Some(lm_head) = &self.lm_head {
            for v in &lm_head.data {
                v.to_bits().hash(&mut hasher);
            }
        }

        let hash = hasher.finish();
        self.metadata.commitment[..8].copy_from_slice(&hash.to_le_bytes());
        
        // Fill rest with deterministic values
        self.metadata.version.0.hash(&mut hasher);
        let hash2 = hasher.finish();
        self.metadata.commitment[8..16].copy_from_slice(&hash2.to_le_bytes());
        
        self.metadata.training_iterations.hash(&mut hasher);
        let hash3 = hasher.finish();
        self.metadata.commitment[16..24].copy_from_slice(&hash3.to_le_bytes());
        
        self.metadata.num_parameters.hash(&mut hasher);
        let hash4 = hasher.finish();
        self.metadata.commitment[24..32].copy_from_slice(&hash4.to_le_bytes());
    }

    /// Saves weights to a file.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), ModelError> {
        let serializable = SerializableModelWeights::from_model(self);
        let json = serde_json::to_string(&serializable)
            .map_err(|e| ModelError::SerializationFailed(e.to_string()))?;
        std::fs::write(path, json)
            .map_err(|e| ModelError::IoError(e.to_string()))?;
        Ok(())
    }

    /// Loads weights from a file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ModelError> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| ModelError::IoError(e.to_string()))?;
        let serializable: SerializableModelWeights = serde_json::from_str(&json)
            .map_err(|e| ModelError::DeserializationFailed(e.to_string()))?;
        Ok(serializable.to_model())
    }
}

/// Serializable form of model weights.
#[derive(Debug, Serialize, Deserialize)]
struct SerializableModelWeights {
    metadata: ModelMetadata,
    embeddings: Option<WeightData>,
    layers: Vec<LayerWeights>,
    lm_head: Option<WeightData>,
    extra_weights: Vec<(String, WeightData)>,
}

impl SerializableModelWeights {
    fn from_model(model: &ModelWeights) -> Self {
        Self {
            metadata: model.metadata.clone(),
            embeddings: model.embeddings.clone(),
            layers: model.layers.clone(),
            lm_head: model.lm_head.clone(),
            extra_weights: model.extra_weights
                .iter()
                .map(|(k, v)| (k.0.clone(), v.clone()))
                .collect(),
        }
    }

    fn to_model(self) -> ModelWeights {
        ModelWeights {
            metadata: self.metadata,
            embeddings: self.embeddings,
            layers: self.layers,
            lm_head: self.lm_head,
            extra_weights: self.extra_weights
                .into_iter()
                .map(|(k, v)| (WeightId::new(k), v))
                .collect(),
        }
    }
}

/// Gradient data for a layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerGradient {
    /// Layer index.
    pub layer_idx: usize,
    /// Gradients for each weight.
    pub gradients: HashMap<String, WeightData>,
}

/// Complete model gradient.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelGradient {
    /// Embedding gradient.
    pub embeddings: Option<WeightData>,
    /// Layer gradients.
    pub layers: Vec<LayerGradient>,
    /// LM head gradient.
    pub lm_head: Option<WeightData>,
    /// Global error bound for this gradient.
    pub error_bound: f64,
}

impl ModelGradient {
    /// Creates a zero gradient matching model structure.
    pub fn zeros_like(model: &ModelWeights) -> Self {
        let embeddings = model.embeddings.as_ref().map(|e| WeightData {
            shape: e.shape.clone(),
            data: vec![0.0; e.data.len()],
            error_bound: 0.0,
        });

        let layers: Vec<LayerGradient> = model.layers.iter().map(|layer| {
            let gradients = layer.weights.iter().map(|(name, weight)| {
                (name.clone(), WeightData {
                    shape: weight.shape.clone(),
                    data: vec![0.0; weight.data.len()],
                    error_bound: 0.0,
                })
            }).collect();
            LayerGradient {
                layer_idx: layer.layer_idx,
                gradients,
            }
        }).collect();

        let lm_head = model.lm_head.as_ref().map(|l| WeightData {
            shape: l.shape.clone(),
            data: vec![0.0; l.data.len()],
            error_bound: 0.0,
        });

        Self {
            embeddings,
            layers,
            lm_head,
            error_bound: 0.0,
        }
    }

    /// Computes gradient hash for commitment.
    pub fn commitment(&self) -> [u8; 32] {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        
        if let Some(embed) = &self.embeddings {
            for v in &embed.data {
                v.to_bits().hash(&mut hasher);
            }
        }
        
        for layer in &self.layers {
            for grad in layer.gradients.values() {
                for v in &grad.data {
                    v.to_bits().hash(&mut hasher);
                }
            }
        }

        let hash1 = hasher.finish();
        let mut commitment = [0u8; 32];
        commitment[..8].copy_from_slice(&hash1.to_le_bytes());
        
        if let Some(lm_head) = &self.lm_head {
            for v in &lm_head.data {
                v.to_bits().hash(&mut hasher);
            }
        }
        let hash2 = hasher.finish();
        commitment[8..16].copy_from_slice(&hash2.to_le_bytes());
        
        self.error_bound.to_bits().hash(&mut hasher);
        let hash3 = hasher.finish();
        commitment[16..24].copy_from_slice(&hash3.to_le_bytes());
        commitment[24..32].copy_from_slice(&hash3.to_le_bytes());
        
        commitment
    }
}

/// Applies gradient to a weight tensor.
fn apply_gradient_to_weight(
    weight: &mut WeightData,
    gradient: &WeightData,
    learning_rate: f64,
) -> Result<(), ModelError> {
    if weight.shape != gradient.shape {
        return Err(ModelError::ShapeMismatch {
            expected: weight.shape.clone(),
            got: gradient.shape.clone(),
        });
    }

    for (w, g) in weight.data.iter_mut().zip(gradient.data.iter()) {
        *w -= (learning_rate * (*g as f64)) as f32;
    }

    // Update error bound
    weight.error_bound = (weight.error_bound + gradient.error_bound * learning_rate).min(1.0);

    Ok(())
}

/// Model operation errors.
#[derive(Debug)]
pub enum ModelError {
    /// Shape mismatch during gradient application.
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    /// Serialization failed.
    SerializationFailed(String),
    /// Deserialization failed.
    DeserializationFailed(String),
    /// IO error.
    IoError(String),
    /// Weight not found.
    WeightNotFound(String),
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ShapeMismatch { expected, got } => {
                write!(f, "Shape mismatch: expected {:?}, got {:?}", expected, got)
            }
            Self::SerializationFailed(msg) => write!(f, "Serialization failed: {}", msg),
            Self::DeserializationFailed(msg) => write!(f, "Deserialization failed: {}", msg),
            Self::IoError(msg) => write!(f, "IO error: {}", msg),
            Self::WeightNotFound(name) => write!(f, "Weight not found: {}", name),
        }
    }
}

impl std::error::Error for ModelError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_model_creation() {
        let metadata = ModelMetadata {
            name: "test-model".to_string(),
            hidden_dim: 64,
            num_layers: 2,
            num_heads: 2,
            vocab_size: 100,
            ..Default::default()
        };

        let model = ModelWeights::random(metadata, 42);
        
        assert!(model.embeddings.is_some());
        assert_eq!(model.layers.len(), 2);
        assert!(model.lm_head.is_some());
        assert!(model.count_parameters() > 0);
    }

    #[test]
    fn test_gradient_application() {
        let metadata = ModelMetadata {
            hidden_dim: 8,
            num_layers: 1,
            num_heads: 1,
            vocab_size: 10,
            ..Default::default()
        };

        let mut model = ModelWeights::random(metadata, 42);
        let old_version = model.metadata.version;
        let old_commitment = model.metadata.commitment;

        let gradient = ModelGradient::zeros_like(&model);
        model.apply_gradient(&gradient, 0.01).unwrap();

        assert!(model.metadata.version > old_version);
        // Note: commitment may or may not change with zero gradient
    }

    #[test]
    fn test_weight_data_conversion() {
        let tensor = BoundedTensor::from_approximate(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
            0.01,
        );

        let weight_data = WeightData::from_tensor(&tensor);
        assert_eq!(weight_data.shape, vec![2, 2]);
        assert_eq!(weight_data.data.len(), 4);

        let reconstructed = weight_data.to_tensor();
        assert_eq!(reconstructed.shape(), &vec![2, 2]);
    }
}
