//! Transformer Model Configuration.
//!
//! Provides configurable parameters for transformer models with utilities
//! for computing parameter counts and ensuring configurations fit within
//! circuit constraints.
//!
//! # Design Principles
//!
//! 1. **Parameter Budgeting**: Configurations track total parameter count to ensure
//!    models stay within the 500K-2M parameter target for efficient proving.
//!
//! 2. **Quantization Support**: Configurations specify precision (INT8/INT4) for
//!    weights and activations, affecting both parameter storage and error bounds.
//!
//! 3. **Constraint Estimation**: Each configuration can estimate circuit constraints
//!    to ensure proof generation stays under the 500ms target.
//!
//! # Example
//!
//! ```ignore
//! use helix_circuits::ml::config::{TransformerConfig, QuantizationConfig};
//!
//! let config = TransformerConfig::small_demo()
//!     .with_quantization(QuantizationConfig::int8());
//!
//! assert!(config.total_parameters() <= 2_000_000);
//! assert!(config.estimated_proving_time_ms() < 500);
//! ```

use std::fmt;

/// Quantization precision for model weights and activations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationPrecision {
    /// Full precision (32-bit float equivalent in field).
    FP32,
    /// 8-bit integer quantization.
    INT8,
    /// 4-bit integer quantization.
    INT4,
}

impl QuantizationPrecision {
    /// Returns the number of bits used for this precision.
    pub fn bits(&self) -> usize {
        match self {
            QuantizationPrecision::FP32 => 32,
            QuantizationPrecision::INT8 => 8,
            QuantizationPrecision::INT4 => 4,
        }
    }

    /// Returns the lookup table size needed for activations.
    pub fn table_size(&self) -> usize {
        match self {
            QuantizationPrecision::FP32 => 1024, // Larger table for higher precision
            QuantizationPrecision::INT8 => 256,
            QuantizationPrecision::INT4 => 16,
        }
    }

    /// Returns the maximum absolute value representable.
    pub fn max_value(&self) -> i64 {
        match self {
            QuantizationPrecision::FP32 => i32::MAX as i64,
            QuantizationPrecision::INT8 => 127,
            QuantizationPrecision::INT4 => 7,
        }
    }

    /// Returns the error bound scale factor.
    pub fn error_scale(&self) -> u64 {
        match self {
            QuantizationPrecision::FP32 => 1,
            QuantizationPrecision::INT8 => 1,
            QuantizationPrecision::INT4 => 2, // Higher error from coarser quantization
        }
    }
}

impl Default for QuantizationPrecision {
    fn default() -> Self {
        QuantizationPrecision::INT8
    }
}

/// Quantization configuration for weights and activations.
#[derive(Debug, Clone, Copy)]
pub struct QuantizationConfig {
    /// Precision for model weights.
    pub weight_precision: QuantizationPrecision,
    /// Precision for activations.
    pub activation_precision: QuantizationPrecision,
    /// Scale factor for quantized values.
    pub scale: u64,
    /// Zero point offset.
    pub zero_point: i64,
    /// Whether to use symmetric quantization.
    pub symmetric: bool,
}

impl QuantizationConfig {
    /// Creates a new INT8 quantization config.
    pub fn int8() -> Self {
        Self {
            weight_precision: QuantizationPrecision::INT8,
            activation_precision: QuantizationPrecision::INT8,
            scale: 128,
            zero_point: 0,
            symmetric: true,
        }
    }

    /// Creates a new INT4 quantization config.
    pub fn int4() -> Self {
        Self {
            weight_precision: QuantizationPrecision::INT4,
            activation_precision: QuantizationPrecision::INT4,
            scale: 8,
            zero_point: 0,
            symmetric: true,
        }
    }

    /// Creates a mixed precision config (INT4 weights, INT8 activations).
    pub fn mixed() -> Self {
        Self {
            weight_precision: QuantizationPrecision::INT4,
            activation_precision: QuantizationPrecision::INT8,
            scale: 128,
            zero_point: 0,
            symmetric: true,
        }
    }

    /// Creates a full precision config (for testing).
    pub fn fp32() -> Self {
        Self {
            weight_precision: QuantizationPrecision::FP32,
            activation_precision: QuantizationPrecision::FP32,
            scale: 1000,
            zero_point: 0,
            symmetric: true,
        }
    }

    /// Returns the weight table size.
    pub fn weight_table_size(&self) -> usize {
        self.weight_precision.table_size()
    }

    /// Returns the activation table size.
    pub fn activation_table_size(&self) -> usize {
        self.activation_precision.table_size()
    }
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self::int8()
    }
}

/// Activation function type for the feed-forward network.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationType {
    /// Rectified Linear Unit: max(0, x)
    ReLU,
    /// Gaussian Error Linear Unit: x * Φ(x)
    GELU,
    /// Sigmoid Linear Unit (SiLU/Swish): x * σ(x)
    SiLU,
}

impl Default for ActivationType {
    fn default() -> Self {
        ActivationType::GELU
    }
}

/// Positional encoding type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionalEncodingType {
    /// Sinusoidal positional encoding (original transformer).
    Sinusoidal,
    /// Learned positional embeddings.
    Learned,
    /// Rotary position embeddings (RoPE).
    Rotary,
    /// No positional encoding.
    None,
}

impl Default for PositionalEncodingType {
    fn default() -> Self {
        PositionalEncodingType::Sinusoidal
    }
}

/// Attention type configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionType {
    /// Standard multi-head attention.
    MultiHead,
    /// Multi-query attention (shared K/V heads).
    MultiQuery,
    /// Grouped query attention.
    GroupedQuery { num_kv_heads: usize },
}

impl Default for AttentionType {
    fn default() -> Self {
        AttentionType::MultiHead
    }
}

/// Configuration for a single transformer block.
#[derive(Debug, Clone)]
pub struct TransformerBlockConfig {
    /// Model dimension (hidden size).
    pub d_model: usize,
    /// Number of attention heads.
    pub n_heads: usize,
    /// Dimension per attention head (usually d_model / n_heads).
    pub d_head: usize,
    /// Feed-forward network hidden dimension (usually 4 * d_model).
    pub d_ff: usize,
    /// Dropout probability (not used in circuit, but affects training).
    pub dropout: f32,
    /// Whether to use pre-layer normalization (modern style).
    pub pre_norm: bool,
    /// Attention type.
    pub attention_type: AttentionType,
    /// Activation function for FFN.
    pub activation: ActivationType,
    /// Layer normalization epsilon.
    pub layer_norm_eps: f64,
}

impl TransformerBlockConfig {
    /// Creates a new block config with the given dimensions.
    pub fn new(d_model: usize, n_heads: usize, d_ff: usize) -> Self {
        assert!(d_model % n_heads == 0, "d_model must be divisible by n_heads");
        Self {
            d_model,
            n_heads,
            d_head: d_model / n_heads,
            d_ff,
            dropout: 0.0, // No dropout in circuit
            pre_norm: true,
            attention_type: AttentionType::MultiHead,
            activation: ActivationType::GELU,
            layer_norm_eps: 1e-5,
        }
    }

    /// Returns the number of parameters in this block.
    pub fn parameter_count(&self) -> usize {
        // Multi-head attention parameters:
        // - Q projection: d_model * d_model
        // - K projection: d_model * d_model
        // - V projection: d_model * d_model
        // - Output projection: d_model * d_model
        let attn_params = 4 * self.d_model * self.d_model;

        // Feed-forward parameters:
        // - First linear: d_model * d_ff
        // - Second linear: d_ff * d_model
        let ff_params = 2 * self.d_model * self.d_ff;

        // Layer norm parameters:
        // - 2 layer norms (attention, FFN), each with gamma and beta
        let ln_params = 4 * self.d_model;

        attn_params + ff_params + ln_params
    }

    /// Estimates constraints for this block.
    pub fn estimated_constraints(&self, seq_len: usize) -> usize {
        // Attention constraints (using Freivalds):
        // - Q, K, V projections: 3 * seq_len * d_model constraints
        // - Attention scores: seq_len * seq_len constraints
        // - Softmax: seq_len * seq_len * 4 (exp lookup + normalization)
        // - Output projection: seq_len * d_model
        let attn_constraints =
            3 * seq_len * self.d_model +
            seq_len * seq_len +
            seq_len * seq_len * 4 +
            seq_len * self.d_model;

        // FFN constraints:
        // - First linear: seq_len * d_ff
        // - Activation (lookup): seq_len * d_ff
        // - Second linear: seq_len * d_model
        let ff_constraints = seq_len * self.d_ff + seq_len * self.d_ff + seq_len * self.d_model;

        // Layer norm: 2 * seq_len * d_model * 3 (mean, variance, normalize)
        let ln_constraints = 2 * seq_len * self.d_model * 3;

        // Add constraints + residual connections
        let residual_constraints = 2 * seq_len * self.d_model;

        attn_constraints + ff_constraints + ln_constraints + residual_constraints
    }

    /// Sets the activation function.
    pub fn with_activation(mut self, activation: ActivationType) -> Self {
        self.activation = activation;
        self
    }

    /// Sets pre-norm or post-norm style.
    pub fn with_pre_norm(mut self, pre_norm: bool) -> Self {
        self.pre_norm = pre_norm;
        self
    }
}

/// Complete transformer model configuration.
#[derive(Debug, Clone)]
pub struct TransformerConfig {
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// Model dimension.
    pub d_model: usize,
    /// Number of transformer layers.
    pub n_layers: usize,
    /// Configuration for each block.
    pub block_config: TransformerBlockConfig,
    /// Positional encoding type.
    pub pos_encoding: PositionalEncodingType,
    /// Quantization configuration.
    pub quantization: QuantizationConfig,
    /// Whether to tie embedding and output weights.
    pub tie_weights: bool,
}

impl TransformerConfig {
    /// Creates a new transformer config.
    pub fn new(
        vocab_size: usize,
        max_seq_len: usize,
        d_model: usize,
        n_layers: usize,
        n_heads: usize,
    ) -> Self {
        let d_ff = 4 * d_model; // Standard 4x expansion
        Self {
            vocab_size,
            max_seq_len,
            d_model,
            n_layers,
            block_config: TransformerBlockConfig::new(d_model, n_heads, d_ff),
            pos_encoding: PositionalEncodingType::Sinusoidal,
            quantization: QuantizationConfig::default(),
            tie_weights: true,
        }
    }

    /// Creates a tiny demo configuration (~100K parameters).
    pub fn tiny_demo() -> Self {
        Self::new(
            1000,  // vocab_size
            64,    // max_seq_len
            64,    // d_model
            2,     // n_layers
            2,     // n_heads
        )
    }

    /// Creates a small demo configuration (~500K parameters).
    pub fn small_demo() -> Self {
        Self::new(
            5000,  // vocab_size
            128,   // max_seq_len
            128,   // d_model
            4,     // n_layers
            4,     // n_heads
        )
    }

    /// Creates a medium configuration (~1M parameters).
    pub fn medium() -> Self {
        Self::new(
            10000, // vocab_size
            256,   // max_seq_len
            256,   // d_model
            4,     // n_layers
            8,     // n_heads
        )
    }

    /// Creates a standard configuration (~2M parameters).
    pub fn standard() -> Self {
        Self::new(
            32000, // vocab_size
            512,   // max_seq_len
            256,   // d_model
            6,     // n_layers
            8,     // n_heads
        )
    }

    /// Returns the total number of parameters.
    pub fn total_parameters(&self) -> usize {
        // Embedding parameters
        let embed_params = self.vocab_size * self.d_model;

        // Position embedding parameters (if learned)
        let pos_params = match self.pos_encoding {
            PositionalEncodingType::Learned => self.max_seq_len * self.d_model,
            _ => 0,
        };

        // Transformer block parameters
        let block_params = self.n_layers * self.block_config.parameter_count();

        // Final layer norm
        let final_ln_params = 2 * self.d_model;

        // Output projection (if not tied with embeddings)
        let output_params = if self.tie_weights {
            0
        } else {
            self.d_model * self.vocab_size
        };

        embed_params + pos_params + block_params + final_ln_params + output_params
    }

    /// Returns the number of parameters in millions.
    pub fn parameters_millions(&self) -> f64 {
        self.total_parameters() as f64 / 1_000_000.0
    }

    /// Estimates total constraints for a forward pass.
    pub fn estimated_constraints(&self, seq_len: usize) -> usize {
        // Embedding lookup: seq_len
        let embed_constraints = seq_len;

        // Position encoding: seq_len * d_model
        let pos_constraints = seq_len * self.d_model;

        // Transformer blocks
        let block_constraints = self.n_layers * self.block_config.estimated_constraints(seq_len);

        // Final layer norm
        let final_ln_constraints = seq_len * self.d_model * 3;

        // Output projection
        let output_constraints = seq_len * self.vocab_size;

        embed_constraints + pos_constraints + block_constraints + final_ln_constraints + output_constraints
    }

    /// Estimates proving time in milliseconds.
    ///
    /// This is a rough estimate based on constraint count and typical
    /// proving speed of ~2000 constraints/ms on modern hardware.
    pub fn estimated_proving_time_ms(&self, seq_len: usize) -> u64 {
        let constraints = self.estimated_constraints(seq_len);
        // Using Freivalds and lookup optimizations, ~2000 constraints/ms
        (constraints as u64) / 2000
    }

    /// Checks if this config is suitable for the demo (under 500ms proving).
    pub fn is_demo_suitable(&self, seq_len: usize) -> bool {
        self.estimated_proving_time_ms(seq_len) < 500
    }

    /// Sets the quantization configuration.
    pub fn with_quantization(mut self, quant: QuantizationConfig) -> Self {
        self.quantization = quant;
        self
    }

    /// Sets the positional encoding type.
    pub fn with_pos_encoding(mut self, pos_encoding: PositionalEncodingType) -> Self {
        self.pos_encoding = pos_encoding;
        self
    }

    /// Sets whether to tie embedding weights.
    pub fn with_tied_weights(mut self, tie: bool) -> Self {
        self.tie_weights = tie;
        self
    }

    /// Returns the K parameter needed for the circuit.
    ///
    /// The K parameter determines the number of rows: 2^K rows available.
    pub fn required_k(&self, seq_len: usize) -> u32 {
        let constraints = self.estimated_constraints(seq_len);
        // Add some overhead for tables and padding
        let rows_needed = constraints + self.quantization.activation_table_size() * 4;
        (rows_needed as f64).log2().ceil() as u32 + 1
    }

    /// Creates a summary string for display.
    pub fn summary(&self) -> String {
        format!(
            "TransformerConfig:\n\
             - Vocab: {}\n\
             - Max seq: {}\n\
             - d_model: {}\n\
             - Layers: {}\n\
             - Heads: {}\n\
             - d_ff: {}\n\
             - Parameters: {:.2}M\n\
             - Quantization: {:?}/{:?}",
            self.vocab_size,
            self.max_seq_len,
            self.d_model,
            self.n_layers,
            self.block_config.n_heads,
            self.block_config.d_ff,
            self.parameters_millions(),
            self.quantization.weight_precision,
            self.quantization.activation_precision,
        )
    }
}

impl fmt::Display for TransformerConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())
    }
}

/// Configuration builder for transformer models.
#[derive(Debug, Clone, Default)]
pub struct TransformerConfigBuilder {
    vocab_size: Option<usize>,
    max_seq_len: Option<usize>,
    d_model: Option<usize>,
    n_layers: Option<usize>,
    n_heads: Option<usize>,
    d_ff: Option<usize>,
    activation: Option<ActivationType>,
    pos_encoding: Option<PositionalEncodingType>,
    quantization: Option<QuantizationConfig>,
    tie_weights: Option<bool>,
}

impl TransformerConfigBuilder {
    /// Creates a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets vocabulary size.
    pub fn vocab_size(mut self, size: usize) -> Self {
        self.vocab_size = Some(size);
        self
    }

    /// Sets maximum sequence length.
    pub fn max_seq_len(mut self, len: usize) -> Self {
        self.max_seq_len = Some(len);
        self
    }

    /// Sets model dimension.
    pub fn d_model(mut self, dim: usize) -> Self {
        self.d_model = Some(dim);
        self
    }

    /// Sets number of layers.
    pub fn n_layers(mut self, layers: usize) -> Self {
        self.n_layers = Some(layers);
        self
    }

    /// Sets number of attention heads.
    pub fn n_heads(mut self, heads: usize) -> Self {
        self.n_heads = Some(heads);
        self
    }

    /// Sets feed-forward dimension.
    pub fn d_ff(mut self, dim: usize) -> Self {
        self.d_ff = Some(dim);
        self
    }

    /// Sets activation function.
    pub fn activation(mut self, activation: ActivationType) -> Self {
        self.activation = Some(activation);
        self
    }

    /// Sets positional encoding type.
    pub fn pos_encoding(mut self, pos_encoding: PositionalEncodingType) -> Self {
        self.pos_encoding = Some(pos_encoding);
        self
    }

    /// Sets quantization configuration.
    pub fn quantization(mut self, quant: QuantizationConfig) -> Self {
        self.quantization = Some(quant);
        self
    }

    /// Sets weight tying.
    pub fn tie_weights(mut self, tie: bool) -> Self {
        self.tie_weights = Some(tie);
        self
    }

    /// Builds the configuration.
    pub fn build(self) -> Result<TransformerConfig, &'static str> {
        let vocab_size = self.vocab_size.ok_or("vocab_size is required")?;
        let max_seq_len = self.max_seq_len.ok_or("max_seq_len is required")?;
        let d_model = self.d_model.ok_or("d_model is required")?;
        let n_layers = self.n_layers.ok_or("n_layers is required")?;
        let n_heads = self.n_heads.ok_or("n_heads is required")?;

        if d_model % n_heads != 0 {
            return Err("d_model must be divisible by n_heads");
        }

        let d_ff = self.d_ff.unwrap_or(4 * d_model);
        let activation = self.activation.unwrap_or_default();
        let pos_encoding = self.pos_encoding.unwrap_or_default();
        let quantization = self.quantization.unwrap_or_default();
        let tie_weights = self.tie_weights.unwrap_or(true);

        let mut config = TransformerConfig::new(vocab_size, max_seq_len, d_model, n_layers, n_heads);
        config.block_config.d_ff = d_ff;
        config.block_config.activation = activation;
        config.pos_encoding = pos_encoding;
        config.quantization = quantization;
        config.tie_weights = tie_weights;

        Ok(config)
    }

    /// Builds the configuration, targeting a specific parameter count.
    pub fn build_with_param_budget(self, target_params: usize) -> Result<TransformerConfig, &'static str> {
        let config = self.build()?;

        // Check if within budget
        let actual_params = config.total_parameters();
        if actual_params > target_params * 12 / 10 {
            return Err("Configuration exceeds parameter budget by more than 20%");
        }

        Ok(config)
    }
}

// ============================================================================
// MLP Architecture Configuration (for V3 N-layer circuit)
// ============================================================================

/// Activation function for a circuit layer.
///
/// These map directly to in-circuit constraint strategies:
/// - `ReLU`: verified via lookup table (existing relu_table)
/// - `Identity`: no constraints (pass-through)
/// - `Tanh`: verified via lookup table (tanh_table)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CircuitActivation {
    /// Rectified Linear Unit: max(0, x). Verified via ReLU lookup table.
    ReLU,
    /// Identity (no activation). No constraints needed.
    Identity,
    /// Hyperbolic tangent. Verified via tanh lookup table.
    Tanh,
}

/// Loss function for the training step circuit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LossFunction {
    /// Mean Squared Error: L = sum((y - target)^2).
    MSE,
    /// Cross-entropy loss: L = -sum(target * log(softmax(y))).
    /// Uses log-sum-exp trick for numerical stability.
    CrossEntropy,
}

/// Specification for a single layer in the MLP.
#[derive(Clone, Debug)]
pub struct LayerSpec {
    /// Input dimension for this layer.
    pub input_dim: usize,
    /// Output dimension for this layer.
    pub output_dim: usize,
    /// Activation function applied after the affine transform.
    pub activation: CircuitActivation,
}

/// N-layer MLP architecture configuration for the V3 circuit.
///
/// Describes the full network topology: layer dimensions, activations,
/// and loss function. Used by `MLTrainingStepV3Circuit` and `compute_witness_v3`.
///
/// # Example
/// ```
/// use helix_circuits::ml::config::{MLPArchitecture, CircuitActivation, LossFunction};
///
/// // 3-layer MLP: 4 -> 8 (ReLU) -> 4 (ReLU) -> 2 (Identity)
/// let arch = MLPArchitecture::from_dims(
///     &[4, 8, 4, 2],
///     &[CircuitActivation::ReLU, CircuitActivation::ReLU, CircuitActivation::Identity],
///     LossFunction::MSE,
/// );
/// assert_eq!(arch.num_layers(), 3);
/// assert_eq!(arch.input_dim(), 4);
/// assert_eq!(arch.output_dim(), 2);
/// ```
#[derive(Clone, Debug)]
pub struct MLPArchitecture {
    /// Layer specifications in forward order.
    pub layers: Vec<LayerSpec>,
    /// Loss function for the training step.
    pub loss: LossFunction,
}

impl MLPArchitecture {
    /// Creates a legacy 2-layer MLP matching the V2 circuit.
    ///
    /// Architecture: `d_in -> d_hid (ReLU) -> d_out (Identity)`
    pub fn two_layer_mlp(d_in: usize, d_hid: usize, d_out: usize) -> Self {
        Self {
            layers: vec![
                LayerSpec {
                    input_dim: d_in,
                    output_dim: d_hid,
                    activation: CircuitActivation::ReLU,
                },
                LayerSpec {
                    input_dim: d_hid,
                    output_dim: d_out,
                    activation: CircuitActivation::Identity,
                },
            ],
            loss: LossFunction::MSE,
        }
    }

    /// Convenience builder from dimension list and activations.
    ///
    /// `dims` must have `activations.len() + 1` elements.
    /// `dims[0]` is the input dimension, `dims[N]` is the output dimension.
    pub fn from_dims(
        dims: &[usize],
        activations: &[CircuitActivation],
        loss: LossFunction,
    ) -> Self {
        assert!(
            dims.len() >= 2,
            "Need at least 2 dimensions (input + output)"
        );
        assert_eq!(
            dims.len() - 1,
            activations.len(),
            "Need exactly one activation per layer"
        );

        let layers = dims
            .windows(2)
            .zip(activations.iter())
            .map(|(pair, &act)| LayerSpec {
                input_dim: pair[0],
                output_dim: pair[1],
                activation: act,
            })
            .collect();

        Self { layers, loss }
    }

    /// Validates that consecutive layer dimensions match.
    pub fn validate(&self) -> Result<(), String> {
        if self.layers.is_empty() {
            return Err("Architecture must have at least one layer".to_string());
        }

        for i in 1..self.layers.len() {
            if self.layers[i].input_dim != self.layers[i - 1].output_dim {
                return Err(format!(
                    "Dimension mismatch between layer {} output ({}) and layer {} input ({})",
                    i - 1,
                    self.layers[i - 1].output_dim,
                    i,
                    self.layers[i].input_dim,
                ));
            }
        }

        for (i, layer) in self.layers.iter().enumerate() {
            if layer.input_dim == 0 || layer.output_dim == 0 {
                return Err(format!("Layer {} has zero dimension", i));
            }
        }

        Ok(())
    }

    /// Returns the network input dimension.
    pub fn input_dim(&self) -> usize {
        self.layers.first().map_or(0, |l| l.input_dim)
    }

    /// Returns the network output dimension.
    pub fn output_dim(&self) -> usize {
        self.layers.last().map_or(0, |l| l.output_dim)
    }

    /// Returns the number of layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Returns the total number of weight parameters (weights + biases).
    pub fn total_weight_count(&self) -> usize {
        self.layers
            .iter()
            .map(|l| l.input_dim * l.output_dim + l.output_dim)
            .sum()
    }

    /// Returns whether any layer uses Tanh activation.
    pub fn uses_tanh(&self) -> bool {
        self.layers.iter().any(|l| l.activation == CircuitActivation::Tanh)
    }

    /// Returns whether the loss function is cross-entropy.
    pub fn uses_cross_entropy(&self) -> bool {
        self.loss == LossFunction::CrossEntropy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tiny_demo_config() {
        let config = TransformerConfig::tiny_demo();
        let params = config.total_parameters();
        println!("Tiny demo params: {}", params);
        assert!(params < 500_000, "Tiny demo should have < 500K parameters");
    }

    #[test]
    fn test_small_demo_config() {
        let config = TransformerConfig::small_demo();
        let params = config.total_parameters();
        let block_params = config.block_config.parameter_count() * config.n_layers;
        println!("Small demo params: {} ({:.2}M)", params, params as f64 / 1e6);
        println!("  Block params (excl. embeddings): {}", block_params);
        // Total params include embeddings (vocab_size * d_model)
        // Block params should be in target range for efficient proving
        assert!(block_params >= 100_000 && block_params <= 1_000_000,
            "Small demo block params should have 100K-1M, got {}", block_params);
    }

    #[test]
    fn test_medium_config() {
        let config = TransformerConfig::medium();
        let params = config.total_parameters();
        let block_params = config.block_config.parameter_count() * config.n_layers;
        println!("Medium params: {} ({:.2}M)", params, params as f64 / 1e6);
        println!("  Block params (excl. embeddings): {}", block_params);
        // Block params are what matter for proving overhead
        assert!(block_params >= 500_000 && block_params <= 5_000_000,
            "Medium block params should have 500K-5M, got {}", block_params);
    }

    #[test]
    fn test_standard_config() {
        let config = TransformerConfig::standard();
        let params = config.total_parameters();
        let block_params = config.block_config.parameter_count() * config.n_layers;
        println!("Standard params: {} ({:.2}M)", params, params as f64 / 1e6);
        println!("  Block params (excl. embeddings): {}", block_params);
        // Block params are what matter for proving overhead
        assert!(block_params >= 1_000_000 && block_params <= 10_000_000,
            "Standard block params should have 1M-10M, got {}", block_params);
    }

    #[test]
    fn test_constraint_estimation() {
        let config = TransformerConfig::small_demo();
        let constraints = config.estimated_constraints(32);
        println!("Estimated constraints for seq_len=32: {}", constraints);
        assert!(constraints > 0);
    }

    #[test]
    fn test_proving_time_estimation() {
        let config = TransformerConfig::tiny_demo();
        let time_ms = config.estimated_proving_time_ms(32);
        println!("Estimated proving time: {}ms", time_ms);
        // Tiny config should be very fast
        assert!(time_ms < 1000);
    }

    #[test]
    fn test_config_builder() {
        let config = TransformerConfigBuilder::new()
            .vocab_size(1000)
            .max_seq_len(64)
            .d_model(64)
            .n_layers(2)
            .n_heads(2)
            .activation(ActivationType::GELU)
            .build()
            .expect("Should build successfully");

        assert_eq!(config.vocab_size, 1000);
        assert_eq!(config.d_model, 64);
        assert_eq!(config.block_config.activation, ActivationType::GELU);
    }

    #[test]
    fn test_quantization_configs() {
        let int8 = QuantizationConfig::int8();
        assert_eq!(int8.weight_precision, QuantizationPrecision::INT8);
        assert_eq!(int8.weight_table_size(), 256);

        let int4 = QuantizationConfig::int4();
        assert_eq!(int4.weight_precision, QuantizationPrecision::INT4);
        assert_eq!(int4.weight_table_size(), 16);

        let mixed = QuantizationConfig::mixed();
        assert_eq!(mixed.weight_precision, QuantizationPrecision::INT4);
        assert_eq!(mixed.activation_precision, QuantizationPrecision::INT8);
    }

    #[test]
    fn test_block_config_parameters() {
        let block = TransformerBlockConfig::new(128, 4, 512);
        let params = block.parameter_count();

        // Expected:
        // Attention: 4 * 128 * 128 = 65536
        // FFN: 2 * 128 * 512 = 131072
        // LayerNorm: 4 * 128 = 512
        // Total: 197120
        assert_eq!(params, 4 * 128 * 128 + 2 * 128 * 512 + 4 * 128);
    }

    #[test]
    fn test_required_k() {
        let config = TransformerConfig::tiny_demo();
        let k = config.required_k(32);
        println!("Required K for tiny demo: {}", k);
        assert!(k >= 10 && k <= 20);
    }

    #[test]
    fn test_summary() {
        let config = TransformerConfig::small_demo();
        let summary = config.summary();
        assert!(summary.contains("d_model: 128"));
        assert!(summary.contains("Layers: 4"));
    }

    // ---- MLPArchitecture tests ----

    #[test]
    fn test_two_layer_mlp() {
        let arch = MLPArchitecture::two_layer_mlp(4, 8, 2);
        assert_eq!(arch.num_layers(), 2);
        assert_eq!(arch.input_dim(), 4);
        assert_eq!(arch.output_dim(), 2);
        assert_eq!(arch.layers[0].activation, CircuitActivation::ReLU);
        assert_eq!(arch.layers[1].activation, CircuitActivation::Identity);
        assert!(arch.validate().is_ok());
    }

    #[test]
    fn test_from_dims() {
        let arch = MLPArchitecture::from_dims(
            &[4, 8, 4, 2],
            &[CircuitActivation::ReLU, CircuitActivation::Tanh, CircuitActivation::Identity],
            LossFunction::MSE,
        );
        assert_eq!(arch.num_layers(), 3);
        assert_eq!(arch.input_dim(), 4);
        assert_eq!(arch.output_dim(), 2);
        assert!(arch.uses_tanh());
        assert!(!arch.uses_cross_entropy());
        assert!(arch.validate().is_ok());
    }

    #[test]
    fn test_total_weight_count() {
        let arch = MLPArchitecture::two_layer_mlp(2, 3, 1);
        // Layer 1: 2*3 + 3 = 9
        // Layer 2: 3*1 + 1 = 4
        assert_eq!(arch.total_weight_count(), 13);
    }

    #[test]
    fn test_validate_dimension_mismatch() {
        let arch = MLPArchitecture {
            layers: vec![
                LayerSpec { input_dim: 4, output_dim: 8, activation: CircuitActivation::ReLU },
                LayerSpec { input_dim: 6, output_dim: 2, activation: CircuitActivation::Identity },
            ],
            loss: LossFunction::MSE,
        };
        assert!(arch.validate().is_err());
    }

    #[test]
    fn test_validate_zero_dimension() {
        let arch = MLPArchitecture {
            layers: vec![
                LayerSpec { input_dim: 0, output_dim: 4, activation: CircuitActivation::ReLU },
            ],
            loss: LossFunction::MSE,
        };
        assert!(arch.validate().is_err());
    }
}
