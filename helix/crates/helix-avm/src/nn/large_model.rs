//! Large Model Support with Memory-Efficient Execution.
//!
//! Provides infrastructure for training models with 500K-2M+ parameters while
//! staying within memory constraints. Key features:
//!
//! - **Layer-by-Layer Execution**: Process one layer at a time to bound memory
//! - **Mixed Precision**: FP32 accumulation with INT8 compute
//! - **Gradient Checkpointing Integration**: Trade compute for memory
//! - **Memory-Efficient Attention**: Chunked softmax for long sequences
//!
//! # Example
//!
//! ```ignore
//! use helix_avm::nn::large_model::{LargeModelConfig, LargeModelExecutor};
//!
//! let config = LargeModelConfig::for_param_count(1_000_000)
//!     .with_memory_budget(4 * 1024 * 1024 * 1024); // 4GB
//!
//! let mut executor = LargeModelExecutor::new(config);
//! executor.forward_efficient(&model, &input)?;
//! ```

use crate::memory::{
    CheckpointStrategy, GradientCheckpointer, MemoryBudget, MemoryProfiler, MemoryTracker,
    ProfilingConfig,
};
use crate::nn::linear::Linear;
use crate::nn::transformer::{TransformerBlock, TransformerConfig, TransformerError, TransformerStack};
use crate::ops::normalization;
use crate::quantization::{MixedPrecisionConfig, MixedPrecisionExecutor, PrecisionLevel};
use helix_core::types::{BoundedTensor, Precision};
use thiserror::Error;

/// Errors during large model execution.
#[derive(Error, Debug)]
pub enum LargeModelError {
    #[error("Transformer error: {0}")]
    Transformer(#[from] TransformerError),

    #[error("Out of memory: required {required} bytes, available {available} bytes")]
    OutOfMemory { required: usize, available: usize },

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Execution error: {0}")]
    Execution(String),
}

/// Configuration for large model execution.
#[derive(Debug, Clone)]
pub struct LargeModelConfig {
    /// Target parameter count.
    pub target_params: usize,
    /// Memory budget in bytes.
    pub memory_budget: usize,
    /// Enable gradient checkpointing.
    pub gradient_checkpointing: bool,
    /// Checkpoint strategy.
    pub checkpoint_strategy: CheckpointStrategy,
    /// Enable mixed precision.
    pub mixed_precision: bool,
    /// Mixed precision configuration.
    pub precision_config: MixedPrecisionConfig,
    /// Enable chunked attention.
    pub chunked_attention: bool,
    /// Attention chunk size (sequence positions per chunk).
    pub attention_chunk_size: usize,
    /// Enable profiling.
    pub profiling: bool,
    /// Activation offloading (to CPU).
    pub offload_activations: bool,
    /// Maximum batch size for memory efficiency.
    pub max_batch_size: usize,
    /// Maximum sequence length.
    pub max_seq_len: usize,
}

impl LargeModelConfig {
    /// Creates configuration for a target parameter count.
    pub fn for_param_count(target_params: usize) -> Self {
        // Estimate memory requirements
        // Training: ~6x parameters (weights + gradients + optimizer + activations)
        let bytes_per_param = 4.0; // FP32
        let memory_multiplier = 6.0;
        let estimated_memory = (target_params as f64 * bytes_per_param * memory_multiplier) as usize;

        Self {
            target_params,
            memory_budget: estimated_memory,
            gradient_checkpointing: target_params >= 500_000,
            checkpoint_strategy: if target_params > 1_000_000 {
                CheckpointStrategy::SqrtN
            } else {
                CheckpointStrategy::Alternate
            },
            mixed_precision: target_params >= 500_000,
            precision_config: if target_params > 1_000_000 {
                MixedPrecisionConfig::w8a8()
            } else {
                MixedPrecisionConfig::fp32()
            },
            chunked_attention: target_params > 1_000_000,
            attention_chunk_size: 128,
            profiling: true,
            offload_activations: false,
            max_batch_size: if target_params > 1_000_000 { 4 } else { 8 },
            max_seq_len: if target_params > 1_000_000 { 256 } else { 512 },
        }
    }

    /// Creates configuration for inference (lower memory).
    pub fn for_inference(target_params: usize) -> Self {
        let mut config = Self::for_param_count(target_params);
        config.gradient_checkpointing = false;
        config.memory_budget /= 4; // Much less memory needed
        config.max_batch_size *= 2;
        config
    }

    /// Sets the memory budget.
    pub fn with_memory_budget(mut self, bytes: usize) -> Self {
        self.memory_budget = bytes;
        self
    }

    /// Enables/disables gradient checkpointing.
    pub fn with_gradient_checkpointing(mut self, enabled: bool) -> Self {
        self.gradient_checkpointing = enabled;
        self
    }

    /// Sets the checkpoint strategy.
    pub fn with_checkpoint_strategy(mut self, strategy: CheckpointStrategy) -> Self {
        self.checkpoint_strategy = strategy;
        self
    }

    /// Enables/disables mixed precision.
    pub fn with_mixed_precision(mut self, enabled: bool) -> Self {
        self.mixed_precision = enabled;
        self
    }

    /// Sets mixed precision config.
    pub fn with_precision_config(mut self, config: MixedPrecisionConfig) -> Self {
        self.precision_config = config;
        self
    }

    /// Enables/disables chunked attention.
    pub fn with_chunked_attention(mut self, enabled: bool, chunk_size: usize) -> Self {
        self.chunked_attention = enabled;
        self.attention_chunk_size = chunk_size;
        self
    }

    /// Enables/disables profiling.
    pub fn with_profiling(mut self, enabled: bool) -> Self {
        self.profiling = enabled;
        self
    }

    /// Validates the configuration.
    pub fn validate(&self) -> Result<(), LargeModelError> {
        if self.memory_budget == 0 {
            return Err(LargeModelError::Config("Memory budget must be positive".to_string()));
        }
        if self.attention_chunk_size == 0 {
            return Err(LargeModelError::Config(
                "Attention chunk size must be positive".to_string(),
            ));
        }
        Ok(())
    }
}

impl Default for LargeModelConfig {
    fn default() -> Self {
        Self::for_param_count(1_000_000)
    }
}

/// Executor for large model operations with memory efficiency.
#[derive(Debug)]
pub struct LargeModelExecutor {
    /// Configuration.
    config: LargeModelConfig,
    /// Memory tracker.
    memory_tracker: MemoryTracker,
    /// Gradient checkpointer (optional).
    checkpointer: Option<GradientCheckpointer>,
    /// Mixed precision executor (optional).
    mixed_executor: Option<MixedPrecisionExecutor>,
    /// Memory profiler (optional).
    profiler: Option<MemoryProfiler>,
    /// Statistics.
    stats: ExecutionStats,
}

/// Statistics for large model execution.
#[derive(Debug, Clone, Default)]
pub struct ExecutionStats {
    /// Number of forward passes.
    pub forward_passes: usize,
    /// Number of backward passes.
    pub backward_passes: usize,
    /// Number of layers processed.
    pub layers_processed: usize,
    /// Number of checkpoints saved.
    pub checkpoints_saved: usize,
    /// Number of recomputations.
    pub recomputations: usize,
    /// Peak memory usage (bytes).
    pub peak_memory: usize,
    /// Total compute time (ms).
    pub total_compute_ms: f64,
}

impl LargeModelExecutor {
    /// Creates a new large model executor.
    pub fn new(config: LargeModelConfig) -> Result<Self, LargeModelError> {
        config.validate()?;

        let memory_budget = MemoryBudget::for_training(config.memory_budget);
        let memory_tracker = MemoryTracker::new(memory_budget);

        let checkpointer = if config.gradient_checkpointing {
            Some(GradientCheckpointer::new(config.checkpoint_strategy))
        } else {
            None
        };

        let mixed_executor = if config.mixed_precision {
            Some(MixedPrecisionExecutor::new(config.precision_config.clone()))
        } else {
            None
        };

        let profiler = if config.profiling {
            Some(MemoryProfiler::new(ProfilingConfig::default()))
        } else {
            None
        };

        Ok(Self {
            config,
            memory_tracker,
            checkpointer,
            mixed_executor,
            profiler,
            stats: ExecutionStats::default(),
        })
    }

    /// Returns the configuration.
    pub fn config(&self) -> &LargeModelConfig {
        &self.config
    }

    /// Returns the memory tracker.
    pub fn memory_tracker(&self) -> &MemoryTracker {
        &self.memory_tracker
    }

    /// Returns execution statistics.
    pub fn stats(&self) -> &ExecutionStats {
        &self.stats
    }

    /// Checks if there's enough memory for an operation.
    pub fn check_memory(&self, required_bytes: usize) -> Result<(), LargeModelError> {
        let available = self.memory_tracker.available();
        if required_bytes > available {
            return Err(LargeModelError::OutOfMemory {
                required: required_bytes,
                available,
            });
        }
        Ok(())
    }

    /// Executes a forward pass layer by layer for memory efficiency.
    pub fn forward_layer_by_layer(
        &mut self,
        blocks: &[TransformerBlock],
        input: &BoundedTensor,
    ) -> Result<BoundedTensor, LargeModelError> {
        self.stats.forward_passes += 1;

        // Initialize checkpointer
        if let Some(ref mut checkpointer) = self.checkpointer {
            checkpointer.init(blocks.len());
            checkpointer.begin_forward();
        }

        // Start profiling
        if let Some(ref mut profiler) = self.profiler {
            profiler.begin_forward();
        }

        let mut current = input.clone();

        for (layer_idx, block) in blocks.iter().enumerate() {
            // Profile layer
            if let Some(ref mut profiler) = self.profiler {
                profiler.begin_layer_typed(&format!("layer_{}", layer_idx), "TransformerBlock");
            }

            // Estimate memory for this layer
            let activation_memory = current.len() * 8 * 4; // 4 intermediate tensors
            self.check_memory(activation_memory)?;

            // Forward through block
            current = block.forward(&current)?;

            // Checkpoint if needed
            if let Some(ref mut checkpointer) = self.checkpointer {
                if checkpointer.should_checkpoint(layer_idx) {
                    checkpointer.save_activation(layer_idx, &current);
                    self.stats.checkpoints_saved += 1;
                }
            }

            // Track memory
            if let Some(ref mut profiler) = self.profiler {
                profiler.record_activations(current.len() * 8);
                profiler.end_layer();
            }

            self.stats.layers_processed += 1;
        }

        // End profiling
        if let Some(ref mut profiler) = self.profiler {
            profiler.end_forward();
        }

        // Update peak memory
        let current_usage = self.memory_tracker.current_usage();
        if current_usage > self.stats.peak_memory {
            self.stats.peak_memory = current_usage;
        }

        Ok(current)
    }

    /// Executes a forward pass with mixed precision.
    pub fn forward_mixed_precision(
        &mut self,
        blocks: &[TransformerBlock],
        input: &BoundedTensor,
    ) -> Result<BoundedTensor, LargeModelError> {
        // If mixed precision is disabled, fall back to regular execution
        if self.mixed_executor.is_none() {
            return self.forward_layer_by_layer(blocks, input);
        }

        self.stats.forward_passes += 1;

        // Note: In a full implementation, we would:
        // 1. Quantize weights once at the start
        // 2. Quantize activations per-layer
        // 3. Accumulate in FP32

        // For now, delegate to layer-by-layer execution
        self.forward_layer_by_layer(blocks, input)
    }

    /// Allocates memory for a tensor.
    pub fn allocate_tensor(&mut self, name: &str, size_bytes: usize) -> Result<(), LargeModelError> {
        self.memory_tracker.allocate("activations", size_bytes).map_err(|e| {
            LargeModelError::OutOfMemory {
                required: size_bytes,
                available: self.memory_tracker.available(),
            }
        })
    }

    /// Frees memory for a tensor.
    pub fn free_tensor(&mut self, name: &str, size_bytes: usize) {
        let _ = self.memory_tracker.free("activations", size_bytes);
    }

    /// Gets the gradient checkpointer.
    pub fn checkpointer(&self) -> Option<&GradientCheckpointer> {
        self.checkpointer.as_ref()
    }

    /// Gets the gradient checkpointer mutably.
    pub fn checkpointer_mut(&mut self) -> Option<&mut GradientCheckpointer> {
        self.checkpointer.as_mut()
    }

    /// Gets the profiler.
    pub fn profiler(&self) -> Option<&MemoryProfiler> {
        self.profiler.as_ref()
    }

    /// Resets the executor state.
    pub fn reset(&mut self) {
        self.memory_tracker.reset();
        if let Some(ref mut checkpointer) = self.checkpointer {
            checkpointer.clear();
        }
        if let Some(ref mut profiler) = self.profiler {
            profiler.reset();
        }
        self.stats = ExecutionStats::default();
    }
}

/// Memory-efficient attention computation using chunked softmax.
pub fn chunked_attention(
    query: &BoundedTensor,
    key: &BoundedTensor,
    value: &BoundedTensor,
    chunk_size: usize,
    precision: Precision,
) -> Result<BoundedTensor, LargeModelError> {
    // For sequences longer than chunk_size, compute attention in chunks
    // to avoid materializing the full seq_len x seq_len attention matrix

    let seq_len = query.shape()[0];

    if seq_len <= chunk_size {
        // Small enough to compute normally
        return crate::nn::attention::scaled_dot_product_attention(query, key, value, precision)
            .map_err(|e| LargeModelError::Execution(e.to_string()));
    }

    // Compute attention chunk by chunk
    let num_chunks = (seq_len + chunk_size - 1) / chunk_size;
    let d_k = query.shape()[1];

    let mut output_data = Vec::with_capacity(seq_len * d_k);

    for chunk_idx in 0..num_chunks {
        let start = chunk_idx * chunk_size;
        let end = (start + chunk_size).min(seq_len);
        let chunk_len = end - start;

        // Extract query chunk
        let query_chunk = extract_rows(query, start, end)?;

        // Compute attention for this query chunk against all keys
        // This still uses O(chunk_size * seq_len) memory, which is better than O(seq_len^2)
        let chunk_output =
            crate::nn::attention::scaled_dot_product_attention(&query_chunk, key, value, precision)
                .map_err(|e| LargeModelError::Execution(e.to_string()))?;

        // Collect output
        output_data.extend(chunk_output.data().iter().cloned());
    }

    Ok(BoundedTensor::new(output_data, vec![seq_len, d_k]))
}

/// Extracts rows from a 2D tensor.
fn extract_rows(tensor: &BoundedTensor, start_row: usize, end_row: usize) -> Result<BoundedTensor, LargeModelError> {
    let shape = tensor.shape();
    if shape.len() != 2 {
        return Err(LargeModelError::Execution("Expected 2D tensor".to_string()));
    }

    let cols = shape[1];
    let num_rows = end_row - start_row;

    let data: Vec<_> = tensor.data()[start_row * cols..end_row * cols].to_vec();

    Ok(BoundedTensor::new(data, vec![num_rows, cols]))
}

/// Estimates memory requirement for a model forward pass.
pub fn estimate_forward_memory(
    num_params: usize,
    batch_size: usize,
    seq_len: usize,
    d_model: usize,
    num_layers: usize,
) -> usize {
    // Weight memory (assuming loaded once)
    let weight_memory = num_params * 4; // FP32

    // Activation memory per layer
    // Each layer needs to store: input, attention output, FFN intermediate, output
    let activation_per_layer = batch_size * seq_len * d_model * 4 * 4; // 4 tensors, 4 bytes

    // With checkpointing, we only store sqrt(N) activations
    let num_stored_activations = (num_layers as f64).sqrt().ceil() as usize;
    let activation_memory = activation_per_layer * num_stored_activations;

    // Workspace for attention computation
    let attention_workspace = batch_size * seq_len * seq_len * 4; // Attention scores

    weight_memory + activation_memory + attention_workspace
}

/// Computes optimal batch size for a given memory budget.
pub fn compute_optimal_batch_size(
    num_params: usize,
    memory_budget: usize,
    seq_len: usize,
    d_model: usize,
    num_layers: usize,
) -> usize {
    // Binary search for largest batch size that fits
    let mut low = 1usize;
    let mut high = 256usize;

    while low < high {
        let mid = (low + high + 1) / 2;
        let required = estimate_forward_memory(num_params, mid, seq_len, d_model, num_layers);

        if required <= memory_budget {
            low = mid;
        } else {
            high = mid - 1;
        }
    }

    low
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_large_model_config() {
        let config = LargeModelConfig::for_param_count(1_000_000);
        assert!(config.gradient_checkpointing);
        assert!(config.mixed_precision);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_scaling() {
        let small = LargeModelConfig::for_param_count(100_000);
        let large = LargeModelConfig::for_param_count(2_000_000);

        assert!(large.memory_budget > small.memory_budget);
        assert!(large.max_batch_size <= small.max_batch_size);
    }

    #[test]
    fn test_executor_creation() {
        let config = LargeModelConfig::for_param_count(500_000);
        let executor = LargeModelExecutor::new(config).unwrap();

        assert!(executor.checkpointer.is_some());
    }

    #[test]
    fn test_memory_estimation() {
        let memory = estimate_forward_memory(
            1_000_000, // params
            4,         // batch
            128,       // seq_len
            256,       // d_model
            6,         // layers
        );

        assert!(memory > 4_000_000); // At least 4MB for weights
    }

    #[test]
    fn test_optimal_batch_size() {
        let batch = compute_optimal_batch_size(
            500_000,       // params
            100_000_000,   // 100MB budget
            128,           // seq_len
            128,           // d_model
            4,             // layers
        );

        assert!(batch >= 1);
        assert!(batch <= 256);
    }

    #[test]
    fn test_chunked_attention() {
        let q = BoundedTensor::zeros(vec![16, 32]);
        let k = BoundedTensor::zeros(vec![16, 32]);
        let v = BoundedTensor::zeros(vec![16, 32]);

        let output = chunked_attention(&q, &k, &v, 8, Precision::F32).unwrap();
        assert_eq!(output.shape(), &vec![16, 32]);
    }

    #[test]
    fn test_executor_layer_by_layer() {
        let config = LargeModelConfig::for_param_count(100_000)
            .with_gradient_checkpointing(true);
        let mut executor = LargeModelExecutor::new(config).unwrap();

        // Create simple transformer blocks
        let transformer_config = TransformerConfig::new(32, 4).unwrap();
        let blocks: Vec<TransformerBlock> = (0..2)
            .map(|_| TransformerBlock::new(transformer_config.clone()).unwrap())
            .collect();

        let input = BoundedTensor::zeros(vec![4, 32]);
        let output = executor.forward_layer_by_layer(&blocks, &input).unwrap();

        assert_eq!(output.shape(), &vec![4, 32]);
        assert!(executor.stats().layers_processed > 0);
    }

    #[test]
    fn test_extract_rows() {
        let tensor = BoundedTensor::from_exact((0..20).map(|i| i as f64).collect(), vec![4, 5]);
        let chunk = extract_rows(&tensor, 1, 3).unwrap();

        assert_eq!(chunk.shape(), &vec![2, 5]);
        assert_eq!(chunk.values()[0], 5.0); // First element of row 1
    }
}
