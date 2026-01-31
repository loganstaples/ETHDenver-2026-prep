//! Gradient Checkpointing for Memory-Efficient Training.
//!
//! Implements gradient checkpointing (activation recomputation) to trade compute
//! for memory during training. Instead of storing all intermediate activations
//! for the backward pass, we only store activations at checkpoint boundaries
//! and recompute the rest during backpropagation.
//!
//! # Memory Savings
//!
//! For a model with N layers:
//! - Without checkpointing: O(N) activation memory
//! - With sqrt(N) checkpointing: O(sqrt(N)) activation memory
//! - With uniform checkpointing: O(k) activation memory, where k is checkpoint interval
//!
//! # Example
//!
//! ```ignore
//! use helix_avm::memory::{GradientCheckpointer, CheckpointStrategy};
//!
//! // Create checkpointer with sqrt(N) strategy
//! let mut checkpointer = GradientCheckpointer::new(CheckpointStrategy::SqrtN);
//!
//! // During forward pass
//! for (i, layer) in layers.iter().enumerate() {
//!     let output = layer.forward(&input)?;
//!     if checkpointer.should_checkpoint(i) {
//!         checkpointer.save_activation(i, &output);
//!     }
//!     input = output;
//! }
//!
//! // During backward pass
//! for i in (0..layers.len()).rev() {
//!     let activation = checkpointer.get_or_recompute(i, &recompute_fn)?;
//!     // Compute gradients using activation
//! }
//! ```

use helix_core::types::BoundedTensor;
use std::collections::HashMap;
use thiserror::Error;

/// Errors during gradient checkpointing.
#[derive(Error, Debug)]
pub enum CheckpointError {
    #[error("Activation not found for layer {layer}")]
    ActivationNotFound { layer: usize },

    #[error("Recomputation failed for layer {layer}: {reason}")]
    RecomputeFailed { layer: usize, reason: String },

    #[error("Invalid checkpoint configuration: {0}")]
    InvalidConfig(String),

    #[error("Layer index {index} exceeds total layers {total}")]
    LayerOutOfBounds { index: usize, total: usize },
}

/// Strategy for selecting checkpoint locations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointStrategy {
    /// Checkpoint every N layers.
    Uniform(usize),
    /// Checkpoint at sqrt(N) evenly spaced points.
    SqrtN,
    /// Checkpoint at specific layers.
    Custom,
    /// No checkpointing (store all activations).
    None,
    /// Checkpoint every other layer.
    Alternate,
    /// Memory-optimal for transformers (checkpoint attention outputs).
    TransformerOptimal,
}

impl CheckpointStrategy {
    /// Computes checkpoint positions for the given number of layers.
    pub fn compute_checkpoints(&self, num_layers: usize) -> Vec<usize> {
        match self {
            CheckpointStrategy::Uniform(interval) => {
                (0..num_layers).step_by(*interval).collect()
            }
            CheckpointStrategy::SqrtN => {
                let sqrt_n = (num_layers as f64).sqrt().ceil() as usize;
                let interval = (num_layers / sqrt_n).max(1);
                (0..num_layers).step_by(interval).collect()
            }
            CheckpointStrategy::None => {
                // Store all activations
                (0..num_layers).collect()
            }
            CheckpointStrategy::Alternate => {
                (0..num_layers).step_by(2).collect()
            }
            CheckpointStrategy::TransformerOptimal => {
                // For transformers: checkpoint before each attention layer
                // Assuming attention at even layers, MLP at odd layers
                (0..num_layers).filter(|i| i % 2 == 0).collect()
            }
            CheckpointStrategy::Custom => {
                // Custom checkpoints are set manually
                Vec::new()
            }
        }
    }

    /// Estimates memory savings factor compared to no checkpointing.
    pub fn memory_savings_factor(&self, num_layers: usize) -> f64 {
        let checkpoints = self.compute_checkpoints(num_layers).len();
        if checkpoints == 0 {
            1.0
        } else {
            num_layers as f64 / checkpoints as f64
        }
    }
}

/// Defines the boundary where a checkpoint is saved.
#[derive(Debug, Clone)]
pub struct CheckpointBoundary {
    /// Layer index.
    pub layer_idx: usize,
    /// Name of the checkpoint (for debugging).
    pub name: String,
    /// Whether this checkpoint is currently stored.
    pub is_stored: bool,
    /// Memory cost of storing this checkpoint (bytes).
    pub memory_cost: usize,
}

impl CheckpointBoundary {
    /// Creates a new checkpoint boundary.
    pub fn new(layer_idx: usize, name: impl Into<String>) -> Self {
        Self {
            layer_idx,
            name: name.into(),
            is_stored: false,
            memory_cost: 0,
        }
    }
}

/// Schedule for recomputing activations.
#[derive(Debug, Clone)]
pub struct RecomputeSchedule {
    /// Layers that need to be recomputed to get activation at target.
    pub layers_to_recompute: Vec<usize>,
    /// Starting checkpoint for recomputation.
    pub start_checkpoint: usize,
    /// Target layer index.
    pub target_layer: usize,
    /// Estimated compute cost (relative).
    pub compute_cost: usize,
}

impl RecomputeSchedule {
    /// Returns the number of layers to recompute.
    pub fn recompute_count(&self) -> usize {
        self.layers_to_recompute.len()
    }
}

/// Cache for storing activations.
#[derive(Debug, Clone)]
pub struct ActivationCache {
    /// Stored activations by layer index.
    activations: HashMap<usize, BoundedTensor>,
    /// Maximum number of activations to store.
    max_size: usize,
    /// Total memory used.
    memory_used: usize,
    /// Memory limit.
    memory_limit: usize,
    /// Statistics.
    stats: CacheStats,
}

/// Cache statistics.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Number of cache hits.
    pub hits: usize,
    /// Number of cache misses.
    pub misses: usize,
    /// Number of evictions.
    pub evictions: usize,
    /// Number of recomputations.
    pub recomputations: usize,
}

impl ActivationCache {
    /// Creates a new activation cache with unbounded size.
    pub fn new() -> Self {
        Self {
            activations: HashMap::new(),
            max_size: usize::MAX,
            memory_used: 0,
            memory_limit: usize::MAX,
            stats: CacheStats::default(),
        }
    }

    /// Creates a cache with a maximum number of entries.
    pub fn with_max_size(max_size: usize) -> Self {
        Self {
            activations: HashMap::new(),
            max_size,
            memory_used: 0,
            memory_limit: usize::MAX,
            stats: CacheStats::default(),
        }
    }

    /// Creates a cache with a memory limit.
    pub fn with_memory_limit(memory_limit: usize) -> Self {
        Self {
            activations: HashMap::new(),
            max_size: usize::MAX,
            memory_used: 0,
            memory_limit,
            stats: CacheStats::default(),
        }
    }

    /// Stores an activation.
    pub fn store(&mut self, layer: usize, activation: BoundedTensor) {
        let memory = activation.len() * 8; // Assuming f64

        // Evict if necessary
        while self.activations.len() >= self.max_size || self.memory_used + memory > self.memory_limit {
            if let Some((&oldest, _)) = self.activations.iter().next() {
                self.evict(oldest);
            } else {
                break;
            }
        }

        self.memory_used += memory;
        self.activations.insert(layer, activation);
    }

    /// Retrieves an activation if present.
    pub fn get(&mut self, layer: usize) -> Option<&BoundedTensor> {
        if self.activations.contains_key(&layer) {
            self.stats.hits += 1;
            self.activations.get(&layer)
        } else {
            self.stats.misses += 1;
            None
        }
    }

    /// Checks if an activation is cached.
    pub fn contains(&self, layer: usize) -> bool {
        self.activations.contains_key(&layer)
    }

    /// Evicts an activation.
    pub fn evict(&mut self, layer: usize) -> Option<BoundedTensor> {
        if let Some(activation) = self.activations.remove(&layer) {
            self.memory_used = self.memory_used.saturating_sub(activation.len() * 8);
            self.stats.evictions += 1;
            Some(activation)
        } else {
            None
        }
    }

    /// Clears all activations.
    pub fn clear(&mut self) {
        self.activations.clear();
        self.memory_used = 0;
    }

    /// Returns the number of cached activations.
    pub fn len(&self) -> usize {
        self.activations.len()
    }

    /// Returns whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.activations.is_empty()
    }

    /// Returns memory usage.
    pub fn memory_usage(&self) -> usize {
        self.memory_used
    }

    /// Returns cache statistics.
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Returns all cached layer indices.
    pub fn cached_layers(&self) -> Vec<usize> {
        self.activations.keys().copied().collect()
    }
}

impl Default for ActivationCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Gradient checkpointer for memory-efficient training.
#[derive(Debug)]
pub struct GradientCheckpointer {
    /// Checkpoint strategy.
    strategy: CheckpointStrategy,
    /// Total number of layers.
    num_layers: usize,
    /// Checkpoint positions.
    checkpoints: Vec<usize>,
    /// Activation cache.
    cache: ActivationCache,
    /// Custom checkpoint boundaries.
    boundaries: Vec<CheckpointBoundary>,
    /// Statistics.
    stats: CheckpointerStats,
}

/// Checkpointer statistics.
#[derive(Debug, Clone, Default)]
pub struct CheckpointerStats {
    /// Total forward passes through checkpointer.
    pub forward_passes: usize,
    /// Total backward passes.
    pub backward_passes: usize,
    /// Total recomputations during backward.
    pub recomputations: usize,
    /// Layers recomputed.
    pub layers_recomputed: usize,
    /// Peak memory usage (bytes).
    pub peak_memory: usize,
}

impl GradientCheckpointer {
    /// Creates a new checkpointer with the given strategy.
    pub fn new(strategy: CheckpointStrategy) -> Self {
        Self {
            strategy,
            num_layers: 0,
            checkpoints: Vec::new(),
            cache: ActivationCache::new(),
            boundaries: Vec::new(),
            stats: CheckpointerStats::default(),
        }
    }

    /// Creates a checkpointer with a memory limit.
    pub fn with_memory_limit(strategy: CheckpointStrategy, memory_limit: usize) -> Self {
        Self {
            strategy,
            num_layers: 0,
            checkpoints: Vec::new(),
            cache: ActivationCache::with_memory_limit(memory_limit),
            boundaries: Vec::new(),
            stats: CheckpointerStats::default(),
        }
    }

    /// Initializes the checkpointer for a model with N layers.
    pub fn init(&mut self, num_layers: usize) {
        self.num_layers = num_layers;
        self.checkpoints = self.strategy.compute_checkpoints(num_layers);
        self.boundaries.clear();

        for &idx in &self.checkpoints {
            self.boundaries.push(CheckpointBoundary::new(idx, format!("layer_{}", idx)));
        }
    }

    /// Sets custom checkpoint positions.
    pub fn set_custom_checkpoints(&mut self, positions: Vec<usize>) {
        self.strategy = CheckpointStrategy::Custom;
        self.checkpoints = positions.clone();
        self.boundaries = positions
            .into_iter()
            .map(|idx| CheckpointBoundary::new(idx, format!("layer_{}", idx)))
            .collect();
    }

    /// Checks if a layer should be checkpointed.
    pub fn should_checkpoint(&self, layer: usize) -> bool {
        self.checkpoints.contains(&layer)
    }

    /// Saves an activation for a layer.
    pub fn save_activation(&mut self, layer: usize, activation: &BoundedTensor) {
        self.cache.store(layer, activation.clone());

        if let Some(boundary) = self.boundaries.iter_mut().find(|b| b.layer_idx == layer) {
            boundary.is_stored = true;
            boundary.memory_cost = activation.len() * 8;
        }

        let memory = self.cache.memory_usage();
        if memory > self.stats.peak_memory {
            self.stats.peak_memory = memory;
        }
    }

    /// Gets an activation, recomputing if necessary.
    pub fn get_or_recompute<F>(
        &mut self,
        layer: usize,
        recompute_fn: F,
    ) -> Result<BoundedTensor, CheckpointError>
    where
        F: Fn(usize, &BoundedTensor) -> Result<BoundedTensor, String>,
    {
        // Check if activation is cached
        if let Some(activation) = self.cache.get(layer) {
            return Ok(activation.clone());
        }

        // Need to recompute
        let schedule = self.compute_recompute_schedule(layer)?;
        self.stats.recomputations += 1;
        self.stats.layers_recomputed += schedule.recompute_count();

        // Get starting checkpoint
        let start_activation = self
            .cache
            .get(schedule.start_checkpoint)
            .cloned()
            .ok_or(CheckpointError::ActivationNotFound {
                layer: schedule.start_checkpoint,
            })?;

        // Recompute forward from checkpoint to target
        let mut current = start_activation;
        for &recompute_layer in &schedule.layers_to_recompute {
            current = recompute_fn(recompute_layer, &current).map_err(|reason| {
                CheckpointError::RecomputeFailed {
                    layer: recompute_layer,
                    reason,
                }
            })?;
        }

        Ok(current)
    }

    /// Computes the schedule for recomputing an activation.
    fn compute_recompute_schedule(&self, target_layer: usize) -> Result<RecomputeSchedule, CheckpointError> {
        if target_layer >= self.num_layers {
            return Err(CheckpointError::LayerOutOfBounds {
                index: target_layer,
                total: self.num_layers,
            });
        }

        // Find the nearest checkpoint before target
        let start_checkpoint = self
            .checkpoints
            .iter()
            .filter(|&&cp| cp <= target_layer)
            .max()
            .copied()
            .unwrap_or(0);

        // Layers to recompute (from checkpoint to target, exclusive of start)
        let layers_to_recompute: Vec<usize> = ((start_checkpoint + 1)..=target_layer).collect();
        let compute_cost = layers_to_recompute.len();

        Ok(RecomputeSchedule {
            layers_to_recompute,
            start_checkpoint,
            target_layer,
            compute_cost,
        })
    }

    /// Called at the start of a forward pass.
    pub fn begin_forward(&mut self) {
        self.stats.forward_passes += 1;
    }

    /// Called at the start of a backward pass.
    pub fn begin_backward(&mut self) {
        self.stats.backward_passes += 1;
    }

    /// Clears all saved activations.
    pub fn clear(&mut self) {
        self.cache.clear();
        for boundary in &mut self.boundaries {
            boundary.is_stored = false;
            boundary.memory_cost = 0;
        }
    }

    /// Returns the checkpointing strategy.
    pub fn strategy(&self) -> CheckpointStrategy {
        self.strategy
    }

    /// Returns checkpoint positions.
    pub fn checkpoints(&self) -> &[usize] {
        &self.checkpoints
    }

    /// Returns the number of checkpoints.
    pub fn num_checkpoints(&self) -> usize {
        self.checkpoints.len()
    }

    /// Returns current memory usage.
    pub fn memory_usage(&self) -> usize {
        self.cache.memory_usage()
    }

    /// Returns checkpointer statistics.
    pub fn stats(&self) -> &CheckpointerStats {
        &self.stats
    }

    /// Returns estimated memory savings factor.
    pub fn memory_savings(&self) -> f64 {
        if self.num_layers > 0 {
            self.num_layers as f64 / self.checkpoints.len().max(1) as f64
        } else {
            1.0
        }
    }

    /// Returns the number of layers that need to be recomputed on average.
    pub fn avg_recompute_distance(&self) -> f64 {
        if self.checkpoints.len() <= 1 {
            return 0.0;
        }

        let mut total_distance = 0;
        for window in self.checkpoints.windows(2) {
            total_distance += window[1] - window[0];
        }

        total_distance as f64 / (self.checkpoints.len() - 1) as f64
    }
}

/// Helper function to wrap a forward function with checkpointing.
pub fn checkpoint_forward<F, E>(
    checkpointer: &mut GradientCheckpointer,
    layer_idx: usize,
    input: &BoundedTensor,
    forward_fn: F,
) -> Result<BoundedTensor, E>
where
    F: Fn(&BoundedTensor) -> Result<BoundedTensor, E>,
{
    let output = forward_fn(input)?;

    if checkpointer.should_checkpoint(layer_idx) {
        checkpointer.save_activation(layer_idx, &output);
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_strategy_uniform() {
        let strategy = CheckpointStrategy::Uniform(3);
        let checkpoints = strategy.compute_checkpoints(10);
        assert_eq!(checkpoints, vec![0, 3, 6, 9]);
    }

    #[test]
    fn test_checkpoint_strategy_sqrt() {
        let strategy = CheckpointStrategy::SqrtN;
        let checkpoints = strategy.compute_checkpoints(16);
        // sqrt(16) = 4, so we should have ~4 checkpoints
        assert!(checkpoints.len() <= 5);
        assert!(checkpoints.len() >= 3);
    }

    #[test]
    fn test_checkpointer_creation() {
        let mut checkpointer = GradientCheckpointer::new(CheckpointStrategy::SqrtN);
        checkpointer.init(12);

        assert_eq!(checkpointer.num_layers, 12);
        assert!(!checkpointer.checkpoints.is_empty());
        assert!(checkpointer.memory_savings() > 1.0);
    }

    #[test]
    fn test_activation_cache() {
        let mut cache = ActivationCache::with_max_size(3);

        cache.store(0, BoundedTensor::zeros(vec![10, 10]));
        cache.store(1, BoundedTensor::zeros(vec![10, 10]));
        cache.store(2, BoundedTensor::zeros(vec![10, 10]));

        assert_eq!(cache.len(), 3);
        assert!(cache.contains(0));
        assert!(cache.contains(1));

        // Add one more, should evict oldest
        cache.store(3, BoundedTensor::zeros(vec![10, 10]));
        assert_eq!(cache.len(), 3);
    }

    #[test]
    fn test_checkpoint_save_and_retrieve() {
        let mut checkpointer = GradientCheckpointer::new(CheckpointStrategy::Uniform(2));
        checkpointer.init(6);

        // Layer 0 should be a checkpoint
        assert!(checkpointer.should_checkpoint(0));

        let activation = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0], vec![3]);
        checkpointer.save_activation(0, &activation);

        // Retrieve it
        let retrieved = checkpointer.get_or_recompute(0, |_, _| {
            Err("Should not be called".to_string())
        }).unwrap();

        assert_eq!(retrieved.values(), activation.values());
    }

    #[test]
    fn test_recompute_schedule() {
        let mut checkpointer = GradientCheckpointer::new(CheckpointStrategy::Uniform(3));
        checkpointer.init(9);
        // Checkpoints at 0, 3, 6

        let schedule = checkpointer.compute_recompute_schedule(5).unwrap();
        assert_eq!(schedule.start_checkpoint, 3);
        assert_eq!(schedule.layers_to_recompute, vec![4, 5]);
    }

    #[test]
    fn test_memory_limit_cache() {
        let mut cache = ActivationCache::with_memory_limit(1000);

        // Each tensor is 800 bytes (100 elements * 8 bytes)
        cache.store(0, BoundedTensor::zeros(vec![10, 10]));
        assert_eq!(cache.len(), 1);

        // Adding another should evict the first due to memory limit
        cache.store(1, BoundedTensor::zeros(vec![10, 10]));
        assert_eq!(cache.len(), 1);
        assert!(cache.stats().evictions > 0);
    }

    #[test]
    fn test_transformer_optimal_strategy() {
        let strategy = CheckpointStrategy::TransformerOptimal;
        let checkpoints = strategy.compute_checkpoints(12);

        // Should checkpoint at attention layers (even indices)
        for cp in &checkpoints {
            assert_eq!(cp % 2, 0);
        }
    }
}
