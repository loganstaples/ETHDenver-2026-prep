//! Model Profiling with Layer-by-Layer Breakdown.
//!
//! Provides detailed profiling of neural network models including:
//! - Memory usage per layer
//! - Compute cost per layer
//! - Parameter counts
//! - Activation sizes
//! - Error bound accumulation
//!
//! # Example
//!
//! ```ignore
//! use helix_avm::memory::{MemoryProfiler, ProfilingConfig};
//!
//! let mut profiler = MemoryProfiler::new(ProfilingConfig::default());
//!
//! // Profile forward pass
//! profiler.begin_forward();
//! for layer in &model.layers {
//!     profiler.begin_layer(&layer.name);
//!     let output = layer.forward(&input)?;
//!     profiler.end_layer(output.len() * 8, layer.param_count() * 8);
//!     input = output;
//! }
//! profiler.end_forward();
//!
//! // Get profile
//! let profile = profiler.get_profile();
//! println!("{}", profile.format_summary());
//! ```

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Configuration for profiling.
#[derive(Debug, Clone)]
pub struct ProfilingConfig {
    /// Track timing information.
    pub track_timing: bool,
    /// Track memory allocations.
    pub track_memory: bool,
    /// Track error bounds.
    pub track_error_bounds: bool,
    /// Track compute operations (FLOPs).
    pub track_compute: bool,
    /// Sample rate (1.0 = always profile, 0.1 = 10% of calls).
    pub sample_rate: f64,
    /// Minimum memory allocation to track (bytes).
    pub min_allocation: usize,
}

impl Default for ProfilingConfig {
    fn default() -> Self {
        Self {
            track_timing: true,
            track_memory: true,
            track_error_bounds: true,
            track_compute: true,
            sample_rate: 1.0,
            min_allocation: 0,
        }
    }
}

impl ProfilingConfig {
    /// Creates a lightweight profiling config.
    pub fn lightweight() -> Self {
        Self {
            track_timing: false,
            track_memory: true,
            track_error_bounds: false,
            track_compute: false,
            sample_rate: 0.1,
            min_allocation: 1024,
        }
    }

    /// Creates a detailed profiling config.
    pub fn detailed() -> Self {
        Self {
            track_timing: true,
            track_memory: true,
            track_error_bounds: true,
            track_compute: true,
            sample_rate: 1.0,
            min_allocation: 0,
        }
    }

    /// Disables all profiling (noop profiler).
    pub fn disabled() -> Self {
        Self {
            track_timing: false,
            track_memory: false,
            track_error_bounds: false,
            track_compute: false,
            sample_rate: 0.0,
            min_allocation: usize::MAX,
        }
    }
}

/// Types of memory events.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryEventType {
    /// Allocation of memory.
    Allocate,
    /// Freeing of memory.
    Free,
    /// Memory peak recorded.
    Peak,
    /// Checkpoint saved.
    Checkpoint,
    /// Recomputation occurred.
    Recompute,
}

/// A single memory event.
#[derive(Debug, Clone)]
pub struct MemoryEvent {
    /// Type of event.
    pub event_type: MemoryEventType,
    /// Layer name associated with event.
    pub layer: String,
    /// Bytes involved.
    pub bytes: usize,
    /// Timestamp of event.
    pub timestamp: Duration,
    /// Category (weights, activations, gradients, etc.).
    pub category: String,
}

/// Profile for a single layer.
#[derive(Debug, Clone)]
pub struct LayerProfile {
    /// Layer name.
    pub name: String,
    /// Layer type (Linear, Attention, MLP, etc.).
    pub layer_type: String,
    /// Number of parameters.
    pub param_count: usize,
    /// Parameter memory (bytes).
    pub param_memory: usize,
    /// Activation memory (bytes).
    pub activation_memory: usize,
    /// Gradient memory (bytes).
    pub gradient_memory: usize,
    /// Forward pass duration.
    pub forward_time: Duration,
    /// Backward pass duration.
    pub backward_time: Duration,
    /// Floating point operations (forward).
    pub forward_flops: usize,
    /// Floating point operations (backward).
    pub backward_flops: usize,
    /// Input shape.
    pub input_shape: Vec<usize>,
    /// Output shape.
    pub output_shape: Vec<usize>,
    /// Maximum error bound after this layer.
    pub max_error_bound: f64,
    /// Error bound increase from this layer.
    pub error_increase: f64,
}

impl LayerProfile {
    /// Creates a new layer profile.
    pub fn new(name: impl Into<String>, layer_type: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            layer_type: layer_type.into(),
            param_count: 0,
            param_memory: 0,
            activation_memory: 0,
            gradient_memory: 0,
            forward_time: Duration::ZERO,
            backward_time: Duration::ZERO,
            forward_flops: 0,
            backward_flops: 0,
            input_shape: Vec::new(),
            output_shape: Vec::new(),
            max_error_bound: 0.0,
            error_increase: 0.0,
        }
    }

    /// Returns total memory usage for this layer.
    pub fn total_memory(&self) -> usize {
        self.param_memory + self.activation_memory + self.gradient_memory
    }

    /// Returns total FLOPs for this layer.
    pub fn total_flops(&self) -> usize {
        self.forward_flops + self.backward_flops
    }

    /// Returns memory per parameter (bytes).
    pub fn memory_per_param(&self) -> f64 {
        if self.param_count == 0 {
            0.0
        } else {
            self.total_memory() as f64 / self.param_count as f64
        }
    }

    /// Returns FLOPs per parameter.
    pub fn flops_per_param(&self) -> f64 {
        if self.param_count == 0 {
            0.0
        } else {
            self.total_flops() as f64 / self.param_count as f64
        }
    }
}

/// Profile for an entire model.
#[derive(Debug, Clone)]
pub struct ModelProfile {
    /// Model name.
    pub name: String,
    /// Per-layer profiles.
    pub layers: Vec<LayerProfile>,
    /// Memory events.
    pub events: Vec<MemoryEvent>,
    /// Total parameters.
    pub total_params: usize,
    /// Total parameter memory.
    pub total_param_memory: usize,
    /// Peak activation memory.
    pub peak_activation_memory: usize,
    /// Peak gradient memory.
    pub peak_gradient_memory: usize,
    /// Total forward time.
    pub total_forward_time: Duration,
    /// Total backward time.
    pub total_backward_time: Duration,
    /// Total forward FLOPs.
    pub total_forward_flops: usize,
    /// Total backward FLOPs.
    pub total_backward_flops: usize,
    /// Final error bound.
    pub final_error_bound: f64,
    /// Number of forward passes profiled.
    pub forward_passes: usize,
    /// Number of backward passes profiled.
    pub backward_passes: usize,
}

impl ModelProfile {
    /// Creates a new model profile.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            layers: Vec::new(),
            events: Vec::new(),
            total_params: 0,
            total_param_memory: 0,
            peak_activation_memory: 0,
            peak_gradient_memory: 0,
            total_forward_time: Duration::ZERO,
            total_backward_time: Duration::ZERO,
            total_forward_flops: 0,
            total_backward_flops: 0,
            final_error_bound: 0.0,
            forward_passes: 0,
            backward_passes: 0,
        }
    }

    /// Returns total memory usage.
    pub fn total_memory(&self) -> usize {
        self.total_param_memory + self.peak_activation_memory + self.peak_gradient_memory
    }

    /// Returns total FLOPs.
    pub fn total_flops(&self) -> usize {
        self.total_forward_flops + self.total_backward_flops
    }

    /// Returns average forward time per layer.
    pub fn avg_forward_time_per_layer(&self) -> Duration {
        if self.layers.is_empty() {
            Duration::ZERO
        } else {
            self.total_forward_time / self.layers.len() as u32
        }
    }

    /// Formats a summary of the profile.
    pub fn format_summary(&self) -> String {
        let mut lines = Vec::new();

        lines.push(format!("Model Profile: {}", self.name));
        lines.push(format!("  Total Parameters: {:>12} ({:.2} M)",
            self.total_params,
            self.total_params as f64 / 1_000_000.0
        ));
        lines.push(format!("  Parameter Memory: {:>12} bytes ({:.2} MB)",
            self.total_param_memory,
            self.total_param_memory as f64 / 1024.0 / 1024.0
        ));
        lines.push(format!("  Peak Activation:  {:>12} bytes ({:.2} MB)",
            self.peak_activation_memory,
            self.peak_activation_memory as f64 / 1024.0 / 1024.0
        ));
        lines.push(format!("  Peak Gradient:    {:>12} bytes ({:.2} MB)",
            self.peak_gradient_memory,
            self.peak_gradient_memory as f64 / 1024.0 / 1024.0
        ));
        lines.push(format!("  Total Memory:     {:>12} bytes ({:.2} MB)",
            self.total_memory(),
            self.total_memory() as f64 / 1024.0 / 1024.0
        ));
        lines.push(format!("  Forward FLOPs:    {:>12} ({:.2} G)",
            self.total_forward_flops,
            self.total_forward_flops as f64 / 1e9
        ));
        lines.push(format!("  Forward Time:     {:>12.3} ms",
            self.total_forward_time.as_secs_f64() * 1000.0
        ));
        lines.push(format!("  Final Error Bound: {:>12.2e}", self.final_error_bound));
        lines.push(format!("  Number of Layers: {:>12}", self.layers.len()));

        lines.join("\n")
    }

    /// Formats a detailed layer-by-layer breakdown.
    pub fn format_layer_breakdown(&self) -> String {
        let mut lines = Vec::new();

        lines.push("Layer-by-Layer Breakdown:".to_string());
        lines.push(format!(
            "{:<30} {:>12} {:>12} {:>12} {:>10} {:>12}",
            "Layer", "Params", "Param MB", "Act MB", "Time ms", "Error"
        ));
        lines.push("-".repeat(100));

        for layer in &self.layers {
            lines.push(format!(
                "{:<30} {:>12} {:>12.2} {:>12.2} {:>10.3} {:>12.2e}",
                truncate_string(&layer.name, 30),
                layer.param_count,
                layer.param_memory as f64 / 1024.0 / 1024.0,
                layer.activation_memory as f64 / 1024.0 / 1024.0,
                layer.forward_time.as_secs_f64() * 1000.0,
                layer.max_error_bound
            ));
        }

        lines.join("\n")
    }

    /// Returns the top N layers by memory usage.
    pub fn top_memory_layers(&self, n: usize) -> Vec<&LayerProfile> {
        let mut layers: Vec<_> = self.layers.iter().collect();
        layers.sort_by(|a, b| b.total_memory().cmp(&a.total_memory()));
        layers.into_iter().take(n).collect()
    }

    /// Returns the top N layers by compute time.
    pub fn top_time_layers(&self, n: usize) -> Vec<&LayerProfile> {
        let mut layers: Vec<_> = self.layers.iter().collect();
        layers.sort_by(|a, b| b.forward_time.cmp(&a.forward_time));
        layers.into_iter().take(n).collect()
    }

    /// Returns layers that contribute most to error.
    pub fn top_error_layers(&self, n: usize) -> Vec<&LayerProfile> {
        let mut layers: Vec<_> = self.layers.iter().collect();
        layers.sort_by(|a, b| b.error_increase.partial_cmp(&a.error_increase).unwrap_or(std::cmp::Ordering::Equal));
        layers.into_iter().take(n).collect()
    }
}

/// Truncates a string to a maximum length.
fn truncate_string(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}

/// Memory profiler for neural network models.
#[derive(Debug)]
pub struct MemoryProfiler {
    /// Profiling configuration.
    config: ProfilingConfig,
    /// Current model profile being built.
    current_profile: ModelProfile,
    /// Current layer being profiled.
    current_layer: Option<LayerProfile>,
    /// Layer start time.
    layer_start_time: Option<Instant>,
    /// Forward pass start time.
    forward_start_time: Option<Instant>,
    /// Backward pass start time.
    backward_start_time: Option<Instant>,
    /// Profiling start time.
    profiling_start: Instant,
    /// Current memory usage by category.
    memory_by_category: HashMap<String, usize>,
    /// Peak memory by category.
    peak_memory_by_category: HashMap<String, usize>,
    /// Current error bound.
    current_error: f64,
    /// Is currently in forward pass.
    in_forward: bool,
    /// Is currently in backward pass.
    in_backward: bool,
}

impl MemoryProfiler {
    /// Creates a new memory profiler.
    pub fn new(config: ProfilingConfig) -> Self {
        Self {
            config,
            current_profile: ModelProfile::new("unnamed"),
            current_layer: None,
            layer_start_time: None,
            forward_start_time: None,
            backward_start_time: None,
            profiling_start: Instant::now(),
            memory_by_category: HashMap::new(),
            peak_memory_by_category: HashMap::new(),
            current_error: 0.0,
            in_forward: false,
            in_backward: false,
        }
    }

    /// Creates a new profiler with default configuration.
    pub fn default_profiler() -> Self {
        Self::new(ProfilingConfig::default())
    }

    /// Sets the model name.
    pub fn set_model_name(&mut self, name: impl Into<String>) {
        self.current_profile.name = name.into();
    }

    /// Begins profiling a forward pass.
    pub fn begin_forward(&mut self) {
        if self.config.track_timing {
            self.forward_start_time = Some(Instant::now());
        }
        self.in_forward = true;
        self.current_profile.forward_passes += 1;
    }

    /// Ends profiling a forward pass.
    pub fn end_forward(&mut self) {
        if let Some(start) = self.forward_start_time.take() {
            self.current_profile.total_forward_time += start.elapsed();
        }
        self.in_forward = false;
    }

    /// Begins profiling a backward pass.
    pub fn begin_backward(&mut self) {
        if self.config.track_timing {
            self.backward_start_time = Some(Instant::now());
        }
        self.in_backward = true;
        self.current_profile.backward_passes += 1;
    }

    /// Ends profiling a backward pass.
    pub fn end_backward(&mut self) {
        if let Some(start) = self.backward_start_time.take() {
            self.current_profile.total_backward_time += start.elapsed();
        }
        self.in_backward = false;
    }

    /// Begins profiling a layer.
    pub fn begin_layer(&mut self, name: &str) {
        self.begin_layer_typed(name, "Unknown");
    }

    /// Begins profiling a layer with a type.
    pub fn begin_layer_typed(&mut self, name: &str, layer_type: &str) {
        self.current_layer = Some(LayerProfile::new(name, layer_type));
        if self.config.track_timing {
            self.layer_start_time = Some(Instant::now());
        }
    }

    /// Records layer input shape.
    pub fn record_input_shape(&mut self, shape: &[usize]) {
        if let Some(ref mut layer) = self.current_layer {
            layer.input_shape = shape.to_vec();
        }
    }

    /// Records layer output shape.
    pub fn record_output_shape(&mut self, shape: &[usize]) {
        if let Some(ref mut layer) = self.current_layer {
            layer.output_shape = shape.to_vec();
        }
    }

    /// Records layer parameters.
    pub fn record_params(&mut self, count: usize, memory: usize) {
        if let Some(ref mut layer) = self.current_layer {
            layer.param_count = count;
            layer.param_memory = memory;
        }
    }

    /// Records activation memory.
    pub fn record_activations(&mut self, memory: usize) {
        if let Some(ref mut layer) = self.current_layer {
            layer.activation_memory = memory;
        }
        self.record_memory_event(MemoryEventType::Allocate, "activation", memory);
    }

    /// Records gradient memory.
    pub fn record_gradients(&mut self, memory: usize) {
        if let Some(ref mut layer) = self.current_layer {
            layer.gradient_memory = memory;
        }
        self.record_memory_event(MemoryEventType::Allocate, "gradient", memory);
    }

    /// Records FLOPs for the current layer.
    pub fn record_flops(&mut self, forward_flops: usize, backward_flops: usize) {
        if let Some(ref mut layer) = self.current_layer {
            layer.forward_flops = forward_flops;
            layer.backward_flops = backward_flops;
        }
    }

    /// Records error bound for the current layer.
    pub fn record_error_bound(&mut self, error_bound: f64) {
        if !self.config.track_error_bounds {
            return;
        }

        let error_increase = error_bound - self.current_error;
        self.current_error = error_bound;

        if let Some(ref mut layer) = self.current_layer {
            layer.max_error_bound = error_bound;
            layer.error_increase = error_increase;
        }
    }

    /// Ends profiling a layer.
    pub fn end_layer(&mut self) {
        if let Some(start) = self.layer_start_time.take() {
            if let Some(ref mut layer) = self.current_layer {
                if self.in_forward {
                    layer.forward_time = start.elapsed();
                } else if self.in_backward {
                    layer.backward_time = start.elapsed();
                }
            }
        }

        if let Some(layer) = self.current_layer.take() {
            // Update totals
            self.current_profile.total_params += layer.param_count;
            self.current_profile.total_param_memory += layer.param_memory;
            self.current_profile.total_forward_flops += layer.forward_flops;
            self.current_profile.total_backward_flops += layer.backward_flops;

            // Update peaks
            if layer.activation_memory > self.current_profile.peak_activation_memory {
                self.current_profile.peak_activation_memory = layer.activation_memory;
            }
            if layer.gradient_memory > self.current_profile.peak_gradient_memory {
                self.current_profile.peak_gradient_memory = layer.gradient_memory;
            }

            self.current_profile.layers.push(layer);
        }
    }

    /// Ends profiling a layer with specific memory values.
    pub fn end_layer_with_memory(&mut self, activation_memory: usize, param_memory: usize) {
        if let Some(ref mut layer) = self.current_layer {
            layer.activation_memory = activation_memory;
            layer.param_memory = param_memory;
        }
        self.end_layer();
    }

    /// Records a memory event.
    fn record_memory_event(&mut self, event_type: MemoryEventType, category: &str, bytes: usize) {
        if !self.config.track_memory || bytes < self.config.min_allocation {
            return;
        }

        let layer_name = self
            .current_layer
            .as_ref()
            .map(|l| l.name.clone())
            .unwrap_or_else(|| "global".to_string());

        let event = MemoryEvent {
            event_type,
            layer: layer_name,
            bytes,
            timestamp: self.profiling_start.elapsed(),
            category: category.to_string(),
        };

        self.current_profile.events.push(event);

        // Update category tracking
        let current = self.memory_by_category.entry(category.to_string()).or_insert(0);
        match event_type {
            MemoryEventType::Allocate => *current += bytes,
            MemoryEventType::Free => *current = current.saturating_sub(bytes),
            _ => {}
        }

        let peak = self.peak_memory_by_category.entry(category.to_string()).or_insert(0);
        if *current > *peak {
            *peak = *current;
        }
    }

    /// Records a memory allocation.
    pub fn record_allocation(&mut self, category: &str, bytes: usize) {
        self.record_memory_event(MemoryEventType::Allocate, category, bytes);
    }

    /// Records a memory free.
    pub fn record_free(&mut self, category: &str, bytes: usize) {
        self.record_memory_event(MemoryEventType::Free, category, bytes);
    }

    /// Returns the current profile.
    pub fn get_profile(&self) -> &ModelProfile {
        &self.current_profile
    }

    /// Finalizes and returns the profile.
    pub fn finalize(mut self) -> ModelProfile {
        self.current_profile.final_error_bound = self.current_error;
        self.current_profile
    }

    /// Resets the profiler for a new model.
    pub fn reset(&mut self) {
        self.current_profile = ModelProfile::new("unnamed");
        self.current_layer = None;
        self.layer_start_time = None;
        self.forward_start_time = None;
        self.backward_start_time = None;
        self.profiling_start = Instant::now();
        self.memory_by_category.clear();
        self.peak_memory_by_category.clear();
        self.current_error = 0.0;
        self.in_forward = false;
        self.in_backward = false;
    }

    /// Returns current memory by category.
    pub fn memory_by_category(&self) -> &HashMap<String, usize> {
        &self.memory_by_category
    }

    /// Returns peak memory by category.
    pub fn peak_memory_by_category(&self) -> &HashMap<String, usize> {
        &self.peak_memory_by_category
    }

    /// Estimates FLOPs for a linear layer.
    pub fn estimate_linear_flops(input_dim: usize, output_dim: usize, batch_size: usize) -> (usize, usize) {
        // Forward: batch_size * output_dim * (2 * input_dim - 1) ≈ 2 * batch_size * output_dim * input_dim
        let forward = 2 * batch_size * output_dim * input_dim;
        // Backward: similar to forward (grad_input + grad_weights)
        let backward = 2 * forward;
        (forward, backward)
    }

    /// Estimates FLOPs for attention.
    pub fn estimate_attention_flops(
        batch_size: usize,
        seq_len: usize,
        d_model: usize,
        num_heads: usize,
    ) -> (usize, usize) {
        let d_k = d_model / num_heads;

        // Q, K, V projections: 3 * (batch * seq * d_model * d_model)
        let projections = 3 * 2 * batch_size * seq_len * d_model * d_model;

        // Attention scores: batch * heads * seq * seq * d_k
        let attention_scores = 2 * batch_size * num_heads * seq_len * seq_len * d_k;

        // Softmax: ~5 ops per element
        let softmax = 5 * batch_size * num_heads * seq_len * seq_len;

        // Attention output: batch * heads * seq * d_k * seq
        let attention_output = 2 * batch_size * num_heads * seq_len * d_k * seq_len;

        // Output projection: batch * seq * d_model * d_model
        let output_proj = 2 * batch_size * seq_len * d_model * d_model;

        let forward = projections + attention_scores + softmax + attention_output + output_proj;
        let backward = 2 * forward; // Approximate

        (forward, backward)
    }

    /// Estimates memory for a transformer layer.
    pub fn estimate_transformer_memory(
        batch_size: usize,
        seq_len: usize,
        d_model: usize,
        d_ff: usize,
        training: bool,
    ) -> usize {
        let bytes_per_element = 4; // f32

        // Attention weights: 4 * d_model * d_model (Q, K, V, O projections)
        let attn_params = 4 * d_model * d_model * bytes_per_element;

        // MLP weights: d_model * d_ff * 2 (up and down projections)
        let mlp_params = 2 * d_model * d_ff * bytes_per_element;

        // Activations (if stored)
        let activations = if training {
            // Attention: Q, K, V, attention scores, attention output
            let attn_activations = batch_size * seq_len * d_model * 3 // Q, K, V
                + batch_size * seq_len * seq_len // attention scores
                + batch_size * seq_len * d_model; // attention output

            // MLP: intermediate
            let mlp_activations = batch_size * seq_len * d_ff;

            (attn_activations + mlp_activations) * bytes_per_element
        } else {
            0
        };

        attn_params + mlp_params + activations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiling_config() {
        let config = ProfilingConfig::default();
        assert!(config.track_memory);
        assert!(config.track_timing);
    }

    #[test]
    fn test_layer_profile() {
        let mut profile = LayerProfile::new("test_layer", "Linear");
        profile.param_count = 1000;
        profile.param_memory = 4000;
        profile.activation_memory = 2000;

        assert_eq!(profile.total_memory(), 6000);
    }

    #[test]
    fn test_model_profile() {
        let mut profile = ModelProfile::new("test_model");
        profile.total_params = 1_000_000;
        profile.total_param_memory = 4_000_000;
        profile.peak_activation_memory = 2_000_000;

        assert_eq!(profile.total_memory(), 6_000_000);

        let summary = profile.format_summary();
        assert!(summary.contains("test_model"));
    }

    #[test]
    fn test_profiler_basic() {
        let mut profiler = MemoryProfiler::new(ProfilingConfig::default());
        profiler.set_model_name("test");

        profiler.begin_forward();
        profiler.begin_layer("layer1");
        profiler.record_params(100, 400);
        profiler.record_activations(200);
        profiler.end_layer();
        profiler.end_forward();

        let profile = profiler.get_profile();
        assert_eq!(profile.layers.len(), 1);
        assert_eq!(profile.total_params, 100);
    }

    #[test]
    fn test_profiler_multiple_layers() {
        let mut profiler = MemoryProfiler::new(ProfilingConfig::default());

        profiler.begin_forward();

        for i in 0..5 {
            profiler.begin_layer_typed(&format!("layer_{}", i), "Linear");
            profiler.record_params(100 * (i + 1), 400 * (i + 1));
            profiler.record_activations(200 * (i + 1));
            profiler.record_flops(1000 * (i + 1), 2000 * (i + 1));
            profiler.end_layer();
        }

        profiler.end_forward();

        let profile = profiler.get_profile();
        assert_eq!(profile.layers.len(), 5);
        assert_eq!(profile.total_params, 100 + 200 + 300 + 400 + 500);
    }

    #[test]
    fn test_flops_estimation() {
        let (forward, backward) = MemoryProfiler::estimate_linear_flops(512, 2048, 32);
        assert!(forward > 0);
        assert!(backward > forward);
    }

    #[test]
    fn test_attention_flops_estimation() {
        let (forward, backward) = MemoryProfiler::estimate_attention_flops(4, 128, 256, 8);
        assert!(forward > 0);
        assert!(backward > 0);
    }

    #[test]
    fn test_transformer_memory_estimation() {
        let memory = MemoryProfiler::estimate_transformer_memory(4, 128, 256, 1024, true);
        assert!(memory > 0);

        // Inference should use less memory
        let inference_memory = MemoryProfiler::estimate_transformer_memory(4, 128, 256, 1024, false);
        assert!(inference_memory < memory);
    }

    #[test]
    fn test_error_bound_tracking() {
        let mut profiler = MemoryProfiler::new(ProfilingConfig::default());

        profiler.begin_forward();

        profiler.begin_layer("layer1");
        profiler.record_error_bound(0.001);
        profiler.end_layer();

        profiler.begin_layer("layer2");
        profiler.record_error_bound(0.003);
        profiler.end_layer();

        profiler.end_forward();

        let profile = profiler.get_profile();
        assert_eq!(profile.layers[0].max_error_bound, 0.001);
        assert_eq!(profile.layers[1].max_error_bound, 0.003);
        assert!((profile.layers[1].error_increase - 0.002).abs() < 1e-10);
    }
}
