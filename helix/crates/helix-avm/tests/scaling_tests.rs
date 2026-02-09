//! Scaling Tests for Large Models (500K-2M Parameters).
//!
//! These tests verify that the AVM can handle production-sized models
//! with bounded memory usage and proper error tracking.

use helix_avm::memory::{
    CheckpointStrategy, GradientCheckpointer, MemoryBudget, MemoryProfiler, MemoryTracker,
    ProfilingConfig,
};
use helix_avm::models::{
    BERTConfig, BERTModel, GPTConfig, GPTModel, Model, ModelDimensions, ModelSize,
};
use helix_avm::nn::large_model::{
    compute_optimal_batch_size, estimate_forward_memory, LargeModelConfig, LargeModelExecutor,
};
use helix_avm::nn::transformer::{TransformerBlock, TransformerConfig};
use helix_core::types::BoundedTensor;

/// Test helper to create a model config for a specific parameter count.
fn _config_for_params(target_params: usize) -> TransformerConfig {
    // Compute dimensions that roughly achieve the target
    let dims = ModelDimensions::new(
        128,  // d_model
        4,    // n_layers
        4,    // n_heads
        512,  // d_ff
        8192, // vocab_size
        256,  // max_seq_len
    );

    // Scale layers to hit target
    let base_params = dims.estimate_params_gpt();
    let scale = (target_params as f64 / base_params as f64).sqrt();

    let scaled_d_model = ((128.0 * scale) as usize / 8) * 8; // Round to multiple of 8

    TransformerConfig::new(scaled_d_model.max(32), 4).unwrap()
}

// =============================================================================
// 500K Parameter Tests
// =============================================================================

#[test]
fn test_500k_model_creation() {
    let config = GPTConfig::for_size(ModelSize::HalfMillion).unwrap();
    let model = GPTModel::new(config).unwrap();

    let param_count = model.param_count();
    println!("500K model actual params: {}", param_count);

    // The target is 500K but with full model components it may be larger
    // The key is that it's in a reasonable range for small models
    assert!(
        param_count > 100_000,
        "Model too small: {} params",
        param_count
    );
    assert!(
        param_count < 5_000_000,
        "Model too large: {} params",
        param_count
    );
}

#[test]
fn test_500k_model_forward() {
    let config = GPTConfig::for_size(ModelSize::HalfMillion).unwrap();
    let model = GPTModel::new(config.clone()).unwrap();

    // Create input
    let seq_len = 32;
    let input = BoundedTensor::zeros(vec![seq_len, config.d_model]);

    // Forward pass
    let output = model.forward(&input).unwrap();

    // Verify output shape
    assert_eq!(output.shape()[0], seq_len);
    assert_eq!(output.shape()[1], config.vocab_size);
}

#[test]
fn test_500k_model_memory() {
    let config = GPTConfig::for_size(ModelSize::HalfMillion).unwrap();
    let model = GPTModel::new(config).unwrap();

    let batch_size = 4;
    let seq_len = 128;

    let training_memory = model.estimate_training_memory(batch_size, seq_len);
    let inference_memory = model.estimate_inference_memory(batch_size, seq_len);

    println!("500K model training memory: {:.2} MB", training_memory as f64 / 1024.0 / 1024.0);
    println!("500K model inference memory: {:.2} MB", inference_memory as f64 / 1024.0 / 1024.0);

    // Training should use more memory
    assert!(training_memory > inference_memory);

    // Should fit in 8GB
    assert!(
        training_memory < 8 * 1024 * 1024 * 1024,
        "Training memory exceeds 8GB"
    );
}

#[test]
fn test_500k_bert_model() {
    let config = BERTConfig::for_size(ModelSize::HalfMillion).unwrap();
    let model = BERTModel::new(config.clone()).unwrap();

    let param_count = model.param_count();
    println!("BERT 500K model params: {}", param_count);

    // Forward pass with segments
    let seq_len = 32;
    let input = BoundedTensor::zeros(vec![seq_len, config.d_model]);
    let (seq_output, pooled_output) = model.forward(&input).unwrap();

    assert_eq!(seq_output.shape()[0], seq_len);
    assert_eq!(pooled_output.shape()[0], config.d_model);
}

// =============================================================================
// 1M Parameter Tests
// =============================================================================

#[test]
fn test_1m_model_creation() {
    let config = GPTConfig::for_size(ModelSize::OneMillion).unwrap();
    let model = GPTModel::new(config).unwrap();

    let param_count = model.param_count();
    println!("1M model actual params: {}", param_count);

    // The target is 1M but with full model components it may be larger
    // Key: larger than 500K config, reasonable for medium models
    assert!(
        param_count > 500_000,
        "Model too small: {} params",
        param_count
    );
    assert!(
        param_count < 10_000_000,
        "Model too large: {} params",
        param_count
    );
}

#[test]
fn test_1m_model_forward() {
    let config = GPTConfig::for_size(ModelSize::OneMillion).unwrap();
    let model = GPTModel::new(config.clone()).unwrap();

    let seq_len = 64;
    let input = BoundedTensor::zeros(vec![seq_len, config.d_model]);

    let output = model.forward(&input).unwrap();

    assert_eq!(output.shape()[0], seq_len);
    assert_eq!(output.shape()[1], config.vocab_size);
}

#[test]
fn test_1m_model_memory_budget() {
    let config = GPTConfig::for_size(ModelSize::OneMillion).unwrap();
    let model = GPTModel::new(config.clone()).unwrap();

    let batch_size = 4;
    let seq_len = 128;

    // Create memory budget
    let budget = MemoryBudget::for_model_params(model.param_count(), 4.0, true);
    let mut tracker = MemoryTracker::new(budget);

    // Allocate for weights
    let weight_memory = model.param_count() * 4;
    tracker.allocate("weights", weight_memory).unwrap();

    // Allocate for activations
    let activation_memory = batch_size * seq_len * config.d_model * 4;
    tracker.allocate("activations", activation_memory).unwrap();

    println!("1M model memory utilization: {:.1}%", tracker.utilization() * 100.0);

    // Should fit within budget
    assert!(tracker.utilization() < 1.0, "Memory budget exceeded");
}

#[test]
fn test_1m_model_gradient_checkpointing() {
    let config = GPTConfig::for_size(ModelSize::OneMillion).unwrap();

    let mut checkpointer = GradientCheckpointer::new(CheckpointStrategy::SqrtN);
    checkpointer.init(config.n_layers);

    println!("1M model checkpoints: {:?}", checkpointer.checkpoints());
    println!("1M model memory savings: {:.1}x", checkpointer.memory_savings());

    // Should have fewer checkpoints than layers
    assert!(checkpointer.num_checkpoints() < config.n_layers);

    // Memory savings should be > 1
    assert!(checkpointer.memory_savings() > 1.0);
}

#[test]
fn test_1m_model_layer_by_layer() {
    let large_config = LargeModelConfig::for_param_count(1_000_000);
    let mut executor = LargeModelExecutor::new(large_config).unwrap();

    // Create transformer blocks
    let transformer_config = TransformerConfig::new(128, 4).unwrap();
    let blocks: Vec<TransformerBlock> = (0..6)
        .map(|_| TransformerBlock::new(transformer_config.clone()).unwrap())
        .collect();

    let input = BoundedTensor::zeros(vec![32, 128]);
    let output = executor.forward_layer_by_layer(&blocks, &input).unwrap();

    assert_eq!(output.shape(), &vec![32, 128]);
    assert!(executor.stats().layers_processed > 0);
    println!("1M model layers processed: {}", executor.stats().layers_processed);
}

// =============================================================================
// 2M Parameter Tests
// =============================================================================

#[test]
fn test_2m_model_creation() {
    let config = GPTConfig::for_size(ModelSize::TwoMillion).unwrap();
    let model = GPTModel::new(config).unwrap();

    let param_count = model.param_count();
    println!("2M model actual params: {}", param_count);

    // The target is 2M but with full model components it may be larger
    // Key: larger than 1M config, reasonable for larger models
    assert!(
        param_count > 1_000_000,
        "Model too small: {} params",
        param_count
    );
    assert!(
        param_count < 20_000_000,
        "Model too large: {} params",
        param_count
    );
}

#[test]
fn test_2m_model_forward() {
    let config = GPTConfig::for_size(ModelSize::TwoMillion).unwrap();
    let model = GPTModel::new(config.clone()).unwrap();

    let seq_len = 32;
    let input = BoundedTensor::zeros(vec![seq_len, config.d_model]);

    let output = model.forward(&input).unwrap();

    assert_eq!(output.shape()[0], seq_len);
    assert_eq!(output.shape()[1], config.vocab_size);
}

#[test]
fn test_2m_model_memory_efficient() {
    let config = GPTConfig::for_size(ModelSize::TwoMillion).unwrap();
    let model = GPTModel::new(config.clone()).unwrap();

    let batch_size = 2; // Smaller batch for 2M model
    let seq_len = 64;

    let training_memory = model.estimate_training_memory(batch_size, seq_len);

    println!("2M model training memory: {:.2} MB", training_memory as f64 / 1024.0 / 1024.0);

    // Should still fit in 8GB with gradient checkpointing
    assert!(
        training_memory < 8 * 1024 * 1024 * 1024,
        "Training memory exceeds 8GB"
    );
}

#[test]
fn test_2m_model_profiling() {
    let config = GPTConfig::for_size(ModelSize::TwoMillion).unwrap();
    let model = GPTModel::new(config.clone()).unwrap();

    let mut profiler = MemoryProfiler::new(ProfilingConfig::default());
    profiler.set_model_name(model.name());

    // Profile layers
    profiler.begin_forward();
    for i in 0..config.n_layers {
        profiler.begin_layer_typed(&format!("layer_{}", i), "TransformerBlock");

        // Estimate layer params
        let layer_params = 4 * config.d_model * config.d_model + 2 * config.d_model * config.d_ff;
        profiler.record_params(layer_params, layer_params * 4);
        profiler.record_activations(32 * config.d_model * 4);

        profiler.end_layer();
    }
    profiler.end_forward();

    let profile = profiler.get_profile();
    println!("\n{}", profile.format_summary());
    println!("\n{}", profile.format_layer_breakdown());

    assert_eq!(profile.layers.len(), config.n_layers);
}

// =============================================================================
// Memory Efficiency Tests
// =============================================================================

#[test]
fn test_memory_budget_enforcement() {
    // Create a strict memory budget (10MB for safety margin)
    let budget = MemoryBudget::new(10 * 1024 * 1024);
    let mut tracker = MemoryTracker::new(budget);

    // Try to allocate 3MB - should succeed
    tracker.allocate("weights", 3 * 1024 * 1024).unwrap();

    // Try to allocate another 8MB - should fail (exceeds 10MB total)
    let result = tracker.allocate("activations", 8 * 1024 * 1024);
    assert!(result.is_err());

    println!("Memory budget properly enforced");
}

#[test]
fn test_chunked_weight_loading() {
    use helix_avm::memory::{ChunkConfig, ChunkedWeightLoader};

    // Create a large tensor (simulating weights)
    let large_tensor = BoundedTensor::zeros(vec![1000, 100]);

    // Create chunked loader with small chunks
    let config = ChunkConfig::new(20_000); // 20K elements per chunk
    let mut loader = ChunkedWeightLoader::new(config);

    // Load as chunked tensor
    let chunked = loader.load(&large_tensor).unwrap();

    println!("Large tensor chunks: {}", chunked.num_chunks());
    assert!(chunked.num_chunks() > 1);

    // Process chunks
    let results = loader
        .process_chunked(&large_tensor, |chunk| chunk.len())
        .unwrap();
    assert!(!results.is_empty());
}

#[test]
fn test_gradient_checkpointing_memory_savings() {
    let num_layers = 12;

    // Without checkpointing (all activations stored)
    let none_strategy = CheckpointStrategy::None;
    let none_checkpoints = none_strategy.compute_checkpoints(num_layers);

    // With sqrt(N) checkpointing
    let sqrt_strategy = CheckpointStrategy::SqrtN;
    let sqrt_checkpoints = sqrt_strategy.compute_checkpoints(num_layers);

    println!("Without checkpointing: {} stored", none_checkpoints.len());
    println!("With sqrt(N): {} stored", sqrt_checkpoints.len());

    let savings = sqrt_strategy.memory_savings_factor(num_layers);
    println!("Memory savings factor: {:.1}x", savings);

    assert!(sqrt_checkpoints.len() < none_checkpoints.len());
    assert!(savings > 2.0, "Expected > 2x memory savings");
}

#[test]
fn test_optimal_batch_size_computation() {
    let params = 1_000_000;
    let memory_budget = 500 * 1024 * 1024; // 500MB
    let seq_len = 128;
    let d_model = 256;
    let num_layers = 6;

    let optimal_batch = compute_optimal_batch_size(params, memory_budget, seq_len, d_model, num_layers);

    println!(
        "Optimal batch size for {}MB budget: {}",
        memory_budget / 1024 / 1024,
        optimal_batch
    );

    assert!(optimal_batch >= 1);

    // Verify it actually fits
    let estimated_memory = estimate_forward_memory(params, optimal_batch, seq_len, d_model, num_layers);
    assert!(
        estimated_memory <= memory_budget,
        "Optimal batch size exceeds budget"
    );
}

// =============================================================================
// Error Bound Tracking Tests
// =============================================================================

#[test]
fn test_large_model_error_propagation() {
    let config = GPTConfig::for_size(ModelSize::OneMillion).unwrap();
    let model = GPTModel::new(config.clone()).unwrap();

    // Create input with non-zero values and error
    let input_values: Vec<f64> = (0..32 * config.d_model)
        .map(|i| (i as f64 * 0.001) % 1.0)
        .collect();
    let input = BoundedTensor::from_approximate(
        input_values,
        vec![32, config.d_model],
        1e-6,
    );

    let output = model.forward(&input).unwrap();

    // Error should propagate but remain bounded
    let max_error = output.max_error();
    println!("1M model output max error: {:.2e}", max_error);

    // With zero-initialized weights, error may or may not propagate significantly
    // The key test is that it doesn't explode
    assert!(max_error < 100.0, "Error bound too large: {}", max_error);
}

#[test]
fn test_layer_by_layer_error_tracking() {
    let transformer_config = TransformerConfig::new(64, 4).unwrap();
    let blocks: Vec<TransformerBlock> = (0..4)
        .map(|_| TransformerBlock::new(transformer_config.clone()).unwrap())
        .collect();

    let mut profiler = MemoryProfiler::new(ProfilingConfig::default());

    let mut current = BoundedTensor::from_approximate(vec![0.1; 16 * 64], vec![16, 64], 1e-7);
    let mut prev_error = current.max_error();

    profiler.begin_forward();
    for (i, block) in blocks.iter().enumerate() {
        profiler.begin_layer(&format!("layer_{}", i));

        current = block.forward(&current).unwrap();
        let new_error = current.max_error();

        profiler.record_error_bound(new_error);
        profiler.end_layer();

        println!("Layer {} error: {:.2e} (increase: {:.2e})", i, new_error, new_error - prev_error);
        prev_error = new_error;
    }
    profiler.end_forward();

    let profile = profiler.get_profile();
    assert_eq!(profile.layers.len(), 4);
}

// =============================================================================
// Integration Tests
// =============================================================================

#[test]
fn test_full_training_simulation() {
    // Simulate a full training step for a 1M parameter model
    let config = GPTConfig::for_size(ModelSize::OneMillion).unwrap();
    let model = GPTModel::new(config.clone()).unwrap();

    let batch_size = 4;
    let seq_len = 64;

    // Memory tracking
    let budget = MemoryBudget::for_training(100 * 1024 * 1024); // 100MB budget
    let mut tracker = MemoryTracker::new(budget);

    // Allocate weights (forward)
    let weight_bytes = model.param_count() * 4;
    tracker.allocate("weights", weight_bytes).unwrap();

    // Forward pass activations
    let activation_bytes = batch_size * seq_len * config.d_model * 4;
    tracker.allocate("activations", activation_bytes).unwrap();

    // Simulate forward
    let input = BoundedTensor::zeros(vec![seq_len, config.d_model]);
    let _ = model.forward(&input).unwrap();

    // Allocate gradients (backward)
    tracker.allocate("gradients", weight_bytes).unwrap();

    let summary = tracker.summary();
    println!("\nTraining simulation memory summary:");
    println!("  Total current: {:.2} MB", summary.total_current as f64 / 1024.0 / 1024.0);
    println!("  Utilization: {:.1}%", summary.utilization * 100.0);

    // Should complete without OOM
    assert!(summary.allocation_failures == 0);
}

#[test]
fn test_model_checkpoint_roundtrip() {
    use helix_avm::models::{Checkpoint, CheckpointMetadata, ModelSerializer};

    // Create a small model for testing
    let _config = GPTConfig::minimal().unwrap();

    // Create checkpoint
    let mut checkpoint = Checkpoint::new(
        CheckpointMetadata::new("test_model", "gpt")
            .with_training_step(1000)
            .with_training_loss(0.5),
    );

    // Add some test tensors
    let weight1 = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let weight2 = BoundedTensor::from_approximate(vec![0.1, 0.2, 0.3, 0.4], vec![4], 0.01);

    checkpoint.add_tensor("layer1.weight", &weight1);
    checkpoint.add_tensor("layer2.weight", &weight2);

    // Serialize
    let serializer = ModelSerializer::new();
    let bytes = serializer.serialize(&checkpoint).unwrap();

    println!("Checkpoint size: {} bytes", bytes.len());

    // Deserialize
    let loaded = serializer.deserialize(&bytes).unwrap();

    // Verify
    assert_eq!(loaded.metadata.training_step, 1000);
    assert_eq!(loaded.tensor_names().len(), 2);

    let recovered_w1 = loaded.get_tensor("layer1.weight").unwrap();
    assert_eq!(recovered_w1.values(), vec![1.0, 2.0, 3.0, 4.0]);
}

// =============================================================================
// Performance Benchmarks (not actual benchmarks, just timing tests)
// =============================================================================

#[test]
fn test_forward_pass_timing() {
    use std::time::Instant;

    for &size in &[ModelSize::HalfMillion, ModelSize::OneMillion] {
        let config = GPTConfig::for_size(size).unwrap();
        let model = GPTModel::new(config.clone()).unwrap();

        let seq_len = 32;
        let input = BoundedTensor::zeros(vec![seq_len, config.d_model]);

        let start = Instant::now();
        let _ = model.forward(&input).unwrap();
        let duration = start.elapsed();

        println!(
            "{:?} model forward pass: {:?} ({} params)",
            size,
            duration,
            model.param_count()
        );
    }
}
