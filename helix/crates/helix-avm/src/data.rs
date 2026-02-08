//! Data Pipeline for training with batching, shuffling, and augmentation.
//!
//! Wraps helix-core's `DataLoader` to produce `BoundedTensor` batches
//! suitable for the AVM training loop.
//!
//! # Example
//!
//! ```ignore
//! use helix_avm::data::{DataPipeline, PipelineConfig, Augmentation};
//! use helix_core::data::InMemoryDataLoader;
//!
//! let loader = InMemoryDataLoader::from_flat(inputs, labels, 32, 784);
//! let config = PipelineConfig::new(32)
//!     .with_shuffle(true, Some(42))
//!     .with_augmentation(Augmentation::GaussianNoise { std: 0.01 })
//!     .with_normalize(0.0, 1.0);
//!
//! let mut pipeline = DataPipeline::new(Box::new(loader), config);
//! for batch in pipeline.iter() {
//!     let (inputs, targets) = batch;
//!     // ... train
//! }
//! ```

use helix_core::data::DataLoader;
use helix_core::types::{BoundedTensor, BoundedValue, ErrorMargin};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Data augmentation operations applied to input tensors.
#[derive(Debug, Clone)]
pub enum Augmentation {
    /// Adds Gaussian noise with the given standard deviation.
    GaussianNoise { std: f64 },
    /// Randomly scales values by a factor in [min, max].
    RandomScale { min: f64, max: f64 },
    /// Normalizes to zero mean and unit variance using provided statistics.
    Normalize { mean: f64, std: f64 },
    /// Randomly zeroes elements with given probability (dropout-style).
    RandomDropout { probability: f64 },
}

/// Configuration for the data pipeline.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Batch size for training.
    pub batch_size: usize,
    /// Whether to shuffle batch order each epoch.
    pub shuffle: bool,
    /// Seed for reproducible shuffling.
    pub seed: Option<u64>,
    /// Data augmentations to apply to inputs (not targets).
    pub augmentations: Vec<Augmentation>,
    /// Whether to drop the last incomplete batch.
    pub drop_last: bool,
    /// Error margin to apply to input data (quantization noise model).
    pub input_error: f64,
}

impl PipelineConfig {
    /// Creates a new pipeline configuration.
    pub fn new(batch_size: usize) -> Self {
        Self {
            batch_size,
            shuffle: false,
            seed: None,
            augmentations: Vec::new(),
            drop_last: false,
            input_error: 0.0,
        }
    }

    /// Enables shuffling with an optional seed.
    pub fn with_shuffle(mut self, shuffle: bool, seed: Option<u64>) -> Self {
        self.shuffle = shuffle;
        self.seed = seed;
        self
    }

    /// Adds a data augmentation.
    pub fn with_augmentation(mut self, augmentation: Augmentation) -> Self {
        self.augmentations.push(augmentation);
        self
    }

    /// Sets whether to drop the last batch if it's incomplete.
    pub fn with_drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
        self
    }

    /// Adds a normalization augmentation.
    pub fn with_normalize(self, mean: f64, std: f64) -> Self {
        self.with_augmentation(Augmentation::Normalize { mean, std })
    }

    /// Sets the input error margin for BoundedTensor creation.
    pub fn with_input_error(mut self, error: f64) -> Self {
        self.input_error = error;
        self
    }
}

/// A training data batch as BoundedTensors.
#[derive(Debug, Clone)]
pub struct TrainingBatch {
    /// Input features as a BoundedTensor with shape [batch_size, feature_dim].
    pub inputs: BoundedTensor,
    /// Target labels as a BoundedTensor with shape [batch_size, label_dim].
    pub targets: BoundedTensor,
    /// Batch index within the epoch.
    pub batch_idx: usize,
}

/// Data pipeline that wraps a `DataLoader` and produces `BoundedTensor` batches.
pub struct DataPipeline {
    loader: Box<dyn DataLoader>,
    config: PipelineConfig,
    rng: StdRng,
    /// Shuffled batch indices for the current epoch.
    indices: Vec<u64>,
    /// Current position in the indices list.
    position: usize,
    /// Total number of batches in the underlying loader.
    total_batches: u64,
    /// Feature dimension (inferred from first batch).
    feature_dim: Option<usize>,
    /// Label dimension (inferred from first batch).
    label_dim: Option<usize>,
    /// Current epoch number.
    epoch: usize,
}

impl DataPipeline {
    /// Creates a new data pipeline wrapping a DataLoader.
    pub fn new(loader: Box<dyn DataLoader>, config: PipelineConfig) -> Self {
        let total_batches = loader.num_batches();
        let rng = StdRng::seed_from_u64(config.seed.unwrap_or(0));
        let indices: Vec<u64> = (0..total_batches).collect();

        let mut pipeline = Self {
            loader,
            config,
            rng,
            indices,
            position: 0,
            total_batches,
            feature_dim: None,
            label_dim: None,
            epoch: 0,
        };

        if pipeline.config.shuffle {
            pipeline.shuffle_indices();
        }

        pipeline
    }

    /// Returns the number of batches per epoch.
    pub fn num_batches(&self) -> usize {
        self.total_batches as usize
    }

    /// Returns the current epoch number.
    pub fn epoch(&self) -> usize {
        self.epoch
    }

    /// Resets the pipeline for a new epoch (reshuffles if configured).
    pub fn reset_epoch(&mut self) {
        self.position = 0;
        self.epoch += 1;
        if self.config.shuffle {
            self.shuffle_indices();
        }
    }

    /// Shuffles the batch indices using Fisher-Yates.
    fn shuffle_indices(&mut self) {
        let n = self.indices.len();
        for i in (1..n).rev() {
            let j = self.rng.gen_range(0..=i);
            self.indices.swap(i, j);
        }
    }

    /// Fetches and processes the next batch. Returns `None` at end of epoch.
    pub fn next_batch(&mut self) -> Option<TrainingBatch> {
        if self.position >= self.indices.len() {
            return None;
        }

        let batch_idx = self.position;
        let loader_idx = self.indices[self.position];
        self.position += 1;

        let (raw_inputs, raw_targets) = match self.loader.load_batch(loader_idx) {
            Ok(batch) => batch,
            Err(_) => return None,
        };

        // Infer dimensions from first batch
        if self.feature_dim.is_none() && !raw_inputs.is_empty() {
            let samples_in_batch = if self.config.batch_size > 0 {
                raw_inputs.len() / self.config.batch_size
            } else {
                raw_inputs.len()
            };
            if samples_in_batch > 0 {
                self.feature_dim = Some(samples_in_batch);
                self.label_dim = Some(
                    if self.config.batch_size > 0 && !raw_targets.is_empty() {
                        raw_targets.len() / self.config.batch_size
                    } else {
                        raw_targets.len()
                    },
                );
            }
        }

        // Apply augmentations to inputs
        let augmented_inputs = self.apply_augmentations(&raw_inputs);

        // Convert to BoundedTensors
        let input_error = self.config.input_error;
        let inputs = if input_error > 0.0 {
            let data: Vec<BoundedValue<f64>> = augmented_inputs
                .iter()
                .map(|&v| BoundedValue::new(v, ErrorMargin::absolute(input_error)))
                .collect();
            BoundedTensor::new(data, vec![augmented_inputs.len()])
        } else {
            BoundedTensor::from_exact(augmented_inputs, vec![raw_inputs.len()])
        };

        let target_len = raw_targets.len();
        let targets = BoundedTensor::from_exact(raw_targets, vec![target_len]);

        Some(TrainingBatch {
            inputs,
            targets,
            batch_idx,
        })
    }

    /// Creates an iterator over all batches in the current epoch.
    pub fn iter(&mut self) -> PipelineIterator<'_> {
        self.position = 0;
        if self.config.shuffle {
            self.shuffle_indices();
        }
        PipelineIterator { pipeline: self }
    }

    /// Applies all configured augmentations to the input data.
    fn apply_augmentations(&mut self, inputs: &[f64]) -> Vec<f64> {
        if self.config.augmentations.is_empty() {
            return inputs.to_vec();
        }

        let mut data = inputs.to_vec();

        for aug in &self.config.augmentations {
            match aug {
                Augmentation::GaussianNoise { std } => {
                    for val in data.iter_mut() {
                        let noise: f64 = self.rng.gen::<f64>() * 2.0 - 1.0;
                        *val += noise * std;
                    }
                }
                Augmentation::RandomScale { min, max } => {
                    let scale = self.rng.gen::<f64>() * (max - min) + min;
                    for val in data.iter_mut() {
                        *val *= scale;
                    }
                }
                Augmentation::Normalize { mean, std } => {
                    if *std > 0.0 {
                        for val in data.iter_mut() {
                            *val = (*val - mean) / std;
                        }
                    }
                }
                Augmentation::RandomDropout { probability } => {
                    for val in data.iter_mut() {
                        if self.rng.gen::<f64>() < *probability {
                            *val = 0.0;
                        }
                    }
                }
            }
        }

        data
    }
}

/// Iterator over pipeline batches within a single epoch.
pub struct PipelineIterator<'a> {
    pipeline: &'a mut DataPipeline,
}

impl<'a> Iterator for PipelineIterator<'a> {
    type Item = TrainingBatch;

    fn next(&mut self) -> Option<Self::Item> {
        self.pipeline.next_batch()
    }
}

/// Creates a synthetic dataset suitable for testing.
///
/// Generates `num_samples` random samples with the given dimensions.
/// Labels are one-hot encoded class assignments based on the first feature.
pub fn create_synthetic_loader(
    num_samples: usize,
    feature_dim: usize,
    num_classes: usize,
    batch_size: usize,
    seed: u64,
) -> Box<dyn DataLoader> {
    use helix_core::data::InMemoryDataLoader;

    let mut rng = StdRng::seed_from_u64(seed);
    let mut batches: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
    let mut batch_inputs = Vec::new();
    let mut batch_labels = Vec::new();
    let mut in_batch = 0;

    for _ in 0..num_samples {
        // Generate random features
        let features: Vec<f64> = (0..feature_dim).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect();

        // Assign class based on first feature's sign and magnitude
        let class_idx = ((features[0] + 1.0) / 2.0 * num_classes as f64)
            .floor()
            .clamp(0.0, (num_classes - 1) as f64) as usize;

        // One-hot encode
        let mut label = vec![0.0; num_classes];
        label[class_idx] = 1.0;

        batch_inputs.extend(features);
        batch_labels.extend(label);
        in_batch += 1;

        if in_batch == batch_size {
            batches.push((
                std::mem::take(&mut batch_inputs),
                std::mem::take(&mut batch_labels),
            ));
            in_batch = 0;
        }
    }

    // Don't drop the last incomplete batch
    if !batch_inputs.is_empty() {
        batches.push((batch_inputs, batch_labels));
    }

    Box::new(InMemoryDataLoader::new(batches))
}

/// Creates an MNIST-like synthetic dataset for testing.
///
/// Generates 28x28 grayscale "images" with simple patterns for 10 classes.
/// Each class has a distinct pattern (horizontal stripes, vertical stripes, etc.).
pub fn create_mnist_like_loader(
    num_samples: usize,
    batch_size: usize,
    seed: u64,
) -> Box<dyn DataLoader> {
    use helix_core::data::InMemoryDataLoader;

    let mut rng = StdRng::seed_from_u64(seed);
    let feature_dim = 784; // 28 * 28
    let num_classes = 10;

    let mut batches: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
    let mut batch_inputs = Vec::new();
    let mut batch_labels = Vec::new();
    let mut in_batch = 0;

    for i in 0..num_samples {
        let class_idx = i % num_classes;
        let mut pixels = vec![0.0f64; feature_dim];

        // Generate class-specific patterns
        for row in 0..28 {
            for col in 0..28 {
                let idx = row * 28 + col;
                let base = match class_idx {
                    0 => if row < 14 { 0.8 } else { 0.2 },         // top half
                    1 => if col < 14 { 0.8 } else { 0.2 },         // left half
                    2 => if row % 4 < 2 { 0.8 } else { 0.2 },      // h-stripes
                    3 => if col % 4 < 2 { 0.8 } else { 0.2 },      // v-stripes
                    4 => if (row + col) % 2 == 0 { 0.8 } else { 0.2 }, // checker
                    5 => {
                        let dr = (row as f64 - 14.0).abs();
                        let dc = (col as f64 - 14.0).abs();
                        if dr + dc < 10.0 { 0.8 } else { 0.2 }     // diamond
                    }
                    6 => {
                        let dr = row as f64 - 14.0;
                        let dc = col as f64 - 14.0;
                        if (dr * dr + dc * dc).sqrt() < 10.0 { 0.8 } else { 0.2 } // circle
                    }
                    7 => if row == col || row == 27 - col { 0.8 } else { 0.2 }, // X
                    8 => if row < 5 || row > 22 || col < 5 || col > 22 { 0.8 } else { 0.2 }, // border
                    9 => (row as f64 / 28.0 + col as f64 / 28.0) / 2.0, // gradient
                    _ => 0.0,
                };
                // Add noise
                let noise: f64 = rng.gen::<f64>() * 0.1 - 0.05;
                pixels[idx] = (base + noise).clamp(0.0, 1.0);
            }
        }

        // One-hot label
        let mut label = vec![0.0; num_classes];
        label[class_idx] = 1.0;

        batch_inputs.extend(pixels);
        batch_labels.extend(label);
        in_batch += 1;

        if in_batch == batch_size {
            batches.push((
                std::mem::take(&mut batch_inputs),
                std::mem::take(&mut batch_labels),
            ));
            in_batch = 0;
        }
    }

    if !batch_inputs.is_empty() {
        batches.push((batch_inputs, batch_labels));
    }

    Box::new(InMemoryDataLoader::new(batches))
}

#[cfg(test)]
mod tests {
    use super::*;
    use helix_core::data::InMemoryDataLoader;

    fn make_test_loader() -> Box<dyn DataLoader> {
        // 4 samples, 3 features, 2 classes
        let inputs = vec![
            1.0, 0.0, 0.0, // sample 0
            0.0, 1.0, 0.0, // sample 1
            0.0, 0.0, 1.0, // sample 2
            1.0, 1.0, 0.0, // sample 3
        ];
        let labels = vec![
            1.0, 0.0, // sample 0
            0.0, 1.0, // sample 1
            0.0, 1.0, // sample 2
            1.0, 0.0, // sample 3
        ];
        Box::new(InMemoryDataLoader::from_flat(inputs, labels, 2, 3))
    }

    #[test]
    fn test_pipeline_basic() {
        let loader = make_test_loader();
        let config = PipelineConfig::new(2);
        let mut pipeline = DataPipeline::new(loader, config);

        let mut count = 0;
        while let Some(batch) = pipeline.next_batch() {
            assert!(!batch.inputs.is_empty());
            assert!(!batch.targets.is_empty());
            count += 1;
        }
        assert_eq!(count, 2);
    }

    #[test]
    fn test_pipeline_shuffle() {
        let loader = make_test_loader();
        let config = PipelineConfig::new(2).with_shuffle(true, Some(42));
        let mut pipeline = DataPipeline::new(loader, config);

        let batch1 = pipeline.next_batch().unwrap();
        let _batch2 = pipeline.next_batch().unwrap();

        // Reset and re-iterate
        pipeline.reset_epoch();
        let batch1_new = pipeline.next_batch().unwrap();

        // After reset, batches may be in different order (shuffled)
        // Just verify we get valid batches
        assert!(!batch1.inputs.is_empty());
        assert!(!batch1_new.inputs.is_empty());
    }

    #[test]
    fn test_pipeline_augmentation_noise() {
        let loader = make_test_loader();
        let config = PipelineConfig::new(2)
            .with_augmentation(Augmentation::GaussianNoise { std: 0.1 });
        let mut pipeline = DataPipeline::new(loader, config);

        let batch = pipeline.next_batch().unwrap();
        // Values should be slightly different from original due to noise
        assert!(!batch.inputs.is_empty());
    }

    #[test]
    fn test_pipeline_augmentation_normalize() {
        let loader = make_test_loader();
        let config = PipelineConfig::new(2).with_normalize(0.5, 0.5);
        let mut pipeline = DataPipeline::new(loader, config);

        let batch = pipeline.next_batch().unwrap();
        let values = batch.inputs.values();
        // After normalizing with mean=0.5, std=0.5: (1.0-0.5)/0.5 = 1.0, (0.0-0.5)/0.5 = -1.0
        assert!((values[0] - 1.0).abs() < 1e-10);
        assert!((values[1] - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_pipeline_iterator() {
        let loader = make_test_loader();
        let config = PipelineConfig::new(2);
        let mut pipeline = DataPipeline::new(loader, config);

        let batches: Vec<TrainingBatch> = pipeline.iter().collect();
        assert_eq!(batches.len(), 2);
    }

    #[test]
    fn test_pipeline_with_input_error() {
        let loader = make_test_loader();
        let config = PipelineConfig::new(2).with_input_error(0.01);
        let mut pipeline = DataPipeline::new(loader, config);

        let batch = pipeline.next_batch().unwrap();
        assert!(batch.inputs.max_error() > 0.0);
    }

    #[test]
    fn test_synthetic_loader() {
        let loader = create_synthetic_loader(100, 10, 5, 25, 42);
        assert_eq!(loader.num_batches(), 4);

        let (inputs, labels) = loader.load_batch(0).unwrap();
        assert_eq!(inputs.len(), 25 * 10);
        assert_eq!(labels.len(), 25 * 5);
    }

    #[test]
    fn test_mnist_like_loader() {
        let loader = create_mnist_like_loader(20, 10, 42);
        assert_eq!(loader.num_batches(), 2);

        let (inputs, labels) = loader.load_batch(0).unwrap();
        assert_eq!(inputs.len(), 10 * 784);
        assert_eq!(labels.len(), 10 * 10);

        // Check pixel values are in [0, 1]
        for &v in &inputs {
            assert!(v >= 0.0 && v <= 1.0);
        }
    }

    #[test]
    fn test_pipeline_epoch_reset() {
        let loader = make_test_loader();
        let config = PipelineConfig::new(2);
        let mut pipeline = DataPipeline::new(loader, config);

        assert_eq!(pipeline.epoch(), 0);

        // Consume all batches
        while pipeline.next_batch().is_some() {}

        pipeline.reset_epoch();
        assert_eq!(pipeline.epoch(), 1);

        // Should be able to iterate again
        let batch = pipeline.next_batch();
        assert!(batch.is_some());
    }
}
