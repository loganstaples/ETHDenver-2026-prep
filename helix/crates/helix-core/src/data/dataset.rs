//! Dataset Loading Module.
//!
//! Provides functionality for loading, preprocessing, and batching datasets
//! for federated learning training.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Dataset metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMetadata {
    /// Dataset name.
    pub name: String,
    /// Number of samples.
    pub num_samples: usize,
    /// Feature dimensions.
    pub feature_dims: Vec<usize>,
    /// Label dimensions.
    pub label_dims: Vec<usize>,
    /// Data type.
    pub dtype: DataType,
    /// Additional metadata.
    pub extra: HashMap<String, String>,
}

/// Supported data types.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DataType {
    Float32,
    Float64,
    Int32,
    Int64,
    UInt8,
}

impl DataType {
    /// Returns the size in bytes of this data type.
    pub fn size_bytes(&self) -> usize {
        match self {
            DataType::Float32 => 4,
            DataType::Float64 => 8,
            DataType::Int32 => 4,
            DataType::Int64 => 8,
            DataType::UInt8 => 1,
        }
    }
}

/// A single data sample.
#[derive(Debug, Clone)]
pub struct Sample {
    /// Sample ID.
    pub id: usize,
    /// Feature data as bytes.
    pub features: Vec<u8>,
    /// Label data as bytes.
    pub labels: Vec<u8>,
}

impl Sample {
    /// Creates a new sample from f32 features and labels.
    pub fn from_f32(id: usize, features: Vec<f32>, labels: Vec<f32>) -> Self {
        let feat_bytes: Vec<u8> = features
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let label_bytes: Vec<u8> = labels
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        
        Self {
            id,
            features: feat_bytes,
            labels: label_bytes,
        }
    }

    /// Converts features to f32 vector.
    pub fn features_as_f32(&self) -> Vec<f32> {
        self.features
            .chunks(4)
            .map(|chunk| {
                let arr: [u8; 4] = chunk.try_into().unwrap_or([0; 4]);
                f32::from_le_bytes(arr)
            })
            .collect()
    }

    /// Converts labels to f32 vector.
    pub fn labels_as_f32(&self) -> Vec<f32> {
        self.labels
            .chunks(4)
            .map(|chunk| {
                let arr: [u8; 4] = chunk.try_into().unwrap_or([0; 4]);
                f32::from_le_bytes(arr)
            })
            .collect()
    }
}

/// A batch of samples.
#[derive(Debug, Clone)]
pub struct Batch {
    /// Batch ID.
    pub id: usize,
    /// Samples in this batch.
    pub samples: Vec<Sample>,
}

impl Batch {
    /// Creates a new batch.
    pub fn new(id: usize, samples: Vec<Sample>) -> Self {
        Self { id, samples }
    }

    /// Returns the number of samples.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Returns true if empty.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Gets features as a 2D matrix (batch_size x feature_dim).
    pub fn features_matrix_f32(&self) -> Vec<Vec<f32>> {
        self.samples
            .iter()
            .map(|s| s.features_as_f32())
            .collect()
    }

    /// Gets labels as a 2D matrix.
    pub fn labels_matrix_f32(&self) -> Vec<Vec<f32>> {
        self.samples
            .iter()
            .map(|s| s.labels_as_f32())
            .collect()
    }
}

/// Configuration for dataset loading.
#[derive(Debug, Clone)]
pub struct DatasetConfig {
    /// Batch size.
    pub batch_size: usize,
    /// Whether to shuffle.
    pub shuffle: bool,
    /// Random seed for shuffling.
    pub seed: Option<u64>,
    /// Whether to drop last incomplete batch.
    pub drop_last: bool,
    /// Number of prefetch batches.
    pub prefetch: usize,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            shuffle: true,
            seed: None,
            drop_last: false,
            prefetch: 2,
        }
    }
}

/// In-memory dataset.
pub struct InMemoryDataset {
    /// Dataset metadata.
    metadata: DatasetMetadata,
    /// All samples.
    samples: Vec<Sample>,
    /// Configuration.
    config: DatasetConfig,
    /// Current position.
    position: usize,
    /// Shuffled indices.
    indices: Vec<usize>,
}

impl InMemoryDataset {
    /// Creates a new in-memory dataset.
    pub fn new(metadata: DatasetMetadata, samples: Vec<Sample>, config: DatasetConfig) -> Self {
        let indices: Vec<usize> = (0..samples.len()).collect();
        
        Self {
            metadata,
            samples,
            config,
            position: 0,
            indices,
        }
    }

    /// Returns the metadata.
    pub fn metadata(&self) -> &DatasetMetadata {
        &self.metadata
    }

    /// Returns the number of samples.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Returns true if empty.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Returns the number of batches.
    pub fn num_batches(&self) -> usize {
        let n = self.samples.len();
        let bs = self.config.batch_size;
        if self.config.drop_last {
            n / bs
        } else {
            (n + bs - 1) / bs
        }
    }

    /// Shuffles the dataset.
    pub fn shuffle(&mut self) {
        use rand::seq::SliceRandom;
        use rand::SeedableRng;

        if let Some(seed) = self.config.seed {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            self.indices.shuffle(&mut rng);
        } else {
            self.indices.shuffle(&mut rand::thread_rng());
        }
    }

    /// Resets to the beginning.
    pub fn reset(&mut self) {
        self.position = 0;
        if self.config.shuffle {
            self.shuffle();
        }
    }

    /// Gets the next batch.
    pub fn next_batch(&mut self) -> Option<Batch> {
        if self.position >= self.samples.len() {
            return None;
        }

        let end = (self.position + self.config.batch_size).min(self.samples.len());
        
        // Check if we should drop incomplete batch
        if self.config.drop_last && end - self.position < self.config.batch_size {
            return None;
        }

        let batch_samples: Vec<Sample> = self.indices[self.position..end]
            .iter()
            .map(|&i| self.samples[i].clone())
            .collect();

        let batch_id = self.position / self.config.batch_size;
        self.position = end;

        Some(Batch::new(batch_id, batch_samples))
    }

    /// Gets a specific sample by index.
    pub fn get(&self, index: usize) -> Option<&Sample> {
        self.samples.get(index)
    }

    /// Creates an iterator over batches.
    pub fn iter(&mut self) -> DatasetIterator {
        self.reset();
        DatasetIterator { dataset: self }
    }
}

/// Iterator over dataset batches.
pub struct DatasetIterator<'a> {
    dataset: &'a mut InMemoryDataset,
}

impl<'a> Iterator for DatasetIterator<'a> {
    type Item = Batch;

    fn next(&mut self) -> Option<Self::Item> {
        self.dataset.next_batch()
    }
}

/// Loads a dataset from a CSV file (simplified).
pub fn load_csv(
    path: &str,
    feature_cols: &[usize],
    label_cols: &[usize],
    has_header: bool,
) -> Result<(DatasetMetadata, Vec<Sample>), String> {
    // Note: This is a simplified implementation
    // In production, would use csv crate
    
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read file: {}", e))?;

    let mut lines: Vec<&str> = content.lines().collect();
    
    if has_header && !lines.is_empty() {
        lines.remove(0);
    }

    let mut samples = Vec::new();
    
    for (id, line) in lines.iter().enumerate() {
        let values: Vec<f32> = line
            .split(',')
            .filter_map(|s| s.trim().parse::<f32>().ok())
            .collect();

        if values.is_empty() {
            continue;
        }

        let features: Vec<f32> = feature_cols
            .iter()
            .filter_map(|&i| values.get(i).copied())
            .collect();

        let labels: Vec<f32> = label_cols
            .iter()
            .filter_map(|&i| values.get(i).copied())
            .collect();

        samples.push(Sample::from_f32(id, features, labels));
    }

    let metadata = DatasetMetadata {
        name: path.to_string(),
        num_samples: samples.len(),
        feature_dims: vec![feature_cols.len()],
        label_dims: vec![label_cols.len()],
        dtype: DataType::Float32,
        extra: HashMap::new(),
    };

    Ok((metadata, samples))
}

/// Creates a synthetic dataset for testing.
pub fn create_synthetic(
    num_samples: usize,
    feature_dim: usize,
    num_classes: usize,
) -> (DatasetMetadata, Vec<Sample>) {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let samples: Vec<Sample> = (0..num_samples)
        .map(|id| {
            let features: Vec<f32> = (0..feature_dim)
                .map(|_| rng.gen_range(-1.0..1.0))
                .collect();
            
            let mut labels = vec![0.0f32; num_classes];
            let class = rng.gen_range(0..num_classes);
            labels[class] = 1.0;
            
            Sample::from_f32(id, features, labels)
        })
        .collect();

    let metadata = DatasetMetadata {
        name: "synthetic".to_string(),
        num_samples,
        feature_dims: vec![feature_dim],
        label_dims: vec![num_classes],
        dtype: DataType::Float32,
        extra: HashMap::new(),
    };

    (metadata, samples)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_roundtrip() {
        let features = vec![1.0, 2.0, 3.0];
        let labels = vec![0.0, 1.0];
        
        let sample = Sample::from_f32(0, features.clone(), labels.clone());
        
        let recovered_features = sample.features_as_f32();
        let recovered_labels = sample.labels_as_f32();
        
        assert_eq!(features, recovered_features);
        assert_eq!(labels, recovered_labels);
    }

    #[test]
    fn test_synthetic_dataset() {
        let (metadata, samples) = create_synthetic(100, 10, 3);
        
        assert_eq!(metadata.num_samples, 100);
        assert_eq!(samples.len(), 100);
    }

    #[test]
    fn test_dataset_batching() {
        let (metadata, samples) = create_synthetic(100, 10, 3);
        let config = DatasetConfig {
            batch_size: 32,
            shuffle: false,
            ..Default::default()
        };
        
        let mut dataset = InMemoryDataset::new(metadata, samples, config);
        
        let batch1 = dataset.next_batch().unwrap();
        assert_eq!(batch1.len(), 32);
        
        let batch2 = dataset.next_batch().unwrap();
        assert_eq!(batch2.len(), 32);
        
        let batch3 = dataset.next_batch().unwrap();
        assert_eq!(batch3.len(), 32);
        
        // Last batch has only 4 samples
        let batch4 = dataset.next_batch().unwrap();
        assert_eq!(batch4.len(), 4);
    }

    #[test]
    fn test_num_batches() {
        let (metadata, samples) = create_synthetic(100, 10, 3);
        
        let config = DatasetConfig {
            batch_size: 32,
            drop_last: false,
            ..Default::default()
        };
        let dataset = InMemoryDataset::new(metadata.clone(), samples.clone(), config);
        assert_eq!(dataset.num_batches(), 4);
        
        let config = DatasetConfig {
            batch_size: 32,
            drop_last: true,
            ..Default::default()
        };
        let dataset = InMemoryDataset::new(metadata, samples, config);
        assert_eq!(dataset.num_batches(), 3);
    }
}
