//! Data Sharding Module.
//!
//! Provides functionality for partitioning datasets across multiple nodes
//! in federated learning scenarios.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::dataset::{DatasetMetadata, Sample};

/// Shard identifier.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct ShardId(pub u32);

impl std::fmt::Display for ShardId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "shard_{}", self.0)
    }
}

/// Sharding strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ShardingStrategy {
    /// Random assignment with uniform distribution.
    Random { seed: u64 },
    /// Round-robin assignment.
    RoundRobin,
    /// Hash-based assignment using sample ID.
    HashBased,
    /// IID (independent and identically distributed) - uniform label distribution.
    IID { seed: u64 },
    /// Non-IID with label skew.
    NonIID {
        /// Number of classes per shard.
        classes_per_shard: usize,
        seed: u64,
    },
    /// Dirichlet distribution for label imbalance.
    Dirichlet {
        /// Concentration parameter (lower = more imbalanced).
        alpha: f64,
        seed: u64,
    },
}

impl Default for ShardingStrategy {
    fn default() -> Self {
        Self::Random { seed: 42 }
    }
}

/// Configuration for data sharding.
#[derive(Debug, Clone)]
pub struct ShardingConfig {
    /// Number of shards.
    pub num_shards: u32,
    /// Sharding strategy.
    pub strategy: ShardingStrategy,
    /// Minimum samples per shard.
    pub min_samples: usize,
    /// Maximum samples per shard (None for no limit).
    pub max_samples: Option<usize>,
}

impl Default for ShardingConfig {
    fn default() -> Self {
        Self {
            num_shards: 10,
            strategy: ShardingStrategy::default(),
            min_samples: 1,
            max_samples: None,
        }
    }
}

/// A data shard containing a subset of samples.
#[derive(Debug, Clone)]
pub struct DataShard {
    /// Shard ID.
    pub id: ShardId,
    /// Sample indices in the original dataset.
    pub sample_indices: Vec<usize>,
    /// Shard statistics.
    pub stats: ShardStats,
}

/// Statistics about a shard.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ShardStats {
    /// Number of samples.
    pub num_samples: usize,
    /// Label distribution (label -> count).
    pub label_distribution: HashMap<usize, usize>,
    /// Total data size in bytes.
    pub size_bytes: usize,
}

impl DataShard {
    /// Creates a new shard.
    pub fn new(id: ShardId, sample_indices: Vec<usize>) -> Self {
        let stats = ShardStats {
            num_samples: sample_indices.len(),
            ..Default::default()
        };
        
        Self {
            id,
            sample_indices,
            stats,
        }
    }

    /// Returns the number of samples.
    pub fn len(&self) -> usize {
        self.sample_indices.len()
    }

    /// Returns true if empty.
    pub fn is_empty(&self) -> bool {
        self.sample_indices.is_empty()
    }
}

/// Data sharder for partitioning datasets.
pub struct DataSharder {
    /// Configuration.
    config: ShardingConfig,
}

impl DataSharder {
    /// Creates a new sharder with the given configuration.
    pub fn new(config: ShardingConfig) -> Self {
        Self { config }
    }

    /// Shards a dataset according to the strategy.
    pub fn shard(&self, samples: &[Sample]) -> Vec<DataShard> {
        match &self.config.strategy {
            ShardingStrategy::Random { seed } => self.shard_random(samples, *seed),
            ShardingStrategy::RoundRobin => self.shard_round_robin(samples),
            ShardingStrategy::HashBased => self.shard_hash_based(samples),
            ShardingStrategy::IID { seed } => self.shard_iid(samples, *seed),
            ShardingStrategy::NonIID { classes_per_shard, seed } => {
                self.shard_non_iid(samples, *classes_per_shard, *seed)
            }
            ShardingStrategy::Dirichlet { alpha, seed } => {
                self.shard_dirichlet(samples, *alpha, *seed)
            }
        }
    }

    fn shard_random(&self, samples: &[Sample], seed: u64) -> Vec<DataShard> {
        use rand::seq::SliceRandom;
        use rand::SeedableRng;

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut indices: Vec<usize> = (0..samples.len()).collect();
        indices.shuffle(&mut rng);

        self.distribute_indices(indices)
    }

    fn shard_round_robin(&self, samples: &[Sample]) -> Vec<DataShard> {
        let mut shards: Vec<Vec<usize>> = (0..self.config.num_shards)
            .map(|_| Vec::new())
            .collect();

        for (i, _) in samples.iter().enumerate() {
            let shard_idx = i % self.config.num_shards as usize;
            shards[shard_idx].push(i);
        }

        shards
            .into_iter()
            .enumerate()
            .map(|(id, indices)| DataShard::new(ShardId(id as u32), indices))
            .collect()
    }

    fn shard_hash_based(&self, samples: &[Sample]) -> Vec<DataShard> {
        let mut shards: Vec<Vec<usize>> = (0..self.config.num_shards)
            .map(|_| Vec::new())
            .collect();

        for (i, sample) in samples.iter().enumerate() {
            let hash = self.simple_hash(sample.id);
            let shard_idx = (hash % self.config.num_shards as u64) as usize;
            shards[shard_idx].push(i);
        }

        shards
            .into_iter()
            .enumerate()
            .map(|(id, indices)| DataShard::new(ShardId(id as u32), indices))
            .collect()
    }

    fn shard_iid(&self, samples: &[Sample], seed: u64) -> Vec<DataShard> {
        // For IID, we want each shard to have similar class distribution
        // This is essentially random sharding with balancing
        self.shard_random(samples, seed)
    }

    fn shard_non_iid(&self, samples: &[Sample], classes_per_shard: usize, seed: u64) -> Vec<DataShard> {
        use rand::seq::SliceRandom;
        use rand::SeedableRng;

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        
        // Group samples by their class (using first label byte as class indicator)
        let mut class_samples: HashMap<u8, Vec<usize>> = HashMap::new();
        for (i, sample) in samples.iter().enumerate() {
            let class = sample.labels.first().copied().unwrap_or(0);
            class_samples.entry(class).or_default().push(i);
        }

        let num_classes = class_samples.len();
        let mut shards: Vec<Vec<usize>> = (0..self.config.num_shards)
            .map(|_| Vec::new())
            .collect();

        // Assign classes to shards
        let classes: Vec<u8> = class_samples.keys().copied().collect();
        for (shard_idx, chunk) in classes.chunks(classes_per_shard.max(1)).enumerate() {
            if shard_idx >= self.config.num_shards as usize {
                break;
            }
            for &class in chunk {
                if let Some(indices) = class_samples.get_mut(&class) {
                    indices.shuffle(&mut rng);
                    shards[shard_idx].extend(indices.drain(..));
                }
            }
        }

        // Distribute remaining samples
        let remaining: Vec<usize> = class_samples.values().flatten().copied().collect();
        for (i, idx) in remaining.into_iter().enumerate() {
            let shard_idx = i % self.config.num_shards as usize;
            shards[shard_idx].push(idx);
        }

        shards
            .into_iter()
            .enumerate()
            .map(|(id, indices)| DataShard::new(ShardId(id as u32), indices))
            .collect()
    }

    fn shard_dirichlet(&self, samples: &[Sample], alpha: f64, seed: u64) -> Vec<DataShard> {
        use rand::Rng;
        use rand::SeedableRng;

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        
        // Simplified Dirichlet-like distribution
        // In practice, would use proper Dirichlet sampling
        let mut weights: Vec<f64> = (0..self.config.num_shards)
            .map(|_| {
                // Gamma approximation
                let u: f64 = rng.gen();
                (-u.ln()).powf(alpha)
            })
            .collect();

        let sum: f64 = weights.iter().sum();
        for w in &mut weights {
            *w /= sum;
        }

        // Assign samples according to weights
        let mut shards: Vec<Vec<usize>> = (0..self.config.num_shards)
            .map(|_| Vec::new())
            .collect();

        for (i, _) in samples.iter().enumerate() {
            let p: f64 = rng.gen();
            let mut cumsum = 0.0;
            for (shard_idx, &w) in weights.iter().enumerate() {
                cumsum += w;
                if p <= cumsum {
                    shards[shard_idx].push(i);
                    break;
                }
            }
        }

        shards
            .into_iter()
            .enumerate()
            .map(|(id, indices)| DataShard::new(ShardId(id as u32), indices))
            .collect()
    }

    fn distribute_indices(&self, indices: Vec<usize>) -> Vec<DataShard> {
        let chunk_size = (indices.len() + self.config.num_shards as usize - 1) 
            / self.config.num_shards as usize;

        indices
            .chunks(chunk_size)
            .enumerate()
            .map(|(id, chunk)| DataShard::new(ShardId(id as u32), chunk.to_vec()))
            .collect()
    }

    fn simple_hash(&self, value: usize) -> u64 {
        let mut h = value as u64;
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51afd7ed558ccd);
        h ^= h >> 33;
        h = h.wrapping_mul(0xc4ceb9fe1a85ec53);
        h ^= h >> 33;
        h
    }
}

/// Shard assignment for distributed training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardAssignment {
    /// Node ID to shard mapping.
    pub node_shards: HashMap<String, Vec<ShardId>>,
    /// Total number of shards.
    pub total_shards: u32,
}

impl ShardAssignment {
    /// Creates a new assignment.
    pub fn new(num_shards: u32) -> Self {
        Self {
            node_shards: HashMap::new(),
            total_shards: num_shards,
        }
    }

    /// Assigns a shard to a node.
    pub fn assign(&mut self, node_id: String, shard_id: ShardId) {
        self.node_shards.entry(node_id).or_default().push(shard_id);
    }

    /// Gets shards for a node.
    pub fn get_shards(&self, node_id: &str) -> Vec<ShardId> {
        self.node_shards.get(node_id).cloned().unwrap_or_default()
    }

    /// Returns the number of assigned nodes.
    pub fn num_nodes(&self) -> usize {
        self.node_shards.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::dataset::{create_synthetic, DatasetConfig, InMemoryDataset};

    #[test]
    fn test_round_robin_sharding() {
        let (_, samples) = create_synthetic(100, 10, 5);
        
        let sharder = DataSharder::new(ShardingConfig {
            num_shards: 4,
            strategy: ShardingStrategy::RoundRobin,
            ..Default::default()
        });
        
        let shards = sharder.shard(&samples);
        
        assert_eq!(shards.len(), 4);
        assert_eq!(shards[0].len(), 25);
        assert_eq!(shards[1].len(), 25);
    }

    #[test]
    fn test_random_sharding() {
        let (_, samples) = create_synthetic(100, 10, 5);
        
        let sharder = DataSharder::new(ShardingConfig {
            num_shards: 5,
            strategy: ShardingStrategy::Random { seed: 42 },
            ..Default::default()
        });
        
        let shards = sharder.shard(&samples);
        
        assert_eq!(shards.len(), 5);
        
        // All samples should be assigned
        let total: usize = shards.iter().map(|s| s.len()).sum();
        assert_eq!(total, 100);
    }

    #[test]
    fn test_shard_assignment() {
        let mut assignment = ShardAssignment::new(4);
        
        assignment.assign("node1".to_string(), ShardId(0));
        assignment.assign("node1".to_string(), ShardId(1));
        assignment.assign("node2".to_string(), ShardId(2));
        
        assert_eq!(assignment.get_shards("node1").len(), 2);
        assert_eq!(assignment.get_shards("node2").len(), 1);
        assert_eq!(assignment.num_nodes(), 2);
    }
}
