//! Computation Chunking for Large Models.
//!
//! Splits ML computations into smaller chunks that can be proven independently
//! and then aggregated. Essential for proving large neural network operations.

use std::collections::HashMap;
use helix_circuits::halo2curves::bn256::Fr;
use serde::{Deserialize, Serialize};

/// A chunk of computation to be proven.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputationChunk {
    /// Unique identifier for this chunk.
    pub id: ChunkId,
    /// Parent chunk ID (for dependency tracking).
    pub parent_id: Option<ChunkId>,
    /// Input state commitment.
    pub input_commitment: [u8; 32],
    /// Output state commitment.
    pub output_commitment: [u8; 32],
    /// Type of computation in this chunk.
    pub computation_type: ComputationType,
    /// Layer indices covered by this chunk.
    pub layer_range: (usize, usize),
    /// Error bound for this chunk.
    pub error_bound: f64,
}

/// Unique identifier for a chunk.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct ChunkId(pub u64);

impl ChunkId {
    /// Creates a new chunk ID from components.
    pub fn new(batch_id: u32, layer_id: u32) -> Self {
        Self(((batch_id as u64) << 32) | (layer_id as u64))
    }

    /// Gets the batch portion of the ID.
    pub fn batch_id(&self) -> u32 {
        (self.0 >> 32) as u32
    }

    /// Gets the layer portion of the ID.
    pub fn layer_id(&self) -> u32 {
        self.0 as u32
    }
}

/// Type of computation in a chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComputationType {
    /// Forward pass computation.
    Forward,
    /// Backward pass (gradient computation).
    Backward,
    /// Weight update.
    WeightUpdate,
    /// Error accumulation.
    ErrorAccumulation,
    /// Aggregation of multiple gradients.
    GradientAggregation,
}

/// Configuration for chunking strategy.
#[derive(Debug, Clone)]
pub struct ChunkingConfig {
    /// Maximum number of operations per chunk.
    pub max_ops_per_chunk: usize,
    /// Maximum memory footprint per chunk (in bytes).
    pub max_memory_per_chunk: usize,
    /// Number of layers per chunk (for layer-based chunking).
    pub layers_per_chunk: usize,
    /// Whether to enable parallel chunking.
    pub parallel: bool,
    /// Number of parallel workers.
    pub num_workers: usize,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            max_ops_per_chunk: 1_000_000,
            max_memory_per_chunk: 512 * 1024 * 1024, // 512MB
            layers_per_chunk: 4,
            parallel: true,
            num_workers: 4,
        }
    }
}

/// Chunker for splitting model computations.
pub struct ComputationChunker {
    /// Configuration.
    config: ChunkingConfig,
    /// Generated chunks.
    chunks: Vec<ComputationChunk>,
    /// Chunk ID counter.
    next_id: u64,
    /// Dependency graph: chunk_id -> dependent chunk_ids.
    dependencies: HashMap<ChunkId, Vec<ChunkId>>,
}

impl ComputationChunker {
    /// Creates a new chunker with default config.
    pub fn new() -> Self {
        Self::with_config(ChunkingConfig::default())
    }

    /// Creates a chunker with custom config.
    pub fn with_config(config: ChunkingConfig) -> Self {
        Self {
            config,
            chunks: Vec::new(),
            next_id: 0,
            dependencies: HashMap::new(),
        }
    }

    /// Chunks a forward pass computation.
    pub fn chunk_forward_pass(
        &mut self,
        num_layers: usize,
        input_commitment: [u8; 32],
    ) -> Vec<ChunkId> {
        let mut chunk_ids = Vec::new();
        let mut current_commitment = input_commitment;
        let mut parent_id = None;

        for start in (0..num_layers).step_by(self.config.layers_per_chunk) {
            let end = (start + self.config.layers_per_chunk).min(num_layers);
            
            let chunk_id = self.allocate_chunk_id();
            let output_commitment = self.compute_intermediate_commitment(start, end);

            let chunk = ComputationChunk {
                id: chunk_id,
                parent_id,
                input_commitment: current_commitment,
                output_commitment,
                computation_type: ComputationType::Forward,
                layer_range: (start, end),
                error_bound: 0.0, // Will be computed during proving
            };

            if let Some(parent) = parent_id {
                self.dependencies.entry(parent).or_default().push(chunk_id);
            }

            self.chunks.push(chunk);
            chunk_ids.push(chunk_id);

            current_commitment = output_commitment;
            parent_id = Some(chunk_id);
        }

        chunk_ids
    }

    /// Chunks a backward pass computation.
    pub fn chunk_backward_pass(
        &mut self,
        num_layers: usize,
        gradient_commitment: [u8; 32],
        forward_chunk_ids: &[ChunkId],
    ) -> Vec<ChunkId> {
        let mut chunk_ids = Vec::new();
        let mut current_commitment = gradient_commitment;
        let mut parent_id = None;

        // Backward pass goes in reverse order
        for (i, start) in (0..num_layers).step_by(self.config.layers_per_chunk).rev().enumerate() {
            let end = (start + self.config.layers_per_chunk).min(num_layers);
            
            let chunk_id = self.allocate_chunk_id();
            let output_commitment = self.compute_intermediate_commitment(end, start);

            let chunk = ComputationChunk {
                id: chunk_id,
                parent_id,
                input_commitment: current_commitment,
                output_commitment,
                computation_type: ComputationType::Backward,
                layer_range: (start, end),
                error_bound: 0.0,
            };

            // Backward chunk depends on corresponding forward chunk
            if let Some(&forward_id) = forward_chunk_ids.get(forward_chunk_ids.len() - 1 - i) {
                self.dependencies.entry(forward_id).or_default().push(chunk_id);
            }

            self.chunks.push(chunk);
            chunk_ids.push(chunk_id);

            current_commitment = output_commitment;
            parent_id = Some(chunk_id);
        }

        chunk_ids
    }

    /// Chunks a weight update computation.
    pub fn chunk_weight_update(
        &mut self,
        num_parameters: usize,
        gradient_chunk_ids: &[ChunkId],
    ) -> ChunkId {
        let chunk_id = self.allocate_chunk_id();

        let chunk = ComputationChunk {
            id: chunk_id,
            parent_id: gradient_chunk_ids.last().copied(),
            input_commitment: [0; 32],
            output_commitment: [0; 32],
            computation_type: ComputationType::WeightUpdate,
            layer_range: (0, num_parameters),
            error_bound: 0.0,
        };

        // Weight update depends on all gradient chunks
        for &grad_id in gradient_chunk_ids {
            self.dependencies.entry(grad_id).or_default().push(chunk_id);
        }

        self.chunks.push(chunk);
        chunk_id
    }

    /// Returns chunks ready for proving (no unproven dependencies).
    pub fn get_ready_chunks(&self, proven: &[ChunkId]) -> Vec<&ComputationChunk> {
        let proven_set: std::collections::HashSet<_> = proven.iter().collect();
        
        self.chunks.iter().filter(|chunk| {
            if proven_set.contains(&chunk.id) {
                return false;
            }
            
            match chunk.parent_id {
                Some(parent) => proven_set.contains(&parent),
                None => true,
            }
        }).collect()
    }

    /// Returns all chunks.
    pub fn chunks(&self) -> &[ComputationChunk] {
        &self.chunks
    }

    /// Returns the dependency graph.
    pub fn dependencies(&self) -> &HashMap<ChunkId, Vec<ChunkId>> {
        &self.dependencies
    }

    /// Computes a topological ordering of chunks for execution.
    pub fn topological_order(&self) -> Vec<ChunkId> {
        let mut result = Vec::new();
        let mut visited = std::collections::HashSet::new();
        
        for chunk in &self.chunks {
            self.topo_visit(chunk.id, &mut visited, &mut result);
        }
        
        result
    }

    fn topo_visit(
        &self,
        chunk_id: ChunkId,
        visited: &mut std::collections::HashSet<ChunkId>,
        result: &mut Vec<ChunkId>,
    ) {
        if visited.contains(&chunk_id) {
            return;
        }
        
        visited.insert(chunk_id);
        
        if let Some(deps) = self.dependencies.get(&chunk_id) {
            for &dep in deps {
                self.topo_visit(dep, visited, result);
            }
        }
        
        result.push(chunk_id);
    }

    fn allocate_chunk_id(&mut self) -> ChunkId {
        let id = ChunkId(self.next_id);
        self.next_id += 1;
        id
    }

    fn compute_intermediate_commitment(&self, _start: usize, _end: usize) -> [u8; 32] {
        // Placeholder - actual implementation would hash intermediate state
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        self.next_id.hash(&mut hasher);
        let hash = hasher.finish();
        
        let mut commitment = [0u8; 32];
        commitment[..8].copy_from_slice(&hash.to_le_bytes());
        commitment
    }
}

impl Default for ComputationChunker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_forward_pass() {
        let mut chunker = ComputationChunker::with_config(ChunkingConfig {
            layers_per_chunk: 2,
            ..Default::default()
        });

        let input = [1u8; 32];
        let chunk_ids = chunker.chunk_forward_pass(6, input);

        assert_eq!(chunk_ids.len(), 3); // 6 layers / 2 per chunk = 3 chunks
    }

    #[test]
    fn test_chunk_backward_pass() {
        let mut chunker = ComputationChunker::with_config(ChunkingConfig {
            layers_per_chunk: 2,
            ..Default::default()
        });

        let input = [1u8; 32];
        let forward_ids = chunker.chunk_forward_pass(4, input);
        let backward_ids = chunker.chunk_backward_pass(4, [2u8; 32], &forward_ids);

        assert_eq!(backward_ids.len(), 2);
    }

    #[test]
    fn test_topological_order() {
        let mut chunker = ComputationChunker::with_config(ChunkingConfig {
            layers_per_chunk: 2,
            ..Default::default()
        });

        let input = [1u8; 32];
        chunker.chunk_forward_pass(4, input);

        let order = chunker.topological_order();
        assert_eq!(order.len(), 2);
    }
}
