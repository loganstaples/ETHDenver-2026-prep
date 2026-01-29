//! Proof Aggregation.
//!
//! Combines multiple proofs into a single aggregated proof.
//! Essential for recursive proving and reducing verification costs.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

use super::chunking::ChunkId;
use super::parallel::ChunkProof;

/// An aggregated proof combining multiple chunk proofs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedProof {
    /// Unique identifier for this aggregated proof.
    pub id: AggregationId,
    /// Root commitment (hash of all chunk commitments).
    pub root_commitment: [u8; 32],
    /// The aggregated proof bytes.
    pub proof: Vec<u8>,
    /// Public inputs for verification.
    pub public_inputs: Vec<[u8; 32]>,
    /// Total error bound (sum of chunk bounds).
    pub total_error_bound: f64,
    /// IDs of chunks included in this aggregation.
    pub chunk_ids: Vec<ChunkId>,
    /// Number of layers covered.
    pub num_layers: usize,
    /// Aggregation depth (for recursive aggregation).
    pub depth: u32,
}

/// Unique identifier for an aggregated proof.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct AggregationId(pub u64);

/// Configuration for proof aggregation.
#[derive(Debug, Clone)]
pub struct AggregationConfig {
    /// Maximum number of proofs per aggregation.
    pub max_proofs_per_aggregation: usize,
    /// Whether to use recursive aggregation.
    pub recursive: bool,
    /// Target depth for recursive aggregation.
    pub target_depth: u32,
    /// Commitment scheme (KZG, IPA, etc).
    pub commitment_scheme: CommitmentScheme,
}

impl Default for AggregationConfig {
    fn default() -> Self {
        Self {
            max_proofs_per_aggregation: 8,
            recursive: true,
            target_depth: 3,
            commitment_scheme: CommitmentScheme::KZG,
        }
    }
}

/// Commitment scheme for aggregation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum CommitmentScheme {
    /// KZG (Kate) polynomial commitments.
    KZG,
    /// Inner Product Argument.
    IPA,
    /// FRI-based.
    FRI,
}

/// Proof aggregator.
pub struct ProofAggregator {
    /// Configuration.
    config: AggregationConfig,
    /// Pending proofs to aggregate.
    pending_proofs: Vec<ChunkProof>,
    /// Completed aggregations.
    aggregations: HashMap<AggregationId, AggregatedProof>,
    /// Next aggregation ID.
    next_id: u64,
}

impl ProofAggregator {
    /// Creates a new aggregator.
    pub fn new() -> Self {
        Self::with_config(AggregationConfig::default())
    }

    /// Creates an aggregator with custom config.
    pub fn with_config(config: AggregationConfig) -> Self {
        Self {
            config,
            pending_proofs: Vec::new(),
            aggregations: HashMap::new(),
            next_id: 0,
        }
    }

    /// Adds a proof for aggregation.
    pub fn add_proof(&mut self, proof: ChunkProof) {
        self.pending_proofs.push(proof);
    }

    /// Adds multiple proofs.
    pub fn add_proofs(&mut self, proofs: Vec<ChunkProof>) {
        self.pending_proofs.extend(proofs);
    }

    /// Aggregates all pending proofs.
    pub fn aggregate_all(&mut self) -> Vec<AggregationId> {
        if self.pending_proofs.is_empty() {
            return Vec::new();
        }

        let mut aggregation_ids = Vec::new();

        // Split into groups
        let groups: Vec<Vec<ChunkProof>> = self.pending_proofs
            .chunks(self.config.max_proofs_per_aggregation)
            .map(|c| c.to_vec())
            .collect();

        self.pending_proofs.clear();

        for group in groups {
            let agg = self.aggregate_group(group, 0);
            aggregation_ids.push(agg.id);
            self.aggregations.insert(agg.id, agg);
        }

        // Recursive aggregation if enabled
        if self.config.recursive && aggregation_ids.len() > 1 {
            let recursive_ids = self.recursive_aggregate(&aggregation_ids);
            aggregation_ids.extend(recursive_ids);
        }

        aggregation_ids
    }

    /// Aggregates pending proofs into a single proof.
    pub fn aggregate_single(&mut self) -> Option<AggregatedProof> {
        if self.pending_proofs.is_empty() {
            return None;
        }

        let proofs = std::mem::take(&mut self.pending_proofs);
        let agg = self.aggregate_group(proofs, 0);
        let id = agg.id;
        self.aggregations.insert(id, agg.clone());
        Some(agg)
    }

    /// Gets an aggregated proof by ID.
    pub fn get_aggregation(&self, id: AggregationId) -> Option<&AggregatedProof> {
        self.aggregations.get(&id)
    }

    /// Gets all aggregated proofs.
    pub fn all_aggregations(&self) -> Vec<&AggregatedProof> {
        self.aggregations.values().collect()
    }

    /// Verifies an aggregated proof.
    pub fn verify(&self, agg: &AggregatedProof) -> bool {
        // Placeholder verification
        // Actual implementation would verify the aggregated proof
        // against the root commitment and public inputs
        
        if agg.proof.is_empty() {
            return false;
        }

        // Check root commitment matches chunk commitments
        let computed_root = self.compute_root(&agg.public_inputs);
        computed_root == agg.root_commitment
    }

    /// Computes the final root for the entire training step.
    pub fn final_root(&self) -> Option<[u8; 32]> {
        if self.aggregations.is_empty() {
            return None;
        }

        // Find the top-level aggregation (highest depth)
        let top = self.aggregations.values()
            .max_by_key(|a| a.depth)?;
        
        Some(top.root_commitment)
    }

    fn aggregate_group(&mut self, proofs: Vec<ChunkProof>, depth: u32) -> AggregatedProof {
        let id = AggregationId(self.next_id);
        self.next_id += 1;

        // Collect public inputs
        let public_inputs: Vec<[u8; 32]> = proofs.iter()
            .flat_map(|p| p.public_inputs.clone())
            .collect();

        // Compute root commitment
        let root_commitment = self.compute_root(&public_inputs);

        // Sum error bounds
        let total_error_bound: f64 = proofs.iter()
            .map(|p| p.error_bound)
            .sum();

        // Collect chunk IDs
        let chunk_ids: Vec<ChunkId> = proofs.iter()
            .map(|p| p.chunk_id)
            .collect();

        // Generate aggregated proof (placeholder)
        let aggregated_proof = self.generate_aggregated_proof(&proofs);

        AggregatedProof {
            id,
            root_commitment,
            proof: aggregated_proof,
            public_inputs,
            total_error_bound,
            chunk_ids,
            num_layers: proofs.len(),
            depth,
        }
    }

    fn recursive_aggregate(&mut self, ids: &[AggregationId]) -> Vec<AggregationId> {
        if ids.len() <= 1 {
            return Vec::new();
        }

        let mut new_ids = Vec::new();
        let aggregations: Vec<AggregatedProof> = ids.iter()
            .filter_map(|id| self.aggregations.get(id).cloned())
            .collect();

        // Convert aggregations to "fake" chunk proofs for re-aggregation
        let fake_proofs: Vec<ChunkProof> = aggregations.iter()
            .map(|a| ChunkProof {
                chunk_id: ChunkId(a.id.0),
                proof: a.proof.clone(),
                public_inputs: vec![a.root_commitment],
                error_bound: a.total_error_bound,
                generation_time_ms: 0,
            })
            .collect();

        let depth = aggregations.first().map(|a| a.depth + 1).unwrap_or(1);

        // Group and aggregate
        let groups: Vec<Vec<ChunkProof>> = fake_proofs
            .chunks(self.config.max_proofs_per_aggregation)
            .map(|c| c.to_vec())
            .collect();

        for group in groups {
            let agg = self.aggregate_group(group, depth);
            new_ids.push(agg.id);
            self.aggregations.insert(agg.id, agg);
        }

        // Continue recursively if needed
        if new_ids.len() > 1 && depth < self.config.target_depth {
            let recursive = self.recursive_aggregate(&new_ids);
            new_ids.extend(recursive);
        }

        new_ids
    }

    fn compute_root(&self, inputs: &[[u8; 32]]) -> [u8; 32] {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        for input in inputs {
            input.hash(&mut hasher);
        }
        let hash = hasher.finish();

        let mut root = [0u8; 32];
        root[..8].copy_from_slice(&hash.to_le_bytes());
        
        // Double hash for Merkle-like structure
        hasher = DefaultHasher::new();
        root.hash(&mut hasher);
        let hash2 = hasher.finish();
        root[8..16].copy_from_slice(&hash2.to_le_bytes());
        
        root
    }

    fn generate_aggregated_proof(&self, proofs: &[ChunkProof]) -> Vec<u8> {
        // Placeholder - actual implementation would use folding or SNARK composition
        let mut aggregated = Vec::new();
        
        // Header
        aggregated.extend_from_slice(b"HELIX_AGG_V1:");
        aggregated.extend_from_slice(&(proofs.len() as u32).to_le_bytes());
        
        // Include proof hashes (simulating commitment aggregation)
        for proof in proofs {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            
            let mut hasher = DefaultHasher::new();
            proof.proof.hash(&mut hasher);
            aggregated.extend_from_slice(&hasher.finish().to_le_bytes());
        }
        
        aggregated
    }
}

impl Default for ProofAggregator {
    fn default() -> Self {
        Self::new()
    }
}

/// Merkle tree for commitment aggregation.
pub struct CommitmentTree {
    /// Leaf commitments.
    leaves: Vec<[u8; 32]>,
    /// Internal nodes (level-by-level).
    nodes: Vec<Vec<[u8; 32]>>,
    /// Root commitment.
    root: Option<[u8; 32]>,
}

impl CommitmentTree {
    /// Creates a new empty tree.
    pub fn new() -> Self {
        Self {
            leaves: Vec::new(),
            nodes: Vec::new(),
            root: None,
        }
    }

    /// Adds a leaf commitment.
    pub fn add_leaf(&mut self, commitment: [u8; 32]) {
        self.leaves.push(commitment);
        self.root = None; // Invalidate root
    }

    /// Builds the tree and returns the root.
    pub fn build(&mut self) -> [u8; 32] {
        if self.leaves.is_empty() {
            return [0; 32];
        }

        self.nodes.clear();
        
        let mut current_level = self.leaves.clone();
        
        while current_level.len() > 1 {
            let mut next_level = Vec::new();
            
            for chunk in current_level.chunks(2) {
                let node = if chunk.len() == 2 {
                    Self::hash_pair(&chunk[0], &chunk[1])
                } else {
                    Self::hash_pair(&chunk[0], &chunk[0])
                };
                next_level.push(node);
            }
            
            self.nodes.push(current_level);
            current_level = next_level;
        }
        
        self.root = Some(current_level[0]);
        current_level[0]
    }

    /// Gets the root commitment.
    pub fn root(&self) -> Option<[u8; 32]> {
        self.root
    }

    /// Generates a Merkle proof for a leaf at the given index.
    pub fn proof(&self, leaf_index: usize) -> Option<Vec<[u8; 32]>> {
        if leaf_index >= self.leaves.len() || self.root.is_none() {
            return None;
        }

        let mut proof = Vec::new();
        let mut index = leaf_index;

        for level in &self.nodes {
            let sibling_index = if index % 2 == 0 { index + 1 } else { index - 1 };
            if sibling_index < level.len() {
                proof.push(level[sibling_index]);
            }
            index /= 2;
        }

        Some(proof)
    }

    fn hash_pair(a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        a.hash(&mut hasher);
        b.hash(&mut hasher);
        let hash = hasher.finish();

        let mut result = [0u8; 32];
        result[..8].copy_from_slice(&hash.to_le_bytes());
        result
    }
}

impl Default for CommitmentTree {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunking::ComputationType;

    fn make_test_proof(id: u64) -> ChunkProof {
        ChunkProof {
            chunk_id: ChunkId(id),
            proof: vec![1, 2, 3, 4],
            public_inputs: vec![[id as u8; 32]],
            error_bound: 0.01,
            generation_time_ms: 100,
        }
    }

    #[test]
    fn test_aggregator_basic() {
        let mut aggregator = ProofAggregator::new();
        
        for i in 0..4 {
            aggregator.add_proof(make_test_proof(i));
        }
        
        let ids = aggregator.aggregate_all();
        assert!(!ids.is_empty());
    }

    #[test]
    fn test_commitment_tree() {
        let mut tree = CommitmentTree::new();
        
        for i in 0..4 {
            tree.add_leaf([i as u8; 32]);
        }
        
        let root = tree.build();
        assert_ne!(root, [0; 32]);
        
        let proof = tree.proof(0);
        assert!(proof.is_some());
    }

    #[test]
    fn test_verify_aggregation() {
        let mut aggregator = ProofAggregator::new();
        
        for i in 0..4 {
            aggregator.add_proof(make_test_proof(i));
        }
        
        let agg = aggregator.aggregate_single().unwrap();
        assert!(aggregator.verify(&agg));
    }
}
