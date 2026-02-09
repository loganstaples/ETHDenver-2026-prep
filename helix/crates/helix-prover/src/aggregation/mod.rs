//! Proof Aggregation.
//!
//! Combines multiple proofs into a single aggregated proof.
//! Essential for recursive proving and reducing verification costs.
//!
//! ## Submodules
//!
//! - `gkr_to_halo2`: Aggregates GKR proofs into Halo2 proofs for on-chain verification

pub mod gkr_to_halo2;

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use tracing;

use super::chunking::ChunkId;
use super::parallel::ChunkProof;

// Re-export GKR aggregation types
pub use gkr_to_halo2::{
    GKRToHalo2Aggregator, AggregatedGKRProof, GKRProofCommitment,
    AggregationConfig as GKRAggregationConfig, BatchAggregator,
};

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

    /// Verifies an aggregated proof by checking:
    /// 1. The proof is non-empty and has the correct format header.
    /// 2. The embedded Merkle root matches the `root_commitment`.
    /// 3. Each chunk's Merkle path is valid against the root.
    /// 4. The independently computed root from `public_inputs` matches.
    pub fn verify(&self, agg: &AggregatedProof) -> bool {
        if agg.proof.is_empty() {
            tracing::warn!("verify: empty aggregated proof");
            return false;
        }

        // --- Parse the aggregated proof format ---
        // Layout: [version:1] [num_proofs:4 LE] [merkle_root:32] [paths...]
        let proof = &agg.proof;
        if proof.len() < 37 {
            tracing::warn!("verify: aggregated proof too short ({} bytes)", proof.len());
            return false;
        }

        let version = proof[0];
        if version != 2 && version != 3 {
            tracing::warn!("verify: unexpected aggregation version {version}");
            return false;
        }

        let num_proofs = u32::from_le_bytes(proof[1..5].try_into().unwrap()) as usize;
        if num_proofs != agg.chunk_ids.len() {
            tracing::warn!(
                "verify: proof claims {} chunks but aggregation has {}",
                num_proofs,
                agg.chunk_ids.len()
            );
            return false;
        }

        let embedded_root: [u8; 32] = proof[5..37].try_into().unwrap();
        if embedded_root != agg.root_commitment {
            tracing::warn!("verify: embedded Merkle root does not match root_commitment");
            return false;
        }

        // --- For version 3, skip past embedded chunk proofs section ---
        let mut offset = 37;
        if version == 3 {
            for _ in 0..num_proofs {
                if offset + 8 > proof.len() {
                    tracing::warn!("verify: v3 proof truncated at chunk id");
                    return false;
                }
                offset += 8; // chunk_id

                if offset + 4 > proof.len() {
                    tracing::warn!("verify: v3 proof truncated at proof length");
                    return false;
                }
                let proof_len = u32::from_le_bytes(
                    proof[offset..offset + 4].try_into().unwrap(),
                ) as usize;
                offset += 4;

                if offset + proof_len > proof.len() {
                    tracing::warn!("verify: v3 proof truncated at proof data");
                    return false;
                }
                // Verify embedded proof is non-empty
                if proof_len == 0 {
                    tracing::warn!("verify: v3 embedded proof is empty");
                    return false;
                }
                offset += proof_len;

                if offset + 4 > proof.len() {
                    tracing::warn!("verify: v3 proof truncated at PI count");
                    return false;
                }
                let num_pis = u32::from_le_bytes(
                    proof[offset..offset + 4].try_into().unwrap(),
                ) as usize;
                offset += 4;
                offset += num_pis * 32;

                if offset + 8 > proof.len() {
                    tracing::warn!("verify: v3 proof truncated at error bound");
                    return false;
                }
                offset += 8; // error_bound f64
            }
        }

        // --- Verify each public input's Merkle path ---
        let num_leaves = agg.public_inputs.len();
        for leaf_idx in 0..num_leaves {
            if offset + 4 > proof.len() {
                tracing::warn!("verify: proof truncated at leaf {leaf_idx} path length");
                return false;
            }
            let path_len = u32::from_le_bytes(
                proof[offset..offset + 4].try_into().unwrap(),
            ) as usize;
            offset += 4;

            let path_bytes_needed = path_len * 32;
            if offset + path_bytes_needed > proof.len() {
                tracing::warn!(
                    "verify: proof truncated at leaf {leaf_idx} path data (need {path_bytes_needed}, have {})",
                    proof.len() - offset,
                );
                return false;
            }

            let mut path: Vec<[u8; 32]> = Vec::with_capacity(path_len);
            for i in 0..path_len {
                let start = offset + i * 32;
                let node: [u8; 32] = proof[start..start + 32].try_into().unwrap();
                path.push(node);
            }
            offset += path_bytes_needed;

            let leaf = agg.public_inputs[leaf_idx];
            if !verify_merkle_path(&leaf, leaf_idx, &path, &embedded_root) {
                tracing::warn!("verify: Merkle path verification failed for leaf {leaf_idx}");
                return false;
            }
        }

        // --- Cross-check: independently computed root from public_inputs ---
        let computed_root = self.compute_root(&agg.public_inputs);
        if computed_root != agg.root_commitment {
            tracing::warn!("verify: recomputed root from public_inputs does not match root_commitment");
            return false;
        }

        true
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

        // Generate aggregated proof with Merkle tree structure
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
        let mut tree = CommitmentTree::new();
        for input in inputs {
            tree.add_leaf(*input);
        }
        tree.build()
    }

    fn generate_aggregated_proof(&self, proofs: &[ChunkProof]) -> Vec<u8> {
        // Build Merkle tree over chunk public inputs (matching compute_root).
        let mut tree = CommitmentTree::new();
        for proof in proofs {
            for pi in &proof.public_inputs {
                tree.add_leaf(*pi);
            }
        }
        let root = tree.build();

        let mut aggregated = Vec::new();

        // Header: version 3 = self-contained aggregation with embedded proofs
        aggregated.push(3u8);
        aggregated.extend_from_slice(&(proofs.len() as u32).to_le_bytes());

        // Merkle root
        aggregated.extend_from_slice(&root);

        // Section 1: Embedded chunk proofs (makes the aggregation self-contained)
        for proof in proofs {
            // Chunk ID
            aggregated.extend_from_slice(&proof.chunk_id.0.to_le_bytes());
            // Proof bytes (length-prefixed)
            aggregated.extend_from_slice(&(proof.proof.len() as u32).to_le_bytes());
            aggregated.extend_from_slice(&proof.proof);
            // Number of public inputs
            aggregated.extend_from_slice(&(proof.public_inputs.len() as u32).to_le_bytes());
            for pi in &proof.public_inputs {
                aggregated.extend_from_slice(pi);
            }
            // Error bound (f64)
            aggregated.extend_from_slice(&proof.error_bound.to_le_bytes());
        }

        // Section 2: Merkle paths for each public input leaf
        let mut leaf_idx = 0;
        for proof in proofs {
            for _pi in &proof.public_inputs {
                if let Some(path) = tree.proof(leaf_idx) {
                    aggregated.extend_from_slice(&(path.len() as u32).to_le_bytes());
                    for node in &path {
                        aggregated.extend_from_slice(node);
                    }
                } else {
                    aggregated.extend_from_slice(&0u32.to_le_bytes());
                }
                leaf_idx += 1;
            }
        }

        aggregated
    }

    /// Extracts the embedded chunk proofs from an aggregated proof.
    ///
    /// Returns `None` if the proof format is invalid or not version 3.
    pub fn extract_chunk_proofs(agg: &AggregatedProof) -> Option<Vec<ChunkProof>> {
        let proof = &agg.proof;
        if proof.len() < 37 {
            return None;
        }

        let version = proof[0];
        if version != 3 {
            return None;
        }

        let num_proofs = u32::from_le_bytes(proof[1..5].try_into().ok()?) as usize;
        let mut offset = 37; // past header + root

        let mut chunk_proofs = Vec::with_capacity(num_proofs);
        for _ in 0..num_proofs {
            if offset + 8 > proof.len() {
                return None;
            }
            let chunk_id = u64::from_le_bytes(proof[offset..offset + 8].try_into().ok()?);
            offset += 8;

            if offset + 4 > proof.len() {
                return None;
            }
            let proof_len = u32::from_le_bytes(proof[offset..offset + 4].try_into().ok()?) as usize;
            offset += 4;

            if offset + proof_len > proof.len() {
                return None;
            }
            let proof_bytes = proof[offset..offset + proof_len].to_vec();
            offset += proof_len;

            if offset + 4 > proof.len() {
                return None;
            }
            let num_pis = u32::from_le_bytes(proof[offset..offset + 4].try_into().ok()?) as usize;
            offset += 4;

            let mut public_inputs = Vec::with_capacity(num_pis);
            for _ in 0..num_pis {
                if offset + 32 > proof.len() {
                    return None;
                }
                let pi: [u8; 32] = proof[offset..offset + 32].try_into().ok()?;
                public_inputs.push(pi);
                offset += 32;
            }

            if offset + 8 > proof.len() {
                return None;
            }
            let error_bound = f64::from_le_bytes(proof[offset..offset + 8].try_into().ok()?);
            offset += 8;

            chunk_proofs.push(ChunkProof {
                chunk_id: ChunkId(chunk_id),
                proof: proof_bytes,
                public_inputs,
                error_bound,
                generation_time_ms: 0,
            });
        }

        Some(chunk_proofs)
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

    pub fn hash_pair(a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(a);
        hasher.update(b);
        hasher.finalize().into()
    }
}

impl Default for CommitmentTree {
    fn default() -> Self {
        Self::new()
    }
}

/// Verifies a Merkle proof path from leaf to root.
///
/// `leaf_index` determines which side of each hash the accumulator is placed on
/// (left if even, right if odd), walking up the tree level-by-level.
pub fn verify_merkle_path(
    leaf: &[u8; 32],
    leaf_index: usize,
    path: &[[u8; 32]],
    expected_root: &[u8; 32],
) -> bool {
    let mut current = *leaf;
    let mut index = leaf_index;

    for sibling in path {
        current = if index % 2 == 0 {
            CommitmentTree::hash_pair(&current, sibling)
        } else {
            CommitmentTree::hash_pair(sibling, &current)
        };
        index /= 2;
    }

    current == *expected_root
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_aggregated_proof_is_version_3() {
        let mut aggregator = ProofAggregator::new();
        for i in 0..3 {
            aggregator.add_proof(make_test_proof(i));
        }
        let agg = aggregator.aggregate_single().unwrap();
        assert_eq!(agg.proof[0], 3, "aggregated proof should be version 3");
    }

    #[test]
    fn test_extract_chunk_proofs() {
        let mut aggregator = ProofAggregator::new();
        for i in 0..4 {
            aggregator.add_proof(make_test_proof(i));
        }
        let agg = aggregator.aggregate_single().unwrap();

        let extracted = ProofAggregator::extract_chunk_proofs(&agg).unwrap();
        assert_eq!(extracted.len(), 4);
        for (i, cp) in extracted.iter().enumerate() {
            assert_eq!(cp.chunk_id.0, i as u64);
            assert_eq!(cp.proof, vec![1, 2, 3, 4]);
            assert_eq!(cp.public_inputs, vec![[i as u8; 32]]);
            assert!((cp.error_bound - 0.01).abs() < 1e-10);
        }
    }

    #[test]
    fn test_recursive_aggregation_v3() {
        let mut aggregator = ProofAggregator::with_config(super::AggregationConfig {
            max_proofs_per_aggregation: 2,
            recursive: true,
            target_depth: 2,
            ..Default::default()
        });

        for i in 0..6 {
            aggregator.add_proof(make_test_proof(i));
        }

        let ids = aggregator.aggregate_all();
        assert!(ids.len() > 1, "recursive aggregation should produce multiple IDs");

        for id in &ids {
            let agg = aggregator.get_aggregation(*id).unwrap();
            assert!(aggregator.verify(agg), "each aggregation level should verify");
        }
    }

    #[test]
    fn test_verify_rejects_empty_proof() {
        let aggregator = ProofAggregator::new();
        let agg = AggregatedProof {
            id: AggregationId(0),
            root_commitment: [0; 32],
            proof: vec![],
            public_inputs: vec![],
            total_error_bound: 0.0,
            chunk_ids: vec![],
            num_layers: 0,
            depth: 0,
        };
        assert!(!aggregator.verify(&agg));
    }

    #[test]
    fn test_verify_rejects_wrong_root() {
        let mut aggregator = ProofAggregator::new();
        for i in 0..2 {
            aggregator.add_proof(make_test_proof(i));
        }
        let mut agg = aggregator.aggregate_single().unwrap();
        agg.root_commitment = [0xFF; 32]; // corrupt
        assert!(!aggregator.verify(&agg));
    }
}
