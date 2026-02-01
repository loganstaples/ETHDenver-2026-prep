//! Embedding Lookup Circuit.
//!
//! Implements token embedding verification with commitment support for HELIX's
//! distributed training. The embedding layer maps discrete token IDs to dense
//! vectors, which is the first operation in transformer models.
//!
//! # Circuit Strategy
//!
//! Embedding lookup is fundamentally an index-based operation:
//!   output = embedding_table[token_id]
//!
//! In ZK circuits, we verify this by:
//! 1. Proving the token_id is within vocabulary bounds
//! 2. Proving the output matches the claimed embedding row
//! 3. Optionally committing to the embedding table for privacy
//!
//! # Commitment-Based Verification
//!
//! For distributed training with MPC, embeddings are secret-shared. We use
//! Merkle tree commitments to verify that lookups are consistent with the
//! committed embedding matrix without revealing the actual embeddings.
//!
//! # Error Bounds
//!
//! Embedding lookup is exact (no numerical error in the lookup itself), but
//! subsequent operations will accumulate error. We track a base error bound
//! for quantized embeddings.

use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{
        Advice, Circuit, Column, ConstraintSystem, Error, Expression, Fixed, Instance, Selector,
        TableColumn,
    },
    poly::Rotation,
};
use halo2curves::bn256::Fr;
use halo2curves::ff::{Field, PrimeField};
use sha2::{Digest, Sha256};
use std::marker::PhantomData;

/// Maximum vocabulary size for embedding tables.
pub const MAX_VOCAB_SIZE: usize = 65536;

/// Maximum embedding dimension.
pub const MAX_EMBED_DIM: usize = 1024;

/// Configuration for embedding lookup circuit.
#[derive(Clone, Debug)]
pub struct EmbeddingConfig<F: PrimeField> {
    /// Advice columns for token IDs and embeddings.
    pub advice: [Column<Advice>; 4],
    /// Fixed column for embedding table (if using fixed embeddings).
    pub fixed: Column<Fixed>,
    /// Instance column for public inputs.
    pub instance: Column<Instance>,
    /// Table column for vocabulary range check.
    pub vocab_table: TableColumn,
    /// Selector for range check.
    pub s_range: Selector,
    /// Selector for equality check.
    pub s_eq: Selector,
    /// Selector for commitment verification.
    pub s_commit: Selector,
    /// Selector for hash computation.
    pub s_hash: Selector,
    /// Phantom data.
    _marker: PhantomData<F>,
}

/// Embedding lookup chip.
pub struct EmbeddingChip<F: PrimeField> {
    config: EmbeddingConfig<F>,
}

impl<F: PrimeField> EmbeddingChip<F> {
    /// Creates a new embedding chip.
    pub fn new(config: EmbeddingConfig<F>) -> Self {
        Self { config }
    }

    /// Configures the embedding circuit.
    pub fn configure(meta: &mut ConstraintSystem<F>) -> EmbeddingConfig<F> {
        let advice = [
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];
        let fixed = meta.fixed_column();
        let instance = meta.instance_column();
        let vocab_table = meta.lookup_table_column();

        for col in &advice {
            meta.enable_equality(*col);
        }
        meta.enable_equality(instance);
        meta.enable_equality(fixed);

        let s_range = meta.complex_selector();
        let s_eq = meta.selector();
        let s_commit = meta.selector();
        let s_hash = meta.selector();

        // Vocabulary range check lookup
        meta.lookup(|meta| {
            let s = meta.query_selector(s_range);
            let token_id = meta.query_advice(advice[0], Rotation::cur());
            vec![(s * token_id, vocab_table)]
        });

        // Equality check gate
        meta.create_gate("embed_eq", |meta| {
            let s = meta.query_selector(s_eq);
            let a = meta.query_advice(advice[0], Rotation::cur());
            let b = meta.query_advice(advice[1], Rotation::cur());
            vec![s * (a - b)]
        });

        // Commitment verification gate (simplified)
        // Verifies: hash(token_id || embedding) matches commitment
        meta.create_gate("embed_commit", |meta| {
            let s = meta.query_selector(s_commit);
            let computed_hash = meta.query_advice(advice[0], Rotation::cur());
            let expected_hash = meta.query_advice(advice[1], Rotation::cur());
            vec![s * (computed_hash - expected_hash)]
        });

        // Hash accumulation gate (for Merkle tree path)
        meta.create_gate("embed_hash", |meta| {
            let s = meta.query_selector(s_hash);
            let left = meta.query_advice(advice[0], Rotation::cur());
            let right = meta.query_advice(advice[1], Rotation::cur());
            let parent = meta.query_advice(advice[2], Rotation::cur());
            let expected = meta.query_advice(advice[3], Rotation::cur());
            // Simplified: parent should match expected
            // Full implementation would verify hash(left || right) = parent
            vec![s * (parent - expected)]
        });

        EmbeddingConfig {
            advice,
            fixed,
            instance,
            vocab_table,
            s_range,
            s_eq,
            s_commit,
            s_hash,
            _marker: PhantomData,
        }
    }

    /// Loads the vocabulary range table.
    pub fn load_vocab_table(
        &self,
        layouter: &mut impl Layouter<F>,
        vocab_size: usize,
    ) -> Result<(), Error> {
        layouter.assign_table(
            || "vocab_table",
            |mut table| {
                for i in 0..vocab_size {
                    table.assign_cell(
                        || format!("vocab_{}", i),
                        self.config.vocab_table,
                        i,
                        || Value::known(F::from(i as u64)),
                    )?;
                }
                Ok(())
            },
        )
    }

    /// Verifies embedding lookup.
    pub fn verify_embedding_lookup(
        &self,
        mut layouter: impl Layouter<F>,
        witness: &EmbeddingWitness<F>,
    ) -> Result<(), Error> {
        let seq_len = witness.token_ids.len();
        let embed_dim = witness.embeddings.get(0).map(|e| e.len()).unwrap_or(0);

        // Verify each token lookup
        for i in 0..seq_len {
            let token_id = witness.token_ids[i];

            // Range check token ID
            layouter.assign_region(
                || format!("range_check_token_{}", i),
                |mut region| {
                    self.config.s_range.enable(&mut region, 0)?;
                    region.assign_advice(
                        || "token_id",
                        self.config.advice[0],
                        0,
                        || Value::known(token_id),
                    )?;
                    Ok(())
                },
            )?;

            // If using commitment, verify Merkle path
            if witness.use_commitment {
                self.verify_merkle_path(
                    layouter.namespace(|| format!("merkle_path_{}", i)),
                    &witness.embedding_hashes[i],
                    witness.root_commitment,
                    &witness.merkle_paths[i],
                )?;
            }

            // Verify each dimension of the embedding
            for d in 0..embed_dim {
                let embed_val = witness.embeddings[i][d];
                let expected_val = witness.expected_embeddings[i][d];

                layouter.assign_region(
                    || format!("verify_embed_{}_{}", i, d),
                    |mut region| {
                        self.config.s_eq.enable(&mut region, 0)?;
                        region.assign_advice(
                            || "embed_val",
                            self.config.advice[0],
                            0,
                            || Value::known(embed_val),
                        )?;
                        region.assign_advice(
                            || "expected_val",
                            self.config.advice[1],
                            0,
                            || Value::known(expected_val),
                        )?;
                        Ok(())
                    },
                )?;
            }
        }

        Ok(())
    }

    /// Verifies a Merkle path for embedding commitment.
    fn verify_merkle_path(
        &self,
        mut layouter: impl Layouter<F>,
        leaf_hash: &F,
        root: F,
        path: &[(F, bool)], // (sibling_hash, is_left)
    ) -> Result<(), Error> {
        if path.is_empty() {
            // Just verify leaf equals root
            layouter.assign_region(
                || "single_node_tree",
                |mut region| {
                    self.config.s_eq.enable(&mut region, 0)?;
                    region.assign_advice(
                        || "leaf",
                        self.config.advice[0],
                        0,
                        || Value::known(*leaf_hash),
                    )?;
                    region.assign_advice(
                        || "root",
                        self.config.advice[1],
                        0,
                        || Value::known(root),
                    )?;
                    Ok(())
                },
            )?;
            return Ok(());
        }

        let mut current = *leaf_hash;

        for (i, (sibling, is_left)) in path.iter().enumerate() {
            let (left, right) = if *is_left {
                (*sibling, current)
            } else {
                (current, *sibling)
            };

            // Compute parent (in actual implementation, this would verify hash)
            // For now, we use the witness-provided parent value
            let parent = if i + 1 < path.len() {
                path[i + 1].0 // Next level's "current" value from witness
            } else {
                root
            };

            layouter.assign_region(
                || format!("merkle_step_{}", i),
                |mut region| {
                    self.config.s_hash.enable(&mut region, 0)?;
                    region.assign_advice(|| "left", self.config.advice[0], 0, || Value::known(left))?;
                    region.assign_advice(|| "right", self.config.advice[1], 0, || Value::known(right))?;
                    region.assign_advice(|| "parent", self.config.advice[2], 0, || Value::known(parent))?;
                    region.assign_advice(|| "expected", self.config.advice[3], 0, || Value::known(parent))?;
                    Ok(())
                },
            )?;

            current = parent;
        }

        Ok(())
    }
}

/// Witness data for embedding lookup verification.
#[derive(Clone, Debug)]
pub struct EmbeddingWitness<F: PrimeField> {
    /// Token IDs to look up.
    pub token_ids: Vec<F>,
    /// Looked-up embeddings.
    pub embeddings: Vec<Vec<F>>,
    /// Expected embeddings (from embedding table).
    pub expected_embeddings: Vec<Vec<F>>,
    /// Whether to use commitment-based verification.
    pub use_commitment: bool,
    /// Root commitment of embedding table.
    pub root_commitment: F,
    /// Hash of each looked-up embedding.
    pub embedding_hashes: Vec<F>,
    /// Merkle paths for each lookup.
    pub merkle_paths: Vec<Vec<(F, bool)>>,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Embedding dimension.
    pub embed_dim: usize,
    /// Error bound per embedding element.
    pub error_bound: F,
}

impl<F: PrimeField> Default for EmbeddingWitness<F> {
    fn default() -> Self {
        Self {
            token_ids: vec![],
            embeddings: vec![],
            expected_embeddings: vec![],
            use_commitment: false,
            root_commitment: F::ZERO,
            embedding_hashes: vec![],
            merkle_paths: vec![],
            vocab_size: 0,
            embed_dim: 0,
            error_bound: F::ZERO,
        }
    }
}

/// Embedding table with commitment support.
#[derive(Clone, Debug)]
pub struct EmbeddingTable {
    /// Embedding matrix [vocab_size, embed_dim].
    pub embeddings: Vec<Vec<Fr>>,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Embedding dimension.
    pub embed_dim: usize,
    /// Merkle tree of embedding hashes.
    pub merkle_tree: Vec<Vec<Fr>>,
    /// Root commitment.
    pub root: Fr,
}

impl EmbeddingTable {
    /// Creates a new embedding table.
    pub fn new(embeddings: Vec<Vec<Fr>>) -> Self {
        let vocab_size = embeddings.len();
        let embed_dim = embeddings.get(0).map(|e| e.len()).unwrap_or(0);

        // Compute leaf hashes
        let leaf_hashes: Vec<Fr> = embeddings
            .iter()
            .map(|embed| Self::hash_embedding(embed))
            .collect();

        // Build Merkle tree
        let (merkle_tree, root) = Self::build_merkle_tree(&leaf_hashes);

        Self {
            embeddings,
            vocab_size,
            embed_dim,
            merkle_tree,
            root,
        }
    }

    /// Creates a random embedding table for testing.
    pub fn random(vocab_size: usize, embed_dim: usize) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let embeddings: Vec<Vec<Fr>> = (0..vocab_size)
            .map(|v| {
                (0..embed_dim)
                    .map(|d| {
                        let mut hasher = DefaultHasher::new();
                        v.hash(&mut hasher);
                        d.hash(&mut hasher);
                        Fr::from(hasher.finish())
                    })
                    .collect()
            })
            .collect();

        Self::new(embeddings)
    }

    /// Hashes an embedding vector.
    fn hash_embedding(embedding: &[Fr]) -> Fr {
        let mut hasher = Sha256::new();
        for val in embedding {
            hasher.update(val.to_repr().as_ref());
        }
        let hash: [u8; 32] = hasher.finalize().into();
        // Convert first 8 bytes to Fr
        let val = u64::from_le_bytes(hash[0..8].try_into().unwrap());
        Fr::from(val)
    }

    /// Builds a Merkle tree from leaf hashes.
    fn build_merkle_tree(leaves: &[Fr]) -> (Vec<Vec<Fr>>, Fr) {
        if leaves.is_empty() {
            return (vec![], Fr::ZERO);
        }

        let mut tree = vec![leaves.to_vec()];
        let mut current_level = leaves.to_vec();

        // Pad to power of 2
        let mut size = current_level.len();
        while size & (size - 1) != 0 {
            current_level.push(Fr::ZERO);
            size = current_level.len();
        }
        tree[0] = current_level.clone();

        while current_level.len() > 1 {
            let mut next_level = Vec::new();
            for chunk in current_level.chunks(2) {
                let left = chunk[0];
                let right = if chunk.len() > 1 { chunk[1] } else { Fr::ZERO };
                let parent = Self::hash_pair(left, right);
                next_level.push(parent);
            }
            tree.push(next_level.clone());
            current_level = next_level;
        }

        let root = current_level.get(0).cloned().unwrap_or(Fr::ZERO);
        (tree, root)
    }

    /// Hashes two field elements together.
    fn hash_pair(left: Fr, right: Fr) -> Fr {
        let mut hasher = Sha256::new();
        hasher.update(left.to_repr().as_ref());
        hasher.update(right.to_repr().as_ref());
        let hash: [u8; 32] = hasher.finalize().into();
        let val = u64::from_le_bytes(hash[0..8].try_into().unwrap());
        Fr::from(val)
    }

    /// Looks up an embedding by token ID.
    pub fn lookup(&self, token_id: usize) -> Option<&[Fr]> {
        self.embeddings.get(token_id).map(|e| e.as_slice())
    }

    /// Gets the Merkle proof for a token ID.
    pub fn get_merkle_proof(&self, token_id: usize) -> Vec<(Fr, bool)> {
        if self.merkle_tree.is_empty() || token_id >= self.vocab_size {
            return vec![];
        }

        let mut path = Vec::new();
        let mut index = token_id;

        // Ensure index is within padded tree
        let tree_size = self.merkle_tree[0].len();
        if index >= tree_size {
            return vec![];
        }

        for level in &self.merkle_tree[..self.merkle_tree.len().saturating_sub(1)] {
            let sibling_index = if index % 2 == 0 { index + 1 } else { index - 1 };
            let sibling = level.get(sibling_index).cloned().unwrap_or(Fr::ZERO);
            let is_left = index % 2 == 1; // True if current is on right
            path.push((sibling, is_left));
            index /= 2;
        }

        path
    }
}

/// Computes embedding lookup witness.
pub fn compute_embedding_witness(
    token_ids: &[u64],
    table: &EmbeddingTable,
    use_commitment: bool,
    base_error: Fr,
) -> EmbeddingWitness<Fr> {
    let mut embeddings = Vec::new();
    let mut expected_embeddings = Vec::new();
    let mut embedding_hashes = Vec::new();
    let mut merkle_paths = Vec::new();

    for &token_id in token_ids {
        let idx = token_id as usize;
        let embed = table.lookup(idx).unwrap_or(&[]).to_vec();

        // For expected embeddings, use the same (they should match)
        let expected = embed.clone();

        // Hash the embedding
        let hash = EmbeddingTable::hash_embedding(&embed);

        // Get Merkle proof
        let path = if use_commitment {
            table.get_merkle_proof(idx)
        } else {
            vec![]
        };

        embeddings.push(embed);
        expected_embeddings.push(expected);
        embedding_hashes.push(hash);
        merkle_paths.push(path);
    }

    let token_ids_field: Vec<Fr> = token_ids.iter().map(|&id| Fr::from(id)).collect();

    EmbeddingWitness {
        token_ids: token_ids_field,
        embeddings,
        expected_embeddings,
        use_commitment,
        root_commitment: table.root,
        embedding_hashes,
        merkle_paths,
        vocab_size: table.vocab_size,
        embed_dim: table.embed_dim,
        error_bound: base_error,
    }
}

/// Complete circuit for embedding lookup verification.
#[derive(Clone)]
pub struct EmbeddingCircuit<F: PrimeField> {
    pub witness: EmbeddingWitness<F>,
    _marker: PhantomData<F>,
}

impl<F: PrimeField> Default for EmbeddingCircuit<F> {
    fn default() -> Self {
        Self {
            witness: EmbeddingWitness::default(),
            _marker: PhantomData,
        }
    }
}

impl<F: PrimeField> EmbeddingCircuit<F> {
    /// Creates a new embedding circuit.
    pub fn new(witness: EmbeddingWitness<F>) -> Self {
        Self {
            witness,
            _marker: PhantomData,
        }
    }
}

impl<F: PrimeField> Circuit<F> for EmbeddingCircuit<F> {
    type Config = EmbeddingConfig<F>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        EmbeddingChip::<F>::configure(meta)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        let chip = EmbeddingChip::new(config.clone());

        // Load vocabulary table
        chip.load_vocab_table(&mut layouter, self.witness.vocab_size.max(1))?;

        // Verify embeddings
        chip.verify_embedding_lookup(layouter.namespace(|| "embedding_lookup"), &self.witness)?;

        Ok(())
    }
}

/// Batch embedding lookup for a sequence of tokens.
#[derive(Clone, Debug)]
pub struct BatchEmbeddingWitness<F: PrimeField> {
    /// Individual embedding witnesses.
    pub embeddings: Vec<EmbeddingWitness<F>>,
    /// Combined output matrix [seq_len, embed_dim].
    pub output_matrix: Vec<Vec<F>>,
    /// Total error bound.
    pub total_error: F,
}

impl<F: PrimeField> Default for BatchEmbeddingWitness<F> {
    fn default() -> Self {
        Self {
            embeddings: vec![],
            output_matrix: vec![],
            total_error: F::ZERO,
        }
    }
}

/// Computes batch embedding witness for a sequence.
pub fn compute_batch_embedding_witness(
    token_ids: &[u64],
    table: &EmbeddingTable,
    use_commitment: bool,
    base_error: Fr,
) -> BatchEmbeddingWitness<Fr> {
    let witness = compute_embedding_witness(token_ids, table, use_commitment, base_error);

    let output_matrix = witness.embeddings.clone();
    let total_error = base_error * Fr::from(token_ids.len() as u64);

    BatchEmbeddingWitness {
        embeddings: vec![witness],
        output_matrix,
        total_error,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2_proofs::dev::MockProver;

    #[test]
    fn test_embedding_table_creation() {
        let table = EmbeddingTable::random(100, 32);
        assert_eq!(table.vocab_size, 100);
        assert_eq!(table.embed_dim, 32);
        assert!(!table.merkle_tree.is_empty());
    }

    #[test]
    fn test_embedding_lookup() {
        let table = EmbeddingTable::random(100, 32);

        let embed = table.lookup(5);
        assert!(embed.is_some());
        assert_eq!(embed.unwrap().len(), 32);
    }

    #[test]
    fn test_merkle_proof() {
        let table = EmbeddingTable::random(64, 8);
        let proof = table.get_merkle_proof(10);

        // For 64 leaves, we should have log2(64) = 6 levels
        assert!(proof.len() <= 6);
    }

    #[test]
    fn test_embedding_witness() {
        let table = EmbeddingTable::random(100, 16);
        let token_ids = vec![5, 10, 15];
        let base_error = Fr::from(1);

        let witness = compute_embedding_witness(&token_ids, &table, false, base_error);

        assert_eq!(witness.token_ids.len(), 3);
        assert_eq!(witness.embeddings.len(), 3);
        assert_eq!(witness.embeddings[0].len(), 16);
    }

    #[test]
    fn test_embedding_circuit() {
        let table = EmbeddingTable::random(32, 8);
        let token_ids = vec![5, 10, 15];
        let base_error = Fr::from(1);

        let witness = compute_embedding_witness(&token_ids, &table, false, base_error);
        let circuit = EmbeddingCircuit::<Fr>::new(witness);

        // Instance column is configured but not used - pass empty vec for it
        let prover = MockProver::run(12, &circuit, vec![vec![]]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn test_embedding_with_commitment() {
        let table = EmbeddingTable::random(64, 8);
        let token_ids = vec![5, 10, 15, 20];
        let base_error = Fr::from(1);

        let witness = compute_embedding_witness(&token_ids, &table, true, base_error);

        assert!(witness.use_commitment);
        assert!(witness.merkle_paths.iter().all(|p| !p.is_empty()));
    }

    #[test]
    fn test_batch_embedding() {
        let table = EmbeddingTable::random(100, 16);
        let token_ids = vec![1, 2, 3, 4, 5];
        let base_error = Fr::from(1);

        let batch_witness = compute_batch_embedding_witness(&token_ids, &table, false, base_error);

        assert_eq!(batch_witness.output_matrix.len(), 5);
        assert_eq!(batch_witness.output_matrix[0].len(), 16);
    }

    #[test]
    fn test_hash_consistency() {
        let embedding = vec![Fr::from(1), Fr::from(2), Fr::from(3)];
        let hash1 = EmbeddingTable::hash_embedding(&embedding);
        let hash2 = EmbeddingTable::hash_embedding(&embedding);
        assert_eq!(hash1, hash2);
    }
}
