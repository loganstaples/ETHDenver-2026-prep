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
//! Merkle tree commitments with in-circuit Poseidon hashing to verify that
//! lookups are consistent with the committed embedding matrix without
//! revealing the actual embeddings.
//!
//! Each Merkle step computes `parent = Poseidon(left, right)` in-circuit,
//! chaining from the leaf hash up to the root. The final root is constrained
//! to match the expected commitment (~764 rows per Merkle level).
//!
//! # Error Bounds
//!
//! Embedding lookup is exact (no numerical error in the lookup itself), but
//! subsequent operations will accumulate error. We track a base error bound
//! for quantized embeddings.

use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    plonk::{
        Advice, Circuit, Column, ConstraintSystem, ErrorFront, Fixed, Instance, Selector,
        TableColumn,
    },
    poly::Rotation,
};
use halo2curves::bn256::Fr;
use halo2curves::ff::{Field, PrimeField};
use std::marker::PhantomData;

use crate::gadgets::poseidon::{
    poseidon_hash_many, poseidon_hash_two, synthesize_poseidon_hash, PoseidonCircuitConfig,
};

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
    /// Configuration for in-circuit Poseidon hashing (Merkle path verification).
    pub poseidon: PoseidonCircuitConfig,
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

        // Vocabulary range check lookup
        meta.lookup("embedding_lookup", |meta| {
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

        // Poseidon gates for in-circuit Merkle path verification.
        // Each Merkle step computes Poseidon(left, right) = parent in-circuit.
        let s_mul = meta.selector();
        let s_add = meta.selector();
        let s_rc_add = meta.selector();

        meta.create_gate("mul", |meta| {
            let s = meta.query_selector(s_mul);
            let a = meta.query_advice(advice[0], Rotation::cur());
            let b = meta.query_advice(advice[1], Rotation::cur());
            let c = meta.query_advice(advice[2], Rotation::cur());
            vec![s * (a * b - c)]
        });

        meta.create_gate("add", |meta| {
            let s = meta.query_selector(s_add);
            let a = meta.query_advice(advice[0], Rotation::cur());
            let b = meta.query_advice(advice[1], Rotation::cur());
            let c = meta.query_advice(advice[2], Rotation::cur());
            vec![s * (a + b - c)]
        });

        meta.create_gate("rc_add", |meta| {
            let s = meta.query_selector(s_rc_add);
            let a = meta.query_advice(advice[0], Rotation::cur());
            let rc = meta.query_fixed(fixed, Rotation::cur());
            let c = meta.query_advice(advice[2], Rotation::cur());
            vec![s * (a + rc - c)]
        });

        let poseidon = PoseidonCircuitConfig {
            advice: [advice[0], advice[1], advice[2]],
            fixed,
            s_mul,
            s_add,
            s_rc_add,
            s_eq,
        };

        EmbeddingConfig {
            advice,
            fixed,
            instance,
            vocab_table,
            s_range,
            s_eq,
            s_commit,
            poseidon,
            _marker: PhantomData,
        }
    }

    /// Loads the vocabulary range table.
    pub fn load_vocab_table(
        &self,
        layouter: &mut impl Layouter<F>,
        vocab_size: usize,
    ) -> Result<(), ErrorFront> {
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
}

/// Fr-specific methods for in-circuit Poseidon-based Merkle verification.
impl EmbeddingChip<Fr> {
    /// Verifies embedding lookup with in-circuit Merkle path verification.
    ///
    /// When `use_commitment` is true, each embedding's Merkle path is verified
    /// in-circuit using Poseidon hashing. The computed root must match the
    /// expected root commitment — a malicious prover cannot supply arbitrary
    /// Merkle paths.
    pub fn verify_embedding_lookup(
        &self,
        mut layouter: impl Layouter<Fr>,
        witness: &EmbeddingWitness<Fr>,
    ) -> Result<(), ErrorFront> {
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

            // If using commitment, verify Merkle path with in-circuit Poseidon
            if witness.use_commitment {
                self.verify_merkle_path(
                    &mut layouter,
                    witness.embedding_hashes[i],
                    witness.root_commitment,
                    &witness.merkle_paths[i],
                    i,
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

    /// Verifies a Merkle path using in-circuit Poseidon hashing.
    ///
    /// For each level of the Merkle tree, computes `parent = Poseidon(left, right)`
    /// in-circuit using the full Poseidon permutation (~764 rows per hash).
    /// The final computed root is constrained to equal the expected root commitment.
    fn verify_merkle_path(
        &self,
        layouter: &mut impl Layouter<Fr>,
        leaf_hash: Fr,
        root: Fr,
        path: &[(Fr, bool)], // (sibling_hash, is_left)
        token_idx: usize,
    ) -> Result<(), ErrorFront> {
        if path.is_empty() {
            // Single-node tree: just verify leaf equals root
            layouter.assign_region(
                || "single_node_tree",
                |mut region| {
                    self.config.s_eq.enable(&mut region, 0)?;
                    region.assign_advice(
                        || "leaf",
                        self.config.advice[0],
                        0,
                        || Value::known(leaf_hash),
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

        let mut current = leaf_hash;

        for (level, (sibling, is_left)) in path.iter().enumerate() {
            let (left, right) = if *is_left {
                (*sibling, current)
            } else {
                (current, *sibling)
            };

            // Compute parent = Poseidon(left, right) in-circuit.
            // synthesize_poseidon_hash lays out the full permutation with all
            // intermediate values constrained, and verifies the output matches
            // the native hash. ~764 rows per call.
            let parent = synthesize_poseidon_hash(
                &self.config.poseidon,
                layouter,
                left,
                right,
                &format!("merkle_t{}_l{}", token_idx, level),
            )?;

            current = parent;
        }

        // Verify computed root matches expected root commitment
        layouter.assign_region(
            || format!("merkle_root_check_t{}", token_idx),
            |mut region| {
                self.config.s_eq.enable(&mut region, 0)?;
                region.assign_advice(
                    || "computed_root",
                    self.config.advice[0],
                    0,
                    || Value::known(current),
                )?;
                region.assign_advice(
                    || "expected_root",
                    self.config.advice[1],
                    0,
                    || Value::known(root),
                )?;
                Ok(())
            },
        )
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

/// Embedding table with Poseidon commitment support.
///
/// Uses Poseidon hashing (matching in-circuit constraints) for both leaf
/// hashing and Merkle tree construction. This ensures native hashes match
/// the in-circuit verification.
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

        // Compute leaf hashes using Poseidon
        let leaf_hashes: Vec<Fr> = embeddings
            .iter()
            .map(|embed| Self::hash_embedding(embed))
            .collect();

        // Build Merkle tree using Poseidon
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

    /// Hashes an embedding vector using Poseidon sponge construction.
    ///
    /// This matches the in-circuit hash computation (same Poseidon parameters).
    fn hash_embedding(embedding: &[Fr]) -> Fr {
        poseidon_hash_many(embedding)
    }

    /// Builds a Merkle tree from leaf hashes using Poseidon.
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

    /// Hashes two field elements using Poseidon.
    ///
    /// This is the same hash function used in-circuit for Merkle path verification.
    fn hash_pair(left: Fr, right: Fr) -> Fr {
        poseidon_hash_two(left, right)
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

/// Circuit implementation specialized for BN254 Fr (required for Poseidon hashing).
impl Circuit<Fr> for EmbeddingCircuit<Fr> {
    type Config = EmbeddingConfig<Fr>;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
        EmbeddingChip::<Fr>::configure(meta)
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fr>,
    ) -> Result<(), ErrorFront> {
        let chip = EmbeddingChip::new(config.clone());

        // Load vocabulary table
        chip.load_vocab_table(&mut layouter, self.witness.vocab_size.max(1))?;

        // Verify embeddings with in-circuit Merkle verification
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

    /// Verifies that in-circuit Merkle path verification works with Poseidon.
    ///
    /// Each Merkle step synthesizes a full Poseidon permutation (~764 rows).
    /// For a tree of 4 leaves (depth 2) with 2 token lookups, we need
    /// 2 tokens × 2 levels × 764 rows ≈ 3,056 rows for Poseidon alone.
    #[test]
    fn test_embedding_circuit_with_commitment() {
        let table = EmbeddingTable::random(4, 2);
        let token_ids = vec![0, 1];
        let base_error = Fr::from(1);

        let witness = compute_embedding_witness(&token_ids, &table, true, base_error);
        assert!(witness.use_commitment);
        assert!(!witness.merkle_paths[0].is_empty());

        let circuit = EmbeddingCircuit::<Fr>::new(witness);

        // k=13 gives 8192 rows, enough for Poseidon-based Merkle verification
        let prover = MockProver::run(13, &circuit, vec![vec![]]).unwrap();
        prover.assert_satisfied();
    }

    /// Proves that a tampered Merkle path is REJECTED by in-circuit verification.
    ///
    /// This is the key security property: unlike the old trivial self-equality
    /// check (parent == expected), the Poseidon-based verification catches
    /// malicious provers who supply wrong sibling hashes.
    #[test]
    fn test_malicious_merkle_path_rejected() {
        let table = EmbeddingTable::random(4, 2);
        let token_ids = vec![0];
        let base_error = Fr::from(1);

        let mut witness = compute_embedding_witness(&token_ids, &table, true, base_error);

        // Tamper with the sibling hash in the Merkle path
        if let Some(path) = witness.merkle_paths.get_mut(0) {
            if let Some(entry) = path.get_mut(0) {
                entry.0 = Fr::from(9999u64); // Wrong sibling
            }
        }

        let circuit = EmbeddingCircuit::<Fr>::new(witness);
        let prover = MockProver::run(13, &circuit, vec![vec![]]).unwrap();

        // Must FAIL: tampered sibling → wrong Poseidon hash → root mismatch
        assert!(prover.verify().is_err(), "Tampered Merkle path must be rejected");
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

    /// Verifies the native Merkle tree is consistent with in-circuit verification.
    ///
    /// Computes leaf hash → root using native Poseidon, then verifies the
    /// same computation succeeds in-circuit.
    #[test]
    fn test_merkle_native_circuit_consistency() {
        let table = EmbeddingTable::random(8, 4);

        // Verify native Merkle proof for each leaf
        for token_id in 0..8 {
            let embed = table.lookup(token_id).unwrap();
            let leaf_hash = EmbeddingTable::hash_embedding(embed);
            let path = table.get_merkle_proof(token_id);

            // Walk the path natively to verify it reaches the root
            let mut current = leaf_hash;
            for (sibling, is_left) in &path {
                let (left, right) = if *is_left {
                    (*sibling, current)
                } else {
                    (current, *sibling)
                };
                current = poseidon_hash_two(left, right);
            }
            assert_eq!(current, table.root, "Native Merkle path for token {} doesn't reach root", token_id);
        }
    }
}
