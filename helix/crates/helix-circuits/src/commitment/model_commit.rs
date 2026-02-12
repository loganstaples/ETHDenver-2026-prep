//! Merkle commitment to model weights using Poseidon and SHA-256 hashing.
//!
//! Provides both:
//! - **Native SHA-256**: For out-of-circuit model weight commitments (256-bit hash)
//! - **In-circuit Poseidon**: For ZK-verified Merkle path verification
//!
//! The `ModelCommitChip` implements in-circuit Merkle path verification using
//! Poseidon hashing (~764 rows per Merkle level). Each step constrains
//! `parent = Poseidon(left, right)` and the final root is checked against
//! the expected commitment.

use halo2_proofs::{
    circuit::{Layouter, Value},
    plonk::{Advice, Column, ConstraintSystem, ErrorFront, Instance},
};
use halo2curves::bn256::Fr;
use halo2curves::ff::Field;
use sha2::{Sha256, Digest};

use crate::gadgets::poseidon::{
    poseidon_hash_two, poseidon_hash_many, synthesize_poseidon_hash, PoseidonCircuitConfig,
};

/// Configuration for the ModelCommit chip.
#[derive(Clone, Debug)]
pub struct ModelCommitConfig {
    pub advice: [Column<Advice>; 3],
    pub instance: Column<Instance>,
    pub poseidon: PoseidonCircuitConfig,
}

/// Chip for proving membership in a Merkle tree of model weights.
///
/// Uses Poseidon hashing for in-circuit Merkle path verification.
/// Each `verify_path` call synthesizes the full Poseidon permutation
/// at each tree level, constraining `parent = Poseidon(left, right)`.
pub struct ModelCommitChip {
    config: ModelCommitConfig,
}

impl ModelCommitChip {
    pub fn configure(
        meta: &mut ConstraintSystem<Fr>,
        instance: Column<Instance>,
    ) -> ModelCommitConfig {
        let advice = [
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];
        let fixed = meta.fixed_column();

        for col in &advice {
            meta.enable_equality(*col);
        }
        meta.enable_equality(instance);
        meta.enable_equality(fixed);

        // Poseidon gates
        let s_mul = meta.selector();
        let s_add = meta.selector();
        let s_rc_add = meta.selector();
        let s_eq = meta.selector();

        meta.create_gate("mul", |meta| {
            let s = meta.query_selector(s_mul);
            let a = meta.query_advice(advice[0], halo2_proofs::poly::Rotation::cur());
            let b = meta.query_advice(advice[1], halo2_proofs::poly::Rotation::cur());
            let c = meta.query_advice(advice[2], halo2_proofs::poly::Rotation::cur());
            vec![s * (a * b - c)]
        });

        meta.create_gate("add", |meta| {
            let s = meta.query_selector(s_add);
            let a = meta.query_advice(advice[0], halo2_proofs::poly::Rotation::cur());
            let b = meta.query_advice(advice[1], halo2_proofs::poly::Rotation::cur());
            let c = meta.query_advice(advice[2], halo2_proofs::poly::Rotation::cur());
            vec![s * (a + b - c)]
        });

        meta.create_gate("rc_add", |meta| {
            let s = meta.query_selector(s_rc_add);
            let a = meta.query_advice(advice[0], halo2_proofs::poly::Rotation::cur());
            let rc = meta.query_fixed(fixed, halo2_proofs::poly::Rotation::cur());
            let c = meta.query_advice(advice[2], halo2_proofs::poly::Rotation::cur());
            vec![s * (a + rc - c)]
        });

        meta.create_gate("eq", |meta| {
            let s = meta.query_selector(s_eq);
            let a = meta.query_advice(advice[0], halo2_proofs::poly::Rotation::cur());
            let b = meta.query_advice(advice[1], halo2_proofs::poly::Rotation::cur());
            vec![s * (a - b)]
        });

        let poseidon = PoseidonCircuitConfig {
            advice,
            fixed,
            s_mul,
            s_add,
            s_rc_add,
            s_eq,
        };

        ModelCommitConfig {
            advice,
            instance,
            poseidon,
        }
    }

    pub fn new(config: ModelCommitConfig) -> Self {
        Self { config }
    }

    /// Verifies that a leaf is in the Merkle tree at the given path.
    ///
    /// Uses in-circuit Poseidon hashing to compute each Merkle tree level:
    /// `parent = Poseidon(left, right)`. The final computed root is constrained
    /// to equal the expected root.
    ///
    /// Each Merkle level requires ~764 rows for the Poseidon permutation.
    pub fn verify_path(
        &self,
        layouter: &mut impl Layouter<Fr>,
        leaf: Fr,
        root: Fr,
        path_elements: &[Fr],
        path_indices: &[bool],
    ) -> Result<(), ErrorFront> {
        assert_eq!(
            path_elements.len(),
            path_indices.len(),
            "path_elements and path_indices must have the same length"
        );

        if path_elements.is_empty() {
            // Single-node tree: verify leaf == root
            layouter.assign_region(
                || "single_node_commit",
                |mut region| {
                    self.config.poseidon.s_eq.enable(&mut region, 0)?;
                    region.assign_advice(
                        || "leaf",
                        self.config.advice[0],
                        0,
                        || Value::known(leaf),
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

        let mut current = leaf;

        for (level, (sibling, is_left)) in
            path_elements.iter().zip(path_indices.iter()).enumerate()
        {
            let (left, right) = if *is_left {
                (*sibling, current)
            } else {
                (current, *sibling)
            };

            // Compute parent = Poseidon(left, right) in-circuit
            let parent = synthesize_poseidon_hash(
                &self.config.poseidon,
                layouter,
                left,
                right,
                &format!("commit_merkle_l{}", level),
            )?;

            current = parent;
        }

        // Verify computed root matches expected
        layouter.assign_region(
            || "commit_root_check",
            |mut region| {
                self.config.poseidon.s_eq.enable(&mut region, 0)?;
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

/// Computes a SHA-256 commitment of model weights (native, not in-circuit).
///
/// This is the production-ready native hash used for weight commitments.
/// Returns (lo, hi) as two 128-bit halves of the 256-bit hash.
pub fn compute_model_commitment(weights: &[u8]) -> ([u8; 16], [u8; 16]) {
    let hash = Sha256::digest(weights);
    let mut lo = [0u8; 16];
    let mut hi = [0u8; 16];
    lo.copy_from_slice(&hash[..16]);
    hi.copy_from_slice(&hash[16..]);
    (lo, hi)
}

/// Computes a SHA-256 Merkle root from leaf hashes (native, not in-circuit).
pub fn compute_merkle_root(leaves: &[[u8; 32]]) -> [u8; 32] {
    if leaves.is_empty() {
        return [0u8; 32];
    }
    if leaves.len() == 1 {
        return leaves[0];
    }

    let mut current_level: Vec<[u8; 32]> = leaves.to_vec();
    while current_level.len() > 1 {
        let mut next_level = Vec::new();
        for pair in current_level.chunks(2) {
            let mut hasher = Sha256::new();
            hasher.update(&pair[0]);
            if pair.len() > 1 {
                hasher.update(&pair[1]);
            } else {
                hasher.update(&pair[0]); // duplicate last for odd count
            }
            let hash: [u8; 32] = hasher.finalize().into();
            next_level.push(hash);
        }
        current_level = next_level;
    }
    current_level[0]
}

/// Computes a Poseidon Merkle root from Fr leaf values (native, matching in-circuit).
///
/// This uses the same Poseidon hash as the in-circuit verification,
/// ensuring native computation matches the ZK constraints.
pub fn compute_merkle_root_poseidon(leaves: &[Fr]) -> Fr {
    if leaves.is_empty() {
        return Fr::ZERO;
    }
    if leaves.len() == 1 {
        return leaves[0];
    }

    let mut current_level = leaves.to_vec();

    // Pad to power of 2
    while current_level.len() & (current_level.len() - 1) != 0 {
        current_level.push(Fr::ZERO);
    }

    while current_level.len() > 1 {
        let mut next_level = Vec::new();
        for chunk in current_level.chunks(2) {
            let left = chunk[0];
            let right = if chunk.len() > 1 { chunk[1] } else { Fr::ZERO };
            next_level.push(poseidon_hash_two(left, right));
        }
        current_level = next_level;
    }
    current_level[0]
}

/// Hashes model weights into a single Fr using Poseidon sponge.
///
/// Converts weight bytes to Fr elements (32 bytes each, top bits cleared)
/// and hashes them using Poseidon sponge construction.
pub fn hash_weights_poseidon(weights: &[Fr]) -> Fr {
    poseidon_hash_many(weights)
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2_proofs::{
        circuit::SimpleFloorPlanner,
        dev::MockProver,
        plonk::Circuit,
    };

    #[test]
    fn test_model_commitment() {
        let weights = vec![1u8, 2, 3, 4, 5];
        let (lo, hi) = compute_model_commitment(&weights);
        assert_ne!(lo, [0u8; 16]);
        assert_ne!(hi, [0u8; 16]);
    }

    #[test]
    fn test_merkle_root_single_leaf() {
        let leaf = [42u8; 32];
        let root = compute_merkle_root(&[leaf]);
        assert_eq!(root, leaf);
    }

    #[test]
    fn test_merkle_root_two_leaves() {
        let leaf1 = [1u8; 32];
        let leaf2 = [2u8; 32];
        let root = compute_merkle_root(&[leaf1, leaf2]);
        assert_ne!(root, leaf1);
        assert_ne!(root, leaf2);
    }

    #[test]
    fn test_poseidon_merkle_root() {
        let leaves = vec![Fr::from(1u64), Fr::from(2u64), Fr::from(3u64), Fr::from(4u64)];
        let root = compute_merkle_root_poseidon(&leaves);
        assert_ne!(root, Fr::ZERO);

        // Verify determinism
        let root2 = compute_merkle_root_poseidon(&leaves);
        assert_eq!(root, root2);

        // Verify different leaves produce different root
        let leaves2 = vec![Fr::from(5u64), Fr::from(6u64), Fr::from(7u64), Fr::from(8u64)];
        let root3 = compute_merkle_root_poseidon(&leaves2);
        assert_ne!(root, root3);
    }

    #[test]
    fn test_poseidon_merkle_root_single() {
        let root = compute_merkle_root_poseidon(&[Fr::from(42u64)]);
        assert_eq!(root, Fr::from(42u64));
    }

    /// Test circuit for ModelCommitChip::verify_path.
    #[derive(Clone, Default)]
    struct TestCommitCircuit {
        leaf: Fr,
        root: Fr,
        path_elements: Vec<Fr>,
        path_indices: Vec<bool>,
    }

    impl Circuit<Fr> for TestCommitCircuit {
        type Config = ModelCommitConfig;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            Self::default()
        }

        fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
            let instance = meta.instance_column();
            ModelCommitChip::configure(meta, instance)
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<Fr>,
        ) -> Result<(), ErrorFront> {
            let chip = ModelCommitChip::new(config);
            chip.verify_path(
                &mut layouter,
                self.leaf,
                self.root,
                &self.path_elements,
                &self.path_indices,
            )
        }
    }

    #[test]
    fn test_verify_path_single_node() {
        let leaf = Fr::from(42u64);
        let circuit = TestCommitCircuit {
            leaf,
            root: leaf, // leaf == root for single node
            path_elements: vec![],
            path_indices: vec![],
        };

        let prover = MockProver::run(12, &circuit, vec![vec![]]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn test_verify_path_depth_1() {
        let left = Fr::from(10u64);
        let right = Fr::from(20u64);
        let root = poseidon_hash_two(left, right);

        // Prove left is in the tree (sibling = right, is_left = false means current is left)
        let circuit = TestCommitCircuit {
            leaf: left,
            root,
            path_elements: vec![right],
            path_indices: vec![false], // sibling is NOT on left, so current is left child
        };

        let prover = MockProver::run(12, &circuit, vec![vec![]]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn test_verify_path_depth_2() {
        // Build a 4-leaf tree: [a, b, c, d]
        let a = Fr::from(1u64);
        let b = Fr::from(2u64);
        let c = Fr::from(3u64);
        let d = Fr::from(4u64);

        let ab = poseidon_hash_two(a, b);
        let cd = poseidon_hash_two(c, d);
        let root = poseidon_hash_two(ab, cd);

        // Prove 'a' is in the tree
        // Level 0: sibling = b, current is left (is_left = false)
        // Level 1: sibling = cd, current is left (is_left = false)
        let circuit = TestCommitCircuit {
            leaf: a,
            root,
            path_elements: vec![b, cd],
            path_indices: vec![false, false],
        };

        let prover = MockProver::run(13, &circuit, vec![vec![]]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn test_verify_path_wrong_root_rejected() {
        let left = Fr::from(10u64);
        let right = Fr::from(20u64);
        let _correct_root = poseidon_hash_two(left, right);

        let circuit = TestCommitCircuit {
            leaf: left,
            root: Fr::from(9999u64), // WRONG root
            path_elements: vec![right],
            path_indices: vec![false],
        };

        let prover = MockProver::run(12, &circuit, vec![vec![]]).unwrap();
        assert!(prover.verify().is_err(), "Wrong root must be rejected");
    }

    #[test]
    fn test_verify_path_wrong_sibling_rejected() {
        let left = Fr::from(10u64);
        let right = Fr::from(20u64);
        let root = poseidon_hash_two(left, right);

        let circuit = TestCommitCircuit {
            leaf: left,
            root,
            path_elements: vec![Fr::from(999u64)], // WRONG sibling
            path_indices: vec![false],
        };

        let prover = MockProver::run(12, &circuit, vec![vec![]]).unwrap();
        assert!(prover.verify().is_err(), "Wrong sibling must be rejected");
    }
}
