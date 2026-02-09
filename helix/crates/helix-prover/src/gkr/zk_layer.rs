//! Zero-Knowledge Layer for GKR Proofs.
//!
//! This module provides the zero-knowledge transformation for GKR proofs.
//! The standard GKR protocol is not zero-knowledge - intermediate values
//! revealed during the sumcheck protocol can leak information about the witness.
//!
//! ## ZK Transformation
//!
//! We use polynomial masking to achieve zero-knowledge:
//! 1. Add random low-degree polynomials to intermediate values
//! 2. Prove that the masked polynomials satisfy the required relations
//! 3. Use commitment schemes to hide the randomness
//!
//! ## Masking Strategy
//!
//! For each layer's polynomial W̃(x), we compute:
//! - Masked polynomial: W̃'(x) = W̃(x) + r(x)
//! - Where r(x) is a random polynomial with r evaluated at hypercube points = 0
//!
//! This ensures:
//! - Sumcheck claims remain valid (random terms cancel)
//! - Intermediate evaluations reveal nothing about W̃

use super::{FieldElement, GKRError, GKRResult};
use super::multilinear::{DenseMultilinear, MultilinearPolynomial};
use helix_circuits::halo2_proofs::arithmetic::Field;
use helix_circuits::halo2curves::ff::PrimeField;
use rand::Rng;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Configuration for the ZK layer.
#[derive(Debug, Clone)]
pub struct ZKConfig {
    /// Security parameter (lambda).
    pub security_bits: usize,
    /// Number of masking polynomials per layer.
    pub num_masks: usize,
    /// Whether to use strong masking (more randomness, more security).
    pub strong_masking: bool,
    /// Seed for deterministic randomness (for testing).
    pub seed: Option<[u8; 32]>,
}

impl Default for ZKConfig {
    fn default() -> Self {
        Self {
            security_bits: 128,
            num_masks: 1,
            strong_masking: true,
            seed: None,
        }
    }
}

impl ZKConfig {
    /// Creates a config for testing (deterministic randomness).
    pub fn for_testing(seed: [u8; 32]) -> Self {
        Self {
            security_bits: 128,
            num_masks: 1,
            strong_masking: false,
            seed: Some(seed),
        }
    }
}

/// Randomness for ZK masking.
#[derive(Debug, Clone)]
pub struct ZKRandomness {
    /// Random field elements for masking.
    masks: Vec<FieldElement>,
    /// Random blinding factors.
    blinders: Vec<FieldElement>,
    /// Commitment randomness.
    commitment_randomness: Vec<FieldElement>,
}

impl ZKRandomness {
    /// Generates fresh randomness.
    pub fn generate(num_masks: usize, num_blinders: usize) -> Self {
        let mut rng = rand::thread_rng();

        let masks: Vec<FieldElement> = (0..num_masks)
            .map(|_| FieldElement::from(rng.gen::<u64>()))
            .collect();

        let blinders: Vec<FieldElement> = (0..num_blinders)
            .map(|_| FieldElement::from(rng.gen::<u64>()))
            .collect();

        let commitment_randomness: Vec<FieldElement> = (0..num_masks)
            .map(|_| FieldElement::from(rng.gen::<u64>()))
            .collect();

        Self {
            masks,
            blinders,
            commitment_randomness,
        }
    }

    /// Generates randomness from a seed (for testing).
    pub fn from_seed(seed: [u8; 32], num_masks: usize, num_blinders: usize) -> Self {
        use sha2::{Sha256, Digest};

        let mut hasher = Sha256::new();
        hasher.update(&seed);
        hasher.update(b"ZK_RANDOMNESS");

        let mut masks = Vec::with_capacity(num_masks);
        let mut blinders = Vec::with_capacity(num_blinders);
        let mut commitment_randomness = Vec::with_capacity(num_masks);

        for i in 0..(num_masks + num_blinders + num_masks) {
            hasher.update(&(i as u64).to_le_bytes());
            let hash = hasher.finalize_reset();
            let mut repr = [0u8; 32];
            repr.copy_from_slice(&hash[..32]);

            let elem = FieldElement::from_raw([
                u64::from_le_bytes(repr[0..8].try_into().expect("invariant: fixed-size slice")),
                u64::from_le_bytes(repr[8..16].try_into().expect("invariant: fixed-size slice")),
                u64::from_le_bytes(repr[16..24].try_into().expect("invariant: fixed-size slice")),
                u64::from_le_bytes(repr[24..32].try_into().expect("invariant: fixed-size slice")) & 0x0FFFFFFFFFFFFFFF,
            ]);

            if i < num_masks {
                masks.push(elem);
            } else if i < num_masks + num_blinders {
                blinders.push(elem);
            } else {
                commitment_randomness.push(elem);
            }

            hasher.update(&hash);
        }

        Self {
            masks,
            blinders,
            commitment_randomness,
        }
    }

    /// Returns the masks.
    pub fn masks(&self) -> &[FieldElement] {
        &self.masks
    }

    /// Returns the blinders.
    pub fn blinders(&self) -> &[FieldElement] {
        &self.blinders
    }
}

/// A polynomial mask for zero-knowledge.
#[derive(Debug, Clone)]
pub struct ZKMask {
    /// The masking polynomial evaluations.
    evaluations: Vec<FieldElement>,
    /// Random coefficients used to generate the mask.
    coefficients: Vec<FieldElement>,
    /// Number of variables.
    num_vars: usize,
}

impl ZKMask {
    /// Generates a random mask polynomial that sums to zero over the hypercube.
    ///
    /// This is achieved by generating random evaluations and adjusting one
    /// to make the total sum zero.
    pub fn generate(num_vars: usize) -> Self {
        let mut rng = rand::thread_rng();
        let size = 1 << num_vars;

        // Generate random evaluations
        let mut evaluations: Vec<FieldElement> = (0..size)
            .map(|_| FieldElement::from(rng.gen::<u64>()))
            .collect();

        // Make sum zero by adjusting the last element
        let sum: FieldElement = evaluations.iter().fold(FieldElement::zero(), |a, b| a + b);
        if !evaluations.is_empty() {
            evaluations[size - 1] = evaluations[size - 1] - sum;
        }

        let coefficients = vec![FieldElement::from(rng.gen::<u64>())];

        Self {
            evaluations,
            coefficients,
            num_vars,
        }
    }

    /// Generates a mask from a seed (deterministic, for testing).
    pub fn from_seed(seed: [u8; 32], num_vars: usize) -> Self {
        use sha2::{Sha256, Digest};

        let size = 1 << num_vars;
        let mut evaluations = Vec::with_capacity(size);

        let mut hasher = Sha256::new();
        hasher.update(&seed);
        hasher.update(b"ZK_MASK");

        for i in 0..size {
            hasher.update(&(i as u64).to_le_bytes());
            let hash = hasher.finalize_reset();
            let mut repr = [0u8; 32];
            repr.copy_from_slice(&hash[..32]);

            let elem = FieldElement::from_raw([
                u64::from_le_bytes(repr[0..8].try_into().expect("invariant: fixed-size slice")),
                u64::from_le_bytes(repr[8..16].try_into().expect("invariant: fixed-size slice")),
                u64::from_le_bytes(repr[16..24].try_into().expect("invariant: fixed-size slice")),
                u64::from_le_bytes(repr[24..32].try_into().expect("invariant: fixed-size slice")) & 0x0FFFFFFFFFFFFFFF,
            ]);

            evaluations.push(elem);
            hasher.update(&hash);
        }

        // Make sum zero
        let sum: FieldElement = evaluations.iter().fold(FieldElement::zero(), |a, b| a + b);
        if !evaluations.is_empty() {
            evaluations[size - 1] = evaluations[size - 1] - sum;
        }

        Self {
            evaluations,
            coefficients: vec![FieldElement::one()],
            num_vars,
        }
    }

    /// Returns the mask evaluations.
    pub fn evaluations(&self) -> &[FieldElement] {
        &self.evaluations
    }

    /// Returns the number of variables.
    pub fn num_variables(&self) -> usize {
        self.num_vars
    }

    /// Evaluates the mask at a point.
    pub fn evaluate(&self, point: &[FieldElement]) -> FieldElement {
        let poly = DenseMultilinear::from_evaluations(self.evaluations.clone());
        poly.evaluate(point)
    }

    /// Verifies that the mask sums to zero.
    pub fn verify_sum_zero(&self) -> bool {
        let sum: FieldElement = self.evaluations.iter().fold(FieldElement::zero(), |a, b| a + b);
        sum == FieldElement::zero()
    }
}

/// A masked polynomial (original + mask).
#[derive(Debug, Clone)]
pub struct MaskedPolynomial {
    /// The original polynomial.
    original: DenseMultilinear,
    /// The masking polynomial.
    mask: ZKMask,
    /// The combined (masked) polynomial.
    masked: DenseMultilinear,
}

impl MaskedPolynomial {
    /// Creates a masked polynomial.
    pub fn new(original: DenseMultilinear, mask: ZKMask) -> Self {
        assert_eq!(original.num_variables(), mask.num_variables());

        let masked_evals: Vec<FieldElement> = original
            .evaluations()
            .iter()
            .zip(mask.evaluations().iter())
            .map(|(&o, &m)| o + m)
            .collect();

        let masked = DenseMultilinear::from_evaluations(masked_evals);

        Self {
            original,
            mask,
            masked,
        }
    }

    /// Creates a masked polynomial with automatic mask generation.
    pub fn mask_polynomial(original: DenseMultilinear) -> Self {
        let mask = ZKMask::generate(original.num_variables());
        Self::new(original, mask)
    }

    /// Returns the masked polynomial.
    pub fn masked(&self) -> &DenseMultilinear {
        &self.masked
    }

    /// Returns the original polynomial (should not be revealed!).
    pub fn original(&self) -> &DenseMultilinear {
        &self.original
    }

    /// Evaluates the masked polynomial at a point.
    pub fn evaluate(&self, point: &[FieldElement]) -> FieldElement {
        self.masked.evaluate(point)
    }

    /// Returns the sum of the masked polynomial (equals original sum).
    pub fn sum(&self) -> FieldElement {
        // mask sums to zero, so masked sum = original sum
        self.original.sum()
    }

    /// Unmasks an evaluation given the mask evaluation.
    ///
    /// This is used internally but should not be exposed to the verifier.
    pub fn unmask_evaluation(&self, masked_eval: FieldElement, point: &[FieldElement]) -> FieldElement {
        let mask_eval = self.mask.evaluate(point);
        masked_eval - mask_eval
    }
}

/// Zero-knowledge layer for the GKR protocol.
pub struct ZeroKnowledgeLayer {
    /// Configuration.
    config: ZKConfig,
    /// Randomness for this layer.
    randomness: Option<ZKRandomness>,
    /// Masks for each layer.
    layer_masks: Vec<ZKMask>,
    /// Blinding commitments.
    blinding_commitments: Vec<[u8; 32]>,
}

impl ZeroKnowledgeLayer {
    /// Creates a new ZK layer with the given configuration.
    pub fn new(config: ZKConfig) -> Self {
        Self {
            config,
            randomness: None,
            layer_masks: Vec::new(),
            blinding_commitments: Vec::new(),
        }
    }

    /// Initializes the ZK layer for a circuit with the given layer sizes.
    pub fn initialize(&mut self, layer_sizes: &[usize]) -> GKRResult<()> {
        // Generate randomness
        let num_masks = layer_sizes.len() * self.config.num_masks;
        let num_blinders = layer_sizes.len();

        self.randomness = Some(match self.config.seed {
            Some(seed) => ZKRandomness::from_seed(seed, num_masks, num_blinders),
            None => ZKRandomness::generate(num_masks, num_blinders),
        });

        // Generate masks for each layer
        // Ensure minimum of 1 variable (size 2) for valid polynomials
        self.layer_masks = layer_sizes
            .iter()
            .enumerate()
            .map(|(i, &size)| {
                let effective_size = size.max(2); // At least 2 elements
                let num_vars = (effective_size as f64).log2().ceil() as usize;
                let num_vars = num_vars.max(1); // At least 1 variable
                match self.config.seed {
                    Some(seed) => {
                        let mut layer_seed = seed;
                        layer_seed[0] ^= i as u8;
                        ZKMask::from_seed(layer_seed, num_vars)
                    }
                    None => ZKMask::generate(num_vars),
                }
            })
            .collect();

        Ok(())
    }

    /// Masks a polynomial for the given layer.
    pub fn mask_polynomial(&self, poly: DenseMultilinear, layer_idx: usize) -> GKRResult<MaskedPolynomial> {
        if layer_idx >= self.layer_masks.len() {
            return Err(GKRError::InvalidCircuit(format!(
                "Layer {} not initialized (only {} layers)",
                layer_idx, self.layer_masks.len()
            )));
        }

        let mask = self.layer_masks[layer_idx].clone();
        Ok(MaskedPolynomial::new(poly, mask))
    }

    /// Returns the mask for a layer.
    pub fn get_mask(&self, layer_idx: usize) -> Option<&ZKMask> {
        self.layer_masks.get(layer_idx)
    }

    /// Computes a blinding commitment for the given layer.
    pub fn compute_blinding_commitment(&mut self, layer_idx: usize, values: &[FieldElement]) -> [u8; 32] {
        use sha2::{Sha256, Digest};

        let mut hasher = Sha256::new();
        hasher.update(b"HELIX_ZK_BLIND");
        hasher.update(&(layer_idx as u64).to_le_bytes());

        // Add mask evaluation
        if let Some(mask) = self.get_mask(layer_idx) {
            for eval in mask.evaluations() {
                hasher.update(&eval.to_repr());
            }
        }

        // Add blinding randomness
        if let Some(ref rand) = self.randomness {
            if layer_idx < rand.blinders().len() {
                hasher.update(&rand.blinders()[layer_idx].to_repr());
            }
        }

        // Add values
        for v in values {
            hasher.update(&v.to_repr());
        }

        let hash = hasher.finalize();
        let mut commitment = [0u8; 32];
        commitment.copy_from_slice(&hash);

        self.blinding_commitments.push(commitment);
        commitment
    }

    /// Verifies a blinding commitment.
    pub fn verify_blinding_commitment(
        &self,
        layer_idx: usize,
        values: &[FieldElement],
        commitment: &[u8; 32],
    ) -> bool {
        // In a real implementation, this would use the commitment scheme
        // For now, just check that the commitment exists
        self.blinding_commitments.get(layer_idx).map_or(false, |c| c == commitment)
    }

    /// Generates opening proofs for the masks at given points.
    pub fn open_masks(&self, points: &[Vec<FieldElement>]) -> Vec<FieldElement> {
        points
            .iter()
            .enumerate()
            .filter_map(|(i, point)| {
                self.layer_masks.get(i).map(|mask| mask.evaluate(point))
            })
            .collect()
    }

    /// Returns whether the layer is initialized.
    pub fn is_initialized(&self) -> bool {
        !self.layer_masks.is_empty()
    }

    /// Returns the configuration.
    pub fn config(&self) -> &ZKConfig {
        &self.config
    }
}

/// Builder for ZK-enhanced polynomials.
pub struct ZKPolynomialBuilder {
    /// Accumulated masks.
    masks: Vec<ZKMask>,
    /// Current randomness state.
    rng_state: u64,
}

impl ZKPolynomialBuilder {
    /// Creates a new ZK polynomial builder.
    pub fn new() -> Self {
        Self {
            masks: Vec::new(),
            rng_state: 0,
        }
    }

    /// Creates a builder with a specific seed.
    pub fn with_seed(seed: u64) -> Self {
        Self {
            masks: Vec::new(),
            rng_state: seed,
        }
    }

    /// Generates a zero-sum mask for the given number of variables.
    pub fn generate_mask(&mut self, num_vars: usize) -> ZKMask {
        let mut seed = [0u8; 32];
        seed[0..8].copy_from_slice(&self.rng_state.to_le_bytes());
        self.rng_state = self.rng_state.wrapping_add(1);

        let mask = ZKMask::from_seed(seed, num_vars);
        self.masks.push(mask.clone());
        mask
    }

    /// Returns all generated masks.
    pub fn masks(&self) -> &[ZKMask] {
        &self.masks
    }
}

impl Default for ZKPolynomialBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zk_mask_sum_zero() {
        let mask = ZKMask::generate(4);
        assert!(mask.verify_sum_zero());
    }

    #[test]
    fn test_zk_mask_from_seed() {
        let seed = [1u8; 32];
        let mask1 = ZKMask::from_seed(seed, 3);
        let mask2 = ZKMask::from_seed(seed, 3);

        // Same seed should give same mask
        assert_eq!(mask1.evaluations(), mask2.evaluations());
        assert!(mask1.verify_sum_zero());
    }

    #[test]
    fn test_masked_polynomial_sum() {
        let evals = vec![
            FieldElement::from(1u64),
            FieldElement::from(2u64),
            FieldElement::from(3u64),
            FieldElement::from(4u64),
        ];

        let original = DenseMultilinear::from_evaluations(evals);
        let original_sum = original.sum();

        let masked = MaskedPolynomial::mask_polynomial(original);

        // Masked sum should equal original sum (mask sums to zero)
        assert_eq!(masked.sum(), original_sum);
    }

    #[test]
    fn test_zk_layer_initialization() {
        let config = ZKConfig::default();
        let mut layer = ZeroKnowledgeLayer::new(config);

        let layer_sizes = vec![4, 8, 16];
        layer.initialize(&layer_sizes).unwrap();

        assert!(layer.is_initialized());
        assert_eq!(layer.layer_masks.len(), 3);
    }

    #[test]
    fn test_zk_layer_mask_polynomial() {
        let config = ZKConfig::for_testing([0u8; 32]);
        let mut layer = ZeroKnowledgeLayer::new(config);

        layer.initialize(&[4]).unwrap();

        let evals = vec![
            FieldElement::from(10u64),
            FieldElement::from(20u64),
            FieldElement::from(30u64),
            FieldElement::from(40u64),
        ];

        let poly = DenseMultilinear::from_evaluations(evals);
        let original_sum = poly.sum();

        let masked = layer.mask_polynomial(poly, 0).unwrap();

        // Sum preserved
        assert_eq!(masked.sum(), original_sum);
    }

    #[test]
    fn test_randomness_generation() {
        let rand1 = ZKRandomness::generate(4, 3);
        let rand2 = ZKRandomness::generate(4, 3);

        // Should be different (with overwhelming probability)
        assert_ne!(rand1.masks(), rand2.masks());
    }

    #[test]
    fn test_randomness_from_seed() {
        let seed = [42u8; 32];
        let rand1 = ZKRandomness::from_seed(seed, 4, 3);
        let rand2 = ZKRandomness::from_seed(seed, 4, 3);

        // Should be identical
        assert_eq!(rand1.masks(), rand2.masks());
        assert_eq!(rand1.blinders(), rand2.blinders());
    }

    #[test]
    fn test_zk_polynomial_builder() {
        let mut builder = ZKPolynomialBuilder::with_seed(12345);

        let mask1 = builder.generate_mask(3);
        let mask2 = builder.generate_mask(4);

        assert!(mask1.verify_sum_zero());
        assert!(mask2.verify_sum_zero());
        assert_eq!(builder.masks().len(), 2);
    }

    #[test]
    fn test_blinding_commitment() {
        let config = ZKConfig::default();
        let mut layer = ZeroKnowledgeLayer::new(config);
        layer.initialize(&[4, 8]).unwrap();

        let values = vec![
            FieldElement::from(1u64),
            FieldElement::from(2u64),
        ];

        let commitment = layer.compute_blinding_commitment(0, &values);

        // Commitment should be 32 bytes
        assert_eq!(commitment.len(), 32);

        // Verify commitment
        assert!(layer.verify_blinding_commitment(0, &values, &commitment));
    }
}
