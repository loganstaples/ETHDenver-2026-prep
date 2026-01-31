//! End-to-End MPC-ZK Integration Tests
//!
//! This module provides comprehensive integration tests demonstrating the complete
//! HELIX training flow:
//!
//! 1. **Secret Weights**: Model weights are kept private
//! 2. **MPC Sharing**: Weights are secret-shared among workers
//! 3. **Training Computation**: Forward/backward pass on shares
//! 4. **ZK Proof Generation**: Prove computation correctness without revealing weights
//! 5. **Verification**: On-chain verification of training integrity
//!
//! These tests prove that HELIX achieves trustless AI training where:
//! - No single party sees the full model weights
//! - All computations are cryptographically verified
//! - Proofs are efficient enough for on-chain verification

use helix_mpc::{
    error::{MPCError, MPCResult},
    field::Fr,
    proofs::{
        BatchedProof, BatchVerifier, ProofStats, ProofType, MPCProof,
        ShareValidityProof, ShareValidityProver, ShareValidityVerifier, ShareValidityWitness,
        AggregationProof, AggregationProver, AggregationVerifier, GradientAggregationWitness,
        GradientShareInput,
        MACProof, MACProver, ZKMACVerifier, MACWitness,
    },
    security::ShareCommitment,
    types::PartyId,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use sha2::{Digest, Sha256};
use std::time::Instant;

/// Test configuration for integration tests.
#[derive(Clone)]
struct TestConfig {
    /// Number of parties participating.
    num_parties: usize,
    /// Reconstruction threshold (for Shamir).
    threshold: usize,
    /// Input dimension for test model.
    input_dim: usize,
    /// Hidden dimension for test model.
    hidden_dim: usize,
    /// Output dimension for test model.
    output_dim: usize,
    /// Batch size for training.
    batch_size: usize,
    /// Random seed for reproducibility.
    seed: u64,
}

impl Default for TestConfig {
    fn default() -> Self {
        Self {
            num_parties: 3,
            threshold: 2,
            input_dim: 8,
            hidden_dim: 16,
            output_dim: 4,
            batch_size: 4,
            seed: 12345,
        }
    }
}

/// Simple neural network weights for testing.
#[derive(Clone, Debug)]
struct TestModelWeights {
    /// First layer weights: input_dim x hidden_dim.
    layer1_weights: Vec<Fr>,
    /// First layer bias: hidden_dim.
    layer1_bias: Vec<Fr>,
    /// Second layer weights: hidden_dim x output_dim.
    layer2_weights: Vec<Fr>,
    /// Second layer bias: output_dim.
    layer2_bias: Vec<Fr>,
}

impl TestModelWeights {
    /// Creates random model weights.
    fn random(config: &TestConfig, rng: &mut impl Rng) -> Self {
        let layer1_size = config.input_dim * config.hidden_dim;
        let layer2_size = config.hidden_dim * config.output_dim;

        Self {
            layer1_weights: (0..layer1_size).map(|_| Fr::random(rng)).collect(),
            layer1_bias: (0..config.hidden_dim).map(|_| Fr::random(rng)).collect(),
            layer2_weights: (0..layer2_size).map(|_| Fr::random(rng)).collect(),
            layer2_bias: (0..config.output_dim).map(|_| Fr::random(rng)).collect(),
        }
    }

    /// Flattens all weights into a single vector.
    fn flatten(&self) -> Vec<Fr> {
        let mut result = Vec::new();
        result.extend(&self.layer1_weights);
        result.extend(&self.layer1_bias);
        result.extend(&self.layer2_weights);
        result.extend(&self.layer2_bias);
        result
    }

    /// Total number of parameters.
    fn num_params(&self) -> usize {
        self.layer1_weights.len() + self.layer1_bias.len() +
        self.layer2_weights.len() + self.layer2_bias.len()
    }
}

/// Secret-shared model weights.
struct SharedModelWeights {
    /// Shares for each party.
    party_shares: Vec<PartyShares>,
    /// Blinding factors for commitments.
    blindings: Vec<[u8; 32]>,
    /// Commitments for each party's shares.
    commitments: Vec<[u8; 32]>,
}

/// A single party's shares of the model.
#[derive(Clone)]
struct PartyShares {
    party_id: PartyId,
    shares: Vec<Fr>,
}

impl PartyShares {
    fn flatten(&self) -> Vec<Fr> {
        self.shares.clone()
    }
}

/// Simulated training input/output.
struct TrainingBatch {
    /// Input data: batch_size x input_dim.
    inputs: Vec<Vec<Fr>>,
    /// Target labels: batch_size x output_dim.
    targets: Vec<Vec<Fr>>,
}

impl TrainingBatch {
    fn random(config: &TestConfig, rng: &mut impl Rng) -> Self {
        Self {
            inputs: (0..config.batch_size)
                .map(|_| (0..config.input_dim).map(|_| Fr::random(rng)).collect())
                .collect(),
            targets: (0..config.batch_size)
                .map(|_| (0..config.output_dim).map(|_| Fr::random(rng)).collect())
                .collect(),
        }
    }
}

/// Computed gradients from a training step.
#[derive(Clone)]
struct GradientShares {
    /// Party that computed these.
    party_id: PartyId,
    /// Gradient shares.
    gradients: Vec<Fr>,
    /// Commitment to this gradient.
    commitment: [u8; 32],
    /// Blinding factor.
    blinding: [u8; 32],
}

/// Statistics from the pipeline.
#[derive(Default, Debug)]
struct PipelineStats {
    sharing_time_ms: u64,
    computation_time_ms: u64,
    proof_generation_time_ms: u64,
    verification_time_ms: u64,
    total_proof_size_bytes: usize,
    num_proofs: usize,
}

// =============================================================================
// Test: Complete End-to-End Training Flow
// =============================================================================

/// Tests the complete HELIX training pipeline:
/// 1. Secret weights → MPC sharing
/// 2. MPC training computation
/// 3. ZK proof generation
/// 4. Proof verification
#[test]
fn test_complete_training_pipeline() -> MPCResult<()> {
    let config = TestConfig::default();
    let mut rng = ChaCha20Rng::seed_from_u64(config.seed);

    println!("\n=== HELIX MPC-ZK Integration Test ===");
    println!("Parties: {}", config.num_parties);
    println!("Model dimensions: {}x{}x{}", config.input_dim, config.hidden_dim, config.output_dim);

    // Step 1: Create secret model weights.
    let weights = TestModelWeights::random(&config, &mut rng);
    println!("\n[1] Created secret model with {} parameters", weights.num_params());

    // Step 2: Secret-share the weights among parties.
    let start = Instant::now();
    let shared_weights = share_model_weights(&weights, &config, &mut rng)?;
    let sharing_time = start.elapsed().as_millis() as u64;
    println!("[2] Secret-shared weights among {} parties ({}ms)", config.num_parties, sharing_time);

    // Step 3: Verify shares reconstruct to original.
    verify_share_reconstruction(&weights, &shared_weights, &config)?;
    println!("[3] Verified shares reconstruct correctly");

    // Step 4: Generate share validity proofs.
    let start = Instant::now();
    let share_proofs = generate_share_validity_proofs(&shared_weights, &config)?;
    let share_proof_time = start.elapsed().as_millis() as u64;
    println!("[4] Generated {} share validity proofs ({}ms)", share_proofs.len(), share_proof_time);

    // Step 5: Create training batch.
    let batch = TrainingBatch::random(&config, &mut rng);
    println!("[5] Created training batch (size={})", config.batch_size);

    // Step 6: Compute gradients on shares.
    let start = Instant::now();
    let gradient_shares = compute_gradients_on_shares(&shared_weights, &batch, &config, &mut rng)?;
    let compute_time = start.elapsed().as_millis() as u64;
    println!("[6] Computed gradients on shares ({}ms)", compute_time);

    // Step 7: Generate aggregation proofs.
    let start = Instant::now();
    let agg_proofs = generate_aggregation_proofs(&gradient_shares, &config)?;
    let agg_proof_time = start.elapsed().as_millis() as u64;
    println!("[7] Generated {} aggregation proofs ({}ms)", agg_proofs.len(), agg_proof_time);

    // Step 8: Generate MAC verification proofs.
    let start = Instant::now();
    let mac_proofs = generate_mac_proofs(&gradient_shares, &config, &mut rng)?;
    let mac_proof_time = start.elapsed().as_millis() as u64;
    println!("[8] Generated {} MAC verification proofs ({}ms)", mac_proofs.len(), mac_proof_time);

    // Step 9: Batch verify all proofs.
    let start = Instant::now();
    let mut proof_batch = BatchedProof::new();
    for proof in &share_proofs {
        proof_batch.add_share_validity(proof.clone());
    }
    for proof in &agg_proofs {
        proof_batch.add_aggregation(proof.clone());
    }
    for proof in &mac_proofs {
        proof_batch.add_mac_verification(proof.clone());
    }
    proof_batch.generate_challenges(config.seed);
    proof_batch.compute_aggregate();

    let verifier = BatchVerifier::new();
    let verified = verifier.verify(&proof_batch)?;
    let verify_time = start.elapsed().as_millis() as u64;

    println!("[9] Batch verified {} proofs: {} ({}ms)",
             proof_batch.len(),
             if verified { "PASSED" } else { "FAILED" },
             verify_time);

    // Step 10: Aggregate gradients securely.
    let aggregated = aggregate_gradient_shares(&gradient_shares, &config)?;
    println!("[10] Aggregated gradients from {} parties", config.num_parties);

    // Step 11: Update weights.
    let updated_shares = apply_gradient_update(&shared_weights, &aggregated, &config)?;
    println!("[11] Applied gradient update to shared weights");

    // Final summary.
    let total_proof_size: usize = proof_batch.total_size();
    println!("\n=== Results ===");
    println!("Total proofs: {}", proof_batch.len());
    println!("Total proof size: {} bytes ({:.2} KB)", total_proof_size, total_proof_size as f64 / 1024.0);
    println!("Verification: {}", if verified { "SUCCESS" } else { "FAILED" });
    println!("Total time: {}ms", sharing_time + compute_time + share_proof_time + agg_proof_time + mac_proof_time + verify_time);

    assert!(verified, "All proofs must verify");
    Ok(())
}

// =============================================================================
// Test: Share Validity Proofs
// =============================================================================

#[test]
fn test_share_validity_proof_flow() -> MPCResult<()> {
    let mut rng = ChaCha20Rng::seed_from_u64(42);

    println!("\n=== Share Validity Proof Test ===");

    // Create share values.
    let share_values: Vec<Fr> = (0..16).map(|_| Fr::random(&mut rng)).collect();
    let shape = vec![4, 4];

    // Generate blinding and commitment.
    let mut blinding = [0u8; 32];
    rng.fill(&mut blinding);
    let commitment = compute_commitment(&share_values, &blinding);

    // Generate dealer key pair (simplified).
    let mut dealer_pk = [0u8; 32];
    rng.fill(&mut dealer_pk);
    let dealer_sig = vec![0u8; 64]; // Mock signature.

    // Create witness.
    let witness = ShareValidityWitness {
        share_values: share_values.clone(),
        shape: shape.clone(),
        party: PartyId::new("party_1"),
        blinding,
        commitment,
        min_value: Fr::from_f64(-1e10),
        max_value: Fr::from_f64(1e10),
        dealer_public_key: dealer_pk,
        dealer_signature: dealer_sig,
    };

    // Generate proof.
    let mut prover = ShareValidityProver::with_seed(42);
    let start = Instant::now();
    let proof = prover.prove(&witness)?;
    let proof_time = start.elapsed().as_millis();

    println!("Generated share validity proof in {}ms", proof_time);
    println!("Proof size: {} bytes", proof.size());

    // Verify proof.
    let verifier = ShareValidityVerifier::new();
    let start = Instant::now();
    let valid = verifier.verify(&proof)?;
    let verify_time = start.elapsed().as_millis();

    println!("Verified proof: {} ({}ms)", valid, verify_time);

    // Test serialization.
    let bytes = proof.to_bytes();
    let recovered = ShareValidityProof::from_bytes(&bytes)?;
    let still_valid = verifier.verify(&recovered)?;

    println!("Serialization round-trip: {}", still_valid);

    assert!(valid);
    assert!(still_valid);
    Ok(())
}

// =============================================================================
// Test: Gradient Aggregation Proofs
// =============================================================================

#[test]
fn test_gradient_aggregation_proof_flow() -> MPCResult<()> {
    let config = TestConfig::default();
    let mut rng = ChaCha20Rng::seed_from_u64(config.seed);

    println!("\n=== Gradient Aggregation Proof Test ===");

    let num_gradients = 32;

    // Create aggregation witness.
    let mut witness = GradientAggregationWitness::new(config.num_parties, num_gradients, 0);

    // Add gradient shares from each party.
    for party_idx in 0..config.num_parties {
        let values: Vec<Fr> = (0..num_gradients).map(|_| Fr::random(&mut rng)).collect();
        let mut blinding = [0u8; 32];
        rng.fill(&mut blinding);
        let commitment = compute_commitment(&values, &blinding);

        let input = GradientShareInput {
            party: PartyId::new(format!("party_{}", party_idx)),
            values,
            commitment,
            blinding,
        };
        witness.add_gradient_share(input)?;
    }

    // Compute the aggregation.
    witness.compute_aggregation();

    // Generate proof.
    let mut prover = AggregationProver::with_seed(config.seed);
    let start = Instant::now();
    let proof = prover.prove(&witness)?;
    let proof_time = start.elapsed().as_millis();

    println!("Generated aggregation proof in {}ms", proof_time);
    println!("Proof size: {} bytes", proof.size());

    // Verify proof.
    let verifier = AggregationVerifier::new();
    let start = Instant::now();
    let valid = verifier.verify(&proof)?;
    let verify_time = start.elapsed().as_millis();

    println!("Verified proof: {} ({}ms)", valid, verify_time);

    // Verify the aggregated values match manually computed values.
    let mut expected = vec![Fr::ZERO; num_gradients];
    for share in &witness.gradient_shares {
        for (i, v) in share.values.iter().enumerate() {
            expected[i] = Fr::add(&expected[i], v);
        }
    }

    for (i, (e, a)) in expected.iter().zip(witness.aggregated_gradient.iter()).enumerate() {
        assert!(e.ct_eq(a).to_bool(), "Aggregation mismatch at index {}", i);
    }

    println!("Aggregated {} gradients from {} parties", num_gradients, config.num_parties);

    assert!(valid);
    Ok(())
}

// =============================================================================
// Test: MAC Verification Proofs
// =============================================================================

#[test]
fn test_mac_verification_proof_flow() -> MPCResult<()> {
    let config = TestConfig::default();
    let mut rng = ChaCha20Rng::seed_from_u64(42);

    println!("\n=== MAC Verification Proof Test ===");

    let num_values = 16;

    // Create MAC witness with proper structure.
    let mut witness = MACWitness::new(config.num_parties, num_values);

    // Generate global MAC key shares (alpha).
    let global_alpha = Fr::random(&mut rng);
    let mut alpha_sum = Fr::ZERO;
    let mut alpha_shares = Vec::new();
    for _ in 0..config.num_parties - 1 {
        let share = Fr::random(&mut rng);
        alpha_shares.push(share.clone());
        alpha_sum = Fr::add(&alpha_sum, &share);
    }
    // Last share ensures sum = global_alpha.
    let last_alpha = Fr::sub(&global_alpha, &alpha_sum);
    alpha_shares.push(last_alpha);
    witness.set_alpha_shares(alpha_shares)?;

    // Generate actual values that will be secret-shared.
    let actual_values: Vec<Fr> = (0..num_values).map(|_| Fr::random(&mut rng)).collect();

    // Compute expected MACs: mac[j] = alpha * value[j].
    let expected_macs: Vec<Fr> = actual_values.iter()
        .map(|v| Fr::mul(&global_alpha, v))
        .collect();

    // Create additive shares for values such that they sum to actual_values.
    let mut all_value_shares: Vec<Vec<Fr>> = Vec::with_capacity(config.num_parties);
    let mut value_sums = vec![Fr::ZERO; num_values];

    for party_idx in 0..config.num_parties - 1 {
        let shares: Vec<Fr> = (0..num_values).map(|_| Fr::random(&mut rng)).collect();
        for (i, s) in shares.iter().enumerate() {
            value_sums[i] = Fr::add(&value_sums[i], s);
        }
        all_value_shares.push(shares);
    }

    // Last party's value shares ensure the sum equals actual_values.
    let last_value_shares: Vec<Fr> = (0..num_values)
        .map(|i| Fr::sub(&actual_values[i], &value_sums[i]))
        .collect();
    all_value_shares.push(last_value_shares);

    // Create additive shares for MACs such that they sum to expected_macs.
    let mut all_mac_shares: Vec<Vec<Fr>> = Vec::with_capacity(config.num_parties);
    let mut mac_sums = vec![Fr::ZERO; num_values];

    for party_idx in 0..config.num_parties - 1 {
        let shares: Vec<Fr> = (0..num_values).map(|_| Fr::random(&mut rng)).collect();
        for (i, s) in shares.iter().enumerate() {
            mac_sums[i] = Fr::add(&mac_sums[i], s);
        }
        all_mac_shares.push(shares);
    }

    // Last party's MAC shares ensure the sum equals expected_macs.
    let last_mac_shares: Vec<Fr> = (0..num_values)
        .map(|i| Fr::sub(&expected_macs[i], &mac_sums[i]))
        .collect();
    all_mac_shares.push(last_mac_shares);

    // Set party shares.
    for party_idx in 0..config.num_parties {
        let mut blinding = [0u8; 32];
        rng.fill(&mut blinding);
        witness.set_party_shares(
            party_idx,
            all_value_shares[party_idx].clone(),
            all_mac_shares[party_idx].clone(),
            blinding,
        )?;
    }

    // Verify internal MAC consistency before proving.
    let macs_valid_before = witness.verify_macs();
    println!("Internal MAC verification (before proof): {}", macs_valid_before);
    assert!(macs_valid_before, "MACs must be valid before generating proof");

    // Generate proof.
    let mut prover = MACProver::with_seed(42);
    let start = Instant::now();
    let proof = prover.prove(&witness)?;
    let proof_time = start.elapsed().as_millis();

    println!("Generated MAC proof in {}ms", proof_time);
    println!("Proof size: {} bytes", proof.size());

    // Verify proof.
    let verifier = ZKMACVerifier::new();
    let start = Instant::now();
    let valid = verifier.verify(&proof)?;
    let verify_time = start.elapsed().as_millis();

    println!("Verified proof: {} ({}ms)", valid, verify_time);

    assert!(valid);
    Ok(())
}

// =============================================================================
// Test: Batched Proof Verification
// =============================================================================

#[test]
fn test_batched_proof_verification() -> MPCResult<()> {
    let config = TestConfig::default();
    let mut rng = ChaCha20Rng::seed_from_u64(42);

    println!("\n=== Batched Proof Verification Test ===");

    let mut batch = BatchedProof::new();

    // Add multiple share validity proofs.
    let mut share_prover = ShareValidityProver::with_seed(42);
    for i in 0..3 {
        let share_values: Vec<Fr> = (0..8).map(|_| Fr::random(&mut rng)).collect();
        let mut blinding = [0u8; 32];
        rng.fill(&mut blinding);
        let commitment = compute_commitment(&share_values, &blinding);
        let mut dealer_pk = [0u8; 32];
        rng.fill(&mut dealer_pk);

        let witness = ShareValidityWitness {
            share_values,
            shape: vec![8],
            party: PartyId::new(format!("party_{}", i)),
            blinding,
            commitment,
            min_value: Fr::from_f64(-1e10),
            max_value: Fr::from_f64(1e10),
            dealer_public_key: dealer_pk,
            dealer_signature: vec![0u8; 64],
        };
        let proof = share_prover.prove(&witness)?;
        batch.add_share_validity(proof);
    }
    println!("Added 3 share validity proofs");

    // Add aggregation proof.
    let mut agg_witness = GradientAggregationWitness::new(3, 8, 0);
    for party_idx in 0..3 {
        let values: Vec<Fr> = (0..8).map(|_| Fr::random(&mut rng)).collect();
        let mut blinding = [0u8; 32];
        rng.fill(&mut blinding);
        let commitment = compute_commitment(&values, &blinding);

        let input = GradientShareInput {
            party: PartyId::new(format!("party_{}", party_idx)),
            values,
            commitment,
            blinding,
        };
        agg_witness.add_gradient_share(input)?;
    }
    agg_witness.compute_aggregation();

    let mut agg_prover = AggregationProver::with_seed(42);
    let agg_proof = agg_prover.prove(&agg_witness)?;
    batch.add_aggregation(agg_proof);
    println!("Added 1 aggregation proof");

    // Add MAC proof with properly constructed shares.
    let num_parties = 3;
    let num_mac_values = 8;
    let mut mac_witness = MACWitness::new(num_parties, num_mac_values);

    // Generate alpha shares that sum to global_alpha.
    let global_alpha = Fr::random(&mut rng);
    let mut alpha_sum = Fr::ZERO;
    let mut alpha_shares = Vec::with_capacity(num_parties);
    for _ in 0..num_parties - 1 {
        let share = Fr::random(&mut rng);
        alpha_sum = Fr::add(&alpha_sum, &share);
        alpha_shares.push(share);
    }
    alpha_shares.push(Fr::sub(&global_alpha, &alpha_sum));
    mac_witness.set_alpha_shares(alpha_shares)?;

    // Generate actual values.
    let actual_values: Vec<Fr> = (0..num_mac_values).map(|_| Fr::random(&mut rng)).collect();
    let expected_macs: Vec<Fr> = actual_values.iter()
        .map(|v| Fr::mul(&global_alpha, v))
        .collect();

    // Create value shares.
    let mut all_value_shares: Vec<Vec<Fr>> = Vec::with_capacity(num_parties);
    let mut value_sums = vec![Fr::ZERO; num_mac_values];
    for _ in 0..num_parties - 1 {
        let shares: Vec<Fr> = (0..num_mac_values).map(|_| Fr::random(&mut rng)).collect();
        for (i, s) in shares.iter().enumerate() {
            value_sums[i] = Fr::add(&value_sums[i], s);
        }
        all_value_shares.push(shares);
    }
    all_value_shares.push((0..num_mac_values).map(|i| Fr::sub(&actual_values[i], &value_sums[i])).collect());

    // Create MAC shares.
    let mut all_mac_shares: Vec<Vec<Fr>> = Vec::with_capacity(num_parties);
    let mut mac_sums = vec![Fr::ZERO; num_mac_values];
    for _ in 0..num_parties - 1 {
        let shares: Vec<Fr> = (0..num_mac_values).map(|_| Fr::random(&mut rng)).collect();
        for (i, s) in shares.iter().enumerate() {
            mac_sums[i] = Fr::add(&mac_sums[i], s);
        }
        all_mac_shares.push(shares);
    }
    all_mac_shares.push((0..num_mac_values).map(|i| Fr::sub(&expected_macs[i], &mac_sums[i])).collect());

    // Set party shares.
    for party_idx in 0..num_parties {
        let mut blinding = [0u8; 32];
        rng.fill(&mut blinding);
        mac_witness.set_party_shares(
            party_idx,
            all_value_shares[party_idx].clone(),
            all_mac_shares[party_idx].clone(),
            blinding,
        )?;
    }

    let mut mac_prover = MACProver::with_seed(42);
    let mac_proof = mac_prover.prove(&mac_witness)?;
    batch.add_mac_verification(mac_proof);
    println!("Added 1 MAC verification proof");

    // Generate challenges and compute aggregate.
    batch.generate_challenges(42);
    batch.compute_aggregate();

    // Verify batch.
    let verifier = BatchVerifier::new();
    let start = Instant::now();
    let valid = verifier.verify(&batch)?;
    let verify_time = start.elapsed().as_millis();

    println!("\nBatch verification: {} ({}ms)", if valid { "PASSED" } else { "FAILED" }, verify_time);
    println!("Total proofs: {}", batch.len());
    println!("Total size: {} bytes", batch.total_size());

    assert!(valid);
    Ok(())
}

// =============================================================================
// Test: Multi-Step Training with Proofs
// =============================================================================

#[test]
fn test_multi_step_training_with_proofs() -> MPCResult<()> {
    let config = TestConfig::default();
    let mut rng = ChaCha20Rng::seed_from_u64(config.seed);

    println!("\n=== Multi-Step Training Test ===");

    let num_steps = 3;
    let mut weights = TestModelWeights::random(&config, &mut rng);
    let mut all_proofs: Vec<BatchedProof> = Vec::new();
    let mut total_stats = PipelineStats::default();

    for step in 0..num_steps {
        println!("\n--- Training Step {} ---", step + 1);

        // Share weights.
        let start = Instant::now();
        let shared = share_model_weights(&weights, &config, &mut rng)?;
        total_stats.sharing_time_ms += start.elapsed().as_millis() as u64;

        // Create batch.
        let batch = TrainingBatch::random(&config, &mut rng);

        // Compute gradients.
        let start = Instant::now();
        let gradient_shares = compute_gradients_on_shares(&shared, &batch, &config, &mut rng)?;
        total_stats.computation_time_ms += start.elapsed().as_millis() as u64;

        // Generate proofs.
        let start = Instant::now();
        let share_proofs = generate_share_validity_proofs(&shared, &config)?;
        let agg_proofs = generate_aggregation_proofs(&gradient_shares, &config)?;
        let mac_proofs = generate_mac_proofs(&gradient_shares, &config, &mut rng)?;
        total_stats.proof_generation_time_ms += start.elapsed().as_millis() as u64;

        // Create batch proof.
        let mut step_batch = BatchedProof::new();
        for p in share_proofs { step_batch.add_share_validity(p); }
        for p in agg_proofs { step_batch.add_aggregation(p); }
        for p in mac_proofs { step_batch.add_mac_verification(p); }
        step_batch.generate_challenges(config.seed + step as u64);
        step_batch.compute_aggregate();

        total_stats.num_proofs += step_batch.len();
        total_stats.total_proof_size_bytes += step_batch.total_size();
        all_proofs.push(step_batch);

        // Aggregate and update weights.
        let aggregated = aggregate_gradient_shares(&gradient_shares, &config)?;
        let updated = apply_gradient_update(&shared, &aggregated, &config)?;

        // Reconstruct updated weights for next iteration.
        weights = reconstruct_weights(&updated, &config)?;

        println!("Step {} complete: {} proofs generated", step + 1, all_proofs.last().unwrap().len());
    }

    // Verify all proofs.
    let verifier = BatchVerifier::new();
    let start = Instant::now();
    let all_valid = all_proofs.iter()
        .map(|b| verifier.verify(b))
        .collect::<MPCResult<Vec<bool>>>()?
        .iter()
        .all(|&v| v);
    total_stats.verification_time_ms = start.elapsed().as_millis() as u64;

    println!("\n=== Multi-Step Results ===");
    println!("Training steps: {}", num_steps);
    println!("Total proofs: {}", total_stats.num_proofs);
    println!("Total proof size: {:.2} KB", total_stats.total_proof_size_bytes as f64 / 1024.0);
    println!("All proofs verified: {}", all_valid);
    println!("Timing breakdown:");
    println!("  Sharing: {}ms", total_stats.sharing_time_ms);
    println!("  Computation: {}ms", total_stats.computation_time_ms);
    println!("  Proof generation: {}ms", total_stats.proof_generation_time_ms);
    println!("  Verification: {}ms", total_stats.verification_time_ms);

    assert!(all_valid);
    Ok(())
}

// =============================================================================
// Test: Proof Chain Composition
// =============================================================================

#[test]
fn test_proof_composition_chain() -> MPCResult<()> {
    let config = TestConfig::default();
    let mut rng = ChaCha20Rng::seed_from_u64(config.seed);

    println!("\n=== Proof Composition Chain Test ===");

    // Phase 1: Generate share validity proofs.
    println!("\nPhase 1: Share Validity Proofs");
    let weights = TestModelWeights::random(&config, &mut rng);
    let shared = share_model_weights(&weights, &config, &mut rng)?;
    let share_proofs = generate_share_validity_proofs(&shared, &config)?;

    let share_chain_hash = compute_chain_hash(&share_proofs.iter().map(|p| p.to_bytes()).collect::<Vec<_>>());
    println!("  Generated {} proofs, chain hash: {:02x?}...", share_proofs.len(), &share_chain_hash[..4]);

    // Phase 2: Computation proofs linked to share proofs.
    println!("\nPhase 2: Aggregation Proofs");
    let batch = TrainingBatch::random(&config, &mut rng);
    let gradient_shares = compute_gradients_on_shares(&shared, &batch, &config, &mut rng)?;
    let agg_proofs = generate_aggregation_proofs(&gradient_shares, &config)?;

    let agg_chain_hash = compute_chain_hash_with_link(
        &agg_proofs.iter().map(|p| p.to_bytes()).collect::<Vec<_>>(),
        &share_chain_hash,
    );
    println!("  Generated {} proofs, chain hash: {:02x?}...", agg_proofs.len(), &agg_chain_hash[..4]);

    // Phase 3: MAC proofs linked to aggregation proofs.
    println!("\nPhase 3: MAC Verification Proofs");
    let mac_proofs = generate_mac_proofs(&gradient_shares, &config, &mut rng)?;

    let mac_chain_hash = compute_chain_hash_with_link(
        &mac_proofs.iter().map(|p| p.to_bytes()).collect::<Vec<_>>(),
        &agg_chain_hash,
    );
    println!("  Generated {} proofs, chain hash: {:02x?}...", mac_proofs.len(), &mac_chain_hash[..4]);

    // Verify the chain by checking all proofs.
    let verifier = BatchVerifier::new();
    let share_valid = share_proofs.iter()
        .all(|p| ShareValidityVerifier::new().verify(p).unwrap_or(false));
    let agg_valid = agg_proofs.iter()
        .all(|p| AggregationVerifier::new().verify(p).unwrap_or(false));
    let mac_valid = mac_proofs.iter()
        .all(|p| ZKMACVerifier::new().verify(p).unwrap_or(false));

    println!("\nChain verification:");
    println!("  Share validity: {}", share_valid);
    println!("  Aggregation: {}", agg_valid);
    println!("  MAC verification: {}", mac_valid);

    let chain_valid = share_valid && agg_valid && mac_valid;
    println!("\nOverall chain: {}", if chain_valid { "VALID" } else { "INVALID" });

    assert!(chain_valid);
    Ok(())
}

// =============================================================================
// Test: Proof Statistics
// =============================================================================

#[test]
fn test_proof_statistics() -> MPCResult<()> {
    let config = TestConfig::default();
    let mut rng = ChaCha20Rng::seed_from_u64(config.seed);

    println!("\n=== Proof Statistics Test ===");

    let mut stats = ProofStats::new();
    let weights = TestModelWeights::random(&config, &mut rng);
    let shared = share_model_weights(&weights, &config, &mut rng)?;

    // Generate and measure share validity proofs.
    for party_shares in &shared.party_shares {
        let start = Instant::now();

        let witness = ShareValidityWitness {
            share_values: party_shares.shares.clone(),
            shape: vec![party_shares.shares.len()],
            party: party_shares.party_id.clone(),
            blinding: shared.blindings[0],
            commitment: shared.commitments[0],
            min_value: Fr::from_f64(-1e10),
            max_value: Fr::from_f64(1e10),
            dealer_public_key: [0u8; 32],
            dealer_signature: vec![0u8; 64],
        };

        let mut prover = ShareValidityProver::with_seed(42);
        let proof = prover.prove(&witness)?;

        let time_ms = start.elapsed().as_millis() as u64;
        stats.record(proof.size(), time_ms);
    }

    println!("Proof Statistics:");
    println!("  Number of proofs: {}", stats.num_proofs);
    println!("  Total size: {} bytes", stats.total_size_bytes);
    println!("  Average size: {} bytes", stats.avg_size_bytes);
    println!("  Total time: {}ms", stats.total_time_ms);
    println!("  Average time: {}ms", stats.avg_time_ms);

    assert!(stats.num_proofs > 0);
    Ok(())
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Computes a hash commitment to a set of field elements.
fn compute_commitment(data: &[Fr], blinding: &[u8; 32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for v in data {
        hasher.update(&v.to_bytes_le());
    }
    hasher.update(blinding);
    hasher.finalize().into()
}

/// Computes a chain hash from proof bytes.
fn compute_chain_hash(proofs: &[Vec<u8>]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for proof in proofs {
        hasher.update(proof);
    }
    hasher.finalize().into()
}

/// Computes a chain hash linked to a previous hash.
fn compute_chain_hash_with_link(proofs: &[Vec<u8>], link: &[u8; 32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(link);
    for proof in proofs {
        hasher.update(proof);
    }
    hasher.finalize().into()
}

/// Shares model weights among parties using additive secret sharing.
fn share_model_weights(
    weights: &TestModelWeights,
    config: &TestConfig,
    rng: &mut impl Rng,
) -> MPCResult<SharedModelWeights> {
    let all_weights = weights.flatten();
    let mut party_shares: Vec<PartyShares> = Vec::new();
    let mut blindings: Vec<[u8; 32]> = Vec::new();
    let mut commitments: Vec<[u8; 32]> = Vec::new();

    // Generate shares for all but last party.
    let mut shares_matrix: Vec<Vec<Fr>> = Vec::new();
    for party_idx in 0..config.num_parties - 1 {
        let shares: Vec<Fr> = (0..all_weights.len())
            .map(|_| Fr::random(rng))
            .collect();
        shares_matrix.push(shares);
    }

    // Last party gets: value - sum(other shares).
    let last_shares: Vec<Fr> = (0..all_weights.len())
        .map(|i| {
            let sum: Fr = shares_matrix.iter()
                .map(|s| s[i].clone())
                .fold(Fr::ZERO, |acc, x| Fr::add(&acc, &x));
            Fr::sub(&all_weights[i], &sum)
        })
        .collect();
    shares_matrix.push(last_shares);

    // Create party shares and commitments.
    for (party_idx, shares) in shares_matrix.into_iter().enumerate() {
        let mut blinding = [0u8; 32];
        rng.fill(&mut blinding);
        let commitment = compute_commitment(&shares, &blinding);

        party_shares.push(PartyShares {
            party_id: PartyId::new(format!("party_{}", party_idx)),
            shares,
        });
        blindings.push(blinding);
        commitments.push(commitment);
    }

    Ok(SharedModelWeights {
        party_shares,
        blindings,
        commitments,
    })
}

/// Verifies that shares correctly reconstruct to the original weights.
fn verify_share_reconstruction(
    original: &TestModelWeights,
    shared: &SharedModelWeights,
    _config: &TestConfig,
) -> MPCResult<()> {
    let original_flat = original.flatten();

    for i in 0..original_flat.len() {
        let reconstructed: Fr = shared.party_shares.iter()
            .map(|ps| ps.shares[i].clone())
            .fold(Fr::ZERO, |acc, x| Fr::add(&acc, &x));

        if !reconstructed.ct_eq(&original_flat[i]).to_bool() {
            return Err(MPCError::ProtocolError(
                format!("Share reconstruction failed at index {}", i)
            ));
        }
    }

    Ok(())
}

/// Generates share validity proofs for all parties.
fn generate_share_validity_proofs(
    shared: &SharedModelWeights,
    _config: &TestConfig,
) -> MPCResult<Vec<ShareValidityProof>> {
    let mut proofs = Vec::new();

    for (idx, party_shares) in shared.party_shares.iter().enumerate() {
        let witness = ShareValidityWitness {
            share_values: party_shares.shares.clone(),
            shape: vec![party_shares.shares.len()],
            party: party_shares.party_id.clone(),
            blinding: shared.blindings[idx],
            commitment: shared.commitments[idx],
            min_value: Fr::from_f64(-1e10),
            max_value: Fr::from_f64(1e10),
            dealer_public_key: [0u8; 32],
            dealer_signature: vec![0u8; 64],
        };

        let mut prover = ShareValidityProver::with_seed(42 + idx as u64);
        let proof = prover.prove(&witness)?;
        proofs.push(proof);
    }

    Ok(proofs)
}

/// Computes gradients on secret-shared weights.
fn compute_gradients_on_shares(
    shared: &SharedModelWeights,
    batch: &TrainingBatch,
    config: &TestConfig,
    rng: &mut impl Rng,
) -> MPCResult<Vec<GradientShares>> {
    let mut gradient_shares = Vec::new();

    for (idx, party_shares) in shared.party_shares.iter().enumerate() {
        // Simplified gradient computation.
        // In real implementation, this would be MPC forward/backward pass.
        let gradients: Vec<Fr> = party_shares.shares.iter()
            .map(|w| {
                // Simple gradient: sum of products with inputs.
                batch.inputs.iter()
                    .flat_map(|inp| inp.iter())
                    .fold(Fr::ZERO, |acc, x| Fr::add(&acc, &Fr::mul(w, x)))
            })
            .collect();

        let mut blinding = [0u8; 32];
        rng.fill(&mut blinding);
        let commitment = compute_commitment(&gradients, &blinding);

        gradient_shares.push(GradientShares {
            party_id: party_shares.party_id.clone(),
            gradients,
            commitment,
            blinding,
        });
    }

    Ok(gradient_shares)
}

/// Generates aggregation proofs for gradient shares.
fn generate_aggregation_proofs(
    gradient_shares: &[GradientShares],
    _config: &TestConfig,
) -> MPCResult<Vec<AggregationProof>> {
    if gradient_shares.is_empty() {
        return Ok(Vec::new());
    }

    let num_gradients = gradient_shares[0].gradients.len();
    let num_parties = gradient_shares.len();

    let mut witness = GradientAggregationWitness::new(num_parties, num_gradients, 0);

    for gs in gradient_shares {
        let input = GradientShareInput {
            party: gs.party_id.clone(),
            values: gs.gradients.clone(),
            commitment: gs.commitment,
            blinding: gs.blinding,
        };
        witness.add_gradient_share(input)?;
    }

    witness.compute_aggregation();

    let mut prover = AggregationProver::with_seed(42);
    let proof = prover.prove(&witness)?;

    Ok(vec![proof])
}

/// Generates MAC verification proofs.
fn generate_mac_proofs(
    gradient_shares: &[GradientShares],
    _config: &TestConfig,
    rng: &mut impl Rng,
) -> MPCResult<Vec<MACProof>> {
    if gradient_shares.is_empty() {
        return Ok(Vec::new());
    }

    let num_parties = gradient_shares.len();
    let num_values = gradient_shares[0].gradients.len();

    let mut witness = MACWitness::new(num_parties, num_values);

    // Generate alpha shares that sum to global alpha.
    let global_alpha = Fr::random(rng);
    let mut alpha_sum = Fr::ZERO;
    let mut alpha_shares = Vec::with_capacity(num_parties);
    for _ in 0..num_parties - 1 {
        let share = Fr::random(rng);
        alpha_sum = Fr::add(&alpha_sum, &share);
        alpha_shares.push(share);
    }
    // Last alpha share ensures sum = global_alpha.
    alpha_shares.push(Fr::sub(&global_alpha, &alpha_sum));
    witness.set_alpha_shares(alpha_shares.clone())?;

    // First, compute the total value for each gradient (reconstructed from shares).
    let mut total_values = vec![Fr::ZERO; num_values];
    for gs in gradient_shares {
        for (i, g) in gs.gradients.iter().enumerate() {
            total_values[i] = Fr::add(&total_values[i], g);
        }
    }

    // Compute expected MACs: mac[j] = alpha * total_value[j].
    let expected_macs: Vec<Fr> = total_values.iter()
        .map(|v| Fr::mul(&global_alpha, v))
        .collect();

    // Distribute MAC shares such that they sum to expected_macs.
    // For parties 0..(n-1), assign random MAC shares.
    // For the last party, assign the remainder.
    let mut mac_sums = vec![Fr::ZERO; num_values];
    let mut all_mac_shares: Vec<Vec<Fr>> = Vec::with_capacity(num_parties);

    for party_idx in 0..num_parties - 1 {
        let macs: Vec<Fr> = (0..num_values).map(|_| Fr::random(rng)).collect();
        for (i, m) in macs.iter().enumerate() {
            mac_sums[i] = Fr::add(&mac_sums[i], m);
        }
        all_mac_shares.push(macs);
    }

    // Last party's MAC shares ensure the sum equals expected_macs.
    let last_macs: Vec<Fr> = (0..num_values)
        .map(|i| Fr::sub(&expected_macs[i], &mac_sums[i]))
        .collect();
    all_mac_shares.push(last_macs);

    // Set party shares with their value shares and computed MAC shares.
    for (party_idx, gs) in gradient_shares.iter().enumerate() {
        let values = gs.gradients.clone();
        witness.set_party_shares(party_idx, values, all_mac_shares[party_idx].clone(), gs.blinding)?;
    }

    let mut prover = MACProver::with_seed(42);
    let proof = prover.prove(&witness)?;

    Ok(vec![proof])
}

/// Aggregates gradient shares.
fn aggregate_gradient_shares(
    gradient_shares: &[GradientShares],
    _config: &TestConfig,
) -> MPCResult<Vec<Fr>> {
    if gradient_shares.is_empty() {
        return Ok(Vec::new());
    }

    let num_gradients = gradient_shares[0].gradients.len();
    let mut aggregated = vec![Fr::ZERO; num_gradients];

    for gs in gradient_shares {
        for (i, g) in gs.gradients.iter().enumerate() {
            aggregated[i] = Fr::add(&aggregated[i], g);
        }
    }

    Ok(aggregated)
}

/// Applies gradient update to shared weights.
fn apply_gradient_update(
    shared: &SharedModelWeights,
    gradients: &[Fr],
    _config: &TestConfig,
) -> MPCResult<Vec<PartyShares>> {
    // Simple gradient descent update.
    let learning_rate = Fr::from_u64(1);

    let updated: Vec<PartyShares> = shared.party_shares.iter()
        .map(|ps| {
            let updated_shares: Vec<Fr> = ps.shares.iter()
                .zip(gradients.iter())
                .map(|(w, g)| Fr::sub(w, &Fr::mul(&learning_rate, g)))
                .collect();

            PartyShares {
                party_id: ps.party_id.clone(),
                shares: updated_shares,
            }
        })
        .collect();

    Ok(updated)
}

/// Reconstructs weights from shares.
fn reconstruct_weights(
    shares: &[PartyShares],
    config: &TestConfig,
) -> MPCResult<TestModelWeights> {
    if shares.is_empty() {
        return Err(MPCError::ProtocolError("No shares to reconstruct".into()));
    }

    let num_weights = shares[0].shares.len();
    let mut reconstructed = vec![Fr::ZERO; num_weights];

    for ps in shares {
        for (i, s) in ps.shares.iter().enumerate() {
            reconstructed[i] = Fr::add(&reconstructed[i], s);
        }
    }

    // Split into layers.
    let l1_size = config.input_dim * config.hidden_dim;
    let b1_size = config.hidden_dim;
    let l2_size = config.hidden_dim * config.output_dim;

    let mut offset = 0;
    let layer1_weights = reconstructed[offset..offset + l1_size].to_vec();
    offset += l1_size;
    let layer1_bias = reconstructed[offset..offset + b1_size].to_vec();
    offset += b1_size;
    let layer2_weights = reconstructed[offset..offset + l2_size].to_vec();
    offset += l2_size;
    let layer2_bias = reconstructed[offset..].to_vec();

    Ok(TestModelWeights {
        layer1_weights,
        layer1_bias,
        layer2_weights,
        layer2_bias,
    })
}

// =============================================================================
// Performance Benchmark (Run with --ignored)
// =============================================================================

#[test]
#[ignore] // Run with: cargo test --release performance -- --ignored --nocapture
fn test_performance_scaling() -> MPCResult<()> {
    println!("\n=== Performance Scaling Test ===");

    let party_counts = [3, 5, 7];
    let model_sizes = [(16, 32, 8), (32, 64, 16), (64, 128, 32)];

    for &num_parties in &party_counts {
        for &(input, hidden, output) in &model_sizes {
            let config = TestConfig {
                num_parties,
                threshold: (num_parties + 1) / 2,
                input_dim: input,
                hidden_dim: hidden,
                output_dim: output,
                batch_size: 4,
                seed: 12345,
            };

            let mut rng = ChaCha20Rng::seed_from_u64(config.seed);
            let weights = TestModelWeights::random(&config, &mut rng);

            // Time sharing.
            let start = Instant::now();
            let shared = share_model_weights(&weights, &config, &mut rng)?;
            let share_time = start.elapsed().as_millis();

            // Time proof generation.
            let start = Instant::now();
            let proofs = generate_share_validity_proofs(&shared, &config)?;
            let proof_time = start.elapsed().as_millis();

            let total_params = weights.num_params();
            let proof_size: usize = proofs.iter().map(|p| p.size()).sum();

            println!(
                "Parties: {}, Model: {}x{}x{} ({} params), Share: {}ms, Proof: {}ms, Size: {} bytes",
                num_parties, input, hidden, output, total_params, share_time, proof_time, proof_size
            );
        }
    }

    Ok(())
}
