use crate::gadgets::range::{RangeChip, RangeConfig};
use crate::gadgets::arithmetic::{ArithmeticChip, ArithmeticConfig};
use crate::approximate::bounded_add::{BoundedAddChip, BoundedAddConfig};
use crate::approximate::bounded_mul::{BoundedMulChip, BoundedMulConfig};
use crate::approximate::bounded_matmul::{BoundedMatMulChip, BoundedMatMulConfig};
use crate::approximate::activation::{ReLUChip, ReLUConfig};

use halo2curves::bn256::Fr;
use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    dev::MockProver,
    plonk::{Circuit, ConstraintSystem, Error, ErrorFront},
};

#[derive(Clone)]
struct TestCircuitConfig {
    add: BoundedAddConfig<Fr, 100>, 
    mul: BoundedMulConfig<Fr, 100>,
    matmul: BoundedMatMulConfig<Fr, 100>,
    relu: ReLUConfig<Fr, 100>,
}

#[derive(Default)]
struct TestCircuit {
    // Inputs (Option allows testing missing values behavior if needed, generally Some)
    a_val: Option<u64>,
    a_err: Option<u64>,
    b_val: Option<u64>,
    b_err: Option<u64>,
    
    // Expected Output override (if None, computes correct one)
    c_val: Option<u64>,
    c_err: Option<u64>,

    // Mode switch (0=add, 1=mul, 2=matmul, 3=relu)
    mode: usize,
}

impl Circuit<Fr> for TestCircuit {
    type Config = TestCircuitConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self::default()
    }

    fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
        let a = meta.advice_column();
        let b = meta.advice_column();
        let c = meta.advice_column();
        let table = meta.lookup_table_column();
        
        let arith_config = ArithmeticChip::configure(meta, a, b, c);
        // Range check is on column 'c', which is the output of arithmetic usually.
        // But gadgets might copy values there.
        let range_config = RangeConfig::configure(meta, table, c);
        let s_relu = meta.selector();

        TestCircuitConfig {
            add: BoundedAddConfig {
                arithmetic: arith_config.clone(),
                range: range_config.clone(),
            },
            mul: BoundedMulConfig {
                arithmetic: arith_config.clone(),
                range: range_config.clone(),
            },
            matmul: BoundedMatMulConfig {
                arithmetic: arith_config.clone(),
                range: range_config.clone(),
            },
            relu: ReLUConfig {
                range: range_config,
                arithmetic: arith_config,
                s_relu,
            }
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fr>,
    ) -> Result<(), ErrorFront> {
        let add_chip = BoundedAddChip::new(config.add.clone());
        let mul_chip = BoundedMulChip::new(config.mul.clone());
        let matmul_chip = BoundedMatMulChip::new(config.matmul.clone());
        let relu_chip = ReLUChip::new(config.relu.clone());
        
        // Load range table once
        add_chip.range_chip.load(&mut layouter)?;

        let val_a = Value::known(Fr::from(self.a_val.unwrap_or(0)));
        let err_a = Value::known(Fr::from(self.a_err.unwrap_or(0)));
        let val_b = Value::known(Fr::from(self.b_val.unwrap_or(0)));
        let err_b = Value::known(Fr::from(self.b_err.unwrap_or(0)));

        if self.mode == 0 {
            // Add
            // Use expected values if provided, else compute correct ones
            let val_c = self.c_val.map(|v| Value::known(Fr::from(v)))
                .unwrap_or_else(|| val_a + val_b);
            let err_c = self.c_err.map(|v| Value::known(Fr::from(v)))
                .unwrap_or_else(|| err_a + err_b);

            add_chip.assign(layouter, val_a, err_a, val_b, err_b, val_c, err_c)?;
        } else if self.mode == 1 {
            // Mul
            let val_c = self.c_val.map(|v| Value::known(Fr::from(v)))
                .unwrap_or_else(|| val_a * val_b);
                
            let err_c = self.c_err.map(|v| Value::known(Fr::from(v)))
                .unwrap_or_else(|| {
                    let term1 = val_a * err_b;
                    let term2 = val_b * err_a;
                    let term3 = err_a * err_b;
                    term1 + term2 + term3
                });

            mul_chip.assign(layouter, val_a, err_a, val_b, err_b, val_c, err_c)?;
        } else if self.mode == 2 {
            // MatMul (1x1 for simplicity)
            let row_a_v = vec![val_a];
            let row_a_e = vec![err_a];
            let col_b_v = vec![val_b];
            let col_b_e = vec![err_b];
            
            let res_val = self.c_val.map(|v| Value::known(Fr::from(v)))
                .unwrap_or_else(|| val_a * val_b);

            let err_c = self.c_err.map(|v| Value::known(Fr::from(v)))
                .unwrap_or_else(|| {
                    let term1 = val_a * err_b;
                    let term2 = val_b * err_a;
                    let term3 = err_a * err_b;
                    term1 + term2 + term3
                });
            
            matmul_chip.assign_dot_product(layouter, &row_a_v, &row_a_e, &col_b_v, &col_b_e, res_val, err_c)?;
        } else if self.mode == 3 {
            // ReLU
            // If checking failure, inputs might be manipulated.
            // For valid ReLU: if a > 0, output=a. If a < 0 (large field element), output=0.
            let out_val = self.c_val.map(|v| Value::known(Fr::from(v)))
                .unwrap_or(val_a); // Default to identity (positive case)
            
            let out_err = self.c_err.map(|v| Value::known(Fr::from(v)))
                .unwrap_or(err_a);

            relu_chip.assign(layouter, val_a, err_a, out_val, out_err)?;
        }

        Ok(())
    }
}

// --- Positive Tests ---

#[test]
fn test_bounded_add_valid() {
    let circuit = TestCircuit { 
        a_val: Some(10), a_err: Some(1),
        b_val: Some(20), b_err: Some(2),
        c_val: None, c_err: None,
        mode: 0 
    };
    let prover = MockProver::run(8, &circuit, vec![]).unwrap();
    assert_eq!(prover.verify(), Ok(()));
}

#[test]
fn test_bounded_mul_valid() {
    let circuit = TestCircuit { 
        a_val: Some(10), a_err: Some(1),
        b_val: Some(20), b_err: Some(2),
        c_val: None, c_err: None,
        mode: 1 
    };
    // Expected err: 10*2 + 20*1 + 1*2 = 20 + 20 + 2 = 42
    // 42 < 100 (range)
    let prover = MockProver::run(8, &circuit, vec![]).unwrap();
    assert_eq!(prover.verify(), Ok(()));
}

#[test]
fn test_bounded_matmul_valid() {
    let circuit = TestCircuit { 
        a_val: Some(2), a_err: Some(1),
        b_val: Some(3), b_err: Some(1),
        c_val: None, c_err: None,
        mode: 2 
    };
    // err = 2*1 + 3*1 + 1*1 = 6.
    let prover = MockProver::run(8, &circuit, vec![]).unwrap();
    assert_eq!(prover.verify(), Ok(()));
}

#[test]
fn test_relu_valid() {
    let circuit = TestCircuit {
        a_val: Some(10), a_err: Some(5),
        b_val: None, b_err: None,
        c_val: Some(10), c_err: Some(5), // Identity for positive
        mode: 3
    };
    let prover = MockProver::run(8, &circuit, vec![]).unwrap();
    assert_eq!(prover.verify(), Ok(()));
}

// --- Negative Tests (Must Fail) ---

#[test]
fn test_bounded_add_invalid_value() {
    let circuit = TestCircuit { 
        a_val: Some(10), a_err: Some(1),
        b_val: Some(20), b_err: Some(2),
        // 10 + 20 != 31
        c_val: Some(31), c_err: None, 
        mode: 0 
    };
    let prover = MockProver::run(8, &circuit, vec![]).unwrap();
    // verification should fail implies Ok(()) is NOT equal to prover.verify()
    assert!(prover.verify().is_err());
}

#[test]
fn test_bounded_add_invalid_error() {
    let circuit = TestCircuit { 
        a_val: Some(10), a_err: Some(1),
        b_val: Some(20), b_err: Some(2),
        // 1 + 2 != 5
        c_val: None, c_err: Some(5),
        mode: 0 
    };
    let prover = MockProver::run(8, &circuit, vec![]).unwrap();
    assert!(prover.verify().is_err());
}

#[test]
fn test_bounded_add_error_out_of_range() {
    let circuit = TestCircuit { 
        a_val: Some(10), a_err: Some(60),
        b_val: Some(20), b_err: Some(50), 
        // 60 + 50 = 110. Range is 100. Should fail range check.
        c_val: None, c_err: None,
        mode: 0 
    };
    let prover = MockProver::run(8, &circuit, vec![]).unwrap();
    assert!(prover.verify().is_err());
}

#[test]
fn test_bounded_mul_invalid_value() {
    let circuit = TestCircuit { 
        a_val: Some(10), a_err: Some(1),
        b_val: Some(20), b_err: Some(2),
        c_val: Some(201), c_err: None, // 10*20 != 201
        mode: 1 
    };
    let prover = MockProver::run(8, &circuit, vec![]).unwrap();
    assert!(prover.verify().is_err());
}

#[test]
fn test_bounded_mul_invalid_error() {
    let circuit = TestCircuit { 
        a_val: Some(10), a_err: Some(1),
        b_val: Some(20), b_err: Some(2),
        c_val: None, c_err: Some(10), // Expected 42
        mode: 1 
    };
    let prover = MockProver::run(8, &circuit, vec![]).unwrap();
    assert!(prover.verify().is_err());
}

#[test]
fn test_state_transition() {
    // Placeholder to confirm compilation of new modules
    assert_eq!(1, 1);
}

// ============================================================================
// EVM Format Compatibility Tests
// ============================================================================

mod evm_format_tests {
    use crate::ml::training_step_v2::{
        MLTrainingStepV2Circuit, MLTrainingStepV2Witness, compute_witness_v2, compute_state_hash_v2,
        ToEvmPublicInputs,
    };
    use crate::verifier::{
        EvmProof, EvmProofBuilder, EvmPublicInputsArray,
        MIN_PROOF_SIZE, NUM_ADVICE_COMMITS, G1_POINT_SIZE, NUM_PUBLIC_INPUTS,
        fr_to_evm_bytes, evm_bytes_to_fr, g1_to_evm_bytes, evm_bytes_to_g1,
        validate_proof_format, validate_public_inputs, compute_hash_pair,
        ProofFormatError, ProofStructure,
        create_test_proof, create_test_public_inputs,
    };
    use halo2curves::bn256::{Fr, G1Affine};
    use halo2curves::ff::{PrimeField, Field};
    use halo2curves::group::Curve;
    use halo2curves::group::prime::PrimeCurveAffine;

    /// Test that proof length validation works correctly.
    #[test]
    fn test_proof_length_validation() {
        // Valid length
        let valid_proof = create_test_proof(1);
        assert!(valid_proof.validate_length().is_ok());
        assert_eq!(valid_proof.len(), MIN_PROOF_SIZE);

        // Too short
        let short_bytes = vec![0u8; 100];
        let result = EvmProof::from_bytes(short_bytes);
        assert!(matches!(
            result,
            Err(ProofFormatError::TooShort { got: 100, min: 320 })
        ));
    }

    /// Test that proof structure matches contract expectations.
    #[test]
    fn test_proof_structure_matches_contract() {
        let proof = create_test_proof(42);
        let structure = ProofStructure::parse(proof.as_bytes()).unwrap();

        // Verify layout matches Halo2Verifier.sol expectations
        assert_eq!(structure.advice_section.offset, 0);
        assert_eq!(structure.advice_section.length, NUM_ADVICE_COMMITS * G1_POINT_SIZE);
        assert_eq!(structure.advice_section.length, 192); // 3 * 64 bytes

        assert_eq!(structure.opening_section.offset, 192);
        assert_eq!(structure.opening_section.length, 2 * G1_POINT_SIZE);
        assert_eq!(structure.opening_section.length, 128); // 2 * 64 bytes

        assert_eq!(structure.total_size, 320);
    }

    /// Test that advice commitments can be extracted correctly.
    #[test]
    fn test_advice_commitment_extraction() {
        let g1 = G1Affine::generator();
        let c0 = (g1 * Fr::from(100u64)).to_affine();
        let c1 = (g1 * Fr::from(200u64)).to_affine();
        let c2 = (g1 * Fr::from(300u64)).to_affine();
        let w = (g1 * Fr::from(400u64)).to_affine();
        let w_prime = (g1 * Fr::from(500u64)).to_affine();

        let proof = EvmProof::from_points([c0, c1, c2], w, w_prime);

        let extracted = proof.advice_commits().unwrap();
        assert_eq!(extracted[0], c0);
        assert_eq!(extracted[1], c1);
        assert_eq!(extracted[2], c2);

        let extracted_w = proof.w().unwrap();
        let extracted_w_prime = proof.w_prime().unwrap();
        assert_eq!(extracted_w, w);
        assert_eq!(extracted_w_prime, w_prime);
    }

    /// Test G1 point serialization roundtrip.
    #[test]
    fn test_g1_serialization_roundtrip() {
        let g1 = G1Affine::generator();
        let point = (g1 * Fr::from(12345u64)).to_affine();

        let bytes = g1_to_evm_bytes(&point);
        assert_eq!(bytes.len(), G1_POINT_SIZE);

        let recovered = evm_bytes_to_g1(&bytes).unwrap();
        assert_eq!(point, recovered);
    }

    /// Test Fr serialization to big-endian EVM format.
    #[test]
    fn test_fr_big_endian_encoding() {
        // Test a known value
        let val = Fr::from(0x0102030405060708u64);
        let bytes = fr_to_evm_bytes(&val);

        // Should be big-endian: most significant bytes first
        // The value fits in 8 bytes, so bytes[24..32] should contain it
        assert_eq!(
            &bytes[24..32],
            &[0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]
        );

        // Leading bytes should be zero
        assert!(bytes[..24].iter().all(|&b| b == 0));

        // Roundtrip
        let recovered = evm_bytes_to_fr(&bytes).unwrap();
        assert_eq!(val, recovered);
    }

    /// Test public inputs array formatting.
    #[test]
    fn test_public_inputs_formatting() {
        let inputs = create_test_public_inputs(1);

        // Check array length
        assert_eq!(inputs.values().len(), NUM_PUBLIC_INPUTS);

        // Check byte encoding length
        let bytes = inputs.to_evm_bytes();
        assert_eq!(bytes.len(), NUM_PUBLIC_INPUTS * 32);

        // Check hex encoding format
        let hex_arr = inputs.to_hex_array();
        assert_eq!(hex_arr.len(), NUM_PUBLIC_INPUTS);
        for hex in &hex_arr {
            assert!(hex.starts_with("0x"));
            assert_eq!(hex.len(), 2 + 64); // 0x + 32 bytes as hex
        }

        // Check Solidity literal format
        let literal = inputs.to_solidity_literal();
        assert!(literal.starts_with('['));
        assert!(literal.ends_with(']'));
    }

    /// Test that the circuit produces valid EVM public inputs.
    #[test]
    fn test_circuit_evm_public_inputs() {
        let d_in = 2;
        let d_hid = 2;
        let d_out = 1;

        let w1 = vec![Fr::from(1), Fr::from(2), Fr::from(3), Fr::from(1)];
        let b1 = vec![Fr::from(0), Fr::from(0)];
        let w2 = vec![Fr::from(1), Fr::from(1)];
        let b2 = vec![Fr::from(0)];
        let x = vec![Fr::from(1), Fr::from(1)];
        let target = vec![Fr::from(5)];
        let lr = Fr::from(1);
        let base_error = Fr::from(1);

        let old_hash = compute_state_hash_v2(&w1, &b1, &w2, &b2);
        let witness = compute_witness_v2(
            d_in, d_hid, d_out, &x, &target, &w1, &b1, &w2, &b2, lr,
            old_hash, (Fr::zero(), Fr::zero()), 1, base_error,
        );

        let evm_inputs = witness.to_evm_public_inputs();

        // Verify structure
        let (old_lo, old_hi) = evm_inputs.old_state_hash();
        assert_eq!(old_lo, old_hash.0);
        assert_eq!(old_hi, old_hash.1);

        assert_eq!(evm_inputs.loss(), witness.loss);
        assert_eq!(evm_inputs.error_bound(), witness.total_error);
        assert_eq!(evm_inputs.step_number(), Fr::from(1u64));
    }

    /// Test hash pair computation matches Solidity's _hashPair.
    #[test]
    fn test_hash_pair_consistency() {
        // The Solidity contract computes:
        // keccak256(abi.encodePacked(lo, hi))
        // where lo and hi are each 32-byte big-endian uint256

        let lo = Fr::from(1u64);
        let hi = Fr::from(2u64);

        let hash = compute_hash_pair(&lo, &hi);

        // Verify it's 32 bytes (keccak256 output)
        assert_eq!(hash.len(), 32);

        // Verify determinism
        let hash2 = compute_hash_pair(&lo, &hi);
        assert_eq!(hash, hash2);

        // Different inputs should produce different hashes
        let hash3 = compute_hash_pair(&Fr::from(3u64), &Fr::from(4u64));
        assert_ne!(hash, hash3);
    }

    /// Test mock proof creation from circuit.
    #[test]
    fn test_circuit_mock_proof() {
        let d_in = 2;
        let d_hid = 2;
        let d_out = 1;

        let w1 = vec![Fr::from(1), Fr::from(2), Fr::from(3), Fr::from(1)];
        let b1 = vec![Fr::from(0), Fr::from(0)];
        let w2 = vec![Fr::from(1), Fr::from(1)];
        let b2 = vec![Fr::from(0)];
        let x = vec![Fr::from(1), Fr::from(1)];
        let target = vec![Fr::from(5)];
        let lr = Fr::from(1);
        let base_error = Fr::from(1);

        let old_hash = compute_state_hash_v2(&w1, &b1, &w2, &b2);
        let witness = compute_witness_v2(
            d_in, d_hid, d_out, &x, &target, &w1, &b1, &w2, &b2, lr,
            old_hash, (Fr::zero(), Fr::zero()), 1, base_error,
        );

        let circuit = MLTrainingStepV2Circuit {
            witness,
            relu_range: 128,
            exp_range: 64,
            exp_scale: 32,
            use_freivalds: true,
        };

        // Create mock proof
        let mock_proof = circuit.create_mock_evm_proof();

        // Verify structure
        assert_eq!(mock_proof.len(), MIN_PROOF_SIZE);
        assert!(mock_proof.validate_length().is_ok());

        // Verify points can be extracted
        let advice = mock_proof.advice_commits().unwrap();
        assert_eq!(advice.len(), NUM_ADVICE_COMMITS);

        let w = mock_proof.w();
        let w_prime = mock_proof.w_prime();
        assert!(w.is_ok());
        assert!(w_prime.is_ok());
    }

    /// Test proof builder pattern.
    #[test]
    fn test_proof_builder() {
        let g1 = G1Affine::generator();

        let proof = EvmProofBuilder::new()
            .add_advice_commit((g1 * Fr::from(1u64)).to_affine())
            .add_advice_commit((g1 * Fr::from(2u64)).to_affine())
            .add_advice_commit((g1 * Fr::from(3u64)).to_affine())
            .with_w((g1 * Fr::from(4u64)).to_affine())
            .with_w_prime((g1 * Fr::from(5u64)).to_affine())
            .build();

        assert_eq!(proof.len(), MIN_PROOF_SIZE);
    }

    /// Test proof dump for debugging.
    #[test]
    fn test_proof_dump() {
        let proof = create_test_proof(99999);
        let dump = proof.dump();

        // Verify dump contains expected sections
        assert!(dump.contains("Total size: 320"));
        assert!(dump.contains("Advice Commitments"));
        assert!(dump.contains("Opening Proofs"));
        assert!(dump.contains("Point C0"));
        assert!(dump.contains("Point C1"));
        assert!(dump.contains("Point C2"));
        assert!(dump.contains("Point W"));
        assert!(dump.contains("Point W'"));
    }

    /// Test witness EVM public inputs dump.
    #[test]
    fn test_witness_evm_dump() {
        let d_in = 2;
        let d_hid = 2;
        let d_out = 1;

        let w1 = vec![Fr::from(1), Fr::from(2), Fr::from(3), Fr::from(1)];
        let b1 = vec![Fr::from(0), Fr::from(0)];
        let w2 = vec![Fr::from(1), Fr::from(1)];
        let b2 = vec![Fr::from(0)];
        let x = vec![Fr::from(1), Fr::from(1)];
        let target = vec![Fr::from(5)];
        let lr = Fr::from(1);
        let base_error = Fr::from(1);

        let old_hash = compute_state_hash_v2(&w1, &b1, &w2, &b2);
        let witness = compute_witness_v2(
            d_in, d_hid, d_out, &x, &target, &w1, &b1, &w2, &b2, lr,
            old_hash, (Fr::zero(), Fr::zero()), 42, base_error,
        );

        let dump = witness.dump_evm_public_inputs();

        // Verify dump contains expected elements
        assert!(dump.contains("EVM Public Inputs"));
        assert!(dump.contains("old_state_hash_lo"));
        assert!(dump.contains("old_state_hash_hi"));
        assert!(dump.contains("new_state_hash_lo"));
        assert!(dump.contains("new_state_hash_hi"));
        assert!(dump.contains("loss"));
        assert!(dump.contains("error_bound"));
        assert!(dump.contains("step_number"));
        assert!(dump.contains("Reconstructed Commitments"));
    }

    /// Test that all curve points in a proof are valid (on curve).
    #[test]
    fn test_all_points_on_curve() {
        let proof = create_test_proof(7777);

        // validate_proof_format checks all 5 points
        let result = validate_proof_format(proof.as_bytes());
        assert!(result.is_ok());
    }

    /// Test invalid point detection.
    #[test]
    fn test_invalid_point_detection() {
        // Create a proof with an invalid point (not on curve)
        let mut bytes = vec![0u8; MIN_PROOF_SIZE];

        // Set first point's x to 1 and y to 1 (not on curve: 1² ≠ 1³ + 3)
        bytes[31] = 1; // x = 1 (big-endian)
        bytes[63] = 1; // y = 1 (big-endian)

        let result = validate_proof_format(&bytes);
        assert!(matches!(result, Err(ProofFormatError::PointNotOnCurve { .. })));
    }

    /// Test proof hex encoding for Solidity.
    #[test]
    fn test_proof_hex_for_solidity() {
        let proof = create_test_proof(12345);
        let hex = proof.to_hex();

        // Should start with 0x
        assert!(hex.starts_with("0x"));

        // Should be correct length (0x + 2 chars per byte)
        assert_eq!(hex.len(), 2 + MIN_PROOF_SIZE * 2);

        // Should only contain valid hex characters
        assert!(hex[2..].chars().all(|c| c.is_ascii_hexdigit()));
    }

    /// Test that ToEvmPublicInputs trait is implemented correctly.
    #[test]
    fn test_to_evm_public_inputs_trait() {
        let d_in = 2;
        let d_hid = 2;
        let d_out = 1;

        let w1 = vec![Fr::from(1), Fr::from(2), Fr::from(3), Fr::from(1)];
        let b1 = vec![Fr::from(0), Fr::from(0)];
        let w2 = vec![Fr::from(1), Fr::from(1)];
        let b2 = vec![Fr::from(0)];
        let x = vec![Fr::from(1), Fr::from(1)];
        let target = vec![Fr::from(5)];
        let lr = Fr::from(1);
        let base_error = Fr::from(1);

        let old_hash = compute_state_hash_v2(&w1, &b1, &w2, &b2);
        let witness = compute_witness_v2(
            d_in, d_hid, d_out, &x, &target, &w1, &b1, &w2, &b2, lr,
            old_hash, (Fr::zero(), Fr::zero()), 1, base_error,
        );

        // Test via trait
        let inputs: EvmPublicInputsArray = ToEvmPublicInputs::to_evm_public_inputs(&witness);
        let bytes = ToEvmPublicInputs::to_evm_public_inputs_bytes(&witness);
        let literal = ToEvmPublicInputs::to_evm_public_inputs_literal(&witness);

        assert_eq!(inputs.values().len(), NUM_PUBLIC_INPUTS);
        assert_eq!(bytes.len(), NUM_PUBLIC_INPUTS * 32);
        assert!(literal.starts_with('['));
    }

    /// Integration test: verify complete proof + public inputs structure.
    #[test]
    fn test_complete_evm_submission_format() {
        // This test simulates what would be sent to the contract

        // 1. Create circuit with witness
        let d_in = 4;
        let d_hid = 4;
        let d_out = 2;

        let w1: Vec<Fr> = (0..d_hid * d_in).map(|i| Fr::from((i % 3 + 1) as u64)).collect();
        let b1 = vec![Fr::from(0); d_hid];
        let w2: Vec<Fr> = (0..d_out * d_hid).map(|i| Fr::from((i % 2 + 1) as u64)).collect();
        let b2 = vec![Fr::from(0); d_out];

        let x: Vec<Fr> = (0..d_in).map(|i| Fr::from((i + 1) as u64)).collect();
        let target: Vec<Fr> = (0..d_out).map(|_| Fr::from(10u64)).collect();
        let lr = Fr::from(1);
        let base_error = Fr::from(1);

        let old_hash = compute_state_hash_v2(&w1, &b1, &w2, &b2);
        let witness = compute_witness_v2(
            d_in, d_hid, d_out, &x, &target, &w1, &b1, &w2, &b2, lr,
            old_hash, (Fr::zero(), Fr::zero()), 1, base_error,
        );

        let new_hash = compute_state_hash_v2(
            &witness.w1_new, &witness.b1_new, &witness.w2_new, &witness.b2_new
        );

        let witness = compute_witness_v2(
            d_in, d_hid, d_out, &x, &target, &w1, &b1, &w2, &b2, lr,
            old_hash, new_hash, 1, base_error,
        );

        let circuit = MLTrainingStepV2Circuit {
            witness: witness.clone(),
            relu_range: 256,
            exp_range: 128,
            exp_scale: 64,
            use_freivalds: true,
        };

        // 2. Generate mock proof
        let proof = circuit.create_mock_evm_proof();
        assert!(proof.len() >= MIN_PROOF_SIZE);

        // 3. Generate public inputs
        let public_inputs = circuit.to_evm_public_inputs();
        assert_eq!(public_inputs.values().len(), NUM_PUBLIC_INPUTS);

        // 4. Verify the commitment reconstruction would work in Solidity
        let (old_lo, old_hi) = public_inputs.old_state_hash();
        let (new_lo, new_hi) = public_inputs.new_state_hash();

        // These would be the values used by the contract to verify commitments
        let old_commitment = compute_hash_pair(&old_lo, &old_hi);
        let new_commitment = compute_hash_pair(&new_lo, &new_hi);

        // Commitments should be 32 bytes
        assert_eq!(old_commitment.len(), 32);
        assert_eq!(new_commitment.len(), 32);

        // 5. Verify the proof structure is valid
        assert!(validate_proof_format(proof.as_bytes()).is_ok());

        // 6. Verify public inputs are valid
        let inputs_vec: Vec<Fr> = public_inputs.values().to_vec();
        assert!(validate_public_inputs(&inputs_vec).is_ok());

        // 7. Print what would be sent to contract (for debugging)
        println!("\n=== Contract Submission Format ===");
        println!("Proof hex: {} ({} bytes)", &proof.to_hex()[..66], proof.len());
        println!("Public inputs: {}", public_inputs.to_solidity_literal());
    }
}
