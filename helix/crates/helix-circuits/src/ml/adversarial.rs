//! Adversarial tests for MLTrainingStepV2Circuit.
//!
//! Each test creates a valid witness via `compute_witness_v2`, tampers with
//! exactly one field, and asserts that `MockProver::verify()` rejects it.
//! This ensures that the constraint system actually catches each class of
//! malicious prover behaviour rather than silently accepting bad witnesses.

#[cfg(test)]
mod tests {
    use halo2_proofs::dev::MockProver;
    use halo2_proofs::arithmetic::Field;
    use halo2curves::bn256::Fr;

    use crate::ml::training_step_v2::{
        MLTrainingStepV2Circuit,
        compute_state_hash_v2, compute_witness_v2,
    };

    /// The k parameter (log2 of rows) used for all adversarial tests.
    const K: u32 = 14;

    // -----------------------------------------------------------------------
    // Helper: build a valid (circuit, public_inputs) pair for a 2×2×1 MLP.
    // -----------------------------------------------------------------------

    fn make_valid_circuit() -> (MLTrainingStepV2Circuit, Vec<Fr>) {
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

        // First pass to compute new weights (for new_hash)
        let tmp = compute_witness_v2(
            d_in, d_hid, d_out, &x, &target, &w1, &b1, &w2, &b2, lr,
            old_hash, (Fr::ZERO, Fr::ZERO), 1, base_error,
        );
        let new_hash = compute_state_hash_v2(
            &tmp.w1_new, &tmp.b1_new, &tmp.w2_new, &tmp.b2_new,
        );

        // Second pass with correct new_hash
        let witness = compute_witness_v2(
            d_in, d_hid, d_out, &x, &target, &w1, &b1, &w2, &b2, lr,
            old_hash, new_hash, 1, base_error,
        );

        let pi = witness.public_inputs();
        let circuit = MLTrainingStepV2Circuit {
            witness,
            relu_range: 128,
            exp_range: 64,
            exp_scale: 32,
            use_freivalds: true,
        };
        (circuit, pi)
    }

    /// Sanity check: the base circuit passes verification.
    #[test]
    fn adversarial_baseline_passes() {
        let (circuit, pi) = make_valid_circuit();
        let prover = MockProver::run(K, &circuit, vec![pi]).unwrap();
        prover.assert_satisfied();
    }

    // ===================================================================
    // 1. Wrong old_state_hash (PI[0-1])
    // ===================================================================

    #[test]
    fn adversarial_wrong_old_state_hash_lo() {
        let (circuit, mut pi) = make_valid_circuit();
        // Tamper with PI[0]: old_state_hash_lo
        pi[0] = Fr::from(0xDEADBEEFu64);
        let prover = MockProver::run(K, &circuit, vec![pi]).unwrap();
        assert!(
            prover.verify().is_err(),
            "wrong old_state_hash_lo (PI[0]) must be rejected"
        );
    }

    #[test]
    fn adversarial_wrong_old_state_hash_hi() {
        let (circuit, mut pi) = make_valid_circuit();
        // Tamper with PI[1]: old_state_hash_hi
        pi[1] = Fr::from(0xCAFEBABEu64);
        let prover = MockProver::run(K, &circuit, vec![pi]).unwrap();
        assert!(
            prover.verify().is_err(),
            "wrong old_state_hash_hi (PI[1]) must be rejected"
        );
    }

    // ===================================================================
    // 2. Wrong new_state_hash (PI[2-3])
    // ===================================================================

    #[test]
    fn adversarial_wrong_new_state_hash_lo() {
        let (circuit, mut pi) = make_valid_circuit();
        // Tamper with PI[2]: new_state_hash_lo
        pi[2] = Fr::from(0xBAADF00Du64);
        let prover = MockProver::run(K, &circuit, vec![pi]).unwrap();
        assert!(
            prover.verify().is_err(),
            "wrong new_state_hash_lo (PI[2]) must be rejected"
        );
    }

    #[test]
    fn adversarial_wrong_new_state_hash_hi() {
        let (circuit, mut pi) = make_valid_circuit();
        // Tamper with PI[3]: new_state_hash_hi
        pi[3] = Fr::from(0xFEEDFACEu64);
        let prover = MockProver::run(K, &circuit, vec![pi]).unwrap();
        assert!(
            prover.verify().is_err(),
            "wrong new_state_hash_hi (PI[3]) must be rejected"
        );
    }

    // ===================================================================
    // 3. Wrong loss value (PI[4])
    // ===================================================================

    #[test]
    fn adversarial_wrong_loss() {
        let (circuit, mut pi) = make_valid_circuit();
        // Claim zero loss when the real loss is nonzero
        pi[4] = Fr::from(0u64);
        let prover = MockProver::run(K, &circuit, vec![pi]).unwrap();
        assert!(
            prover.verify().is_err(),
            "wrong loss (PI[4]) must be rejected"
        );
    }

    // ===================================================================
    // 4. Wrong error_bound (PI[5])
    // ===================================================================

    #[test]
    fn adversarial_wrong_error_bound() {
        let (circuit, mut pi) = make_valid_circuit();
        // Claim zero error bound when there is accumulated error
        pi[5] = Fr::from(0u64);
        let prover = MockProver::run(K, &circuit, vec![pi]).unwrap();
        assert!(
            prover.verify().is_err(),
            "wrong error_bound (PI[5]) must be rejected"
        );
    }

    // ===================================================================
    // 5. Wrong step_number (PI[6])
    // ===================================================================

    #[test]
    fn adversarial_wrong_step_number() {
        let (circuit, mut pi) = make_valid_circuit();
        // Claim a different step number
        pi[6] = Fr::from(999u64);
        let prover = MockProver::run(K, &circuit, vec![pi]).unwrap();
        assert!(
            prover.verify().is_err(),
            "wrong step_number (PI[6]) must be rejected"
        );
    }

    // ===================================================================
    // 6. Wrong error_checksum (PI[7])
    // ===================================================================

    #[test]
    fn adversarial_wrong_error_checksum() {
        let (circuit, mut pi) = make_valid_circuit();
        // Arbitrary checksum that doesn't match Poseidon(total_error, step, ...)
        pi[7] = Fr::from(0x1234u64);
        let prover = MockProver::run(K, &circuit, vec![pi]).unwrap();
        assert!(
            prover.verify().is_err(),
            "wrong error_checksum (PI[7]) must be rejected"
        );
    }

    // ===================================================================
    // 7. Incorrect gradient computation (dW doesn't match chain rule)
    // ===================================================================

    #[test]
    fn adversarial_wrong_gradient_dw2() {
        let (mut circuit, pi) = make_valid_circuit();
        // Tamper with dw2[0]: should be dy[0] * h[0], set to something else.
        // The circuit has an s_eq gate checking dw2[j*d_hid+k] == dy[j]*h[k].
        let honest_dw2_0 = circuit.witness.dw2[0];
        circuit.witness.dw2[0] = honest_dw2_0 + Fr::from(42u64);
        let prover = MockProver::run(K, &circuit, vec![pi]).unwrap();
        assert!(
            prover.verify().is_err(),
            "tampered gradient dW2 must be rejected"
        );
    }

    // ===================================================================
    // 8. Incorrect weight update (W_new != W_old - lr * dW)
    // ===================================================================

    #[test]
    fn adversarial_wrong_weight_update() {
        let (circuit, _pi) = make_valid_circuit();
        // Tamper with w1_new: the circuit constrains w1_new = w1 - lr*dw1
        // via s_sub gate. Change w1_new[0] and rebuild PI to match the
        // tampered new_state_hash, so only the weight update gate should fail.
        let mut bad_witness = circuit.witness.clone();
        bad_witness.w1_new[0] = bad_witness.w1_new[0] + Fr::from(7u64);

        // Recompute new_state_hash from tampered weights so PI[2-3] match
        // the tampered witness (isolating the weight-update gate failure).
        let bad_new_hash = compute_state_hash_v2(
            &bad_witness.w1_new, &bad_witness.b1_new,
            &bad_witness.w2_new, &bad_witness.b2_new,
        );
        bad_witness.new_state_hash = bad_new_hash;

        let bad_pi = bad_witness.public_inputs();
        let bad_circuit = MLTrainingStepV2Circuit {
            witness: bad_witness,
            relu_range: 128,
            exp_range: 64,
            exp_scale: 32,
            use_freivalds: true,
        };

        let prover = MockProver::run(K, &bad_circuit, vec![bad_pi]).unwrap();
        assert!(
            prover.verify().is_err(),
            "tampered weight update (w1_new) must be rejected by s_sub gate"
        );
    }

    // ===================================================================
    // 9. Incorrect ReLU (positive input mapped to zero)
    // ===================================================================

    #[test]
    fn adversarial_wrong_relu_positive_to_zero() {
        let (circuit, pi) = make_valid_circuit();
        let mut bad_witness = circuit.witness.clone();

        // Find a hidden unit that has a positive h_pre value
        // In our test case: w1 = [1,2,3,1], x = [1,1]
        // h_pre[0] = 1*1 + 2*1 + b1[0] = 3, h_pre[1] = 3*1 + 1*1 + b1[1] = 4
        // Both are positive, so h[0]=3, h[1]=4.
        // Tamper: set h[0] = 0 (claiming ReLU of a positive is zero)
        assert_ne!(bad_witness.h[0], Fr::ZERO, "h[0] should be positive for this test");
        bad_witness.h[0] = Fr::ZERO;

        // Use the original PI (which includes original state hashes, loss, etc.)
        // The ReLU lookup will reject (3, 0) since the table says (3, 3).
        let bad_circuit = MLTrainingStepV2Circuit {
            witness: bad_witness,
            relu_range: 128,
            exp_range: 64,
            exp_scale: 32,
            use_freivalds: true,
        };

        let prover = MockProver::run(K, &bad_circuit, vec![pi]).unwrap();
        assert!(
            prover.verify().is_err(),
            "ReLU: positive input mapped to zero must be rejected by lookup"
        );
    }

    // ===================================================================
    // 10. Swapped old/new hashes (replay with reversed direction)
    // ===================================================================

    #[test]
    fn adversarial_swapped_old_new_hashes() {
        let (circuit, mut pi) = make_valid_circuit();
        // Swap PI[0-1] with PI[2-3]: claim old hash is new and vice versa.
        // This simulates a replay attack where the prover reverses the
        // training direction.
        let old_lo = pi[0];
        let old_hi = pi[1];
        let new_lo = pi[2];
        let new_hi = pi[3];

        pi[0] = new_lo;
        pi[1] = new_hi;
        pi[2] = old_lo;
        pi[3] = old_hi;

        let prover = MockProver::run(K, &circuit, vec![pi]).unwrap();
        assert!(
            prover.verify().is_err(),
            "swapped old/new state hashes must be rejected"
        );
    }

    // ===================================================================
    // Additional adversarial tests beyond the 10 minimum
    // ===================================================================

    /// Tamper with the forward pass: claim wrong h_pre (matmul output).
    /// With the Freivalds n=1 fix, this should now be caught by
    /// verify_dot_product constraints.
    #[test]
    fn adversarial_wrong_h_pre_with_freivalds() {
        let (circuit, pi) = make_valid_circuit();
        let mut bad_witness = circuit.witness.clone();

        // Tamper h_pre[0]: the real value is 3 (=1*1+2*1), set to 100.
        // The Freivalds n=1 path now uses verify_dot_product which
        // creates s_mul + s_add gates for each element of the dot product.
        // The c[i] passed to verify_matmul_freivalds is h_pre[j] - b1[j],
        // so tampering h_pre[j] changes c[i] and breaks the dot product check.
        bad_witness.h_pre[0] = Fr::from(100u64);

        let bad_circuit = MLTrainingStepV2Circuit {
            witness: bad_witness,
            relu_range: 128,
            exp_range: 64,
            exp_scale: 32,
            use_freivalds: true,
        };

        let prover = MockProver::run(K, &bad_circuit, vec![pi]).unwrap();
        assert!(
            prover.verify().is_err(),
            "tampered h_pre must be rejected (Freivalds n=1 fix)"
        );
    }

    /// Tamper with backward pass: wrong dh (hidden gradient from W2^T * dy).
    #[test]
    fn adversarial_wrong_hidden_gradient_dh() {
        let (circuit, pi) = make_valid_circuit();
        let mut bad_witness = circuit.witness.clone();

        // dh[k] = sum_j(W2[j,d_hid+k] * dy[j]) is verified via verify_dot_product.
        let honest_dh_0 = bad_witness.dh[0];
        bad_witness.dh[0] = honest_dh_0 + Fr::from(50u64);

        let bad_circuit = MLTrainingStepV2Circuit {
            witness: bad_witness,
            relu_range: 128,
            exp_range: 64,
            exp_scale: 32,
            use_freivalds: true,
        };

        let prover = MockProver::run(K, &bad_circuit, vec![pi]).unwrap();
        assert!(
            prover.verify().is_err(),
            "tampered hidden gradient dh must be rejected"
        );
    }

    /// Tamper with the output gradient: wrong dy.
    #[test]
    fn adversarial_wrong_output_gradient_dy() {
        let (circuit, pi) = make_valid_circuit();
        let mut bad_witness = circuit.witness.clone();

        // dy[j] = 2*(y[j] - target[j]) is checked by s_mul + s_eq gates.
        let honest_dy = bad_witness.dy[0];
        bad_witness.dy[0] = honest_dy + Fr::ONE;

        let bad_circuit = MLTrainingStepV2Circuit {
            witness: bad_witness,
            relu_range: 128,
            exp_range: 64,
            exp_scale: 32,
            use_freivalds: true,
        };

        let prover = MockProver::run(K, &bad_circuit, vec![pi]).unwrap();
        assert!(
            prover.verify().is_err(),
            "tampered output gradient dy must be rejected"
        );
    }

    /// Tamper with the ReLU mask: set mask to 1 for a zero-output unit.
    /// In our test all h_pre are positive so this test uses a different
    /// setup where at least one h_pre is negative.
    #[test]
    fn adversarial_wrong_relu_mask() {
        // Build a model where at least one h_pre is negative.
        let d_in = 2;
        let d_hid = 2;
        let d_out = 1;

        // With w1 = [-1, -1, 1, 1], x = [1, 1]:
        // h_pre[0] = -1*1 + -1*1 = -2 (negative)
        // h_pre[1] = 1*1 + 1*1 = 2 (positive)
        // Note: use field negation for -1
        let neg1 = Fr::ZERO - Fr::ONE;
        let w1 = vec![neg1, neg1, Fr::from(1), Fr::from(1)];
        let b1 = vec![Fr::from(0), Fr::from(0)];
        let w2 = vec![Fr::from(1), Fr::from(1)];
        let b2 = vec![Fr::from(0)];

        let x = vec![Fr::from(1), Fr::from(1)];
        let target = vec![Fr::from(5)];
        let lr = Fr::from(1);
        let base_error = Fr::from(1);

        let old_hash = compute_state_hash_v2(&w1, &b1, &w2, &b2);
        let tmp = compute_witness_v2(
            d_in, d_hid, d_out, &x, &target, &w1, &b1, &w2, &b2, lr,
            old_hash, (Fr::ZERO, Fr::ZERO), 1, base_error,
        );
        let new_hash = compute_state_hash_v2(
            &tmp.w1_new, &tmp.b1_new, &tmp.w2_new, &tmp.b2_new,
        );
        let mut witness = compute_witness_v2(
            d_in, d_hid, d_out, &x, &target, &w1, &b1, &w2, &b2, lr,
            old_hash, new_hash, 1, base_error,
        );

        // Verify h[0] is indeed zero (ReLU of negative)
        assert_eq!(witness.h[0], Fr::ZERO);
        assert_eq!(witness.relu_mask[0], Fr::ZERO);

        // Tamper: flip relu_mask[0] to 1, which would let the gradient
        // flow through when it shouldn't. The s_mul gate for
        // dh_pre[k] = dh[k] * relu_mask[k] will now compute wrong.
        witness.relu_mask[0] = Fr::ONE;

        let pi = witness.public_inputs();
        let bad_circuit = MLTrainingStepV2Circuit {
            witness,
            relu_range: 128,
            exp_range: 64,
            exp_scale: 32,
            use_freivalds: true,
        };

        let prover = MockProver::run(K, &bad_circuit, vec![pi]).unwrap();
        assert!(
            prover.verify().is_err(),
            "tampered ReLU mask must be rejected"
        );
    }

    /// Tamper with bias gradient: db1 should equal dh_pre but we change it.
    #[test]
    fn adversarial_wrong_bias_gradient_db1() {
        let (circuit, pi) = make_valid_circuit();
        let mut bad_witness = circuit.witness.clone();

        // db1[j] is constrained to equal dh_pre[j] via s_eq gate
        bad_witness.db1[0] = bad_witness.db1[0] + Fr::from(13u64);

        let bad_circuit = MLTrainingStepV2Circuit {
            witness: bad_witness,
            relu_range: 128,
            exp_range: 64,
            exp_scale: 32,
            use_freivalds: true,
        };

        let prover = MockProver::run(K, &bad_circuit, vec![pi]).unwrap();
        assert!(
            prover.verify().is_err(),
            "tampered bias gradient db1 must be rejected"
        );
    }
}
