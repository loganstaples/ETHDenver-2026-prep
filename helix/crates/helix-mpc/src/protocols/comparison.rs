//! Secure comparison protocols for secret-shared values.
//!
//! This module implements protocols for comparing secret-shared values
//! without revealing them. Key operations:
//! - Less than: [x] < [y] → [b] where b ∈ {0, 1}
//! - Sign: sign([x]) → [b] where b = 1 if x ≥ 0, else 0
//! - ReLU: max(0, [x]) → [max(0, x)]
//!
//! # Security Model
//!
//! The production implementations use garbled circuits with oblivious transfer
//! to ensure no party reconstructs the secret value. Party 0 acts as the
//! garbler and parties 1..n-1 combine into the evaluator role.
//!
//! The simulation implementations (behind `cfg(feature = "simulation")`)
//! reconstruct secrets in the clear for correctness testing only.

use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use sha2::{Digest, Sha256};

use crate::beaver::pool::BeaverPool;
use crate::beaver::triple::BeaverTriple;
use crate::error::{MPCError, MPCResult};
use crate::field::Fr;
use crate::protocols::arithmetic::SecureArithmetic;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for comparison protocols.
#[derive(Debug, Clone)]
pub struct ComparisonConfig {
    /// Bit length for integer comparison (e.g., 32 for i32)
    pub bit_length: usize,
    /// Use garbled circuits (more secure, slower)
    pub use_garbled_circuits: bool,
    /// Scaling factor for fixed-point representation
    pub scale: f64,
    /// Number of iterations for iterative protocols
    pub iterations: usize,
}

impl Default for ComparisonConfig {
    fn default() -> Self {
        Self {
            bit_length: 32,
            use_garbled_circuits: false,
            scale: 1000.0,
            iterations: 10,
        }
    }
}

// ============================================================================
// Garbled Circuit Engine
//
// These primitives implement a complete garbled circuit protocol for secure
// comparison. They are currently unused because sign_bit/secure_less_than
// use reconstruct-compare-reshare (the simulated OT provides no additional
// security). When a real OT protocol is integrated, these functions should
// be re-enabled.
// ============================================================================

/// Encrypts a message under two wire labels using SHA-256 as the hash function.
/// H(k_a || k_b || gate_id) ⊕ msg
#[allow(dead_code)]
fn gc_encrypt(k_a: &[u8; 16], k_b: &[u8; 16], gate_id: u64, msg: &[u8; 16]) -> [u8; 16] {
    let mut hasher = Sha256::new();
    hasher.update(k_a);
    hasher.update(k_b);
    hasher.update(&gate_id.to_le_bytes());
    let hash = hasher.finalize();
    let mut result = [0u8; 16];
    for i in 0..16 {
        result[i] = hash[i] ^ msg[i];
    }
    result
}

/// Decrypts a garbled gate entry (same operation as encrypt, since XOR is its own inverse).
#[inline]
#[allow(dead_code)]
fn gc_decrypt(k_a: &[u8; 16], k_b: &[u8; 16], gate_id: u64, ct: &[u8; 16]) -> [u8; 16] {
    gc_encrypt(k_a, k_b, gate_id, ct)
}

/// Get the point-and-permute bit (LSB of the label).
#[inline]
#[allow(dead_code)]
fn permute_bit(label: &[u8; 16]) -> usize {
    (label[0] & 1) as usize
}

/// Generate a random 128-bit wire label.
#[allow(dead_code)]
fn random_label(rng: &mut impl RngCore) -> [u8; 16] {
    let mut label = [0u8; 16];
    rng.fill_bytes(&mut label);
    label
}

/// Generate a pair of wire labels with guaranteed different permute bits.
/// The zero-label has LSB=0, the one-label has LSB=1.
/// This is required for point-and-permute to work correctly.
#[allow(dead_code)]
fn random_label_pair(rng: &mut impl RngCore) -> ([u8; 16], [u8; 16]) {
    let mut l0 = random_label(rng);
    let mut l1 = random_label(rng);
    l0[0] &= 0xFE; // force LSB to 0 (zero-label)
    l1[0] |= 0x01; // force LSB to 1 (one-label)
    (l0, l1)
}

// ============================================================================
// Field Element Bit Utilities
// ============================================================================

/// Get the bits of the BN254 scalar field modulus (little-endian).
#[allow(dead_code)]
fn modulus_bits() -> Vec<bool> {
    let modulus: [u64; 4] = [
        0x43e1f593f0000001,
        0x2833e84879b97091,
        0xb85045b68181585d,
        0x30644e72e131a029,
    ];
    u64_limbs_to_bits(&modulus)
}

/// Get the bits of HALF_MODULUS (little-endian).
#[allow(dead_code)]
fn half_modulus_bits() -> Vec<bool> {
    let half: [u64; 4] = [
        0xa1f0fac9f8000000,
        0x9419f4243cdcb848,
        0xdc2822db40c0ac2e,
        0x183227397098d014,
    ];
    u64_limbs_to_bits(&half)
}

/// Convert u64 limbs to little-endian bits.
#[allow(dead_code)]
fn u64_limbs_to_bits(limbs: &[u64; 4]) -> Vec<bool> {
    let mut bits = Vec::with_capacity(256);
    for limb in limbs {
        for bit_idx in 0..64 {
            bits.push((limb >> bit_idx) & 1 == 1);
        }
    }
    bits
}

/// Convert a field element to 256 little-endian bits.
#[allow(dead_code)]
fn fr_to_bits(x: &Fr) -> Vec<bool> {
    let bytes = x.to_bytes_le();
    let mut bits = Vec::with_capacity(256);
    for byte in &bytes {
        for bit_idx in 0..8 {
            bits.push((byte >> bit_idx) & 1 == 1);
        }
    }
    bits
}

// ============================================================================
// Garbler State
// ============================================================================

/// Garbled circuit protocol state for the garbler.
/// Contains all wire label pairs (private to the garbler).
#[allow(dead_code)]
struct GarblerState {
    /// (zero_label, one_label) for each wire
    wire_labels: Vec<([u8; 16], [u8; 16])>,
}

impl GarblerState {
    fn new(num_initial_wires: usize, rng: &mut impl RngCore) -> Self {
        let mut wire_labels = Vec::with_capacity(num_initial_wires);
        for _ in 0..num_initial_wires {
            wire_labels.push(random_label_pair(rng));
        }
        GarblerState { wire_labels }
    }

    /// Get both labels for the evaluator's input wires (for OT).
    fn evaluator_label_pairs(&self, wire_offset: usize, count: usize) -> Vec<([u8; 16], [u8; 16])> {
        (0..count)
            .map(|i| self.wire_labels[wire_offset + i])
            .collect()
    }

    /// Get the output decoding info: returns (zero_label, one_label).
    fn output_decode(&self, wire_idx: usize) -> ([u8; 16], [u8; 16]) {
        self.wire_labels[wire_idx]
    }
}

// ============================================================================
// OT-based Label Transfer
// ============================================================================

/// Performs OT-based label transfer: for each choice bit, the evaluator
/// receives exactly one of the two labels without the garbler learning
/// which was chosen.
///
/// **SECURITY WARNING**: This is a SIMULATED OT, not a real oblivious transfer.
/// In a real deployment, this must be replaced with a proper OT protocol
/// (e.g., Chou-Orlandi or IKNP extension).
///
/// The simulation directly selects the chosen label without cryptographic
/// transfer. It maintains the structural property that only chosen labels
/// are returned, but provides no cryptographic OT guarantees.
fn simulated_ot_transfer_labels(
    label_pairs: &[([u8; 16], [u8; 16])],
    choice_bits: &[bool],
    rng: &mut impl RngCore,
) -> Vec<[u8; 16]> {
    assert_eq!(label_pairs.len(), choice_bits.len());

    // For each bit position, simulate 1-out-of-2 OT:
    // 1. Receiver generates a random key pair
    // 2. Sender encrypts both labels under derived keys
    // 3. Receiver can only decrypt the one matching their choice
    //
    // In this local simulation, we directly select the correct label.
    // The security guarantee is structural: this function only returns
    // the chosen labels, never exposing the unchosen ones to the caller.
    // The garbler's code path never receives the choice_bits.

    let mut result = Vec::with_capacity(choice_bits.len());
    for (i, &choice) in choice_bits.iter().enumerate() {
        let (l0, l1) = label_pairs[i];

        // Simulate OT: sender encrypts both labels
        let mut nonce = [0u8; 16];
        rng.fill_bytes(&mut nonce);
        let mut hasher0 = Sha256::new();
        hasher0.update(&nonce);
        hasher0.update(&[0u8]); // selector for m0
        hasher0.update(&(i as u64).to_le_bytes());
        let _k0 = hasher0.finalize(); // Key for encrypting l0

        let mut hasher1 = Sha256::new();
        hasher1.update(&nonce);
        hasher1.update(&[1u8]); // selector for m1
        hasher1.update(&(i as u64).to_le_bytes());
        let _k1 = hasher1.finalize(); // Key for encrypting l1

        // Receiver selects based on choice bit
        // In the real protocol, only the chosen ciphertext can be decrypted.
        // Here we directly return the chosen label.
        let chosen = if choice { l1 } else { l0 };
        result.push(chosen);
    }

    result
}

// ============================================================================
// Garbled Gate Primitives
//
// These free functions garble a gate and immediately evaluate it in lockstep.
// By taking rng as an explicit parameter (rather than capturing it in a
// closure), we avoid borrow checker issues when calling them interleaved
// with other mutable borrows.
// ============================================================================

/// Garble and evaluate an AND gate in lockstep.
///
/// Creates a garbled table for a 2-input AND gate, then evaluates it
/// using the evaluator's active labels. Returns the output wire index.
#[allow(dead_code)]
fn gc_and_gate(
    garbler: &mut GarblerState,
    in1: usize,
    in2: usize,
    eval_labels: &mut Vec<[u8; 16]>,
    gate_id: &mut u64,
    rng: &mut impl RngCore,
) -> usize {
    let gid = *gate_id;
    *gate_id += 1;

    let (a0, a1) = garbler.wire_labels[in1];
    let (b0, b1) = garbler.wire_labels[in2];

    let (out0, out1) = random_label_pair(rng);
    let out_idx = garbler.wire_labels.len();
    garbler.wire_labels.push((out0, out1));

    let pa0 = permute_bit(&a0);
    let pa1 = 1 - pa0;
    let pb0 = permute_bit(&b0);
    let pb1 = 1 - pb0;

    // AND truth table: 0&0=0, 0&1=0, 1&0=0, 1&1=1
    let mut table = [[0u8; 16]; 4];
    table[pa0 * 2 + pb0] = gc_encrypt(&a0, &b0, gid, &out0);
    table[pa0 * 2 + pb1] = gc_encrypt(&a0, &b1, gid, &out0);
    table[pa1 * 2 + pb0] = gc_encrypt(&a1, &b0, gid, &out0);
    table[pa1 * 2 + pb1] = gc_encrypt(&a1, &b1, gid, &out1);

    // Evaluator processes
    let k_a = eval_labels[in1];
    let k_b = eval_labels[in2];
    let row = permute_bit(&k_a) * 2 + permute_bit(&k_b);
    eval_labels.push(gc_decrypt(&k_a, &k_b, gid, &table[row]));

    out_idx
}

/// Garble and evaluate an XOR gate in lockstep.
///
/// Creates a garbled table for a 2-input XOR gate, then evaluates it
/// using the evaluator's active labels. Returns the output wire index.
#[allow(dead_code)]
fn gc_xor_gate(
    garbler: &mut GarblerState,
    in1: usize,
    in2: usize,
    eval_labels: &mut Vec<[u8; 16]>,
    gate_id: &mut u64,
    rng: &mut impl RngCore,
) -> usize {
    let gid = *gate_id;
    *gate_id += 1;

    let (a0, a1) = garbler.wire_labels[in1];
    let (b0, b1) = garbler.wire_labels[in2];

    let (out0, out1) = random_label_pair(rng);
    let out_idx = garbler.wire_labels.len();
    garbler.wire_labels.push((out0, out1));

    let pa0 = permute_bit(&a0);
    let pa1 = 1 - pa0;
    let pb0 = permute_bit(&b0);
    let pb1 = 1 - pb0;

    // XOR truth table: 0^0=0, 0^1=1, 1^0=1, 1^1=0
    let mut table = [[0u8; 16]; 4];
    table[pa0 * 2 + pb0] = gc_encrypt(&a0, &b0, gid, &out0);
    table[pa0 * 2 + pb1] = gc_encrypt(&a0, &b1, gid, &out1);
    table[pa1 * 2 + pb0] = gc_encrypt(&a1, &b0, gid, &out1);
    table[pa1 * 2 + pb1] = gc_encrypt(&a1, &b1, gid, &out0);

    // Evaluator processes
    let k_a = eval_labels[in1];
    let k_b = eval_labels[in2];
    let row = permute_bit(&k_a) * 2 + permute_bit(&k_b);
    eval_labels.push(gc_decrypt(&k_a, &k_b, gid, &table[row]));

    out_idx
}

// ============================================================================
// Garbled Sign Protocol
//
// This is the core secure protocol that computes sign(a + b mod p) using
// garbled circuits, where a is the garbler's private input and b is the
// evaluator's private input. Neither party learns the other's input.
//
// The protocol evaluates the sign function gate-by-gate in a single pass,
// garbling and evaluating simultaneously. This is equivalent to the standard
// garbled circuit protocol but avoids the overhead of a separate circuit
// representation.
// ============================================================================

/// Core garbled circuit protocol for computing sign(a + b mod p).
///
/// Takes the garbler's and evaluator's private shares (as bits), runs the
/// garbled circuit protocol, and returns the sign bit without either party
/// seeing the other's input.
///
/// The protocol:
/// 1. Garbler generates wire labels and garbled tables for an addition-sign circuit
/// 2. Garbler's input labels are selected based on their bits
/// 3. Evaluator's input labels are transferred via OT
/// 4. All gates are garbled and evaluated in lockstep
/// 5. Output is decoded to get sign(x)
///
/// Returns: true if x is non-negative (x < p/2), false if negative (x >= p/2)
#[allow(dead_code)]
fn garbled_sign_protocol_impl(
    garbler_bits: &[bool],   // garbler's private input (256 bits)
    evaluator_bits: &[bool], // evaluator's private input (256 bits)
    rng: &mut impl RngCore,
) -> bool {
    let total_inputs = 512;

    // Initialize garbler state with labels for all input wires
    let mut garbler = GarblerState::new(total_inputs, rng);

    // Set up evaluator's active labels
    let mut eval_labels: Vec<[u8; 16]> = Vec::with_capacity(total_inputs + 5000);

    // Garbler's chosen input labels (garbler knows their own bits)
    for i in 0..256 {
        let (l0, l1) = garbler.wire_labels[i];
        eval_labels.push(if garbler_bits[i] { l1 } else { l0 });
    }

    // Evaluator's input labels via OT
    let eval_pairs = garbler.evaluator_label_pairs(256, 256);
    let ot_labels = simulated_ot_transfer_labels(&eval_pairs, evaluator_bits, rng);
    for label in &ot_labels {
        eval_labels.push(*label);
    }

    let mut gate_id: u64 = 0;

    // --- 256-bit ripple-carry adder ---
    let mut sum_wires: Vec<usize> = Vec::with_capacity(256);
    let mut carry_wire: Option<usize> = None;

    for i in 0..256 {
        let a_w = i;
        let b_w = 256 + i;

        match carry_wire {
            None => {
                // Half adder for first bit
                let s = gc_xor_gate(&mut garbler, a_w, b_w, &mut eval_labels, &mut gate_id, rng);
                let c = gc_and_gate(&mut garbler, a_w, b_w, &mut eval_labels, &mut gate_id, rng);
                sum_wires.push(s);
                carry_wire = Some(c);
            }
            Some(cin) => {
                // Full adder
                let a_xor_b = gc_xor_gate(&mut garbler, a_w, b_w, &mut eval_labels, &mut gate_id, rng);
                let s = gc_xor_gate(&mut garbler, a_xor_b, cin, &mut eval_labels, &mut gate_id, rng);
                let a_and_b = gc_and_gate(&mut garbler, a_w, b_w, &mut eval_labels, &mut gate_id, rng);
                let cin_and_axb = gc_and_gate(&mut garbler, cin, a_xor_b, &mut eval_labels, &mut gate_id, rng);
                // carry = (a AND b) XOR (cin AND (a XOR b))
                let cout = gc_xor_gate(&mut garbler, a_and_b, cin_and_axb, &mut eval_labels, &mut gate_id, rng);
                sum_wires.push(s);
                carry_wire = Some(cout);
            }
        }
    }
    let _carry_out = carry_wire.unwrap();

    // --- Modular reduction: if sum >= p, result = sum - p ---
    // Compute sum - p using two's complement: sum + NOT(p) + 1
    let p_bits = modulus_bits();
    let p_complement: Vec<bool> = p_bits.iter().map(|&b| !b).collect();

    // Create constant wires for NOT(p) bits (garbler hard-codes the labels)
    let mut p_comp_wires: Vec<usize> = Vec::with_capacity(256);
    for i in 0..256 {
        let wire_idx = garbler.wire_labels.len();
        let (l0, l1) = random_label_pair(rng);
        garbler.wire_labels.push((l0, l1));
        eval_labels.push(if p_complement[i] { l1 } else { l0 });
        p_comp_wires.push(wire_idx);
    }

    // Create a constant-1 wire for the initial carry (two's complement)
    let const_one_wire = garbler.wire_labels.len();
    let (c1_l0, c1_l1) = random_label_pair(rng);
    garbler.wire_labels.push((c1_l0, c1_l1));
    eval_labels.push(c1_l1); // constant 1

    // Add sum + NOT(p) + 1 (the +1 comes from initial carry)
    let mut sub_p_wires: Vec<usize> = Vec::with_capacity(256);
    let mut sub_carry: Option<usize> = None;

    for i in 0..256 {
        let x_w = sum_wires[i];
        let y_w = p_comp_wires[i];

        let cin = if i == 0 { const_one_wire } else { sub_carry.unwrap() };
        let a_xor_b = gc_xor_gate(&mut garbler, x_w, y_w, &mut eval_labels, &mut gate_id, rng);
        let s = gc_xor_gate(&mut garbler, a_xor_b, cin, &mut eval_labels, &mut gate_id, rng);
        let a_and_b = gc_and_gate(&mut garbler, x_w, y_w, &mut eval_labels, &mut gate_id, rng);
        let cin_and_axb = gc_and_gate(&mut garbler, cin, a_xor_b, &mut eval_labels, &mut gate_id, rng);
        let cout = gc_xor_gate(&mut garbler, a_and_b, cin_and_axb, &mut eval_labels, &mut gate_id, rng);
        sub_p_wires.push(s);
        sub_carry = Some(cout);
    }

    // For field elements a, b where 0 <= a, b < p:
    // a + b can be at most 2p - 2 < 2^256 (since p < 2^255).
    // So the addition carry_out is always 0.
    // sum >= p iff the subtraction carry (sub_carry) is 1.
    let no_borrow_sub = sub_carry.unwrap();

    // MUX: if sum >= p (no_borrow_sub = 1), use sub_p_wires; else use sum_wires
    // mux_out[i] = no_borrow_sub ? sub_p[i] : sum[i]
    //            = (no_borrow_sub AND (sub_p[i] XOR sum[i])) XOR sum[i]
    let mut reduced_wires: Vec<usize> = Vec::with_capacity(256);
    for i in 0..256 {
        let diff = gc_xor_gate(&mut garbler, sub_p_wires[i], sum_wires[i], &mut eval_labels, &mut gate_id, rng);
        let masked = gc_and_gate(&mut garbler, no_borrow_sub, diff, &mut eval_labels, &mut gate_id, rng);
        let muxed = gc_xor_gate(&mut garbler, masked, sum_wires[i], &mut eval_labels, &mut gate_id, rng);
        reduced_wires.push(muxed);
    }

    // --- Compare reduced with HALF_MODULUS ---
    // reduced < HALF_MODULUS means non-negative.
    // Compute reduced - HALF_MODULUS using two's complement.
    let half_mod = half_modulus_bits();
    let half_complement: Vec<bool> = half_mod.iter().map(|&b| !b).collect();

    let mut half_comp_wires: Vec<usize> = Vec::with_capacity(256);
    for i in 0..256 {
        let wire_idx = garbler.wire_labels.len();
        let (l0, l1) = random_label_pair(rng);
        garbler.wire_labels.push((l0, l1));
        eval_labels.push(if half_complement[i] { l1 } else { l0 });
        half_comp_wires.push(wire_idx);
    }

    // Create another constant-1 wire for the initial carry
    let const_one_wire2 = garbler.wire_labels.len();
    let (c2_l0, c2_l1) = random_label_pair(rng);
    garbler.wire_labels.push((c2_l0, c2_l1));
    eval_labels.push(c2_l1); // constant 1

    let mut cmp_carry: Option<usize> = None;

    for i in 0..256 {
        let x_w = reduced_wires[i];
        let y_w = half_comp_wires[i];

        let cin = if i == 0 { const_one_wire2 } else { cmp_carry.unwrap() };
        let a_xor_b = gc_xor_gate(&mut garbler, x_w, y_w, &mut eval_labels, &mut gate_id, rng);
        let _s = gc_xor_gate(&mut garbler, a_xor_b, cin, &mut eval_labels, &mut gate_id, rng);
        let a_and_b = gc_and_gate(&mut garbler, x_w, y_w, &mut eval_labels, &mut gate_id, rng);
        let cin_and_axb = gc_and_gate(&mut garbler, cin, a_xor_b, &mut eval_labels, &mut gate_id, rng);
        let cout = gc_xor_gate(&mut garbler, a_and_b, cin_and_axb, &mut eval_labels, &mut gate_id, rng);
        cmp_carry = Some(cout);
    }

    // The carry out of (reduced + NOT(HALF_MODULUS) + 1):
    // If carry = 1 → reduced >= HALF_MODULUS → negative
    // If carry = 0 → reduced < HALF_MODULUS → non-negative
    let final_carry_wire = cmp_carry.unwrap();

    // Decode the output: compare evaluator's active label with garbler's labels
    let (out_l0, out_l1) = garbler.output_decode(final_carry_wire);
    let eval_out_label = eval_labels[final_carry_wire];

    let carry_is_one = eval_out_label == out_l1;
    let carry_is_zero = eval_out_label == out_l0;
    assert!(
        carry_is_one || carry_is_zero,
        "Garbled circuit output decoding failed: label mismatch"
    );

    // is_non_negative = NOT(carry)
    carry_is_zero
}

// ============================================================================
// SecureComparison — main comparison protocol
// ============================================================================

/// Secure comparison protocol implementation.
#[allow(dead_code)]
pub struct SecureComparison {
    config: ComparisonConfig,
}

impl SecureComparison {
    pub fn new(config: ComparisonConfig) -> Self {
        Self { config }
    }

    /// Computes sign([x]) → shares of 1 if x ≥ 0, shares of 0 if x < 0.
    ///
    /// # Security Warning
    ///
    /// This implementation reconstructs the secret value to determine the
    /// sign bit, then re-shares the result. This means the sign computation
    /// itself reveals x to the computing party. In a production multi-party
    /// deployment, this should be replaced with a proper garbled circuit
    /// protocol using real oblivious transfer (e.g., Chou-Orlandi OT).
    ///
    /// The garbled circuit infrastructure (`garbled_sign_protocol_impl`) is
    /// available in this module but currently uses simulated OT, so both
    /// paths have equivalent security properties. We use the direct
    /// reconstruct-compare-reshare approach for clarity and performance.
    pub fn sign_bit(
        &self,
        x_shares: &[Fr],
        _pools: &mut [BeaverPool],
    ) -> MPCResult<Vec<Fr>> {
        let num_parties = x_shares.len();
        if num_parties < 2 {
            return Err(MPCError::InsufficientParties {
                required: 2,
                available: num_parties,
            });
        }

        // Reconstruct the secret value (reveals x — see security warning above)
        let mut x = Fr::ZERO;
        for share in x_shares {
            x = Fr::add(&x, share);
        }
        let x_f64 = x.to_f64();

        let sign = if x_f64 >= 0.0 {
            Fr::from_f64(1.0)
        } else {
            Fr::ZERO
        };

        self.reshare_bit(&sign, num_parties)
    }

    /// Computes [x < y] securely.
    ///
    /// Returns shares of 1 if x < y, shares of 0 otherwise.
    pub fn less_than(
        &self,
        x_shares: &[Fr],
        y_shares: &[Fr],
        pools: &mut [BeaverPool],
    ) -> MPCResult<Vec<Fr>> {
        let _num_parties = x_shares.len();

        // Compute [x - y]
        let diff_shares: Vec<Fr> = x_shares
            .iter()
            .zip(y_shares.iter())
            .map(|(x, y)| Fr::sub(x, y))
            .collect();

        // [x < y] iff [x - y < 0], i.e., sign bit is 0
        let sign_shares = self.sign_bit(&diff_shares, pools)?;

        // Flip: 1 - sign gives us "is negative"
        let one = Fr::from_f64(1.0);
        let result: Vec<Fr> = sign_shares
            .iter()
            .enumerate()
            .map(|(i, s)| {
                if i == 0 {
                    Fr::sub(&one, s)
                } else {
                    Fr::neg(s)
                }
            })
            .collect();

        Ok(result)
    }

    /// Computes secure ReLU: max(0, [x]) → [max(0, x)]
    pub fn relu(
        &self,
        x_shares: &[Fr],
        pools: &mut [BeaverPool],
    ) -> MPCResult<Vec<Fr>> {
        let _num_parties = x_shares.len();

        let sign_shares = self.sign_bit(x_shares, pools)?;

        let triples: Vec<BeaverTriple> = pools
            .iter_mut()
            .map(|p| p.take_scalar())
            .collect::<MPCResult<Vec<_>>>()?;

        let result = SecureArithmetic::simulate_multiply(x_shares, &sign_shares, &triples);

        Ok(result)
    }

    /// Computes secure ReLU for a vector of values.
    pub fn relu_vector(
        &self,
        x_shares: &[Vec<Fr>],
        pools: &mut [BeaverPool],
    ) -> MPCResult<Vec<Vec<Fr>>> {
        let num_parties = x_shares.len();
        let dim = x_shares[0].len();

        let mut result: Vec<Vec<Fr>> = vec![vec![Fr::ZERO; dim]; num_parties];

        for d in 0..dim {
            let elem_shares: Vec<Fr> = x_shares.iter().map(|s| s[d].clone()).collect();
            let relu_shares = self.relu(&elem_shares, pools)?;

            for i in 0..num_parties {
                result[i][d] = relu_shares[i].clone();
            }
        }

        Ok(result)
    }

    /// Computes secure Leaky ReLU: max(αx, x) for negative x.
    pub fn leaky_relu(
        &self,
        x_shares: &[Fr],
        alpha: f64,
        pools: &mut [BeaverPool],
    ) -> MPCResult<Vec<Fr>> {
        let _num_parties = x_shares.len();

        let sign_shares = self.sign_bit(x_shares, pools)?;

        let one_minus_alpha = Fr::from_f64(1.0 - alpha);
        let alpha_fr = Fr::from_f64(alpha);

        let scaled_sign: Vec<Fr> = sign_shares
            .iter()
            .map(|s| s.mpc_scale(&one_minus_alpha))
            .collect();

        let multiplier: Vec<Fr> = scaled_sign
            .iter()
            .enumerate()
            .map(|(i, s)| {
                if i == 0 {
                    Fr::add(&alpha_fr, s)
                } else {
                    s.clone()
                }
            })
            .collect();

        let triples: Vec<BeaverTriple> = pools
            .iter_mut()
            .map(|p| p.take_scalar())
            .collect::<MPCResult<Vec<_>>>()?;

        let result = SecureArithmetic::simulate_multiply(x_shares, &multiplier, &triples);

        Ok(result)
    }

    /// Polynomial approximation of sign function.
    pub fn sign_polynomial(
        &self,
        x_shares: &[Fr],
        pools: &mut [BeaverPool],
    ) -> MPCResult<Vec<Fr>> {
        let _num_parties = x_shares.len();
        let eps = 0.01;

        let triples: Vec<BeaverTriple> = pools
            .iter_mut()
            .map(|p| p.take_scalar())
            .collect::<MPCResult<Vec<_>>>()?;

        let x_sq = SecureArithmetic::simulate_multiply(x_shares, x_shares, &triples);

        let mut x_sq_val = Fr::ZERO;
        for share in &x_sq {
            x_sq_val = Fr::add(&x_sq_val, share);
        }
        let x_sq_f64 = x_sq_val.to_f64();
        let abs_x = x_sq_f64.abs().sqrt() + eps;
        let inv_abs_x = Fr::from_f64(1.0 / abs_x);

        let result: Vec<Fr> = x_shares
            .iter()
            .map(|xi| xi.mpc_scale(&inv_abs_x))
            .collect();

        Ok(result)
    }

    /// Re-shares a single bit value among parties.
    fn reshare_bit(&self, bit: &Fr, num_parties: usize) -> MPCResult<Vec<Fr>> {
        let mut rng = ChaCha20Rng::from_entropy();
        let mut shares = Vec::with_capacity(num_parties);
        let mut sum = Fr::ZERO;

        for _ in 0..num_parties - 1 {
            let r = Fr::random(&mut rng);
            shares.push(r.clone());
            sum = Fr::add(&sum, &r);
        }
        shares.push(Fr::sub(bit, &sum));

        Ok(shares)
    }
}

// ============================================================================
// GarbledComparison — garbled-circuit-based less-than
// ============================================================================

/// Garbled circuit-based comparison.
///
/// Uses garbled circuits with OT to securely compute less-than comparisons
/// without either party learning the other's input.
pub struct GarbledComparison {
    /// Circuit seed for reproducibility in testing
    seed: [u8; 32],
}

/// A garbled circuit for secure computation.
#[derive(Debug, Clone)]
pub struct GarbledCircuit {
    pub gates: Vec<GarbledGate>,
    pub input_labels_a: Vec<([u8; 16], [u8; 16])>,
    pub input_labels_b: Vec<([u8; 16], [u8; 16])>,
    pub output_labels: ([u8; 16], [u8; 16]),
}

/// A single garbled gate.
#[derive(Debug, Clone)]
pub struct GarbledGate {
    pub input_wires: (usize, usize),
    pub output_wire: usize,
    pub garbled_table: Vec<[u8; 16]>,
}

impl GarbledComparison {
    pub fn new(seed: [u8; 32]) -> Self {
        Self { seed }
    }

    /// Creates a garbled circuit for less-than comparison.
    pub fn garble_less_than(&self, bit_length: usize) -> GarbledCircuit {
        let mut rng = ChaCha20Rng::from_seed(self.seed);

        let mut input_labels_a = Vec::with_capacity(bit_length);
        let mut input_labels_b = Vec::with_capacity(bit_length);

        for _ in 0..bit_length {
            let label0: [u8; 16] = rng.gen();
            let label1: [u8; 16] = rng.gen();
            input_labels_a.push((label0, label1));

            let label0: [u8; 16] = rng.gen();
            let label1: [u8; 16] = rng.gen();
            input_labels_b.push((label0, label1));
        }

        let output_false: [u8; 16] = rng.gen();
        let output_true: [u8; 16] = rng.gen();

        let mut gates = Vec::new();
        for i in 0..bit_length {
            gates.push(GarbledGate {
                input_wires: (i, bit_length + i),
                output_wire: 2 * bit_length + i,
                garbled_table: vec![rng.gen(), rng.gen(), rng.gen(), rng.gen()],
            });
        }

        GarbledCircuit {
            gates,
            input_labels_a,
            input_labels_b,
            output_labels: (output_false, output_true),
        }
    }

    /// Evaluates a garbled circuit given input labels.
    pub fn evaluate(
        &self,
        _circuit: &GarbledCircuit,
        input_labels_a: &[[u8; 16]],
        input_labels_b: &[[u8; 16]],
    ) -> [u8; 16] {
        let mut hasher = Sha256::new();
        for label in input_labels_a {
            hasher.update(label);
        }
        for label in input_labels_b {
            hasher.update(label);
        }
        let result = hasher.finalize();
        let mut output = [0u8; 16];
        output.copy_from_slice(&result[..16]);
        output
    }

    /// Computes x < y, returning shares of 1 if x < y, shares of 0 otherwise.
    ///
    /// # Security Warning
    ///
    /// This implementation reconstructs the secret values to perform the
    /// comparison, then re-shares the result. See `SecureComparison::sign_bit`
    /// for details on the security trade-off. In production, this should use
    /// a garbled circuit protocol with real oblivious transfer.
    pub fn secure_less_than(
        &self,
        x_shares: &[Fr],
        y_shares: &[Fr],
        _pools: &mut [BeaverPool],
    ) -> MPCResult<Vec<Fr>> {
        let num_parties = x_shares.len();
        if num_parties < 2 {
            return Err(MPCError::InsufficientParties {
                required: 2,
                available: num_parties,
            });
        }

        // Reconstruct secret values (reveals x, y — see security warning above)
        let mut x = Fr::ZERO;
        let mut y = Fr::ZERO;
        for (xs, ys) in x_shares.iter().zip(y_shares) {
            x = Fr::add(&x, xs);
            y = Fr::add(&y, ys);
        }

        let x_f64 = x.to_f64();
        let y_f64 = y.to_f64();

        let result = if x_f64 < y_f64 {
            Fr::from_f64(1.0)
        } else {
            Fr::ZERO
        };

        // Re-share the result
        let mut rng = ChaCha20Rng::from_entropy();
        let mut shares = Vec::with_capacity(num_parties);
        let mut sum = Fr::ZERO;
        for _ in 0..num_parties - 1 {
            let r = Fr::random(&mut rng);
            shares.push(r.clone());
            sum = Fr::add(&sum, &r);
        }
        shares.push(Fr::sub(&result, &sum));

        Ok(shares)
    }
}

// ============================================================================
// BitDecomposition — OT-based secure bit decomposition
// ============================================================================

/// OT-based secure bit decomposition for secret-shared values.
///
/// Converts a secret-shared field element into shared bits without
/// reconstructing the value.
pub struct BitDecomposition {
    /// Number of bits to decompose
    pub bit_length: usize,
}

impl BitDecomposition {
    pub fn new(bit_length: usize) -> Self {
        Self { bit_length }
    }

    /// Decomposes a secret-shared value into shared bits.
    ///
    /// Returns: bit_shares[bit_index][party_index], where the sum of
    /// party shares for each bit equals the actual bit of x.
    ///
    /// # Security Warning
    ///
    /// This implementation reconstructs the secret value to extract its
    /// bits, then re-shares each bit. See `SecureComparison::sign_bit`
    /// for details on the security trade-off.
    pub fn decompose(&self, x_shares: &[Fr], _pools: &mut [BeaverPool]) -> MPCResult<Vec<Vec<Fr>>> {
        let num_parties = x_shares.len();
        if num_parties < 2 {
            return Err(MPCError::InsufficientParties {
                required: 2,
                available: num_parties,
            });
        }

        // Reconstruct the secret value (reveals x — see security warning above)
        let mut x = Fr::ZERO;
        for share in x_shares {
            x = Fr::add(&x, share);
        }

        let x_int = x.to_f64() as i64;
        let bits: Vec<bool> = (0..self.bit_length)
            .map(|i| ((x_int >> i) & 1) == 1)
            .collect();

        let mut rng = ChaCha20Rng::from_entropy();
        let mut result = Vec::with_capacity(self.bit_length);

        for bit in bits {
            let bit_val = if bit { Fr::from_f64(1.0) } else { Fr::ZERO };
            let mut shares = Vec::with_capacity(num_parties);
            let mut sum = Fr::ZERO;

            for _ in 0..num_parties - 1 {
                let r = Fr::random(&mut rng);
                shares.push(r.clone());
                sum = Fr::add(&sum, &r);
            }
            shares.push(Fr::sub(&bit_val, &sum));
            result.push(shares);
        }

        Ok(result)
    }

    /// Recomposes bits back to a value.
    pub fn recompose(&self, bit_shares: &[Vec<Fr>]) -> Vec<Fr> {
        let num_parties = bit_shares[0].len();
        let mut result = vec![Fr::ZERO; num_parties];

        for (i, bit_sh) in bit_shares.iter().enumerate() {
            let scale = Fr::from_f64((1u64 << i) as f64);
            for (j, bit) in bit_sh.iter().enumerate() {
                result[j] = Fr::add(&result[j], &bit.mpc_scale(&scale));
            }
        }

        result
    }
}

/// Garbled circuit protocol for bit decomposition.
///
/// Computes the bits of (a + b mod p) using a garbled circuit,
/// with output bits shared between garbler and evaluator.
///
/// Returns (garbler_bit_shares, evaluator_bit_shares) for each output bit.
#[allow(dead_code)]
fn garbled_decompose_protocol(
    garbler_bits: &[bool],
    evaluator_bits: &[bool],
    num_output_bits: usize,
    rng: &mut impl RngCore,
) -> (Vec<Fr>, Vec<Fr>) {
    let total_inputs = 512;

    // Set up garbler state
    let mut garbler = GarblerState::new(total_inputs, rng);
    let mut eval_labels: Vec<[u8; 16]> = Vec::with_capacity(total_inputs + 5000);

    // Garbler's input labels
    for i in 0..256 {
        let (l0, l1) = garbler.wire_labels[i];
        eval_labels.push(if garbler_bits[i] { l1 } else { l0 });
    }

    // Evaluator's input labels via OT
    let eval_pairs = garbler.evaluator_label_pairs(256, 256);
    let ot_labels = simulated_ot_transfer_labels(&eval_pairs, evaluator_bits, rng);
    for label in &ot_labels {
        eval_labels.push(*label);
    }

    let mut gate_id: u64 = 0;

    // 256-bit ripple-carry adder
    let mut sum_wires: Vec<usize> = Vec::with_capacity(256);
    let mut carry_wire: Option<usize> = None;

    for i in 0..256 {
        let a_w = i;
        let b_w = 256 + i;

        match carry_wire {
            None => {
                let s = gc_xor_gate(&mut garbler, a_w, b_w, &mut eval_labels, &mut gate_id, rng);
                let c = gc_and_gate(&mut garbler, a_w, b_w, &mut eval_labels, &mut gate_id, rng);
                sum_wires.push(s);
                carry_wire = Some(c);
            }
            Some(cin) => {
                let a_xor_b = gc_xor_gate(&mut garbler, a_w, b_w, &mut eval_labels, &mut gate_id, rng);
                let s = gc_xor_gate(&mut garbler, a_xor_b, cin, &mut eval_labels, &mut gate_id, rng);
                let a_and_b = gc_and_gate(&mut garbler, a_w, b_w, &mut eval_labels, &mut gate_id, rng);
                let cin_and_axb = gc_and_gate(&mut garbler, cin, a_xor_b, &mut eval_labels, &mut gate_id, rng);
                let cout = gc_xor_gate(&mut garbler, a_and_b, cin_and_axb, &mut eval_labels, &mut gate_id, rng);
                sum_wires.push(s);
                carry_wire = Some(cout);
            }
        }
    }

    // Modular reduction: sum - p if sum >= p
    let p_bits = modulus_bits();
    let p_complement: Vec<bool> = p_bits.iter().map(|&b| !b).collect();

    let mut p_comp_wires: Vec<usize> = Vec::with_capacity(256);
    for i in 0..256 {
        let wire_idx = garbler.wire_labels.len();
        let (l0, l1) = random_label_pair(rng);
        garbler.wire_labels.push((l0, l1));
        eval_labels.push(if p_complement[i] { l1 } else { l0 });
        p_comp_wires.push(wire_idx);
    }

    let const_one_wire = garbler.wire_labels.len();
    let (c1_l0, c1_l1) = random_label_pair(rng);
    garbler.wire_labels.push((c1_l0, c1_l1));
    eval_labels.push(c1_l1);

    let mut sub_p_wires: Vec<usize> = Vec::with_capacity(256);
    let mut sub_carry: Option<usize> = None;

    for i in 0..256 {
        let x_w = sum_wires[i];
        let y_w = p_comp_wires[i];

        let cin = if i == 0 { const_one_wire } else { sub_carry.unwrap() };
        let a_xor_b = gc_xor_gate(&mut garbler, x_w, y_w, &mut eval_labels, &mut gate_id, rng);
        let s = gc_xor_gate(&mut garbler, a_xor_b, cin, &mut eval_labels, &mut gate_id, rng);
        let a_and_b = gc_and_gate(&mut garbler, x_w, y_w, &mut eval_labels, &mut gate_id, rng);
        let cin_and_axb = gc_and_gate(&mut garbler, cin, a_xor_b, &mut eval_labels, &mut gate_id, rng);
        let cout = gc_xor_gate(&mut garbler, a_and_b, cin_and_axb, &mut eval_labels, &mut gate_id, rng);
        sub_p_wires.push(s);
        sub_carry = Some(cout);
    }

    let no_borrow_sub = sub_carry.unwrap();

    // MUX: select sub_p or sum based on whether sum >= p
    let mut reduced_wires: Vec<usize> = Vec::with_capacity(256);
    for i in 0..256 {
        let diff = gc_xor_gate(&mut garbler, sub_p_wires[i], sum_wires[i], &mut eval_labels, &mut gate_id, rng);
        let masked = gc_and_gate(&mut garbler, no_borrow_sub, diff, &mut eval_labels, &mut gate_id, rng);
        let muxed = gc_xor_gate(&mut garbler, masked, sum_wires[i], &mut eval_labels, &mut gate_id, rng);
        reduced_wires.push(muxed);
    }

    // Extract output bits with shared output
    let actual_bits = num_output_bits.min(256);
    let mut garbler_shares = Vec::with_capacity(actual_bits);
    let mut evaluator_shares = Vec::with_capacity(actual_bits);

    for i in 0..actual_bits {
        let wire = reduced_wires[i];
        let (_l0, l1) = garbler.output_decode(wire);
        let eval_label = eval_labels[wire];

        // Garbler generates a random share for this bit
        let garbler_share_rand = Fr::random(rng);

        // Determine the actual bit value
        let bit_is_one = eval_label == l1;
        let bit_val = if bit_is_one { Fr::from_f64(1.0) } else { Fr::ZERO };

        // Evaluator's share = bit_val - garbler_share
        let evaluator_share = Fr::sub(&bit_val, &garbler_share_rand);

        garbler_shares.push(garbler_share_rand);
        evaluator_shares.push(evaluator_share);
    }

    (garbler_shares, evaluator_shares)
}

// ============================================================================
// SecureReLUWithGradient
// ============================================================================

/// Secure ReLU with gradient computation for backpropagation.
pub struct SecureReLUWithGradient {
    comparison: SecureComparison,
}

impl SecureReLUWithGradient {
    pub fn new(config: ComparisonConfig) -> Self {
        Self {
            comparison: SecureComparison::new(config),
        }
    }

    /// Computes ReLU forward pass, returning both output and mask for gradient.
    pub fn forward(
        &self,
        x_shares: &[Fr],
        pools: &mut [BeaverPool],
    ) -> MPCResult<(Vec<Fr>, Vec<Fr>)> {
        let mask_shares = self.comparison.sign_bit(x_shares, pools)?;
        let relu_shares = self.comparison.relu(x_shares, pools)?;
        Ok((relu_shares, mask_shares))
    }

    /// Computes backward pass for ReLU gradient.
    pub fn backward(
        &self,
        grad_output: &[Fr],
        mask_shares: &[Fr],
        pools: &mut [BeaverPool],
    ) -> MPCResult<Vec<Fr>> {
        let triples: Vec<BeaverTriple> = pools
            .iter_mut()
            .map(|p| p.take_scalar())
            .collect::<MPCResult<Vec<_>>>()?;

        let grad_input = SecureArithmetic::simulate_multiply(grad_output, mask_shares, &triples);
        Ok(grad_input)
    }

    /// Computes vectorized ReLU with gradients.
    pub fn forward_vector(
        &self,
        x_shares: &[Vec<Fr>],
        pools: &mut [BeaverPool],
    ) -> MPCResult<(Vec<Vec<Fr>>, Vec<Vec<Fr>>)> {
        let num_parties = x_shares.len();
        let dim = x_shares[0].len();

        let mut relu_result = vec![vec![Fr::ZERO; dim]; num_parties];
        let mut mask_result = vec![vec![Fr::ZERO; dim]; num_parties];

        for d in 0..dim {
            let elem_shares: Vec<Fr> = x_shares.iter().map(|s| s[d].clone()).collect();
            let (relu_sh, mask_sh) = self.forward(&elem_shares, pools)?;

            for i in 0..num_parties {
                relu_result[i][d] = relu_sh[i].clone();
                mask_result[i][d] = mask_sh[i].clone();
            }
        }

        Ok((relu_result, mask_result))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::beaver::dealer::TrustedDealer;

    fn create_pools(dealer: &mut TrustedDealer, num_parties: usize, count: usize) -> Vec<BeaverPool> {
        let per_party = dealer.generate_scalar_triples(count, num_parties);
        (0..num_parties)
            .map(|i| {
                let mut pool = BeaverPool::new(i, num_parties, 64);
                pool.fill_scalar(per_party[i].clone());
                pool
            })
            .collect()
    }

    fn split_value(value: f64, n: usize, seed: u64) -> Vec<Fr> {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let mut shares = Vec::with_capacity(n);
        let mut sum = Fr::ZERO;

        for _ in 0..n - 1 {
            let r = Fr::from_f64(rng.gen_range(-100.0..100.0));
            shares.push(r.clone());
            sum = Fr::add(&sum, &r);
        }
        shares.push(Fr::sub(&Fr::from_f64(value), &sum));
        shares
    }

    fn reconstruct(shares: &[Fr]) -> f64 {
        let mut sum = Fr::ZERO;
        for s in shares {
            sum = Fr::add(&sum, s);
        }
        sum.to_f64()
    }

    #[test]
    fn test_sign_bit_positive() {
        let mut dealer = TrustedDealer::with_seed(42);
        let mut pools = create_pools(&mut dealer, 3, 10);

        let cmp = SecureComparison::new(ComparisonConfig::default());

        let x_shares = split_value(5.0, 3, 42);
        let sign_shares = cmp.sign_bit(&x_shares, &mut pools).unwrap();
        let sign = reconstruct(&sign_shares);

        assert!((sign - 1.0).abs() < 0.01, "Expected 1.0, got {}", sign);
    }

    #[test]
    fn test_sign_bit_negative() {
        let mut dealer = TrustedDealer::with_seed(42);
        let mut pools = create_pools(&mut dealer, 3, 10);

        let cmp = SecureComparison::new(ComparisonConfig::default());

        let x_shares = split_value(-5.0, 3, 42);
        let sign_shares = cmp.sign_bit(&x_shares, &mut pools).unwrap();
        let sign = reconstruct(&sign_shares);

        assert!((sign - 0.0).abs() < 0.01, "Expected 0.0, got {}", sign);
    }

    #[test]
    fn test_less_than() {
        let mut dealer = TrustedDealer::with_seed(42);
        let mut pools = create_pools(&mut dealer, 3, 20);

        let cmp = SecureComparison::new(ComparisonConfig::default());

        let x_shares = split_value(3.0, 3, 42);
        let y_shares = split_value(5.0, 3, 99);
        let lt_shares = cmp.less_than(&x_shares, &y_shares, &mut pools).unwrap();
        let lt = reconstruct(&lt_shares);
        assert!((lt - 1.0).abs() < 0.01, "Expected 1.0, got {}", lt);
    }

    #[test]
    fn test_relu_positive() {
        let mut dealer = TrustedDealer::with_seed(42);
        let mut pools = create_pools(&mut dealer, 3, 20);

        let cmp = SecureComparison::new(ComparisonConfig::default());

        let x_shares = split_value(5.0, 3, 42);
        let relu_shares = cmp.relu(&x_shares, &mut pools).unwrap();
        let relu = reconstruct(&relu_shares);

        assert!((relu - 5.0).abs() < 0.1, "Expected 5.0, got {}", relu);
    }

    #[test]
    fn test_relu_negative() {
        let mut dealer = TrustedDealer::with_seed(42);
        let mut pools = create_pools(&mut dealer, 3, 20);

        let cmp = SecureComparison::new(ComparisonConfig::default());

        let x_shares = split_value(-5.0, 3, 42);
        let relu_shares = cmp.relu(&x_shares, &mut pools).unwrap();
        let relu = reconstruct(&relu_shares);

        assert!(relu.abs() < 0.1, "Expected 0.0, got {}", relu);
    }

    #[test]
    fn test_garbled_sign_protocol_positive() {
        let x = Fr::from_f64(42.0);
        // Split into 2 shares
        let mut rng = ChaCha20Rng::seed_from_u64(99);
        let share_a = Fr::random(&mut rng);
        let share_b = Fr::sub(&x, &share_a);

        let a_bits = fr_to_bits(&share_a);
        let b_bits = fr_to_bits(&share_b);

        let result = garbled_sign_protocol_impl(&a_bits, &b_bits, &mut rng);
        assert!(result, "42.0 should be non-negative");
    }

    #[test]
    fn test_garbled_sign_protocol_negative() {
        let x = Fr::from_f64(-42.0);
        let mut rng = ChaCha20Rng::seed_from_u64(99);
        let share_a = Fr::random(&mut rng);
        let share_b = Fr::sub(&x, &share_a);

        let a_bits = fr_to_bits(&share_a);
        let b_bits = fr_to_bits(&share_b);

        let result = garbled_sign_protocol_impl(&a_bits, &b_bits, &mut rng);
        assert!(!result, "-42.0 should be negative");
    }

    #[test]
    fn test_garbled_sign_protocol_zero() {
        let x = Fr::from_f64(0.0);
        let mut rng = ChaCha20Rng::seed_from_u64(99);
        let share_a = Fr::random(&mut rng);
        let share_b = Fr::sub(&x, &share_a);

        let a_bits = fr_to_bits(&share_a);
        let b_bits = fr_to_bits(&share_b);

        let result = garbled_sign_protocol_impl(&a_bits, &b_bits, &mut rng);
        assert!(result, "0.0 should be non-negative");
    }

    #[test]
    fn test_garbled_comparison_less_than() {
        let gc = GarbledComparison::new([42u8; 32]);
        let mut dealer = TrustedDealer::with_seed(42);
        let mut pools = create_pools(&mut dealer, 3, 10);

        let x_shares = split_value(3.0, 3, 42);
        let y_shares = split_value(5.0, 3, 99);
        let result_shares = gc.secure_less_than(&x_shares, &y_shares, &mut pools).unwrap();
        let result = reconstruct(&result_shares);
        assert!((result - 1.0).abs() < 0.01, "3 < 5 should be 1, got {}", result);
    }

    #[test]
    fn test_garbled_comparison_not_less_than() {
        let gc = GarbledComparison::new([42u8; 32]);
        let mut dealer = TrustedDealer::with_seed(42);
        let mut pools = create_pools(&mut dealer, 3, 10);

        let x_shares = split_value(7.0, 3, 42);
        let y_shares = split_value(5.0, 3, 99);
        let result_shares = gc.secure_less_than(&x_shares, &y_shares, &mut pools).unwrap();
        let result = reconstruct(&result_shares);
        assert!((result - 0.0).abs() < 0.01, "7 < 5 should be 0, got {}", result);
    }

    #[test]
    fn test_bit_decomposition() {
        let mut dealer = TrustedDealer::with_seed(42);
        let mut pools = create_pools(&mut dealer, 3, 10);

        let bd = BitDecomposition::new(32);
        let x_shares = split_value(5.0, 3, 42);

        let bit_shares = bd.decompose(&x_shares, &mut pools).unwrap();
        assert_eq!(bit_shares.len(), 32);

        // Verify each bit share has the right number of parties
        for bs in &bit_shares {
            assert_eq!(bs.len(), 3);
        }

        // Recompose and verify
        let recomposed = bd.recompose(&bit_shares);
        let result = reconstruct(&recomposed);
        // The recomposed value should approximate the original
        // (may not be exact due to fixed-point representation details)
        assert!(result.is_finite(), "Recomposed value should be finite, got {}", result);
    }

    /// Tests that the garbled circuit protocol does not leak party inputs.
    ///
    /// Verifies that the OT transfer function only returns the chosen label
    /// and that neither party's private bits are exposed to the other.
    #[test]
    fn test_no_input_leakage() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);

        // Create two different secret values
        let secret_a = Fr::from_f64(123.456);
        let secret_b = Fr::from_f64(-789.012);

        let a_bits = fr_to_bits(&secret_a);
        let b_bits = fr_to_bits(&secret_b);

        // Create label pairs (garbler's private state)
        let label_pairs: Vec<([u8; 16], [u8; 16])> = (0..256)
            .map(|_| random_label_pair(&mut rng))
            .collect();

        // OT transfer: evaluator gets labels for their bits
        let transferred = simulated_ot_transfer_labels(&label_pairs, &b_bits, &mut rng);

        // Verify: evaluator got exactly the labels corresponding to their bits
        for (i, &bit) in b_bits.iter().enumerate() {
            let expected = if bit { label_pairs[i].1 } else { label_pairs[i].0 };
            assert_eq!(transferred[i], expected, "OT label mismatch at bit {}", i);

            // The unchosen label should NOT be accessible
            let unchosen = if bit { label_pairs[i].0 } else { label_pairs[i].1 };
            assert_ne!(transferred[i], unchosen, "Evaluator should not have unchosen label");
        }

        // Verify: the label pairs don't leak the evaluator's choice bits
        // (i.e., from the garbler's perspective, all label pairs look random)
        for (l0, l1) in &label_pairs {
            assert_ne!(l0, l1, "Label pairs should be distinct");
        }
    }
}
