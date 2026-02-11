# HELIX-MPC Security Guarantees

This document describes the security properties, threat model, and guarantees provided by the HELIX MPC implementation.

## Overview

HELIX-MPC enables privacy-preserving machine learning training where model weights remain secret-shared among parties. The implementation provides the following high-level guarantees:

1. **Weight Privacy**: No single party ever sees the complete model weights
2. **Computation Integrity**: All computations are verifiable; cheating is detectable
3. **Replay Protection**: Old messages cannot be replayed to attack the protocol
4. **Forward Secrecy**: Compromise of long-term keys doesn't reveal past sessions

## Threat Model

### Adversary Capabilities

The implementation is designed to resist adversaries who can:

- **Semi-Honest (Passive)**: Follow the protocol but try to learn extra information
- **Malicious (Active)**: Deviate from the protocol arbitrarily (with detection)
- **Network Control**: Observe, delay, reorder, or drop messages
- **Corruption**: Compromise up to t-1 out of n parties (for t-of-n security)

### Assumptions

Security relies on the following assumptions:

1. **Discrete Log Hardness**: For key exchange and Pedersen commitments
2. **Hash Function Security**: SHA-256 is collision-resistant and acts as a random oracle
3. **AES-GCM Security**: Authenticated encryption is secure
4. **Honest Majority**: At least (n+1)/2 parties are honest (for some protocols)
5. **Secure Channels**: TLS provides confidentiality and integrity (when enabled)

## Security Properties by Component

### 1. Secret Sharing (`sharing` module)

**Additive Sharing**:
- Information-theoretic privacy: Any subset of shares reveals nothing about the secret
- Requires all n shares to reconstruct
- Linear homomorphism preserved

**Security Level**: Information-theoretic (unconditional)

### 2. Beaver Triple Generation (`beaver` module)

**Trusted Dealer**:
- Suitable for demos only
- Single point of trust
- Dealer knows all secrets

**OT-Based Generation** (⚠️ DEPRECATED):
- ~~No trusted party required~~
- ~~Based on Oblivious Transfer security~~
- The OT implementation has known security limitations and is gated behind `#[cfg(feature = "experimental-ot")]`
- Use `DistributedTripleGen::simulate_distributed_batch()` for non-trusted-dealer generation

**Distributed Generation**:
- Simulated multi-party OT-style generation
- Each party contributes local randomness
- Uses `mpc_scale` for field-exact multiplication (matches TrustedDealer)

**Security Level**: Computational (DDH-based for full OT); Trusted-dealer or simulated for current deployment

### 3. Secure Arithmetic (`protocols/arithmetic`)

**Addition/Subtraction**: Local operations, no leakage

**Multiplication (Beaver)**:
- Reveals: d = x - a, e = y - b (random-looking values)
- Does not reveal: x, y, or the product xy
- Requires correct Beaver triple

**Security Level**: Information-theoretic given correct triples

### 4. Secure Comparison (`protocols/comparison`)

**⚠️ Current Implementation (Reconstruct-Compare-Reshare)**:
- All three operations (`sign_bit`, `secure_less_than`, `decompose`) use a unified approach
- The secret value is reconstructed among all parties, the operation is applied in cleartext, and the result is re-shared
- **This reveals the secret value to all parties** — it is NOT fully secure comparison
- The garbled circuit infrastructure exists in the codebase but is gated as dead code pending a production-quality OT implementation

**Polynomial Approximation**:
- Reveals: Nothing (approximate computation on shares)
- Trade-off: Lower accuracy for better efficiency

**Security Level**: ⚠️ **Reveals values** in current implementation. Garbled circuit path requires production OT.

### 5. MAC Authentication (`security/mac`)

**SPDZ-Style MACs**:
- Information-theoretic security for share authentication
- Global MAC key α is secret-shared
- Each value x has MAC α·x
- Detects any modification of shares

**Message Authentication**:
- HMAC-SHA256 for message integrity
- Sequence numbers prevent replay
- Pairwise keys prevent cross-party forgery

**Security Level**: Information-theoretic (for value MACs), Computational (for HMAC)

### 6. Commitment Schemes (`security/commitment`)

**Hash Commitments**:
- Computationally hiding: H(value || blinding)
- Computationally binding: Can't find collisions
- Used for: Share commitments, gradient verification

**Pedersen Commitments**:
- Information-theoretically hiding
- Computationally binding (DL assumption)
- Additively homomorphic: C(a) · C(b) = C(a+b)
- Used for: Gradient sum verification

**Merkle Trees**:
- O(log n) proof size
- Batch verification
- Used for: Large gradient vectors

**Security Level**: Computational binding, IT hiding (Pedersen) or Computational hiding (Hash)

### 7. Byzantine Fault Detection (`security/byzantine`)

**Detected Attacks**:
- Invalid MAC on messages
- Commitment-value mismatch
- Replay attacks (sequence numbers)
- Protocol violations (wrong phase)
- Gradient poisoning (bound checks)
- Incorrect Beaver triples
- Inconsistent values across operations

**Response**:
- Fault evidence generated
- Party excluded after threshold
- Slashing accusation for on-chain penalty

**Security Level**: Computational (signature verification)

### 8. Session Establishment (`session/establishment`)

**Key Exchange**:
- X25519 Diffie-Hellman
- Ephemeral keys for forward secrecy
- Pairwise shared secrets derived

**Authentication**:
- Ed25519 signatures
- Long-term public keys verified
- Timestamp freshness check

**Session Key Derivation**:
- All parties contribute randomness
- Session ID = H(all commitments)
- Session key = H(session_id || all randoms)

**Security Level**: Computational (DDH, signature security)

### 9. Network Channel (`session/network`)

**TLS Protection**:
- Confidentiality and integrity
- Server authentication
- Optional client authentication

**Message Framing**:
- Length-prefixed messages with maximum size enforcement (64 MB default, prevents OOM attacks)
- HMAC-SHA256 on each message with sequence number binding (prevents replay)
- Per-peer sequence tracking rejects out-of-order or replayed messages

**Security Level**: Computational (TLS security)

## What is Revealed

For transparency, here's what information may leak during protocol execution:

| Operation | Revealed | Not Revealed |
|-----------|----------|--------------|
| Sharing | Share count, shape | Values, sum |
| Beaver multiply | d=x-a, e=y-b | x, y, a, b, xy |
| Comparison/sign (current impl) | Reconstructed values | Nothing (⚠️ values leak) |
| Activation (reconstruct-reshare) | Activation values | Weights |
| Matmul | Matrix dimensions | Matrix contents |
| Gradient aggregation | Sum of all gradients | Individual gradients |
| Session establishment | Party IDs, timing | Session key |

**Important**: Activation values are revealed in the reconstruct-reshare activation approach. This is a standard trade-off in MPC-ML that leaks intermediate activations but not model weights. Use polynomial approximations for higher privacy at a computational cost.

## Known Limitations

1. **Trusted Dealer Mode**: The default demo mode uses a trusted dealer for Beaver triples. For production, use distributed generation or implement full OT.

2. ~~**f64 Arithmetic**: The demo uses f64 for convenience.~~ **FIXED (Round 6)**: SPDZ MAC system and Shamir secret sharing now use BN254 Fr field arithmetic internally. The Shamir `SecretSharingScheme` trait retains f64 interface for compatibility, with Fr conversion at sharing boundaries.

3. **Activation Leakage**: Reconstruct-reshare activations reveal intermediate values. For full privacy, use polynomial approximations (with accuracy trade-off).

4. ~~**Timing Side Channels**: Operations are not constant-time.~~ **PARTIALLY FIXED (Round 6)**: Commitment verification and fingerprint checks now use constant-time hash comparison (`ct_eq_hash`). Pedersen commitments use halo2curves EC operations which are inherently constant-time. Some non-critical operations may still have variable timing.

5. **Network Metadata**: Message sizes and timing patterns may leak information about computation structure.

6. **OT Implementation Deprecated**: The existing OT (`beaver/ot.rs`) has known security limitations and is gated behind `experimental-ot` feature flag. Do not use in production without a full audit.

7. **Comparison Reveals Values**: `sign_bit`, `secure_less_than`, and `decompose` currently reconstruct the secret value to all parties. This is a standard demo simplification — for production, a garbled circuit or fully secret-shared comparison is needed.

8. **Aggregation is Plaintext**: Gradient aggregation sums are visible to all parties. Individual gradients remain private but the aggregate is revealed.

## Security Recommendations

### For Demo/Testing

```rust
// Acceptable for demos:
let dealer = TrustedDealer::new();
let triples = dealer.generate_scalar_triples(100, 3);
```

### For Production

```rust
// Use distributed triple generation (or implement full OT):
let triples = DistributedTripleGen::simulate_distributed_batch(100, 3, seed);

// Use field arithmetic with mpc_scale for Beaver protocol:
let c = a.mpc_scale(&b); // Field-exact fixed-point multiplication

// Enable all verification:
let detector = ByzantineDetector::new(timeout, max_faults);
let mac_keys = MACKey::generate_shares(num_parties, seed);

// Use secure session establishment:
let sessions = simulate_session_establishment(&parties, duration)?;

// Network channels enforce message size limits and HMAC replay protection
```

## Formal Security

The security of HELIX-MPC can be proven under standard cryptographic assumptions:

**Theorem (Informal)**: In the semi-honest model with an honest majority, HELIX-MPC reveals only the function output to all parties.

**Theorem (Informal)**: In the malicious model with SPDZ-style MACs, any deviation from the protocol is detected with overwhelming probability.

For formal proofs, see:
- Damgård et al., "Multiparty Computation from Somewhat Homomorphic Encryption"
- Keller et al., "MASCOT: Faster Malicious Arithmetic Secure Computation with Oblivious Transfer"

## Audit Status

This implementation has NOT been formally audited. It is a prototype for demonstration purposes. Before production use:

- [ ] Independent security audit
- [ ] Formal verification of critical paths
- [ ] Penetration testing
- [ ] Side-channel analysis
- [ ] Performance hardening

## Reporting Vulnerabilities

If you discover a security vulnerability, please report it privately. Do not open a public issue.

Contact: security@helix.network (placeholder)

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2025-01 | Initial implementation |
| 0.2.0 | 2026-02 | Production readiness: standardized mpc_scale multiplication, fixed OOM/replay in network, deprecated OT, unified comparison to reconstruct-compare-reshare, removed private shares from BeaverWitness, added gradient updates to circuit bridge |
