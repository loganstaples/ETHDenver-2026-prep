// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/// @title PoseidonHasher
/// @notice Poseidon hash function over BN254 Fr matching helix-circuits Rust implementation.
/// @dev Parameters: width=3, rate=2, 8 full rounds (4+4), 57 partial rounds, x^5 S-box.
///      MDS matrix: [[2,1,1],[1,2,1],[1,1,2]].
///      Round constants derived via SHA256("HELIX_POSEIDON_RC_V1" || round_le64 || i_le64)
///      with top 3 bits cleared (repr[31] &= 0x1F) to fit BN254 scalar field.
library PoseidonHasher {
    /// BN254 scalar field modulus
    uint256 internal constant R = 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001;

    uint256 internal constant WIDTH = 3;
    uint256 internal constant FULL_ROUNDS = 8;
    uint256 internal constant PARTIAL_ROUNDS = 57;
    uint256 internal constant TOTAL_ROUNDS = 65; // 8 + 57

    /// @notice Computes a single round constant.
    /// @dev Matches Rust: SHA256("HELIX_POSEIDON_RC_V1" || round.to_le_bytes() || i.to_le_bytes())
    ///      then repr[31] &= 0x1F, interpreted as little-endian Fr.
    function _roundConstant(uint256 round, uint256 i) internal pure returns (uint256) {
        // Build the preimage: "HELIX_POSEIDON_RC_V1" (20 bytes) || round_le64 (8 bytes) || i_le64 (8 bytes)
        // Total: 36 bytes
        bytes memory preimage = new bytes(36);

        // Copy domain separator
        bytes20 domain = bytes20("HELIX_POSEIDON_RC_V");
        assembly {
            mstore(add(preimage, 32), domain)
        }
        // The 20th byte is '1' (0x31) - domain is "HELIX_POSEIDON_RC_V1"
        preimage[19] = 0x31;

        // Write round as LE u64 at offset 20
        for (uint256 b = 0; b < 8; b++) {
            preimage[20 + b] = bytes1(uint8(round >> (b * 8)));
        }

        // Write i as LE u64 at offset 28
        for (uint256 b = 0; b < 8; b++) {
            preimage[28 + b] = bytes1(uint8(i >> (b * 8)));
        }

        bytes32 hash = sha256(preimage);

        // Convert SHA256 bytes to uint256 matching Rust's little-endian Fr interpretation.
        // Rust does: repr.copy_from_slice(&hash); repr[31] &= 0x1F; Fr::from_repr(repr)
        // where Fr::from_repr interprets repr as little-endian (repr[0] = LSB).
        // In Solidity bytes32, hash[0] is the first SHA256 output byte (same as Rust hash[0]).
        // So we place hash[b] at bit position b*8 to match Rust's LE interpretation.
        uint256 result = 0;
        for (uint256 b = 0; b < 32; b++) {
            uint8 byteVal = uint8(hash[b]);
            if (b == 31) {
                byteVal &= 0x1F; // Clear top 3 bits (same as Rust repr[31] &= 0x1F)
            }
            result |= uint256(byteVal) << (b * 8);
        }

        return result % R;
    }

    /// @notice S-box: x^5 mod R
    function _sbox(uint256 x) internal pure returns (uint256) {
        uint256 x2 = mulmod(x, x, R);
        uint256 x4 = mulmod(x2, x2, R);
        return mulmod(x4, x, R);
    }

    /// @notice MDS matrix multiplication: M = [[2,1,1],[1,2,1],[1,1,2]]
    /// @dev out[i] = state[i] + sum(state)
    function _mds(
        uint256 s0, uint256 s1, uint256 s2
    ) internal pure returns (uint256, uint256, uint256) {
        uint256 sum = addmod(addmod(s0, s1, R), s2, R);
        return (
            addmod(s0, sum, R),
            addmod(s1, sum, R),
            addmod(s2, sum, R)
        );
    }

    /// @notice Full Poseidon permutation.
    function _permute(
        uint256 s0, uint256 s1, uint256 s2
    ) internal pure returns (uint256, uint256, uint256) {
        uint256 halfFull = FULL_ROUNDS / 2; // 4
        uint256 round = 0;

        // First half of full rounds (4 rounds)
        for (uint256 r = 0; r < halfFull; r++) {
            // AddRoundConstants
            s0 = addmod(s0, _roundConstant(round, 0), R);
            s1 = addmod(s1, _roundConstant(round, 1), R);
            s2 = addmod(s2, _roundConstant(round, 2), R);
            // Full S-box
            s0 = _sbox(s0);
            s1 = _sbox(s1);
            s2 = _sbox(s2);
            // MDS
            (s0, s1, s2) = _mds(s0, s1, s2);
            round++;
        }

        // Partial rounds (57 rounds)
        for (uint256 r = 0; r < PARTIAL_ROUNDS; r++) {
            // AddRoundConstants
            s0 = addmod(s0, _roundConstant(round, 0), R);
            s1 = addmod(s1, _roundConstant(round, 1), R);
            s2 = addmod(s2, _roundConstant(round, 2), R);
            // Partial S-box (only s0)
            s0 = _sbox(s0);
            // MDS
            (s0, s1, s2) = _mds(s0, s1, s2);
            round++;
        }

        // Second half of full rounds (4 rounds)
        for (uint256 r = 0; r < halfFull; r++) {
            // AddRoundConstants
            s0 = addmod(s0, _roundConstant(round, 0), R);
            s1 = addmod(s1, _roundConstant(round, 1), R);
            s2 = addmod(s2, _roundConstant(round, 2), R);
            // Full S-box
            s0 = _sbox(s0);
            s1 = _sbox(s1);
            s2 = _sbox(s2);
            // MDS
            (s0, s1, s2) = _mds(s0, s1, s2);
            round++;
        }

        return (s0, s1, s2);
    }

    /// @notice Hash two field elements. State = [left, right, 0], output = state[0].
    /// @dev Matches Rust `poseidon_hash_two(left, right)`.
    function hashTwo(uint256 left, uint256 right) internal pure returns (uint256) {
        (uint256 s0,,) = _permute(left, right, 0);
        return s0;
    }

    /// @notice Compute the error checksum matching the circuit's Poseidon computation.
    /// @dev checksum = Poseidon(Poseidon(errorBound, stepNumber), Poseidon(modelId, errorBudget))
    ///      Matches `compute_error_checksum()` in training_step_v2.rs.
    function computeErrorChecksum(
        uint256 errorBound,
        uint256 stepNumber,
        uint256 modelId,
        uint256 errorBudget
    ) internal pure returns (uint256) {
        uint256 h1 = hashTwo(errorBound, stepNumber);
        uint256 h2 = hashTwo(modelId, errorBudget);
        return hashTwo(h1, h2);
    }
}
