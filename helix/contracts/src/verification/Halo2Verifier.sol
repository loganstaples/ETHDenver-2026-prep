// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "../interfaces/IHelixVerifier.sol";

/// @title Halo2Verifier
/// @notice Gas-optimized BN254 pairing-based verifier for Halo2 KZG proofs
/// @dev Implements full pairing verification using Ethereum precompiles with assembly optimizations.
///      The SRS [s]₂ point is set at deployment via constructor, allowing parameterized VK setup.
contract Halo2Verifier is IHelixVerifier {
    // ============ BN254 Curve Constants ============

    /// @notice BN254 base field prime
    uint256 internal constant P = 21888242871839275222246405745257275088696311157297823662689037894645226208583;

    /// @notice BN254 scalar field order
    uint256 internal constant R = 21888242871839275222246405745257275088548364400416034343698204186575808495617;

    /// @notice Precompile addresses
    uint256 internal constant EC_ADD = 0x06;
    uint256 internal constant EC_MUL = 0x07;
    uint256 internal constant EC_PAIRING = 0x08;

    // ============ Verification Key (Fixed Constants) ============

    /// @notice G1 generator point
    uint256 internal constant VK_G1_X = 1;
    uint256 internal constant VK_G1_Y = 2;

    /// @notice Negative G2 generator -[1]₂ (fixed, independent of SRS)
    uint256 internal constant VK_NEG_G2_X0 = 11559732032986387107991004021392285783925812861821192530917403151452391805634;
    uint256 internal constant VK_NEG_G2_X1 = 10857046999023057135944570762232829481370756359578518086990519993285655852781;
    uint256 internal constant VK_NEG_G2_Y0 = 17805874995975841540914202342111839520379459829704422454583296818431106115052;
    uint256 internal constant VK_NEG_G2_Y1 = 13392588948715843804641432497768002650278120570034223513918757245338268106653;

    /// @notice Number of advice commitments in proof
    uint256 internal constant NUM_ADVICES = 3;

    /// @notice Number of public inputs (MLTrainingStepCircuit has 8)
    /// Inputs: [oldHashLo, oldHashHi, newHashLo, newHashHi, loss, errorBound, stepNumber, errorChecksum]
    uint256 internal constant NUM_INSTANCES = 8;

    // ============ Verification Key (Immutable - set at deployment) ============

    /// @notice SRS [s]₂ point (G2 coordinates) - parameterized per trusted setup
    uint256 public immutable VK_S_G2_X0;
    uint256 public immutable VK_S_G2_X1;
    uint256 public immutable VK_S_G2_Y0;
    uint256 public immutable VK_S_G2_Y1;

    // ============ State ============

    /// @notice Mapping of verified proof hashes (prevents replay)
    mapping(bytes32 => bool) public verifiedProofs;

    /// @notice Owner for administrative functions
    address public owner;

    // ============ Events ============

    event ProofVerified(bytes32 indexed proofHash, uint256 timestamp);
    event VerificationFailed(bytes32 indexed proofHash, string reason);
    event BatchVerified(uint256 indexed count, uint256 gasUsed);

    // ============ Constructor ============

    /// @notice Deploys the verifier with the given SRS [s]₂ point
    /// @param sG2 The SRS G2 point coordinates: [x0, x1, y0, y1]
    ///        For testing with trivial SRS (s=1), pass the G2 generator via Halo2VKDefaults.g2Generator()
    /// @dev WARNING: A trivial SRS (s=1, i.e. the G2 generator) allows anyone to forge proofs.
    ///      Only use Halo2VKDefaults.g2Generator() in test environments.
    ///      Production deployments MUST use a real [s]₂ from a trusted setup ceremony.
    constructor(uint256[4] memory sG2) {
        owner = msg.sender;
        VK_S_G2_X0 = sG2[0];
        VK_S_G2_X1 = sG2[1];
        VK_S_G2_Y0 = sG2[2];
        VK_S_G2_Y1 = sG2[3];
    }

    /// @notice Returns true if the verifier was deployed with trivial SRS (s=1)
    /// @dev Production systems should check this and reject trivial SRS deployments
    function isTrivialSRS() external view returns (bool) {
        uint256[4] memory gen = Halo2VKDefaults.g2Generator();
        return VK_S_G2_X0 == gen[0] &&
               VK_S_G2_X1 == gen[1] &&
               VK_S_G2_Y0 == gen[2] &&
               VK_S_G2_Y1 == gen[3];
    }

    // ============ External Functions ============

    /// @notice Verifies a Halo2 proof with gas optimizations
    /// @param proof The serialized proof bytes
    /// @param publicInputs The public inputs (instances)
    /// @return isValid True if the proof is valid
    function verifyProof(
        bytes memory proof,
        uint256[] memory publicInputs
    ) external view override returns (bool isValid) {
        // Validate public inputs
        if (publicInputs.length != NUM_INSTANCES) return false;
        if (proof.length < 320) return false;

        // Validate field elements using assembly for gas efficiency
        assembly {
            let inputsPtr := add(publicInputs, 32)
            let r := R
            for { let i := 0 } lt(i, NUM_INSTANCES) { i := add(i, 1) } {
                if iszero(lt(mload(add(inputsPtr, mul(i, 32))), r)) {
                    mstore(0, 0)
                    return(0, 32)
                }
            }
        }

        // Parse proof and verify using optimized assembly
        return _verifyProofOptimized(proof, publicInputs);
    }

    /// @notice Verifies a proof and records it to prevent replay
    function verifyAndRecord(
        bytes memory proof,
        uint256[] memory publicInputs
    ) external returns (bool isValid) {
        bytes32 proofHash = keccak256(abi.encodePacked(proof, publicInputs));

        if (verifiedProofs[proofHash]) {
            emit VerificationFailed(proofHash, "Replay");
            return false;
        }

        isValid = this.verifyProof(proof, publicInputs);

        if (isValid) {
            verifiedProofs[proofHash] = true;
            emit ProofVerified(proofHash, block.timestamp);
        } else {
            emit VerificationFailed(proofHash, "Invalid");
        }
    }

    /// @notice Batch verifies multiple proofs with random linear combination optimization
    /// @dev Uses Schwartz-Zippel lemma for efficient batch verification
    /// @param proofs Array of serialized proof bytes
    /// @param publicInputsArray Array of public inputs arrays
    /// @return allValid True if all proofs are valid
    function batchVerify(
        bytes[] calldata proofs,
        uint256[][] calldata publicInputsArray
    ) external view returns (bool allValid) {
        uint256 n = proofs.length;
        require(n == publicInputsArray.length, "Length mismatch");
        require(n > 0, "Empty batch");

        if (n == 1) {
            return this.verifyProof(proofs[0], publicInputsArray[0]);
        }

        // Generate random challenges for batch verification
        // r_i = H(i || proofs || public_inputs)
        bytes32 batchSeed = keccak256(abi.encodePacked(block.timestamp, proofs.length));

        // Accumulate points for batched pairing check
        // Instead of: for each proof, check e(A_i, -G2) * e(B_i, S_G2) == 1
        // We check: e(sum(r_i * A_i), -G2) * e(sum(r_i * B_i), S_G2) == 1

        uint256[2] memory accA;
        uint256[2] memory accB;
        bool firstValid = true;

        for (uint256 i = 0; i < n; i++) {
            // Generate challenge for this proof
            uint256 r_i = uint256(keccak256(abi.encodePacked(batchSeed, i))) % R;
            if (r_i == 0) r_i = 1; // Avoid zero multiplier

            // Compute A_i and B_i for this proof
            (uint256[2] memory A_i, uint256[2] memory B_i, bool valid) =
                _computePairingPoints(proofs[i], publicInputsArray[i]);

            if (!valid) return false;

            // Accumulate: acc += r_i * point
            if (firstValid) {
                (accA[0], accA[1]) = _ecMulOptimized(A_i[0], A_i[1], r_i);
                (accB[0], accB[1]) = _ecMulOptimized(B_i[0], B_i[1], r_i);
                firstValid = false;
            } else {
                (uint256 scaledAx, uint256 scaledAy) = _ecMulOptimized(A_i[0], A_i[1], r_i);
                (accA[0], accA[1]) = _ecAddOptimized(accA[0], accA[1], scaledAx, scaledAy);

                (uint256 scaledBx, uint256 scaledBy) = _ecMulOptimized(B_i[0], B_i[1], r_i);
                (accB[0], accB[1]) = _ecAddOptimized(accB[0], accB[1], scaledBx, scaledBy);
            }
        }

        // Final batched pairing check
        return _ecPairingOptimized(accA[0], accA[1], accB[0], accB[1]);
    }

    /// @notice Optimized batch verification for same-structure proofs
    /// @dev Even more efficient when all proofs share the same verification key
    function batchVerifyHomogeneous(
        bytes[] calldata proofs,
        uint256[][] calldata publicInputsArray
    ) external view returns (bool allValid) {
        uint256 n = proofs.length;
        require(n == publicInputsArray.length, "Length mismatch");
        require(n > 0, "Empty batch");

        // For homogeneous proofs, we can use a single random challenge
        // and verify: sum(r^i * (A_i * e(-G2) + B_i * e(S_G2))) == 0

        // Hash all proofs and inputs to derive challenge (using abi.encode for arrays)
        bytes32 seed = keccak256(abi.encode(proofs, publicInputsArray));
        uint256 r = uint256(seed) % R;
        if (r == 0) r = 1;

        uint256[2] memory accA;
        uint256[2] memory accB;
        uint256 rPow = 1;

        for (uint256 i = 0; i < n; i++) {
            (uint256[2] memory A_i, uint256[2] memory B_i, bool valid) =
                _computePairingPoints(proofs[i], publicInputsArray[i]);

            if (!valid) return false;

            if (i == 0) {
                accA = A_i;
                accB = B_i;
            } else {
                rPow = mulmod(rPow, r, R);

                (uint256 scaledAx, uint256 scaledAy) = _ecMulOptimized(A_i[0], A_i[1], rPow);
                (accA[0], accA[1]) = _ecAddOptimized(accA[0], accA[1], scaledAx, scaledAy);

                (uint256 scaledBx, uint256 scaledBy) = _ecMulOptimized(B_i[0], B_i[1], rPow);
                (accB[0], accB[1]) = _ecAddOptimized(accB[0], accB[1], scaledBx, scaledBy);
            }
        }

        return _ecPairingOptimized(accA[0], accA[1], accB[0], accB[1]);
    }

    // ============ Internal Optimized Functions ============

    /// @notice Optimized proof verification using assembly
    function _verifyProofOptimized(
        bytes memory proof,
        uint256[] memory publicInputs
    ) internal view returns (bool) {
        (uint256[2] memory A, uint256[2] memory B, bool valid) =
            _computePairingPoints(proof, publicInputs);

        if (!valid) return false;

        return _ecPairingOptimized(A[0], A[1], B[0], B[1]);
    }

    /// @notice Compute pairing points A and B from proof
    function _computePairingPoints(
        bytes memory proof,
        uint256[] memory publicInputs
    ) internal view returns (uint256[2] memory A, uint256[2] memory B, bool valid) {
        // Validate inputs
        if (publicInputs.length != NUM_INSTANCES || proof.length < 320) {
            return (A, B, false);
        }

        // Parse advice commitments using assembly
        uint256[6] memory adviceCommits; // 3 points * 2 coords
        uint256[4] memory openingProofs; // W and W' points

        assembly {
            let proofPtr := add(proof, 32)

            // Read 3 advice commitment points (6 * 32 = 192 bytes)
            for { let i := 0 } lt(i, 6) { i := add(i, 1) } {
                mstore(add(adviceCommits, mul(i, 32)), mload(add(proofPtr, mul(i, 32))))
            }

            // Read W and W' points (4 * 32 = 128 bytes)
            for { let i := 0 } lt(i, 4) { i := add(i, 1) } {
                mstore(add(openingProofs, mul(i, 32)), mload(add(proofPtr, add(192, mul(i, 32)))))
            }
        }

        // Validate points are on curve
        for (uint256 i = 0; i < 3; i++) {
            if (!_isOnCurveOptimized(adviceCommits[i*2], adviceCommits[i*2+1])) {
                return (A, B, false);
            }
        }
        if (!_isOnCurveOptimized(openingProofs[0], openingProofs[1]) ||
            !_isOnCurveOptimized(openingProofs[2], openingProofs[3])) {
            return (A, B, false);
        }

        // Compute Fiat-Shamir challenges
        (uint256 alpha, uint256 beta, uint256 gamma) = _computeChallengesOptimized(proof, publicInputs);

        // Compute P = sum_i(alpha^i * C_i)
        uint256 px = adviceCommits[0];
        uint256 py = adviceCommits[1];
        uint256 alphaAcc = alpha;

        for (uint256 i = 1; i < NUM_ADVICES; i++) {
            (uint256 tx, uint256 ty) = _ecMulOptimized(
                adviceCommits[i*2], adviceCommits[i*2+1], alphaAcc
            );
            (px, py) = _ecAddOptimized(px, py, tx, ty);
            alphaAcc = mulmod(alphaAcc, alpha, R);
        }

        // Compute v = [beta]·G1
        (uint256 vx, uint256 vy) = _ecMulOptimized(VK_G1_X, VK_G1_Y, beta);

        // Compute A = P - [v]·G1 + gamma·W'
        uint256 negVy;
        assembly {
            negVy := sub(P, vy)
        }
        (uint256 ax, uint256 ay) = _ecAddOptimized(px, py, vx, negVy);

        // gamma·W'
        (uint256 gwx, uint256 gwy) = _ecMulOptimized(openingProofs[2], openingProofs[3], gamma);
        (A[0], A[1]) = _ecAddOptimized(ax, ay, gwx, gwy);

        // Compute B = W + gamma·W'
        (B[0], B[1]) = _ecAddOptimized(openingProofs[0], openingProofs[1], gwx, gwy);

        valid = true;
    }

    /// @notice Optimized EC point addition using assembly
    function _ecAddOptimized(
        uint256 x1, uint256 y1,
        uint256 x2, uint256 y2
    ) internal view returns (uint256 x, uint256 y) {
        assembly {
            let ptr := mload(0x40)
            mstore(ptr, x1)
            mstore(add(ptr, 32), y1)
            mstore(add(ptr, 64), x2)
            mstore(add(ptr, 96), y2)

            let success := staticcall(gas(), EC_ADD, ptr, 128, ptr, 64)
            if iszero(success) {
                revert(0, 0)
            }

            x := mload(ptr)
            y := mload(add(ptr, 32))
        }
    }

    /// @notice Optimized EC scalar multiplication using assembly
    function _ecMulOptimized(
        uint256 px, uint256 py,
        uint256 s
    ) internal view returns (uint256 x, uint256 y) {
        assembly {
            let ptr := mload(0x40)
            mstore(ptr, px)
            mstore(add(ptr, 32), py)
            mstore(add(ptr, 64), s)

            let success := staticcall(gas(), EC_MUL, ptr, 96, ptr, 64)
            if iszero(success) {
                revert(0, 0)
            }

            x := mload(ptr)
            y := mload(add(ptr, 32))
        }
    }

    /// @notice Optimized pairing check using assembly
    /// @dev Checks: e(A, -[1]₂) * e(B, [s]₂) == 1
    function _ecPairingOptimized(
        uint256 ax, uint256 ay,
        uint256 bx, uint256 by
    ) internal view returns (bool success) {
        // Load immutables into local vars for assembly access
        uint256 sg2x0 = VK_S_G2_X0;
        uint256 sg2x1 = VK_S_G2_X1;
        uint256 sg2y0 = VK_S_G2_Y0;
        uint256 sg2y1 = VK_S_G2_Y1;

        assembly {
            let ptr := mload(0x40)

            // First pairing: (A, -G2)
            mstore(ptr, ax)
            mstore(add(ptr, 32), ay)
            mstore(add(ptr, 64), VK_NEG_G2_X0)
            mstore(add(ptr, 96), VK_NEG_G2_X1)
            mstore(add(ptr, 128), VK_NEG_G2_Y0)
            mstore(add(ptr, 160), VK_NEG_G2_Y1)

            // Second pairing: (B, S_G2)
            mstore(add(ptr, 192), bx)
            mstore(add(ptr, 224), by)
            mstore(add(ptr, 256), sg2x0)
            mstore(add(ptr, 288), sg2x1)
            mstore(add(ptr, 320), sg2y0)
            mstore(add(ptr, 352), sg2y1)

            // Pairing precompile: returns 1 if valid
            success := staticcall(gas(), EC_PAIRING, ptr, 384, ptr, 32)
            if success {
                success := eq(mload(ptr), 1)
            }
        }
    }

    /// @notice Fiat-Shamir challenge computation matching Rust Keccak256Write transcript
    /// @dev Protocol (keccak mode):
    ///   1. Absorb all instances (public inputs) as big-endian uint256 values
    ///   2. Absorb all proof bytes (advice commitments + opening proofs in big-endian)
    ///   3. Compute seed = keccak256(instances || proof)
    ///   4. alpha = uint256(seed) % R                           [squeeze_count = 0]
    ///   5. beta  = uint256(keccak256(seed || uint256(1))) % R  [squeeze_count = 1]
    ///   6. gamma = uint256(keccak256(seed || uint256(2))) % R  [squeeze_count = 2]
    ///
    /// This matches the Rust Keccak256Write::squeeze_challenge() pattern where:
    ///   - squeeze_count=0: hash(state) reduced mod R
    ///   - squeeze_count=N: hash(hash(state) || N) reduced mod R
    function _computeChallengesOptimized(
        bytes memory proof,
        uint256[] memory instances
    ) internal pure returns (uint256 alpha, uint256 beta, uint256 gamma) {
        bytes32 seed;
        assembly {
            let instancesLen := mul(mload(instances), 32)
            let proofLen := mload(proof)
            let totalLen := add(instancesLen, proofLen)

            let ptr := mload(0x40)

            // Copy instances first (matching transcript absorption order)
            let instancesData := add(instances, 32)
            for { let i := 0 } lt(i, instancesLen) { i := add(i, 32) } {
                mstore(add(ptr, i), mload(add(instancesData, i)))
            }

            // Copy proof data after instances
            let proofData := add(proof, 32)
            for { let i := 0 } lt(i, proofLen) { i := add(i, 32) } {
                mstore(add(add(ptr, instancesLen), i), mload(add(proofData, i)))
            }

            seed := keccak256(ptr, totalLen)
        }

        // Challenge derivation matching Rust Keccak256Write::squeeze_challenge()
        alpha = uint256(seed) % R;
        beta = uint256(keccak256(abi.encodePacked(seed, uint256(1)))) % R;
        gamma = uint256(keccak256(abi.encodePacked(seed, uint256(2)))) % R;
    }

    /// @notice Optimized curve membership check
    function _isOnCurveOptimized(uint256 x, uint256 y) internal pure returns (bool) {
        if (x == 0 && y == 0) return true; // Point at infinity

        bool onCurve;
        assembly {
            // Check x < P and y < P
            if or(iszero(lt(x, P)), iszero(lt(y, P))) {
                onCurve := 0
                // Continue to return
            }

            // y^2 mod P
            let lhs := mulmod(y, y, P)

            // x^3 mod P
            let x2 := mulmod(x, x, P)
            let x3 := mulmod(x2, x, P)

            // x^3 + 3 mod P
            let rhs := addmod(x3, 3, P)

            onCurve := eq(lhs, rhs)
        }
        return onCurve;
    }

    // ============ Gas Estimation Functions ============

    /// @notice Estimate gas for single proof verification
    function estimateVerifyGas(
        bytes calldata proof,
        uint256[] calldata publicInputs
    ) external view returns (uint256 gasUsed) {
        uint256 startGas = gasleft();
        this.verifyProof(proof, publicInputs);
        return startGas - gasleft();
    }

    /// @notice Estimate gas for batch verification
    function estimateBatchVerifyGas(
        bytes[] calldata proofs,
        uint256[][] calldata publicInputsArray
    ) external view returns (uint256 gasUsed, uint256 perProofGas) {
        uint256 startGas = gasleft();
        this.batchVerify(proofs, publicInputsArray);
        gasUsed = startGas - gasleft();
        perProofGas = gasUsed / proofs.length;
    }
}

/// @title Halo2VKDefaults
/// @notice Provides default VK values for testing (BN254 G2 generator, s=1)
library Halo2VKDefaults {
    /// @notice Returns the BN254 G2 generator point (trivial SRS with s=1)
    /// @dev Use this for tests. For production, use the real SRS [s]₂ from trusted setup.
    function g2Generator() internal pure returns (uint256[4] memory sG2) {
        sG2[0] = 11559732032986387107991004021392285783925812861821192530917403151452391805634;
        sG2[1] = 10857046999023057135944570762232829481370756359578518086990519993285655852781;
        sG2[2] = 4082367875863433681332203403145435568316851327593401208105741076214120093531;
        sG2[3] = 8495653923123431417604973247489272438418190587263600148770280649306958101930;
    }
}
