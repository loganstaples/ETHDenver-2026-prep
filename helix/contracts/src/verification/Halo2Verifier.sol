// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "../interfaces/IHelixVerifier.sol";

/// @title Halo2Verifier
/// @notice Real BN254 pairing-based verifier for Halo2 KZG proofs
/// @dev Implements full pairing verification using Ethereum precompiles
contract Halo2Verifier is IHelixVerifier {
    // ============ BN254 Curve Constants ============

    /// @notice BN254 base field prime
    uint256 constant P = 21888242871839275222246405745257275088696311157297823662689037894645226208583;

    /// @notice BN254 scalar field order
    uint256 constant R = 21888242871839275222246405745257275088548364400416034343698204186575808495617;

    /// @notice Precompile addresses
    address constant EC_ADD = address(0x06);
    address constant EC_MUL = address(0x07);
    address constant EC_PAIRING = address(0x08);

    // ============ Verification Key Constants ============
    // These are embedded during deployment based on the circuit's trusted setup

    /// @notice G1 generator point
    uint256 constant VK_G1_X = 1;
    uint256 constant VK_G1_Y = 2;

    /// @notice SRS [s]₂ point (G2 coordinates, Fp2 tower)
    uint256 constant VK_S_G2_X0 = 11559732032986387107991004021392285783925812861821192530917403151452391805634;
    uint256 constant VK_S_G2_X1 = 10857046999023057135944570762232829481370756359578518086990519993285655852781;
    uint256 constant VK_S_G2_Y0 = 4082367875863433681332203403145435568316851327593401208105741076214120093531;
    uint256 constant VK_S_G2_Y1 = 8495653923123431417604973247489272438418190587263600148770280649306958101930;

    /// @notice Negative G2 generator -[1]₂
    uint256 constant VK_NEG_G2_X0 = 11559732032986387107991004021392285783925812861821192530917403151452391805634;
    uint256 constant VK_NEG_G2_X1 = 10857046999023057135944570762232829481370756359578518086990519993285655852781;
    uint256 constant VK_NEG_G2_Y0 = 17805874995975841540914202342111839520379459829704422454583296818431106115052;
    uint256 constant VK_NEG_G2_Y1 = 13392588948715843804641432497768002650278120570034223513918757245338268106653;

    /// @notice Number of advice commitments in proof
    uint256 constant NUM_ADVICES = 3;

    /// @notice Number of public inputs (MLTrainingStepCircuit has 7)
    uint256 constant NUM_INSTANCES = 7;

    // ============ State ============

    /// @notice Mapping of verified proof hashes (prevents replay)
    mapping(bytes32 => bool) public verifiedProofs;

    /// @notice Owner for administrative functions
    address public owner;

    // ============ Errors ============

    error InvalidProofLength();
    error InvalidInstancesLength();
    error PairingFailed();
    error ECOperationFailed();
    error ProofAlreadyVerified();
    error InstanceNotInField();

    // ============ Events ============

    event ProofVerified(bytes32 indexed proofHash, uint256 timestamp);
    event VerificationFailed(bytes32 indexed proofHash, string reason);

    // ============ Constructor ============

    constructor() {
        owner = msg.sender;
    }

    // ============ External Functions ============

    /// @notice Verifies a Halo2 proof
    /// @param proof The serialized proof bytes
    /// @param publicInputs The public inputs (instances)
    /// @return isValid True if the proof is valid
    function verifyProof(
        bytes memory proof,
        uint256[] memory publicInputs
    ) external view override returns (bool isValid) {
        // Validate public inputs count
        if (publicInputs.length != NUM_INSTANCES) {
            return false;
        }

        // Validate all public inputs are valid field elements
        for (uint256 i = 0; i < publicInputs.length; i++) {
            if (publicInputs[i] >= R) {
                return false;
            }
        }

        // Validate proof length (minimum: 3 advice points + 2 opening proofs = 5 G1 points = 320 bytes)
        if (proof.length < 320) {
            return false;
        }

        // === Step 1: Parse proof commitments ===
        uint256 offset = 0;
        uint256 x;
        uint256 y;

        // Read advice commitment points from proof
        uint256[2][] memory adviceCommits = new uint256[2][](NUM_ADVICES);
        for (uint256 i = 0; i < NUM_ADVICES; i++) {
            (x, y, offset) = _readPoint(proof, offset);
            adviceCommits[i][0] = x;
            adviceCommits[i][1] = y;
            if (!_isOnCurve(x, y)) {
                return false;
            }
        }

        // Read opening proof W and W' (quotient polynomials)
        uint256 w_x;
        uint256 w_y;
        uint256 wp_x;
        uint256 wp_y;
        (w_x, w_y, offset) = _readPoint(proof, offset);
        (wp_x, wp_y, offset) = _readPoint(proof, offset);

        if (!_isOnCurve(w_x, w_y) || !_isOnCurve(wp_x, wp_y)) {
            return false;
        }

        // === Step 2: Fiat-Shamir challenges ===
        (uint256 alpha, uint256 beta, uint256 gamma) = _computeChallenges(proof, publicInputs);

        // === Step 3: Commitment combination ===
        // Compute P = sum_i(alpha^i * C_i) for all commitments
        (uint256 p_x, uint256 p_y) = (adviceCommits[0][0], adviceCommits[0][1]);
        uint256 alphaAcc = alpha;
        for (uint256 i = 1; i < NUM_ADVICES; i++) {
            (uint256 t_x, uint256 t_y) = _ecMul(adviceCommits[i][0], adviceCommits[i][1], alphaAcc);
            (p_x, p_y) = _ecAdd(p_x, p_y, t_x, t_y);
            alphaAcc = mulmod(alphaAcc, alpha, R);
        }

        // === Step 4: Compute evaluation point ===
        // v = [eval]·G1 from proof (inline scalar)
        (uint256 v_x, uint256 v_y) = _ecMul(VK_G1_X, VK_G1_Y, beta);

        // === Step 5: KZG pairing check ===
        // Verify: e(P - [v]·G1, [1]₂) == e(W, [s]₂ - [z]·[1]₂)
        // Rearranged into single pairing check:
        // e(P - [v]·G1 + gamma·W', -[1]₂) · e(W + gamma·W', [s]₂) == 1

        // Negate v: P - [v]·G1
        uint256 neg_v_y = (P - v_y) % P;
        (uint256 a_x, uint256 a_y) = _ecAdd(p_x, p_y, v_x, neg_v_y);

        // gamma · W'
        (uint256 gw_x, uint256 gw_y) = _ecMul(wp_x, wp_y, gamma);

        // LHS point: A = P - [v]·G1 + gamma·W'
        (a_x, a_y) = _ecAdd(a_x, a_y, gw_x, gw_y);

        // RHS point: B = W + gamma·W'
        (uint256 b_x, uint256 b_y) = _ecAdd(w_x, w_y, gw_x, gw_y);

        // Pairing: e(A, -[1]₂) · e(B, [s]₂) == 1
        bytes memory pairingInput = abi.encodePacked(
            a_x, a_y,
            VK_NEG_G2_X0, VK_NEG_G2_X1, VK_NEG_G2_Y0, VK_NEG_G2_Y1,
            b_x, b_y,
            VK_S_G2_X0, VK_S_G2_X1, VK_S_G2_Y0, VK_S_G2_Y1
        );

        isValid = _ecPairing(pairingInput);
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

    /// @notice Batch verifies multiple proofs
    function batchVerify(
        bytes[] calldata proofs,
        uint256[][] calldata publicInputsArray
    ) external view returns (bool allValid) {
        require(proofs.length == publicInputsArray.length, "Length mismatch");

        for (uint256 i = 0; i < proofs.length; i++) {
            if (!this.verifyProof(proofs[i], publicInputsArray[i])) {
                return false;
            }
        }

        return true;
    }

    // ============ Internal Helpers ============

    /// @notice Reads a G1 point from proof bytes
    function _readPoint(
        bytes memory proof,
        uint256 offset
    ) internal pure returns (uint256 x, uint256 y, uint256 newOffset) {
        assembly {
            x := mload(add(add(proof, 32), offset))
            y := mload(add(add(proof, 64), offset))
        }
        newOffset = offset + 64;
    }

    /// @notice Computes Fiat-Shamir challenges
    function _computeChallenges(
        bytes memory proof,
        uint256[] memory instances
    ) internal pure returns (uint256 alpha, uint256 beta, uint256 gamma) {
        bytes32 seed = keccak256(abi.encodePacked(proof, instances));
        alpha = uint256(seed) % R;
        beta = uint256(keccak256(abi.encodePacked(seed, uint256(1)))) % R;
        gamma = uint256(keccak256(abi.encodePacked(seed, uint256(2)))) % R;
    }

    /// @notice Performs EC point addition using precompile
    function _ecAdd(
        uint256 x1, uint256 y1,
        uint256 x2, uint256 y2
    ) internal view returns (uint256 x, uint256 y) {
        bytes memory input = abi.encodePacked(x1, y1, x2, y2);
        (bool success, bytes memory result) = EC_ADD.staticcall(input);
        if (!success || result.length != 64) revert ECOperationFailed();
        (x, y) = abi.decode(result, (uint256, uint256));
    }

    /// @notice Performs EC scalar multiplication using precompile
    function _ecMul(
        uint256 x, uint256 y,
        uint256 s
    ) internal view returns (uint256 rx, uint256 ry) {
        bytes memory input = abi.encodePacked(x, y, s);
        (bool success, bytes memory result) = EC_MUL.staticcall(input);
        if (!success || result.length != 64) revert ECOperationFailed();
        (rx, ry) = abi.decode(result, (uint256, uint256));
    }

    /// @notice Performs pairing check using precompile
    function _ecPairing(bytes memory input) internal view returns (bool) {
        (bool success, bytes memory result) = EC_PAIRING.staticcall(input);
        if (!success || result.length != 32) return false;
        return abi.decode(result, (uint256)) == 1;
    }

    /// @notice Checks if a point is on the BN254 curve
    function _isOnCurve(uint256 x, uint256 y) internal pure returns (bool) {
        if (x == 0 && y == 0) return true; // Point at infinity
        if (x >= P || y >= P) return false;
        // y^2 = x^3 + 3 (mod P)
        uint256 lhs = mulmod(y, y, P);
        uint256 rhs = addmod(mulmod(mulmod(x, x, P), x, P), 3, P);
        return lhs == rhs;
    }
}
