// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "./Halo2VerifierCore.sol";
import "../interfaces/IHelixVerifier.sol";

/// @title RLCAggregationVerifier
/// @notice Verifies aggregated ZK proofs from the RLC (Random Linear Combination) aggregation circuit.
/// @dev Wraps a Halo2VerifierCore with a SEPARATE verifying key for the aggregation circuit
///      (SHPLONKAggregationCircuit). A single verification call replaces N individual verifications.
///      The aggregation circuit enforces PI chaining, Fiat-Shamir RLC commitment, and error accumulation
///      internally — the verifier only needs to check the single aggregated proof.
contract RLCAggregationVerifier is IHelixVerifier {
    // ============ Immutable State ============

    /// @notice The PSE-generated core verifier contract (shared with individual proof verifier)
    Halo2VerifierCore public immutable core;

    /// @notice The aggregation circuit's verifying key contract address
    /// @dev Different from individual proof VK — generated from SHPLONKAggregationCircuit
    address public immutable aggregationVK;

    /// @notice Contract deployer
    address public immutable owner;

    // ============ Mutable State ============

    /// @notice Tracks used aggregate proof hashes for replay protection
    mapping(bytes32 => bool) public usedAggregateProofs;

    // ============ Events ============

    /// @notice Emitted when an aggregated proof is successfully verified and recorded
    event AggregatedProofVerified(
        uint256 indexed roundId,
        uint256 numSteps,
        bytes32 indexed proofHash,
        uint256 finalCommitmentLo,
        uint256 finalCommitmentHi,
        uint256 totalErrorBound
    );

    /// @notice Emitted when a replay attack is detected
    event ReplayAttemptDetected(bytes32 indexed proofHash, address indexed submitter);

    // ============ Constructor ============

    /// @param _core Address of the Halo2VerifierCore contract
    /// @param _aggregationVK Address of the aggregation circuit's verifying key contract
    constructor(address _core, address _aggregationVK) {
        require(_core != address(0), "Invalid core verifier");
        require(_aggregationVK != address(0), "Invalid aggregation VK");
        core = Halo2VerifierCore(_core);
        aggregationVK = _aggregationVK;
        owner = msg.sender;
    }

    // ============ IHelixVerifier Interface ============

    /// @notice Verify a single aggregated ZK proof
    /// @dev Delegates to the core verifier with the aggregation circuit VK
    /// @param proof The aggregated proof bytes (~1856 bytes for PSE SHPLONK format)
    /// @param publicInputs The 8 public inputs matching contract interface
    /// @return isValid True if the proof is valid
    function verifyProof(
        bytes memory proof,
        uint256[] memory publicInputs
    ) external view override returns (bool isValid) {
        if (publicInputs.length != 8) return false;
        try core.verifyProof(aggregationVK, proof, publicInputs) returns (bool result) {
            return result;
        } catch {
            return false;
        }
    }

    // ============ Aggregation-Specific Verification ============

    /// @notice Verify an aggregated proof with batch metadata validation
    /// @param proof The aggregated ZK proof bytes
    /// @param publicInputs 8 public inputs [oldHashLo, oldHashHi, newHashLo, newHashHi, totalLoss, totalError, stepInfo, rlcCommitment]
    /// @param numSteps Number of training steps aggregated (1-32)
    /// @param roundId Training round identifier
    /// @return isValid True if proof is valid and metadata checks pass
    function verifyAggregatedProof(
        bytes memory proof,
        uint256[] memory publicInputs,
        uint256 numSteps,
        uint256 roundId
    ) external view returns (bool isValid) {
        if (numSteps == 0 || numSteps > 32) return false;
        if (publicInputs.length != 8) return false;

        try core.verifyProof(aggregationVK, proof, publicInputs) returns (bool result) {
            return result;
        } catch {
            return false;
        }
    }

    /// @notice Verify and record an aggregated proof to prevent replay
    /// @param proof The aggregated ZK proof bytes
    /// @param publicInputs 8 public inputs
    /// @param numSteps Number of training steps aggregated
    /// @param roundId Training round identifier
    /// @return isValid True if proof is valid, not replayed, and metadata is correct
    function verifyAndRecordAggregated(
        bytes memory proof,
        uint256[] memory publicInputs,
        uint256 numSteps,
        uint256 roundId
    ) external returns (bool isValid) {
        require(numSteps > 0 && numSteps <= 32, "Invalid batch size");
        require(publicInputs.length == 8, "Invalid public inputs count");

        bytes32 proofHash = keccak256(abi.encodePacked(proof, publicInputs, numSteps, roundId));
        if (usedAggregateProofs[proofHash]) {
            emit ReplayAttemptDetected(proofHash, msg.sender);
            return false;
        }

        bool valid;
        try core.verifyProof(aggregationVK, proof, publicInputs) returns (bool result) {
            valid = result;
        } catch {
            valid = false;
        }

        if (valid) {
            usedAggregateProofs[proofHash] = true;
            emit AggregatedProofVerified(
                roundId, numSteps, proofHash,
                publicInputs[2], publicInputs[3], publicInputs[5]
            );
        }
        return valid;
    }

    // ============ Gas Estimation ============

    /// @notice Estimate gas cost for verifying a single aggregated proof
    function estimateVerifyGas(
        bytes memory proof,
        uint256[] memory publicInputs
    ) external view returns (uint256 gasUsed) {
        uint256 gasBefore = gasleft();
        try core.verifyProof(aggregationVK, proof, publicInputs) {} catch {}
        gasUsed = gasBefore - gasleft();
    }

    // ============ View Functions ============

    /// @notice Check if a proof hash has been used
    function isProofUsed(bytes32 proofHash) external view returns (bool) {
        return usedAggregateProofs[proofHash];
    }
}
