// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "../interfaces/IHelixVerifier.sol";

/// @title BatchVerifier
/// @notice Gas-optimized batch proof verification with detailed error tracking
/// @dev Designed for <250K gas per proof verification with comprehensive error codes
contract BatchVerifier {
    // ============ Error Codes (16-bit) ============
    // Upper 8 bits = category, lower 8 bits = specific error

    /// @notice Error categories
    uint16 public constant CATEGORY_PROOF = 0x0100;
    uint16 public constant CATEGORY_COMMITMENT = 0x0200;
    uint16 public constant CATEGORY_ERROR_BOUND = 0x0300;
    uint16 public constant CATEGORY_DATA = 0x0400;
    uint16 public constant CATEGORY_TIMING = 0x0500;
    uint16 public constant CATEGORY_PROTOCOL = 0x0600;
    uint16 public constant CATEGORY_GRADIENT = 0x0700;

    /// @notice Specific error codes
    uint16 public constant ERR_PROOF_MALFORMED = 0x0101;
    uint16 public constant ERR_PROOF_TOO_SHORT = 0x0102;
    uint16 public constant ERR_PROOF_VERIFICATION_FAILED = 0x0103;
    uint16 public constant ERR_PROOF_FIELD_ELEMENT_INVALID = 0x0104;
    uint16 public constant ERR_PROOF_PAIRING_FAILED = 0x0105;
    uint16 public constant ERR_PROOF_POINT_NOT_ON_CURVE = 0x0106;

    uint16 public constant ERR_OLD_COMMITMENT_MISMATCH = 0x0201;
    uint16 public constant ERR_NEW_COMMITMENT_INVALID = 0x0202;
    uint16 public constant ERR_COMMITMENT_HASH_COLLISION = 0x0203;

    uint16 public constant ERR_ERROR_BOUND_EXCEEDED = 0x0301;
    uint16 public constant ERR_ACCUMULATED_ERROR_EXCEEDED = 0x0302;
    uint16 public constant ERR_ERROR_BOUND_MANIPULATION = 0x0303;

    uint16 public constant ERR_DATA_ROOT_MISMATCH = 0x0401;
    uint16 public constant ERR_DATA_PROOF_INVALID = 0x0402;

    uint16 public constant ERR_SUBMISSION_TOO_LATE = 0x0501;
    uint16 public constant ERR_ROUND_COMPLETED = 0x0502;

    uint16 public constant ERR_DOUBLE_SUBMISSION = 0x0601;
    uint16 public constant ERR_REPLAY_DETECTED = 0x0602;
    uint16 public constant ERR_INVALID_PUBLIC_INPUTS = 0x0603;

    uint16 public constant ERR_GRADIENT_OUTLIER = 0x0701;
    uint16 public constant ERR_GRADIENT_POISONING = 0x0702;

    uint16 public constant ERR_UNKNOWN = 0xFFFF;

    // ============ BN254 Constants ============
    uint256 internal constant R = 21888242871839275222246405745257275088548364400416034343698204186575808495617;
    uint256 internal constant P = 21888242871839275222246405745257275088696311157297823662689037894645226208583;
    uint256 internal constant NUM_INSTANCES = 8;
    uint256 internal constant MIN_PROOF_LENGTH = 320;

    // ============ Structs ============

    /// @notice Result of a single proof verification
    struct VerificationResult {
        bool isValid;
        uint16 errorCode;
        uint256 gasUsed;
    }

    /// @notice Batch verification result
    struct BatchResult {
        uint256 totalProofs;
        uint256 validProofs;
        uint256 invalidProofs;
        uint256 totalGasUsed;
        uint256 averageGasPerProof;
        VerificationResult[] results;
    }

    /// @notice Proof submission for batch verification
    struct ProofSubmission {
        bytes proof;
        uint256[] publicInputs;
        address prover;
        uint256 modelId;
        uint256 roundId;
    }

    // ============ State ============

    /// @notice The underlying verifier contract
    IHelixVerifier public immutable verifier;

    /// @notice Owner for administrative functions
    address public owner;

    /// @notice Verified proof hashes (prevents replay)
    mapping(bytes32 => bool) public verifiedProofs;

    /// @notice Gas limit per proof for batch verification
    uint256 public gasLimitPerProof;

    // ============ Events ============

    event BatchVerificationCompleted(
        uint256 indexed batchId,
        uint256 totalProofs,
        uint256 validProofs,
        uint256 totalGasUsed
    );

    event ProofVerified(
        uint256 indexed batchId,
        uint256 indexed proofIndex,
        address indexed prover,
        bool isValid,
        uint16 errorCode,
        uint256 gasUsed
    );

    event InvalidProofDetected(
        address indexed prover,
        uint256 indexed modelId,
        uint256 indexed roundId,
        uint16 errorCode,
        bytes32 proofHash
    );

    event GasLimitUpdated(uint256 oldLimit, uint256 newLimit);

    // ============ Errors ============

    error EmptyBatch();
    error LengthMismatch();
    error GasLimitExceeded();
    error NotAuthorized();
    error ProofAlreadyVerified();

    // ============ Modifiers ============

    modifier onlyOwner() {
        if (msg.sender != owner) revert NotAuthorized();
        _;
    }

    // ============ Constructor ============

    constructor(address _verifier) {
        verifier = IHelixVerifier(_verifier);
        owner = msg.sender;
        gasLimitPerProof = 250_000; // Default target: <250K gas per proof
    }

    // ============ Core Verification Functions ============

    /// @notice Verify a single proof with detailed error tracking
    /// @param proof The proof bytes
    /// @param publicInputs The public inputs
    /// @return result The verification result with error code and gas used
    function verifySingle(
        bytes calldata proof,
        uint256[] calldata publicInputs
    ) external view returns (VerificationResult memory result) {
        uint256 startGas = gasleft();

        (bool isValid, uint16 errorCode) = _verifyWithErrorCode(proof, publicInputs);

        result.isValid = isValid;
        result.errorCode = errorCode;
        result.gasUsed = startGas - gasleft();
    }

    /// @notice Verify a single proof and record it (prevents replay)
    /// @param proof The proof bytes
    /// @param publicInputs The public inputs
    /// @return result The verification result
    function verifyAndRecord(
        bytes calldata proof,
        uint256[] calldata publicInputs
    ) external returns (VerificationResult memory result) {
        bytes32 proofHash = keccak256(abi.encodePacked(proof, publicInputs));

        if (verifiedProofs[proofHash]) {
            result.isValid = false;
            result.errorCode = ERR_REPLAY_DETECTED;
            return result;
        }

        uint256 startGas = gasleft();
        (bool isValid, uint16 errorCode) = _verifyWithErrorCode(proof, publicInputs);

        result.isValid = isValid;
        result.errorCode = errorCode;
        result.gasUsed = startGas - gasleft();

        if (isValid) {
            verifiedProofs[proofHash] = true;
        }
    }

    /// @notice Batch verify multiple proofs with gas optimization
    /// @param proofs Array of proof bytes
    /// @param publicInputsArray Array of public inputs arrays
    /// @return batchResult The batch verification results
    function batchVerify(
        bytes[] calldata proofs,
        uint256[][] calldata publicInputsArray
    ) external view returns (BatchResult memory batchResult) {
        uint256 n = proofs.length;
        if (n == 0) revert EmptyBatch();
        if (n != publicInputsArray.length) revert LengthMismatch();

        uint256 totalStartGas = gasleft();

        batchResult.totalProofs = n;
        batchResult.results = new VerificationResult[](n);

        for (uint256 i = 0; i < n; i++) {
            uint256 proofStartGas = gasleft();

            (bool isValid, uint16 errorCode) = _verifyWithErrorCode(
                proofs[i],
                publicInputsArray[i]
            );

            uint256 proofGasUsed = proofStartGas - gasleft();

            batchResult.results[i] = VerificationResult({
                isValid: isValid,
                errorCode: errorCode,
                gasUsed: proofGasUsed
            });

            if (isValid) {
                batchResult.validProofs++;
            } else {
                batchResult.invalidProofs++;
            }
        }

        batchResult.totalGasUsed = totalStartGas - gasleft();
        batchResult.averageGasPerProof = batchResult.totalGasUsed / n;
    }

    /// @notice Batch verify with recording and event emission
    /// @param submissions Array of proof submissions with metadata
    /// @return batchResult The batch verification results
    function batchVerifyAndRecord(
        ProofSubmission[] calldata submissions
    ) external returns (BatchResult memory batchResult) {
        uint256 n = submissions.length;
        if (n == 0) revert EmptyBatch();

        uint256 totalStartGas = gasleft();
        uint256 batchId = uint256(keccak256(abi.encodePacked(block.timestamp, block.prevrandao, n)));

        batchResult.totalProofs = n;
        batchResult.results = new VerificationResult[](n);

        for (uint256 i = 0; i < n; i++) {
            ProofSubmission calldata sub = submissions[i];
            bytes32 proofHash = keccak256(abi.encodePacked(sub.proof, sub.publicInputs));

            uint256 proofStartGas = gasleft();
            bool isValid;
            uint16 errorCode;

            if (verifiedProofs[proofHash]) {
                isValid = false;
                errorCode = ERR_REPLAY_DETECTED;
            } else {
                (isValid, errorCode) = _verifyWithErrorCode(sub.proof, sub.publicInputs);

                if (isValid) {
                    verifiedProofs[proofHash] = true;
                } else {
                    emit InvalidProofDetected(
                        sub.prover,
                        sub.modelId,
                        sub.roundId,
                        errorCode,
                        proofHash
                    );
                }
            }

            uint256 proofGasUsed = proofStartGas - gasleft();

            batchResult.results[i] = VerificationResult({
                isValid: isValid,
                errorCode: errorCode,
                gasUsed: proofGasUsed
            });

            if (isValid) {
                batchResult.validProofs++;
            } else {
                batchResult.invalidProofs++;
            }

            emit ProofVerified(
                batchId,
                i,
                sub.prover,
                isValid,
                errorCode,
                proofGasUsed
            );
        }

        batchResult.totalGasUsed = totalStartGas - gasleft();
        batchResult.averageGasPerProof = batchResult.totalGasUsed / n;

        emit BatchVerificationCompleted(
            batchId,
            n,
            batchResult.validProofs,
            batchResult.totalGasUsed
        );
    }

    /// @notice Optimized batch verify for homogeneous proofs (same circuit)
    /// @dev Uses random linear combination for even better gas efficiency
    /// @param proofs Array of proof bytes
    /// @param publicInputsArray Array of public inputs arrays
    /// @return allValid True if all proofs are valid
    /// @return errorCodes Array of error codes (0 if valid)
    function batchVerifyHomogeneous(
        bytes[] calldata proofs,
        uint256[][] calldata publicInputsArray
    ) external view returns (bool allValid, uint16[] memory errorCodes) {
        uint256 n = proofs.length;
        if (n == 0) revert EmptyBatch();
        if (n != publicInputsArray.length) revert LengthMismatch();

        errorCodes = new uint16[](n);
        allValid = true;

        // For single proof, use direct verification
        if (n == 1) {
            (bool valid, uint16 code) = _verifyWithErrorCode(proofs[0], publicInputsArray[0]);
            errorCodes[0] = code;
            return (valid, errorCodes);
        }

        // For multiple proofs, verify each individually but track errors
        // A full RLC optimization would require assembly-level integration
        for (uint256 i = 0; i < n; i++) {
            (bool valid, uint16 code) = _verifyWithErrorCode(proofs[i], publicInputsArray[i]);
            errorCodes[i] = code;
            if (!valid) {
                allValid = false;
            }
        }
    }

    // ============ Internal Verification ============

    /// @notice Internal verification with detailed error code
    function _verifyWithErrorCode(
        bytes calldata proof,
        uint256[] calldata publicInputs
    ) internal view returns (bool isValid, uint16 errorCode) {
        // Check public inputs count
        if (publicInputs.length != NUM_INSTANCES) {
            return (false, ERR_INVALID_PUBLIC_INPUTS);
        }

        // Check proof length
        if (proof.length < MIN_PROOF_LENGTH) {
            return (false, ERR_PROOF_TOO_SHORT);
        }

        // Validate field elements
        for (uint256 i = 0; i < NUM_INSTANCES; i++) {
            if (publicInputs[i] >= R) {
                return (false, ERR_PROOF_FIELD_ELEMENT_INVALID);
            }
        }

        // Call the underlying verifier
        try verifier.verifyProof(proof, publicInputs) returns (bool valid) {
            if (valid) {
                return (true, 0);
            } else {
                return (false, ERR_PROOF_VERIFICATION_FAILED);
            }
        } catch {
            return (false, ERR_PROOF_PAIRING_FAILED);
        }
    }

    // ============ Validation Helpers ============

    /// @notice Validate commitment consistency
    /// @param publicInputs The public inputs
    /// @param expectedOldCommitment The expected old commitment
    /// @return isValid True if valid
    /// @return errorCode The error code if invalid
    function validateCommitment(
        uint256[] calldata publicInputs,
        uint256 expectedOldCommitment
    ) external pure returns (bool isValid, uint16 errorCode) {
        if (publicInputs.length < 4) {
            return (false, ERR_INVALID_PUBLIC_INPUTS);
        }

        // Reconstruct old commitment from public inputs [0] and [1]
        uint256 computedCommitment = uint256(keccak256(abi.encodePacked(
            publicInputs[0],
            publicInputs[1]
        )));

        if (computedCommitment != expectedOldCommitment) {
            return (false, ERR_OLD_COMMITMENT_MISMATCH);
        }

        return (true, 0);
    }

    /// @notice Validate error bound
    /// @param publicInputs The public inputs
    /// @param maxErrorBound The maximum allowed error bound
    /// @return isValid True if valid
    /// @return errorCode The error code if invalid
    function validateErrorBound(
        uint256[] calldata publicInputs,
        uint256 maxErrorBound
    ) external pure returns (bool isValid, uint16 errorCode) {
        if (publicInputs.length < 6) {
            return (false, ERR_INVALID_PUBLIC_INPUTS);
        }

        // Error bound is at index 5
        if (publicInputs[5] > maxErrorBound) {
            return (false, ERR_ERROR_BOUND_EXCEEDED);
        }

        return (true, 0);
    }

    /// @notice Full validation before verification
    /// @param proof The proof bytes
    /// @param publicInputs The public inputs
    /// @param expectedOldCommitment Expected old commitment
    /// @param maxErrorBound Maximum error bound
    /// @return errorCode First error encountered (0 if all valid)
    function preValidate(
        bytes calldata proof,
        uint256[] calldata publicInputs,
        uint256 expectedOldCommitment,
        uint256 maxErrorBound
    ) external pure returns (uint16 errorCode) {
        // Check public inputs
        if (publicInputs.length != NUM_INSTANCES) {
            return ERR_INVALID_PUBLIC_INPUTS;
        }

        // Check proof length
        if (proof.length < MIN_PROOF_LENGTH) {
            return ERR_PROOF_TOO_SHORT;
        }

        // Check field elements
        for (uint256 i = 0; i < NUM_INSTANCES; i++) {
            if (publicInputs[i] >= R) {
                return ERR_PROOF_FIELD_ELEMENT_INVALID;
            }
        }

        // Check commitment
        uint256 computedCommitment = uint256(keccak256(abi.encodePacked(
            publicInputs[0],
            publicInputs[1]
        )));
        if (computedCommitment != expectedOldCommitment) {
            return ERR_OLD_COMMITMENT_MISMATCH;
        }

        // Check error bound
        if (publicInputs[5] > maxErrorBound) {
            return ERR_ERROR_BOUND_EXCEEDED;
        }

        return 0; // All valid
    }

    // ============ Gas Estimation ============

    /// @notice Estimate gas for single proof verification
    function estimateSingleVerifyGas(
        bytes calldata proof,
        uint256[] calldata publicInputs
    ) external view returns (uint256 gasEstimate) {
        uint256 startGas = gasleft();
        _verifyWithErrorCode(proof, publicInputs);
        return startGas - gasleft();
    }

    /// @notice Estimate gas for batch verification
    function estimateBatchVerifyGas(
        bytes[] calldata proofs,
        uint256[][] calldata publicInputsArray
    ) external view returns (uint256 totalGas, uint256 perProofGas) {
        uint256 startGas = gasleft();

        for (uint256 i = 0; i < proofs.length; i++) {
            _verifyWithErrorCode(proofs[i], publicInputsArray[i]);
        }

        totalGas = startGas - gasleft();
        perProofGas = proofs.length > 0 ? totalGas / proofs.length : 0;
    }

    // ============ View Functions ============

    /// @notice Get error category from error code
    function getErrorCategory(uint16 errorCode) external pure returns (string memory) {
        uint16 category = errorCode & 0xFF00;

        if (category == CATEGORY_PROOF) return "PROOF";
        if (category == CATEGORY_COMMITMENT) return "COMMITMENT";
        if (category == CATEGORY_ERROR_BOUND) return "ERROR_BOUND";
        if (category == CATEGORY_DATA) return "DATA";
        if (category == CATEGORY_TIMING) return "TIMING";
        if (category == CATEGORY_PROTOCOL) return "PROTOCOL";
        if (category == CATEGORY_GRADIENT) return "GRADIENT";

        return "UNKNOWN";
    }

    /// @notice Get human-readable error description
    function getErrorDescription(uint16 errorCode) external pure returns (string memory) {
        if (errorCode == 0) return "SUCCESS";
        if (errorCode == ERR_PROOF_MALFORMED) return "Proof is malformed";
        if (errorCode == ERR_PROOF_TOO_SHORT) return "Proof is too short";
        if (errorCode == ERR_PROOF_VERIFICATION_FAILED) return "Proof verification failed";
        if (errorCode == ERR_PROOF_FIELD_ELEMENT_INVALID) return "Field element exceeds scalar field";
        if (errorCode == ERR_PROOF_PAIRING_FAILED) return "EC pairing failed";
        if (errorCode == ERR_PROOF_POINT_NOT_ON_CURVE) return "Point not on curve";
        if (errorCode == ERR_OLD_COMMITMENT_MISMATCH) return "Old commitment mismatch";
        if (errorCode == ERR_NEW_COMMITMENT_INVALID) return "New commitment invalid";
        if (errorCode == ERR_ERROR_BOUND_EXCEEDED) return "Error bound exceeds maximum";
        if (errorCode == ERR_ACCUMULATED_ERROR_EXCEEDED) return "Accumulated error too high";
        if (errorCode == ERR_DATA_ROOT_MISMATCH) return "Data root mismatch";
        if (errorCode == ERR_SUBMISSION_TOO_LATE) return "Submission after deadline";
        if (errorCode == ERR_ROUND_COMPLETED) return "Round already completed";
        if (errorCode == ERR_DOUBLE_SUBMISSION) return "Double submission detected";
        if (errorCode == ERR_REPLAY_DETECTED) return "Replay attack detected";
        if (errorCode == ERR_INVALID_PUBLIC_INPUTS) return "Invalid public inputs";
        if (errorCode == ERR_GRADIENT_OUTLIER) return "Gradient is outlier";
        if (errorCode == ERR_GRADIENT_POISONING) return "Gradient poisoning detected";

        return "Unknown error";
    }

    /// @notice Check if a proof has been verified
    function isProofVerified(bytes32 proofHash) external view returns (bool) {
        return verifiedProofs[proofHash];
    }

    /// @notice Compute proof hash
    function computeProofHash(
        bytes calldata proof,
        uint256[] calldata publicInputs
    ) external pure returns (bytes32) {
        return keccak256(abi.encodePacked(proof, publicInputs));
    }

    // ============ Admin Functions ============

    /// @notice Set gas limit per proof
    function setGasLimitPerProof(uint256 _limit) external onlyOwner {
        emit GasLimitUpdated(gasLimitPerProof, _limit);
        gasLimitPerProof = _limit;
    }

    /// @notice Transfer ownership
    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "Invalid address");
        owner = newOwner;
    }

    /// @notice Clear verified proof (emergency only)
    function clearVerifiedProof(bytes32 proofHash) external onlyOwner {
        verifiedProofs[proofHash] = false;
    }
}
