// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "forge-std/Test.sol";
import "../src/verification/BatchVerifier.sol";
import "../src/verification/Halo2Verifier.sol";
import "../src/mocks/MockVerifier.sol";
import "./ProofFixtures.t.sol";

/// @title BatchVerificationTest
/// @notice Comprehensive tests for batch proof verification
/// @dev Tests gas optimization, error codes, batch operations, and edge cases
contract BatchVerificationTest is Test {
    // ============ Contracts ============
    BatchVerifier public batchVerifier;
    MockVerifier public mockVerifier;
    Halo2Verifier public realVerifier;

    // ============ Actors ============
    address public owner;
    address public prover1;
    address public prover2;
    address public prover3;

    // ============ Constants ============
    uint256 constant GAS_TARGET = 250_000; // Target: <250K gas per proof

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

    function setUp() public {
        owner = address(this);
        prover1 = makeAddr("prover1");
        prover2 = makeAddr("prover2");
        prover3 = makeAddr("prover3");

        // Deploy contracts
        mockVerifier = new MockVerifier();
        realVerifier = new Halo2Verifier();
        batchVerifier = new BatchVerifier(address(mockVerifier));
    }

    // ============ Helper Functions ============

    function _createValidProof() internal pure returns (bytes memory) {
        return new bytes(320);
    }

    function _createValidPublicInputs() internal pure returns (uint256[] memory) {
        uint256[] memory inputs = new uint256[](8);
        inputs[0] = 12345;  // oldHashLo
        inputs[1] = 67890;  // oldHashHi
        inputs[2] = 1111;   // newHashLo
        inputs[3] = 2222;   // newHashHi
        inputs[4] = 100;    // loss
        inputs[5] = 10;     // errorBound
        inputs[6] = 1;      // stepNumber
        inputs[7] = 0;      // errorChecksum (placeholder for verifier-only tests)
        return inputs;
    }

    function _createInvalidFieldElementInputs() internal pure returns (uint256[] memory) {
        uint256[] memory inputs = new uint256[](8);
        inputs[0] = 12345;
        inputs[1] = 67890;
        inputs[2] = 1111;
        inputs[3] = 2222;
        inputs[4] = type(uint256).max; // Exceeds scalar field
        inputs[5] = 10;
        inputs[6] = 1;
        inputs[7] = 0;      // errorChecksum (placeholder)
        return inputs;
    }

    function _createWrongCountInputs() internal pure returns (uint256[] memory) {
        return new uint256[](5); // Only 5 instead of 8
    }

    // ============ Single Verification Tests ============

    /// @notice Test successful single proof verification
    function test_SingleVerification_Success() public {
        mockVerifier.setShouldPass(true);

        bytes memory proof = _createValidProof();
        uint256[] memory inputs = _createValidPublicInputs();

        BatchVerifier.VerificationResult memory result = batchVerifier.verifySingle(proof, inputs);

        assertTrue(result.isValid, "Proof should be valid");
        assertEq(result.errorCode, 0, "Error code should be 0");
        assertLt(result.gasUsed, GAS_TARGET, "Should be under gas target");
    }

    /// @notice Test failed single proof verification
    function test_SingleVerification_Failure() public {
        mockVerifier.setShouldPass(false);

        bytes memory proof = _createValidProof();
        uint256[] memory inputs = _createValidPublicInputs();

        BatchVerifier.VerificationResult memory result = batchVerifier.verifySingle(proof, inputs);

        assertFalse(result.isValid, "Proof should be invalid");
        assertEq(result.errorCode, batchVerifier.ERR_PROOF_VERIFICATION_FAILED());
    }

    /// @notice Test verification with invalid public inputs count
    function test_SingleVerification_InvalidInputsCount() public {
        bytes memory proof = _createValidProof();
        uint256[] memory inputs = _createWrongCountInputs();

        BatchVerifier.VerificationResult memory result = batchVerifier.verifySingle(proof, inputs);

        assertFalse(result.isValid, "Should fail with wrong inputs count");
        assertEq(result.errorCode, batchVerifier.ERR_INVALID_PUBLIC_INPUTS());
    }

    /// @notice Test verification with proof too short
    function test_SingleVerification_ProofTooShort() public {
        bytes memory shortProof = new bytes(100); // Less than 320
        uint256[] memory inputs = _createValidPublicInputs();

        BatchVerifier.VerificationResult memory result = batchVerifier.verifySingle(shortProof, inputs);

        assertFalse(result.isValid, "Should fail with short proof");
        assertEq(result.errorCode, batchVerifier.ERR_PROOF_TOO_SHORT());
    }

    /// @notice Test verification with invalid field element (using real verifier)
    function test_SingleVerification_InvalidFieldElement() public {
        BatchVerifier realBatchVerifier = new BatchVerifier(address(realVerifier));

        bytes memory proof = _createValidProof();
        uint256[] memory inputs = _createInvalidFieldElementInputs();

        BatchVerifier.VerificationResult memory result = realBatchVerifier.verifySingle(proof, inputs);

        assertFalse(result.isValid, "Should fail with invalid field element");
        assertEq(result.errorCode, realBatchVerifier.ERR_PROOF_FIELD_ELEMENT_INVALID());
    }

    /// @notice Test verify and record prevents replay
    function test_VerifyAndRecord_PreventsReplay() public {
        mockVerifier.setShouldPass(true);

        bytes memory proof = _createValidProof();
        uint256[] memory inputs = _createValidPublicInputs();

        // First verification should succeed
        BatchVerifier.VerificationResult memory result1 = batchVerifier.verifyAndRecord(proof, inputs);
        assertTrue(result1.isValid, "First verification should succeed");

        // Second verification should fail (replay)
        BatchVerifier.VerificationResult memory result2 = batchVerifier.verifyAndRecord(proof, inputs);
        assertFalse(result2.isValid, "Replay should fail");
        assertEq(result2.errorCode, batchVerifier.ERR_REPLAY_DETECTED());
    }

    // ============ Batch Verification Tests ============

    /// @notice Test batch verification with all valid proofs
    function test_BatchVerification_AllValid() public {
        mockVerifier.setShouldPass(true);

        bytes[] memory proofs = new bytes[](3);
        uint256[][] memory inputsArray = new uint256[][](3);

        for (uint256 i = 0; i < 3; i++) {
            proofs[i] = _createValidProof();
            inputsArray[i] = _createValidPublicInputs();
        }

        BatchVerifier.BatchResult memory result = batchVerifier.batchVerify(proofs, inputsArray);

        assertEq(result.totalProofs, 3);
        assertEq(result.validProofs, 3);
        assertEq(result.invalidProofs, 0);

        for (uint256 i = 0; i < 3; i++) {
            assertTrue(result.results[i].isValid);
            assertEq(result.results[i].errorCode, 0);
        }
    }

    /// @notice Test batch verification with mixed results
    function test_BatchVerification_MixedResults() public {
        bytes[] memory proofs = new bytes[](4);
        uint256[][] memory inputsArray = new uint256[][](4);

        // Set up 4 proofs: 2 valid, 2 invalid
        proofs[0] = _createValidProof();
        inputsArray[0] = _createValidPublicInputs();

        proofs[1] = new bytes(100); // Too short
        inputsArray[1] = _createValidPublicInputs();

        proofs[2] = _createValidProof();
        inputsArray[2] = _createValidPublicInputs();

        proofs[3] = _createValidProof();
        inputsArray[3] = _createWrongCountInputs(); // Wrong count

        mockVerifier.setShouldPass(true);

        BatchVerifier.BatchResult memory result = batchVerifier.batchVerify(proofs, inputsArray);

        assertEq(result.totalProofs, 4);
        assertEq(result.validProofs, 2);
        assertEq(result.invalidProofs, 2);

        assertTrue(result.results[0].isValid);
        assertFalse(result.results[1].isValid);
        assertEq(result.results[1].errorCode, batchVerifier.ERR_PROOF_TOO_SHORT());
        assertTrue(result.results[2].isValid);
        assertFalse(result.results[3].isValid);
        assertEq(result.results[3].errorCode, batchVerifier.ERR_INVALID_PUBLIC_INPUTS());
    }

    /// @notice Test batch verification with all invalid proofs
    function test_BatchVerification_AllInvalid() public {
        mockVerifier.setShouldPass(false);

        bytes[] memory proofs = new bytes[](3);
        uint256[][] memory inputsArray = new uint256[][](3);

        for (uint256 i = 0; i < 3; i++) {
            proofs[i] = _createValidProof();
            inputsArray[i] = _createValidPublicInputs();
        }

        BatchVerifier.BatchResult memory result = batchVerifier.batchVerify(proofs, inputsArray);

        assertEq(result.totalProofs, 3);
        assertEq(result.validProofs, 0);
        assertEq(result.invalidProofs, 3);
    }

    /// @notice Test empty batch reverts
    function test_BatchVerification_EmptyBatch() public {
        bytes[] memory proofs = new bytes[](0);
        uint256[][] memory inputsArray = new uint256[][](0);

        vm.expectRevert(BatchVerifier.EmptyBatch.selector);
        batchVerifier.batchVerify(proofs, inputsArray);
    }

    /// @notice Test length mismatch reverts
    function test_BatchVerification_LengthMismatch() public {
        bytes[] memory proofs = new bytes[](2);
        uint256[][] memory inputsArray = new uint256[][](3);

        vm.expectRevert(BatchVerifier.LengthMismatch.selector);
        batchVerifier.batchVerify(proofs, inputsArray);
    }

    // ============ Batch Verify And Record Tests ============

    /// @notice Test batch verify and record with events
    function test_BatchVerifyAndRecord_EmitsEvents() public {
        mockVerifier.setShouldPass(true);

        BatchVerifier.ProofSubmission[] memory submissions = new BatchVerifier.ProofSubmission[](2);

        submissions[0] = BatchVerifier.ProofSubmission({
            proof: _createValidProof(),
            publicInputs: _createValidPublicInputs(),
            prover: prover1,
            modelId: 1,
            roundId: 1
        });

        // Use different public inputs for second proof to avoid same hash
        uint256[] memory inputs2 = new uint256[](8);
        inputs2[0] = 22222;  // Different oldHashLo
        inputs2[1] = 33333;  // Different oldHashHi
        inputs2[2] = 4444;   // Different newHashLo
        inputs2[3] = 5555;   // Different newHashHi
        inputs2[4] = 200;    // Different loss
        inputs2[5] = 20;     // Different errorBound
        inputs2[6] = 2;      // Different stepNumber
        inputs2[7] = 0;      // errorChecksum (placeholder)

        submissions[1] = BatchVerifier.ProofSubmission({
            proof: _createValidProof(),
            publicInputs: inputs2,
            prover: prover2,
            modelId: 1,
            roundId: 2
        });

        BatchVerifier.BatchResult memory result = batchVerifier.batchVerifyAndRecord(submissions);

        assertEq(result.validProofs, 2);
        assertEq(result.invalidProofs, 0);
    }

    /// @notice Test batch verify and record detects invalid proofs
    function test_BatchVerifyAndRecord_DetectsInvalid() public {
        mockVerifier.setShouldPass(false);

        BatchVerifier.ProofSubmission[] memory submissions = new BatchVerifier.ProofSubmission[](1);

        submissions[0] = BatchVerifier.ProofSubmission({
            proof: _createValidProof(),
            publicInputs: _createValidPublicInputs(),
            prover: prover1,
            modelId: 1,
            roundId: 1
        });

        BatchVerifier.BatchResult memory result = batchVerifier.batchVerifyAndRecord(submissions);

        assertEq(result.validProofs, 0);
        assertEq(result.invalidProofs, 1);
        assertEq(result.results[0].errorCode, batchVerifier.ERR_PROOF_VERIFICATION_FAILED());
    }

    /// @notice Test batch verify and record prevents replay
    function test_BatchVerifyAndRecord_PreventsReplay() public {
        mockVerifier.setShouldPass(true);

        BatchVerifier.ProofSubmission[] memory submissions = new BatchVerifier.ProofSubmission[](1);

        submissions[0] = BatchVerifier.ProofSubmission({
            proof: _createValidProof(),
            publicInputs: _createValidPublicInputs(),
            prover: prover1,
            modelId: 1,
            roundId: 1
        });

        // First batch
        BatchVerifier.BatchResult memory result1 = batchVerifier.batchVerifyAndRecord(submissions);
        assertEq(result1.validProofs, 1);

        // Second batch with same proof
        BatchVerifier.BatchResult memory result2 = batchVerifier.batchVerifyAndRecord(submissions);
        assertEq(result2.validProofs, 0);
        assertEq(result2.invalidProofs, 1);
        assertEq(result2.results[0].errorCode, batchVerifier.ERR_REPLAY_DETECTED());
    }

    // ============ Homogeneous Batch Verification Tests ============

    /// @notice Test homogeneous batch verification
    function test_HomogeneousBatchVerify() public {
        mockVerifier.setShouldPass(true);

        bytes[] memory proofs = new bytes[](3);
        uint256[][] memory inputsArray = new uint256[][](3);

        for (uint256 i = 0; i < 3; i++) {
            proofs[i] = _createValidProof();
            inputsArray[i] = _createValidPublicInputs();
        }

        (bool allValid, uint16[] memory errorCodes) = batchVerifier.batchVerifyHomogeneous(
            proofs,
            inputsArray
        );

        assertTrue(allValid);
        for (uint256 i = 0; i < 3; i++) {
            assertEq(errorCodes[i], 0);
        }
    }

    /// @notice Test homogeneous batch with failure
    function test_HomogeneousBatchVerify_WithFailure() public {
        mockVerifier.setShouldPass(false);

        bytes[] memory proofs = new bytes[](2);
        uint256[][] memory inputsArray = new uint256[][](2);

        for (uint256 i = 0; i < 2; i++) {
            proofs[i] = _createValidProof();
            inputsArray[i] = _createValidPublicInputs();
        }

        (bool allValid, uint16[] memory errorCodes) = batchVerifier.batchVerifyHomogeneous(
            proofs,
            inputsArray
        );

        assertFalse(allValid);
        for (uint256 i = 0; i < 2; i++) {
            assertEq(errorCodes[i], batchVerifier.ERR_PROOF_VERIFICATION_FAILED());
        }
    }

    /// @notice Test homogeneous batch with single proof
    function test_HomogeneousBatchVerify_SingleProof() public {
        mockVerifier.setShouldPass(true);

        bytes[] memory proofs = new bytes[](1);
        uint256[][] memory inputsArray = new uint256[][](1);

        proofs[0] = _createValidProof();
        inputsArray[0] = _createValidPublicInputs();

        (bool allValid, uint16[] memory errorCodes) = batchVerifier.batchVerifyHomogeneous(
            proofs,
            inputsArray
        );

        assertTrue(allValid);
        assertEq(errorCodes[0], 0);
    }

    // ============ Validation Helper Tests ============

    /// @notice Test commitment validation
    function test_ValidateCommitment_Success() public {
        uint256[] memory inputs = _createValidPublicInputs();
        uint256 expectedCommitment = uint256(keccak256(abi.encodePacked(inputs[0], inputs[1])));

        (bool isValid, uint16 errorCode) = batchVerifier.validateCommitment(inputs, expectedCommitment);

        assertTrue(isValid);
        assertEq(errorCode, 0);
    }

    /// @notice Test commitment validation failure
    function test_ValidateCommitment_Mismatch() public {
        uint256[] memory inputs = _createValidPublicInputs();
        uint256 wrongCommitment = 999999;

        (bool isValid, uint16 errorCode) = batchVerifier.validateCommitment(inputs, wrongCommitment);

        assertFalse(isValid);
        assertEq(errorCode, batchVerifier.ERR_OLD_COMMITMENT_MISMATCH());
    }

    /// @notice Test error bound validation
    function test_ValidateErrorBound_Success() public {
        uint256[] memory inputs = _createValidPublicInputs();
        uint256 maxErrorBound = 100;

        (bool isValid, uint16 errorCode) = batchVerifier.validateErrorBound(inputs, maxErrorBound);

        assertTrue(isValid);
        assertEq(errorCode, 0);
    }

    /// @notice Test error bound validation failure
    function test_ValidateErrorBound_Exceeded() public {
        uint256[] memory inputs = _createValidPublicInputs();
        inputs[5] = 200; // Set error bound to 200
        uint256 maxErrorBound = 100;

        (bool isValid, uint16 errorCode) = batchVerifier.validateErrorBound(inputs, maxErrorBound);

        assertFalse(isValid);
        assertEq(errorCode, batchVerifier.ERR_ERROR_BOUND_EXCEEDED());
    }

    /// @notice Test pre-validation
    function test_PreValidate_AllValid() public {
        bytes memory proof = _createValidProof();
        uint256[] memory inputs = _createValidPublicInputs();
        uint256 expectedCommitment = uint256(keccak256(abi.encodePacked(inputs[0], inputs[1])));
        uint256 maxErrorBound = 100;

        uint16 errorCode = batchVerifier.preValidate(proof, inputs, expectedCommitment, maxErrorBound);

        assertEq(errorCode, 0);
    }

    /// @notice Test pre-validation catches all errors in order
    function test_PreValidate_CatchesErrors() public {
        // Test wrong inputs count
        bytes memory proof = _createValidProof();
        uint256[] memory wrongInputs = _createWrongCountInputs();

        uint16 code1 = batchVerifier.preValidate(proof, wrongInputs, 0, 100);
        assertEq(code1, batchVerifier.ERR_INVALID_PUBLIC_INPUTS());

        // Test short proof
        bytes memory shortProof = new bytes(100);
        uint256[] memory validInputs = _createValidPublicInputs();

        uint16 code2 = batchVerifier.preValidate(shortProof, validInputs, 0, 100);
        assertEq(code2, batchVerifier.ERR_PROOF_TOO_SHORT());
    }

    // ============ Gas Estimation Tests ============

    /// @notice Test gas estimation for single verification
    function test_EstimateSingleVerifyGas() public {
        mockVerifier.setShouldPass(true);

        bytes memory proof = _createValidProof();
        uint256[] memory inputs = _createValidPublicInputs();

        uint256 gasEstimate = batchVerifier.estimateSingleVerifyGas(proof, inputs);

        assertGt(gasEstimate, 0, "Gas estimate should be positive");
        assertLt(gasEstimate, GAS_TARGET, "Should be under gas target");
    }

    /// @notice Test gas estimation for batch verification
    function test_EstimateBatchVerifyGas() public {
        mockVerifier.setShouldPass(true);

        bytes[] memory proofs = new bytes[](5);
        uint256[][] memory inputsArray = new uint256[][](5);

        for (uint256 i = 0; i < 5; i++) {
            proofs[i] = _createValidProof();
            inputsArray[i] = _createValidPublicInputs();
        }

        (uint256 totalGas, uint256 perProofGas) = batchVerifier.estimateBatchVerifyGas(proofs, inputsArray);

        assertGt(totalGas, 0, "Total gas should be positive");
        assertGt(perProofGas, 0, "Per proof gas should be positive");
        assertLt(perProofGas, GAS_TARGET, "Per proof gas should be under target");
    }

    /// @notice Test that batch verification is more gas efficient per proof
    function test_BatchVerification_GasEfficiency() public {
        mockVerifier.setShouldPass(true);

        bytes memory proof = _createValidProof();
        uint256[] memory inputs = _createValidPublicInputs();

        // Single verification gas
        uint256 singleGas = batchVerifier.estimateSingleVerifyGas(proof, inputs);

        // Batch verification gas (5 proofs)
        bytes[] memory proofs = new bytes[](5);
        uint256[][] memory inputsArray = new uint256[][](5);

        for (uint256 i = 0; i < 5; i++) {
            proofs[i] = _createValidProof();
            inputsArray[i] = _createValidPublicInputs();
        }

        (, uint256 perProofGas) = batchVerifier.estimateBatchVerifyGas(proofs, inputsArray);

        // Batch per-proof gas should not be significantly higher than single
        assertLt(perProofGas, singleGas * 2, "Batch should not be 2x worse than single");
    }

    // ============ Error Code Tests ============

    /// @notice Test error category lookup
    function test_GetErrorCategory() public {
        assertEq(batchVerifier.getErrorCategory(0x0103), "PROOF");
        assertEq(batchVerifier.getErrorCategory(0x0201), "COMMITMENT");
        assertEq(batchVerifier.getErrorCategory(0x0301), "ERROR_BOUND");
        assertEq(batchVerifier.getErrorCategory(0x0401), "DATA");
        assertEq(batchVerifier.getErrorCategory(0x0501), "TIMING");
        assertEq(batchVerifier.getErrorCategory(0x0601), "PROTOCOL");
        assertEq(batchVerifier.getErrorCategory(0x0701), "GRADIENT");
        assertEq(batchVerifier.getErrorCategory(0xFFFF), "UNKNOWN");
    }

    /// @notice Test error description lookup
    function test_GetErrorDescription() public {
        assertEq(batchVerifier.getErrorDescription(0), "SUCCESS");
        assertEq(batchVerifier.getErrorDescription(0x0102), "Proof is too short");
        assertEq(batchVerifier.getErrorDescription(0x0103), "Proof verification failed");
        assertEq(batchVerifier.getErrorDescription(0x0201), "Old commitment mismatch");
        assertEq(batchVerifier.getErrorDescription(0x0301), "Error bound exceeds maximum");
        assertEq(batchVerifier.getErrorDescription(0x0602), "Replay attack detected");
    }

    // ============ Proof Hash Tests ============

    /// @notice Test proof hash computation
    function test_ComputeProofHash() public {
        bytes memory proof = _createValidProof();
        uint256[] memory inputs = _createValidPublicInputs();

        bytes32 hash1 = batchVerifier.computeProofHash(proof, inputs);
        bytes32 hash2 = batchVerifier.computeProofHash(proof, inputs);

        assertEq(hash1, hash2, "Same input should produce same hash");

        // Different inputs should produce different hash
        inputs[0] = 999;
        bytes32 hash3 = batchVerifier.computeProofHash(proof, inputs);
        assertTrue(hash1 != hash3, "Different input should produce different hash");
    }

    /// @notice Test isProofVerified
    function test_IsProofVerified() public {
        mockVerifier.setShouldPass(true);

        bytes memory proof = _createValidProof();
        uint256[] memory inputs = _createValidPublicInputs();
        bytes32 proofHash = batchVerifier.computeProofHash(proof, inputs);

        assertFalse(batchVerifier.isProofVerified(proofHash), "Should not be verified initially");

        batchVerifier.verifyAndRecord(proof, inputs);

        assertTrue(batchVerifier.isProofVerified(proofHash), "Should be verified after recording");
    }

    // ============ Admin Function Tests ============

    /// @notice Test gas limit update
    function test_SetGasLimitPerProof() public {
        uint256 newLimit = 300_000;
        batchVerifier.setGasLimitPerProof(newLimit);
        assertEq(batchVerifier.gasLimitPerProof(), newLimit);
    }

    /// @notice Test ownership transfer
    function test_TransferOwnership() public {
        address newOwner = makeAddr("newOwner");
        batchVerifier.transferOwnership(newOwner);
        assertEq(batchVerifier.owner(), newOwner);
    }

    /// @notice Test only owner can transfer ownership
    function test_TransferOwnership_OnlyOwner() public {
        address newOwner = makeAddr("newOwner");

        vm.prank(prover1);
        vm.expectRevert(BatchVerifier.NotAuthorized.selector);
        batchVerifier.transferOwnership(newOwner);
    }

    /// @notice Test clear verified proof (emergency)
    function test_ClearVerifiedProof() public {
        mockVerifier.setShouldPass(true);

        bytes memory proof = _createValidProof();
        uint256[] memory inputs = _createValidPublicInputs();

        batchVerifier.verifyAndRecord(proof, inputs);

        bytes32 proofHash = batchVerifier.computeProofHash(proof, inputs);
        assertTrue(batchVerifier.isProofVerified(proofHash));

        batchVerifier.clearVerifiedProof(proofHash);
        assertFalse(batchVerifier.isProofVerified(proofHash));
    }

    // ============ Edge Case Tests ============

    /// @notice Test large batch verification
    function test_LargeBatchVerification() public {
        mockVerifier.setShouldPass(true);

        uint256 batchSize = 20;
        bytes[] memory proofs = new bytes[](batchSize);
        uint256[][] memory inputsArray = new uint256[][](batchSize);

        for (uint256 i = 0; i < batchSize; i++) {
            proofs[i] = _createValidProof();
            inputsArray[i] = _createValidPublicInputs();
        }

        BatchVerifier.BatchResult memory result = batchVerifier.batchVerify(proofs, inputsArray);

        assertEq(result.totalProofs, batchSize);
        assertEq(result.validProofs, batchSize);
    }

    /// @notice Test very large proof
    function test_VeryLargeProof() public {
        mockVerifier.setShouldPass(true);

        bytes memory largeProof = new bytes(10000); // 10KB proof
        uint256[] memory inputs = _createValidPublicInputs();

        BatchVerifier.VerificationResult memory result = batchVerifier.verifySingle(largeProof, inputs);

        // Should still verify (mock verifier ignores proof content)
        assertTrue(result.isValid);
        assertLt(result.gasUsed, 1_000_000, "Should complete within reasonable gas");
    }

    /// @notice Test concurrent batch verification
    function test_ConcurrentBatchVerification() public {
        mockVerifier.setShouldPass(true);

        // Simulate concurrent verification by creating multiple batch results
        BatchVerifier.BatchResult memory result1;
        BatchVerifier.BatchResult memory result2;

        bytes[] memory proofs1 = new bytes[](2);
        uint256[][] memory inputs1 = new uint256[][](2);
        proofs1[0] = _createValidProof();
        proofs1[1] = _createValidProof();
        inputs1[0] = _createValidPublicInputs();
        inputs1[1] = _createValidPublicInputs();

        bytes[] memory proofs2 = new bytes[](2);
        uint256[][] memory inputs2 = new uint256[][](2);
        proofs2[0] = _createValidProof();
        proofs2[1] = _createValidProof();
        inputs2[0] = _createValidPublicInputs();
        inputs2[1] = _createValidPublicInputs();

        result1 = batchVerifier.batchVerify(proofs1, inputs1);
        result2 = batchVerifier.batchVerify(proofs2, inputs2);

        assertEq(result1.validProofs, 2);
        assertEq(result2.validProofs, 2);
    }

    // ============ Gas Target Compliance Test ============

    /// @notice Critical test: verify gas usage is under 250K target
    function test_GasTarget_Compliance() public {
        mockVerifier.setShouldPass(true);

        bytes memory proof = _createValidProof();
        uint256[] memory inputs = _createValidPublicInputs();

        uint256 startGas = gasleft();
        batchVerifier.verifySingle(proof, inputs);
        uint256 gasUsed = startGas - gasleft();

        assertLt(gasUsed, GAS_TARGET, "Single verification must be under 250K gas");

        // Batch verification per-proof should also be under target
        bytes[] memory proofs = new bytes[](10);
        uint256[][] memory inputsArray = new uint256[][](10);

        for (uint256 i = 0; i < 10; i++) {
            proofs[i] = _createValidProof();
            inputsArray[i] = _createValidPublicInputs();
        }

        startGas = gasleft();
        batchVerifier.batchVerify(proofs, inputsArray);
        uint256 batchGasUsed = startGas - gasleft();
        uint256 perProofGas = batchGasUsed / 10;

        assertLt(perProofGas, GAS_TARGET, "Batch per-proof must be under 250K gas");
    }
}
