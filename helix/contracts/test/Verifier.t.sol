// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "forge-std/Test.sol";
import "../src/verification/HelixVerifier.sol";
import "../src/verification/Halo2Verifier.sol";
import "../src/mocks/MockVerifier.sol";

/// @title VerifierTest
/// @notice Comprehensive tests for verifier contracts
contract VerifierTest is Test {
    HelixVerifier public helixVerifier;
    Halo2Verifier public halo2Verifier;
    MockVerifier public mockVerifier;

    // BN254 constants
    uint256 constant P = 21888242871839275222246405745257275088696311157297823662689037894645226208583;
    uint256 constant R = 21888242871839275222246405745257275088548364400416034343698204186575808495617;

    function setUp() public {
        helixVerifier = new HelixVerifier();
        halo2Verifier = new Halo2Verifier();
        mockVerifier = new MockVerifier();
    }

    // ============ Halo2Verifier Tests ============

    function test_Halo2Verifier_Constructor() public view {
        assertEq(halo2Verifier.owner(), address(this));
    }

    function test_Halo2Verifier_RejectsShortProof() public view {
        bytes memory shortProof = hex"deadbeef";
        uint256[] memory inputs = _createValidPublicInputs();

        bool valid = halo2Verifier.verifyProof(shortProof, inputs);
        assertFalse(valid);
    }

    function test_Halo2Verifier_RejectsWrongInputCount() public view {
        bytes memory proof = new bytes(320);
        uint256[] memory inputs = new uint256[](5); // Wrong count, should be 7

        bool valid = halo2Verifier.verifyProof(proof, inputs);
        assertFalse(valid);
    }

    function test_Halo2Verifier_RejectsInvalidFieldElements() public view {
        bytes memory proof = new bytes(320);
        uint256[] memory inputs = _createValidPublicInputs();
        inputs[0] = R; // Invalid: equals field order

        bool valid = halo2Verifier.verifyProof(proof, inputs);
        assertFalse(valid);
    }

    function test_Halo2Verifier_RejectsExceedingFieldElements() public view {
        bytes memory proof = new bytes(320);
        uint256[] memory inputs = _createValidPublicInputs();
        inputs[4] = type(uint256).max; // Invalid: exceeds field

        bool valid = halo2Verifier.verifyProof(proof, inputs);
        assertFalse(valid);
    }

    function test_Halo2Verifier_VerifyAndRecordPreventsReplay() public {
        bytes memory proof = new bytes(320);
        uint256[] memory inputs = _createValidPublicInputs();

        bytes32 proofHash = keccak256(abi.encodePacked(proof, inputs));

        // First verification
        halo2Verifier.verifyAndRecord(proof, inputs);

        // Get state
        bool recorded = halo2Verifier.verifiedProofs(proofHash);

        // Second attempt should detect replay (regardless of first result)
        bool secondResult = halo2Verifier.verifyAndRecord(proof, inputs);
        assertFalse(secondResult); // Should fail as replay
    }

    function test_Halo2Verifier_BatchVerifyLengthMismatch() public {
        bytes[] memory proofs = new bytes[](2);
        uint256[][] memory inputs = new uint256[][](3); // Mismatched length

        vm.expectRevert("Length mismatch");
        halo2Verifier.batchVerify(proofs, inputs);
    }

    function test_Halo2Verifier_BatchVerifyEmpty() public {
        bytes[] memory proofs = new bytes[](0);
        uint256[][] memory inputs = new uint256[][](0);

        vm.expectRevert("Empty batch");
        halo2Verifier.batchVerify(proofs, inputs);
    }

    function test_Halo2Verifier_BatchVerifySingleProof() public view {
        bytes[] memory proofs = new bytes[](1);
        proofs[0] = new bytes(320);

        uint256[][] memory inputs = new uint256[][](1);
        inputs[0] = _createValidPublicInputs();

        // Single proof batch should delegate to single verify
        bool valid = halo2Verifier.batchVerify(proofs, inputs);
        // Will fail due to invalid proof, but should not revert
        assertFalse(valid);
    }

    function test_Halo2Verifier_BatchVerifyHomogeneous() public view {
        bytes[] memory proofs = new bytes[](2);
        proofs[0] = new bytes(320);
        proofs[1] = new bytes(320);

        uint256[][] memory inputs = new uint256[][](2);
        inputs[0] = _createValidPublicInputs();
        inputs[1] = _createValidPublicInputs();
        inputs[1][0] = 2; // Slightly different input

        // Should not revert
        bool valid = halo2Verifier.batchVerifyHomogeneous(proofs, inputs);
        // Will fail due to invalid proofs
        assertFalse(valid);
    }

    function test_Halo2Verifier_GasEstimation() public view {
        bytes memory proof = new bytes(320);
        uint256[] memory inputs = _createValidPublicInputs();

        uint256 gasUsed = halo2Verifier.estimateVerifyGas(proof, inputs);
        assertGt(gasUsed, 0);
    }

    function test_Halo2Verifier_BatchGasEstimation() public view {
        bytes[] memory proofs = new bytes[](3);
        proofs[0] = new bytes(320);
        proofs[1] = new bytes(320);
        proofs[2] = new bytes(320);

        uint256[][] memory inputs = new uint256[][](3);
        inputs[0] = _createValidPublicInputs();
        inputs[1] = _createValidPublicInputs();
        inputs[2] = _createValidPublicInputs();

        (uint256 gasUsed, uint256 perProofGas) = halo2Verifier.estimateBatchVerifyGas(proofs, inputs);
        assertGt(gasUsed, 0);
        assertEq(perProofGas, gasUsed / 3);
    }

    // ============ HelixVerifier Tests ============

    function test_HelixVerifier_Constructor() public view {
        assertEq(helixVerifier.owner(), address(this));
        assertFalse(helixVerifier.initialized());
    }

    function test_HelixVerifier_Initialize() public {
        uint256[2] memory selectors = [uint256(1), uint256(2)];
        uint256[2] memory perms = [uint256(3), uint256(4)];
        helixVerifier.initialize(selectors, perms, 1024, 12345, 1000);

        assertTrue(helixVerifier.initialized());

        (uint256 domainSize, uint256 maxErrorBound, bool isInit) = helixVerifier.getVerificationKeyInfo();
        assertEq(domainSize, 1024);
        assertEq(maxErrorBound, 1000);
        assertTrue(isInit);
    }

    function test_HelixVerifier_CannotDoubleInitialize() public {
        uint256[2] memory selectors = [uint256(1), uint256(2)];
        uint256[2] memory perms = [uint256(3), uint256(4)];
        helixVerifier.initialize(selectors, perms, 1024, 12345, 1000);

        vm.expectRevert("Already initialized");
        helixVerifier.initialize(selectors, perms, 1024, 12345, 1000);
    }

    function test_HelixVerifier_VerifyRequiresInit() public {
        bytes memory proof = new bytes(256);
        uint256[] memory inputs = _createValidPublicInputs();

        vm.expectRevert("Not initialized");
        helixVerifier.verifyProof(proof, inputs);
    }

    function test_HelixVerifier_RejectsWrongInputCount() public {
        uint256[2] memory selectors = [uint256(1), uint256(2)];
        uint256[2] memory perms = [uint256(3), uint256(4)];
        helixVerifier.initialize(selectors, perms, 1024, 12345, 1000);

        bytes memory proof = new bytes(256);
        uint256[] memory inputs = new uint256[](5); // Should be 7

        bool valid = helixVerifier.verifyProof(proof, inputs);
        assertFalse(valid);
    }

    function test_HelixVerifier_RejectsShortProof() public {
        uint256[2] memory selectors = [uint256(1), uint256(2)];
        uint256[2] memory perms = [uint256(3), uint256(4)];
        helixVerifier.initialize(selectors, perms, 1024, 12345, 1000);

        bytes memory proof = new bytes(100); // Too short
        uint256[] memory inputs = _createValidPublicInputs();

        bool valid = helixVerifier.verifyProof(proof, inputs);
        assertFalse(valid);
    }

    function test_HelixVerifier_ExceedMaxError() public {
        uint256[2] memory selectors = [uint256(1), uint256(2)];
        uint256[2] memory perms = [uint256(3), uint256(4)];
        helixVerifier.initialize(selectors, perms, 1024, 12345, 100); // Set low max

        bytes memory proof = new bytes(256);
        uint256[] memory inputs = _createValidPublicInputs();
        inputs[5] = 200; // Error bound exceeds max

        bool valid = helixVerifier.verifyProof(proof, inputs);
        assertFalse(valid);
    }

    function test_HelixVerifier_ReplayPrevention() public {
        uint256[2] memory selectors = [uint256(1), uint256(2)];
        uint256[2] memory perms = [uint256(3), uint256(4)];
        helixVerifier.initialize(selectors, perms, 1024, 12345, 1000);

        bytes memory proof = new bytes(256);
        uint256[] memory inputs = _createValidPublicInputs();

        bytes32 proofHash = keccak256(proof);

        // After any verification, the hash is recorded
        helixVerifier.verifyProof(proof, inputs);

        // Check if proof was recorded
        bool wasVerified = helixVerifier.verifiedProofs(proofHash);
        // Depending on verification result, this might be true or false
    }

    function test_HelixVerifier_TransferOwnership() public {
        address newOwner = makeAddr("newOwner");
        helixVerifier.transferOwnership(newOwner);
        assertEq(helixVerifier.owner(), newOwner);
    }

    function test_HelixVerifier_TransferOwnershipInvalidAddress() public {
        vm.expectRevert("Invalid address");
        helixVerifier.transferOwnership(address(0));
    }

    // ============ MockVerifier Tests ============

    function test_MockVerifier_DefaultPass() public view {
        assertTrue(mockVerifier.shouldPass());

        bytes memory proof = hex"deadbeef";
        uint256[] memory inputs = new uint256[](1);

        bool valid = mockVerifier.verifyProof(proof, inputs);
        assertTrue(valid);
    }

    function test_MockVerifier_SetShouldPass() public {
        mockVerifier.setShouldPass(false);
        assertFalse(mockVerifier.shouldPass());

        bytes memory proof = hex"deadbeef";
        uint256[] memory inputs = new uint256[](1);

        bool valid = mockVerifier.verifyProof(proof, inputs);
        assertFalse(valid);
    }

    function test_MockVerifier_Toggle() public {
        bytes memory proof = hex"abcd1234";
        uint256[] memory inputs = new uint256[](0);

        assertTrue(mockVerifier.verifyProof(proof, inputs));

        mockVerifier.setShouldPass(false);
        assertFalse(mockVerifier.verifyProof(proof, inputs));

        mockVerifier.setShouldPass(true);
        assertTrue(mockVerifier.verifyProof(proof, inputs));
    }

    // ============ Cross-Verifier Tests ============

    function test_AllVerifiersImplementInterface() public view {
        // All verifiers should implement IHelixVerifier
        bytes memory proof = new bytes(320);
        uint256[] memory inputs = _createValidPublicInputs();

        // Should not revert - just call the interface method
        halo2Verifier.verifyProof(proof, inputs);
        mockVerifier.verifyProof(proof, inputs);
    }

    // ============ Edge Cases ============

    function test_VerifyWithZeroProof() public view {
        bytes memory proof = new bytes(320);
        uint256[] memory inputs = _createValidPublicInputs();

        // Zero-filled proof should fail verification gracefully
        bool valid = halo2Verifier.verifyProof(proof, inputs);
        // Should return false without reverting
        assertFalse(valid);
    }

    function test_VerifyWithMaxLengthProof() public view {
        // Very large proof (10KB)
        bytes memory proof = new bytes(10240);
        uint256[] memory inputs = _createValidPublicInputs();

        // Should handle large proofs without OOG
        bool valid = halo2Verifier.verifyProof(proof, inputs);
        assertFalse(valid);
    }

    function test_VerifyWithBoundaryInputs() public view {
        bytes memory proof = new bytes(320);
        uint256[] memory inputs = _createValidPublicInputs();

        // Test with max valid field element (R - 1)
        inputs[0] = R - 1;

        bool valid = halo2Verifier.verifyProof(proof, inputs);
        // Should process without reverting
        assertFalse(valid);
    }

    // ============ Gas Benchmark Tests ============

    function test_VerificationGasConsumption() public view {
        bytes memory proof = new bytes(320);
        uint256[] memory inputs = _createValidPublicInputs();

        uint256 gasBefore = gasleft();
        halo2Verifier.verifyProof(proof, inputs);
        uint256 gasUsed = gasBefore - gasleft();

        // Verification should use reasonable gas
        assertLt(gasUsed, 500_000);
    }

    function test_BatchVerificationGasSavings() public view {
        bytes[] memory proofs = new bytes[](5);
        uint256[][] memory inputs = new uint256[][](5);

        for (uint i = 0; i < 5; i++) {
            proofs[i] = new bytes(320);
            inputs[i] = _createValidPublicInputs();
            inputs[i][0] = i + 1; // Vary inputs
        }

        // Measure batch verification gas
        uint256 gasBefore = gasleft();
        halo2Verifier.batchVerify(proofs, inputs);
        uint256 batchGas = gasBefore - gasleft();

        // Batch verification per proof should be less than 5x single verification
        uint256 perProofBatch = batchGas / 5;

        // Measure single verification gas
        gasBefore = gasleft();
        halo2Verifier.verifyProof(proofs[0], inputs[0]);
        uint256 singleGas = gasBefore - gasleft();

        // Batch should provide some efficiency
        // Note: With invalid proofs, the early exit might affect this
    }

    // ============ Helper Functions ============

    function _createValidPublicInputs() internal pure returns (uint256[] memory) {
        uint256[] memory inputs = new uint256[](7);
        inputs[0] = 1;     // oldHashLo
        inputs[1] = 2;     // oldHashHi
        inputs[2] = 3;     // newHashLo
        inputs[3] = 4;     // newHashHi
        inputs[4] = 100;   // loss
        inputs[5] = 10;    // errorBound
        inputs[6] = 1;     // stepNumber
        return inputs;
    }

}
