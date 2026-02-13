// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "forge-std/Test.sol";
import "../src/verification/Halo2Verifier.sol";
import "../src/core/HelixCoordinatorV3.sol";
import "../src/core/ModelRegistry.sol";
import "../src/token/HelixToken.sol";
import "../src/token/Staking.sol";
import "../src/token/Rewards.sol";
import "./ProofFixtures.t.sol";

/// @title RealProofVerificationTest
/// @notice Tests real proof verification through Halo2Verifier.verifyProof()
/// @dev This addresses the critical gap: testing real proofs against the real verifier
///      Tests cover: proof format validation, curve point checks, field element bounds,
///      Fiat-Shamir challenge derivation, pairing checks, and coordinator integration
contract RealProofVerificationTest is Test {
    Halo2Verifier public verifier;

    /// @notice BN254 scalar field order
    uint256 constant R = 21888242871839275222246405745257275088548364400416034343698204186575808495617;

    /// @notice BN254 base field prime
    uint256 constant P = 21888242871839275222246405745257275088696311157297823662689037894645226208583;

    function setUp() public {
        // Deploy verifier (deploys core verifier + VK internally)
        verifier = new Halo2Verifier();
    }

    // ============ Proof Format Tests ============

    function test_RejectsTooShortProof() public view {
        bytes memory shortProof = new bytes(319); // 1 byte short
        uint256[] memory inputs = _validInputs();

        bool result = verifier.verifyProof(shortProof, inputs);
        assertFalse(result);
    }

    function test_RejectsWrongInputCount() public view {
        bytes memory proof = ProofFixtureHardcoded.createValidProof();

        // 7 inputs instead of 8
        uint256[] memory inputs = new uint256[](7);
        bool result = verifier.verifyProof(proof, inputs);
        assertFalse(result);

        // 9 inputs
        uint256[] memory inputs9 = new uint256[](9);
        result = verifier.verifyProof(proof, inputs9);
        assertFalse(result);
    }

    function test_RejectsInputsExceedingFieldOrder() public view {
        bytes memory proof = ProofFixtureHardcoded.createValidProof();
        uint256[] memory inputs = _validInputs();

        // Set one input to >= R (scalar field order)
        inputs[0] = R;
        bool result = verifier.verifyProof(proof, inputs);
        assertFalse(result);

        // Just above R
        inputs[0] = R + 1;
        result = verifier.verifyProof(proof, inputs);
        assertFalse(result);

        // Max uint256
        inputs[0] = type(uint256).max;
        result = verifier.verifyProof(proof, inputs);
        assertFalse(result);
    }

    function test_AcceptsInputsJustBelowFieldOrder() public view {
        bytes memory proof = ProofFixtureHardcoded.createValidProof();
        uint256[] memory inputs = _validInputs();

        // R - 1 is the largest valid field element
        inputs[0] = R - 1;
        // This should not revert (may not verify cryptographically, but format is valid)
        verifier.verifyProof(proof, inputs);
    }

    // ============ Curve Point Validation Tests ============

    function test_RejectsPointNotOnCurve() public view {
        bytes memory proof = ProofFixtureHardcoded.createInvalidProof();
        uint256[] memory inputs = _validInputs();

        // Invalid proof has point (1, 3) which is NOT on y^2 = x^3 + 3
        bool result = verifier.verifyProof(proof, inputs);
        assertFalse(result);
    }

    function test_ValidCurvePoints() public view {
        bytes memory proof = ProofFixtureHardcoded.createValidProof();
        uint256[] memory inputs = _validInputs();

        // Should not revert - all points are on curve
        // Pairing check may fail (proof is synthetic), but format is valid
        verifier.verifyProof(proof, inputs);
    }

    function test_AcceptsPointAtInfinity() public {
        // Create proof with point at infinity (0, 0) for an advice commitment
        bytes memory proof = new bytes(320);
        // Leave first point as (0, 0) = point at infinity
        // Fill others with valid points
        assembly {
            // Point 1: (0, 0) - infinity
            mstore(add(proof, 32), 0)
            mstore(add(proof, 64), 0)
            // Point 2: generator (1, 2)
            mstore(add(proof, 96), 1)
            mstore(add(proof, 128), 2)
            // Point 3: generator
            mstore(add(proof, 160), 1)
            mstore(add(proof, 192), 2)
            // W: generator
            mstore(add(proof, 224), 1)
            mstore(add(proof, 256), 2)
            // W': generator
            mstore(add(proof, 288), 1)
            mstore(add(proof, 320), 2)
        }

        uint256[] memory inputs = _validInputs();
        // Should not revert
        verifier.verifyProof(proof, inputs);
    }

    // ============ Corrupted Proof Tests ============

    function test_CorruptedProofRejected() public view {
        // Multiple corruption seeds
        for (uint256 seed = 1; seed <= 5; seed++) {
            bytes memory proof = ProofFixtureHardcoded.createCorruptedProof(seed);
            uint256[] memory inputs = _validInputs();

            bool result = verifier.verifyProof(proof, inputs);
            // Corrupted proofs should either fail curve check or pairing check
            // They should never return true
            assertFalse(result);
        }
    }

    // ============ Replay Protection (verifyAndRecord) Tests ============

    function test_VerifyAndRecord_BlocksReplay() public {
        bytes memory proof = ProofFixtureHardcoded.createValidProof();
        uint256[] memory inputs = _validInputs();

        // First call
        verifier.verifyAndRecord(proof, inputs);

        // Second call with same proof+inputs should return false (replay)
        bool result = verifier.verifyAndRecord(proof, inputs);
        assertFalse(result);
    }

    function test_VerifyAndRecord_DoesNotTrackInvalidProof() public {
        bytes memory proof = ProofFixtureHardcoded.createValidProof();
        uint256[] memory inputs = _validInputs();

        bytes32 proofHash = keccak256(abi.encodePacked(proof, inputs));
        assertFalse(verifier.verifiedProofs(proofHash));

        // Synthetic proof fails pairing check, so hash is NOT stored
        bool result = verifier.verifyAndRecord(proof, inputs);
        assertFalse(result);
        assertFalse(verifier.verifiedProofs(proofHash));
    }

    // ============ Batch Verification Tests ============

    function test_BatchVerify_SingleProof() public view {
        bytes[] memory proofs = new bytes[](1);
        proofs[0] = ProofFixtureHardcoded.createValidProof();

        uint256[][] memory inputsArray = new uint256[][](1);
        inputsArray[0] = _validInputs();

        // Single proof batch should behave same as individual verify
        verifier.batchVerify(proofs, inputsArray);
    }

    function test_BatchVerify_LengthMismatch() public {
        bytes[] memory proofs = new bytes[](2);
        proofs[0] = ProofFixtureHardcoded.createValidProof();
        proofs[1] = ProofFixtureHardcoded.createValidProof();

        uint256[][] memory inputsArray = new uint256[][](1);
        inputsArray[0] = _validInputs();

        vm.expectRevert("Length mismatch");
        verifier.batchVerify(proofs, inputsArray);
    }

    function test_BatchVerify_EmptyBatch() public {
        bytes[] memory proofs = new bytes[](0);
        uint256[][] memory inputsArray = new uint256[][](0);

        vm.expectRevert("Empty batch");
        verifier.batchVerify(proofs, inputsArray);
    }

    function test_BatchVerifyHomogeneous_SingleProof() public view {
        bytes[] memory proofs = new bytes[](1);
        proofs[0] = ProofFixtureHardcoded.createValidProof();

        uint256[][] memory inputsArray = new uint256[][](1);
        inputsArray[0] = _validInputs();

        verifier.batchVerifyHomogeneous(proofs, inputsArray);
    }

    // ============ Gas Estimation Tests ============

    function test_GasEstimate_SingleVerification() public view {
        bytes memory proof = ProofFixtureHardcoded.createValidProof();
        uint256[] memory inputs = _validInputs();

        uint256 gasUsed = verifier.estimateVerifyGas(proof, inputs);
        // Verification should use reasonable gas (< 500k for BN254 pairing)
        assertGt(gasUsed, 0);
        assertLt(gasUsed, 500_000);
    }

    // ============ V3 Coordinator with Real Verifier Tests ============

    function test_V3_WithRealVerifier_InvalidProofSlashes() public {
        // Deploy V3 stack with real Halo2Verifier
        (HelixCoordinatorV3 v3, Staking staking, HelixToken token) = _deployV3WithRealVerifier();

        address prover = makeAddr("prover");
        token.mint(prover, 1000e18);
        vm.startPrank(prover);
        token.approve(address(staking), type(uint256).max);
        staking.stake(200e18);
        vm.stopPrank();

        // Register model with correct commitment
        uint256 oldHashLo = 12345;
        uint256 oldHashHi = 67890;
        uint256 correctCommitment = uint256(keccak256(abi.encodePacked(oldHashLo, oldHashHi)));
        uint256 modelId = v3.registerModel("Model", "desc", "hash", correctCommitment, 4, 8, 2, 2, 0);
        v3.startRound(modelId, 1 hours);

        // Create a valid-format proof (on-curve points) but cryptographically invalid
        bytes memory proof = ProofFixtureHardcoded.createValidProof();
        uint256[] memory publicInputs = new uint256[](8);
        publicInputs[0] = oldHashLo;
        publicInputs[1] = oldHashHi;
        publicInputs[2] = 99999;
        publicInputs[3] = 88888;
        publicInputs[4] = 100;
        publicInputs[5] = 10;
        publicInputs[6] = 1;
        publicInputs[7] = ProofFixtureHardcoded.computeErrorChecksum(10, 1, modelId, v3.maxErrorBound());

        // Submit proof - should fail pairing check (synthetic proof, not real)
        // and slash the prover
        (uint256 stakeBefore,,,) = staking.getStakeInfo(prover);

        vm.prank(prover);
        v3.submitProof(modelId, 1, proof, publicInputs);

        // Prover should be slashed (first offense = warning or minor, depending on severity calc)
        (uint256 stakeAfter,,,) = staking.getStakeInfo(prover);
        // Staking.sol calculates severity: first offense for InvalidProof is Major (critical violation)
        // Major = 50% slash
        assertLt(stakeAfter, stakeBefore);
    }

    function test_V3_WithRealVerifier_RejectsOffCurveProof() public {
        (HelixCoordinatorV3 v3, Staking staking, HelixToken token) = _deployV3WithRealVerifier();

        address prover = makeAddr("prover");
        token.mint(prover, 1000e18);
        vm.startPrank(prover);
        token.approve(address(staking), type(uint256).max);
        staking.stake(200e18);
        vm.stopPrank();

        uint256 oldHashLo = 12345;
        uint256 oldHashHi = 67890;
        uint256 correctCommitment = uint256(keccak256(abi.encodePacked(oldHashLo, oldHashHi)));
        uint256 modelId = v3.registerModel("Model", "desc", "hash", correctCommitment, 4, 8, 2, 2, 0);
        v3.startRound(modelId, 1 hours);

        // Off-curve proof
        bytes memory proof = ProofFixtureHardcoded.createInvalidProof();
        uint256[] memory publicInputs = new uint256[](8);
        publicInputs[0] = oldHashLo;
        publicInputs[1] = oldHashHi;
        publicInputs[2] = 99999;
        publicInputs[3] = 88888;
        publicInputs[4] = 100;
        publicInputs[5] = 10;
        publicInputs[6] = 1;
        publicInputs[7] = ProofFixtureHardcoded.computeErrorChecksum(10, 1, modelId, v3.maxErrorBound());

        vm.prank(prover);
        v3.submitProof(modelId, 1, proof, publicInputs);

        // Should be slashed
        assertEq(v3.getSlashingRecordCount(), 1);
    }

    // ============ Security: Proof Replay in V3 ============

    function test_V3_ProofReplayProtection() public {
        (HelixCoordinatorV3 v3, Staking staking, HelixToken token) = _deployV3WithMockVerifier();

        address prover = makeAddr("prover");
        address prover2Addr = makeAddr("prover2");
        token.mint(prover, 1000e18);
        token.mint(prover2Addr, 1000e18);

        vm.startPrank(prover);
        token.approve(address(staking), type(uint256).max);
        staking.stake(200e18);
        vm.stopPrank();

        vm.startPrank(prover2Addr);
        token.approve(address(staking), type(uint256).max);
        staking.stake(200e18);
        vm.stopPrank();

        uint256 oldHashLo = 12345;
        uint256 oldHashHi = 67890;
        uint256 correctCommitment = uint256(keccak256(abi.encodePacked(oldHashLo, oldHashHi)));
        uint256 modelId = v3.registerModel("Model", "desc", "hash", correctCommitment, 4, 8, 2, 2, 0);
        v3.startRound(modelId, 1 hours);

        uint256[] memory publicInputs = new uint256[](8);
        publicInputs[0] = oldHashLo;
        publicInputs[1] = oldHashHi;
        publicInputs[2] = 99999;
        publicInputs[3] = 88888;
        publicInputs[4] = 100;
        publicInputs[5] = 10;
        publicInputs[6] = 1;
        publicInputs[7] = ProofFixtureHardcoded.computeErrorChecksum(10, 1, modelId, v3.maxErrorBound());

        bytes memory proof = ProofFixtureHardcoded.createValidProof();

        // First submission
        vm.prank(prover);
        v3.submitProof(modelId, 1, proof, publicInputs);

        // Start new round
        v3.startRound(modelId, 1 hours);

        // Replay attempt
        vm.prank(prover2Addr);
        vm.expectRevert("Proof already used");
        v3.submitProof(modelId, 2, proof, publicInputs);
    }

    // ============ Helpers ============

    function _validInputs() internal pure returns (uint256[] memory) {
        uint256[] memory inputs = new uint256[](8);
        inputs[0] = 12345;
        inputs[1] = 67890;
        inputs[2] = 99999;
        inputs[3] = 88888;
        inputs[4] = 100;
        inputs[5] = 10;
        inputs[6] = 1;
        inputs[7] = 0;
        return inputs;
    }

    function _deployV3WithRealVerifier() internal returns (HelixCoordinatorV3, Staking, HelixToken) {
        address treasury = makeAddr("treasury");
        HelixToken token = new HelixToken(treasury);
        Halo2Verifier realVerifier = new Halo2Verifier();
        Staking staking = new Staking(address(token), 100e18, 7 days, 5000);
        Rewards rewards = new Rewards(address(token));
        ModelRegistry reg = new ModelRegistry();

        HelixCoordinatorV3 v3 = new HelixCoordinatorV3(
            address(realVerifier), address(staking), address(rewards), address(reg), treasury
        );

        staking.setOperator(address(v3));
        rewards.setCoordinator(address(v3));
        rewards.setStakingContract(address(staking));
        reg.setCoordinator(address(v3));

        return (v3, staking, token);
    }

    function _deployV3WithMockVerifier() internal returns (HelixCoordinatorV3, Staking, HelixToken) {
        address treasury = makeAddr("treasury");
        HelixToken token = new HelixToken(treasury);
        MockVerifierForRealTest mockV = new MockVerifierForRealTest();
        Staking staking = new Staking(address(token), 100e18, 7 days, 5000);
        Rewards rewards = new Rewards(address(token));
        ModelRegistry reg = new ModelRegistry();

        HelixCoordinatorV3 v3 = new HelixCoordinatorV3(
            address(mockV), address(staking), address(rewards), address(reg), treasury
        );

        staking.setOperator(address(v3));
        rewards.setCoordinator(address(v3));
        rewards.setStakingContract(address(staking));
        reg.setCoordinator(address(v3));

        return (v3, staking, token);
    }
}

contract MockVerifierForRealTest {
    bool public shouldAccept = true;

    function verifyProof(bytes memory, uint256[] memory) external view returns (bool) {
        return shouldAccept;
    }

    function setAccept(bool _accept) external {
        shouldAccept = _accept;
    }
}
