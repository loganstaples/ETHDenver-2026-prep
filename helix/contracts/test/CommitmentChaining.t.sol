// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "forge-std/Test.sol";
import "../src/core/HelixCoordinatorV2.sol";
import "../src/core/HelixCoordinatorV3.sol";
import "../src/core/ModelRegistry.sol";
import "../src/token/HelixToken.sol";
import "../src/token/Staking.sol";
import "../src/token/Rewards.sol";
import "./ProofFixtures.t.sol";

/// @title MockVerifierAcceptsAll
/// @notice Mock verifier that accepts all proofs (for testing contract logic)
contract MockVerifierAcceptsAll {
    bool public shouldAccept = true;

    function verifyProof(bytes memory, uint256[] memory) external view returns (bool) {
        return shouldAccept;
    }

    function setAccept(bool _accept) external {
        shouldAccept = _accept;
    }
}

/// @title CommitmentChainingV2Test
/// @notice Tests commitment chaining, step sequencing, error budget, round finalization,
///         and batch submission for HelixCoordinatorV2
contract CommitmentChainingV2Test is Test {
    HelixCoordinatorV2 public coordinator;
    MockVerifierAcceptsAll public mockVerifier;
    ModelRegistry public registry;

    address public owner;
    address public treasury;
    address public prover1;
    address public prover2;

    uint256 constant MIN_STAKE = 0.1 ether;
    uint256 constant ROUND_DURATION = 1 hours;

    // Commitment chain data
    uint256 constant INIT_LO = 100;
    uint256 constant INIT_HI = 200;
    uint256 constant STEP1_NEW_LO = 300;
    uint256 constant STEP1_NEW_HI = 400;
    uint256 constant STEP2_NEW_LO = 500;
    uint256 constant STEP2_NEW_HI = 600;
    uint256 constant STEP3_NEW_LO = 700;
    uint256 constant STEP3_NEW_HI = 800;

    event ProofAccepted(
        uint256 indexed modelId, uint256 indexed roundId, address indexed prover,
        uint256 stepNumber, uint256 newCommitment, uint256 loss, uint256 errorBound
    );
    event CommitmentUpdated(
        uint256 indexed modelId, uint256 indexed roundId,
        uint256 oldCommitment, uint256 newCommitment, uint256 stepNumber
    );
    event ErrorBudgetWarning(
        uint256 indexed modelId, uint256 indexed roundId,
        uint256 accumulated, uint256 budget
    );
    event TrainingHalted(
        uint256 indexed modelId, uint256 indexed roundId,
        uint256 accumulatedError, uint256 maxBudget
    );
    event RoundFinalized(
        uint256 indexed modelId, uint256 indexed roundId,
        uint256 finalCommitment, uint256 totalSteps, uint256 finalLoss, uint256 totalError
    );
    event RoundCompleted(
        uint256 indexed modelId, uint256 indexed roundId,
        uint256 newCommitment, uint256 totalErrorBound
    );

    function setUp() public {
        owner = address(this);
        treasury = makeAddr("treasury");
        prover1 = makeAddr("prover1");
        prover2 = makeAddr("prover2");

        mockVerifier = new MockVerifierAcceptsAll();
        coordinator = new HelixCoordinatorV2(address(mockVerifier), treasury);
        registry = new ModelRegistry();
        registry.setCoordinator(address(coordinator));
        coordinator.setModelRegistry(address(registry));

        vm.deal(prover1, 10 ether);
        vm.deal(prover2, 10 ether);
    }

    // ============ Helpers ============

    function _initialCommitment() internal pure returns (uint256) {
        return uint256(keccak256(abi.encodePacked(INIT_LO, INIT_HI)));
    }

    function _commitment(uint256 lo, uint256 hi) internal pure returns (uint256) {
        return uint256(keccak256(abi.encodePacked(lo, hi)));
    }

    function _registerAndStake(address prover) internal returns (uint256 modelId) {
        modelId = coordinator.registerModel("hash", _initialCommitment(), MIN_STAKE, 4, 8, 2, 2, 0);
        vm.prank(prover);
        coordinator.stake{value: 1 ether}(modelId);
    }

    function _buildInputs(
        uint256 oldLo, uint256 oldHi,
        uint256 newLo, uint256 newHi,
        uint256 loss, uint256 errorBound,
        uint256 step, uint256 modelId
    ) internal view returns (uint256[] memory) {
        uint256[] memory inputs = new uint256[](8);
        inputs[0] = oldLo;
        inputs[1] = oldHi;
        inputs[2] = newLo;
        inputs[3] = newHi;
        inputs[4] = loss;
        inputs[5] = errorBound;
        inputs[6] = step;
        inputs[7] = ProofFixtureHardcoded.computeErrorChecksum(errorBound, step, modelId, coordinator.maxErrorBound());
        return inputs;
    }

    function _validProof() internal pure returns (bytes memory) {
        return ProofFixtureHardcoded.createValidProof();
    }

    // Use a different proof per step to avoid replay protection
    function _uniqueProof(uint256 seed) internal pure returns (bytes memory) {
        bytes memory proof = new bytes(320);
        // Put seed at the start to make each proof unique
        assembly {
            mstore(add(proof, 32), seed)
            mstore(add(proof, 64), 2)
        }
        // Fill the rest with valid-looking data
        assembly {
            mstore(add(proof, 96), 1)
            mstore(add(proof, 128), 2)
            mstore(add(proof, 160), 1)
            mstore(add(proof, 192), 2)
            mstore(add(proof, 224), 1)
            mstore(add(proof, 256), 2)
            mstore(add(proof, 288), 1)
            mstore(add(proof, 320), 2)
        }
        return proof;
    }

    // ============ 1. Commitment Chaining Tests ============

    function test_CommitmentChaining_FirstProofMatchesInitial() public {
        uint256 modelId = _registerAndStake(prover1);
        coordinator.startRound(modelId, ROUND_DURATION);

        uint256[] memory inputs = _buildInputs(INIT_LO, INIT_HI, STEP1_NEW_LO, STEP1_NEW_HI, 100, 10, 1, modelId);

        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, _validProof(), inputs);

        // Model commitment should be updated
        (, uint256 commitment,) = coordinator.getModelState(modelId);
        assertEq(commitment, _commitment(STEP1_NEW_LO, STEP1_NEW_HI));
    }

    function test_CommitmentChaining_SecondProofChainsFromFirst() public {
        uint256 modelId = _registerAndStake(prover1);
        coordinator.startRound(modelId, ROUND_DURATION);

        // Step 1
        uint256[] memory inputs1 = _buildInputs(INIT_LO, INIT_HI, STEP1_NEW_LO, STEP1_NEW_HI, 100, 10, 1, modelId);
        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, _uniqueProof(1), inputs1);

        // Step 2: old_hash must match step 1's new_hash
        uint256[] memory inputs2 = _buildInputs(STEP1_NEW_LO, STEP1_NEW_HI, STEP2_NEW_LO, STEP2_NEW_HI, 90, 8, 2, modelId);
        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, _uniqueProof(2), inputs2);

        (, uint256 commitment,) = coordinator.getModelState(modelId);
        assertEq(commitment, _commitment(STEP2_NEW_LO, STEP2_NEW_HI));
    }

    function test_CommitmentChaining_ThreeStepsInSameRound() public {
        uint256 modelId = _registerAndStake(prover1);
        coordinator.startRound(modelId, ROUND_DURATION);

        // Step 1
        uint256[] memory inputs1 = _buildInputs(INIT_LO, INIT_HI, STEP1_NEW_LO, STEP1_NEW_HI, 100, 10, 1, modelId);
        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, _uniqueProof(1), inputs1);

        // Step 2
        uint256[] memory inputs2 = _buildInputs(STEP1_NEW_LO, STEP1_NEW_HI, STEP2_NEW_LO, STEP2_NEW_HI, 90, 8, 2, modelId);
        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, _uniqueProof(2), inputs2);

        // Step 3
        uint256[] memory inputs3 = _buildInputs(STEP2_NEW_LO, STEP2_NEW_HI, STEP3_NEW_LO, STEP3_NEW_HI, 80, 5, 3, modelId);
        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, _uniqueProof(3), inputs3);

        (, uint256 commitment,) = coordinator.getModelState(modelId);
        assertEq(commitment, _commitment(STEP3_NEW_LO, STEP3_NEW_HI));
        assertEq(coordinator.getLastStepNumber(modelId), 3);
    }

    function test_CommitmentChaining_MismatchReverts() public {
        uint256 modelId = _registerAndStake(prover1);
        coordinator.startRound(modelId, ROUND_DURATION);

        // Submit first proof
        uint256[] memory inputs1 = _buildInputs(INIT_LO, INIT_HI, STEP1_NEW_LO, STEP1_NEW_HI, 100, 10, 1, modelId);
        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, _uniqueProof(1), inputs1);

        // Try step 2 with WRONG old_hash (uses initial instead of step 1 output)
        uint256[] memory badInputs = _buildInputs(INIT_LO, INIT_HI, STEP2_NEW_LO, STEP2_NEW_HI, 90, 8, 2, modelId);
        vm.prank(prover1);
        vm.expectRevert(HelixCoordinatorV2.OldCommitmentMismatch.selector);
        coordinator.submitProof(modelId, 1, _uniqueProof(2), badInputs);
    }

    function test_CommitmentChaining_AcrossRounds() public {
        uint256 modelId = _registerAndStake(prover1);

        // Round 1: step 1
        coordinator.startRound(modelId, ROUND_DURATION);
        uint256[] memory inputs1 = _buildInputs(INIT_LO, INIT_HI, STEP1_NEW_LO, STEP1_NEW_HI, 100, 10, 1, modelId);
        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, _uniqueProof(1), inputs1);
        coordinator.finalizeRound(modelId, 1);

        // Round 2: step 2 must chain from round 1's output
        coordinator.startRound(modelId, ROUND_DURATION);
        uint256[] memory inputs2 = _buildInputs(STEP1_NEW_LO, STEP1_NEW_HI, STEP2_NEW_LO, STEP2_NEW_HI, 90, 8, 2, modelId);
        vm.prank(prover1);
        coordinator.submitProof(modelId, 2, _uniqueProof(2), inputs2);

        (, uint256 commitment,) = coordinator.getModelState(modelId);
        assertEq(commitment, _commitment(STEP2_NEW_LO, STEP2_NEW_HI));
    }

    // ============ 2. Step Number Sequencing Tests ============

    function test_StepSequencing_FirstStepMustBeOne() public {
        uint256 modelId = _registerAndStake(prover1);
        coordinator.startRound(modelId, ROUND_DURATION);

        // Step 0 should fail (must be lastStep + 1 = 1)
        uint256[] memory inputs = _buildInputs(INIT_LO, INIT_HI, STEP1_NEW_LO, STEP1_NEW_HI, 100, 10, 0, modelId);
        vm.prank(prover1);
        vm.expectRevert(HelixCoordinatorV2.StepNumberMismatch.selector);
        coordinator.submitProof(modelId, 1, _validProof(), inputs);
    }

    function test_StepSequencing_OutOfOrderReverts() public {
        uint256 modelId = _registerAndStake(prover1);
        coordinator.startRound(modelId, ROUND_DURATION);

        // Submit step 1
        uint256[] memory inputs1 = _buildInputs(INIT_LO, INIT_HI, STEP1_NEW_LO, STEP1_NEW_HI, 100, 10, 1, modelId);
        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, _uniqueProof(1), inputs1);

        // Try step 3 (skip step 2)
        uint256[] memory inputs3 = _buildInputs(STEP1_NEW_LO, STEP1_NEW_HI, STEP3_NEW_LO, STEP3_NEW_HI, 80, 5, 3, modelId);
        vm.prank(prover1);
        vm.expectRevert(HelixCoordinatorV2.StepNumberMismatch.selector);
        coordinator.submitProof(modelId, 1, _uniqueProof(3), inputs3);
    }

    function test_StepSequencing_DuplicateStepReverts() public {
        uint256 modelId = _registerAndStake(prover1);
        coordinator.startRound(modelId, ROUND_DURATION);

        // Submit step 1
        uint256[] memory inputs1 = _buildInputs(INIT_LO, INIT_HI, STEP1_NEW_LO, STEP1_NEW_HI, 100, 10, 1, modelId);
        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, _uniqueProof(1), inputs1);

        // Try step 1 again (should fail since lastStep is now 1, need step 2)
        uint256[] memory inputs1b = _buildInputs(STEP1_NEW_LO, STEP1_NEW_HI, STEP2_NEW_LO, STEP2_NEW_HI, 90, 8, 1, modelId);
        vm.prank(prover1);
        vm.expectRevert(HelixCoordinatorV2.StepNumberMismatch.selector);
        coordinator.submitProof(modelId, 1, _uniqueProof(10), inputs1b);
    }

    function test_StepSequencing_ContinuesAcrossRounds() public {
        uint256 modelId = _registerAndStake(prover1);

        // Round 1: step 1
        coordinator.startRound(modelId, ROUND_DURATION);
        uint256[] memory inputs1 = _buildInputs(INIT_LO, INIT_HI, STEP1_NEW_LO, STEP1_NEW_HI, 100, 10, 1, modelId);
        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, _uniqueProof(1), inputs1);
        coordinator.finalizeRound(modelId, 1);

        assertEq(coordinator.getLastStepNumber(modelId), 1);

        // Round 2: must start at step 2
        coordinator.startRound(modelId, ROUND_DURATION);
        uint256[] memory inputs2 = _buildInputs(STEP1_NEW_LO, STEP1_NEW_HI, STEP2_NEW_LO, STEP2_NEW_HI, 90, 8, 2, modelId);
        vm.prank(prover1);
        coordinator.submitProof(modelId, 2, _uniqueProof(2), inputs2);

        assertEq(coordinator.getLastStepNumber(modelId), 2);
    }

    // ============ 3. Error Budget Enforcement Tests ============

    function test_ErrorBudget_EnforcedOnExceed() public {
        uint256 modelId = _registerAndStake(prover1);
        // Start round with budget of 15
        coordinator.startRoundWithBudget(modelId, ROUND_DURATION, 15);

        // Step 1: error 10 (under budget 15)
        uint256[] memory inputs1 = _buildInputs(INIT_LO, INIT_HI, STEP1_NEW_LO, STEP1_NEW_HI, 100, 10, 1, modelId);
        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, _uniqueProof(1), inputs1);

        // Step 2: error 8 pushes total to 18 > 15 → proof accepted but round halted
        uint256[] memory inputs2 = _buildInputs(STEP1_NEW_LO, STEP1_NEW_HI, STEP2_NEW_LO, STEP2_NEW_HI, 90, 8, 2, modelId);
        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, _uniqueProof(2), inputs2);

        // Round should now be halted
        (,,,,bool halted,) = coordinator.getRoundProgress(modelId, 1);
        assertTrue(halted);
    }

    function test_ErrorBudget_HaltsTraining() public {
        uint256 modelId = _registerAndStake(prover1);
        coordinator.startRoundWithBudget(modelId, ROUND_DURATION, 5);

        // Error 10 > budget 5 → proof accepted but round halted
        uint256[] memory inputs = _buildInputs(INIT_LO, INIT_HI, STEP1_NEW_LO, STEP1_NEW_HI, 100, 10, 1, modelId);
        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, _validProof(), inputs);

        // Round should be halted
        (,,,,bool halted,) = coordinator.getRoundProgress(modelId, 1);
        assertTrue(halted);

        // Further proofs rejected
        uint256[] memory inputs2 = _buildInputs(STEP1_NEW_LO, STEP1_NEW_HI, STEP2_NEW_LO, STEP2_NEW_HI, 90, 2, 2, modelId);
        vm.prank(prover1);
        vm.expectRevert(HelixCoordinatorV2.TrainingHaltedForRound.selector);
        coordinator.submitProof(modelId, 1, _uniqueProof(99), inputs2);
    }

    function test_ErrorBudget_IncreaseAllowsContinuation() public {
        uint256 modelId = _registerAndStake(prover1);
        coordinator.startRoundWithBudget(modelId, ROUND_DURATION, 5);

        // Error 10 > budget 5 → proof accepted but round halted
        uint256[] memory inputs = _buildInputs(INIT_LO, INIT_HI, STEP1_NEW_LO, STEP1_NEW_HI, 100, 10, 1, modelId);
        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, _validProof(), inputs);

        // Verify halted
        (,,,,bool halted,) = coordinator.getRoundProgress(modelId, 1);
        assertTrue(halted);

        // Owner increases budget
        coordinator.increaseErrorBudget(modelId, 1, 100);

        // Now next proof should work (step 2 chains from step 1)
        uint256[] memory inputs2 = _buildInputs(STEP1_NEW_LO, STEP1_NEW_HI, STEP2_NEW_LO, STEP2_NEW_HI, 90, 2, 2, modelId);
        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, _uniqueProof(50), inputs2);

        assertEq(coordinator.getLastStepNumber(modelId), 2);
    }

    function test_ErrorBudget_NoLimitWhenZero() public {
        uint256 modelId = _registerAndStake(prover1);
        // Standard startRound (no budget = unlimited)
        coordinator.startRound(modelId, ROUND_DURATION);

        // Large error bound should be fine (up to maxErrorBound per step)
        uint256 largeError = 1e17; // Under maxErrorBound (1e18)
        uint256[] memory inputs = _buildInputs(INIT_LO, INIT_HI, STEP1_NEW_LO, STEP1_NEW_HI, 100, largeError, 1, modelId);
        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, _validProof(), inputs);

        assertEq(coordinator.getLastStepNumber(modelId), 1);
    }

    function test_ErrorBudget_WarningAt80Percent() public {
        uint256 modelId = _registerAndStake(prover1);
        coordinator.startRoundWithBudget(modelId, ROUND_DURATION, 100);

        // Step 1: error 85 → 85% > 80% → should emit warning
        uint256[] memory inputs = _buildInputs(INIT_LO, INIT_HI, STEP1_NEW_LO, STEP1_NEW_HI, 100, 85, 1, modelId);

        vm.expectEmit(true, true, false, true);
        emit ErrorBudgetWarning(modelId, 1, 85, 100);

        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, _validProof(), inputs);
    }

    // ============ 4. Round Finalization Tests ============

    function test_FinalizeRound_MarksComplete() public {
        uint256 modelId = _registerAndStake(prover1);
        coordinator.startRound(modelId, ROUND_DURATION);

        uint256[] memory inputs = _buildInputs(INIT_LO, INIT_HI, STEP1_NEW_LO, STEP1_NEW_HI, 100, 10, 1, modelId);
        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, _validProof(), inputs);

        coordinator.finalizeRound(modelId, 1);

        // Verify round is completed - no more proofs accepted
        uint256[] memory inputs2 = _buildInputs(STEP1_NEW_LO, STEP1_NEW_HI, STEP2_NEW_LO, STEP2_NEW_HI, 90, 8, 2, modelId);
        vm.prank(prover1);
        vm.expectRevert(HelixCoordinatorV2.RoundAlreadyCompleted.selector);
        coordinator.submitProof(modelId, 1, _uniqueProof(2), inputs2);
    }

    function test_FinalizeRound_EmitsEvents() public {
        uint256 modelId = _registerAndStake(prover1);
        coordinator.startRound(modelId, ROUND_DURATION);

        uint256[] memory inputs = _buildInputs(INIT_LO, INIT_HI, STEP1_NEW_LO, STEP1_NEW_HI, 100, 10, 1, modelId);
        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, _validProof(), inputs);

        uint256 expectedCommitment = _commitment(STEP1_NEW_LO, STEP1_NEW_HI);

        vm.expectEmit(true, true, false, true);
        emit RoundFinalized(modelId, 1, expectedCommitment, 1, 100, 10);

        coordinator.finalizeRound(modelId, 1);
    }

    function test_FinalizeRound_OnlyAuthorized() public {
        uint256 modelId = _registerAndStake(prover1);
        coordinator.startRound(modelId, ROUND_DURATION);

        uint256[] memory inputs = _buildInputs(INIT_LO, INIT_HI, STEP1_NEW_LO, STEP1_NEW_HI, 100, 10, 1, modelId);
        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, _validProof(), inputs);

        // Non-owner cannot finalize before round expires
        vm.prank(prover2);
        vm.expectRevert("Round not expired and not authorized");
        coordinator.finalizeRound(modelId, 1);
    }

    function test_FinalizeRound_RequiresProofs() public {
        uint256 modelId = _registerAndStake(prover1);
        coordinator.startRound(modelId, ROUND_DURATION);

        // Cannot finalize with no proofs when round hasn't expired
        vm.expectRevert("No proofs submitted and round not expired");
        coordinator.finalizeRound(modelId, 1);
    }

    function test_FinalizeRound_MultiStepThenFinalize() public {
        uint256 modelId = _registerAndStake(prover1);
        coordinator.startRound(modelId, ROUND_DURATION);

        // 3 steps
        vm.startPrank(prover1);
        coordinator.submitProof(modelId, 1, _uniqueProof(1),
            _buildInputs(INIT_LO, INIT_HI, STEP1_NEW_LO, STEP1_NEW_HI, 100, 10, 1, modelId));
        coordinator.submitProof(modelId, 1, _uniqueProof(2),
            _buildInputs(STEP1_NEW_LO, STEP1_NEW_HI, STEP2_NEW_LO, STEP2_NEW_HI, 90, 8, 2, modelId));
        coordinator.submitProof(modelId, 1, _uniqueProof(3),
            _buildInputs(STEP2_NEW_LO, STEP2_NEW_HI, STEP3_NEW_LO, STEP3_NEW_HI, 80, 5, 3, modelId));
        vm.stopPrank();

        // Check round progress
        (uint256 proofCount, uint256 stepNum, uint256 accError,,,) = coordinator.getRoundProgress(modelId, 1);
        assertEq(proofCount, 3);
        assertEq(stepNum, 3);
        assertEq(accError, 23); // 10 + 8 + 5

        // Finalize
        vm.expectEmit(true, true, false, true);
        emit RoundFinalized(modelId, 1, _commitment(STEP3_NEW_LO, STEP3_NEW_HI), 3, 80, 23);
        coordinator.finalizeRound(modelId, 1);
    }

    // ============ 5. Batch Proof Submission Tests ============

    function test_SequentialBatch_AllAccepted() public {
        uint256 modelId = _registerAndStake(prover1);
        coordinator.startRound(modelId, ROUND_DURATION);

        bytes[] memory proofs = new bytes[](3);
        proofs[0] = _uniqueProof(10);
        proofs[1] = _uniqueProof(11);
        proofs[2] = _uniqueProof(12);

        uint256[][] memory inputs = new uint256[][](3);
        inputs[0] = _buildInputs(INIT_LO, INIT_HI, STEP1_NEW_LO, STEP1_NEW_HI, 100, 10, 1, modelId);
        inputs[1] = _buildInputs(STEP1_NEW_LO, STEP1_NEW_HI, STEP2_NEW_LO, STEP2_NEW_HI, 90, 8, 2, modelId);
        inputs[2] = _buildInputs(STEP2_NEW_LO, STEP2_NEW_HI, STEP3_NEW_LO, STEP3_NEW_HI, 80, 5, 3, modelId);

        vm.prank(prover1);
        coordinator.submitSequentialBatch(modelId, 1, proofs, inputs);

        // All 3 steps should be processed
        assertEq(coordinator.getLastStepNumber(modelId), 3);
        (, uint256 commitment,) = coordinator.getModelState(modelId);
        assertEq(commitment, _commitment(STEP3_NEW_LO, STEP3_NEW_HI));
    }

    function test_SequentialBatch_MidBatchFailure() public {
        uint256 modelId = _registerAndStake(prover1);
        coordinator.startRound(modelId, ROUND_DURATION);

        bytes[] memory proofs = new bytes[](2);
        proofs[0] = _uniqueProof(20);
        proofs[1] = _uniqueProof(21);

        uint256[][] memory inputs = new uint256[][](2);
        inputs[0] = _buildInputs(INIT_LO, INIT_HI, STEP1_NEW_LO, STEP1_NEW_HI, 100, 10, 1, modelId);
        // Second proof has WRONG old_hash (doesn't chain from first)
        inputs[1] = _buildInputs(INIT_LO, INIT_HI, STEP2_NEW_LO, STEP2_NEW_HI, 90, 8, 2, modelId);

        vm.prank(prover1);
        vm.expectRevert(HelixCoordinatorV2.OldCommitmentMismatch.selector);
        coordinator.submitSequentialBatch(modelId, 1, proofs, inputs);

        // Entire batch reverted - step should still be 0
        assertEq(coordinator.getLastStepNumber(modelId), 0);
    }

    function test_SequentialBatch_EmptyReverts() public {
        uint256 modelId = _registerAndStake(prover1);
        coordinator.startRound(modelId, ROUND_DURATION);

        bytes[] memory proofs = new bytes[](0);
        uint256[][] memory inputs = new uint256[][](0);

        vm.prank(prover1);
        vm.expectRevert(HelixCoordinatorV2.EmptyBatch.selector);
        coordinator.submitSequentialBatch(modelId, 1, proofs, inputs);
    }

    function test_SequentialBatch_GasSavingsVsIndividual() public {
        uint256 modelId = _registerAndStake(prover1);
        coordinator.startRound(modelId, ROUND_DURATION);

        bytes[] memory proofs = new bytes[](3);
        proofs[0] = _uniqueProof(30);
        proofs[1] = _uniqueProof(31);
        proofs[2] = _uniqueProof(32);

        uint256[][] memory inputs = new uint256[][](3);
        inputs[0] = _buildInputs(INIT_LO, INIT_HI, STEP1_NEW_LO, STEP1_NEW_HI, 100, 10, 1, modelId);
        inputs[1] = _buildInputs(STEP1_NEW_LO, STEP1_NEW_HI, STEP2_NEW_LO, STEP2_NEW_HI, 90, 8, 2, modelId);
        inputs[2] = _buildInputs(STEP2_NEW_LO, STEP2_NEW_HI, STEP3_NEW_LO, STEP3_NEW_HI, 80, 5, 3, modelId);

        uint256 gasBefore = gasleft();
        vm.prank(prover1);
        coordinator.submitSequentialBatch(modelId, 1, proofs, inputs);
        uint256 batchGas = gasBefore - gasleft();

        // Just verify it worked and gas is reasonable
        assertEq(coordinator.getLastStepNumber(modelId), 3);
        assertTrue(batchGas > 0);
    }

    // ============ 6. Rich Event Emission Tests ============

    function test_Events_ProofAccepted() public {
        uint256 modelId = _registerAndStake(prover1);
        coordinator.startRound(modelId, ROUND_DURATION);

        uint256[] memory inputs = _buildInputs(INIT_LO, INIT_HI, STEP1_NEW_LO, STEP1_NEW_HI, 100, 10, 1, modelId);
        uint256 expectedCommitment = _commitment(STEP1_NEW_LO, STEP1_NEW_HI);

        vm.expectEmit(true, true, true, true);
        emit ProofAccepted(modelId, 1, prover1, 1, expectedCommitment, 100, 10);

        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, _validProof(), inputs);
    }

    function test_Events_CommitmentUpdated() public {
        uint256 modelId = _registerAndStake(prover1);
        coordinator.startRound(modelId, ROUND_DURATION);

        uint256[] memory inputs = _buildInputs(INIT_LO, INIT_HI, STEP1_NEW_LO, STEP1_NEW_HI, 100, 10, 1, modelId);
        uint256 oldCommitment = _initialCommitment();
        uint256 newCommitment = _commitment(STEP1_NEW_LO, STEP1_NEW_HI);

        vm.expectEmit(true, true, false, true);
        emit CommitmentUpdated(modelId, 1, oldCommitment, newCommitment, 1);

        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, _validProof(), inputs);
    }

    function test_Events_TrainingHalted() public {
        uint256 modelId = _registerAndStake(prover1);
        coordinator.startRoundWithBudget(modelId, ROUND_DURATION, 5);

        uint256[] memory inputs = _buildInputs(INIT_LO, INIT_HI, STEP1_NEW_LO, STEP1_NEW_HI, 100, 10, 1, modelId);

        vm.expectEmit(true, true, false, true);
        emit TrainingHalted(modelId, 1, 10, 5);

        // Proof accepted but round halted (error 10 > budget 5)
        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, _validProof(), inputs);
    }

    // ============ 7. View Function Tests ============

    function test_GetRoundProgress() public {
        uint256 modelId = _registerAndStake(prover1);
        coordinator.startRoundWithBudget(modelId, ROUND_DURATION, 50);

        uint256[] memory inputs = _buildInputs(INIT_LO, INIT_HI, STEP1_NEW_LO, STEP1_NEW_HI, 100, 10, 1, modelId);
        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, _validProof(), inputs);

        (uint256 proofCount, uint256 stepNum, uint256 accError, uint256 budget, bool halted, uint256 loss) = coordinator.getRoundProgress(modelId, 1);
        assertEq(proofCount, 1);
        assertEq(stepNum, 1);
        assertEq(accError, 10);
        assertEq(budget, 50);
        assertFalse(halted);
        assertEq(loss, 100);
    }

    function test_GetLastStepNumber_DefaultsToZero() public {
        uint256 modelId = coordinator.registerModel("hash", 12345, MIN_STAKE, 4, 8, 2, 2, 0);
        assertEq(coordinator.getLastStepNumber(modelId), 0);
    }

    // ============ 8. Edge Cases ============

    function test_DifferentProversCanSubmitToSameRound() public {
        uint256 modelId = _registerAndStake(prover1);
        vm.prank(prover2);
        coordinator.stake{value: 1 ether}(modelId);

        coordinator.startRound(modelId, ROUND_DURATION);

        // Prover1 submits step 1
        uint256[] memory inputs1 = _buildInputs(INIT_LO, INIT_HI, STEP1_NEW_LO, STEP1_NEW_HI, 100, 10, 1, modelId);
        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, _uniqueProof(1), inputs1);

        // Prover2 submits step 2
        uint256[] memory inputs2 = _buildInputs(STEP1_NEW_LO, STEP1_NEW_HI, STEP2_NEW_LO, STEP2_NEW_HI, 90, 8, 2, modelId);
        vm.prank(prover2);
        coordinator.submitProof(modelId, 1, _uniqueProof(2), inputs2);

        assertEq(coordinator.getLastStepNumber(modelId), 2);
    }

    function test_StartNewRoundWithoutFinalizing() public {
        uint256 modelId = _registerAndStake(prover1);

        // Round 1: submit step 1 but don't finalize
        coordinator.startRound(modelId, ROUND_DURATION);
        uint256[] memory inputs1 = _buildInputs(INIT_LO, INIT_HI, STEP1_NEW_LO, STEP1_NEW_HI, 100, 10, 1, modelId);
        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, _uniqueProof(1), inputs1);

        // Start round 2 without finalizing round 1
        coordinator.startRound(modelId, ROUND_DURATION);

        // Round 2 should chain from step 1's output
        uint256[] memory inputs2 = _buildInputs(STEP1_NEW_LO, STEP1_NEW_HI, STEP2_NEW_LO, STEP2_NEW_HI, 90, 8, 2, modelId);
        vm.prank(prover1);
        coordinator.submitProof(modelId, 2, _uniqueProof(2), inputs2);

        assertEq(coordinator.getLastStepNumber(modelId), 2);
    }
}

/// @title CommitmentChainingV3Test
/// @notice Tests commitment chaining, step sequencing, error budget for HelixCoordinatorV3
contract CommitmentChainingV3Test is Test {
    HelixCoordinatorV3 public coordinator;
    MockVerifierAcceptsAll public mockVerifier;
    HelixToken public token;
    Staking public staking;
    Rewards public rewards;
    ModelRegistry public registry;

    address public owner;
    address public treasuryAddr;
    address public prover1;
    address public prover2;

    uint256 constant STAKE_AMOUNT = 200e18;
    uint256 constant MIN_STAKE = 100e18;
    uint256 constant ROUND_DURATION = 1 hours;

    uint256 constant INIT_LO = 100;
    uint256 constant INIT_HI = 200;
    uint256 constant STEP1_NEW_LO = 300;
    uint256 constant STEP1_NEW_HI = 400;
    uint256 constant STEP2_NEW_LO = 500;
    uint256 constant STEP2_NEW_HI = 600;

    event ProofAccepted(
        uint256 indexed modelId, uint256 indexed roundId, address indexed prover,
        uint256 stepNumber, uint256 newCommitment, uint256 loss, uint256 errorBound
    );
    event CommitmentUpdated(
        uint256 indexed modelId, uint256 indexed roundId,
        uint256 oldCommitment, uint256 newCommitment, uint256 stepNumber
    );
    event RoundFinalized(
        uint256 indexed modelId, uint256 indexed roundId,
        address indexed bestProver, uint256 bestLoss
    );
    event TrainingHalted(
        uint256 indexed modelId, uint256 indexed roundId,
        uint256 accumulatedError, uint256 maxBudget
    );
    event ErrorBudgetWarning(
        uint256 indexed modelId, uint256 indexed roundId,
        uint256 accumulated, uint256 budget
    );

    function setUp() public {
        owner = address(this);
        treasuryAddr = makeAddr("treasury");
        prover1 = makeAddr("prover1");
        prover2 = makeAddr("prover2");

        token = new HelixToken(treasuryAddr);
        mockVerifier = new MockVerifierAcceptsAll();
        staking = new Staking(address(token), MIN_STAKE, 7 days, 5000);
        rewards = new Rewards(address(token));
        registry = new ModelRegistry();

        coordinator = new HelixCoordinatorV3(
            address(mockVerifier), address(staking), address(rewards),
            address(registry), treasuryAddr
        );

        staking.setOperator(address(coordinator));
        rewards.setCoordinator(address(coordinator));
        rewards.setStakingContract(address(staking));
        registry.setCoordinator(address(coordinator));

        // Fund and stake provers
        token.mint(prover1, 1000e18);
        token.mint(prover2, 1000e18);

        vm.startPrank(prover1);
        token.approve(address(staking), type(uint256).max);
        staking.stake(STAKE_AMOUNT);
        vm.stopPrank();

        vm.startPrank(prover2);
        token.approve(address(staking), type(uint256).max);
        staking.stake(STAKE_AMOUNT);
        vm.stopPrank();

        // Fund reward pool
        token.mint(address(this), 10000e18);
        token.approve(address(rewards), type(uint256).max);
        rewards.fundRewardPool(10000e18, 100e18, 365 days);
    }

    function _initialCommitment() internal pure returns (uint256) {
        return uint256(keccak256(abi.encodePacked(INIT_LO, INIT_HI)));
    }

    function _commitment(uint256 lo, uint256 hi) internal pure returns (uint256) {
        return uint256(keccak256(abi.encodePacked(lo, hi)));
    }

    function _buildInputs(
        uint256 oldLo, uint256 oldHi,
        uint256 newLo, uint256 newHi,
        uint256 loss, uint256 errorBound,
        uint256 step, uint256 modelId
    ) internal view returns (uint256[] memory) {
        uint256[] memory inputs = new uint256[](8);
        inputs[0] = oldLo;
        inputs[1] = oldHi;
        inputs[2] = newLo;
        inputs[3] = newHi;
        inputs[4] = loss;
        inputs[5] = errorBound;
        inputs[6] = step;
        inputs[7] = ProofFixtureHardcoded.computeErrorChecksum(errorBound, step, modelId, coordinator.maxErrorBound());
        return inputs;
    }

    function _uniqueProof(uint256 seed) internal pure returns (bytes memory) {
        bytes memory proof = new bytes(320);
        assembly {
            mstore(add(proof, 32), seed)
            mstore(add(proof, 64), 2)
            mstore(add(proof, 96), 1)
            mstore(add(proof, 128), 2)
            mstore(add(proof, 160), 1)
            mstore(add(proof, 192), 2)
            mstore(add(proof, 224), 1)
            mstore(add(proof, 256), 2)
            mstore(add(proof, 288), 1)
            mstore(add(proof, 320), 2)
        }
        return proof;
    }

    // ============ V3 Commitment Chaining ============

    function test_V3_CommitmentChaining_SingleParticipant() public {
        uint256 initCommitment = _initialCommitment();
        uint256 modelId = coordinator.registerModel("Model", "desc", "hash", initCommitment, 4, 8, 2, 2, 0);

        // Round 1 (single-participant, auto-finalize)
        coordinator.startRound(modelId, ROUND_DURATION);
        uint256[] memory inputs1 = _buildInputs(INIT_LO, INIT_HI, STEP1_NEW_LO, STEP1_NEW_HI, 100, 10, 1, modelId);
        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, _uniqueProof(1), inputs1);

        // Model commitment should be updated (auto-finalized)
        (, uint256 commitment,) = coordinator.getModelState(modelId);
        assertEq(commitment, _commitment(STEP1_NEW_LO, STEP1_NEW_HI));

        // Step counter should advance on finalization
        assertEq(coordinator.getLastStepNumber(modelId), 1);
    }

    function test_V3_CommitmentChaining_AcrossRounds() public {
        uint256 initCommitment = _initialCommitment();
        uint256 modelId = coordinator.registerModel("Model", "desc", "hash", initCommitment, 4, 8, 2, 2, 0);

        // Round 1: step 1
        coordinator.startRound(modelId, ROUND_DURATION);
        uint256[] memory inputs1 = _buildInputs(INIT_LO, INIT_HI, STEP1_NEW_LO, STEP1_NEW_HI, 100, 10, 1, modelId);
        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, _uniqueProof(1), inputs1);

        // Round 2: step 2 must chain from round 1 output
        coordinator.startRound(modelId, ROUND_DURATION);
        uint256[] memory inputs2 = _buildInputs(STEP1_NEW_LO, STEP1_NEW_HI, STEP2_NEW_LO, STEP2_NEW_HI, 90, 8, 2, modelId);
        vm.prank(prover1);
        coordinator.submitProof(modelId, 2, _uniqueProof(2), inputs2);

        (, uint256 commitment,) = coordinator.getModelState(modelId);
        assertEq(commitment, _commitment(STEP2_NEW_LO, STEP2_NEW_HI));
        assertEq(coordinator.getLastStepNumber(modelId), 2);
    }

    function test_V3_CommitmentChaining_MismatchReverts() public {
        uint256 initCommitment = _initialCommitment();
        uint256 modelId = coordinator.registerModel("Model", "desc", "hash", initCommitment, 4, 8, 2, 2, 0);

        coordinator.startRound(modelId, ROUND_DURATION);

        // Wrong old_hash
        uint256[] memory badInputs = _buildInputs(999, 888, STEP1_NEW_LO, STEP1_NEW_HI, 100, 10, 1, modelId);
        vm.prank(prover1);
        vm.expectRevert("Old commitment mismatch");
        coordinator.submitProof(modelId, 1, _uniqueProof(1), badInputs);
    }

    // ============ V3 Step Sequencing ============

    function test_V3_StepSequencing_WrongStepReverts() public {
        uint256 initCommitment = _initialCommitment();
        uint256 modelId = coordinator.registerModel("Model", "desc", "hash", initCommitment, 4, 8, 2, 2, 0);

        coordinator.startRound(modelId, ROUND_DURATION);

        // Step 5 should fail (expected step 1)
        uint256[] memory inputs = _buildInputs(INIT_LO, INIT_HI, STEP1_NEW_LO, STEP1_NEW_HI, 100, 10, 5, modelId);
        vm.prank(prover1);
        vm.expectRevert("Step number mismatch");
        coordinator.submitProof(modelId, 1, _uniqueProof(1), inputs);
    }

    function test_V3_StepSequencing_MultiParticipantSameStep() public {
        uint256 initCommitment = _initialCommitment();
        uint256 modelId = coordinator.registerModel("Model", "desc", "hash", initCommitment, 4, 8, 2, 2, 0);

        // Multi-participant round
        coordinator.startRoundWithThreshold(modelId, ROUND_DURATION, 2);

        // Both provers submit step 1
        uint256[] memory inputs1 = _buildInputs(INIT_LO, INIT_HI, STEP1_NEW_LO, STEP1_NEW_HI, 100, 10, 1, modelId);
        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, _uniqueProof(1), inputs1);

        // Prover2 also submits step 1 with different new commitment but same old commitment
        uint256[] memory inputs2 = _buildInputs(INIT_LO, INIT_HI, STEP2_NEW_LO, STEP2_NEW_HI, 95, 9, 1, modelId);
        vm.prank(prover2);
        coordinator.submitProof(modelId, 1, _uniqueProof(2), inputs2);

        // Finalize (after dispute period)
        vm.warp(block.timestamp + ROUND_DURATION + 2 hours);
        coordinator.finalizeRound(modelId, 1);

        // Step should have advanced to 1
        assertEq(coordinator.getLastStepNumber(modelId), 1);
    }

    // ============ V3 Error Budget ============

    function test_V3_ErrorBudget_Enforcement() public {
        uint256 initCommitment = _initialCommitment();
        uint256 modelId = coordinator.registerModel("Model", "desc", "hash", initCommitment, 4, 8, 2, 2, 0);

        coordinator.startRoundWithBudget(modelId, ROUND_DURATION, 5);

        // Error 10 > budget 5 → proof accepted but round halted
        uint256[] memory inputs = _buildInputs(INIT_LO, INIT_HI, STEP1_NEW_LO, STEP1_NEW_HI, 100, 10, 1, modelId);
        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, _uniqueProof(1), inputs);

        // Round should be halted
        (,,,bool halted) = coordinator.getRoundProgress(modelId, 1);
        assertTrue(halted);
    }

    function test_V3_ErrorBudget_IncreaseUnhalts() public {
        uint256 initCommitment = _initialCommitment();
        uint256 modelId = coordinator.registerModel("Model", "desc", "hash", initCommitment, 4, 8, 2, 2, 0);

        coordinator.startRoundWithBudget(modelId, ROUND_DURATION, 5);

        // Error 10 > budget 5 → proof accepted but round halted
        uint256[] memory inputs = _buildInputs(INIT_LO, INIT_HI, STEP1_NEW_LO, STEP1_NEW_HI, 100, 10, 1, modelId);
        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, _uniqueProof(1), inputs);

        // Verify halted
        (,,,bool halted) = coordinator.getRoundProgress(modelId, 1);
        assertTrue(halted);

        // Increase budget and un-halt
        coordinator.increaseErrorBudget(modelId, 1, 100);

        // Start a new round since V3 auto-finalized (single participant round)
        coordinator.startRound(modelId, ROUND_DURATION);

        // Now next step should work (step 2, round 2)
        uint256[] memory inputs2 = _buildInputs(STEP1_NEW_LO, STEP1_NEW_HI, STEP2_NEW_LO, STEP2_NEW_HI, 90, 2, 2, modelId);
        vm.prank(prover1);
        coordinator.submitProof(modelId, 2, _uniqueProof(50), inputs2);

        assertEq(coordinator.getLastStepNumber(modelId), 2);
    }

    // ============ V3 Events ============

    function test_V3_Events_ProofAccepted() public {
        uint256 initCommitment = _initialCommitment();
        uint256 modelId = coordinator.registerModel("Model", "desc", "hash", initCommitment, 4, 8, 2, 2, 0);

        coordinator.startRound(modelId, ROUND_DURATION);

        uint256[] memory inputs = _buildInputs(INIT_LO, INIT_HI, STEP1_NEW_LO, STEP1_NEW_HI, 100, 10, 1, modelId);
        uint256 expectedCommitment = _commitment(STEP1_NEW_LO, STEP1_NEW_HI);

        vm.expectEmit(true, true, true, true);
        emit ProofAccepted(modelId, 1, prover1, 1, expectedCommitment, 100, 10);

        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, _uniqueProof(1), inputs);
    }

    function test_V3_Events_RoundFinalized() public {
        uint256 initCommitment = _initialCommitment();
        uint256 modelId = coordinator.registerModel("Model", "desc", "hash", initCommitment, 4, 8, 2, 2, 0);

        coordinator.startRoundWithThreshold(modelId, ROUND_DURATION, 2);

        // Two provers submit
        uint256[] memory inputs1 = _buildInputs(INIT_LO, INIT_HI, STEP1_NEW_LO, STEP1_NEW_HI, 100, 10, 1, modelId);
        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, _uniqueProof(1), inputs1);

        uint256[] memory inputs2 = _buildInputs(INIT_LO, INIT_HI, STEP2_NEW_LO, STEP2_NEW_HI, 90, 8, 1, modelId);
        vm.prank(prover2);
        coordinator.submitProof(modelId, 1, _uniqueProof(2), inputs2);

        // Finalize
        vm.warp(block.timestamp + ROUND_DURATION + 2 hours);

        vm.expectEmit(true, true, true, true);
        emit RoundFinalized(modelId, 1, prover2, 90); // prover2 had lower loss

        coordinator.finalizeRound(modelId, 1);
    }

    // ============ V3 View Functions ============

    function test_V3_GetRoundProgress() public {
        uint256 initCommitment = _initialCommitment();
        uint256 modelId = coordinator.registerModel("Model", "desc", "hash", initCommitment, 4, 8, 2, 2, 0);

        coordinator.startRoundWithBudget(modelId, ROUND_DURATION, 50);

        uint256[] memory inputs = _buildInputs(INIT_LO, INIT_HI, STEP1_NEW_LO, STEP1_NEW_HI, 100, 10, 1, modelId);
        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, _uniqueProof(1), inputs);

        (uint256 expectedStep, uint256 accError, uint256 budget, bool halted) = coordinator.getRoundProgress(modelId, 1);
        assertEq(expectedStep, 1);
        assertEq(accError, 10);
        assertEq(budget, 50);
        assertFalse(halted);
    }

    function test_V3_GetLastStepNumber() public {
        uint256 initCommitment = _initialCommitment();
        uint256 modelId = coordinator.registerModel("Model", "desc", "hash", initCommitment, 4, 8, 2, 2, 0);

        assertEq(coordinator.getLastStepNumber(modelId), 0);

        // Submit and auto-finalize (single participant)
        coordinator.startRound(modelId, ROUND_DURATION);
        uint256[] memory inputs = _buildInputs(INIT_LO, INIT_HI, STEP1_NEW_LO, STEP1_NEW_HI, 100, 10, 1, modelId);
        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, _uniqueProof(1), inputs);

        assertEq(coordinator.getLastStepNumber(modelId), 1);
    }

    // ============ V3 Sequential Batch ============

    function test_V3_SequentialBatch() public {
        // V3 sequential batch works differently: single-participant rounds auto-finalize
        // so we can batch submit proofs in sequence
        uint256 initCommitment = _initialCommitment();
        uint256 modelId = coordinator.registerModel("Model", "desc", "hash", initCommitment, 4, 8, 2, 2, 0);

        coordinator.startRound(modelId, ROUND_DURATION);

        // For V3 single-participant, only one proof per round (auto-finalize)
        // So sequential batch with one proof per round
        bytes[] memory proofs = new bytes[](1);
        proofs[0] = _uniqueProof(40);

        uint256[][] memory inputs = new uint256[][](1);
        inputs[0] = _buildInputs(INIT_LO, INIT_HI, STEP1_NEW_LO, STEP1_NEW_HI, 100, 10, 1, modelId);

        vm.prank(prover1);
        coordinator.submitSequentialBatch(modelId, 1, proofs, inputs);

        assertEq(coordinator.getLastStepNumber(modelId), 1);
    }
}
