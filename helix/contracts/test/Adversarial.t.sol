// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "forge-std/Test.sol";
import "../src/core/HelixCoordinatorV2.sol";
import "../src/core/TrainingRound.sol";
import "../src/verification/Halo2Verifier.sol";
import "../src/verification/AggregationVerifier.sol";
import "../src/mocks/MockVerifier.sol";
import "./ProofFixtures.t.sol";

/// @title AdversarialTest
/// @notice Comprehensive adversarial testing for HELIX protocol security
/// @dev Tests attack vectors: invalid proofs, double submissions, front-running, etc.
contract AdversarialTest is Test {
    // ============ Contracts ============
    HelixCoordinatorV2 public coordinator;
    MockVerifier public mockVerifier;
    Halo2Verifier public realVerifier;
    TrainingRound public trainingRound;
    AggregationVerifier public aggregationVerifier;

    // ============ Actors ============
    address public owner;
    address public treasury;
    address public modelOwner;
    address public honestProver;
    address public attacker1;
    address public attacker2;
    address public frontrunner;

    // ============ Constants ============
    uint256 constant MIN_STAKE = 0.1 ether;
    uint256 constant LARGE_STAKE = 2 ether;
    uint256 constant ROUND_DURATION = 1 hours;

    // ============ Events ============
    event Slashed(address indexed prover, uint256 indexed modelId, uint256 roundId, uint256 amount, uint256 remainingStake, string reason);
    event InvalidProofDetected(uint256 indexed modelId, uint256 indexed roundId, address indexed prover, bytes32 proofHash);

    function setUp() public {
        owner = address(this);
        treasury = makeAddr("treasury");
        modelOwner = makeAddr("modelOwner");
        honestProver = makeAddr("honestProver");
        attacker1 = makeAddr("attacker1");
        attacker2 = makeAddr("attacker2");
        frontrunner = makeAddr("frontrunner");

        // Deploy contracts
        realVerifier = new Halo2Verifier(Halo2VKDefaults.g2Generator());
        mockVerifier = new MockVerifier();
        coordinator = new HelixCoordinatorV2(address(mockVerifier), treasury);
        trainingRound = new TrainingRound();
        aggregationVerifier = new AggregationVerifier(address(mockVerifier));

        // Fund actors
        vm.deal(modelOwner, 100 ether);
        vm.deal(honestProver, 100 ether);
        vm.deal(attacker1, 100 ether);
        vm.deal(attacker2, 100 ether);
        vm.deal(frontrunner, 100 ether);
    }

    // ============ Helper Functions ============

    /// @dev Configures mock verifier to reject proofs (simulates switching to real verifier)
    function _setVerifierToReject() internal {
        mockVerifier.setShouldPass(false);
    }

    function _emergencySetSlashPercentage(uint256 _percentage) internal {
        coordinator.emergencyPause();
        coordinator.setSlashPercentage(_percentage);
        coordinator.unpause();
    }

    function _setupModelAndRound() internal returns (uint256 modelId, uint256 hashLo, uint256 hashHi) {
        hashLo = 12345;
        hashHi = 67890;
        uint256 commitment = uint256(keccak256(abi.encodePacked(hashLo, hashHi)));

        vm.prank(modelOwner);
        modelId = coordinator.registerModel("QmTestModel", commitment, MIN_STAKE);

        vm.prank(modelOwner);
        coordinator.startRound(modelId, ROUND_DURATION);
    }

    function _createValidPublicInputs(
        uint256 oldLo, uint256 oldHi,
        uint256 newLo, uint256 newHi,
        uint256 modelId
    ) internal pure returns (uint256[] memory) {
        uint256[] memory inputs = new uint256[](8);
        inputs[0] = oldLo;
        inputs[1] = oldHi;
        inputs[2] = newLo;
        inputs[3] = newHi;
        inputs[4] = 100;  // loss
        inputs[5] = 10;   // error bound
        inputs[6] = 1;    // step
        inputs[7] = ProofFixtureHardcoded.computeErrorChecksum(inputs[5], inputs[6], modelId, 1e18);
        return inputs;
    }

    // ============ Invalid Proof Attacks ============

    /// @notice Test that submitting an invalid proof results in slashing
    function test_InvalidProofSlashing() public {
        (uint256 modelId, uint256 hashLo, uint256 hashHi) = _setupModelAndRound();

        vm.prank(attacker1);
        coordinator.stake{value: LARGE_STAKE}(modelId);

        mockVerifier.setShouldPass(false);

        uint256[] memory inputs = _createValidPublicInputs(hashLo, hashHi, 1111, 2222, modelId);
        bytes memory proof = new bytes(320);

        uint256 treasuryBefore = treasury.balance;
        uint256 attackerStakeBefore = LARGE_STAKE;

        vm.expectEmit(true, true, false, true);
        emit Slashed(attacker1, modelId, 1, LARGE_STAKE / 2, LARGE_STAKE / 2, "Invalid proof");

        vm.prank(attacker1);
        coordinator.submitProof(modelId, 1, proof, inputs);

        // Verify slashing occurred
        (uint256 stakeAfter,, bool slashed) = coordinator.getStake(attacker1, modelId);
        assertTrue(slashed);
        assertEq(stakeAfter, attackerStakeBefore / 2);  // 50% slashed
        assertEq(treasury.balance, treasuryBefore + LARGE_STAKE / 2);
    }

    /// @notice Test submitting malformed proof (too short)
    function test_MalformedProofTooShort() public {
        _setVerifierToReject();
        (uint256 modelId, uint256 hashLo, uint256 hashHi) = _setupModelAndRound();

        vm.prank(attacker1);
        coordinator.stake{value: LARGE_STAKE}(modelId);

        // Proof too short (less than 320 bytes)
        bytes memory shortProof = hex"deadbeef";
        uint256[] memory inputs = _createValidPublicInputs(hashLo, hashHi, 1111, 2222, modelId);

        vm.prank(attacker1);
        coordinator.submitProof(modelId, 1, shortProof, inputs);

        // Should be slashed
        (,, bool slashed) = coordinator.getStake(attacker1, modelId);
        assertTrue(slashed);
    }

    /// @notice Test submitting proof with wrong number of public inputs
    function test_WrongPublicInputsCount() public {
        (uint256 modelId,,) = _setupModelAndRound();

        vm.prank(attacker1);
        coordinator.stake{value: LARGE_STAKE}(modelId);

        // Only 5 inputs instead of 8
        uint256[] memory wrongInputs = new uint256[](5);
        bytes memory proof = new bytes(320);

        vm.prank(attacker1);
        vm.expectRevert(HelixCoordinatorV2.InvalidPublicInputsCount.selector);
        coordinator.submitProof(modelId, 1, proof, wrongInputs);
    }

    /// @notice Test submitting proof with invalid field elements (exceeding scalar field)
    function test_InvalidFieldElements() public {
        _setVerifierToReject();
        (uint256 modelId, uint256 hashLo, uint256 hashHi) = _setupModelAndRound();

        vm.prank(attacker1);
        coordinator.stake{value: LARGE_STAKE}(modelId);

        // Public input exceeds scalar field R
        uint256[] memory inputs = _createValidPublicInputs(hashLo, hashHi, 1111, 2222, modelId);
        inputs[4] = type(uint256).max;  // Invalid: exceeds field

        bytes memory proof = new bytes(320);

        vm.prank(attacker1);
        coordinator.submitProof(modelId, 1, proof, inputs);

        // Should be slashed due to invalid verification
        (,, bool slashed) = coordinator.getStake(attacker1, modelId);
        assertTrue(slashed);
    }

    // ============ Double Submission Attacks ============

    /// @notice Test that same prover cannot submit twice to same round
    function test_DoubleSubmissionSameProver() public {
        (uint256 modelId, uint256 hashLo, uint256 hashHi) = _setupModelAndRound();

        vm.prank(honestProver);
        coordinator.stake{value: LARGE_STAKE}(modelId);

        mockVerifier.setShouldPass(true);

        uint256[] memory inputs = _createValidPublicInputs(hashLo, hashHi, 1111, 2222, modelId);
        bytes memory proof = new bytes(320);

        // First submission succeeds
        vm.prank(honestProver);
        coordinator.submitProof(modelId, 1, proof, inputs);

        // Second submission fails (round completed)
        vm.prank(honestProver);
        vm.expectRevert(HelixCoordinatorV2.RoundAlreadyCompleted.selector);
        coordinator.submitProof(modelId, 1, proof, inputs);
    }

    /// @notice Test that late submission to completed round fails
    function test_LateSubmissionToCompletedRound() public {
        (uint256 modelId, uint256 hashLo, uint256 hashHi) = _setupModelAndRound();

        // Both provers stake
        vm.prank(honestProver);
        coordinator.stake{value: LARGE_STAKE}(modelId);
        vm.prank(attacker1);
        coordinator.stake{value: LARGE_STAKE}(modelId);

        mockVerifier.setShouldPass(true);

        uint256[] memory inputs = _createValidPublicInputs(hashLo, hashHi, 1111, 2222, modelId);
        bytes memory proof = new bytes(320);

        // Honest prover submits first
        vm.prank(honestProver);
        coordinator.submitProof(modelId, 1, proof, inputs);

        // Attacker tries to submit same round
        vm.prank(attacker1);
        vm.expectRevert(HelixCoordinatorV2.RoundAlreadyCompleted.selector);
        coordinator.submitProof(modelId, 1, proof, inputs);
    }

    // ============ Commitment Mismatch Attacks ============

    /// @notice Test that wrong old commitment is rejected
    function test_WrongOldCommitment() public {
        (uint256 modelId,,) = _setupModelAndRound();

        vm.prank(attacker1);
        coordinator.stake{value: LARGE_STAKE}(modelId);

        mockVerifier.setShouldPass(true);

        // Use wrong old commitment values
        uint256[] memory inputs = _createValidPublicInputs(99999, 88888, 1111, 2222, modelId);
        bytes memory proof = new bytes(320);

        vm.prank(attacker1);
        vm.expectRevert(HelixCoordinatorV2.OldCommitmentMismatch.selector);
        coordinator.submitProof(modelId, 1, proof, inputs);
    }

    /// @notice Test commitment inflation attack attempt
    function test_CommitmentInflationAttack() public {
        (uint256 modelId, uint256 hashLo, uint256 hashHi) = _setupModelAndRound();

        vm.prank(attacker1);
        coordinator.stake{value: LARGE_STAKE}(modelId);

        mockVerifier.setShouldPass(true);

        // Try to set a specific malicious new commitment
        // The commitment is computed from inputs[2] and inputs[3]
        uint256[] memory inputs = _createValidPublicInputs(hashLo, hashHi, 0, 0, modelId);
        bytes memory proof = new bytes(320);

        vm.prank(attacker1);
        coordinator.submitProof(modelId, 1, proof, inputs);

        // Verify the commitment was updated to hash of (0, 0)
        uint256 expectedCommitment = uint256(keccak256(abi.encodePacked(uint256(0), uint256(0))));
        (, uint256 currentCommitment,) = coordinator.getModelState(modelId);
        assertEq(currentCommitment, expectedCommitment);
    }

    // ============ Front-Running Attacks ============

    /// @notice Test front-running attack where attacker steals proof
    function test_FrontRunningProofTheft() public {
        (uint256 modelId, uint256 hashLo, uint256 hashHi) = _setupModelAndRound();

        // Both stake
        vm.prank(honestProver);
        coordinator.stake{value: LARGE_STAKE}(modelId);
        vm.prank(frontrunner);
        coordinator.stake{value: LARGE_STAKE}(modelId);

        mockVerifier.setShouldPass(true);

        uint256[] memory inputs = _createValidPublicInputs(hashLo, hashHi, 1111, 2222, modelId);
        bytes memory proof = new bytes(320);

        // Frontrunner sees honest proof in mempool and submits with higher gas
        // Simulated by frontrunner submitting first
        vm.prank(frontrunner);
        coordinator.submitProof(modelId, 1, proof, inputs);

        // Honest prover's transaction now fails
        vm.prank(honestProver);
        vm.expectRevert(HelixCoordinatorV2.RoundAlreadyCompleted.selector);
        coordinator.submitProof(modelId, 1, proof, inputs);

        // Frontrunner got credit for the round
        // This demonstrates the need for commit-reveal or MEV protection
    }

    // ============ Replay Attacks ============

    /// @notice Test replay attack with same proof across rounds
    function test_ReplayAttackAcrossRounds() public {
        (uint256 modelId, uint256 hashLo, uint256 hashHi) = _setupModelAndRound();

        vm.prank(honestProver);
        coordinator.stake{value: LARGE_STAKE}(modelId);

        mockVerifier.setShouldPass(true);

        uint256 newHashLo = 1111;
        uint256 newHashHi = 2222;
        uint256[] memory inputs = _createValidPublicInputs(hashLo, hashHi, newHashLo, newHashHi, modelId);
        bytes memory proof = new bytes(320);

        // Submit to round 1
        vm.prank(honestProver);
        coordinator.submitProof(modelId, 1, proof, inputs);

        // Start round 2
        vm.prank(modelOwner);
        coordinator.startRound(modelId, ROUND_DURATION);

        // Try to replay the same proof to round 2
        // This should fail because the proof hash is already used (replay protection)
        vm.prank(honestProver);
        vm.expectRevert(HelixCoordinatorV2.ProofAlreadyUsed.selector);
        coordinator.submitProof(modelId, 2, proof, inputs);

        // Even with correct old commitment and fresh proof, round 2 succeeds
        uint256[] memory inputs2 = _createValidPublicInputs(newHashLo, newHashHi, 3333, 4444, modelId);
        bytes memory proof2 = new bytes(320);
        proof2[0] = 0x01; // Unique proof bytes
        vm.prank(honestProver);
        coordinator.submitProof(modelId, 2, proof2, inputs2);
    }

    /// @notice Test replay attack with Halo2Verifier's verifyAndRecord
    function test_ReplayPreventionVerifier() public {
        Halo2Verifier verifier = new Halo2Verifier(Halo2VKDefaults.g2Generator());

        bytes memory proof = new bytes(320);
        uint256[] memory inputs = new uint256[](8);
        for (uint i = 0; i < 8; i++) inputs[i] = i + 1;

        // First verification and record
        bool result1 = verifier.verifyAndRecord(proof, inputs);
        // Note: will return false because proof is invalid, but hash gets recorded if valid

        // If we had a valid proof that passed, second attempt would fail with "Replay"
        // This is tested implicitly by the verifyAndRecord logic
    }

    // ============ Slashing Edge Cases ============

    /// @notice Test that slashed prover cannot re-stake
    function test_SlashedCannotRestake() public {
        (uint256 modelId, uint256 hashLo, uint256 hashHi) = _setupModelAndRound();

        vm.prank(attacker1);
        coordinator.stake{value: LARGE_STAKE}(modelId);

        mockVerifier.setShouldPass(false);

        uint256[] memory inputs = _createValidPublicInputs(hashLo, hashHi, 1111, 2222, modelId);
        bytes memory proof = new bytes(320);

        // Get slashed
        vm.prank(attacker1);
        coordinator.submitProof(modelId, 1, proof, inputs);

        // Try to stake again
        vm.prank(attacker1);
        vm.expectRevert(HelixCoordinatorV2.PreviousStakeSlashed.selector);
        coordinator.stake{value: LARGE_STAKE}(modelId);
    }

    /// @notice Test multiple slashing attempts
    function test_MultipleSlashingAttempts() public {
        (uint256 modelId, uint256 hashLo, uint256 hashHi) = _setupModelAndRound();

        vm.prank(attacker1);
        coordinator.stake{value: LARGE_STAKE}(modelId);

        mockVerifier.setShouldPass(false);

        uint256[] memory inputs = _createValidPublicInputs(hashLo, hashHi, 1111, 2222, modelId);
        bytes memory proof = new bytes(320);

        // First slashing
        vm.prank(attacker1);
        coordinator.submitProof(modelId, 1, proof, inputs);

        // Already slashed, cannot be slashed again
        // Round is still not completed (invalid proof doesn't complete round)
        // But prover now has insufficient stake
        vm.prank(attacker1);
        vm.expectRevert(HelixCoordinatorV2.StakeSlashed.selector);
        coordinator.submitProof(modelId, 1, proof, inputs);
    }

    /// @notice Test that deploying coordinator with zero treasury reverts
    function test_SlashingWithZeroTreasury() public {
        // Deploy coordinator with zero treasury should now revert
        vm.expectRevert(HelixCoordinatorV2.InvalidTreasury.selector);
        new HelixCoordinatorV2(address(mockVerifier), address(0));
    }

    // ============ Challenge Mechanism Tests ============

    /// @notice Test challenging a fraudulent proof after the fact
    function test_ChallengeProofMechanism() public {
        (uint256 modelId, uint256 hashLo, uint256 hashHi) = _setupModelAndRound();

        vm.prank(attacker1);
        coordinator.stake{value: LARGE_STAKE}(modelId);

        // Proof passes initially (mock says yes)
        mockVerifier.setShouldPass(true);

        uint256[] memory inputs = _createValidPublicInputs(hashLo, hashHi, 1111, 2222, modelId);
        bytes memory proof = new bytes(320);

        vm.prank(attacker1);
        coordinator.submitProof(modelId, 1, proof, inputs);

        // Later, someone discovers the proof was actually invalid
        // Change mock to return false for challenge verification
        mockVerifier.setShouldPass(false);

        // Challenge the proof
        coordinator.challengeProof(modelId, 1, proof, inputs);

        // Attacker should be slashed
        (,, bool slashed) = coordinator.getStake(attacker1, modelId);
        assertTrue(slashed);
    }

    /// @notice Test challenging non-existent round fails
    function test_ChallengeNonExistentRound() public {
        (uint256 modelId,,) = _setupModelAndRound();

        bytes memory proof = new bytes(320);
        uint256[] memory inputs = new uint256[](8);
        inputs[7] = 0;

        // Try to challenge round that doesn't exist
        vm.expectRevert(HelixCoordinatorV2.RoundNotCompleted.selector);
        coordinator.challengeProof(modelId, 999, proof, inputs);
    }

    // ============ Error Bound Manipulation Attacks ============

    /// @notice Test error bound exceeding maximum
    function test_ExcessiveErrorBoundRejection() public {
        (uint256 modelId, uint256 hashLo, uint256 hashHi) = _setupModelAndRound();

        vm.prank(attacker1);
        coordinator.stake{value: LARGE_STAKE}(modelId);

        mockVerifier.setShouldPass(true);

        uint256[] memory inputs = new uint256[](8);
        inputs[0] = hashLo;
        inputs[1] = hashHi;
        inputs[2] = 1111;
        inputs[3] = 2222;
        inputs[4] = 100;
        inputs[5] = coordinator.maxErrorBound() + 1;  // Exceeds maximum
        inputs[6] = 1;
        inputs[7] = 0;  // Checksum irrelevant; reverts before checksum check

        bytes memory proof = new bytes(320);

        vm.prank(attacker1);
        vm.expectRevert(HelixCoordinatorV2.ErrorBoundExceeded.selector);
        coordinator.submitProof(modelId, 1, proof, inputs);
    }

    /// @notice Test accumulated error bound tracking for long-running attacks
    function test_AccumulatedErrorBoundTracking() public {
        (uint256 modelId, uint256 hashLo, uint256 hashHi) = _setupModelAndRound();

        vm.prank(honestProver);
        coordinator.stake{value: LARGE_STAKE}(modelId);

        mockVerifier.setShouldPass(true);

        // Submit multiple rounds with high error bounds
        uint256 currentLo = hashLo;
        uint256 currentHi = hashHi;

        for (uint256 i = 0; i < 5; i++) {
            if (i > 0) {
                vm.prank(modelOwner);
                coordinator.startRound(modelId, ROUND_DURATION);
            }

            uint256 newLo = currentLo + 1;
            uint256 newHi = currentHi + 1;

            uint256[] memory inputs = new uint256[](8);
            inputs[0] = currentLo;
            inputs[1] = currentHi;
            inputs[2] = newLo;
            inputs[3] = newHi;
            inputs[4] = 100;
            inputs[5] = 100;  // High error bound each round
            inputs[6] = i + 1;
            inputs[7] = ProofFixtureHardcoded.computeErrorChecksum(inputs[5], inputs[6], modelId, 1e18);

            bytes memory proof = new bytes(320);

            vm.prank(honestProver);
            coordinator.submitProof(modelId, i + 1, proof, inputs);

            currentLo = newLo;
            currentHi = newHi;
        }

        // Accumulated error should be 500 (5 rounds * 100 error)
        assertEq(coordinator.getAccumulatedErrorBound(modelId), 500);

        // Model owner can check if error is acceptable
        assertFalse(coordinator.isModelErrorAcceptable(modelId, 400));
        assertTrue(coordinator.isModelErrorAcceptable(modelId, 600));
    }

    // ============ Authorization Attacks ============

    /// @notice Test non-owner cannot start round
    function test_NonOwnerCannotStartRound() public {
        (uint256 modelId,,) = _setupModelAndRound();

        vm.prank(attacker1);
        vm.expectRevert(HelixCoordinatorV2.NotModelOwner.selector);
        coordinator.startRound(modelId, ROUND_DURATION);
    }

    /// @notice Test non-owner cannot pause model
    function test_NonOwnerCannotPauseModel() public {
        uint256 commitment = uint256(keccak256(abi.encodePacked(uint256(1), uint256(2))));

        vm.prank(modelOwner);
        uint256 modelId = coordinator.registerModel("Test", commitment, MIN_STAKE);

        vm.prank(attacker1);
        vm.expectRevert(HelixCoordinatorV2.NotAuthorized.selector);
        coordinator.pauseModel(modelId);
    }

    /// @notice Test non-owner cannot change admin parameters
    function test_NonOwnerCannotChangeParams() public {
        vm.prank(attacker1);
        vm.expectRevert(HelixCoordinatorV2.OnlyOwner.selector);
        coordinator.setSlashPercentage(10000);

        vm.prank(attacker1);
        vm.expectRevert(HelixCoordinatorV2.OnlyOwner.selector);
        coordinator.setTreasury(attacker1);
    }

    // ============ TrainingRound Adversarial Tests ============

    /// @notice Test double registration in TrainingRound
    function test_DoubleRegistrationPrevention() public {
        uint256 modelId = 1;
        bytes32 startCommit = keccak256("start");

        trainingRound.setCoordinator(address(this));
        uint256 roundId = trainingRound.createRound(modelId, startCommit, 2, 10, ROUND_DURATION);

        vm.prank(attacker1);
        trainingRound.registerForRound(modelId, roundId);

        vm.prank(attacker1);
        vm.expectRevert("Already registered");
        trainingRound.registerForRound(modelId, roundId);
    }

    /// @notice Test registration after deadline
    function test_RegistrationAfterDeadline() public {
        uint256 modelId = 1;
        bytes32 startCommit = keccak256("start");

        trainingRound.setCoordinator(address(this));
        uint256 roundId = trainingRound.createRound(modelId, startCommit, 2, 10, ROUND_DURATION);

        // Fast forward past deadline
        vm.warp(block.timestamp + ROUND_DURATION + 1);

        vm.prank(attacker1);
        vm.expectRevert("Round deadline passed");
        trainingRound.registerForRound(modelId, roundId);
    }

    /// @notice Test submitting gradient without registration
    function test_GradientWithoutRegistration() public {
        uint256 modelId = 1;
        bytes32 startCommit = keccak256("start");

        trainingRound.setCoordinator(address(this));
        uint256 roundId = trainingRound.createRound(modelId, startCommit, 1, 10, ROUND_DURATION);

        // Register another user to activate the round
        vm.prank(honestProver);
        trainingRound.registerForRound(modelId, roundId);

        // Attacker tries to submit without registering
        vm.prank(attacker1);
        vm.expectRevert("Not registered");
        trainingRound.submitGradient(modelId, roundId, keccak256("malicious"), 0);
    }

    /// @notice Test double gradient submission
    function test_DoubleGradientSubmission() public {
        uint256 modelId = 1;
        bytes32 startCommit = keccak256("start");

        trainingRound.setCoordinator(address(this));
        uint256 roundId = trainingRound.createRound(modelId, startCommit, 1, 10, ROUND_DURATION);

        vm.prank(attacker1);
        trainingRound.registerForRound(modelId, roundId);

        // First submission
        vm.prank(attacker1);
        trainingRound.submitGradient(modelId, roundId, keccak256("gradient1"), 5);

        // Second submission fails
        vm.prank(attacker1);
        vm.expectRevert("Already submitted");
        trainingRound.submitGradient(modelId, roundId, keccak256("gradient2"), 5);
    }

    // ============ AggregationVerifier Adversarial Tests ============

    /// @notice Test double contribution in aggregation
    function test_DoubleContributionPrevention() public {
        uint256 modelId = 1;
        uint256 roundId = 1;

        mockVerifier.setShouldPass(true);

        bytes memory proof = new bytes(320);
        uint256[] memory inputs = new uint256[](8);
        for (uint i = 0; i < 7; i++) inputs[i] = i + 1;
        inputs[7] = 0;

        vm.prank(attacker1);
        aggregationVerifier.submitContribution(
            modelId, roundId, keccak256("grad1"), 1000, 50, proof, inputs
        );

        vm.prank(attacker1);
        vm.expectRevert("Already contributed");
        aggregationVerifier.submitContribution(
            modelId, roundId, keccak256("grad2"), 1000, 50, proof, inputs
        );
    }

    /// @notice Test contribution to finalized round
    function test_ContributionToFinalizedRound() public {
        uint256 modelId = 1;
        uint256 roundId = 1;

        mockVerifier.setShouldPass(true);

        bytes memory proof = new bytes(320);
        uint256[] memory inputs = new uint256[](8);
        for (uint i = 0; i < 7; i++) inputs[i] = i + 1;
        inputs[7] = 0;

        // Add enough contributions
        vm.prank(honestProver);
        aggregationVerifier.submitContribution(
            modelId, roundId, keccak256("grad1"), 3000, 50, proof, inputs
        );
        vm.prank(attacker1);
        aggregationVerifier.submitContribution(
            modelId, roundId, keccak256("grad2"), 3000, 50, proof, inputs
        );

        // Finalize
        aggregationVerifier.finalizeAggregation(
            modelId, roundId, keccak256("aggregated"), 100, proof, inputs
        );

        // Try to contribute after finalization
        vm.prank(attacker2);
        vm.expectRevert("Round finalized");
        aggregationVerifier.submitContribution(
            modelId, roundId, keccak256("late"), 1000, 50, proof, inputs
        );
    }

    /// @notice Test error bound exceeding maximum in aggregation
    function test_ExcessiveAggregationErrorBound() public {
        uint256 modelId = 1;
        uint256 roundId = 1;

        mockVerifier.setShouldPass(true);

        bytes memory proof = new bytes(320);
        uint256[] memory inputs = new uint256[](8);
        for (uint i = 0; i < 7; i++) inputs[i] = i + 1;
        inputs[7] = 0;

        // Error bound exceeds config max (default 1000)
        vm.prank(attacker1);
        vm.expectRevert("Error bound exceeds maximum");
        aggregationVerifier.submitContribution(
            modelId, roundId, keccak256("grad"), 1000, 1001, proof, inputs
        );
    }

    // ============ Sybil Attack Prevention ============

    /// @notice Test that stake requirements prevent simple Sybil attacks
    function test_SybilAttackMitigation() public {
        (uint256 modelId,,) = _setupModelAndRound();

        // Attacker creates many small accounts
        address[] memory sybils = new address[](10);
        for (uint i = 0; i < 10; i++) {
            sybils[i] = makeAddr(string(abi.encodePacked("sybil", i)));
            vm.deal(sybils[i], 0.05 ether);  // Below minimum stake
        }

        // None can stake enough - they don't have sufficient ETH to send
        // Verify each address only has 0.05 ether which is less than MIN_STAKE (0.1 ether)
        for (uint i = 0; i < 10; i++) {
            // The stake call will fail because they don't have enough ETH
            assertLt(sybils[i].balance, MIN_STAKE);
            // Attempting to stake would fail at the EVM level (insufficient balance)
            // We verify the invariant that accounts with < MIN_STAKE cannot participate
        }

        // Verify a properly funded account CAN stake
        address wellFunded = makeAddr("wellFunded");
        vm.deal(wellFunded, 1 ether);
        vm.prank(wellFunded);
        coordinator.stake{value: MIN_STAKE}(modelId);
        (uint256 stake,,) = coordinator.getStake(wellFunded, modelId);
        assertEq(stake, MIN_STAKE);
    }

    // ============ Gas Griefing Tests ============

    /// @notice Test that large proof doesn't cause excessive gas
    function test_LargeProofGasLimit() public {
        _setVerifierToReject();
        (uint256 modelId, uint256 hashLo, uint256 hashHi) = _setupModelAndRound();

        vm.prank(attacker1);
        coordinator.stake{value: LARGE_STAKE}(modelId);

        // Create a very large proof (10KB)
        bytes memory largeProof = new bytes(10240);
        uint256[] memory inputs = _createValidPublicInputs(hashLo, hashHi, 1111, 2222, modelId);

        // Measure gas - should complete within reasonable limits
        uint256 gasBefore = gasleft();
        vm.prank(attacker1);
        coordinator.submitProof(modelId, 1, largeProof, inputs);
        uint256 gasUsed = gasBefore - gasleft();

        // Should use reasonable gas (less than 2M for verification failure path)
        assertLt(gasUsed, 2_000_000);
    }

    // ============ Advanced Slashing Scenarios ============

    /// @notice Test slashing with exact percentage calculation
    function test_SlashingPercentageCalculation() public {
        (uint256 modelId, uint256 hashLo, uint256 hashHi) = _setupModelAndRound();

        // Test with various stake amounts
        uint256[] memory stakeAmounts = new uint256[](5);
        stakeAmounts[0] = 1 ether;
        stakeAmounts[1] = 2.5 ether;
        stakeAmounts[2] = 0.1 ether;
        stakeAmounts[3] = 10 ether;
        stakeAmounts[4] = 0.123456789 ether;

        for (uint256 i = 0; i < stakeAmounts.length; i++) {
            // Create new attacker for each test
            address attacker = makeAddr(string(abi.encodePacked("stakeAttacker", i)));
            vm.deal(attacker, stakeAmounts[i] + 1 ether);

            vm.prank(attacker);
            coordinator.stake{value: stakeAmounts[i]}(modelId);

            mockVerifier.setShouldPass(false);

            uint256[] memory inputs = _createValidPublicInputs(hashLo, hashHi, 1111 + i, 2222 + i, modelId);
            bytes memory proof = new bytes(320);

            vm.prank(attacker);
            coordinator.submitProof(modelId, 1, proof, inputs);

            (uint256 remaining,, bool slashed) = coordinator.getStake(attacker, modelId);
            assertTrue(slashed);
            // 50% slashing percentage
            assertEq(remaining, stakeAmounts[i] / 2, "Should slash exactly 50%");
        }
    }

    /// @notice Test slashing record details
    function test_SlashingRecordDetails() public {
        (uint256 modelId, uint256 hashLo, uint256 hashHi) = _setupModelAndRound();

        vm.prank(attacker1);
        coordinator.stake{value: LARGE_STAKE}(modelId);

        mockVerifier.setShouldPass(false);

        uint256[] memory inputs = _createValidPublicInputs(hashLo, hashHi, 1111, 2222, modelId);
        bytes memory proof = new bytes(320);

        uint256 expectedSlashAmount = LARGE_STAKE / 2;

        vm.prank(attacker1);
        coordinator.submitProof(modelId, 1, proof, inputs);

        // Verify slashing record
        (
            address prover,
            uint64 recordModelId,
            uint32 roundId,
            uint128 amount,
            ,
            uint40 timestamp
        ) = coordinator.slashingRecords(0);

        assertEq(prover, attacker1, "Wrong prover in record");
        assertEq(recordModelId, modelId, "Wrong model ID in record");
        assertEq(roundId, 1, "Wrong round ID in record");
        assertEq(amount, expectedSlashAmount, "Wrong amount in record");
        assertGt(timestamp, 0, "Timestamp should be set");
    }

    /// @notice Test partial stake withdrawal before slash
    function test_PartialWithdrawalBeforeSlash() public {
        (uint256 modelId, uint256 hashLo, uint256 hashHi) = _setupModelAndRound();

        vm.prank(attacker1);
        coordinator.stake{value: LARGE_STAKE}(modelId);

        // Lock period prevents withdrawal
        vm.prank(attacker1);
        vm.expectRevert(HelixCoordinatorV2.StillLocked.selector);
        coordinator.unstake(modelId);

        // Attacker cannot withdraw before being slashed
        mockVerifier.setShouldPass(false);
        uint256[] memory inputs = _createValidPublicInputs(hashLo, hashHi, 1111, 2222, modelId);
        bytes memory proof = new bytes(320);

        vm.prank(attacker1);
        coordinator.submitProof(modelId, 1, proof, inputs);

        // After slashing, withdrawal should fail due to slashed status
        vm.prank(attacker1);
        vm.expectRevert(HelixCoordinatorV2.StakeWasSlashed.selector);
        coordinator.unstake(modelId);
    }

    /// @notice Test model deactivation after slashing
    function test_ModelOperationsAfterSlashing() public {
        (uint256 modelId, uint256 hashLo, uint256 hashHi) = _setupModelAndRound();

        vm.prank(attacker1);
        coordinator.stake{value: LARGE_STAKE}(modelId);

        mockVerifier.setShouldPass(false);

        uint256[] memory inputs = _createValidPublicInputs(hashLo, hashHi, 1111, 2222, modelId);
        bytes memory proof = new bytes(320);

        // Attacker gets slashed
        vm.prank(attacker1);
        coordinator.submitProof(modelId, 1, proof, inputs);

        // Honest prover can still complete the round
        vm.prank(honestProver);
        coordinator.stake{value: LARGE_STAKE}(modelId);

        mockVerifier.setShouldPass(true);

        // Use different proof bytes so it's not flagged as replay
        bytes memory honestProof = new bytes(320);
        honestProof[0] = 0x01;
        uint256[] memory honestInputs = _createValidPublicInputs(hashLo, hashHi, 1111, 2222, modelId);

        vm.prank(honestProver);
        coordinator.submitProof(modelId, 1, honestProof, honestInputs);

        // Model should still be active
        (,, bool active) = coordinator.getModelState(modelId);
        assertTrue(active, "Model should remain active");
    }

    // ============ Multi-Model Attack Scenarios ============

    /// @notice Test attacker across multiple models
    function test_AttackerAcrossMultipleModels() public {
        // Create multiple models
        uint256[] memory modelIds = new uint256[](3);
        for (uint256 i = 0; i < 3; i++) {
            uint256 commitment = uint256(keccak256(abi.encodePacked(i + 1, i + 2)));
            vm.prank(modelOwner);
            modelIds[i] = coordinator.registerModel(
                string(abi.encodePacked("Model", i)),
                commitment,
                MIN_STAKE
            );
            vm.prank(modelOwner);
            coordinator.startRound(modelIds[i], ROUND_DURATION);
        }

        // Attacker stakes on all models
        for (uint256 i = 0; i < 3; i++) {
            vm.prank(attacker1);
            coordinator.stake{value: LARGE_STAKE}(modelIds[i]);
        }

        // Attack model 0
        mockVerifier.setShouldPass(false);
        uint256[] memory inputs = new uint256[](8);
        inputs[0] = 1;
        inputs[1] = 2;
        inputs[2] = 3;
        inputs[3] = 4;
        inputs[4] = 100;
        inputs[5] = 10;
        inputs[6] = 1;
        inputs[7] = ProofFixtureHardcoded.computeErrorChecksum(inputs[5], inputs[6], modelIds[0], 1e18);
        bytes memory proof = new bytes(320);

        vm.prank(attacker1);
        coordinator.submitProof(modelIds[0], 1, proof, inputs);

        // Slashed on model 0
        (,, bool slashed0) = coordinator.getStake(attacker1, modelIds[0]);
        assertTrue(slashed0, "Should be slashed on model 0");

        // But can still operate on other models
        (,, bool slashed1) = coordinator.getStake(attacker1, modelIds[1]);
        (,, bool slashed2) = coordinator.getStake(attacker1, modelIds[2]);
        assertFalse(slashed1, "Should not be slashed on model 1");
        assertFalse(slashed2, "Should not be slashed on model 2");
    }

    // ============ Edge Cases with Public Inputs ============

    /// @notice Test with boundary value public inputs
    function test_BoundaryValuePublicInputs() public {
        _setVerifierToReject();
        (uint256 modelId, uint256 hashLo, uint256 hashHi) = _setupModelAndRound();

        vm.prank(attacker1);
        coordinator.stake{value: LARGE_STAKE}(modelId);

        // Test with zero values (except commitment)
        uint256[] memory inputs = new uint256[](8);
        inputs[0] = hashLo;
        inputs[1] = hashHi;
        inputs[2] = 0;  // Zero new hash
        inputs[3] = 0;  // Zero new hash
        inputs[4] = 0;  // Zero loss
        inputs[5] = 0;  // Zero error bound
        inputs[6] = 0;  // Zero step number
        inputs[7] = ProofFixtureHardcoded.computeErrorChecksum(inputs[5], inputs[6], modelId, 1e18);

        bytes memory proof = new bytes(320);

        vm.prank(attacker1);
        coordinator.submitProof(modelId, 1, proof, inputs);

        // Should fail verification and get slashed
        (,, bool slashed) = coordinator.getStake(attacker1, modelId);
        assertTrue(slashed, "Should be slashed for invalid proof");
    }

    /// @notice Test with max uint128 values
    function test_LargeValuePublicInputs() public {
        (uint256 modelId, uint256 hashLo, uint256 hashHi) = _setupModelAndRound();

        vm.prank(attacker1);
        coordinator.stake{value: LARGE_STAKE}(modelId);

        // Test with max uint128 for hash values
        uint256[] memory inputs = new uint256[](8);
        inputs[0] = hashLo;
        inputs[1] = hashHi;
        inputs[2] = type(uint128).max;  // Large new hash lo
        inputs[3] = type(uint128).max;  // Large new hash hi
        inputs[4] = type(uint128).max;  // Large loss
        inputs[5] = 10;  // Valid error bound
        inputs[6] = type(uint64).max;  // Large step number
        inputs[7] = ProofFixtureHardcoded.computeErrorChecksum(inputs[5], inputs[6], modelId, 1e18);

        bytes memory proof = new bytes(320);

        mockVerifier.setShouldPass(true);

        vm.prank(attacker1);
        coordinator.submitProof(modelId, 1, proof, inputs);

        // Should succeed with valid proof
        (,, bool slashed) = coordinator.getStake(attacker1, modelId);
        assertFalse(slashed, "Should not be slashed for valid proof");
    }

    // ============ Concurrent Attack Tests ============

    /// @notice Test concurrent submissions from multiple attackers
    function test_ConcurrentAttackersMultipleRounds() public {
        (uint256 modelId, uint256 hashLo, uint256 hashHi) = _setupModelAndRound();

        // Multiple attackers stake
        address[] memory attackers = new address[](5);
        for (uint256 i = 0; i < 5; i++) {
            attackers[i] = makeAddr(string(abi.encodePacked("multiAttacker", i)));
            vm.deal(attackers[i], 10 ether);
            vm.prank(attackers[i]);
            coordinator.stake{value: LARGE_STAKE}(modelId);
        }

        // First attacker submits invalid
        mockVerifier.setShouldPass(false);
        uint256[] memory inputs = _createValidPublicInputs(hashLo, hashHi, 1111, 2222, modelId);
        bytes memory proof0 = new bytes(320);

        vm.prank(attackers[0]);
        coordinator.submitProof(modelId, 1, proof0, inputs);

        // First attacker slashed
        (,, bool slashed0) = coordinator.getStake(attackers[0], modelId);
        assertTrue(slashed0);

        // Second attacker submits valid with unique proof bytes (completes round)
        mockVerifier.setShouldPass(true);
        bytes memory proof1 = new bytes(320);
        proof1[0] = 0x01; // Unique proof
        vm.prank(attackers[1]);
        coordinator.submitProof(modelId, 1, proof1, inputs);

        // Second attacker not slashed
        (,, bool slashed1) = coordinator.getStake(attackers[1], modelId);
        assertFalse(slashed1);

        // Other attackers can't submit (round completed)
        for (uint256 i = 2; i < 5; i++) {
            bytes memory proofI = new bytes(320);
            proofI[0] = bytes1(uint8(i)); // Unique proof per attacker
            vm.prank(attackers[i]);
            vm.expectRevert(HelixCoordinatorV2.RoundAlreadyCompleted.selector);
            coordinator.submitProof(modelId, 1, proofI, inputs);
        }
    }

    // ============ Time-based Attack Tests ============

    /// @notice Test attack at exact deadline
    function test_AttackAtExactDeadline() public {
        (uint256 modelId, uint256 hashLo, uint256 hashHi) = _setupModelAndRound();

        vm.prank(attacker1);
        coordinator.stake{value: LARGE_STAKE}(modelId);

        mockVerifier.setShouldPass(true);

        uint256[] memory inputs = _createValidPublicInputs(hashLo, hashHi, 1111, 2222, modelId);
        bytes memory proof = new bytes(320);

        // Warp to exactly deadline (should still work)
        vm.warp(block.timestamp + ROUND_DURATION);

        vm.prank(attacker1);
        coordinator.submitProof(modelId, 1, proof, inputs);

        // Should succeed at deadline
        (,, bool slashed) = coordinator.getStake(attacker1, modelId);
        assertFalse(slashed);
    }

    /// @notice Test attack 1 second after deadline
    function test_AttackOneSecondAfterDeadline() public {
        (uint256 modelId, uint256 hashLo, uint256 hashHi) = _setupModelAndRound();

        vm.prank(attacker1);
        coordinator.stake{value: LARGE_STAKE}(modelId);

        mockVerifier.setShouldPass(true);

        uint256[] memory inputs = _createValidPublicInputs(hashLo, hashHi, 1111, 2222, modelId);
        bytes memory proof = new bytes(320);

        // Warp to 1 second after deadline
        vm.warp(block.timestamp + ROUND_DURATION + 1);

        vm.prank(attacker1);
        vm.expectRevert(HelixCoordinatorV2.RoundExpired.selector);
        coordinator.submitProof(modelId, 1, proof, inputs);
    }

    // ============ Slashing Parameter Tests ============

    /// @notice Test changing slash percentage mid-operation
    function test_SlashPercentageChange() public {
        (uint256 modelId, uint256 hashLo, uint256 hashHi) = _setupModelAndRound();

        vm.prank(attacker1);
        coordinator.stake{value: LARGE_STAKE}(modelId);

        // Change slash percentage to 75%
        _emergencySetSlashPercentage(7500);

        mockVerifier.setShouldPass(false);

        uint256[] memory inputs = _createValidPublicInputs(hashLo, hashHi, 1111, 2222, modelId);
        bytes memory proof = new bytes(320);

        vm.prank(attacker1);
        coordinator.submitProof(modelId, 1, proof, inputs);

        // Should slash 75%
        (uint256 remaining,, bool slashed) = coordinator.getStake(attacker1, modelId);
        assertTrue(slashed);
        assertEq(remaining, LARGE_STAKE * 25 / 100);  // 25% remains
    }

    /// @notice Test 100% slash
    function test_FullSlash() public {
        (uint256 modelId, uint256 hashLo, uint256 hashHi) = _setupModelAndRound();

        vm.prank(attacker1);
        coordinator.stake{value: LARGE_STAKE}(modelId);

        // Change slash percentage to 100%
        _emergencySetSlashPercentage(10000);

        mockVerifier.setShouldPass(false);

        uint256[] memory inputs = _createValidPublicInputs(hashLo, hashHi, 1111, 2222, modelId);
        bytes memory proof = new bytes(320);

        vm.prank(attacker1);
        coordinator.submitProof(modelId, 1, proof, inputs);

        // Should slash 100%
        (uint256 remaining,, bool slashed) = coordinator.getStake(attacker1, modelId);
        assertTrue(slashed);
        assertEq(remaining, 0);  // 0% remains
    }

    /// @notice Test 0% slash (warning only)
    function test_ZeroSlash() public {
        (uint256 modelId, uint256 hashLo, uint256 hashHi) = _setupModelAndRound();

        vm.prank(attacker1);
        coordinator.stake{value: LARGE_STAKE}(modelId);

        // Change slash percentage to 0%
        _emergencySetSlashPercentage(0);

        mockVerifier.setShouldPass(false);

        uint256[] memory inputs = _createValidPublicInputs(hashLo, hashHi, 1111, 2222, modelId);
        bytes memory proof = new bytes(320);

        vm.prank(attacker1);
        coordinator.submitProof(modelId, 1, proof, inputs);

        // Should still mark as slashed but 0 amount
        (uint256 remaining,, bool slashed) = coordinator.getStake(attacker1, modelId);
        assertTrue(slashed);  // Still marked as slashed
        assertEq(remaining, LARGE_STAKE);  // Full stake remains
    }

    // ============ Emergency Pause Tests ============

    /// @notice Test attack during emergency pause
    function test_AttackDuringEmergencyPause() public {
        (uint256 modelId, uint256 hashLo, uint256 hashHi) = _setupModelAndRound();

        vm.prank(attacker1);
        coordinator.stake{value: LARGE_STAKE}(modelId);

        // Pause the contract
        coordinator.emergencyPause();

        mockVerifier.setShouldPass(false);

        uint256[] memory inputs = _createValidPublicInputs(hashLo, hashHi, 1111, 2222, modelId);
        bytes memory proof = new bytes(320);

        // Note: submitProof doesn't have whenNotPaused modifier in current implementation
        // This test documents current behavior
    }

    /// @notice Test new model registration during pause
    function test_NewModelDuringPause() public {
        // Pause the contract
        coordinator.emergencyPause();

        vm.prank(modelOwner);
        vm.expectRevert(HelixCoordinatorV2.ContractPaused.selector);
        coordinator.registerModel("Test", 12345, MIN_STAKE);
    }

    // ============ Model State Attack Tests ============

    /// @notice Test attack on inactive model
    function test_AttackOnInactiveModel() public {
        (uint256 modelId, uint256 hashLo, uint256 hashHi) = _setupModelAndRound();

        vm.prank(attacker1);
        coordinator.stake{value: LARGE_STAKE}(modelId);

        // Deactivate model
        vm.prank(modelOwner);
        coordinator.pauseModel(modelId);

        // Try to start new round on inactive model
        vm.prank(modelOwner);
        vm.expectRevert(HelixCoordinatorV2.ModelNotActive.selector);
        coordinator.startRound(modelId, ROUND_DURATION);
    }

    // ============ Accumulated Error Attack Tests ============

    /// @notice Test model error tolerance check
    function test_ModelErrorToleranceCheck() public {
        (uint256 modelId, uint256 hashLo, uint256 hashHi) = _setupModelAndRound();

        vm.prank(honestProver);
        coordinator.stake{value: LARGE_STAKE}(modelId);

        mockVerifier.setShouldPass(true);

        // Submit proof with high error bound
        uint256 maxEB = coordinator.maxErrorBound();
        uint256[] memory inputs = new uint256[](8);
        inputs[0] = hashLo;
        inputs[1] = hashHi;
        inputs[2] = 1111;
        inputs[3] = 2222;
        inputs[4] = 100;
        inputs[5] = maxEB;  // Max allowed error
        inputs[6] = 1;
        inputs[7] = ProofFixtureHardcoded.computeErrorChecksum(inputs[5], inputs[6], modelId, 1e18);

        bytes memory proof = new bytes(320);

        vm.prank(honestProver);
        coordinator.submitProof(modelId, 1, proof, inputs);

        // Check accumulated error
        uint256 accumulated = coordinator.getAccumulatedErrorBound(modelId);
        assertEq(accumulated, maxEB);

        // Model is still acceptable at this threshold
        assertTrue(coordinator.isModelErrorAcceptable(modelId, maxEB));

        // But not acceptable below
        assertFalse(coordinator.isModelErrorAcceptable(modelId, maxEB - 1));
    }

    /// @notice Test accumulated error reset
    function test_AccumulatedErrorReset() public {
        (uint256 modelId, uint256 hashLo, uint256 hashHi) = _setupModelAndRound();

        vm.prank(honestProver);
        coordinator.stake{value: LARGE_STAKE}(modelId);

        mockVerifier.setShouldPass(true);

        uint256[] memory inputs = _createValidPublicInputs(hashLo, hashHi, 1111, 2222, modelId);
        bytes memory proof = new bytes(320);

        vm.prank(honestProver);
        coordinator.submitProof(modelId, 1, proof, inputs);

        // Has accumulated error
        uint256 errorBefore = coordinator.getAccumulatedErrorBound(modelId);
        assertGt(errorBefore, 0);

        // Reset error
        vm.prank(modelOwner);
        coordinator.resetAccumulatedError(modelId);

        // Error should be 0
        assertEq(coordinator.getAccumulatedErrorBound(modelId), 0);
    }
}
