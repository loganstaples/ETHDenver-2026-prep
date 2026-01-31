// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "forge-std/Test.sol";
import "../src/core/TrainingRound.sol";

/// @title TrainingRoundTest
/// @notice Comprehensive tests for TrainingRound state machine
contract TrainingRoundTest is Test {
    TrainingRound public round;

    address public owner;
    address public coordinator;
    address public participant1;
    address public participant2;
    address public participant3;
    address public nonParticipant;

    uint256 constant ROUND_DURATION = 1 hours;

    // Events
    event RoundCreated(uint256 indexed modelId, uint256 indexed roundId, bytes32 startCommitment, uint256 deadline);
    event ParticipantRegistered(uint256 indexed modelId, uint256 indexed roundId, address indexed participant);
    event GradientSubmitted(uint256 indexed modelId, uint256 indexed roundId, address indexed participant, bytes32 gradientCommitment);
    event RoundActivated(uint256 indexed modelId, uint256 indexed roundId);
    event RoundAggregating(uint256 indexed modelId, uint256 indexed roundId);
    event RoundCompleted(uint256 indexed modelId, uint256 indexed roundId, bytes32 endCommitment, uint256 participantCount);
    event RoundFailed(uint256 indexed modelId, uint256 indexed roundId, string reason);

    function setUp() public {
        owner = address(this);
        coordinator = makeAddr("coordinator");
        participant1 = makeAddr("participant1");
        participant2 = makeAddr("participant2");
        participant3 = makeAddr("participant3");
        nonParticipant = makeAddr("nonParticipant");

        round = new TrainingRound();
        round.setCoordinator(coordinator);
    }

    // ============ Round Creation Tests ============

    function test_CreateRound() public {
        uint256 modelId = 1;
        bytes32 startCommit = keccak256("initial_state");

        vm.prank(coordinator);
        uint256 roundId = round.createRound(modelId, startCommit, 2, 10, ROUND_DURATION);

        assertEq(roundId, 1);
        assertEq(round.roundCount(modelId), 1);

        TrainingRound.Round memory r = round.getRound(modelId, roundId);
        assertEq(r.modelId, modelId);
        assertEq(r.roundId, roundId);
        assertEq(r.startCommitment, startCommit);
        assertEq(uint256(r.state), uint256(TrainingRound.RoundState.Created));
        assertEq(r.minParticipants, 2);
        assertEq(r.maxParticipants, 10);
        assertEq(r.deadline, block.timestamp + ROUND_DURATION);
    }

    function test_CreateRound_DefaultDuration() public {
        uint256 modelId = 1;
        bytes32 startCommit = keccak256("initial");

        vm.prank(coordinator);
        uint256 roundId = round.createRound(modelId, startCommit, 2, 10, 0);

        TrainingRound.Round memory r = round.getRound(modelId, roundId);
        assertEq(r.deadline, block.timestamp + round.defaultRoundDuration());
    }

    function test_CreateRound_OnlyCoordinator() public {
        vm.prank(participant1);
        vm.expectRevert("Only coordinator");
        round.createRound(1, keccak256("test"), 2, 10, ROUND_DURATION);
    }

    function test_CreateMultipleRounds() public {
        uint256 modelId = 1;

        vm.startPrank(coordinator);
        uint256 round1 = round.createRound(modelId, keccak256("r1"), 2, 10, ROUND_DURATION);
        uint256 round2 = round.createRound(modelId, keccak256("r2"), 2, 10, ROUND_DURATION);
        uint256 round3 = round.createRound(modelId, keccak256("r3"), 2, 10, ROUND_DURATION);
        vm.stopPrank();

        assertEq(round1, 1);
        assertEq(round2, 2);
        assertEq(round3, 3);
        assertEq(round.roundCount(modelId), 3);
    }

    function test_CreateRoundsForMultipleModels() public {
        vm.startPrank(coordinator);
        uint256 r1 = round.createRound(1, keccak256("m1r1"), 2, 10, ROUND_DURATION);
        uint256 r2 = round.createRound(2, keccak256("m2r1"), 2, 10, ROUND_DURATION);
        uint256 r3 = round.createRound(1, keccak256("m1r2"), 2, 10, ROUND_DURATION);
        vm.stopPrank();

        assertEq(r1, 1);
        assertEq(r2, 1);  // Different model, resets to 1
        assertEq(r3, 2);  // Same model, increments
    }

    // ============ Registration Tests ============

    function test_RegisterForRound() public {
        uint256 modelId = 1;
        vm.prank(coordinator);
        uint256 roundId = round.createRound(modelId, keccak256("test"), 2, 10, ROUND_DURATION);

        vm.expectEmit(true, true, true, true);
        emit ParticipantRegistered(modelId, roundId, participant1);

        vm.prank(participant1);
        round.registerForRound(modelId, roundId);

        TrainingRound.Participant memory p = round.getParticipant(modelId, roundId, participant1);
        assertTrue(p.registered);
        assertFalse(p.submitted);
        assertEq(p.addr, participant1);

        address[] memory participants = round.getParticipants(modelId, roundId);
        assertEq(participants.length, 1);
        assertEq(participants[0], participant1);
    }

    function test_RegisterActivatesRound() public {
        uint256 modelId = 1;
        vm.prank(coordinator);
        uint256 roundId = round.createRound(modelId, keccak256("test"), 2, 10, ROUND_DURATION);

        // First registration - still Created
        vm.prank(participant1);
        round.registerForRound(modelId, roundId);

        TrainingRound.Round memory r = round.getRound(modelId, roundId);
        assertEq(uint256(r.state), uint256(TrainingRound.RoundState.Created));

        // Second registration - activates round
        vm.expectEmit(true, true, false, false);
        emit RoundActivated(modelId, roundId);

        vm.prank(participant2);
        round.registerForRound(modelId, roundId);

        r = round.getRound(modelId, roundId);
        assertEq(uint256(r.state), uint256(TrainingRound.RoundState.Active));
        assertEq(r.registeredCount, 2);
    }

    function test_CannotDoubleRegister() public {
        uint256 modelId = 1;
        vm.prank(coordinator);
        uint256 roundId = round.createRound(modelId, keccak256("test"), 2, 10, ROUND_DURATION);

        vm.prank(participant1);
        round.registerForRound(modelId, roundId);

        vm.prank(participant1);
        vm.expectRevert("Already registered");
        round.registerForRound(modelId, roundId);
    }

    function test_CannotRegisterWhenFull() public {
        uint256 modelId = 1;
        vm.prank(coordinator);
        uint256 roundId = round.createRound(modelId, keccak256("test"), 1, 2, ROUND_DURATION);

        vm.prank(participant1);
        round.registerForRound(modelId, roundId);
        vm.prank(participant2);
        round.registerForRound(modelId, roundId);

        vm.prank(participant3);
        vm.expectRevert("Round full");
        round.registerForRound(modelId, roundId);
    }

    function test_CannotRegisterAfterDeadline() public {
        uint256 modelId = 1;
        vm.prank(coordinator);
        uint256 roundId = round.createRound(modelId, keccak256("test"), 2, 10, ROUND_DURATION);

        vm.warp(block.timestamp + ROUND_DURATION + 1);

        vm.prank(participant1);
        vm.expectRevert("Round deadline passed");
        round.registerForRound(modelId, roundId);
    }

    function test_CannotRegisterInCompletedRound() public {
        uint256 modelId = 1;
        vm.prank(coordinator);
        uint256 roundId = round.createRound(modelId, keccak256("test"), 1, 10, ROUND_DURATION);

        // Register and submit
        vm.prank(participant1);
        round.registerForRound(modelId, roundId);

        vm.prank(participant1);
        round.submitGradient(modelId, roundId, keccak256("grad"), 5);

        // Complete round
        vm.prank(coordinator);
        round.startAggregation(modelId, roundId);
        vm.prank(coordinator);
        round.completeRound(modelId, roundId, keccak256("final"), 10);

        // Try to register
        vm.prank(participant2);
        vm.expectRevert("Round not accepting registrations");
        round.registerForRound(modelId, roundId);
    }

    // ============ Gradient Submission Tests ============

    function test_SubmitGradient() public {
        uint256 modelId = 1;
        vm.prank(coordinator);
        uint256 roundId = round.createRound(modelId, keccak256("test"), 1, 10, ROUND_DURATION);

        vm.prank(participant1);
        round.registerForRound(modelId, roundId);

        bytes32 gradCommit = keccak256("gradient_data");

        vm.expectEmit(true, true, true, true);
        emit GradientSubmitted(modelId, roundId, participant1, gradCommit);

        vm.prank(participant1);
        round.submitGradient(modelId, roundId, gradCommit, 100);

        TrainingRound.Participant memory p = round.getParticipant(modelId, roundId, participant1);
        assertTrue(p.submitted);
        assertEq(p.gradientCommitment, gradCommit);
        assertEq(p.errorBound, 100);
        assertGt(p.submittedAt, 0);

        TrainingRound.Round memory r = round.getRound(modelId, roundId);
        assertEq(r.submittedCount, 1);
    }

    function test_CannotSubmitWithoutRegistration() public {
        uint256 modelId = 1;
        vm.prank(coordinator);
        uint256 roundId = round.createRound(modelId, keccak256("test"), 1, 10, ROUND_DURATION);

        // Activate round with a different participant
        vm.prank(participant1);
        round.registerForRound(modelId, roundId);

        // Non-registered tries to submit
        vm.prank(nonParticipant);
        vm.expectRevert("Not registered");
        round.submitGradient(modelId, roundId, keccak256("malicious"), 0);
    }

    function test_CannotDoubleSubmit() public {
        uint256 modelId = 1;
        vm.prank(coordinator);
        uint256 roundId = round.createRound(modelId, keccak256("test"), 1, 10, ROUND_DURATION);

        vm.prank(participant1);
        round.registerForRound(modelId, roundId);

        vm.prank(participant1);
        round.submitGradient(modelId, roundId, keccak256("grad1"), 5);

        vm.prank(participant1);
        vm.expectRevert("Already submitted");
        round.submitGradient(modelId, roundId, keccak256("grad2"), 5);
    }

    function test_CannotSubmitAfterDeadline() public {
        uint256 modelId = 1;
        vm.prank(coordinator);
        uint256 roundId = round.createRound(modelId, keccak256("test"), 1, 10, ROUND_DURATION);

        vm.prank(participant1);
        round.registerForRound(modelId, roundId);

        vm.warp(block.timestamp + ROUND_DURATION + 1);

        vm.prank(participant1);
        vm.expectRevert("Round deadline passed");
        round.submitGradient(modelId, roundId, keccak256("grad"), 5);
    }

    function test_CannotSubmitInCreatedState() public {
        uint256 modelId = 1;
        vm.prank(coordinator);
        uint256 roundId = round.createRound(modelId, keccak256("test"), 2, 10, ROUND_DURATION);

        // Only register one participant (need 2 to activate)
        vm.prank(participant1);
        round.registerForRound(modelId, roundId);

        TrainingRound.Round memory r = round.getRound(modelId, roundId);
        assertEq(uint256(r.state), uint256(TrainingRound.RoundState.Created));

        vm.prank(participant1);
        vm.expectRevert("Round not active");
        round.submitGradient(modelId, roundId, keccak256("grad"), 5);
    }

    // ============ Aggregation Tests ============

    function test_StartAggregation() public {
        uint256 modelId = 1;
        vm.prank(coordinator);
        uint256 roundId = round.createRound(modelId, keccak256("test"), 2, 10, ROUND_DURATION);

        // Register and submit
        vm.prank(participant1);
        round.registerForRound(modelId, roundId);
        vm.prank(participant2);
        round.registerForRound(modelId, roundId);

        vm.prank(participant1);
        round.submitGradient(modelId, roundId, keccak256("grad1"), 5);
        vm.prank(participant2);
        round.submitGradient(modelId, roundId, keccak256("grad2"), 5);

        vm.expectEmit(true, true, false, false);
        emit RoundAggregating(modelId, roundId);

        vm.prank(coordinator);
        round.startAggregation(modelId, roundId);

        TrainingRound.Round memory r = round.getRound(modelId, roundId);
        assertEq(uint256(r.state), uint256(TrainingRound.RoundState.Aggregating));
    }

    function test_CannotStartAggregationWithoutEnoughSubmissions() public {
        uint256 modelId = 1;
        vm.prank(coordinator);
        uint256 roundId = round.createRound(modelId, keccak256("test"), 2, 10, ROUND_DURATION);

        // Register 2 participants but only 1 submits
        vm.prank(participant1);
        round.registerForRound(modelId, roundId);
        vm.prank(participant2);
        round.registerForRound(modelId, roundId);

        vm.prank(participant1);
        round.submitGradient(modelId, roundId, keccak256("grad"), 5);

        vm.prank(coordinator);
        vm.expectRevert("Not enough submissions");
        round.startAggregation(modelId, roundId);
    }

    function test_StartAggregationAfterDeadline() public {
        uint256 modelId = 1;
        vm.prank(coordinator);
        uint256 roundId = round.createRound(modelId, keccak256("test"), 1, 10, ROUND_DURATION);

        // Register and submit
        vm.prank(participant1);
        round.registerForRound(modelId, roundId);
        vm.prank(participant1);
        round.submitGradient(modelId, roundId, keccak256("grad"), 5);

        // Pass deadline
        vm.warp(block.timestamp + ROUND_DURATION + 1);

        // Can still start aggregation after deadline if enough submissions
        vm.prank(coordinator);
        round.startAggregation(modelId, roundId);

        TrainingRound.Round memory r = round.getRound(modelId, roundId);
        assertEq(uint256(r.state), uint256(TrainingRound.RoundState.Aggregating));
    }

    // ============ Round Completion Tests ============

    function test_CompleteRound() public {
        uint256 modelId = 1;
        vm.prank(coordinator);
        uint256 roundId = round.createRound(modelId, keccak256("start"), 1, 10, ROUND_DURATION);

        vm.prank(participant1);
        round.registerForRound(modelId, roundId);
        vm.prank(participant1);
        round.submitGradient(modelId, roundId, keccak256("grad"), 5);

        vm.prank(coordinator);
        round.startAggregation(modelId, roundId);

        bytes32 endCommit = keccak256("final_state");

        vm.expectEmit(true, true, true, true);
        emit RoundCompleted(modelId, roundId, endCommit, 1);

        vm.prank(coordinator);
        round.completeRound(modelId, roundId, endCommit, 100);

        TrainingRound.Round memory r = round.getRound(modelId, roundId);
        assertEq(uint256(r.state), uint256(TrainingRound.RoundState.Completed));
        assertEq(r.endCommitment, endCommit);
        assertEq(r.errorBound, 100);
        assertGt(r.endTime, 0);
    }

    function test_CannotCompleteWithoutAggregating() public {
        uint256 modelId = 1;
        vm.prank(coordinator);
        uint256 roundId = round.createRound(modelId, keccak256("test"), 1, 10, ROUND_DURATION);

        vm.prank(participant1);
        round.registerForRound(modelId, roundId);
        vm.prank(participant1);
        round.submitGradient(modelId, roundId, keccak256("grad"), 5);

        // Try to complete without starting aggregation
        vm.prank(coordinator);
        vm.expectRevert("Not aggregating");
        round.completeRound(modelId, roundId, keccak256("final"), 10);
    }

    // ============ Round Failure Tests ============

    function test_FailRound() public {
        uint256 modelId = 1;
        vm.prank(coordinator);
        uint256 roundId = round.createRound(modelId, keccak256("test"), 2, 10, ROUND_DURATION);

        vm.expectEmit(true, true, false, true);
        emit RoundFailed(modelId, roundId, "Timeout");

        vm.prank(coordinator);
        round.failRound(modelId, roundId, "Timeout");

        TrainingRound.Round memory r = round.getRound(modelId, roundId);
        assertEq(uint256(r.state), uint256(TrainingRound.RoundState.Failed));
        assertGt(r.endTime, 0);
    }

    function test_CannotFailCompletedRound() public {
        uint256 modelId = 1;
        vm.prank(coordinator);
        uint256 roundId = round.createRound(modelId, keccak256("test"), 1, 10, ROUND_DURATION);

        vm.prank(participant1);
        round.registerForRound(modelId, roundId);
        vm.prank(participant1);
        round.submitGradient(modelId, roundId, keccak256("grad"), 5);

        vm.prank(coordinator);
        round.startAggregation(modelId, roundId);
        vm.prank(coordinator);
        round.completeRound(modelId, roundId, keccak256("final"), 10);

        vm.prank(coordinator);
        vm.expectRevert("Round already finalized");
        round.failRound(modelId, roundId, "Late failure");
    }

    function test_CannotFailAlreadyFailedRound() public {
        uint256 modelId = 1;
        vm.prank(coordinator);
        uint256 roundId = round.createRound(modelId, keccak256("test"), 2, 10, ROUND_DURATION);

        vm.prank(coordinator);
        round.failRound(modelId, roundId, "First failure");

        vm.prank(coordinator);
        vm.expectRevert("Round already finalized");
        round.failRound(modelId, roundId, "Second failure");
    }

    // ============ View Function Tests ============

    function test_GetParticipants() public {
        uint256 modelId = 1;
        vm.prank(coordinator);
        uint256 roundId = round.createRound(modelId, keccak256("test"), 1, 10, ROUND_DURATION);

        vm.prank(participant1);
        round.registerForRound(modelId, roundId);
        vm.prank(participant2);
        round.registerForRound(modelId, roundId);

        address[] memory participants = round.getParticipants(modelId, roundId);
        assertEq(participants.length, 2);
        assertEq(participants[0], participant1);
        assertEq(participants[1], participant2);
    }

    function test_IsDeadlinePassed() public {
        uint256 modelId = 1;
        vm.prank(coordinator);
        uint256 roundId = round.createRound(modelId, keccak256("test"), 2, 10, ROUND_DURATION);

        assertFalse(round.isDeadlinePassed(modelId, roundId));

        vm.warp(block.timestamp + ROUND_DURATION + 1);

        assertTrue(round.isDeadlinePassed(modelId, roundId));
    }

    // ============ Admin Tests ============

    function test_SetCoordinator() public {
        address newCoord = makeAddr("newCoord");
        round.setCoordinator(newCoord);
        assertEq(round.coordinator(), newCoord);
    }

    function test_SetCoordinator_OnlyOwner() public {
        vm.prank(participant1);
        vm.expectRevert("Only owner");
        round.setCoordinator(participant1);
    }

    function test_SetCoordinator_InvalidAddress() public {
        vm.expectRevert("Invalid address");
        round.setCoordinator(address(0));
    }

    function test_SetDefaultRoundDuration() public {
        round.setDefaultRoundDuration(2 hours);
        assertEq(round.defaultRoundDuration(), 2 hours);
    }

    function test_SetDefaultRoundDuration_InvalidDuration() public {
        vm.expectRevert("Invalid duration");
        round.setDefaultRoundDuration(0);
    }

    function test_TransferOwnership() public {
        address newOwner = makeAddr("newOwner");
        round.transferOwnership(newOwner);
        assertEq(round.owner(), newOwner);

        vm.prank(newOwner);
        round.setDefaultRoundDuration(3 hours);
        assertEq(round.defaultRoundDuration(), 3 hours);
    }

    function test_TransferOwnership_InvalidAddress() public {
        vm.expectRevert("Invalid address");
        round.transferOwnership(address(0));
    }

    // ============ Full Lifecycle Test ============

    function test_FullRoundLifecycle() public {
        uint256 modelId = 42;
        bytes32 startCommit = keccak256("model_v1");
        bytes32 endCommit = keccak256("model_v2");

        // Create round
        vm.prank(coordinator);
        uint256 roundId = round.createRound(modelId, startCommit, 3, 10, ROUND_DURATION);

        // Register 3 participants
        vm.prank(participant1);
        round.registerForRound(modelId, roundId);
        vm.prank(participant2);
        round.registerForRound(modelId, roundId);
        vm.prank(participant3);
        round.registerForRound(modelId, roundId);

        // All submit gradients
        vm.prank(participant1);
        round.submitGradient(modelId, roundId, keccak256("grad1"), 10);
        vm.prank(participant2);
        round.submitGradient(modelId, roundId, keccak256("grad2"), 15);
        vm.prank(participant3);
        round.submitGradient(modelId, roundId, keccak256("grad3"), 12);

        // Start aggregation
        vm.prank(coordinator);
        round.startAggregation(modelId, roundId);

        // Complete round
        vm.prank(coordinator);
        round.completeRound(modelId, roundId, endCommit, 37);

        // Verify final state
        TrainingRound.Round memory r = round.getRound(modelId, roundId);
        assertEq(uint256(r.state), uint256(TrainingRound.RoundState.Completed));
        assertEq(r.startCommitment, startCommit);
        assertEq(r.endCommitment, endCommit);
        assertEq(r.registeredCount, 3);
        assertEq(r.submittedCount, 3);
        assertEq(r.errorBound, 37);
    }
}
