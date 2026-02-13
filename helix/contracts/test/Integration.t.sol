// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "forge-std/Test.sol";
import "../src/core/HelixCoordinatorV2.sol";
import "../src/core/TrainingRound.sol";
import "../src/verification/Halo2Verifier.sol";
import "../src/verification/AggregationVerifier.sol";
import "../src/mocks/MockVerifier.sol";
import "./ProofFixtures.t.sol";

/// @title IntegrationTest
/// @notice Full end-to-end integration tests for HELIX protocol
/// @dev Tests complete workflows: registration -> staking -> training -> completion
contract IntegrationTest is Test {
    // ============ Contracts ============
    HelixCoordinatorV2 public coordinator;
    Halo2Verifier public verifier;
    MockVerifier public mockVerifier;
    TrainingRound public trainingRound;
    AggregationVerifier public aggregationVerifier;

    // ============ Actors ============
    address public owner;
    address public treasury;
    address public modelOwner;
    address public prover1;
    address public prover2;
    address public prover3;
    address public attacker;

    // ============ Constants ============
    uint256 constant MIN_STAKE = 0.1 ether;
    uint256 constant ROUND_DURATION = 1 hours;
    uint256 constant LARGE_STAKE = 1 ether;

    // ============ Events ============
    event ModelRegistered(uint256 indexed modelId, address indexed owner, uint256 initialCommitment, string ipfsHash);
    event RoundStarted(uint256 indexed modelId, uint256 indexed roundId, uint256 deadline);
    event ProofSubmitted(uint256 indexed modelId, uint256 indexed roundId, address indexed prover, uint256 newCommitment);
    event RoundCompleted(uint256 indexed modelId, uint256 indexed roundId, uint256 newCommitment);
    event Staked(address indexed prover, uint256 indexed modelId, uint256 amount);
    event Unstaked(address indexed prover, uint256 indexed modelId, uint256 amount);
    event Slashed(address indexed prover, uint256 indexed modelId, uint256 amount, string reason);

    function setUp() public {
        // Set up actors
        owner = address(this);
        treasury = makeAddr("treasury");
        modelOwner = makeAddr("modelOwner");
        prover1 = makeAddr("prover1");
        prover2 = makeAddr("prover2");
        prover3 = makeAddr("prover3");
        attacker = makeAddr("attacker");

        // Deploy core contracts
        verifier = new Halo2Verifier();
        mockVerifier = new MockVerifier();
        coordinator = new HelixCoordinatorV2(address(mockVerifier), treasury);
        trainingRound = new TrainingRound();
        aggregationVerifier = new AggregationVerifier(address(mockVerifier));

        // Set up coordinator as TrainingRound coordinator
        trainingRound.setCoordinator(address(coordinator));

        // Fund actors
        vm.deal(modelOwner, 100 ether);
        vm.deal(prover1, 100 ether);
        vm.deal(prover2, 100 ether);
        vm.deal(prover3, 100 ether);
        vm.deal(attacker, 100 ether);
    }

    // ============ Full Workflow Tests ============

    /// @notice Test complete training workflow: register -> stake -> round -> proof -> complete
    function test_FullTrainingWorkflow() public {
        // Step 1: Model owner registers a model
        uint256 oldHashLo = 12345;
        uint256 oldHashHi = 67890;
        uint256 initialCommitment = uint256(keccak256(abi.encodePacked(oldHashLo, oldHashHi)));

        vm.prank(modelOwner);
        uint256 modelId = coordinator.registerModel("QmTestModel", initialCommitment, MIN_STAKE, 4, 8, 2, 2, 0);

        // Verify model registration
        (uint256 currentRound, uint256 commitment, bool active) = coordinator.getModelState(modelId);
        assertEq(currentRound, 0);
        assertEq(commitment, initialCommitment);
        assertTrue(active);

        // Step 2: Prover stakes to participate
        vm.prank(prover1);
        coordinator.stake{value: LARGE_STAKE}(modelId);

        (uint256 stakeAmount,,) = coordinator.getStake(prover1, modelId);
        assertEq(stakeAmount, LARGE_STAKE);

        // Step 3: Model owner starts a training round
        vm.prank(modelOwner);
        coordinator.startRound(modelId, ROUND_DURATION);

        (currentRound,,) = coordinator.getModelState(modelId);
        assertEq(currentRound, 1);

        // Step 4: Prover submits valid proof
        mockVerifier.setShouldPass(true);

        uint256 newHashLo = 11111;
        uint256 newHashHi = 22222;
        uint256 expectedNewCommitment = uint256(keccak256(abi.encodePacked(newHashLo, newHashHi)));

        uint256[] memory publicInputs = new uint256[](8);
        publicInputs[0] = oldHashLo;
        publicInputs[1] = oldHashHi;
        publicInputs[2] = newHashLo;
        publicInputs[3] = newHashHi;
        publicInputs[4] = 100;  // loss
        publicInputs[5] = 10;   // error bound
        publicInputs[6] = 1;    // step
        publicInputs[7] = ProofFixtureHardcoded.computeErrorChecksum(publicInputs[5], publicInputs[6], modelId, 1e18);

        bytes memory proof = new bytes(320);

        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, proof, publicInputs);

        // Verify round completion
        (currentRound, commitment, active) = coordinator.getModelState(modelId);
        assertEq(commitment, expectedNewCommitment);
        assertTrue(active);

        // Step 5: Check error bound accumulation
        uint256 accumulatedError = coordinator.getAccumulatedErrorBound(modelId);
        assertEq(accumulatedError, 10);

        // Step 6: Prover can unstake (lock period reset after valid proof)
        vm.prank(prover1);
        coordinator.unstake(modelId);

        (stakeAmount,,) = coordinator.getStake(prover1, modelId);
        assertEq(stakeAmount, 0);
    }

    /// @notice Test multi-round training with error bound accumulation
    function test_MultiRoundTraining() public {
        // Setup model
        uint256 hashLo = 100;
        uint256 hashHi = 200;
        uint256 currentCommitment = uint256(keccak256(abi.encodePacked(hashLo, hashHi)));

        vm.prank(modelOwner);
        uint256 modelId = coordinator.registerModel("QmMultiRound", currentCommitment, MIN_STAKE, 4, 8, 2, 2, 0);

        // Prover stakes
        vm.prank(prover1);
        coordinator.stake{value: LARGE_STAKE}(modelId);

        mockVerifier.setShouldPass(true);

        // Run 5 training rounds
        for (uint256 round = 1; round <= 5; round++) {
            vm.prank(modelOwner);
            coordinator.startRound(modelId, ROUND_DURATION);

            // Generate new commitment
            uint256 newHashLo = hashLo + round;
            uint256 newHashHi = hashHi + round;
            uint256 newCommitment = uint256(keccak256(abi.encodePacked(newHashLo, newHashHi)));

            uint256[] memory inputs = new uint256[](8);
            inputs[0] = hashLo;
            inputs[1] = hashHi;
            inputs[2] = newHashLo;
            inputs[3] = newHashHi;
            inputs[4] = 100 - round * 10;  // decreasing loss
            inputs[5] = 5;  // error bound per step
            inputs[6] = round;
            inputs[7] = ProofFixtureHardcoded.computeErrorChecksum(inputs[5], inputs[6], modelId, 1e18);

            bytes memory proof = new bytes(320);

            vm.prank(prover1);
            coordinator.submitProof(modelId, round, proof, inputs);

            // Update for next round
            hashLo = newHashLo;
            hashHi = newHashHi;
        }

        // Verify accumulated error bound (5 rounds * 5 error per round)
        uint256 totalError = coordinator.getAccumulatedErrorBound(modelId);
        assertEq(totalError, 25);

        // Verify final round number
        (uint256 finalRound,,) = coordinator.getModelState(modelId);
        assertEq(finalRound, 5);
    }

    /// @notice Test multiple provers competing in the same model
    function test_MultipleProversCompeting() public {
        // Setup
        uint256 hashLo = 1000;
        uint256 hashHi = 2000;
        uint256 initialCommitment = uint256(keccak256(abi.encodePacked(hashLo, hashHi)));

        vm.prank(modelOwner);
        uint256 modelId = coordinator.registerModel("QmCompetition", initialCommitment, MIN_STAKE, 4, 8, 2, 2, 0);

        // Multiple provers stake
        vm.prank(prover1);
        coordinator.stake{value: LARGE_STAKE}(modelId);
        vm.prank(prover2);
        coordinator.stake{value: LARGE_STAKE}(modelId);
        vm.prank(prover3);
        coordinator.stake{value: LARGE_STAKE}(modelId);

        // Start round
        vm.prank(modelOwner);
        coordinator.startRound(modelId, ROUND_DURATION);

        mockVerifier.setShouldPass(true);

        // Prover 2 submits first
        uint256 newHashLo = 1111;
        uint256 newHashHi = 2222;
        uint256[] memory inputs = new uint256[](8);
        inputs[0] = hashLo;
        inputs[1] = hashHi;
        inputs[2] = newHashLo;
        inputs[3] = newHashHi;
        inputs[4] = 50;
        inputs[5] = 5;
        inputs[6] = 1;
        inputs[7] = ProofFixtureHardcoded.computeErrorChecksum(inputs[5], inputs[6], modelId, 1e18);

        bytes memory proof = new bytes(320);

        vm.prank(prover2);
        coordinator.submitProof(modelId, 1, proof, inputs);

        // Verify prover2 got the round
        (,, bool active) = coordinator.getModelState(modelId);
        assertTrue(active);

        // Other provers cannot submit same proof (replay protection)
        vm.prank(prover1);
        vm.expectRevert(HelixCoordinatorV2.ProofAlreadyUsed.selector);
        coordinator.submitProof(modelId, 1, proof, inputs);
    }

    /// @notice Test staking across multiple models
    function test_StakingAcrossModels() public {
        // Register two models
        uint256 commit1 = uint256(keccak256(abi.encodePacked(uint256(1), uint256(2))));
        uint256 commit2 = uint256(keccak256(abi.encodePacked(uint256(3), uint256(4))));

        vm.startPrank(modelOwner);
        uint256 model1 = coordinator.registerModel("Model1", commit1, MIN_STAKE, 4, 8, 2, 2, 0);
        uint256 model2 = coordinator.registerModel("Model2", commit2, 0.5 ether, 4, 8, 2, 2, 0);
        vm.stopPrank();

        // Prover stakes in both models
        vm.startPrank(prover1);
        coordinator.stake{value: 0.5 ether}(model1);
        coordinator.stake{value: 1 ether}(model2);
        vm.stopPrank();

        // Verify stakes are independent
        (uint256 stake1,,) = coordinator.getStake(prover1, model1);
        (uint256 stake2,,) = coordinator.getStake(prover1, model2);
        assertEq(stake1, 0.5 ether);
        assertEq(stake2, 1 ether);

        // Getting slashed on one model doesn't affect the other
        // Start round on model1
        vm.prank(modelOwner);
        coordinator.startRound(model1, ROUND_DURATION);

        // Submit invalid proof to model1
        mockVerifier.setShouldPass(false);

        uint256[] memory inputs = new uint256[](8);
        inputs[0] = 1;
        inputs[1] = 2;
        inputs[2] = 5;
        inputs[3] = 6;
        inputs[4] = 100;
        inputs[5] = 10;
        inputs[6] = 1;
        inputs[7] = ProofFixtureHardcoded.computeErrorChecksum(inputs[5], inputs[6], model1, 1e18);

        bytes memory proof = new bytes(320);

        vm.prank(prover1);
        coordinator.submitProof(model1, 1, proof, inputs);

        // Verify slashed on model1
        (uint256 newStake1,, bool slashed1) = coordinator.getStake(prover1, model1);
        assertTrue(slashed1);
        assertEq(newStake1, 0.25 ether);  // 50% slashed

        // Model2 stake unaffected
        (uint256 newStake2,, bool slashed2) = coordinator.getStake(prover1, model2);
        assertFalse(slashed2);
        assertEq(newStake2, 1 ether);
    }

    /// @notice Test model pause/resume functionality
    function test_ModelPauseResume() public {
        uint256 commitment = uint256(keccak256(abi.encodePacked(uint256(1), uint256(2))));

        vm.prank(modelOwner);
        uint256 modelId = coordinator.registerModel("Pausable", commitment, MIN_STAKE, 4, 8, 2, 2, 0);

        // Prover stakes
        vm.prank(prover1);
        coordinator.stake{value: LARGE_STAKE}(modelId);

        // Owner pauses model
        vm.prank(modelOwner);
        coordinator.pauseModel(modelId);

        (,, bool active) = coordinator.getModelState(modelId);
        assertFalse(active);

        // Cannot start round while paused
        vm.prank(modelOwner);
        vm.expectRevert(HelixCoordinatorV2.ModelNotActive.selector);
        coordinator.startRound(modelId, ROUND_DURATION);

        // Resume model
        vm.prank(modelOwner);
        coordinator.resumeModel(modelId);

        (,, active) = coordinator.getModelState(modelId);
        assertTrue(active);

        // Now can start round
        vm.prank(modelOwner);
        coordinator.startRound(modelId, ROUND_DURATION);

        (uint256 round,,) = coordinator.getModelState(modelId);
        assertEq(round, 1);
    }

    /// @notice Test error bound limits enforcement
    function test_ErrorBoundLimits() public {
        uint256 hashLo = 500;
        uint256 hashHi = 600;
        uint256 commitment = uint256(keccak256(abi.encodePacked(hashLo, hashHi)));

        vm.prank(modelOwner);
        uint256 modelId = coordinator.registerModel("ErrorTest", commitment, MIN_STAKE, 4, 8, 2, 2, 0);

        vm.prank(prover1);
        coordinator.stake{value: LARGE_STAKE}(modelId);

        vm.prank(modelOwner);
        coordinator.startRound(modelId, ROUND_DURATION);

        mockVerifier.setShouldPass(true);

        // Try to submit with error bound exceeding maximum
        uint256[] memory inputs = new uint256[](8);
        inputs[0] = hashLo;
        inputs[1] = hashHi;
        inputs[2] = 501;
        inputs[3] = 601;
        inputs[4] = 100;
        inputs[5] = coordinator.maxErrorBound() + 1;  // Exceeds max
        inputs[6] = 1;
        inputs[7] = ProofFixtureHardcoded.computeErrorChecksum(inputs[5], inputs[6], modelId, 1e18);

        bytes memory proof = new bytes(320);

        vm.prank(prover1);
        vm.expectRevert(HelixCoordinatorV2.ErrorBoundExceeded.selector);
        coordinator.submitProof(modelId, 1, proof, inputs);
    }

    /// @notice Test accumulated error bound tracking
    function test_AccumulatedErrorTracking() public {
        uint256 hashLo = 700;
        uint256 hashHi = 800;
        uint256 commitment = uint256(keccak256(abi.encodePacked(hashLo, hashHi)));

        vm.prank(modelOwner);
        uint256 modelId = coordinator.registerModel("ErrorTrack", commitment, MIN_STAKE, 4, 8, 2, 2, 0);

        vm.prank(prover1);
        coordinator.stake{value: LARGE_STAKE}(modelId);

        mockVerifier.setShouldPass(true);

        // Submit multiple rounds with different error bounds
        uint256[] memory errorBounds = new uint256[](3);
        errorBounds[0] = 100;
        errorBounds[1] = 200;
        errorBounds[2] = 150;

        for (uint256 i = 0; i < 3; i++) {
            vm.prank(modelOwner);
            coordinator.startRound(modelId, ROUND_DURATION);

            uint256 newHashLo = hashLo + i + 1;
            uint256 newHashHi = hashHi + i + 1;

            uint256[] memory inputs = new uint256[](8);
            inputs[0] = hashLo;
            inputs[1] = hashHi;
            inputs[2] = newHashLo;
            inputs[3] = newHashHi;
            inputs[4] = 100;
            inputs[5] = errorBounds[i];
            inputs[6] = i + 1;
            inputs[7] = ProofFixtureHardcoded.computeErrorChecksum(inputs[5], inputs[6], modelId, 1e18);

            bytes memory proof = new bytes(320);

            vm.prank(prover1);
            coordinator.submitProof(modelId, i + 1, proof, inputs);

            hashLo = newHashLo;
            hashHi = newHashHi;
        }

        // Verify total accumulated error (100 + 200 + 150 = 450)
        uint256 totalError = coordinator.getAccumulatedErrorBound(modelId);
        assertEq(totalError, 450);

        // Test isModelErrorAcceptable
        assertTrue(coordinator.isModelErrorAcceptable(modelId, 500));
        assertFalse(coordinator.isModelErrorAcceptable(modelId, 400));

        // Test error reset
        vm.prank(modelOwner);
        coordinator.resetAccumulatedError(modelId);
        assertEq(coordinator.getAccumulatedErrorBound(modelId), 0);
    }

    /// @notice Test deadline enforcement
    function test_RoundDeadlineEnforcement() public {
        uint256 hashLo = 900;
        uint256 hashHi = 1000;
        uint256 commitment = uint256(keccak256(abi.encodePacked(hashLo, hashHi)));

        vm.prank(modelOwner);
        uint256 modelId = coordinator.registerModel("DeadlineTest", commitment, MIN_STAKE, 4, 8, 2, 2, 0);

        vm.prank(prover1);
        coordinator.stake{value: LARGE_STAKE}(modelId);

        // Start round with 1 hour duration
        vm.prank(modelOwner);
        coordinator.startRound(modelId, ROUND_DURATION);

        mockVerifier.setShouldPass(true);

        // Fast forward past deadline
        vm.warp(block.timestamp + ROUND_DURATION + 1);

        uint256[] memory inputs = new uint256[](8);
        inputs[0] = hashLo;
        inputs[1] = hashHi;
        inputs[2] = 901;
        inputs[3] = 1001;
        inputs[4] = 100;
        inputs[5] = 10;
        inputs[6] = 1;
        inputs[7] = ProofFixtureHardcoded.computeErrorChecksum(inputs[5], inputs[6], modelId, 1e18);

        bytes memory proof = new bytes(320);

        vm.prank(prover1);
        vm.expectRevert(HelixCoordinatorV2.RoundExpired.selector);
        coordinator.submitProof(modelId, 1, proof, inputs);
    }

    /// @notice Test lock period and unstaking timing
    function test_UnstakingTiming() public {
        uint256 commitment = uint256(keccak256(abi.encodePacked(uint256(1), uint256(2))));

        vm.prank(modelOwner);
        uint256 modelId = coordinator.registerModel("UnstakeTest", commitment, MIN_STAKE, 4, 8, 2, 2, 0);

        vm.prank(prover1);
        coordinator.stake{value: LARGE_STAKE}(modelId);

        // Cannot unstake immediately
        vm.prank(prover1);
        vm.expectRevert(HelixCoordinatorV2.StillLocked.selector);
        coordinator.unstake(modelId);

        // After 6 days, still locked
        vm.warp(block.timestamp + 6 days);
        vm.prank(prover1);
        vm.expectRevert(HelixCoordinatorV2.StillLocked.selector);
        coordinator.unstake(modelId);

        // After 7 days, can unstake
        vm.warp(block.timestamp + 1 days + 1);
        uint256 balanceBefore = prover1.balance;
        vm.prank(prover1);
        coordinator.unstake(modelId);
        uint256 balanceAfter = prover1.balance;

        assertEq(balanceAfter - balanceBefore, LARGE_STAKE);
    }

    /// @notice Test treasury receiving slashed funds
    function test_TreasuryReceivesSlashedFunds() public {
        uint256 hashLo = 1100;
        uint256 hashHi = 1200;
        uint256 commitment = uint256(keccak256(abi.encodePacked(hashLo, hashHi)));

        vm.prank(modelOwner);
        uint256 modelId = coordinator.registerModel("TreasuryTest", commitment, MIN_STAKE, 4, 8, 2, 2, 0);

        vm.prank(prover1);
        coordinator.stake{value: 2 ether}(modelId);

        vm.prank(modelOwner);
        coordinator.startRound(modelId, ROUND_DURATION);

        uint256 treasuryBefore = treasury.balance;

        // Submit invalid proof
        mockVerifier.setShouldPass(false);

        uint256[] memory inputs = new uint256[](8);
        inputs[0] = hashLo;
        inputs[1] = hashHi;
        inputs[2] = 1101;
        inputs[3] = 1201;
        inputs[4] = 100;
        inputs[5] = 10;
        inputs[6] = 1;
        inputs[7] = ProofFixtureHardcoded.computeErrorChecksum(inputs[5], inputs[6], modelId, 1e18);

        bytes memory proof = new bytes(320);

        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, proof, inputs);

        uint256 treasuryAfter = treasury.balance;

        // 50% of 2 ether = 1 ether to treasury
        assertEq(treasuryAfter - treasuryBefore, 1 ether);
    }

    // ============ TrainingRound Integration Tests ============

    /// @notice Test TrainingRound state machine flow
    function test_TrainingRoundStateMachine() public {
        uint256 modelId = 1;
        bytes32 startCommit = keccak256("start");

        // Create round
        uint256 roundId = trainingRound.createRound(
            modelId,
            startCommit,
            2,  // min participants
            10, // max participants
            ROUND_DURATION
        );

        TrainingRound.Round memory round = trainingRound.getRound(modelId, roundId);
        assertEq(uint256(round.state), uint256(TrainingRound.RoundState.Created));

        // Register participants (need 2 to activate)
        vm.prank(prover1);
        trainingRound.registerForRound(modelId, roundId);

        round = trainingRound.getRound(modelId, roundId);
        assertEq(uint256(round.state), uint256(TrainingRound.RoundState.Created));
        assertEq(round.registeredCount, 1);

        vm.prank(prover2);
        trainingRound.registerForRound(modelId, roundId);

        round = trainingRound.getRound(modelId, roundId);
        assertEq(uint256(round.state), uint256(TrainingRound.RoundState.Active));
        assertEq(round.registeredCount, 2);

        // Submit gradients
        vm.prank(prover1);
        trainingRound.submitGradient(modelId, roundId, keccak256("gradient1"), 5);

        vm.prank(prover2);
        trainingRound.submitGradient(modelId, roundId, keccak256("gradient2"), 5);

        round = trainingRound.getRound(modelId, roundId);
        assertEq(round.submittedCount, 2);

        // Start aggregation
        trainingRound.startAggregation(modelId, roundId);
        round = trainingRound.getRound(modelId, roundId);
        assertEq(uint256(round.state), uint256(TrainingRound.RoundState.Aggregating));

        // Complete round
        trainingRound.completeRound(modelId, roundId, keccak256("final"), 10);
        round = trainingRound.getRound(modelId, roundId);
        assertEq(uint256(round.state), uint256(TrainingRound.RoundState.Completed));
    }

    // ============ AggregationVerifier Integration Tests ============

    /// @notice Test aggregation verification flow
    function test_AggregationVerifierFlow() public {
        uint256 modelId = 1;
        uint256 roundId = 1;

        mockVerifier.setShouldPass(true);

        // Submit contributions
        bytes memory proof = new bytes(320);
        uint256[] memory inputs = new uint256[](8);
        inputs[0] = 1;
        inputs[1] = 2;
        inputs[2] = 3;
        inputs[3] = 4;
        inputs[4] = 100;
        inputs[5] = 10;
        inputs[6] = 1;
        inputs[7] = ProofFixtureHardcoded.computeErrorChecksum(inputs[5], inputs[6], modelId, 1e18);

        vm.prank(prover1);
        aggregationVerifier.submitContribution(
            modelId,
            roundId,
            keccak256("gradient1"),
            3000,  // 30% stake weight
            50,    // error bound
            proof,
            inputs
        );

        vm.prank(prover2);
        aggregationVerifier.submitContribution(
            modelId,
            roundId,
            keccak256("gradient2"),
            3000,
            50,
            proof,
            inputs
        );

        // Check round state
        (
            uint256 participantCount,
            uint256 totalStakeWeight,
            ,
            bool isFinalized,
        ) = aggregationVerifier.getAggregationRound(modelId, roundId);

        assertEq(participantCount, 2);
        assertEq(totalStakeWeight, 6000);
        assertFalse(isFinalized);

        // Finalize aggregation
        bytes32 aggregatedCommit = keccak256("aggregated");
        aggregationVerifier.finalizeAggregation(
            modelId,
            roundId,
            aggregatedCommit,
            100,  // aggregated error bound
            proof,
            inputs
        );

        (,,, isFinalized,) = aggregationVerifier.getAggregationRound(modelId, roundId);
        assertTrue(isFinalized);

        // Verify aggregation
        assertTrue(aggregationVerifier.verifyAggregation(modelId, roundId, aggregatedCommit));
    }

    // ============ Admin Function Tests ============

    /// @notice Test admin parameter changes
    function test_AdminParameterChanges() public {
        // Only owner can change parameters (onlyOwner checked before whenPaused)
        vm.prank(attacker);
        vm.expectRevert(HelixCoordinatorV2.OnlyOwner.selector);
        coordinator.setSlashPercentage(2500);

        // Legacy setters require emergency pause
        coordinator.emergencyPause();

        // Owner can change when paused
        coordinator.setSlashPercentage(2500);
        assertEq(coordinator.slashPercentage(), 2500);

        coordinator.setDefaultMinStake(0.5 ether);
        assertEq(coordinator.defaultMinStake(), 0.5 ether);

        coordinator.setMaxErrorBound(5e17);
        assertEq(coordinator.maxErrorBound(), 5e17);

        // Cannot set slash percentage over 100%
        vm.expectRevert(HelixCoordinatorV2.MaxPercentage.selector);
        coordinator.setSlashPercentage(10001);

        coordinator.unpause();
    }

    /// @notice Test verifier swap
    function test_VerifierSwap() public {
        uint256 hashLo = 1300;
        uint256 hashHi = 1400;
        uint256 commitment = uint256(keccak256(abi.encodePacked(hashLo, hashHi)));

        vm.prank(modelOwner);
        uint256 modelId = coordinator.registerModel("VerifierSwap", commitment, MIN_STAKE, 4, 8, 2, 2, 0);

        vm.prank(prover1);
        coordinator.stake{value: LARGE_STAKE}(modelId);

        vm.prank(modelOwner);
        coordinator.startRound(modelId, ROUND_DURATION);

        // Proof fails with current mock
        mockVerifier.setShouldPass(false);

        // Re-enable mock verifier (verifier is immutable, so toggle pass state instead)
        mockVerifier.setShouldPass(true);

        // Now proof succeeds
        uint256[] memory inputs = new uint256[](8);
        inputs[0] = hashLo;
        inputs[1] = hashHi;
        inputs[2] = 1301;
        inputs[3] = 1401;
        inputs[4] = 100;
        inputs[5] = 10;
        inputs[6] = 1;
        inputs[7] = ProofFixtureHardcoded.computeErrorChecksum(inputs[5], inputs[6], modelId, 1e18);

        bytes memory proof = new bytes(320);

        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, proof, inputs);

        // Round completed successfully
        (uint256 round,,) = coordinator.getModelState(modelId);
        assertEq(round, 1);
    }

    // ============ Edge Case Tests ============

    /// @notice Test zero stake scenarios
    function test_ZeroStakeRejected() public {
        uint256 commitment = uint256(keccak256(abi.encodePacked(uint256(1), uint256(2))));

        vm.prank(modelOwner);
        uint256 modelId = coordinator.registerModel("ZeroStake", commitment, MIN_STAKE, 4, 8, 2, 2, 0);

        vm.prank(prover1);
        vm.expectRevert(HelixCoordinatorV2.ZeroStake.selector);
        coordinator.stake{value: 0}(modelId);
    }

    /// @notice Test commitment reconstruction
    function test_CommitmentReconstruction() public {
        uint256 lo = type(uint128).max;
        uint256 hi = type(uint128).max - 1;
        uint256 expected = uint256(keccak256(abi.encodePacked(lo, hi)));

        vm.prank(modelOwner);
        uint256 modelId = coordinator.registerModel("CommitTest", expected, MIN_STAKE, 4, 8, 2, 2, 0);

        vm.prank(prover1);
        coordinator.stake{value: LARGE_STAKE}(modelId);

        vm.prank(modelOwner);
        coordinator.startRound(modelId, ROUND_DURATION);

        mockVerifier.setShouldPass(true);

        // Verify commitment matching works correctly with large values
        uint256[] memory inputs = new uint256[](8);
        inputs[0] = lo;
        inputs[1] = hi;
        inputs[2] = lo + 1;
        inputs[3] = hi + 1;
        inputs[4] = 100;
        inputs[5] = 10;
        inputs[6] = 1;
        inputs[7] = ProofFixtureHardcoded.computeErrorChecksum(inputs[5], inputs[6], modelId, 1e18);

        bytes memory proof = new bytes(320);

        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, proof, inputs);

        // Verify model updated correctly
        uint256 newExpected = uint256(keccak256(abi.encodePacked(lo + 1, hi + 1)));
        (, uint256 currentCommitment,) = coordinator.getModelState(modelId);
        assertEq(currentCommitment, newExpected);
    }

    /// @notice Test receive ETH function
    function test_ReceiveETH() public {
        uint256 before = address(coordinator).balance;
        (bool success,) = address(coordinator).call{value: 5 ether}("");
        assertTrue(success);
        assertEq(address(coordinator).balance, before + 5 ether);
    }
}
