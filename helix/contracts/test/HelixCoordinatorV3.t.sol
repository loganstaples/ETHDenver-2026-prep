// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "forge-std/Test.sol";
import "../src/core/HelixCoordinatorV3.sol";
import "../src/core/ModelRegistry.sol";
import "../src/verification/Halo2Verifier.sol";
import "../src/token/HelixToken.sol";
import "../src/token/Staking.sol";
import "../src/token/Rewards.sol";
import "./ProofFixtures.t.sol";

/// @title HelixCoordinatorV3Test
/// @notice Integration tests for HelixCoordinatorV3 with full token stack
contract HelixCoordinatorV3Test is Test {
    HelixCoordinatorV3 public coordinator;
    Halo2Verifier public verifier;
    HelixToken public token;
    Staking public staking;
    Rewards public rewards;
    ModelRegistry public registry;
    MockVerifierForV3Test public mockVerifier;

    address public owner;
    address public treasuryAddr;
    address public prover1;
    address public prover2;
    address public challenger;

    uint256 constant STAKE_AMOUNT = 200e18;
    uint256 constant MIN_STAKE = 100e18;
    uint256 constant ROUND_DURATION = 1 hours;

    event ModelRegistered(uint256 indexed modelId, address indexed owner, uint256 initialCommitment, string ipfsHash);
    event RoundStarted(uint256 indexed modelId, uint256 indexed roundId, uint256 deadline, uint256 modelCommitment);
    event ProofSubmitted(uint256 indexed modelId, uint256 indexed roundId, address indexed prover, uint256 newCommitment, uint256 errorBound);
    event RoundCompleted(uint256 indexed modelId, uint256 indexed roundId, uint256 newCommitment, uint256 totalErrorBound);
    event InvalidProofDetected(uint256 indexed modelId, uint256 indexed roundId, address indexed prover, bytes32 proofHash);
    event ProofReplayBlocked(bytes32 indexed proofHash, address indexed submitter);

    function setUp() public {
        owner = address(this);
        treasuryAddr = makeAddr("treasury");
        prover1 = makeAddr("prover1");
        prover2 = makeAddr("prover2");
        challenger = makeAddr("challenger");

        // Deploy token
        token = new HelixToken(treasuryAddr);

        // Deploy mock verifier (accepts all proofs by default)
        mockVerifier = new MockVerifierForV3Test();

        // Deploy staking
        staking = new Staking(address(token), MIN_STAKE, 7 days, 5000);

        // Deploy rewards
        rewards = new Rewards(address(token));

        // Deploy registry
        registry = new ModelRegistry();

        // Deploy coordinator
        coordinator = new HelixCoordinatorV3(
            address(mockVerifier),
            address(staking),
            address(rewards),
            address(registry),
            treasuryAddr
        );

        // Wire contracts
        staking.setOperator(address(coordinator));
        rewards.setCoordinator(address(coordinator));
        rewards.setStakingContract(address(staking));
        registry.setCoordinator(address(coordinator));

        // Fund provers with HELIX tokens
        token.mint(prover1, 1000e18);
        token.mint(prover2, 1000e18);

        // Provers approve and stake
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

    // ============ Constructor Tests ============

    function test_Constructor_SetsAddresses() public view {
        assertEq(address(coordinator.verifier()), address(mockVerifier));
        assertEq(address(coordinator.stakingContract()), address(staking));
        assertEq(address(coordinator.rewardsContract()), address(rewards));
        assertEq(address(coordinator.modelRegistry()), address(registry));
        assertEq(coordinator.treasury(), treasuryAddr);
        assertEq(coordinator.owner(), owner);
    }

    function test_Constructor_RejectsZeroAddresses() public {
        vm.expectRevert("Invalid verifier");
        new HelixCoordinatorV3(address(0), address(staking), address(rewards), address(registry), treasuryAddr);

        vm.expectRevert("Invalid staking");
        new HelixCoordinatorV3(address(mockVerifier), address(0), address(rewards), address(registry), treasuryAddr);

        vm.expectRevert("Invalid rewards");
        new HelixCoordinatorV3(address(mockVerifier), address(staking), address(0), address(registry), treasuryAddr);

        vm.expectRevert("Invalid registry");
        new HelixCoordinatorV3(address(mockVerifier), address(staking), address(rewards), address(0), treasuryAddr);

        vm.expectRevert("Invalid treasury");
        new HelixCoordinatorV3(address(mockVerifier), address(staking), address(rewards), address(registry), address(0));
    }

    // ============ Model Registration Tests ============

    function test_RegisterModel() public {
        uint256 modelId = coordinator.registerModel("TestModel", "A test model", "QmHash123", 12345, 4, 8, 2, 2, 0);
        assertEq(modelId, 0);

        (uint256 round, uint256 commitment, bool active) = coordinator.getModelState(modelId);
        assertEq(round, 0);
        assertEq(commitment, 12345);
        assertTrue(active);
    }

    function test_RegisterModel_CreatesCheckpoint() public {
        coordinator.registerModel("TestModel", "A test model", "QmHash123", 12345, 4, 8, 2, 2, 0);

        // Verify ModelRegistry has the checkpoint
        uint256 checkpointCount = registry.getCheckpointCount(0);
        assertEq(checkpointCount, 1);
    }

    function test_RegisterMultipleModels() public {
        uint256 m1 = coordinator.registerModel("Model1", "desc1", "hash1", 100, 4, 8, 2, 2, 0);
        uint256 m2 = coordinator.registerModel("Model2", "desc2", "hash2", 200, 4, 8, 2, 2, 0);
        assertEq(m1, 0);
        assertEq(m2, 1);
    }

    // ============ Round Tests ============

    function test_StartRound() public {
        uint256 modelId = coordinator.registerModel("Model", "desc", "hash", 100, 4, 8, 2, 2, 0);
        coordinator.startRound(modelId, ROUND_DURATION);

        (uint256 round,,) = coordinator.getModelState(modelId);
        assertEq(round, 1);
    }

    function test_StartRound_OnlyModelOwner() public {
        uint256 modelId = coordinator.registerModel("Model", "desc", "hash", 100, 4, 8, 2, 2, 0);

        vm.prank(prover1);
        vm.expectRevert("Only model owner");
        coordinator.startRound(modelId, ROUND_DURATION);
    }

    // ============ Proof Submission (Valid Proof) ============

    function test_SubmitProof_ValidAccepted() public {
        // Set up commitment
        uint256 oldHashLo = 12345;
        uint256 oldHashHi = 67890;
        uint256 correctCommitment = uint256(keccak256(abi.encodePacked(oldHashLo, oldHashHi)));

        uint256 modelId = coordinator.registerModel("Model", "desc", "hash", correctCommitment, 4, 8, 2, 2, 0);
        coordinator.startRound(modelId, ROUND_DURATION);

        // Build public inputs
        uint256[] memory publicInputs = new uint256[](8);
        publicInputs[0] = oldHashLo;
        publicInputs[1] = oldHashHi;
        publicInputs[2] = 99999;
        publicInputs[3] = 88888;
        publicInputs[4] = 100; // loss
        publicInputs[5] = 10;  // error bound
        publicInputs[6] = 1;   // step
        publicInputs[7] = _computeChecksum(10, 1, modelId, coordinator.maxErrorBound());

        bytes memory proof = ProofFixtureHardcoded.createValidProof();

        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, proof, publicInputs);

        // Verify state updated
        uint256 expectedNewCommitment = uint256(keccak256(abi.encodePacked(uint256(99999), uint256(88888))));
        (, uint256 commitment, ) = coordinator.getModelState(modelId);
        assertEq(commitment, expectedNewCommitment);

        // Verify error bound accumulated
        assertEq(coordinator.getAccumulatedErrorBound(modelId), 10);
    }

    function test_SubmitProof_UpdatesRegistry() public {
        uint256 oldHashLo = 12345;
        uint256 oldHashHi = 67890;
        uint256 correctCommitment = uint256(keccak256(abi.encodePacked(oldHashLo, oldHashHi)));

        uint256 modelId = coordinator.registerModel("Model", "desc", "hash", correctCommitment, 4, 8, 2, 2, 0);
        coordinator.startRound(modelId, ROUND_DURATION);

        uint256[] memory publicInputs = _buildPublicInputs(oldHashLo, oldHashHi, 99999, 88888, 100, 10, 1, modelId);
        bytes memory proof = ProofFixtureHardcoded.createValidProof();

        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, proof, publicInputs);

        // ModelRegistry should have 2 checkpoints (initial + update)
        uint256 checkpointCount = registry.getCheckpointCount(modelId);
        assertEq(checkpointCount, 2);
    }

    // ============ Proof Replay Protection ============

    function test_SubmitProof_ReplayBlocked() public {
        uint256 oldHashLo = 12345;
        uint256 oldHashHi = 67890;
        uint256 correctCommitment = uint256(keccak256(abi.encodePacked(oldHashLo, oldHashHi)));

        uint256 modelId = coordinator.registerModel("Model", "desc", "hash", correctCommitment, 4, 8, 2, 2, 0);
        coordinator.startRound(modelId, ROUND_DURATION);

        uint256[] memory publicInputs = _buildPublicInputs(oldHashLo, oldHashHi, 99999, 88888, 100, 10, 1, modelId);
        bytes memory proof = ProofFixtureHardcoded.createValidProof();

        // First submission succeeds
        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, proof, publicInputs);

        // Start a new round with the new commitment
        uint256 newCommitment = uint256(keccak256(abi.encodePacked(uint256(99999), uint256(88888))));
        coordinator.startRound(modelId, ROUND_DURATION);

        // Attempt replay with same proof+inputs
        vm.prank(prover2);
        vm.expectRevert("Proof already used");
        coordinator.submitProof(modelId, 2, proof, publicInputs);
    }

    function test_IsProofUsed() public {
        bytes memory proof = ProofFixtureHardcoded.createValidProof();
        uint256[] memory publicInputs = new uint256[](8);
        bytes32 proofHash = keccak256(abi.encodePacked(proof, publicInputs));

        assertFalse(coordinator.isProofUsed(proofHash));
    }

    // ============ Slashing via Staking.sol ============

    function test_SubmitProof_InvalidSlashesViaStaking() public {
        uint256 oldHashLo = 12345;
        uint256 oldHashHi = 67890;
        uint256 correctCommitment = uint256(keccak256(abi.encodePacked(oldHashLo, oldHashHi)));

        uint256 modelId = coordinator.registerModel("Model", "desc", "hash", correctCommitment, 4, 8, 2, 2, 0);
        coordinator.startRound(modelId, ROUND_DURATION);

        uint256[] memory publicInputs = _buildPublicInputs(oldHashLo, oldHashHi, 99999, 88888, 100, 10, 1, modelId);
        bytes memory proof = ProofFixtureHardcoded.createValidProof();

        // Make verifier reject
        mockVerifier.setAccept(false);

        uint256 stakeBefore = _getStakeAmount(prover1);

        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, proof, publicInputs);

        // Staking.sol should have slashed
        uint256 stakeAfter = _getStakeAmount(prover1);
        assertLt(stakeAfter, stakeBefore);

        // Slashing record should exist
        assertEq(coordinator.getSlashingRecordCount(), 1);
    }

    // ============ Challenge Tests ============

    function test_ChallengeProof_SlashesViaStaking() public {
        // Setup and submit a valid proof
        uint256 oldHashLo = 12345;
        uint256 oldHashHi = 67890;
        uint256 correctCommitment = uint256(keccak256(abi.encodePacked(oldHashLo, oldHashHi)));

        uint256 modelId = coordinator.registerModel("Model", "desc", "hash", correctCommitment, 4, 8, 2, 2, 0);
        coordinator.startRound(modelId, ROUND_DURATION);

        uint256[] memory publicInputs = _buildPublicInputs(oldHashLo, oldHashHi, 99999, 88888, 100, 10, 1, modelId);
        bytes memory proof = ProofFixtureHardcoded.createValidProof();

        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, proof, publicInputs);

        // Now make verifier reject for the challenge
        mockVerifier.setAccept(false);

        // Challenger challenges
        vm.prank(challenger);
        coordinator.challengeProof(modelId, 1, proof, publicInputs);

        // Prover1 should be slashed
        assertEq(coordinator.getSlashingRecordCount(), 1);
    }

    function test_ChallengeProof_CannotChallengeSelf() public {
        uint256 oldHashLo = 12345;
        uint256 oldHashHi = 67890;
        uint256 correctCommitment = uint256(keccak256(abi.encodePacked(oldHashLo, oldHashHi)));

        uint256 modelId = coordinator.registerModel("Model", "desc", "hash", correctCommitment, 4, 8, 2, 2, 0);
        coordinator.startRound(modelId, ROUND_DURATION);

        uint256[] memory publicInputs = _buildPublicInputs(oldHashLo, oldHashHi, 99999, 88888, 100, 10, 1, modelId);
        bytes memory proof = ProofFixtureHardcoded.createValidProof();

        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, proof, publicInputs);

        vm.prank(prover1);
        vm.expectRevert("Cannot challenge self");
        coordinator.challengeProof(modelId, 1, proof, publicInputs);
    }

    // ============ Staking Check Tests ============

    function test_SubmitProof_RequiresTokenStake() public {
        address unstaked = makeAddr("unstaked");
        uint256 modelId = coordinator.registerModel("Model", "desc", "hash", 100, 4, 8, 2, 2, 0);
        coordinator.startRound(modelId, ROUND_DURATION);

        bytes memory proof = new bytes(320);
        uint256[] memory publicInputs = new uint256[](8);

        vm.prank(unstaked);
        vm.expectRevert("Insufficient stake or not active");
        coordinator.submitProof(modelId, 1, proof, publicInputs);
    }

    // ============ Treasury Zero-Address Protection ============

    function test_SetTreasury_RejectsZeroAddress() public {
        vm.expectRevert("Invalid treasury");
        coordinator.setTreasury(address(0));
    }

    // ============ Pause Tests ============

    function test_EmergencyPause() public {
        coordinator.emergencyPause();
        assertTrue(coordinator.paused());

        // Cannot register model when paused
        vm.expectRevert("Contract is paused");
        coordinator.registerModel("Model", "desc", "hash", 100, 4, 8, 2, 2, 0);
    }

    function test_Unpause() public {
        coordinator.emergencyPause();
        coordinator.unpause();
        assertFalse(coordinator.paused());
    }

    // ============ Admin Tests ============

    function test_SetStakingContract() public {
        Staking newStaking = new Staking(address(token), MIN_STAKE, 7 days, 5000);
        coordinator.setStakingContract(address(newStaking));
        assertEq(address(coordinator.stakingContract()), address(newStaking));
    }

    function test_SetRewardsContract() public {
        Rewards newRewards = new Rewards(address(token));
        coordinator.setRewardsContract(address(newRewards));
        assertEq(address(coordinator.rewardsContract()), address(newRewards));
    }

    function test_SetModelRegistry() public {
        ModelRegistry newRegistry = new ModelRegistry();
        coordinator.setModelRegistry(address(newRegistry));
        assertEq(address(coordinator.modelRegistry()), address(newRegistry));
    }

    function test_PauseModel() public {
        uint256 modelId = coordinator.registerModel("Model", "desc", "hash", 100, 4, 8, 2, 2, 0);
        coordinator.pauseModel(modelId);

        (,, bool active) = coordinator.getModelState(modelId);
        assertFalse(active);

        coordinator.resumeModel(modelId);
        (,, active) = coordinator.getModelState(modelId);
        assertTrue(active);
    }

    // ============ Error Bound Tests ============

    function test_ErrorBoundAccumulates() public {
        uint256 oldHashLo = 12345;
        uint256 oldHashHi = 67890;
        uint256 correctCommitment = uint256(keccak256(abi.encodePacked(oldHashLo, oldHashHi)));

        uint256 modelId = coordinator.registerModel("Model", "desc", "hash", correctCommitment, 4, 8, 2, 2, 0);
        coordinator.startRound(modelId, ROUND_DURATION);

        uint256[] memory publicInputs = _buildPublicInputs(oldHashLo, oldHashHi, 99999, 88888, 100, 10, 1, modelId);
        bytes memory proof = ProofFixtureHardcoded.createValidProof();

        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, proof, publicInputs);
        assertEq(coordinator.getAccumulatedErrorBound(modelId), 10);
        assertTrue(coordinator.isModelErrorAcceptable(modelId, 100));
    }

    function test_ErrorBoundExceedsMaxReverts() public {
        uint256 oldHashLo = 12345;
        uint256 oldHashHi = 67890;
        uint256 correctCommitment = uint256(keccak256(abi.encodePacked(oldHashLo, oldHashHi)));

        uint256 modelId = coordinator.registerModel("Model", "desc", "hash", correctCommitment, 4, 8, 2, 2, 0);
        coordinator.startRound(modelId, ROUND_DURATION);

        // Set error bound higher than max
        coordinator.setMaxErrorBound(5);

        uint256[] memory publicInputs = _buildPublicInputs(oldHashLo, oldHashHi, 99999, 88888, 100, 10, 1, modelId);
        bytes memory proof = ProofFixtureHardcoded.createValidProof();

        vm.prank(prover1);
        vm.expectRevert("Error bound exceeds maximum");
        coordinator.submitProof(modelId, 1, proof, publicInputs);
    }

    // ============ Guardian Tests ============

    function test_AddGuardian() public {
        address guardian2 = makeAddr("guardian2");
        coordinator.addGuardian(guardian2);
        assertTrue(coordinator.getGuardianStatus(guardian2));
    }

    function test_RemoveGuardian() public {
        address guardian2 = makeAddr("guardian2");
        address guardian3 = makeAddr("guardian3");
        coordinator.addGuardian(guardian2);
        coordinator.addGuardian(guardian3);

        coordinator.removeGuardian(guardian2);
        assertFalse(coordinator.getGuardianStatus(guardian2));
    }

    // ============ Data Commitment Tests ============

    function test_SetModelDataCommitment() public {
        uint256 modelId = coordinator.registerModel("Model", "desc", "hash", 100, 4, 8, 2, 2, 0);
        bytes32 dataRoot = keccak256("training_data");
        coordinator.setModelDataCommitment(modelId, dataRoot);
        assertEq(coordinator.getModelDataCommitment(modelId), dataRoot);
    }

    // ============ View Functions ============

    function test_GetSlashingRecordCount() public view {
        assertEq(coordinator.getSlashingRecordCount(), 0);
    }

    function test_GetChallengerConfig() public view {
        (uint16 pct, bool enabled) = coordinator.getChallengerConfig();
        assertEq(pct, 1000); // 10%
        assertTrue(enabled);
    }

    // ============ Helpers ============

    function _buildPublicInputs(
        uint256 oldHashLo, uint256 oldHashHi,
        uint256 newHashLo, uint256 newHashHi,
        uint256 loss, uint256 errorBound,
        uint256 step, uint256 modelId
    ) internal view returns (uint256[] memory) {
        uint256[] memory inputs = new uint256[](8);
        inputs[0] = oldHashLo;
        inputs[1] = oldHashHi;
        inputs[2] = newHashLo;
        inputs[3] = newHashHi;
        inputs[4] = loss;
        inputs[5] = errorBound;
        inputs[6] = step;
        inputs[7] = _computeChecksum(errorBound, step, modelId, coordinator.maxErrorBound());
        return inputs;
    }

    function _computeChecksum(
        uint256 errorBound, uint256 stepNumber, uint256 modelId, uint256 errorBudget
    ) internal pure returns (uint256) {
        return ProofFixtureHardcoded.computeErrorChecksum(errorBound, stepNumber, modelId, errorBudget);
    }

    function _getStakeAmount(address staker) internal view returns (uint256) {
        (uint256 amount,,,) = staking.getStakeInfo(staker);
        return amount;
    }

    // ============ Checkpoint Interval Tests ============

    function test_RegisterModelDefaultCheckpointInterval() public {
        uint256 modelId = coordinator.registerModel("Model", "desc", "hash", 100, 4, 8, 2, 2, 0);
        assertEq(coordinator.getCheckpointInterval(modelId), 1);
        assertEq(coordinator.modelCheckpointInterval(modelId), 1);
    }

    function test_RegisterModelWithCheckpointInterval() public {
        uint256 oldHashLo = 12345;
        uint256 oldHashHi = 67890;
        uint256 correctCommitment = uint256(keccak256(abi.encodePacked(oldHashLo, oldHashHi)));

        uint256 modelId = coordinator.registerModelWithCheckpoint(
            "Model", "desc", "hash", correctCommitment, 4, 8, 2, 2, 0, 10
        );
        assertEq(coordinator.getCheckpointInterval(modelId), 10);
    }

    function test_RegisterModelZeroIntervalReverts() public {
        vm.expectRevert("Invalid checkpoint interval");
        coordinator.registerModelWithCheckpoint(
            "Model", "desc", "hash", 100, 4, 8, 2, 2, 0, 0
        );
    }

    function test_RegisterModelExceedsMaxIntervalReverts() public {
        vm.expectRevert("Invalid checkpoint interval");
        coordinator.registerModelWithCheckpoint(
            "Model", "desc", "hash", 100, 4, 8, 2, 2, 0, 1001
        );
    }

    function test_StepSequencingWithCheckpointInterval() public {
        uint256 oldHashLo = 12345;
        uint256 oldHashHi = 67890;
        uint256 correctCommitment = uint256(keccak256(abi.encodePacked(oldHashLo, oldHashHi)));

        // Register with interval=5 (proof required every 5 steps)
        uint256 modelId = coordinator.registerModelWithCheckpoint(
            "Model", "desc", "hash", correctCommitment, 4, 8, 2, 2, 0, 5
        );
        coordinator.startRound(modelId, ROUND_DURATION);

        // Step should be 5 (0 + interval=5)
        uint256[] memory publicInputs = _buildPublicInputs(oldHashLo, oldHashHi, 99999, 88888, 100, 10, 5, modelId);
        bytes memory proof = ProofFixtureHardcoded.createValidProof();

        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, proof, publicInputs);

        // Verify step advanced to 5
        assertEq(coordinator.lastStepNumber(modelId), 5);
    }

    function test_WrongStepNumberWithIntervalReverts() public {
        uint256 oldHashLo = 12345;
        uint256 oldHashHi = 67890;
        uint256 correctCommitment = uint256(keccak256(abi.encodePacked(oldHashLo, oldHashHi)));

        // Register with interval=5
        uint256 modelId = coordinator.registerModelWithCheckpoint(
            "Model", "desc", "hash", correctCommitment, 4, 8, 2, 2, 0, 5
        );
        coordinator.startRound(modelId, ROUND_DURATION);

        // Try submitting with step=1 (should be step=5 with interval=5)
        uint256[] memory publicInputs = _buildPublicInputs(oldHashLo, oldHashHi, 99999, 88888, 100, 10, 1, modelId);
        bytes memory proof = ProofFixtureHardcoded.createValidProof();

        vm.prank(prover1);
        vm.expectRevert("Step number mismatch");
        coordinator.submitProof(modelId, 1, proof, publicInputs);
    }

    function test_ErrorBoundScalesWithInterval() public {
        uint256 oldHashLo = 12345;
        uint256 oldHashHi = 67890;
        uint256 correctCommitment = uint256(keccak256(abi.encodePacked(oldHashLo, oldHashHi)));

        // Register with interval=10
        uint256 modelId = coordinator.registerModelWithCheckpoint(
            "Model", "desc", "hash", correctCommitment, 4, 8, 2, 2, 0, 10
        );

        // Set a low maxErrorBound
        coordinator.setMaxErrorBound(100);

        coordinator.startRound(modelId, ROUND_DURATION);

        // Error bound of 500 should be accepted (100 * interval=10 = 1000 max)
        uint256[] memory publicInputs = _buildPublicInputs(oldHashLo, oldHashHi, 99999, 88888, 100, 500, 10, modelId);
        bytes memory proof = ProofFixtureHardcoded.createValidProof();

        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, proof, publicInputs);

        assertEq(coordinator.getAccumulatedErrorBound(modelId), 500);
    }

    function test_ErrorBoundExceedsScaledMaxReverts() public {
        uint256 oldHashLo = 12345;
        uint256 oldHashHi = 67890;
        uint256 correctCommitment = uint256(keccak256(abi.encodePacked(oldHashLo, oldHashHi)));

        // Register with interval=5
        uint256 modelId = coordinator.registerModelWithCheckpoint(
            "Model", "desc", "hash", correctCommitment, 4, 8, 2, 2, 0, 5
        );

        // Set maxErrorBound to 100 (so max for interval=5 is 500)
        coordinator.setMaxErrorBound(100);

        coordinator.startRound(modelId, ROUND_DURATION);

        // Error bound of 501 should fail (100 * 5 = 500 max)
        uint256[] memory publicInputs = _buildPublicInputs(oldHashLo, oldHashHi, 99999, 88888, 100, 501, 5, modelId);
        bytes memory proof = ProofFixtureHardcoded.createValidProof();

        vm.prank(prover1);
        vm.expectRevert("Error bound exceeds maximum");
        coordinator.submitProof(modelId, 1, proof, publicInputs);
    }

    function test_SetCheckpointInterval() public {
        uint256 modelId = coordinator.registerModel("Model", "desc", "hash", 100, 4, 8, 2, 2, 0);
        assertEq(coordinator.getCheckpointInterval(modelId), 1);

        coordinator.setCheckpointInterval(modelId, 20);
        assertEq(coordinator.getCheckpointInterval(modelId), 20);
    }

    function test_SetCheckpointIntervalOnlyOwner() public {
        uint256 modelId = coordinator.registerModel("Model", "desc", "hash", 100, 4, 8, 2, 2, 0);

        vm.prank(prover1);
        vm.expectRevert("Only model owner");
        coordinator.setCheckpointInterval(modelId, 10);
    }

    function test_SetCheckpointIntervalDuringActiveRoundReverts() public {
        uint256 modelId = coordinator.registerModel("Model", "desc", "hash", 100, 4, 8, 2, 2, 0);
        coordinator.startRound(modelId, ROUND_DURATION);

        vm.expectRevert("Cannot change during active round");
        coordinator.setCheckpointInterval(modelId, 10);
    }

    function test_CrossRoundCheckpointContinuity() public {
        uint256 oldHashLo = 12345;
        uint256 oldHashHi = 67890;
        uint256 correctCommitment = uint256(keccak256(abi.encodePacked(oldHashLo, oldHashHi)));

        // Register with interval=3
        uint256 modelId = coordinator.registerModelWithCheckpoint(
            "Model", "desc", "hash", correctCommitment, 4, 8, 2, 2, 0, 3
        );

        // Round 1: step should be 3
        coordinator.startRound(modelId, ROUND_DURATION);

        uint256 newHashLo1 = 99999;
        uint256 newHashHi1 = 88888;
        uint256[] memory publicInputs1 = _buildPublicInputs(oldHashLo, oldHashHi, newHashLo1, newHashHi1, 100, 10, 3, modelId);
        bytes memory proof1 = ProofFixtureHardcoded.createValidProof();

        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, proof1, publicInputs1);
        assertEq(coordinator.lastStepNumber(modelId), 3);

        // Round 2: step should be 6 (3 + interval=3)
        uint256 newCommitment1 = uint256(keccak256(abi.encodePacked(newHashLo1, newHashHi1)));
        coordinator.startRound(modelId, ROUND_DURATION);

        // Need new commitment as old hash
        uint256 newHashLo2 = 77777;
        uint256 newHashHi2 = 66666;
        uint256[] memory publicInputs2 = _buildPublicInputs(newHashLo1, newHashHi1, newHashLo2, newHashHi2, 90, 8, 6, modelId);
        bytes memory proof2 = ProofFixtureHardcoded.createValidProof();

        vm.prank(prover1);
        coordinator.submitProof(modelId, 2, proof2, publicInputs2);
        assertEq(coordinator.lastStepNumber(modelId), 6);
    }

    function test_CheckpointIntervalBackwardCompat() public {
        // Standard registration (no checkpoint parameter) should work exactly as before
        uint256 oldHashLo = 12345;
        uint256 oldHashHi = 67890;
        uint256 correctCommitment = uint256(keccak256(abi.encodePacked(oldHashLo, oldHashHi)));

        uint256 modelId = coordinator.registerModel("Model", "desc", "hash", correctCommitment, 4, 8, 2, 2, 0);
        assertEq(coordinator.getCheckpointInterval(modelId), 1);

        coordinator.startRound(modelId, ROUND_DURATION);

        // Step=1 should work (interval=1, backward compatible)
        uint256[] memory publicInputs = _buildPublicInputs(oldHashLo, oldHashHi, 99999, 88888, 100, 10, 1, modelId);
        bytes memory proof = ProofFixtureHardcoded.createValidProof();

        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, proof, publicInputs);

        assertEq(coordinator.lastStepNumber(modelId), 1);
        assertEq(coordinator.getAccumulatedErrorBound(modelId), 10);
    }
}

/// @notice Mock verifier for V3 tests
contract MockVerifierForV3Test {
    bool public shouldAccept = true;

    function verifyProof(bytes memory, uint256[] memory) external view returns (bool) {
        return shouldAccept;
    }

    function setAccept(bool _accept) external {
        shouldAccept = _accept;
    }
}
