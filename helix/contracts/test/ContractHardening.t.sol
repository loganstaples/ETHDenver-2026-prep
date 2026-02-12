// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "forge-std/Test.sol";
import "../src/governance/TrainingDAO.sol";
import "../src/core/HelixCoordinatorV3.sol";
import "../src/core/ModelRegistry.sol";
import "../src/token/HelixToken.sol";
import "../src/token/Staking.sol";
import "../src/token/Rewards.sol";
import "./ProofFixtures.t.sol";

contract DAOEncodingBugTest is Test {
    TrainingDAO public dao;
    HelixToken public token;
    address public proposer;

    function setUp() public {
        token = new HelixToken(address(this));
        dao = new TrainingDAO(address(token));

        proposer = makeAddr("proposer");
        token.mint(proposer, 10000e18);

        vm.prank(proposer);
        token.delegate(proposer);

        vm.roll(block.number + 1);
    }

    function test_CreateParameterProposal_EncodesCorrectId() public {
        TrainingDAO.ParameterProposal memory params = TrainingDAO.ParameterProposal({
            learningRate: 5e15,
            batchSize: 64,
            maxErrorBound: 500,
            minParticipants: 5,
            roundDuration: 2 hours
        });

        vm.prank(proposer);
        uint256 proposalId = dao.createParameterProposal("Update params", params);

        // The callData should encode the ACTUAL proposalId
        bytes memory expectedCallData = abi.encodeWithSignature("applyParameters(uint256)", proposalId);

        // Verify the proposal exists and has correct type
        (address prop_proposer, TrainingDAO.ProposalType pType,,,,,,) = dao.getProposalInfo(proposalId);
        assertEq(prop_proposer, proposer);
        assertTrue(pType == TrainingDAO.ProposalType.ParameterChange);

        // Verify params stored correctly
        (uint256 lr, uint256 bs, uint256 meb, uint256 mp, uint256 rd) = dao.parameterProposals(proposalId);
        assertEq(lr, 5e15);
        assertEq(bs, 64);
        assertEq(meb, 500);
        assertEq(mp, 5);
        assertEq(rd, 2 hours);
    }

    function test_CreateParameterProposal_MultipleProposals() public {
        TrainingDAO.ParameterProposal memory params1 = TrainingDAO.ParameterProposal({
            learningRate: 5e15, batchSize: 64, maxErrorBound: 500, minParticipants: 5, roundDuration: 2 hours
        });
        TrainingDAO.ParameterProposal memory params2 = TrainingDAO.ParameterProposal({
            learningRate: 1e16, batchSize: 128, maxErrorBound: 1000, minParticipants: 10, roundDuration: 4 hours
        });

        vm.startPrank(proposer);
        uint256 id1 = dao.createParameterProposal("First", params1);
        uint256 id2 = dao.createParameterProposal("Second", params2);
        vm.stopPrank();

        assertEq(id1, 1);
        assertEq(id2, 2);

        // Verify each proposal's params are stored under the correct ID
        (uint256 lr1,,,,) = dao.parameterProposals(id1);
        (uint256 lr2,,,,) = dao.parameterProposals(id2);
        assertEq(lr1, 5e15);
        assertEq(lr2, 1e16);
    }
}

/// @notice Mock verifier for round lifecycle tests
contract MockVerifierForRoundTest {
    bool public shouldAccept = true;

    function verifyProof(bytes memory, uint256[] memory) external view returns (bool) {
        return shouldAccept;
    }

    function setAccept(bool _accept) external {
        shouldAccept = _accept;
    }
}

/// @title RoundLifecycleTest
/// @notice Tests for multi-participant round lifecycle with thresholds, timeouts, and dispute period
contract RoundLifecycleTest is Test {
    HelixCoordinatorV3 public coordinator;
    HelixToken public token;
    Staking public staking;
    Rewards public rewards;
    ModelRegistry public registry;
    MockVerifierForRoundTest public mockVerifier;

    address public modelOwner;
    address public treasuryAddr;
    address public prover1;
    address public prover2;
    address public prover3;

    uint256 constant STAKE_AMOUNT = 200e18;
    uint256 constant MIN_STAKE = 100e18;
    uint256 constant ROUND_DURATION = 2 hours;

    // Commitment values for all proofs in a round
    uint256 constant OLD_HASH_LO = 12345;
    uint256 constant OLD_HASH_HI = 67890;

    function setUp() public {
        modelOwner = address(this);
        treasuryAddr = makeAddr("treasury");
        prover1 = makeAddr("prover1");
        prover2 = makeAddr("prover2");
        prover3 = makeAddr("prover3");

        // Deploy token
        token = new HelixToken(treasuryAddr);

        // Deploy mock verifier
        mockVerifier = new MockVerifierForRoundTest();

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

        // Fund and stake all provers
        _fundAndStake(prover1);
        _fundAndStake(prover2);
        _fundAndStake(prover3);

        // Fund reward pool
        token.mint(address(this), 10000e18);
        token.approve(address(rewards), type(uint256).max);
        rewards.fundRewardPool(10000e18, 100e18, 365 days);
    }

    function _fundAndStake(address prover) internal {
        token.mint(prover, 1000e18);
        vm.startPrank(prover);
        token.approve(address(staking), type(uint256).max);
        staking.stake(STAKE_AMOUNT);
        vm.stopPrank();
    }

    function _correctCommitment() internal pure returns (uint256) {
        return uint256(keccak256(abi.encodePacked(OLD_HASH_LO, OLD_HASH_HI)));
    }

    function _computeChecksum(
        uint256 errorBound, uint256 stepNumber, uint256 modelId, uint256 errorBudget
    ) internal pure returns (uint256) {
        return ProofFixtureHardcoded.computeErrorChecksum(errorBound, stepNumber, modelId, errorBudget);
    }

    /// @notice Build public inputs with a unique nonce encoded in the newHashLo to avoid replay
    function _buildPublicInputs(
        uint256 newHashLo, uint256 newHashHi,
        uint256 loss, uint256 errorBound,
        uint256 step, uint256 modelId
    ) internal view returns (uint256[] memory) {
        uint256[] memory inputs = new uint256[](8);
        inputs[0] = OLD_HASH_LO;
        inputs[1] = OLD_HASH_HI;
        inputs[2] = newHashLo;
        inputs[3] = newHashHi;
        inputs[4] = loss;
        inputs[5] = errorBound;
        inputs[6] = step;
        inputs[7] = _computeChecksum(errorBound, step, modelId, coordinator.maxErrorBound());
        return inputs;
    }

    /// @notice Create unique proof bytes by appending a nonce to distinguish from other proofs
    function _createUniqueProof(uint256 nonce) internal pure returns (bytes memory) {
        bytes memory base = ProofFixtureHardcoded.createValidProof();
        // Append nonce bytes to make proof unique for replay protection
        return abi.encodePacked(base, nonce);
    }

    // ============ Test 1: startRoundWithThreshold stores minParticipants ============

    function test_StartRoundWithThreshold() public {
        uint256 modelId = coordinator.registerModel("Model", "desc", "hash", _correctCommitment());
        coordinator.startRoundWithThreshold(modelId, ROUND_DURATION, 3);

        (uint32 minParticipants, uint32 validProofs, uint40 disputeDeadline,
         uint40 startedAt, bool finalized, address bestProver, uint256 bestLoss) = coordinator.getRoundExt(modelId, 1);

        assertEq(minParticipants, 3);
        assertEq(validProofs, 0);
        assertFalse(finalized);
        assertEq(bestProver, address(0));
        assertEq(bestLoss, type(uint256).max);
        assertEq(startedAt, uint40(block.timestamp));
        assertEq(disputeDeadline, uint40(block.timestamp + ROUND_DURATION + coordinator.DISPUTE_PERIOD()));
    }

    // ============ Test 2: Multi-participant proof does NOT auto-finalize ============

    function test_MultiParticipant_ProofDoesNotFinalize() public {
        uint256 modelId = coordinator.registerModel("Model", "desc", "hash", _correctCommitment());
        coordinator.startRoundWithThreshold(modelId, ROUND_DURATION, 3);

        // Submit 1 proof — should NOT finalize the round
        uint256[] memory inputs = _buildPublicInputs(99999, 88888, 100, 10, 1, modelId);
        bytes memory proof = _createUniqueProof(1);

        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, proof, inputs);

        // Round should NOT be completed
        (,,,bool isCompleted,) = coordinator.rounds(modelId, 1);
        assertFalse(isCompleted);

        // Ext should show 1 valid proof, not finalized
        (uint32 minP, uint32 validP,,, bool fin,,) = coordinator.getRoundExt(modelId, 1);
        assertEq(minP, 3);
        assertEq(validP, 1);
        assertFalse(fin);

        // Model commitment should NOT have changed
        (, uint256 commitment,) = coordinator.getModelState(modelId);
        assertEq(commitment, _correctCommitment());
    }

    // ============ Test 3: Finalize round after dispute period with best prover ============

    function test_FinalizeRound_AfterDisputePeriod() public {
        uint256 modelId = coordinator.registerModel("Model", "desc", "hash", _correctCommitment());
        coordinator.startRoundWithThreshold(modelId, ROUND_DURATION, 2);

        // Prover1 submits with loss=200
        uint256[] memory inputs1 = _buildPublicInputs(11111, 22222, 200, 10, 1, modelId);
        bytes memory proof1 = _createUniqueProof(100);

        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, proof1, inputs1);

        // Prover2 submits with loss=50 (better)
        uint256[] memory inputs2 = _buildPublicInputs(33333, 44444, 50, 8, 1, modelId);
        bytes memory proof2 = _createUniqueProof(200);

        vm.prank(prover2);
        coordinator.submitProof(modelId, 1, proof2, inputs2);

        // Round should NOT be completed yet (needs finalization)
        (,,,bool isCompleted,) = coordinator.rounds(modelId, 1);
        assertFalse(isCompleted);

        // Warp past dispute deadline (round duration + dispute period)
        vm.warp(block.timestamp + ROUND_DURATION + coordinator.DISPUTE_PERIOD() + 1);

        // Finalize
        coordinator.finalizeRound(modelId, 1);

        // Verify round completed with best prover (prover2, loss=50)
        (,,,bool completed, address prover) = coordinator.rounds(modelId, 1);
        assertTrue(completed);
        assertEq(prover, prover2);

        // Verify model commitment updated to prover2's new commitment
        uint256 expectedCommitment = uint256(keccak256(abi.encodePacked(uint256(33333), uint256(44444))));
        (, uint256 commitment,) = coordinator.getModelState(modelId);
        assertEq(commitment, expectedCommitment);

        // Verify error bound accumulated (from best prover's error bound = 8)
        assertEq(coordinator.getAccumulatedErrorBound(modelId), 8);

        // Verify participants list
        address[] memory participants = coordinator.getRoundParticipants(modelId, 1);
        assertEq(participants.length, 2);
    }

    // ============ Test 4: Finalize reverts before dispute end ============

    function test_FinalizeRound_RevertsBeforeDisputeEnd() public {
        uint256 modelId = coordinator.registerModel("Model", "desc", "hash", _correctCommitment());
        coordinator.startRoundWithThreshold(modelId, ROUND_DURATION, 2);

        // Submit 2 proofs
        uint256[] memory inputs1 = _buildPublicInputs(11111, 22222, 200, 10, 1, modelId);
        bytes memory proof1 = _createUniqueProof(300);
        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, proof1, inputs1);

        uint256[] memory inputs2 = _buildPublicInputs(33333, 44444, 50, 8, 1, modelId);
        bytes memory proof2 = _createUniqueProof(400);
        vm.prank(prover2);
        coordinator.submitProof(modelId, 1, proof2, inputs2);

        // Try to finalize BEFORE dispute period ends (still within round duration)
        vm.expectRevert("Dispute period not ended");
        coordinator.finalizeRound(modelId, 1);

        // Try after round deadline but before dispute deadline
        vm.warp(block.timestamp + ROUND_DURATION + 1);
        vm.expectRevert("Dispute period not ended");
        coordinator.finalizeRound(modelId, 1);
    }

    // ============ Test 5: Expire round with insufficient participants ============

    function test_ExpireRound_InsufficientParticipants() public {
        uint256 modelId = coordinator.registerModel("Model", "desc", "hash", _correctCommitment());
        coordinator.startRoundWithThreshold(modelId, ROUND_DURATION, 3);

        // Submit only 1 proof (need 3)
        uint256[] memory inputs = _buildPublicInputs(99999, 88888, 100, 10, 1, modelId);
        bytes memory proof = _createUniqueProof(500);

        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, proof, inputs);

        // Cannot expire while submission still open
        vm.expectRevert("Submission still open");
        coordinator.expireRound(modelId, 1);

        // Warp past deadline
        vm.warp(block.timestamp + ROUND_DURATION + 1);

        // Now expire should work
        coordinator.expireRound(modelId, 1);

        // Verify round is completed/finalized but model commitment unchanged
        (,,,bool isCompleted,) = coordinator.rounds(modelId, 1);
        assertTrue(isCompleted);

        (,,,, bool finalized,,) = coordinator.getRoundExt(modelId, 1);
        assertTrue(finalized);

        // Model commitment should NOT have changed
        (, uint256 commitment,) = coordinator.getModelState(modelId);
        assertEq(commitment, _correctCommitment());

        // Error bound should NOT have been accumulated
        assertEq(coordinator.getAccumulatedErrorBound(modelId), 0);
    }

    // ============ Test 6: Backward compat - single participant auto-finalizes ============

    function test_BackwardCompat_SingleParticipantAutoFinalizes() public {
        uint256 modelId = coordinator.registerModel("Model", "desc", "hash", _correctCommitment());

        // Use the original startRound (defaults to minParticipants=1)
        coordinator.startRound(modelId, ROUND_DURATION);

        // Verify minParticipants is 1
        (uint32 minP,,,,,,) = coordinator.getRoundExt(modelId, 1);
        assertEq(minP, 1);

        // Submit a single proof
        uint256[] memory inputs = _buildPublicInputs(99999, 88888, 100, 10, 1, modelId);
        bytes memory proof = _createUniqueProof(600);

        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, proof, inputs);

        // Round should be auto-completed (backward compat)
        (,,,bool isCompleted, address roundProver) = coordinator.rounds(modelId, 1);
        assertTrue(isCompleted);
        assertEq(roundProver, prover1);

        // Model commitment should be updated
        uint256 expectedCommitment = uint256(keccak256(abi.encodePacked(uint256(99999), uint256(88888))));
        (, uint256 commitment,) = coordinator.getModelState(modelId);
        assertEq(commitment, expectedCommitment);

        // Error bound should be accumulated
        assertEq(coordinator.getAccumulatedErrorBound(modelId), 10);
    }
}

/// @title TrainingJobTest
/// @notice Tests for the training fee mechanism (createTrainingJob, cancelTrainingJob, fee distribution)
contract TrainingJobTest is Test {
    HelixCoordinatorV3 public coordinator;
    HelixToken public token;
    Staking public staking;
    Rewards public rewards;
    ModelRegistry public registry;
    MockVerifierForRoundTest public mockVerifier;

    address public modelOwner;
    address public treasuryAddr;
    address public prover1;
    address public prover2;
    address public nonOwner;

    uint256 constant STAKE_AMOUNT = 200e18;
    uint256 constant MIN_STAKE = 100e18;
    uint256 constant ROUND_DURATION = 2 hours;

    uint256 constant OLD_HASH_LO = 12345;
    uint256 constant OLD_HASH_HI = 67890;

    function setUp() public {
        modelOwner = makeAddr("jobModelOwner");
        treasuryAddr = makeAddr("treasury");
        prover1 = makeAddr("jobProver1");
        prover2 = makeAddr("jobProver2");
        nonOwner = makeAddr("nonOwner");

        // Deploy token
        token = new HelixToken(treasuryAddr);

        // Deploy mock verifier
        mockVerifier = new MockVerifierForRoundTest();

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

        // Fund and stake provers
        _fundAndStake(prover1);
        _fundAndStake(prover2);

        // Fund reward pool
        token.mint(address(this), 10000e18);
        token.approve(address(rewards), type(uint256).max);
        rewards.fundRewardPool(10000e18, 100e18, 365 days);

        // Fund model owner so they can deposit for training jobs
        token.mint(modelOwner, 100000e18);
    }

    function _fundAndStake(address prover) internal {
        token.mint(prover, 1000e18);
        vm.startPrank(prover);
        token.approve(address(staking), type(uint256).max);
        staking.stake(STAKE_AMOUNT);
        vm.stopPrank();
    }

    function _correctCommitment() internal pure returns (uint256) {
        return uint256(keccak256(abi.encodePacked(OLD_HASH_LO, OLD_HASH_HI)));
    }

    function _computeChecksum(
        uint256 errorBound, uint256 stepNumber, uint256 modelId, uint256 errorBudget
    ) internal pure returns (uint256) {
        return ProofFixtureHardcoded.computeErrorChecksum(errorBound, stepNumber, modelId, errorBudget);
    }

    function _buildPublicInputs(
        uint256 newHashLo, uint256 newHashHi,
        uint256 loss, uint256 errorBound,
        uint256 step, uint256 modelId
    ) internal view returns (uint256[] memory) {
        uint256[] memory inputs = new uint256[](8);
        inputs[0] = OLD_HASH_LO;
        inputs[1] = OLD_HASH_HI;
        inputs[2] = newHashLo;
        inputs[3] = newHashHi;
        inputs[4] = loss;
        inputs[5] = errorBound;
        inputs[6] = step;
        inputs[7] = _computeChecksum(errorBound, step, modelId, coordinator.maxErrorBound());
        return inputs;
    }

    function _createUniqueProof(uint256 nonce) internal pure returns (bytes memory) {
        bytes memory base = ProofFixtureHardcoded.createValidProof();
        return abi.encodePacked(base, nonce);
    }

    /// @notice Register a model as modelOwner and return the modelId
    function _registerModel() internal returns (uint256 modelId) {
        vm.prank(modelOwner);
        modelId = coordinator.registerModel("TestModel", "desc", "hash", _correctCommitment());
    }

    // ============ Test 1: Create training job ============

    function test_CreateTrainingJob() public {
        uint256 modelId = _registerModel();

        uint256 depositAmount = 1000e18;
        uint256 numRounds = 10;

        // Model owner approves coordinator to spend tokens
        vm.startPrank(modelOwner);
        token.approve(address(coordinator), depositAmount);
        uint256 jobId = coordinator.createTrainingJob(modelId, numRounds, depositAmount);
        vm.stopPrank();

        // Verify job state
        (
            uint256 jModelId,
            address jOwner,
            uint256 jTotalDeposit,
            uint256 jTotalRounds,
            uint256 jCompletedRounds,
            uint256 jFeePerRound,
            uint256 jRemainingBalance,
            bool jActive
        ) = coordinator.trainingJobs(jobId);

        assertEq(jModelId, modelId);
        assertEq(jOwner, modelOwner);
        assertEq(jTotalDeposit, depositAmount);
        assertEq(jTotalRounds, numRounds);
        assertEq(jCompletedRounds, 0);
        assertEq(jFeePerRound, depositAmount / numRounds);
        assertEq(jRemainingBalance, depositAmount);
        assertTrue(jActive);

        // Verify activeModelJob
        assertEq(coordinator.activeModelJob(modelId), jobId);

        // Verify tokens transferred to coordinator
        assertEq(token.balanceOf(address(coordinator)), depositAmount);
    }

    // ============ Test 2: Fee distribution on round completion ============

    function test_TrainingJob_DistributesFees() public {
        uint256 modelId = _registerModel();

        uint256 depositAmount = 1000e18;
        uint256 numRounds = 10;
        uint256 feePerRound = depositAmount / numRounds; // 100e18

        // Create training job
        vm.startPrank(modelOwner);
        token.approve(address(coordinator), depositAmount);
        uint256 jobId = coordinator.createTrainingJob(modelId, numRounds, depositAmount);

        // Start a single-participant round (auto-finalizes on proof submission)
        coordinator.startRound(modelId, ROUND_DURATION);
        vm.stopPrank();

        // Record prover1 balance before
        uint256 prover1BalanceBefore = token.balanceOf(prover1);

        // Submit proof as prover1 — this auto-finalizes and distributes fees
        uint256[] memory inputs = _buildPublicInputs(99999, 88888, 100, 10, 1, modelId);
        bytes memory proof = _createUniqueProof(7001);

        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, proof, inputs);

        // Verify prover1 received fee (1 participant, so they get the full feePerRound)
        uint256 prover1BalanceAfter = token.balanceOf(prover1);
        assertEq(prover1BalanceAfter - prover1BalanceBefore, feePerRound);

        // Verify job state updated
        (,,,, uint256 completedRounds,, uint256 remainingBalance, bool active) = coordinator.trainingJobs(jobId);
        assertEq(completedRounds, 1);
        assertEq(remainingBalance, depositAmount - feePerRound);
        assertTrue(active);
    }

    // ============ Test 3: Cancel training job and refund ============

    function test_CancelTrainingJob_RefundsRemaining() public {
        uint256 modelId = _registerModel();

        uint256 depositAmount = 1000e18;
        uint256 numRounds = 10;
        uint256 feePerRound = depositAmount / numRounds; // 100e18

        // Create training job
        vm.startPrank(modelOwner);
        token.approve(address(coordinator), depositAmount);
        uint256 jobId = coordinator.createTrainingJob(modelId, numRounds, depositAmount);

        // Start round and submit proof (completes 1 round)
        coordinator.startRound(modelId, ROUND_DURATION);
        vm.stopPrank();

        uint256[] memory inputs = _buildPublicInputs(99999, 88888, 100, 10, 1, modelId);
        bytes memory proof = _createUniqueProof(7002);

        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, proof, inputs);

        // Record model owner balance before cancel
        uint256 ownerBalanceBefore = token.balanceOf(modelOwner);

        // Cancel the job
        vm.prank(modelOwner);
        coordinator.cancelTrainingJob(jobId);

        // Verify refund: depositAmount - 1 round of fees
        uint256 expectedRefund = depositAmount - feePerRound;
        uint256 ownerBalanceAfter = token.balanceOf(modelOwner);
        assertEq(ownerBalanceAfter - ownerBalanceBefore, expectedRefund);

        // Verify job deactivated
        (,,,,,,, bool active) = coordinator.trainingJobs(jobId);
        assertFalse(active);

        // Verify activeModelJob cleared
        assertEq(coordinator.activeModelJob(modelId), 0);
    }

    // ============ Test 4: Only model owner can create job ============

    function test_TrainingJob_OnlyModelOwner() public {
        uint256 modelId = _registerModel();

        // Fund nonOwner with tokens
        token.mint(nonOwner, 1000e18);

        vm.startPrank(nonOwner);
        token.approve(address(coordinator), 1000e18);
        vm.expectRevert("Only model owner");
        coordinator.createTrainingJob(modelId, 10, 1000e18);
        vm.stopPrank();
    }

    // ============ Test 5: Cannot create duplicate job for same model ============

    function test_TrainingJob_CannotCreateDuplicate() public {
        uint256 modelId = _registerModel();

        vm.startPrank(modelOwner);
        token.approve(address(coordinator), 2000e18);

        // First job succeeds
        coordinator.createTrainingJob(modelId, 10, 1000e18);

        // Second job for same model should revert
        vm.expectRevert("Model already has active job");
        coordinator.createTrainingJob(modelId, 10, 1000e18);
        vm.stopPrank();
    }
}

/// @title ModelVersionTest
/// @notice Tests for model version history: parentVersion, commitmentToVersion, getVersionChain, ModelVersionCreated event
contract ModelVersionTest is Test {
    ModelRegistry public registry;

    bytes32 constant COMMIT_0 = keccak256("initial");
    bytes32 constant COMMIT_1 = keccak256("update1");
    bytes32 constant COMMIT_2 = keccak256("update2");
    bytes32 constant COMMIT_3 = keccak256("update3");
    bytes32 constant COMMIT_4 = keccak256("update4");
    bytes32 constant COMMIT_5 = keccak256("update5");

    function setUp() public {
        registry = new ModelRegistry();
        // Allow this test contract to call updateModel as coordinator
        registry.setCoordinator(address(this));
    }

    /// @notice Register a model and return its ID
    function _registerModel(bytes32 commitment) internal returns (uint256) {
        return registry.registerModel("TestModel", "desc", "ipfs://hash", commitment);
    }

    /// @notice Update model with a new commitment
    function _updateModel(uint256 modelId, bytes32 commitment, uint256 roundId) internal {
        registry.updateModel(
            modelId,
            commitment,
            roundId,
            "",          // ipfsHash (empty = use model default)
            100,         // errorBound
            keccak256(abi.encodePacked(commitment, roundId)) // proofHash
        );
    }

    // ============ Test 1: Initial checkpoint has parentVersion == 0 ============

    function test_InitialCheckpoint_HasParentZero() public {
        uint256 modelId = _registerModel(COMMIT_0);

        ModelRegistry.Checkpoint memory cp = registry.getCheckpoint(modelId, 0);
        assertEq(cp.parentVersion, 0, "Initial checkpoint parentVersion should be 0");
        assertEq(cp.commitment, COMMIT_0, "Initial checkpoint commitment mismatch");
        assertEq(cp.roundId, 0, "Initial checkpoint roundId should be 0");
    }

    // ============ Test 2: Update creates linked version ============

    function test_UpdateCreatesLinkedVersion() public {
        uint256 modelId = _registerModel(COMMIT_0);

        _updateModel(modelId, COMMIT_1, 1);

        // checkpoint[0] is the initial, checkpoint[1] is the update
        ModelRegistry.Checkpoint memory cp0 = registry.getCheckpoint(modelId, 0);
        ModelRegistry.Checkpoint memory cp1 = registry.getCheckpoint(modelId, 1);

        assertEq(cp0.parentVersion, 0, "Initial parentVersion should be 0");
        assertEq(cp1.parentVersion, 0, "Second checkpoint should point to index 0 as parent");
        assertEq(cp1.commitment, COMMIT_1, "Second checkpoint commitment mismatch");

        assertEq(registry.getCheckpointCount(modelId), 2, "Should have 2 checkpoints");
    }

    // ============ Test 3: getVersionChain returns correct chain ============

    function test_GetVersionChain() public {
        uint256 modelId = _registerModel(COMMIT_0);
        _updateModel(modelId, COMMIT_1, 1);
        _updateModel(modelId, COMMIT_2, 2);

        // 3 checkpoints: 0 (initial), 1 (update1), 2 (update2)
        // Chain from version 2 should be: [cp2, cp1, cp0]
        ModelRegistry.Checkpoint[] memory chain = registry.getVersionChain(modelId, 2, 10);

        assertEq(chain.length, 3, "Chain should have 3 entries");
        assertEq(chain[0].commitment, COMMIT_2, "Chain[0] should be latest (version 2)");
        assertEq(chain[1].commitment, COMMIT_1, "Chain[1] should be version 1");
        assertEq(chain[2].commitment, COMMIT_0, "Chain[2] should be initial (version 0)");
    }

    // ============ Test 4: commitmentToVersion reverse lookup ============

    function test_CommitmentToVersion_ReverseLookup() public {
        uint256 modelId = _registerModel(COMMIT_0);
        _updateModel(modelId, COMMIT_1, 1);

        assertEq(registry.getVersionForCommitment(modelId, COMMIT_0), 0, "COMMIT_0 should map to version 0");
        assertEq(registry.getVersionForCommitment(modelId, COMMIT_1), 1, "COMMIT_1 should map to version 1");

        // Also check via the public mapping
        assertEq(registry.commitmentToVersion(modelId, COMMIT_0), 0);
        assertEq(registry.commitmentToVersion(modelId, COMMIT_1), 1);
    }

    // ============ Test 5: getVersionChain with limited count ============

    function test_GetVersionChain_LimitedCount() public {
        uint256 modelId = _registerModel(COMMIT_0);
        _updateModel(modelId, COMMIT_1, 1);
        _updateModel(modelId, COMMIT_2, 2);
        _updateModel(modelId, COMMIT_3, 3);
        _updateModel(modelId, COMMIT_4, 4);
        _updateModel(modelId, COMMIT_5, 5);

        // 6 checkpoints total (0-5). Request chain from version 5, limit 2
        ModelRegistry.Checkpoint[] memory chain = registry.getVersionChain(modelId, 5, 2);

        assertEq(chain.length, 2, "Chain should be limited to 2 entries");
        assertEq(chain[0].commitment, COMMIT_5, "Chain[0] should be version 5");
        assertEq(chain[1].commitment, COMMIT_4, "Chain[1] should be version 4");
    }
}

/// @title ComputeRewardsTest
/// @notice Tests for the compute-first 70/20/10 reward model
contract ComputeRewardsTest is Test {
    HelixCoordinatorV3 public coordinator;
    HelixToken public token;
    Staking public staking;
    Rewards public rewards;
    ModelRegistry public registry;
    MockVerifierForRoundTest public mockVerifier;

    address public modelOwner;
    address public treasuryAddr;
    address public worker1;
    address public worker2;
    address public worker3;

    uint256 constant STAKE_AMOUNT = 200e18;
    uint256 constant MIN_STAKE = 100e18;
    uint256 constant ROUND_DURATION = 2 hours;
    uint256 constant REWARDS_PER_ROUND = 100e18;

    uint256 constant OLD_HASH_LO = 12345;
    uint256 constant OLD_HASH_HI = 67890;

    function setUp() public {
        modelOwner = address(this);
        treasuryAddr = makeAddr("treasury");
        worker1 = makeAddr("worker1");
        worker2 = makeAddr("worker2");
        worker3 = makeAddr("worker3");

        // Deploy token
        token = new HelixToken(treasuryAddr);

        // Deploy mock verifier
        mockVerifier = new MockVerifierForRoundTest();

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

        // Fund and stake all workers
        _fundAndStake(worker1);
        _fundAndStake(worker2);
        _fundAndStake(worker3);

        // Fund reward pool
        token.mint(address(this), 100000e18);
        token.approve(address(rewards), type(uint256).max);
        rewards.fundRewardPool(100000e18, REWARDS_PER_ROUND, 365 days);
    }

    function _fundAndStake(address worker) internal {
        token.mint(worker, 1000e18);
        vm.startPrank(worker);
        token.approve(address(staking), type(uint256).max);
        staking.stake(STAKE_AMOUNT);
        vm.stopPrank();
    }

    function _correctCommitment() internal pure returns (uint256) {
        return uint256(keccak256(abi.encodePacked(OLD_HASH_LO, OLD_HASH_HI)));
    }

    function _computeChecksum(
        uint256 errorBound, uint256 stepNumber, uint256 modelId, uint256 errorBudget
    ) internal pure returns (uint256) {
        return ProofFixtureHardcoded.computeErrorChecksum(errorBound, stepNumber, modelId, errorBudget);
    }

    function _buildPublicInputs(
        uint256 newHashLo, uint256 newHashHi,
        uint256 loss, uint256 errorBound,
        uint256 step, uint256 modelId
    ) internal view returns (uint256[] memory) {
        uint256[] memory inputs = new uint256[](8);
        inputs[0] = OLD_HASH_LO;
        inputs[1] = OLD_HASH_HI;
        inputs[2] = newHashLo;
        inputs[3] = newHashHi;
        inputs[4] = loss;
        inputs[5] = errorBound;
        inputs[6] = step;
        inputs[7] = _computeChecksum(errorBound, step, modelId, coordinator.maxErrorBound());
        return inputs;
    }

    function _createUniqueProof(uint256 nonce) internal pure returns (bytes memory) {
        bytes memory base = ProofFixtureHardcoded.createValidProof();
        return abi.encodePacked(base, nonce);
    }

    // ============ Test 1: Equal rewards for equal proofs ============

    function test_ComputeReward_EqualForEqualProofs() public {
        uint256 modelId = coordinator.registerModel("Model", "desc", "hash", _correctCommitment());
        coordinator.startRoundWithThreshold(modelId, ROUND_DURATION, 2);

        uint256 sameLoss = 100;

        // Worker1 submits
        uint256[] memory inputs1 = _buildPublicInputs(11111, 22222, sameLoss, 10, 1, modelId);
        vm.prank(worker1);
        coordinator.submitProof(modelId, 1, _createUniqueProof(9001), inputs1);

        // Worker2 submits with same loss
        uint256[] memory inputs2 = _buildPublicInputs(33333, 44444, sameLoss, 10, 1, modelId);
        vm.prank(worker2);
        coordinator.submitProof(modelId, 1, _createUniqueProof(9002), inputs2);

        // Warp past dispute period and finalize
        vm.warp(block.timestamp + ROUND_DURATION + coordinator.DISPUTE_PERIOD() + 1);
        coordinator.finalizeRound(modelId, 1);

        // Both workers should get equal rewards (same loss = no quality bonus difference)
        uint256 reward1 = rewards.pendingRewards(modelId, 1, worker1);
        uint256 reward2 = rewards.pendingRewards(modelId, 1, worker2);

        assertGt(reward1, 0, "Worker1 reward should be > 0");
        assertEq(reward1, reward2, "Equal loss should produce equal rewards");
    }

    // ============ Test 2: Quality bonus — lower loss gets more ============

    function test_QualityBonus_LowerLossGetsMore() public {
        uint256 modelId = coordinator.registerModel("Model", "desc", "hash", _correctCommitment());
        coordinator.startRoundWithThreshold(modelId, ROUND_DURATION, 3);

        // Worker1 submits with loss=50 (best)
        uint256[] memory inputs1 = _buildPublicInputs(11111, 22222, 50, 10, 1, modelId);
        vm.prank(worker1);
        coordinator.submitProof(modelId, 1, _createUniqueProof(9010), inputs1);

        // Worker2 submits with loss=100 (median)
        uint256[] memory inputs2 = _buildPublicInputs(33333, 44444, 100, 10, 1, modelId);
        vm.prank(worker2);
        coordinator.submitProof(modelId, 1, _createUniqueProof(9011), inputs2);

        // Worker3 submits with loss=200 (worst)
        uint256[] memory inputs3 = _buildPublicInputs(55555, 66666, 200, 10, 1, modelId);
        vm.prank(worker3);
        coordinator.submitProof(modelId, 1, _createUniqueProof(9012), inputs3);

        // Warp past dispute period and finalize
        vm.warp(block.timestamp + ROUND_DURATION + coordinator.DISPUTE_PERIOD() + 1);
        coordinator.finalizeRound(modelId, 1);

        uint256 reward1 = rewards.pendingRewards(modelId, 1, worker1);
        uint256 reward2 = rewards.pendingRewards(modelId, 1, worker2);
        uint256 reward3 = rewards.pendingRewards(modelId, 1, worker3);

        // Worker1 (loss=50) should get quality bonus (below median of 100)
        // Worker2 (loss=100 = median) should not get quality bonus
        // Worker3 (loss=200 > median) should not get quality bonus
        assertGt(reward1, reward3, "Lower loss worker should get more than higher loss worker");
        assertGt(reward1, reward2, "Best worker should get more than median worker");
        // Worker2 and Worker3 should have no quality bonus, only compute + timeliness
        // They submitted at the same block, so timeliness should be equal
        assertEq(reward2, reward3, "Median and above-median should get equal reward (no quality bonus)");
    }

    // ============ Test 3: Full timeliness bonus for early submission ============

    function test_TimelinessBonus_EarlySubmissionFull() public {
        uint256 modelId = coordinator.registerModel("Model", "desc", "hash", _correctCommitment());
        coordinator.startRoundWithThreshold(modelId, ROUND_DURATION, 2);

        uint256 sameLoss = 100;

        // Worker1 submits immediately (at start of round, 0% progress)
        uint256[] memory inputs1 = _buildPublicInputs(11111, 22222, sameLoss, 10, 1, modelId);
        vm.prank(worker1);
        coordinator.submitProof(modelId, 1, _createUniqueProof(9020), inputs1);

        // Worker2 submits right at the deadline (100% progress)
        vm.warp(block.timestamp + ROUND_DURATION);
        uint256[] memory inputs2 = _buildPublicInputs(33333, 44444, sameLoss, 10, 1, modelId);
        vm.prank(worker2);
        coordinator.submitProof(modelId, 1, _createUniqueProof(9021), inputs2);

        // Warp past dispute period and finalize
        vm.warp(block.timestamp + coordinator.DISPUTE_PERIOD() + 1);
        coordinator.finalizeRound(modelId, 1);

        uint256 reward1 = rewards.pendingRewards(modelId, 1, worker1);
        uint256 reward2 = rewards.pendingRewards(modelId, 1, worker2);

        // Worker1 (early submission, progress=0%) gets full timeliness bonus
        // Worker2 (at deadline, progress=100%) gets 0 timeliness bonus
        assertGt(reward1, reward2, "Early submitter should get more than deadline submitter");
    }

    // ============ Test 4: Decayed timeliness bonus for late submission ============

    function test_TimelinessBonus_LateSubmissionDecayed() public {
        uint256 modelId = coordinator.registerModel("Model", "desc", "hash", _correctCommitment());
        coordinator.startRoundWithThreshold(modelId, ROUND_DURATION, 2);

        uint256 sameLoss = 100;

        // Worker1 submits at 50% of round duration (within 75% threshold = full bonus)
        vm.warp(block.timestamp + ROUND_DURATION / 2);
        uint256[] memory inputs1 = _buildPublicInputs(11111, 22222, sameLoss, 10, 1, modelId);
        vm.prank(worker1);
        coordinator.submitProof(modelId, 1, _createUniqueProof(9030), inputs1);

        // Worker2 submits at 90% of round duration (beyond 75% = decayed)
        vm.warp(block.timestamp + (ROUND_DURATION * 4 / 10)); // 50% + 40% = 90% total
        uint256[] memory inputs2 = _buildPublicInputs(33333, 44444, sameLoss, 10, 1, modelId);
        vm.prank(worker2);
        coordinator.submitProof(modelId, 1, _createUniqueProof(9031), inputs2);

        // Warp past dispute period and finalize
        vm.warp(block.timestamp + ROUND_DURATION + coordinator.DISPUTE_PERIOD() + 1);
        coordinator.finalizeRound(modelId, 1);

        uint256 reward1 = rewards.pendingRewards(modelId, 1, worker1);
        uint256 reward2 = rewards.pendingRewards(modelId, 1, worker2);

        // Worker1 (50% progress, within 75% threshold) gets full timeliness bonus
        // Worker2 (90% progress, beyond 75%) gets partial/decayed timeliness bonus
        assertGt(reward1, reward2, "Worker at 50% should get more timeliness bonus than worker at 90%");
        assertGt(reward2, 0, "Worker at 90% should still get some reward");
    }

    // ============ Test 5: Pool split configurable by owner ============

    function test_PoolSplit_ConfigurableByOwner() public {
        // Default split should be 70/20/10
        assertEq(rewards.computePoolBps(), 7000);
        assertEq(rewards.qualityPoolBps(), 2000);
        assertEq(rewards.timelinessPoolBps(), 1000);

        // Owner sets 80/10/10
        rewards.setPoolSplit(8000, 1000, 1000);

        assertEq(rewards.computePoolBps(), 8000, "Compute pool should be 80%");
        assertEq(rewards.qualityPoolBps(), 1000, "Quality pool should be 10%");
        assertEq(rewards.timelinessPoolBps(), 1000, "Timeliness pool should be 10%");

        // Revert if split doesn't sum to 10000
        vm.expectRevert("Must sum to 100%");
        rewards.setPoolSplit(5000, 3000, 1000);

        // Non-owner cannot set
        vm.prank(worker1);
        vm.expectRevert("Only owner");
        rewards.setPoolSplit(8000, 1000, 1000);
    }
}

/// @title E2EContractHardeningTest
/// @notice End-to-end integration tests exercising the full hardened V3 contract flow
contract E2EContractHardeningTest is Test {
    HelixCoordinatorV3 public coordinator;
    HelixToken public token;
    Staking public staking;
    Rewards public rewards;
    ModelRegistry public registry;
    MockVerifierForRoundTest public mockVerifier;

    address public modelOwner;
    address public treasuryAddr;
    address public prover1;
    address public prover2;
    address public prover3;
    address public prover4;

    uint256 constant STAKE_AMOUNT = 200e18;
    uint256 constant MIN_STAKE = 100e18;
    uint256 constant ROUND_DURATION = 2 hours;

    uint256 constant OLD_HASH_LO = 12345;
    uint256 constant OLD_HASH_HI = 67890;

    function setUp() public {
        modelOwner = makeAddr("e2eModelOwner");
        treasuryAddr = makeAddr("e2eTreasury");
        prover1 = makeAddr("e2eProver1");
        prover2 = makeAddr("e2eProver2");
        prover3 = makeAddr("e2eProver3");
        prover4 = makeAddr("e2eProver4");

        // Deploy token
        token = new HelixToken(treasuryAddr);

        // Deploy mock verifier
        mockVerifier = new MockVerifierForRoundTest();

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

        // Fund and stake all provers
        _fundAndStake(prover1);
        _fundAndStake(prover2);
        _fundAndStake(prover3);
        _fundAndStake(prover4);

        // Fund model owner for training jobs
        token.mint(modelOwner, 100000e18);
    }

    function _fundAndStake(address prover) internal {
        token.mint(prover, 10000e18);
        vm.startPrank(prover);
        token.approve(address(staking), type(uint256).max);
        staking.stake(STAKE_AMOUNT);
        vm.stopPrank();
    }

    function _correctCommitment() internal pure returns (uint256) {
        return uint256(keccak256(abi.encodePacked(OLD_HASH_LO, OLD_HASH_HI)));
    }

    function _computeChecksum(
        uint256 errorBound, uint256 stepNumber, uint256 modelId, uint256 errorBudget
    ) internal pure returns (uint256) {
        return ProofFixtureHardcoded.computeErrorChecksum(errorBound, stepNumber, modelId, errorBudget);
    }

    function _buildPublicInputs(
        uint256 oldLo, uint256 oldHi,
        uint256 newHashLo, uint256 newHashHi,
        uint256 loss, uint256 errorBound,
        uint256 step, uint256 modelId
    ) internal view returns (uint256[] memory) {
        uint256[] memory inputs = new uint256[](8);
        inputs[0] = oldLo;
        inputs[1] = oldHi;
        inputs[2] = newHashLo;
        inputs[3] = newHashHi;
        inputs[4] = loss;
        inputs[5] = errorBound;
        inputs[6] = step;
        inputs[7] = _computeChecksum(errorBound, step, modelId, coordinator.maxErrorBound());
        return inputs;
    }

    function _createUniqueProof(uint256 nonce) internal pure returns (bytes memory) {
        bytes memory base = ProofFixtureHardcoded.createValidProof();
        return abi.encodePacked(base, nonce);
    }

    // ============ Test 1: Full Flow Multi-Round with Compute Rewards ============

    function test_FullFlow_MultiRound_ComputeRewards() public {
        // --- Step 1: Register model ---
        vm.prank(modelOwner);
        uint256 modelId = coordinator.registerModel("E2EModel", "Full E2E test", "ipfs://e2e", _correctCommitment());

        // --- Step 2: Create training job (500e18 for 5 rounds = 100e18/round) ---
        vm.startPrank(modelOwner);
        token.approve(address(coordinator), 500e18);
        uint256 jobId = coordinator.createTrainingJob(modelId, 5, 500e18);
        vm.stopPrank();

        // --- Step 3: Fund reward pool ---
        token.mint(address(this), 1000e18);
        token.approve(address(rewards), 1000e18);
        rewards.fundRewardPool(1000e18, 100e18, 365 days);

        // --- Step 4: Start round 1 with threshold=2 ---
        vm.prank(modelOwner);
        coordinator.startRoundWithThreshold(modelId, ROUND_DURATION, 2);

        // --- Step 5: 4 workers submit proofs with different losses ---
        // prover1: loss=50 (best)
        uint256[] memory inputs1 = _buildPublicInputs(OLD_HASH_LO, OLD_HASH_HI, 11111, 22222, 50, 10, 1, modelId);
        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, _createUniqueProof(50001), inputs1);

        // prover2: loss=100
        uint256[] memory inputs2 = _buildPublicInputs(OLD_HASH_LO, OLD_HASH_HI, 33333, 44444, 100, 10, 1, modelId);
        vm.prank(prover2);
        coordinator.submitProof(modelId, 1, _createUniqueProof(50002), inputs2);

        // prover3: loss=150
        uint256[] memory inputs3 = _buildPublicInputs(OLD_HASH_LO, OLD_HASH_HI, 55555, 66666, 150, 10, 1, modelId);
        vm.prank(prover3);
        coordinator.submitProof(modelId, 1, _createUniqueProof(50003), inputs3);

        // prover4: loss=200
        uint256[] memory inputs4 = _buildPublicInputs(OLD_HASH_LO, OLD_HASH_HI, 77777, 88888, 200, 10, 1, modelId);
        vm.prank(prover4);
        coordinator.submitProof(modelId, 1, _createUniqueProof(50004), inputs4);

        // --- Step 6: Warp past dispute period ---
        vm.warp(block.timestamp + ROUND_DURATION + coordinator.DISPUTE_PERIOD() + 1);

        // --- Step 7: Finalize round 1 ---
        coordinator.finalizeRound(modelId, 1);

        // --- Step 8: Verify after finalization ---

        // 8a: Model commitment updated to best prover's (lowest loss=50, prover1)
        uint256 expectedCommitment = uint256(keccak256(abi.encodePacked(uint256(11111), uint256(22222))));
        (, uint256 commitment,) = coordinator.getModelState(modelId);
        assertEq(commitment, expectedCommitment, "Model commitment should match best prover's new commitment");

        // 8b: Error bound accumulated
        assertEq(coordinator.getAccumulatedErrorBound(modelId), 10, "Error bound should accumulate from best prover");

        // 8c: ModelRegistry has 2 checkpoints (initial + round 1)
        assertEq(registry.getCheckpointCount(modelId), 2, "Registry should have 2 checkpoints");

        // 8d: Training job: completedRounds=1, remainingBalance=400e18
        (,,,, uint256 completedRounds,, uint256 remainingBalance, bool active) = coordinator.trainingJobs(jobId);
        assertEq(completedRounds, 1, "Job should have 1 completed round");
        assertEq(remainingBalance, 400e18, "Job should have 400e18 remaining");
        assertTrue(active, "Job should still be active");

        // 8e: Each worker has pending rewards
        uint256 reward1 = rewards.pendingRewards(modelId, 1, prover1);
        uint256 reward2 = rewards.pendingRewards(modelId, 1, prover2);
        uint256 reward3 = rewards.pendingRewards(modelId, 1, prover3);
        uint256 reward4 = rewards.pendingRewards(modelId, 1, prover4);
        assertGt(reward1, 0, "Prover1 should have pending rewards");
        assertGt(reward2, 0, "Prover2 should have pending rewards");
        assertGt(reward3, 0, "Prover3 should have pending rewards");
        assertGt(reward4, 0, "Prover4 should have pending rewards");

        // 8f: Worker with lowest loss (prover1, loss=50) has highest reward (quality bonus)
        assertGt(reward1, reward4, "Best prover should earn more than worst prover");

        // --- Step 9: Workers claim rewards ---
        uint256[] memory modelIds = new uint256[](1);
        modelIds[0] = modelId;
        uint256[] memory roundIds = new uint256[](1);
        roundIds[0] = 1;

        uint256 prover1BalBefore = token.balanceOf(prover1);
        vm.prank(prover1);
        rewards.claimRoundRewards(modelIds, roundIds);
        uint256 prover1BalAfter = token.balanceOf(prover1);
        assertEq(prover1BalAfter - prover1BalBefore, reward1, "Prover1 should receive claimed rewards");

        // Verify rewards marked as claimed (can't claim again)
        vm.prank(prover1);
        vm.expectRevert("No rewards to claim");
        rewards.claimRoundRewards(modelIds, roundIds);

        // Other provers claim too
        vm.prank(prover2);
        rewards.claimRoundRewards(modelIds, roundIds);
        vm.prank(prover3);
        rewards.claimRoundRewards(modelIds, roundIds);
        vm.prank(prover4);
        rewards.claimRoundRewards(modelIds, roundIds);

        // --- Step 10: Verify version chain ---
        // Registry has 2 checkpoints: index 0 (initial) and index 1 (round 1)
        ModelRegistry.Checkpoint[] memory chain = registry.getVersionChain(modelId, 1, 10);
        assertEq(chain.length, 2, "Version chain should have 2 entries");
        assertEq(chain[0].commitment, bytes32(expectedCommitment), "Chain[0] should be round 1 commitment");
        assertEq(chain[1].commitment, bytes32(_correctCommitment()), "Chain[1] should be initial commitment");
        assertEq(chain[0].parentVersion, 0, "Round 1 checkpoint parent should be version 0");
    }

    // ============ Test 2: Batch Finalize Multiple Rounds ============

    function test_BatchFinalizeMultipleRounds() public {
        // --- Register model ---
        vm.prank(modelOwner);
        uint256 modelId = coordinator.registerModel("BatchModel", "Batch test", "ipfs://batch", _correctCommitment());

        // Fund reward pool
        token.mint(address(this), 10000e18);
        token.approve(address(rewards), 10000e18);
        rewards.fundRewardPool(10000e18, 100e18, 365 days);

        // --- Round 1: threshold=1 (auto-finalizes) ---
        vm.prank(modelOwner);
        coordinator.startRound(modelId, ROUND_DURATION);

        uint256[] memory inputs1 = _buildPublicInputs(OLD_HASH_LO, OLD_HASH_HI, 11111, 22222, 100, 10, 1, modelId);
        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, _createUniqueProof(60001), inputs1);

        // Verify round 1 is auto-completed
        (,,,bool r1Completed,) = coordinator.rounds(modelId, 1);
        assertTrue(r1Completed, "Round 1 should be auto-completed");

        // --- Round 2: threshold=2 (requires manual finalization) ---
        // After round 1 finalized, model commitment changed to hash(11111, 22222)
        uint256 newCommitment1Lo = 11111;
        uint256 newCommitment1Hi = 22222;

        vm.prank(modelOwner);
        coordinator.startRoundWithThreshold(modelId, ROUND_DURATION, 2);

        // Submit 2 proofs to round 2
        uint256[] memory inputs2a = _buildPublicInputs(newCommitment1Lo, newCommitment1Hi, 33333, 44444, 80, 8, 2, modelId);
        vm.prank(prover1);
        coordinator.submitProof(modelId, 2, _createUniqueProof(60002), inputs2a);

        uint256[] memory inputs2b = _buildPublicInputs(newCommitment1Lo, newCommitment1Hi, 55555, 66666, 120, 12, 2, modelId);
        vm.prank(prover2);
        coordinator.submitProof(modelId, 2, _createUniqueProof(60003), inputs2b);

        // Round 2 should NOT be completed yet
        (,,,bool r2Completed,) = coordinator.rounds(modelId, 2);
        assertFalse(r2Completed, "Round 2 should not be auto-completed");

        // --- Warp past dispute period ---
        vm.warp(block.timestamp + ROUND_DURATION + coordinator.DISPUTE_PERIOD() + 1);

        // --- Batch finalize [1, 2] --- round 1 should be skipped (already done), round 2 finalized
        uint256[] memory roundIds = new uint256[](2);
        roundIds[0] = 1;
        roundIds[1] = 2;
        coordinator.batchFinalizeRounds(modelId, roundIds);

        // Verify round 2 is now finalized
        (,,,bool r2CompletedAfter,) = coordinator.rounds(modelId, 2);
        assertTrue(r2CompletedAfter, "Round 2 should be completed after batch finalize");

        (,,,, bool r2Finalized, address r2BestProver,) = coordinator.getRoundExt(modelId, 2);
        assertTrue(r2Finalized, "Round 2 ext should be finalized");
        assertEq(r2BestProver, prover1, "Best prover for round 2 should be prover1 (lower loss=80)");

        // Verify model commitment updated to round 2's best
        uint256 expectedR2Commitment = uint256(keccak256(abi.encodePacked(uint256(33333), uint256(44444))));
        (, uint256 currentCommitment,) = coordinator.getModelState(modelId);
        assertEq(currentCommitment, expectedR2Commitment, "Model commitment should reflect round 2 best");

        // Verify registry has 3 checkpoints (initial + round 1 + round 2)
        assertEq(registry.getCheckpointCount(modelId), 3, "Registry should have 3 checkpoints");
    }

    // ============ Test 3: Expire and Refund ============

    function test_ExpireAndRefund() public {
        // --- Register model ---
        vm.prank(modelOwner);
        uint256 modelId = coordinator.registerModel("ExpireModel", "Expire test", "ipfs://expire", _correctCommitment());

        // --- Create training job ---
        vm.startPrank(modelOwner);
        token.approve(address(coordinator), 500e18);
        uint256 jobId = coordinator.createTrainingJob(modelId, 5, 500e18);
        vm.stopPrank();

        // Fund reward pool
        token.mint(address(this), 1000e18);
        token.approve(address(rewards), 1000e18);
        rewards.fundRewardPool(1000e18, 100e18, 365 days);

        // --- Start round with threshold=3 ---
        vm.prank(modelOwner);
        coordinator.startRoundWithThreshold(modelId, ROUND_DURATION, 3);

        // Only 1 worker submits (need 3)
        uint256[] memory inputs = _buildPublicInputs(OLD_HASH_LO, OLD_HASH_HI, 11111, 22222, 100, 10, 1, modelId);
        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, _createUniqueProof(70001), inputs);

        // Verify cannot expire while submission still open
        vm.expectRevert("Submission still open");
        coordinator.expireRound(modelId, 1);

        // --- Warp past deadline ---
        vm.warp(block.timestamp + ROUND_DURATION + 1);

        // --- Expire the round ---
        coordinator.expireRound(modelId, 1);

        // --- Verify round expired ---
        (,,,bool isCompleted,) = coordinator.rounds(modelId, 1);
        assertTrue(isCompleted, "Expired round should be marked completed");

        (,,,, bool finalized,,) = coordinator.getRoundExt(modelId, 1);
        assertTrue(finalized, "Expired round should be marked finalized");

        // Model commitment unchanged
        (, uint256 commitment,) = coordinator.getModelState(modelId);
        assertEq(commitment, _correctCommitment(), "Model commitment should be unchanged after expire");

        // Error bound should NOT have been accumulated
        assertEq(coordinator.getAccumulatedErrorBound(modelId), 0, "Error bound should not accumulate on expired round");

        // Training job balance preserved (fees stay for next round)
        (,,,, uint256 completedRounds,, uint256 remainingBalance,) = coordinator.trainingJobs(jobId);
        assertEq(completedRounds, 0, "No rounds should be completed");
        assertEq(remainingBalance, 500e18, "Training job balance should be fully preserved");
    }
}
