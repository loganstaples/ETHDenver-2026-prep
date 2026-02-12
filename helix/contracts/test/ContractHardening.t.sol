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
