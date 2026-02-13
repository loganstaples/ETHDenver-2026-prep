// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "forge-std/Test.sol";
import "../src/core/HelixCoordinatorV2.sol";
import "../src/core/HelixCoordinatorV3.sol";
import "../src/core/ModelRegistry.sol";
import "../src/token/HelixToken.sol";
import "../src/token/Staking.sol";
import "../src/token/Rewards.sol";
import "../src/governance/TrainingDAO.sol";
import "./ProofFixtures.t.sol";

// ============================================================================
// Mock Verifier
// ============================================================================

contract MockVerifierForSecurity {
    bool public shouldAccept = true;

    function verifyProof(bytes memory, uint256[] memory) external view returns (bool) {
        return shouldAccept;
    }

    function setAccept(bool _accept) external {
        shouldAccept = _accept;
    }
}

// ============================================================================
// 1. TrainingDAO Flash Loan Snapshot Fix Tests
// ============================================================================

/// @title DAOSnapshotFlashLoanTest
/// @notice Verifies that snapshotBlock = block.number - 1 prevents flash loan voting attacks
contract DAOSnapshotFlashLoanTest is Test {
    HelixToken public token;
    TrainingDAO public dao;

    address public legitimateVoter;
    address public flashAttacker;
    address public proposalCreator;

    uint256 constant VOTER_BALANCE = 10_000 ether;
    uint256 constant THRESHOLD = 1000 ether;

    function setUp() public {
        legitimateVoter = makeAddr("legitimateVoter");
        flashAttacker = makeAddr("flashAttacker");
        proposalCreator = makeAddr("proposalCreator");

        token = new HelixToken(makeAddr("treasury"));
        dao = new TrainingDAO(address(token));

        // Give legitimate voter tokens and delegate
        token.mint(legitimateVoter, VOTER_BALANCE);
        vm.prank(legitimateVoter);
        token.delegate(legitimateVoter);

        // Give proposal creator tokens and delegate
        token.mint(proposalCreator, THRESHOLD * 2);
        vm.prank(proposalCreator);
        token.delegate(proposalCreator);

        // Advance block so delegations are finalized
        vm.roll(block.number + 2);
    }

    /// @notice Flash loan in same block as proposal creation cannot influence voting
    function test_FlashLoan_SameBlock_ZeroVotingPower() public {
        // Block N: proposalCreator creates proposal
        vm.prank(proposalCreator);
        uint256 proposalId = dao.createProposal(
            TrainingDAO.ProposalType.Custom,
            "Test proposal",
            address(0),
            ""
        );

        // In the SAME block N: attacker flash-borrows tokens and delegates
        token.mint(flashAttacker, VOTER_BALANCE);
        vm.prank(flashAttacker);
        token.delegate(flashAttacker);

        // snapshotBlock is block.number - 1 = N-1
        // At block N-1, the attacker had 0 tokens
        uint256 votingPower = dao.getVotingPower(flashAttacker, proposalId);
        assertEq(votingPower, 0, "Flash attacker should have 0 voting power at snapshot");
    }

    /// @notice Flash loan attacker cannot vote even after voting opens
    function test_FlashLoan_CannotVoteAfterDelay() public {
        // Block N: proposal is created
        vm.prank(proposalCreator);
        uint256 proposalId = dao.createProposal(
            TrainingDAO.ProposalType.Custom,
            "Test proposal",
            address(0),
            ""
        );

        // Same block: attacker flash-borrows and delegates
        token.mint(flashAttacker, VOTER_BALANCE);
        vm.prank(flashAttacker);
        token.delegate(flashAttacker);

        // Advance time past voting delay
        vm.warp(block.timestamp + 1 days + 1);
        vm.roll(block.number + 100);

        // Attacker tries to vote - should revert because 0 voting power at snapshot
        vm.prank(flashAttacker);
        vm.expectRevert("No voting power");
        dao.castVote(proposalId, true);
    }

    /// @notice Legitimate voter who had tokens before proposal creation can vote
    function test_LegitimateVoter_CanVote() public {
        vm.prank(proposalCreator);
        uint256 proposalId = dao.createProposal(
            TrainingDAO.ProposalType.Custom,
            "Test proposal",
            address(0),
            ""
        );

        // Advance past voting delay
        vm.warp(block.timestamp + 1 days + 1);
        vm.roll(block.number + 10);

        // Legitimate voter should have voting power
        uint256 votingPower = dao.getVotingPower(legitimateVoter, proposalId);
        assertEq(votingPower, VOTER_BALANCE, "Legitimate voter should have full voting power");

        vm.prank(legitimateVoter);
        dao.castVote(proposalId, true);
    }

    /// @notice Snapshot uses block.number - 1, not block.number
    function test_SnapshotBlock_IsPreviousBlock() public {
        uint256 expectedSnapshot = block.number - 1;

        vm.prank(proposalCreator);
        uint256 proposalId = dao.createProposal(
            TrainingDAO.ProposalType.Custom,
            "Test",
            address(0),
            ""
        );

        uint256 snapshot = dao.getProposalSnapshot(proposalId);
        assertEq(snapshot, expectedSnapshot, "Snapshot should be block.number - 1");
    }
}

// ============================================================================
// 2. createParameterProposal Validation Tests
// ============================================================================

/// @title DAOParameterValidationTest
/// @notice Verifies parameter bounds checking in createParameterProposal
contract DAOParameterValidationTest is Test {
    HelixToken public token;
    TrainingDAO public dao;
    address public voter;

    function setUp() public {
        voter = makeAddr("voter");
        token = new HelixToken(makeAddr("treasury"));
        dao = new TrainingDAO(address(token));

        token.mint(voter, 10000 ether);
        vm.prank(voter);
        token.delegate(voter);
        vm.roll(block.number + 2);
    }

    function test_RejectsZeroLearningRate() public {
        TrainingDAO.ParameterProposal memory params = TrainingDAO.ParameterProposal({
            learningRate: 0,
            batchSize: 32,
            maxErrorBound: 1000,
            minParticipants: 3,
            roundDuration: 1 hours
        });

        vm.prank(voter);
        vm.expectRevert("Learning rate must be positive");
        dao.createParameterProposal("test", params);
    }

    function test_RejectsZeroBatchSize() public {
        TrainingDAO.ParameterProposal memory params = TrainingDAO.ParameterProposal({
            learningRate: 1e15,
            batchSize: 0,
            maxErrorBound: 1000,
            minParticipants: 3,
            roundDuration: 1 hours
        });

        vm.prank(voter);
        vm.expectRevert("Batch size must be positive");
        dao.createParameterProposal("test", params);
    }

    function test_RejectsTooShortRoundDuration() public {
        TrainingDAO.ParameterProposal memory params = TrainingDAO.ParameterProposal({
            learningRate: 1e15,
            batchSize: 32,
            maxErrorBound: 1000,
            minParticipants: 3,
            roundDuration: 1 minutes // too short
        });

        vm.prank(voter);
        vm.expectRevert("Round duration too short");
        dao.createParameterProposal("test", params);
    }

    function test_RejectsTooLongRoundDuration() public {
        TrainingDAO.ParameterProposal memory params = TrainingDAO.ParameterProposal({
            learningRate: 1e15,
            batchSize: 32,
            maxErrorBound: 1000,
            minParticipants: 3,
            roundDuration: 60 days // too long
        });

        vm.prank(voter);
        vm.expectRevert("Round duration too long");
        dao.createParameterProposal("test", params);
    }

    function test_AcceptsValidParameters() public {
        TrainingDAO.ParameterProposal memory params = TrainingDAO.ParameterProposal({
            learningRate: 1e15,
            batchSize: 32,
            maxErrorBound: 1000,
            minParticipants: 3,
            roundDuration: 1 hours
        });

        vm.prank(voter);
        uint256 proposalId = dao.createParameterProposal("test", params);
        assertGt(proposalId, 0);
    }
}

// ============================================================================
// 3. Error Budget Enforcement Tests (V2)
// ============================================================================

/// @title ErrorBudgetEnforcementV2Test
/// @notice Verifies model-level accumulated error budget enforcement
contract ErrorBudgetEnforcementV2Test is Test {
    HelixCoordinatorV2 public coordinator;
    MockVerifierForSecurity public mockVerifier;
    ModelRegistry public registry;

    address public owner;
    address public treasury;
    address public prover;

    uint256 constant INIT_LO = 100;
    uint256 constant INIT_HI = 200;

    function setUp() public {
        owner = address(this);
        treasury = makeAddr("treasury");
        prover = makeAddr("prover");

        mockVerifier = new MockVerifierForSecurity();
        registry = new ModelRegistry();
        coordinator = new HelixCoordinatorV2(address(mockVerifier), treasury);
        coordinator.setModelRegistry(address(registry));
        registry.setCoordinator(address(coordinator));

        // Register model from test contract (owner = address(this))
        uint256 initCommitment = uint256(keccak256(abi.encodePacked(INIT_LO, INIT_HI)));
        coordinator.registerModel("hash", initCommitment, 0.1 ether, 4, 8, 2, 2, 0);

        // Fund prover and stake
        vm.deal(prover, 10 ether);
        vm.prank(prover);
        coordinator.stake{value: 1 ether}(0);
    }

    function _buildInputs(
        uint256 oldLo, uint256 oldHi, uint256 newLo, uint256 newHi,
        uint256 loss, uint256 errorBound, uint256 step, uint256 modelId
    ) internal view returns (uint256[] memory inputs) {
        inputs = new uint256[](8);
        inputs[0] = oldLo;
        inputs[1] = oldHi;
        inputs[2] = newLo;
        inputs[3] = newHi;
        inputs[4] = loss;
        inputs[5] = errorBound;
        inputs[6] = step;
        inputs[7] = ProofFixtureHardcoded.computeErrorChecksum(
            errorBound, step, modelId, coordinator.maxErrorBound()
        );
    }

    function test_ModelErrorBudget_RevertsWhenExceeded() public {
        uint256 modelId = 0;

        // Set model max accumulated error to 15
        coordinator.setModelMaxAccumulatedError(modelId, 15);

        coordinator.startRound(modelId, 1 hours);

        // Submit proof with error bound 10 (within budget of 15)
        uint256[] memory inputs = _buildInputs(INIT_LO, INIT_HI, 300, 400, 100, 10, 1, modelId);
        vm.prank(prover);
        coordinator.submitProof(modelId, 1, abi.encodePacked(uint256(1)), inputs);

        // Now accumulated = 10. Submit another with error 10 (total 20 > 15)
        assertEq(coordinator.getAccumulatedErrorBound(modelId), 10);

        // Second proof should revert because model budget exceeded
        uint256[] memory inputs2 = _buildInputs(300, 400, 500, 600, 90, 10, 2, modelId);
        vm.prank(prover);
        vm.expectRevert(HelixCoordinatorV2.ModelErrorBudgetExceeded.selector);
        coordinator.submitProof(modelId, 1, abi.encodePacked(uint256(2)), inputs2);
    }

    function test_ModelErrorBudget_ZeroMeansNoLimit() public {
        uint256 modelId = 0;

        // By default, modelMaxAccumulatedError is 0 (no limit)
        assertEq(coordinator.modelMaxAccumulatedError(modelId), 0);

        coordinator.startRound(modelId, 1 hours);

        // Should succeed with any error bound under per-step max
        uint256[] memory inputs = _buildInputs(INIT_LO, INIT_HI, 300, 400, 100, 10, 1, modelId);
        vm.prank(prover);
        coordinator.submitProof(modelId, 1, abi.encodePacked(uint256(1)), inputs);

        assertEq(coordinator.getAccumulatedErrorBound(modelId), 10);
    }

    function test_ModelErrorBudget_OnlyAuthorizedCanSet() public {
        uint256 modelId = 0;

        // Random user cannot set
        address random = makeAddr("random");
        vm.prank(random);
        vm.expectRevert(HelixCoordinatorV2.NotAuthorized.selector);
        coordinator.setModelMaxAccumulatedError(modelId, 100);

        // Owner can set
        coordinator.setModelMaxAccumulatedError(modelId, 100);
        assertEq(coordinator.modelMaxAccumulatedError(modelId), 100);
    }
}

// ============================================================================
// 4. Round Timeout Enforcement Tests (V2)
// ============================================================================

/// @title RoundTimeoutV2Test
/// @notice Verifies that anyone can finalize expired rounds
contract RoundTimeoutV2Test is Test {
    HelixCoordinatorV2 public coordinator;
    MockVerifierForSecurity public mockVerifier;
    ModelRegistry public registry;

    address public modelOwner;
    address public treasury;
    address public prover;
    address public random;

    uint256 constant INIT_LO = 100;
    uint256 constant INIT_HI = 200;

    function setUp() public {
        modelOwner = address(this);
        treasury = makeAddr("treasury");
        prover = makeAddr("prover");
        random = makeAddr("random");

        mockVerifier = new MockVerifierForSecurity();
        registry = new ModelRegistry();
        coordinator = new HelixCoordinatorV2(address(mockVerifier), treasury);
        coordinator.setModelRegistry(address(registry));
        registry.setCoordinator(address(coordinator));

        vm.deal(prover, 10 ether);
    }

    function _registerModelAndStake() internal returns (uint256 modelId) {
        uint256 initCommitment = uint256(keccak256(abi.encodePacked(INIT_LO, INIT_HI)));
        modelId = coordinator.registerModel("hash", initCommitment, 0.1 ether, 4, 8, 2, 2, 0);
        vm.prank(prover);
        coordinator.stake{value: 1 ether}(modelId);
    }

    function test_AnyoneCanFinalizeExpiredRound_WithProofs() public {
        uint256 modelId = _registerModelAndStake();
        coordinator.startRound(modelId, 1 hours);

        // Submit a proof
        uint256[] memory inputs = new uint256[](8);
        inputs[0] = INIT_LO;
        inputs[1] = INIT_HI;
        inputs[2] = 300;
        inputs[3] = 400;
        inputs[4] = 100;
        inputs[5] = 10;
        inputs[6] = 1;
        inputs[7] = ProofFixtureHardcoded.computeErrorChecksum(
            10, 1, modelId, coordinator.maxErrorBound()
        );

        vm.prank(prover);
        coordinator.submitProof(modelId, 1, abi.encodePacked(uint256(1)), inputs);

        // Warp past deadline
        vm.warp(block.timestamp + 1 hours + 1);

        // Random user can finalize
        vm.prank(random);
        coordinator.finalizeRound(modelId, 1);

        // Verify round is completed
        (,,,bool isCompleted,) = coordinator.rounds(modelId, 1);
        assertTrue(isCompleted);
    }

    function test_AnyoneCanFinalizeExpiredRound_WithoutProofs() public {
        uint256 modelId = _registerModelAndStake();
        coordinator.startRound(modelId, 1 hours);

        // Warp past deadline
        vm.warp(block.timestamp + 1 hours + 1);

        // Random user can close the round even with no proofs
        vm.prank(random);
        coordinator.finalizeRound(modelId, 1);

        (,,,bool isCompleted,) = coordinator.rounds(modelId, 1);
        assertTrue(isCompleted);
    }

    function test_NonOwner_CannotFinalize_BeforeExpiry() public {
        uint256 modelId = _registerModelAndStake();
        coordinator.startRound(modelId, 1 hours);

        // Submit a proof
        uint256[] memory inputs = new uint256[](8);
        inputs[0] = INIT_LO;
        inputs[1] = INIT_HI;
        inputs[2] = 300;
        inputs[3] = 400;
        inputs[4] = 100;
        inputs[5] = 10;
        inputs[6] = 1;
        inputs[7] = ProofFixtureHardcoded.computeErrorChecksum(
            10, 1, modelId, coordinator.maxErrorBound()
        );

        vm.prank(prover);
        coordinator.submitProof(modelId, 1, abi.encodePacked(uint256(1)), inputs);

        // Random user cannot finalize before expiry
        vm.prank(random);
        vm.expectRevert("Round not expired and not authorized");
        coordinator.finalizeRound(modelId, 1);
    }
}

// ============================================================================
// 5. Round Timeout Enforcement Tests (V3)
// ============================================================================

/// @title RoundTimeoutV3Test
/// @notice Verifies timeoutRound() in V3 for stuck rounds
contract RoundTimeoutV3Test is Test {
    HelixCoordinatorV3 public coordinator;
    MockVerifierForSecurity public mockVerifier;
    HelixToken public token;
    Staking public staking;
    Rewards public rewards;
    ModelRegistry public registry;

    address public modelOwner;
    address public treasuryAddr;
    address public prover;
    address public random;

    function setUp() public {
        modelOwner = address(this);
        treasuryAddr = makeAddr("treasury");
        prover = makeAddr("prover");
        random = makeAddr("random");

        token = new HelixToken(treasuryAddr);
        mockVerifier = new MockVerifierForSecurity();
        staking = new Staking(address(token), 100e18, 7 days, 5000);
        rewards = new Rewards(address(token));
        registry = new ModelRegistry();

        coordinator = new HelixCoordinatorV3(
            address(mockVerifier),
            address(staking),
            address(rewards),
            address(registry),
            treasuryAddr
        );

        staking.setOperator(address(coordinator));
        rewards.setCoordinator(address(coordinator));
        registry.setCoordinator(address(coordinator));

        // Fund and stake prover
        token.mint(prover, 1000e18);
        vm.startPrank(prover);
        token.approve(address(staking), type(uint256).max);
        staking.stake(200e18);
        vm.stopPrank();

        // Fund reward pool
        token.mint(address(this), 10000e18);
        token.approve(address(rewards), type(uint256).max);
        rewards.fundRewardPool(10000e18, 100e18, 365 days);
    }

    function test_TimeoutRound_ExpiresStuckRound() public {
        // Register model and start round requiring 5 participants
        uint256 modelId = coordinator.registerModel(
            "Test", "desc", "hash", 12345, 4, 8, 2, 2, 0
        );

        coordinator.startRoundWithThreshold(modelId, 1 hours, 5);

        // Warp past deadline + dispute period
        vm.warp(block.timestamp + 1 hours + 1 hours + 1);

        // Anyone can timeout the round
        vm.prank(random);
        coordinator.timeoutRound(modelId, 1);

        // Round should be finalized/expired
        (,,,bool isCompleted,) = coordinator.rounds(modelId, 1);
        assertTrue(isCompleted, "Round should be completed after timeout");
    }

    function test_TimeoutRound_CannotTimeoutBeforeDispute() public {
        uint256 modelId = coordinator.registerModel(
            "Test", "desc", "hash", 12345, 4, 8, 2, 2, 0
        );

        coordinator.startRoundWithThreshold(modelId, 1 hours, 5);

        // Warp to just before dispute deadline
        vm.warp(block.timestamp + 1 hours + 30 minutes);

        vm.prank(random);
        vm.expectRevert("Dispute period not ended");
        coordinator.timeoutRound(modelId, 1);
    }
}

// ============================================================================
// 6. Batch Proof Verification Tests
// ============================================================================

/// @title BatchVerificationV2Test
/// @notice Tests the optimized verifyAndSubmitBatch function
contract BatchVerificationV2Test is Test {
    HelixCoordinatorV2 public coordinator;
    MockVerifierForSecurity public mockVerifier;
    ModelRegistry public registry;

    address public prover;
    address public treasury;

    uint256 constant INIT_LO = 100;
    uint256 constant INIT_HI = 200;

    function setUp() public {
        treasury = makeAddr("treasury");
        prover = makeAddr("prover");

        mockVerifier = new MockVerifierForSecurity();
        registry = new ModelRegistry();
        coordinator = new HelixCoordinatorV2(address(mockVerifier), treasury);
        coordinator.setModelRegistry(address(registry));
        registry.setCoordinator(address(coordinator));

        vm.deal(prover, 10 ether);
    }

    function test_BatchVerification_EmptyBatchReverts() public {
        uint256 initCommitment = uint256(keccak256(abi.encodePacked(INIT_LO, INIT_HI)));
        coordinator.registerModel("hash", initCommitment, 0.1 ether, 4, 8, 2, 2, 0);

        vm.prank(prover);
        coordinator.stake{value: 1 ether}(0);

        coordinator.startRound(0, 1 hours);

        bytes[] memory proofs = new bytes[](0);
        uint256[][] memory inputs = new uint256[][](0);

        vm.prank(prover);
        vm.expectRevert(HelixCoordinatorV2.EmptyBatch.selector);
        coordinator.verifyAndSubmitBatch(0, 1, proofs, inputs);
    }

    function test_BatchVerification_ValidBatch() public {
        uint256 initCommitment = uint256(keccak256(abi.encodePacked(INIT_LO, INIT_HI)));
        uint256 modelId = coordinator.registerModel("hash", initCommitment, 0.1 ether, 4, 8, 2, 2, 0);

        vm.prank(prover);
        coordinator.stake{value: 1 ether}(modelId);

        coordinator.startRound(modelId, 1 hours);

        // Build 2 sequential proofs
        bytes[] memory proofs = new bytes[](2);
        proofs[0] = abi.encodePacked(uint256(1));
        proofs[1] = abi.encodePacked(uint256(2));

        uint256[][] memory inputs = new uint256[][](2);

        // Proof 1
        inputs[0] = new uint256[](8);
        inputs[0][0] = INIT_LO;
        inputs[0][1] = INIT_HI;
        inputs[0][2] = 300;
        inputs[0][3] = 400;
        inputs[0][4] = 100;
        inputs[0][5] = 10;
        inputs[0][6] = 1;
        inputs[0][7] = ProofFixtureHardcoded.computeErrorChecksum(
            10, 1, modelId, coordinator.maxErrorBound()
        );

        // Proof 2 chains from proof 1
        inputs[1] = new uint256[](8);
        inputs[1][0] = 300;
        inputs[1][1] = 400;
        inputs[1][2] = 500;
        inputs[1][3] = 600;
        inputs[1][4] = 90;
        inputs[1][5] = 10;
        inputs[1][6] = 2;
        inputs[1][7] = ProofFixtureHardcoded.computeErrorChecksum(
            10, 2, modelId, coordinator.maxErrorBound()
        );

        vm.prank(prover);
        coordinator.verifyAndSubmitBatch(modelId, 1, proofs, inputs);

        // Verify both proofs were accepted
        assertEq(coordinator.getLastStepNumber(modelId), 2);
        assertEq(coordinator.getAccumulatedErrorBound(modelId), 20);
    }
}

// ============================================================================
// 7. Emergency Pause Tests for All Contracts
// ============================================================================

/// @title PauseAllContractsTest
/// @notice Verifies emergency pause works on TrainingDAO, Staking, and Rewards
contract PauseAllContractsTest is Test {
    HelixToken public token;
    TrainingDAO public dao;
    Staking public staking;
    Rewards public rewards;

    address public owner;
    address public voter;

    function setUp() public {
        owner = address(this);
        voter = makeAddr("voter");

        token = new HelixToken(makeAddr("treasury"));
        dao = new TrainingDAO(address(token));
        staking = new Staking(address(token), 100e18, 7 days, 5000);
        rewards = new Rewards(address(token));

        // Setup voter
        token.mint(voter, 10000 ether);
        vm.prank(voter);
        token.delegate(voter);
        vm.roll(block.number + 2);
    }

    // ---- TrainingDAO Pause Tests ----

    function test_DAO_PauseBlocksProposalCreation() public {
        dao.emergencyPause();

        vm.prank(voter);
        vm.expectRevert("Contract is paused");
        dao.createProposal(TrainingDAO.ProposalType.Custom, "test", address(0), "");
    }

    function test_DAO_PauseBlocksVoting() public {
        // Create proposal before pause
        vm.prank(voter);
        uint256 proposalId = dao.createProposal(
            TrainingDAO.ProposalType.Custom, "test", address(0), ""
        );

        // Advance past voting delay
        vm.warp(block.timestamp + 1 days + 1);
        vm.roll(block.number + 10);

        // Pause
        dao.emergencyPause();

        // Voting should be blocked
        vm.prank(voter);
        vm.expectRevert("Contract is paused");
        dao.castVote(proposalId, true);
    }

    function test_DAO_UnpauseRestoresFunction() public {
        dao.emergencyPause();
        dao.unpause();

        vm.prank(voter);
        uint256 proposalId = dao.createProposal(
            TrainingDAO.ProposalType.Custom, "test", address(0), ""
        );
        assertGt(proposalId, 0);
    }

    function test_DAO_OnlyOwnerCanPause() public {
        vm.prank(voter);
        vm.expectRevert("Only owner");
        dao.emergencyPause();
    }

    // ---- Staking Pause Tests ----

    function test_Staking_PauseBlocksStaking() public {
        staking.emergencyPause();

        token.mint(voter, 200e18);
        vm.startPrank(voter);
        token.approve(address(staking), 200e18);
        vm.expectRevert("Contract is paused");
        staking.stake(200e18);
        vm.stopPrank();
    }

    function test_Staking_PauseBlocksUnbonding() public {
        // Stake first
        token.mint(voter, 200e18);
        vm.startPrank(voter);
        token.approve(address(staking), 200e18);
        staking.stake(200e18);
        vm.stopPrank();

        // Pause
        staking.emergencyPause();

        // Try to start unbonding
        vm.prank(voter);
        vm.expectRevert("Contract is paused");
        staking.startUnbonding();
    }

    function test_Staking_UnstakeStillWorksWhenPaused() public {
        // Stake and start unbonding
        token.mint(voter, 200e18);
        vm.startPrank(voter);
        token.approve(address(staking), 200e18);
        staking.stake(200e18);
        staking.startUnbonding();
        vm.stopPrank();

        // Wait for unbonding period
        vm.warp(block.timestamp + 7 days + 1);

        // Pause
        staking.emergencyPause();

        // Unstake should still work (don't lock people's money during pause)
        vm.prank(voter);
        staking.unstake();
    }

    // ---- Rewards Pause Tests ----

    function test_Rewards_PauseBlocksFunding() public {
        rewards.emergencyPause();

        token.mint(voter, 1000e18);
        vm.startPrank(voter);
        token.approve(address(rewards), 1000e18);
        vm.expectRevert("Contract is paused");
        rewards.fundRewardPool(1000e18, 100e18, 365 days);
        vm.stopPrank();
    }

    function test_Rewards_PauseBlocksClaims() public {
        // Setup rewards
        rewards.setCoordinator(address(this));
        token.mint(address(this), 10000e18);
        token.approve(address(rewards), 10000e18);
        rewards.fundRewardPool(10000e18, 100e18, 365 days);

        rewards.registerParticipant(0, 1, voter);
        rewards.allocateRoundRewards(0, 1);

        // Pause
        rewards.emergencyPause();

        uint256[] memory modelIds = new uint256[](1);
        uint256[] memory roundIds = new uint256[](1);
        modelIds[0] = 0;
        roundIds[0] = 1;

        vm.prank(voter);
        vm.expectRevert("Contract is paused");
        rewards.claimRoundRewards(modelIds, roundIds);
    }
}

// ============================================================================
// 8. Reward Fairness Tests
// ============================================================================

/// @title RewardFairnessTest
/// @notice Verifies proportional reward distribution based on proof count
contract RewardFairnessTest is Test {
    HelixToken public token;
    Rewards public rewards;

    address public coordinator;
    address public worker1;
    address public worker2;

    function setUp() public {
        coordinator = address(this);
        worker1 = makeAddr("worker1");
        worker2 = makeAddr("worker2");

        token = new HelixToken(makeAddr("treasury"));
        rewards = new Rewards(address(token));
        rewards.setCoordinator(coordinator);

        // Fund reward pool
        token.mint(address(this), 100000e18);
        token.approve(address(rewards), 100000e18);
        rewards.fundRewardPool(100000e18, 1000e18, 365 days);
    }

    function test_ProportionalRewards_MoreProofsMoreReward() public {
        // Worker1 submits 3 proofs, Worker2 submits 1 proof
        rewards.registerParticipantWithData(0, 1, worker1, 100, uint40(block.timestamp), uint40(block.timestamp), uint40(block.timestamp + 1 hours));
        rewards.registerParticipantWithData(0, 1, worker1, 90, uint40(block.timestamp), uint40(block.timestamp), uint40(block.timestamp + 1 hours));
        rewards.registerParticipantWithData(0, 1, worker1, 85, uint40(block.timestamp), uint40(block.timestamp), uint40(block.timestamp + 1 hours));
        rewards.registerParticipantWithData(0, 1, worker2, 95, uint40(block.timestamp), uint40(block.timestamp), uint40(block.timestamp + 1 hours));

        rewards.allocateRoundRewards(0, 1);

        // Worker1 should have more pending rewards (3/4 of compute pool vs 1/4)
        uint256 w1Earned = rewards.pendingRewards(0, 1, worker1);
        uint256 w2Earned = rewards.pendingRewards(0, 1, worker2);

        // Worker1 has 3x the proof count, so should get roughly 3x the compute pool share
        assertGt(w1Earned, w2Earned, "Worker with more proofs should earn more");
    }

    function test_EqualProofs_EqualRewards() public {
        // Both workers submit 1 proof with same loss
        rewards.registerParticipantWithData(0, 1, worker1, 100, uint40(block.timestamp), uint40(block.timestamp), uint40(block.timestamp + 1 hours));
        rewards.registerParticipantWithData(0, 1, worker2, 100, uint40(block.timestamp), uint40(block.timestamp), uint40(block.timestamp + 1 hours));

        rewards.allocateRoundRewards(0, 1);

        uint256 w1Earned = rewards.pendingRewards(0, 1, worker1);
        uint256 w2Earned = rewards.pendingRewards(0, 1, worker2);

        assertEq(w1Earned, w2Earned, "Equal work should get equal rewards");
    }
}

// ============================================================================
// 9. V2 Pause on Previously Unprotected Functions
// ============================================================================

/// @title V2PauseProtectionTest
/// @notice Verifies V2 pause protection on stake and challengeProof
contract V2PauseProtectionTest is Test {
    HelixCoordinatorV2 public coordinator;
    MockVerifierForSecurity public mockVerifier;
    ModelRegistry public registry;

    address public prover;
    address public treasury;

    function setUp() public {
        treasury = makeAddr("treasury");
        prover = makeAddr("prover");

        mockVerifier = new MockVerifierForSecurity();
        registry = new ModelRegistry();
        coordinator = new HelixCoordinatorV2(address(mockVerifier), treasury);
        coordinator.setModelRegistry(address(registry));
        registry.setCoordinator(address(coordinator));

        uint256 initCommitment = uint256(keccak256(abi.encodePacked(uint256(100), uint256(200))));
        coordinator.registerModel("hash", initCommitment, 0.1 ether, 4, 8, 2, 2, 0);

        vm.deal(prover, 10 ether);
    }

    function test_PauseBlocksStaking() public {
        coordinator.emergencyPause();

        vm.prank(prover);
        vm.expectRevert(HelixCoordinatorV2.ContractPaused.selector);
        coordinator.stake{value: 1 ether}(0);
    }

    function test_PauseBlocksChallenge() public {
        // First stake and submit proof before pause
        vm.prank(prover);
        coordinator.stake{value: 1 ether}(0);

        coordinator.emergencyPause();

        vm.prank(makeAddr("challenger"));
        vm.expectRevert(HelixCoordinatorV2.ContractPaused.selector);
        coordinator.challengeProof(0, 1, "", new uint256[](8));
    }
}

// ============================================================================
// 10. V3 Error Budget Enforcement Tests
// ============================================================================

/// @title ErrorBudgetEnforcementV3Test
/// @notice Verifies model-level error budget in V3
contract ErrorBudgetEnforcementV3Test is Test {
    HelixCoordinatorV3 public coordinator;
    MockVerifierForSecurity public mockVerifier;
    HelixToken public token;
    Staking public staking;
    Rewards public rewards;
    ModelRegistry public registry;

    address public modelOwner;
    address public treasuryAddr;

    function setUp() public {
        modelOwner = address(this);
        treasuryAddr = makeAddr("treasury");

        token = new HelixToken(treasuryAddr);
        mockVerifier = new MockVerifierForSecurity();
        staking = new Staking(address(token), 100e18, 7 days, 5000);
        rewards = new Rewards(address(token));
        registry = new ModelRegistry();

        coordinator = new HelixCoordinatorV3(
            address(mockVerifier),
            address(staking),
            address(rewards),
            address(registry),
            treasuryAddr
        );

        staking.setOperator(address(coordinator));
        rewards.setCoordinator(address(coordinator));
        registry.setCoordinator(address(coordinator));
    }

    function test_V3_SetModelErrorBudget_OnlyAuthorized() public {
        uint256 modelId = coordinator.registerModel(
            "Test", "desc", "hash", 12345, 4, 8, 2, 2, 0
        );

        // Random user cannot set
        address random = makeAddr("random");
        vm.prank(random);
        vm.expectRevert("Not authorized");
        coordinator.setModelMaxAccumulatedError(modelId, 100);

        // Model owner (this) can set
        coordinator.setModelMaxAccumulatedError(modelId, 100);
        assertEq(coordinator.modelMaxAccumulatedError(modelId), 100);
    }
}
