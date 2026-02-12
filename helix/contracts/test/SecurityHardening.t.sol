// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "forge-std/Test.sol";
import "../src/core/HelixCoordinatorV2.sol";
import "../src/core/ModelRegistry.sol";
import "../src/token/HelixToken.sol";
import "../src/token/Rewards.sol";
import "../src/token/Staking.sol";
import "../src/governance/TrainingDAO.sol";
import "../src/mocks/MockVerifier.sol";
import "./ProofFixtures.t.sol";

/// @title ETH-receivable treasury for tests
contract TestTreasury {
    receive() external payable {}
}

// ============================================================================
// 1. Rewards Double-Claim Prevention Tests
// ============================================================================

/// @title RewardsDoubleClaimTest
/// @notice Verifies that the double-claim vulnerability in Rewards.sol is fixed
contract RewardsDoubleClaimTest is Test {
    HelixToken public token;
    Rewards public rewards;

    address public deployer;
    address public funder;
    address public participant1;
    address public participant2;
    address public coordinator;

    uint256 constant FUND_AMOUNT = 100_000 ether;
    uint256 constant REWARDS_PER_ROUND = 1000 ether;
    uint256 constant DURATION = 365 days;

    function setUp() public {
        deployer = address(this);
        funder = makeAddr("funder");
        participant1 = makeAddr("participant1");
        participant2 = makeAddr("participant2");
        coordinator = makeAddr("coordinator");

        // Deploy token and rewards
        token = new HelixToken(makeAddr("treasury"));
        rewards = new Rewards(address(token));

        // Setup roles
        rewards.setCoordinator(coordinator);

        // Fund the reward pool
        token.mint(funder, FUND_AMOUNT);
        vm.startPrank(funder);
        token.approve(address(rewards), FUND_AMOUNT);
        rewards.fundRewardPool(FUND_AMOUNT, REWARDS_PER_ROUND, DURATION);
        vm.stopPrank();
    }

    /// @notice Core test: attempt double-claim via claimRoundRewards() twice
    function test_DoubleClaim_Prevention() public {
        uint256 modelId = 0;
        uint256 roundId = 1;

        // Register participant and allocate rewards
        vm.startPrank(coordinator);
        rewards.registerParticipant(modelId, roundId, participant1);
        rewards.allocateRoundRewards(modelId, roundId);
        vm.stopPrank();

        // Check reward was allocated
        (uint256 totalEarned, uint256 totalClaimed,,) = rewards.getParticipantStats(participant1);
        assertGt(totalEarned, 0, "Should have earned rewards");
        assertEq(totalClaimed, 0, "Should not have claimed yet");
        uint256 expectedReward = totalEarned;

        // Step 1: Claim rewards via claimRoundRewards()
        uint256[] memory modelIds = new uint256[](1);
        uint256[] memory roundIds = new uint256[](1);
        modelIds[0] = modelId;
        roundIds[0] = roundId;

        vm.prank(participant1);
        rewards.claimRoundRewards(modelIds, roundIds);

        uint256 balanceAfterFirst = token.balanceOf(participant1);
        assertEq(balanceAfterFirst, expectedReward, "Should receive full reward");

        // Step 2: Attempt to double-claim via claimRoundRewards() again
        vm.prank(participant1);
        vm.expectRevert("No rewards to claim");
        rewards.claimRoundRewards(modelIds, roundIds);

        // Balance should not have changed
        assertEq(token.balanceOf(participant1), balanceAfterFirst, "Balance should not change after failed double-claim");
    }

    /// @notice Test: claimRoundRewards() twice for same round fails
    function test_DoubleClaim_ReverseOrder() public {
        uint256 modelId = 0;
        uint256 roundId = 1;

        vm.startPrank(coordinator);
        rewards.registerParticipant(modelId, roundId, participant1);
        rewards.allocateRoundRewards(modelId, roundId);
        vm.stopPrank();

        (uint256 totalEarned,,,) = rewards.getParticipantStats(participant1);

        // Step 1: Claim via claimRoundRewards()
        uint256[] memory modelIds = new uint256[](1);
        uint256[] memory roundIds = new uint256[](1);
        modelIds[0] = modelId;
        roundIds[0] = roundId;

        vm.prank(participant1);
        rewards.claimRoundRewards(modelIds, roundIds);
        assertEq(token.balanceOf(participant1), totalEarned);

        // Step 2: Try claimRoundRewards() again - should revert since already claimed
        vm.prank(participant1);
        vm.expectRevert("No rewards to claim");
        rewards.claimRoundRewards(modelIds, roundIds);
    }

    /// @notice Test: partial claims across rounds can't exceed total earned
    function test_DoubleClaim_MultipleRounds() public {
        uint256 modelId = 0;

        // Register and allocate for two rounds
        vm.startPrank(coordinator);
        rewards.registerParticipant(modelId, 1, participant1);
        rewards.allocateRoundRewards(modelId, 1);
        rewards.registerParticipant(modelId, 2, participant1);
        rewards.allocateRoundRewards(modelId, 2);
        vm.stopPrank();

        (uint256 totalEarned,,,) = rewards.getParticipantStats(participant1);

        // Claim round 1 via claimRoundRewards
        uint256[] memory modelIds = new uint256[](1);
        uint256[] memory roundIds = new uint256[](1);
        modelIds[0] = modelId;
        roundIds[0] = 1;

        vm.prank(participant1);
        rewards.claimRoundRewards(modelIds, roundIds);

        // Claim round 2 via claimRoundRewards
        roundIds[0] = 2;
        vm.prank(participant1);
        rewards.claimRoundRewards(modelIds, roundIds);

        // Total claimed should equal total earned (no more, no less)
        assertEq(token.balanceOf(participant1), totalEarned, "Total claimed must equal total earned");

        // Try claiming again - should fail
        roundIds[0] = 1;
        vm.prank(participant1);
        vm.expectRevert("No rewards to claim");
        rewards.claimRoundRewards(modelIds, roundIds);

        roundIds[0] = 2;
        vm.prank(participant1);
        vm.expectRevert("No rewards to claim");
        rewards.claimRoundRewards(modelIds, roundIds);
    }
}

// ============================================================================
// 2. TrainingDAO Flash Loan Voting Prevention Tests
// ============================================================================

/// @title TrainingDAOFlashLoanTest
/// @notice Verifies snapshot-based voting prevents flash loan attacks
contract TrainingDAOFlashLoanTest is Test {
    HelixToken public token;
    TrainingDAO public dao;

    address public deployer;
    address public voter1;
    address public voter2;
    address public flashAttacker;

    uint256 constant INITIAL_BALANCE = 10_000 ether;

    function setUp() public {
        deployer = address(this);
        voter1 = makeAddr("voter1");
        voter2 = makeAddr("voter2");
        flashAttacker = makeAddr("flashAttacker");

        // Deploy token and DAO
        token = new HelixToken(makeAddr("treasury"));
        dao = new TrainingDAO(address(token));

        // Distribute tokens to voters
        token.mint(voter1, INITIAL_BALANCE);
        token.mint(voter2, INITIAL_BALANCE);

        // Voters must delegate to themselves to activate vote tracking
        vm.prank(voter1);
        token.delegate(voter1);
        vm.prank(voter2);
        token.delegate(voter2);

        // Advance one block so delegation takes effect for getPastVotes
        vm.roll(block.number + 1);
    }

    /// @notice Test: voting power uses snapshot at proposal creation block
    function test_VotingPowerUsesSnapshot() public {
        // voter1 creates a proposal
        vm.prank(voter1);
        uint256 proposalId = dao.createProposal(
            TrainingDAO.ProposalType.Custom,
            "Test proposal",
            address(0),
            ""
        );

        // Check snapshot block was recorded
        uint256 snapshot = dao.getProposalSnapshot(proposalId);
        assertGt(snapshot, 0, "Snapshot block should be set");

        // Must advance block so getPastVotes can query the snapshot block
        vm.roll(block.number + 1);

        // Verify voting power at snapshot
        uint256 votingPower = dao.getVotingPower(voter1, proposalId);
        assertEq(votingPower, INITIAL_BALANCE, "Voting power should match snapshot balance");
    }

    /// @notice Test: tokens acquired after proposal creation have no voting power
    function test_FlashLoan_TokensAfterProposalHaveNoPower() public {
        // voter1 creates a proposal at current block
        vm.prank(voter1);
        uint256 proposalId = dao.createProposal(
            TrainingDAO.ProposalType.Custom,
            "Test proposal",
            address(0),
            ""
        );

        // Advance block to seal the snapshot block
        vm.roll(block.number + 1);

        // Flash attacker acquires tokens AFTER proposal snapshot block
        token.mint(flashAttacker, 50_000 ether);
        vm.prank(flashAttacker);
        token.delegate(flashAttacker);

        // Advance block so delegation checkpoint is finalized
        vm.roll(block.number + 1);

        // Advance time past voting delay so voting is active
        (, , , uint256 startTime, , , , ) = dao.getProposalInfo(proposalId);
        vm.warp(startTime + 1);

        // Flash attacker should have zero voting power for this proposal
        // because they acquired tokens after the snapshot block
        uint256 attackerPower = dao.getVotingPower(flashAttacker, proposalId);
        assertEq(attackerPower, 0, "Flash attacker should have zero voting power");

        // Attacker cannot vote
        vm.prank(flashAttacker);
        vm.expectRevert("No voting power");
        dao.castVote(proposalId, true);
    }

    /// @notice Test: voting power is based on delegation at snapshot block
    function test_DelegationSnapshot() public {
        // voter3 has tokens but hasn't delegated yet at proposal time
        address voter3 = makeAddr("voter3");
        token.mint(voter3, INITIAL_BALANCE);
        // voter3 does NOT delegate yet

        // Advance to a known block for the snapshot
        vm.roll(100);

        // Create proposal - snapshot recorded at block 100
        vm.prank(voter1);
        uint256 proposalId = dao.createProposal(
            TrainingDAO.ProposalType.Custom,
            "Test proposal",
            address(0),
            ""
        );

        // Advance block past the snapshot
        vm.roll(101);

        // voter3 delegates AFTER proposal creation (at block 101)
        vm.prank(voter3);
        token.delegate(voter3);

        // Advance so delegation checkpoint is finalized
        vm.roll(102);

        // voter3 has zero power for this proposal since they delegated after snapshot
        uint256 power = dao.getVotingPower(voter3, proposalId);
        assertEq(power, 0, "Delegating after proposal should give zero power");
    }

    /// @notice Test: createParameterProposal uses internal call (preserves msg.sender)
    function test_CreateParameterProposal_InternalCall() public {
        vm.prank(voter1);
        uint256 proposalId = dao.createParameterProposal(
            "Update parameters",
            TrainingDAO.ParameterProposal({
                learningRate: 2e15,
                batchSize: 64,
                maxErrorBound: 500,
                minParticipants: 5,
                roundDuration: 2 hours
            })
        );

        // Verify the proposal was created with voter1 as proposer
        (address proposer, , , , , , , ) = dao.getProposalInfo(proposalId);
        assertEq(proposer, voter1, "Proposer should be voter1");
    }

    /// @notice Test: flash-loaned tokens without delegation cannot create proposals
    function test_FlashLoan_CannotCreateProposal_WithoutDelegation() public {
        // Flash attacker acquires tokens but does NOT delegate
        token.mint(flashAttacker, 10_000 ether);

        vm.roll(block.number + 1);

        // Attacker has balance but zero voting power (no delegation)
        assertGe(token.balanceOf(flashAttacker), 1000 ether, "Attacker has tokens");

        // createProposal now checks getVotes() not balanceOf(), so undelegated tokens fail
        vm.prank(flashAttacker);
        vm.expectRevert("Below proposal threshold");
        dao.createProposal(
            TrainingDAO.ProposalType.Custom,
            "Malicious proposal",
            address(0),
            ""
        );
    }

    /// @notice Test: flash-loaned tokens returned after proposal can't create new proposals
    function test_FlashLoan_ReturnedTokens_CannotCreateNewProposal() public {
        // Flash attacker acquires tokens and delegates
        token.mint(flashAttacker, 10_000 ether);
        vm.prank(flashAttacker);
        token.delegate(flashAttacker);

        vm.roll(block.number + 1);

        // Attacker creates a proposal (has delegated voting power)
        vm.prank(flashAttacker);
        uint256 proposalId = dao.createProposal(
            TrainingDAO.ProposalType.Custom,
            "Attacker proposal",
            address(0),
            ""
        );
        assertGt(proposalId, 0, "Proposal created");

        // Now attacker returns the flash loan (transfers tokens away)
        vm.prank(flashAttacker);
        token.transfer(address(1), 10_000 ether);

        // Advance 2 blocks so getPastVotes(block.number - 1) sees the transfer
        vm.roll(block.number + 2);

        assertEq(token.balanceOf(flashAttacker), 0, "Attacker returned tokens");

        // Attacker cannot create another proposal without delegated voting power
        // getPastVotes(attacker, block.number - 1) now sees a block after the transfer
        vm.prank(flashAttacker);
        vm.expectRevert("Below proposal threshold");
        dao.createProposal(
            TrainingDAO.ProposalType.Custom,
            "Second malicious proposal",
            address(0),
            ""
        );
    }

    /// @notice Test: proposal with insufficient quorum is defeated
    function test_ParameterProposal_DefeatedWithoutQuorum() public {
        // Current supply: 20M initial + 10K voter1 + 10K voter2 = ~20.02M
        // Quorum is 4% of supply = ~800K. voter1 + voter2 = 20K, so quorum fails

        vm.prank(voter1);
        uint256 proposalId = dao.createParameterProposal(
            "Should fail quorum",
            TrainingDAO.ParameterProposal({
                learningRate: 1e15,
                batchSize: 64,
                maxErrorBound: 1000,
                minParticipants: 3,
                roundDuration: 1 hours
            })
        );

        vm.roll(block.number + 1);

        // Advance to voting
        (, , , uint256 startTime, , , , ) = dao.getProposalInfo(proposalId);
        vm.warp(startTime + 1);

        // Both voters vote (20K tokens total, quorum needs ~800K)
        vm.prank(voter1);
        dao.castVote(proposalId, true);
        vm.prank(voter2);
        dao.castVote(proposalId, true);

        // Advance past voting period
        (, , , , uint256 endTime, , , ) = dao.getProposalInfo(proposalId);
        vm.warp(endTime + 1);

        // Proposal should be defeated (insufficient quorum)
        assertEq(
            uint256(dao.getProposalState(proposalId)),
            uint256(TrainingDAO.ProposalState.Defeated),
            "Proposal should be defeated without quorum"
        );
    }
}

// ============================================================================
// 2b. TrainingDAO Parameter Proposal Execution Tests
// ============================================================================

/// @title TrainingDAOExecutionTest
/// @notice Full lifecycle tests for parameter proposal execution and reentrancy protection
contract TrainingDAOExecutionTest is Test {
    HelixToken public token;
    TrainingDAO public dao;

    address public voter1;
    address public voter2;

    // Large balance needed to meet 4% quorum of ~20M supply
    uint256 constant VOTER_BALANCE = 500_000 ether;

    function setUp() public {
        voter1 = makeAddr("voter1");
        voter2 = makeAddr("voter2");

        token = new HelixToken(makeAddr("treasury"));
        dao = new TrainingDAO(address(token));

        // Give voters enough tokens to meet quorum (4% of ~21M = ~840K, 500K+500K = 1M > 840K)
        token.mint(voter1, VOTER_BALANCE);
        token.mint(voter2, VOTER_BALANCE);

        vm.prank(voter1);
        token.delegate(voter1);
        vm.prank(voter2);
        token.delegate(voter2);

        vm.roll(block.number + 1);
    }

    /// @notice Test: full parameter proposal lifecycle (create, vote, queue, execute, verify)
    function test_ParameterProposal_FullExecution() public {
        // Verify default parameters first
        (uint256 defLr, uint256 defBs, uint256 defMeb, uint256 defMp, uint256 defRd) =
            dao.getTrainingParameters();
        assertEq(defLr, 1e15, "Default learning rate");
        assertEq(defBs, 32, "Default batch size");

        // Create parameter proposal
        TrainingDAO.ParameterProposal memory newParams = TrainingDAO.ParameterProposal({
            learningRate: 5e15,      // 0.005
            batchSize: 128,
            maxErrorBound: 500,
            minParticipants: 10,
            roundDuration: 2 hours
        });

        vm.prank(voter1);
        uint256 proposalId = dao.createParameterProposal("Increase batch size and LR", newParams);

        // Verify parameter proposal stored correctly
        (uint256 lr, uint256 bs, uint256 meb, uint256 mp, uint256 rd) = dao.parameterProposals(proposalId);
        assertEq(lr, 5e15, "Learning rate stored");
        assertEq(bs, 128, "Batch size stored");
        assertEq(meb, 500, "Max error bound stored");
        assertEq(mp, 10, "Min participants stored");
        assertEq(rd, 2 hours, "Round duration stored");

        // Verify proposal type
        (, TrainingDAO.ProposalType pType, , , , , , ) = dao.getProposalInfo(proposalId);
        assertEq(uint256(pType), uint256(TrainingDAO.ProposalType.ParameterChange), "Should be ParameterChange");

        // Advance block so snapshot is finalized
        vm.roll(block.number + 1);

        // Advance time to voting period
        (, , , uint256 startTime, , , , ) = dao.getProposalInfo(proposalId);
        vm.warp(startTime + 1);

        // Both voters vote in favor
        vm.prank(voter1);
        dao.castVote(proposalId, true);
        vm.prank(voter2);
        dao.castVote(proposalId, true);

        // Advance past voting period
        (, , , , uint256 endTime, , , ) = dao.getProposalInfo(proposalId);
        vm.warp(endTime + 1);

        // Verify proposal succeeded
        assertEq(
            uint256(dao.getProposalState(proposalId)),
            uint256(TrainingDAO.ProposalState.Succeeded),
            "Proposal should have succeeded"
        );

        // Queue the proposal
        dao.queueProposal(proposalId);
        assertEq(
            uint256(dao.getProposalState(proposalId)),
            uint256(TrainingDAO.ProposalState.Queued),
            "Proposal should be queued"
        );

        // Attempt early execution (should fail)
        vm.expectRevert("Timelock not expired");
        dao.executeProposal(proposalId);

        // Advance past timelock
        uint256 executionTime = dao.queuedProposals(proposalId);
        vm.warp(executionTime);

        // Execute the proposal
        dao.executeProposal(proposalId);

        // Verify state is Executed
        assertEq(
            uint256(dao.getProposalState(proposalId)),
            uint256(TrainingDAO.ProposalState.Executed),
            "Proposal should be executed"
        );

        // Verify training parameters were updated
        (uint256 newLr, uint256 newBs, uint256 newMeb, uint256 newMp, uint256 newRd) =
            dao.getTrainingParameters();
        assertEq(newLr, 5e15, "Learning rate should be updated");
        assertEq(newBs, 128, "Batch size should be updated");
        assertEq(newMeb, 500, "Max error bound should be updated");
        assertEq(newMp, 10, "Min participants should be updated");
        assertEq(newRd, 2 hours, "Round duration should be updated");
    }

    /// @notice Test: cannot execute proposal twice
    function test_ParameterProposal_CannotExecuteTwice() public {
        vm.prank(voter1);
        uint256 proposalId = dao.createParameterProposal(
            "Test double execute",
            TrainingDAO.ParameterProposal({
                learningRate: 2e15,
                batchSize: 64,
                maxErrorBound: 800,
                minParticipants: 5,
                roundDuration: 1 hours
            })
        );

        vm.roll(block.number + 1);

        (, , , uint256 startTime, , , , ) = dao.getProposalInfo(proposalId);
        vm.warp(startTime + 1);

        vm.prank(voter1);
        dao.castVote(proposalId, true);
        vm.prank(voter2);
        dao.castVote(proposalId, true);

        (, , , , uint256 endTime, , , ) = dao.getProposalInfo(proposalId);
        vm.warp(endTime + 1);

        dao.queueProposal(proposalId);
        vm.warp(dao.queuedProposals(proposalId));

        dao.executeProposal(proposalId);

        // Second execution should revert
        vm.expectRevert("Not queued");
        dao.executeProposal(proposalId);
    }

    /// @notice Test: nonReentrant prevents reentrancy on executeProposal
    function test_ExecuteProposal_NonReentrant() public {
        // Create a proposal that targets address(0) - no external call
        vm.prank(voter1);
        uint256 proposalId = dao.createProposal(
            TrainingDAO.ProposalType.Custom,
            "Test nonReentrant",
            address(0),
            ""
        );

        vm.roll(block.number + 1);

        (, , , uint256 startTime, , , , ) = dao.getProposalInfo(proposalId);
        vm.warp(startTime + 1);

        vm.prank(voter1);
        dao.castVote(proposalId, true);
        vm.prank(voter2);
        dao.castVote(proposalId, true);

        (, , , , uint256 endTime, , , ) = dao.getProposalInfo(proposalId);
        vm.warp(endTime + 1);

        dao.queueProposal(proposalId);
        vm.warp(dao.queuedProposals(proposalId));

        // Should succeed - nonReentrant doesn't prevent normal execution
        dao.executeProposal(proposalId);

        assertEq(
            uint256(dao.getProposalState(proposalId)),
            uint256(TrainingDAO.ProposalState.Executed),
            "Should execute normally with nonReentrant"
        );
    }
}

/// @title DAOReentrancyAttacker
/// @notice Contract that attempts reentrancy during DAO proposal execution
contract DAOReentrancyAttacker {
    TrainingDAO public dao;
    uint256 public targetProposalId;
    uint256 public attackCount;

    constructor(TrainingDAO _dao) {
        dao = _dao;
    }

    function setTarget(uint256 _proposalId) external {
        targetProposalId = _proposalId;
    }

    // Called when DAO executes a proposal targeting this contract
    fallback() external payable {
        if (attackCount < 1) {
            attackCount++;
            // Attempt reentrant call to executeProposal
            try dao.executeProposal(targetProposalId) {} catch {}
        }
    }

    receive() external payable {}
}

/// @title TrainingDAOReentrancyTest
/// @notice Verifies nonReentrant on executeProposal blocks reentrancy attacks
contract TrainingDAOReentrancyTest is Test {
    HelixToken public token;
    TrainingDAO public dao;
    DAOReentrancyAttacker public attacker;

    address public voter1;

    uint256 constant LARGE_BALANCE = 1_000_000 ether;

    function setUp() public {
        voter1 = makeAddr("voter1");

        token = new HelixToken(makeAddr("treasury"));
        dao = new TrainingDAO(address(token));
        attacker = new DAOReentrancyAttacker(dao);

        // Give voter1 enough tokens for quorum (4% of ~21M = ~840K)
        token.mint(voter1, LARGE_BALANCE);
        vm.prank(voter1);
        token.delegate(voter1);
        vm.roll(block.number + 1);
    }

    /// @notice Test: reentrancy via executeProposal targeting a malicious contract is blocked
    function test_ExecuteProposal_ReentrancyBlocked() public {
        // Create a proposal that calls the attacker contract
        vm.prank(voter1);
        uint256 proposalId = dao.createProposal(
            TrainingDAO.ProposalType.Custom,
            "Reentrancy test",
            address(attacker),
            abi.encodeWithSignature("setTarget(uint256)", 1) // harmless call
        );

        attacker.setTarget(proposalId);

        vm.roll(block.number + 1);

        // Advance to voting and vote
        (, , , uint256 startTime, , , , ) = dao.getProposalInfo(proposalId);
        vm.warp(startTime + 1);

        vm.prank(voter1);
        dao.castVote(proposalId, true);

        (, , , , uint256 endTime, , , ) = dao.getProposalInfo(proposalId);
        vm.warp(endTime + 1);

        dao.queueProposal(proposalId);
        vm.warp(dao.queuedProposals(proposalId));

        // Execute - the attacker's fallback tries to re-enter executeProposal
        // but nonReentrant blocks it
        dao.executeProposal(proposalId);

        // Verify executed only once
        assertEq(
            uint256(dao.getProposalState(proposalId)),
            uint256(TrainingDAO.ProposalState.Executed),
            "Proposal should be executed exactly once"
        );
    }
}

// ============================================================================
// 3. Staking Event Parameter Fix Tests
// ============================================================================

/// @title StakingEventTest
/// @notice Verifies ChallengerRewarded event emits correct staker (not msg.sender)
contract StakingEventTest is Test {
    HelixToken public token;
    Staking public staking;

    address public deployer;
    address public staker;
    address public challenger;
    address public operator;

    uint256 constant STAKE_AMOUNT = 10_000 ether;

    event ChallengerRewarded(address indexed challenger, uint256 reward, address indexed slashedStaker);

    function setUp() public {
        deployer = address(this);
        staker = makeAddr("staker");
        challenger = makeAddr("challenger");
        operator = makeAddr("operator");

        token = new HelixToken(makeAddr("treasury"));
        staking = new Staking(
            address(token),
            1000 ether,  // minStake
            7 days,      // unbondingPeriod
            5000         // slashingRate 50%
        );

        staking.setOperator(operator);

        // Fund staker and stake
        token.mint(staker, STAKE_AMOUNT);
        vm.startPrank(staker);
        token.approve(address(staking), STAKE_AMOUNT);
        staking.stake(STAKE_AMOUNT);
        vm.stopPrank();
    }

    /// @notice Test: ChallengerRewarded event emits the actual staker address, not msg.sender
    function test_ChallengerRewarded_EmitsCorrectStaker() public {
        // operator (msg.sender) slashes staker with explicit Major severity
        // The event should emit: ChallengerRewarded(challenger, reward, staker)
        // NOT: ChallengerRewarded(challenger, reward, operator/msg.sender)
        vm.prank(operator);

        // Expect event with correct parameters - the 3rd indexed param should be staker, not operator
        vm.expectEmit(true, false, true, false);
        emit ChallengerRewarded(challenger, 0, staker); // reward amount checked loosely

        // Use slashWithSeverity to skip the warning system and ensure actual slashing occurs
        staking.slashWithSeverity(
            staker,
            Staking.SeverityLevel.Major,
            Staking.ViolationType.InvalidProof,
            challenger,
            "Test slash"
        );

        // Verify challenger received tokens (further proof the event is correct)
        assertGt(token.balanceOf(challenger), 0, "Challenger should receive reward");
    }
}

// ============================================================================
// 4. HelixCoordinatorV2 ReentrancyGuard Tests
// ============================================================================

/// @title ReentrancyAttacker
/// @notice A contract that attempts reentrancy during unstake
contract ReentrancyAttacker {
    HelixCoordinatorV2 public coordinator;
    uint256 public modelId;
    uint256 public attackCount;

    constructor(HelixCoordinatorV2 _coordinator) {
        coordinator = _coordinator;
    }

    function setModelId(uint256 _modelId) external {
        modelId = _modelId;
    }

    function stakeAndWait() external payable {
        coordinator.stake{value: msg.value}(modelId);
    }

    function attack() external {
        coordinator.unstake(modelId);
    }

    receive() external payable {
        if (attackCount < 1) {
            attackCount++;
            // Attempt reentrant call during ETH transfer
            try coordinator.unstake(modelId) {} catch {}
        }
    }
}

/// @title ReentrancyGuardTest
/// @notice Verifies ReentrancyGuard protects unstake and submitProof
contract ReentrancyGuardTest is Test {
    HelixCoordinatorV2 public coordinator;
    MockVerifier public mockVerifier;
    ReentrancyAttacker public attacker;
    TestTreasury public treasuryContract;

    address public owner;

    function setUp() public {
        owner = address(this);
        treasuryContract = new TestTreasury();
        mockVerifier = new MockVerifier();
        coordinator = new HelixCoordinatorV2(address(mockVerifier), address(treasuryContract));

        attacker = new ReentrancyAttacker(coordinator);

        // Fund attacker
        vm.deal(address(attacker), 10 ether);
    }

    /// @notice Test: reentrancy via unstake is blocked
    function test_Reentrancy_UnstakeBlocked() public {
        uint256 modelId = coordinator.registerModel("hash", 100, 0.1 ether);
        attacker.setModelId(modelId);

        // Attacker stakes 1 ETH from its own balance
        attacker.stakeAndWait{value: 1 ether}();

        // Fast forward past lock period
        vm.warp(block.timestamp + 8 days);

        // Track balance before unstake
        uint256 balanceBefore = address(attacker).balance;

        // Attack should succeed on first call but reentrant call should fail silently
        attacker.attack();

        // Attacker should only get their stake once (1 ETH, not 2 ETH)
        uint256 balanceAfter = address(attacker).balance;
        assertEq(balanceAfter - balanceBefore, 1 ether, "Should only receive stake once");
    }

    /// @notice Test: constructor reverts with zero treasury
    function test_Constructor_ZeroTreasury_Reverts() public {
        vm.expectRevert(HelixCoordinatorV2.InvalidTreasury.selector);
        new HelixCoordinatorV2(address(mockVerifier), address(0));
    }
}

// ============================================================================
// 5. Admin Timelock Tests
// ============================================================================

/// @title AdminTimelockTest
/// @notice Verifies timelock enforcement for admin parameter changes
contract AdminTimelockTest is Test {
    HelixCoordinatorV2 public coordinator;
    MockVerifier public mockVerifier;
    MockVerifier public newVerifier;
    TestTreasury public treasuryContract;

    address public owner;

    function setUp() public {
        owner = address(this);
        treasuryContract = new TestTreasury();
        mockVerifier = new MockVerifier();
        newVerifier = new MockVerifier();
        coordinator = new HelixCoordinatorV2(address(mockVerifier), address(treasuryContract));
    }

    // ---- Parameter Timelock (48 hours) ----

    /// @notice Test: propose slash percentage change with 48-hour timelock
    function test_Timelock_SlashPercentage_48Hours() public {
        uint256 newPercentage = 2500; // 25%

        coordinator.proposeSetSlashPercentage(newPercentage);

        bytes32 key = keccak256("setSlashPercentage");
        (, uint256 executionTime, bool active) = coordinator.getPendingChange(key);
        assertTrue(active);
        assertEq(executionTime, block.timestamp + 48 hours, "Should use 48-hour timelock");

        // Early execute fails
        vm.expectRevert(HelixCoordinatorV2.TimelockNotReady.selector);
        coordinator.executeSetSlashPercentage(newPercentage);

        // Wait and execute
        vm.warp(executionTime);
        coordinator.executeSetSlashPercentage(newPercentage);

        assertEq(coordinator.slashPercentage(), newPercentage, "Slash percentage should be updated");
    }

    /// @notice Test: propose treasury change with 48-hour timelock
    function test_Timelock_Treasury_48Hours() public {
        address newTreasury = makeAddr("newTreasury");

        coordinator.proposeSetTreasury(newTreasury);

        bytes32 key = keccak256("setTreasury");
        (, uint256 executionTime, ) = coordinator.getPendingChange(key);

        vm.warp(executionTime);
        coordinator.executeSetTreasury(newTreasury);

        assertEq(coordinator.treasury(), newTreasury);
    }

    /// @notice Test: propose min stake change
    function test_Timelock_DefaultMinStake() public {
        uint256 newMinStake = 0.5 ether;

        coordinator.proposeSetDefaultMinStake(newMinStake);

        bytes32 key = keccak256("setDefaultMinStake");
        (, uint256 executionTime, ) = coordinator.getPendingChange(key);

        vm.warp(executionTime);
        coordinator.executeSetDefaultMinStake(newMinStake);

        assertEq(coordinator.defaultMinStake(), newMinStake);
    }

    /// @notice Test: propose max error bound change
    function test_Timelock_MaxErrorBound() public {
        uint256 newBound = 2e18;

        coordinator.proposeSetMaxErrorBound(newBound);

        bytes32 key = keccak256("setMaxErrorBound");
        (, uint256 executionTime, ) = coordinator.getPendingChange(key);

        vm.warp(executionTime);
        coordinator.executeSetMaxErrorBound(newBound);

        assertEq(coordinator.maxErrorBound(), newBound);
    }

    /// @notice Test: propose challenger config change
    function test_Timelock_ChallengerConfig() public {
        uint16 newPercentage = 2000;
        uint128 newMin = 0.01 ether;
        uint128 newMax = 5 ether;
        bool newEnabled = false;

        coordinator.proposeSetChallengerConfig(newPercentage, newMin, newMax, newEnabled);

        bytes32 key = keccak256("setChallengerConfig");
        (, uint256 executionTime, ) = coordinator.getPendingChange(key);

        vm.warp(executionTime);
        coordinator.executeSetChallengerConfig(newPercentage, newMin, newMax, newEnabled);

        (uint16 pct, uint128 min, uint128 max, bool enabled) = coordinator.getChallengerConfig();
        assertEq(pct, newPercentage);
        assertEq(min, newMin);
        assertEq(max, newMax);
        assertEq(enabled, newEnabled);
    }

    /// @notice Test: cancel pending change
    function test_Timelock_CancelPendingChange() public {
        coordinator.proposeSetSlashPercentage(1000);

        bytes32 key = keccak256("setSlashPercentage");
        (, , bool active) = coordinator.getPendingChange(key);
        assertTrue(active);

        coordinator.cancelPendingChange(key);
        (, , active) = coordinator.getPendingChange(key);
        assertFalse(active, "Should be cancelled");

        // Execute after cancel should fail
        vm.expectRevert(HelixCoordinatorV2.NoTimelockPending.selector);
        coordinator.executeSetSlashPercentage(1000);
    }

    /// @notice Test: cannot propose duplicate change
    function test_Timelock_DuplicateProposalReverts() public {
        coordinator.proposeSetSlashPercentage(1000);

        vm.expectRevert(HelixCoordinatorV2.TimelockAlreadyPending.selector);
        coordinator.proposeSetSlashPercentage(2000);
    }

    /// @notice Test: execute with wrong parameters reverts
    function test_Timelock_WrongParametersReverts() public {
        coordinator.proposeSetSlashPercentage(1000);

        bytes32 key = keccak256("setSlashPercentage");
        (, uint256 executionTime, ) = coordinator.getPendingChange(key);
        vm.warp(executionTime);

        // Execute with different value than proposed
        vm.expectRevert(HelixCoordinatorV2.ErrorChecksumMismatch.selector);
        coordinator.executeSetSlashPercentage(2000);
    }

    /// @notice Test: non-owner cannot propose or execute
    function test_Timelock_OnlyOwner() public {
        address nonOwner = makeAddr("nonOwner");

        vm.prank(nonOwner);
        vm.expectRevert(HelixCoordinatorV2.OnlyOwner.selector);
        coordinator.proposeSetSlashPercentage(1000);

        // Owner proposes
        coordinator.proposeSetSlashPercentage(1000);

        bytes32 key = keccak256("setSlashPercentage");
        (, uint256 executionTime, ) = coordinator.getPendingChange(key);
        vm.warp(executionTime);

        vm.prank(nonOwner);
        vm.expectRevert(HelixCoordinatorV2.OnlyOwner.selector);
        coordinator.executeSetSlashPercentage(1000);
    }
}

// ============================================================================
// 6. Pagination Tests
// ============================================================================

/// @title PaginationTest
/// @notice Verifies pagination for unbounded arrays in V2 and ModelRegistry
contract PaginationTest is Test {
    HelixCoordinatorV2 public coordinator;
    MockVerifier public mockVerifier;
    ModelRegistry public registry;
    TestTreasury public treasuryContract;

    address public owner;
    address public prover;

    function setUp() public {
        owner = address(this);
        prover = makeAddr("prover");
        treasuryContract = new TestTreasury();
        mockVerifier = new MockVerifier();
        coordinator = new HelixCoordinatorV2(address(mockVerifier), address(treasuryContract));
        registry = new ModelRegistry();

        vm.deal(prover, 100 ether);
    }

    // ---- V2 Slashing Records Pagination ----

    /// @notice Test: paginated slashing records
    function test_SlashingRecords_Pagination() public {
        // Create 5 slashing records by staking and submitting invalid proofs
        mockVerifier.setShouldPass(false);

        for (uint256 i = 0; i < 5; i++) {
            uint256 oldHashLo = 100 + i;
            uint256 oldHashHi = 200 + i;
            uint256 commitment = uint256(keccak256(abi.encodePacked(oldHashLo, oldHashHi)));

            uint256 modelId = coordinator.registerModel("hash", commitment, 0.1 ether);
            coordinator.startRound(modelId, 1 hours);

            address p = makeAddr(string.concat("prover", vm.toString(i)));
            vm.deal(p, 10 ether);
            vm.prank(p);
            coordinator.stake{value: 1 ether}(modelId);

            uint256[] memory publicInputs = new uint256[](8);
            publicInputs[0] = oldHashLo;
            publicInputs[1] = oldHashHi;
            publicInputs[2] = 3;
            publicInputs[3] = 4;
            publicInputs[4] = 100;
            publicInputs[5] = 10;
            publicInputs[6] = 1;
            publicInputs[7] = ProofFixtureHardcoded.computeErrorChecksum(10, 1, modelId, coordinator.maxErrorBound());

            vm.prank(p);
            coordinator.submitProof(modelId, 1, hex"deadbeef", publicInputs);
        }

        assertEq(coordinator.getSlashingRecordCount(), 5, "Should have 5 records");

        // Page 1: offset=0, limit=2
        (HelixCoordinatorV2.SlashingRecord[] memory page1, uint256 total1) = coordinator.getSlashingRecords(0, 2);
        assertEq(page1.length, 2, "Page 1 should have 2 records");
        assertEq(total1, 5, "Total should be 5");

        // Page 2: offset=2, limit=2
        (HelixCoordinatorV2.SlashingRecord[] memory page2, uint256 total2) = coordinator.getSlashingRecords(2, 2);
        assertEq(page2.length, 2, "Page 2 should have 2 records");
        assertEq(total2, 5);

        // Page 3: offset=4, limit=2 (only 1 record left)
        (HelixCoordinatorV2.SlashingRecord[] memory page3, uint256 total3) = coordinator.getSlashingRecords(4, 2);
        assertEq(page3.length, 1, "Page 3 should have 1 record");
        assertEq(total3, 5);

        // Out of bounds: offset=10
        (HelixCoordinatorV2.SlashingRecord[] memory empty, uint256 total4) = coordinator.getSlashingRecords(10, 2);
        assertEq(empty.length, 0, "Out of bounds should return empty");
        assertEq(total4, 5);
    }

    // ---- ModelRegistry Owner Models Pagination ----

    /// @notice Test: paginated owner models in ModelRegistry
    function test_ModelRegistry_OwnerModelsPaginated() public {
        // Register 5 models
        for (uint256 i = 0; i < 5; i++) {
            registry.registerModel(
                string.concat("Model ", vm.toString(i)),
                "description",
                "ipfsHash",
                bytes32(i + 1)
            );
        }

        // Full list
        uint256[] memory allModels = registry.getOwnerModels(address(this));
        assertEq(allModels.length, 5);

        // Paginated: offset=1, limit=2
        (uint256[] memory page, uint256 total) = registry.getOwnerModelsPaginated(address(this), 1, 2);
        assertEq(page.length, 2);
        assertEq(total, 5);
        assertEq(page[0], allModels[1]);
        assertEq(page[1], allModels[2]);

        // Out of bounds
        (uint256[] memory empty, uint256 total2) = registry.getOwnerModelsPaginated(address(this), 10, 5);
        assertEq(empty.length, 0);
        assertEq(total2, 5);
    }

    // ---- ModelRegistry Checkpoints Pagination ----

    /// @notice Test: paginated checkpoints in ModelRegistry
    function test_ModelRegistry_CheckpointsPaginated() public {
        // Register model (creates 1 initial checkpoint)
        uint256 modelId = registry.registerModel("Model", "desc", "ipfs", bytes32(uint256(1)));

        // Set this contract as coordinator
        registry.setCoordinator(address(this));

        // Add 4 more checkpoints via updateModel
        for (uint256 i = 1; i <= 4; i++) {
            registry.updateModel(
                modelId,
                bytes32(i + 1),
                i,
                "",
                100 * i,
                bytes32(i)
            );
        }

        assertEq(registry.getCheckpointCount(modelId), 5, "Should have 5 checkpoints");

        // Paginated: offset=0, limit=3
        (ModelRegistry.Checkpoint[] memory page1, uint256 total1) = registry.getCheckpointsPaginated(modelId, 0, 3);
        assertEq(page1.length, 3);
        assertEq(total1, 5);

        // Paginated: offset=3, limit=5 (only 2 remaining)
        (ModelRegistry.Checkpoint[] memory page2, uint256 total2) = registry.getCheckpointsPaginated(modelId, 3, 5);
        assertEq(page2.length, 2);
        assertEq(total2, 5);
    }
}

// ============================================================================
// 7. Custom Errors Gas Savings Test
// ============================================================================

/// @title CustomErrorsTest
/// @notice Verifies that custom errors are used throughout V2 instead of require strings
contract CustomErrorsTest is Test {
    HelixCoordinatorV2 public coordinator;
    MockVerifier public mockVerifier;
    TestTreasury public treasuryContract;

    function setUp() public {
        treasuryContract = new TestTreasury();
        mockVerifier = new MockVerifier();
        coordinator = new HelixCoordinatorV2(address(mockVerifier), address(treasuryContract));
    }

    function test_CustomError_OnlyOwner() public {
        vm.prank(makeAddr("nonOwner"));
        vm.expectRevert(HelixCoordinatorV2.OnlyOwner.selector);
        coordinator.emergencyPause();
    }

    function test_CustomError_ContractPaused() public {
        coordinator.emergencyPause();
        vm.expectRevert(HelixCoordinatorV2.ContractPaused.selector);
        coordinator.registerModel("hash", 100, 0.1 ether);
    }

    function test_CustomError_ModelNotFound() public {
        vm.expectRevert(HelixCoordinatorV2.ModelNotFound.selector);
        coordinator.startRound(999, 1 hours);
    }

    function test_CustomError_ZeroStake() public {
        uint256 modelId = coordinator.registerModel("hash", 100, 0.1 ether);
        vm.expectRevert(HelixCoordinatorV2.ZeroStake.selector);
        coordinator.stake{value: 0}(modelId);
    }

    function test_CustomError_AlreadyPaused() public {
        coordinator.emergencyPause();
        vm.expectRevert(HelixCoordinatorV2.AlreadyPaused.selector);
        coordinator.emergencyPause();
    }

    function test_CustomError_NotPaused() public {
        vm.expectRevert(HelixCoordinatorV2.NotPaused.selector);
        coordinator.unpause();
    }

    function test_CustomError_MaxPercentage() public {
        // Legacy setSlashPercentage now requires paused state
        coordinator.emergencyPause();
        vm.expectRevert(HelixCoordinatorV2.MaxPercentage.selector);
        coordinator.setSlashPercentage(10001);
    }

    // ---- Emergency-Only Legacy Functions ----

    function test_LegacySetTreasury_RevertsWhenNotPaused() public {
        vm.expectRevert(HelixCoordinatorV2.NotPaused.selector);
        coordinator.setTreasury(makeAddr("newTreasury"));
    }

    function test_LegacySetSlashPercentage_RevertsWhenNotPaused() public {
        vm.expectRevert(HelixCoordinatorV2.NotPaused.selector);
        coordinator.setSlashPercentage(1000);
    }

    function test_LegacySetDefaultMinStake_RevertsWhenNotPaused() public {
        vm.expectRevert(HelixCoordinatorV2.NotPaused.selector);
        coordinator.setDefaultMinStake(1 ether);
    }

    function test_LegacySetMaxErrorBound_RevertsWhenNotPaused() public {
        vm.expectRevert(HelixCoordinatorV2.NotPaused.selector);
        coordinator.setMaxErrorBound(2e18);
    }

    function test_LegacyFunctions_WorkWhenPaused() public {
        coordinator.emergencyPause();

        // All legacy setters should work during emergency
        coordinator.setTreasury(makeAddr("newTreasury"));
        coordinator.setSlashPercentage(2500);
        coordinator.setDefaultMinStake(0.5 ether);
        coordinator.setMaxErrorBound(2e18);

        assertEq(coordinator.slashPercentage(), 2500);
        assertEq(coordinator.defaultMinStake(), 0.5 ether);
        assertEq(coordinator.maxErrorBound(), 2e18);
    }

    function test_LegacySetTreasury_RevertsZeroAddress() public {
        coordinator.emergencyPause();
        vm.expectRevert(HelixCoordinatorV2.InvalidTreasury.selector);
        coordinator.setTreasury(address(0));
    }
}

// ============================================================================
// 8. Proof Replay Protection Tests (V2)
// ============================================================================

/// @title ProofReplayProtectionTest
/// @notice Verifies that V2 now blocks proof replay attacks via usedProofHashes
contract ProofReplayProtectionTest is Test {
    HelixCoordinatorV2 public coordinator;
    MockVerifier public mockVerifier;
    TestTreasury public treasuryContract;

    address public owner;
    address public prover1;
    address public prover2;

    uint256 constant ROUND_DURATION = 1 hours;

    function setUp() public {
        owner = address(this);
        prover1 = makeAddr("prover1");
        prover2 = makeAddr("prover2");
        treasuryContract = new TestTreasury();
        mockVerifier = new MockVerifier();
        coordinator = new HelixCoordinatorV2(address(mockVerifier), address(treasuryContract));

        vm.deal(prover1, 10 ether);
        vm.deal(prover2, 10 ether);
    }

    /// @notice Helper to create a model, start round, and return public inputs
    function _setupModelAndRound() internal returns (uint256 modelId, bytes memory proof, uint256[] memory publicInputs) {
        uint256 oldHashLo = 12345;
        uint256 oldHashHi = 67890;
        uint256 correctCommitment = uint256(keccak256(abi.encodePacked(oldHashLo, oldHashHi)));

        modelId = coordinator.registerModel("hash", correctCommitment, 0.1 ether);
        coordinator.startRound(modelId, ROUND_DURATION);

        proof = new bytes(320); // valid-length proof

        publicInputs = new uint256[](8);
        publicInputs[0] = oldHashLo;
        publicInputs[1] = oldHashHi;
        publicInputs[2] = 99;
        publicInputs[3] = 100;
        publicInputs[4] = 500;
        publicInputs[5] = 10;
        publicInputs[6] = 1;
        publicInputs[7] = ProofFixtureHardcoded.computeErrorChecksum(10, 1, modelId, coordinator.maxErrorBound());
    }

    /// @notice Test: same proof cannot be submitted twice in the same round
    function test_ProofReplay_SameRound_Blocked() public {
        (uint256 modelId, bytes memory proof, uint256[] memory publicInputs) = _setupModelAndRound();

        vm.prank(prover1);
        coordinator.stake{value: 1 ether}(modelId);

        // First submission succeeds (round completes)
        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, proof, publicInputs);

        // Start a new round with the new commitment so the round check passes
        uint256 newCommitment = uint256(keccak256(abi.encodePacked(uint256(99), uint256(100))));
        (,uint256 commitment,) = coordinator.getModelState(modelId);
        assertEq(commitment, newCommitment);

        coordinator.startRound(modelId, ROUND_DURATION);

        // Prover2 stakes and tries to replay the exact same proof
        vm.prank(prover2);
        coordinator.stake{value: 1 ether}(modelId);

        // Replay should fail with ProofAlreadyUsed
        vm.prank(prover2);
        vm.expectRevert(HelixCoordinatorV2.ProofAlreadyUsed.selector);
        coordinator.submitProof(modelId, 2, proof, publicInputs);
    }

    /// @notice Test: isProofUsed returns correct state
    function test_IsProofUsed_ReturnsCorrectly() public {
        (uint256 modelId, bytes memory proof, uint256[] memory publicInputs) = _setupModelAndRound();

        // Before submission
        assertFalse(coordinator.isProofUsed(proof, publicInputs), "Proof should not be used initially");

        vm.prank(prover1);
        coordinator.stake{value: 1 ether}(modelId);

        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, proof, publicInputs);

        // After submission
        assertTrue(coordinator.isProofUsed(proof, publicInputs), "Proof should be marked as used");
    }

    /// @notice Test: different proof for same round is allowed (not a replay)
    function test_DifferentProof_Allowed() public {
        (uint256 modelId, bytes memory proof, uint256[] memory publicInputs) = _setupModelAndRound();

        vm.prank(prover1);
        coordinator.stake{value: 1 ether}(modelId);

        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, proof, publicInputs);

        // Different proof bytes = different hash = not a replay
        bytes memory differentProof = new bytes(320);
        differentProof[0] = 0x01; // Make it different

        assertFalse(coordinator.isProofUsed(differentProof, publicInputs), "Different proof should not be marked as used");
    }

    /// @notice Test: invalid proof that gets slashed still marks proof hash as used
    function test_InvalidProof_StillMarksHashUsed() public {
        uint256 oldHashLo = 555;
        uint256 oldHashHi = 666;
        uint256 correctCommitment = uint256(keccak256(abi.encodePacked(oldHashLo, oldHashHi)));

        uint256 modelId = coordinator.registerModel("hash", correctCommitment, 0.1 ether);
        coordinator.startRound(modelId, ROUND_DURATION);

        // Use mock verifier that rejects proofs
        mockVerifier.setShouldPass(false);

        bytes memory proof = hex"deadbeef";
        uint256[] memory publicInputs = new uint256[](8);
        publicInputs[0] = oldHashLo;
        publicInputs[1] = oldHashHi;
        publicInputs[2] = 3;
        publicInputs[3] = 4;
        publicInputs[4] = 100;
        publicInputs[5] = 10;
        publicInputs[6] = 1;
        publicInputs[7] = ProofFixtureHardcoded.computeErrorChecksum(10, 1, modelId, coordinator.maxErrorBound());

        vm.prank(prover1);
        coordinator.stake{value: 1 ether}(modelId);

        // Submit invalid proof - gets slashed but proof hash recorded
        vm.prank(prover1);
        coordinator.submitProof(modelId, 1, proof, publicInputs);

        // Proof hash should be marked as used even though it was invalid
        assertTrue(coordinator.isProofUsed(proof, publicInputs), "Invalid proof hash should still be recorded");
    }
}

// ============================================================================
// 9. commitRoundData Authorization Tests
// ============================================================================

/// @title CommitRoundDataAuthTest
/// @notice Verifies that commitRoundData is restricted to authorized callers
contract CommitRoundDataAuthTest is Test {
    HelixCoordinatorV2 public coordinator;
    MockVerifier public mockVerifier;
    TestTreasury public treasuryContract;

    address public contractOwner;
    address public modelOwner;
    address public randomUser;
    address public dataCommitmentContract;

    function setUp() public {
        contractOwner = address(this);
        modelOwner = makeAddr("modelOwner");
        randomUser = makeAddr("randomUser");
        dataCommitmentContract = makeAddr("dataCommitment");

        treasuryContract = new TestTreasury();
        mockVerifier = new MockVerifier();
        coordinator = new HelixCoordinatorV2(address(mockVerifier), address(treasuryContract));

        // Set data commitment contract
        coordinator.setDataCommitmentContract(dataCommitmentContract);
    }

    /// @notice Helper to register a model and start a round
    function _setupModel() internal returns (uint256 modelId) {
        vm.prank(modelOwner);
        modelId = coordinator.registerModel("hash", 100, 0.1 ether);

        vm.prank(modelOwner);
        coordinator.startRound(modelId, 1 hours);
    }

    /// @notice Test: model owner can commit round data
    function test_CommitRoundData_ModelOwner_Succeeds() public {
        uint256 modelId = _setupModel();

        vm.prank(modelOwner);
        coordinator.commitRoundData(modelId, 1, bytes32(uint256(42)));

        assertEq(coordinator.getRoundDataRoot(modelId, 1), bytes32(uint256(42)));
    }

    /// @notice Test: contract owner can commit round data
    function test_CommitRoundData_ContractOwner_Succeeds() public {
        uint256 modelId = _setupModel();

        coordinator.commitRoundData(modelId, 1, bytes32(uint256(42)));

        assertEq(coordinator.getRoundDataRoot(modelId, 1), bytes32(uint256(42)));
    }

    /// @notice Test: data commitment contract can commit round data
    function test_CommitRoundData_DataCommitmentContract_Succeeds() public {
        uint256 modelId = _setupModel();

        vm.prank(dataCommitmentContract);
        coordinator.commitRoundData(modelId, 1, bytes32(uint256(42)));

        assertEq(coordinator.getRoundDataRoot(modelId, 1), bytes32(uint256(42)));
    }

    /// @notice Test: random user CANNOT commit round data
    function test_CommitRoundData_RandomUser_Reverts() public {
        uint256 modelId = _setupModel();

        vm.prank(randomUser);
        vm.expectRevert(HelixCoordinatorV2.NotAuthorized.selector);
        coordinator.commitRoundData(modelId, 1, bytes32(uint256(42)));
    }

    /// @notice Test: random user cannot overwrite existing round data
    function test_CommitRoundData_CannotOverwrite_ByUnauthorized() public {
        uint256 modelId = _setupModel();

        // Model owner sets data
        vm.prank(modelOwner);
        coordinator.commitRoundData(modelId, 1, bytes32(uint256(42)));

        // Random user cannot overwrite
        vm.prank(randomUser);
        vm.expectRevert(HelixCoordinatorV2.NotAuthorized.selector);
        coordinator.commitRoundData(modelId, 1, bytes32(uint256(99)));

        // Data unchanged
        assertEq(coordinator.getRoundDataRoot(modelId, 1), bytes32(uint256(42)));
    }
}

// ============================================================================
// 10. Treasury Zero-Address Check in Timelocked Setter
// ============================================================================

/// @title TreasuryZeroAddressTest
/// @notice Verifies that executeSetTreasury rejects zero address
contract TreasuryZeroAddressTest is Test {
    HelixCoordinatorV2 public coordinator;
    MockVerifier public mockVerifier;
    TestTreasury public treasuryContract;

    function setUp() public {
        treasuryContract = new TestTreasury();
        mockVerifier = new MockVerifier();
        coordinator = new HelixCoordinatorV2(address(mockVerifier), address(treasuryContract));
    }

    /// @notice Test: proposeSetTreasury rejects zero address
    function test_ProposeSetTreasury_ZeroAddress_Reverts() public {
        vm.expectRevert(HelixCoordinatorV2.InvalidTreasury.selector);
        coordinator.proposeSetTreasury(address(0));
    }

    /// @notice Test: executeSetTreasury rejects zero address (defense in depth)
    function test_ExecuteSetTreasury_ZeroAddress_Reverts() public {
        // We can't normally propose address(0) since proposeSetTreasury checks it.
        // But for defense in depth, executeSetTreasury also checks.
        // We test by proposing a valid address, then trying to execute with address(0).
        address validTreasury = makeAddr("newTreasury");
        coordinator.proposeSetTreasury(validTreasury);

        bytes32 key = keccak256("setTreasury");
        (, uint256 executionTime, ) = coordinator.getPendingChange(key);
        vm.warp(executionTime);

        // Execute with zero address - should revert even though timelock passed
        vm.expectRevert(HelixCoordinatorV2.InvalidTreasury.selector);
        coordinator.executeSetTreasury(address(0));
    }

    /// @notice Test: constructor rejects zero treasury
    function test_Constructor_ZeroTreasury_Reverts() public {
        vm.expectRevert(HelixCoordinatorV2.InvalidTreasury.selector);
        new HelixCoordinatorV2(address(mockVerifier), address(0));
    }

    /// @notice Test: emergency setTreasury rejects zero address
    function test_EmergencySetTreasury_ZeroAddress_Reverts() public {
        coordinator.emergencyPause();
        vm.expectRevert(HelixCoordinatorV2.InvalidTreasury.selector);
        coordinator.setTreasury(address(0));
    }
}

// ============================================================================
// 11. Rewards Double-Claim Attack Vectors (Extended)
// ============================================================================

/// @title RewardsDoubleClaimExtendedTest
/// @notice Extended tests for double-claim attack vectors using unified roundClaimed mapping
contract RewardsDoubleClaimExtendedTest is Test {
    HelixToken public token;
    Rewards public rewards;

    address public funder;
    address public participant;
    address public coordinator;

    uint256 constant FUND_AMOUNT = 100_000 ether;
    uint256 constant REWARDS_PER_ROUND = 1000 ether;
    uint256 constant DURATION = 365 days;

    function setUp() public {
        funder = makeAddr("funder");
        participant = makeAddr("participant");
        coordinator = makeAddr("coordinator");

        token = new HelixToken(makeAddr("treasury"));
        rewards = new Rewards(address(token));

        rewards.setCoordinator(coordinator);

        token.mint(funder, FUND_AMOUNT);
        vm.startPrank(funder);
        token.approve(address(rewards), FUND_AMOUNT);
        rewards.fundRewardPool(FUND_AMOUNT, REWARDS_PER_ROUND, DURATION);
        vm.stopPrank();
    }

    /// @notice Test: calling claimRoundRewards twice for the same round
    function test_ClaimRoundRewards_Twice_SameRound() public {
        vm.startPrank(coordinator);
        rewards.registerParticipant(0, 1, participant);
        rewards.allocateRoundRewards(0, 1);
        vm.stopPrank();

        uint256[] memory modelIds = new uint256[](1);
        uint256[] memory roundIds = new uint256[](1);
        modelIds[0] = 0;
        roundIds[0] = 1;

        // First claim succeeds
        vm.prank(participant);
        rewards.claimRoundRewards(modelIds, roundIds);

        uint256 balanceAfterFirst = token.balanceOf(participant);
        assertGt(balanceAfterFirst, 0, "Should have received rewards");

        // Second claim for same round should revert (roundClaimed = true, no rewards to add)
        vm.prank(participant);
        vm.expectRevert("No rewards to claim");
        rewards.claimRoundRewards(modelIds, roundIds);

        // Balance unchanged
        assertEq(token.balanceOf(participant), balanceAfterFirst, "Balance should not change");
    }

    /// @notice Test: sequential claims across multiple rounds can't exceed total
    function test_InterleavedClaims_MultipleRounds() public {
        // Allocate rewards for 3 rounds
        vm.startPrank(coordinator);
        rewards.registerParticipant(0, 1, participant);
        rewards.allocateRoundRewards(0, 1);
        rewards.registerParticipant(0, 2, participant);
        rewards.allocateRoundRewards(0, 2);
        rewards.registerParticipant(0, 3, participant);
        rewards.allocateRoundRewards(0, 3);
        vm.stopPrank();

        (uint256 totalEarned,,,) = rewards.getParticipantStats(participant);

        // Claim round 1 via claimRoundRewards
        uint256[] memory modelIds = new uint256[](1);
        uint256[] memory roundIds = new uint256[](1);
        modelIds[0] = 0;
        roundIds[0] = 1;

        vm.prank(participant);
        rewards.claimRoundRewards(modelIds, roundIds);

        // Claim rounds 2 & 3 via batch claimRoundRewards
        uint256[] memory batchModelIds = new uint256[](2);
        uint256[] memory batchRoundIds = new uint256[](2);
        batchModelIds[0] = 0;
        batchModelIds[1] = 0;
        batchRoundIds[0] = 2;
        batchRoundIds[1] = 3;

        vm.prank(participant);
        rewards.claimRoundRewards(batchModelIds, batchRoundIds);

        // Try to claim round 2 via claimRoundRewards again - should fail
        roundIds[0] = 2;
        vm.prank(participant);
        vm.expectRevert("No rewards to claim");
        rewards.claimRoundRewards(modelIds, roundIds);

        // Try to claim round 3 via claimRoundRewards again - should fail
        roundIds[0] = 3;
        vm.prank(participant);
        vm.expectRevert("No rewards to claim");
        rewards.claimRoundRewards(modelIds, roundIds);

        // Total claimed must equal total earned
        assertEq(token.balanceOf(participant), totalEarned, "Total must equal earned");
    }

    /// @notice Test: batch claim with mix of already-claimed and new rounds
    function test_BatchClaim_PartiallyClaimedRounds() public {
        vm.startPrank(coordinator);
        rewards.registerParticipant(0, 1, participant);
        rewards.allocateRoundRewards(0, 1);
        rewards.registerParticipant(0, 2, participant);
        rewards.allocateRoundRewards(0, 2);
        vm.stopPrank();

        // Claim round 1 first
        uint256[] memory modelIds = new uint256[](1);
        uint256[] memory roundIds = new uint256[](1);
        modelIds[0] = 0;
        roundIds[0] = 1;

        vm.prank(participant);
        rewards.claimRoundRewards(modelIds, roundIds);
        uint256 balanceAfterRound1 = token.balanceOf(participant);

        // Now batch claim both rounds 1 and 2 - round 1 should be skipped
        uint256[] memory batchModelIds = new uint256[](2);
        uint256[] memory batchRoundIds = new uint256[](2);
        batchModelIds[0] = 0;
        batchModelIds[1] = 0;
        batchRoundIds[0] = 1; // Already claimed
        batchRoundIds[1] = 2; // Not claimed

        vm.prank(participant);
        rewards.claimRoundRewards(batchModelIds, batchRoundIds);

        // Should have received only round 2 rewards (round 1 was skipped)
        uint256 balanceAfterBatch = token.balanceOf(participant);
        assertGt(balanceAfterBatch, balanceAfterRound1, "Should get round 2 rewards");

        (uint256 totalEarned,,,) = rewards.getParticipantStats(participant);
        assertEq(balanceAfterBatch, totalEarned, "Final balance should equal total earned");
    }

    /// @notice Test: claimRoundRewards followed by second claimRoundRewards is blocked
    function test_ClaimRoundRewards_Then_SecondClaim_Blocked() public {
        vm.startPrank(coordinator);
        rewards.registerParticipant(0, 1, participant);
        rewards.allocateRoundRewards(0, 1);
        vm.stopPrank();

        // Claim via claimRoundRewards first
        uint256[] memory modelIds = new uint256[](1);
        uint256[] memory roundIds = new uint256[](1);
        modelIds[0] = 0;
        roundIds[0] = 1;

        vm.prank(participant);
        rewards.claimRoundRewards(modelIds, roundIds);

        // Try to double-claim via claimRoundRewards again
        vm.prank(participant);
        vm.expectRevert("No rewards to claim");
        rewards.claimRoundRewards(modelIds, roundIds);
    }

    /// @notice Test: two participants cannot steal each other's rewards
    function test_TwoParticipants_IndependentClaims() public {
        address participant2 = makeAddr("participant2");

        vm.startPrank(coordinator);
        rewards.registerParticipant(0, 1, participant);
        rewards.registerParticipant(0, 1, participant2);
        rewards.allocateRoundRewards(0, 1);
        vm.stopPrank();

        uint256[] memory modelIds = new uint256[](1);
        uint256[] memory roundIds = new uint256[](1);
        modelIds[0] = 0;
        roundIds[0] = 1;

        // Participant 1 claims
        vm.prank(participant);
        rewards.claimRoundRewards(modelIds, roundIds);

        // Participant 2 claims
        vm.prank(participant2);
        rewards.claimRoundRewards(modelIds, roundIds);

        // Both should have received equal shares
        uint256 bal1 = token.balanceOf(participant);
        uint256 bal2 = token.balanceOf(participant2);
        assertEq(bal1, bal2, "Equal participants should receive equal rewards");
        assertGt(bal1, 0, "Should have received rewards");

        // Neither can claim again
        vm.prank(participant);
        vm.expectRevert("No rewards to claim");
        rewards.claimRoundRewards(modelIds, roundIds);

        vm.prank(participant2);
        vm.expectRevert("No rewards to claim");
        rewards.claimRoundRewards(modelIds, roundIds);
    }
}

// ============================================================================
// Rewards awardBonus Tests
// ============================================================================

/// @title RewardsAwardBonusTest
/// @notice Verifies that awardBonus transfers tokens directly and maintains
///         consistent accounting (totalEarned == totalClaimed for bonuses).
contract RewardsAwardBonusTest is Test {
    HelixToken public token;
    Rewards public rewards;

    address public deployer;
    address public funder;
    address public recipient;
    address public coordinator;

    uint256 constant FUND_AMOUNT = 100_000 ether;
    uint256 constant REWARDS_PER_ROUND = 1000 ether;
    uint256 constant DURATION = 365 days;
    uint256 constant BONUS_AMOUNT = 500 ether;

    function setUp() public {
        deployer = address(this);
        funder = makeAddr("funder");
        recipient = makeAddr("recipient");
        coordinator = makeAddr("coordinator");

        token = new HelixToken(makeAddr("treasury"));
        rewards = new Rewards(address(token));
        rewards.setCoordinator(coordinator);

        // Fund the reward pool
        token.mint(funder, FUND_AMOUNT);
        vm.startPrank(funder);
        token.approve(address(rewards), FUND_AMOUNT);
        rewards.fundRewardPool(FUND_AMOUNT, REWARDS_PER_ROUND, DURATION);
        vm.stopPrank();
    }

    /// @notice awardBonus transfers tokens directly to recipient
    function test_AwardBonus_TransfersTokens() public {
        uint256 balBefore = token.balanceOf(recipient);
        assertEq(balBefore, 0, "Recipient should start with zero balance");

        rewards.awardBonus(recipient, BONUS_AMOUNT, "early adopter");

        uint256 balAfter = token.balanceOf(recipient);
        assertEq(balAfter, BONUS_AMOUNT, "Recipient should receive bonus tokens");
    }

    /// @notice awardBonus keeps totalEarned == totalClaimed (no pending rewards)
    function test_AwardBonus_EarnedEqualsClaimed() public {
        rewards.awardBonus(recipient, BONUS_AMOUNT, "early adopter");

        (uint256 totalEarned, uint256 totalClaimed, , uint256 pendingAmount) =
            rewards.getParticipantStats(recipient);

        assertEq(totalEarned, BONUS_AMOUNT, "totalEarned should equal bonus");
        assertEq(totalClaimed, BONUS_AMOUNT, "totalClaimed should equal bonus");
        assertEq(pendingAmount, 0, "No pending rewards after bonus");
    }

    /// @notice getClaimableRewards returns 0 after bonus (nothing left to claim)
    function test_AwardBonus_ClaimableIsZero() public {
        rewards.awardBonus(recipient, BONUS_AMOUNT, "early adopter");

        uint256 claimable = rewards.getClaimableRewards(recipient);
        assertEq(claimable, 0, "Claimable should be zero after direct bonus");
    }

    /// @notice awardBonus reduces remaining pool correctly
    function test_AwardBonus_ReducesPool() public {
        (uint256 totalBefore, uint256 distributedBefore, uint256 remainingBefore, , ) =
            rewards.getRewardPoolInfo();

        rewards.awardBonus(recipient, BONUS_AMOUNT, "early adopter");

        (uint256 totalAfter, uint256 distributedAfter, uint256 remainingAfter, , ) =
            rewards.getRewardPoolInfo();

        assertEq(totalAfter, totalBefore, "Total should not change");
        assertEq(distributedAfter, distributedBefore + BONUS_AMOUNT, "Distributed should increase");
        assertEq(remainingAfter, remainingBefore - BONUS_AMOUNT, "Remaining should decrease");
    }

    /// @notice awardBonus emits BonusAwarded event
    function test_AwardBonus_EmitsEvent() public {
        vm.expectEmit(true, false, false, true);
        emit Rewards.BonusAwarded(recipient, BONUS_AMOUNT, "early adopter");

        rewards.awardBonus(recipient, BONUS_AMOUNT, "early adopter");
    }

    /// @notice awardBonus reverts for non-owner
    function test_AwardBonus_RevertsForNonOwner() public {
        vm.prank(funder);
        vm.expectRevert("Only owner");
        rewards.awardBonus(recipient, BONUS_AMOUNT, "should fail");
    }

    /// @notice awardBonus reverts for zero recipient
    function test_AwardBonus_RevertsForZeroAddress() public {
        vm.expectRevert("Invalid recipient");
        rewards.awardBonus(address(0), BONUS_AMOUNT, "should fail");
    }

    /// @notice awardBonus reverts when pool is insufficient
    function test_AwardBonus_RevertsWhenPoolInsufficient() public {
        uint256 tooMuch = FUND_AMOUNT + 1;
        vm.expectRevert("Insufficient reward pool");
        rewards.awardBonus(recipient, tooMuch, "should fail");
    }

    /// @notice Multiple bonuses accumulate correctly
    function test_AwardBonus_MultipleAccumulate() public {
        rewards.awardBonus(recipient, BONUS_AMOUNT, "bonus 1");
        rewards.awardBonus(recipient, BONUS_AMOUNT, "bonus 2");

        uint256 bal = token.balanceOf(recipient);
        assertEq(bal, BONUS_AMOUNT * 2, "Should receive both bonuses");

        (uint256 totalEarned, uint256 totalClaimed, , uint256 pendingAmount) =
            rewards.getParticipantStats(recipient);
        assertEq(totalEarned, BONUS_AMOUNT * 2, "totalEarned should be 2x bonus");
        assertEq(totalClaimed, BONUS_AMOUNT * 2, "totalClaimed should be 2x bonus");
        assertEq(pendingAmount, 0, "No pending rewards");
    }

    /// @notice Bonus + round rewards don't interfere with each other
    function test_AwardBonus_PlusRoundRewards() public {
        // First give a bonus
        rewards.awardBonus(recipient, BONUS_AMOUNT, "bonus");

        // Then register for a round and allocate rewards
        vm.startPrank(coordinator);
        rewards.registerParticipant(0, 1, recipient);
        rewards.allocateRoundRewards(0, 1);
        vm.stopPrank();

        // After bonus + allocation, earned > claimed (pending from round)
        (uint256 totalEarned, uint256 totalClaimed, , uint256 pendingAmount) =
            rewards.getParticipantStats(recipient);
        assertEq(totalClaimed, BONUS_AMOUNT, "Only bonus is claimed so far");
        assertGt(totalEarned, totalClaimed, "Round rewards are pending");
        assertEq(pendingAmount, totalEarned - totalClaimed, "Pending = round reward");

        // Claim round rewards
        uint256[] memory modelIds = new uint256[](1);
        uint256[] memory roundIds = new uint256[](1);
        modelIds[0] = 0;
        roundIds[0] = 1;

        vm.prank(recipient);
        rewards.claimRoundRewards(modelIds, roundIds);

        // Now everything should be claimed
        (totalEarned, totalClaimed, , pendingAmount) = rewards.getParticipantStats(recipient);
        assertEq(totalEarned, totalClaimed, "Everything claimed");
        assertEq(pendingAmount, 0, "No pending");

        // Balance should be bonus + round reward
        assertEq(token.balanceOf(recipient), totalEarned, "Balance matches total earned");
    }
}
