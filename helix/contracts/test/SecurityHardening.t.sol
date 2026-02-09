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

    /// @notice Core test: attempt double-claim via claimRewards() then claimRoundRewards()
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

        // Step 1: Claim all rewards via claimRewards()
        vm.prank(participant1);
        rewards.claimRewards();

        uint256 balanceAfterFirst = token.balanceOf(participant1);
        assertEq(balanceAfterFirst, expectedReward, "Should receive full reward");

        // Step 2: Attempt to double-claim via claimRoundRewards()
        uint256[] memory modelIds = new uint256[](1);
        uint256[] memory roundIds = new uint256[](1);
        modelIds[0] = modelId;
        roundIds[0] = roundId;

        vm.prank(participant1);
        vm.expectRevert("Already claimed");
        rewards.claimRoundRewards(modelIds, roundIds);

        // Balance should not have changed
        assertEq(token.balanceOf(participant1), balanceAfterFirst, "Balance should not change after failed double-claim");
    }

    /// @notice Test: claim via claimRoundRewards() first, then claimRewards() gives nothing
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

        // Step 2: Try claimRewards() - should revert since nothing left
        vm.prank(participant1);
        vm.expectRevert("No rewards to claim");
        rewards.claimRewards();
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

        // Claim remaining via claimRewards
        vm.prank(participant1);
        rewards.claimRewards();

        // Total claimed should equal total earned (no more, no less)
        assertEq(token.balanceOf(participant1), totalEarned, "Total claimed must equal total earned");

        // Try claiming again - both paths should fail
        vm.prank(participant1);
        vm.expectRevert("No rewards to claim");
        rewards.claimRewards();

        roundIds[0] = 2;
        vm.prank(participant1);
        vm.expectRevert("Already claimed");
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

    // ---- Verifier Timelock (7 days) ----

    /// @notice Test: propose verifier change, attempt early execute, fail, wait, succeed
    function test_Timelock_Verifier_FullCycle() public {
        // Propose
        coordinator.proposeSetVerifier(address(newVerifier));

        // Verify pending change exists
        bytes32 key = keccak256("setVerifier");
        (, uint256 executionTime, bool active) = coordinator.getPendingChange(key);
        assertTrue(active, "Change should be active");
        assertEq(executionTime, block.timestamp + 7 days, "Should use 7-day timelock");

        // Attempt early execution - should revert
        vm.expectRevert(HelixCoordinatorV2.TimelockNotReady.selector);
        coordinator.executeSetVerifier(address(newVerifier));

        // Warp to just before timelock expiry
        vm.warp(executionTime - 1);
        vm.expectRevert(HelixCoordinatorV2.TimelockNotReady.selector);
        coordinator.executeSetVerifier(address(newVerifier));

        // Warp past timelock
        vm.warp(executionTime);
        coordinator.executeSetVerifier(address(newVerifier));

        // Verify change took effect
        assertEq(address(coordinator.verifier()), address(newVerifier), "Verifier should be updated");

        // Verify pending change cleared
        (, , active) = coordinator.getPendingChange(key);
        assertFalse(active, "Pending change should be cleared");
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
        vm.expectRevert(HelixCoordinatorV2.MaxPercentage.selector);
        coordinator.setSlashPercentage(10001);
    }
}
